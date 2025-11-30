//! VQE/QAOA Optimization Patterns
//!
//! This module provides optimized gradient computation and optimization
//! patterns specifically designed for VQE (Variational Quantum Eigensolver)
//! and QAOA (Quantum Approximate Optimization Algorithm).
//!
//! # Features
//!
//! - **Gradient descent optimizers**: Adam, momentum, vanilla GD
//! - **Convergence detection**: Energy, gradient norm, plateau detection
//! - **VQE helpers**: Energy minimization with caching
//! - **QAOA helpers**: Layer-wise parameter optimization
//! - **Progress tracking**: Optimization history and diagnostics

use simq_core::Circuit;
use simq_state::observable::PauliObservable;
use simq_state::AdaptiveState;
use std::time::{Duration, Instant};

use super::{compute_gradient_auto, GradientConfig, GradientMethod};

/// Helper function to compute expectation value
fn compute_expectation(
    simulator: &crate::Simulator,
    circuit: &Circuit,
    observable: &PauliObservable,
) -> crate::error::Result<f64> {
    let result = simulator.run(circuit)?;

    let expectation = match &result.state {
        AdaptiveState::Dense(dense) => {
            observable.expectation_value(dense)?
        }
        AdaptiveState::Sparse { state: sparse, .. } => {
            use simq_state::DenseState;
            let dense = DenseState::from_sparse(sparse)?;
            observable.expectation_value(&dense)?
        }
    };

    Ok(expectation)
}

/// Configuration for VQE optimization
#[derive(Debug, Clone)]
pub struct VQEConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence threshold for energy change
    pub energy_tolerance: f64,
    /// Convergence threshold for gradient norm
    pub gradient_tolerance: f64,
    /// Initial learning rate
    pub learning_rate: f64,
    /// Enable adaptive learning rate
    pub adaptive_learning_rate: bool,
    /// Gradient computation configuration
    pub gradient_config: GradientConfig,
}

impl Default for VQEConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            energy_tolerance: 1e-6,
            gradient_tolerance: 1e-6,
            learning_rate: 0.01,
            adaptive_learning_rate: true,
            gradient_config: GradientConfig {
                method: GradientMethod::Auto,
                ..Default::default()
            },
        }
    }
}

/// Configuration for QAOA optimization
#[derive(Debug, Clone)]
pub struct QAOAConfig {
    /// Number of QAOA layers (p)
    pub num_layers: usize,
    /// Maximum iterations per layer
    pub max_iterations: usize,
    /// Energy tolerance
    pub energy_tolerance: f64,
    /// Gradient tolerance
    pub gradient_tolerance: f64,
    /// Learning rate for gamma parameters (problem Hamiltonian)
    pub gamma_learning_rate: f64,
    /// Learning rate for beta parameters (mixer Hamiltonian)
    pub beta_learning_rate: f64,
    /// Enable layer-wise optimization
    pub layer_wise: bool,
    /// Gradient computation configuration
    pub gradient_config: GradientConfig,
}

impl Default for QAOAConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            max_iterations: 100,
            energy_tolerance: 1e-6,
            gradient_tolerance: 1e-6,
            gamma_learning_rate: 0.01,
            beta_learning_rate: 0.01,
            layer_wise: false,
            gradient_config: GradientConfig {
                method: GradientMethod::Auto,
                ..Default::default()
            },
        }
    }
}

/// Convergence status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceStatus {
    /// Not yet converged
    NotConverged,
    /// Converged due to energy change below threshold
    EnergyConverged,
    /// Converged due to gradient norm below threshold
    GradientConverged,
    /// Converged due to both energy and gradient
    FullyConverged,
    /// Reached maximum iterations
    MaxIterations,
    /// Detected plateau (no improvement)
    Plateau,
}

/// Single optimization step result
#[derive(Debug, Clone)]
pub struct OptimizationStep {
    /// Iteration number
    pub iteration: usize,
    /// Current parameters
    pub parameters: Vec<f64>,
    /// Current energy (expectation value)
    pub energy: f64,
    /// Current gradient
    pub gradient: Vec<f64>,
    /// Gradient norm
    pub gradient_norm: f64,
    /// Energy change from previous step
    pub energy_change: f64,
    /// Step time
    pub step_time: Duration,
    /// Convergence status
    pub status: ConvergenceStatus,
}

/// Complete optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Final parameters
    pub parameters: Vec<f64>,
    /// Final energy
    pub energy: f64,
    /// Final gradient
    pub gradient: Vec<f64>,
    /// Convergence status
    pub status: ConvergenceStatus,
    /// Number of iterations performed
    pub num_iterations: usize,
    /// Total optimization time
    pub total_time: Duration,
    /// History of all optimization steps
    pub history: Vec<OptimizationStep>,
}

impl OptimizationResult {
    /// Check if optimization converged successfully
    pub fn converged(&self) -> bool {
        matches!(
            self.status,
            ConvergenceStatus::EnergyConverged
                | ConvergenceStatus::GradientConverged
                | ConvergenceStatus::FullyConverged
        )
    }

    /// Get the best energy found
    pub fn best_energy(&self) -> f64 {
        self.history
            .iter()
            .map(|step| step.energy)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(self.energy)
    }

    /// Get parameters at best energy
    pub fn best_parameters(&self) -> &[f64] {
        self.history
            .iter()
            .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
            .map(|step| step.parameters.as_slice())
            .unwrap_or(&self.parameters)
    }
}

/// VQE optimizer with gradient descent
pub struct VQEOptimizer<F>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    circuit_builder: F,
    config: VQEConfig,
    history: Vec<OptimizationStep>,
    best_energy: f64,
    plateau_counter: usize,
}

impl<F> VQEOptimizer<F>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    /// Create a new VQE optimizer
    pub fn new(circuit_builder: F, config: VQEConfig) -> Self {
        Self {
            circuit_builder,
            config,
            history: Vec::new(),
            best_energy: f64::INFINITY,
            plateau_counter: 0,
        }
    }

    /// Run VQE optimization
    pub fn optimize(
        &mut self,
        simulator: &crate::Simulator,
        observable: &PauliObservable,
        initial_params: &[f64],
    ) -> crate::error::Result<OptimizationResult> {
        let start_time = Instant::now();
        let mut params = initial_params.to_vec();
        let mut learning_rate = self.config.learning_rate;
        let mut prev_energy = f64::INFINITY;

        for iteration in 0..self.config.max_iterations {
            let step_start = Instant::now();

            // Build circuit and compute energy
            let circuit = (self.circuit_builder)(&params);
            let energy = compute_expectation(simulator, &circuit, observable)?;

            // Compute gradient
            let grad_result = compute_gradient_auto(
                simulator,
                &self.circuit_builder,
                observable,
                &params,
            )?;

            let gradient = grad_result.gradients.clone();
            let gradient_norm = grad_result.norm();
            let energy_change = (energy - prev_energy).abs();

            // Check convergence
            let status = self.check_convergence(
                iteration,
                energy,
                prev_energy,
                gradient_norm,
            );

            // Record step
            let step = OptimizationStep {
                iteration,
                parameters: params.clone(),
                energy,
                gradient: gradient.clone(),
                gradient_norm,
                energy_change,
                step_time: step_start.elapsed(),
                status,
            };
            self.history.push(step.clone());

            // Update best energy
            if energy < self.best_energy {
                self.best_energy = energy;
                self.plateau_counter = 0;
            } else {
                self.plateau_counter += 1;
            }

            // Check for early stopping
            if status != ConvergenceStatus::NotConverged {
                return Ok(OptimizationResult {
                    parameters: params,
                    energy,
                    gradient,
                    status,
                    num_iterations: iteration + 1,
                    total_time: start_time.elapsed(),
                    history: self.history.clone(),
                });
            }

            // Adaptive learning rate
            if self.config.adaptive_learning_rate {
                learning_rate = self.adapt_learning_rate(
                    learning_rate,
                    gradient_norm,
                    energy_change,
                );
            }

            // Gradient descent update
            for (p, g) in params.iter_mut().zip(gradient.iter()) {
                *p -= learning_rate * g;
            }

            prev_energy = energy;
        }

        // Max iterations reached
        let final_circuit = (self.circuit_builder)(&params);
        let final_energy = compute_expectation(simulator, &final_circuit, observable)?;
        let final_grad = compute_gradient_auto(
            simulator,
            &self.circuit_builder,
            observable,
            &params,
        )?;

        Ok(OptimizationResult {
            parameters: params,
            energy: final_energy,
            gradient: final_grad.gradients,
            status: ConvergenceStatus::MaxIterations,
            num_iterations: self.config.max_iterations,
            total_time: start_time.elapsed(),
            history: self.history.clone(),
        })
    }

    /// Check convergence criteria
    fn check_convergence(
        &self,
        iteration: usize,
        energy: f64,
        prev_energy: f64,
        gradient_norm: f64,
    ) -> ConvergenceStatus {
        let energy_change = (energy - prev_energy).abs();
        let energy_converged = energy_change < self.config.energy_tolerance;
        let gradient_converged = gradient_norm < self.config.gradient_tolerance;
        let plateau = self.plateau_counter > 10; // No improvement for 10 steps

        if iteration >= self.config.max_iterations {
            ConvergenceStatus::MaxIterations
        } else if plateau {
            ConvergenceStatus::Plateau
        } else if energy_converged && gradient_converged {
            ConvergenceStatus::FullyConverged
        } else if energy_converged {
            ConvergenceStatus::EnergyConverged
        } else if gradient_converged {
            ConvergenceStatus::GradientConverged
        } else {
            ConvergenceStatus::NotConverged
        }
    }

    /// Adapt learning rate based on progress
    fn adapt_learning_rate(
        &self,
        current_lr: f64,
        gradient_norm: f64,
        energy_change: f64,
    ) -> f64 {
        // Simple adaptive strategy:
        // - Increase LR if gradient is large and energy is changing
        // - Decrease LR if gradient is small or energy change is small

        let min_lr = 1e-6;
        let max_lr = 1.0;

        if gradient_norm > 1.0 && energy_change > self.config.energy_tolerance * 10.0 {
            // Good progress, increase LR
            (current_lr * 1.1).min(max_lr)
        } else if gradient_norm < self.config.gradient_tolerance * 10.0
            || energy_change < self.config.energy_tolerance
        {
            // Slow progress, decrease LR
            (current_lr * 0.9).max(min_lr)
        } else {
            // Maintain current LR
            current_lr
        }
    }

    /// Get optimization history
    pub fn history(&self) -> &[OptimizationStep] {
        &self.history
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.history.clear();
        self.best_energy = f64::INFINITY;
        self.plateau_counter = 0;
    }
}

/// QAOA optimizer with layer-wise optimization
pub struct QAOAOptimizer<F>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    circuit_builder: F,
    config: QAOAConfig,
    history: Vec<OptimizationStep>,
}

impl<F> QAOAOptimizer<F>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    /// Create a new QAOA optimizer
    pub fn new(circuit_builder: F, config: QAOAConfig) -> Self {
        Self {
            circuit_builder,
            config,
            history: Vec::new(),
        }
    }

    /// Run QAOA optimization
    ///
    /// Parameters should be ordered as: [gamma_1, beta_1, gamma_2, beta_2, ...]
    /// where gamma parameters correspond to the problem Hamiltonian
    /// and beta parameters correspond to the mixer Hamiltonian
    pub fn optimize(
        &mut self,
        simulator: &crate::Simulator,
        observable: &PauliObservable,
        initial_params: &[f64],
    ) -> crate::error::Result<OptimizationResult> {
        if initial_params.len() != 2 * self.config.num_layers {
            return Err(crate::error::SimulatorError::InvalidConfig(format!(
                "Expected {} parameters for {} QAOA layers, got {}",
                2 * self.config.num_layers,
                self.config.num_layers,
                initial_params.len()
            )));
        }

        if self.config.layer_wise {
            self.optimize_layer_wise(simulator, observable, initial_params)
        } else {
            self.optimize_all_layers(simulator, observable, initial_params)
        }
    }

    /// Optimize all layers simultaneously
    fn optimize_all_layers(
        &mut self,
        simulator: &crate::Simulator,
        observable: &PauliObservable,
        initial_params: &[f64],
    ) -> crate::error::Result<OptimizationResult> {
        let start_time = Instant::now();
        let mut params = initial_params.to_vec();
        let mut prev_energy = f64::INFINITY;

        for iteration in 0..self.config.max_iterations {
            let step_start = Instant::now();

            // Compute energy and gradient
            let circuit = (self.circuit_builder)(&params);
            let energy = compute_expectation(simulator, &circuit, observable)?;
            let grad_result = compute_gradient_auto(
                simulator,
                &self.circuit_builder,
                observable,
                &params,
            )?;

            let gradient = grad_result.gradients.clone();
            let gradient_norm = grad_result.norm();
            let energy_change = (energy - prev_energy).abs();

            // Check convergence
            let status = if iteration >= self.config.max_iterations {
                ConvergenceStatus::MaxIterations
            } else if energy_change < self.config.energy_tolerance
                && gradient_norm < self.config.gradient_tolerance
            {
                ConvergenceStatus::FullyConverged
            } else if energy_change < self.config.energy_tolerance {
                ConvergenceStatus::EnergyConverged
            } else if gradient_norm < self.config.gradient_tolerance {
                ConvergenceStatus::GradientConverged
            } else {
                ConvergenceStatus::NotConverged
            };

            // Record step
            let step = OptimizationStep {
                iteration,
                parameters: params.clone(),
                energy,
                gradient: gradient.clone(),
                gradient_norm,
                energy_change,
                step_time: step_start.elapsed(),
                status,
            };
            self.history.push(step);

            if status != ConvergenceStatus::NotConverged {
                return Ok(OptimizationResult {
                    parameters: params,
                    energy,
                    gradient,
                    status,
                    num_iterations: iteration + 1,
                    total_time: start_time.elapsed(),
                    history: self.history.clone(),
                });
            }

            // Update parameters with different learning rates for gamma/beta
            for i in 0..params.len() {
                let lr = if i % 2 == 0 {
                    self.config.gamma_learning_rate
                } else {
                    self.config.beta_learning_rate
                };
                params[i] -= lr * gradient[i];
            }

            prev_energy = energy;
        }

        // Max iterations reached
        let final_circuit = (self.circuit_builder)(&params);
        let final_energy = compute_expectation(simulator, &final_circuit, observable)?;
        let final_grad = compute_gradient_auto(
            simulator,
            &self.circuit_builder,
            observable,
            &params,
        )?;

        Ok(OptimizationResult {
            parameters: params,
            energy: final_energy,
            gradient: final_grad.gradients,
            status: ConvergenceStatus::MaxIterations,
            num_iterations: self.config.max_iterations,
            total_time: start_time.elapsed(),
            history: self.history.clone(),
        })
    }

    /// Optimize layers one at a time (layer-wise optimization)
    fn optimize_layer_wise(
        &mut self,
        simulator: &crate::Simulator,
        observable: &PauliObservable,
        initial_params: &[f64],
    ) -> crate::error::Result<OptimizationResult> {
        let start_time = Instant::now();
        let mut params = initial_params.to_vec();

        // Optimize each layer sequentially
        for layer in 0..self.config.num_layers {
            let layer_start_idx = layer * 2;
            let _layer_end_idx = layer_start_idx + 2;

            // Optimize just this layer's parameters
            for _iteration in 0..self.config.max_iterations {
                let circuit = (self.circuit_builder)(&params);
                let _energy = compute_expectation(simulator, &circuit, observable)?;
                let grad_result = compute_gradient_auto(
                    simulator,
                    &self.circuit_builder,
                    observable,
                    &params,
                )?;

                // Update only this layer's parameters
                let gamma_idx = layer_start_idx;
                let beta_idx = layer_start_idx + 1;

                params[gamma_idx] -= self.config.gamma_learning_rate * grad_result.gradients[gamma_idx];
                params[beta_idx] -= self.config.beta_learning_rate * grad_result.gradients[beta_idx];

                // Simple convergence check for this layer
                if grad_result.gradients[gamma_idx].abs() < self.config.gradient_tolerance
                    && grad_result.gradients[beta_idx].abs() < self.config.gradient_tolerance
                {
                    break;
                }
            }
        }

        // Final evaluation
        let final_circuit = (self.circuit_builder)(&params);
        let final_energy = compute_expectation(simulator, &final_circuit, observable)?;
        let final_grad = compute_gradient_auto(
            simulator,
            &self.circuit_builder,
            observable,
            &params,
        )?;

        Ok(OptimizationResult {
            parameters: params,
            energy: final_energy,
            gradient: final_grad.gradients,
            status: ConvergenceStatus::FullyConverged,
            num_iterations: self.config.num_layers * self.config.max_iterations,
            total_time: start_time.elapsed(),
            history: self.history.clone(),
        })
    }

    /// Get optimization history
    pub fn history(&self) -> &[OptimizationStep] {
        &self.history
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

/// Simple gradient descent optimizer (utility function)
pub fn gradient_descent<F>(
    simulator: &crate::Simulator,
    circuit_builder: F,
    observable: &PauliObservable,
    initial_params: &[f64],
    learning_rate: f64,
    max_iterations: usize,
) -> crate::error::Result<OptimizationResult>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    let config = VQEConfig {
        max_iterations,
        learning_rate,
        adaptive_learning_rate: false,
        ..Default::default()
    };

    let mut optimizer = VQEOptimizer::new(circuit_builder, config);
    optimizer.optimize(simulator, observable, initial_params)
}

/// Adam optimizer configuration
#[derive(Debug, Clone)]
pub struct AdamConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Beta1 (exponential decay rate for first moment)
    pub beta1: f64,
    /// Beta2 (exponential decay rate for second moment)
    pub beta2: f64,
    /// Epsilon (small constant for numerical stability)
    pub epsilon: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Energy convergence tolerance
    pub energy_tolerance: f64,
    /// Gradient convergence tolerance
    pub gradient_tolerance: f64,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            max_iterations: 1000,
            energy_tolerance: 1e-6,
            gradient_tolerance: 1e-6,
        }
    }
}

/// Adam optimizer for quantum circuits
///
/// Adaptive Moment Estimation (Adam) is an efficient stochastic optimization
/// algorithm that computes adaptive learning rates for each parameter.
pub struct AdamOptimizer<F>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    circuit_builder: F,
    config: AdamConfig,
    history: Vec<OptimizationStep>,
    // First moment estimate
    m: Vec<f64>,
    // Second moment estimate
    v: Vec<f64>,
    // Time step
    t: usize,
}

impl<F> AdamOptimizer<F>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    /// Create a new Adam optimizer
    pub fn new(circuit_builder: F, config: AdamConfig) -> Self {
        Self {
            circuit_builder,
            config,
            history: Vec::new(),
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    /// Run Adam optimization
    pub fn optimize(
        &mut self,
        simulator: &crate::Simulator,
        observable: &PauliObservable,
        initial_params: &[f64],
    ) -> crate::error::Result<OptimizationResult> {
        let start_time = Instant::now();
        let mut params = initial_params.to_vec();
        let num_params = params.len();

        // Initialize moment estimates
        self.m = vec![0.0; num_params];
        self.v = vec![0.0; num_params];
        self.t = 0;

        let mut prev_energy = f64::INFINITY;

        for iteration in 0..self.config.max_iterations {
            let step_start = Instant::now();
            self.t += 1;

            // Compute energy and gradient
            let circuit = (self.circuit_builder)(&params);
            let energy = compute_expectation(simulator, &circuit, observable)?;
            let grad_result = compute_gradient_auto(
                simulator,
                &self.circuit_builder,
                observable,
                &params,
            )?;

            let gradient = grad_result.gradients.clone();
            let gradient_norm = grad_result.norm();
            let energy_change = (energy - prev_energy).abs();

            // Check convergence
            let status = if iteration >= self.config.max_iterations {
                ConvergenceStatus::MaxIterations
            } else if energy_change < self.config.energy_tolerance
                && gradient_norm < self.config.gradient_tolerance
            {
                ConvergenceStatus::FullyConverged
            } else if energy_change < self.config.energy_tolerance {
                ConvergenceStatus::EnergyConverged
            } else if gradient_norm < self.config.gradient_tolerance {
                ConvergenceStatus::GradientConverged
            } else {
                ConvergenceStatus::NotConverged
            };

            // Record step
            let step = OptimizationStep {
                iteration,
                parameters: params.clone(),
                energy,
                gradient: gradient.clone(),
                gradient_norm,
                energy_change,
                step_time: step_start.elapsed(),
                status,
            };
            self.history.push(step);

            if status != ConvergenceStatus::NotConverged {
                return Ok(OptimizationResult {
                    parameters: params,
                    energy,
                    gradient,
                    status,
                    num_iterations: iteration + 1,
                    total_time: start_time.elapsed(),
                    history: self.history.clone(),
                });
            }

            // Adam update
            for i in 0..num_params {
                let g = gradient[i];

                // Update biased first moment estimate
                self.m[i] = self.config.beta1 * self.m[i] + (1.0 - self.config.beta1) * g;

                // Update biased second moment estimate
                self.v[i] = self.config.beta2 * self.v[i] + (1.0 - self.config.beta2) * g * g;

                // Bias-corrected first moment
                let m_hat = self.m[i] / (1.0 - self.config.beta1.powi(self.t as i32));

                // Bias-corrected second moment
                let v_hat = self.v[i] / (1.0 - self.config.beta2.powi(self.t as i32));

                // Update parameter
                params[i] -= self.config.learning_rate * m_hat / (v_hat.sqrt() + self.config.epsilon);
            }

            prev_energy = energy;
        }

        // Max iterations reached
        let final_circuit = (self.circuit_builder)(&params);
        let final_energy = compute_expectation(simulator, &final_circuit, observable)?;
        let final_grad = compute_gradient_auto(
            simulator,
            &self.circuit_builder,
            observable,
            &params,
        )?;

        Ok(OptimizationResult {
            parameters: params,
            energy: final_energy,
            gradient: final_grad.gradients,
            status: ConvergenceStatus::MaxIterations,
            num_iterations: self.config.max_iterations,
            total_time: start_time.elapsed(),
            history: self.history.clone(),
        })
    }

    /// Get optimization history
    pub fn history(&self) -> &[OptimizationStep] {
        &self.history
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.history.clear();
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }
}

/// Momentum optimizer configuration
#[derive(Debug, Clone)]
pub struct MomentumConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum coefficient (typically 0.9)
    pub momentum: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Energy convergence tolerance
    pub energy_tolerance: f64,
    /// Gradient convergence tolerance
    pub gradient_tolerance: f64,
}

impl Default for MomentumConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            max_iterations: 1000,
            energy_tolerance: 1e-6,
            gradient_tolerance: 1e-6,
        }
    }
}

/// Momentum-based optimizer for quantum circuits
///
/// Gradient descent with momentum helps accelerate convergence by
/// accumulating a velocity vector in directions of persistent reduction.
pub struct MomentumOptimizer<F>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    circuit_builder: F,
    config: MomentumConfig,
    history: Vec<OptimizationStep>,
    // Velocity vector
    velocity: Vec<f64>,
}

impl<F> MomentumOptimizer<F>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    /// Create a new momentum optimizer
    pub fn new(circuit_builder: F, config: MomentumConfig) -> Self {
        Self {
            circuit_builder,
            config,
            history: Vec::new(),
            velocity: Vec::new(),
        }
    }

    /// Run momentum-based optimization
    pub fn optimize(
        &mut self,
        simulator: &crate::Simulator,
        observable: &PauliObservable,
        initial_params: &[f64],
    ) -> crate::error::Result<OptimizationResult> {
        let start_time = Instant::now();
        let mut params = initial_params.to_vec();
        let num_params = params.len();

        // Initialize velocity
        self.velocity = vec![0.0; num_params];

        let mut prev_energy = f64::INFINITY;

        for iteration in 0..self.config.max_iterations {
            let step_start = Instant::now();

            // Compute energy and gradient
            let circuit = (self.circuit_builder)(&params);
            let energy = compute_expectation(simulator, &circuit, observable)?;
            let grad_result = compute_gradient_auto(
                simulator,
                &self.circuit_builder,
                observable,
                &params,
            )?;

            let gradient = grad_result.gradients.clone();
            let gradient_norm = grad_result.norm();
            let energy_change = (energy - prev_energy).abs();

            // Check convergence
            let status = if iteration >= self.config.max_iterations {
                ConvergenceStatus::MaxIterations
            } else if energy_change < self.config.energy_tolerance
                && gradient_norm < self.config.gradient_tolerance
            {
                ConvergenceStatus::FullyConverged
            } else if energy_change < self.config.energy_tolerance {
                ConvergenceStatus::EnergyConverged
            } else if gradient_norm < self.config.gradient_tolerance {
                ConvergenceStatus::GradientConverged
            } else {
                ConvergenceStatus::NotConverged
            };

            // Record step
            let step = OptimizationStep {
                iteration,
                parameters: params.clone(),
                energy,
                gradient: gradient.clone(),
                gradient_norm,
                energy_change,
                step_time: step_start.elapsed(),
                status,
            };
            self.history.push(step);

            if status != ConvergenceStatus::NotConverged {
                return Ok(OptimizationResult {
                    parameters: params,
                    energy,
                    gradient,
                    status,
                    num_iterations: iteration + 1,
                    total_time: start_time.elapsed(),
                    history: self.history.clone(),
                });
            }

            // Momentum update
            for i in 0..num_params {
                // Update velocity: v = momentum * v + learning_rate * gradient
                self.velocity[i] = self.config.momentum * self.velocity[i]
                    + self.config.learning_rate * gradient[i];

                // Update parameter: theta = theta - velocity
                params[i] -= self.velocity[i];
            }

            prev_energy = energy;
        }

        // Max iterations reached
        let final_circuit = (self.circuit_builder)(&params);
        let final_energy = compute_expectation(simulator, &final_circuit, observable)?;
        let final_grad = compute_gradient_auto(
            simulator,
            &self.circuit_builder,
            observable,
            &params,
        )?;

        Ok(OptimizationResult {
            parameters: params,
            energy: final_energy,
            gradient: final_grad.gradients,
            status: ConvergenceStatus::MaxIterations,
            num_iterations: self.config.max_iterations,
            total_time: start_time.elapsed(),
            history: self.history.clone(),
        })
    }

    /// Get optimization history
    pub fn history(&self) -> &[OptimizationStep] {
        &self.history
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.history.clear();
        self.velocity.clear();
    }
}
