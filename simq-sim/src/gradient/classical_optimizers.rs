//! Classical Optimization Algorithms for VQE/QAOA
//!
//! This module provides wrappers around classical optimization algorithms
//! specifically designed for variational quantum algorithms. These optimizers
//! are gradient-free or use numerical gradients, making them ideal for noisy
//! quantum hardware or when analytical gradients are unavailable.
//!
//! # Available Optimizers
//!
//! - **L-BFGS**: Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm
//!   - Quasi-Newton method that approximates the BFGS algorithm using limited memory
//!   - Excellent for smooth, differentiable objective functions
//!   - Uses numerical gradients via finite differences
//!
//! - **Nelder-Mead**: Simplex-based gradient-free optimization
//!   - Also known as the downhill simplex method or amoeba method
//!   - Doesn't require gradients, making it robust to noise
//!   - Good for non-smooth or noisy objective functions (ideal for NISQ devices)
//!   - Similar to COBYLA but simpler to implement
//!
//! # Example
//!
//! ```ignore
//! use simq_sim::gradient::classical::{LBFGSOptimizer, LBFGSConfig};
//!
//! let config = LBFGSConfig {
//!     max_iterations: 100,
//!     tolerance: 1e-6,
//!     ..Default::default()
//! };
//!
//! let mut optimizer = LBFGSOptimizer::new(|params| build_circuit(params), config);
//! let result = optimizer.optimize(&simulator, &observable, &initial_params)?;
//! ```

use simq_core::Circuit;
use simq_state::observable::PauliObservable;
use simq_state::AdaptiveState;
use std::time::{Duration, Instant};

use super::{ConvergenceStatus, OptimizationResult, OptimizationStep};

/// Helper function to compute expectation value
fn compute_expectation(
    simulator: &crate::Simulator,
    circuit: &Circuit,
    observable: &PauliObservable,
) -> crate::error::Result<f64> {
    let result = simulator.run(circuit)?;

    let expectation = match &result.state {
        AdaptiveState::Dense(dense) => observable.expectation_value(dense)?,
        AdaptiveState::Sparse { state: sparse, .. } => {
            use simq_state::DenseState;
            let dense = DenseState::from_sparse(sparse);
            observable.expectation_value(&dense)?
        }
    };

    Ok(expectation)
}

// ============================================================================
// L-BFGS Optimizer
// ============================================================================

/// Configuration for L-BFGS optimizer
#[derive(Debug, Clone)]
pub struct LBFGSConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,

    /// Convergence tolerance for cost function
    pub tolerance: f64,

    /// Number of corrections to approximate the inverse Hessian (typically 5-20)
    pub memory_size: usize,

    /// Step size for numerical gradient computation
    pub gradient_epsilon: f64,

    /// Line search tolerance
    pub line_search_tolerance: f64,

    /// Maximum number of line search iterations
    pub max_line_search_iterations: usize,
}

impl Default for LBFGSConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            memory_size: 10,
            gradient_epsilon: 1e-7,
            line_search_tolerance: 1e-4,
            max_line_search_iterations: 20,
        }
    }
}

/// L-BFGS optimizer for quantum circuits
///
/// Limited-memory BFGS is a quasi-Newton optimization algorithm that approximates
/// the inverse Hessian using limited memory. It's particularly effective for
/// smooth optimization problems with many parameters.
///
/// # Algorithm
///
/// 1. Compute numerical gradient via finite differences
/// 2. Use L-BFGS update to approximate inverse Hessian
/// 3. Compute search direction
/// 4. Perform line search to find optimal step size
/// 5. Update parameters
///
/// # When to Use
///
/// - Smooth objective functions (noiseless simulators)
/// - Many parameters (>10)
/// - When gradients can be computed efficiently
/// - When you need fast convergence
pub struct LBFGSOptimizer<F>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    circuit_builder: F,
    config: LBFGSConfig,
    history: Vec<OptimizationStep>,
}

impl<F> LBFGSOptimizer<F>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    /// Create a new L-BFGS optimizer
    pub fn new(circuit_builder: F, config: LBFGSConfig) -> Self {
        Self {
            circuit_builder,
            config,
            history: Vec::new(),
        }
    }

    /// Run L-BFGS optimization
    pub fn optimize(
        &mut self,
        simulator: &crate::Simulator,
        observable: &PauliObservable,
        initial_params: &[f64],
    ) -> crate::error::Result<OptimizationResult> {
        let start_time = Instant::now();
        let mut params = initial_params.to_vec();
        let num_params = params.len();

        // Storage for L-BFGS history (s_k and y_k vectors)
        let mut s_history: Vec<Vec<f64>> = Vec::new();
        let mut y_history: Vec<Vec<f64>> = Vec::new();
        let mut rho_history: Vec<f64> = Vec::new();

        let mut prev_energy = f64::INFINITY;
        let mut prev_gradient = vec![0.0; num_params];

        for iteration in 0..self.config.max_iterations {
            let step_start = Instant::now();

            // Compute current energy
            let circuit = (self.circuit_builder)(&params);
            let energy = compute_expectation(simulator, &circuit, observable)?;

            // Compute gradient via finite differences
            let gradient = self.compute_numerical_gradient(simulator, observable, &params)?;
            let gradient_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

            // Check convergence
            let energy_change = (energy - prev_energy).abs();
            let status = self.check_convergence(iteration, energy_change, gradient_norm);

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

            // Compute search direction using L-BFGS two-loop recursion
            let search_direction = if iteration == 0 {
                // First iteration: use steepest descent
                gradient.iter().map(|&g| -g).collect()
            } else {
                self.compute_lbfgs_direction(
                    &gradient,
                    &s_history,
                    &y_history,
                    &rho_history,
                )
            };

            // Line search to find optimal step size
            let step_size = self.line_search(
                simulator,
                observable,
                &params,
                &search_direction,
                energy,
                &gradient,
            )?;

            // Update parameters
            let mut new_params = params.clone();
            for i in 0..num_params {
                new_params[i] += step_size * search_direction[i];
            }

            // Update L-BFGS history
            if iteration > 0 {
                let s_k: Vec<f64> = new_params
                    .iter()
                    .zip(params.iter())
                    .map(|(new, old)| new - old)
                    .collect();

                let y_k: Vec<f64> = gradient
                    .iter()
                    .zip(prev_gradient.iter())
                    .map(|(new, old)| new - old)
                    .collect();

                let rho_k = 1.0 / s_k.iter().zip(y_k.iter()).map(|(s, y)| s * y).sum::<f64>();

                // Add to history (maintain fixed size)
                s_history.push(s_k);
                y_history.push(y_k);
                rho_history.push(rho_k);

                if s_history.len() > self.config.memory_size {
                    s_history.remove(0);
                    y_history.remove(0);
                    rho_history.remove(0);
                }
            }

            params = new_params;
            prev_energy = energy;
            prev_gradient = gradient;
        }

        // Max iterations reached
        let final_circuit = (self.circuit_builder)(&params);
        let final_energy = compute_expectation(simulator, &final_circuit, observable)?;
        let final_gradient = self.compute_numerical_gradient(simulator, observable, &params)?;

        Ok(OptimizationResult {
            parameters: params,
            energy: final_energy,
            gradient: final_gradient,
            status: ConvergenceStatus::MaxIterations,
            num_iterations: self.config.max_iterations,
            total_time: start_time.elapsed(),
            history: self.history.clone(),
        })
    }

    /// Compute numerical gradient via central finite differences
    fn compute_numerical_gradient(
        &self,
        simulator: &crate::Simulator,
        observable: &PauliObservable,
        params: &[f64],
    ) -> crate::error::Result<Vec<f64>> {
        let eps = self.config.gradient_epsilon;
        let num_params = params.len();
        let mut gradient = vec![0.0; num_params];

        for i in 0..num_params {
            // Forward perturbation
            let mut params_plus = params.to_vec();
            params_plus[i] += eps;
            let circuit_plus = (self.circuit_builder)(&params_plus);
            let energy_plus = compute_expectation(simulator, &circuit_plus, observable)?;

            // Backward perturbation
            let mut params_minus = params.to_vec();
            params_minus[i] -= eps;
            let circuit_minus = (self.circuit_builder)(&params_minus);
            let energy_minus = compute_expectation(simulator, &circuit_minus, observable)?;

            // Central difference
            gradient[i] = (energy_plus - energy_minus) / (2.0 * eps);
        }

        Ok(gradient)
    }

    /// Compute L-BFGS search direction using two-loop recursion
    fn compute_lbfgs_direction(
        &self,
        gradient: &[f64],
        s_history: &[Vec<f64>],
        y_history: &[Vec<f64>],
        rho_history: &[f64],
    ) -> Vec<f64> {
        let m = s_history.len();
        let mut q = gradient.to_vec();
        let mut alpha = vec![0.0; m];

        // First loop (backward)
        for i in (0..m).rev() {
            alpha[i] = rho_history[i]
                * s_history[i]
                    .iter()
                    .zip(q.iter())
                    .map(|(s, q)| s * q)
                    .sum::<f64>();

            for j in 0..q.len() {
                q[j] -= alpha[i] * y_history[i][j];
            }
        }

        // Scale by approximate Hessian diagonal (H_0)
        let gamma = if m > 0 {
            let s_last = &s_history[m - 1];
            let y_last = &y_history[m - 1];
            let sy: f64 = s_last.iter().zip(y_last.iter()).map(|(s, y)| s * y).sum();
            let yy: f64 = y_last.iter().map(|y| y * y).sum();
            sy / yy
        } else {
            1.0
        };

        let mut r: Vec<f64> = q.iter().map(|&qi| gamma * qi).collect();

        // Second loop (forward)
        for i in 0..m {
            let beta = rho_history[i]
                * y_history[i]
                    .iter()
                    .zip(r.iter())
                    .map(|(y, r)| y * r)
                    .sum::<f64>();

            for j in 0..r.len() {
                r[j] += s_history[i][j] * (alpha[i] - beta);
            }
        }

        // Return negative direction (for minimization)
        r.iter().map(|&ri| -ri).collect()
    }

    /// Simple backtracking line search
    fn line_search(
        &self,
        simulator: &crate::Simulator,
        observable: &PauliObservable,
        params: &[f64],
        direction: &[f64],
        current_energy: f64,
        gradient: &[f64],
    ) -> crate::error::Result<f64> {
        let c1 = 1e-4; // Armijo condition constant
        let tau = 0.5; // Backtracking factor
        let mut alpha = 1.0; // Initial step size

        // Compute directional derivative
        let dir_deriv: f64 = gradient
            .iter()
            .zip(direction.iter())
            .map(|(g, d)| g * d)
            .sum();

        for _ in 0..self.config.max_line_search_iterations {
            // Test new parameters
            let new_params: Vec<f64> = params
                .iter()
                .zip(direction.iter())
                .map(|(&p, &d)| p + alpha * d)
                .collect();

            let circuit = (self.circuit_builder)(&new_params);
            let new_energy = compute_expectation(simulator, &circuit, observable)?;

            // Armijo condition
            if new_energy <= current_energy + c1 * alpha * dir_deriv {
                return Ok(alpha);
            }

            // Backtrack
            alpha *= tau;
        }

        // If line search fails, return small step
        Ok(alpha)
    }

    /// Check convergence criteria
    fn check_convergence(
        &self,
        iteration: usize,
        energy_change: f64,
        gradient_norm: f64,
    ) -> ConvergenceStatus {
        if iteration >= self.config.max_iterations {
            ConvergenceStatus::MaxIterations
        } else if energy_change < self.config.tolerance && gradient_norm < self.config.tolerance {
            ConvergenceStatus::FullyConverged
        } else if energy_change < self.config.tolerance {
            ConvergenceStatus::EnergyConverged
        } else if gradient_norm < self.config.tolerance {
            ConvergenceStatus::GradientConverged
        } else {
            ConvergenceStatus::NotConverged
        }
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

// ============================================================================
// Nelder-Mead Optimizer
// ============================================================================

/// Configuration for Nelder-Mead optimizer
#[derive(Debug, Clone)]
pub struct NelderMeadConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,

    /// Convergence tolerance for simplex size
    pub tolerance: f64,

    /// Reflection coefficient (default: 1.0)
    pub alpha: f64,

    /// Expansion coefficient (default: 2.0)
    pub gamma: f64,

    /// Contraction coefficient (default: 0.5)
    pub rho: f64,

    /// Shrink coefficient (default: 0.5)
    pub sigma: f64,
}

impl Default for NelderMeadConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-6,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }
}

/// Nelder-Mead simplex optimizer
///
/// The Nelder-Mead method is a gradient-free optimization algorithm that maintains
/// a simplex of n+1 points in n-dimensional space and iteratively transforms the
/// simplex to find the minimum.
///
/// # Algorithm
///
/// 1. Initialize simplex with n+1 vertices
/// 2. Order vertices by function value
/// 3. Try reflection, expansion, contraction, or shrink operations
/// 4. Update simplex based on which operation succeeds
/// 5. Repeat until convergence
///
/// # When to Use
///
/// - Noisy objective functions (ideal for NISQ devices)
/// - Non-smooth or discontinuous functions
/// - Few parameters (<10-15)
/// - When gradients are unavailable or unreliable
/// - When you need robustness over speed
pub struct NelderMeadOptimizer<F>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    circuit_builder: F,
    config: NelderMeadConfig,
    history: Vec<OptimizationStep>,
}

impl<F> NelderMeadOptimizer<F>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    /// Create a new Nelder-Mead optimizer
    pub fn new(circuit_builder: F, config: NelderMeadConfig) -> Self {
        Self {
            circuit_builder,
            config,
            history: Vec::new(),
        }
    }

    /// Run Nelder-Mead optimization
    pub fn optimize(
        &mut self,
        simulator: &crate::Simulator,
        observable: &PauliObservable,
        initial_params: &[f64],
    ) -> crate::error::Result<OptimizationResult> {
        let start_time = Instant::now();
        let n = initial_params.len();

        // Initialize simplex
        let mut simplex = self.initialize_simplex(initial_params);
        let mut energies: Vec<f64> = simplex
            .iter()
            .map(|params| {
                let circuit = (self.circuit_builder)(params);
                compute_expectation(simulator, &circuit, observable).unwrap_or(f64::INFINITY)
            })
            .collect();

        let mut iteration = 0;

        while iteration < self.config.max_iterations {
            let step_start = Instant::now();

            // Sort simplex by energy
            let mut indices: Vec<usize> = (0..simplex.len()).collect();
            indices.sort_by(|&a, &b| energies[a].partial_cmp(&energies[b]).unwrap());

            let best_idx = indices[0];
            let worst_idx = indices[n];
            let second_worst_idx = indices[n - 1];

            let best_energy = energies[best_idx];
            let worst_energy = energies[worst_idx];

            // Check convergence (simplex size)
            let simplex_size = self.compute_simplex_size(&simplex, &indices);
            let energy_change = if iteration > 0 {
                (best_energy - self.history.last().unwrap().energy).abs()
            } else {
                f64::INFINITY
            };

            let status = if iteration >= self.config.max_iterations {
                ConvergenceStatus::MaxIterations
            } else if simplex_size < self.config.tolerance {
                ConvergenceStatus::FullyConverged
            } else {
                ConvergenceStatus::NotConverged
            };

            // Record step (use best point)
            let step = OptimizationStep {
                iteration,
                parameters: simplex[best_idx].clone(),
                energy: best_energy,
                gradient: vec![0.0; n], // Gradient-free method
                gradient_norm: 0.0,
                energy_change,
                step_time: step_start.elapsed(),
                status,
            };
            self.history.push(step);

            if status != ConvergenceStatus::NotConverged {
                return Ok(OptimizationResult {
                    parameters: simplex[best_idx].clone(),
                    energy: best_energy,
                    gradient: vec![0.0; n],
                    status,
                    num_iterations: iteration + 1,
                    total_time: start_time.elapsed(),
                    history: self.history.clone(),
                });
            }

            // Compute centroid of all points except worst
            let centroid = self.compute_centroid(&simplex, &indices[..n]);

            // Try reflection
            let reflected = self.reflect(&simplex[worst_idx], &centroid);
            let reflected_circuit = (self.circuit_builder)(&reflected);
            let reflected_energy = compute_expectation(simulator, &reflected_circuit, observable)?;

            if reflected_energy < energies[best_idx] {
                // Try expansion
                let expanded = self.expand(&reflected, &centroid);
                let expanded_circuit = (self.circuit_builder)(&expanded);
                let expanded_energy = compute_expectation(simulator, &expanded_circuit, observable)?;

                if expanded_energy < reflected_energy {
                    simplex[worst_idx] = expanded;
                    energies[worst_idx] = expanded_energy;
                } else {
                    simplex[worst_idx] = reflected;
                    energies[worst_idx] = reflected_energy;
                }
            } else if reflected_energy < energies[second_worst_idx] {
                simplex[worst_idx] = reflected;
                energies[worst_idx] = reflected_energy;
            } else {
                // Try contraction
                let contracted = if reflected_energy < worst_energy {
                    self.contract_outside(&reflected, &centroid)
                } else {
                    self.contract_inside(&simplex[worst_idx], &centroid)
                };

                let contracted_circuit = (self.circuit_builder)(&contracted);
                let contracted_energy =
                    compute_expectation(simulator, &contracted_circuit, observable)?;

                if contracted_energy < worst_energy.min(reflected_energy) {
                    simplex[worst_idx] = contracted;
                    energies[worst_idx] = contracted_energy;
                } else {
                    // Shrink simplex toward best point
                    let best_point = simplex[best_idx].clone();
                    self.shrink(&mut simplex, &best_point);
                    for i in 0..simplex.len() {
                        if i != best_idx {
                            let circuit = (self.circuit_builder)(&simplex[i]);
                            energies[i] = compute_expectation(simulator, &circuit, observable)?;
                        }
                    }
                }
            }

            iteration += 1;
        }

        // Return best point
        let best_idx = energies
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        Ok(OptimizationResult {
            parameters: simplex[best_idx].clone(),
            energy: energies[best_idx],
            gradient: vec![0.0; n],
            status: ConvergenceStatus::MaxIterations,
            num_iterations: self.config.max_iterations,
            total_time: start_time.elapsed(),
            history: self.history.clone(),
        })
    }

    /// Initialize simplex using standard basis
    fn initialize_simplex(&self, initial: &[f64]) -> Vec<Vec<f64>> {
        let n = initial.len();
        let mut simplex = vec![initial.to_vec()];

        // Create n additional vertices by perturbing each dimension
        let scale = 0.05; // 5% perturbation
        for i in 0..n {
            let mut vertex = initial.to_vec();
            vertex[i] += if vertex[i].abs() > 1e-10 {
                vertex[i] * scale
            } else {
                scale
            };
            simplex.push(vertex);
        }

        simplex
    }

    /// Compute centroid of given points
    fn compute_centroid(&self, simplex: &[Vec<f64>], indices: &[usize]) -> Vec<f64> {
        let n = simplex[0].len();
        let mut centroid = vec![0.0; n];

        for &idx in indices {
            for i in 0..n {
                centroid[i] += simplex[idx][i];
            }
        }

        for i in 0..n {
            centroid[i] /= indices.len() as f64;
        }

        centroid
    }

    /// Reflection operation
    fn reflect(&self, worst: &[f64], centroid: &[f64]) -> Vec<f64> {
        worst
            .iter()
            .zip(centroid.iter())
            .map(|(&w, &c)| c + self.config.alpha * (c - w))
            .collect()
    }

    /// Expansion operation
    fn expand(&self, reflected: &[f64], centroid: &[f64]) -> Vec<f64> {
        reflected
            .iter()
            .zip(centroid.iter())
            .map(|(&r, &c)| c + self.config.gamma * (r - c))
            .collect()
    }

    /// Outside contraction
    fn contract_outside(&self, reflected: &[f64], centroid: &[f64]) -> Vec<f64> {
        reflected
            .iter()
            .zip(centroid.iter())
            .map(|(&r, &c)| c + self.config.rho * (r - c))
            .collect()
    }

    /// Inside contraction
    fn contract_inside(&self, worst: &[f64], centroid: &[f64]) -> Vec<f64> {
        worst
            .iter()
            .zip(centroid.iter())
            .map(|(&w, &c)| c + self.config.rho * (w - c))
            .collect()
    }

    /// Shrink simplex toward best point
    fn shrink(&self, simplex: &mut [Vec<f64>], best: &[f64]) {
        for vertex in simplex.iter_mut() {
            for i in 0..vertex.len() {
                vertex[i] = best[i] + self.config.sigma * (vertex[i] - best[i]);
            }
        }
    }

    /// Compute simplex size (max distance from centroid)
    fn compute_simplex_size(&self, simplex: &[Vec<f64>], indices: &[usize]) -> f64 {
        let centroid = self.compute_centroid(simplex, indices);
        let mut max_dist = 0.0;

        for &idx in indices {
            let dist: f64 = simplex[idx]
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            max_dist = max_dist.max(dist);
        }

        max_dist
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbfgs_config_default() {
        let config = LBFGSConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.memory_size, 10);
        assert!(config.tolerance > 0.0);
    }

    #[test]
    fn test_nelder_mead_config_default() {
        let config = NelderMeadConfig::default();
        assert_eq!(config.max_iterations, 200);
        assert!(config.alpha > 0.0);
        assert!(config.gamma > 0.0);
    }
}
