//! Gradient computation for variational quantum algorithms
//!
//! This module provides efficient gradient computation methods for parametric
//! quantum circuits, including:
//! - Parameter shift rule (exact gradients for quantum circuits)
//! - Finite differences (fallback method)
//! - Batched evaluation (parallel gradient computation)

pub mod autodiff;
pub mod batch;
pub mod batch_advanced;
pub mod classical_optimizers;
pub mod convergence;
pub mod finite_difference;
pub mod parameter_shift;
pub mod vqe_qaoa;

pub use autodiff::{gradient_forward, Dual, HybridAD};
pub use batch::evaluate_batch_expectation;
pub use batch_advanced::{
    latin_hypercube_sampling, line_search, verify_gradients, AdaptiveBatchEvaluator, BatchConfig,
    ImportanceSampler,
};
pub use classical_optimizers::{
    LBFGSConfig, LBFGSOptimizer, NelderMeadConfig, NelderMeadOptimizer,
};
pub use convergence::{
    energy_logger, progress_callback, target_energy_callback, BestTracker, ConvergenceMonitor,
    ConvergenceReport, MonitorConfig, StepMetrics, StoppingCriterion, TrackedOptimizationResult,
};
pub use finite_difference::{
    compute_gradient_finite_difference, FiniteDifferenceConfig, FiniteDifferenceMethod,
};
pub use parameter_shift::{compute_gradient_parameter_shift, ParameterShiftConfig};
pub use vqe_qaoa::{
    gradient_descent, AdamConfig, AdamOptimizer, ConvergenceStatus, MomentumConfig,
    MomentumOptimizer, OptimizationResult, OptimizationStep, QAOAConfig, QAOAOptimizer, VQEConfig,
    VQEOptimizer,
};

use simq_core::Circuit;
use simq_state::observable::PauliObservable;

/// Configuration for gradient computation
#[derive(Debug, Clone)]
pub struct GradientConfig {
    /// Method to use for gradient computation
    pub method: GradientMethod,
    /// Shift value for parameter shift rule (default: π/2)
    pub shift: f64,
    /// Epsilon for finite differences (default: 1e-7)
    pub epsilon: f64,
    /// Enable parallel gradient computation
    pub parallel: bool,
    /// Cache circuit compilations
    pub cache_circuits: bool,
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            method: GradientMethod::ParameterShift,
            shift: std::f64::consts::FRAC_PI_2,
            epsilon: 1e-7,
            parallel: true,
            cache_circuits: true,
        }
    }
}

/// Gradient computation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientMethod {
    /// Parameter shift rule (exact, quantum-native)
    ParameterShift,
    /// Finite differences (approximate, fallback)
    FiniteDifference,
    /// Automatic selection based on circuit properties
    Auto,
}

/// Result of gradient computation
#[derive(Debug, Clone)]
pub struct GradientResult {
    /// Gradient vector (∂⟨ψ|H|ψ⟩/∂θᵢ for each parameter)
    pub gradients: Vec<f64>,
    /// Number of circuit evaluations performed
    pub num_evaluations: usize,
    /// Computation time
    pub computation_time: std::time::Duration,
    /// Method used
    pub method_used: GradientMethod,
}

impl GradientResult {
    /// Get the gradient vector
    pub fn as_slice(&self) -> &[f64] {
        &self.gradients
    }

    /// Get the number of parameters
    pub fn len(&self) -> usize {
        self.gradients.len()
    }

    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        self.gradients.is_empty()
    }

    /// Get the L2 norm of the gradient
    pub fn norm(&self) -> f64 {
        self.gradients.iter().map(|g| g * g).sum::<f64>().sqrt()
    }
}

/// Compute gradients with automatic method selection and fallback
///
/// This is the recommended high-level interface for gradient computation.
/// It will try parameter shift rule first, and automatically fall back to
/// finite differences if parameter shift fails.
///
/// # Arguments
///
/// * `simulator` - The quantum simulator
/// * `circuit_builder` - Function that builds a circuit from parameters
/// * `observable` - Observable to measure
/// * `params` - Current parameter values
/// * `config` - Gradient computation configuration
///
/// # Returns
///
/// Result containing the gradient vector and metadata
///
/// # Example
///
/// ```ignore
/// use simq_sim::gradient::{compute_gradient, GradientConfig, GradientMethod};
///
/// let config = GradientConfig {
///     method: GradientMethod::Auto,
///     ..Default::default()
/// };
///
/// let result = compute_gradient(
///     &simulator,
///     |params| build_vqe_circuit(params),
///     &hamiltonian,
///     &params,
///     &config,
/// )?;
///
/// println!("Gradient: {:?}", result.gradients);
/// println!("Method used: {:?}", result.method_used);
/// ```
pub fn compute_gradient<F>(
    simulator: &crate::Simulator,
    circuit_builder: F,
    observable: &PauliObservable,
    params: &[f64],
    config: &GradientConfig,
) -> crate::error::Result<GradientResult>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    match config.method {
        GradientMethod::ParameterShift => {
            // Use parameter shift rule
            let ps_config = parameter_shift::ParameterShiftConfig {
                shift: config.shift,
                parallel: config.parallel,
                batch_size: 0, // Auto
            };
            compute_gradient_parameter_shift(
                simulator,
                circuit_builder,
                observable,
                params,
                &ps_config,
            )
        },

        GradientMethod::FiniteDifference => {
            // Use finite differences
            let fd_config = finite_difference::FiniteDifferenceConfig {
                method: finite_difference::FiniteDifferenceMethod::Central,
                epsilon: config.epsilon,
                parallel: config.parallel,
            };
            compute_gradient_finite_difference(
                simulator,
                circuit_builder,
                observable,
                params,
                &fd_config,
            )
        },

        GradientMethod::Auto => {
            // Try parameter shift first, fall back to finite differences
            let ps_config = parameter_shift::ParameterShiftConfig {
                shift: config.shift,
                parallel: config.parallel,
                batch_size: 0, // Auto
            };

            match compute_gradient_parameter_shift(
                simulator,
                &circuit_builder,
                observable,
                params,
                &ps_config,
            ) {
                Ok(result) => Ok(result),
                Err(_) => {
                    // Parameter shift failed, fall back to finite differences
                    let fd_config = finite_difference::FiniteDifferenceConfig {
                        method: finite_difference::FiniteDifferenceMethod::Central,
                        epsilon: config.epsilon,
                        parallel: config.parallel,
                    };
                    compute_gradient_finite_difference(
                        simulator,
                        circuit_builder,
                        observable,
                        params,
                        &fd_config,
                    )
                },
            }
        },
    }
}

/// Compute gradients with automatic fallback (simplified interface)
///
/// Uses default configuration with automatic method selection.
/// This is the simplest way to compute gradients.
///
/// # Arguments
///
/// * `simulator` - The quantum simulator
/// * `circuit_builder` - Function that builds a circuit from parameters
/// * `observable` - Observable to measure
/// * `params` - Current parameter values
///
/// # Returns
///
/// Result containing the gradient vector and metadata
///
/// # Example
///
/// ```ignore
/// let result = compute_gradient_auto(
///     &simulator,
///     |params| build_circuit(params),
///     &observable,
///     &params,
/// )?;
/// ```
pub fn compute_gradient_auto<F>(
    simulator: &crate::Simulator,
    circuit_builder: F,
    observable: &PauliObservable,
    params: &[f64],
) -> crate::error::Result<GradientResult>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    let config = GradientConfig {
        method: GradientMethod::Auto,
        ..Default::default()
    };
    compute_gradient(simulator, circuit_builder, observable, params, &config)
}
