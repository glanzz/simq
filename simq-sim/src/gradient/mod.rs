//! Gradient computation for variational quantum algorithms
//!
//! This module provides efficient gradient computation methods for parametric
//! quantum circuits, including:
//! - Parameter shift rule (exact gradients for quantum circuits)
//! - Finite differences (fallback method)
//! - Batched evaluation (parallel gradient computation)

pub mod parameter_shift;
pub mod finite_difference;
pub mod batch;

pub use parameter_shift::{compute_gradient_parameter_shift, ParameterShiftConfig};
pub use finite_difference::compute_gradient_finite_difference;
pub use batch::evaluate_batch_expectation;

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
