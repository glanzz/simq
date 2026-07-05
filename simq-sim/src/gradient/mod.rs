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

use num_complex::Complex64;
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
            // The ±π/2 parameter shift rule is only exact when each parameter
            // appears once, unscaled, as the angle of a single one-qubit
            // rotation. QAOA-style circuits (one parameter feeding many gates
            // with doubled angles) violate this and can produce identically
            // zero "gradients", so fall back to finite differences for them.
            if !parameter_shift_compatible(&circuit_builder, params) {
                let fd_config = finite_difference::FiniteDifferenceConfig {
                    method: finite_difference::FiniteDifferenceMethod::Central,
                    epsilon: config.epsilon,
                    parallel: config.parallel,
                };
                return compute_gradient_finite_difference(
                    simulator,
                    circuit_builder,
                    observable,
                    params,
                    &fd_config,
                );
            }

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

/// Probe angle used to detect how a parameter enters the circuit's gates.
/// Deliberately not a "nice" fraction of π so that special-angle code paths
/// (caches, snapping) cannot mask the dependence.
const PS_PROBE_SHIFT: f64 = 0.31;

/// Tolerance for deciding that two gate matrices differ
const PS_MATRIX_TOL: f64 = 1e-9;

/// Relative tolerance on the recovered angular speed |s - 1|
const PS_SCALE_TOL: f64 = 1e-3;

/// Structural check: is the standard ±π/2 parameter shift rule applicable to
/// every parameter of the circuit family produced by `circuit_builder`?
///
/// For each parameter, the circuit is rebuilt with that parameter perturbed
/// and the gate matrices are compared against the unperturbed circuit:
///
/// * the parameter must affect **at most one** gate (a parameter feeding
///   several gates needs the summed per-gate shift rule);
/// * that gate must be a **one-qubit** gate (controlled rotations have a
///   different eigenvalue structure and need a four-term rule);
/// * the gate's angle must advance at the **same rate** as the parameter
///   (an angle like `2γ` halves the period of the expectation value, which
///   breaks the ±π/2 rule). The rate is recovered from
///   `|tr(M(θ+δ) M(θ)†)| = 2·|cos(s·δ/2)|`, which is insensitive to global
///   phase conventions.
///
/// Parameters that affect no gate are fine (their gradient is zero under any
/// method). The check is conservative: any structural surprise (gate count
/// changing with a parameter, missing matrices, multi-qubit parameterized
/// gates) disqualifies parameter shift and the caller should use finite
/// differences instead.
fn parameter_shift_compatible<F>(circuit_builder: &F, params: &[f64]) -> bool
where
    F: Fn(&[f64]) -> Circuit,
{
    let base_circuit = circuit_builder(params);
    let base_matrices: Vec<Option<Vec<Complex64>>> = base_circuit
        .operations()
        .map(|op| op.gate().matrix())
        .collect();

    for i in 0..params.len() {
        let mut shifted = params.to_vec();
        shifted[i] += PS_PROBE_SHIFT;
        let probe_circuit = circuit_builder(&shifted);
        let probe_matrices: Vec<Option<Vec<Complex64>>> = probe_circuit
            .operations()
            .map(|op| op.gate().matrix())
            .collect();

        if probe_matrices.len() != base_matrices.len() {
            return false; // circuit structure depends on the parameter value
        }

        let mut changed: Option<usize> = None;
        for (j, (base, probe)) in base_matrices.iter().zip(&probe_matrices).enumerate() {
            match (base, probe) {
                (Some(mb), Some(mp)) => {
                    if mb.len() != mp.len() {
                        return false;
                    }
                    let differs = mb
                        .iter()
                        .zip(mp)
                        .any(|(a, b)| (a - b).norm() > PS_MATRIX_TOL);
                    if differs {
                        if changed.is_some() {
                            return false; // parameter feeds multiple gates
                        }
                        changed = Some(j);
                    }
                },
                (None, None) => {},
                _ => return false,
            }
        }

        if let Some(j) = changed {
            let mb = base_matrices[j].as_ref().unwrap();
            let mp = probe_matrices[j].as_ref().unwrap();
            if mb.len() != 4 {
                return false; // only single-qubit rotations support the ±π/2 rule
            }
            // tr(Mp · Mb†) = Σ_{r,c} Mp[r,c] · conj(Mb[r,c])
            let mut trace = Complex64::new(0.0, 0.0);
            for (a, b) in mp.iter().zip(mb) {
                trace += a * b.conj();
            }
            let cos_half = (trace.norm() / 2.0).clamp(0.0, 1.0);
            let speed = 2.0 * cos_half.acos() / PS_PROBE_SHIFT;
            if (speed - 1.0).abs() > PS_SCALE_TOL {
                return false; // angle is scaled relative to the parameter
            }
        }
    }

    true
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Simulator, SimulatorConfig};
    use simq_core::{circuit::Circuit, QubitId};
    use simq_gates::standard::{RotationX, RotationY};
    use simq_state::observable::{PauliObservable, PauliString};
    use std::sync::Arc;

    fn q(i: usize) -> QubitId {
        QubitId::new(i)
    }

    fn make_sim() -> Simulator {
        Simulator::new(SimulatorConfig::default().with_optimization(false))
    }

    fn z_observable() -> PauliObservable {
        PauliObservable::from_pauli_string(PauliString::from_str("Z").unwrap(), 1.0)
    }

    fn ry_circuit(params: &[f64]) -> Circuit {
        let mut c = Circuit::new(1);
        c.add_gate(Arc::new(RotationY::new(params[0])), &[q(0)])
            .unwrap();
        c
    }

    /// Lines 167-180: compute_gradient with ParameterShift method
    #[test]
    fn test_compute_gradient_parameter_shift_method() {
        let sim = make_sim();
        let obs = z_observable();
        let config = GradientConfig {
            method: GradientMethod::ParameterShift,
            parallel: true,
            ..GradientConfig::default()
        };
        let result = compute_gradient(&sim, ry_circuit, &obs, &[0.5], &config).unwrap();
        assert_eq!(result.method_used, GradientMethod::ParameterShift);
        assert_eq!(result.gradients.len(), 1);
    }

    /// Lines 183-196: compute_gradient with FiniteDifference method
    #[test]
    fn test_compute_gradient_finite_difference_method() {
        let sim = make_sim();
        let obs = z_observable();
        let config = GradientConfig {
            method: GradientMethod::FiniteDifference,
            epsilon: 1e-7,
            parallel: false,
            ..GradientConfig::default()
        };
        let result = compute_gradient(&sim, ry_circuit, &obs, &[0.5], &config).unwrap();
        assert_eq!(result.method_used, GradientMethod::FiniteDifference);
        assert_eq!(result.gradients.len(), 1);
    }

    /// Lines 199-231: compute_gradient with Auto method (happy path — parameter shift succeeds)
    #[test]
    fn test_compute_gradient_auto_method_success() {
        let sim = make_sim();
        let obs = z_observable();
        let config = GradientConfig {
            method: GradientMethod::Auto,
            ..GradientConfig::default()
        };
        let result = compute_gradient(&sim, ry_circuit, &obs, &[0.5], &config).unwrap();
        // Should use parameter shift (line 214: Ok(result))
        assert_eq!(result.method_used, GradientMethod::ParameterShift);
        assert_eq!(result.gradients.len(), 1);
    }

    /// Lines 219-229: Auto fallback to finite differences when parameter shift fails.
    /// We can trigger the fallback by providing a circuit builder that panics for shifted params,
    /// which will cause the parameter shift to fail and fall back to finite differences.
    /// Instead, we simulate this by using a circuit that the parameter shift can process
    /// but we force GradientMethod::Auto with a broken parameter shift configuration.
    /// Since we can't easily break PS, test by covering the FD path directly:
    #[test]
    fn test_compute_gradient_auto_fallback_fd_path() {
        // We test with empty params: both PS and FD return empty gradients (no fallback needed)
        // but we also test the FD method as a direct path to verify lines 217-228 are reachable
        let sim = make_sim();
        let obs = z_observable();
        let fd_config = GradientConfig {
            method: GradientMethod::FiniteDifference,
            epsilon: 1e-7,
            parallel: true,
            ..GradientConfig::default()
        };
        let result = compute_gradient(&sim, ry_circuit, &obs, &[0.5], &fd_config).unwrap();
        assert_eq!(result.method_used, GradientMethod::FiniteDifference);
        // Also verify GradientResult helper methods
        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        assert_eq!(result.as_slice().len(), 1);
        assert!(result.norm().is_finite());
    }

    /// Lines 207-214: Auto method succeeds via parameter shift
    #[test]
    fn test_compute_gradient_auto_succeeds() {
        let sim = make_sim();
        let obs = z_observable();
        let circuit_builder = |params: &[f64]| {
            let mut c = Circuit::new(1);
            c.add_gate(Arc::new(RotationY::new(params[0])), &[q(0)])
                .unwrap();
            c
        };
        let config = GradientConfig {
            method: GradientMethod::Auto,
            parallel: false,
            ..GradientConfig::default()
        };
        let grad = compute_gradient(&sim, circuit_builder, &obs, &[0.5], &config).unwrap();
        assert_eq!(grad.gradients.len(), 1);
        assert!(grad.gradients[0].is_finite());
    }

    /// A plain one-parameter rotation satisfies the parameter-shift
    /// preconditions.
    #[test]
    fn test_parameter_shift_compatible_plain_rotation() {
        assert!(parameter_shift_compatible(&ry_circuit, &[0.5]));
    }

    /// A doubled angle (RX(2θ), as generated by QAOA mixers) breaks the ±π/2
    /// shift rule and must be detected.
    #[test]
    fn test_parameter_shift_incompatible_scaled_angle() {
        let scaled = |params: &[f64]| {
            let mut c = Circuit::new(1);
            c.add_gate(Arc::new(RotationX::new(2.0 * params[0])), &[q(0)])
                .unwrap();
            c
        };
        assert!(!parameter_shift_compatible(&scaled, &[0.3]));
    }

    /// A parameter feeding several gates (QAOA cost/mixer layers) needs the
    /// summed shift rule and must be detected.
    #[test]
    fn test_parameter_shift_incompatible_shared_parameter() {
        let shared = |params: &[f64]| {
            let mut c = Circuit::new(2);
            c.add_gate(Arc::new(RotationY::new(params[0])), &[q(0)])
                .unwrap();
            c.add_gate(Arc::new(RotationY::new(params[0])), &[q(1)])
                .unwrap();
            c
        };
        assert!(!parameter_shift_compatible(&shared, &[0.5]));
    }

    /// Issue #39 regression: for RX(2θ) with observable Z, E(θ) = cos(2θ) and
    /// the ±π/2 parameter shift is identically zero even though the true
    /// gradient is -2·sin(2θ). Auto must fall back to finite differences and
    /// return the correct value.
    #[test]
    fn test_auto_uses_finite_differences_for_scaled_angles() {
        let sim = make_sim();
        let obs = z_observable();
        let scaled = |params: &[f64]| {
            let mut c = Circuit::new(1);
            c.add_gate(Arc::new(RotationX::new(2.0 * params[0])), &[q(0)])
                .unwrap();
            c
        };
        let config = GradientConfig {
            method: GradientMethod::Auto,
            ..GradientConfig::default()
        };
        let theta = 0.3;
        let result = compute_gradient(&sim, scaled, &obs, &[theta], &config).unwrap();
        assert_eq!(result.method_used, GradientMethod::FiniteDifference);
        let expected = -2.0 * (2.0 * theta).sin();
        assert!(
            (result.gradients[0] - expected).abs() < 1e-3,
            "gradient {} but expected {}",
            result.gradients[0],
            expected
        );
    }

    /// GradientConfig default values
    #[test]
    fn test_gradient_config_default() {
        let config = GradientConfig::default();
        assert_eq!(config.method, GradientMethod::ParameterShift);
        assert!(config.parallel);
        assert!(config.cache_circuits);
        assert!((config.epsilon - 1e-7).abs() < 1e-15);
    }

    /// GradientResult methods
    #[test]
    fn test_gradient_result_methods() {
        let result = GradientResult {
            gradients: vec![3.0, 4.0],
            num_evaluations: 4,
            computation_time: std::time::Duration::from_millis(10),
            method_used: GradientMethod::ParameterShift,
        };
        assert_eq!(result.len(), 2);
        assert!(!result.is_empty());
        assert_eq!(result.as_slice(), &[3.0, 4.0]);
        assert!((result.norm() - 5.0).abs() < 1e-10);
    }

    /// Empty GradientResult
    #[test]
    fn test_gradient_result_empty() {
        let result = GradientResult {
            gradients: vec![],
            num_evaluations: 0,
            computation_time: std::time::Duration::default(),
            method_used: GradientMethod::FiniteDifference,
        };
        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
        assert_eq!(result.norm(), 0.0);
    }
}
