//! Parameter shift rule for quantum gradient computation
//!
//! The parameter shift rule provides exact gradients for parametric quantum circuits
//! without requiring classical automatic differentiation.
//!
//! For a parametric gate U(θ) = exp(-iθG/2) where G is a generator with eigenvalues ±1/2:
//!
//! ∂⟨ψ(θ)|H|ψ(θ)⟩/∂θ = [⟨ψ(θ+s)|H|ψ(θ+s)⟩ - ⟨ψ(θ-s)|H|ψ(θ-s)⟩] / (2 sin(s))
//!
//! For s = π/2, this simplifies to:
//! ∂⟨ψ(θ)|H|ψ(θ)⟩/∂θ = [⟨ψ(θ+π/2)|H|ψ(θ+π/2)⟩ - ⟨ψ(θ-π/2)|H|ψ(θ-π/2)⟩] / 2

use rayon::prelude::*;
use std::time::Instant;
use simq_core::Circuit;
use simq_state::AdaptiveState;
use simq_state::observable::PauliObservable;
use crate::Simulator;
use crate::error::{SimulatorError, Result};
use super::{GradientResult, GradientMethod};

/// Configuration for parameter shift rule
#[derive(Debug, Clone)]
pub struct ParameterShiftConfig {
    /// Shift value (default: π/2 for standard gates)
    pub shift: f64,
    /// Enable parallel evaluation of shifted circuits
    pub parallel: bool,
    /// Batch size for parallel evaluation (0 = automatic)
    pub batch_size: usize,
}

impl Default for ParameterShiftConfig {
    fn default() -> Self {
        Self {
            shift: std::f64::consts::FRAC_PI_2,
            parallel: true,
            batch_size: 0, // Auto
        }
    }
}

/// Compute gradient using the parameter shift rule
///
/// # Arguments
///
/// * `simulator` - The quantum simulator
/// * `circuit_builder` - Function that builds a circuit given parameter values
/// * `observable` - The observable to measure (Hamiltonian)
/// * `params` - Current parameter values
/// * `config` - Configuration for parameter shift rule
///
/// # Returns
///
/// A `GradientResult` containing the gradient vector and metadata
///
/// # Example
///
/// ```ignore
/// use simq_sim::gradient::{compute_gradient_parameter_shift, ParameterShiftConfig};
///
/// let params = vec![0.5, 1.2, 0.8];
/// let config = ParameterShiftConfig::default();
///
/// let result = compute_gradient_parameter_shift(
///     &simulator,
///     |p| build_vqe_circuit(p),
///     &hamiltonian,
///     &params,
///     &config,
/// )?;
///
/// println!("Gradient: {:?}", result.gradients);
/// println!("Norm: {}", result.norm());
/// ```
pub fn compute_gradient_parameter_shift<F>(
    simulator: &Simulator,
    circuit_builder: F,
    observable: &PauliObservable,
    params: &[f64],
    config: &ParameterShiftConfig,
) -> Result<GradientResult>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    let start_time = Instant::now();
    let n_params = params.len();

    if n_params == 0 {
        return Ok(GradientResult {
            gradients: vec![],
            num_evaluations: 0,
            computation_time: start_time.elapsed(),
            method_used: GradientMethod::ParameterShift,
        });
    }

    let shift = config.shift;
    let gradients: Vec<f64> = if config.parallel {
        // Parallel computation: evaluate all shifted circuits in parallel
        (0..n_params)
            .into_par_iter()
            .map(|i| {
                compute_single_parameter_gradient(
                    simulator,
                    &circuit_builder,
                    observable,
                    params,
                    i,
                    shift,
                )
            })
            .collect::<Result<Vec<f64>>>()?
    } else {
        // Sequential computation
        (0..n_params)
            .map(|i| {
                compute_single_parameter_gradient(
                    simulator,
                    &circuit_builder,
                    observable,
                    params,
                    i,
                    shift,
                )
            })
            .collect::<Result<Vec<f64>>>()?
    };

    Ok(GradientResult {
        gradients,
        num_evaluations: n_params * 2, // Two evaluations per parameter
        computation_time: start_time.elapsed(),
        method_used: GradientMethod::ParameterShift,
    })
}

/// Compute gradient for a single parameter using parameter shift rule
fn compute_single_parameter_gradient<F>(
    simulator: &Simulator,
    circuit_builder: &F,
    observable: &PauliObservable,
    params: &[f64],
    param_index: usize,
    shift: f64,
) -> Result<f64>
where
    F: Fn(&[f64]) -> Circuit,
{
    // Create shifted parameter vectors
    let mut params_plus = params.to_vec();
    let mut params_minus = params.to_vec();
    params_plus[param_index] += shift;
    params_minus[param_index] -= shift;

    // Build shifted circuits
    let circuit_plus = circuit_builder(&params_plus);
    let circuit_minus = circuit_builder(&params_minus);

    // Simulate and measure expectation values
    let exp_plus = evaluate_expectation(simulator, &circuit_plus, observable)?;
    let exp_minus = evaluate_expectation(simulator, &circuit_minus, observable)?;

    // Compute gradient using parameter shift formula
    // For shift = π/2: grad = (exp_plus - exp_minus) / 2
    let gradient = if (shift - std::f64::consts::FRAC_PI_2).abs() < 1e-10 {
        (exp_plus - exp_minus) / 2.0
    } else {
        // General formula: grad = (exp_plus - exp_minus) / (2 * sin(shift))
        (exp_plus - exp_minus) / (2.0 * shift.sin())
    };

    Ok(gradient)
}

/// Evaluate expectation value ⟨ψ|H|ψ⟩ for a circuit and observable
fn evaluate_expectation(
    simulator: &Simulator,
    circuit: &Circuit,
    observable: &PauliObservable,
) -> Result<f64> {
    // Run simulation
    let result = simulator.run(circuit)?;

    // Compute expectation value
    let expectation = match &result.state {
        AdaptiveState::Dense(dense) => {
            observable.expectation_value_dense(dense.amplitudes())
        }
        AdaptiveState::Sparse { state: sparse, .. } => {
            observable.expectation_value_sparse(
                sparse.amplitudes(),
                circuit.num_qubits(),
            )
        }
    };

    Ok(expectation)
}

/// Batch parameter shift rule for multiple observables
///
/// Computes gradients for all observables simultaneously, which is more
/// efficient than computing them separately.
pub fn batch_parameter_shift<F>(
    simulator: &Simulator,
    circuit_builder: F,
    observables: &[PauliObservable],
    params: &[f64],
    config: &ParameterShiftConfig,
) -> Result<Vec<GradientResult>>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    if config.parallel {
        observables
            .par_iter()
            .map(|obs| compute_gradient_parameter_shift(simulator, &circuit_builder, obs, params, config))
            .collect()
    } else {
        observables
            .iter()
            .map(|obs| compute_gradient_parameter_shift(simulator, &circuit_builder, obs, params, config))
            .collect()
    }
}

/// Two-point parameter shift rule (for gates with eigenvalues ±r/2)
///
/// For more general gates, the parameter shift rule requires more evaluation points.
/// This implements the two-term parameter shift rule for gates with eigenvalues ±r/2.
pub fn parameter_shift_general<F>(
    simulator: &Simulator,
    circuit_builder: F,
    observable: &PauliObservable,
    params: &[f64],
    eigenvalue_scaling: f64, // r value (typically 1.0)
    config: &ParameterShiftConfig,
) -> Result<GradientResult>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    let shift = std::f64::consts::PI / (4.0 * eigenvalue_scaling);

    let mut modified_config = config.clone();
    modified_config.shift = shift;

    compute_gradient_parameter_shift(simulator, circuit_builder, observable, params, &modified_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_core::QubitId;
    use simq_gates::standard::{Hadamard, RotationY};
    use std::sync::Arc;
    use crate::SimulatorConfig;

    #[test]
    fn test_parameter_shift_single_param() {
        let simulator = Simulator::new(SimulatorConfig::default());

        // Simple circuit: H - RY(θ) - H
        let circuit_builder = |params: &[f64]| {
            let mut circuit = Circuit::new(1);
            circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
            circuit.add_gate(Arc::new(RotationY::new(params[0])), &[QubitId::new(0)]).unwrap();
            circuit
        };

        // Observable: Z
        let observable = PauliObservable::from_string("Z", &[0]).unwrap();

        let params = vec![0.5];
        let config = ParameterShiftConfig::default();

        let result = compute_gradient_parameter_shift(
            &simulator,
            circuit_builder,
            &observable,
            &params,
            &config,
        ).unwrap();

        assert_eq!(result.gradients.len(), 1);
        assert_eq!(result.num_evaluations, 2);
        assert!(result.computation_time.as_secs_f64() > 0.0);
    }

    #[test]
    fn test_parameter_shift_multiple_params() {
        let simulator = Simulator::new(SimulatorConfig::default());

        let circuit_builder = |params: &[f64]| {
            let mut circuit = Circuit::new(2);
            circuit.add_gate(Arc::new(RotationY::new(params[0])), &[QubitId::new(0)]).unwrap();
            circuit.add_gate(Arc::new(RotationY::new(params[1])), &[QubitId::new(1)]).unwrap();
            circuit
        };

        let observable = PauliObservable::from_string("ZZ", &[0, 1]).unwrap();

        let params = vec![0.3, 0.7];
        let config = ParameterShiftConfig::default();

        let result = compute_gradient_parameter_shift(
            &simulator,
            circuit_builder,
            &observable,
            &params,
            &config,
        ).unwrap();

        assert_eq!(result.gradients.len(), 2);
        assert_eq!(result.num_evaluations, 4); // 2 params × 2 evaluations
    }

    #[test]
    fn test_gradient_norm() {
        let result = GradientResult {
            gradients: vec![3.0, 4.0],
            num_evaluations: 4,
            computation_time: std::time::Duration::from_millis(100),
            method_used: GradientMethod::ParameterShift,
        };

        assert_eq!(result.norm(), 5.0); // √(3² + 4²) = 5
    }

    #[test]
    fn test_zero_params() {
        let simulator = Simulator::new(SimulatorConfig::default());
        let circuit_builder = |_: &[f64]| Circuit::new(1);
        let observable = PauliObservable::from_string("Z", &[0]).unwrap();

        let result = compute_gradient_parameter_shift(
            &simulator,
            circuit_builder,
            &observable,
            &[],
            &ParameterShiftConfig::default(),
        ).unwrap();

        assert_eq!(result.len(), 0);
        assert_eq!(result.num_evaluations, 0);
    }
}
