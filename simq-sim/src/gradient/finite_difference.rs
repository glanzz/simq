//! Finite difference gradient computation
//!
//! Provides gradient computation using finite differences as a fallback when
//! the parameter shift rule is not applicable.

use super::{GradientMethod, GradientResult};
use crate::error::Result;
use crate::Simulator;
use rayon::prelude::*;
use simq_core::Circuit;
use simq_state::observable::PauliObservable;
use simq_state::AdaptiveState;
use std::time::Instant;

/// Finite difference method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiniteDifferenceMethod {
    /// Forward difference: f'(x) ≈ [f(x+ε) - f(x)] / ε
    Forward,
    /// Central difference: f'(x) ≈ [f(x+ε) - f(x-ε)] / (2ε)
    Central,
    /// Backward difference: f'(x) ≈ [f(x) - f(x-ε)] / ε
    Backward,
}

/// Configuration for finite difference gradient computation
#[derive(Debug, Clone)]
pub struct FiniteDifferenceConfig {
    /// Method to use
    pub method: FiniteDifferenceMethod,
    /// Epsilon value for finite differences
    pub epsilon: f64,
    /// Enable parallel evaluation
    pub parallel: bool,
}

impl Default for FiniteDifferenceConfig {
    fn default() -> Self {
        Self {
            method: FiniteDifferenceMethod::Central,
            epsilon: 1e-7,
            parallel: true,
        }
    }
}

/// Compute gradient using finite differences
///
/// # Arguments
///
/// * `simulator` - The quantum simulator
/// * `circuit_builder` - Function that builds a circuit given parameter values
/// * `observable` - The observable to measure
/// * `params` - Current parameter values
/// * `config` - Finite difference configuration
///
/// # Returns
///
/// A `GradientResult` containing the gradient vector
pub fn compute_gradient_finite_difference<F>(
    simulator: &Simulator,
    circuit_builder: F,
    observable: &PauliObservable,
    params: &[f64],
    config: &FiniteDifferenceConfig,
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
            method_used: GradientMethod::FiniteDifference,
        });
    }

    // Compute base expectation value for forward/backward methods
    let base_expectation = match config.method {
        FiniteDifferenceMethod::Forward | FiniteDifferenceMethod::Backward => {
            let circuit = circuit_builder(params);
            Some(evaluate_expectation(simulator, &circuit, observable)?)
        },
        FiniteDifferenceMethod::Central => None,
    };

    let _num_evals_per_param = match config.method {
        FiniteDifferenceMethod::Forward | FiniteDifferenceMethod::Backward => 1,
        FiniteDifferenceMethod::Central => 2,
    };

    let gradients: Vec<f64> = if config.parallel {
        (0..n_params)
            .into_par_iter()
            .map(|i| {
                compute_single_param_fd(
                    simulator,
                    &circuit_builder,
                    observable,
                    params,
                    i,
                    config.epsilon,
                    config.method,
                    base_expectation,
                )
            })
            .collect::<Result<Vec<f64>>>()?
    } else {
        (0..n_params)
            .map(|i| {
                compute_single_param_fd(
                    simulator,
                    &circuit_builder,
                    observable,
                    params,
                    i,
                    config.epsilon,
                    config.method,
                    base_expectation,
                )
            })
            .collect::<Result<Vec<f64>>>()?
    };

    let total_evaluations = match config.method {
        FiniteDifferenceMethod::Forward | FiniteDifferenceMethod::Backward => {
            1 + n_params // Base + one per parameter
        },
        FiniteDifferenceMethod::Central => n_params * 2, // Two per parameter
    };

    Ok(GradientResult {
        gradients,
        num_evaluations: total_evaluations,
        computation_time: start_time.elapsed(),
        method_used: GradientMethod::FiniteDifference,
    })
}

/// Compute finite difference gradient for a single parameter
fn compute_single_param_fd<F>(
    simulator: &Simulator,
    circuit_builder: &F,
    observable: &PauliObservable,
    params: &[f64],
    param_index: usize,
    epsilon: f64,
    method: FiniteDifferenceMethod,
    base_expectation: Option<f64>,
) -> Result<f64>
where
    F: Fn(&[f64]) -> Circuit,
{
    match method {
        FiniteDifferenceMethod::Forward => {
            let mut params_plus = params.to_vec();
            params_plus[param_index] += epsilon;

            let circuit_plus = circuit_builder(&params_plus);
            let exp_plus = evaluate_expectation(simulator, &circuit_plus, observable)?;
            let exp_base = base_expectation.unwrap();

            Ok((exp_plus - exp_base) / epsilon)
        },
        FiniteDifferenceMethod::Backward => {
            let mut params_minus = params.to_vec();
            params_minus[param_index] -= epsilon;

            let circuit_minus = circuit_builder(&params_minus);
            let exp_minus = evaluate_expectation(simulator, &circuit_minus, observable)?;
            let exp_base = base_expectation.unwrap();

            Ok((exp_base - exp_minus) / epsilon)
        },
        FiniteDifferenceMethod::Central => {
            let mut params_plus = params.to_vec();
            let mut params_minus = params.to_vec();
            params_plus[param_index] += epsilon;
            params_minus[param_index] -= epsilon;

            let circuit_plus = circuit_builder(&params_plus);
            let circuit_minus = circuit_builder(&params_minus);

            let exp_plus = evaluate_expectation(simulator, &circuit_plus, observable)?;
            let exp_minus = evaluate_expectation(simulator, &circuit_minus, observable)?;

            Ok((exp_plus - exp_minus) / (2.0 * epsilon))
        },
    }
}

/// Evaluate expectation value
fn evaluate_expectation(
    simulator: &Simulator,
    circuit: &Circuit,
    observable: &PauliObservable,
) -> Result<f64> {
    let result = simulator.run(circuit)?;

    let expectation = match &result.state {
        AdaptiveState::Dense(dense) => observable.expectation_value(dense)?,
        AdaptiveState::Sparse { state: sparse, .. } => {
            use simq_state::DenseState;
            let dense = DenseState::from_sparse(sparse)?;
            observable.expectation_value(&dense)?
        },
    };

    Ok(expectation)
}

/// Automatic epsilon selection based on parameter magnitude
pub fn auto_epsilon(params: &[f64], base_epsilon: f64) -> Vec<f64> {
    params
        .iter()
        .map(|&p| {
            let magnitude = p.abs();
            if magnitude > 1.0 {
                base_epsilon * magnitude
            } else {
                base_epsilon
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimulatorConfig;
    use simq_core::QubitId;
    use simq_gates::standard::{Hadamard, RotationY};
    use simq_state::observable::PauliString;
    use std::sync::Arc;

    #[test]
    fn test_finite_difference_central() {
        let simulator = Simulator::new(SimulatorConfig::default());

        let circuit_builder = |params: &[f64]| {
            let mut circuit = Circuit::new(1);
            circuit
                .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
                .unwrap();
            circuit
                .add_gate(Arc::new(RotationY::new(params[0])), &[QubitId::new(0)])
                .unwrap();
            circuit
        };

        let observable =
            PauliObservable::from_pauli_string(PauliString::from_str("Z").unwrap(), 1.0);
        let params = vec![0.5];

        let config = FiniteDifferenceConfig {
            method: FiniteDifferenceMethod::Central,
            epsilon: 1e-5,
            parallel: false,
        };

        let result = compute_gradient_finite_difference(
            &simulator,
            circuit_builder,
            &observable,
            &params,
            &config,
        )
        .unwrap();

        assert_eq!(result.gradients.len(), 1);
        assert_eq!(result.num_evaluations, 2); // Central: 2 evaluations
    }

    #[test]
    fn test_finite_difference_forward() {
        let simulator = Simulator::new(SimulatorConfig::default());

        let circuit_builder = |params: &[f64]| {
            let mut circuit = Circuit::new(1);
            circuit
                .add_gate(Arc::new(RotationY::new(params[0])), &[QubitId::new(0)])
                .unwrap();
            circuit
        };

        let observable =
            PauliObservable::from_pauli_string(PauliString::from_str("Z").unwrap(), 1.0);
        let params = vec![0.5];

        let config = FiniteDifferenceConfig {
            method: FiniteDifferenceMethod::Forward,
            epsilon: 1e-5,
            parallel: false,
        };

        let result = compute_gradient_finite_difference(
            &simulator,
            circuit_builder,
            &observable,
            &params,
            &config,
        )
        .unwrap();

        assert_eq!(result.gradients.len(), 1);
        assert_eq!(result.num_evaluations, 2); // Forward: base + 1
    }

    #[test]
    fn test_auto_epsilon() {
        let params = vec![0.1, 1.0, 10.0];
        let epsilons = auto_epsilon(&params, 1e-7);

        assert_eq!(epsilons[0], 1e-7); // Small param: use base
        assert_eq!(epsilons[1], 1e-7); // Magnitude 1: use base
        assert_eq!(epsilons[2], 1e-6); // Large param: scale epsilon
    }
}
