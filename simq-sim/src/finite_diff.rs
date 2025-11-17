//! Finite difference fallback for gradient computation in variational quantum algorithms.

use rayon::prelude::*;
use crate::simulator::Simulator;
use crate::circuit::ParametricCircuit;
use crate::observable::Observable;

/// Computes gradients of expectation values using finite differences.
/// For each parameter θᵢ, evaluates the expectation value at θᵢ + ε and θᵢ - ε, then computes the gradient.
pub fn compute_gradient_finite_difference(
    simulator: &Simulator,
    circuit: &ParametricCircuit,
    observable: &Observable,
    params: &[f64],
    epsilon: f64,
) -> Vec<f64> {
    params
        .par_iter()
        .enumerate()
        .map(|(i, &theta)| {
            let mut params_plus = params.to_vec();
            let mut params_minus = params.to_vec();
            params_plus[i] += epsilon;
            params_minus[i] -= epsilon;

            let exp_plus = simulator.expectation_value(circuit, &params_plus, observable);
            let exp_minus = simulator.expectation_value(circuit, &params_minus, observable);

            (exp_plus - exp_minus) / (2.0 * epsilon)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    // ...existing code...
    // Add tests for finite difference gradient correctness here.
}
