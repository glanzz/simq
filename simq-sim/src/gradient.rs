//! Gradient computation using the parameter shift rule for variational quantum algorithms.

use rayon::prelude::*;
use crate::simulator::Simulator;
use crate::circuit::ParametricCircuit;
use crate::observable::Observable;

/// Computes gradients of expectation values with respect to circuit parameters using the parameter shift rule.
/// For each parameter θᵢ, evaluates the expectation value at θᵢ + π/2 and θᵢ - π/2, then computes the gradient.
pub fn compute_gradient_parameter_shift(
    simulator: &Simulator,
    circuit: &ParametricCircuit,
    observable: &Observable,
    params: &[f64],
) -> Vec<f64> {
    let shift = std::f64::consts::FRAC_PI_2;
    params
        .par_iter()
        .enumerate()
        .map(|(i, &theta)| {
            let mut params_plus = params.to_vec();
            let mut params_minus = params.to_vec();
            params_plus[i] += shift;
            params_minus[i] -= shift;

            let exp_plus = simulator.expectation_value(circuit, &params_plus, observable);
            let exp_minus = simulator.expectation_value(circuit, &params_minus, observable);

            (exp_plus - exp_minus) / 2.0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    // ...existing code...
    // Add tests for single RX gate and VQE ansatz gradient correctness here.
}
