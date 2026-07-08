//! Batched parameter evaluation for variational quantum algorithms.

use rayon::prelude::*;
use crate::simulator::Simulator;
use crate::circuit::ParametricCircuit;
use crate::observable::Observable;

/// Evaluates expectation values for a batch of parameter sets in parallel.
/// Returns a vector of expectation values, one for each parameter set.
pub fn evaluate_batch_expectation(
    simulator: &Simulator,
    circuit: &ParametricCircuit,
    observable: &Observable,
    batch_params: &[Vec<f64>],
) -> Vec<f64> {
    batch_params
        .par_iter()
        .map(|params| simulator.expectation_value(circuit, params, observable))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    // ...existing code...
    // Add tests for batch evaluation correctness here.
}
