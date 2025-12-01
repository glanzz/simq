//! Batch evaluation for efficient gradient computation
//!
//! This module provides utilities for evaluating multiple parameter sets
//! or multiple observables in parallel, which is critical for optimization
//! algorithms like VQE and QAOA.

use crate::error::Result;
use crate::Simulator;
use rayon::prelude::*;
use simq_core::Circuit;
use simq_state::observable::PauliObservable;
use simq_state::AdaptiveState;

/// Batch evaluation result
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Expectation values for each parameter set
    pub values: Vec<f64>,
    /// Total number of circuit evaluations
    pub num_evaluations: usize,
    /// Computation time
    pub computation_time: std::time::Duration,
}

/// Evaluate expectation values for a batch of parameter sets
///
/// This is useful for optimization algorithms that need to evaluate
/// the objective function at multiple points.
///
/// # Arguments
///
/// * `simulator` - The quantum simulator
/// * `circuit_builder` - Function to build circuit from parameters
/// * `observable` - Observable to measure
/// * `batch_params` - Vector of parameter sets to evaluate
///
/// # Returns
///
/// Vector of expectation values, one for each parameter set
///
/// # Example
///
/// ```ignore
/// let param_sets = vec![
///     vec![0.1, 0.2],
///     vec![0.3, 0.4],
///     vec![0.5, 0.6],
/// ];
///
/// let results = evaluate_batch_expectation(
///     &simulator,
///     |p| build_circuit(p),
///     &hamiltonian,
///     &param_sets,
/// )?;
/// ```
pub fn evaluate_batch_expectation<F>(
    simulator: &Simulator,
    circuit_builder: F,
    observable: &PauliObservable,
    batch_params: &[Vec<f64>],
) -> Result<BatchResult>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    let start_time = std::time::Instant::now();

    let values: Vec<f64> = batch_params
        .par_iter()
        .map(|params| {
            let circuit = circuit_builder(params);
            evaluate_single_expectation(simulator, &circuit, observable)
        })
        .collect::<Result<Vec<f64>>>()?;

    Ok(BatchResult {
        values,
        num_evaluations: batch_params.len(),
        computation_time: start_time.elapsed(),
    })
}

/// Evaluate expectation value for a single circuit
fn evaluate_single_expectation(
    simulator: &Simulator,
    circuit: &Circuit,
    observable: &PauliObservable,
) -> Result<f64> {
    let result = simulator.run(circuit)?;

    // Convert to dense state for expectation value computation
    let expectation = match &result.state {
        AdaptiveState::Dense(dense) => observable.expectation_value(dense)?,
        AdaptiveState::Sparse { state: sparse, .. } => {
            // Convert sparse to dense for expectation value
            use simq_state::DenseState;
            let dense = DenseState::from_sparse(sparse)?;
            observable.expectation_value(&dense)?
        },
    };

    Ok(expectation)
}

/// Evaluate multiple observables for the same circuit
///
/// This is more efficient than evaluating each observable separately
/// because the circuit only needs to be simulated once.
pub fn evaluate_multi_observable<F>(
    simulator: &Simulator,
    circuit_builder: F,
    observables: &[PauliObservable],
    params: &[f64],
) -> Result<Vec<f64>>
where
    F: Fn(&[f64]) -> Circuit,
{
    let circuit = circuit_builder(params);
    let result = simulator.run(&circuit)?;

    let expectation_values: Vec<f64> = observables
        .iter()
        .map(|observable| match &result.state {
            AdaptiveState::Dense(dense) => observable.expectation_value(dense),
            AdaptiveState::Sparse { state: sparse, .. } => {
                use simq_state::DenseState;
                let dense = DenseState::from_sparse(sparse)?;
                observable.expectation_value(&dense)
            },
        })
        .collect::<std::result::Result<Vec<f64>, simq_state::error::StateError>>()?;

    Ok(expectation_values)
}

/// Batch gradient computation for multiple parameter sets
///
/// Computes gradients for multiple initial points, useful for
/// parallel optimization or ensemble methods.
pub fn batch_gradient<F, G>(
    simulator: &Simulator,
    circuit_builder: F,
    gradient_fn: G,
    observable: &PauliObservable,
    batch_params: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
    G: Fn(&Simulator, &F, &PauliObservable, &[f64]) -> Result<Vec<f64>> + Send + Sync,
{
    batch_params
        .par_iter()
        .map(|params| gradient_fn(simulator, &circuit_builder, observable, params))
        .collect()
}

/// Grid search evaluation
///
/// Evaluates expectation value over a grid of parameter values,
/// useful for visualization and debugging.
#[derive(Debug, Clone)]
pub struct GridSearchResult {
    /// Grid of parameter values
    pub param_grid: Vec<Vec<f64>>,
    /// Expectation values at each grid point
    pub values: Vec<f64>,
    /// Optimal parameters (minimum energy)
    pub optimal_params: Vec<f64>,
    /// Optimal value
    pub optimal_value: f64,
}

pub fn grid_search<F>(
    simulator: &Simulator,
    circuit_builder: F,
    observable: &PauliObservable,
    param_ranges: &[(f64, f64)], // (min, max) for each parameter
    num_points: usize,           // Number of points per dimension
) -> Result<GridSearchResult>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    // Generate grid points
    let mut param_grid = Vec::new();
    generate_grid(param_ranges, num_points, &mut vec![], &mut param_grid);

    // Evaluate all grid points
    let batch_result =
        evaluate_batch_expectation(simulator, circuit_builder, observable, &param_grid)?;

    // Find optimal point
    let (min_idx, &optimal_value) = batch_result
        .values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    let optimal_params = param_grid[min_idx].clone();

    Ok(GridSearchResult {
        param_grid,
        values: batch_result.values,
        optimal_params,
        optimal_value,
    })
}

/// Recursively generate grid points
fn generate_grid(
    ranges: &[(f64, f64)],
    num_points: usize,
    current: &mut Vec<f64>,
    result: &mut Vec<Vec<f64>>,
) {
    if current.len() == ranges.len() {
        result.push(current.clone());
        return;
    }

    let dim = current.len();
    let (min, max) = ranges[dim];
    let step = (max - min) / (num_points - 1) as f64;

    for i in 0..num_points {
        let value = min + i as f64 * step;
        current.push(value);
        generate_grid(ranges, num_points, current, result);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimulatorConfig;
    use simq_core::QubitId;
    use simq_gates::standard::RotationY;
    use simq_state::observable::PauliString;
    use std::sync::Arc;

    #[test]
    fn test_batch_evaluation() {
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

        let batch_params = vec![
            vec![0.0],
            vec![std::f64::consts::PI / 2.0],
            vec![std::f64::consts::PI],
        ];

        let result =
            evaluate_batch_expectation(&simulator, circuit_builder, &observable, &batch_params)
                .unwrap();

        assert_eq!(result.values.len(), 3);
        assert_eq!(result.num_evaluations, 3);
    }

    #[test]
    fn test_grid_generation() {
        let ranges = vec![(0.0, 1.0), (-1.0, 1.0)];
        let mut grid = Vec::new();
        generate_grid(&ranges, 3, &mut vec![], &mut grid);

        assert_eq!(grid.len(), 9); // 3 × 3 grid
        assert_eq!(grid[0], vec![0.0, -1.0]);
        assert_eq!(grid[8], vec![1.0, 1.0]);
    }
}
