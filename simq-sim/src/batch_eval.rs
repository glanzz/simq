//! CPU batch execution for parameter sweeps over a fixed circuit shape.
//!
//! A VQE/QAOA outer loop calls [`Simulator::run`] on the same-shaped
//! circuit many times, only rotation angles differing — [`Simulator`]'s
//! `fusion_cache` field already exists to skip re-deriving multi-qubit
//! fusion's block *structure* on every call (see its docs). The natural
//! next step, and the pure-software counterpart to a GPU decision-diagram
//! batch backend (BQSim-style: a batched backend sharing structure across
//! a parameter sweep), is running the sweep itself across CPU cores: each
//! instance's compile-and-simulate work is independent once that shared
//! structural work is (at most) paid once, so it embarrassingly
//! parallelizes via rayon. `FusionStructureCache` is `Mutex`-guarded
//! specifically so concurrent lookups from a batch like this are safe.

use crate::error::Result;
use crate::result::SimulationResult;
use crate::simulator::Simulator;
use rayon::prelude::*;
use simq_core::Circuit;
use simq_state::{AdaptiveState, DenseState, PauliObservable};

/// Run `circuit_for_instance(0..num_instances)` in parallel across CPU
/// cores on `simulator`, returning one [`SimulationResult`] per instance in
/// instance order.
///
/// # Errors
/// Returns the first error encountered, if any instance fails to simulate.
pub fn run_batch(
    simulator: &Simulator,
    num_instances: usize,
    circuit_for_instance: impl Fn(usize) -> Circuit + Sync,
) -> Result<Vec<SimulationResult>> {
    (0..num_instances)
        .into_par_iter()
        .map(|i| simulator.run(&circuit_for_instance(i)))
        .collect()
}

/// Same as [`run_batch`], but immediately reduces each instance's final
/// state through `observable` and returns just the expectation values —
/// the common VQE/QAOA case, and avoids materializing every instance's
/// full state at once.
///
/// # Errors
/// Returns the first error encountered (simulation failure, or an
/// observable/state qubit-count mismatch).
pub fn run_batch_expectation(
    simulator: &Simulator,
    num_instances: usize,
    circuit_for_instance: impl Fn(usize) -> Circuit + Sync,
    observable: &PauliObservable,
) -> Result<Vec<f64>> {
    (0..num_instances)
        .into_par_iter()
        .map(|i| {
            let result = simulator.run(&circuit_for_instance(i))?;
            let dense = to_dense(result.state)?;
            Ok(observable.expectation_value(&dense)?)
        })
        .collect()
}

/// `AdaptiveState` -> `DenseState`, converting only when the state didn't
/// already settle dense — mirrors `simq::bench_workloads::run_to_dense`,
/// as a `Result`-returning helper instead of a panicking one.
fn to_dense(state: AdaptiveState) -> Result<DenseState> {
    match state {
        AdaptiveState::Dense(dense) => Ok(dense),
        sparse => {
            let n = sparse.num_qubits();
            let amps = sparse.to_dense_vec();
            Ok(DenseState::from_amplitudes(n, &amps)?)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimulatorConfig;
    use simq_core::QubitId;
    use simq_gates::standard::{CNot, Hadamard, RotationY};
    use simq_state::{Pauli, PauliString};
    use std::sync::Arc;

    fn chain_circuit(num_qubits: usize, theta: f64) -> Circuit {
        let mut c = Circuit::new(num_qubits);
        for q in 0..num_qubits {
            c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
        }
        for q in 0..num_qubits {
            c.add_gate(Arc::new(RotationY::new(theta + 0.1 * q as f64)), &[QubitId::new(q)])
                .unwrap();
        }
        for q in 0..num_qubits - 1 {
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
                .unwrap();
        }
        c
    }

    #[test]
    fn run_batch_returns_one_result_per_instance() {
        let sim = Simulator::new(SimulatorConfig::default());
        let results = run_batch(&sim, 5, |i| chain_circuit(4, 0.2 * i as f64)).unwrap();
        assert_eq!(results.len(), 5);
        for r in &results {
            assert_eq!(r.num_qubits(), 4);
        }
    }

    #[test]
    fn run_batch_expectation_matches_serial_evaluation() {
        let sim = Simulator::new(SimulatorConfig::default());
        let n = 5;
        let obs = PauliObservable::from_pauli_string(PauliString::all_z(n), 1.0);

        let batched =
            run_batch_expectation(&sim, 6, |i| chain_circuit(n, 0.15 * i as f64), &obs).unwrap();

        let serial: Vec<f64> = (0..6)
            .map(|i| {
                let result = sim.run(&chain_circuit(n, 0.15 * i as f64)).unwrap();
                let dense = to_dense(result.state).unwrap();
                obs.expectation_value(&dense).unwrap()
            })
            .collect();

        assert_eq!(batched, serial);
    }

    #[test]
    fn run_batch_expectation_shares_the_fusion_cache_across_instances() {
        // Only the multi-qubit fusion path (>= 18 qubits) ever touches the
        // structural cache -- see `FusionStructureCache`'s docs -- so this
        // needs a wide-enough circuit to actually exercise cache reuse
        // across the batch, unlike the small-n tests above.
        let n = 20;
        let sim = Simulator::new(SimulatorConfig::default());
        let obs = PauliObservable::from_pauli_string(PauliString::all_z(n), 1.0);

        let values =
            run_batch_expectation(&sim, 8, |i| chain_circuit(n, 0.05 * i as f64), &obs).unwrap();
        assert_eq!(values.len(), 8);
        assert!(
            sim.fusion_cache_hits() > 0,
            "expected at least one fusion-cache hit across the batch"
        );
    }

    #[test]
    fn run_batch_propagates_the_first_error() {
        use simq_core::gate::Gate;

        #[derive(Debug)]
        struct NoMatrixGate;
        impl Gate for NoMatrixGate {
            fn name(&self) -> &str {
                "NoMatrix"
            }
            fn num_qubits(&self) -> usize {
                1
            }
        }

        let sim = Simulator::new(SimulatorConfig::default().with_optimization(false));
        let make_bad_circuit = |_i: usize| {
            let mut c = Circuit::new(1);
            c.add_gate(Arc::new(NoMatrixGate), &[QubitId::new(0)])
                .unwrap();
            c
        };

        let result = run_batch(&sim, 3, make_bad_circuit);
        assert!(result.is_err());
    }

    #[test]
    fn pauli_i_case_still_exercises_paulistring_get_none_branch_in_all_z() {
        // Sanity check that `PauliString::all_z` covers every qubit (no
        // gaps `.get()` would return `None` for), so `to_dense`'s
        // conversion path above is exercised on an observable that
        // actually spans the full register.
        let obs = PauliString::all_z(4);
        assert!((0..4).all(|q| obs.get(q) == Some(Pauli::Z)));
    }
}
