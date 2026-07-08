//! # SimQ — high-performance quantum computing SDK in Rust
//!
//! This is the umbrella crate: it re-exports the SimQ subcrates and provides
//! [`QuantumCircuit`], a fluent circuit builder, so a single dependency on
//! `simq` is all you need to build and simulate circuits.
//!
//! # Quick start
//!
//! ```
//! use simq::QuantumCircuit;
//!
//! // Bell state: H on qubit 0, then CNOT(0 -> 1)
//! let mut qc = QuantumCircuit::new(2);
//! qc.h(0).cnot(0, 1);
//!
//! let result = qc.simulate_with_shots(1024).unwrap();
//! let counts = result.measurements.unwrap();
//! // Only "00" and "11" occur, each with probability ~0.5
//! assert_eq!(counts.get("01") + counts.get("10"), 0);
//! ```
//!
//! Parameterized circuits chain just as naturally:
//!
//! ```
//! use simq::QuantumCircuit;
//!
//! let theta = 0.8;
//! let mut qc = QuantumCircuit::new(2);
//! qc.ry(theta, 0).cnot(0, 1).rz(theta, 1);
//! let circuit = qc.build().unwrap();
//! assert_eq!(circuit.len(), 3);
//! ```
//!
//! # Crate layout
//!
//! | Module | Source crate | Contents |
//! |--------|--------------|----------|
//! | [`core`] | `simq-core` | [`Circuit`], [`QubitId`], the [`Gate`] trait, builders |
//! | [`gates`] | `simq-gates` | Standard gate library (Hadamard, CNOT, rotations, ...) |
//! | [`sim`] | `simq-sim` | [`Simulator`], VQE/QAOA helpers, gradients |
//! | [`state`] | `simq-state` | State vectors, density matrices, measurement, observables |
//! | [`compiler`] | `simq-compiler` | Circuit optimization passes |
//! | [`backend`] | `simq-backend` | Backend abstraction, transpiler, local simulator backend |
//!
//! The most common types are re-exported at the crate root; `use
//! simq::prelude::*;` pulls in everything needed for typical usage,
//! including the full standard gate set.

pub use simq_backend as backend;
pub use simq_compiler as compiler;
pub use simq_core as core;
pub use simq_gates as gates;
pub use simq_sim as sim;
pub use simq_state as state;

mod circuit;

pub use circuit::QuantumCircuit;

/// Workload definitions shared by the end-to-end benchmarks and the Qiskit
/// cross-validation harness (see `BENCHMARKS.md`).
pub mod bench_workloads;

// Most-used types at the crate root
pub use simq_core::{
    Circuit, CircuitBuilder, Complex64, DynamicCircuitBuilder, Gate, GateOp, QuantumError, QubitId,
};
pub use simq_sim::{
    MeasurementCounts, SimulationResult, Simulator, SimulatorConfig, SimulatorError,
};
pub use simq_state::{Pauli, PauliObservable, PauliString, StateVector};

/// Convenience re-exports for typical usage: `use simq::prelude::*;`
pub mod prelude {
    pub use crate::circuit::QuantumCircuit;
    pub use simq_core::{Circuit, CircuitBuilder, DynamicCircuitBuilder, Gate, QubitId};
    pub use simq_gates::standard::*;
    pub use simq_sim::{
        MeasurementCounts, SimulationResult, Simulator, SimulatorConfig, SimulatorError,
    };
    pub use simq_state::{Pauli, PauliObservable, PauliString};
    pub use std::sync::Arc;
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn readme_quick_start_compiles_and_runs() {
        let mut qc = QuantumCircuit::new(2);
        qc.h(0).cnot(0, 1);

        let result = qc.simulate_with_shots(1024).unwrap();
        let counts = result.measurements.unwrap();
        assert_eq!(counts.total_shots(), 1024);
        assert_eq!(counts.get("01") + counts.get("10"), 0);
    }

    #[test]
    fn low_level_api_reachable_through_facade() {
        // The Arc-based subcrate API stays fully accessible
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();

        let result = Simulator::new(SimulatorConfig::default())
            .run(&circuit)
            .unwrap();
        assert_eq!(result.num_qubits(), 2);
    }
}
