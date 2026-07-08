//! Shared workload definitions for the end-to-end benchmark suite.
//!
//! These builders are used by BOTH `benches/end_to_end.rs` (timing) and
//! `examples/xcheck_bench.rs` (cross-validation against the Qiskit baseline
//! in `benchmarks/qiskit_baseline.py`). Keeping them in one place guarantees
//! that the circuits we time are exactly the circuits whose expectation
//! values are checked against Qiskit before any numbers are published.
//!
//! The Python side re-implements the same circuits; `benchmarks/compare.py`
//! refuses to print a comparison table unless all cross-check values agree
//! to 1e-12.

use simq_core::{Circuit, QubitId};
use simq_gates::standard::{CNot, Hadamard, RotationX, RotationY, RotationZ};
use simq_sim::{Simulator, SimulatorConfig};
use simq_state::{
    measurement::ComputationalBasis, AdaptiveState, DenseState, Pauli, PauliObservable,
    PauliString,
};
use std::sync::Arc;

/// Deterministic RY angle for (layer, qubit) — same formula as the Python side.
pub fn vqe_theta(layer: usize, qubit: usize, num_qubits: usize) -> f64 {
    0.1 + 0.37 * (layer * num_qubits + qubit) as f64
}

/// Deterministic RZ angle for (layer, qubit) — same formula as the Python side.
pub fn vqe_phi(layer: usize, qubit: usize, num_qubits: usize) -> f64 {
    0.05 + 0.21 * (layer * num_qubits + qubit) as f64
}

/// QAOA parameters (gamma, beta) per layer — same values as the Python side.
pub const QAOA_GAMMA: [f64; 2] = [0.8, 0.4];
pub const QAOA_BETA: [f64; 2] = [0.7, 0.35];

/// Number of VQE ansatz layers.
pub const VQE_LAYERS: usize = 3;

/// Hardware-efficient VQE ansatz: H on all qubits, then `VQE_LAYERS` layers of
/// [RY(θ_lq) on each qubit, CNOT chain, RZ(φ_lq) on each qubit].
///
/// The per-qubit angles are deliberately asymmetric so any qubit-ordering
/// mismatch against the Qiskit implementation shows up in the cross-check.
pub fn vqe_circuit(num_qubits: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits);
    for q in 0..num_qubits {
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
    }
    for l in 0..VQE_LAYERS {
        for q in 0..num_qubits {
            c.add_gate(
                Arc::new(RotationY::new(vqe_theta(l, q, num_qubits))),
                &[QubitId::new(q)],
            )
            .unwrap();
        }
        for q in 0..num_qubits - 1 {
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
                .unwrap();
        }
        for q in 0..num_qubits {
            c.add_gate(
                Arc::new(RotationZ::new(vqe_phi(l, q, num_qubits))),
                &[QubitId::new(q)],
            )
            .unwrap();
        }
    }
    c
}

/// VQE observable: H = Σ_{q} Z_q Z_{q+1} + 0.5 Σ_q X_q
///
/// Mixes diagonal (ZZ) and off-diagonal (X) terms so both expectation paths
/// are exercised.
pub fn vqe_observable(num_qubits: usize) -> PauliObservable {
    let mut obs = PauliObservable::new();
    for q in 0..num_qubits - 1 {
        let mut paulis = vec![Pauli::I; num_qubits];
        paulis[q] = Pauli::Z;
        paulis[q + 1] = Pauli::Z;
        obs.add_term(PauliString::from_paulis(paulis), 1.0);
    }
    for q in 0..num_qubits {
        let mut paulis = vec![Pauli::I; num_qubits];
        paulis[q] = Pauli::X;
        obs.add_term(PauliString::from_paulis(paulis), 0.5);
    }
    obs
}

/// QAOA MaxCut circuit on the ring graph C_n at depth p=2.
///
/// Cost layers implement exp(-i γ Z_q Z_r) via CNOT·RZ(2γ)·CNOT, mixer layers
/// are RX(2β) on every qubit — the textbook construction, gate-for-gate the
/// same as the Python side.
pub fn qaoa_circuit(num_qubits: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits);
    for q in 0..num_qubits {
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
    }
    for l in 0..QAOA_GAMMA.len() {
        for q in 0..num_qubits {
            let r = (q + 1) % num_qubits;
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(r)])
                .unwrap();
            c.add_gate(Arc::new(RotationZ::new(2.0 * QAOA_GAMMA[l])), &[QubitId::new(r)])
                .unwrap();
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(r)])
                .unwrap();
        }
        for q in 0..num_qubits {
            c.add_gate(Arc::new(RotationX::new(2.0 * QAOA_BETA[l])), &[QubitId::new(q)])
                .unwrap();
        }
    }
    c
}

/// QAOA MaxCut cost observable on C_n: C = Σ_edges 0.5 (1 - Z_q Z_r).
///
/// Returned as the ZZ part only; use [`qaoa_cost_from_zz`] to fold in the
/// constant.
pub fn qaoa_zz_observable(num_qubits: usize) -> PauliObservable {
    let mut obs = PauliObservable::new();
    for q in 0..num_qubits {
        let r = (q + 1) % num_qubits;
        let mut paulis = vec![Pauli::I; num_qubits];
        paulis[q] = Pauli::Z;
        paulis[r] = Pauli::Z;
        obs.add_term(PauliString::from_paulis(paulis), 1.0);
    }
    obs
}

/// Fold the ⟨ΣZZ⟩ expectation into the MaxCut cost value.
pub fn qaoa_cost_from_zz(num_qubits: usize, zz_expectation: f64) -> f64 {
    0.5 * (num_qubits as f64) - 0.5 * zz_expectation
}

/// GHZ preparation circuit: H(0) then a CNOT chain.
pub fn ghz_circuit(num_qubits: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits);
    c.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
    for q in 0..num_qubits - 1 {
        c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
            .unwrap();
    }
    c
}

/// Run a circuit and return the final state as a `DenseState`.
pub fn run_to_dense(sim: &Simulator, circuit: &Circuit) -> DenseState {
    let result = sim.run(circuit).expect("simulation failed");
    match result.state {
        AdaptiveState::Dense(dense) => dense,
        sparse => {
            let amps = sparse.to_dense_vec();
            DenseState::from_amplitudes(sparse.num_qubits(), &amps).unwrap()
        },
    }
}

/// One full VQE energy evaluation: simulate the ansatz and take ⟨H⟩.
pub fn vqe_energy(sim: &Simulator, num_qubits: usize) -> f64 {
    let circuit = vqe_circuit(num_qubits);
    let state = run_to_dense(sim, &circuit);
    vqe_observable(num_qubits)
        .expectation_value(&state)
        .expect("expectation failed")
}

/// One full QAOA cost evaluation: simulate the p=2 circuit and take the cut value.
pub fn qaoa_cost(sim: &Simulator, num_qubits: usize) -> f64 {
    let circuit = qaoa_circuit(num_qubits);
    let state = run_to_dense(sim, &circuit);
    let zz = qaoa_zz_observable(num_qubits)
        .expectation_value(&state)
        .expect("expectation failed");
    qaoa_cost_from_zz(num_qubits, zz)
}

/// GHZ shot sampling: simulate and draw `shots` samples.
pub fn ghz_sample(sim: &Simulator, num_qubits: usize, shots: usize, seed: u64) -> usize {
    let circuit = ghz_circuit(num_qubits);
    let state = run_to_dense(sim, &circuit);
    // xorshift* PRNG: deterministic and dependency-free
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let mut rng = move || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let x = s.wrapping_mul(0x2545F4914F6CDD1D);
        (x >> 11) as f64 / (1u64 << 53) as f64
    };
    let result = ComputationalBasis::new()
        .sample(&state, shots, &mut rng)
        .expect("sampling failed");
    result.sorted_outcomes().len()
}

/// The default simulator used across the suite (out-of-the-box settings).
pub fn default_simulator() -> Simulator {
    Simulator::new(SimulatorConfig::default())
}
