//! Cross-validation of the benchmark workloads against Qiskit.
//!
//! Prints the exact expectation values of the `BENCHMARKS.md` workloads.
//! `benchmarks/compare.py` runs this and diffs the output against the same
//! values computed by Qiskit, proving both suites simulate identical
//! circuits before any timings are reported. Keep the circuit definitions
//! in sync with `simq/benches/end_to_end.rs` and
//! `benchmarks/qiskit_baseline.py`.
use simq::{Pauli, PauliObservable, PauliString, QuantumCircuit};
use std::f64::consts::PI;

fn param(e: u64, j: u64) -> f64 {
    let x = ((e.wrapping_mul(131) + j.wrapping_mul(31)) % 1000) as f64 / 1000.0;
    (2.0 * x - 1.0) * PI
}

fn vqe_ansatz(n: usize, layers: usize, eval: u64) -> QuantumCircuit {
    let mut qc = QuantumCircuit::new(n);
    let mut j = 0u64;
    for _ in 0..layers {
        for q in 0..n {
            qc.ry(param(eval, j), q);
            j += 1;
        }
        for q in 0..n {
            qc.rz(param(eval, j), q);
            j += 1;
        }
        for q in 0..n - 1 {
            qc.cnot(q, q + 1);
        }
    }
    qc
}

fn ising_hamiltonian(n: usize) -> PauliObservable {
    let mut paulis = vec![Pauli::I; n];
    paulis[0] = Pauli::Z;
    paulis[1] = Pauli::Z;
    let mut h = PauliObservable::from_pauli_string(PauliString::from_paulis(paulis), 1.0);
    for i in 1..n - 1 {
        let mut p = vec![Pauli::I; n];
        p[i] = Pauli::Z;
        p[i + 1] = Pauli::Z;
        h.add_term(PauliString::from_paulis(p), 1.0);
    }
    for i in 0..n {
        let mut p = vec![Pauli::I; n];
        p[i] = Pauli::X;
        h.add_term(PauliString::from_paulis(p), 0.5);
    }
    h
}

fn qaoa_circuit(n: usize, p: usize, eval: u64) -> QuantumCircuit {
    let mut qc = QuantumCircuit::new(n);
    for q in 0..n {
        qc.h(q);
    }
    let mut j = 0u64;
    for _ in 0..p {
        let gamma = param(eval, j);
        j += 1;
        for q in 0..n {
            qc.rzz(gamma, q, (q + 1) % n);
        }
        let beta = param(eval, j);
        j += 1;
        for q in 0..n {
            qc.rx(beta, q);
        }
    }
    qc
}

fn maxcut_cost(n: usize) -> PauliObservable {
    let mut h = PauliObservable::from_pauli_string(PauliString::identity(n), n as f64 * 0.5);
    for q in 0..n {
        let mut p = vec![Pauli::I; n];
        p[q] = Pauli::Z;
        p[(q + 1) % n] = Pauli::Z;
        h.add_term(PauliString::from_paulis(p), -0.5);
    }
    h
}

fn main() {
    for e in [1u64, 2, 7] {
        for n in [4usize, 8] {
            let v = vqe_ansatz(n, 3, e).expectation_value(&ising_hamiltonian(n)).unwrap();
            println!("vqe {n}q eval{e}: {v:.12}");
            let q = qaoa_circuit(n, 2, e).expectation_value(&maxcut_cost(n)).unwrap();
            println!("qaoa {n}q eval{e}: {q:.12}");
        }
    }
}
