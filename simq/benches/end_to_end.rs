//! End-to-end benchmarks mirrored by `benchmarks/qiskit_baseline.py`.
//!
//! Every workload here is defined, gate for gate, in `BENCHMARKS.md` and
//! implemented identically in the Qiskit baseline script so the two suites
//! are directly comparable. If you change a circuit here, change it there
//! and in the writeup too.
//!
//! One iteration always includes circuit *construction* as well as
//! simulation — that matches how a variational optimizer drives a
//! simulator, which is the workload SimQ optimizes for.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use simq::{Pauli, PauliObservable, PauliString, QuantumCircuit, SimulatorConfig};
use std::f64::consts::PI;

/// Deterministic, cache-unfriendly parameter schedule shared with the
/// Qiskit baseline: evaluation `e`, parameter slot `j`.
///
/// The angles it produces are "generic" (irrational multiples of π), so
/// SimQ's compile-time gate caches for common angles do NOT kick in and
/// the comparison measures the runtime-computation path.
fn param(e: u64, j: u64) -> f64 {
    let x = ((e.wrapping_mul(131) + j.wrapping_mul(31)) % 1000) as f64 / 1000.0;
    (2.0 * x - 1.0) * PI
}

/// Hardware-efficient VQE ansatz: `layers` × (RY + RZ on every qubit,
/// then a linear CNOT chain).
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

/// Transverse-field Ising Hamiltonian on a line:
/// H = Σ Z_i Z_{i+1} + 0.5 Σ X_i
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

/// QAOA MaxCut circuit on a ring graph, depth `p`.
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

/// MaxCut cost observable on the ring: C = Σ_edges (I − Z_i Z_j) / 2
fn maxcut_cost(n: usize) -> PauliObservable {
    let mut h =
        PauliObservable::from_pauli_string(PauliString::identity(n), n as f64 * 0.5);
    for q in 0..n {
        let mut p = vec![Pauli::I; n];
        p[q] = Pauli::Z;
        p[(q + 1) % n] = Pauli::Z;
        h.add_term(PauliString::from_paulis(p), -0.5);
    }
    h
}

/// VQE energy evaluation: build ansatz + statevector simulation + exact ⟨H⟩.
fn bench_vqe_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("vqe_energy");
    for &n in &[4usize, 8, 12, 16] {
        let hamiltonian = ising_hamiltonian(n);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            let mut eval = 0u64;
            b.iter(|| {
                eval += 1;
                let qc = vqe_ansatz(n, 3, eval);
                qc.expectation_value(&hamiltonian).unwrap()
            });
        });
    }
    group.finish();
}

/// QAOA cost evaluation: build p=2 circuit + simulation + exact ⟨C⟩.
fn bench_qaoa_maxcut(c: &mut Criterion) {
    let mut group = c.benchmark_group("qaoa_maxcut");
    for &n in &[4usize, 8, 12, 16] {
        let cost = maxcut_cost(n);
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            let mut eval = 0u64;
            b.iter(|| {
                eval += 1;
                let qc = qaoa_circuit(n, 2, eval);
                qc.expectation_value(&cost).unwrap()
            });
        });
    }
    group.finish();
}

/// GHZ preparation + 1024 measurement shots.
fn bench_ghz_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("ghz_sampling");
    for &n in &[10usize, 16, 20] {
        if n >= 20 {
            group.sample_size(20);
        }
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| {
                let mut qc = QuantumCircuit::new(n);
                qc.h(0);
                for q in 0..n - 1 {
                    qc.cnot(q, q + 1);
                }
                let config = SimulatorConfig::default().with_shots(1024).with_seed(42);
                qc.simulate_with_config(config).unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_vqe_energy, bench_qaoa_maxcut, bench_ghz_sampling);
criterion_main!(benches);
