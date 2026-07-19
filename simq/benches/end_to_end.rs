//! End-to-end benchmarks: full VQE/QAOA energy evaluations and GHZ sampling.
//!
//! These are the SimQ half of the cross-validated suite described in
//! `BENCHMARKS.md`. The circuits come from `simq::bench_workloads`, the same
//! module used by `examples/xcheck_bench.rs`, so the timed workloads are
//! provably the workloads whose expectation values are checked against
//! Qiskit (to 1e-12) before any comparison table is printed.
//!
//! Run the full suite (bench + baseline + cross-check + table) with
//! `./benchmarks/run.sh`, or just these timings with
//! `cargo bench -p simq --bench end_to_end`.
//!
//! Each iteration measures what a variational optimizer pays per cost-function
//! call: circuit construction + compilation/optimization + simulation +
//! expectation value. Nothing is cached between iterations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use simq::bench_workloads as wl;
use std::hint::black_box;

const QUBIT_SIZES: [usize; 4] = [4, 8, 12, 16];
const GHZ_SHOTS: usize = 1024;
/// Qubit count for the multi-instance groups below -- see
/// `wl::NUM_INSTANCES` docs on why this is one representative size rather
/// than the full `QUBIT_SIZES` sweep (bounding how much slower this adds to
/// `benchmarks/run.sh`, which re-runs this suite against Qiskit and qsim
/// for every group/size).
const MULTI_INSTANCE_SIZE: usize = 8;

fn bench_vqe_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("vqe_energy");
    for &n in &QUBIT_SIZES {
        if n >= 12 {
            group.sample_size(20);
        }
        let sim = wl::default_simulator();
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::vqe_energy(&sim, n)));
        });
    }
    group.finish();
}

fn bench_qaoa_maxcut(c: &mut Criterion) {
    let mut group = c.benchmark_group("qaoa_maxcut");
    for &n in &QUBIT_SIZES {
        if n >= 12 {
            group.sample_size(20);
        }
        let sim = wl::default_simulator();
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::qaoa_cost(&sim, n)));
        });
    }
    group.finish();
}

fn bench_ghz_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("ghz_sampling");
    for &n in &QUBIT_SIZES {
        if n >= 12 {
            group.sample_size(20);
        }
        let sim = wl::default_simulator();
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::ghz_sample(&sim, n, GHZ_SHOTS, 0xB1A2)));
        });
    }
    group.finish();
}

/// QFT: long-range (non-nearest-neighbor) entangling structure, the
/// counterpoint to the three local workloads above -- see
/// `wl::qft_circuit`'s docs and BENCHMARKS.md's methodology notes.
fn bench_qft_probe(c: &mut Criterion) {
    let mut group = c.benchmark_group("qft_probe");
    for &n in &QUBIT_SIZES {
        if n >= 12 {
            group.sample_size(20);
        }
        let sim = wl::default_simulator();
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::qft_probe(&sim, n)));
        });
    }
    group.finish();
}

/// Random-circuit-sampling-style workload: structure-agnostic stress test,
/// the counterpoint to VQE/QAOA/GHZ's fixed linear-chain locality -- see
/// `wl::random_circuit`'s docs and BENCHMARKS.md's methodology notes.
fn bench_random_circuit(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_circuit");
    for &n in &QUBIT_SIZES {
        if n >= 12 {
            group.sample_size(20);
        }
        let sim = wl::default_simulator();
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::random_circuit_p0(&sim, n)));
        });
    }
    group.finish();
}

/// Multi-instance VQE/QAOA/GHZ: times the whole `NUM_INSTANCES`-instance
/// batch as one unit per iteration -- see `wl::NUM_INSTANCES` docs on why
/// this exists (QED-C-style overfitting guard) and why it's one
/// representative qubit count rather than the full `QUBIT_SIZES` sweep.
fn bench_multi_instance(c: &mut Criterion) {
    let sim = wl::default_simulator();
    let n = MULTI_INSTANCE_SIZE;

    let mut group = c.benchmark_group("vqe_energy_multi_instance");
    group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
        b.iter(|| black_box(wl::vqe_energy_instances(&sim, n)));
    });
    group.finish();

    let mut group = c.benchmark_group("qaoa_cost_multi_instance");
    group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
        b.iter(|| black_box(wl::qaoa_cost_instances(&sim, n)));
    });
    group.finish();

    let mut group = c.benchmark_group("ghz_sampling_multi_instance");
    group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
        b.iter(|| black_box(wl::ghz_sample_instances(&sim, n, GHZ_SHOTS)));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_vqe_energy,
    bench_qaoa_maxcut,
    bench_ghz_sampling,
    bench_qft_probe,
    bench_random_circuit,
    bench_multi_instance
);
criterion_main!(benches);
