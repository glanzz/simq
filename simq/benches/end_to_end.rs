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

criterion_group!(benches, bench_vqe_energy, bench_qaoa_maxcut, bench_ghz_sampling);
criterion_main!(benches);
