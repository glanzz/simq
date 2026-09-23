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

/// Same GHZ workload as `bench_ghz_sampling`, but on the Clifford/stabilizer
/// tableau backend and at qubit counts a statevector cannot reach (past the
/// ~30-qubit wall documented in BENCHMARKS.md) -- demonstrating the new
/// backend's whole point: `O(n^2)`, not `O(2^n)`, for Clifford-only
/// circuits like GHZ preparation. See `simq_sim::stabilizer`.
const STABILIZER_QUBIT_SIZES: [usize; 4] = [16, 50, 100, 200];

/// Fewer shots than `GHZ_SHOTS`: this group's point is demonstrating the
/// tableau backend scales to qubit counts a statevector cannot reach at
/// all, not precisely timing shot throughput -- and each shot clones and
/// collapses a full `O(n^2)`-bit tableau, so cost scales with both.
const STABILIZER_SHOTS: usize = 128;

fn bench_ghz_sampling_stabilizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("ghz_sampling_stabilizer");
    for &n in &STABILIZER_QUBIT_SIZES {
        if n >= 50 {
            group.sample_size(10);
        }
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::ghz_sample_stabilizer(n, STABILIZER_SHOTS, 0xB1A2)));
        });
    }
    group.finish();
}

/// Same VQE/QAOA workloads as `bench_vqe_energy`/`bench_qaoa_maxcut`, computed
/// on `simq_state::SinglePrecisionState` (`Complex32`, half the memory of the
/// default `Complex64` statevector) instead of the default `f64` one. Same
/// `QUBIT_SIZES` as those two groups, so the comparison isolates the
/// precision/memory tradeoff from any qubit-count effect.
fn bench_vqe_energy_single_precision(c: &mut Criterion) {
    let mut group = c.benchmark_group("vqe_energy_single_precision");
    for &n in &QUBIT_SIZES {
        if n >= 12 {
            group.sample_size(20);
        }
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::vqe_energy_single_precision(n)));
        });
    }
    group.finish();
}

fn bench_qaoa_cost_single_precision(c: &mut Criterion) {
    let mut group = c.benchmark_group("qaoa_cost_single_precision");
    for &n in &QUBIT_SIZES {
        if n >= 12 {
            group.sample_size(20);
        }
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::qaoa_cost_single_precision(n)));
        });
    }
    group.finish();
}

/// Same VQE/QAOA workloads as `bench_vqe_energy`/`bench_qaoa_maxcut`, computed
/// via the Pauli-propagation observable engine (`simq_sim::pauli_propagation`)
/// instead of the statevector simulator, at the *same* `QUBIT_SIZES` as those
/// two groups -- a correctness/overhead comparison, not a scaling claim.
/// Both ansätze put one non-Clifford rotation on every qubit in every layer,
/// which is exactly the regime `pauli_propagation::recommend_backend` (and
/// `wl::vqe_energy_pauli_propagation`'s docs) flag as a poor fit; see
/// `bench_near_clifford_pauli_propagation` below for this engine's actual
/// scaling showcase.
fn bench_vqe_energy_pauli_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("vqe_energy_pauli_propagation");
    for &n in &QUBIT_SIZES {
        if n >= 12 {
            group.sample_size(10);
        }
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::vqe_energy_pauli_propagation(n)));
        });
    }
    group.finish();
}

fn bench_qaoa_cost_pauli_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("qaoa_cost_pauli_propagation");
    for &n in &QUBIT_SIZES {
        if n >= 12 {
            group.sample_size(10);
        }
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::qaoa_cost_pauli_propagation(n)));
        });
    }
    group.finish();
}

/// `wl::near_clifford_circuit`: a fixed, small non-Clifford gate count
/// regardless of qubit count -- the regime Pauli propagation actually wins
/// in. Goes past the ~30-qubit statevector wall documented in
/// BENCHMARKS.md, same spirit as `bench_ghz_sampling_stabilizer`.
const NEAR_CLIFFORD_QUBIT_SIZES: [usize; 4] = [16, 30, 60, 100];

fn bench_near_clifford_pauli_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("near_clifford_pauli_propagation");
    for &n in &NEAR_CLIFFORD_QUBIT_SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::near_clifford_expectation_pauli_propagation(n)));
        });
    }
    group.finish();
}

/// Same VQE/QAOA workloads, computed on the matrix-product-state backend
/// (`simq_sim::mps`). Both ansätze are near-1D (VQE: linear CNOT chain, QAOA:
/// a ring) -- see `simq_sim::mps::is_1d_candidate` -- so MPS represents them
/// at a small bond dimension far past the statevector wall.
const MPS_QUBIT_SIZES: [usize; 4] = [16, 30, 50, 80];

fn bench_vqe_energy_mps(c: &mut Criterion) {
    let mut group = c.benchmark_group("vqe_energy_mps");
    for &n in &MPS_QUBIT_SIZES {
        if n >= 50 {
            group.sample_size(10);
        }
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::vqe_energy_mps(n)));
        });
    }
    group.finish();
}

fn bench_qaoa_cost_mps(c: &mut Criterion) {
    let mut group = c.benchmark_group("qaoa_cost_mps");
    for &n in &MPS_QUBIT_SIZES {
        if n >= 50 {
            group.sample_size(10);
        }
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::qaoa_cost_mps(n)));
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

/// Same VQE/QAOA multi-instance workloads as `bench_multi_instance`, but
/// computed via `simq_sim::batch_eval`'s parallel batch executor instead of
/// a serial loop over instances -- see `wl::vqe_energy_instances_batched`'s
/// docs. `BATCH_QUBIT_SIZES` includes both `MULTI_INSTANCE_SIZE` (8q, below
/// the fusion-structure cache's 18-qubit engagement threshold, so any win
/// here is pure CPU parallelism) and 20q (above it, so a win there also
/// reflects the batch sharing one instance's compiled fusion structure with
/// the rest -- see `FusionStructureCache`'s docs).
const BATCH_QUBIT_SIZES: [usize; 2] = [MULTI_INSTANCE_SIZE, 20];

fn bench_multi_instance_batched(c: &mut Criterion) {
    let sim = wl::default_simulator();

    let mut group = c.benchmark_group("vqe_energy_multi_instance_serial");
    for &n in &BATCH_QUBIT_SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::vqe_energy_instances(&sim, n)));
        });
    }
    group.finish();

    let mut group = c.benchmark_group("vqe_energy_multi_instance_batched");
    for &n in &BATCH_QUBIT_SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::vqe_energy_instances_batched(&sim, n)));
        });
    }
    group.finish();

    let mut group = c.benchmark_group("qaoa_cost_multi_instance_serial");
    for &n in &BATCH_QUBIT_SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::qaoa_cost_instances(&sim, n)));
        });
    }
    group.finish();

    let mut group = c.benchmark_group("qaoa_cost_multi_instance_batched");
    for &n in &BATCH_QUBIT_SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(format!("{n}q")), &n, |b, &n| {
            b.iter(|| black_box(wl::qaoa_cost_instances_batched(&sim, n)));
        });
    }
    group.finish();
}

/// Compiler-only comparison (no simulation), on `wl::redundant_circuit` --
/// the one workload in this suite with redundant fixed-gate chains for
/// `simq_compiler::egraph::EqualitySaturation` to reduce (VQE/QAOA/GHZ/QFT/
/// random_circuit all use parameterized rotations, out of that pass's
/// scope; see its module docs).
///
/// Two pairs, deliberately kept separate -- see BENCHMARKS.md for why:
/// `rewrite_only` compares pure symbolic rewriting (this crate's existing
/// `TemplateSubstitution`+`AdvancedTemplateMatching` vs. equality
/// saturation) with no `GateFusion` in either pipeline, which is the fair,
/// apples-to-apples comparison for what this pass actually adds.
/// `full_o3` compares the complete default O3 pipeline with and without
/// the pass added -- included for transparency: `GateFusion`'s numeric
/// matrix multiplication already collapses this benchmark's chains to
/// their minimum regardless of how they got shortened first, so this pair
/// mostly measures the pass's added compile-time overhead, not a gate-count
/// win, once fusion is already in the pipeline.
fn bench_redundant_circuit_compile(c: &mut Criterion) {
    use simq::compiler::pipeline::{
        create_compiler, create_o3_egraph_compiler, OptimizationLevel, PipelineBuilder,
    };

    let mut group = c.benchmark_group("redundant_circuit_compile_rewrite_only");
    for &n in &QUBIT_SIZES {
        let circuit = wl::redundant_circuit(n);

        let templates = PipelineBuilder::new()
            .with_dead_code_elimination()
            .with_template_substitution()
            .with_advanced_template_matching()
            .max_iterations(10)
            .build();
        group.bench_with_input(BenchmarkId::new("templates", format!("{n}q")), &n, |b, _| {
            b.iter(|| {
                let mut circuit = circuit.clone();
                templates.compile(&mut circuit).unwrap();
                black_box(circuit.len())
            });
        });

        let egraph = PipelineBuilder::new()
            .with_dead_code_elimination()
            .with_equality_saturation()
            .max_iterations(10)
            .build();
        group.bench_with_input(BenchmarkId::new("egraph", format!("{n}q")), &n, |b, _| {
            b.iter(|| {
                let mut circuit = circuit.clone();
                egraph.compile(&mut circuit).unwrap();
                black_box(circuit.len())
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("redundant_circuit_compile_full_o3");
    for &n in &QUBIT_SIZES {
        let circuit = wl::redundant_circuit(n);

        let o3 = create_compiler(OptimizationLevel::O3);
        group.bench_with_input(BenchmarkId::new("o3", format!("{n}q")), &n, |b, _| {
            b.iter(|| {
                let mut circuit = circuit.clone();
                o3.compile(&mut circuit).unwrap();
                black_box(circuit.len())
            });
        });

        let o3_egraph = create_o3_egraph_compiler();
        group.bench_with_input(BenchmarkId::new("o3_egraph", format!("{n}q")), &n, |b, _| {
            b.iter(|| {
                let mut circuit = circuit.clone();
                o3_egraph.compile(&mut circuit).unwrap();
                black_box(circuit.len())
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_vqe_energy,
    bench_qaoa_maxcut,
    bench_ghz_sampling,
    bench_ghz_sampling_stabilizer,
    bench_vqe_energy_single_precision,
    bench_qaoa_cost_single_precision,
    bench_vqe_energy_pauli_propagation,
    bench_qaoa_cost_pauli_propagation,
    bench_near_clifford_pauli_propagation,
    bench_vqe_energy_mps,
    bench_qaoa_cost_mps,
    bench_qft_probe,
    bench_random_circuit,
    bench_redundant_circuit_compile,
    bench_multi_instance,
    bench_multi_instance_batched
);
criterion_main!(benches);
