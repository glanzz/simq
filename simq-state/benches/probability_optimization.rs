//! Benchmark for probability distribution computation optimization
//!
//! Compares SIMD-optimized vs scalar implementation for computing
//! probability distributions from quantum state amplitudes.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use num_complex::Complex64;
use simq_state::DenseState;

/// Create a random quantum state for benchmarking
fn create_random_state(num_qubits: usize, seed: u64) -> DenseState {
    let dimension = 1 << num_qubits;
    let mut rng_state = seed;

    let amplitudes: Vec<Complex64> = (0..dimension)
        .map(|_| {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let re = ((rng_state / 65536) % 32768) as f64 / 32768.0 - 0.5;
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let im = ((rng_state / 65536) % 32768) as f64 / 32768.0 - 0.5;
            Complex64::new(re, im)
        })
        .collect();

    let mut state = DenseState::from_amplitudes(num_qubits, &amplitudes).unwrap();
    state.normalize();
    state
}

/// Benchmark probability distribution computation (SIMD-optimized)
fn bench_probability_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("probability_computation_simd");

    for num_qubits in [5, 10, 15, 20].iter() {
        let state = create_random_state(*num_qubits, 42);
        let dimension = 1 << num_qubits;

        group.throughput(Throughput::Elements(dimension as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}q", num_qubits)),
            num_qubits,
            |b, _| {
                b.iter(|| {
                    let probs = state.get_all_probabilities();
                    black_box(probs);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark probability distribution computation (scalar fallback)
fn bench_probability_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("probability_computation_scalar");

    for num_qubits in [5, 10, 15, 20].iter() {
        let state = create_random_state(*num_qubits, 42);
        let dimension = 1 << num_qubits;

        group.throughput(Throughput::Elements(dimension as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}q", num_qubits)),
            num_qubits,
            |b, _| {
                b.iter(|| {
                    // Scalar implementation
                    let probs: Vec<f64> = state
                        .amplitudes()
                        .iter()
                        .map(|amp| amp.norm_sqr())
                        .collect();
                    black_box(probs);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark full sampling workflow (includes probability + alias table)
fn bench_sampling_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_workflow");

    for num_qubits in [8, 10, 12, 15].iter() {
        let state = create_random_state(*num_qubits, 42);
        let dimension = 1 << num_qubits;

        group.throughput(Throughput::Elements(dimension as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}q", num_qubits)),
            num_qubits,
            |b, _| {
                b.iter(|| {
                    // Simulate measurement setup
                    let probs = state.get_all_probabilities();
                    black_box(probs);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark just the SIMD kernel directly
fn bench_simd_kernel_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_kernel_direct");

    for num_qubits in [5, 10, 15, 20].iter() {
        let state = create_random_state(*num_qubits, 42);
        let dimension = 1 << num_qubits;
        let amplitudes = state.amplitudes();
        let mut output = vec![0.0; dimension];

        group.throughput(Throughput::Elements(dimension as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}q", num_qubits)),
            num_qubits,
            |b, _| {
                b.iter(|| {
                    simq_state::simd::kernels::compute_probabilities(
                        black_box(amplitudes),
                        black_box(&mut output),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory allocation overhead
fn bench_with_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("probability_with_allocation");

    for num_qubits in [10, 12, 15].iter() {
        let state = create_random_state(*num_qubits, 42);
        let dimension = 1 << num_qubits;

        group.throughput(Throughput::Elements(dimension as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}q", num_qubits)),
            num_qubits,
            |b, _| {
                b.iter(|| {
                    // Full workflow including allocation
                    let probs = state.get_all_probabilities();
                    black_box(probs);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark without allocation (reuse buffer)
fn bench_without_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("probability_no_allocation");

    for num_qubits in [10, 12, 15].iter() {
        let state = create_random_state(*num_qubits, 42);
        let dimension = 1 << num_qubits;
        let amplitudes = state.amplitudes();
        let mut output = vec![0.0; dimension];

        group.throughput(Throughput::Elements(dimension as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}q", num_qubits)),
            num_qubits,
            |b, _| {
                b.iter(|| {
                    simq_state::simd::kernels::compute_probabilities(
                        black_box(amplitudes),
                        black_box(&mut output),
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_probability_simd,
    bench_probability_scalar,
    bench_sampling_workflow,
    bench_simd_kernel_direct,
    bench_with_allocation,
    bench_without_allocation
);
criterion_main!(benches);
