//! Benchmarks for state vector operations
//!
//! Compares SIMD-optimized vs scalar implementations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_complex::Complex64;
use simq_state::simd::{apply_single_qubit_gate, norm_simd, normalize_simd};
use simq_state::simd::single_qubit::apply_gate_scalar;
use simq_state::StateVector;

fn hadamard_matrix() -> [[Complex64; 2]; 2] {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    [
        [Complex64::new(inv_sqrt2, 0.0), Complex64::new(inv_sqrt2, 0.0)],
        [Complex64::new(inv_sqrt2, 0.0), Complex64::new(-inv_sqrt2, 0.0)],
    ]
}

fn rotation_x_matrix(theta: f64) -> [[Complex64; 2]; 2] {
    let half_theta = theta / 2.0;
    let cos_val = half_theta.cos();
    let sin_val = half_theta.sin();

    [
        [
            Complex64::new(cos_val, 0.0),
            Complex64::new(0.0, -sin_val),
        ],
        [
            Complex64::new(0.0, -sin_val),
            Complex64::new(cos_val, 0.0),
        ],
    ]
}

fn bench_single_qubit_gate_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_qubit_gate_scalar");

    for num_qubits in [10, 15, 20].iter() {
        let size = 1 << num_qubits;
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let mut state = vec![Complex64::new(0.0, 0.0); 1 << num_qubits];
                state[0] = Complex64::new(1.0, 0.0);
                let h = hadamard_matrix();

                b.iter(|| {
                    apply_gate_scalar(black_box(&mut state), &h, 0, num_qubits);
                })
            },
        );
    }

    group.finish();
}

fn bench_single_qubit_gate_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_qubit_gate_simd");

    for num_qubits in [10, 15, 20].iter() {
        let size = 1 << num_qubits;
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let mut state = vec![Complex64::new(0.0, 0.0); 1 << num_qubits];
                state[0] = Complex64::new(1.0, 0.0);
                let h = hadamard_matrix();

                b.iter(|| {
                    apply_single_qubit_gate(black_box(&mut state), &h, 0, num_qubits);
                })
            },
        );
    }

    group.finish();
}

fn bench_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("norm");

    for num_qubits in [10, 15, 20].iter() {
        let size = 1 << num_qubits;
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar", num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let state = vec![Complex64::new(1.0, 0.0); 1 << num_qubits];

                b.iter(|| {
                    let norm: f64 = black_box(&state)
                        .iter()
                        .map(|z| z.norm_sqr())
                        .sum::<f64>()
                        .sqrt();
                    black_box(norm);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd", num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let state = vec![Complex64::new(1.0, 0.0); 1 << num_qubits];

                b.iter(|| {
                    let norm = norm_simd(black_box(&state));
                    black_box(norm);
                })
            },
        );
    }

    group.finish();
}

fn bench_state_vector_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_vector_creation");

    for num_qubits in [10, 15, 20].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, &num_qubits| {
                b.iter(|| {
                    let state = StateVector::new(black_box(num_qubits)).unwrap();
                    black_box(state);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_qubit_gate_scalar,
    bench_single_qubit_gate_simd,
    bench_norm,
    bench_state_vector_creation,
);
criterion_main!(benches);
