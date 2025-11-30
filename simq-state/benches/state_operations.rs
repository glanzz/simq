//! Benchmarks for state vector operations
//!
//! Compares SIMD-optimized vs scalar implementations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_complex::Complex64;
use simq_state::simd::diagonal::apply_diagonal_gate_scalar;
use simq_state::simd::single_qubit::apply_gate_scalar;
use simq_state::simd::{apply_diagonal_gate, apply_single_qubit_gate, norm_simd, normalize_simd};
use simq_state::{DenseState, StateVector};

fn hadamard_matrix() -> [[Complex64; 2]; 2] {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    [
        [
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ],
        [
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(-inv_sqrt2, 0.0),
        ],
    ]
}

fn rotation_x_matrix(theta: f64) -> [[Complex64; 2]; 2] {
    let half_theta = theta / 2.0;
    let cos_val = half_theta.cos();
    let sin_val = half_theta.sin();

    [
        [Complex64::new(cos_val, 0.0), Complex64::new(0.0, -sin_val)],
        [Complex64::new(0.0, -sin_val), Complex64::new(cos_val, 0.0)],
    ]
}

fn phase_diagonal(theta: f64) -> [Complex64; 2] {
    [
        Complex64::new(1.0, 0.0),
        Complex64::new(theta.cos(), theta.sin()),
    ]
}

fn rotation_z_diagonal(theta: f64) -> [Complex64; 2] {
    let half_theta = theta / 2.0;
    [
        Complex64::new(half_theta.cos(), -half_theta.sin()),
        Complex64::new(half_theta.cos(), half_theta.sin()),
    ]
}

fn pauli_z_diagonal() -> [Complex64; 2] {
    [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)]
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

fn bench_dense_state_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_state_creation");

    for num_qubits in [10, 15, 20].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, &num_qubits| {
                b.iter(|| {
                    let state = DenseState::new(black_box(num_qubits)).unwrap();
                    black_box(state);
                })
            },
        );
    }

    group.finish();
}

fn bench_dense_state_gate_application(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_state_gate_application");

    for num_qubits in [10, 15, 20].iter() {
        let size = 1 << num_qubits;
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let mut state = DenseState::new(num_qubits).unwrap();
                let h = hadamard_matrix();

                b.iter(|| {
                    state
                        .apply_single_qubit_gate(black_box(&h), black_box(0))
                        .unwrap();
                })
            },
        );
    }

    group.finish();
}

fn bench_dense_state_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_state_measurement");

    for num_qubits in [10, 15, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("get_probability", num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let state = DenseState::new(num_qubits).unwrap();

                b.iter(|| {
                    let prob = state.get_probability(black_box(0)).unwrap();
                    black_box(prob);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("get_all_probabilities", num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let state = DenseState::new(num_qubits).unwrap();

                b.iter(|| {
                    let probs = state.get_all_probabilities();
                    black_box(probs);
                })
            },
        );
    }

    group.finish();
}

fn bench_diagonal_gate_vs_general(c: &mut Criterion) {
    let mut group = c.benchmark_group("diagonal_gate_comparison");

    for num_qubits in [10, 15, 20].iter() {
        let size = 1 << num_qubits;
        group.throughput(Throughput::Elements(size as u64));

        // Benchmark general gate application for Z gate
        group.bench_with_input(
            BenchmarkId::new("general_gate_z", num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let mut state = vec![Complex64::new(0.0, 0.0); 1 << num_qubits];
                state[0] = Complex64::new(1.0, 0.0);
                // Z gate as full matrix
                let z_matrix = [
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
                ];

                b.iter(|| {
                    apply_single_qubit_gate(black_box(&mut state), &z_matrix, 0, num_qubits);
                })
            },
        );

        // Benchmark diagonal optimized gate application for Z gate
        group.bench_with_input(
            BenchmarkId::new("diagonal_gate_z", num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let mut state = vec![Complex64::new(0.0, 0.0); 1 << num_qubits];
                state[0] = Complex64::new(1.0, 0.0);
                let z_diagonal = pauli_z_diagonal();

                b.iter(|| {
                    apply_diagonal_gate(black_box(&mut state), z_diagonal, 0, num_qubits);
                })
            },
        );

        // Benchmark general gate application for Phase gate
        group.bench_with_input(
            BenchmarkId::new("general_gate_phase", num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let mut state = vec![Complex64::new(0.0, 0.0); 1 << num_qubits];
                state[0] = Complex64::new(1.0, 0.0);
                let theta = std::f64::consts::PI / 4.0;
                // Phase gate as full matrix
                let phase_matrix = [
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [
                        Complex64::new(0.0, 0.0),
                        Complex64::new(theta.cos(), theta.sin()),
                    ],
                ];

                b.iter(|| {
                    apply_single_qubit_gate(black_box(&mut state), &phase_matrix, 0, num_qubits);
                })
            },
        );

        // Benchmark diagonal optimized gate application for Phase gate
        group.bench_with_input(
            BenchmarkId::new("diagonal_gate_phase", num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let mut state = vec![Complex64::new(0.0, 0.0); 1 << num_qubits];
                state[0] = Complex64::new(1.0, 0.0);
                let theta = std::f64::consts::PI / 4.0;
                let phase_diag = phase_diagonal(theta);

                b.iter(|| {
                    apply_diagonal_gate(black_box(&mut state), phase_diag, 0, num_qubits);
                })
            },
        );
    }

    group.finish();
}

fn bench_diagonal_gate_scalar_vs_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("diagonal_gate_scalar_vs_simd");

    for num_qubits in [10, 15, 20].iter() {
        let size = 1 << num_qubits;
        group.throughput(Throughput::Elements(size as u64));

        // Benchmark scalar diagonal gate
        group.bench_with_input(
            BenchmarkId::new("scalar", num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let mut state = vec![Complex64::new(0.0, 0.0); 1 << num_qubits];
                state[0] = Complex64::new(1.0, 0.0);
                let theta = std::f64::consts::PI / 4.0;
                let rz_diag = rotation_z_diagonal(theta);

                b.iter(|| {
                    apply_diagonal_gate_scalar(black_box(&mut state), rz_diag, 0, num_qubits);
                })
            },
        );

        // Benchmark SIMD optimized diagonal gate
        group.bench_with_input(
            BenchmarkId::new("simd", num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let mut state = vec![Complex64::new(0.0, 0.0); 1 << num_qubits];
                state[0] = Complex64::new(1.0, 0.0);
                let theta = std::f64::consts::PI / 4.0;
                let rz_diag = rotation_z_diagonal(theta);

                b.iter(|| {
                    apply_diagonal_gate(black_box(&mut state), rz_diag, 0, num_qubits);
                })
            },
        );
    }

    group.finish();
}

fn bench_dense_state_diagonal_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_state_diagonal_gates");

    for num_qubits in [10, 15, 20].iter() {
        let size = 1 << num_qubits;
        group.throughput(Throughput::Elements(size as u64));

        // Benchmark using general gate method
        group.bench_with_input(
            BenchmarkId::new("apply_single_qubit_gate", num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let mut state = DenseState::new(num_qubits).unwrap();
                let theta = std::f64::consts::PI / 4.0;
                let rz_matrix = [
                    [
                        Complex64::new(theta.cos(), -theta.sin()),
                        Complex64::new(0.0, 0.0),
                    ],
                    [
                        Complex64::new(0.0, 0.0),
                        Complex64::new(theta.cos(), theta.sin()),
                    ],
                ];

                b.iter(|| {
                    state
                        .apply_single_qubit_gate(black_box(&rz_matrix), black_box(0))
                        .unwrap();
                })
            },
        );

        // Benchmark using optimized diagonal gate method
        group.bench_with_input(
            BenchmarkId::new("apply_diagonal_gate", num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let mut state = DenseState::new(num_qubits).unwrap();
                let theta = std::f64::consts::PI / 4.0;
                let rz_diag = rotation_z_diagonal(theta);

                b.iter(|| {
                    state
                        .apply_diagonal_gate(black_box(rz_diag), black_box(0))
                        .unwrap();
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
    bench_diagonal_gate_vs_general,
    bench_diagonal_gate_scalar_vs_simd,
    bench_dense_state_diagonal_gates,
    bench_norm,
    bench_state_vector_creation,
    bench_dense_state_creation,
    bench_dense_state_gate_application,
    bench_dense_state_measurement,
);
criterion_main!(benches);
