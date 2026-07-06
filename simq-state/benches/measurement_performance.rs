use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_complex::Complex64;
use simq_state::{ComputationalBasis, DenseState};

// Linear congruential generator for reproducible benchmarks
struct BenchRng {
    state: u64,
}

impl BenchRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.state / 65536) % 32768) as f64 / 32768.0
    }
}

fn create_random_state(num_qubits: usize, seed: u64) -> DenseState {
    let dimension = 1 << num_qubits;
    let mut rng = BenchRng::new(seed);

    let amplitudes: Vec<Complex64> = (0..dimension)
        .map(|_| Complex64::new(rng.next() - 0.5, rng.next() - 0.5))
        .collect();

    let mut state = DenseState::from_amplitudes(num_qubits, &amplitudes).unwrap();
    state.normalize();
    state
}

fn bench_single_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_measurement");

    for num_qubits in [5, 10, 15, 20].iter() {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_qubits", num_qubits)),
            num_qubits,
            |b, &num_qubits| {
                let state = create_random_state(num_qubits, 42);
                let measurement = ComputationalBasis::new();
                let mut rng = BenchRng::new(123);

                b.iter(|| {
                    let mut state_copy = state.clone_state().unwrap();
                    measurement
                        .measure_once(black_box(&mut state_copy), &mut || rng.next())
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_batch_sampling_shots(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_sampling_shots");

    let num_qubits = 10;
    let state = create_random_state(num_qubits, 42);
    let measurement = ComputationalBasis::new().with_collapse(false);

    for &shots in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(shots as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_shots", shots)),
            &shots,
            |b, &shots| {
                let mut rng = BenchRng::new(123);

                b.iter(|| {
                    measurement
                        .sample(black_box(&state), shots, &mut || rng.next())
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_batch_sampling_qubits(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_sampling_qubits");

    let shots: usize = 1000;

    for &num_qubits in [5, 10, 15, 20].iter() {
        group.throughput(Throughput::Elements(shots as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_qubits", num_qubits)),
            &num_qubits,
            |b, &num_qubits| {
                let state = create_random_state(num_qubits, 42);
                let measurement = ComputationalBasis::new().with_collapse(false);
                let mut rng = BenchRng::new(123);

                b.iter(|| {
                    measurement
                        .sample(black_box(&state), shots, &mut || rng.next())
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_probability_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("probability_computation");

    for num_qubits in [5, 10, 15, 20].iter() {
        let dimension = 1 << num_qubits;
        group.throughput(Throughput::Elements(dimension as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_qubits", num_qubits)),
            num_qubits,
            |b, &num_qubits| {
                let state = create_random_state(num_qubits, 42);

                b.iter(|| black_box(state.get_all_probabilities()));
            },
        );
    }

    group.finish();
}

fn bench_alias_table_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("alias_table_construction");

    for num_qubits in [5, 10, 15, 20].iter() {
        let dimension = 1 << num_qubits;
        group.throughput(Throughput::Elements(dimension as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_qubits", num_qubits)),
            num_qubits,
            |b, &num_qubits| {
                let state = create_random_state(num_qubits, 42);
                let probabilities = state.get_all_probabilities();

                b.iter(|| {
                    // Simulate alias table construction time
                    // (actual AliasTable is private, so we measure the sample operation)
                    let measurement = ComputationalBasis::new().with_collapse(false);
                    let mut rng = BenchRng::new(123);
                    measurement
                        .sample(black_box(&state), 1, &mut || rng.next())
                        .unwrap();
                    black_box(&probabilities)
                });
            },
        );
    }

    group.finish();
}

fn bench_sampling_vs_repeated_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_vs_repeated");

    let num_qubits = 10;
    let shots = 1000;
    let state = create_random_state(num_qubits, 42);

    group.throughput(Throughput::Elements(shots as u64));

    // Batch sampling
    group.bench_function("batch_sampling", |b| {
        let measurement = ComputationalBasis::new().with_collapse(false);
        let mut rng = BenchRng::new(123);

        b.iter(|| {
            measurement
                .sample(black_box(&state), shots, &mut || rng.next())
                .unwrap()
        });
    });

    // Repeated individual measurements
    group.bench_function("repeated_individual", |b| {
        let measurement = ComputationalBasis::new().with_collapse(false);
        let mut rng = BenchRng::new(123);

        b.iter(|| {
            let mut counts = std::collections::HashMap::new();
            for _ in 0..shots {
                let mut state_copy = state.clone_state().unwrap();
                let result = measurement
                    .measure_once(black_box(&mut state_copy), &mut || rng.next())
                    .unwrap();
                *counts.entry(result.outcome).or_insert(0) += 1;
            }
            counts
        });
    });

    group.finish();
}

fn bench_sparse_vs_dense_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vs_dense_state");

    let num_qubits = 15;
    let shots = 1000;

    // Dense state (uniform superposition)
    let dimension = 1 << num_qubits;
    let amplitude = Complex64::new(1.0 / (dimension as f64).sqrt(), 0.0);
    let dense_amplitudes = vec![amplitude; dimension];
    let dense_state = DenseState::from_amplitudes(num_qubits, &dense_amplitudes).unwrap();

    // Sparse state (mostly zeros, a few non-zero amplitudes)
    let mut sparse_amplitudes = vec![Complex64::new(0.0, 0.0); dimension];
    sparse_amplitudes[0] = Complex64::new(0.7, 0.0);
    sparse_amplitudes[1] = Complex64::new(0.7, 0.0);
    sparse_amplitudes[100] = Complex64::new(0.1, 0.0);
    let mut sparse_state = DenseState::from_amplitudes(num_qubits, &sparse_amplitudes).unwrap();
    sparse_state.normalize();

    group.throughput(Throughput::Elements(shots as u64));

    group.bench_function("dense_state", |b| {
        let measurement = ComputationalBasis::new().with_collapse(false);
        let mut rng = BenchRng::new(123);

        b.iter(|| {
            measurement
                .sample(black_box(&dense_state), shots, &mut || rng.next())
                .unwrap()
        });
    });

    group.bench_function("sparse_state", |b| {
        let measurement = ComputationalBasis::new().with_collapse(false);
        let mut rng = BenchRng::new(123);

        b.iter(|| {
            measurement
                .sample(black_box(&sparse_state), shots, &mut || rng.next())
                .unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_measurement,
    bench_batch_sampling_shots,
    bench_batch_sampling_qubits,
    bench_probability_computation,
    bench_alias_table_construction,
    bench_sampling_vs_repeated_measurement,
    bench_sparse_vs_dense_state
);

criterion_main!(benches);
