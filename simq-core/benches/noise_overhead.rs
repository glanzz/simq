use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use simq_core::noise::{
    HardwareNoiseModel, QubitTimeTracker, DepolarizingChannel, AmplitudeDamping,
    PhaseDamping, NoiseChannel, GateTiming,
};

/// Benchmark single-qubit gate noise generation
fn bench_single_qubit_noise_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_qubit_noise_generation");

    for num_qubits in [5, 10, 20, 50, 100].iter() {
        let model = HardwareNoiseModel::ibm_washington();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("ibm_washington", num_qubits),
            num_qubits,
            |b, _| {
                b.iter(|| {
                    let noise = model.single_qubit_gate_noise(black_box(0)).unwrap();
                    black_box(noise);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark two-qubit gate noise generation
fn bench_two_qubit_noise_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("two_qubit_noise_generation");

    for num_qubits in [5, 10, 20, 50, 100].iter() {
        let mut model = HardwareNoiseModel::new(*num_qubits);
        model.set_two_qubit_gate(0, 1, 0.99, 0.3);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("with_config", num_qubits),
            num_qubits,
            |b, _| {
                b.iter(|| {
                    let noise = model.two_qubit_gate_noise(black_box(0), black_box(1)).unwrap();
                    black_box(noise);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark idle noise calculation
fn bench_idle_noise_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("idle_noise_calculation");

    let model = HardwareNoiseModel::ibm_washington();

    for idle_time in [0.1, 1.0, 10.0, 100.0].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("ibm_washington", idle_time),
            idle_time,
            |b, &time| {
                b.iter(|| {
                    let (amp, phase) = model.idle_noise(black_box(0), black_box(time)).unwrap();
                    black_box((amp, phase));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark time tracking overhead
fn bench_time_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("time_tracking");

    for num_qubits in [5, 10, 20, 50, 100].iter() {
        let timing = GateTiming::default();
        let mut tracker = QubitTimeTracker::new(*num_qubits, timing);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("single_gate", num_qubits),
            num_qubits,
            |b, _| {
                b.iter(|| {
                    tracker.apply_single_qubit_gate(black_box(0));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark idle time calculation
fn bench_idle_time_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("idle_time_calculation");

    for num_qubits in [5, 10, 20, 50, 100].iter() {
        let timing = GateTiming::default();
        let mut tracker = QubitTimeTracker::new(*num_qubits, timing);

        // Create some time differential
        tracker.apply_single_qubit_gate(0);
        tracker.apply_two_qubit_gate(1, 2);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("single_qubit", num_qubits),
            num_qubits,
            |b, _| {
                b.iter(|| {
                    let idle = tracker.idle_time_since_last_operation(black_box(3));
                    black_box(idle);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark all idle times calculation
fn bench_all_idle_times(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_idle_times");

    for num_qubits in [5, 10, 20, 50, 100].iter() {
        let timing = GateTiming::default();
        let mut tracker = QubitTimeTracker::new(*num_qubits, timing);

        tracker.apply_single_qubit_gate(0);

        group.throughput(Throughput::Elements(*num_qubits as u64));
        group.bench_with_input(
            BenchmarkId::new("all_qubits", num_qubits),
            num_qubits,
            |b, _| {
                b.iter(|| {
                    let idle_times = tracker.all_idle_times();
                    black_box(idle_times);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Kraus operator generation
fn bench_kraus_operator_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("kraus_operator_generation");

    group.bench_function("depolarizing", |b| {
        b.iter(|| {
            let channel = DepolarizingChannel::new(black_box(0.01)).unwrap();
            let kraus = channel.kraus_operators();
            black_box(kraus);
        });
    });

    group.bench_function("amplitude_damping", |b| {
        b.iter(|| {
            let channel = AmplitudeDamping::new(black_box(0.02)).unwrap();
            let kraus = channel.kraus_operators();
            black_box(kraus);
        });
    });

    group.bench_function("phase_damping", |b| {
        b.iter(|| {
            let channel = PhaseDamping::new(black_box(0.015)).unwrap();
            let kraus = channel.kraus_operators();
            black_box(kraus);
        });
    });

    group.finish();
}

/// Benchmark device preset creation
fn bench_device_preset_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_preset_creation");

    group.bench_function("ibm_washington_127q", |b| {
        b.iter(|| {
            let model = HardwareNoiseModel::ibm_washington();
            black_box(model);
        });
    });

    group.bench_function("google_sycamore_53q", |b| {
        b.iter(|| {
            let model = HardwareNoiseModel::google_sycamore();
            black_box(model);
        });
    });

    group.bench_function("ionq_aria_25q", |b| {
        b.iter(|| {
            let model = HardwareNoiseModel::ionq_aria();
            black_box(model);
        });
    });

    group.bench_function("ibm_falcon_5q", |b| {
        b.iter(|| {
            let model = HardwareNoiseModel::ibm_falcon_5q();
            black_box(model);
        });
    });

    group.finish();
}

/// Benchmark circuit fidelity estimation
fn bench_circuit_fidelity_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_fidelity_estimation");

    let model = HardwareNoiseModel::ibm_washington();

    for circuit_depth in [10, 50, 100, 500].iter() {
        let single_gates = vec![*circuit_depth; 5];
        let two_gates: Vec<(usize, usize)> = (0..*circuit_depth).map(|i| (i % 4, (i % 4) + 1)).collect();
        let total_time = (*circuit_depth as f64) * 0.1;

        group.throughput(Throughput::Elements(*circuit_depth as u64));
        group.bench_with_input(
            BenchmarkId::new("depth", circuit_depth),
            circuit_depth,
            |b, _| {
                b.iter(|| {
                    let fidelity = model.estimate_circuit_fidelity(
                        black_box(&single_gates),
                        black_box(&two_gates),
                        black_box(total_time),
                    );
                    black_box(fidelity);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark synchronization operations
fn bench_synchronization(c: &mut Criterion) {
    let mut group = c.benchmark_group("synchronization");

    for num_qubits in [5, 10, 20, 50, 100].iter() {
        let timing = GateTiming::default();
        let mut tracker = QubitTimeTracker::new(*num_qubits, timing);

        // Create time differential
        for i in 0..*num_qubits {
            tracker.apply_single_qubit_gate(i % 3);
        }

        group.throughput(Throughput::Elements(*num_qubits as u64));
        group.bench_with_input(
            BenchmarkId::new("sync_all", num_qubits),
            num_qubits,
            |b, _| {
                b.iter(|| {
                    tracker.synchronize_all_qubits();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark complete noise workflow
fn bench_complete_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_workflow");

    for num_qubits in [5, 10, 20].iter() {
        let model = HardwareNoiseModel::ibm_washington();
        let timing = model.timing().clone();
        let mut tracker = QubitTimeTracker::new(*num_qubits, timing);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("single_gate_full", num_qubits),
            num_qubits,
            |b, _| {
                b.iter(|| {
                    // Simulate a single-qubit gate with full noise workflow
                    let qubit = 0;

                    // 1. Get idle noise for other qubits
                    for q in 1..*num_qubits {
                        let idle_time = tracker.idle_time_since_last_operation(q);
                        if idle_time > 0.0 {
                            let _idle_noise = model.idle_noise(q, idle_time).unwrap();
                        }
                    }

                    // 2. Get gate noise
                    let _gate_noise = model.single_qubit_gate_noise(qubit).unwrap();

                    // 3. Update time tracker
                    tracker.apply_single_qubit_gate(qubit);
                    tracker.synchronize_qubits(&[qubit]);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_qubit_noise_generation,
    bench_two_qubit_noise_generation,
    bench_idle_noise_calculation,
    bench_time_tracking,
    bench_idle_time_calculation,
    bench_all_idle_times,
    bench_kraus_operator_generation,
    bench_device_preset_creation,
    bench_circuit_fidelity_estimation,
    bench_synchronization,
    bench_complete_workflow,
);

criterion_main!(benches);
