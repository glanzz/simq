//! Benchmark parallel scaling for SimQ simulator

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use simq_sim::{Simulator, config::SimulatorConfig};
use simq_core::{Circuit, QubitId};
use simq_gates::standard::Hadamard;
use std::sync::Arc;

fn generate_circuit(num_qubits: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);
    for d in 0..depth {
        for q in 0..num_qubits {
            circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
        }
    }
    circuit
}

fn bench_parallel_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("ParallelScaling");
    let qubit_sizes = [10, 12, 14, 16, 18, 20];
    let depth = 20;
    let thread_counts = [1, 2, 4, 8, 16];
    let thresholds = [0, 4, 8, 16];

    for &num_qubits in &qubit_sizes {
        let circuit = generate_circuit(num_qubits, depth);

        // Single-threaded
        let config = SimulatorConfig {
            parallel_threshold: usize::MAX, // disables parallelism
            use_gpu: false,
            ..Default::default()
        };
        let simulator = Simulator::new(config.clone());
        group.bench_with_input(BenchmarkId::new("CPU_Serial", num_qubits), &num_qubits, |b, &_|
            b.iter(|| simulator.run(&circuit)));

        // Rayon parallel: sweep thread counts and thresholds
        for &threads in &thread_counts {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();
            for &threshold in &thresholds {
                let config = SimulatorConfig {
                    parallel_threshold: threshold,
                    use_gpu: false,
                    ..Default::default()
                };
                let simulator = Simulator::new(config.clone());
                let label = format!("CPU_Rayon_{}thr_thresh{}", threads, threshold);
                group.bench_with_input(BenchmarkId::new(label, num_qubits), &num_qubits, |b, &_|
                    b.iter(|| simulator.run(&circuit)));
            }
        }

        // GPU (if available)
        let config = SimulatorConfig {
            parallel_threshold: 0,
            use_gpu: true,
            ..Default::default()
        };
        let simulator = Simulator::new(config.clone());
        group.bench_with_input(BenchmarkId::new("GPU", num_qubits), &num_qubits, |b, &_|
            b.iter(|| simulator.run(&circuit)));
    }
    group.finish();
}

criterion_group!(benches, bench_parallel_scaling);
criterion_main!(benches);
