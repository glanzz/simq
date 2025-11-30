use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use simq_core::{Circuit, QubitId};
use simq_gates::standard::{RotationX, RotationZ, CNot, Hadamard};
use simq_sim::Simulator;
use std::sync::Arc;
use simq_core::gate::Gate;

// Create a parameterized ansatz (Hardware Efficient Ansatz)
fn create_ansatz(num_qubits: usize, depth: usize, params: &[f64]) -> Circuit {
    let mut circuit = Circuit::with_capacity(num_qubits, num_qubits * depth * 2);
    let mut param_idx = 0;

    for d in 0..depth {
        // Rotation layer
        for i in 0..num_qubits {
            let qubit = QubitId::new(i);
            circuit.add_gate(Arc::new(RotationX::new(params[param_idx])) as Arc<dyn Gate>, &[qubit]).unwrap();
            param_idx += 1;
            circuit.add_gate(Arc::new(RotationZ::new(params[param_idx])) as Arc<dyn Gate>, &[qubit]).unwrap();
            param_idx += 1;
        }

        // Entangling layer
        if d < depth - 1 {
            for i in 0..(num_qubits - 1) {
                let q1 = QubitId::new(i);
                let q2 = QubitId::new(i + 1);
                circuit.add_gate(Arc::new(CNot) as Arc<dyn Gate>, &[q1, q2]).unwrap();
            }
        }
    }

    circuit
}

fn bench_vqe_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("vqe_iteration");

    for num_qubits in [4, 8, 12].iter() {
        let depth = 3;
        let num_params = num_qubits * depth * 2;
        let params: Vec<f64> = (0..num_params).map(|i| (i as f64) * 0.1).collect();
        
        // Pre-create circuit to measure just the execution time (simulating one VQE step where circuit is updated)
        // In a real VQE, we'd update parameters, but for benchmarking the simulation cost, we can just run it.
        // Or better, we can include the circuit creation cost if we want to measure "full iteration including transpilation"
        // but usually the simulation is the bottleneck. Let's measure simulation only for now as "energy evaluation".
        
        let circuit = create_ansatz(*num_qubits, depth, &params);

        group.bench_with_input(
            BenchmarkId::new("energy_eval", format!("{}q_d{}", num_qubits, depth)),
            &circuit,
            |b, circuit| {
                b.iter(|| {
                    let mut sim = Simulator::new(black_box(*num_qubits));
                    sim.run(black_box(circuit)).unwrap();
                    // In a real VQE we would compute expectation value here.
                    // For now, let's assume measuring all qubits is part of the cost
                    let _probs = sim.get_probabilities(); 
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_vqe_iteration);
criterion_main!(benches);
