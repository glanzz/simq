use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use simq_core::{Circuit, QubitId};
use simq_gates::standard::{Hadamard, PauliX, CNot, RotationZ};
use simq_sim::Simulator;
use std::sync::Arc;
use simq_core::gate::Gate;

fn create_random_circuit(num_qubits: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::with_capacity(num_qubits, num_qubits * depth);

    for d in 0..depth {
        // Single qubit gates layer
        for i in 0..num_qubits {
            let qubit = QubitId::new(i);
            if (i + d) % 2 == 0 {
                circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[qubit]).unwrap();
            } else {
                circuit.add_gate(Arc::new(RotationZ::new(0.5)) as Arc<dyn Gate>, &[qubit]).unwrap();
            }
        }

        // Entangling layer
        for i in 0..(num_qubits - 1) {
            if (i + d) % 2 == 0 {
                let q1 = QubitId::new(i);
                let q2 = QubitId::new(i + 1);
                circuit.add_gate(Arc::new(CNot) as Arc<dyn Gate>, &[q1, q2]).unwrap();
            }
        }
    }

    circuit
}

fn create_ghz_circuit(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::with_capacity(num_qubits, num_qubits);
    
    // H on first qubit
    circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[QubitId::new(0)]).unwrap();
    
    // CNOT chain
    for i in 0..(num_qubits - 1) {
        let q1 = QubitId::new(i);
        let q2 = QubitId::new(i + 1);
        circuit.add_gate(Arc::new(CNot) as Arc<dyn Gate>, &[q1, q2]).unwrap();
    }
    
    circuit
}

fn bench_full_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_simulation");
    
    // Benchmark Random Circuits
    for num_qubits in [10, 15, 20].iter() {
        let depth = 20;
        let circuit = create_random_circuit(*num_qubits, depth);
        
        group.bench_with_input(
            BenchmarkId::new("random_circuit", format!("{}q_d{}", num_qubits, depth)),
            &circuit,
            |b, circuit| {
                b.iter(|| {
                    let mut sim = Simulator::new(black_box(*num_qubits));
                    sim.run(black_box(circuit)).unwrap();
                })
            },
        );
    }

    // Benchmark GHZ State Preparation
    for num_qubits in [10, 15, 20, 25].iter() {
        let circuit = create_ghz_circuit(*num_qubits);
        
        group.bench_with_input(
            BenchmarkId::new("ghz_prep", format!("{}q", num_qubits)),
            &circuit,
            |b, circuit| {
                b.iter(|| {
                    let mut sim = Simulator::new(black_box(*num_qubits));
                    sim.run(black_box(circuit)).unwrap();
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_full_simulation);
criterion_main!(benches);
