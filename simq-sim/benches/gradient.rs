use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use simq_core::{Circuit, QubitId};
use simq_gates::standard::{RotationX, RotationZ, CNot};
use simq_sim::Simulator;
use std::sync::Arc;
use simq_core::gate::Gate;

// Helper to create a parameterized circuit
fn create_param_circuit(num_qubits: usize, depth: usize, params: &[f64]) -> Circuit {
    let mut circuit = Circuit::with_capacity(num_qubits, num_qubits * depth * 2);
    let mut param_idx = 0;

    for d in 0..depth {
        for i in 0..num_qubits {
            let qubit = QubitId::new(i);
            circuit.add_gate(Arc::new(RotationX::new(params[param_idx])) as Arc<dyn Gate>, &[qubit]).unwrap();
            param_idx += 1;
        }

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

fn bench_gradient_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_computation");

    for num_qubits in [4, 8].iter() {
        let depth = 2;
        let num_params = num_qubits * depth; // One RX per qubit per layer
        let params: Vec<f64> = (0..num_params).map(|i| (i as f64) * 0.1).collect();
        
        // Parameter Shift Rule Benchmark
        // For each parameter, we need 2 evaluations (f(x+s) - f(x-s)) / 2s
        // Total evaluations = 2 * num_params
        
        group.bench_with_input(
            BenchmarkId::new("parameter_shift", format!("{}q_d{}_{}params", num_qubits, depth, num_params)),
            &params,
            |b, params| {
                b.iter(|| {
                    // Simulate gradient computation via parameter shift
                    let mut gradients = vec![0.0; params.len()];
                    
                    for i in 0..params.len() {
                        // Shift +
                        let mut params_plus = params.to_vec();
                        params_plus[i] += std::f64::consts::PI / 2.0;
                        let circuit_plus = create_param_circuit(*num_qubits, depth, &params_plus);
                        let mut sim_plus = Simulator::new(black_box(*num_qubits));
                        sim_plus.run(black_box(circuit_plus)).unwrap();
                        let _energy_plus = sim_plus.get_probability(0).unwrap(); // Mock energy
                        
                        // Shift -
                        let mut params_minus = params.to_vec();
                        params_minus[i] -= std::f64::consts::PI / 2.0;
                        let circuit_minus = create_param_circuit(*num_qubits, depth, &params_minus);
                        let mut sim_minus = Simulator::new(black_box(*num_qubits));
                        sim_minus.run(black_box(circuit_minus)).unwrap();
                        let _energy_minus = sim_minus.get_probability(0).unwrap(); // Mock energy
                        
                        gradients[i] = (_energy_plus - _energy_minus) / 2.0;
                    }
                    
                    black_box(gradients);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_gradient_computation);
criterion_main!(benches);
