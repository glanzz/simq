use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use simq_compiler::fusion::fuse_single_qubit_gates;
use simq_core::{Circuit, QubitId};
use simq_gates::standard::{Hadamard, PauliX, PauliY, PauliZ, RotationX, RotationZ, SGate, TGate};
use std::sync::Arc;
use simq_core::gate::Gate;

/// Create a circuit with many single-qubit gates that can be fused
fn create_fuseable_circuit(num_qubits: usize, gates_per_qubit: usize) -> Circuit {
    let mut circuit = Circuit::with_capacity(num_qubits, num_qubits * gates_per_qubit);

    let gates: Vec<Arc<dyn Gate>> = vec![
        Arc::new(Hadamard),
        Arc::new(PauliX),
        Arc::new(PauliY),
        Arc::new(PauliZ),
        Arc::new(SGate),
        Arc::new(TGate),
    ];

    for qubit_idx in 0..num_qubits {
        let qubit = QubitId::new(qubit_idx);
        for i in 0..gates_per_qubit {
            let gate = Arc::clone(&gates[i % gates.len()]);
            circuit.add_gate(gate, &[qubit]).unwrap();
        }
    }

    circuit
}

/// Create a circuit with parameterized rotation gates
fn create_rotation_circuit(num_qubits: usize, rotations_per_qubit: usize) -> Circuit {
    let mut circuit = Circuit::with_capacity(num_qubits, num_qubits * rotations_per_qubit);

    for qubit_idx in 0..num_qubits {
        let qubit = QubitId::new(qubit_idx);
        for i in 0..rotations_per_qubit {
            let angle = (i as f64) * 0.1;
            if i % 2 == 0 {
                circuit.add_gate(Arc::new(RotationX::new(angle)) as Arc<dyn Gate>, &[qubit]).unwrap();
            } else {
                circuit.add_gate(Arc::new(RotationZ::new(angle)) as Arc<dyn Gate>, &[qubit]).unwrap();
            }
        }
    }

    circuit
}

/// Create a circuit with mixed single and two-qubit gates
fn create_mixed_circuit(num_qubits: usize, gates_per_qubit: usize) -> Circuit {
    let mut circuit = Circuit::with_capacity(num_qubits, num_qubits * gates_per_qubit);

    for qubit_idx in 0..num_qubits {
        let qubit = QubitId::new(qubit_idx);

        // Add some single-qubit gates
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[qubit]).unwrap();
        circuit.add_gate(Arc::new(TGate) as Arc<dyn Gate>, &[qubit]).unwrap();
        circuit.add_gate(Arc::new(SGate) as Arc<dyn Gate>, &[qubit]).unwrap();

        // Add a two-qubit gate if possible (breaks fusion chain)
        if qubit_idx + 1 < num_qubits {
            let next_qubit = QubitId::new(qubit_idx + 1);
            circuit.add_gate(
                Arc::new(simq_gates::standard::CNot) as Arc<dyn Gate>,
                &[qubit, next_qubit]
            ).unwrap();
        }

        // Add more single-qubit gates
        circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[qubit]).unwrap();
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[qubit]).unwrap();
    }

    circuit
}

fn bench_gate_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("gate_fusion");

    // Benchmark different circuit sizes
    for num_qubits in [5, 10, 20, 50].iter() {
        for gates_per_qubit in [10, 20, 50].iter() {
            let circuit = create_fuseable_circuit(*num_qubits, *gates_per_qubit);
            let total_gates = circuit.len();

            group.bench_with_input(
                BenchmarkId::new("fuseable", format!("{}q_{}g_total{}", num_qubits, gates_per_qubit, total_gates)),
                &circuit,
                |b, circuit| {
                    b.iter(|| {
                        let optimized = fuse_single_qubit_gates(black_box(circuit), None).unwrap();
                        black_box(optimized)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_rotation_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotation_fusion");

    for num_qubits in [5, 10, 20].iter() {
        for rotations in [10, 20, 30].iter() {
            let circuit = create_rotation_circuit(*num_qubits, *rotations);

            group.bench_with_input(
                BenchmarkId::new("rotations", format!("{}q_{}rot", num_qubits, rotations)),
                &circuit,
                |b, circuit| {
                    b.iter(|| {
                        let optimized = fuse_single_qubit_gates(black_box(circuit), None).unwrap();
                        black_box(optimized)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_mixed_circuit(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_circuit_fusion");

    for num_qubits in [10, 20, 50].iter() {
        let circuit = create_mixed_circuit(*num_qubits, 6);

        group.bench_with_input(
            BenchmarkId::new("mixed", format!("{}qubits", num_qubits)),
            &circuit,
            |b, circuit| {
                b.iter(|| {
                    let optimized = fuse_single_qubit_gates(black_box(circuit), None).unwrap();
                    black_box(optimized)
                })
            },
        );
    }

    group.finish();
}

fn bench_matrix_multiplication(c: &mut Criterion) {
    use num_complex::Complex64;
    use simq_compiler::matrix_utils::multiply_2x2;

    let hadamard = [
        [Complex64::new(0.7071067811865476, 0.0), Complex64::new(0.7071067811865476, 0.0)],
        [Complex64::new(0.7071067811865476, 0.0), Complex64::new(-0.7071067811865476, 0.0)],
    ];

    let pauli_x = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];

    c.bench_function("matrix_mult_2x2", |b| {
        b.iter(|| {
            let result = multiply_2x2(black_box(&hadamard), black_box(&pauli_x));
            black_box(result)
        })
    });
}

fn bench_fusion_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_overhead");

    // Compare circuits with no fusion opportunities vs many
    let no_fusion = {
        let mut circuit = Circuit::new(10);
        for i in 0..10 {
            circuit.add_gate(
                Arc::new(Hadamard) as Arc<dyn Gate>,
                &[QubitId::new(i)]
            ).unwrap();
        }
        circuit
    };

    let many_fusion = create_fuseable_circuit(10, 20);

    group.bench_function("no_fusion_opportunities", |b| {
        b.iter(|| {
            let optimized = fuse_single_qubit_gates(black_box(&no_fusion), None).unwrap();
            black_box(optimized)
        })
    });

    group.bench_function("many_fusion_opportunities", |b| {
        b.iter(|| {
            let optimized = fuse_single_qubit_gates(black_box(&many_fusion), None).unwrap();
            black_box(optimized)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_gate_fusion,
    bench_rotation_fusion,
    bench_mixed_circuit,
    bench_matrix_multiplication,
    bench_fusion_overhead
);
criterion_main!(benches);
