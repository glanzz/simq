use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use simq_compiler::{
    fusion::fuse_single_qubit_gates,
    passes::{
        AdvancedTemplateMatching, DeadCodeElimination, GateCommutation, GateFusion,
        OptimizationPass,
    },
    create_compiler, OptimizationLevel,
};
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

/// Create a circuit with many self-inverse patterns (for dead code elimination)
fn create_self_inverse_circuit(num_qubits: usize, pairs_per_qubit: usize) -> Circuit {
    let mut circuit = Circuit::with_capacity(num_qubits, num_qubits * pairs_per_qubit * 2);

    for qubit_idx in 0..num_qubits {
        let qubit = QubitId::new(qubit_idx);
        for i in 0..pairs_per_qubit {
            match i % 4 {
                0 => {
                    // X-X pairs
                    circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[qubit]).unwrap();
                }
                1 => {
                    // Y-Y pairs
                    circuit.add_gate(Arc::new(PauliY) as Arc<dyn Gate>, &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(PauliY) as Arc<dyn Gate>, &[qubit]).unwrap();
                }
                2 => {
                    // H-H pairs
                    circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[qubit]).unwrap();
                }
                _ => {
                    // Z-Z pairs
                    circuit.add_gate(Arc::new(PauliZ) as Arc<dyn Gate>, &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(PauliZ) as Arc<dyn Gate>, &[qubit]).unwrap();
                }
            }
        }
    }

    circuit
}

/// Create a circuit with template patterns (for template matching)
fn create_template_circuit(num_qubits: usize, patterns_per_qubit: usize) -> Circuit {
    let mut circuit = Circuit::with_capacity(num_qubits, num_qubits * patterns_per_qubit * 3);

    for qubit_idx in 0..num_qubits {
        let qubit = QubitId::new(qubit_idx);
        for i in 0..patterns_per_qubit {
            match i % 3 {
                0 => {
                    // H-Z-H → X
                    circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(PauliZ) as Arc<dyn Gate>, &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[qubit]).unwrap();
                }
                1 => {
                    // H-X-H → Z
                    circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[qubit]).unwrap();
                }
                _ => {
                    // X-Y-X → Y
                    circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(PauliY) as Arc<dyn Gate>, &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[qubit]).unwrap();
                }
            }
        }
    }

    circuit
}

/// Create a circuit that benefits from commutation
fn create_commutation_circuit(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::with_capacity(num_qubits, num_qubits * 10);

    // Create interleaved gates on different qubits that can be reordered
    for round in 0..5 {
        for qubit_idx in 0..num_qubits {
            let qubit = QubitId::new(qubit_idx);
            if round % 2 == 0 {
                circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[qubit]).unwrap();
            } else {
                circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[qubit]).unwrap();
            }
        }
    }

    circuit
}

/// Create a realistic quantum circuit with various patterns
fn create_realistic_circuit(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::with_capacity(num_qubits, num_qubits * 20);

    for qubit_idx in 0..num_qubits {
        let qubit = QubitId::new(qubit_idx);

        // Start with some single-qubit gates
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[qubit]).unwrap();
        circuit.add_gate(Arc::new(RotationZ::new(0.5)) as Arc<dyn Gate>, &[qubit]).unwrap();
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[qubit]).unwrap();

        // Some that can be fused
        circuit.add_gate(Arc::new(SGate) as Arc<dyn Gate>, &[qubit]).unwrap();
        circuit.add_gate(Arc::new(TGate) as Arc<dyn Gate>, &[qubit]).unwrap();
        circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[qubit]).unwrap();

        // Self-inverse pair
        circuit.add_gate(Arc::new(PauliY) as Arc<dyn Gate>, &[qubit]).unwrap();
        circuit.add_gate(Arc::new(PauliY) as Arc<dyn Gate>, &[qubit]).unwrap();

        // Template pattern: H-Z-H
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[qubit]).unwrap();
        circuit.add_gate(Arc::new(PauliZ) as Arc<dyn Gate>, &[qubit]).unwrap();
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[qubit]).unwrap();

        // Add entangling gates
        if qubit_idx + 1 < num_qubits {
            let next_qubit = QubitId::new(qubit_idx + 1);
            circuit.add_gate(
                Arc::new(simq_gates::standard::CNot) as Arc<dyn Gate>,
                &[qubit, next_qubit]
            ).unwrap();
        }
    }

    circuit
}

// ===== Individual Pass Benchmarks =====

fn bench_dead_code_elimination(c: &mut Criterion) {
    let mut group = c.benchmark_group("dead_code_elimination");

    let pass = DeadCodeElimination::new();

    for num_qubits in [5, 10, 20, 50].iter() {
        for pairs in [5, 10, 20].iter() {
            let circuit = create_self_inverse_circuit(*num_qubits, *pairs);
            let total_gates = circuit.len();

            group.bench_with_input(
                BenchmarkId::new("self_inverse", format!("{}q_{}pairs_{}gates", num_qubits, pairs, total_gates)),
                &circuit,
                |b, circuit| {
                    b.iter(|| {
                        let mut c = circuit.clone();
                        let result = pass.apply(black_box(&mut c)).unwrap();
                        black_box((c, result))
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_template_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("template_matching");

    let pass = AdvancedTemplateMatching::new();

    for num_qubits in [5, 10, 20, 50].iter() {
        for patterns in [5, 10, 20].iter() {
            let circuit = create_template_circuit(*num_qubits, *patterns);
            let total_gates = circuit.len();

            group.bench_with_input(
                BenchmarkId::new("patterns", format!("{}q_{}pat_{}gates", num_qubits, patterns, total_gates)),
                &circuit,
                |b, circuit| {
                    b.iter(|| {
                        let mut c = circuit.clone();
                        let result = pass.apply(black_box(&mut c)).unwrap();
                        black_box((c, result))
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_gate_commutation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gate_commutation");

    let pass = GateCommutation::new();

    for num_qubits in [5, 10, 20, 50].iter() {
        let circuit = create_commutation_circuit(*num_qubits);
        let total_gates = circuit.len();

        group.bench_with_input(
            BenchmarkId::new("commutation", format!("{}q_{}gates", num_qubits, total_gates)),
            &circuit,
            |b, circuit| {
                b.iter(|| {
                    let mut c = circuit.clone();
                    let result = pass.apply(black_box(&mut c)).unwrap();
                    black_box((c, result))
                })
            },
        );
    }

    group.finish();
}

fn bench_gate_fusion_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("gate_fusion_pass");

    let pass = GateFusion::new();

    for num_qubits in [5, 10, 20, 50].iter() {
        for gates_per_qubit in [10, 20, 50].iter() {
            let circuit = create_fuseable_circuit(*num_qubits, *gates_per_qubit);
            let total_gates = circuit.len();

            group.bench_with_input(
                BenchmarkId::new("fusion", format!("{}q_{}gpq_{}gates", num_qubits, gates_per_qubit, total_gates)),
                &circuit,
                |b, circuit| {
                    b.iter(|| {
                        let mut c = circuit.clone();
                        let result = pass.apply(black_box(&mut c)).unwrap();
                        black_box((c, result))
                    })
                },
            );
        }
    }

    group.finish();
}

// ===== Full Pipeline Benchmarks =====

fn bench_optimization_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_levels");

    for num_qubits in [5, 10, 20].iter() {
        let circuit = create_realistic_circuit(*num_qubits);
        let total_gates = circuit.len();

        // Benchmark each optimization level
        for level in [OptimizationLevel::O0, OptimizationLevel::O1, OptimizationLevel::O2, OptimizationLevel::O3] {
            let compiler = create_compiler(level);
            let level_name = match level {
                OptimizationLevel::O0 => "O0",
                OptimizationLevel::O1 => "O1",
                OptimizationLevel::O2 => "O2",
                OptimizationLevel::O3 => "O3",
            };

            group.bench_with_input(
                BenchmarkId::new(level_name, format!("{}q_{}gates", num_qubits, total_gates)),
                &circuit,
                |b, circuit| {
                    b.iter(|| {
                        let mut c = circuit.clone();
                        let result = compiler.compile(black_box(&mut c)).unwrap();
                        black_box((c, result))
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_pass_combinations(c: &mut Criterion) {
    let mut group = c.benchmark_group("pass_combinations");

    let circuit = create_realistic_circuit(10);
    let total_gates = circuit.len();

    // Benchmark individual passes
    group.bench_with_input(
        BenchmarkId::new("single_pass", format!("dce_{}gates", total_gates)),
        &circuit,
        |b, circuit| {
            let pass = DeadCodeElimination::new();
            b.iter(|| {
                let mut c = circuit.clone();
                let result = pass.apply(black_box(&mut c)).unwrap();
                black_box((c, result))
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("single_pass", format!("template_{}gates", total_gates)),
        &circuit,
        |b, circuit| {
            let pass = AdvancedTemplateMatching::new();
            b.iter(|| {
                let mut c = circuit.clone();
                let result = pass.apply(black_box(&mut c)).unwrap();
                black_box((c, result))
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("single_pass", format!("fusion_{}gates", total_gates)),
        &circuit,
        |b, circuit| {
            let pass = GateFusion::new();
            b.iter(|| {
                let mut c = circuit.clone();
                let result = pass.apply(black_box(&mut c)).unwrap();
                black_box((c, result))
            })
        },
    );

    // Benchmark pass combinations
    group.bench_with_input(
        BenchmarkId::new("two_passes", format!("dce_template_{}gates", total_gates)),
        &circuit,
        |b, circuit| {
            let dce = DeadCodeElimination::new();
            let template = AdvancedTemplateMatching::new();
            b.iter(|| {
                let mut c = circuit.clone();
                dce.apply(&mut c).unwrap();
                template.apply(black_box(&mut c)).unwrap();
                black_box(c)
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("full_pipeline", format!("all_passes_{}gates", total_gates)),
        &circuit,
        |b, circuit| {
            let compiler = create_compiler(OptimizationLevel::O3);
            b.iter(|| {
                let mut c = circuit.clone();
                let result = compiler.compile(black_box(&mut c)).unwrap();
                black_box((c, result))
            })
        },
    );

    group.finish();
}

fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    group.sample_size(20); // Reduce sample size for large circuits

    let compiler = create_compiler(OptimizationLevel::O2);

    // Test how the compiler scales with circuit size
    for num_qubits in [10, 25, 50, 100].iter() {
        let circuit = create_realistic_circuit(*num_qubits);
        let total_gates = circuit.len();

        group.bench_with_input(
            BenchmarkId::new("circuit_size", format!("{}q_{}gates", num_qubits, total_gates)),
            &circuit,
            |b, circuit| {
                b.iter(|| {
                    let mut c = circuit.clone();
                    let result = compiler.compile(black_box(&mut c)).unwrap();
                    black_box((c, result))
                })
            },
        );
    }

    group.finish();
}

// ===== Legacy Gate Fusion Benchmarks =====

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
    // Individual pass benchmarks
    bench_dead_code_elimination,
    bench_template_matching,
    bench_gate_commutation,
    bench_gate_fusion_pass,
    // Full pipeline benchmarks
    bench_optimization_levels,
    bench_pass_combinations,
    bench_scalability,
    // Legacy gate fusion benchmarks
    bench_gate_fusion,
    bench_rotation_fusion,
    bench_mixed_circuit,
    bench_matrix_multiplication,
    bench_fusion_overhead
);
criterion_main!(benches);
