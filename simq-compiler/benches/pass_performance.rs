//! Detailed performance benchmarking for individual optimization passes
//!
//! This benchmark suite provides in-depth performance analysis for each optimization
//! pass, including:
//! - Performance scaling with circuit size
//! - Impact of gate density and patterns
//! - Pass-specific optimization scenarios
//! - Comparative analysis across passes

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use simq_compiler::passes::{
    AdvancedTemplateMatching, DeadCodeElimination, GateCommutation, GateFusion, OptimizationPass,
};
use simq_core::{Circuit, QubitId};
use simq_gates::standard::{
    CNot, CZ, Hadamard, PauliX, PauliY, PauliZ, RotationX, RotationY, RotationZ, SGate, TGate,
    Swap,
};
use std::sync::Arc;

// ===== Circuit Generators =====

/// Generate a circuit with self-inverse gate pairs for DCE testing
fn circuit_with_inverse_pairs(num_qubits: usize, pairs_per_qubit: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    for q in 0..num_qubits {
        let qubit = QubitId::new(q);
        for i in 0..pairs_per_qubit {
            match i % 4 {
                0 => {
                    circuit.add_gate(Arc::new(PauliX), &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(PauliX), &[qubit]).unwrap();
                }
                1 => {
                    circuit.add_gate(Arc::new(PauliY), &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(PauliY), &[qubit]).unwrap();
                }
                2 => {
                    circuit.add_gate(Arc::new(PauliZ), &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(PauliZ), &[qubit]).unwrap();
                }
                _ => {
                    circuit.add_gate(Arc::new(Hadamard), &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(Hadamard), &[qubit]).unwrap();
                }
            }
        }
    }

    circuit
}

/// Generate a circuit with diagonal gates that can commute
fn circuit_with_diagonal_gates(num_qubits: usize, gates_per_qubit: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    for q in 0..num_qubits {
        let qubit = QubitId::new(q);
        for i in 0..gates_per_qubit {
            match i % 3 {
                0 => circuit.add_gate(Arc::new(PauliZ), &[qubit]).unwrap(),
                1 => circuit.add_gate(Arc::new(SGate), &[qubit]).unwrap(),
                _ => circuit.add_gate(Arc::new(TGate), &[qubit]).unwrap(),
            }
        }
    }

    circuit
}

/// Generate a circuit with rotation gates on same axis (commutable)
fn circuit_with_same_axis_rotations(num_qubits: usize, rotations_per_qubit: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    for q in 0..num_qubits {
        let qubit = QubitId::new(q);
        for i in 0..rotations_per_qubit {
            let angle = (i as f64) * 0.1;
            circuit
                .add_gate(Arc::new(RotationX::new(angle)), &[qubit])
                .unwrap();
        }
    }

    circuit
}

/// Generate a circuit with CNOT gates that can commute
fn circuit_with_commuting_cnots(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    if num_qubits < 3 {
        return circuit;
    }

    let control = QubitId::new(0);

    // Add multiple CNOTs with same control, different targets (these commute)
    for target_idx in 1..num_qubits {
        let target = QubitId::new(target_idx);
        circuit.add_gate(Arc::new(CNot), &[control, target]).unwrap();
    }

    circuit
}

/// Generate a circuit with CZ gates (always commute)
fn circuit_with_cz_gates(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    for i in 0..num_qubits - 1 {
        for j in i + 1..num_qubits {
            circuit
                .add_gate(Arc::new(CZ), &[QubitId::new(i), QubitId::new(j)])
                .unwrap();
        }
    }

    circuit
}

/// Generate a circuit with interleaved gates (needs commutation)
fn circuit_with_interleaved_gates(num_qubits: usize, iterations: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    for _ in 0..iterations {
        // Add gates in a pattern that benefits from commutation
        for q in 0..num_qubits {
            let qubit = QubitId::new(q);
            circuit.add_gate(Arc::new(Hadamard), &[qubit]).unwrap();
        }

        // Add diagonal gates that can be grouped
        for q in 0..num_qubits {
            let qubit = QubitId::new(q);
            circuit.add_gate(Arc::new(PauliZ), &[qubit]).unwrap();
            circuit.add_gate(Arc::new(SGate), &[qubit]).unwrap();
        }
    }

    circuit
}

/// Generate a circuit with fuseable gates
fn circuit_with_fuseable_gates(num_qubits: usize, gates_per_qubit: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    for q in 0..num_qubits {
        let qubit = QubitId::new(q);
        for i in 0..gates_per_qubit {
            match i % 4 {
                0 => circuit.add_gate(Arc::new(Hadamard), &[qubit]).unwrap(),
                1 => circuit.add_gate(Arc::new(PauliX), &[qubit]).unwrap(),
                2 => circuit.add_gate(Arc::new(SGate), &[qubit]).unwrap(),
                _ => circuit.add_gate(Arc::new(TGate), &[qubit]).unwrap(),
            }
        }
    }

    circuit
}

/// Generate a circuit with template patterns
fn circuit_with_template_patterns(num_qubits: usize, patterns_per_qubit: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    for q in 0..num_qubits {
        let qubit = QubitId::new(q);
        for i in 0..patterns_per_qubit {
            match i % 3 {
                0 => {
                    // H-Z-H -> X pattern
                    circuit.add_gate(Arc::new(Hadamard), &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(PauliZ), &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(Hadamard), &[qubit]).unwrap();
                }
                1 => {
                    // H-X-H -> Z pattern
                    circuit.add_gate(Arc::new(Hadamard), &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(PauliX), &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(Hadamard), &[qubit]).unwrap();
                }
                _ => {
                    // Self-inverse pairs
                    circuit.add_gate(Arc::new(PauliX), &[qubit]).unwrap();
                    circuit.add_gate(Arc::new(PauliX), &[qubit]).unwrap();
                }
            }
        }
    }

    circuit
}

/// Generate a realistic mixed circuit
fn circuit_realistic(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    // Initial layer of Hadamards
    for q in 0..num_qubits {
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(q)])
            .unwrap();
    }

    // Add some entanglement with CNOTs
    for q in 0..num_qubits - 1 {
        circuit
            .add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
            .unwrap();
    }

    // Add rotations and phase gates
    for q in 0..num_qubits {
        let qubit = QubitId::new(q);
        circuit
            .add_gate(Arc::new(RotationX::new(0.5)), &[qubit])
            .unwrap();
        circuit.add_gate(Arc::new(SGate), &[qubit]).unwrap();
        circuit
            .add_gate(Arc::new(RotationZ::new(0.25)), &[qubit])
            .unwrap();
    }

    // Add some inverse pairs for optimization
    for q in 0..num_qubits / 2 {
        let qubit = QubitId::new(q);
        circuit.add_gate(Arc::new(PauliX), &[qubit]).unwrap();
        circuit.add_gate(Arc::new(PauliX), &[qubit]).unwrap();
    }

    circuit
}

// ===== Dead Code Elimination Benchmarks =====

fn bench_dce_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dce_scaling");

    let pass = DeadCodeElimination::new();

    for num_qubits in [5, 10, 20, 50, 100] {
        let pairs_per_qubit = 20;
        let circuit = circuit_with_inverse_pairs(num_qubits, pairs_per_qubit);
        let total_gates = circuit.len();

        group.throughput(Throughput::Elements(total_gates as u64));
        group.bench_with_input(
            BenchmarkId::new("inverse_pairs", format!("{}q_{}gates", num_qubits, total_gates)),
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

fn bench_dce_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("dce_density");

    let pass = DeadCodeElimination::new();
    let num_qubits = 10;

    // Vary the density of removable patterns
    for pairs_per_qubit in [5, 10, 20, 50, 100] {
        let circuit = circuit_with_inverse_pairs(num_qubits, pairs_per_qubit);
        let total_gates = circuit.len();

        group.throughput(Throughput::Elements(total_gates as u64));
        group.bench_with_input(
            BenchmarkId::new("density", format!("{}pairs_{}gates", pairs_per_qubit, total_gates)),
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

// ===== Gate Commutation Benchmarks =====

fn bench_commutation_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("commutation_scaling");

    let pass = GateCommutation::new();

    for num_qubits in [5, 10, 20, 50, 100] {
        let circuit = circuit_with_interleaved_gates(num_qubits, 5);
        let total_gates = circuit.len();

        group.throughput(Throughput::Elements(total_gates as u64));
        group.bench_with_input(
            BenchmarkId::new("interleaved", format!("{}q_{}gates", num_qubits, total_gates)),
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

fn bench_commutation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("commutation_patterns");

    let pass = GateCommutation::new();
    let num_qubits = 10;

    // Test different commutation patterns

    // Diagonal gates
    let diagonal_circuit = circuit_with_diagonal_gates(num_qubits, 20);
    group.bench_with_input(
        BenchmarkId::new("pattern", format!("diagonal_{}gates", diagonal_circuit.len())),
        &diagonal_circuit,
        |b, circuit| {
            b.iter(|| {
                let mut c = circuit.clone();
                let result = pass.apply(black_box(&mut c)).unwrap();
                black_box((c, result))
            })
        },
    );

    // Same-axis rotations
    let rotation_circuit = circuit_with_same_axis_rotations(num_qubits, 20);
    group.bench_with_input(
        BenchmarkId::new("pattern", format!("rotations_{}gates", rotation_circuit.len())),
        &rotation_circuit,
        |b, circuit| {
            b.iter(|| {
                let mut c = circuit.clone();
                let result = pass.apply(black_box(&mut c)).unwrap();
                black_box((c, result))
            })
        },
    );

    // Commuting CNOTs
    let cnot_circuit = circuit_with_commuting_cnots(num_qubits);
    group.bench_with_input(
        BenchmarkId::new("pattern", format!("cnots_{}gates", cnot_circuit.len())),
        &cnot_circuit,
        |b, circuit| {
            b.iter(|| {
                let mut c = circuit.clone();
                let result = pass.apply(black_box(&mut c)).unwrap();
                black_box((c, result))
            })
        },
    );

    // CZ gates
    let cz_circuit = circuit_with_cz_gates(num_qubits);
    group.bench_with_input(
        BenchmarkId::new("pattern", format!("cz_{}gates", cz_circuit.len())),
        &cz_circuit,
        |b, circuit| {
            b.iter(|| {
                let mut c = circuit.clone();
                let result = pass.apply(black_box(&mut c)).unwrap();
                black_box((c, result))
            })
        },
    );

    group.finish();
}

// ===== Gate Fusion Benchmarks =====

fn bench_fusion_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_scaling");

    let pass = GateFusion::new();

    for num_qubits in [5, 10, 20, 50, 100] {
        let circuit = circuit_with_fuseable_gates(num_qubits, 20);
        let total_gates = circuit.len();

        group.throughput(Throughput::Elements(total_gates as u64));
        group.bench_with_input(
            BenchmarkId::new("fuseable", format!("{}q_{}gates", num_qubits, total_gates)),
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

fn bench_fusion_chain_length(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_chain_length");

    let pass = GateFusion::new();
    let num_qubits = 10;

    // Vary the length of fuseable chains
    for gates_per_qubit in [5, 10, 20, 50, 100] {
        let circuit = circuit_with_fuseable_gates(num_qubits, gates_per_qubit);
        let total_gates = circuit.len();

        group.throughput(Throughput::Elements(total_gates as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "chain_length",
                format!("{}gates_per_q_{}total", gates_per_qubit, total_gates),
            ),
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

// ===== Template Matching Benchmarks =====

fn bench_template_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("template_scaling");

    let pass = AdvancedTemplateMatching::new();

    for num_qubits in [5, 10, 20, 50, 100] {
        let circuit = circuit_with_template_patterns(num_qubits, 10);
        let total_gates = circuit.len();

        group.throughput(Throughput::Elements(total_gates as u64));
        group.bench_with_input(
            BenchmarkId::new("patterns", format!("{}q_{}gates", num_qubits, total_gates)),
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

fn bench_template_pattern_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("template_pattern_density");

    let pass = AdvancedTemplateMatching::new();
    let num_qubits = 10;

    // Vary the density of template patterns
    for patterns_per_qubit in [5, 10, 20, 50] {
        let circuit = circuit_with_template_patterns(num_qubits, patterns_per_qubit);
        let total_gates = circuit.len();

        group.throughput(Throughput::Elements(total_gates as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "density",
                format!("{}patterns_{}gates", patterns_per_qubit, total_gates),
            ),
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

// ===== Comparative Analysis =====

fn bench_pass_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("pass_comparison");

    let circuit = circuit_realistic(20);
    let total_gates = circuit.len();

    // Benchmark each pass on the same realistic circuit
    let dce = DeadCodeElimination::new();
    group.bench_with_input(
        BenchmarkId::new("pass_type", format!("dce_{}gates", total_gates)),
        &circuit,
        |b, circuit| {
            b.iter(|| {
                let mut c = circuit.clone();
                let result = dce.apply(black_box(&mut c)).unwrap();
                black_box((c, result))
            })
        },
    );

    let commutation = GateCommutation::new();
    group.bench_with_input(
        BenchmarkId::new("pass_type", format!("commutation_{}gates", total_gates)),
        &circuit,
        |b, circuit| {
            b.iter(|| {
                let mut c = circuit.clone();
                let result = commutation.apply(black_box(&mut c)).unwrap();
                black_box((c, result))
            })
        },
    );

    let fusion = GateFusion::new();
    group.bench_with_input(
        BenchmarkId::new("pass_type", format!("fusion_{}gates", total_gates)),
        &circuit,
        |b, circuit| {
            b.iter(|| {
                let mut c = circuit.clone();
                let result = fusion.apply(black_box(&mut c)).unwrap();
                black_box((c, result))
            })
        },
    );

    let template = AdvancedTemplateMatching::new();
    group.bench_with_input(
        BenchmarkId::new("pass_type", format!("template_{}gates", total_gates)),
        &circuit,
        |b, circuit| {
            b.iter(|| {
                let mut c = circuit.clone();
                let result = template.apply(black_box(&mut c)).unwrap();
                black_box((c, result))
            })
        },
    );

    group.finish();
}

fn bench_worst_case_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("worst_case");

    // DCE worst case: no patterns to remove
    let dce = DeadCodeElimination::new();
    let dce_worst = circuit_realistic(50); // No inverse pairs
    group.bench_with_input(
        BenchmarkId::new("dce", format!("no_patterns_{}gates", dce_worst.len())),
        &dce_worst,
        |b, circuit| {
            b.iter(|| {
                let mut c = circuit.clone();
                let result = dce.apply(black_box(&mut c)).unwrap();
                black_box((c, result))
            })
        },
    );

    // Commutation worst case: no commuting gates
    let commutation = GateCommutation::new();
    let comm_worst = circuit_realistic(50); // Mixed gates, hard to commute
    group.bench_with_input(
        BenchmarkId::new("commutation", format!("no_commute_{}gates", comm_worst.len())),
        &comm_worst,
        |b, circuit| {
            b.iter(|| {
                let mut c = circuit.clone();
                let result = commutation.apply(black_box(&mut c)).unwrap();
                black_box((c, result))
            })
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    // Dead Code Elimination
    bench_dce_scaling,
    bench_dce_density,
    // Gate Commutation
    bench_commutation_scaling,
    bench_commutation_patterns,
    // Gate Fusion
    bench_fusion_scaling,
    bench_fusion_chain_length,
    // Template Matching
    bench_template_scaling,
    bench_template_pattern_density,
    // Comparative Analysis
    bench_pass_comparison,
    bench_worst_case_scenarios
);

criterion_main!(benches);
