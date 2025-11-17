//! Execution Planning and Compilation Overhead Benchmarks
//!
//! This benchmark suite measures:
//! - Execution plan generation performance
//! - Compilation overhead with and without planning
//! - Parallelization analysis performance
//! - Scaling with circuit size and complexity

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use simq_compiler::{
    execution_plan::ExecutionPlanner,
    pipeline::{create_compiler, OptimizationLevel},
    CachedCompiler,
};
use simq_core::{Circuit, QubitId};
use simq_gates::standard::{CNot, Hadamard, PauliX, PauliY, PauliZ, SGate, TGate};
use std::sync::Arc;

// ===== Circuit Generators =====

/// Create a sequential circuit (no parallelism)
fn create_sequential_circuit(num_gates: usize) -> Circuit {
    let mut circuit = Circuit::new(2);

    for i in 0..num_gates {
        let gate: Arc<dyn simq_core::gate::Gate> = match i % 4 {
            0 => Arc::new(Hadamard),
            1 => Arc::new(PauliX),
            2 => Arc::new(PauliY),
            _ => Arc::new(PauliZ),
        };
        circuit.add_gate(gate, &[QubitId::new(0)]).unwrap();
    }

    circuit
}

/// Create a parallel circuit (maximum parallelism)
fn create_parallel_circuit(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    // Each qubit gets gates that can execute in parallel
    for q in 0..num_qubits {
        circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
        circuit.add_gate(Arc::new(TGate), &[QubitId::new(q)]).unwrap();
        circuit.add_gate(Arc::new(SGate), &[QubitId::new(q)]).unwrap();
    }

    circuit
}

/// Create a mixed circuit (realistic workload)
fn create_mixed_circuit(num_qubits: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    for layer in 0..depth {
        // Single-qubit gates on all qubits (parallel)
        for q in 0..num_qubits {
            let gate: Arc<dyn simq_core::gate::Gate> = match layer % 3 {
                0 => Arc::new(Hadamard),
                1 => Arc::new(SGate),
                _ => Arc::new(TGate),
            };
            circuit.add_gate(gate, &[QubitId::new(q)]).unwrap();
        }

        // Two-qubit gates (creates dependencies)
        for q in 0..num_qubits-1 {
            if (layer + q) % 2 == 0 {
                circuit.add_gate(
                    Arc::new(CNot),
                    &[QubitId::new(q), QubitId::new(q + 1)]
                ).unwrap();
            }
        }
    }

    circuit
}

/// Create a circuit with redundancy (for optimization benchmarks)
fn create_redundant_circuit(num_qubits: usize, redundancy_factor: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    for q in 0..num_qubits {
        let qubit = QubitId::new(q);

        // Add redundant gate pairs
        for _ in 0..redundancy_factor {
            // H-H pair (cancels)
            circuit.add_gate(Arc::new(Hadamard), &[qubit]).unwrap();
            circuit.add_gate(Arc::new(Hadamard), &[qubit]).unwrap();

            // X-X pair (cancels)
            circuit.add_gate(Arc::new(PauliX), &[qubit]).unwrap();
            circuit.add_gate(Arc::new(PauliX), &[qubit]).unwrap();
        }

        // Add some useful gates
        circuit.add_gate(Arc::new(Hadamard), &[qubit]).unwrap();
        circuit.add_gate(Arc::new(TGate), &[qubit]).unwrap();
    }

    circuit
}

/// Create a deep circuit (tests scaling with depth)
fn create_deep_circuit(num_qubits: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    for d in 0..depth {
        for q in 0..num_qubits {
            let gate: Arc<dyn simq_core::gate::Gate> = match (d + q) % 5 {
                0 => Arc::new(Hadamard),
                1 => Arc::new(PauliX),
                2 => Arc::new(PauliY),
                3 => Arc::new(PauliZ),
                _ => Arc::new(TGate),
            };
            circuit.add_gate(gate, &[QubitId::new(q)]).unwrap();
        }
    }

    circuit
}

// ===== Execution Plan Generation Benchmarks =====

fn bench_plan_generation_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution_plan/sequential");

    for size in [10, 50, 100, 500, 1000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                let circuit = create_sequential_circuit(size);
                let planner = ExecutionPlanner::new();

                b.iter(|| {
                    let plan = planner.generate_plan(black_box(&circuit));
                    black_box(plan);
                });
            },
        );
    }

    group.finish();
}

fn bench_plan_generation_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution_plan/parallel");

    for num_qubits in [5, 10, 20, 50, 100] {
        group.throughput(Throughput::Elements((num_qubits * 3) as u64)); // 3 gates per qubit
        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            &num_qubits,
            |b, &num_qubits| {
                let circuit = create_parallel_circuit(num_qubits);
                let planner = ExecutionPlanner::new();

                b.iter(|| {
                    let plan = planner.generate_plan(black_box(&circuit));
                    black_box(plan);
                });
            },
        );
    }

    group.finish();
}

fn bench_plan_generation_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution_plan/mixed");

    for (qubits, depth) in [(5, 10), (10, 10), (20, 10), (10, 50), (10, 100)] {
        let size = qubits * depth;
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("q{}_d{}", qubits, depth), size),
            &(qubits, depth),
            |b, &(qubits, depth)| {
                let circuit = create_mixed_circuit(qubits, depth);
                let planner = ExecutionPlanner::new();

                b.iter(|| {
                    let plan = planner.generate_plan(black_box(&circuit));
                    black_box(plan);
                });
            },
        );
    }

    group.finish();
}

// ===== Compilation Overhead Benchmarks =====

fn bench_compilation_without_planning(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation/without_planning");

    for size in [10, 50, 100, 500] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                let circuit = create_redundant_circuit(size / 10, 5);
                let compiler = create_compiler(OptimizationLevel::O2);

                b.iter(|| {
                    let mut test_circuit = circuit.clone();
                    let result = compiler.compile(black_box(&mut test_circuit));
                    black_box(&result);
                });
            },
        );
    }

    group.finish();
}

fn bench_compilation_with_planning(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation/with_planning");

    for size in [10, 50, 100, 500] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                let circuit = create_redundant_circuit(size / 10, 5);
                let compiler = create_compiler(OptimizationLevel::O2);
                let planner = ExecutionPlanner::new();

                b.iter(|| {
                    let mut test_circuit = circuit.clone();
                    let result = compiler.compile(black_box(&mut test_circuit));
                    black_box(&result);

                    // Generate execution plan after compilation
                    let plan = planner.generate_plan(black_box(&test_circuit));
                    black_box(plan);
                });
            },
        );
    }

    group.finish();
}

fn bench_planning_overhead_percentage(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation/planning_overhead");

    for size in [10, 50, 100, 500] {
        let circuit = create_redundant_circuit(size / 10, 5);
        let planner = ExecutionPlanner::new();

        group.bench_with_input(
            BenchmarkId::new("planning_only", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let plan = planner.generate_plan(black_box(&circuit));
                    black_box(plan);
                });
            },
        );
    }

    group.finish();
}

// ===== Optimization Level Comparison =====

fn bench_optimization_levels_with_planning(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation/opt_levels");

    let circuit = create_mixed_circuit(10, 20);
    let planner = ExecutionPlanner::new();

    for level in [
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    ] {
        group.bench_with_input(
            BenchmarkId::new("with_planning", format!("{:?}", level)),
            &level,
            |b, &level| {
                let compiler = create_compiler(level);

                b.iter(|| {
                    let mut test_circuit = circuit.clone();
                    compiler.compile(black_box(&mut test_circuit)).unwrap();
                    let plan = planner.generate_plan(black_box(&test_circuit));
                    black_box(plan);
                });
            },
        );
    }

    group.finish();
}

// ===== Caching with Execution Planning =====

fn bench_cached_compilation_with_planning(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation/cached_with_planning");

    let circuit = create_redundant_circuit(10, 5);
    let compiler = create_compiler(OptimizationLevel::O2);
    let mut cached_compiler = CachedCompiler::new(compiler, 100);
    let planner = ExecutionPlanner::new();

    // Warm up cache
    let mut warmup = circuit.clone();
    cached_compiler.compile(&mut warmup).unwrap();

    group.bench_function("cache_hit", |b| {
        b.iter(|| {
            let mut test_circuit = circuit.clone();
            let result = cached_compiler.compile(black_box(&mut test_circuit));
            black_box(&result);

            let plan = planner.generate_plan(black_box(&test_circuit));
            black_box(plan);
        });
    });

    group.bench_function("cache_miss", |b| {
        b.iter(|| {
            // Create unique circuit each time
            let mut unique_circuit = create_mixed_circuit(5, 5);
            let result = cached_compiler.compile(black_box(&mut unique_circuit));
            black_box(&result);

            let plan = planner.generate_plan(black_box(&unique_circuit));
            black_box(plan);
        });
    });

    group.finish();
}

// ===== Parallelization Analysis Benchmarks =====

fn bench_parallelization_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution_plan/metrics");

    let planner = ExecutionPlanner::new();

    for size in [10, 50, 100, 500] {
        let circuit = create_mixed_circuit(size / 10, 10);

        group.bench_with_input(
            BenchmarkId::new("full_analysis", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let plan = planner.generate_plan(black_box(&circuit));

                    // Compute all metrics
                    let efficiency = plan.parallelization_efficiency();
                    let avg_size = plan.average_layer_size();
                    let bottleneck = plan.bottleneck_layer();

                    black_box((efficiency, avg_size, bottleneck));
                });
            },
        );
    }

    group.finish();
}

// ===== Deep Circuit Benchmarks =====

fn bench_deep_circuits(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution_plan/deep_circuits");

    for depth in [10, 50, 100, 500, 1000] {
        let num_qubits = 5;
        let total_gates = num_qubits * depth;

        group.throughput(Throughput::Elements(total_gates as u64));
        group.bench_with_input(
            BenchmarkId::new("depth", depth),
            &depth,
            |b, &depth| {
                let circuit = create_deep_circuit(num_qubits, depth);
                let planner = ExecutionPlanner::new();

                b.iter(|| {
                    let plan = planner.generate_plan(black_box(&circuit));
                    black_box(plan);
                });
            },
        );
    }

    group.finish();
}

// ===== Custom Gate Timing Benchmarks =====

fn bench_custom_gate_timing(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution_plan/custom_timing");

    let circuit = create_mixed_circuit(10, 20);

    group.bench_function("default_timing", |b| {
        let planner = ExecutionPlanner::new();

        b.iter(|| {
            let plan = planner.generate_plan(black_box(&circuit));
            black_box(plan);
        });
    });

    group.bench_function("custom_timing", |b| {
        let mut planner = ExecutionPlanner::new();
        planner.set_gate_time("H", 0.05);
        planner.set_gate_time("CNOT", 0.5);
        planner.set_gate_time("T", 0.1);

        b.iter(|| {
            let plan = planner.generate_plan(black_box(&circuit));
            black_box(plan);
        });
    });

    group.finish();
}

// ===== End-to-End Workflow Benchmarks =====

fn bench_full_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("workflow/end_to_end");

    for size in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("optimize_and_plan", size),
            &size,
            |b, &size| {
                let original_circuit = create_redundant_circuit(size / 10, 5);
                let compiler = create_compiler(OptimizationLevel::O2);
                let planner = ExecutionPlanner::new();

                b.iter(|| {
                    // 1. Generate plan for original circuit
                    let original_plan = planner.generate_plan(black_box(&original_circuit));
                    black_box(&original_plan);

                    // 2. Optimize circuit
                    let mut optimized = original_circuit.clone();
                    compiler.compile(black_box(&mut optimized)).unwrap();

                    // 3. Generate plan for optimized circuit
                    let optimized_plan = planner.generate_plan(black_box(&optimized));
                    black_box(&optimized_plan);

                    // 4. Compare metrics
                    let improvement = original_plan.total_time / optimized_plan.total_time;
                    black_box(improvement);
                });
            },
        );
    }

    group.finish();
}

// ===== Criterion Groups =====

criterion_group!(
    execution_planning,
    bench_plan_generation_sequential,
    bench_plan_generation_parallel,
    bench_plan_generation_mixed,
    bench_parallelization_metrics,
);

criterion_group!(
    compilation_overhead,
    bench_compilation_without_planning,
    bench_compilation_with_planning,
    bench_planning_overhead_percentage,
    bench_optimization_levels_with_planning,
);

criterion_group!(
    advanced_features,
    bench_cached_compilation_with_planning,
    bench_deep_circuits,
    bench_custom_gate_timing,
);

criterion_group!(
    workflows,
    bench_full_workflow,
);

criterion_main!(
    execution_planning,
    compilation_overhead,
    advanced_features,
    workflows,
);
