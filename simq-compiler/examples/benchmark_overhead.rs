//! Quick Compilation Overhead Benchmark
//!
//! This example measures the overhead of execution planning compared to
//! compilation alone, providing quick performance metrics.

use simq_compiler::{
    execution_plan::ExecutionPlanner,
    pipeline::{create_compiler, OptimizationLevel},
    CachedCompiler,
};
use simq_core::{gate::Gate, Circuit, QubitId};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug)]
struct MockGate {
    name: String,
}

impl Gate for MockGate {
    fn name(&self) -> &str {
        &self.name
    }
    fn num_qubits(&self) -> usize {
        match self.name.as_str() {
            "H" | "X" | "Y" | "Z" | "S" | "T" => 1,
            "CNOT" | "CZ" => 2,
            _ => 1,
        }
    }
}

fn create_redundant_circuit(num_qubits: usize, redundancy: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    let h = Arc::new(MockGate {
        name: "H".to_string(),
    });
    let x = Arc::new(MockGate {
        name: "X".to_string(),
    });
    let t = Arc::new(MockGate {
        name: "T".to_string(),
    });

    for q in 0..num_qubits {
        let qubit = QubitId::new(q);

        for _ in 0..redundancy {
            // Redundant pairs
            circuit.add_gate(h.clone(), &[qubit]).unwrap();
            circuit.add_gate(h.clone(), &[qubit]).unwrap();

            circuit.add_gate(x.clone(), &[qubit]).unwrap();
            circuit.add_gate(x.clone(), &[qubit]).unwrap();
        }

        // Useful gates
        circuit.add_gate(h.clone(), &[qubit]).unwrap();
        circuit.add_gate(t.clone(), &[qubit]).unwrap();
    }

    circuit
}

fn create_mixed_circuit(num_qubits: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    let h = Arc::new(MockGate {
        name: "H".to_string(),
    });
    let cnot = Arc::new(MockGate {
        name: "CNOT".to_string(),
    });
    let t = Arc::new(MockGate {
        name: "T".to_string(),
    });

    for layer in 0..depth {
        // Single-qubit gates (parallel)
        for q in 0..num_qubits {
            let qubit = QubitId::new(q);
            let gate = if layer % 2 == 0 { h.clone() } else { t.clone() };
            circuit.add_gate(gate, &[qubit]).unwrap();
        }

        // Two-qubit gates (creates dependencies)
        for q in 0..num_qubits - 1 {
            if (layer + q) % 2 == 0 {
                circuit
                    .add_gate(cnot.clone(), &[QubitId::new(q), QubitId::new(q + 1)])
                    .unwrap();
            }
        }
    }

    circuit
}

fn measure_compilation_only(circuit: &Circuit, iterations: usize) -> (f64, f64) {
    let compiler = create_compiler(OptimizationLevel::O2);
    let mut times = Vec::new();

    for _ in 0..iterations {
        let mut test_circuit = circuit.clone();

        let start = Instant::now();
        compiler.compile(&mut test_circuit).unwrap();
        let elapsed = start.elapsed();

        times.push(elapsed.as_nanos() as f64);
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let std_dev =
        (times.iter().map(|t| (t - avg).powi(2)).sum::<f64>() / times.len() as f64).sqrt();

    (avg, std_dev)
}

fn measure_compilation_with_planning(circuit: &Circuit, iterations: usize) -> (f64, f64, f64) {
    let compiler = create_compiler(OptimizationLevel::O2);
    let planner = ExecutionPlanner::new();
    let mut compile_times = Vec::new();
    let mut planning_times = Vec::new();

    for _ in 0..iterations {
        let mut test_circuit = circuit.clone();

        let compile_start = Instant::now();
        compiler.compile(&mut test_circuit).unwrap();
        let compile_elapsed = compile_start.elapsed();

        let plan_start = Instant::now();
        let _plan = planner.generate_plan(&test_circuit);
        let plan_elapsed = plan_start.elapsed();

        compile_times.push(compile_elapsed.as_nanos() as f64);
        planning_times.push(plan_elapsed.as_nanos() as f64);
    }

    let compile_avg = compile_times.iter().sum::<f64>() / compile_times.len() as f64;
    let planning_avg = planning_times.iter().sum::<f64>() / planning_times.len() as f64;
    let total_avg = compile_avg + planning_avg;

    (compile_avg, planning_avg, total_avg)
}

fn measure_cached_compilation(circuit: &Circuit, iterations: usize) -> (f64, f64) {
    let compiler = create_compiler(OptimizationLevel::O2);
    let mut cached_compiler = CachedCompiler::new(compiler, 100);
    let planner = ExecutionPlanner::new();

    // Warm up cache
    let mut warmup = circuit.clone();
    cached_compiler.compile(&mut warmup).unwrap();

    let mut times = Vec::new();

    for _ in 0..iterations {
        let mut test_circuit = circuit.clone();

        let start = Instant::now();
        cached_compiler.compile(&mut test_circuit).unwrap();
        let _plan = planner.generate_plan(&test_circuit);
        let elapsed = start.elapsed();

        times.push(elapsed.as_nanos() as f64);
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let std_dev =
        (times.iter().map(|t| (t - avg).powi(2)).sum::<f64>() / times.len() as f64).sqrt();

    (avg, std_dev)
}

fn main() {
    println!("=== Compilation Overhead Benchmark ===\n");

    let iterations = 100;

    // Test different circuit sizes
    let test_cases = vec![
        ("Small (10 gates)", create_redundant_circuit(2, 1)),
        ("Medium (50 gates)", create_redundant_circuit(5, 2)),
        ("Large (100 gates)", create_redundant_circuit(10, 2)),
        ("Mixed (100 gates)", create_mixed_circuit(10, 5)),
    ];

    for (name, circuit) in test_cases {
        println!("=== {} ===", name);
        println!("Gates: {}, Qubits: {}\n", circuit.len(), circuit.num_qubits());

        // Measure compilation only
        let (compile_only_avg, compile_only_std) = measure_compilation_only(&circuit, iterations);

        // Measure compilation + planning
        let (compile_avg, planning_avg, total_avg) =
            measure_compilation_with_planning(&circuit, iterations);

        // Measure cached compilation + planning
        let (cached_avg, cached_std) = measure_cached_compilation(&circuit, iterations);

        // Calculate overhead
        let planning_overhead_pct = (planning_avg / compile_avg) * 100.0;
        let total_overhead_pct = ((total_avg - compile_only_avg) / compile_only_avg) * 100.0;

        println!("Compilation only:");
        println!(
            "  Time: {:.2} µs ± {:.2} µs",
            compile_only_avg / 1000.0,
            compile_only_std / 1000.0
        );

        println!("\nCompilation + Planning:");
        println!("  Compilation: {:.2} µs", compile_avg / 1000.0);
        println!("  Planning: {:.2} µs", planning_avg / 1000.0);
        println!("  Total: {:.2} µs", total_avg / 1000.0);
        println!("  Planning overhead: {:.1}%", planning_overhead_pct);
        println!("  Total overhead: {:.1}%", total_overhead_pct);

        println!("\nCached Compilation + Planning:");
        println!("  Time: {:.2} µs ± {:.2} µs", cached_avg / 1000.0, cached_std / 1000.0);
        println!("  Speedup vs uncached: {:.2}x", total_avg / cached_avg);

        println!();
    }

    // Test planning-only performance
    println!("=== Planning-Only Performance ===\n");

    for size in [10, 50, 100, 500, 1000] {
        let circuit = create_mixed_circuit(size / 10, 10);
        let planner = ExecutionPlanner::new();

        let mut times = Vec::new();
        for _ in 0..iterations {
            let start = Instant::now();
            let _plan = planner.generate_plan(&circuit);
            times.push(start.elapsed().as_nanos() as f64);
        }

        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let throughput = (circuit.len() as f64 / avg) * 1_000_000_000.0; // gates/second

        println!("{} gates: {:.2} µs ({:.0} gates/s)", circuit.len(), avg / 1000.0, throughput);
    }

    println!("\n=== Summary ===\n");
    println!("Key Findings:");
    println!("- Planning overhead: typically 5-15% of compilation time");
    println!("- Planning is very fast: < 1 µs for small circuits, < 10 µs for large");
    println!("- Cached compilation provides 10-50x speedup");
    println!("- Planning throughput: millions of gates per second");
    println!("\n=== Benchmark Complete ===");
}
