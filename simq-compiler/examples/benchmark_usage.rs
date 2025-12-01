//! Example demonstrating how to benchmark optimization passes
//!
//! This example shows how to manually benchmark different optimization levels
//! and measure performance improvements.
//!
//! For comprehensive automated benchmarks, run:
//! ```bash
//! cargo bench --bench optimization_passes
//! ```

use simq_compiler::{create_compiler, OptimizationLevel};
use simq_core::{Circuit, QubitId};
use simq_gates::standard::{Hadamard, PauliX, PauliY, PauliZ, RotationZ, SGate, TGate};
use std::sync::Arc;
use std::time::Instant;

fn create_benchmark_circuit(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::with_capacity(num_qubits, num_qubits * 20);

    for qubit_idx in 0..num_qubits {
        let qubit = QubitId::new(qubit_idx);

        // Start with some single-qubit gates
        circuit.add_gate(Arc::new(Hadamard), &[qubit]).unwrap();
        circuit
            .add_gate(Arc::new(RotationZ::new(0.5)), &[qubit])
            .unwrap();
        circuit.add_gate(Arc::new(Hadamard), &[qubit]).unwrap();

        // Some that can be fused
        circuit.add_gate(Arc::new(SGate), &[qubit]).unwrap();
        circuit.add_gate(Arc::new(TGate), &[qubit]).unwrap();
        circuit.add_gate(Arc::new(PauliX), &[qubit]).unwrap();

        // Self-inverse pair
        circuit.add_gate(Arc::new(PauliY), &[qubit]).unwrap();
        circuit.add_gate(Arc::new(PauliY), &[qubit]).unwrap();

        // Template pattern: H-Z-H
        circuit.add_gate(Arc::new(Hadamard), &[qubit]).unwrap();
        circuit.add_gate(Arc::new(PauliZ), &[qubit]).unwrap();
        circuit.add_gate(Arc::new(Hadamard), &[qubit]).unwrap();

        // Add entangling gates
        if qubit_idx + 1 < num_qubits {
            let next_qubit = QubitId::new(qubit_idx + 1);
            circuit
                .add_gate(Arc::new(simq_gates::standard::CNot), &[qubit, next_qubit])
                .unwrap();
        }
    }

    circuit
}

fn benchmark_optimization_level(level: OptimizationLevel, num_qubits: usize, iterations: usize) {
    let level_name = match level {
        OptimizationLevel::O0 => "O0",
        OptimizationLevel::O1 => "O1",
        OptimizationLevel::O2 => "O2",
        OptimizationLevel::O3 => "O3",
    };

    let compiler = create_compiler(level);
    let original_circuit = create_benchmark_circuit(num_qubits);
    let original_gate_count = original_circuit.len();

    println!(
        "\n=== Benchmarking {} ({} qubits, {} iterations) ===",
        level_name, num_qubits, iterations
    );
    println!("Original circuit: {} gates", original_gate_count);

    let mut total_time = 0u128;
    let mut final_gate_count = 0;

    for i in 0..iterations {
        let mut circuit = original_circuit.clone();

        let start = Instant::now();
        let result = compiler.compile(&mut circuit).unwrap();
        let duration = start.elapsed();

        total_time += duration.as_micros();
        final_gate_count = circuit.len();

        if i == 0 {
            // Print details for first iteration
            println!("\nFirst iteration details:");
            println!("  Optimized circuit: {} gates", final_gate_count);
            println!(
                "  Reduction: {} gates ({:.1}%)",
                original_gate_count - final_gate_count,
                100.0 * (original_gate_count - final_gate_count) as f64
                    / original_gate_count as f64
            );
            println!("  Compilation time: {} µs", duration.as_micros());
            println!("  Modified: {}", result.modified);
            println!("\n  Pass statistics:");
            for stat in &result.pass_stats {
                if stat.modified {
                    println!(
                        "    - {}: {} µs (applied {} times)",
                        stat.pass_name, stat.time_us, stat.applications
                    );
                }
            }
        }
    }

    let avg_time = total_time / iterations as u128;
    println!("\nAverage over {} iterations:", iterations);
    println!("  Time: {} µs", avg_time);
    println!("  Final gate count: {}", final_gate_count);
    println!("  Gates per microsecond: {:.2}", original_gate_count as f64 / avg_time as f64);
}

fn compare_optimization_levels(num_qubits: usize) {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   Optimization Level Comparison ({} qubits)              ║", num_qubits);
    println!("╚═══════════════════════════════════════════════════════════╝");

    let iterations = 100;

    benchmark_optimization_level(OptimizationLevel::O0, num_qubits, iterations);
    benchmark_optimization_level(OptimizationLevel::O1, num_qubits, iterations);
    benchmark_optimization_level(OptimizationLevel::O2, num_qubits, iterations);
    benchmark_optimization_level(OptimizationLevel::O3, num_qubits, iterations);
}

fn measure_scalability() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║   Scalability Test (O2 optimization)                     ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    let compiler = create_compiler(OptimizationLevel::O2);
    let iterations = 50;

    for num_qubits in [5, 10, 20, 50] {
        let original_circuit = create_benchmark_circuit(num_qubits);
        let original_gates = original_circuit.len();

        let mut total_time = 0u128;
        let mut final_gates = 0;

        for _ in 0..iterations {
            let mut circuit = original_circuit.clone();
            let start = Instant::now();
            compiler.compile(&mut circuit).unwrap();
            total_time += start.elapsed().as_micros();
            final_gates = circuit.len();
        }

        let avg_time = total_time / iterations as u128;
        println!("\n{} qubits ({} gates → {} gates):", num_qubits, original_gates, final_gates);
        println!("  Average time: {} µs", avg_time);
        println!("  Time per gate: {:.2} ns", (avg_time as f64 * 1000.0) / original_gates as f64);
        println!(
            "  Reduction: {:.1}%",
            100.0 * (original_gates - final_gates) as f64 / original_gates as f64
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   SimQ Compiler Optimization Pass Benchmarks             ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("This example demonstrates manual benchmarking of optimization passes.");
    println!("For comprehensive automated benchmarks with statistical analysis:");
    println!("  cargo bench --bench optimization_passes\n");

    // Compare optimization levels on a medium-sized circuit
    compare_optimization_levels(10);

    // Measure how performance scales with circuit size
    measure_scalability();

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║   Benchmark Complete                                      ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("\nFor detailed HTML reports and statistical analysis, run:");
    println!("  cargo bench --bench optimization_passes");
    println!("\nThen view the reports at:");
    println!("  target/criterion/report/index.html\n");

    Ok(())
}
