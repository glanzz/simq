//! Example: Circuit validation with DAG analysis
//!
//! This example demonstrates how to validate quantum circuits using DAG analysis,
//! including cycle detection, dependency validation, and parallelism analysis.

use simq_core::gate::Gate;
use simq_core::{Circuit, QubitId, Result};
use std::sync::Arc;

// Mock gate implementations for demonstration
#[derive(Debug)]
struct HadamardGate;

impl Gate for HadamardGate {
    fn name(&self) -> &str {
        "H"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn is_hermitian(&self) -> bool {
        true
    }
}

#[derive(Debug)]
struct CnotGate;

impl Gate for CnotGate {
    fn name(&self) -> &str {
        "CNOT"
    }

    fn num_qubits(&self) -> usize {
        2
    }
}

#[derive(Debug)]
struct XGate;

impl Gate for XGate {
    fn name(&self) -> &str {
        "X"
    }

    fn num_qubits(&self) -> usize {
        1
    }
}

fn main() -> Result<()> {
    println!("=== SimQ Circuit Validation Example ===\n");

    // Example 1: Basic validation
    println!("1. Basic Circuit Validation");
    println!("----------------------------");
    let mut circuit = Circuit::new(2);
    let h_gate = Arc::new(HadamardGate);
    let cnot_gate = Arc::new(CnotGate);

    circuit.add_gate(h_gate, &[QubitId::new(0)])?;
    circuit.add_gate(cnot_gate, &[QubitId::new(0), QubitId::new(1)])?;

    match circuit.validate() {
        Ok(()) => println!("✓ Circuit validation passed"),
        Err(e) => println!("✗ Circuit validation failed: {}", e),
    }
    println!();

    // Example 2: DAG validation
    println!("2. DAG Validation");
    println!("-----------------");
    match circuit.validate_dag() {
        Ok(report) => {
            println!("✓ DAG validation passed");
            if report.has_warnings() {
                println!("Warnings:");
                for warning in report.warnings() {
                    println!("  - {}", warning.format());
                }
            }
        }
        Err(e) => println!("✗ DAG validation failed: {}", e),
    }
    println!();

    // Example 3: Circuit depth computation
    println!("3. Circuit Depth Analysis");
    println!("-------------------------");
    match circuit.compute_depth() {
        Ok(depth) => {
            println!("Circuit depth: {} (considering parallelism)", depth);
            println!("Sequential depth: {} (number of operations)", circuit.operations().count());
        }
        Err(e) => println!("Failed to compute depth: {}", e),
    }
    println!();

    // Example 4: Parallelism analysis
    println!("4. Parallelism Analysis");
    println!("-----------------------");
    match circuit.analyze_parallelism() {
        Ok(analysis) => {
            println!("Parallelism factor: {:.2}", analysis.parallelism_factor);
            println!("Max parallelism: {} operations", analysis.max_parallelism);
            println!("Number of layers: {}", analysis.num_layers());
            println!("Average parallelism per layer: {:.2}", analysis.avg_parallelism());
            println!("\nExecution layers:");
            for (i, layer) in analysis.layers.iter().enumerate() {
                println!("  Layer {}: {} operations", i, layer.len());
            }
        }
        Err(e) => println!("Failed to analyze parallelism: {}", e),
    }
    println!();

    // Example 5: Parallel gates
    println!("5. Parallel Gates Example");
    println!("-------------------------");
    let mut parallel_circuit = Circuit::new(3);
    let h_gate = Arc::new(HadamardGate);

    // Three parallel H gates on different qubits
    parallel_circuit.add_gate(h_gate.clone(), &[QubitId::new(0)])?;
    parallel_circuit.add_gate(h_gate.clone(), &[QubitId::new(1)])?;
    parallel_circuit.add_gate(h_gate, &[QubitId::new(2)])?;

    match parallel_circuit.analyze_parallelism() {
        Ok(analysis) => {
            println!("Circuit with 3 parallel H gates:");
            println!("  Depth: {} (all gates can run in parallel)", analysis.num_layers());
            println!("  Parallelism factor: {:.2}", analysis.parallelism_factor);
            println!("  Max parallelism: {} operations", analysis.max_parallelism);
        }
        Err(e) => println!("Failed to analyze: {}", e),
    }
    println!();

    // Example 6: Sequential circuit
    println!("6. Sequential Circuit Example");
    println!("-----------------------------");
    let mut sequential_circuit = Circuit::new(2);
    let h_gate = Arc::new(HadamardGate);
    let cnot_gate = Arc::new(CnotGate);
    let x_gate = Arc::new(XGate);

    // Sequential: H(0) -> CNOT(0,1) -> X(1)
    sequential_circuit.add_gate(h_gate, &[QubitId::new(0)])?;
    sequential_circuit.add_gate(cnot_gate, &[QubitId::new(0), QubitId::new(1)])?;
    sequential_circuit.add_gate(x_gate, &[QubitId::new(1)])?;

    match sequential_circuit.analyze_parallelism() {
        Ok(analysis) => {
            println!("Sequential circuit (H -> CNOT -> X):");
            println!("  Depth: {} (all gates are sequential)", analysis.num_layers());
            println!("  Parallelism factor: {:.2}", analysis.parallelism_factor);
            println!("  Max parallelism: {} operations", analysis.max_parallelism);
        }
        Err(e) => println!("Failed to analyze: {}", e),
    }
    println!();

    // Example 7: Acyclic check
    println!("7. Acyclic Check");
    println!("----------------");
    match circuit.is_acyclic() {
        Ok(true) => println!("✓ Circuit is acyclic (no cycles detected)"),
        Ok(false) => println!("✗ Circuit contains cycles"),
        Err(e) => println!("Failed to check: {}", e),
    }
    println!();

    println!("✓ Validation example completed successfully!");

    Ok(())
}

