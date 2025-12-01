//! Circuit analysis example
//!
//! This example demonstrates how to use the circuit analysis module
//! to analyze quantum circuits for:
//! - Gate statistics (counts, depth)
//! - Resource estimation (memory, time)
//! - Parallelism opportunities
//!
//! Run with:
//! ```bash
//! cargo run --example circuit_analysis
//! ```

use simq_compiler::{CircuitAnalysis, ResourceEstimate};
use simq_core::{Circuit, QubitId};
use simq_gates::{CNot, Hadamard, PauliX, RotationZ};
use std::sync::Arc;

fn main() {
    println!("=== SimQ Circuit Analysis Example ===\n");

    // Example 1: Small circuit (Bell state preparation)
    println!("Example 1: Bell State Circuit");
    println!("{}", "=".repeat(50));
    let bell_circuit = create_bell_state_circuit();
    analyze_circuit(&bell_circuit);
    println!("\n");

    // Example 2: Larger circuit (QAOA-like circuit)
    println!("Example 2: QAOA-style Circuit");
    println!("{}", "=".repeat(50));
    let qaoa_circuit = create_qaoa_style_circuit(5, 2);
    analyze_circuit(&qaoa_circuit);
    println!("\n");

    // Example 3: Deep circuit (many layers)
    println!("Example 3: Deep Circuit");
    println!("{}", "=".repeat(50));
    let deep_circuit = create_deep_circuit(3, 10);
    analyze_circuit(&deep_circuit);
    println!("\n");

    // Example 4: Resource estimation for different qubit counts
    println!("Example 4: Resource Scaling");
    println!("{}", "=".repeat(50));
    analyze_resource_scaling();
}

/// Create a Bell state circuit: H(0), CNOT(0, 1)
fn create_bell_state_circuit() -> Circuit {
    let mut circuit = Circuit::new(2);

    circuit
        .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
        .unwrap();

    circuit
}

/// Create a QAOA-style circuit with alternating layers
fn create_qaoa_style_circuit(num_qubits: usize, num_layers: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    // Initial Hadamard layer
    for i in 0..num_qubits {
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(i)])
            .unwrap();
    }

    // Alternating layers
    for _ in 0..num_layers {
        // Problem layer (RZ gates)
        for i in 0..num_qubits {
            circuit
                .add_gate(Arc::new(RotationZ::new(0.5)), &[QubitId::new(i)])
                .unwrap();
        }

        // Entangling layer (CNOT ladder)
        for i in 0..num_qubits - 1 {
            circuit
                .add_gate(Arc::new(CNot), &[QubitId::new(i), QubitId::new(i + 1)])
                .unwrap();
        }

        // Mixer layer (X rotations)
        for i in 0..num_qubits {
            circuit
                .add_gate(Arc::new(PauliX), &[QubitId::new(i)])
                .unwrap();
        }
    }

    circuit
}

/// Create a deep circuit with many sequential layers
fn create_deep_circuit(num_qubits: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);

    for layer in 0..depth {
        for i in 0..num_qubits {
            if layer % 2 == 0 {
                circuit
                    .add_gate(Arc::new(Hadamard), &[QubitId::new(i)])
                    .unwrap();
            } else {
                circuit
                    .add_gate(Arc::new(PauliX), &[QubitId::new(i)])
                    .unwrap();
            }
        }
    }

    circuit
}

/// Analyze and display circuit information
fn analyze_circuit(circuit: &Circuit) {
    // Full analysis
    match CircuitAnalysis::analyze(circuit) {
        Ok(analysis) => {
            println!("{}", analysis);

            // Additional insights
            println!("Additional Insights:");
            println!(
                "  Gate efficiency: {:.1}%",
                (analysis.parallelism_factor() / analysis.statistics.total_gates as f64) * 100.0
            );

            if analysis.statistics.two_qubit_fraction() > 0.3 {
                println!("  ⚠️  High two-qubit gate fraction (expensive on hardware)");
            }

            // Memory warning
            if analysis.resources.dense_memory_bytes > 1024 * 1024 * 1024 {
                println!("  ⚠️  Circuit requires > 1GB memory");
            } else {
                println!("  ✓ Circuit fits comfortably in memory");
            }
        },
        Err(e) => {
            eprintln!("Error analyzing circuit: {}", e);
        },
    }
}

/// Analyze how resources scale with qubit count
fn analyze_resource_scaling() {
    println!("Memory requirements by qubit count:\n");

    for qubits in [10, 20, 25, 30, 35, 40] {
        let circuit = Circuit::new(qubits);
        if let Ok(resources) = ResourceEstimate::from_circuit(&circuit) {
            let memory_str = ResourceEstimate::format_memory(resources.dense_memory_bytes);
            let fits_in_32gb = resources.fits_in_memory(32 * 1024 * 1024 * 1024);
            let indicator = if fits_in_32gb { "✓" } else { "✗" };

            println!("  {} qubits: {} {}", qubits, memory_str, indicator);
        }
    }

    println!("\n✓ = Fits in 32GB RAM");
    println!("✗ = Requires > 32GB RAM\n");

    // Max qubits calculation
    let max_qubits_32gb = ResourceEstimate::max_qubits_for_memory(32 * 1024 * 1024 * 1024);
    let max_qubits_64gb = ResourceEstimate::max_qubits_for_memory(64 * 1024 * 1024 * 1024);

    println!("Maximum qubits:");
    println!("  32 GB RAM: {} qubits", max_qubits_32gb);
    println!("  64 GB RAM: {} qubits", max_qubits_64gb);
}
