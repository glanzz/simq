//! Example demonstrating gate fusion optimization
//!
//! This example shows how to use the gate fusion optimization pass to combine
//! adjacent single-qubit gates into fused gates, reducing circuit depth.

use simq_compiler::fusion::{fuse_single_qubit_gates, FusionConfig};
use simq_core::{Circuit, QubitId};
use simq_gates::standard::{Hadamard, PauliX, PauliY, PauliZ, RotationX, RotationZ, SGate, TGate};
use std::sync::Arc;
use simq_core::gate::Gate;

fn main() {
    println!("=== Gate Fusion Optimization Example ===\n");

    // Example 1: Simple fusion of Clifford gates
    example_simple_fusion();

    // Example 2: Fusion with rotation gates
    example_rotation_fusion();

    // Example 3: Mixed circuit with two-qubit gates
    example_mixed_circuit();

    // Example 4: Identity elimination
    example_identity_elimination();

    // Example 5: Custom fusion configuration
    example_custom_config();
}

fn example_simple_fusion() {
    println!("Example 1: Simple Fusion of Clifford Gates");
    println!("-------------------------------------------");

    let mut circuit = Circuit::new(2);
    let q0 = QubitId::new(0);
    let q1 = QubitId::new(1);

    // Add a sequence of gates on q0
    circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();
    circuit.add_gate(Arc::new(SGate) as Arc<dyn Gate>, &[q0]).unwrap();
    circuit.add_gate(Arc::new(TGate) as Arc<dyn Gate>, &[q0]).unwrap();
    circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();

    // Add gates on q1
    circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[q1]).unwrap();
    circuit.add_gate(Arc::new(PauliY) as Arc<dyn Gate>, &[q1]).unwrap();
    circuit.add_gate(Arc::new(PauliZ) as Arc<dyn Gate>, &[q1]).unwrap();

    println!("Original circuit:");
    println!("  Qubits: {}", circuit.num_qubits());
    println!("  Operations: {}", circuit.len());
    for (i, op) in circuit.operations().enumerate() {
        println!("    {}: {}", i, op);
    }

    // Apply fusion
    let optimized = fuse_single_qubit_gates(&circuit, None).unwrap();

    println!("\nOptimized circuit:");
    println!("  Qubits: {}", optimized.num_qubits());
    println!("  Operations: {}", optimized.len());
    for (i, op) in optimized.operations().enumerate() {
        println!("    {}: {} - {}", i, op, op.gate().description());
    }

    let reduction = ((circuit.len() - optimized.len()) as f64 / circuit.len() as f64) * 100.0;
    println!("\nReduction: {:.1}% fewer operations\n", reduction);
}

fn example_rotation_fusion() {
    println!("Example 2: Fusion of Rotation Gates");
    println!("------------------------------------");

    let mut circuit = Circuit::new(1);
    let q0 = QubitId::new(0);

    // Add multiple rotation gates
    circuit.add_gate(Arc::new(RotationX::new(0.5)) as Arc<dyn Gate>, &[q0]).unwrap();
    circuit.add_gate(Arc::new(RotationZ::new(0.25)) as Arc<dyn Gate>, &[q0]).unwrap();
    circuit.add_gate(Arc::new(RotationX::new(0.3)) as Arc<dyn Gate>, &[q0]).unwrap();
    circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();
    circuit.add_gate(Arc::new(RotationZ::new(0.1)) as Arc<dyn Gate>, &[q0]).unwrap();

    println!("Original circuit with {} operations", circuit.len());

    let optimized = fuse_single_qubit_gates(&circuit, None).unwrap();

    println!("Optimized circuit with {} operations", optimized.len());
    println!("\nNote: Rotation gates can be fused into a single unitary matrix,");
    println!("which can be more efficient than computing each rotation separately.\n");
}

fn example_mixed_circuit() {
    println!("Example 3: Mixed Circuit with Two-Qubit Gates");
    println!("----------------------------------------------");

    let mut circuit = Circuit::new(3);
    let q0 = QubitId::new(0);
    let q1 = QubitId::new(1);
    let q2 = QubitId::new(2);

    // Single-qubit gates on q0
    circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();
    circuit.add_gate(Arc::new(TGate) as Arc<dyn Gate>, &[q0]).unwrap();
    circuit.add_gate(Arc::new(SGate) as Arc<dyn Gate>, &[q0]).unwrap();

    // Two-qubit gate (breaks fusion chain)
    circuit.add_gate(Arc::new(simq_gates::standard::CNot) as Arc<dyn Gate>, &[q0, q1]).unwrap();

    // More single-qubit gates on q0 (new fusion chain)
    circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[q0]).unwrap();
    circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();

    // Single-qubit gates on q1
    circuit.add_gate(Arc::new(PauliZ) as Arc<dyn Gate>, &[q1]).unwrap();
    circuit.add_gate(Arc::new(TGate) as Arc<dyn Gate>, &[q1]).unwrap();

    // Gate on q2
    circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q2]).unwrap();

    println!("Original circuit: {} operations", circuit.len());

    let optimized = fuse_single_qubit_gates(&circuit, None).unwrap();

    println!("Optimized circuit: {} operations", optimized.len());
    println!("\nNote: Two-qubit gates break fusion chains, creating multiple");
    println!("separate fusion opportunities.\n");
}

fn example_identity_elimination() {
    println!("Example 4: Identity Elimination");
    println!("--------------------------------");

    let mut circuit = Circuit::new(1);
    let q0 = QubitId::new(0);

    // Add gates that cancel out (X * X = I)
    circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();
    circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[q0]).unwrap();
    circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[q0]).unwrap(); // Cancels with previous X
    circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap(); // Cancels with first H

    println!("Original circuit: {} operations", circuit.len());
    println!("  (Contains H, X, X, H - which simplifies to identity)");

    let optimized = fuse_single_qubit_gates(&circuit, None).unwrap();

    println!("\nOptimized circuit: {} operations", optimized.len());
    println!("  (Identity gate eliminated automatically)");

    if optimized.len() == 0 {
        println!("\nThe entire sequence was eliminated as it equals the identity!");
    }
    println!();
}

fn example_custom_config() {
    println!("Example 5: Custom Fusion Configuration");
    println!("---------------------------------------");

    let mut circuit = Circuit::new(1);
    let q0 = QubitId::new(0);

    // Add 5 T gates
    for _ in 0..5 {
        circuit.add_gate(Arc::new(TGate) as Arc<dyn Gate>, &[q0]).unwrap();
    }

    println!("Original circuit: {} T gates", circuit.len());

    // Default configuration (no max fusion size)
    let optimized_default = fuse_single_qubit_gates(&circuit, None).unwrap();
    println!("\nWith default config:");
    println!("  Operations: {} (all T gates fused into one)", optimized_default.len());

    // Custom configuration with max fusion size
    let config = FusionConfig {
        min_fusion_size: 2,
        max_fusion_size: Some(3), // Limit fusion chains to 3 gates
        eliminate_identity: true,
        identity_epsilon: 1e-10,
    };

    let optimized_custom = fuse_single_qubit_gates(&circuit, Some(config)).unwrap();
    println!("\nWith max_fusion_size = 3:");
    println!("  Operations: {} (creates multiple fused gates)", optimized_custom.len());
    println!("  (First 3 T gates fused, remaining 2 T gates fused separately)");

    // Configuration that disables identity elimination
    let mut circuit_xx = Circuit::new(1);
    circuit_xx.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[q0]).unwrap();
    circuit_xx.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[q0]).unwrap();

    let config_no_elim = FusionConfig {
        eliminate_identity: false,
        ..Default::default()
    };

    let optimized_no_elim = fuse_single_qubit_gates(&circuit_xx, Some(config_no_elim)).unwrap();
    println!("\nWith identity elimination disabled:");
    println!("  X, X circuit: {} operations (identity kept)", optimized_no_elim.len());

    println!();
}
