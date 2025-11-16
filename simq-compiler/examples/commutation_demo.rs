//! Demonstration of enhanced gate commutation rules
//!
//! This example shows how the enhanced commutation pass can recognize and reorder
//! various quantum gate patterns.

use simq_compiler::passes::{GateCommutation, OptimizationPass};
use simq_core::{Circuit, QubitId};
use simq_gates::standard::{CNot, CZ, Hadamard, PauliX, PauliY, PauliZ, RotationX, SGate, TGate};
use std::sync::Arc;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║       Enhanced Gate Commutation Rules Demonstration           ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Example 1: Diagonal gates commutation
    diagonal_gates_example();

    // Example 2: Rotation gates on same axis
    rotation_gates_example();

    // Example 3: CNOT commutation patterns
    cnot_commutation_example();

    // Example 4: Mixed single/two-qubit gates
    mixed_gates_example();

    // Example 5: Complex optimization scenario
    complex_optimization_example();

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                    Demonstration Complete                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
}

/// Example 1: Diagonal gates (Z, S, T) can be reordered
fn diagonal_gates_example() {
    println!("📊 Example 1: Diagonal Gates Commutation");
    println!("─────────────────────────────────────────\n");

    let mut circuit = Circuit::new(3);

    // Add gates: H, Z, S, T on q0
    // H is a barrier, but Z, S, T can be reordered
    circuit
        .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(PauliZ), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(SGate), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(TGate), &[QubitId::new(0)])
        .unwrap();

    println!("Before commutation:");
    print_circuit(&circuit);

    let pass = GateCommutation::new();
    let modified = pass.apply(&mut circuit).unwrap();

    println!("\nAfter commutation:");
    print_circuit(&circuit);
    println!("Modified: {}\n", modified);

    println!("✅ Diagonal gates (Z, S, T) can commute with each other!");
    println!("   This allows better grouping for fusion optimization.\n");
}

/// Example 2: Rotation gates around the same axis commute
fn rotation_gates_example() {
    println!("📊 Example 2: Rotation Gates (Same Axis)");
    println!("─────────────────────────────────────────\n");

    let mut circuit = Circuit::new(2);

    // Add multiple RX gates on q0 - they commute!
    circuit
        .add_gate(Arc::new(RotationX::new(std::f64::consts::PI / 4.0)), &[
            QubitId::new(0),
        ])
        .unwrap();
    circuit
        .add_gate(Arc::new(PauliX), &[QubitId::new(1)])
        .unwrap(); // On different qubit
    circuit
        .add_gate(Arc::new(RotationX::new(std::f64::consts::PI / 8.0)), &[
            QubitId::new(0),
        ])
        .unwrap();

    println!("Before commutation:");
    print_circuit(&circuit);

    let pass = GateCommutation::new();
    let modified = pass.apply(&mut circuit).unwrap();

    println!("\nAfter commutation:");
    print_circuit(&circuit);
    println!("Modified: {}\n", modified);

    println!("✅ RX(π/4) and RX(π/8) can be grouped together!");
    println!("   Same-axis rotations commute, enabling fusion.\n");
}

/// Example 3: CNOT gates with same control or same target
fn cnot_commutation_example() {
    println!("📊 Example 3: CNOT Commutation Patterns");
    println!("─────────────────────────────────────────\n");

    let mut circuit = Circuit::new(4);

    // CNOT with same control, different targets
    circuit
        .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
        .unwrap();
    circuit
        .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(2)])
        .unwrap();

    // CNOT with same target, different controls
    circuit
        .add_gate(Arc::new(CNot), &[QubitId::new(1), QubitId::new(3)])
        .unwrap();
    circuit
        .add_gate(Arc::new(CNot), &[QubitId::new(2), QubitId::new(3)])
        .unwrap();

    println!("Before commutation:");
    print_circuit(&circuit);

    let pass = GateCommutation::new();
    let modified = pass.apply(&mut circuit).unwrap();

    println!("\nAfter commutation:");
    print_circuit(&circuit);
    println!("Modified: {}\n", modified);

    println!("✅ CNOT gates can commute in two patterns:");
    println!("   1. Same control, different targets");
    println!("   2. Same target, different controls (NEW!)");
    println!("   This enables better parallelization opportunities.\n");
}

/// Example 4: Single-qubit gates with two-qubit gates
fn mixed_gates_example() {
    println!("📊 Example 4: Mixed Single/Two-Qubit Gates");
    println!("─────────────────────────────────────────\n");

    let mut circuit = Circuit::new(3);

    // Z on control commutes with CNOT
    circuit
        .add_gate(Arc::new(PauliZ), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
        .unwrap();

    // X on target commutes with CNOT
    circuit
        .add_gate(Arc::new(PauliX), &[QubitId::new(1)])
        .unwrap();
    circuit
        .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
        .unwrap();

    // Diagonal gates commute with CZ
    circuit
        .add_gate(Arc::new(SGate), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(CZ), &[QubitId::new(0), QubitId::new(2)])
        .unwrap();

    println!("Before commutation:");
    print_circuit(&circuit);

    let pass = GateCommutation::new();
    let modified = pass.apply(&mut circuit).unwrap();

    println!("\nAfter commutation:");
    print_circuit(&circuit);
    println!("Modified: {}\n", modified);

    println!("✅ Mixed gate commutation rules (NEW!):");
    println!("   • Z commutes with CNOT on control");
    println!("   • X commutes with CNOT on target");
    println!("   • Diagonal gates commute with CZ\n");
}

/// Example 5: Complex scenario showing multiple optimization opportunities
fn complex_optimization_example() {
    println!("📊 Example 5: Complex Optimization Scenario");
    println!("─────────────────────────────────────────\n");

    let mut circuit = Circuit::new(4);

    // Interleaved gates that can be reordered
    circuit
        .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(PauliZ), &[QubitId::new(1)])
        .unwrap();
    circuit
        .add_gate(Arc::new(SGate), &[QubitId::new(1)])
        .unwrap(); // Can group with Z
    circuit
        .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(2)])
        .unwrap();
    circuit
        .add_gate(Arc::new(TGate), &[QubitId::new(1)])
        .unwrap(); // Can group with Z,S
    circuit
        .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(3)])
        .unwrap(); // Same control
    circuit
        .add_gate(Arc::new(PauliY), &[QubitId::new(2)])
        .unwrap();

    println!("Before commutation (gates: {}):", circuit.len());
    print_circuit(&circuit);

    let pass = GateCommutation::new();
    let modified = pass.apply(&mut circuit).unwrap();

    println!("\nAfter commutation (gates: {}):", circuit.len());
    print_circuit(&circuit);
    println!("Modified: {}\n", modified);

    println!("✅ Complex optimization demonstrates:");
    println!("   • Grouping diagonal gates (Z, S, T) on q1");
    println!("   • Reordering CNOTs with same control");
    println!("   • Creating fusion opportunities");
    println!("   • Preserving circuit semantics\n");
}

/// Helper function to print circuit gates
fn print_circuit(circuit: &Circuit) {
    for (i, op) in circuit.operations().enumerate() {
        let qubits: Vec<String> = op.qubits().iter().map(|q| format!("q{}", q.index())).collect();
        println!("  [{}] {}({})", i, op.gate().name(), qubits.join(", "));
    }
}
