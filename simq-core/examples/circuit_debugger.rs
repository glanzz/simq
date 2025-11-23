//! Circuit Debugger Demo
//!
//! Demonstrates step-by-step circuit debugging with breakpoints,
//! history tracking, and visualization.
//!
//! Run with: cargo run --example circuit_debugger -p simq-core

use simq_core::{Circuit, CircuitDebugger, DynamicCircuitBuilder, QubitId};
use simq_gates::standard::*;
use std::sync::Arc;

fn main() {
    println!("=== Circuit Debugger Demo ===\n");

    // Create a Bell state circuit
    let mut circuit = Circuit::new(2);
    circuit
        .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
        .unwrap();

    demo_basic_stepping(&circuit);
    demo_breakpoints(&circuit);
    demo_history(&circuit);
    demo_complex_circuit();
}

fn demo_basic_stepping(circuit: &Circuit) {
    println!("=== Demo 1: Basic Stepping ===\n");

    let mut debugger = CircuitDebugger::new(circuit);

    println!("Initial state:");
    println!("{}\n", debugger.status());

    // Step through each gate
    while debugger.has_next() {
        println!("Before step {}:", debugger.step_number());
        if let Some(gate) = debugger.current_gate_name() {
            println!("  Next gate: {} on qubits {:?}", gate, debugger.current_qubits());
        }

        debugger.step();

        println!("After step {}:", debugger.step_number());
        println!("  Executed: {}\n", debugger.history().last().unwrap());
    }

    println!("Final state:");
    println!("{}\n", debugger.status());
    println!();
}

fn demo_breakpoints(circuit: &Circuit) {
    println!("=== Demo 2: Breakpoints ===\n");

    let mut debugger = CircuitDebugger::new(circuit);

    // Set breakpoint at step 1
    debugger.add_breakpoint(1);
    println!("Added breakpoint at step 1");
    println!("Breakpoints: {:?}\n", debugger.breakpoints());

    // Continue until breakpoint
    println!("Continuing execution...");
    if debugger.continue_execution() {
        println!("Hit breakpoint at step {}", debugger.step_number());
        println!("Current gate: {:?}\n", debugger.current_gate_name());
    }

    // Continue to end
    println!("Continuing to end...");
    debugger.continue_execution();
    println!("Reached end at step {}\n", debugger.step_number());
    println!();
}

fn demo_history(circuit: &Circuit) {
    println!("=== Demo 3: Execution History ===\n");

    let mut debugger = CircuitDebugger::new(circuit);

    // Execute all gates
    while debugger.step() {}

    // Print execution trace
    debugger.print_trace();
    println!();

    // Get executed circuit
    let executed = debugger.to_executed_circuit();
    println!("Executed circuit has {} gates", executed.len());
    println!();
}

fn demo_complex_circuit() {
    println!("=== Demo 4: Complex Circuit with Visualization ===\n");

    // Create a more complex circuit
    let mut builder = DynamicCircuitBuilder::new(3);

    // Prepare superposition
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(Hadamard), &[1]).unwrap();
    builder.apply_gate(Arc::new(Hadamard), &[2]).unwrap();

    // Apply some rotations
    builder
        .apply_gate(Arc::new(RotationX::new(std::f64::consts::PI / 4.0)), &[0])
        .unwrap();
    builder
        .apply_gate(Arc::new(RotationY::new(std::f64::consts::PI / 2.0)), &[1])
        .unwrap();

    // Entangle qubits
    builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[1, 2]).unwrap();

    // Some final single-qubit gates
    builder.apply_gate(Arc::new(PauliZ), &[0]).unwrap();
    builder.apply_gate(Arc::new(SGate), &[1]).unwrap();
    builder.apply_gate(Arc::new(TGate), &[2]).unwrap();

    let circuit = builder.build();

    let mut debugger = CircuitDebugger::new(&circuit);

    // Set breakpoints at interesting points
    debugger.add_breakpoint(3); // After superposition
    debugger.add_breakpoint(5); // After rotations

    println!("Circuit has {} gates", debugger.total_gates());
    println!("Breakpoints: {:?}\n", debugger.breakpoints());

    // Run until first breakpoint
    debugger.continue_execution();
    println!("\n--- Breakpoint 1 (after superposition) ---");
    println!("{}", debugger.visualize_current_position());

    // Run until next breakpoint
    debugger.continue_execution();
    println!("\n--- Breakpoint 2 (after rotations) ---");
    println!("Executed so far:");
    for step in debugger.history() {
        println!("  {}", step);
    }
    println!();

    // Step through remaining gates one by one
    println!("Stepping through remaining gates:");
    while debugger.step() {
        if let Some(last_step) = debugger.history().last() {
            println!("  Executed: {}", last_step);
        }
    }

    println!("\n--- Final State ---");
    debugger.print_trace();
}
