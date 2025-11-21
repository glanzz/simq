//! ASCII Circuit Rendering Demo
//!
//! Demonstrates how to render quantum circuits as ASCII art.
//!
//! Run with: cargo run --example ascii_circuit -p simq-core

use simq_core::{AsciiConfig, Circuit, DynamicCircuitBuilder, Gate};
use std::sync::Arc;

// Simple mock gates for demonstration
#[derive(Debug)]
struct H;
impl Gate for H {
    fn name(&self) -> &str { "H" }
    fn num_qubits(&self) -> usize { 1 }
}

#[derive(Debug)]
struct X;
impl Gate for X {
    fn name(&self) -> &str { "X" }
    fn num_qubits(&self) -> usize { 1 }
}

#[derive(Debug)]
struct T;
impl Gate for T {
    fn name(&self) -> &str { "T" }
    fn num_qubits(&self) -> usize { 1 }
}

#[derive(Debug)]
struct CNOT;
impl Gate for CNOT {
    fn name(&self) -> &str { "CNOT" }
    fn num_qubits(&self) -> usize { 2 }
}

#[derive(Debug)]
struct CZ;
impl Gate for CZ {
    fn name(&self) -> &str { "CZ" }
    fn num_qubits(&self) -> usize { 2 }
}

#[derive(Debug)]
struct RZ(f64);
impl Gate for RZ {
    fn name(&self) -> &str { "RZ" }
    fn num_qubits(&self) -> usize { 1 }
    fn description(&self) -> String {
        format!("RZ({:.3})", self.0)
    }
}

fn main() {
    println!("=== ASCII Circuit Renderer Demo ===\n");

    // Build a sample circuit
    let mut builder = DynamicCircuitBuilder::new(4);

    builder.apply_gate(Arc::new(H), &[0]).unwrap();
    builder.apply_gate(Arc::new(H), &[1]).unwrap();
    builder.apply_gate(Arc::new(CNOT), &[0, 2]).unwrap();
    builder.apply_gate(Arc::new(X), &[3]).unwrap();
    builder.apply_gate(Arc::new(CZ), &[1, 3]).unwrap();
    builder.apply_gate(Arc::new(T), &[0]).unwrap();
    builder
        .apply_gate(Arc::new(RZ(std::f64::consts::PI / 4.0)), &[2])
        .unwrap();

    let circuit = builder.build();

    // Default rendering (auto-detects terminal width)
    println!("Default (auto terminal width):");
    println!("{}\n", circuit.to_ascii());

    // Custom width
    println!("Custom width (80 chars):");
    let config = AsciiConfig {
        max_width: 80,
        ..Default::default()
    };
    println!("{}\n", circuit.to_ascii_with_config(&config));

    // Narrow terminal
    println!("Narrow terminal (50 chars):");
    let config = AsciiConfig {
        max_width: 50,
        ..Default::default()
    };
    println!("{}\n", circuit.to_ascii_with_config(&config));

    // Compact mode for very narrow
    println!("Compact mode (40 chars):");
    let config = AsciiConfig {
        max_width: 40,
        compact: true,
        ..Default::default()
    };
    println!("{}\n", circuit.to_ascii_with_config(&config));

    // No labels
    println!("Without qubit labels:");
    let config = AsciiConfig {
        max_width: 60,
        show_labels: false,
        ..Default::default()
    };
    println!("{}\n", circuit.to_ascii_with_config(&config));

    // Empty circuit
    println!("Empty circuit:");
    let empty = Circuit::new(3);
    println!("{}", empty.to_ascii());
}
