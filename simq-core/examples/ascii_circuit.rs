//! ASCII Circuit Rendering Demo
//!
//! Demonstrates how to render quantum circuits as ASCII art with all gate types.
//!
//! Run with: cargo run --example ascii_circuit -p simq-core

use simq_core::{AsciiConfig, Circuit, DynamicCircuitBuilder, Gate, QubitId};
use simq_gates::standard::*;
use std::sync::Arc;

// Custom gate examples
#[derive(Debug)]
struct GroverOracle;
impl Gate for GroverOracle {
    fn name(&self) -> &str { "Oracle" }
    fn num_qubits(&self) -> usize { 1 }
    fn description(&self) -> String { "Grover".to_string() }
}

#[derive(Debug)]
struct GroverDiffusion;
impl Gate for GroverDiffusion {
    fn name(&self) -> &str { "Diffusion" }
    fn num_qubits(&self) -> usize { 1 }
    fn description(&self) -> String { "Diffusion".to_string() }
}

#[derive(Debug)]
struct XXGate(f64);
impl Gate for XXGate {
    fn name(&self) -> &str { "XX" }
    fn num_qubits(&self) -> usize { 2 }
    fn description(&self) -> String { format!("XX({:.4})", self.0) }
}

#[derive(Debug)]
struct QFTGate;
impl Gate for QFTGate {
    fn name(&self) -> &str { "QFT" }
    fn num_qubits(&self) -> usize { 1 }
    fn description(&self) -> String { "QFT".to_string() }
}

fn main() {
    println!("=== ASCII Circuit Renderer Demo ===\n");

    // Demo 1: Basic single-qubit gates
    println!("1. Single-Qubit Gates:");
    let mut builder = DynamicCircuitBuilder::new(4);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliX), &[1]).unwrap();
    builder.apply_gate(Arc::new(PauliY), &[2]).unwrap();
    builder.apply_gate(Arc::new(PauliZ), &[3]).unwrap();
    builder.apply_gate(Arc::new(SGate), &[0]).unwrap();
    builder.apply_gate(Arc::new(TGate), &[1]).unwrap();
    builder.apply_gate(Arc::new(SXGate), &[2]).unwrap();
    builder.apply_gate(Arc::new(Identity), &[3]).unwrap();
    println!("{}\n", builder.build().to_ascii());

    // Demo 2: Rotation gates
    println!("2. Rotation Gates:");
    let mut builder = DynamicCircuitBuilder::new(3);
    builder.apply_gate(Arc::new(RotationX::new(std::f64::consts::PI / 4.0)), &[0]).unwrap();
    builder.apply_gate(Arc::new(RotationY::new(std::f64::consts::PI / 2.0)), &[1]).unwrap();
    builder.apply_gate(Arc::new(RotationZ::new(std::f64::consts::PI)), &[2]).unwrap();
    builder.apply_gate(Arc::new(Phase::new(0.5)), &[0]).unwrap();
    builder.apply_gate(Arc::new(U3::new(1.0, 2.0, 3.0)), &[1]).unwrap();
    println!("{}\n", builder.build().to_ascii());

    // Demo 3: Two-qubit gates
    println!("3. Two-Qubit Gates:");
    let mut builder = DynamicCircuitBuilder::new(4);
    builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    builder.apply_gate(Arc::new(CZ), &[2, 3]).unwrap();
    builder.apply_gate(Arc::new(Swap), &[0, 2]).unwrap();
    builder.apply_gate(Arc::new(ISwap), &[1, 3]).unwrap();
    builder.apply_gate(Arc::new(CY), &[0, 3]).unwrap();
    println!("{}\n", builder.build().to_ascii());

    // Demo 4: Three-qubit gates
    println!("4. Three-Qubit Gates:");
    let mut builder = DynamicCircuitBuilder::new(4);
    builder.apply_gate(Arc::new(Toffoli), &[0, 1, 2]).unwrap();
    builder.apply_gate(Arc::new(Fredkin), &[1, 2, 3]).unwrap();
    println!("{}\n", builder.build().to_ascii());

    // Demo 5: Complex circuit
    println!("5. Bell State + Teleportation-like Circuit:");
    let mut builder = DynamicCircuitBuilder::new(3);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[1, 2]).unwrap();
    builder.apply_gate(Arc::new(Hadamard), &[1]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[0, 2]).unwrap();
    builder.apply_gate(Arc::new(CZ), &[1, 2]).unwrap();
    println!("{}\n", builder.build().to_ascii());

    // Demo 6: Width adaptation
    println!("6. Narrow Terminal (40 chars):");
    let config = AsciiConfig {
        max_width: 40,
        compact: true,
        ..Default::default()
    };
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(RotationX::new(1.234)), &[1]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    println!("{}\n", builder.build().to_ascii_with_config(&config));

    // Demo 7: Without labels
    println!("7. Without Qubit Labels:");
    let config = AsciiConfig {
        max_width: 60,
        show_labels: false,
        ..Default::default()
    };
    let mut builder = DynamicCircuitBuilder::new(3);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(Hadamard), &[1]).unwrap();
    builder.apply_gate(Arc::new(Hadamard), &[2]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[1, 2]).unwrap();
    println!("{}\n", builder.build().to_ascii_with_config(&config));

    // Demo 8: Custom gates
    println!("8. Custom Gates:");
    let mut circuit = Circuit::new(3);
    circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(1)]).unwrap();
    circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(2)]).unwrap();
    circuit.add_gate(Arc::new(GroverOracle), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(Arc::new(GroverOracle), &[QubitId::new(1)]).unwrap();
    circuit.add_gate(Arc::new(GroverOracle), &[QubitId::new(2)]).unwrap();
    circuit.add_gate(Arc::new(GroverDiffusion), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(Arc::new(GroverDiffusion), &[QubitId::new(1)]).unwrap();
    circuit.add_gate(Arc::new(GroverDiffusion), &[QubitId::new(2)]).unwrap();
    println!("{}\n", circuit.to_ascii());

    // Demo 9: Parametric custom gate (XX interaction)
    println!("9. Parametric Custom Gate (XX):");
    let mut builder = DynamicCircuitBuilder::new(3);
    builder.apply_gate(Arc::new(XXGate(std::f64::consts::PI / 4.0)), &[0, 1]).unwrap();
    builder.apply_gate(Arc::new(XXGate(std::f64::consts::PI / 2.0)), &[1, 2]).unwrap();
    println!("{}\n", builder.build().to_ascii());

    // Demo 10: QFT-like circuit
    println!("10. QFT-like Custom Circuit:");
    let mut circuit = Circuit::new(4);
    for i in 0..4 {
        circuit.add_gate(Arc::new(QFTGate), &[QubitId::new(i)]).unwrap();
    }
    println!("{}\n", circuit.to_ascii());

    // Demo 11: Empty circuit
    println!("11. Empty Circuit:");
    let empty = Circuit::new(2);
    println!("{}", empty.to_ascii());
}
