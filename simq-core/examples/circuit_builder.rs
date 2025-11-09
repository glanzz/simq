//! Circuit builder examples demonstrating type-safe and dynamic circuit construction

use simq_core::{CircuitBuilder, DynamicCircuitBuilder, Gate};
use std::sync::Arc;

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
struct PhaseGate(f64);

impl Gate for PhaseGate {
    fn name(&self) -> &str {
        "P"
    }
    fn num_qubits(&self) -> usize {
        1
    }
    fn description(&self) -> String {
        format!("Phase gate with angle {}", self.0)
    }
}

fn main() {
    println!("=== SimQ Circuit Builder Examples ===\n");

    example_typed_builder();
    println!();

    example_dynamic_builder();
    println!();

    example_bell_state();
    println!();

    example_ghz_state();
    println!();

    example_parameterized_circuit();
}

fn example_typed_builder() {
    println!("Example 1: Type-Safe CircuitBuilder<N>");
    println!("----------------------------------------");
    println!("Using const generics for compile-time size checking\n");

    let mut builder = CircuitBuilder::<2>::new();
    let [q0, q1] = builder.qubits();

    let h_gate = Arc::new(HadamardGate);
    let cnot_gate = Arc::new(CnotGate);

    builder
        .apply_gate(h_gate, &[q0])
        .unwrap()
        .apply_gate(cnot_gate, &[q0, q1])
        .unwrap();

    let circuit = builder.build();
    println!("{}", circuit);
    println!("✓ Circuit built with compile-time type safety!");
}

fn example_dynamic_builder() {
    println!("Example 2: DynamicCircuitBuilder");
    println!("---------------------------------");
    println!("Using runtime-determined circuit size\n");

    // Circuit size determined at runtime (e.g., from config, user input)
    let num_qubits = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    println!("Building circuit with {} qubits (specify via CLI arg)", num_qubits);

    let mut builder = DynamicCircuitBuilder::new(num_qubits);
    let h_gate = Arc::new(HadamardGate);

    // Apply Hadamard to all qubits
    for i in 0..num_qubits {
        builder.apply_gate(h_gate.clone(), &[i]).unwrap();
    }

    let circuit = builder.build();
    println!("{}", circuit);
    println!("✓ Circuit built with {} qubits", num_qubits);
}

fn example_bell_state() {
    println!("Example 3: Bell State Preparation");
    println!("----------------------------------");
    println!("Creating maximally entangled state: (|00⟩ + |11⟩)/√2\n");

    let mut builder = CircuitBuilder::<2>::new();
    let [q0, q1] = builder.qubits();

    // Bell state circuit
    builder
        .apply_gate(Arc::new(HadamardGate), &[q0])
        .unwrap()
        .apply_gate(Arc::new(CnotGate), &[q0, q1])
        .unwrap();

    let circuit = builder.build();
    println!("{}", circuit);
    println!("✓ Bell state circuit created!");
    println!("  This prepares the maximally entangled EPR pair");
}

fn example_ghz_state() {
    println!("Example 4: GHZ State Preparation");
    println!("---------------------------------");
    println!("Creating 3-qubit GHZ state: (|000⟩ + |111⟩)/√2\n");

    let mut builder = CircuitBuilder::<3>::new();
    let [q0, q1, q2] = builder.qubits();

    let h_gate = Arc::new(HadamardGate);
    let cnot_gate = Arc::new(CnotGate);

    // GHZ state circuit
    builder
        .apply_gate(h_gate, &[q0])
        .unwrap()
        .apply_gate(cnot_gate.clone(), &[q0, q1])
        .unwrap()
        .apply_gate(cnot_gate, &[q1, q2])
        .unwrap();

    let circuit = builder.build();
    println!("{}", circuit);
    println!("✓ GHZ state circuit created!");
    println!("  This is a 3-qubit entangled state used in quantum protocols");
}

fn example_parameterized_circuit() {
    println!("Example 5: Parameterized Circuit");
    println!("---------------------------------");
    println!("Building a circuit with parameterized gates\n");

    let mut builder = CircuitBuilder::<3>::new();
    let qubits = builder.qubits();

    // Apply phase gates with different angles
    let angles = [0.0, std::f64::consts::PI / 4.0, std::f64::consts::PI / 2.0];

    for (q, angle) in qubits.iter().zip(angles.iter()) {
        let phase_gate = Arc::new(PhaseGate(*angle));
        builder.apply_gate(phase_gate, &[*q]).unwrap();
    }

    let circuit = builder.build();
    println!("{}", circuit);
    println!("✓ Parameterized circuit created with different phase angles!");
}
