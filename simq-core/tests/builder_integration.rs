//! Integration tests for circuit builders

use simq_core::{CircuitBuilder, DynamicCircuitBuilder, Gate, Qubit};
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
struct ToffoliGate;

impl Gate for ToffoliGate {
    fn name(&self) -> &str {
        "TOFFOLI"
    }

    fn num_qubits(&self) -> usize {
        3
    }
}

#[test]
fn test_typed_builder_basic() {
    let mut builder = CircuitBuilder::<3>::new();
    let [q0, q1, q2] = builder.qubits();

    let h_gate = Arc::new(HadamardGate);
    let cnot_gate = Arc::new(CnotGate);

    builder
        .apply_gate(h_gate.clone(), &[q0])
        .unwrap()
        .apply_gate(cnot_gate.clone(), &[q0, q1])
        .unwrap()
        .apply_gate(cnot_gate, &[q1, q2])
        .unwrap()
        .apply_gate(h_gate, &[q2])
        .unwrap();

    let circuit = builder.build();
    assert_eq!(circuit.num_qubits(), 3);
    assert_eq!(circuit.len(), 4);
}

#[test]
fn test_dynamic_builder_basic() {
    let mut builder = DynamicCircuitBuilder::new(3);
    let h_gate = Arc::new(HadamardGate);
    let cnot_gate = Arc::new(CnotGate);

    builder
        .apply_gate(h_gate.clone(), &[0])
        .unwrap()
        .apply_gate(cnot_gate.clone(), &[0, 1])
        .unwrap()
        .apply_gate(cnot_gate, &[1, 2])
        .unwrap()
        .apply_gate(h_gate, &[2])
        .unwrap();

    let circuit = builder.build();
    assert_eq!(circuit.num_qubits(), 3);
    assert_eq!(circuit.len(), 4);
}

#[test]
fn test_typed_builder_prevents_out_of_bounds() {
    let builder = CircuitBuilder::<3>::new();

    // This should fail at qubit creation time
    let result = builder.qubit(5);
    assert!(result.is_err());
}

#[test]
fn test_dynamic_builder_runtime_size() {
    let sizes = vec![2, 5, 10, 100];

    for size in sizes {
        let builder = DynamicCircuitBuilder::new(size);
        assert_eq!(builder.num_qubits(), size);

        let circuit = builder.build();
        assert_eq!(circuit.num_qubits(), size);
    }
}

#[test]
fn test_builder_with_capacity() {
    let mut builder = CircuitBuilder::<5>::with_capacity(100);
    let h_gate = Arc::new(HadamardGate);

    // Add many gates without reallocation
    let qubits = builder.qubits();
    for q in &qubits {
        for _ in 0..20 {
            builder.apply_gate(h_gate.clone(), &[*q]).unwrap();
        }
    }

    assert_eq!(builder.num_operations(), 100);
}

#[test]
fn test_qubit_type_safety() {
    let builder3 = CircuitBuilder::<3>::new();
    let builder5 = CircuitBuilder::<5>::new();

    let q0_3: Qubit<3> = builder3.qubit(0).unwrap();
    let q0_5: Qubit<5> = builder5.qubit(0).unwrap();

    // These are different types and cannot be mixed
    // (This is verified at compile time, but we can verify the types are different)
    assert_eq!(Qubit::<3>::circuit_size(), 3);
    assert_eq!(Qubit::<5>::circuit_size(), 5);

    // Verify they're logically equal even though type-different
    assert_eq!(q0_3.index(), q0_5.index());
}

#[test]
fn test_bell_state_circuit_typed() {
    let mut builder = CircuitBuilder::<2>::new();
    let [q0, q1] = builder.qubits();

    // Create Bell state: (|00⟩ + |11⟩)/√2
    builder
        .apply_gate(Arc::new(HadamardGate), &[q0])
        .unwrap()
        .apply_gate(Arc::new(CnotGate), &[q0, q1])
        .unwrap();

    let circuit = builder.build();
    assert_eq!(circuit.num_qubits(), 2);
    assert_eq!(circuit.len(), 2);
}

#[test]
fn test_bell_state_circuit_dynamic() {
    let mut builder = DynamicCircuitBuilder::new(2);

    builder
        .apply_gate(Arc::new(HadamardGate), &[0])
        .unwrap()
        .apply_gate(Arc::new(CnotGate), &[0, 1])
        .unwrap();

    let circuit = builder.build();
    assert_eq!(circuit.num_qubits(), 2);
    assert_eq!(circuit.len(), 2);
}

#[test]
fn test_ghz_state_circuit() {
    let mut builder = CircuitBuilder::<3>::new();
    let [q0, q1, q2] = builder.qubits();

    // Create GHZ state: (|000⟩ + |111⟩)/√2
    let h_gate = Arc::new(HadamardGate);
    let cnot_gate = Arc::new(CnotGate);

    builder
        .apply_gate(h_gate, &[q0])
        .unwrap()
        .apply_gate(cnot_gate.clone(), &[q0, q1])
        .unwrap()
        .apply_gate(cnot_gate, &[q1, q2])
        .unwrap();

    let circuit = builder.build();
    assert_eq!(circuit.num_qubits(), 3);
    assert_eq!(circuit.len(), 3);
}

#[test]
fn test_three_qubit_gate() {
    let mut builder = CircuitBuilder::<5>::new();
    let qubits = builder.qubits();

    let toffoli = Arc::new(ToffoliGate);
    builder
        .apply_gate(toffoli, &[qubits[0], qubits[1], qubits[2]])
        .unwrap();

    assert_eq!(builder.num_operations(), 1);
}

#[test]
fn test_empty_circuit_validation() {
    let builder = CircuitBuilder::<5>::new();
    assert!(builder.validate().is_ok());
    assert!(builder.is_empty());

    let circuit = builder.build();
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_builder_equivalence() {
    // Build the same circuit with both builders
    let mut typed_builder = CircuitBuilder::<3>::new();
    let [q0, q1, q2] = typed_builder.qubits();

    let mut dynamic_builder = DynamicCircuitBuilder::new(3);

    let h_gate = Arc::new(HadamardGate);
    let cnot_gate = Arc::new(CnotGate);

    // Add same operations to both
    typed_builder
        .apply_gate(h_gate.clone(), &[q0])
        .unwrap()
        .apply_gate(cnot_gate.clone(), &[q0, q1])
        .unwrap()
        .apply_gate(h_gate.clone(), &[q2])
        .unwrap();

    dynamic_builder
        .apply_gate(h_gate.clone(), &[0])
        .unwrap()
        .apply_gate(cnot_gate.clone(), &[0, 1])
        .unwrap()
        .apply_gate(h_gate, &[2])
        .unwrap();

    let typed_circuit = typed_builder.build();
    let dynamic_circuit = dynamic_builder.build();

    // Both should produce equivalent circuits
    assert_eq!(typed_circuit.num_qubits(), dynamic_circuit.num_qubits());
    assert_eq!(typed_circuit.len(), dynamic_circuit.len());
}

#[test]
fn test_large_circuit() {
    let mut builder = CircuitBuilder::<20>::with_capacity(1000);
    let qubits = builder.qubits();
    let h_gate = Arc::new(HadamardGate);

    // Build a large circuit
    for _ in 0..50 {
        for q in &qubits {
            builder.apply_gate(h_gate.clone(), &[*q]).unwrap();
        }
    }

    assert_eq!(builder.num_operations(), 1000);
    let circuit = builder.build();
    assert_eq!(circuit.len(), 1000);
}

#[test]
fn test_dynamic_builder_from_variable() {
    // Simulate reading circuit size from environment or config
    let circuit_sizes = vec![1, 2, 5, 10, 50];

    for size in circuit_sizes {
        let builder = DynamicCircuitBuilder::new(size);
        assert_eq!(builder.num_qubits(), size);

        let circuit = builder.build();
        assert_eq!(circuit.num_qubits(), size);
        assert!(circuit.is_empty());
    }
}
