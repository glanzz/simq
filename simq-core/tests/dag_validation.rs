//! Integration tests for DAG validation

use simq_core::gate::Gate;
use simq_core::{Circuit, QubitId};
use std::sync::Arc;

// Mock gate implementations for testing
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

#[test]
fn test_basic_validation() {
    let mut circuit = Circuit::new(2);
    let h_gate = Arc::new(HadamardGate);
    circuit.add_gate(h_gate, &[QubitId::new(0)]).unwrap();

    assert!(circuit.validate().is_ok());
}

#[test]
fn test_dag_validation_simple() {
    let mut circuit = Circuit::new(2);
    let h_gate = Arc::new(HadamardGate);
    let cnot_gate = Arc::new(CnotGate);

    circuit.add_gate(h_gate, &[QubitId::new(0)]).unwrap();
    circuit
        .add_gate(cnot_gate, &[QubitId::new(0), QubitId::new(1)])
        .unwrap();

    let report = circuit.validate_dag().unwrap();
    assert!(report.is_valid());
}

#[test]
fn test_dag_validation_parallel_gates() {
    // Create circuit with parallel gates (H on different qubits)
    let mut circuit = Circuit::new(3);
    let h_gate = Arc::new(HadamardGate);

    circuit
        .add_gate(h_gate.clone(), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(h_gate.clone(), &[QubitId::new(1)])
        .unwrap();
    circuit.add_gate(h_gate, &[QubitId::new(2)]).unwrap();

    let report = circuit.validate_dag().unwrap();
    assert!(report.is_valid());
}

#[test]
fn test_compute_depth() {
    let mut circuit = Circuit::new(2);
    let h_gate = Arc::new(HadamardGate);
    let cnot_gate = Arc::new(CnotGate);

    circuit.add_gate(h_gate, &[QubitId::new(0)]).unwrap();
    circuit
        .add_gate(cnot_gate, &[QubitId::new(0), QubitId::new(1)])
        .unwrap();

    let depth = circuit.compute_depth().unwrap();
    assert!(depth >= 2);
    assert!(depth <= 2); // Sequential: H then CNOT
}

#[test]
fn test_compute_depth_parallel() {
    // Three parallel H gates should have depth 1
    let mut circuit = Circuit::new(3);
    let h_gate = Arc::new(HadamardGate);

    circuit
        .add_gate(h_gate.clone(), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(h_gate.clone(), &[QubitId::new(1)])
        .unwrap();
    circuit.add_gate(h_gate, &[QubitId::new(2)]).unwrap();

    let depth = circuit.compute_depth().unwrap();
    // All three gates can run in parallel, so depth should be 1
    assert_eq!(depth, 1);
}

#[test]
fn test_is_acyclic() {
    let mut circuit = Circuit::new(2);
    let h_gate = Arc::new(HadamardGate);
    circuit.add_gate(h_gate, &[QubitId::new(0)]).unwrap();

    assert!(circuit.is_acyclic().unwrap());
}

#[test]
fn test_analyze_parallelism() {
    // Create circuit with some parallel gates
    let mut circuit = Circuit::new(3);
    let h_gate = Arc::new(HadamardGate);
    let cnot_gate = Arc::new(CnotGate);

    // Parallel H gates
    circuit
        .add_gate(h_gate.clone(), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(h_gate.clone(), &[QubitId::new(1)])
        .unwrap();
    circuit.add_gate(h_gate, &[QubitId::new(2)]).unwrap();

    // Then CNOT
    circuit
        .add_gate(cnot_gate, &[QubitId::new(0), QubitId::new(1)])
        .unwrap();

    let analysis = circuit.analyze_parallelism().unwrap();
    assert!(analysis.parallelism_factor > 1.0);
    assert!(analysis.max_parallelism >= 3); // Three H gates can run in parallel
    assert_eq!(analysis.num_layers(), 2); // Layer 1: H gates, Layer 2: CNOT
}

#[test]
fn test_depth_method() {
    let mut circuit = Circuit::new(2);
    let h_gate = Arc::new(HadamardGate);
    let cnot_gate = Arc::new(CnotGate);

    circuit.add_gate(h_gate, &[QubitId::new(0)]).unwrap();
    circuit
        .add_gate(cnot_gate, &[QubitId::new(0), QubitId::new(1)])
        .unwrap();

    // depth() should use DAG analysis if possible
    let depth = circuit.depth();
    assert!(depth >= 1);
    assert!(depth <= 2);
}

#[test]
fn test_validation_invalid_qubit() {
    let mut circuit = Circuit::new(2);
    let h_gate = Arc::new(HadamardGate);

    // This should fail validation
    let result = circuit.add_gate(h_gate, &[QubitId::new(5)]);
    assert!(result.is_err());
}

#[test]
fn test_empty_circuit_validation() {
    let circuit = Circuit::new(2);
    assert!(circuit.validate().is_ok());
    assert!(circuit.validate_dag().is_ok());
    assert!(circuit.is_acyclic().unwrap());
}

#[test]
fn test_sequential_circuit_depth() {
    // Create a sequential circuit: H(0) -> CNOT(0,1) -> X(1)
    let mut circuit = Circuit::new(2);
    let h_gate = Arc::new(HadamardGate);
    let cnot_gate = Arc::new(CnotGate);
    let x_gate = Arc::new(XGate);

    circuit.add_gate(h_gate, &[QubitId::new(0)]).unwrap();
    circuit
        .add_gate(cnot_gate, &[QubitId::new(0), QubitId::new(1)])
        .unwrap();
    circuit.add_gate(x_gate, &[QubitId::new(1)]).unwrap();

    let depth = circuit.compute_depth().unwrap();
    // Sequential circuit should have depth equal to number of operations
    assert_eq!(depth, 3);
}

#[test]
fn test_bell_state_circuit() {
    // Bell state: H(0) -> CNOT(0,1)
    let mut circuit = Circuit::new(2);
    let h_gate = Arc::new(HadamardGate);
    let cnot_gate = Arc::new(CnotGate);

    circuit.add_gate(h_gate, &[QubitId::new(0)]).unwrap();
    circuit
        .add_gate(cnot_gate, &[QubitId::new(0), QubitId::new(1)])
        .unwrap();

    let report = circuit.validate_dag().unwrap();
    assert!(report.is_valid());

    let depth = circuit.compute_depth().unwrap();
    assert_eq!(depth, 2);

    let analysis = circuit.analyze_parallelism().unwrap();
    assert_eq!(analysis.num_layers(), 2);
}

#[test]
fn test_ghz_state_circuit() {
    // GHZ state: H(0) -> CNOT(0,1) -> CNOT(1,2)
    let mut circuit = Circuit::new(3);
    let h_gate = Arc::new(HadamardGate);
    let cnot_gate = Arc::new(CnotGate);

    circuit.add_gate(h_gate, &[QubitId::new(0)]).unwrap();
    circuit
        .add_gate(cnot_gate.clone(), &[QubitId::new(0), QubitId::new(1)])
        .unwrap();
    circuit
        .add_gate(cnot_gate, &[QubitId::new(1), QubitId::new(2)])
        .unwrap();

    let report = circuit.validate_dag().unwrap();
    assert!(report.is_valid());

    let depth = circuit.compute_depth().unwrap();
    assert_eq!(depth, 3);

    let analysis = circuit.analyze_parallelism().unwrap();
    assert_eq!(analysis.num_layers(), 3);
}
