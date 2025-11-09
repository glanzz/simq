//! Performance tests for validation

use simq_core::gate::Gate;
use simq_core::{Circuit, QubitId};
use std::sync::Arc;

#[derive(Debug)]
struct MockGate {
    name: String,
    num_qubits: usize,
}

impl Gate for MockGate {
    fn name(&self) -> &str {
        &self.name
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

#[test]
fn test_validation_performance_small() {
    // Small circuit: 10 gates
    let mut circuit = Circuit::new(5);
    let h_gate = Arc::new(MockGate {
        name: "H".to_string(),
        num_qubits: 1,
    });
    let cnot_gate = Arc::new(MockGate {
        name: "CNOT".to_string(),
        num_qubits: 2,
    });

    for i in 0..10 {
        if i % 2 == 0 {
            circuit.add_gate(h_gate.clone(), &[QubitId::new(i % 5)]).unwrap();
        } else {
            circuit.add_gate(
                cnot_gate.clone(),
                &[QubitId::new(i % 5), QubitId::new((i + 1) % 5)],
            )
            .unwrap();
        }
    }

    // Should complete quickly
    let start = std::time::Instant::now();
    let _report = circuit.validate_dag().unwrap();
    let elapsed = start.elapsed();
    
    // Should complete in < 1ms for small circuits
    assert!(elapsed.as_millis() < 10, "Validation took too long: {:?}", elapsed);
}

#[test]
fn test_dag_construction_performance() {
    use simq_core::validation::dag::DependencyGraph;

    // Medium circuit: 100 gates
    let mut circuit = Circuit::new(10);
    let h_gate = Arc::new(MockGate {
        name: "H".to_string(),
        num_qubits: 1,
    });

    for i in 0..100 {
        circuit.add_gate(h_gate.clone(), &[QubitId::new(i % 10)]).unwrap();
    }

    let start = std::time::Instant::now();
    let _dag = DependencyGraph::from_circuit(&circuit).unwrap();
    let elapsed = start.elapsed();

    // Should complete in < 10ms for 100-gate circuits
    assert!(elapsed.as_millis() < 50, "DAG construction took too long: {:?}", elapsed);
}

#[test]
fn test_cycle_detection_performance() {
    use simq_core::validation::dag::DependencyGraph;

    // Circuit with no cycles (typical case)
    let mut circuit = Circuit::new(10);
    let h_gate = Arc::new(MockGate {
        name: "H".to_string(),
        num_qubits: 1,
    });

    for i in 0..50 {
        circuit.add_gate(h_gate.clone(), &[QubitId::new(i % 10)]).unwrap();
    }

    let dag = DependencyGraph::from_circuit(&circuit).unwrap();
    let start = std::time::Instant::now();
    let cycles = dag.find_cycles();
    let elapsed = start.elapsed();

    assert!(cycles.is_empty());
    // Should complete in < 5ms
    assert!(elapsed.as_millis() < 20, "Cycle detection took too long: {:?}", elapsed);
}

#[test]
fn test_topological_sort_performance() {
    use simq_core::validation::dag::DependencyGraph;

    let mut circuit = Circuit::new(10);
    let h_gate = Arc::new(MockGate {
        name: "H".to_string(),
        num_qubits: 1,
    });

    for i in 0..100 {
        circuit.add_gate(h_gate.clone(), &[QubitId::new(i % 10)]).unwrap();
    }

    let dag = DependencyGraph::from_circuit(&circuit).unwrap();
    let start = std::time::Instant::now();
    let _order = dag.topological_sort().unwrap();
    let elapsed = start.elapsed();

    // Should complete in < 5ms
    assert!(elapsed.as_millis() < 20, "Topological sort took too long: {:?}", elapsed);
}

#[test]
fn test_parallelism_analysis_performance() {
    // Circuit with some parallelism
    let mut circuit = Circuit::new(10);
    let h_gate = Arc::new(MockGate {
        name: "H".to_string(),
        num_qubits: 1,
    });

    // Add parallel gates
    for i in 0..10 {
        circuit.add_gate(h_gate.clone(), &[QubitId::new(i)]).unwrap();
    }

    // Then sequential gates
    let cnot_gate = Arc::new(MockGate {
        name: "CNOT".to_string(),
        num_qubits: 2,
    });
    for i in 0..9 {
        circuit.add_gate(
            cnot_gate.clone(),
            &[QubitId::new(i), QubitId::new(i + 1)],
        )
        .unwrap();
    }

    let start = std::time::Instant::now();
    let _analysis = circuit.analyze_parallelism().unwrap();
    let elapsed = start.elapsed();

    // Should complete in < 10ms
    assert!(elapsed.as_millis() < 50, "Parallelism analysis took too long: {:?}", elapsed);
}

