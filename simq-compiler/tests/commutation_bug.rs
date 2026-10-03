use simq_compiler::passes::{GateCommutation, OptimizationPass};
use simq_core::{Circuit, QubitId};
use std::sync::Arc;

#[derive(Debug)]
struct MockGate {
    name: String,
}

impl simq_core::gate::Gate for MockGate {
    fn name(&self) -> &str {
        &self.name
    }
    fn num_qubits(&self) -> usize {
        1
    }
}

#[test]
fn test_gate_commutation_index_bug() {
    let pass = GateCommutation::new();
    let mut circuit = Circuit::new(2);

    let g0 = Arc::new(MockGate {
        name: "G0".to_string(),
    });
    let g1 = Arc::new(MockGate {
        name: "G1".to_string(),
    });
    let g2 = Arc::new(MockGate {
        name: "G2".to_string(),
    });
    let g3 = Arc::new(MockGate {
        name: "G3".to_string(),
    });

    // Circuit: G0(q0), G1(q1), G2(q0), G3(q0)
    circuit.add_gate(g0.clone(), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(g1.clone(), &[QubitId::new(1)]).unwrap();
    circuit.add_gate(g2.clone(), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(g3.clone(), &[QubitId::new(0)]).unwrap();

    // Apply pass
    pass.apply(&mut circuit).unwrap();

    // SimQ Circuit::operations() returns an iterator, so we collect it
    let names: Vec<_> = circuit.operations().map(|op| op.gate().name()).collect();

    println!("Resulting order: {:?}", names);
    assert_eq!(names, vec!["G0", "G2", "G3", "G1"], "Gates should be grouped by qubit");
}
