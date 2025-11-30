//! Integration tests for the optimization pipeline

use num_complex::Complex64;
use simq_compiler::{
    passes::{DeadCodeElimination, GateCommutation, GateFusion, TemplateSubstitution},
    Compiler, CompilerBuilder,
};
use simq_core::{gate::Gate, Circuit, QubitId};
use std::sync::Arc;

// Mock gate for testing
#[derive(Debug)]
struct MockGate {
    name: String,
    matrix: Option<Vec<Complex64>>,
}

impl Gate for MockGate {
    fn name(&self) -> &str {
        &self.name
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        self.matrix.clone()
    }
}

fn pauli_x_matrix() -> Vec<Complex64> {
    vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]
}

fn hadamard_matrix() -> Vec<Complex64> {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    vec![
        Complex64::new(inv_sqrt2, 0.0),
        Complex64::new(inv_sqrt2, 0.0),
        Complex64::new(inv_sqrt2, 0.0),
        Complex64::new(-inv_sqrt2, 0.0),
    ]
}

#[test]
fn test_full_optimization_pipeline() {
    // Create compiler with all optimization passes
    let compiler = CompilerBuilder::new()
        .add_pass(Arc::new(DeadCodeElimination::new()))
        .add_pass(Arc::new(GateCommutation::new()))
        .add_pass(Arc::new(GateFusion::new()))
        .add_pass(Arc::new(TemplateSubstitution::new()))
        .max_iterations(5)
        .build();

    // Create a circuit with optimization opportunities
    let mut circuit = Circuit::new(3);

    let x_gate = Arc::new(MockGate {
        name: "X".to_string(),
        matrix: Some(pauli_x_matrix()),
    });
    let h_gate = Arc::new(MockGate {
        name: "H".to_string(),
        matrix: Some(hadamard_matrix()),
    });

    // Add gates with various optimization opportunities:
    // - X-X pair (dead code elimination)
    // - H-H pair (template substitution)
    // - Gates on different qubits (commutation opportunities)
    circuit
        .add_gate(x_gate.clone(), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(x_gate.clone(), &[QubitId::new(0)])
        .unwrap(); // X-X cancels
    circuit
        .add_gate(h_gate.clone(), &[QubitId::new(1)])
        .unwrap();
    circuit
        .add_gate(h_gate.clone(), &[QubitId::new(1)])
        .unwrap(); // H-H cancels
    circuit.add_gate(x_gate, &[QubitId::new(2)]).unwrap();

    let original_len = circuit.len();
    assert_eq!(original_len, 5);

    // Apply optimization
    let result = compiler.compile(&mut circuit).unwrap();

    // Circuit should be optimized
    assert!(result.modified);
    assert!(circuit.len() < original_len);

    // With dead code elimination and template substitution,
    // X-X and H-H should be removed, leaving only 1 gate
    assert_eq!(circuit.len(), 1);
}

#[test]
fn test_compiler_with_no_passes() {
    let compiler = Compiler::default();

    let mut circuit = Circuit::new(2);
    let x_gate = Arc::new(MockGate {
        name: "X".to_string(),
        matrix: Some(pauli_x_matrix()),
    });

    circuit
        .add_gate(x_gate.clone(), &[QubitId::new(0)])
        .unwrap();
    circuit.add_gate(x_gate, &[QubitId::new(0)]).unwrap();

    let original_len = circuit.len();

    // No passes, so no modification
    let result = compiler.compile(&mut circuit).unwrap();
    assert!(!result.modified);
    assert_eq!(circuit.len(), original_len);
}

#[test]
fn test_fixed_point_iteration() {
    // Create compiler with dead code elimination only
    let compiler = CompilerBuilder::new()
        .add_pass(Arc::new(DeadCodeElimination::new()))
        .max_iterations(10)
        .build();

    let mut circuit = Circuit::new(2);
    let x_gate = Arc::new(MockGate {
        name: "X".to_string(),
        matrix: Some(pauli_x_matrix()),
    });

    // Add multiple X-X pairs
    for _ in 0..5 {
        circuit
            .add_gate(x_gate.clone(), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(x_gate.clone(), &[QubitId::new(0)])
            .unwrap();
    }

    assert_eq!(circuit.len(), 10);

    // All pairs should be eliminated through fixed-point iteration
    let result = compiler.compile(&mut circuit).unwrap();
    assert!(result.modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_compiler_statistics() {
    let compiler = CompilerBuilder::new()
        .add_pass(Arc::new(DeadCodeElimination::new()))
        .add_pass(Arc::new(TemplateSubstitution::new()))
        .enable_timing(true)
        .build();

    let mut circuit = Circuit::new(2);
    let x_gate = Arc::new(MockGate {
        name: "X".to_string(),
        matrix: Some(pauli_x_matrix()),
    });

    circuit
        .add_gate(x_gate.clone(), &[QubitId::new(0)])
        .unwrap();
    circuit.add_gate(x_gate, &[QubitId::new(0)]).unwrap();

    let result = compiler.compile(&mut circuit).unwrap();

    // Check statistics are collected
    assert!(result.modified);
    assert!(!result.pass_stats.is_empty());
    assert!(result.total_time_us > 0);
}
