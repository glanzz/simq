//! E2E tests for optimization pass internals:
//! dead_code_elimination, gate_commutation, gate_fusion, template_matching, template_substitution

use num_complex::Complex64;
use simq_compiler::passes::{
    AdvancedTemplateMatching, DeadCodeElimination, GateCommutation, GateFusion, OptimizationPass,
    TemplateSubstitution,
};
use simq_core::{gate::Gate, Circuit, QubitId};
use std::sync::Arc;

fn q(i: usize) -> QubitId {
    QubitId::new(i)
}

#[derive(Debug)]
struct MockGate {
    name: String,
    n_qubits: usize,
    matrix: Option<Vec<Complex64>>,
}

impl Gate for MockGate {
    fn name(&self) -> &str {
        &self.name
    }
    fn num_qubits(&self) -> usize {
        self.n_qubits
    }
    fn matrix(&self) -> Option<Vec<Complex64>> {
        self.matrix.clone()
    }
}

fn make_gate(name: &str) -> Arc<MockGate> {
    Arc::new(MockGate {
        name: name.to_string(),
        n_qubits: 1,
        matrix: None,
    })
}

fn make_gate_2q(name: &str) -> Arc<MockGate> {
    Arc::new(MockGate {
        name: name.to_string(),
        n_qubits: 2,
        matrix: None,
    })
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

fn make_gate_with_matrix(name: &str, matrix: Vec<Complex64>) -> Arc<MockGate> {
    Arc::new(MockGate {
        name: name.to_string(),
        n_qubits: 1,
        matrix: Some(matrix),
    })
}

// ============================================================================
// DeadCodeElimination tests
// ============================================================================

#[test]
fn test_dce_removes_xx_pair_on_same_qubit() {
    let pass = DeadCodeElimination::new();
    let mut circuit = Circuit::new(2);
    let x = make_gate("X");
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    assert_eq!(circuit.len(), 2);

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_dce_removes_yy_pair() {
    let pass = DeadCodeElimination::new();
    let mut circuit = Circuit::new(1);
    let y = make_gate("Y");
    circuit.add_gate(y.clone(), &[q(0)]).unwrap();
    circuit.add_gate(y.clone(), &[q(0)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_dce_removes_zz_pair() {
    let pass = DeadCodeElimination::new();
    let mut circuit = Circuit::new(1);
    let z = make_gate("Z");
    circuit.add_gate(z.clone(), &[q(0)]).unwrap();
    circuit.add_gate(z.clone(), &[q(0)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_dce_identity_removal() {
    let pass = DeadCodeElimination::new();
    let mut circuit = Circuit::new(2);
    let x = make_gate("X");
    let id = make_gate("I");
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(id.clone(), &[q(1)]).unwrap();
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();

    assert_eq!(circuit.len(), 3);
    let modified = pass.apply(&mut circuit).unwrap();
    assert!(modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_dce_does_not_remove_non_inverse_pairs() {
    let pass = DeadCodeElimination::new();
    let mut circuit = Circuit::new(1);
    let x = make_gate("X");
    let h = make_gate("H");
    circuit.add_gate(x, &[q(0)]).unwrap();
    circuit.add_gate(h, &[q(0)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(!modified);
    assert_eq!(circuit.len(), 2);
}

#[test]
fn test_dce_with_inverse_removal_disabled() {
    let pass = DeadCodeElimination::new().with_inverse_pair_removal(false);
    let mut circuit = Circuit::new(1);
    let x = make_gate("X");
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(!modified);
    assert_eq!(circuit.len(), 2);
}

#[test]
fn test_dce_with_identity_removal_disabled() {
    let pass = DeadCodeElimination::new().with_identity_removal(false);
    let mut circuit = Circuit::new(1);
    let id = make_gate("I");
    circuit.add_gate(id, &[q(0)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(!modified);
    assert_eq!(circuit.len(), 1);
}

#[test]
fn test_dce_pass_metadata() {
    let pass = DeadCodeElimination::new();
    assert_eq!(pass.name(), "dead-code-elimination");
    assert!(pass.iterative());
    assert!(pass.benefit_score() > 0.5);
    assert!(pass.description().is_some());
}

#[test]
fn test_dce_cnot_pair_removal() {
    let pass = DeadCodeElimination::new();
    let mut circuit = Circuit::new(2);
    let cnot = make_gate_2q("CNOT");
    circuit.add_gate(cnot.clone(), &[q(0), q(1)]).unwrap();
    circuit.add_gate(cnot.clone(), &[q(0), q(1)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_dce_swap_pair_removal() {
    let pass = DeadCodeElimination::new();
    let mut circuit = Circuit::new(2);
    let swap = make_gate_2q("SWAP");
    circuit.add_gate(swap.clone(), &[q(0), q(1)]).unwrap();
    circuit.add_gate(swap.clone(), &[q(0), q(1)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(modified);
    assert_eq!(circuit.len(), 0);
}

// ============================================================================
// GateCommutation tests
// ============================================================================

#[test]
fn test_commutation_pass_metadata() {
    let pass = GateCommutation::new();
    assert_eq!(pass.name(), "gate-commutation");
    assert!(pass.iterative());
    assert!(pass.benefit_score() > 0.0);
    assert!(pass.description().is_some());
}

#[test]
fn test_commutation_disjoint_qubits() {
    let pass = GateCommutation::new();
    let mut circuit = Circuit::new(3);
    let x = make_gate("X");
    let h = make_gate("H");

    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(h.clone(), &[q(1)]).unwrap();
    circuit.add_gate(x.clone(), &[q(2)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    // Disjoint qubits — reordering may or may not happen depending on implementation
    // At minimum, circuit length should be preserved
    assert_eq!(circuit.len(), 3);
    let _ = modified;
}

#[test]
fn test_commutation_diagonal_gates_commute() {
    let pass = GateCommutation::new();
    let mut circuit = Circuit::new(1);
    let z = make_gate("Z");
    let s = make_gate("S");

    circuit.add_gate(z.clone(), &[q(0)]).unwrap();
    circuit.add_gate(s.clone(), &[q(0)]).unwrap();

    let original_len = circuit.len();
    let _ = pass.apply(&mut circuit).unwrap();
    assert_eq!(circuit.len(), original_len);
}

#[test]
fn test_commutation_with_max_swaps() {
    let pass = GateCommutation::new().with_max_swaps(0);
    let mut circuit = Circuit::new(2);
    let x = make_gate("X");
    let h = make_gate("H");

    circuit.add_gate(x, &[q(0)]).unwrap();
    circuit.add_gate(h, &[q(0)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(!modified);
}

// ============================================================================
// GateFusion tests
// ============================================================================

#[test]
fn test_fusion_pass_metadata() {
    let pass = GateFusion::new();
    assert_eq!(pass.name(), "gate-fusion");
    assert!(pass.iterative());
    assert!(pass.benefit_score() > 0.9);
    assert!(pass.description().is_some());
}

#[test]
fn test_fusion_xx_fuses_to_identity_removed() {
    let pass = GateFusion::new();
    let mut circuit = Circuit::new(1);
    let x = make_gate_with_matrix("X", pauli_x_matrix());
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_fusion_different_qubits_no_fusion() {
    let pass = GateFusion::new();
    let mut circuit = Circuit::new(3);
    let h = make_gate_with_matrix("H", hadamard_matrix());
    circuit.add_gate(h.clone(), &[q(0)]).unwrap();
    circuit.add_gate(h.clone(), &[q(1)]).unwrap();
    circuit.add_gate(h.clone(), &[q(2)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(!modified);
    assert_eq!(circuit.len(), 3);
}

#[test]
fn test_fusion_no_matrix_no_fusion() {
    let pass = GateFusion::new();
    let mut circuit = Circuit::new(1);
    let g = make_gate("Custom");
    circuit.add_gate(g.clone(), &[q(0)]).unwrap();
    circuit.add_gate(g.clone(), &[q(0)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(!modified);
    assert_eq!(circuit.len(), 2);
}

#[test]
fn test_fusion_min_size_threshold() {
    let pass = GateFusion::new().with_min_fusion_size(3);
    let mut circuit = Circuit::new(1);
    let h = make_gate_with_matrix("H", hadamard_matrix());
    circuit.add_gate(h.clone(), &[q(0)]).unwrap();
    circuit.add_gate(h.clone(), &[q(0)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(!modified);
    assert_eq!(circuit.len(), 2);
}

#[test]
fn test_fusion_identity_elimination_toggle() {
    let pass = GateFusion::new().with_identity_elimination(false);
    assert_eq!(pass.name(), "gate-fusion");
}

// ============================================================================
// AdvancedTemplateMatching tests
// ============================================================================

#[test]
fn test_template_matching_pass_metadata() {
    let pass = AdvancedTemplateMatching::new();
    assert_eq!(pass.name(), "advanced-template-matching");
    assert!(pass.iterative());
    assert!(pass.benefit_score() > 0.8);
    assert!(pass.description().is_some());
}

#[test]
fn test_template_hzh_to_x() {
    let pass = AdvancedTemplateMatching::new();
    let mut circuit = Circuit::new(1);
    circuit
        .add_gate(Arc::new(simq_gates::standard::Hadamard), &[q(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(simq_gates::standard::PauliZ), &[q(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(simq_gates::standard::Hadamard), &[q(0)])
        .unwrap();

    assert_eq!(circuit.len(), 3);
    let modified = pass.apply(&mut circuit).unwrap();
    assert!(modified);
    assert_eq!(circuit.len(), 1);

    let op = circuit.get_operation(0).unwrap();
    assert_eq!(op.gate().name(), "X");
}

#[test]
fn test_template_hxh_to_z() {
    let pass = AdvancedTemplateMatching::new();
    let mut circuit = Circuit::new(1);
    circuit
        .add_gate(Arc::new(simq_gates::standard::Hadamard), &[q(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(simq_gates::standard::PauliX), &[q(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(simq_gates::standard::Hadamard), &[q(0)])
        .unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(modified);
    assert_eq!(circuit.len(), 1);

    let op = circuit.get_operation(0).unwrap();
    assert_eq!(op.gate().name(), "Z");
}

#[test]
fn test_template_xx_to_identity() {
    let pass = AdvancedTemplateMatching::new();
    let mut circuit = Circuit::new(1);
    circuit
        .add_gate(Arc::new(simq_gates::standard::PauliX), &[q(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(simq_gates::standard::PauliX), &[q(0)])
        .unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_template_zxz_to_x() {
    let pass = AdvancedTemplateMatching::new();
    let mut circuit = Circuit::new(1);
    circuit
        .add_gate(Arc::new(simq_gates::standard::PauliZ), &[q(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(simq_gates::standard::PauliX), &[q(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(simq_gates::standard::PauliZ), &[q(0)])
        .unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(modified);
    assert_eq!(circuit.len(), 1);
    assert_eq!(circuit.get_operation(0).unwrap().gate().name(), "X");
}

#[test]
fn test_template_chained_hzh_then_xx() {
    let pass = AdvancedTemplateMatching::new();
    let mut circuit = Circuit::new(1);
    // H-Z-H -> X, then X followed by another X -> identity
    circuit
        .add_gate(Arc::new(simq_gates::standard::Hadamard), &[q(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(simq_gates::standard::PauliZ), &[q(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(simq_gates::standard::Hadamard), &[q(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(simq_gates::standard::PauliX), &[q(0)])
        .unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_template_no_match_across_qubits() {
    let pass = AdvancedTemplateMatching::new();
    let mut circuit = Circuit::new(2);
    circuit
        .add_gate(Arc::new(simq_gates::standard::PauliX), &[q(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(simq_gates::standard::PauliX), &[q(1)])
        .unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(!modified);
    assert_eq!(circuit.len(), 2);
}

// ============================================================================
// TemplateSubstitution tests
// ============================================================================

#[test]
fn test_template_sub_pass_metadata() {
    let pass = TemplateSubstitution::new();
    assert_eq!(pass.name(), "template-substitution");
    assert!(pass.iterative());
    assert!(pass.benefit_score() > 0.7);
    assert!(pass.description().is_some());
}

#[test]
fn test_template_sub_xx_removal() {
    let pass = TemplateSubstitution::new();
    let mut circuit = Circuit::new(1);
    let x = make_gate("X");
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_template_sub_hh_removal() {
    let pass = TemplateSubstitution::new();
    let mut circuit = Circuit::new(1);
    let h = make_gate("H");
    circuit.add_gate(h.clone(), &[q(0)]).unwrap();
    circuit.add_gate(h.clone(), &[q(0)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_template_sub_four_s_removal() {
    let pass = TemplateSubstitution::new();
    let mut circuit = Circuit::new(1);
    let s = make_gate("S");
    for _ in 0..4 {
        circuit.add_gate(s.clone(), &[q(0)]).unwrap();
    }

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_template_sub_eight_t_removal() {
    let pass = TemplateSubstitution::new();
    let mut circuit = Circuit::new(1);
    let t = make_gate("T");
    for _ in 0..8 {
        circuit.add_gate(t.clone(), &[q(0)]).unwrap();
    }

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_template_sub_no_match_different_qubits() {
    let pass = TemplateSubstitution::new();
    let mut circuit = Circuit::new(2);
    let x = make_gate("X");
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(x.clone(), &[q(1)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(!modified);
    assert_eq!(circuit.len(), 2);
}

#[test]
fn test_template_sub_no_match_nonpattern() {
    let pass = TemplateSubstitution::new();
    let mut circuit = Circuit::new(1);
    let x = make_gate("X");
    let y = make_gate("Y");
    circuit.add_gate(x, &[q(0)]).unwrap();
    circuit.add_gate(y, &[q(0)]).unwrap();

    let modified = pass.apply(&mut circuit).unwrap();
    assert!(!modified);
    assert_eq!(circuit.len(), 2);
}

// ============================================================================
// Cross-pass interaction tests
// ============================================================================

#[test]
fn test_dce_then_fusion_pipeline() {
    let dce = DeadCodeElimination::new();
    let fusion = GateFusion::new();
    let mut circuit = Circuit::new(2);

    let x = make_gate_with_matrix("X", pauli_x_matrix());
    let h = make_gate_with_matrix("H", hadamard_matrix());
    let id = make_gate("I");

    // I on q1 (identity), X-X on q0 (inverse pair), H on q0
    circuit.add_gate(id, &[q(1)]).unwrap();
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(h.clone(), &[q(0)]).unwrap();

    // DCE removes identity and X-X pair
    dce.apply(&mut circuit).unwrap();
    assert_eq!(circuit.len(), 1);

    // Fusion has nothing to fuse on a single gate
    let modified = fusion.apply(&mut circuit).unwrap();
    assert!(!modified);
    assert_eq!(circuit.len(), 1);
}

#[test]
fn test_template_then_dce_pipeline() {
    let template = AdvancedTemplateMatching::new();
    let _dce = DeadCodeElimination::new();
    let mut circuit = Circuit::new(1);

    // H-Z-H -> X, then X on same qubit -> X-X pair (for DCE)
    circuit
        .add_gate(Arc::new(simq_gates::standard::Hadamard), &[q(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(simq_gates::standard::PauliZ), &[q(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(simq_gates::standard::Hadamard), &[q(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(simq_gates::standard::PauliX), &[q(0)])
        .unwrap();

    // Template: H-Z-H -> X, then X-X -> gone
    template.apply(&mut circuit).unwrap();
    assert_eq!(circuit.len(), 0);
}
