//! E2E tests for decomposition internals: basis, clifford_t, single_qubit, two_qubit, multi_qubit

use num_complex::Complex64;
use simq_compiler::decomposition::{
    basis::{BasisGate, BasisGateSet, CustomBasis},
    clifford_t::{CliffordTDecomposer, CliffordTGate, GridSynthConfig},
    multi_qubit::{MultiQubitDecomposer, MultiQubitInstruction},
    single_qubit::{EulerAngles, EulerBasis, SingleQubitDecomposer},
    two_qubit::{EntanglementGate, TwoQubitDecomposer},
    Decomposer, DecompositionConfig, UniversalDecomposer,
};
use simq_compiler::matrix_computation::{hadamard_matrix, identity_2x2, pauli_x_matrix, Matrix2};
use simq_core::gate::Gate;
use std::f64::consts::PI;
use std::sync::Arc;

const EPSILON: f64 = 1e-8;

// Helper: MockGate with matrix
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

fn matrix2_to_flat(m: &Matrix2) -> Vec<Complex64> {
    vec![m[0][0], m[0][1], m[1][0], m[1][1]]
}

// ============================================================================
// Basis gate set tests
// ============================================================================

#[test]
fn test_basis_gate_set_ibm_contains_expected_gates() {
    let ibm = BasisGateSet::IBM;
    assert!(ibm.contains(BasisGate::CNOT));
    assert!(ibm.contains(BasisGate::U1));
    assert!(ibm.contains(BasisGate::U2));
    assert!(ibm.contains(BasisGate::U3));
}

#[test]
fn test_basis_gate_set_ibm_qiskit_contains_rz_sx() {
    let ibm_q = BasisGateSet::IBMQiskit;
    assert!(ibm_q.contains(BasisGate::RZ));
    assert!(ibm_q.contains(BasisGate::SX));
    assert!(ibm_q.contains(BasisGate::X));
    assert!(ibm_q.contains(BasisGate::CNOT));
}

#[test]
fn test_basis_gate_set_google_contains_expected_gates() {
    let google = BasisGateSet::Google;
    assert!(google.contains(BasisGate::PhasedXZ));
    assert!(google.contains(BasisGate::SqrtISWAP));
    assert!(google.contains(BasisGate::SYC));
    assert!(!google.contains(BasisGate::CNOT));
}

#[test]
fn test_basis_gate_set_google_cirq_contains_cz() {
    let cirq = BasisGateSet::GoogleCirq;
    assert!(cirq.contains(BasisGate::CZ));
    assert!(cirq.contains(BasisGate::PhasedXZ));
    assert!(cirq.contains(BasisGate::FSIM));
}

#[test]
fn test_basis_gate_set_ionq_contains_ms() {
    let ionq = BasisGateSet::IonQ;
    assert!(ionq.contains(BasisGate::GPI));
    assert!(ionq.contains(BasisGate::GPI2));
    assert!(ionq.contains(BasisGate::MS));
}

#[test]
fn test_basis_gate_set_clifford_t_is_discrete() {
    let ct = BasisGateSet::CliffordT;
    assert!(ct.is_discrete());
    assert!(ct.contains(BasisGate::H));
    assert!(ct.contains(BasisGate::T));
    assert!(ct.contains(BasisGate::S));
    assert!(ct.contains(BasisGate::CNOT));
    assert!(!ct.has_rotation_gates());
}

#[test]
fn test_basis_gate_set_all_contains_everything() {
    let all = BasisGateSet::All;
    assert!(all.contains(BasisGate::H));
    assert!(all.contains(BasisGate::CNOT));
    assert!(all.contains(BasisGate::Toffoli));
    assert!(!all.is_discrete());
}

#[test]
fn test_basis_gate_single_vs_two_qubit_classification() {
    let ibm = BasisGateSet::IBM;
    let singles = ibm.single_qubit_gates();
    let twos = ibm.two_qubit_gates();

    for g in &singles {
        assert_eq!(g.num_qubits(), 1, "Single-qubit gate {:?} reports wrong qubit count", g);
    }
    for g in &twos {
        assert_eq!(g.num_qubits(), 2, "Two-qubit gate {:?} reports wrong qubit count", g);
    }
}

#[test]
fn test_basis_gate_entangling_gate() {
    assert_eq!(BasisGateSet::IBM.entangling_gate(), Some(BasisGate::CNOT));
    assert_eq!(BasisGateSet::Google.entangling_gate(), Some(BasisGate::SqrtISWAP));
    assert_eq!(BasisGateSet::GoogleCirq.entangling_gate(), Some(BasisGate::CZ));
    assert_eq!(BasisGateSet::IonQ.entangling_gate(), Some(BasisGate::MS));
}

#[test]
fn test_basis_gate_descriptions_nonempty() {
    for basis in &[
        BasisGateSet::IBM,
        BasisGateSet::Google,
        BasisGateSet::IonQ,
        BasisGateSet::CliffordT,
        BasisGateSet::Universal,
        BasisGateSet::Rotation,
        BasisGateSet::Pauli,
    ] {
        let desc = basis.description();
        assert!(!desc.is_empty(), "Description for {:?} should not be empty", basis);
    }
}

#[test]
fn test_custom_basis_gate_set() {
    let mut custom = CustomBasis::new();
    custom.add_single_qubit(BasisGate::H);
    custom.add_single_qubit(BasisGate::T);
    custom.add_two_qubit(BasisGate::CZ);

    let basis = BasisGateSet::Custom(custom);
    let gates = basis.gates();
    assert!(gates.len() >= 3);
}

#[test]
fn test_basis_gate_properties() {
    assert!(BasisGate::H.is_clifford());
    assert!(BasisGate::S.is_clifford());
    assert!(BasisGate::CNOT.is_clifford());
    assert!(!BasisGate::T.is_clifford());
    assert!(BasisGate::RX.is_parameterized());
    assert!(BasisGate::RY.is_parameterized());
    assert!(BasisGate::RZ.is_parameterized());
    assert!(!BasisGate::H.is_parameterized());
    assert_eq!(BasisGate::Toffoli.num_qubits(), 3);
    assert_eq!(BasisGate::Fredkin.num_qubits(), 3);
}

// ============================================================================
// SingleQubitDecomposer tests
// ============================================================================

#[test]
fn test_single_qubit_decompose_identity_zyz() {
    let decomposer = SingleQubitDecomposer::new(EulerBasis::ZYZ);
    let id = identity_2x2();
    let angles = decomposer.decompose_to_angles(&id).unwrap();
    assert!(angles.is_identity(), "Identity matrix should decompose to identity angles");
}

#[test]
fn test_single_qubit_decompose_hadamard_zyz() {
    let decomposer = SingleQubitDecomposer::new(EulerBasis::ZYZ);
    let h = hadamard_matrix();
    let angles = decomposer.decompose_to_angles(&h).unwrap();
    assert!(angles.gate_count() > 0, "Hadamard should require non-trivial decomposition");
}

#[test]
fn test_single_qubit_decompose_pauli_x_zxz() {
    let decomposer = SingleQubitDecomposer::new(EulerBasis::ZXZ);
    let x = pauli_x_matrix();
    let angles = decomposer.decompose_to_angles(&x).unwrap();
    assert!(angles.gate_count() > 0);
}

#[test]
fn test_single_qubit_all_euler_bases() {
    let h = hadamard_matrix();
    for basis in &[
        EulerBasis::ZYZ,
        EulerBasis::ZXZ,
        EulerBasis::XYX,
        EulerBasis::YZY,
        EulerBasis::U3,
    ] {
        let decomposer = SingleQubitDecomposer::new(*basis);
        let result = decomposer.decompose_to_angles(&h);
        assert!(result.is_ok(), "Decomposition failed for basis {:?}", basis);
    }
}

#[test]
fn test_euler_angles_gate_count() {
    let zero_angles = EulerAngles::new(0.0, 0.0, 0.0, 0.0);
    assert!(zero_angles.is_identity());
    assert_eq!(zero_angles.gate_count(), 0);

    let nontrivial = EulerAngles::new(0.0, PI / 4.0, PI / 2.0, PI / 4.0);
    assert!(!nontrivial.is_identity());
    assert!(nontrivial.gate_count() > 0);
}

#[test]
fn test_single_qubit_decomposer_as_decomposer_trait() {
    let decomposer = SingleQubitDecomposer::new(EulerBasis::ZYZ);
    let gate = MockGate {
        name: "H".to_string(),
        n_qubits: 1,
        matrix: Some(matrix2_to_flat(&hadamard_matrix())),
    };
    assert!(decomposer.can_decompose(&gate));

    let two_q_gate = MockGate {
        name: "CNOT".to_string(),
        n_qubits: 2,
        matrix: None,
    };
    assert!(!decomposer.can_decompose(&two_q_gate));
}

// ============================================================================
// TwoQubitDecomposer tests
// ============================================================================

#[test]
fn test_two_qubit_decompose_swap_cnot() {
    let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
    let instructions = decomposer.decompose_swap();
    assert!(!instructions.is_empty(), "SWAP decomposition should produce instructions");
}

#[test]
fn test_two_qubit_decompose_swap_cz() {
    let decomposer = TwoQubitDecomposer::new(EntanglementGate::CZ);
    let instructions = decomposer.decompose_swap();
    assert!(!instructions.is_empty());
}

#[test]
fn test_two_qubit_decompose_iswap() {
    let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
    let instructions = decomposer.decompose_iswap();
    assert!(!instructions.is_empty());
}

#[test]
fn test_two_qubit_decompose_sqrt_iswap() {
    let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
    let instructions = decomposer.decompose_sqrt_iswap();
    assert!(!instructions.is_empty());
}

#[test]
fn test_two_qubit_decomposer_trait_gate_check() {
    let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
    let single_gate = MockGate {
        name: "H".to_string(),
        n_qubits: 1,
        matrix: None,
    };
    assert!(!decomposer.can_decompose(&single_gate));

    // can_decompose requires num_qubits == 2 AND matrix().is_some()
    let two_gate_no_matrix = MockGate {
        name: "CZ".to_string(),
        n_qubits: 2,
        matrix: None,
    };
    assert!(!decomposer.can_decompose(&two_gate_no_matrix));

    let cz_matrix: Vec<Complex64> = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(-1.0, 0.0),
    ];
    let two_gate_with_matrix = MockGate {
        name: "CZ".to_string(),
        n_qubits: 2,
        matrix: Some(cz_matrix),
    };
    assert!(decomposer.can_decompose(&two_gate_with_matrix));
}

// ============================================================================
// MultiQubitDecomposer tests
// ============================================================================

#[test]
fn test_multi_qubit_toffoli_relative_phase() {
    let decomposer = MultiQubitDecomposer::new();
    let instructions = decomposer.decompose_toffoli_relative_phase();
    assert!(!instructions.is_empty());

    let has_cnot = instructions
        .iter()
        .any(|i| matches!(i, MultiQubitInstruction::CNOT { .. }));
    assert!(has_cnot, "Toffoli decomposition should contain CNOT gates");
}

#[test]
fn test_multi_qubit_toffoli_with_ancilla() {
    let decomposer = MultiQubitDecomposer::with_ancillas();
    let instructions = decomposer.decompose_toffoli_with_ancilla();
    assert!(!instructions.is_empty());
}

#[test]
fn test_multi_qubit_fredkin_decomposition() {
    let decomposer = MultiQubitDecomposer::new();
    let instructions = decomposer.decompose_fredkin();
    assert!(!instructions.is_empty());

    let has_cnot = instructions
        .iter()
        .any(|i| matches!(i, MultiQubitInstruction::CNOT { .. }));
    assert!(has_cnot, "Fredkin decomposition should contain CNOT gates");
}

#[test]
fn test_multi_qubit_ccz_decomposition() {
    let decomposer = MultiQubitDecomposer::new();
    let instructions = decomposer.decompose_ccz();
    assert!(!instructions.is_empty());
}

#[test]
fn test_multi_qubit_mcx_linear_scaling() {
    let decomposer = MultiQubitDecomposer::new();

    let mcx_3 = decomposer.decompose_mcx_linear(3);
    let mcx_4 = decomposer.decompose_mcx_linear(4);

    assert!(!mcx_3.is_empty());
    assert!(!mcx_4.is_empty());
    assert!(
        mcx_4.len() >= mcx_3.len(),
        "MCX with more controls should produce more or equal instructions"
    );
}

#[test]
fn test_multi_qubit_mcx_cost_estimate() {
    let decomposer = MultiQubitDecomposer::new();

    let cost_2 = decomposer.estimate_mcx_cost(2);
    let cost_3 = decomposer.estimate_mcx_cost(3);
    let cost_5 = decomposer.estimate_mcx_cost(5);

    assert!(cost_3 > cost_2, "MCX cost should increase with controls");
    assert!(cost_5 > cost_3, "MCX cost should increase with controls");
}

#[test]
fn test_multi_qubit_decomposer_trait_gate_check() {
    let decomposer = MultiQubitDecomposer::new();
    let single_gate = MockGate {
        name: "H".to_string(),
        n_qubits: 1,
        matrix: None,
    };
    assert!(!decomposer.can_decompose(&single_gate));

    let three_gate = MockGate {
        name: "Toffoli".to_string(),
        n_qubits: 3,
        matrix: None,
    };
    assert!(decomposer.can_decompose(&three_gate));
}

// ============================================================================
// CliffordTDecomposer tests
// ============================================================================

#[test]
fn test_clifford_t_decompose_rz_special_angles() {
    let decomposer = CliffordTDecomposer::new();

    let gates_pi = decomposer.decompose_rz(PI);
    assert!(!gates_pi.is_empty(), "Rz(pi) should decompose to Clifford+T gates");

    let gates_half_pi = decomposer.decompose_rz(PI / 2.0);
    assert!(!gates_half_pi.is_empty());

    let gates_quarter_pi = decomposer.decompose_rz(PI / 4.0);
    assert!(!gates_quarter_pi.is_empty());
}

#[test]
fn test_clifford_t_gate_classification() {
    assert!(CliffordTGate::H.is_clifford());
    assert!(CliffordTGate::S.is_clifford());
    assert!(CliffordTGate::X.is_clifford());
    assert!(CliffordTGate::Y.is_clifford());
    assert!(CliffordTGate::Z.is_clifford());
    assert!(!CliffordTGate::T.is_clifford());
    assert!(!CliffordTGate::TDagger.is_clifford());
    assert!(CliffordTGate::T.is_t_gate());
    assert!(CliffordTGate::TDagger.is_t_gate());
    assert!(!CliffordTGate::H.is_t_gate());
}

#[test]
fn test_clifford_t_count_t_gates() {
    let gates = vec![
        CliffordTGate::H,
        CliffordTGate::T,
        CliffordTGate::S,
        CliffordTGate::T,
        CliffordTGate::TDagger,
    ];
    assert_eq!(CliffordTDecomposer::count_t_gates(&gates), 3);
}

#[test]
fn test_clifford_t_decompose_single_qubit_hadamard() {
    let decomposer = CliffordTDecomposer::new();
    let h = hadamard_matrix();
    let result = decomposer.decompose_single_qubit(&h);
    assert!(result.is_ok());
    let gates = result.unwrap();
    assert!(!gates.is_empty());
}

#[test]
fn test_clifford_t_with_config() {
    let config = GridSynthConfig {
        epsilon: 1e-3,
        max_gates: 100,
        optimize_t_count: true,
        optimize_t_depth: false,
    };
    let decomposer = CliffordTDecomposer::with_config(config);
    let gates = decomposer.decompose_rz(PI / 7.0);
    assert!(!gates.is_empty());
}

#[test]
fn test_clifford_t_optimize_t_count() {
    let decomposer = CliffordTDecomposer::new();
    let gates = vec![
        CliffordTGate::T,
        CliffordTGate::T,
        CliffordTGate::T,
        CliffordTGate::T,
        CliffordTGate::H,
    ];
    let optimized = decomposer.optimize_t_count(&gates);
    assert!(
        CliffordTDecomposer::count_t_gates(&optimized)
            <= CliffordTDecomposer::count_t_gates(&gates),
        "Optimization should not increase T count"
    );
}

// ============================================================================
// UniversalDecomposer tests
// ============================================================================

#[test]
fn test_universal_decomposer_single_qubit_ibm() {
    let config = DecompositionConfig {
        basis: BasisGateSet::IBM,
        optimization_level: 1,
        max_depth: None,
        fidelity_threshold: 0.99,
        max_gates: None,
        allow_ancillas: false,
        num_ancillas: 0,
        clifford_t_epsilon: 1e-6,
    };
    let decomposer = UniversalDecomposer::new(config);
    let gate = MockGate {
        name: "H".to_string(),
        n_qubits: 1,
        matrix: Some(matrix2_to_flat(&hadamard_matrix())),
    };
    let result = decomposer.decompose_gate(&gate);
    assert!(result.is_ok());
    let dr = result.unwrap();
    assert!(dr.fidelity >= 0.99);
    assert!(dr.gate_count > 0);
}

#[test]
fn test_universal_decomposer_multiple_gates() {
    let config = DecompositionConfig {
        basis: BasisGateSet::IBM,
        optimization_level: 0,
        max_depth: None,
        fidelity_threshold: 0.9,
        max_gates: None,
        allow_ancillas: false,
        num_ancillas: 0,
        clifford_t_epsilon: 1e-6,
    };
    let decomposer = UniversalDecomposer::new(config);
    let gates: Vec<Arc<dyn Gate>> = vec![
        Arc::new(MockGate {
            name: "H".to_string(),
            n_qubits: 1,
            matrix: Some(matrix2_to_flat(&hadamard_matrix())),
        }),
        Arc::new(MockGate {
            name: "X".to_string(),
            n_qubits: 1,
            matrix: Some(matrix2_to_flat(&pauli_x_matrix())),
        }),
    ];
    let results = decomposer.decompose_gates(&gates);
    assert!(results.is_ok());
    let rs = results.unwrap();
    assert_eq!(rs.len(), 2);
}
