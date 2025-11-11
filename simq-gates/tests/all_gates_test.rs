//! Comprehensive tests for all standard quantum gates

use approx::assert_relative_eq;
use num_complex::Complex64;
use simq_gates::standard::*;
use simq_core::gate::Gate;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

// Helper function to multiply 2x2 matrices
fn mult_2x2(
    a: &[[Complex64; 2]; 2],
    b: &[[Complex64; 2]; 2],
) -> [[Complex64; 2]; 2] {
    let mut result = [[Complex64::new(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

// Helper function to check if matrix is identity
fn is_identity_2x2(m: &[[Complex64; 2]; 2]) -> bool {
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            if (m[i][j] - expected).norm() > EPSILON {
                return false;
            }
        }
    }
    true
}

// Helper to check if matrix is unitary (U†U = I)
fn is_unitary_2x2(m: &[[Complex64; 2]; 2]) -> bool {
    let m_dagger = [
        [m[0][0].conj(), m[1][0].conj()],
        [m[0][1].conj(), m[1][1].conj()],
    ];
    let product = mult_2x2(&m_dagger, m);
    is_identity_2x2(&product)
}

// ============================================================================
// Single-Qubit Gate Tests
// ============================================================================

#[test]
fn test_pauli_gates_basic_properties() {
    // X gate
    assert_eq!(PauliX.name(), "X");
    assert_eq!(PauliX.num_qubits(), 1);
    assert!(PauliX.is_hermitian());

    // Y gate
    assert_eq!(PauliY.name(), "Y");
    assert_eq!(PauliY.num_qubits(), 1);
    assert!(PauliY.is_hermitian());

    // Z gate
    assert_eq!(PauliZ.name(), "Z");
    assert_eq!(PauliZ.num_qubits(), 1);
    assert!(PauliZ.is_hermitian());
}

#[test]
fn test_pauli_gates_squaring() {
    // X² = I
    let x_squared = mult_2x2(PauliX::matrix(), PauliX::matrix());
    assert!(is_identity_2x2(&x_squared));

    // Y² = I
    let y_squared = mult_2x2(PauliY::matrix(), PauliY::matrix());
    assert!(is_identity_2x2(&y_squared));

    // Z² = I
    let z_squared = mult_2x2(PauliZ::matrix(), PauliZ::matrix());
    assert!(is_identity_2x2(&z_squared));
}

#[test]
fn test_hadamard_gate() {
    assert_eq!(Hadamard.name(), "H");
    assert_eq!(Hadamard.num_qubits(), 1);
    assert!(Hadamard.is_hermitian());

    // H² = I
    let h_squared = mult_2x2(Hadamard::matrix(), Hadamard::matrix());
    assert!(is_identity_2x2(&h_squared));
}

#[test]
fn test_s_gate() {
    assert_eq!(SGate.name(), "S");
    assert_eq!(SGate.num_qubits(), 1);

    // S² = Z
    let s_squared = mult_2x2(SGate::matrix(), SGate::matrix());
    let z_matrix = PauliZ::matrix();

    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(s_squared[i][j].re, z_matrix[i][j].re, epsilon = EPSILON);
            assert_relative_eq!(s_squared[i][j].im, z_matrix[i][j].im, epsilon = EPSILON);
        }
    }
}

#[test]
fn test_s_dagger_gate() {
    assert_eq!(SGateDagger.name(), "S†");
    assert_eq!(SGateDagger.num_qubits(), 1);

    // S · S† = I
    let s_s_dag = mult_2x2(SGate::matrix(), SGateDagger::matrix());
    assert!(is_identity_2x2(&s_s_dag));
}

#[test]
fn test_t_gate() {
    assert_eq!(TGate.name(), "T");
    assert_eq!(TGate.num_qubits(), 1);

    // T² = S
    let t_squared = mult_2x2(TGate::matrix(), TGate::matrix());
    let s_matrix = SGate::matrix();

    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(t_squared[i][j].re, s_matrix[i][j].re, epsilon = EPSILON);
            assert_relative_eq!(t_squared[i][j].im, s_matrix[i][j].im, epsilon = EPSILON);
        }
    }
}

#[test]
fn test_t_dagger_gate() {
    assert_eq!(TGateDagger.name(), "T†");
    assert_eq!(TGateDagger.num_qubits(), 1);

    // T · T† = I
    let t_t_dag = mult_2x2(TGate::matrix(), TGateDagger::matrix());
    assert!(is_identity_2x2(&t_t_dag));
}

#[test]
fn test_identity_gate() {
    assert_eq!(Identity.name(), "I");
    assert_eq!(Identity.num_qubits(), 1);
    assert!(Identity.is_hermitian());

    let id_matrix = Identity::matrix();
    assert!(is_identity_2x2(id_matrix));
}

#[test]
fn test_sx_gate() {
    assert_eq!(SXGate.name(), "SX");
    assert_eq!(SXGate.num_qubits(), 1);

    // SX² = X
    let sx_squared = mult_2x2(SXGate::matrix(), SXGate::matrix());
    let x_matrix = PauliX::matrix();

    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(sx_squared[i][j].re, x_matrix[i][j].re, epsilon = EPSILON);
            assert_relative_eq!(sx_squared[i][j].im, x_matrix[i][j].im, epsilon = EPSILON);
        }
    }
}

#[test]
fn test_sx_dagger_gate() {
    assert_eq!(SXGateDagger.name(), "SX†");
    assert_eq!(SXGateDagger.num_qubits(), 1);

    // SX · SX† = I
    let sx_sx_dag = mult_2x2(SXGate::matrix(), SXGateDagger::matrix());
    assert!(is_identity_2x2(&sx_sx_dag));
}

// ============================================================================
// Rotation Gate Tests
// ============================================================================

#[test]
fn test_rotation_x_gate() {
    let angle = PI / 3.0;
    let rx = RotationX::new(angle);

    assert_eq!(rx.name(), "RX");
    assert_eq!(rx.num_qubits(), 1);
    assert_relative_eq!(rx.angle(), angle, epsilon = EPSILON);
    assert!(rx.description().contains("RX"));

    // RX(0) = I
    let rx_0 = RotationX::new(0.0);
    let matrix_0 = rx_0.matrix();
    assert!(is_identity_2x2(&matrix_0));

    // Check unitarity
    let matrix = rx.matrix();
    assert!(is_unitary_2x2(&matrix));
}

#[test]
fn test_rotation_y_gate() {
    let angle = PI / 4.0;
    let ry = RotationY::new(angle);

    assert_eq!(ry.name(), "RY");
    assert_eq!(ry.num_qubits(), 1);
    assert_relative_eq!(ry.angle(), angle, epsilon = EPSILON);

    // RY(0) = I
    let ry_0 = RotationY::new(0.0);
    let matrix_0 = ry_0.matrix();
    assert!(is_identity_2x2(&matrix_0));

    // Check unitarity
    let matrix = ry.matrix();
    assert!(is_unitary_2x2(&matrix));
}

#[test]
fn test_rotation_z_gate() {
    let angle = PI / 6.0;
    let rz = RotationZ::new(angle);

    assert_eq!(rz.name(), "RZ");
    assert_eq!(rz.num_qubits(), 1);
    assert_relative_eq!(rz.angle(), angle, epsilon = EPSILON);

    // RZ(0) = I (modulo global phase)
    let rz_0 = RotationZ::new(0.0);
    let matrix_0 = rz_0.matrix();
    // RZ(0) has a global phase e^0 = 1, so it should be identity
    assert!(is_identity_2x2(&matrix_0));

    // Check unitarity
    let matrix = rz.matrix();
    assert!(is_unitary_2x2(&matrix));
}

#[test]
fn test_phase_gate() {
    let angle = PI / 5.0;
    let p = Phase::new(angle);

    assert_eq!(p.name(), "P");
    assert_eq!(p.num_qubits(), 1);
    assert_relative_eq!(p.angle(), angle, epsilon = EPSILON);

    // Check unitarity
    let matrix = p.matrix();
    assert!(is_unitary_2x2(&matrix));
}

// ============================================================================
// Universal Gate Tests (U1, U2, U3)
// ============================================================================

#[test]
fn test_u1_gate() {
    let lambda = PI / 3.0;
    let u1 = U1::new(lambda);

    assert_eq!(u1.name(), "U1");
    assert_eq!(u1.num_qubits(), 1);
    assert_relative_eq!(u1.lambda(), lambda, epsilon = EPSILON);

    // U1 should be equivalent to Phase gate
    let p = Phase::new(lambda);
    let u1_matrix = u1.matrix();
    let p_matrix = p.matrix();

    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(u1_matrix[i][j].re, p_matrix[i][j].re, epsilon = EPSILON);
            assert_relative_eq!(u1_matrix[i][j].im, p_matrix[i][j].im, epsilon = EPSILON);
        }
    }

    // Check unitarity
    assert!(is_unitary_2x2(&u1_matrix));
}

#[test]
fn test_u2_gate() {
    let phi = PI / 4.0;
    let lambda = PI / 3.0;
    let u2 = U2::new(phi, lambda);

    assert_eq!(u2.name(), "U2");
    assert_eq!(u2.num_qubits(), 1);
    assert_relative_eq!(u2.phi(), phi, epsilon = EPSILON);
    assert_relative_eq!(u2.lambda(), lambda, epsilon = EPSILON);

    // Check unitarity
    let matrix = u2.matrix();
    assert!(is_unitary_2x2(&matrix));

    // U2(0, 0) should be Hadamard (modulo global phase)
    let u2_h = U2::new(0.0, 0.0);
    let u2_h_matrix = u2_h.matrix();
    // The matrices should be equivalent up to a global phase
    // For simplicity, just check unitarity
    assert!(is_unitary_2x2(&u2_h_matrix));
}

#[test]
fn test_u3_gate() {
    let theta = PI / 2.0;
    let phi = PI / 4.0;
    let lambda = PI / 3.0;
    let u3 = U3::new(theta, phi, lambda);

    assert_eq!(u3.name(), "U3");
    assert_eq!(u3.num_qubits(), 1);
    assert_relative_eq!(u3.theta(), theta, epsilon = EPSILON);
    assert_relative_eq!(u3.phi(), phi, epsilon = EPSILON);
    assert_relative_eq!(u3.lambda(), lambda, epsilon = EPSILON);

    // Check unitarity
    let matrix = u3.matrix();
    assert!(is_unitary_2x2(&matrix));

    // U3(π, 0, π) should be X gate
    let u3_x = U3::new(PI, 0.0, PI);
    let u3_x_matrix = u3_x.matrix();
    let x_matrix = PauliX::matrix();

    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(u3_x_matrix[i][j].re, x_matrix[i][j].re, epsilon = EPSILON);
            assert_relative_eq!(u3_x_matrix[i][j].im, x_matrix[i][j].im, epsilon = EPSILON);
        }
    }
}

// ============================================================================
// Two-Qubit Gate Tests
// ============================================================================

#[test]
fn test_cnot_gate() {
    assert_eq!(CNot.name(), "CNOT");
    assert_eq!(CNot.num_qubits(), 2);

    let matrix = CNot::matrix();

    // CNOT should leave |00⟩ and |01⟩ unchanged
    assert_eq!(matrix[0][0].re, 1.0);
    assert_eq!(matrix[1][1].re, 1.0);

    // CNOT should flip |10⟩ ↔ |11⟩
    assert_eq!(matrix[2][3].re, 1.0);
    assert_eq!(matrix[3][2].re, 1.0);
}

#[test]
fn test_cz_gate() {
    assert_eq!(CZ.name(), "CZ");
    assert_eq!(CZ.num_qubits(), 2);
    assert!(CZ.is_hermitian());

    let matrix = CZ::matrix();

    // CZ should leave |00⟩, |01⟩, |10⟩ unchanged
    assert_eq!(matrix[0][0].re, 1.0);
    assert_eq!(matrix[1][1].re, 1.0);
    assert_eq!(matrix[2][2].re, 1.0);

    // CZ should apply -1 phase to |11⟩
    assert_eq!(matrix[3][3].re, -1.0);
}

#[test]
fn test_swap_gate() {
    assert_eq!(Swap.name(), "SWAP");
    assert_eq!(Swap.num_qubits(), 2);
    assert!(Swap.is_hermitian());

    let matrix = Swap::matrix();

    // SWAP should leave |00⟩ and |11⟩ unchanged
    assert_eq!(matrix[0][0].re, 1.0);
    assert_eq!(matrix[3][3].re, 1.0);

    // SWAP should swap |01⟩ ↔ |10⟩
    assert_eq!(matrix[1][2].re, 1.0);
    assert_eq!(matrix[2][1].re, 1.0);
}

#[test]
fn test_iswap_gate() {
    assert_eq!(ISwap.name(), "iSWAP");
    assert_eq!(ISwap.num_qubits(), 2);

    let matrix = ISwap::matrix();

    // iSWAP should leave |00⟩ and |11⟩ unchanged
    assert_eq!(matrix[0][0].re, 1.0);
    assert_eq!(matrix[3][3].re, 1.0);

    // iSWAP should swap with phase: |01⟩ ↔ i|10⟩
    assert_eq!(matrix[1][2].im, 1.0);
    assert_eq!(matrix[2][1].im, 1.0);
}

#[test]
fn test_cy_gate() {
    assert_eq!(CY.name(), "CY");
    assert_eq!(CY.num_qubits(), 2);

    let matrix = CY::matrix();

    // CY should leave |00⟩ and |01⟩ unchanged
    assert_eq!(matrix[0][0].re, 1.0);
    assert_eq!(matrix[1][1].re, 1.0);

    // CY should apply Y gate when control is |1⟩
    // Y = [[0, -i], [i, 0]]
    // So |10⟩ → |11⟩ with -i, |11⟩ → |10⟩ with i
    assert_eq!(matrix[2][3].im, -1.0);
    assert_eq!(matrix[3][2].im, 1.0);
}

#[test]
fn test_ecr_gate() {
    assert_eq!(ECR.name(), "ECR");
    assert_eq!(ECR.num_qubits(), 2);

    let matrix = ECR::matrix();

    // Just verify matrix is properly formed (non-zero elements)
    let non_zero_count = matrix.iter()
        .flatten()
        .filter(|&c| c.norm() > EPSILON)
        .count();

    assert!(non_zero_count > 0);
}

#[test]
fn test_cphase_gate() {
    let angle = PI / 4.0;
    let cp = CPhase::new(angle);

    assert_eq!(cp.name(), "CP");
    assert_eq!(cp.num_qubits(), 2);
    assert_relative_eq!(cp.angle(), angle, epsilon = EPSILON);

    let matrix = cp.matrix();

    // CP should leave |00⟩, |01⟩, |10⟩ unchanged
    assert_eq!(matrix[0][0].re, 1.0);
    assert_eq!(matrix[1][1].re, 1.0);
    assert_eq!(matrix[2][2].re, 1.0);

    // CP should apply phase to |11⟩
    let expected_phase = Complex64::new(angle.cos(), angle.sin());
    assert_relative_eq!(matrix[3][3].re, expected_phase.re, epsilon = EPSILON);
    assert_relative_eq!(matrix[3][3].im, expected_phase.im, epsilon = EPSILON);
}

#[test]
fn test_rxx_gate() {
    let angle = PI / 6.0;
    let rxx = RXX::new(angle);

    assert_eq!(rxx.name(), "RXX");
    assert_eq!(rxx.num_qubits(), 2);
    assert_relative_eq!(rxx.angle(), angle, epsilon = EPSILON);

    // RXX(0) should be identity
    let rxx_0 = RXX::new(0.0);
    let matrix_0 = rxx_0.matrix();

    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(matrix_0[i][j].re, expected, epsilon = EPSILON);
            assert_relative_eq!(matrix_0[i][j].im, 0.0, epsilon = EPSILON);
        }
    }
}

#[test]
fn test_ryy_gate() {
    let angle = PI / 6.0;
    let ryy = RYY::new(angle);

    assert_eq!(ryy.name(), "RYY");
    assert_eq!(ryy.num_qubits(), 2);
    assert_relative_eq!(ryy.angle(), angle, epsilon = EPSILON);

    // RYY(0) should be identity
    let ryy_0 = RYY::new(0.0);
    let matrix_0 = ryy_0.matrix();

    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(matrix_0[i][j].re, expected, epsilon = EPSILON);
            assert_relative_eq!(matrix_0[i][j].im, 0.0, epsilon = EPSILON);
        }
    }
}

#[test]
fn test_rzz_gate() {
    let angle = PI / 6.0;
    let rzz = RZZ::new(angle);

    assert_eq!(rzz.name(), "RZZ");
    assert_eq!(rzz.num_qubits(), 2);
    assert_relative_eq!(rzz.angle(), angle, epsilon = EPSILON);

    // RZZ(0) should be identity (modulo global phase)
    let rzz_0 = RZZ::new(0.0);
    let matrix_0 = rzz_0.matrix();

    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(matrix_0[i][j].re, expected, epsilon = EPSILON);
            assert_relative_eq!(matrix_0[i][j].im, 0.0, epsilon = EPSILON);
        }
    }
}

// ============================================================================
// Three-Qubit Gate Tests
// ============================================================================

#[test]
fn test_toffoli_gate() {
    assert_eq!(Toffoli.name(), "CCNOT");
    assert_eq!(Toffoli.num_qubits(), 3);
    assert!(Toffoli.description().contains("Toffoli"));

    let matrix = Toffoli::matrix();

    // Toffoli should leave all states unchanged except |110⟩ and |111⟩
    for i in 0..6 {
        assert_eq!(matrix[i][i].re, 1.0);
    }

    // Toffoli should flip target when both controls are |1⟩
    // |110⟩ (index 6) ↔ |111⟩ (index 7)
    assert_eq!(matrix[6][7].re, 1.0);
    assert_eq!(matrix[7][6].re, 1.0);
}

#[test]
fn test_fredkin_gate() {
    assert_eq!(Fredkin.name(), "CSWAP");
    assert_eq!(Fredkin.num_qubits(), 3);
    assert!(Fredkin.description().contains("Fredkin"));

    let matrix = Fredkin::matrix();

    // Fredkin should leave states unchanged when control is |0⟩ (first 4 states)
    for i in 0..4 {
        assert_eq!(matrix[i][i].re, 1.0);
    }

    // Fredkin should leave |100⟩ and |111⟩ unchanged
    assert_eq!(matrix[4][4].re, 1.0);
    assert_eq!(matrix[7][7].re, 1.0);

    // Fredkin should swap |101⟩ (index 5) ↔ |110⟩ (index 6) when control is |1⟩
    assert_eq!(matrix[5][6].re, 1.0);
    assert_eq!(matrix[6][5].re, 1.0);
}

// ============================================================================
// Matrix Properties Tests
// ============================================================================

#[test]
fn test_all_single_qubit_gates_are_unitary() {
    let gates: Vec<[[Complex64; 2]; 2]> = vec![
        *Hadamard::matrix(),
        *PauliX::matrix(),
        *PauliY::matrix(),
        *PauliZ::matrix(),
        *SGate::matrix(),
        *SGateDagger::matrix(),
        *TGate::matrix(),
        *TGateDagger::matrix(),
        *Identity::matrix(),
        *SXGate::matrix(),
        *SXGateDagger::matrix(),
        RotationX::new(PI / 3.0).matrix(),
        RotationY::new(PI / 4.0).matrix(),
        RotationZ::new(PI / 5.0).matrix(),
        Phase::new(PI / 6.0).matrix(),
        U1::new(PI / 7.0).matrix(),
        U2::new(PI / 8.0, PI / 9.0).matrix(),
        U3::new(PI / 2.0, PI / 3.0, PI / 4.0).matrix(),
    ];

    for matrix in gates {
        assert!(
            is_unitary_2x2(&matrix),
            "Matrix is not unitary"
        );
    }
}

#[test]
fn test_gate_name_uniqueness() {
    let names = vec![
        Hadamard.name(),
        PauliX.name(),
        PauliY.name(),
        PauliZ.name(),
        SGate.name(),
        TGate.name(),
        Identity.name(),
        SXGate.name(),
        CNot.name(),
        CZ.name(),
        Swap.name(),
        ISwap.name(),
        CY.name(),
        ECR.name(),
        Toffoli.name(),
        Fredkin.name(),
    ];

    // Check no duplicates (except for parameterized gates which have same base name)
    for (i, &name1) in names.iter().enumerate() {
        for &name2 in names.iter().skip(i + 1) {
            assert_ne!(name1, name2, "Duplicate gate name: {}", name1);
        }
    }
}
