//! Comprehensive end-to-end tests for ferriq-gates crate
//!
//! Covers: standard gates, custom gates, parametric custom gates,
//! gate registry, matrix operations, caching, lookup tables, and optimized gates.

use num_complex::Complex64;
use ferriq_core::gate::Gate;
use ferriq_gates::standard::*;
use ferriq_gates::*;
use std::f64::consts::PI;
use std::sync::Arc;

const EPSILON: f64 = 1e-8;

fn mult_2x2(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
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

fn is_identity_2x2(m: &[[Complex64; 2]; 2]) -> bool {
    for (i, row) in m.iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            let expected = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            if (val - expected).norm() > EPSILON {
                return false;
            }
        }
    }
    true
}

fn is_unitary_2x2(m: &[[Complex64; 2]; 2]) -> bool {
    let m_dagger = [
        [m[0][0].conj(), m[1][0].conj()],
        [m[0][1].conj(), m[1][1].conj()],
    ];
    is_identity_2x2(&mult_2x2(&m_dagger, m))
}

// ============================================================================
// 1. Standard single-qubit gate properties
// ============================================================================

#[test]
fn hadamard_properties() {
    assert_eq!(Hadamard.name(), "H");
    assert_eq!(Hadamard.num_qubits(), 1);
    assert!(Hadamard.is_hermitian());
    assert!(!Hadamard.is_diagonal());
    let m = Hadamard::matrix();
    assert!(is_unitary_2x2(m));
    assert!(is_identity_2x2(&mult_2x2(m, m)));
}

#[test]
fn pauli_x_properties() {
    assert_eq!(PauliX.name(), "X");
    assert!(PauliX.is_hermitian());
    assert!(!PauliX.is_diagonal());
    let m = PauliX::matrix();
    assert!(is_identity_2x2(&mult_2x2(m, m)));
}

#[test]
fn pauli_y_properties() {
    assert_eq!(PauliY.name(), "Y");
    assert!(PauliY.is_hermitian());
    assert!(!PauliY.is_diagonal());
    let m = PauliY::matrix();
    assert!(is_identity_2x2(&mult_2x2(m, m)));
}

#[test]
fn pauli_z_properties() {
    assert_eq!(PauliZ.name(), "Z");
    assert!(PauliZ.is_hermitian());
    assert!(PauliZ.is_diagonal());
    let m = PauliZ::matrix();
    assert!(is_identity_2x2(&mult_2x2(m, m)));
}

#[test]
fn identity_gate_properties() {
    assert_eq!(Identity.name(), "I");
    assert_eq!(Identity.num_qubits(), 1);
    assert!(Identity.is_hermitian());
    assert!(Identity.is_diagonal());
    let m = Identity::matrix();
    assert!(is_identity_2x2(m));
}

#[test]
fn s_gate_properties() {
    assert_eq!(SGate.name(), "S");
    assert!(!SGate.is_hermitian());
    assert!(SGate.is_diagonal());
    let m = SGate::matrix();
    assert!(is_unitary_2x2(m));
    let s2 = mult_2x2(m, m);
    let z = PauliZ::matrix();
    for i in 0..2 {
        for j in 0..2 {
            assert!((s2[i][j] - z[i][j]).norm() < EPSILON, "S^2 should equal Z");
        }
    }
}

#[test]
fn t_gate_properties() {
    assert_eq!(TGate.name(), "T");
    assert!(!TGate.is_hermitian());
    assert!(TGate.is_diagonal());
    let m = TGate::matrix();
    assert!(is_unitary_2x2(m));
    let t2 = mult_2x2(m, m);
    let s = SGate::matrix();
    for i in 0..2 {
        for j in 0..2 {
            assert!((t2[i][j] - s[i][j]).norm() < EPSILON, "T^2 should equal S");
        }
    }
}

#[test]
fn sx_gate_properties() {
    assert_eq!(SXGate.name(), "SX");
    assert!(!SXGate.is_hermitian());
    assert!(!SXGate.is_diagonal());
    let m = SXGate::matrix();
    assert!(is_unitary_2x2(m));
    let sx2 = mult_2x2(m, m);
    let x = PauliX::matrix();
    for i in 0..2 {
        for j in 0..2 {
            assert!((sx2[i][j] - x[i][j]).norm() < EPSILON, "SX^2 should equal X");
        }
    }
}

#[test]
fn dagger_gates_are_inverse() {
    let s = SGate::matrix();
    let sd = SGateDagger::matrix();
    assert!(is_identity_2x2(&mult_2x2(s, sd)));

    let t = TGate::matrix();
    let td = TGateDagger::matrix();
    assert!(is_identity_2x2(&mult_2x2(t, td)));

    let sx = SXGate::matrix();
    let sxd = SXGateDagger::matrix();
    assert!(is_identity_2x2(&mult_2x2(sx, sxd)));
}

#[test]
fn all_single_qubit_gates_are_unitary() {
    let gates: Vec<Box<dyn Gate>> = vec![
        Box::new(Hadamard),
        Box::new(PauliX),
        Box::new(PauliY),
        Box::new(PauliZ),
        Box::new(SGate),
        Box::new(SGateDagger),
        Box::new(TGate),
        Box::new(TGateDagger),
        Box::new(SXGate),
        Box::new(SXGateDagger),
        Box::new(Identity),
    ];
    for gate in &gates {
        assert_eq!(gate.num_qubits(), 1, "{} should be single-qubit", gate.name());
        let matrix_vec = gate.matrix();
        assert!(matrix_vec.is_some(), "{} should have a matrix", gate.name());
        let flat = matrix_vec.unwrap();
        assert_eq!(flat.len(), 4, "{} matrix should be 2x2 flat", gate.name());
        assert!(is_unitary(&flat, 1e-8), "{} should be unitary", gate.name());
    }
}

// ============================================================================
// 2. Two-qubit gate properties
// ============================================================================

#[test]
fn cnot_gate_properties() {
    assert_eq!(CNot.name(), "CNOT");
    assert_eq!(CNot.num_qubits(), 2);
    let flat = CNot.matrix().unwrap();
    assert_eq!(flat.len(), 16);
    assert!(is_unitary(&flat, 1e-8));
}

#[test]
fn cz_gate_properties() {
    assert_eq!(CZ.name(), "CZ");
    assert_eq!(CZ.num_qubits(), 2);
    assert!(CZ.is_hermitian());
    assert!(CZ.is_diagonal());
    let flat = CZ.matrix().unwrap();
    assert!(is_unitary(&flat, 1e-8));
}

#[test]
fn swap_gate_properties() {
    assert_eq!(Swap.name(), "SWAP");
    assert_eq!(Swap.num_qubits(), 2);
    assert!(Swap.is_hermitian());
    let flat = Swap.matrix().unwrap();
    assert!(is_unitary(&flat, 1e-8));
}

#[test]
fn iswap_gate_properties() {
    assert_eq!(ISwap.name(), "iSWAP");
    assert_eq!(ISwap.num_qubits(), 2);
    let flat = ISwap.matrix().unwrap();
    assert!(is_unitary(&flat, 1e-8));
}

#[test]
fn cy_ecr_properties() {
    assert_eq!(CY.name(), "CY");
    assert_eq!(CY.num_qubits(), 2);
    let cy_flat = CY.matrix().unwrap();
    assert!(is_unitary(&cy_flat, 1e-8));

    assert_eq!(ECR.name(), "ECR");
    assert_eq!(ECR.num_qubits(), 2);
    let ecr_flat = ECR.matrix().unwrap();
    assert!(is_unitary(&ecr_flat, 1e-8));
}

// ============================================================================
// 3. Three-qubit gate properties
// ============================================================================

#[test]
fn toffoli_gate_properties() {
    assert_eq!(Toffoli.name(), "CCNOT");
    assert_eq!(Toffoli.num_qubits(), 3);
    let flat = Toffoli.matrix().unwrap();
    assert_eq!(flat.len(), 64);
    assert!(is_unitary(&flat, 1e-8));
}

#[test]
fn fredkin_gate_properties() {
    assert_eq!(Fredkin.name(), "CSWAP");
    assert_eq!(Fredkin.num_qubits(), 3);
    let flat = Fredkin.matrix().unwrap();
    assert_eq!(flat.len(), 64);
    assert!(is_unitary(&flat, 1e-8));
}

// ============================================================================
// 4. Parameterized rotation gates
// ============================================================================

#[test]
fn rotation_x_at_special_angles() {
    let rx_pi = RotationX::new(PI);
    assert_eq!(rx_pi.name(), "RX");
    assert_eq!(rx_pi.num_qubits(), 1);
    let m = rx_pi.matrix();
    assert!(is_unitary_2x2(&m));
    let x = PauliX::matrix();
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (m[i][j].norm() - x[i][j].norm()).abs() < EPSILON,
                "RX(π) should match X up to global phase"
            );
        }
    }

    let rx_0 = RotationX::new(0.0);
    let m0 = rx_0.matrix();
    assert!(is_identity_2x2(&m0));
}

#[test]
fn rotation_y_at_special_angles() {
    let ry_0 = RotationY::new(0.0);
    assert!(is_identity_2x2(&ry_0.matrix()));

    let ry = RotationY::new(PI / 3.0);
    assert!(is_unitary_2x2(&ry.matrix()));
}

#[test]
fn rotation_z_at_special_angles() {
    let rz_0 = RotationZ::new(0.0);
    assert!(is_identity_2x2(&rz_0.matrix()));

    let rz = RotationZ::new(PI / 4.0);
    assert_eq!(rz.name(), "RZ");
    assert!(rz.is_diagonal());
    assert!(is_unitary_2x2(&rz.matrix()));
}

#[test]
fn phase_gate_at_special_angles() {
    let p0 = Phase::new(0.0);
    assert!(is_identity_2x2(&p0.matrix()));

    let p_pi = Phase::new(PI);
    assert_eq!(p_pi.name(), "P");
    assert!(p_pi.is_diagonal());
    let m = p_pi.matrix();
    assert!(is_unitary_2x2(&m));
    let z = PauliZ::matrix();
    for i in 0..2 {
        for j in 0..2 {
            assert!((m[i][j].norm() - z[i][j].norm()).abs() < EPSILON);
        }
    }
}

#[test]
fn rotation_inverse_check() {
    let theta = 0.7;
    let rx = RotationX::new(theta);
    let rx_neg = RotationX::new(-theta);
    let product = mult_2x2(&rx.matrix_uncached(), &rx_neg.matrix_uncached());
    assert!(is_identity_2x2(&product));

    let ry = RotationY::new(theta);
    let ry_neg = RotationY::new(-theta);
    assert!(is_identity_2x2(&mult_2x2(&ry.matrix_uncached(), &ry_neg.matrix_uncached())));

    let rz = RotationZ::new(theta);
    let rz_neg = RotationZ::new(-theta);
    assert!(is_identity_2x2(&mult_2x2(&rz.matrix_uncached(), &rz_neg.matrix_uncached())));
}

#[test]
fn rotation_additivity() {
    let a = 0.3;
    let b = 0.5;
    let rx_a = RotationX::new(a);
    let rx_b = RotationX::new(b);
    let rx_ab = RotationX::new(a + b);
    let composed = mult_2x2(&rx_b.matrix_uncached(), &rx_a.matrix_uncached());
    let expected = rx_ab.matrix_uncached();
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (composed[i][j] - expected[i][j]).norm() < EPSILON,
                "RX(a)*RX(b) should equal RX(a+b)"
            );
        }
    }
}

#[test]
fn rotation_2pi_is_minus_identity() {
    let rx = RotationX::new(2.0 * PI);
    let m = rx.matrix();
    for (i, row) in m.iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            let expected = if i == j {
                Complex64::new(-1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            assert!((val - expected).norm() < EPSILON);
        }
    }
}

// ============================================================================
// 5. Universal gates (U1, U2, U3)
// ============================================================================

#[test]
fn u1_gate() {
    let u1 = U1::new(PI);
    assert_eq!(u1.name(), "U1");
    assert_eq!(u1.num_qubits(), 1);
    assert!(is_unitary_2x2(&u1.matrix()));
}

#[test]
fn u2_gate() {
    let u2 = U2::new(0.0, PI);
    assert_eq!(u2.name(), "U2");
    assert!(is_unitary_2x2(&u2.matrix()));
}

#[test]
fn u3_gate() {
    let u3 = U3::new(PI, 0.0, PI);
    assert_eq!(u3.name(), "U3");
    assert!(is_unitary_2x2(&u3.matrix()));
}

#[test]
fn u3_is_most_general() {
    let u3_h = U3::new(PI / 2.0, 0.0, PI);
    let h = Hadamard::matrix();
    let m = u3_h.matrix();
    for i in 0..2 {
        for j in 0..2 {
            assert!((m[i][j] - h[i][j]).norm() < EPSILON, "U3(π/2, 0, π) ≈ H");
        }
    }
}

// ============================================================================
// 6. Controlled-phase and two-qubit rotation gates
// ============================================================================

#[test]
fn controlled_phase_gate() {
    let cp = CPhase::new(PI);
    assert_eq!(cp.name(), "CP");
    assert_eq!(cp.num_qubits(), 2);
    let gate_ref: &dyn Gate = &cp;
    let flat = gate_ref.matrix().unwrap();
    assert_eq!(flat.len(), 16);
    assert!(is_unitary(&flat, 1e-8));
}

#[test]
fn rxx_ryy_rzz_gates() {
    let theta = 0.5;
    let rxx = RXX::new(theta);
    assert_eq!(rxx.name(), "RXX");
    assert_eq!(rxx.num_qubits(), 2);
    let rxx_ref: &dyn Gate = &rxx;
    let flat_rxx = rxx_ref.matrix().unwrap();
    assert!(is_unitary(&flat_rxx, 1e-8));

    let ryy = RYY::new(theta);
    assert_eq!(ryy.name(), "RYY");
    let ryy_ref: &dyn Gate = &ryy;
    assert!(is_unitary(&ryy_ref.matrix().unwrap(), 1e-8));

    let rzz = RZZ::new(theta);
    assert_eq!(rzz.name(), "RZZ");
    let rzz_ref: &dyn Gate = &rzz;
    assert!(is_unitary(&rzz_ref.matrix().unwrap(), 1e-8));
}

#[test]
fn two_qubit_rotation_zero_angle_is_identity() {
    let rxx = RXX::new(0.0);
    let rxx_ref: &dyn Gate = &rxx;
    let flat = rxx_ref.matrix().unwrap();
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            assert!((flat[i * 4 + j] - expected).norm() < EPSILON, "RXX(0) should be identity");
        }
    }
}

// ============================================================================
// 7. Gate trait method: matrix() as Option<Vec<Complex64>>
// ============================================================================

#[test]
fn trait_matrix_returns_flat_vec() {
    let h: &dyn Gate = &Hadamard;
    let m = h.matrix().unwrap();
    assert_eq!(m.len(), 4);
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    assert!((m[0] - Complex64::new(inv_sqrt2, 0.0)).norm() < EPSILON);

    let cnot: &dyn Gate = &CNot;
    let cm = cnot.matrix().unwrap();
    assert_eq!(cm.len(), 16);

    let toffoli: &dyn Gate = &Toffoli;
    let tm = toffoli.matrix().unwrap();
    assert_eq!(tm.len(), 64);
}

// ============================================================================
// 8. CustomGate building and validation
// ============================================================================

#[test]
fn custom_gate_builder_hadamard_clone() {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let custom_h = CustomGateBuilder::new("MyH")
        .matrix_2x2([
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
            ],
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        ])
        .build()
        .unwrap();
    assert_eq!(custom_h.name(), "MyH");
    assert_eq!(custom_h.num_qubits(), 1);
    assert!(custom_h.is_hermitian());
}

#[test]
fn custom_gate_non_unitary_rejected() {
    let result = CustomGateBuilder::new("Bad")
        .matrix_2x2([
            [Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ])
        .build();
    assert!(result.is_err());
}

#[test]
fn custom_gate_with_description() {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let gate = CustomGateBuilder::new("MyGate")
        .matrix_2x2([
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
            ],
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        ])
        .description("A custom Hadamard-like gate")
        .build()
        .unwrap();
    assert_eq!(gate.description(), "A custom Hadamard-like gate");
}

#[test]
fn custom_gate_build_arc() {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let gate: Arc<dyn Gate> = CustomGateBuilder::new("ArcGate")
        .matrix_2x2([
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
            ],
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        ])
        .build_arc()
        .unwrap();
    assert_eq!(gate.name(), "ArcGate");
}

#[test]
fn custom_gate_adjoint() {
    let gate = CustomGateBuilder::new("S_custom")
        .matrix_2x2([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
        ])
        .build()
        .unwrap();
    let adj = gate.adjoint();
    assert_eq!(adj.name(), "S_custom†");
    let m = gate.matrix_vec();
    let ma = adj.matrix_vec();
    let flat_orig = [m[0], m[2], m[1], m[3]];
    for i in 0..4 {
        assert!((flat_orig[i].conj() - ma[i]).norm() < EPSILON);
    }
}

#[test]
fn custom_gate_compose() {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let h = CustomGateBuilder::new("H")
        .matrix_2x2([
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
            ],
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        ])
        .build()
        .unwrap();
    let hh = h.compose(&h).unwrap();
    let m = hh.matrix_vec();
    assert!((m[0] - Complex64::new(1.0, 0.0)).norm() < EPSILON);
    assert!(m[1].norm() < EPSILON);
    assert!(m[2].norm() < EPSILON);
    assert!((m[3] - Complex64::new(1.0, 0.0)).norm() < EPSILON);
}

#[test]
fn custom_gate_controlled() {
    let x = CustomGateBuilder::new("X")
        .matrix_2x2([
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ])
        .build()
        .unwrap();
    let cx = x.controlled().unwrap();
    assert_eq!(cx.name(), "CX");
    assert_eq!(cx.num_qubits(), 2);
    let m = cx.matrix_vec();
    assert_eq!(m.len(), 16);
}

#[test]
fn custom_gate_fidelity() {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let h1 = CustomGateBuilder::new("H1")
        .matrix_2x2([
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
            ],
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        ])
        .build()
        .unwrap();
    let h2 = CustomGateBuilder::new("H2")
        .matrix_2x2([
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
            ],
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        ])
        .build()
        .unwrap();
    let f = h1.fidelity(h2.matrix_vec()).unwrap();
    assert!((f - 1.0).abs() < EPSILON);
}

#[test]
fn custom_gate_4x4() {
    let flat: Vec<Complex64> = (0..16).map(|_| Complex64::new(0.0, 0.0)).collect();
    let mut matrix = flat;
    matrix[0] = Complex64::new(1.0, 0.0);
    matrix[5] = Complex64::new(1.0, 0.0);
    matrix[10] = Complex64::new(0.0, 0.0);
    matrix[11] = Complex64::new(1.0, 0.0);
    matrix[14] = Complex64::new(1.0, 0.0);
    matrix[15] = Complex64::new(0.0, 0.0);
    let gate = CustomGateBuilder::new("MyCNOT")
        .num_qubits(2)
        .matrix(matrix)
        .build()
        .unwrap();
    assert_eq!(gate.num_qubits(), 2);
}

#[test]
fn custom_gate_invalid_name() {
    let result = CustomGateBuilder::new("")
        .matrix_2x2([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ])
        .build();
    assert!(result.is_err());
}

#[test]
fn custom_gate_require_hermitian() {
    let result = CustomGateBuilder::new("NonHerm")
        .matrix_2x2([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
        ])
        .require_hermitian(true)
        .build();
    assert!(result.is_err());
}

// ============================================================================
// 9. ParametricCustomGate
// ============================================================================

#[test]
fn parametric_custom_gate_basic() {
    let gate = ParametricCustomGateBuilder::new("MyRX", 1)
        .with_parameters(vec!["theta"])
        .with_matrix_fn(|params| {
            let theta = params[0];
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();
            vec![
                Complex64::new(cos, 0.0),
                Complex64::new(0.0, -sin),
                Complex64::new(0.0, -sin),
                Complex64::new(cos, 0.0),
            ]
        })
        .with_initial_params(vec![0.0])
        .build()
        .unwrap();
    let m = gate.matrix_vec();
    assert_eq!(m.len(), 4);
    assert!(is_unitary(m, 1e-8));
}

#[test]
fn parametric_custom_gate_set_parameters() {
    let mut gate = ParametricCustomGateBuilder::new("MyRY", 1)
        .with_parameters(vec!["theta"])
        .with_matrix_fn(|params| {
            let theta = params[0];
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();
            vec![
                Complex64::new(cos, 0.0),
                Complex64::new(-sin, 0.0),
                Complex64::new(sin, 0.0),
                Complex64::new(cos, 0.0),
            ]
        })
        .with_initial_params(vec![0.0])
        .build()
        .unwrap();

    assert!(gate.set_parameters(vec![PI]).is_ok());
    let m = gate.matrix_vec();
    assert!(m[0].re.abs() < EPSILON, "cos(pi/2) should be ~0");
}

// ============================================================================
// 10. GateRegistry
// ============================================================================

#[test]
fn gate_registry_register_get() {
    let mut registry = GateRegistry::new();
    assert!(registry.is_empty());

    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let h = CustomGateBuilder::new("H")
        .matrix_2x2([
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
            ],
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        ])
        .build()
        .unwrap();
    registry.register("H", h);
    assert!(!registry.is_empty());
    assert_eq!(registry.len(), 1);
    assert!(registry.contains("H"));
    let retrieved = registry.get("H").unwrap();
    assert_eq!(retrieved.name(), "H");
}

#[test]
fn gate_registry_unregister() {
    let mut registry = GateRegistry::new();
    let gate = CustomGateBuilder::new("X")
        .matrix_2x2([
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ])
        .build()
        .unwrap();
    registry.register("X", gate);
    assert!(registry.contains("X"));
    let removed = registry.unregister("X");
    assert!(removed.is_some());
    assert!(!registry.contains("X"));
}

#[test]
fn gate_registry_gate_names() {
    let mut registry = GateRegistry::new();
    let i_gate = CustomGateBuilder::new("I")
        .matrix_2x2([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ])
        .build()
        .unwrap();
    let x_gate = CustomGateBuilder::new("X")
        .matrix_2x2([
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ])
        .build()
        .unwrap();
    registry.register("I", i_gate);
    registry.register("X", x_gate);
    let names = registry.gate_names();
    assert_eq!(names.len(), 2);
}

#[test]
fn gate_registry_gates_for_qubits() {
    let mut registry = GateRegistry::new();
    let one_q = CustomGateBuilder::new("H")
        .matrix_2x2([
            [
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            ],
            [
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
            ],
        ])
        .build()
        .unwrap();
    registry.register("H", one_q);
    let single_qubit_gates = registry.gates_for_qubits(1);
    assert_eq!(single_qubit_gates.len(), 1);
    let two_qubit_gates = registry.gates_for_qubits(2);
    assert_eq!(two_qubit_gates.len(), 0);
}

#[test]
fn gate_registry_list_gates() {
    let mut registry = GateRegistry::new();
    let gate = CustomGateBuilder::new("TestGate")
        .matrix_2x2([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ])
        .build()
        .unwrap();
    registry.register("TestGate", gate);
    let info = registry.list_gates();
    assert_eq!(info.len(), 1);
    assert_eq!(info[0].name, "TestGate");
    assert_eq!(info[0].num_qubits, 1);
}

#[test]
fn gate_registry_clear() {
    let mut registry = GateRegistry::new();
    let gate = CustomGateBuilder::new("G")
        .matrix_2x2([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ])
        .build()
        .unwrap();
    registry.register("G", gate);
    assert!(!registry.is_empty());
    registry.clear();
    assert!(registry.is_empty());
}

// ============================================================================
// 11. Matrix operations
// ============================================================================

#[test]
fn tensor_product_identity() {
    let i = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let result = tensor_product(&i, &i);
    assert_eq!(result.len(), 16);
    for idx in 0..4 {
        assert!((result[idx * 4 + idx] - Complex64::new(1.0, 0.0)).norm() < EPSILON);
    }
}

#[test]
fn matrix_multiply_identity() {
    let i = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let result = matrix_multiply(&i, &i);
    for idx in 0..2 {
        assert!((result[idx * 2 + idx] - Complex64::new(1.0, 0.0)).norm() < EPSILON);
    }
}

#[test]
fn matrix_adjoint_test() {
    let m = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
    ];
    let adj = matrix_adjoint(&m);
    assert!((adj[0] - Complex64::new(1.0, -2.0)).norm() < EPSILON);
    assert!((adj[1] - Complex64::new(5.0, -6.0)).norm() < EPSILON);
    assert!((adj[2] - Complex64::new(3.0, -4.0)).norm() < EPSILON);
    assert!((adj[3] - Complex64::new(7.0, -8.0)).norm() < EPSILON);
}

#[test]
fn matrix_trace_test() {
    let m = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(3.0, 0.0),
    ];
    let tr = matrix_trace(&m);
    assert!((tr - Complex64::new(4.0, 0.0)).norm() < EPSILON);
}

#[test]
fn is_unitary_check() {
    let h_flat = Hadamard.matrix().unwrap();
    assert!(is_unitary(&h_flat, 1e-8));

    let non_unitary = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    assert!(!is_unitary(&non_unitary, 1e-8));
}

#[test]
fn is_hermitian_check() {
    let h_flat = Hadamard.matrix().unwrap();
    assert!(is_hermitian(&h_flat, 1e-8));

    let s_flat = SGate.matrix().unwrap();
    assert!(!is_hermitian(&s_flat, 1e-8));
}

#[test]
fn matrix_to_vec_roundtrip() {
    let m = [
        [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        [Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
    ];
    let flat = matrix_to_vec(&m);
    assert_eq!(flat.len(), 4);
    let back = vec_to_matrix_2x2(&flat);
    for i in 0..2 {
        for j in 0..2 {
            assert!((back[i][j] - m[i][j]).norm() < EPSILON);
        }
    }
}

#[test]
fn embed_gate_matrix_single_qubit() {
    let h_flat = Hadamard.matrix().unwrap();
    let embedded = embed_gate_matrix_vec(&h_flat, 2, &[0]);
    assert_eq!(embedded.len(), 16);
    assert!(is_unitary(&embedded, 1e-8));
}

// ============================================================================
// 12. Compile-time cache
// ============================================================================

#[test]
fn common_angles_exact_match() {
    let pi2 = CommonAngles::rx_lookup(PI / 2.0);
    assert!(pi2.is_some());
    let m = pi2.unwrap();
    assert!(is_unitary_2x2(&m));

    let pi4 = CommonAngles::rz_lookup(PI / 4.0);
    assert!(pi4.is_some());
}

#[test]
fn common_angles_no_match() {
    let result = CommonAngles::rx_lookup(0.123456);
    assert!(result.is_none());
}

#[test]
fn vqe_angles_in_range() {
    let m = VQEAngles::rx_cached(0.1);
    assert!(is_unitary_2x2(&m));

    let m2 = VQEAngles::ry_cached(0.5);
    assert!(is_unitary_2x2(&m2));

    let m3 = VQEAngles::rz_cached(0.3);
    assert!(is_unitary_2x2(&m3));
}

// ============================================================================
// 13. Enhanced universal cache (multi-level fallback)
// ============================================================================

#[test]
fn enhanced_cache_common_angles() {
    let m = EnhancedUniversalCache::rx(PI / 2.0);
    assert!(is_unitary_2x2(&m));

    let m2 = EnhancedUniversalCache::ry(PI / 4.0);
    assert!(is_unitary_2x2(&m2));

    let m3 = EnhancedUniversalCache::rz(PI);
    assert!(is_unitary_2x2(&m3));
}

#[test]
fn enhanced_cache_arbitrary_angle() {
    let m = EnhancedUniversalCache::rx(0.12345);
    assert!(is_unitary_2x2(&m));
}

#[test]
fn enhanced_cache_matches_direct_computation() {
    let theta = 0.789;
    let cached = EnhancedUniversalCache::rx(theta);
    let direct = RotationX::new(theta).matrix_uncached();
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (cached[i][j] - direct[i][j]).norm() < 1e-2,
                "Cached RX should be close to direct computation (nearest-neighbor lookup)"
            );
        }
    }
    assert!(is_unitary_2x2(&cached), "Cached matrix must be unitary");
}

// ============================================================================
// 14. Generated angle cache
// ============================================================================

#[test]
fn generated_cache_clifford_t() {
    let result = GeneratedAngleCache::rx_clifford_t(PI / 4.0);
    assert!(result.is_some());
    assert!(is_unitary_2x2(&result.unwrap()));
}

#[test]
fn generated_cache_pi_fractions() {
    let result = GeneratedAngleCache::rx_pi_fraction(PI / 6.0);
    assert!(result.is_some());
}

// ============================================================================
// 15. Lookup tables
// ============================================================================

#[test]
fn lookup_table_creation() {
    use ferriq_gates::lookup::*;

    let config = LookupConfig::default();
    let table = RotationLookupTable::new(config);
    let stats = table.stats();
    assert!(stats.memory_bytes > 0);
}

#[test]
fn lookup_table_rx_accuracy() {
    use ferriq_gates::lookup::*;

    let config = LookupConfig::default();
    let table = RotationLookupTable::new(config);
    let theta = 0.3;
    let from_table = table.rx_matrix(theta);
    let direct = RotationX::new(theta).matrix_uncached();
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (from_table[i][j] - direct[i][j]).norm() < 1e-4,
                "Lookup table RX should be close to direct computation"
            );
        }
    }
}

// ============================================================================
// 16. Optimized rotation gates
// ============================================================================

#[test]
fn optimized_rotation_gates() {
    use ferriq_gates::optimized::*;

    let table = create_global_lookup_table();
    let orx = OptimizedRotationX::new(0.5, &table);
    assert_eq!(orx.name(), "RX");
    assert_eq!(orx.num_qubits(), 1);
    let m = orx.compute_matrix();
    let direct = RotationX::new(0.5).matrix_uncached();
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (m[i][j] - direct[i][j]).norm() < 1e-3,
                "Optimized RX should approximate direct computation"
            );
        }
    }

    let ory = OptimizedRotationY::new(0.5, &table);
    let direct_y = RotationY::new(0.5).matrix_uncached();
    let my = ory.compute_matrix();
    for i in 0..2 {
        for j in 0..2 {
            assert!((my[i][j] - direct_y[i][j]).norm() < 1e-3);
        }
    }

    let orz = OptimizedRotationZ::new(0.5, &table);
    let direct_z = RotationZ::new(0.5).matrix_uncached();
    let mz = orz.compute_matrix();
    for i in 0..2 {
        for j in 0..2 {
            assert!((mz[i][j] - direct_z[i][j]).norm() < 1e-3);
        }
    }
}

#[test]
fn optimized_compact_vs_high_precision() {
    use ferriq_gates::optimized::*;

    let compact = create_compact_table();
    let hq = create_high_precision_table();

    let theta = 0.3;
    let from_compact = OptimizedRotationX::new(theta, &compact).compute_matrix();
    let from_hq = OptimizedRotationX::new(theta, &hq).compute_matrix();
    let direct = RotationX::new(theta).matrix_uncached();

    let err_compact: f64 = (0..2)
        .flat_map(|i| (0..2).map(move |j| (from_compact[i][j] - direct[i][j]).norm()))
        .sum();
    let err_hq: f64 = (0..2)
        .flat_map(|i| (0..2).map(move |j| (from_hq[i][j] - direct[i][j]).norm()))
        .sum();

    assert!(err_hq <= err_compact + EPSILON, "High precision should be at least as accurate");
}

// ============================================================================
// 17. Circuit matrix computation
// ============================================================================

#[test]
fn circuit_matrix_single_gate() {
    use ferriq_core::{Circuit, QubitId};

    let mut circuit = Circuit::new(1);
    circuit
        .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
        .unwrap();
    let full = circuit_matrix(&circuit).unwrap();
    assert_eq!(full.len(), 4);
    assert!(is_unitary(&full, 1e-8));
}

#[test]
fn circuit_matrix_two_gate_sequence() {
    use ferriq_core::{Circuit, QubitId};

    let mut circuit = Circuit::new(1);
    circuit
        .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
        .unwrap();
    let full = circuit_matrix(&circuit).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            assert!((full[i * 2 + j] - expected).norm() < EPSILON, "H*H should be identity");
        }
    }
}

#[test]
fn circuit_matrix_multi_qubit() {
    use ferriq_core::{Circuit, QubitId};

    let mut circuit = Circuit::new(2);
    circuit
        .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
        .unwrap();
    let full = circuit_matrix(&circuit).unwrap();
    assert_eq!(full.len(), 16);
    assert!(is_unitary(&full, 1e-8));
}

// ============================================================================
// 18. Stress tests
// ============================================================================

#[test]
fn stress_many_rotation_angles() {
    for i in 0..100 {
        let theta = (i as f64) * PI / 50.0;
        let rx = RotationX::new(theta);
        assert!(is_unitary_2x2(&rx.matrix()), "RX({}) cached should be unitary", theta);
        assert!(
            is_unitary_2x2(&rx.matrix_uncached()),
            "RX({}) uncached should be unitary",
            theta
        );
        let ry = RotationY::new(theta);
        assert!(is_unitary_2x2(&ry.matrix()), "RY({}) cached should be unitary", theta);
        assert!(
            is_unitary_2x2(&ry.matrix_uncached()),
            "RY({}) uncached should be unitary",
            theta
        );
        let rz = RotationZ::new(theta);
        assert!(is_unitary_2x2(&rz.matrix()), "RZ({}) cached should be unitary", theta);
        assert!(
            is_unitary_2x2(&rz.matrix_uncached()),
            "RZ({}) uncached should be unitary",
            theta
        );
    }
}

#[test]
fn stress_enhanced_cache_many_angles() {
    for i in 0..200 {
        let theta = (i as f64) * 0.01;
        let m = EnhancedUniversalCache::rx(theta);
        assert!(is_unitary_2x2(&m));
    }
}
