use approx::assert_relative_eq;
use num_complex::Complex64;
use simq_state::{DenseState, Pauli, PauliObservable, PauliString};
use std::f64::consts::FRAC_1_SQRT_2;

fn hadamard() -> [[Complex64; 2]; 2] {
    let h = FRAC_1_SQRT_2;
    [
        [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
        [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
    ]
}

fn x_gate() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ]
}

// ============================================================
// Multi-qubit Pauli string expectation values
// ============================================================

#[test]
fn test_xy_pauli_string_on_bell_state() {
    // Bell state (|00⟩ + |11⟩)/√2
    let amplitudes = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

    // XY on Bell state |Φ+⟩: ⟨Φ+|XY|Φ+⟩
    // XY|00⟩ = X|0⟩⊗Y|0⟩ = |1⟩⊗(i|1⟩) = i|11⟩
    // XY|11⟩ = X|1⟩⊗Y|1⟩ = |0⟩⊗(-i|0⟩) = -i|00⟩
    // XY on state: (i|11⟩ - i|00⟩)/√2
    // ⟨ψ|XY|ψ⟩ = (1/√2)⟨00|·(-i/√2)|00⟩ + (1/√2)⟨11|·(i/√2)|11⟩
    //           = -i/2 + i/2 = 0
    let xy = PauliString::from_str("XY").unwrap();
    let ev = xy.expectation_value(&state).unwrap();
    assert_relative_eq!(ev, 0.0, epsilon = 1e-10);
}

#[test]
fn test_yy_pauli_string_on_bell_state() {
    // Bell state (|00⟩ + |11⟩)/√2
    let amplitudes = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

    // YY on Bell |Φ+⟩: ⟨Φ+|YY|Φ+⟩ = -1
    let yy = PauliString::from_str("YY").unwrap();
    let ev = yy.expectation_value(&state).unwrap();
    assert_relative_eq!(ev, -1.0, epsilon = 1e-10);
}

#[test]
fn test_xyz_pauli_string_on_3_qubit_state() {
    // |000⟩ state
    let state = DenseState::new(3).unwrap();

    // XYZ on |000⟩ is non-diagonal, expect 0 for basis state
    let xyz = PauliString::from_str("XYZ").unwrap();
    let ev = xyz.expectation_value(&state).unwrap();
    assert_relative_eq!(ev, 0.0, epsilon = 1e-10);
}

#[test]
fn test_zzz_on_all_ones_state() {
    // |111⟩ state
    let mut state = DenseState::new(3).unwrap();
    state.apply_single_qubit_gate(&x_gate(), 0).unwrap();
    state.apply_single_qubit_gate(&x_gate(), 1).unwrap();
    state.apply_single_qubit_gate(&x_gate(), 2).unwrap();

    // ZZZ on |111⟩: eigenvalue = (-1)^3 = -1
    let zzz = PauliString::from_str("ZZZ").unwrap();
    let ev = zzz.expectation_value(&state).unwrap();
    assert_relative_eq!(ev, -1.0, epsilon = 1e-10);
}

#[test]
fn test_identity_string_always_one() {
    let state = DenseState::new(3).unwrap();
    let iii = PauliString::identity(3);
    let ev = iii.expectation_value(&state).unwrap();
    assert_relative_eq!(ev, 1.0, epsilon = 1e-10);

    // Also for superposition state
    let mut state2 = DenseState::new(2).unwrap();
    state2.apply_single_qubit_gate(&hadamard(), 0).unwrap();
    let ii = PauliString::identity(2);
    let ev2 = ii.expectation_value(&state2).unwrap();
    assert_relative_eq!(ev2, 1.0, epsilon = 1e-10);
}

#[test]
fn test_pauli_string_with_negative_coeff() {
    let state = DenseState::new(1).unwrap();
    let neg_z = PauliString::from_str("Z").unwrap().with_coeff(-1);
    let ev = neg_z.expectation_value(&state).unwrap();
    assert_relative_eq!(ev, -1.0, epsilon = 1e-10);
}

// ============================================================
// Observable algebra (weighted sums)
// ============================================================

#[test]
fn test_observable_linear_combination() {
    // |+⟩ state
    let amplitudes = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    let state = DenseState::from_amplitudes(1, &amplitudes).unwrap();

    // H = 0.5*X + 0.3*Z + 0.2*I
    // ⟨+|X|+⟩ = 1, ⟨+|Z|+⟩ = 0, ⟨+|I|+⟩ = 1
    // Expect: 0.5*1 + 0.3*0 + 0.2*1 = 0.7
    let mut obs = PauliObservable::new();
    obs.add_term(PauliString::from_str("X").unwrap(), 0.5);
    obs.add_term(PauliString::from_str("Z").unwrap(), 0.3);
    obs.add_term(PauliString::from_str("I").unwrap(), 0.2);

    let ev = obs.expectation_value(&state).unwrap();
    assert_relative_eq!(ev, 0.7, epsilon = 1e-10);
}

#[test]
fn test_observable_scalar_multiplication_via_coeffs() {
    let state = DenseState::new(1).unwrap();

    // 3.0 * Z on |0⟩ -> 3.0
    let obs = PauliObservable::from_pauli_string(
        PauliString::from_str("Z").unwrap(),
        3.0,
    );
    let ev = obs.expectation_value(&state).unwrap();
    assert_relative_eq!(ev, 3.0, epsilon = 1e-10);
}

#[test]
fn test_observable_two_qubit_heisenberg_term() {
    // Bell state (|00⟩ + |11⟩)/√2
    let amplitudes = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

    // Heisenberg term: XX + YY + ZZ
    // For Bell |Φ+⟩: ⟨XX⟩ = 1, ⟨YY⟩ = -1, ⟨ZZ⟩ = 1
    // Total = 1 + (-1) + 1 = 1
    let mut obs = PauliObservable::new();
    obs.add_term(PauliString::from_str("XX").unwrap(), 1.0);
    obs.add_term(PauliString::from_str("YY").unwrap(), 1.0);
    obs.add_term(PauliString::from_str("ZZ").unwrap(), 1.0);

    let ev = obs.expectation_value(&state).unwrap();
    assert_relative_eq!(ev, 1.0, epsilon = 1e-10);
}

#[test]
fn test_observable_is_diagonal_check() {
    let mut diag_obs = PauliObservable::new();
    diag_obs.add_term(PauliString::from_str("ZI").unwrap(), 1.0);
    diag_obs.add_term(PauliString::from_str("IZ").unwrap(), 1.0);
    assert!(diag_obs.is_diagonal());

    let mut non_diag_obs = PauliObservable::new();
    non_diag_obs.add_term(PauliString::from_str("XX").unwrap(), 1.0);
    non_diag_obs.add_term(PauliString::from_str("ZZ").unwrap(), 1.0);
    assert!(!non_diag_obs.is_diagonal());
}

#[test]
fn test_observable_empty_gives_zero() {
    let state = DenseState::new(2).unwrap();
    let obs = PauliObservable::new();
    let ev = obs.expectation_value(&state).unwrap();
    assert_relative_eq!(ev, 0.0, epsilon = 1e-10);
}

#[test]
fn test_pauli_string_from_paulis_matches_from_str() {
    let from_str = PauliString::from_str("XYZIZ").unwrap();
    let from_vec = PauliString::from_paulis(vec![
        Pauli::X, Pauli::Y, Pauli::Z, Pauli::I, Pauli::Z,
    ]);

    let state = DenseState::new(5).unwrap();
    let ev1 = from_str.expectation_value(&state).unwrap();
    let ev2 = from_vec.expectation_value(&state).unwrap();
    assert_relative_eq!(ev1, ev2, epsilon = 1e-10);
}

#[test]
fn test_single_z_observable_on_flipped_qubit() {
    let mut state = DenseState::new(3).unwrap();
    state.apply_single_qubit_gate(&x_gate(), 1).unwrap();

    // Z on qubit 1 should give -1 (flipped)
    let obs = PauliObservable::single_z(3, 1);
    let ev = obs.expectation_value(&state).unwrap();
    assert_relative_eq!(ev, -1.0, epsilon = 1e-10);

    // Z on qubit 0 should give +1 (not flipped)
    let obs0 = PauliObservable::single_z(3, 0);
    let ev0 = obs0.expectation_value(&state).unwrap();
    assert_relative_eq!(ev0, 1.0, epsilon = 1e-10);
}
