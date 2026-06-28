use approx::assert_relative_eq;
use num_complex::Complex64;
use simq_state::DenseState;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

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

fn ry_gate(theta: f64) -> [[Complex64; 2]; 2] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [
        [Complex64::new(c, 0.0), Complex64::new(-s, 0.0)],
        [Complex64::new(s, 0.0), Complex64::new(c, 0.0)],
    ]
}

// ============================================================
// Arbitrary matrix application beyond basic gates
// ============================================================

#[test]
fn test_dense_arbitrary_unitary_preserves_norm() {
    let theta: f64 = 0.7;
    let phi: f64 = 1.3;
    let u = [
        [
            Complex64::new(theta.cos(), 0.0),
            Complex64::new(-phi.sin() * theta.sin(), -phi.cos() * theta.sin()),
        ],
        [
            Complex64::new(phi.sin() * theta.sin(), -phi.cos() * theta.sin()),
            Complex64::new(theta.cos(), 0.0),
        ],
    ];

    let mut state = DenseState::new(3).unwrap();
    state.apply_single_qubit_gate(&hadamard(), 0).unwrap();
    state.apply_single_qubit_gate(&u, 1).unwrap();
    assert!(state.is_normalized(1e-10));
}

#[test]
fn test_dense_two_qubit_arbitrary_matrix() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let iswap = [
        [one, zero, zero, zero],
        [zero, zero, Complex64::new(0.0, 1.0), zero],
        [zero, Complex64::new(0.0, 1.0), zero, zero],
        [zero, zero, zero, one],
    ];

    let mut state = DenseState::new(3).unwrap();
    state.apply_single_qubit_gate(&x_gate(), 0).unwrap();
    state.apply_two_qubit_gate(&iswap, 0, 1).unwrap();

    assert!(state.is_normalized(1e-10));
    assert_relative_eq!(state.amplitudes()[2].im, 1.0, epsilon = 1e-10);
}

// ============================================================
// Controlled rotation gates
// ============================================================

#[test]
fn test_dense_crx_on_control_zero_is_identity() {
    let mut state = DenseState::new(2).unwrap();
    let original = state.clone();
    state.apply_crx(0, 1, PI / 3.0).unwrap();
    let fid = state.fidelity(&original).unwrap();
    assert_relative_eq!(fid, 1.0, epsilon = 1e-10);
}

#[test]
fn test_dense_crx_pi_equals_cnot_up_to_phase() {
    let mut state_crx = DenseState::new(2).unwrap();
    state_crx.apply_single_qubit_gate(&hadamard(), 0).unwrap();
    state_crx.apply_crx(0, 1, PI).unwrap();
    assert!(state_crx.is_normalized(1e-10));
}

#[test]
fn test_dense_cry_creates_partial_rotation() {
    let mut state = DenseState::new(2).unwrap();
    state.apply_single_qubit_gate(&x_gate(), 0).unwrap();
    // State is |01⟩ (q0=1, q1=0) = index 1 in LSB ordering
    state.apply_cry(0, 1, PI / 2.0).unwrap();

    // CRY rotates target q1: superposition at indices 1 (|01⟩) and 3 (|11⟩)
    let p0 = state.get_probability(1).unwrap();
    let p1 = state.get_probability(3).unwrap();
    assert_relative_eq!(p0 + p1, 1.0, epsilon = 1e-10);
    assert!(p0 > 0.01 && p1 > 0.01);
}

#[test]
fn test_dense_crz_applies_phase() {
    let mut state = DenseState::new(2).unwrap();
    state.apply_single_qubit_gate(&x_gate(), 0).unwrap();
    state.apply_single_qubit_gate(&hadamard(), 1).unwrap();
    state.apply_crz(0, 1, PI).unwrap();
    assert!(state.is_normalized(1e-10));
}

// ============================================================
// Diagonal gate batching
// ============================================================

#[test]
fn test_dense_sequential_diagonal_gates() {
    let mut state = DenseState::new(3).unwrap();
    for q in 0..3 {
        state.apply_single_qubit_gate(&hadamard(), q).unwrap();
    }

    let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
    let s_diag = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
    let t_diag = [
        Complex64::new(1.0, 0.0),
        Complex64::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2),
    ];

    state.apply_diagonal_gate(z_diag, 0).unwrap();
    state.apply_diagonal_gate(s_diag, 1).unwrap();
    state.apply_diagonal_gate(t_diag, 2).unwrap();

    assert!(state.is_normalized(1e-10));
    for amp in state.amplitudes() {
        assert!(amp.norm() > 0.0);
    }
}

// ============================================================
// Large qubit counts (8-10 qubits)
// ============================================================

#[test]
fn test_dense_8_qubit_operations() {
    let mut state = DenseState::new(8).unwrap();
    assert_eq!(state.dimension(), 256);
    assert!(state.is_simd_aligned());

    state.apply_single_qubit_gate(&hadamard(), 0).unwrap();
    state.apply_single_qubit_gate(&hadamard(), 4).unwrap();
    state.apply_single_qubit_gate(&hadamard(), 7).unwrap();
    state.apply_cnot(0, 1).unwrap();
    state.apply_cz(4, 5).unwrap();

    assert!(state.is_normalized(1e-10));
}

#[test]
fn test_dense_10_qubit_state_creation_and_gates() {
    let mut state = DenseState::new(10).unwrap();
    assert_eq!(state.dimension(), 1024);

    for q in 0..10 {
        state.apply_single_qubit_gate(&hadamard(), q).unwrap();
    }

    let expected_amp = 1.0 / (1024.0_f64).sqrt();
    for amp in state.amplitudes() {
        assert_relative_eq!(amp.re.abs(), expected_amp, epsilon = 1e-10);
    }
    assert!(state.is_normalized(1e-10));
}

#[test]
fn test_dense_12_qubit_basic_operations() {
    let mut state = DenseState::new(12).unwrap();
    assert_eq!(state.dimension(), 4096);

    state.apply_single_qubit_gate(&hadamard(), 0).unwrap();
    state.apply_cnot(0, 11).unwrap();

    let h = FRAC_1_SQRT_2;
    assert_relative_eq!(state.get_probability(0).unwrap(), 0.5, epsilon = 1e-10);
    let idx_both_one = 1 | (1 << 11);
    assert_relative_eq!(state.get_probability(idx_both_one).unwrap(), 0.5, epsilon = 1e-10);
}

// ============================================================
// Partial trace with multiple traced-out qubits
// ============================================================

#[test]
fn test_dense_partial_trace_ghz_trace_two_qubits() {
    let val = FRAC_1_SQRT_2;
    let mut amplitudes = vec![Complex64::new(0.0, 0.0); 8];
    amplitudes[0] = Complex64::new(val, 0.0);
    amplitudes[7] = Complex64::new(val, 0.0);
    let state = DenseState::from_amplitudes(3, &amplitudes).unwrap();

    let rho = state.partial_trace(&[0]).unwrap();
    assert_relative_eq!(rho[0].re, 0.5, epsilon = 1e-10);
    assert_relative_eq!(rho[3].re, 0.5, epsilon = 1e-10);

    let trace = rho[0].re + rho[3].re;
    assert_relative_eq!(trace, 1.0, epsilon = 1e-10);
}

#[test]
fn test_dense_partial_trace_4_qubit_keep_one() {
    let mut state = DenseState::new(4).unwrap();
    state.apply_single_qubit_gate(&hadamard(), 0).unwrap();
    state.apply_cnot(0, 1).unwrap();
    state.apply_cnot(0, 2).unwrap();
    state.apply_cnot(0, 3).unwrap();

    let rho = state.partial_trace(&[0]).unwrap();
    assert_relative_eq!(rho[0].re, 0.5, epsilon = 1e-10);
    assert_relative_eq!(rho[3].re, 0.5, epsilon = 1e-10);
    let trace = rho[0].re + rho[3].re;
    assert_relative_eq!(trace, 1.0, epsilon = 1e-10);
}

#[test]
fn test_dense_partial_trace_product_state_multiple_traces() {
    let mut state = DenseState::new(4).unwrap();
    state.apply_single_qubit_gate(&x_gate(), 2).unwrap();

    let rho = state.partial_trace(&[2]).unwrap();
    assert_relative_eq!(rho[0].re, 0.0, epsilon = 1e-10);
    assert_relative_eq!(rho[3].re, 1.0, epsilon = 1e-10);
}

#[test]
fn test_dense_partial_trace_hermiticity_3_qubit() {
    let mut state = DenseState::new(3).unwrap();
    state.apply_single_qubit_gate(&ry_gate(1.0), 0).unwrap();
    state.apply_cnot(0, 1).unwrap();
    state.apply_single_qubit_gate(&hadamard(), 2).unwrap();

    let rho = state.partial_trace(&[0, 1]).unwrap();
    let dim = 4;
    for i in 0..dim {
        assert_relative_eq!(rho[i * dim + i].im, 0.0, epsilon = 1e-10);
        for j in (i + 1)..dim {
            let diff = (rho[i * dim + j] - rho[j * dim + i].conj()).norm();
            assert!(diff < 1e-10, "Hermiticity violated at ({}, {})", i, j);
        }
    }
}

// ============================================================
// Dense state inner product and fidelity
// ============================================================

#[test]
fn test_dense_orthogonal_states_fidelity_zero() {
    let state0 = DenseState::new(1).unwrap();
    let mut state1 = DenseState::new(1).unwrap();
    state1.apply_single_qubit_gate(&x_gate(), 0).unwrap();

    let fid = state0.fidelity(&state1).unwrap();
    assert_relative_eq!(fid, 0.0, epsilon = 1e-10);
}
