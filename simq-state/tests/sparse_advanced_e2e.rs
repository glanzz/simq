use approx::assert_relative_eq;
use num_complex::Complex64;
use simq_state::{DenseState, SparseState};
use std::f64::consts::FRAC_1_SQRT_2;

// ============================================================
// Partial traces with multiple qubits
// ============================================================

#[test]
fn test_sparse_partial_trace_4_qubit_keep_two() {
    let mut state = SparseState::new(4).unwrap();
    let val = FRAC_1_SQRT_2;
    state.set_amplitude(0, Complex64::new(val, 0.0)); // |0000⟩
    state.set_amplitude(15, Complex64::new(val, 0.0)); // |1111⟩

    let rho = state.partial_trace(&[0, 1]).unwrap();
    let dim = 4;
    let trace: f64 = (0..dim).map(|i| rho[i * dim + i].re).sum();
    assert_relative_eq!(trace, 1.0, epsilon = 1e-10);

    assert_relative_eq!(rho[0].re, 0.5, epsilon = 1e-10);
    assert_relative_eq!(rho[dim * (dim - 1) + (dim - 1)].re, 0.5, epsilon = 1e-10);
}

#[test]
fn test_sparse_partial_trace_keep_single_qubit_from_4() {
    let mut state = SparseState::new(4).unwrap();
    let val = FRAC_1_SQRT_2;
    state.set_amplitude(0, Complex64::new(val, 0.0));
    state.set_amplitude(15, Complex64::new(val, 0.0));

    for qubit in 0..4 {
        let rho = state.partial_trace(&[qubit]).unwrap();
        let trace = rho[0].re + rho[3].re;
        assert_relative_eq!(trace, 1.0, epsilon = 1e-10);
        assert_relative_eq!(rho[0].re, 0.5, epsilon = 1e-10);
        assert_relative_eq!(rho[3].re, 0.5, epsilon = 1e-10);
    }
}

// ============================================================
// Density threshold edge cases
// ============================================================

#[test]
fn test_sparse_density_exactly_at_threshold() {
    let mut state = SparseState::new(4).unwrap();
    state.set_density_threshold(0.125);

    assert_relative_eq!(state.density(), 0.0625, epsilon = 1e-3);
    assert!(!state.should_convert_to_dense());

    state.set_amplitude(1, Complex64::new(0.5, 0.0));
    assert_relative_eq!(state.density(), 0.125, epsilon = 1e-3);
    assert!(!state.should_convert_to_dense());

    state.set_amplitude(2, Complex64::new(0.5, 0.0));
    assert!(state.should_convert_to_dense());
}

#[test]
fn test_sparse_density_threshold_zero_always_converts() {
    let mut state = SparseState::new(3).unwrap();
    state.set_density_threshold(0.0);
    assert!(state.should_convert_to_dense());
}

#[test]
fn test_sparse_density_threshold_one_never_converts() {
    let mut state = SparseState::new(3).unwrap();
    state.set_density_threshold(1.0);
    for i in 0..8 {
        state.set_amplitude(i, Complex64::new(0.3, 0.0));
    }
    assert!(!state.should_convert_to_dense());
}

// ============================================================
// Large sparse state (10+ qubits)
// ============================================================

#[test]
fn test_sparse_10_qubit_few_amplitudes() {
    let mut state = SparseState::new(10).unwrap();
    assert_eq!(state.dimension(), 1024);
    assert_eq!(state.num_amplitudes(), 1);

    state.set_amplitude(512, Complex64::new(0.5, 0.0));
    state.set_amplitude(1023, Complex64::new(0.5, 0.0));
    assert_eq!(state.num_amplitudes(), 3);
    assert!(state.density() < 0.01);
    assert!(!state.should_convert_to_dense());
}

#[test]
fn test_sparse_15_qubit_creation_and_gate() {
    let mut state = SparseState::new(15).unwrap();
    assert_eq!(state.dimension(), 32768);

    let h_gate = [
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(-FRAC_1_SQRT_2, 0.0),
    ];
    state.apply_single_qubit_gate(&h_gate, 0).unwrap();
    assert_eq!(state.num_amplitudes(), 2);
    assert!(state.is_normalized(1e-10));
}

// ============================================================
// Conversion round-trips (sparse → dense → sparse)
// ============================================================

#[test]
fn test_sparse_dense_sparse_roundtrip_preserves_amplitudes() {
    let mut sparse1 = SparseState::new(3).unwrap();
    sparse1.set_amplitude(0, Complex64::new(0.6, 0.1));
    sparse1.set_amplitude(3, Complex64::new(0.0, 0.7));
    sparse1.set_amplitude(7, Complex64::new(0.3, -0.2));

    let dense = DenseState::from_sparse(&sparse1).unwrap();
    let sparse2 = dense.to_sparse().unwrap();

    assert_eq!(sparse2.num_amplitudes(), sparse1.num_amplitudes());
    for idx in [0, 3, 7] {
        let a1 = sparse1.get_amplitude(idx);
        let a2 = sparse2.get_amplitude(idx);
        assert_relative_eq!(a1.re, a2.re, epsilon = 1e-12);
        assert_relative_eq!(a1.im, a2.im, epsilon = 1e-12);
    }
}

#[test]
fn test_sparse_dense_sparse_with_gate_application() {
    let mut sparse = SparseState::new(3).unwrap();
    let h_gate = [
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(-FRAC_1_SQRT_2, 0.0),
    ];
    sparse.apply_single_qubit_gate(&h_gate, 0).unwrap();

    let dense = DenseState::from_sparse(&sparse).unwrap();
    let sparse2 = dense.to_sparse().unwrap();

    assert_eq!(sparse2.num_amplitudes(), 2);
    assert_relative_eq!(
        sparse2.get_amplitude(0).re,
        FRAC_1_SQRT_2,
        epsilon = 1e-10
    );
    assert_relative_eq!(
        sparse2.get_amplitude(1).re,
        FRAC_1_SQRT_2,
        epsilon = 1e-10
    );
}

// ============================================================
// SparseState gate operations
// ============================================================

#[test]
fn test_sparse_two_qubit_cnot_bell_state() {
    let mut state = SparseState::new(2).unwrap();
    let h_gate = [
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(-FRAC_1_SQRT_2, 0.0),
    ];
    state.apply_single_qubit_gate(&h_gate, 0).unwrap();

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let cnot = [
        one, zero, zero, zero, zero, one, zero, zero, zero, zero, zero, one, zero, zero, one,
        zero,
    ];
    state.apply_two_qubit_gate(&cnot, 0, 1).unwrap();

    assert_eq!(state.num_amplitudes(), 2);
    assert_relative_eq!(
        state.get_amplitude(0).norm_sqr(),
        0.5,
        epsilon = 1e-10
    );
    assert_relative_eq!(
        state.get_amplitude(3).norm_sqr(),
        0.5,
        epsilon = 1e-10
    );
}

#[test]
fn test_sparse_measure_and_collapse_renormalizes() {
    let mut state = SparseState::new(2).unwrap();
    state.set_amplitude(0, Complex64::new(FRAC_1_SQRT_2, 0.0));
    state.set_amplitude(2, Complex64::new(FRAC_1_SQRT_2, 0.0));

    let prob = state.measure_and_collapse(1, 0).unwrap();
    assert_relative_eq!(prob, 0.5, epsilon = 1e-10);
    assert_eq!(state.num_amplitudes(), 1);
    assert!(state.is_normalized(1e-10));
}

#[test]
fn test_sparse_set_amplitude_removes_near_zero() {
    let mut state = SparseState::new(2).unwrap();
    state.set_amplitude(0, Complex64::new(1.0, 0.0));
    state.set_amplitude(1, Complex64::new(0.5, 0.0));
    assert_eq!(state.num_amplitudes(), 2);

    state.set_amplitude(1, Complex64::new(1e-15, 0.0));
    assert_eq!(state.num_amplitudes(), 1);
}
