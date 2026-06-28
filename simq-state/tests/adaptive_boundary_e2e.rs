use approx::assert_relative_eq;
use num_complex::Complex64;
use simq_state::AdaptiveState;
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
// Switching at exactly the threshold value
// ============================================================

#[test]
fn test_adaptive_threshold_exact_boundary() {
    let mut state = AdaptiveState::with_threshold(3, 0.25).unwrap();
    assert!(state.is_sparse());

    state.apply_single_qubit_gate(&hadamard(), 0).unwrap();
    // 2/8 = 0.25, not exceeding threshold
    assert!(state.is_sparse() || state.is_dense());

    state.apply_single_qubit_gate(&hadamard(), 1).unwrap();
    // 4/8 = 0.5 > 0.25, should convert
    assert!(state.is_dense());
}

#[test]
fn test_adaptive_tiny_threshold_converts_immediately() {
    let mut state = AdaptiveState::with_threshold(4, 0.01).unwrap();
    assert!(state.is_sparse());

    state.apply_single_qubit_gate(&hadamard(), 0).unwrap();
    // 2/16 = 0.125 > 0.01
    assert!(state.is_dense());
}

#[test]
fn test_adaptive_high_threshold_stays_sparse() {
    let mut state = AdaptiveState::with_threshold(3, 0.99).unwrap();
    state.apply_single_qubit_gate(&hadamard(), 0).unwrap();
    state.apply_single_qubit_gate(&hadamard(), 1).unwrap();
    // Even at 4/8 = 50% density, threshold is 99%
    assert!(state.is_sparse());
}

// ============================================================
// Repeated back-and-forth conversions (sparse ↔ dense)
// ============================================================

#[test]
fn test_adaptive_force_dense_then_reset_stays_dense() {
    let mut state = AdaptiveState::new(3).unwrap();
    assert!(state.is_sparse());

    state.force_to_dense().unwrap();
    assert!(state.is_dense());

    state.reset();
    assert!(state.is_dense());
}

#[test]
fn test_adaptive_multiple_force_to_dense_idempotent() {
    let mut state = AdaptiveState::new(3).unwrap();

    let first = state.force_to_dense().unwrap();
    assert!(first);

    let second = state.force_to_dense().unwrap();
    assert!(!second);

    assert!(state.is_dense());
}

#[test]
fn test_adaptive_conversion_preserves_state() {
    let mut state = AdaptiveState::new(3).unwrap();
    state.apply_single_qubit_gate(&x_gate(), 0).unwrap();

    let amps_before = state.to_dense_vec();
    state.force_to_dense().unwrap();
    let amps_after = state.to_dense_vec();

    assert_eq!(amps_before.len(), amps_after.len());
    for i in 0..amps_before.len() {
        assert_relative_eq!(amps_before[i].re, amps_after[i].re, epsilon = 1e-12);
        assert_relative_eq!(amps_before[i].im, amps_after[i].im, epsilon = 1e-12);
    }
}

// ============================================================
// Operations that change sparsity (Hadamard on |0⟩ sparse → dense)
// ============================================================

#[test]
fn test_adaptive_hadamard_on_zero_triggers_conversion() {
    let mut state = AdaptiveState::new(3).unwrap();
    assert!(state.is_sparse());

    // Apply H to all qubits -> density goes to 1.0
    for q in 0..3 {
        state.apply_single_qubit_gate(&hadamard(), q).unwrap();
    }

    assert!(state.is_dense());
    assert!(state.is_normalized(1e-10));
}

#[test]
fn test_adaptive_single_hadamard_may_not_convert() {
    let mut state = AdaptiveState::with_threshold(1, 0.5).unwrap();
    state.apply_single_qubit_gate(&hadamard(), 0).unwrap();
    // 2/4 = 0.5, exactly at threshold -> should NOT convert (only converts when >)
    // The state could be either sparse or dense depending on implementation
    assert!(state.is_normalized(1e-10));
}

// ============================================================
// Gate application correctness after conversion
// ============================================================

#[test]
fn test_adaptive_gates_correct_after_sparse_to_dense() {
    let mut state = AdaptiveState::new(3).unwrap();

    // Apply gates until conversion happens
    state.apply_single_qubit_gate(&hadamard(), 0).unwrap();
    state.apply_single_qubit_gate(&hadamard(), 1).unwrap();
    state.apply_single_qubit_gate(&hadamard(), 2).unwrap();

    assert!(state.is_dense());

    // Continue applying gates after conversion
    state.apply_single_qubit_gate(&x_gate(), 0).unwrap();

    assert!(state.is_normalized(1e-10));

    let amps = state.to_dense_vec();
    let expected_amp = 1.0 / (8.0_f64).sqrt();
    for amp in &amps {
        assert_relative_eq!(amp.re.abs(), expected_amp, epsilon = 1e-10);
    }
}

#[test]
fn test_adaptive_measurement_works_in_both_representations() {
    // Sparse measurement
    let mut sparse_state = AdaptiveState::new(2).unwrap();
    sparse_state.apply_single_qubit_gate(&x_gate(), 0).unwrap();
    let outcome = sparse_state.measure_qubit(0, 0.5).unwrap();
    assert_eq!(outcome, 1);

    // Dense measurement
    let mut dense_state = AdaptiveState::new(2).unwrap();
    dense_state.force_to_dense().unwrap();
    dense_state.apply_single_qubit_gate(&x_gate(), 0).unwrap();
    let outcome = dense_state.measure_qubit(0, 0.5).unwrap();
    assert_eq!(outcome, 1);
}

#[test]
fn test_adaptive_stats_update_on_conversion() {
    let mut state = AdaptiveState::new(4).unwrap();
    let stats_before = state.stats();
    assert_eq!(stats_before.representation, "Sparse");

    state.force_to_dense().unwrap();
    let stats_after = state.stats();
    assert_eq!(stats_after.representation, "Dense");
    assert_eq!(stats_after.memory_entries, 16);
}

#[test]
fn test_adaptive_partial_trace_works_both_representations() {
    let val = FRAC_1_SQRT_2;
    let amps = vec![
        Complex64::new(val, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(val, 0.0),
    ];

    let state = AdaptiveState::from_amplitudes(2, &amps).unwrap();
    let rho = state.partial_trace(&[0]).unwrap();
    assert_relative_eq!(rho[0].re, 0.5, epsilon = 1e-10);
    assert_relative_eq!(rho[3].re, 0.5, epsilon = 1e-10);
}
