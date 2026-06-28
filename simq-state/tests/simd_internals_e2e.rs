use approx::assert_relative_eq;
use num_complex::Complex64;
use simq_state::simd;

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

fn y_gate() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
        [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
    ]
}

fn z_gate() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
    ]
}

fn assert_states_equal(a: &[Complex64], b: &[Complex64], epsilon: f64) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert_relative_eq!(a[i].re, b[i].re, epsilon = epsilon);
        assert_relative_eq!(a[i].im, b[i].im, epsilon = epsilon);
    }
}

fn make_state(num_qubits: usize) -> Vec<Complex64> {
    let dim = 1 << num_qubits;
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    state[0] = Complex64::new(1.0, 0.0);
    state
}

fn make_plus_state(num_qubits: usize) -> Vec<Complex64> {
    let dim = 1 << num_qubits;
    let amp = 1.0 / (dim as f64).sqrt();
    vec![Complex64::new(amp, 0.0); dim]
}

// ============================================================
// SIMD single-qubit gate: scalar vs dispatched consistency
// ============================================================

#[test]
fn test_simd_single_qubit_hadamard_matches_scalar() {
    let h = hadamard();
    for num_qubits in 1..=6 {
        for qubit in 0..num_qubits {
            let mut state_scalar = make_state(num_qubits);
            let mut state_simd = state_scalar.clone();

            simq_state::simd::single_qubit::apply_gate_scalar(
                &mut state_scalar,
                &h,
                qubit,
                num_qubits,
            );
            simd::apply_single_qubit_gate(&mut state_simd, &h, qubit, num_qubits);

            assert_states_equal(&state_scalar, &state_simd, 1e-12);
        }
    }
}

#[test]
fn test_simd_single_qubit_all_gates_match_scalar() {
    let gates = [x_gate(), y_gate(), z_gate(), hadamard()];
    let num_qubits = 4;

    for gate in &gates {
        for qubit in 0..num_qubits {
            let mut state_scalar = make_plus_state(num_qubits);
            let mut state_simd = state_scalar.clone();

            simq_state::simd::single_qubit::apply_gate_scalar(
                &mut state_scalar,
                gate,
                qubit,
                num_qubits,
            );
            simd::apply_single_qubit_gate(&mut state_simd, gate, qubit, num_qubits);

            assert_states_equal(&state_scalar, &state_simd, 1e-12);
        }
    }
}

#[test]
fn test_simd_single_qubit_preserves_norm() {
    let gates = [x_gate(), y_gate(), z_gate(), hadamard()];
    let num_qubits = 5;

    for gate in &gates {
        let mut state = make_plus_state(num_qubits);
        simd::apply_single_qubit_gate(&mut state, gate, 2, num_qubits);
        let norm: f64 = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-12);
    }
}

// ============================================================
// SIMD two-qubit gate: scalar vs dispatched consistency
// ============================================================

fn cnot_matrix() -> [[Complex64; 4]; 4] {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    [
        [one, zero, zero, zero],
        [zero, one, zero, zero],
        [zero, zero, zero, one],
        [zero, zero, one, zero],
    ]
}

fn swap_matrix() -> [[Complex64; 4]; 4] {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    [
        [one, zero, zero, zero],
        [zero, zero, one, zero],
        [zero, one, zero, zero],
        [zero, zero, zero, one],
    ]
}

#[test]
fn test_simd_two_qubit_cnot_matches_scalar() {
    let cnot = cnot_matrix();
    for num_qubits in 2..=5 {
        for q1 in 0..num_qubits {
            for q2 in 0..num_qubits {
                if q1 == q2 {
                    continue;
                }
                let mut state_scalar = make_plus_state(num_qubits);
                let mut state_simd = state_scalar.clone();

                simq_state::simd::two_qubit::apply_gate_scalar(
                    &mut state_scalar,
                    &cnot,
                    q1,
                    q2,
                    num_qubits,
                );
                simd::apply_two_qubit_gate(&mut state_simd, &cnot, q1, q2, num_qubits);

                assert_states_equal(&state_scalar, &state_simd, 1e-12);
            }
        }
    }
}

#[test]
fn test_simd_two_qubit_swap_preserves_norm() {
    let swap = swap_matrix();
    let num_qubits = 4;
    let mut state = make_plus_state(num_qubits);
    state[0] = Complex64::new(0.8, 0.0);
    state[3] = Complex64::new(0.6, 0.0);

    let norm_before: f64 = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    simd::apply_two_qubit_gate(&mut state, &swap, 0, 2, num_qubits);
    let norm_after: f64 = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    assert_relative_eq!(norm_before, norm_after, epsilon = 1e-12);
}

// ============================================================
// SIMD controlled gates: CNOT, CZ, CRX, CRY, CRZ
// ============================================================

#[test]
fn test_cnot_creates_bell_state() {
    let num_qubits = 2;
    let mut state = make_state(num_qubits);
    simd::apply_single_qubit_gate(&mut state, &hadamard(), 0, num_qubits);
    simd::apply_cnot(&mut state, 0, 1, num_qubits);

    let h = FRAC_1_SQRT_2;
    assert_relative_eq!(state[0].re, h, epsilon = 1e-12);
    assert_relative_eq!(state[3].re, h, epsilon = 1e-12);
    assert_relative_eq!(state[1].norm(), 0.0, epsilon = 1e-12);
    assert_relative_eq!(state[2].norm(), 0.0, epsilon = 1e-12);
}

#[test]
fn test_cnot_scalar_vs_striped_consistency() {
    let num_qubits = 4;
    for ctrl in 0..num_qubits {
        for tgt in 0..num_qubits {
            if ctrl == tgt {
                continue;
            }
            let mut state_scalar = make_plus_state(num_qubits);
            let mut state_striped = state_scalar.clone();

            simq_state::simd::controlled_gates::apply_cnot_scalar(
                &mut state_scalar,
                ctrl,
                tgt,
                num_qubits,
            );
            simq_state::simd::controlled_gates::apply_cnot_striped(
                &mut state_striped,
                ctrl,
                tgt,
                num_qubits,
            );

            assert_states_equal(&state_scalar, &state_striped, 1e-12);
        }
    }
}

#[test]
fn test_cz_scalar_vs_striped_consistency() {
    let num_qubits = 4;
    for q1 in 0..num_qubits {
        for q2 in 0..num_qubits {
            if q1 == q2 {
                continue;
            }
            let mut state_scalar = make_plus_state(num_qubits);
            let mut state_striped = state_scalar.clone();

            simq_state::simd::controlled_gates::apply_cz_scalar(
                &mut state_scalar,
                q1,
                q2,
                num_qubits,
            );
            simq_state::simd::controlled_gates::apply_cz_striped(
                &mut state_striped,
                q1,
                q2,
                num_qubits,
            );

            assert_states_equal(&state_scalar, &state_striped, 1e-12);
        }
    }
}

#[test]
fn test_cz_only_flips_11_phase() {
    let num_qubits = 2;
    let mut state = make_plus_state(num_qubits);
    let original = state.clone();
    simd::apply_cz(&mut state, 0, 1, num_qubits);

    assert_relative_eq!(state[0].re, original[0].re, epsilon = 1e-12);
    assert_relative_eq!(state[1].re, original[1].re, epsilon = 1e-12);
    assert_relative_eq!(state[2].re, original[2].re, epsilon = 1e-12);
    assert_relative_eq!(state[3].re, -original[3].re, epsilon = 1e-12);
}

#[test]
fn test_controlled_u_scalar_vs_striped() {
    let u_matrix = hadamard();
    let num_qubits = 4;

    for ctrl in 0..num_qubits {
        for tgt in 0..num_qubits {
            if ctrl == tgt {
                continue;
            }
            let mut state_scalar = make_plus_state(num_qubits);
            let mut state_striped = state_scalar.clone();

            simq_state::simd::controlled_gates::apply_controlled_u_scalar(
                &mut state_scalar,
                ctrl,
                tgt,
                &u_matrix,
                num_qubits,
            );
            simq_state::simd::controlled_gates::apply_controlled_u_striped(
                &mut state_striped,
                ctrl,
                tgt,
                &u_matrix,
                num_qubits,
            );

            assert_states_equal(&state_scalar, &state_striped, 1e-12);
        }
    }
}

#[test]
fn test_crx_preserves_norm() {
    let num_qubits = 3;
    for theta in [0.0, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0] {
        let mut state = make_plus_state(num_qubits);
        simd::apply_crx(&mut state, 0, 1, theta, num_qubits);
        let norm: f64 = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-12);
    }
}

#[test]
fn test_cry_preserves_norm() {
    let num_qubits = 3;
    for theta in [0.0, PI / 4.0, PI / 2.0, PI] {
        let mut state = make_plus_state(num_qubits);
        simd::apply_cry(&mut state, 1, 2, theta, num_qubits);
        let norm: f64 = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-12);
    }
}

#[test]
fn test_crz_preserves_norm() {
    let num_qubits = 3;
    for theta in [0.0, PI / 4.0, PI / 2.0, PI] {
        let mut state = make_plus_state(num_qubits);
        simd::apply_crz(&mut state, 2, 0, theta, num_qubits);
        let norm: f64 = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-12);
    }
}

#[test]
fn test_crz_zero_angle_is_identity() {
    let num_qubits = 3;
    let state_original = make_plus_state(num_qubits);
    let mut state = state_original.clone();
    simd::apply_crz(&mut state, 0, 1, 0.0, num_qubits);
    assert_states_equal(&state, &state_original, 1e-12);
}

// ============================================================
// SIMD diagonal gate: consistency tests
// ============================================================

#[test]
fn test_diagonal_z_gate_matches_general_gate() {
    let num_qubits = 4;
    let z = z_gate();
    let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];

    for qubit in 0..num_qubits {
        let mut state_general = make_plus_state(num_qubits);
        let mut state_diag = state_general.clone();

        simd::apply_single_qubit_gate(&mut state_general, &z, qubit, num_qubits);
        simd::apply_diagonal_gate(&mut state_diag, z_diag, qubit, num_qubits);

        assert_states_equal(&state_general, &state_diag, 1e-12);
    }
}

#[test]
fn test_diagonal_s_gate_matches_general() {
    let num_qubits = 3;
    let s_matrix = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
    ];
    let s_diag = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];

    for qubit in 0..num_qubits {
        let mut state_general = make_plus_state(num_qubits);
        let mut state_diag = state_general.clone();

        simd::apply_single_qubit_gate(&mut state_general, &s_matrix, qubit, num_qubits);
        simd::apply_diagonal_gate(&mut state_diag, s_diag, qubit, num_qubits);

        assert_states_equal(&state_general, &state_diag, 1e-12);
    }
}

#[test]
fn test_diagonal_scalar_vs_optimized() {
    let num_qubits = 5;
    let t_diag = [
        Complex64::new(1.0, 0.0),
        Complex64::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2),
    ];

    for qubit in 0..num_qubits {
        let mut state_scalar = make_plus_state(num_qubits);
        let mut state_opt = state_scalar.clone();

        simq_state::simd::diagonal::apply_diagonal_gate_scalar(
            &mut state_scalar,
            t_diag,
            qubit,
            num_qubits,
        );
        simq_state::simd::diagonal::apply_diagonal_gate_optimized(
            &mut state_opt,
            t_diag,
            qubit,
            num_qubits,
        );

        assert_states_equal(&state_scalar, &state_opt, 1e-12);
    }
}

// ============================================================
// SIMD kernels: norm, normalize, probabilities
// ============================================================

#[test]
fn test_norm_simd_matches_manual() {
    let state = vec![
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
    ];
    let manual_norm = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    let simd_norm = simd::norm_simd(&state);
    assert_relative_eq!(simd_norm, manual_norm, epsilon = 1e-12);
}

#[test]
fn test_normalize_simd() {
    let mut state = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(0.0, 3.0),
    ];
    simd::normalize_simd(&mut state);
    let norm = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-12);
}

#[test]
fn test_compute_probabilities_scalar_vs_dispatch() {
    let amplitudes = vec![
        Complex64::new(0.5, 0.5),
        Complex64::new(0.3, -0.4),
        Complex64::new(0.0, 0.1),
        Complex64::new(-0.2, 0.3),
    ];

    let mut probs_scalar = vec![0.0; amplitudes.len()];
    let mut probs_dispatch = vec![0.0; amplitudes.len()];

    simq_state::simd::kernels::compute_probabilities_scalar(&amplitudes, &mut probs_scalar);
    simq_state::simd::kernels::compute_probabilities(&amplitudes, &mut probs_dispatch);

    for i in 0..amplitudes.len() {
        assert_relative_eq!(probs_scalar[i], probs_dispatch[i], epsilon = 1e-12);
    }
}

#[test]
fn test_compute_probabilities_sum_to_one_for_normalized_state() {
    let state = make_plus_state(4);
    let mut probs = vec![0.0; state.len()];
    simq_state::simd::kernels::compute_probabilities(&state, &mut probs);
    let sum: f64 = probs.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
}

// ============================================================
// Larger qubit SIMD tests (8 qubits)
// ============================================================

#[test]
fn test_simd_8_qubit_hadamard_all() {
    let num_qubits = 8;
    let h = hadamard();
    let mut state = make_state(num_qubits);

    for q in 0..num_qubits {
        simd::apply_single_qubit_gate(&mut state, &h, q, num_qubits);
    }

    let expected_amp = 1.0 / (256.0_f64).sqrt();
    for amp in &state {
        assert_relative_eq!(amp.re.abs(), expected_amp, epsilon = 1e-10);
    }
    let norm: f64 = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
}

#[test]
fn test_simd_cnot_chain_ghz_5_qubits() {
    let num_qubits = 5;
    let mut state = make_state(num_qubits);

    simd::apply_single_qubit_gate(&mut state, &hadamard(), 0, num_qubits);
    for q in 0..(num_qubits - 1) {
        simd::apply_cnot(&mut state, q, q + 1, num_qubits);
    }

    let h = FRAC_1_SQRT_2;
    assert_relative_eq!(state[0].re, h, epsilon = 1e-10);
    assert_relative_eq!(state[(1 << num_qubits) - 1].re, h, epsilon = 1e-10);
    for s in &state[1..((1 << num_qubits) - 1)] {
        assert_relative_eq!(s.norm(), 0.0, epsilon = 1e-10);
    }
}
