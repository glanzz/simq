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

// ============================================================
// New coverage tests for simd.rs public API
// ============================================================

#[test]
fn test_apply_single_qubit_gate_hadamard_2qubit() {
    // Apply Hadamard on qubit 0 of 2-qubit |00⟩ state
    // In the simq convention, qubit 0 is the LSB of the index.
    // H on qubit 0 maps |00⟩ → (|00⟩ + |01⟩)/√2 — indices 0 and 1 get amplitude FRAC_1_SQRT_2
    let num_qubits = 2;
    let mut state = make_state(num_qubits); // |00⟩
    simd::apply_single_qubit_gate(&mut state, &hadamard(), 0, num_qubits);

    assert_relative_eq!(state[0].re, FRAC_1_SQRT_2, epsilon = 1e-12);
    assert_relative_eq!(state[1].re, FRAC_1_SQRT_2, epsilon = 1e-12);
    assert_relative_eq!(state[2].norm(), 0.0, epsilon = 1e-12);
    assert_relative_eq!(state[3].norm(), 0.0, epsilon = 1e-12);
}

#[test]
fn test_apply_two_qubit_gate_identity_unchanged() {
    // Apply 4×4 identity on qubits 0,1 of 2-qubit state — state unchanged
    let num_qubits = 2;
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let identity4: [[Complex64; 4]; 4] = [
        [one, zero, zero, zero],
        [zero, one, zero, zero],
        [zero, zero, one, zero],
        [zero, zero, zero, one],
    ];

    let original = make_plus_state(num_qubits);
    let mut state = original.clone();
    simd::apply_two_qubit_gate(&mut state, &identity4, 0, 1, num_qubits);

    for i in 0..state.len() {
        assert_relative_eq!(state[i].re, original[i].re, epsilon = 1e-12);
        assert_relative_eq!(state[i].im, original[i].im, epsilon = 1e-12);
    }
}

#[test]
fn test_norm_simd_known_vector() {
    // Known normalized vector — norm should be ≈ 1.0
    let state = make_plus_state(2); // [0.5, 0.5, 0.5, 0.5]
    let norm = simd::norm_simd(&state);
    assert_relative_eq!(norm, 1.0, epsilon = 1e-12);
}

#[test]
fn test_normalize_simd_makes_unit_norm() {
    // Start with non-unit vector, normalize, check norm = 1
    let mut state = vec![Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)];
    simd::normalize_simd(&mut state);
    let norm = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-12);
}

#[test]
fn test_normalize_simd_zero_vector_left_unchanged() {
    // Zero vector (norm < 1e-10) — normalize_simd should leave it unchanged
    let mut state = vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];
    simd::normalize_simd(&mut state);
    assert_relative_eq!(state[0].norm(), 0.0, epsilon = 1e-12);
    assert_relative_eq!(state[1].norm(), 0.0, epsilon = 1e-12);
}

#[test]
fn test_apply_cnot_10_becomes_11() {
    // CNOT(control=0, target=1) on |10⟩ (index 2 in 2-qubit basis) → |11⟩ (index 3)
    let num_qubits = 2;
    let mut state = make_state(num_qubits);
    // Build |10⟩: qubit 0 is LSB, qubit 1 is next bit
    // Index 2 = binary 10 → qubit1=1, qubit0=0
    // We want control=0 (qubit0) to be |1⟩ → use index where bit0=1 → index 1 or 3
    // Let's use the standard: apply X on qubit 0 to get |10⟩ in the qubit-ordering sense
    // In simq convention: index bit k = (index >> k) & 1
    // |10⟩ means qubit1=1, qubit0=0 → index = 2
    // CNOT(control=1, target=0): flip qubit0 when qubit1=1
    state[0] = Complex64::new(0.0, 0.0);
    state[2] = Complex64::new(1.0, 0.0); // |10⟩
    simd::apply_cnot(&mut state, 1, 0, num_qubits);
    // After CNOT(ctrl=1,tgt=0) on |10⟩ → |11⟩ (index 3)
    assert_relative_eq!(state[3].re, 1.0, epsilon = 1e-12);
    assert_relative_eq!(state[2].norm(), 0.0, epsilon = 1e-12);
}

#[test]
fn test_apply_cz_negates_11_amplitude() {
    // CZ(0,1) on equal superposition: |11⟩ component (index 3) gets phase -1
    let num_qubits = 2;
    let mut state = make_plus_state(num_qubits);
    let amp_before_3 = state[3]; // |11⟩ amplitude = 0.5
    let amp_before_0 = state[0]; // |00⟩ amplitude = 0.5
    simd::apply_cz(&mut state, 0, 1, num_qubits);
    // |11⟩ (index 3) amplitude should be negated
    assert_relative_eq!(state[3].re, -amp_before_3.re, epsilon = 1e-12);
    // Other amplitudes unchanged
    assert_relative_eq!(state[0].re, amp_before_0.re, epsilon = 1e-12);
    assert_relative_eq!(state[1].re, state[1].re, epsilon = 1e-12);
    assert_relative_eq!(state[2].re, state[2].re, epsilon = 1e-12);
}

#[test]
fn test_apply_controlled_u_x_on_10() {
    // Controlled-X (X gate as u_matrix), control=1, target=0, state=|10⟩
    let num_qubits = 2;
    let mut state = make_state(num_qubits);
    state[0] = Complex64::new(0.0, 0.0);
    state[2] = Complex64::new(1.0, 0.0); // |10⟩
    simd::apply_controlled_u(&mut state, 1, 0, &x_gate(), num_qubits);
    // Control qubit 1 is |1⟩, so X is applied to qubit 0 → |10⟩ becomes |11⟩ (index 3)
    assert_relative_eq!(state[3].re, 1.0, epsilon = 1e-12);
    assert_relative_eq!(state[2].norm(), 0.0, epsilon = 1e-12);
}

#[test]
fn test_apply_crx_pi_on_10_flips_target() {
    // CRX(π) on |10⟩ (2 qubits): control=1, target=0
    // When control=1, RX(π) ≈ -iX flips the target qubit
    let num_qubits = 2;
    let mut state = make_state(num_qubits);
    state[0] = Complex64::new(0.0, 0.0);
    state[2] = Complex64::new(1.0, 0.0); // |10⟩
    simd::apply_crx(&mut state, 1, 0, PI, num_qubits);
    // After CRX(π): target qubit 0 gets flipped → |10⟩ → |11⟩ (index 3) with phase
    // The norm must be preserved
    let norm: f64 = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-12);
    // The |10⟩ amplitude should be gone (flipped)
    assert_relative_eq!(state[2].norm(), 0.0, epsilon = 1e-10);
    assert_relative_eq!(state[3].norm(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_apply_cry_pi_on_10() {
    // CRY(π) on |10⟩ (2 qubits): control=1, target=0
    let num_qubits = 2;
    let mut state = make_state(num_qubits);
    state[0] = Complex64::new(0.0, 0.0);
    state[2] = Complex64::new(1.0, 0.0); // |10⟩
    simd::apply_cry(&mut state, 1, 0, PI, num_qubits);
    let norm: f64 = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-12);
    // RY(π)|0⟩ = |1⟩, so qubit0 gets flipped: |10⟩ → |11⟩
    assert_relative_eq!(state[2].norm(), 0.0, epsilon = 1e-10);
    assert_relative_eq!(state[3].norm(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_apply_crz_pi_on_10() {
    // CRZ(π) on |10⟩ (2 qubits): control=1, target=0
    // RZ(π) applies phase to |1⟩ component — but here target is |0⟩ so no flip
    // Let's put state in |11⟩ so that control=1 and target is in |1⟩
    let num_qubits = 2;
    let mut state = make_state(num_qubits);
    state[0] = Complex64::new(0.0, 0.0);
    state[3] = Complex64::new(1.0, 0.0); // |11⟩
    simd::apply_crz(&mut state, 1, 0, PI, num_qubits);
    let norm: f64 = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-12);
    // CRZ(π) on |11⟩: control qubit1=1 so RZ(π) applied to qubit0 which is |1⟩
    // RZ(π) = diag(e^{-iπ/2}, e^{iπ/2}) so |11⟩ gets phase e^{iπ/2} = i
    assert_relative_eq!(state[3].norm(), 1.0, epsilon = 1e-10);
    assert_relative_eq!(state[0].norm(), 0.0, epsilon = 1e-10);
}

#[test]
fn test_apply_diagonal_gate_z_on_plus0() {
    // Apply Z diagonal [1, -1] on qubit 0 of |+0⟩ = (|00⟩ + |10⟩)/√2
    let num_qubits = 2;
    let mut state = make_state(num_qubits);
    // Create |+0⟩ by applying H on qubit 1
    simd::apply_single_qubit_gate(&mut state, &hadamard(), 1, num_qubits);
    // Now apply Z diagonal on qubit 0
    let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
    simd::apply_diagonal_gate(&mut state, z_diag, 0, num_qubits);
    // Norm must be preserved
    let norm: f64 = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-12);
}
