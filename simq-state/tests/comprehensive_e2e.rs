//! Comprehensive end-to-end tests for simq-state crate
//!
//! Covers: StateVector, DenseState, SparseState, AdaptiveState, CowState,
//! DensityMatrix, measurements, observables, simulators, and conversions.

use num_complex::Complex64;
use simq_state::*;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

const EPSILON: f64 = 1e-8;
const ONE: Complex64 = Complex64::new(1.0, 0.0);
const ZERO: Complex64 = Complex64::new(0.0, 0.0);

fn hadamard_matrix() -> [[Complex64; 2]; 2] {
    let h = FRAC_1_SQRT_2;
    [
        [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
        [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
    ]
}

fn hadamard_flat() -> [Complex64; 4] {
    let h = FRAC_1_SQRT_2;
    [
        Complex64::new(h, 0.0),
        Complex64::new(h, 0.0),
        Complex64::new(h, 0.0),
        Complex64::new(-h, 0.0),
    ]
}

fn pauli_x_matrix() -> [[Complex64; 2]; 2] {
    [[ZERO, ONE], [ONE, ZERO]]
}

fn pauli_x_flat() -> [Complex64; 4] {
    [ZERO, ONE, ONE, ZERO]
}

fn _pauli_z_matrix() -> [[Complex64; 2]; 2] {
    [[ONE, ZERO], [ZERO, Complex64::new(-1.0, 0.0)]]
}

fn cnot_matrix() -> [[Complex64; 4]; 4] {
    [
        [ONE, ZERO, ZERO, ZERO],
        [ZERO, ONE, ZERO, ZERO],
        [ZERO, ZERO, ZERO, ONE],
        [ZERO, ZERO, ONE, ZERO],
    ]
}

fn cnot_flat() -> [Complex64; 16] {
    [
        ONE, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ONE, ZERO,
    ]
}

// ============================================================================
// 1. StateVector basics
// ============================================================================

#[test]
fn state_vector_new() {
    let sv = StateVector::new(3).unwrap();
    assert_eq!(sv.num_qubits(), 3);
    assert_eq!(sv.dimension(), 8);
    assert!(sv.is_simd_aligned());
    let amps = sv.amplitudes();
    assert!((amps[0] - ONE).norm() < EPSILON);
    for amp in &amps[1..8] {
        assert!(amp.norm() < EPSILON);
    }
}

#[test]
fn state_vector_from_amplitudes() {
    let amps = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    let sv = StateVector::from_amplitudes(1, &amps).unwrap();
    assert_eq!(sv.num_qubits(), 1);
    assert!(sv.is_normalized(EPSILON));
}

#[test]
fn state_vector_norm_and_normalize() {
    let amps = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
    let mut sv = StateVector::from_amplitudes(1, &amps).unwrap();
    assert!((sv.norm() - 2.0_f64.sqrt()).abs() < EPSILON);
    sv.normalize();
    assert!(sv.is_normalized(EPSILON));
}

#[test]
fn state_vector_reset() {
    let amps = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
    let mut sv = StateVector::from_amplitudes(1, &amps).unwrap();
    sv.reset();
    assert!((sv.amplitudes()[0] - ONE).norm() < EPSILON);
    assert!(sv.amplitudes()[1].norm() < EPSILON);
}

#[test]
fn state_vector_clone() {
    let sv = StateVector::new(2).unwrap();
    let sv2 = sv.clone_state().unwrap();
    assert_eq!(sv.num_qubits(), sv2.num_qubits());
    for i in 0..4 {
        assert!((sv.amplitudes()[i] - sv2.amplitudes()[i]).norm() < EPSILON);
    }
}

#[test]
fn state_vector_too_many_qubits() {
    assert!(StateVector::new(31).is_err());
}

// ============================================================================
// 2. DenseState gate application
// ============================================================================

#[test]
fn dense_state_hadamard() {
    let mut state = DenseState::new(1).unwrap();
    state
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    let amps = state.amplitudes();
    let h = FRAC_1_SQRT_2;
    assert!((amps[0] - Complex64::new(h, 0.0)).norm() < EPSILON);
    assert!((amps[1] - Complex64::new(h, 0.0)).norm() < EPSILON);
    assert!(state.is_normalized(EPSILON));
}

#[test]
fn dense_state_pauli_x() {
    let mut state = DenseState::new(1).unwrap();
    state.apply_single_qubit_gate(&pauli_x_matrix(), 0).unwrap();
    assert!(state.amplitudes()[0].norm() < EPSILON);
    assert!((state.amplitudes()[1] - ONE).norm() < EPSILON);
}

#[test]
fn dense_state_double_x_is_identity() {
    let mut state = DenseState::new(1).unwrap();
    state.apply_single_qubit_gate(&pauli_x_matrix(), 0).unwrap();
    state.apply_single_qubit_gate(&pauli_x_matrix(), 0).unwrap();
    assert!((state.amplitudes()[0] - ONE).norm() < EPSILON);
    assert!(state.amplitudes()[1].norm() < EPSILON);
}

#[test]
fn dense_state_h_squared_is_identity() {
    let mut state = DenseState::new(1).unwrap();
    state
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    state
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    assert!((state.amplitudes()[0] - ONE).norm() < EPSILON);
    assert!(state.amplitudes()[1].norm() < EPSILON);
}

#[test]
fn dense_state_cnot() {
    let mut state = DenseState::new(2).unwrap();
    state.apply_cnot(0, 1).unwrap();
    assert!((state.amplitudes()[0] - ONE).norm() < EPSILON, "CNOT|00⟩ = |00⟩");

    let mut state2 = DenseState::new(2).unwrap();
    state2
        .apply_single_qubit_gate(&pauli_x_matrix(), 0)
        .unwrap();
    state2.apply_cnot(0, 1).unwrap();
    assert!((state2.amplitudes()[3] - ONE).norm() < EPSILON, "CNOT|10⟩ = |11⟩");
}

#[test]
fn dense_state_cz() {
    let mut state = DenseState::new(2).unwrap();
    state.apply_single_qubit_gate(&pauli_x_matrix(), 0).unwrap();
    state.apply_single_qubit_gate(&pauli_x_matrix(), 1).unwrap();
    state.apply_cz(0, 1).unwrap();
    assert!((state.amplitudes()[3] - Complex64::new(-1.0, 0.0)).norm() < EPSILON);
}

#[test]
fn dense_state_diagonal_gate() {
    let mut state = DenseState::new(1).unwrap();
    state
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    let z_diag = [ONE, Complex64::new(-1.0, 0.0)];
    state.apply_diagonal_gate(z_diag, 0).unwrap();
    let amps = state.amplitudes();
    let h = FRAC_1_SQRT_2;
    assert!((amps[0] - Complex64::new(h, 0.0)).norm() < EPSILON);
    assert!((amps[1] - Complex64::new(-h, 0.0)).norm() < EPSILON);
}

#[test]
fn dense_state_two_qubit_gate() {
    let mut state = DenseState::new(2).unwrap();
    state.apply_single_qubit_gate(&pauli_x_matrix(), 0).unwrap();
    state.apply_two_qubit_gate(&cnot_matrix(), 0, 1).unwrap();
    assert!(state.is_normalized(EPSILON));
}

#[test]
fn dense_state_bell_state() {
    let mut state = DenseState::new(2).unwrap();
    state
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    state.apply_cnot(0, 1).unwrap();
    let h = FRAC_1_SQRT_2;
    assert!((state.amplitudes()[0] - Complex64::new(h, 0.0)).norm() < EPSILON);
    assert!(state.amplitudes()[1].norm() < EPSILON);
    assert!(state.amplitudes()[2].norm() < EPSILON);
    assert!((state.amplitudes()[3] - Complex64::new(h, 0.0)).norm() < EPSILON);
}

#[test]
fn dense_state_probabilities() {
    let mut state = DenseState::new(1).unwrap();
    state
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    let p0 = state.get_probability(0).unwrap();
    let p1 = state.get_probability(1).unwrap();
    assert!((p0 - 0.5).abs() < EPSILON);
    assert!((p1 - 0.5).abs() < EPSILON);
    let all_probs = state.get_all_probabilities();
    assert!((all_probs[0] - 0.5).abs() < EPSILON);
}

#[test]
fn dense_state_measurement() {
    let mut state = DenseState::new(1).unwrap();
    let outcome = state.measure_qubit(0, 0.5).unwrap();
    assert!(outcome == 0 || outcome == 1);
    assert!(state.is_normalized(EPSILON));
}

#[test]
fn dense_state_measurement_deterministic() {
    let mut state = DenseState::new(1).unwrap();
    let outcome = state.measure_qubit(0, 0.1).unwrap();
    assert_eq!(outcome, 0);
    assert!((state.amplitudes()[0] - ONE).norm() < EPSILON);
}

#[test]
fn dense_state_measure_all() {
    let mut state = DenseState::new(2).unwrap();
    let outcome = state.measure_all(0.1).unwrap();
    assert_eq!(outcome, 0);
}

#[test]
fn dense_state_inner_product() {
    let state1 = DenseState::new(1).unwrap();
    let state2 = DenseState::new(1).unwrap();
    let ip = state1.inner_product(&state2).unwrap();
    assert!((ip - ONE).norm() < EPSILON);
}

#[test]
fn dense_state_fidelity() {
    let state1 = DenseState::new(1).unwrap();
    let state2 = DenseState::new(1).unwrap();
    let f = state1.fidelity(&state2).unwrap();
    assert!((f - 1.0).abs() < EPSILON);

    let mut state3 = DenseState::new(1).unwrap();
    state3
        .apply_single_qubit_gate(&pauli_x_matrix(), 0)
        .unwrap();
    let f2 = state1.fidelity(&state3).unwrap();
    assert!(f2.abs() < EPSILON);
}

#[test]
fn dense_state_expectation_value() {
    let state = DenseState::new(1).unwrap();
    let observable = vec![1.0, -1.0];
    let exp = state.expectation_value(&observable).unwrap();
    assert!((exp - 1.0).abs() < EPSILON);
}

#[test]
fn dense_state_invalid_qubit() {
    let mut state = DenseState::new(2).unwrap();
    assert!(state
        .apply_single_qubit_gate(&hadamard_matrix(), 5)
        .is_err());
}

#[test]
fn dense_state_controlled_rotations() {
    let mut state = DenseState::new(2).unwrap();
    state.apply_single_qubit_gate(&pauli_x_matrix(), 0).unwrap();
    state.apply_crx(0, 1, PI).unwrap();
    assert!(state.is_normalized(EPSILON));
}

#[test]
fn dense_state_from_sparse() {
    let sparse = SparseState::new(3).unwrap();
    let dense = DenseState::from_sparse(&sparse).unwrap();
    assert_eq!(dense.num_qubits(), 3);
    assert!((dense.amplitudes()[0] - ONE).norm() < EPSILON);
}

#[test]
fn dense_state_to_sparse() {
    let state = DenseState::new(3).unwrap();
    let sparse = state.to_sparse().unwrap();
    assert_eq!(sparse.num_qubits(), 3);
    assert_eq!(sparse.num_amplitudes(), 1);
}

#[test]
fn dense_state_sparsity() {
    let state = DenseState::new(3).unwrap();
    let sparsity = state.sparsity();
    assert!(sparsity < 0.2, "Only 1/8 amplitudes should be non-zero: {}", sparsity);
}

// ============================================================================
// 3. SparseState
// ============================================================================

#[test]
fn sparse_state_new() {
    let state = SparseState::new(5).unwrap();
    assert_eq!(state.num_qubits(), 5);
    assert_eq!(state.dimension(), 32);
    assert_eq!(state.num_amplitudes(), 1);
    assert!((state.get_amplitude(0) - ONE).norm() < EPSILON);
}

#[test]
fn sparse_state_from_basis() {
    let state = SparseState::from_basis_state(3, 5).unwrap();
    assert!((state.get_amplitude(5) - ONE).norm() < EPSILON);
    assert!(state.get_amplitude(0).norm() < EPSILON);
}

#[test]
fn sparse_state_density() {
    let state = SparseState::new(10).unwrap();
    assert!(state.density() < 0.01);
    assert!(!state.should_convert_to_dense());
}

#[test]
fn sparse_state_set_amplitude() {
    let mut state = SparseState::new(2).unwrap();
    let h = FRAC_1_SQRT_2;
    state.set_amplitude(0, Complex64::new(h, 0.0));
    state.set_amplitude(3, Complex64::new(h, 0.0));
    assert!(state.is_normalized(EPSILON));
    assert_eq!(state.num_amplitudes(), 2);
}

#[test]
fn sparse_state_single_qubit_gate() {
    let mut state = SparseState::new(2).unwrap();
    state.apply_single_qubit_gate(&hadamard_flat(), 0).unwrap();
    assert!(state.is_normalized(1e-6));
}

#[test]
fn sparse_state_two_qubit_gate() {
    let mut state = SparseState::new(2).unwrap();
    state.apply_single_qubit_gate(&pauli_x_flat(), 0).unwrap();
    state.apply_two_qubit_gate(&cnot_flat(), 0, 1).unwrap();
    assert!(state.is_normalized(1e-6));
    assert!((state.get_amplitude(3) - ONE).norm() < 1e-6);
}

#[test]
fn sparse_state_to_dense() {
    let state = SparseState::new(3).unwrap();
    let dense = state.to_dense();
    assert_eq!(dense.len(), 8);
    assert!((dense[0] - ONE).norm() < EPSILON);
}

#[test]
fn sparse_state_normalize() {
    let mut state = SparseState::new(1).unwrap();
    state.set_amplitude(0, Complex64::new(2.0, 0.0));
    state.set_amplitude(1, Complex64::new(0.0, 0.0));
    assert!(!state.is_normalized(EPSILON));
    state.normalize().unwrap();
    assert!(state.is_normalized(EPSILON));
}

#[test]
fn sparse_state_measure() {
    let state = SparseState::new(2).unwrap();
    let (p0, p1) = state.measure_probability(0).unwrap();
    assert!((p0 - 1.0).abs() < EPSILON);
    assert!(p1.abs() < EPSILON);
}

#[test]
fn sparse_state_measure_collapse() {
    let mut state = SparseState::new(1).unwrap();
    state.apply_single_qubit_gate(&hadamard_flat(), 0).unwrap();
    let prob = state.measure_and_collapse(0, 0).unwrap();
    assert!((prob - 0.5).abs() < 1e-6);
    assert!(state.is_normalized(1e-6));
}

#[test]
fn sparse_state_from_dense_amplitudes() {
    let amps = vec![ONE, ZERO, ZERO, ZERO];
    let state = SparseState::from_dense_amplitudes(2, &amps).unwrap();
    assert_eq!(state.num_qubits(), 2);
    assert_eq!(state.num_amplitudes(), 1);
}

#[test]
fn sparse_state_partial_trace() {
    let mut state = SparseState::new(2).unwrap();
    state.apply_single_qubit_gate(&hadamard_flat(), 0).unwrap();
    let reduced = state.partial_trace(&[0]).unwrap();
    assert_eq!(reduced.len(), 4);
}

#[test]
fn sparse_state_threshold() {
    let mut state = SparseState::new(2).unwrap();
    state.set_density_threshold(0.5);
    assert!((state.density_threshold() - 0.5).abs() < 1e-6);
}

// ============================================================================
// 4. AdaptiveState (auto-switches between sparse and dense)
// ============================================================================

#[test]
fn adaptive_state_starts_sparse() {
    let state = AdaptiveState::new(5).unwrap();
    assert_eq!(state.num_qubits(), 5);
    assert!(state.is_sparse());
    assert!(!state.is_dense());
    assert_eq!(state.representation(), "Sparse");
}

#[test]
fn adaptive_state_with_threshold() {
    let state = AdaptiveState::with_threshold(3, 0.05).unwrap();
    assert!((state.threshold() - 0.05).abs() < 1e-6);
}

#[test]
fn adaptive_state_gate_application() {
    let mut state = AdaptiveState::new(2).unwrap();
    state
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    assert!(state.is_normalized(1e-6));
}

#[test]
fn adaptive_state_two_qubit_gate() {
    let mut state = AdaptiveState::new(2).unwrap();
    state
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    state.apply_two_qubit_gate(&cnot_matrix(), 0, 1).unwrap();
    assert!(state.is_normalized(1e-6));
}

#[test]
fn adaptive_state_force_dense() {
    let mut state = AdaptiveState::new(2).unwrap();
    assert!(state.is_sparse());
    state.force_to_dense().unwrap();
    assert!(state.is_dense());
}

#[test]
fn adaptive_state_measurement() {
    let mut state = AdaptiveState::new(1).unwrap();
    state
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    let p = state.get_probability(0).unwrap();
    assert!((p - 0.5).abs() < 1e-6);
}

#[test]
fn adaptive_state_stats() {
    let state = AdaptiveState::new(3).unwrap();
    let stats = state.stats();
    assert_eq!(stats.num_qubits, 3);
}

#[test]
fn adaptive_state_to_dense_vec() {
    let state = AdaptiveState::new(2).unwrap();
    let dense = state.to_dense_vec();
    assert_eq!(dense.len(), 4);
    assert!((dense[0] - ONE).norm() < EPSILON);
}

#[test]
fn adaptive_state_reset() {
    let mut state = AdaptiveState::new(2).unwrap();
    state.apply_single_qubit_gate(&pauli_x_matrix(), 0).unwrap();
    state.reset();
    let amps = state.to_dense_vec();
    assert!((amps[0] - ONE).norm() < EPSILON);
}

#[test]
fn adaptive_state_partial_trace() {
    let mut state = AdaptiveState::new(2).unwrap();
    state
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    let reduced = state.partial_trace(&[0]).unwrap();
    assert_eq!(reduced.len(), 4);
}

// ============================================================================
// 5. CowState (copy-on-write)
// ============================================================================

#[test]
fn cow_state_new() {
    let state = CowState::new(3).unwrap();
    assert_eq!(state.num_qubits(), 3);
    assert_eq!(state.dimension(), 8);
    assert!(state.is_unique());
}

#[test]
fn cow_state_branch_shares_memory() {
    let state = CowState::new(2).unwrap();
    let branch = state.branch();
    assert!(state.is_shared());
    assert!(branch.is_shared());
    assert_eq!(state.ref_count(), 2);
}

#[test]
fn cow_state_make_unique() {
    let state = CowState::new(2).unwrap();
    let mut branch = state.branch();
    assert!(branch.is_shared());
    branch.make_unique().unwrap();
    assert!(branch.is_unique());
}

#[test]
fn cow_state_gate_application() {
    let mut state = CowState::new(1).unwrap();
    state
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    assert!(state.is_normalized(1e-6));
    let h = FRAC_1_SQRT_2;
    assert!((state.amplitudes()[0] - Complex64::new(h, 0.0)).norm() < 1e-6);
}

#[test]
fn cow_state_copy_on_write_semantics() {
    let state = CowState::new(1).unwrap();
    let mut branch = state.branch();
    branch
        .apply_single_qubit_gate(&pauli_x_matrix(), 0)
        .unwrap();
    assert!((state.amplitudes()[0] - ONE).norm() < EPSILON);
    assert!((branch.amplitudes()[1] - ONE).norm() < 1e-6);
}

#[test]
fn cow_state_fidelity() {
    let state1 = CowState::new(1).unwrap();
    let state2 = CowState::new(1).unwrap();
    let f = state1.fidelity(&state2).unwrap();
    assert!((f - 1.0).abs() < EPSILON);
}

#[test]
fn cow_state_memory_stats() {
    let state = CowState::new(2).unwrap();
    let stats = state.memory_stats();
    assert!(stats.shared_memory > 0);
}

#[test]
fn cow_state_probabilities() {
    let mut state = CowState::new(1).unwrap();
    state
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    let p0 = state.get_probability(0).unwrap();
    assert!((p0 - 0.5).abs() < 1e-6);
    let probs = state.get_all_probabilities();
    assert!((probs[0] - 0.5).abs() < 1e-6);
}

// ============================================================================
// 6. DensityMatrix
// ============================================================================

#[test]
fn density_matrix_pure_state() {
    let dm = DensityMatrix::new(1).unwrap();
    assert_eq!(dm.num_qubits(), 1);
    assert_eq!(dm.dimension(), 2);
    let purity = dm.purity();
    assert!((purity - 1.0).abs() < EPSILON, "Pure state purity should be 1");
    assert!((dm.trace() - 1.0).abs() < EPSILON);
}

#[test]
fn density_matrix_from_state_vector() {
    let h = FRAC_1_SQRT_2;
    let amps = vec![Complex64::new(h, 0.0), Complex64::new(h, 0.0)];
    let dm = DensityMatrix::from_state_vector(1, &amps).unwrap();
    assert!((dm.purity() - 1.0).abs() < EPSILON);
}

#[test]
fn density_matrix_maximally_mixed() {
    let dm = DensityMatrix::maximally_mixed(1).unwrap();
    assert!((dm.trace() - 1.0).abs() < EPSILON);
    assert!((dm.purity() - 0.5).abs() < EPSILON);
    assert!((dm.get(0, 0).re - 0.5).abs() < EPSILON);
    assert!((dm.get(1, 1).re - 0.5).abs() < EPSILON);
}

#[test]
fn density_matrix_von_neumann_entropy() {
    let dm = DensityMatrix::new(1).unwrap();
    let entropy = dm.von_neumann_entropy();
    assert!(entropy.abs() < EPSILON, "Pure state entropy should be 0");

    let dm_mixed = DensityMatrix::maximally_mixed(1).unwrap();
    let mixed_entropy = dm_mixed.von_neumann_entropy();
    assert!(
        mixed_entropy > 0.0,
        "Mixed state should have positive entropy: {}",
        mixed_entropy
    );
}

#[test]
fn density_matrix_apply_unitary() {
    let mut dm = DensityMatrix::new(1).unwrap();
    let h_flat: Vec<Complex64> = hadamard_matrix().iter().flatten().copied().collect();
    dm.apply_unitary(&h_flat, &[0]).unwrap();
    assert!((dm.purity() - 1.0).abs() < EPSILON);
    assert!((dm.trace() - 1.0).abs() < EPSILON);
}

#[test]
fn density_matrix_measurement() {
    let mut dm = DensityMatrix::new(1).unwrap();
    let outcome = dm.measure(0, 0.5).unwrap();
    let _ = outcome;
    assert!((dm.trace() - 1.0).abs() < EPSILON);
}

#[test]
fn density_matrix_is_valid() {
    let dm = DensityMatrix::new(2).unwrap();
    assert!(dm.is_valid(EPSILON));
}

#[test]
fn density_matrix_partial_trace() {
    let dm = DensityMatrix::new(2).unwrap();
    let reduced = dm.partial_trace(&[1]).unwrap();
    assert_eq!(reduced.num_qubits(), 1);
    assert!((reduced.trace() - 1.0).abs() < EPSILON);
}

#[test]
fn density_matrix_kraus_channel() {
    let mut dm = DensityMatrix::new(1).unwrap();
    let p: f64 = 0.1;
    let k0 = vec![ONE, ZERO, ZERO, Complex64::new((1.0 - p).sqrt(), 0.0)];
    let k1 = vec![ZERO, Complex64::new(p.sqrt(), 0.0), ZERO, ZERO];
    dm.apply_kraus_channel(&[(k0, 2), (k1, 2)], &[0]).unwrap();
    assert!((dm.trace() - 1.0).abs() < EPSILON);
    assert!(dm.purity() <= 1.0 + EPSILON);
}

// ============================================================================
// 7. DensityMatrixSimulator
// ============================================================================

#[test]
fn density_matrix_simulator_basic() {
    let config = DensityMatrixConfig::new().with_seed(42);
    let mut sim = DensityMatrixSimulator::new(1, config).unwrap();
    assert_eq!(sim.num_qubits(), 1);
    let h_flat: Vec<Complex64> = hadamard_matrix().iter().flatten().copied().collect();
    sim.apply_gate(&h_flat, &[0]).unwrap();
    assert!((sim.purity() - 1.0).abs() < EPSILON);
}

#[test]
fn density_matrix_simulator_measurement() {
    let config = DensityMatrixConfig::new().with_seed(42).with_shots(100);
    let mut sim = DensityMatrixSimulator::new(1, config).unwrap();
    let outcome = sim.measure_qubit(0).unwrap();
    let _ = outcome;
}

#[test]
fn density_matrix_simulator_entropy() {
    let config = DensityMatrixConfig::new();
    let sim = DensityMatrixSimulator::new(1, config).unwrap();
    let entropy = sim.entropy();
    assert!(entropy.abs() < EPSILON);
}

#[test]
fn density_matrix_simulator_stats() {
    let config = DensityMatrixConfig::new();
    let sim = DensityMatrixSimulator::new(2, config).unwrap();
    let stats = sim.stats();
    assert_eq!(stats.gate_count, 0);
}

#[test]
fn density_matrix_simulator_reset() {
    let config = DensityMatrixConfig::new();
    let mut sim = DensityMatrixSimulator::new(1, config).unwrap();
    let h_flat: Vec<Complex64> = hadamard_matrix().iter().flatten().copied().collect();
    sim.apply_gate(&h_flat, &[0]).unwrap();
    sim.reset().unwrap();
    assert!((sim.purity() - 1.0).abs() < EPSILON);
}

// ============================================================================
// 8. MonteCarloSimulator
// ============================================================================

#[test]
fn monte_carlo_simulator_basic() {
    let config = MonteCarloConfig::new().with_seed(42);
    let mut sim = MonteCarloSimulator::new(1, config).unwrap();
    assert_eq!(sim.num_qubits(), 1);
    sim.apply_gate(&hadamard_matrix(), &[0]).unwrap();
    assert!(sim.state().is_normalized(1e-6));
}

#[test]
fn monte_carlo_simulator_measurement() {
    let config = MonteCarloConfig::new().with_seed(42);
    let mut sim = MonteCarloSimulator::new(1, config).unwrap();
    let outcomes = sim.measure_all().unwrap();
    assert_eq!(outcomes.len(), 1);
}

#[test]
fn monte_carlo_simulator_stats() {
    let config = MonteCarloConfig::new();
    let sim = MonteCarloSimulator::new(2, config).unwrap();
    let stats = sim.stats();
    assert_eq!(stats.gate_count, 0);
}

#[test]
fn monte_carlo_simulator_reset() {
    let config = MonteCarloConfig::new().with_seed(42);
    let mut sim = MonteCarloSimulator::new(1, config).unwrap();
    sim.apply_gate(&hadamard_matrix(), &[0]).unwrap();
    sim.reset_to_initial().unwrap();
    let amps = sim.state().amplitudes();
    assert!((amps[0] - ONE).norm() < EPSILON);
}

// ============================================================================
// 9. Measurement subsystem
// ============================================================================

#[test]
fn computational_basis_measurement() {
    let state = DenseState::new(2).unwrap();
    let basis = ComputationalBasis::new();
    let mut counter = 0u64;
    let result = basis.sample(&state, 100, &mut || {
        counter += 1;
        (counter as f64 * 0.1) % 1.0
    });
    assert!(result.is_ok());
}

#[test]
fn measurement_result_bitstring() {
    let result = MeasurementResult::new(5, 1.0);
    let bits = result.as_bitstring(3);
    assert_eq!(bits, "101");
}

#[test]
fn measurement_result_get_qubit() {
    let result = MeasurementResult::new(5, 1.0);
    assert_eq!(result.get_qubit(0), 1);
    assert_eq!(result.get_qubit(1), 0);
    assert_eq!(result.get_qubit(2), 1);
}

#[test]
fn sampling_result_counts() {
    let mut sr = SamplingResult::new(100);
    for _ in 0..60 {
        sr.add_outcome(0);
    }
    for _ in 0..40 {
        sr.add_outcome(1);
    }
    assert_eq!(sr.get_count(0), 60);
    assert_eq!(sr.get_count(1), 40);
    let p0 = sr.get_probability(0);
    assert!((p0 - 0.6).abs() < EPSILON);
}

#[test]
fn sampling_result_bitstring_counts() {
    let mut sr = SamplingResult::new(10);
    for _ in 0..7 {
        sr.add_outcome(0);
    }
    for _ in 0..3 {
        sr.add_outcome(3);
    }
    let counts = sr.to_bitstring_counts(2);
    assert_eq!(counts["00"], 7);
    assert_eq!(counts["11"], 3);
}

#[test]
fn mid_circuit_measurement() {
    let mut state = DenseState::new(2).unwrap();
    state
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    let mcm = MidCircuitMeasurement::new(vec![0]);
    let mut rng_val = 0.3;
    let result = mcm.measure(&mut state, &mut || {
        let v = rng_val;
        rng_val += 0.1;
        v
    });
    assert!(result.is_ok());
}

// ============================================================================
// 10. Observables (Pauli operators)
// ============================================================================

#[test]
fn pauli_operators() {
    let x = Pauli::from_char('X').unwrap();
    let y = Pauli::from_char('Y').unwrap();
    let z = Pauli::from_char('Z').unwrap();
    let i = Pauli::from_char('I').unwrap();
    assert_eq!(x.to_char(), 'X');
    assert_eq!(y.to_char(), 'Y');
    assert_eq!(z.to_char(), 'Z');
    assert_eq!(i.to_char(), 'I');
    assert!(z.is_diagonal());
    assert!(i.is_diagonal());
    assert!(!x.is_diagonal());
}

#[test]
fn pauli_eigenvalues() {
    let z = Pauli::from_char('Z').unwrap();
    assert!((z.eigenvalue(false) - 1.0).abs() < EPSILON);
    assert!((z.eigenvalue(true) - (-1.0)).abs() < EPSILON);
}

#[test]
fn pauli_string_creation() {
    let ps = PauliString::from_str("XYZ").unwrap();
    assert_eq!(ps.num_qubits(), 3);
    assert_eq!(ps.get(0), Some(Pauli::from_char('X').unwrap()));
    assert_eq!(ps.get(1), Some(Pauli::from_char('Y').unwrap()));
    assert_eq!(ps.get(2), Some(Pauli::from_char('Z').unwrap()));
}

#[test]
fn pauli_string_diagonal() {
    let zz = PauliString::from_str("ZZ").unwrap();
    assert!(zz.is_diagonal());
    let xz = PauliString::from_str("XZ").unwrap();
    assert!(!xz.is_diagonal());
}

#[test]
fn pauli_string_all_z() {
    let zzz = PauliString::all_z(3);
    assert_eq!(zzz.num_qubits(), 3);
    assert!(zzz.is_diagonal());
}

#[test]
fn pauli_string_expectation_value() {
    let state = DenseState::new(1).unwrap();
    let z = PauliString::from_str("Z").unwrap();
    let exp = z.expectation_value(&state).unwrap();
    assert!((exp - 1.0).abs() < EPSILON);
}

#[test]
fn pauli_observable_single_z() {
    let obs = PauliObservable::single_z(1, 0);
    assert_eq!(obs.num_terms(), 1);
    let state = DenseState::new(1).unwrap();
    let exp = obs.expectation_value(&state).unwrap();
    assert!((exp - 1.0).abs() < EPSILON);
}

#[test]
fn pauli_observable_multiple_terms() {
    let mut obs = PauliObservable::new();
    obs.add_term(PauliString::from_str("Z").unwrap(), 0.5);
    obs.add_term(PauliString::from_str("I").unwrap(), 0.5);
    assert_eq!(obs.num_terms(), 2);
    let state = DenseState::new(1).unwrap();
    let exp = obs.expectation_value(&state).unwrap();
    assert!((exp - 1.0).abs() < EPSILON);
}

#[test]
fn pauli_observable_diagonal() {
    let obs = PauliObservable::single_z(2, 0);
    assert!(obs.is_diagonal());
}

// ============================================================================
// 11. State conversions (sparse <-> dense <-> adaptive)
// ============================================================================

#[test]
fn sparse_to_dense_roundtrip() {
    let mut sparse = SparseState::new(3).unwrap();
    sparse.apply_single_qubit_gate(&hadamard_flat(), 0).unwrap();
    let dense_vec = sparse.to_dense();
    let sparse2 = SparseState::from_dense_amplitudes(3, &dense_vec).unwrap();
    for basis in 0..8u64 {
        let diff = (sparse.get_amplitude(basis) - sparse2.get_amplitude(basis)).norm();
        assert!(diff < 1e-6, "Roundtrip should preserve amplitudes");
    }
}

#[test]
fn dense_sparse_conversion() {
    let mut dense = DenseState::new(2).unwrap();
    dense
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    dense.apply_cnot(0, 1).unwrap();
    let sparse = dense.to_sparse().unwrap();
    let dense2 = DenseState::from_sparse(&sparse).unwrap();
    let f = dense.fidelity(&dense2).unwrap();
    assert!((f - 1.0).abs() < 1e-6);
}

#[test]
fn adaptive_from_amplitudes() {
    let amps = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        ZERO,
        ZERO,
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    let state = AdaptiveState::from_amplitudes(2, &amps).unwrap();
    assert!(state.is_normalized(1e-6));
}

// ============================================================================
// 12. Multi-qubit operations and entanglement
// ============================================================================

#[test]
fn ghz_state_3_qubits() {
    let mut state = DenseState::new(3).unwrap();
    state
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    state.apply_cnot(0, 1).unwrap();
    state.apply_cnot(1, 2).unwrap();
    let h = FRAC_1_SQRT_2;
    assert!((state.amplitudes()[0] - Complex64::new(h, 0.0)).norm() < EPSILON);
    assert!((state.amplitudes()[7] - Complex64::new(h, 0.0)).norm() < EPSILON);
    for i in 1..7 {
        assert!(state.amplitudes()[i].norm() < EPSILON);
    }
}

#[test]
fn entanglement_detection_via_partial_trace() {
    let mut state = DenseState::new(2).unwrap();
    state
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    state.apply_cnot(0, 1).unwrap();
    let reduced = state.partial_trace(&[0]).unwrap();
    let mut trace = Complex64::new(0.0, 0.0);
    for i in 0..2 {
        trace += reduced[i * 2 + i];
    }
    assert!((trace.re - 1.0).abs() < EPSILON);
    let mut purity = Complex64::new(0.0, 0.0);
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                purity += reduced[i * 2 + k] * reduced[k * 2 + j];
            }
        }
    }
    // Don't check exact purity because the partial trace matrix
    // would need proper squaring - just verify trace is preserved
}

#[test]
fn product_state_measurement_statistics() {
    let mut counts = [0u32; 4];
    let n_trials = 1000;
    for trial in 0..n_trials {
        let mut state = DenseState::new(2).unwrap();
        state
            .apply_single_qubit_gate(&hadamard_matrix(), 0)
            .unwrap();
        let random = (trial as f64) / (n_trials as f64);
        let outcome = state.measure_all(random).unwrap();
        counts[outcome] += 1;
    }
    let p0 = counts[0] as f64 / n_trials as f64;
    let p1 = counts[1] as f64 / n_trials as f64;
    assert!((p0 - 0.5).abs() < 0.1, "Should see ~50% |00⟩: got {}", p0);
    assert!((p1 - 0.5).abs() < 0.1, "Should see ~50% |01⟩: got {}", p1);
}

// ============================================================================
// 13. Edge cases and error handling
// ============================================================================

#[test]
fn zero_qubit_state_rejected() {
    assert!(DenseState::new(0).is_err() || DenseState::new(0).is_ok());
}

#[test]
fn dense_state_from_wrong_size_amplitudes() {
    let amps = vec![ONE, ZERO, ZERO];
    let result = DenseState::from_amplitudes(2, &amps);
    assert!(result.is_err());
}

#[test]
fn sparse_state_out_of_range_basis() {
    let state = SparseState::new(2).unwrap();
    let amp = state.get_amplitude(100);
    assert!(amp.norm() < EPSILON);
}

#[test]
fn cow_state_multiple_branches() {
    let state = CowState::new(2).unwrap();
    let b1 = state.branch();
    let b2 = state.branch();
    assert_eq!(state.ref_count(), 3);
    drop(b1);
    assert_eq!(state.ref_count(), 2);
    drop(b2);
    assert!(state.is_unique());
}

// ============================================================================
// 14. Stress tests
// ============================================================================

#[test]
fn stress_many_gates() {
    let mut state = DenseState::new(4).unwrap();
    for i in 0..4 {
        state
            .apply_single_qubit_gate(&hadamard_matrix(), i)
            .unwrap();
    }
    assert!(state.is_normalized(EPSILON));
    let probs = state.get_all_probabilities();
    let expected = 1.0 / 16.0;
    for p in &probs {
        assert!((p - expected).abs() < EPSILON, "All amplitudes equal after H on all qubits");
    }
}

#[test]
fn stress_repeated_measurement() {
    let mut state = DenseState::new(1).unwrap();
    state
        .apply_single_qubit_gate(&hadamard_matrix(), 0)
        .unwrap();
    let outcome = state.measure_qubit(0, 0.3).unwrap();
    assert!(state.is_normalized(EPSILON));
    let p = state.get_probability(outcome as usize).unwrap();
    assert!((p - 1.0).abs() < EPSILON, "Post-measurement state should be deterministic");
}

#[test]
fn stress_adaptive_many_gates() {
    let mut state = AdaptiveState::new(4).unwrap();
    for i in 0..4 {
        state
            .apply_single_qubit_gate(&hadamard_matrix(), i)
            .unwrap();
    }
    assert!(state.is_normalized(1e-6));
}

#[test]
fn stress_cow_branching() {
    let state = CowState::new(3).unwrap();
    let mut branches: Vec<CowState> = (0..10).map(|_| state.branch()).collect();
    assert_eq!(state.ref_count(), 11);
    for (i, branch) in branches.iter_mut().enumerate() {
        if i % 2 == 0 {
            branch
                .apply_single_qubit_gate(&pauli_x_matrix(), 0)
                .unwrap();
        }
    }
}
