use ahash::AHashMap;
use num_complex::Complex64;
use ferriq_sim::execution_engine::kernels::controlled::apply_controlled_gate;
use ferriq_sim::execution_engine::kernels::diagonal::{apply_diagonal_gate, apply_phase_gate};
use ferriq_sim::execution_engine::kernels::matrix::{common, GateMatrix, GateMatrixData};
use ferriq_sim::execution_engine::kernels::single_qubit::{
    apply_hadamard, apply_pauli_x, apply_pauli_z, apply_single_qubit_dense,
};
use ferriq_sim::execution_engine::kernels::sparse::{
    apply_single_qubit_sparse, apply_two_qubit_sparse,
};
use ferriq_sim::execution_engine::kernels::two_qubit::{
    apply_cnot, apply_cz, apply_swap, apply_two_qubit_dense,
};
use ferriq_sim::execution_engine::kernels::Matrix4x4;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn zero() -> Complex64 {
    Complex64::new(0.0, 0.0)
}

fn one() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

fn isqrt2() -> Complex64 {
    Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0)
}

fn state_ket0(n: usize) -> Vec<Complex64> {
    let dim = 1 << n;
    let mut s = vec![zero(); dim];
    s[0] = one();
    s
}

fn state_ket1() -> Vec<Complex64> {
    vec![zero(), one()]
}

fn norm_sq(state: &[Complex64]) -> f64 {
    state.iter().map(|a| a.norm_sqr()).sum()
}

fn assert_close(a: Complex64, b: Complex64, tol: f64) {
    assert!((a - b).norm() < tol, "Expected {:?} to be close to {:?}", a, b);
}

// ============================================================================
// Single-qubit gate kernels
// ============================================================================

#[test]
fn single_qubit_pauli_x_ket0_to_ket1() {
    let mut state = state_ket0(1);
    apply_pauli_x(0, &mut state, false, 1024).unwrap();
    assert_close(state[0], zero(), 1e-12);
    assert_close(state[1], one(), 1e-12);
}

#[test]
fn single_qubit_pauli_x_ket1_to_ket0() {
    let mut state = state_ket1();
    apply_pauli_x(0, &mut state, false, 1024).unwrap();
    assert_close(state[0], one(), 1e-12);
    assert_close(state[1], zero(), 1e-12);
}

#[test]
fn single_qubit_pauli_z_ket0_unchanged() {
    let mut state = state_ket0(1);
    apply_pauli_z(0, &mut state, false, 1024).unwrap();
    assert_close(state[0], one(), 1e-12);
    assert_close(state[1], zero(), 1e-12);
}

#[test]
fn single_qubit_pauli_z_ket1_negated() {
    let mut state = state_ket1();
    apply_pauli_z(0, &mut state, false, 1024).unwrap();
    assert_close(state[0], zero(), 1e-12);
    assert_close(state[1], c(-1.0, 0.0), 1e-12);
}

#[test]
fn single_qubit_hadamard_ket0_to_plus() {
    let mut state = state_ket0(1);
    apply_hadamard(0, &mut state, false, 1024).unwrap();
    assert_close(state[0], isqrt2(), 1e-12);
    assert_close(state[1], isqrt2(), 1e-12);
}

#[test]
fn single_qubit_hadamard_twice_identity() {
    let mut state = state_ket0(1);
    apply_hadamard(0, &mut state, false, 1024).unwrap();
    apply_hadamard(0, &mut state, false, 1024).unwrap();
    assert_close(state[0], one(), 1e-10);
    assert_close(state[1], zero(), 1e-10);
}

#[test]
fn single_qubit_dense_pauli_x_matrix() {
    let x = common::pauli_x();
    let mut state = state_ket0(1);
    apply_single_qubit_dense(&x, 0, &mut state, false, 1024).unwrap();
    // big-endian: qubit 0 = MSB for 1-qubit system = bit 0
    assert!((norm_sq(&state) - 1.0).abs() < 1e-10);
}

#[test]
fn single_qubit_dense_hadamard_matrix() {
    let h = common::hadamard();
    let mut state = state_ket0(1);
    apply_single_qubit_dense(&h, 0, &mut state, false, 1024).unwrap();
    assert!((state[0].re - isqrt2().re).abs() < 1e-10);
    assert!((state[1].re - isqrt2().re).abs() < 1e-10);
    assert!((norm_sq(&state) - 1.0).abs() < 1e-10);
}

#[test]
fn single_qubit_dense_identity() {
    let id = common::identity();
    let mut state = state_ket0(1);
    apply_single_qubit_dense(&id, 0, &mut state, false, 1024).unwrap();
    assert_close(state[0], one(), 1e-12);
    assert_close(state[1], zero(), 1e-12);
}

#[test]
fn single_qubit_dense_out_of_bounds() {
    let h = common::hadamard();
    let mut state = state_ket0(1);
    assert!(apply_single_qubit_dense(&h, 1, &mut state, false, 1024).is_err());
}

#[test]
fn single_qubit_preserves_norm_two_qubit_system() {
    let h = common::hadamard();
    let mut state = state_ket0(2);
    apply_single_qubit_dense(&h, 0, &mut state, false, 1024).unwrap();
    assert!((norm_sq(&state) - 1.0).abs() < 1e-10);
    apply_single_qubit_dense(&h, 1, &mut state, false, 1024).unwrap();
    assert!((norm_sq(&state) - 1.0).abs() < 1e-10);
}

// ============================================================================
// Two-qubit gate kernels
// ============================================================================

#[test]
fn cnot_ket00_unchanged() {
    let mut state = state_ket0(2);
    apply_cnot(0, 1, &mut state, false, 1024).unwrap();
    assert_close(state[0], one(), 1e-12);
    for s in &state[1..4] {
        assert_close(*s, zero(), 1e-12);
    }
}

#[test]
fn cnot_ket10_to_ket11() {
    // Little-endian: qubit k is bit k of the state index, so the control
    // (qubit 0) being set corresponds to index 1.
    let mut state = vec![zero(); 4];
    state[1] = one(); // control (qubit 0) = 1, target (qubit 1) = 0
    apply_cnot(0, 1, &mut state, false, 1024).unwrap();
    // CNOT flips target when control=1 → index 3 (both bits set)
    assert_close(state[3], one(), 1e-12);
    assert_close(state[1], zero(), 1e-12);
}

#[test]
fn cnot_preserves_norm() {
    let mut state = state_ket0(2);
    let h = common::hadamard();
    apply_single_qubit_dense(&h, 0, &mut state, false, 1024).unwrap();
    apply_cnot(0, 1, &mut state, false, 1024).unwrap();
    assert!((norm_sq(&state) - 1.0).abs() < 1e-10);
}

#[test]
fn cz_ket11_negated() {
    // |11⟩ → index 3
    let mut state = vec![zero(); 4];
    state[3] = one();
    apply_cz(0, 1, &mut state, false, 1024).unwrap();
    // CZ applies -1 phase to |11⟩
    assert_close(state[3], c(-1.0, 0.0), 1e-12);
}

#[test]
fn cz_ket00_unchanged() {
    let mut state = state_ket0(2);
    apply_cz(0, 1, &mut state, false, 1024).unwrap();
    assert_close(state[0], one(), 1e-12);
}

#[test]
fn swap_exchanges_qubits() {
    // |10⟩ → |01⟩
    let mut state = vec![zero(); 4];
    state[2] = one(); // |10⟩
    apply_swap(0, 1, &mut state, false, 1024).unwrap();
    // |10⟩ → |01⟩ which is index 1
    assert_close(state[1], one(), 1e-12);
    assert_close(state[2], zero(), 1e-12);
}

#[test]
fn swap_same_qubit_is_identity() {
    let mut state = state_ket0(2);
    apply_swap(0, 0, &mut state, false, 1024).unwrap();
    assert_close(state[0], one(), 1e-12);
}

#[test]
fn swap_preserves_norm() {
    let h = common::hadamard();
    let mut state = state_ket0(2);
    apply_single_qubit_dense(&h, 0, &mut state, false, 1024).unwrap();
    apply_swap(0, 1, &mut state, false, 1024).unwrap();
    assert!((norm_sq(&state) - 1.0).abs() < 1e-10);
}

#[test]
fn two_qubit_dense_out_of_bounds() {
    let id4: Matrix4x4 = [
        [one(), zero(), zero(), zero()],
        [zero(), one(), zero(), zero()],
        [zero(), zero(), one(), zero()],
        [zero(), zero(), zero(), one()],
    ];
    let mut state = state_ket0(2);
    assert!(apply_two_qubit_dense(&id4, 0, 2, &mut state, false, 1024).is_err());
}

#[test]
fn two_qubit_dense_same_qubit_error() {
    let id4: Matrix4x4 = [
        [one(), zero(), zero(), zero()],
        [zero(), one(), zero(), zero()],
        [zero(), zero(), one(), zero()],
        [zero(), zero(), zero(), one()],
    ];
    let mut state = state_ket0(2);
    assert!(apply_two_qubit_dense(&id4, 0, 0, &mut state, false, 1024).is_err());
}

// ============================================================================
// Controlled gate kernels
// ============================================================================

#[test]
fn controlled_x_acts_as_cnot() {
    let x = common::pauli_x();
    let mut state = vec![zero(); 4];
    state[2] = one(); // |10⟩ in big-endian
    apply_controlled_gate(0, 1, &x, &mut state, false, 1024).unwrap();
    // Control=0 is qubit 0 (MSB), target=1 is qubit 1 (LSB)
    // |10⟩ → |11⟩
    assert_close(state[3], one(), 1e-12);
    assert_close(state[2], zero(), 1e-12);
}

#[test]
fn controlled_gate_no_flip_when_control_zero() {
    let x = common::pauli_x();
    let mut state = state_ket0(2);
    apply_controlled_gate(0, 1, &x, &mut state, false, 1024).unwrap();
    assert_close(state[0], one(), 1e-12);
}

#[test]
fn controlled_gate_same_qubit_error() {
    let x = common::pauli_x();
    let mut state = state_ket0(2);
    assert!(apply_controlled_gate(0, 0, &x, &mut state, false, 1024).is_err());
}

#[test]
fn controlled_gate_preserves_norm() {
    let h = common::hadamard();
    let x = common::pauli_x();
    let mut state = state_ket0(2);
    apply_single_qubit_dense(&h, 0, &mut state, false, 1024).unwrap();
    apply_controlled_gate(0, 1, &x, &mut state, false, 1024).unwrap();
    assert!((norm_sq(&state) - 1.0).abs() < 1e-10);
}

// ============================================================================
// Diagonal gate kernels
// ============================================================================

#[test]
fn phase_gate_applies_phase_to_ket1() {
    let mut state = state_ket1();
    let phase = c(0.0, 1.0); // i
    apply_phase_gate(0, phase, &mut state, false, 1024).unwrap();
    assert_close(state[0], zero(), 1e-12);
    assert_close(state[1], c(0.0, 1.0), 1e-12);
}

#[test]
fn phase_gate_ket0_unchanged() {
    let mut state = state_ket0(1);
    let phase = c(0.0, 1.0);
    apply_phase_gate(0, phase, &mut state, false, 1024).unwrap();
    assert_close(state[0], one(), 1e-12);
    assert_close(state[1], zero(), 1e-12);
}

#[test]
fn diagonal_gate_single_qubit() {
    let phases = vec![one(), c(-1.0, 0.0)]; // Z gate as diagonal
    let mut state = state_ket1();
    apply_diagonal_gate(&[0], &phases, &mut state, false, 1024).unwrap();
    assert_close(state[1], c(-1.0, 0.0), 1e-12);
}

#[test]
fn diagonal_gate_invalid_phases_length() {
    let phases = vec![one()]; // Wrong length for 1-qubit gate
    let mut state = state_ket0(1);
    assert!(apply_diagonal_gate(&[0], &phases, &mut state, false, 1024).is_err());
}

#[test]
fn diagonal_gate_preserves_norm() {
    let h = common::hadamard();
    let mut state = state_ket0(1);
    apply_single_qubit_dense(&h, 0, &mut state, false, 1024).unwrap();
    let phases = vec![one(), c(0.0, 1.0)]; // S gate
    apply_diagonal_gate(&[0], &phases, &mut state, false, 1024).unwrap();
    assert!((norm_sq(&state) - 1.0).abs() < 1e-10);
}

// ============================================================================
// Sparse state kernels
// ============================================================================

#[test]
fn sparse_single_qubit_x_gate() {
    let x = common::pauli_x();
    let mut amps: AHashMap<u64, Complex64> = AHashMap::new();
    amps.insert(0, one()); // |0⟩
    apply_single_qubit_sparse(&x, 0, &mut amps, 1).unwrap();
    assert!(amps.get(&1).is_some());
    assert_close(*amps.get(&1).unwrap(), one(), 1e-12);
}

#[test]
fn sparse_single_qubit_preserves_norm() {
    let h = common::hadamard();
    let mut amps: AHashMap<u64, Complex64> = AHashMap::new();
    amps.insert(0, one());
    apply_single_qubit_sparse(&h, 0, &mut amps, 1).unwrap();
    let total: f64 = amps.values().map(|a| a.norm_sqr()).sum();
    assert!((total - 1.0).abs() < 1e-10);
}

#[test]
fn sparse_single_qubit_hadamard_twice() {
    let h = common::hadamard();
    let mut amps: AHashMap<u64, Complex64> = AHashMap::new();
    amps.insert(0, one());
    apply_single_qubit_sparse(&h, 0, &mut amps, 1).unwrap();
    apply_single_qubit_sparse(&h, 0, &mut amps, 1).unwrap();
    let amp0 = amps.get(&0).copied().unwrap_or(zero());
    assert_close(amp0, one(), 1e-10);
}

#[test]
fn sparse_two_qubit_cnot() {
    let cnot_matrix: [[Complex64; 4]; 4] = [
        [one(), zero(), zero(), zero()],
        [zero(), one(), zero(), zero()],
        [zero(), zero(), zero(), one()],
        [zero(), zero(), one(), zero()],
    ];
    let mut amps: AHashMap<u64, Complex64> = AHashMap::new();
    amps.insert(2, one()); // |10⟩
    apply_two_qubit_sparse(&cnot_matrix, 1, 0, &mut amps, 2).unwrap();
    // |10⟩ → |11⟩
    let amp3 = amps.get(&3).copied().unwrap_or(zero());
    assert_close(amp3, one(), 1e-12);
}

#[test]
fn sparse_two_qubit_preserves_norm() {
    let h = common::hadamard();
    let mut amps: AHashMap<u64, Complex64> = AHashMap::new();
    amps.insert(0, one());
    apply_single_qubit_sparse(&h, 0, &mut amps, 2).unwrap();

    let cnot_matrix: [[Complex64; 4]; 4] = [
        [one(), zero(), zero(), zero()],
        [zero(), one(), zero(), zero()],
        [zero(), zero(), zero(), one()],
        [zero(), zero(), one(), zero()],
    ];
    apply_two_qubit_sparse(&cnot_matrix, 0, 1, &mut amps, 2).unwrap();
    let total: f64 = amps.values().map(|a| a.norm_sqr()).sum();
    assert!((total - 1.0).abs() < 1e-10);
}

// ============================================================================
// GateMatrix
// ============================================================================

#[test]
fn gate_matrix_single() {
    let h = common::hadamard();
    let gm = GateMatrix::single(h);
    assert_eq!(gm.num_qubits(), 1);
    assert!(!gm.is_diagonal);
}

#[test]
fn gate_matrix_identity_is_diagonal() {
    let id = common::identity();
    let gm = GateMatrix::single(id);
    assert!(gm.is_diagonal);
}

#[test]
fn gate_matrix_pauli_z_is_diagonal() {
    let z = common::pauli_z();
    let gm = GateMatrix::single(z);
    assert!(gm.is_diagonal);
}

#[test]
fn gate_matrix_two_qubit() {
    let cnot: Matrix4x4 = [
        [one(), zero(), zero(), zero()],
        [zero(), one(), zero(), zero()],
        [zero(), zero(), zero(), one()],
        [zero(), zero(), one(), zero()],
    ];
    let gm = GateMatrix::two(cnot);
    assert_eq!(gm.num_qubits(), 2);
}

#[test]
fn gate_matrix_diagonal() {
    let phases = vec![one(), c(-1.0, 0.0)];
    let gm = GateMatrix::diagonal(phases);
    assert!(gm.is_diagonal);
    assert_eq!(gm.num_qubits(), 1);
}

#[test]
fn gate_matrix_data_variants() {
    let h = common::hadamard();
    let gm = GateMatrix::single(h);
    match &gm.data {
        GateMatrixData::Single(_) => {},
        _ => panic!("Expected Single variant"),
    }
}

// ============================================================================
// Common gate matrices
// ============================================================================

#[test]
fn common_pauli_x_correct() {
    let x = common::pauli_x();
    assert_close(x[0][0], zero(), 1e-12);
    assert_close(x[0][1], one(), 1e-12);
    assert_close(x[1][0], one(), 1e-12);
    assert_close(x[1][1], zero(), 1e-12);
}

#[test]
fn common_pauli_y_correct() {
    let y = common::pauli_y();
    assert_close(y[0][0], zero(), 1e-12);
    assert_close(y[0][1], c(0.0, -1.0), 1e-12);
    assert_close(y[1][0], c(0.0, 1.0), 1e-12);
    assert_close(y[1][1], zero(), 1e-12);
}

#[test]
fn common_pauli_z_correct() {
    let z = common::pauli_z();
    assert_close(z[0][0], one(), 1e-12);
    assert_close(z[0][1], zero(), 1e-12);
    assert_close(z[1][0], zero(), 1e-12);
    assert_close(z[1][1], c(-1.0, 0.0), 1e-12);
}

#[test]
fn common_hadamard_correct() {
    let h = common::hadamard();
    assert_close(h[0][0], isqrt2(), 1e-12);
    assert_close(h[0][1], isqrt2(), 1e-12);
    assert_close(h[1][0], isqrt2(), 1e-12);
    assert_close(h[1][1], -isqrt2(), 1e-12);
}

#[test]
fn common_identity_correct() {
    let id = common::identity();
    assert_close(id[0][0], one(), 1e-12);
    assert_close(id[0][1], zero(), 1e-12);
    assert_close(id[1][0], zero(), 1e-12);
    assert_close(id[1][1], one(), 1e-12);
}

// ============================================================================
// Parallel execution paths
// ============================================================================

#[test]
fn single_qubit_parallel_path() {
    let h = common::hadamard();
    let mut state = state_ket0(1);
    // Force parallel with threshold 0
    apply_single_qubit_dense(&h, 0, &mut state, true, 0).unwrap();
    assert!((norm_sq(&state) - 1.0).abs() < 1e-10);
}

#[test]
fn cnot_parallel_path() {
    let mut state = vec![zero(); 4];
    state[2] = one();
    apply_cnot(0, 1, &mut state, true, 0).unwrap();
    assert!((norm_sq(&state) - 1.0).abs() < 1e-10);
}

#[test]
fn controlled_gate_parallel_path() {
    let x = common::pauli_x();
    let mut state = state_ket0(2);
    let h = common::hadamard();
    apply_single_qubit_dense(&h, 0, &mut state, false, 1024).unwrap();
    apply_controlled_gate(0, 1, &x, &mut state, true, 0).unwrap();
    assert!((norm_sq(&state) - 1.0).abs() < 1e-10);
}
