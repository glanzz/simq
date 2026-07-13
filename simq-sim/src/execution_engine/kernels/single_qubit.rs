//! Single-qubit gate application kernels
//!
//! All kernels process the state in cache-sized blocks. Parallel execution
//! splits the state into blocks of at least [`MIN_PAR_BLOCK`] amplitudes so
//! rayon tasks stay coarse-grained: the old per-pair task splitting spent
//! more time in fork/join than in the actual complex arithmetic (issue #76).

use super::Matrix2x2;
use crate::execution_engine::error::{ExecutionError, Result};
use num_complex::Complex64;
use rayon::prelude::*;

/// Minimum amplitudes per rayon task (128 KiB of Complex64).
///
/// Below this, fork/join overhead dominates the memory-bound kernel work.
pub(crate) const MIN_PAR_BLOCK: usize = 1 << 13;

/// Whether a state of `n` amplitudes should be split into parallel blocks of
/// `unit`-amplitude groups (`unit` = 2*stride for single-qubit gates).
#[inline]
pub(crate) fn par_block_len(n: usize, unit: usize) -> Option<usize> {
    let block = unit.max(MIN_PAR_BLOCK);
    // Require at least two blocks, otherwise parallelism buys nothing.
    if n >= block * 2 {
        Some(block)
    } else {
        None
    }
}

#[inline]
fn check_qubit(qubit: usize, n: usize) -> Result<usize> {
    let num_qubits = n.trailing_zeros() as usize;
    if qubit >= num_qubits {
        return Err(ExecutionError::QubitOutOfBounds {
            qubit: simq_core::QubitId::new(qubit),
            max: num_qubits,
        });
    }
    Ok(num_qubits)
}

/// Apply a single-qubit gate to a dense state vector
///
/// Dispatches to a diagonal fast path when the matrix has zero off-diagonal
/// elements (RZ, Phase, S, T, Z, ...), which halves the memory traffic.
///
/// # Arguments
///
/// * `gate` - The 2x2 gate matrix
/// * `qubit` - Index of the target qubit
/// * `state` - The state vector to modify
/// * `use_parallel` - Whether to use parallel execution
/// * `parallel_threshold` - Minimum state size for parallel execution
///
/// # Errors
///
/// Returns an error if the qubit index is out of bounds.
pub fn apply_single_qubit_dense(
    gate: &Matrix2x2,
    qubit: usize,
    state: &mut [Complex64],
    use_parallel: bool,
    parallel_threshold: usize,
) -> Result<()> {
    let n = state.len();
    check_qubit(qubit, n)?;

    // Little-endian convention: qubit k corresponds to bit k of the state index.
    let stride = 1 << qubit;

    #[cfg(debug_assertions)]
    validate_gate_matrix(gate)?;

    // Diagonal fast path: no amplitude mixing, one multiply per amplitude.
    if gate[0][1] == Complex64::new(0.0, 0.0) && gate[1][0] == Complex64::new(0.0, 0.0) {
        return apply_diagonal(
            [gate[0][0], gate[1][1]],
            qubit,
            state,
            use_parallel,
            parallel_threshold,
        );
    }

    let parallel = use_parallel && n >= parallel_threshold;
    match par_block_len(n, stride * 2).filter(|_| parallel) {
        Some(block) => {
            state
                .par_chunks_mut(block)
                .for_each(|chunk| apply_single_qubit_block(gate, stride, chunk));
        },
        None => apply_single_qubit_block(gate, stride, state),
    }

    Ok(())
}

/// Apply the gate to a block whose length is a multiple of `2 * stride`.
///
/// The zip over two disjoint sub-slices is bounds-check free and
/// auto-vectorizes.
#[inline]
fn apply_single_qubit_block(gate: &Matrix2x2, stride: usize, block: &mut [Complex64]) {
    let [[g00, g01], [g10, g11]] = *gate;
    for group in block.chunks_exact_mut(stride * 2) {
        let (lo, hi) = group.split_at_mut(stride);
        for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
            let x = *a;
            let y = *b;
            *a = g00 * x + g01 * y;
            *b = g10 * x + g11 * y;
        }
    }
}

/// Apply a diagonal single-qubit gate diag(d0, d1)
#[inline]
fn apply_diagonal(
    diag: [Complex64; 2],
    qubit: usize,
    state: &mut [Complex64],
    use_parallel: bool,
    parallel_threshold: usize,
) -> Result<()> {
    let n = state.len();
    let stride = 1 << qubit;
    let parallel = use_parallel && n >= parallel_threshold;

    match par_block_len(n, stride * 2).filter(|_| parallel) {
        Some(block) => {
            state
                .par_chunks_mut(block)
                .for_each(|chunk| apply_diagonal_block(diag, stride, chunk));
        },
        None => apply_diagonal_block(diag, stride, state),
    }
    Ok(())
}

#[inline]
fn apply_diagonal_block(diag: [Complex64; 2], stride: usize, block: &mut [Complex64]) {
    let [d0, d1] = diag;
    let one = Complex64::new(1.0, 0.0);
    let scale_lo = d0 != one;
    for group in block.chunks_exact_mut(stride * 2) {
        let (lo, hi) = group.split_at_mut(stride);
        if scale_lo {
            for a in lo.iter_mut() {
                *a *= d0;
            }
        }
        for b in hi.iter_mut() {
            *b *= d1;
        }
    }
}

/// Validate that a gate matrix is unitary (in debug mode)
#[cfg(debug_assertions)]
fn validate_gate_matrix(gate: &Matrix2x2) -> Result<()> {
    use approx::abs_diff_eq;

    // Compute gate * gate†
    let a00 = gate[0][0] * gate[0][0].conj() + gate[0][1] * gate[0][1].conj();
    let a11 = gate[1][0] * gate[1][0].conj() + gate[1][1] * gate[1][1].conj();
    let a01 = gate[0][0] * gate[1][0].conj() + gate[0][1] * gate[1][1].conj();

    // Should be identity matrix
    if !abs_diff_eq!(a00.re, 1.0, epsilon = 1e-10) || !abs_diff_eq!(a00.im, 0.0, epsilon = 1e-10) {
        return Err(ExecutionError::InvalidGateMatrix {
            gate: "single-qubit".to_string(),
            reason: format!("Matrix not unitary: (0,0) = {}", a00),
        });
    }

    if !abs_diff_eq!(a11.re, 1.0, epsilon = 1e-10) || !abs_diff_eq!(a11.im, 0.0, epsilon = 1e-10) {
        return Err(ExecutionError::InvalidGateMatrix {
            gate: "single-qubit".to_string(),
            reason: format!("Matrix not unitary: (1,1) = {}", a11),
        });
    }

    if !abs_diff_eq!(a01.re, 0.0, epsilon = 1e-10) || !abs_diff_eq!(a01.im, 0.0, epsilon = 1e-10) {
        return Err(ExecutionError::InvalidGateMatrix {
            gate: "single-qubit".to_string(),
            reason: format!("Matrix not unitary: (0,1) = {}", a01),
        });
    }

    Ok(())
}

/// Apply a Pauli-X gate (optimized special case)
pub fn apply_pauli_x(
    qubit: usize,
    state: &mut [Complex64],
    use_parallel: bool,
    parallel_threshold: usize,
) -> Result<()> {
    let n = state.len();
    check_qubit(qubit, n)?;
    let stride = 1 << qubit;

    let x_block = |block: &mut [Complex64]| {
        for group in block.chunks_exact_mut(stride * 2) {
            let (lo, hi) = group.split_at_mut(stride);
            lo.swap_with_slice(hi);
        }
    };

    let parallel = use_parallel && n >= parallel_threshold;
    match par_block_len(n, stride * 2).filter(|_| parallel) {
        Some(block) => state.par_chunks_mut(block).for_each(x_block),
        None => x_block(state),
    }

    Ok(())
}

/// Apply a Pauli-Z gate (optimized special case)
pub fn apply_pauli_z(
    qubit: usize,
    state: &mut [Complex64],
    use_parallel: bool,
    parallel_threshold: usize,
) -> Result<()> {
    let n = state.len();
    check_qubit(qubit, n)?;
    let stride = 1 << qubit;

    let z_block = |block: &mut [Complex64]| {
        for group in block.chunks_exact_mut(stride * 2) {
            for b in &mut group[stride..] {
                *b = -*b;
            }
        }
    };

    let parallel = use_parallel && n >= parallel_threshold;
    match par_block_len(n, stride * 2).filter(|_| parallel) {
        Some(block) => state.par_chunks_mut(block).for_each(z_block),
        None => z_block(state),
    }

    Ok(())
}

/// Apply a Hadamard gate (optimized special case)
pub fn apply_hadamard(
    qubit: usize,
    state: &mut [Complex64],
    use_parallel: bool,
    parallel_threshold: usize,
) -> Result<()> {
    const SQRT_2_INV: f64 = std::f64::consts::FRAC_1_SQRT_2;

    let n = state.len();
    check_qubit(qubit, n)?;
    let stride = 1 << qubit;

    let h_block = |block: &mut [Complex64]| {
        for group in block.chunks_exact_mut(stride * 2) {
            let (lo, hi) = group.split_at_mut(stride);
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let x = *a;
                let y = *b;
                *a = (x + y) * SQRT_2_INV;
                *b = (x - y) * SQRT_2_INV;
            }
        }
    };

    let parallel = use_parallel && n >= parallel_threshold;
    match par_block_len(n, stride * 2).filter(|_| parallel) {
        Some(block) => state.par_chunks_mut(block).for_each(h_block),
        None => h_block(state),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_pauli_x() {
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

        apply_pauli_x(0, &mut state, false, usize::MAX).unwrap();

        assert_abs_diff_eq!(state[0].re, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[1].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pauli_z() {
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];

        apply_pauli_z(0, &mut state, false, usize::MAX).unwrap();

        assert_abs_diff_eq!(state[0].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[1].re, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hadamard() {
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

        apply_hadamard(0, &mut state, false, usize::MAX).unwrap();

        let sqrt_2_inv = std::f64::consts::FRAC_1_SQRT_2;
        assert_abs_diff_eq!(state[0].re, sqrt_2_inv, epsilon = 1e-10);
        assert_abs_diff_eq!(state[1].re, sqrt_2_inv, epsilon = 1e-10);
    }

    #[test]
    fn test_general_single_qubit() {
        // Test with Hadamard matrix
        let h_gate: Matrix2x2 = [
            [
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            ],
            [
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(-std::f64::consts::FRAC_1_SQRT_2, 0.0),
            ],
        ];

        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

        apply_single_qubit_dense(&h_gate, 0, &mut state, false, usize::MAX).unwrap();

        let sqrt_2_inv = std::f64::consts::FRAC_1_SQRT_2;
        assert_abs_diff_eq!(state[0].re, sqrt_2_inv, epsilon = 1e-10);
        assert_abs_diff_eq!(state[1].re, sqrt_2_inv, epsilon = 1e-10);
    }

    /// The diagonal fast path (RZ-like gate) must match the general kernel.
    #[test]
    fn test_diagonal_fast_path_matches_general() {
        let theta: f64 = 0.7;
        let rz: Matrix2x2 = [
            [
                Complex64::from_polar(1.0, -theta / 2.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::from_polar(1.0, theta / 2.0),
            ],
        ];

        // A 4-qubit random-ish state
        let mut state: Vec<Complex64> = (0..16)
            .map(|i| Complex64::new(0.1 + i as f64 * 0.05, 0.02 * i as f64))
            .collect();
        let mut expected = state.clone();

        // Reference: manual dense application
        for (i, amp) in expected.iter_mut().enumerate() {
            if i & 0b10 == 0 {
                *amp *= rz[0][0];
            } else {
                *amp *= rz[1][1];
            }
        }

        apply_single_qubit_dense(&rz, 1, &mut state, false, usize::MAX).unwrap();

        for (got, want) in state.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(got.re, want.re, epsilon = 1e-12);
            assert_abs_diff_eq!(got.im, want.im, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_qubit_out_of_bounds() {
        let gate: Matrix2x2 = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];

        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

        let result = apply_single_qubit_dense(&gate, 10, &mut state, false, usize::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn test_pauli_x_parallel() {
        // threshold=0 forces the parallel branch selection logic
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        apply_pauli_x(0, &mut state, true, 0).unwrap();
        assert_abs_diff_eq!(state[0].re, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[1].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pauli_x_out_of_bounds() {
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let result = apply_pauli_x(5, &mut state, false, usize::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn test_pauli_z_parallel() {
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        apply_pauli_z(0, &mut state, true, 0).unwrap();
        assert_abs_diff_eq!(state[0].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[1].re, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pauli_z_out_of_bounds() {
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let result = apply_pauli_z(5, &mut state, false, usize::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn test_hadamard_parallel() {
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        apply_hadamard(0, &mut state, true, 0).unwrap();
        let sqrt_2_inv = std::f64::consts::FRAC_1_SQRT_2;
        assert_abs_diff_eq!(state[0].re, sqrt_2_inv, epsilon = 1e-10);
        assert_abs_diff_eq!(state[1].re, sqrt_2_inv, epsilon = 1e-10);
    }

    #[test]
    fn test_hadamard_out_of_bounds() {
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let result = apply_hadamard(5, &mut state, false, usize::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn test_general_single_qubit_gate_parallel() {
        let h_gate: Matrix2x2 = [
            [
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            ],
            [
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(-std::f64::consts::FRAC_1_SQRT_2, 0.0),
            ],
        ];
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        apply_single_qubit_dense(&h_gate, 0, &mut state, true, 0).unwrap();
        let sqrt_2_inv = std::f64::consts::FRAC_1_SQRT_2;
        assert_abs_diff_eq!(state[0].re, sqrt_2_inv, epsilon = 1e-10);
        assert_abs_diff_eq!(state[1].re, sqrt_2_inv, epsilon = 1e-10);
    }

    /// Large state: parallel and sequential kernels must agree bit-for-bit.
    #[test]
    fn test_parallel_matches_sequential_large_state() {
        let ry: Matrix2x2 = [
            [Complex64::new(0.8, 0.0), Complex64::new(-0.6, 0.0)],
            [Complex64::new(0.6, 0.0), Complex64::new(0.8, 0.0)],
        ];
        let n = 1 << 15;
        let base: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()) / (n as f64).sqrt())
            .collect();

        for qubit in [0usize, 3, 7, 14] {
            let mut seq = base.clone();
            let mut par = base.clone();
            apply_single_qubit_dense(&ry, qubit, &mut seq, false, usize::MAX).unwrap();
            apply_single_qubit_dense(&ry, qubit, &mut par, true, 0).unwrap();
            assert_eq!(seq, par, "qubit {} mismatch", qubit);
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_validate_gate_matrix_not_unitary_a00() {
        // Non-unitary gate: row norms don't equal 1
        let bad_gate: Matrix2x2 = [
            [Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let result = apply_single_qubit_dense(&bad_gate, 0, &mut state, false, usize::MAX);
        assert!(result.is_err());
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_validate_gate_matrix_not_unitary_a11() {
        // Gate where (0,0) passes but (1,1) fails
        let bad_gate: Matrix2x2 = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(2.0, 0.0)],
        ];
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let result = apply_single_qubit_dense(&bad_gate, 0, &mut state, false, usize::MAX);
        assert!(result.is_err());
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_validate_gate_matrix_not_unitary_a01() {
        // Gate where diagonal norms are 1 but off-diag inner product fails
        // Use: [[1/√2, 1/√2], [1/√2, 1/√2]] — not unitary (rows not orthogonal)
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let bad_gate: Matrix2x2 = [
            [Complex64::new(s, 0.0), Complex64::new(s, 0.0)],
            [Complex64::new(s, 0.0), Complex64::new(s, 0.0)],
        ];
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let result = apply_single_qubit_dense(&bad_gate, 0, &mut state, false, usize::MAX);
        assert!(result.is_err());
    }
}
