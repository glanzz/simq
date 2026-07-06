//! Single-qubit gate application kernels

use super::Matrix2x2;
use crate::execution_engine::error::{ExecutionError, Result};
use num_complex::Complex64;
use rayon::prelude::*;

/// Apply a single-qubit gate to a dense state vector
///
/// This is the non-SIMD version that works on any architecture.
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
    let num_qubits = (n as f64).log2() as usize;

    if qubit >= num_qubits {
        return Err(ExecutionError::QubitOutOfBounds {
            qubit: simq_core::QubitId::new(qubit),
            max: num_qubits,
        });
    }

    // Little-endian convention: qubit k corresponds to bit k of the state index.
    // This matches the sparse kernels, the specialized X/Z/H kernels below, and
    // the simq-state DenseState implementation.
    let stride = 1 << qubit;

    // Validate gate matrix (optional in release builds)
    #[cfg(debug_assertions)]
    validate_gate_matrix(gate)?;

    if use_parallel && n >= parallel_threshold {
        apply_single_qubit_parallel(gate, stride, state);
    } else {
        apply_single_qubit_sequential(gate, stride, state);
    }

    Ok(())
}

/// Apply single-qubit gate sequentially
#[inline]
fn apply_single_qubit_sequential(gate: &Matrix2x2, stride: usize, state: &mut [Complex64]) {
    let n = state.len();
    let mut i = 0;

    while i < n {
        for j in 0..stride {
            let idx0 = i + j;
            let idx1 = idx0 + stride;

            let a = state[idx0];
            let b = state[idx1];

            state[idx0] = gate[0][0] * a + gate[0][1] * b;
            state[idx1] = gate[1][0] * a + gate[1][1] * b;
        }
        i += stride * 2;
    }
}

/// Apply single-qubit gate in parallel
#[inline]
fn apply_single_qubit_parallel(gate: &Matrix2x2, stride: usize, state: &mut [Complex64]) {
    state.par_chunks_mut(stride * 2).for_each(|chunk| {
        for j in 0..stride.min(chunk.len() - stride) {
            let idx0 = j;
            let idx1 = j + stride;

            let a = chunk[idx0];
            let b = chunk[idx1];

            chunk[idx0] = gate[0][0] * a + gate[0][1] * b;
            chunk[idx1] = gate[1][0] * a + gate[1][1] * b;
        }
    });
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

#[cfg(not(debug_assertions))]
#[inline]
#[allow(dead_code)]
fn validate_gate_matrix(_gate: &Matrix2x2) -> Result<()> {
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
    let num_qubits = (n as f64).log2() as usize;

    if qubit >= num_qubits {
        return Err(ExecutionError::QubitOutOfBounds {
            qubit: simq_core::QubitId::new(qubit),
            max: num_qubits,
        });
    }

    let stride = 1 << qubit;

    if use_parallel && n >= parallel_threshold {
        state.par_chunks_mut(stride * 2).for_each(|chunk| {
            for j in 0..stride.min(chunk.len() - stride) {
                chunk.swap(j, j + stride);
            }
        });
    } else {
        let mut i = 0;
        while i < n {
            for j in 0..stride {
                state.swap(i + j, i + j + stride);
            }
            i += stride * 2;
        }
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
    let num_qubits = (n as f64).log2() as usize;

    if qubit >= num_qubits {
        return Err(ExecutionError::QubitOutOfBounds {
            qubit: simq_core::QubitId::new(qubit),
            max: num_qubits,
        });
    }

    let stride = 1 << qubit;

    if use_parallel && n >= parallel_threshold {
        state.par_chunks_mut(stride * 2).for_each(|chunk| {
            for j in stride..stride * 2 {
                if j < chunk.len() {
                    chunk[j] = -chunk[j];
                }
            }
        });
    } else {
        let mut i = 0;
        while i < n {
            for j in stride..stride * 2 {
                state[i + j] = -state[i + j];
            }
            i += stride * 2;
        }
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
    let factor = Complex64::new(SQRT_2_INV, 0.0);

    let n = state.len();
    let num_qubits = (n as f64).log2() as usize;

    if qubit >= num_qubits {
        return Err(ExecutionError::QubitOutOfBounds {
            qubit: simq_core::QubitId::new(qubit),
            max: num_qubits,
        });
    }

    let stride = 1 << qubit;

    if use_parallel && n >= parallel_threshold {
        state.par_chunks_mut(stride * 2).for_each(|chunk| {
            for j in 0..stride.min(chunk.len() - stride) {
                let a = chunk[j];
                let b = chunk[j + stride];
                chunk[j] = (a + b) * factor;
                chunk[j + stride] = (a - b) * factor;
            }
        });
    } else {
        let mut i = 0;
        while i < n {
            for j in 0..stride {
                let a = state[i + j];
                let b = state[i + j + stride];
                state[i + j] = (a + b) * factor;
                state[i + j + stride] = (a - b) * factor;
            }
            i += stride * 2;
        }
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
        // threshold=0 forces parallel path
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
