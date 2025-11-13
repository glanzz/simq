//! SIMD-optimized matrix-vector operations for quantum gates
//!
//! This module provides high-performance implementations of matrix-vector
//! multiplication using SIMD instructions (SSE2, AVX, AVX-512).
//!
//! The operations are specialized for 2×2 and 4×4 complex matrices,
//! which correspond to single-qubit and two-qubit quantum gates.

pub mod single_qubit;
pub mod two_qubit;
pub mod kernels;
pub mod controlled_gates;

use num_complex::Complex64;

/// Apply a 2×2 matrix to a state vector (single-qubit gate)
///
/// This function applies a single-qubit gate to a quantum state vector.
/// It uses SIMD instructions when available for optimal performance.
///
/// # Arguments
/// * `state` - Mutable slice of state amplitudes
/// * `matrix` - 2×2 gate matrix in row-major order
/// * `qubit` - Index of the qubit to apply the gate to
/// * `num_qubits` - Total number of qubits in the state
///
/// # Safety
/// This function uses unsafe SIMD intrinsics. The caller must ensure:
/// - `state.len() == 2^num_qubits`
/// - `qubit < num_qubits`
/// - `state` is properly aligned (64-byte alignment recommended)
///
/// # Example
/// ```ignore
/// use num_complex::Complex64;
/// use simq_state::simd::apply_single_qubit_gate;
///
/// let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
/// let hadamard = [
///     [Complex64::new(0.7071, 0.0), Complex64::new(0.7071, 0.0)],
///     [Complex64::new(0.7071, 0.0), Complex64::new(-0.7071, 0.0)],
/// ];
///
/// apply_single_qubit_gate(&mut state, &hadamard, 0, 1);
/// ```
#[inline]
pub fn apply_single_qubit_gate(
    state: &mut [Complex64],
    matrix: &[[Complex64; 2]; 2],
    qubit: usize,
    num_qubits: usize,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe {
            single_qubit::apply_gate_avx2(state, matrix, qubit, num_qubits);
        }
        return;
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        unsafe {
            single_qubit::apply_gate_sse2(state, matrix, qubit, num_qubits);
        }
        return;
    }

    // Fallback to scalar implementation
    single_qubit::apply_gate_scalar(state, matrix, qubit, num_qubits);
}

/// Apply a 4×4 matrix to a state vector (two-qubit gate)
///
/// This function applies a two-qubit gate to a quantum state vector.
///
/// # Arguments
/// * `state` - Mutable slice of state amplitudes
/// * `matrix` - 4×4 gate matrix in row-major order
/// * `qubit1` - Index of the first qubit
/// * `qubit2` - Index of the second qubit
/// * `num_qubits` - Total number of qubits in the state
///
/// # Safety
/// Similar safety requirements as `apply_single_qubit_gate`
#[inline]
pub fn apply_two_qubit_gate(
    state: &mut [Complex64],
    matrix: &[[Complex64; 4]; 4],
    qubit1: usize,
    qubit2: usize,
    num_qubits: usize,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe {
            two_qubit::apply_gate_avx2(state, matrix, qubit1, qubit2, num_qubits);
        }
        return;
    }

    // Fallback to scalar implementation
    two_qubit::apply_gate_scalar(state, matrix, qubit1, qubit2, num_qubits);
}

/// Compute the norm of a complex vector using SIMD
///
/// # Arguments
/// * `vec` - Complex vector
///
/// # Returns
/// L2 norm of the vector
#[inline]
pub fn norm_simd(vec: &[Complex64]) -> f64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe {
            return kernels::norm_avx2(vec);
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        unsafe {
            return kernels::norm_sse2(vec);
        }
    }

    // Scalar fallback
    vec.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

/// Normalize a complex vector using SIMD
///
/// # Arguments
/// * `vec` - Complex vector to normalize
#[inline]
pub fn normalize_simd(vec: &mut [Complex64]) {
    let norm = norm_simd(vec);
    if norm > 1e-10 {
        let inv_norm = 1.0 / norm;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            unsafe {
                kernels::scale_avx2(vec, inv_norm);
            }
            return;
        }

        // Scalar fallback
        for z in vec.iter_mut() {
            *z *= inv_norm;
        }
    }
}

/// Apply a CNOT (Controlled-NOT) gate optimized for the controlled structure
///
/// This uses direct amplitude manipulation instead of full 4×4 matrix
/// multiplication, providing 3-4x speedup for large state vectors.
///
/// # Arguments
/// * `state` - Mutable slice of state amplitudes
/// * `control` - Index of the control qubit
/// * `target` - Index of the target qubit
/// * `num_qubits` - Total number of qubits
///
/// # Example
/// ```ignore
/// use num_complex::Complex64;
/// use simq_state::simd::apply_cnot;
///
/// let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), ...];
/// apply_cnot(&mut state, 0, 1, 3);
/// ```
#[inline]
pub fn apply_cnot(
    state: &mut [Complex64],
    control: usize,
    target: usize,
    num_qubits: usize,
) {
    controlled_gates::apply_cnot_striped(state, control, target, num_qubits);
}

/// Apply a CZ (Controlled-Z) gate optimized for the controlled structure
///
/// This applies a phase of -1 only to the |11⟩ state, which is much faster
/// than full 4×4 matrix multiplication.
///
/// # Arguments
/// * `state` - Mutable slice of state amplitudes
/// * `qubit1` - Index of the first qubit
/// * `qubit2` - Index of the second qubit
/// * `num_qubits` - Total number of qubits
///
/// # Example
/// ```ignore
/// use num_complex::Complex64;
/// use simq_state::simd::apply_cz;
///
/// let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), ...];
/// apply_cz(&mut state, 0, 1, 3);
/// ```
#[inline]
pub fn apply_cz(
    state: &mut [Complex64],
    qubit1: usize,
    qubit2: usize,
    num_qubits: usize,
) {
    controlled_gates::apply_cz_striped(state, qubit1, qubit2, num_qubits);
}

/// Apply a controlled-U gate (U gate on target if control qubit is 1)
///
/// More general than CNOT but still optimized compared to full 4×4 multiplication.
///
/// # Arguments
/// * `state` - Mutable slice of state amplitudes
/// * `control` - Index of the control qubit
/// * `target` - Index of the target qubit
/// * `u_matrix` - 2×2 unitary matrix to apply to target when control=1
/// * `num_qubits` - Total number of qubits
#[inline]
pub fn apply_controlled_u(
    state: &mut [Complex64],
    control: usize,
    target: usize,
    u_matrix: &[[Complex64; 2]; 2],
    num_qubits: usize,
) {
    controlled_gates::apply_controlled_u_scalar(state, control, target, u_matrix, num_qubits);
}

