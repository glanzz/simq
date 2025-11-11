//! SIMD-optimized single-qubit gate operations

use num_complex::Complex64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Apply a single-qubit gate using scalar operations (fallback)
///
/// This is the reference implementation that works on all platforms.
/// It applies the gate matrix to pairs of amplitudes corresponding to
/// the computational basis states that differ only in the target qubit.
///
/// # Algorithm
/// For a gate on qubit `q` in an n-qubit system:
/// - Group amplitudes into pairs (i, j) where i and j differ only in bit q
/// - Apply 2×2 matrix multiplication to each pair
/// - Complexity: O(2^n) operations
pub fn apply_gate_scalar(
    state: &mut [Complex64],
    matrix: &[[Complex64; 2]; 2],
    qubit: usize,
    num_qubits: usize,
) {
    let dimension = 1 << num_qubits;
    let qubit_mask = 1 << qubit;

    // Extract matrix elements for better cache locality
    let m00 = matrix[0][0];
    let m01 = matrix[0][1];
    let m10 = matrix[1][0];
    let m11 = matrix[1][1];

    // Iterate over all basis states
    for i in 0..dimension {
        // Skip if this is the "high" state (we process pairs)
        if i & qubit_mask != 0 {
            continue;
        }

        let j = i | qubit_mask; // j = i with qubit bit set to 1

        // Load current amplitudes
        let amp0 = state[i];
        let amp1 = state[j];

        // Apply matrix multiplication
        state[i] = m00 * amp0 + m01 * amp1;
        state[j] = m10 * amp0 + m11 * amp1;
    }
}

/// Apply a single-qubit gate using SSE2 instructions
///
/// Uses 128-bit SIMD registers to process complex numbers.
/// Each Complex64 fits in one SSE register (2 f64 values).
///
/// # Safety
/// Requires SSE2 support (available on all x86_64 CPUs)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn apply_gate_sse2(
    state: &mut [Complex64],
    matrix: &[[Complex64; 2]; 2],
    qubit: usize,
    num_qubits: usize,
) {
    let dimension = 1 << num_qubits;
    let qubit_mask = 1 << qubit;

    // Load matrix elements into SSE registers
    // Each complex number occupies one __m128d register (2 f64)
    let m00_re = _mm_set1_pd(matrix[0][0].re);
    let m00_im = _mm_set1_pd(matrix[0][0].im);
    let m01_re = _mm_set1_pd(matrix[0][1].re);
    let m01_im = _mm_set1_pd(matrix[0][1].im);
    let m10_re = _mm_set1_pd(matrix[1][0].re);
    let m10_im = _mm_set1_pd(matrix[1][0].im);
    let m11_re = _mm_set1_pd(matrix[1][1].re);
    let m11_im = _mm_set1_pd(matrix[1][1].im);

    for i in 0..dimension {
        if i & qubit_mask != 0 {
            continue;
        }

        let j = i | qubit_mask;

        // Load amplitudes as [re, im]
        let amp0_ptr = state.as_ptr().add(i) as *const f64;
        let amp1_ptr = state.as_ptr().add(j) as *const f64;

        let amp0 = _mm_loadu_pd(amp0_ptr); // [amp0.re, amp0.im]
        let amp1 = _mm_loadu_pd(amp1_ptr); // [amp1.re, amp1.im]

        // Broadcast real and imaginary parts
        let amp0_re = _mm_shuffle_pd::<0b00>(amp0, amp0); // [amp0.re, amp0.re]
        let amp0_im = _mm_shuffle_pd::<0b11>(amp0, amp0); // [amp0.im, amp0.im]
        let amp1_re = _mm_shuffle_pd::<0b00>(amp1, amp1); // [amp1.re, amp1.re]
        let amp1_im = _mm_shuffle_pd::<0b11>(amp1, amp1); // [amp1.im, amp1.im]

        // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i

        // Compute new state[i] = m00 * amp0 + m01 * amp1
        let t0_re = _mm_mul_pd(m00_re, amp0_re); // m00.re * amp0.re
        let t0_im_neg = _mm_mul_pd(m00_im, amp0_im); // m00.im * amp0.im
        let new_i_re_part1 = _mm_sub_pd(t0_re, t0_im_neg); // (m00.re * amp0.re - m00.im * amp0.im)

        let t0_re_im = _mm_mul_pd(m00_re, amp0_im); // m00.re * amp0.im
        let t0_im_re = _mm_mul_pd(m00_im, amp0_re); // m00.im * amp0.re
        let new_i_im_part1 = _mm_add_pd(t0_re_im, t0_im_re); // (m00.re * amp0.im + m00.im * amp0.re)

        let t1_re = _mm_mul_pd(m01_re, amp1_re);
        let t1_im_neg = _mm_mul_pd(m01_im, amp1_im);
        let new_i_re_part2 = _mm_sub_pd(t1_re, t1_im_neg);

        let t1_re_im = _mm_mul_pd(m01_re, amp1_im);
        let t1_im_re = _mm_mul_pd(m01_im, amp1_re);
        let new_i_im_part2 = _mm_add_pd(t1_re_im, t1_im_re);

        let new_i_re = _mm_add_pd(new_i_re_part1, new_i_re_part2);
        let new_i_im = _mm_add_pd(new_i_im_part1, new_i_im_part2);

        // Interleave real and imaginary parts: [re, im]
        let new_i = _mm_unpacklo_pd(new_i_re, new_i_im);

        // Compute new state[j] = m10 * amp0 + m11 * amp1
        let t2_re = _mm_mul_pd(m10_re, amp0_re);
        let t2_im_neg = _mm_mul_pd(m10_im, amp0_im);
        let new_j_re_part1 = _mm_sub_pd(t2_re, t2_im_neg);

        let t2_re_im = _mm_mul_pd(m10_re, amp0_im);
        let t2_im_re = _mm_mul_pd(m10_im, amp0_re);
        let new_j_im_part1 = _mm_add_pd(t2_re_im, t2_im_re);

        let t3_re = _mm_mul_pd(m11_re, amp1_re);
        let t3_im_neg = _mm_mul_pd(m11_im, amp1_im);
        let new_j_re_part2 = _mm_sub_pd(t3_re, t3_im_neg);

        let t3_re_im = _mm_mul_pd(m11_re, amp1_im);
        let t3_im_re = _mm_mul_pd(m11_im, amp1_re);
        let new_j_im_part2 = _mm_add_pd(t3_re_im, t3_im_re);

        let new_j_re = _mm_add_pd(new_j_re_part1, new_j_re_part2);
        let new_j_im = _mm_add_pd(new_j_im_part1, new_j_im_part2);

        let new_j = _mm_unpacklo_pd(new_j_re, new_j_im);

        // Store results
        let out_i_ptr = state.as_mut_ptr().add(i) as *mut f64;
        let out_j_ptr = state.as_mut_ptr().add(j) as *mut f64;

        _mm_storeu_pd(out_i_ptr, new_i);
        _mm_storeu_pd(out_j_ptr, new_j);
    }
}

/// Apply a single-qubit gate using AVX2 instructions
///
/// Uses 256-bit SIMD registers to process 2 complex numbers at once.
///
/// # Safety
/// Requires AVX2 support
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn apply_gate_avx2(
    state: &mut [Complex64],
    matrix: &[[Complex64; 2]; 2],
    qubit: usize,
    num_qubits: usize,
) {
    // For now, fallback to SSE2 for correctness
    // Full AVX2 implementation would process multiple pairs simultaneously
    apply_gate_sse2(state, matrix, qubit, num_qubits);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn hadamard_matrix() -> [[Complex64; 2]; 2] {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        [
            [Complex64::new(inv_sqrt2, 0.0), Complex64::new(inv_sqrt2, 0.0)],
            [Complex64::new(inv_sqrt2, 0.0), Complex64::new(-inv_sqrt2, 0.0)],
        ]
    }

    #[test]
    fn test_scalar_hadamard() {
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let h = hadamard_matrix();

        apply_gate_scalar(&mut state, &h, 0, 1);

        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert_relative_eq!(state[0].re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(state[1].re, inv_sqrt2, epsilon = 1e-10);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_sse2_hadamard() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let h = hadamard_matrix();

        unsafe {
            apply_gate_sse2(&mut state, &h, 0, 1);
        }

        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert_relative_eq!(state[0].re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(state[1].re, inv_sqrt2, epsilon = 1e-10);
    }

    #[test]
    fn test_scalar_vs_simd() {
        let mut state_scalar = vec![
            Complex64::new(0.5, 0.1),
            Complex64::new(0.3, -0.2),
            Complex64::new(0.2, 0.4),
            Complex64::new(0.1, 0.3),
        ];
        let mut state_simd = state_scalar.clone();

        let h = hadamard_matrix();

        apply_gate_scalar(&mut state_scalar, &h, 0, 2);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                unsafe {
                    apply_gate_sse2(&mut state_simd, &h, 0, 2);
                }

                for i in 0..state_scalar.len() {
                    assert_relative_eq!(state_scalar[i].re, state_simd[i].re, epsilon = 1e-10);
                    assert_relative_eq!(state_scalar[i].im, state_simd[i].im, epsilon = 1e-10);
                }
            }
        }
    }
}
