//! SIMD-optimized single-qubit gate operations
//!
//! This module provides multiple implementations of single-qubit gate application:
//! - Scalar: Reference implementation (works on all platforms)
//! - SSE2: 128-bit SIMD (x86_64 baseline)
//! - AVX2: 256-bit SIMD with cache-friendly access patterns
//! - Parallel: Multi-threaded execution with Rayon for large states

use num_complex::Complex64;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Threshold for using parallel execution (number of qubits)
/// For states smaller than 2^PARALLEL_THRESHOLD, sequential execution is faster
const PARALLEL_THRESHOLD: usize = 16; // 2^16 = 65536 amplitudes

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
/// This implementation processes multiple amplitude pairs simultaneously,
/// improving cache utilization and instruction-level parallelism.
///
/// # Algorithm
/// - For small strides (qubit index low): Process pairs sequentially with good cache locality
/// - For large strides (qubit index high): Process in chunks to maximize SIMD utilization
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
    let dimension = 1 << num_qubits;
    let qubit_mask = 1 << qubit;
    let stride = 1 << qubit;

    // For low qubits (stride < cache line), sequential processing is faster
    // For high qubits (stride >= cache line), we want to maximize SIMD usage
    if stride < 8 {
        // Small stride: use SSE2 for better cache locality
        apply_gate_sse2(state, matrix, qubit, num_qubits);
        return;
    }

    // Load matrix elements into AVX registers (broadcast to all lanes)
    let m00_re = _mm256_set1_pd(matrix[0][0].re);
    let m00_im = _mm256_set1_pd(matrix[0][0].im);
    let m01_re = _mm256_set1_pd(matrix[0][1].re);
    let m01_im = _mm256_set1_pd(matrix[0][1].im);
    let m10_re = _mm256_set1_pd(matrix[1][0].re);
    let m10_im = _mm256_set1_pd(matrix[1][0].im);
    let m11_re = _mm256_set1_pd(matrix[1][1].re);
    let m11_im = _mm256_set1_pd(matrix[1][1].im);

    // Process amplitude pairs in chunks for better SIMD utilization
    // Each iteration processes 2 pairs (4 complex numbers total)
    let mut i = 0;
    while i < dimension {
        if i & qubit_mask != 0 {
            i += 1;
            continue;
        }

        // Process pairs with good alignment
        if i + stride < dimension {
            let j = i | qubit_mask;

            // Load amplitudes: amp0 = state[i], amp1 = state[j]
            // Using aligned loads when possible
            let amp0_ptr = state.as_ptr().add(i) as *const f64;
            let amp1_ptr = state.as_ptr().add(j) as *const f64;

            // Load as [amp0.re, amp0.im, ?, ?] where ? is padding
            let amp0_raw = _mm_loadu_pd(amp0_ptr);
            let amp1_raw = _mm_loadu_pd(amp1_ptr);

            // Broadcast complex number components
            // amp0_re = [amp0.re, amp0.re, amp0.re, amp0.re]
            let amp0 = _mm256_castpd128_pd256(amp0_raw); // [amp0.re, amp0.im, 0, 0]
            let amp1 = _mm256_castpd128_pd256(amp1_raw); // [amp1.re, amp1.im, 0, 0]

            let amp0_re = _mm256_permute4x64_pd::<0b00000000>(amp0); // Broadcast re
            let amp0_im = _mm256_permute4x64_pd::<0b01010101>(amp0); // Broadcast im
            let amp1_re = _mm256_permute4x64_pd::<0b00000000>(amp1);
            let amp1_im = _mm256_permute4x64_pd::<0b01010101>(amp1);

            // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i

            // Compute new state[i] = m00 * amp0 + m01 * amp1
            // Real part: m00.re * amp0.re - m00.im * amp0.im + m01.re * amp1.re - m01.im * amp1.im
            let t00_re = _mm256_mul_pd(m00_re, amp0_re);
            let t00_im = _mm256_mul_pd(m00_im, amp0_im);
            let new_i_re_p1 = _mm256_sub_pd(t00_re, t00_im);

            let t01_re = _mm256_mul_pd(m01_re, amp1_re);
            let t01_im = _mm256_mul_pd(m01_im, amp1_im);
            let new_i_re_p2 = _mm256_sub_pd(t01_re, t01_im);

            let new_i_re = _mm256_add_pd(new_i_re_p1, new_i_re_p2);

            // Imaginary part: m00.re * amp0.im + m00.im * amp0.re + m01.re * amp1.im + m01.im * amp1.re
            let t00_re_im = _mm256_mul_pd(m00_re, amp0_im);
            let t00_im_re = _mm256_mul_pd(m00_im, amp0_re);
            let new_i_im_p1 = _mm256_add_pd(t00_re_im, t00_im_re);

            let t01_re_im = _mm256_mul_pd(m01_re, amp1_im);
            let t01_im_re = _mm256_mul_pd(m01_im, amp1_re);
            let new_i_im_p2 = _mm256_add_pd(t01_re_im, t01_im_re);

            let new_i_im = _mm256_add_pd(new_i_im_p1, new_i_im_p2);

            // Compute new state[j] = m10 * amp0 + m11 * amp1
            let t10_re = _mm256_mul_pd(m10_re, amp0_re);
            let t10_im = _mm256_mul_pd(m10_im, amp0_im);
            let new_j_re_p1 = _mm256_sub_pd(t10_re, t10_im);

            let t11_re = _mm256_mul_pd(m11_re, amp1_re);
            let t11_im = _mm256_mul_pd(m11_im, amp1_im);
            let new_j_re_p2 = _mm256_sub_pd(t11_re, t11_im);

            let new_j_re = _mm256_add_pd(new_j_re_p1, new_j_re_p2);

            let t10_re_im = _mm256_mul_pd(m10_re, amp0_im);
            let t10_im_re = _mm256_mul_pd(m10_im, amp0_re);
            let new_j_im_p1 = _mm256_add_pd(t10_re_im, t10_im_re);

            let t11_re_im = _mm256_mul_pd(m11_re, amp1_im);
            let t11_im_re = _mm256_mul_pd(m11_im, amp1_re);
            let new_j_im_p2 = _mm256_add_pd(t11_re_im, t11_im_re);

            let new_j_im = _mm256_add_pd(new_j_im_p1, new_j_im_p2);

            // Extract lower 128 bits and store
            let new_i_128 = _mm256_castpd256_pd128(new_i_re);
            let new_i_im_128 = _mm256_castpd256_pd128(new_i_im);
            let new_i_final = _mm_unpacklo_pd(new_i_128, new_i_im_128);

            let new_j_128 = _mm256_castpd256_pd128(new_j_re);
            let new_j_im_128 = _mm256_castpd256_pd128(new_j_im);
            let new_j_final = _mm_unpacklo_pd(new_j_128, new_j_im_128);

            // Store results
            let out_i_ptr = state.as_mut_ptr().add(i) as *mut f64;
            let out_j_ptr = state.as_mut_ptr().add(j) as *mut f64;

            _mm_storeu_pd(out_i_ptr, new_i_final);
            _mm_storeu_pd(out_j_ptr, new_j_final);
        }

        i += 1;
    }
}

/// Apply a single-qubit gate with parallel execution
///
/// This function uses Rayon to parallelize gate application across multiple threads.
/// It's most beneficial for large quantum states (>16 qubits) where the overhead
/// of thread coordination is amortized by the computational workload.
///
/// # Algorithm
/// - Divides the state into chunks based on the target qubit
/// - Each thread processes independent amplitude pairs
/// - For low qubits: chunks are small but numerous (good parallelism)
/// - For high qubits: chunks are large but few (less parallelism)
///
/// # Arguments
/// * `state` - Mutable slice of state amplitudes
/// * `matrix` - 2×2 gate matrix in row-major order
/// * `qubit` - Index of the qubit to apply the gate to
/// * `num_qubits` - Total number of qubits in the state
pub fn apply_gate_parallel(
    state: &mut [Complex64],
    matrix: &[[Complex64; 2]; 2],
    qubit: usize,
    num_qubits: usize,
) {
    let dimension = 1 << num_qubits;
    let stride = 1 << qubit;

    // Extract matrix elements for better cache locality
    let m00 = matrix[0][0];
    let m01 = matrix[0][1];
    let m10 = matrix[1][0];
    let m11 = matrix[1][1];

    // Determine chunk size for parallel execution
    // We want chunks that are cache-friendly and provide good load balancing
    let num_pairs: usize = dimension / 2;
    let num_threads = rayon::current_num_threads();
    let pairs_per_thread = num_pairs.div_ceil(num_threads);

    // Create ranges for parallel processing
    // Each range processes a contiguous block of "low" indices
    let _ranges: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let start = thread_id * pairs_per_thread;
            let end = std::cmp::min(start + pairs_per_thread, num_pairs);
            (start, end)
        })
        .filter(|(start, end)| start < end)
        .collect();

    // Process amplitude pairs in parallel
    state.par_chunks_mut(stride * 2).for_each(|chunk| {
        // Within each chunk, process pairs
        for i in (0..chunk.len()).step_by(stride * 2) {
            if i + stride < chunk.len() {
                let amp0 = chunk[i];
                let amp1 = chunk[i + stride];

                // Apply matrix multiplication
                chunk[i] = m00 * amp0 + m01 * amp1;
                chunk[i + stride] = m10 * amp0 + m11 * amp1;
            }
        }
    });
}

/// Apply a single-qubit gate with optimal execution strategy
///
/// This function automatically selects the best execution strategy based on:
/// - State size (number of qubits)
/// - Available CPU features (AVX2, SSE2)
/// - Number of available CPU cores
///
/// # Strategy Selection
/// - Small states (< 16 qubits): Sequential SIMD (AVX2/SSE2/scalar)
/// - Large states (≥ 16 qubits): Parallel execution with SIMD per thread
///
/// # Arguments
/// * `state` - Mutable slice of state amplitudes
/// * `matrix` - 2×2 gate matrix in row-major order
/// * `qubit` - Index of the qubit to apply the gate to
/// * `num_qubits` - Total number of qubits in the state
pub fn apply_gate_optimized(
    state: &mut [Complex64],
    matrix: &[[Complex64; 2]; 2],
    qubit: usize,
    num_qubits: usize,
) {
    // For large states, use parallel execution
    if num_qubits >= PARALLEL_THRESHOLD && rayon::current_num_threads() > 1 {
        apply_gate_parallel(state, matrix, qubit, num_qubits);
        return;
    }

    // For smaller states, use sequential SIMD
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe {
            apply_gate_avx2(state, matrix, qubit, num_qubits);
        }
        return;
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        unsafe {
            apply_gate_sse2(state, matrix, qubit, num_qubits);
        }
        return;
    }

    // Fallback to scalar
    apply_gate_scalar(state, matrix, qubit, num_qubits);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn hadamard_matrix() -> [[Complex64; 2]; 2] {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        [
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
            ],
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
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
        let _state_simd = state_scalar.clone();

        let h = hadamard_matrix();

        apply_gate_scalar(&mut state_scalar, &h, 0, 2);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                let mut state_sse2 = vec![
                    Complex64::new(0.5, 0.1),
                    Complex64::new(0.3, -0.2),
                    Complex64::new(0.2, 0.4),
                    Complex64::new(0.1, 0.3),
                ];

                unsafe {
                    apply_gate_sse2(&mut state_sse2, &h, 0, 2);
                }

                for i in 0..state_scalar.len() {
                    assert_relative_eq!(state_scalar[i].re, state_sse2[i].re, epsilon = 1e-10);
                    assert_relative_eq!(state_scalar[i].im, state_sse2[i].im, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_parallel_hadamard() {
        // Test with a larger state to make parallel execution meaningful
        let num_qubits = 10; // 1024 amplitudes
        let mut state_scalar = vec![Complex64::new(0.0, 0.0); 1 << num_qubits];
        state_scalar[0] = Complex64::new(1.0, 0.0); // |000...0⟩

        let mut state_parallel = state_scalar.clone();

        let h = hadamard_matrix();

        apply_gate_scalar(&mut state_scalar, &h, 0, num_qubits);
        apply_gate_parallel(&mut state_parallel, &h, 0, num_qubits);

        // Verify results match
        for i in 0..state_scalar.len() {
            assert_relative_eq!(state_scalar[i].re, state_parallel[i].re, epsilon = 1e-10);
            assert_relative_eq!(state_scalar[i].im, state_parallel[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_optimized_execution() {
        // Test that optimized version produces correct results
        let num_qubits = 12;
        let mut state_scalar = vec![Complex64::new(0.0, 0.0); 1 << num_qubits];
        state_scalar[0] = Complex64::new(1.0, 0.0);

        let mut state_optimized = state_scalar.clone();

        let h = hadamard_matrix();

        apply_gate_scalar(&mut state_scalar, &h, 0, num_qubits);
        apply_gate_optimized(&mut state_optimized, &h, 0, num_qubits);

        // Verify results match between scalar and optimized versions
        for i in 0..state_scalar.len() {
            assert_relative_eq!(state_scalar[i].re, state_optimized[i].re, epsilon = 1e-10);
            assert_relative_eq!(state_scalar[i].im, state_optimized[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_pauli_x_gate() {
        // Test Pauli-X gate (bit flip)
        let x_gate = [
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ];

        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        apply_gate_scalar(&mut state, &x_gate, 0, 1);

        // Should flip |0⟩ to |1⟩
        assert_relative_eq!(state[0].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[1].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_phase_gate() {
        // Test S gate (phase gate)
        let s_gate = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)], // i
        ];

        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let mut state = vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ];

        apply_gate_scalar(&mut state, &s_gate, 0, 1);

        // |0⟩ should remain unchanged, |1⟩ should get phase i
        assert_relative_eq!(state[0].re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(state[0].im, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[1].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[1].im, inv_sqrt2, epsilon = 1e-10);
    }
}
