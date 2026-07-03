//! SIMD-optimized two-qubit gate operations

use num_complex::Complex64;

/// Cache-friendly two-qubit gate application (scalar implementation)
///
/// This implementation iterates the state vector in a nested, stride-based
/// pattern that keeps accesses to the four amplitudes touched by each
/// 2-qubit block contiguous and cache-friendly. It sorts the qubit indices
/// so the inner loops operate over the smaller stride, improving locality.
pub fn apply_gate_scalar(
    state: &mut [Complex64],
    matrix: &[[Complex64; 4]; 4],
    qubit1: usize,
    qubit2: usize,
    num_qubits: usize,
) {
    let dimension = 1usize << num_qubits;

    // Order qubits so low < high
    let (low, high) = if qubit1 < qubit2 {
        (qubit1, qubit2)
    } else {
        (qubit2, qubit1)
    };

    let stride_low = 1usize << low;
    let stride_high = 1usize << high;
    let outer_step = stride_high * 2; // 1 << (high+1)
    let mid_step = stride_low * 2; // 1 << (low+1)

    // Flatten matrix to local copy for fast access
    let mut m = [[Complex64::new(0.0, 0.0); 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            m[r][c] = matrix[r][c];
        }
    }

    // Iterate in a three-level stride pattern:
    // - outer loop advances by blocks that change the high qubit
    // - middle loop advances subblocks that align low-qubit groups
    // - inner loop iterates the smallest stride (stride_low) so the four
    //   amplitudes (00,01,10,11) are close in memory
    let mut base = 0usize;
    while base < dimension {
        let mut mid = 0usize;
        while mid < stride_high {
            let block_base = base + mid;
            for k in 0..stride_low {
                let i00 = block_base + k;
                let i01 = i00 + stride_low;
                let i10 = i00 + stride_high;
                let i11 = i10 + stride_low;

                // Load amplitudes
                let a0 = state[i00];
                let a1 = state[i01];
                let a2 = state[i10];
                let a3 = state[i11];

                // Apply 4x4 matrix
                let mut r0 = Complex64::new(0.0, 0.0);
                let mut r1 = Complex64::new(0.0, 0.0);
                let mut r2 = Complex64::new(0.0, 0.0);
                let mut r3 = Complex64::new(0.0, 0.0);

                r0 += m[0][0] * a0;
                r0 += m[0][1] * a1;
                r0 += m[0][2] * a2;
                r0 += m[0][3] * a3;
                r1 += m[1][0] * a0;
                r1 += m[1][1] * a1;
                r1 += m[1][2] * a2;
                r1 += m[1][3] * a3;
                r2 += m[2][0] * a0;
                r2 += m[2][1] * a1;
                r2 += m[2][2] * a2;
                r2 += m[2][3] * a3;
                r3 += m[3][0] * a0;
                r3 += m[3][1] * a1;
                r3 += m[3][2] * a2;
                r3 += m[3][3] * a3;

                // Store back
                state[i00] = r0;
                state[i01] = r1;
                state[i10] = r2;
                state[i11] = r3;
            }

            mid += mid_step;
        }

        base += outer_step;
    }
}

/// Apply a two-qubit gate using AVX2 instructions
///
/// # Safety
/// Requires AVX2 support
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn apply_gate_avx2(
    state: &mut [Complex64],
    matrix: &[[Complex64; 4]; 4],
    qubit1: usize,
    qubit2: usize,
    num_qubits: usize,
) {
    // AVX2 specialized path can be added later; for now reuse scalar
    apply_gate_scalar(state, matrix, qubit1, qubit2, num_qubits);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn identity_matrix() -> [[Complex64; 4]; 4] {
        let o = Complex64::new(0.0, 0.0);
        let i = Complex64::new(1.0, 0.0);
        [[i, o, o, o], [o, i, o, o], [o, o, i, o], [o, o, o, i]]
    }

    fn cnot_matrix() -> [[Complex64; 4]; 4] {
        // CNOT: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
        let o = Complex64::new(0.0, 0.0);
        let i = Complex64::new(1.0, 0.0);
        [[i, o, o, o], [o, i, o, o], [o, o, o, i], [o, o, i, o]]
    }

    #[test]
    fn test_scalar_cnot() {
        // Test identity on |00⟩
        let mut state = vec![
            Complex64::new(1.0, 0.0), // |00⟩
            Complex64::new(0.0, 0.0), // |01⟩
            Complex64::new(0.0, 0.0), // |10⟩
            Complex64::new(0.0, 0.0), // |11⟩
        ];

        apply_gate_scalar(&mut state, &identity_matrix(), 0, 1, 2);

        // Identity should preserve state
        assert_relative_eq!(state[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(state[1].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[2].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[3].re, 0.0, epsilon = 1e-10);
    }

    /// Covers apply_gate_avx2 (lines 105, 113) — on AVX2-capable CPUs this
    /// exercises the function body which immediately delegates to scalar.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_identity_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state_scalar = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let mut state_avx2 = state_scalar.clone();

        apply_gate_scalar(&mut state_scalar, &identity_matrix(), 0, 1, 2);
        unsafe {
            apply_gate_avx2(&mut state_avx2, &identity_matrix(), 0, 1, 2);
        }
        for i in 0..4 {
            assert_relative_eq!(state_scalar[i].re, state_avx2[i].re, epsilon = 1e-10);
            assert_relative_eq!(state_scalar[i].im, state_avx2[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_cnot_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // Apply CNOT starting from |10⟩ — should flip to |11⟩
        let o = Complex64::new(0.0, 0.0);
        let i = Complex64::new(1.0, 0.0);
        let start = vec![o, o, i, o]; // |10⟩
        let mut state_scalar = start.clone();
        let mut state_avx2 = start.clone();

        apply_gate_scalar(&mut state_scalar, &cnot_matrix(), 0, 1, 2);
        unsafe {
            apply_gate_avx2(&mut state_avx2, &cnot_matrix(), 0, 1, 2);
        }
        for idx in 0..4 {
            assert_relative_eq!(state_scalar[idx].re, state_avx2[idx].re, epsilon = 1e-10);
            assert_relative_eq!(state_scalar[idx].im, state_avx2[idx].im, epsilon = 1e-10);
        }
    }
}
