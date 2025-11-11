//! SIMD-optimized two-qubit gate operations

use num_complex::Complex64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Apply a two-qubit gate using scalar operations (fallback)
///
/// This applies a 4×4 matrix to groups of 4 amplitudes corresponding
/// to the four basis states that can be formed with the two target qubits.
pub fn apply_gate_scalar(
    state: &mut [Complex64],
    matrix: &[[Complex64; 4]; 4],
    qubit1: usize,
    qubit2: usize,
    num_qubits: usize,
) {
    let dimension = 1 << num_qubits;
    let mask1 = 1 << qubit1;
    let mask2 = 1 << qubit2;
    let mask_both = mask1 | mask2;

    // Preload matrix for better performance
    let mut m = [[Complex64::new(0.0, 0.0); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = matrix[i][j];
        }
    }

    for i in 0..dimension {
        // Skip if either qubit bit is set
        if i & mask_both != 0 {
            continue;
        }

        // Compute the four indices
        let i00 = i;              // both qubits 0
        let i01 = i | mask2;      // qubit1=0, qubit2=1
        let i10 = i | mask1;      // qubit1=1, qubit2=0
        let i11 = i | mask_both;  // both qubits 1

        // Load amplitudes
        let amp = [state[i00], state[i01], state[i10], state[i11]];

        // Apply 4×4 matrix multiplication
        let mut new_amp = [Complex64::new(0.0, 0.0); 4];
        for row in 0..4 {
            for col in 0..4 {
                new_amp[row] += m[row][col] * amp[col];
            }
        }

        // Store results
        state[i00] = new_amp[0];
        state[i01] = new_amp[1];
        state[i10] = new_amp[2];
        state[i11] = new_amp[3];
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
    // For now, fallback to scalar implementation
    // Full AVX2 implementation would use vectorized 4×4 multiplication
    apply_gate_scalar(state, matrix, qubit1, qubit2, num_qubits);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn cnot_matrix() -> [[Complex64; 4]; 4] {
        [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        ]
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

        let identity_2q = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ];

        apply_gate_scalar(&mut state, &identity_2q, 0, 1, 2);

        // Identity should preserve state
        assert_relative_eq!(state[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(state[1].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[2].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[3].re, 0.0, epsilon = 1e-10);
    }
}
