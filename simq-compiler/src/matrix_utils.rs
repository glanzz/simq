//! Matrix utilities for gate composition
//!
//! Provides efficient matrix multiplication and comparison utilities
//! for quantum gate matrices.

use num_complex::Complex64;
use smallvec::{smallvec, SmallVec};

/// Inline-capacity buffer for flat row-major square matrices up to 8x8
/// (3-qubit blocks, 64 entries). Sized to cover this crate's fusion block
/// width cap so composing a fused gate's matrix never heap-allocates.
pub type FlatMatrix = SmallVec<[Complex64; 64]>;

/// Multiply two 2x2 complex matrices
///
/// Computes C = A * B where A and B are 2x2 matrices.
/// This is the fundamental operation for composing single-qubit gates.
///
/// # Arguments
/// * `a` - Left matrix (applied second in the gate sequence)
/// * `b` - Right matrix (applied first in the gate sequence)
///
/// # Returns
/// The product matrix C = A * B
///
/// # Example
/// ```
/// use num_complex::Complex64;
/// use simq_compiler::matrix_utils::multiply_2x2;
///
/// let identity = [[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
///                 [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]];
/// let result = multiply_2x2(&identity, &identity);
/// // result is the identity matrix
/// ```
#[inline]
pub fn multiply_2x2(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// Multiply two flat row-major `dim x dim` complex matrices: C = A * B
///
/// Generalizes [`multiply_2x2`] to arbitrary block widths (used for fused
/// multi-qubit gate blocks, `dim` up to 8 for a 3-qubit block). Panics (via
/// debug assertion) if `a`/`b` don't have exactly `dim * dim` elements.
///
/// # Arguments
/// * `a` - Left matrix (applied second in the gate sequence), flat row-major
/// * `b` - Right matrix (applied first in the gate sequence), flat row-major
/// * `dim` - Matrix dimension (2^k for a k-qubit block)
pub fn multiply_square(a: &[Complex64], b: &[Complex64], dim: usize) -> FlatMatrix {
    debug_assert_eq!(a.len(), dim * dim, "matrix A must be dim x dim");
    debug_assert_eq!(b.len(), dim * dim, "matrix B must be dim x dim");

    let mut result: FlatMatrix = smallvec![Complex64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..dim {
                sum += a[i * dim + k] * b[k * dim + j];
            }
            result[i * dim + j] = sum;
        }
    }
    result
}

/// Flat row-major `dim x dim` identity matrix.
pub fn identity_flat(dim: usize) -> FlatMatrix {
    let mut result: FlatMatrix = smallvec![Complex64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        result[i * dim + i] = Complex64::new(1.0, 0.0);
    }
    result
}

/// Check if a flat row-major `dim x dim` matrix is approximately identity.
///
/// Generalizes [`is_identity`] to arbitrary block widths.
pub fn is_identity_flat(matrix: &[Complex64], dim: usize, epsilon: f64) -> bool {
    debug_assert_eq!(matrix.len(), dim * dim, "matrix must be dim x dim");
    for i in 0..dim {
        for j in 0..dim {
            let expected = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            if (matrix[i * dim + j] - expected).norm() > epsilon {
                return false;
            }
        }
    }
    true
}

/// Convert a 2x2 matrix to a flattened vector in row-major order
///
/// This format is compatible with the Gate trait's matrix() method.
///
/// # Arguments
/// * `matrix` - 2x2 matrix to flatten
///
/// # Returns
/// Vector of length 4 containing matrix elements in row-major order
#[inline]
pub fn matrix_to_vec(matrix: &[[Complex64; 2]; 2]) -> Vec<Complex64> {
    vec![matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]]
}

/// Check if two matrices are approximately equal
///
/// Uses relative epsilon comparison for floating point values.
///
/// # Arguments
/// * `a` - First matrix
/// * `b` - Second matrix
/// * `epsilon` - Relative tolerance for comparison (default: 1e-10)
///
/// # Returns
/// True if matrices are approximately equal
#[inline]
pub fn matrices_approx_eq(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2], epsilon: f64) -> bool {
    for i in 0..2 {
        for j in 0..2 {
            let diff = (a[i][j] - b[i][j]).norm();
            if diff > epsilon {
                return false;
            }
        }
    }
    true
}

/// Check if a matrix is approximately the identity matrix
///
/// # Arguments
/// * `matrix` - Matrix to check
/// * `epsilon` - Relative tolerance (default: 1e-10)
///
/// # Returns
/// True if matrix is approximately identity
#[inline]
pub fn is_identity(matrix: &[[Complex64; 2]; 2], epsilon: f64) -> bool {
    const IDENTITY: [[Complex64; 2]; 2] = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    matrices_approx_eq(matrix, &IDENTITY, epsilon)
}

/// Compute the Frobenius norm of a matrix
///
/// The Frobenius norm is the square root of the sum of squares of all matrix elements.
///
/// # Arguments
/// * `matrix` - Matrix to compute norm of
///
/// # Returns
/// Frobenius norm of the matrix
#[inline]
pub fn frobenius_norm(matrix: &[[Complex64; 2]; 2]) -> f64 {
    let mut sum: f64 = 0.0;
    for row in matrix {
        for &element in row {
            sum += element.norm_sqr();
        }
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const ZERO: Complex64 = Complex64::new(0.0, 0.0);
    const ONE: Complex64 = Complex64::new(1.0, 0.0);
    const I: Complex64 = Complex64::new(0.0, 1.0);
    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

    #[test]
    fn test_multiply_identity() {
        let identity = [[ONE, ZERO], [ZERO, ONE]];
        let result = multiply_2x2(&identity, &identity);
        assert!(is_identity(&result, 1e-10));
    }

    #[test]
    fn test_multiply_pauli_x() {
        // X * X = I
        let pauli_x = [[ZERO, ONE], [ONE, ZERO]];
        let result = multiply_2x2(&pauli_x, &pauli_x);
        assert!(is_identity(&result, 1e-10));
    }

    #[test]
    fn test_multiply_hadamard() {
        // H * H = I
        let hadamard = [
            [
                Complex64::new(INV_SQRT2, 0.0),
                Complex64::new(INV_SQRT2, 0.0),
            ],
            [
                Complex64::new(INV_SQRT2, 0.0),
                Complex64::new(-INV_SQRT2, 0.0),
            ],
        ];
        let result = multiply_2x2(&hadamard, &hadamard);
        assert!(is_identity(&result, 1e-10));
    }

    #[test]
    fn test_multiply_s_gate() {
        // S * S = Z
        let s_gate = [[ONE, ZERO], [ZERO, I]];
        let pauli_z = [[ONE, ZERO], [ZERO, Complex64::new(-1.0, 0.0)]];
        let result = multiply_2x2(&s_gate, &s_gate);
        assert!(matrices_approx_eq(&result, &pauli_z, 1e-10));
    }

    #[test]
    fn test_matrix_to_vec() {
        let matrix = [[ONE, ZERO], [ZERO, ONE]];
        let vec = matrix_to_vec(&matrix);
        assert_eq!(vec.len(), 4);
        assert_eq!(vec[0], ONE);
        assert_eq!(vec[1], ZERO);
        assert_eq!(vec[2], ZERO);
        assert_eq!(vec[3], ONE);
    }

    #[test]
    fn test_frobenius_norm_identity() {
        let identity = [[ONE, ZERO], [ZERO, ONE]];
        let norm = frobenius_norm(&identity);
        assert_relative_eq!(norm, 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_matrices_approx_eq() {
        let a = [[ONE, ZERO], [ZERO, ONE]];
        let b = [
            [Complex64::new(1.0 + 1e-12, 0.0), ZERO],
            [ZERO, Complex64::new(1.0 - 1e-12, 0.0)],
        ];
        assert!(matrices_approx_eq(&a, &b, 1e-10));
    }

    #[test]
    fn test_is_identity() {
        let identity = [[ONE, ZERO], [ZERO, ONE]];
        assert!(is_identity(&identity, 1e-10));

        let not_identity = [[ONE, ONE], [ZERO, ONE]];
        assert!(!is_identity(&not_identity, 1e-10));
    }

    #[test]
    fn test_multiply_square_identity_4x4() {
        let id = identity_flat(4);
        let result = multiply_square(&id, &id, 4);
        assert!(is_identity_flat(&result, 4, 1e-10));
    }

    #[test]
    fn test_multiply_square_matches_multiply_2x2() {
        // X * H via multiply_square(dim=2) should match multiply_2x2(X, H)
        let x = [[ZERO, ONE], [ONE, ZERO]];
        let h = [
            [Complex64::new(INV_SQRT2, 0.0), Complex64::new(INV_SQRT2, 0.0)],
            [Complex64::new(INV_SQRT2, 0.0), Complex64::new(-INV_SQRT2, 0.0)],
        ];
        let expected = multiply_2x2(&x, &h);

        let x_flat = matrix_to_vec(&x);
        let h_flat = matrix_to_vec(&h);
        let result = multiply_square(&x_flat, &h_flat, 2);

        assert_relative_eq!(result[0].re, expected[0][0].re, epsilon = 1e-10);
        assert_relative_eq!(result[1].re, expected[0][1].re, epsilon = 1e-10);
        assert_relative_eq!(result[2].re, expected[1][0].re, epsilon = 1e-10);
        assert_relative_eq!(result[3].re, expected[1][1].re, epsilon = 1e-10);
    }

    #[test]
    fn test_identity_flat_dims() {
        let id2 = identity_flat(2);
        assert_eq!(id2.len(), 4);
        assert!(is_identity_flat(&id2, 2, 1e-10));

        let id8 = identity_flat(8);
        assert_eq!(id8.len(), 64);
        assert!(is_identity_flat(&id8, 8, 1e-10));
    }

    #[test]
    fn test_is_identity_flat_rejects_non_identity() {
        let mut not_identity = identity_flat(2);
        not_identity[1] = ONE;
        assert!(!is_identity_flat(&not_identity, 2, 1e-10));
    }
}
