//! Matrix utilities for gate composition
//!
//! Provides efficient matrix multiplication and comparison utilities
//! for quantum gate matrices.

use num_complex::Complex64;

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
pub fn multiply_2x2(
    a: &[[Complex64; 2]; 2],
    b: &[[Complex64; 2]; 2],
) -> [[Complex64; 2]; 2] {
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
    vec![
        matrix[0][0], matrix[0][1],
        matrix[1][0], matrix[1][1],
    ]
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
pub fn matrices_approx_eq(
    a: &[[Complex64; 2]; 2],
    b: &[[Complex64; 2]; 2],
    epsilon: f64,
) -> bool {
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
    const INV_SQRT2: f64 = 0.7071067811865476;

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
}
