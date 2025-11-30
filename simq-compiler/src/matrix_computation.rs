//! Gate matrix computation utilities
//!
//! This module provides advanced matrix computation operations for quantum gates,
//! including tensor products, controlled gate generation, matrix exponentiation,
//! and decomposition utilities.
//!
//! # Features
//!
//! - **Tensor Product**: Compute Kronecker products of gate matrices
//! - **Controlled Gates**: Generate controlled versions of any gate
//! - **Matrix Exponentiation**: Compute e^(iθM) for gate matrices
//! - **Unitary Verification**: Check if matrices are unitary
//! - **Gate Composition**: Multiply and combine gate matrices
//! - **Decomposition**: Tools for decomposing arbitrary unitaries
//!
//! # Example
//!
//! ```ignore
//! use simq_compiler::matrix_computation::*;
//! use num_complex::Complex64;
//!
//! // Compute tensor product of two gates
//! let h = hadamard_matrix();
//! let x = pauli_x_matrix();
//! let h_tensor_x = tensor_product_2x2(&h, &x);
//!
//! // Generate controlled version of a gate
//! let cx = controlled_gate_2x2(&x);
//! ```

use num_complex::Complex64;
use simq_core::{QuantumError, Result};

// Constants
const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);
const I: Complex64 = Complex64::new(0.0, 1.0);
const EPSILON: f64 = 1e-10;

// ============================================================================
// Matrix Type Definitions
// ============================================================================

/// 2×2 complex matrix (single-qubit gate)
pub type Matrix2 = [[Complex64; 2]; 2];

/// 4×4 complex matrix (two-qubit gate)
pub type Matrix4 = [[Complex64; 4]; 4];

/// 8×8 complex matrix (three-qubit gate)
pub type Matrix8 = [[Complex64; 8]; 8];

/// Dynamic complex matrix (arbitrary size)
pub type DynamicMatrix = Vec<Vec<Complex64>>;

// ============================================================================
// Tensor Product (Kronecker Product)
// ============================================================================

/// Compute tensor product of two 2×2 matrices
///
/// Given matrices A and B, computes A ⊗ B, resulting in a 4×4 matrix.
///
/// # Example
///
/// ```ignore
/// let h = hadamard_matrix();
/// let x = pauli_x_matrix();
/// let h_x = tensor_product_2x2(&h, &x);  // H ⊗ X
/// ```
pub fn tensor_product_2x2(a: &Matrix2, b: &Matrix2) -> Matrix4 {
    let mut result = [[ZERO; 4]; 4];

    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    let row = i * 2 + k;
                    let col = j * 2 + l;
                    result[row][col] = a[i][j] * b[k][l];
                }
            }
        }
    }

    result
}

/// Compute tensor product of two 4×4 matrices
///
/// Given matrices A and B, computes A ⊗ B, resulting in an 8×8 matrix.
pub fn tensor_product_4x4(a: &Matrix4, b: &Matrix2) -> Matrix8 {
    let mut result = [[ZERO; 8]; 8];

    for i in 0..4 {
        for j in 0..4 {
            for k in 0..2 {
                for l in 0..2 {
                    let row = i * 2 + k;
                    let col = j * 2 + l;
                    result[row][col] = a[i][j] * b[k][l];
                }
            }
        }
    }

    result
}

/// Compute tensor product of arbitrary-sized matrices
///
/// General implementation for dynamic matrices.
pub fn tensor_product_dynamic(a: &DynamicMatrix, b: &DynamicMatrix) -> Result<DynamicMatrix> {
    let a_rows = a.len();
    let a_cols = a.first().map(|row| row.len()).unwrap_or(0);
    let b_rows = b.len();
    let b_cols = b.first().map(|row| row.len()).unwrap_or(0);

    if a_rows == 0 || a_cols == 0 || b_rows == 0 || b_cols == 0 {
        return Err(QuantumError::ValidationError("Empty matrix".to_string()));
    }

    let result_rows = a_rows * b_rows;
    let result_cols = a_cols * b_cols;

    let mut result = vec![vec![ZERO; result_cols]; result_rows];

    for i in 0..a_rows {
        for j in 0..a_cols {
            for k in 0..b_rows {
                for l in 0..b_cols {
                    let row = i * b_rows + k;
                    let col = j * b_cols + l;
                    result[row][col] = a[i][j] * b[k][l];
                }
            }
        }
    }

    Ok(result)
}

// ============================================================================
// Controlled Gate Generation
// ============================================================================

/// Generate controlled version of a 2×2 gate (produces 4×4 matrix)
///
/// Creates a controlled-U gate where the control qubit is the first qubit.
/// If control is |0⟩, applies identity to target.
/// If control is |1⟩, applies U to target.
///
/// # Example
///
/// ```ignore
/// let x = pauli_x_matrix();
/// let cnot = controlled_gate_2x2(&x);  // CNOT gate
/// ```
pub fn controlled_gate_2x2(gate: &Matrix2) -> Matrix4 {
    [
        [ONE, ZERO, ZERO, ZERO],
        [ZERO, ONE, ZERO, ZERO],
        [ZERO, ZERO, gate[0][0], gate[0][1]],
        [ZERO, ZERO, gate[1][0], gate[1][1]],
    ]
}

/// Generate controlled version of a 4×4 gate (produces 8×8 matrix)
///
/// Creates a controlled-U gate where the control qubit is the first qubit.
pub fn controlled_gate_4x4(gate: &Matrix4) -> Matrix8 {
    let mut result = [[ZERO; 8]; 8];

    // Identity on first 4×4 block (control = 0)
    for i in 0..4 {
        result[i][i] = ONE;
    }

    // Gate on second 4×4 block (control = 1)
    for i in 0..4 {
        for j in 0..4 {
            result[i + 4][j + 4] = gate[i][j];
        }
    }

    result
}

/// Generate doubly-controlled gate (Toffoli-style)
///
/// Creates a CC-U gate with two control qubits.
/// Applies U only when both controls are |1⟩.
pub fn doubly_controlled_gate_2x2(gate: &Matrix2) -> Matrix8 {
    let mut result = [[ZERO; 8]; 8];

    // Identity on first 6 diagonal elements
    for i in 0..6 {
        result[i][i] = ONE;
    }

    // Apply gate to last 2×2 block (both controls = 1)
    result[6][6] = gate[0][0];
    result[6][7] = gate[0][1];
    result[7][6] = gate[1][0];
    result[7][7] = gate[1][1];

    result
}

// ============================================================================
// Matrix Multiplication
// ============================================================================

/// Multiply two 2×2 matrices
pub fn multiply_2x2(a: &Matrix2, b: &Matrix2) -> Matrix2 {
    let mut result = [[ZERO; 2]; 2];

    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

/// Multiply two 4×4 matrices
pub fn multiply_4x4(a: &Matrix4, b: &Matrix4) -> Matrix4 {
    let mut result = [[ZERO; 4]; 4];

    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

/// Multiply two 8×8 matrices
pub fn multiply_8x8(a: &Matrix8, b: &Matrix8) -> Matrix8 {
    let mut result = [[ZERO; 8]; 8];

    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

// ============================================================================
// Matrix Properties
// ============================================================================

/// Compute adjoint (conjugate transpose) of 2×2 matrix
pub fn adjoint_2x2(m: &Matrix2) -> Matrix2 {
    [
        [m[0][0].conj(), m[1][0].conj()],
        [m[0][1].conj(), m[1][1].conj()],
    ]
}

/// Compute adjoint of 4×4 matrix
pub fn adjoint_4x4(m: &Matrix4) -> Matrix4 {
    let mut result = [[ZERO; 4]; 4];

    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = m[j][i].conj();
        }
    }

    result
}

/// Check if 2×2 matrix is unitary (U†U = I)
pub fn is_unitary_2x2(m: &Matrix2) -> bool {
    let m_dag = adjoint_2x2(m);
    let product = multiply_2x2(&m_dag, m);

    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { ONE } else { ZERO };
            if (product[i][j] - expected).norm() > EPSILON {
                return false;
            }
        }
    }

    true
}

/// Check if 4×4 matrix is unitary
pub fn is_unitary_4x4(m: &Matrix4) -> bool {
    let m_dag = adjoint_4x4(m);
    let product = multiply_4x4(&m_dag, m);

    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { ONE } else { ZERO };
            if (product[i][j] - expected).norm() > EPSILON {
                return false;
            }
        }
    }

    true
}

/// Check if 2×2 matrix is Hermitian (M = M†)
pub fn is_hermitian_2x2(m: &Matrix2) -> bool {
    let m_dag = adjoint_2x2(m);

    for i in 0..2 {
        for j in 0..2 {
            if (m[i][j] - m_dag[i][j]).norm() > EPSILON {
                return false;
            }
        }
    }

    true
}

/// Compute trace of 2×2 matrix
pub fn trace_2x2(m: &Matrix2) -> Complex64 {
    m[0][0] + m[1][1]
}

/// Compute trace of 4×4 matrix
pub fn trace_4x4(m: &Matrix4) -> Complex64 {
    let mut tr = ZERO;
    for i in 0..4 {
        tr += m[i][i];
    }
    tr
}

/// Compute determinant of 2×2 matrix
pub fn determinant_2x2(m: &Matrix2) -> Complex64 {
    m[0][0] * m[1][1] - m[0][1] * m[1][0]
}

// ============================================================================
// Matrix Exponentiation
// ============================================================================

/// Compute matrix exponential exp(M) for 2×2 matrix using Taylor series
///
/// For small matrices, uses the series: exp(M) = I + M + M²/2! + M³/3! + ...
///
/// # Arguments
/// * `m` - Input matrix
/// * `terms` - Number of Taylor series terms (default: 20)
pub fn matrix_exp_2x2(m: &Matrix2, terms: usize) -> Matrix2 {
    let mut result = [[ONE, ZERO], [ZERO, ONE]]; // Identity
    let mut term = [[ONE, ZERO], [ZERO, ONE]]; // Current term
    let mut factorial = 1.0;

    for n in 1..=terms {
        term = multiply_2x2(&term, m);
        factorial *= n as f64;

        for i in 0..2 {
            for j in 0..2 {
                result[i][j] += term[i][j] / factorial;
            }
        }
    }

    result
}

/// Compute matrix exponential exp(iθM) for Hermitian 2×2 matrix
///
/// For Hermitian matrices (like Pauli matrices), we can use eigenvalue decomposition
/// for more efficient and accurate computation.
pub fn matrix_exp_hermitian_2x2(m: &Matrix2, theta: f64) -> Matrix2 {
    // For small angles, use direct computation
    if theta.abs() < EPSILON {
        return [[ONE, ZERO], [ZERO, ONE]];
    }

    // Scale matrix by iθ
    let scaled = [
        [
            m[0][0] * Complex64::new(0.0, theta),
            m[0][1] * Complex64::new(0.0, theta),
        ],
        [
            m[1][0] * Complex64::new(0.0, theta),
            m[1][1] * Complex64::new(0.0, theta),
        ],
    ];

    matrix_exp_2x2(&scaled, 20)
}

// ============================================================================
// Common Gate Matrices (for convenience)
// ============================================================================

/// Identity matrix
pub fn identity_2x2() -> Matrix2 {
    [[ONE, ZERO], [ZERO, ONE]]
}

/// Pauli-X matrix
pub fn pauli_x_matrix() -> Matrix2 {
    [[ZERO, ONE], [ONE, ZERO]]
}

/// Pauli-Y matrix
pub fn pauli_y_matrix() -> Matrix2 {
    [[ZERO, -I], [I, ZERO]]
}

/// Pauli-Z matrix
pub fn pauli_z_matrix() -> Matrix2 {
    [[ONE, ZERO], [ZERO, Complex64::new(-1.0, 0.0)]]
}

/// Hadamard matrix
pub fn hadamard_matrix() -> Matrix2 {
    let inv_sqrt2 = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
    [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]]
}

// ============================================================================
// Gate Decomposition Utilities
// ============================================================================

/// Decompose 2×2 unitary into Euler angles (ZYZ decomposition)
///
/// Any single-qubit unitary can be written as:
/// U = e^(iα) Rz(β) Ry(γ) Rz(δ)
///
/// Returns (alpha, beta, gamma, delta)
pub fn decompose_zyz(u: &Matrix2) -> Result<(f64, f64, f64, f64)> {
    if !is_unitary_2x2(u) {
        return Err(QuantumError::ValidationError("Matrix is not unitary".to_string()));
    }

    // Extract global phase
    let det = determinant_2x2(u);
    let alpha = det.arg() / 2.0;

    // Remove global phase
    let phase_factor = Complex64::new(0.0, -alpha).exp();
    let u_normalized = [
        [u[0][0] * phase_factor, u[0][1] * phase_factor],
        [u[1][0] * phase_factor, u[1][1] * phase_factor],
    ];

    // Compute Euler angles
    let gamma = 2.0 * u_normalized[0][0].norm().acos();

    let beta = if gamma.abs() < EPSILON {
        0.0
    } else {
        let sin_half_gamma = (gamma / 2.0).sin();
        (u_normalized[1][0] / Complex64::new(0.0, sin_half_gamma)).arg()
    };

    let delta = if gamma.abs() < EPSILON {
        (u_normalized[0][1] / Complex64::new(0.0, 1.0)).arg()
    } else {
        let sin_half_gamma = (gamma / 2.0).sin();
        (u_normalized[0][1] / Complex64::new(0.0, -sin_half_gamma)).arg()
    };

    Ok((alpha, beta, gamma, delta))
}

/// Compute fidelity between two 2×2 unitary matrices
///
/// Fidelity: F = |Tr(U†V)| / 2
/// Returns a value between 0 and 1, where 1 means identical gates.
pub fn gate_fidelity_2x2(u: &Matrix2, v: &Matrix2) -> f64 {
    let u_dag = adjoint_2x2(u);
    let product = multiply_2x2(&u_dag, v);
    let tr = trace_2x2(&product);
    (tr.norm() / 2.0).min(1.0)
}

/// Check if two matrices are approximately equal
pub fn matrices_equal_2x2(a: &Matrix2, b: &Matrix2, epsilon: f64) -> bool {
    for i in 0..2 {
        for j in 0..2 {
            if (a[i][j] - b[i][j]).norm() > epsilon {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tensor_product_identity() {
        let id = identity_2x2();
        let x = pauli_x_matrix();

        let result = tensor_product_2x2(&id, &x);

        // I ⊗ X should have X in each 2×2 block
        assert_eq!(result[0][1], ONE);
        assert_eq!(result[1][0], ONE);
        assert_eq!(result[2][3], ONE);
        assert_eq!(result[3][2], ONE);
    }

    #[test]
    fn test_controlled_gate() {
        let x = pauli_x_matrix();
        let cnot = controlled_gate_2x2(&x);

        // CNOT should leave |00⟩ and |01⟩ unchanged
        assert_eq!(cnot[0][0], ONE);
        assert_eq!(cnot[1][1], ONE);

        // CNOT should flip |10⟩ ↔ |11⟩
        assert_eq!(cnot[2][3], ONE);
        assert_eq!(cnot[3][2], ONE);
    }

    #[test]
    fn test_matrix_multiplication() {
        let x = pauli_x_matrix();
        let x_squared = multiply_2x2(&x, &x);

        // X² = I
        assert!(matrices_equal_2x2(&x_squared, &identity_2x2(), EPSILON));
    }

    #[test]
    fn test_unitary_verification() {
        let h = hadamard_matrix();
        assert!(is_unitary_2x2(&h));

        let x = pauli_x_matrix();
        assert!(is_unitary_2x2(&x));

        let non_unitary = [[ONE, ONE], [ZERO, ONE]];
        assert!(!is_unitary_2x2(&non_unitary));
    }

    #[test]
    fn test_hermitian_verification() {
        let x = pauli_x_matrix();
        assert!(is_hermitian_2x2(&x));

        let y = pauli_y_matrix();
        assert!(is_hermitian_2x2(&y));

        let z = pauli_z_matrix();
        assert!(is_hermitian_2x2(&z));
    }

    #[test]
    fn test_trace() {
        let id = identity_2x2();
        let tr = trace_2x2(&id);
        assert_relative_eq!(tr.re, 2.0, epsilon = EPSILON);

        let z = pauli_z_matrix();
        let tr_z = trace_2x2(&z);
        assert_relative_eq!(tr_z.re, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_determinant() {
        let id = identity_2x2();
        let det = determinant_2x2(&id);
        assert_relative_eq!(det.re, 1.0, epsilon = EPSILON);

        let x = pauli_x_matrix();
        let det_x = determinant_2x2(&x);
        assert_relative_eq!(det_x.re, -1.0, epsilon = EPSILON);
    }

    #[test]
    fn test_adjoint() {
        let x = pauli_x_matrix();
        let x_dag = adjoint_2x2(&x);

        // X is Hermitian, so X† = X
        assert!(matrices_equal_2x2(&x, &x_dag, EPSILON));
    }

    #[test]
    fn test_gate_fidelity() {
        let x = pauli_x_matrix();
        let fidelity = gate_fidelity_2x2(&x, &x);
        assert_relative_eq!(fidelity, 1.0, epsilon = EPSILON);

        let h = hadamard_matrix();
        let fidelity_xh = gate_fidelity_2x2(&x, &h);
        assert!(fidelity_xh < 1.0);
        assert!(fidelity_xh > 0.0);
    }

    #[test]
    fn test_doubly_controlled_gate() {
        let x = pauli_x_matrix();
        let ccx = doubly_controlled_gate_2x2(&x);

        // Toffoli should be identity on first 6 basis states
        for i in 0..6 {
            assert_eq!(ccx[i][i], ONE);
        }

        // Should flip |110⟩ ↔ |111⟩
        assert_eq!(ccx[6][7], ONE);
        assert_eq!(ccx[7][6], ONE);
    }

    #[test]
    fn test_matrix_exp_identity() {
        let zero_matrix = [[ZERO, ZERO], [ZERO, ZERO]];
        let exp_zero = matrix_exp_2x2(&zero_matrix, 10);

        // exp(0) = I
        assert!(matrices_equal_2x2(&exp_zero, &identity_2x2(), EPSILON));
    }
}
