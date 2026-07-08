//! Gate matrix utilities

use super::{Matrix2x2, Matrix4x4};
use num_complex::Complex64;

/// Gate matrix wrapper with metadata
#[derive(Debug, Clone)]
pub struct GateMatrix {
    /// The matrix data
    pub data: GateMatrixData,
    /// Whether the gate is diagonal
    pub is_diagonal: bool,
    /// Whether the gate is unitary (for validation)
    pub is_unitary: bool,
}

/// Gate matrix data (different sizes for different gate types)
#[derive(Debug, Clone)]
pub enum GateMatrixData {
    /// 2x2 matrix for single-qubit gates
    Single(Matrix2x2),
    /// 4x4 matrix for two-qubit gates
    Two(Matrix4x4),
    /// Diagonal gate (just the diagonal elements)
    Diagonal(Vec<Complex64>),
}

impl GateMatrix {
    /// Create a single-qubit gate matrix
    pub fn single(matrix: Matrix2x2) -> Self {
        Self {
            data: GateMatrixData::Single(matrix),
            is_diagonal: is_diagonal_2x2(&matrix),
            is_unitary: true, // Assume unitary, validate separately if needed
        }
    }

    /// Create a two-qubit gate matrix
    pub fn two(matrix: Matrix4x4) -> Self {
        Self {
            data: GateMatrixData::Two(matrix),
            is_diagonal: is_diagonal_4x4(&matrix),
            is_unitary: true,
        }
    }

    /// Create a diagonal gate
    pub fn diagonal(diagonal: Vec<Complex64>) -> Self {
        Self {
            data: GateMatrixData::Diagonal(diagonal),
            is_diagonal: true,
            is_unitary: true,
        }
    }

    /// Get the number of qubits this gate acts on
    pub fn num_qubits(&self) -> usize {
        match &self.data {
            GateMatrixData::Single(_) => 1,
            GateMatrixData::Two(_) => 2,
            GateMatrixData::Diagonal(d) => (d.len() as f64).log2() as usize,
        }
    }
}

/// Check if a 2x2 matrix is diagonal
fn is_diagonal_2x2(matrix: &Matrix2x2) -> bool {
    matrix[0][1].norm_sqr() < 1e-15 && matrix[1][0].norm_sqr() < 1e-15
}

/// Check if a 4x4 matrix is diagonal
#[allow(clippy::needless_range_loop)]
fn is_diagonal_4x4(matrix: &Matrix4x4) -> bool {
    for i in 0..4 {
        for j in 0..4 {
            if i != j && matrix[i][j].norm_sqr() > 1e-15 {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_matrix_single_diagonal() {
        // Identity is diagonal
        let identity = common::identity();
        let gm = GateMatrix::single(identity);
        assert!(gm.is_diagonal);
        assert_eq!(gm.num_qubits(), 1);
    }

    #[test]
    fn test_gate_matrix_single_non_diagonal() {
        // Pauli X is not diagonal
        let x = common::pauli_x();
        let gm = GateMatrix::single(x);
        assert!(!gm.is_diagonal);
    }

    #[test]
    fn test_gate_matrix_two_non_diagonal() {
        // CNOT-like matrix is not diagonal
        let mut cnot: Matrix4x4 = [[Complex64::new(0.0, 0.0); 4]; 4];
        cnot[0][0] = Complex64::new(1.0, 0.0);
        cnot[1][1] = Complex64::new(1.0, 0.0);
        cnot[2][3] = Complex64::new(1.0, 0.0); // off-diagonal element
        cnot[3][2] = Complex64::new(1.0, 0.0);
        let gm = GateMatrix::two(cnot);
        assert!(!gm.is_diagonal); // triggers the return false branch in is_diagonal_4x4
    }

    #[test]
    fn test_gate_matrix_diagonal_variant() {
        let diag = GateMatrix::diagonal(vec![Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)]);
        assert!(diag.is_diagonal);
        assert_eq!(diag.num_qubits(), 1);
    }

    #[test]
    fn test_gate_matrix_two_diagonal() {
        // Diagonal 4x4 matrix
        let mut mat: Matrix4x4 = [[Complex64::new(0.0, 0.0); 4]; 4];
        mat[0][0] = Complex64::new(1.0, 0.0);
        mat[1][1] = Complex64::new(1.0, 0.0);
        mat[2][2] = Complex64::new(1.0, 0.0);
        mat[3][3] = Complex64::new(-1.0, 0.0);
        let gm = GateMatrix::two(mat);
        assert!(gm.is_diagonal);
        assert_eq!(gm.num_qubits(), 2);
    }
}

/// Common gate matrices
pub mod common {
    use super::*;

    pub fn pauli_x() -> Matrix2x2 {
        [
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ]
    }

    pub fn pauli_y() -> Matrix2x2 {
        [
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
            [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
        ]
    }

    pub fn pauli_z() -> Matrix2x2 {
        [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
        ]
    }

    pub fn hadamard() -> Matrix2x2 {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        [
            [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
            [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
        ]
    }

    pub fn identity() -> Matrix2x2 {
        [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ]
    }
}
