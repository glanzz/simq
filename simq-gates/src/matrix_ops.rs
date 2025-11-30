//! Matrix operations and computations for quantum gates
//!
//! This module provides utilities for working with quantum gate matrices, including:
//! - Tensor product operations for embedding gates in larger systems
//! - Matrix operations (multiplication, adjoint, trace, etc.)
//! - Full system matrix computation
//! - Matrix validation (unitarity, hermiticity)
//! - Circuit matrix composition
//!
//! # Example
//!
//! ```rust
//! use simq_gates::matrix_ops::{embed_gate_matrix, matrix_multiply, is_unitary};
//! use simq_gates::matrices::{PAULI_X, IDENTITY};
//!
//! // Embed a single-qubit gate in a 2-qubit system
//! let x_on_qubit_0 = embed_gate_matrix(&PAULI_X, 2, &[0]);
//! let x_on_qubit_1 = embed_gate_matrix(&PAULI_X, 2, &[1]);
//!
//! // Verify unitarity
//! assert!(is_unitary(&x_on_qubit_0, 1e-10));
//! ```
//!
//! # Circuit Matrix Composition
//!
//! Compute the full matrix representation of a circuit:
//!
//! ```rust
//! use simq_core::Circuit;
//! use simq_gates::matrix_ops::circuit_matrix;
//! use simq_gates::standard::{Hadamard, PauliX};
//! use std::sync::Arc;
//!
//! let mut circuit = Circuit::new(2);
//! circuit.add_gate(Arc::new(Hadamard), &[simq_core::QubitId::new(0)]).unwrap();
//! circuit.add_gate(Arc::new(PauliX), &[simq_core::QubitId::new(1)]).unwrap();
//!
//! let matrix = circuit_matrix(&circuit).unwrap();
//! ```

use num_complex::Complex64;
use std::f64;

/// Compute the tensor product of two matrices
///
/// For matrices A (m×m) and B (n×n), the tensor product A ⊗ B is (mn)×(mn).
/// The result is stored in row-major order as a flattened vector.
///
/// # Panics
/// Panics if the matrices are not square or have incompatible dimensions.
pub fn tensor_product(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    let n_a = (a.len() as f64).sqrt() as usize;
    let n_b = (b.len() as f64).sqrt() as usize;

    assert_eq!(n_a * n_a, a.len(), "Matrix A must be square");
    assert_eq!(n_b * n_b, b.len(), "Matrix B must be square");

    let n_result = n_a * n_b;
    let mut result = vec![Complex64::new(0.0, 0.0); n_result * n_result];

    for i in 0..n_a {
        for j in 0..n_a {
            let a_ij = a[i * n_a + j];
            for k in 0..n_b {
                for l in 0..n_b {
                    let b_kl = b[k * n_b + l];
                    let result_row = i * n_b + k;
                    let result_col = j * n_b + l;
                    result[result_row * n_result + result_col] = a_ij * b_kl;
                }
            }
        }
    }

    result
}

/// Embed a gate matrix into a larger system
///
/// Given a gate matrix for `gate_qubits`, embeds it into a system with `num_qubits` total qubits.
/// The gate is applied to the specified qubits, with identity operations on all other qubits.
///
/// # Arguments
/// * `gate_matrix` - The gate matrix as a flattened vector (row-major order)
/// * `num_qubits` - Total number of qubits in the system
/// * `qubit_indices` - Indices of qubits the gate acts on (must be sorted and unique)
///
/// # Panics
/// Panics if `qubit_indices` contains invalid qubit indices or if the gate matrix size doesn't match.
pub fn embed_gate_matrix(
    gate_matrix: &[[Complex64; 2]; 2],
    num_qubits: usize,
    qubit_indices: &[usize],
) -> Vec<Complex64> {
    assert!(!qubit_indices.is_empty(), "Qubit indices cannot be empty");
    assert!(qubit_indices.len() == 1, "Currently only single-qubit gates are supported");
    assert!(
        qubit_indices[0] < num_qubits,
        "Qubit index {} out of bounds for {} qubits",
        qubit_indices[0],
        num_qubits
    );

    let gate_vec: Vec<Complex64> = gate_matrix.iter().flatten().copied().collect();
    embed_gate_matrix_vec(&gate_vec, num_qubits, qubit_indices)
}

/// Embed a gate matrix (as vector) into a larger system
///
/// This is a more general version that accepts a flattened matrix vector.
pub fn embed_gate_matrix_vec(
    gate_matrix: &[Complex64],
    num_qubits: usize,
    qubit_indices: &[usize],
) -> Vec<Complex64> {
    let gate_size = (gate_matrix.len() as f64).sqrt() as usize;
    let num_gate_qubits = gate_size.ilog2() as usize;

    assert_eq!(
        qubit_indices.len(),
        num_gate_qubits,
        "Number of qubit indices must match gate size"
    );

    // Create zero matrix for the full system
    let system_size = 1 << num_qubits;
    let mut result = vec![Complex64::new(0.0, 0.0); system_size * system_size];

    // If this is a single-qubit gate, we can optimize
    if num_gate_qubits == 1 {
        let qubit = qubit_indices[0];
        let qubit_mask = 1 << qubit;
        let other_qubits_mask = !qubit_mask & ((1 << num_qubits) - 1);

        // Build the full matrix by iterating over all input basis states (columns)
        for input_state in 0..system_size {
            // Extract the target qubit bit from input state (column)
            let input_qubit_bit = (input_state & qubit_mask) >> qubit;
            let input_other = input_state & other_qubits_mask;

            // Apply the gate to get output states (rows)
            for output_qubit_bit in 0..gate_size {
                // Gate matrix element: gate[output_qubit_bit][input_qubit_bit]
                // Matrix is stored row-major: row * gate_size + col
                let gate_idx = output_qubit_bit * gate_size + input_qubit_bit;
                let gate_value = gate_matrix[gate_idx];

                // Construct the output state
                let output_state = input_other | (output_qubit_bit << qubit);

                // Set matrix element: result[row * system_size + col]
                result[output_state * system_size + input_state] = gate_value;
            }
        }
    } else {
        // For multi-qubit gates, use a more general approach
        // This is less efficient but more general
        for i in 0..system_size {
            for j in 0..system_size {
                // Extract the relevant qubit bits
                let mut gate_input = 0;
                let mut gate_output = 0;
                let mut matches = true;

                for (idx, &qubit) in qubit_indices.iter().enumerate() {
                    let mask = 1 << qubit;
                    let i_bit = (i & mask) >> qubit;
                    let j_bit = (j & mask) >> qubit;
                    gate_input |= (i_bit as usize) << idx;
                    gate_output |= (j_bit as usize) << idx;

                    // Check if non-gate qubits match
                    if (i ^ (i_bit << qubit)) != (j ^ (j_bit << qubit)) {
                        matches = false;
                        break;
                    }
                }

                if matches {
                    let gate_idx = gate_input * gate_size + gate_output;
                    result[i * system_size + j] = gate_matrix[gate_idx];
                } else {
                    result[i * system_size + j] = Complex64::new(0.0, 0.0);
                }
            }
        }
    }

    result
}

/// Create an identity matrix of the given size
pub fn identity_matrix(size: usize) -> Vec<Complex64> {
    let mut matrix = vec![Complex64::new(0.0, 0.0); size * size];
    for i in 0..size {
        matrix[i * size + i] = Complex64::new(1.0, 0.0);
    }
    matrix
}

/// Multiply two matrices
///
/// Computes C = A * B where A, B, and C are square matrices stored as flattened vectors.
pub fn matrix_multiply(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    let n = (a.len() as f64).sqrt() as usize;
    assert_eq!(n * n, a.len(), "Matrix A must be square");
    assert_eq!(n * n, b.len(), "Matrix B must be square");

    let mut result = vec![Complex64::new(0.0, 0.0); n * n];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                result[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }

    result
}

/// Compute the adjoint (Hermitian conjugate) of a matrix
///
/// Returns A† where (A†)ᵢⱼ = (Aⱼᵢ)*
pub fn matrix_adjoint(matrix: &[Complex64]) -> Vec<Complex64> {
    let n = (matrix.len() as f64).sqrt() as usize;
    assert_eq!(n * n, matrix.len(), "Matrix must be square");

    let mut result = vec![Complex64::new(0.0, 0.0); n * n];

    for i in 0..n {
        for j in 0..n {
            result[i * n + j] = matrix[j * n + i].conj();
        }
    }

    result
}

/// Compute the trace of a matrix
pub fn matrix_trace(matrix: &[Complex64]) -> Complex64 {
    let n = (matrix.len() as f64).sqrt() as usize;
    assert_eq!(n * n, matrix.len(), "Matrix must be square");

    let mut trace = Complex64::new(0.0, 0.0);
    for i in 0..n {
        trace += matrix[i * n + i];
    }
    trace
}

/// Compute the determinant of a 2x2 matrix
pub fn determinant_2x2(matrix: &[[Complex64; 2]; 2]) -> Complex64 {
    matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
}

/// Check if a matrix is unitary (U†U = I)
///
/// # Arguments
/// * `matrix` - The matrix to check (flattened vector)
/// * `tolerance` - Numerical tolerance for equality checks
pub fn is_unitary(matrix: &[Complex64], tolerance: f64) -> bool {
    let n = (matrix.len() as f64).sqrt() as usize;
    assert_eq!(n * n, matrix.len(), "Matrix must be square");

    let adjoint = matrix_adjoint(matrix);
    let u_dagger_u = matrix_multiply(&adjoint, matrix);
    let identity = identity_matrix(n);

    // Check if U†U ≈ I
    for i in 0..n {
        for j in 0..n {
            let diff = (u_dagger_u[i * n + j] - identity[i * n + j]).norm();
            if diff > tolerance {
                return false;
            }
        }
    }

    true
}

/// Check if a matrix is Hermitian (A = A†)
///
/// # Arguments
/// * `matrix` - The matrix to check (flattened vector)
/// * `tolerance` - Numerical tolerance for equality checks
pub fn is_hermitian(matrix: &[Complex64], tolerance: f64) -> bool {
    let n = (matrix.len() as f64).sqrt() as usize;
    assert_eq!(n * n, matrix.len(), "Matrix must be square");

    let adjoint = matrix_adjoint(matrix);

    for i in 0..n {
        for j in 0..n {
            let diff = (matrix[i * n + j] - adjoint[i * n + j]).norm();
            if diff > tolerance {
                return false;
            }
        }
    }

    true
}

/// Compute the fidelity between two quantum states or unitaries
///
/// For unitaries U and V, fidelity is |Tr(U†V)|² / d² where d is the dimension.
pub fn fidelity(u: &[Complex64], v: &[Complex64]) -> f64 {
    let n = (u.len() as f64).sqrt() as usize;
    assert_eq!(n * n, u.len(), "Matrix U must be square");
    assert_eq!(n * n, v.len(), "Matrix V must be square");

    let u_dagger = matrix_adjoint(u);
    let u_dagger_v = matrix_multiply(&u_dagger, v);
    let trace = matrix_trace(&u_dagger_v);
    let norm_sq = trace.norm_sqr();
    norm_sq / ((n * n) as f64)
}

/// Convert a matrix from 2D array to flattened vector
pub fn matrix_to_vec<const N: usize>(matrix: &[[Complex64; N]; N]) -> Vec<Complex64> {
    matrix.iter().flatten().copied().collect()
}

/// Convert a flattened vector to 2D array (for small matrices)
pub fn vec_to_matrix_2x2(vec: &[Complex64]) -> [[Complex64; 2]; 2] {
    assert_eq!(vec.len(), 4, "Vector must have 4 elements for 2x2 matrix");
    [[vec[0], vec[1]], [vec[2], vec[3]]]
}

/// Compute the full matrix representation of a circuit
///
/// This function computes the overall unitary matrix for a circuit by composing
/// all gate matrices. Gates are applied in sequence (right to left in matrix multiplication).
///
/// # Arguments
/// * `circuit` - The circuit to compute the matrix for
///
/// # Returns
/// The full circuit matrix as a flattened vector (row-major order), or an error
/// if any gate doesn't have a matrix representation.
///
/// # Example
/// ```rust
/// use simq_core::Circuit;
/// use simq_gates::matrix_ops::circuit_matrix;
/// use simq_gates::standard::{Hadamard, PauliX};
/// use std::sync::Arc;
///
/// let mut circuit = Circuit::new(2);
/// circuit.add_gate(Arc::new(Hadamard), &[simq_core::QubitId::new(0)]).unwrap();
/// circuit.add_gate(Arc::new(PauliX), &[simq_core::QubitId::new(1)]).unwrap();
///
/// let matrix = circuit_matrix(&circuit).unwrap();
/// // Matrix represents the full circuit transformation
/// ```
pub fn circuit_matrix(circuit: &simq_core::Circuit) -> std::result::Result<Vec<Complex64>, String> {
    let num_qubits = circuit.num_qubits();
    let system_size = 1 << num_qubits;

    // Start with identity matrix
    let mut result = identity_matrix(system_size);

    // Apply each gate in sequence
    for op in circuit.operations() {
        // Get the gate matrix
        let gate_matrix = op.gate().matrix().ok_or_else(|| {
            format!("Gate {} does not have a matrix representation", op.gate().name())
        })?;

        // Extract qubit indices
        let qubit_indices: Vec<usize> = op.qubits().iter().map(|q| q.index()).collect();

        // Embed the gate matrix into the full system
        let embedded_matrix = embed_gate_matrix_vec(&gate_matrix, num_qubits, &qubit_indices);

        // Compose with current circuit matrix: result = embedded_matrix * result
        // (gates are applied right to left, so we multiply on the left)
        result = matrix_multiply(&embedded_matrix, &result);
    }

    Ok(result)
}

/// Compute circuit matrix with custom gate matrix provider
///
/// This allows computing circuit matrices when gates might not implement the Gate::matrix() method,
/// by providing a custom function to extract matrices from gates.
pub fn circuit_matrix_with_provider<F>(
    circuit: &simq_core::Circuit,
    matrix_provider: F,
) -> std::result::Result<Vec<Complex64>, String>
where
    F: Fn(&dyn simq_core::gate::Gate) -> Option<Vec<Complex64>>,
{
    let num_qubits = circuit.num_qubits();
    let system_size = 1 << num_qubits;

    // Start with identity matrix
    let mut result = identity_matrix(system_size);

    // Apply each gate in sequence
    for op in circuit.operations() {
        // Get the gate matrix using the provider
        let gate_matrix = matrix_provider(op.gate().as_ref()).ok_or_else(|| {
            format!("Gate {} does not have a matrix representation", op.gate().name())
        })?;

        // Extract qubit indices
        let qubit_indices: Vec<usize> = op.qubits().iter().map(|q| q.index()).collect();

        // Embed the gate matrix into the full system
        let embedded_matrix = embed_gate_matrix_vec(&gate_matrix, num_qubits, &qubit_indices);

        // Compose with current circuit matrix
        result = matrix_multiply(&embedded_matrix, &result);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrices::{CNOT, HADAMARD, IDENTITY, PAULI_X, PAULI_Y, PAULI_Z};
    use approx::assert_relative_eq;

    #[test]
    fn test_tensor_product() {
        let i = matrix_to_vec(&IDENTITY);
        let x = matrix_to_vec(&PAULI_X);

        // I ⊗ X should give us X on the second qubit
        let result = tensor_product(&i, &x);
        assert_eq!(result.len(), 16); // 4x4 matrix

        // Check that |00⟩ → |01⟩, |01⟩ → |00⟩ (X on second qubit)
        assert_relative_eq!(result[1].norm(), 1.0, epsilon = 1e-10); // |00⟩ → |01⟩
        assert_relative_eq!(result[4].norm(), 1.0, epsilon = 1e-10); // |01⟩ → |00⟩
    }

    #[test]
    fn test_embed_gate_matrix() {
        // First, verify PAULI_X matrix structure
        let x_vec = matrix_to_vec(&PAULI_X);
        // PAULI_X = [[0, 1], [1, 0]]
        // Flattened: [0, 1, 1, 0]
        // X[0][0]=0, X[0][1]=1 (|0⟩ → |1⟩), X[1][0]=1 (|1⟩ → |0⟩), X[1][1]=0
        assert_relative_eq!(x_vec[0].re, 0.0, epsilon = 1e-10); // X[0][0]
        assert_relative_eq!(x_vec[1].re, 1.0, epsilon = 1e-10); // X[0][1] - |0⟩ → |1⟩
        assert_relative_eq!(x_vec[2].re, 1.0, epsilon = 1e-10); // X[1][0] - |1⟩ → |0⟩
        assert_relative_eq!(x_vec[3].re, 0.0, epsilon = 1e-10); // X[1][1]

        // Embed X gate on qubit 0 in a 2-qubit system
        // In quantum computing, state |q1 q0⟩ is represented as:
        // |00⟩ = 0, |01⟩ = 1, |10⟩ = 2, |11⟩ = 3
        // So qubit 0 is the least significant bit
        let x_embedded = embed_gate_matrix(&PAULI_X, 2, &[0]);
        assert_eq!(x_embedded.len(), 16); // 4x4 matrix

        // X on qubit 0: |q1 q0⟩ → |q1 (X q0)⟩
        // |00⟩ → |01⟩ (flip qubit 0: 0→1)
        assert_relative_eq!(x_embedded[1 * 4 + 0].re, 1.0, epsilon = 1e-10);

        // |01⟩ → |00⟩ (flip qubit 0: 1→0)
        assert_relative_eq!(x_embedded[0 * 4 + 1].re, 1.0, epsilon = 1e-10);

        // |10⟩ → |11⟩ (flip qubit 0: 0→1)
        assert_relative_eq!(x_embedded[3 * 4 + 2].re, 1.0, epsilon = 1e-10);

        // |11⟩ → |10⟩ (flip qubit 0: 1→0)
        assert_relative_eq!(x_embedded[2 * 4 + 3].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_multiply() {
        let x = matrix_to_vec(&PAULI_X);
        let x_squared = matrix_multiply(&x, &x);
        let identity = matrix_to_vec(&IDENTITY);

        // X² = I
        for idx in 0..4 {
            assert_relative_eq!(x_squared[idx].re, identity[idx].re, epsilon = 1e-10);
            assert_relative_eq!(x_squared[idx].im, identity[idx].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_matrix_adjoint() {
        let y = matrix_to_vec(&PAULI_Y);
        let y_dagger = matrix_adjoint(&y);

        // Y is Hermitian, so Y† = Y
        for i in 0..4 {
            assert_relative_eq!(y_dagger[i].re, y[i].re, epsilon = 1e-10);
            assert_relative_eq!(y_dagger[i].im, y[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_is_unitary() {
        let h = matrix_to_vec(&HADAMARD);
        assert!(is_unitary(&h, 1e-10));

        let x = matrix_to_vec(&PAULI_X);
        assert!(is_unitary(&x, 1e-10));

        let cnot = matrix_to_vec(&CNOT);
        assert!(is_unitary(&cnot, 1e-10));
    }

    #[test]
    fn test_is_hermitian() {
        let x = matrix_to_vec(&PAULI_X);
        assert!(is_hermitian(&x, 1e-10));

        let y = matrix_to_vec(&PAULI_Y);
        assert!(is_hermitian(&y, 1e-10));

        let z = matrix_to_vec(&PAULI_Z);
        assert!(is_hermitian(&z, 1e-10));

        let h = matrix_to_vec(&HADAMARD);
        assert!(is_hermitian(&h, 1e-10));
    }

    #[test]
    fn test_matrix_trace() {
        let i = matrix_to_vec(&IDENTITY);
        let trace = matrix_trace(&i);
        assert_relative_eq!(trace.re, 2.0, epsilon = 1e-10);
        assert_relative_eq!(trace.im, 0.0, epsilon = 1e-10);

        let x = matrix_to_vec(&PAULI_X);
        let trace = matrix_trace(&x);
        assert_relative_eq!(trace.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(trace.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_determinant_2x2() {
        let det_i = determinant_2x2(&IDENTITY);
        assert_relative_eq!(det_i.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(det_i.im, 0.0, epsilon = 1e-10);

        let det_x = determinant_2x2(&PAULI_X);
        assert_relative_eq!(det_x.re, -1.0, epsilon = 1e-10);
        assert_relative_eq!(det_x.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fidelity() {
        let identity = matrix_to_vec(&IDENTITY);
        let fid = fidelity(&identity, &identity);
        assert_relative_eq!(fid, 1.0, epsilon = 1e-10);

        let x = matrix_to_vec(&PAULI_X);
        let x_squared = matrix_multiply(&x, &x);
        let identity = matrix_to_vec(&IDENTITY);
        let fid = fidelity(&x_squared, &identity);
        assert_relative_eq!(fid, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_identity_matrix() {
        let i = identity_matrix(2);
        assert_eq!(i.len(), 4);
        assert_relative_eq!(i[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(i[1].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(i[2].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(i[3].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_circuit_matrix() {
        use crate::standard::{Hadamard, PauliX};
        use simq_core::Circuit;
        use std::sync::Arc;

        // Create a simple circuit: H on qubit 0, X on qubit 1
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(Hadamard), &[simq_core::QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX), &[simq_core::QubitId::new(1)])
            .unwrap();

        // Compute circuit matrix
        let matrix = circuit_matrix(&circuit).unwrap();
        assert_eq!(matrix.len(), 16); // 4x4 matrix for 2 qubits

        // Verify unitarity - the circuit matrix should always be unitary
        // since it's composed of unitary gates
        assert!(is_unitary(&matrix, 1e-10));
    }

    #[test]
    fn test_circuit_matrix_empty() {
        use simq_core::Circuit;

        // Empty circuit should be identity
        let circuit = Circuit::new(2);
        let matrix = circuit_matrix(&circuit).unwrap();
        let identity = identity_matrix(4); // 2^2 = 4

        for i in 0..16 {
            assert_relative_eq!(matrix[i].re, identity[i].re, epsilon = 1e-10);
            assert_relative_eq!(matrix[i].im, identity[i].im, epsilon = 1e-10);
        }
    }
}
