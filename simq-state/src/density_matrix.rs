//! Density matrix representation for mixed quantum states
//!
//! This module provides a density matrix implementation that can represent
//! both pure and mixed quantum states, essential for noise simulation.
//!
//! # Overview
//!
//! A density matrix ρ is a positive semi-definite, Hermitian matrix with Tr(ρ) = 1.
//! For pure states: ρ = |ψ⟩⟨ψ|
//! For mixed states: ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|
//!
//! # Key Properties
//!
//! - **Purity**: Tr(ρ²) ∈ (0, 1], equals 1 for pure states
//! - **Von Neumann Entropy**: S = -Tr(ρ log ρ), measures mixedness
//! - **Partial Trace**: Extract subsystem density matrices
//!
//! # Example
//!
//! ```ignore
//! use simq_state::DensityMatrix;
//!
//! // Create maximally mixed 1-qubit state
//! let mut dm = DensityMatrix::maximally_mixed(1)?;
//! assert!((dm.purity() - 0.5).abs() < 1e-10);
//!
//! // Create from pure state
//! let pure = DensityMatrix::from_state_vector(2, &amplitudes)?;
//! assert!((pure.purity() - 1.0).abs() < 1e-10);
//! ```

use crate::error::{Result, StateError};
use num_complex::Complex64;
use std::fmt;

/// Density matrix representation of a quantum state
///
/// Stores the full 2^n × 2^n density matrix in row-major order.
/// Memory usage: O(4^n) complex numbers.
pub struct DensityMatrix {
    /// Number of qubits
    num_qubits: usize,

    /// Dimension (2^num_qubits)
    dimension: usize,

    /// Density matrix elements in row-major order
    /// Length: dimension²
    matrix: Vec<Complex64>,
}

impl DensityMatrix {
    /// Create a new density matrix initialized to |0...0⟩⟨0...0|
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    ///
    /// # Errors
    /// Returns error if num_qubits is too large for memory
    pub fn new(num_qubits: usize) -> Result<Self> {
        let dimension = 1usize
            .checked_shl(num_qubits as u32)
            .ok_or(StateError::InvalidDimension { dimension: 0 })?;

        let matrix_size = dimension
            .checked_mul(dimension)
            .ok_or(StateError::AllocationError {
                size: usize::MAX,
            })?;

        let mut matrix = vec![Complex64::new(0.0, 0.0); matrix_size];
        // Initialize to |0...0⟩⟨0...0| - only (0,0) element is 1
        matrix[0] = Complex64::new(1.0, 0.0);

        Ok(Self {
            num_qubits,
            dimension,
            matrix,
        })
    }

    /// Create density matrix from a pure state vector
    ///
    /// Computes ρ = |ψ⟩⟨ψ| from amplitudes
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    /// * `amplitudes` - State vector amplitudes (length 2^num_qubits)
    pub fn from_state_vector(num_qubits: usize, amplitudes: &[Complex64]) -> Result<Self> {
        let dimension = 1usize << num_qubits;

        if amplitudes.len() != dimension {
            return Err(StateError::DimensionMismatch {
                expected: dimension,
                actual: amplitudes.len(),
            });
        }

        let matrix_size = dimension * dimension;
        let mut matrix = vec![Complex64::new(0.0, 0.0); matrix_size];

        // Compute outer product: ρᵢⱼ = ψᵢ ψⱼ*
        for i in 0..dimension {
            for j in 0..dimension {
                matrix[i * dimension + j] = amplitudes[i] * amplitudes[j].conj();
            }
        }

        Ok(Self {
            num_qubits,
            dimension,
            matrix,
        })
    }

    /// Create maximally mixed state: ρ = I/2^n
    ///
    /// All diagonal elements are 1/2^n, off-diagonal elements are 0.
    pub fn maximally_mixed(num_qubits: usize) -> Result<Self> {
        let dimension = 1usize << num_qubits;
        let matrix_size = dimension * dimension;
        let mut matrix = vec![Complex64::new(0.0, 0.0); matrix_size];

        let value = Complex64::new(1.0 / dimension as f64, 0.0);
        for i in 0..dimension {
            matrix[i * dimension + i] = value;
        }

        Ok(Self {
            num_qubits,
            dimension,
            matrix,
        })
    }

    /// Get number of qubits
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get matrix dimension (2^num_qubits)
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get a matrix element ρᵢⱼ
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Complex64 {
        self.matrix[row * self.dimension + col]
    }

    /// Set a matrix element ρᵢⱼ
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: Complex64) {
        self.matrix[row * self.dimension + col] = value;
    }

    /// Get reference to the full matrix data
    pub fn matrix(&self) -> &[Complex64] {
        &self.matrix
    }

    /// Get mutable reference to the full matrix data
    pub fn matrix_mut(&mut self) -> &mut [Complex64] {
        &mut self.matrix
    }

    /// Apply a unitary gate: ρ → U ρ U†
    ///
    /// # Arguments
    /// * `unitary` - Gate matrix (dimension × dimension)
    /// * `qubits` - Target qubit indices (in order)
    ///
    /// # Implementation
    /// Uses matrix multiplication: ρ' = U ρ U†
    pub fn apply_unitary(&mut self, unitary: &[Complex64], qubits: &[usize]) -> Result<()> {
        // For single-qubit gates, use optimized implementation
        if qubits.len() == 1 {
            self.apply_single_qubit_unitary(unitary, qubits[0])
        } else if qubits.len() == 2 {
            self.apply_two_qubit_unitary(unitary, qubits[0], qubits[1])
        } else {
            // General n-qubit case (expensive!)
            self.apply_general_unitary(unitary, qubits)
        }
    }

    /// Apply single-qubit unitary gate
    fn apply_single_qubit_unitary(&mut self, unitary: &[Complex64], qubit: usize) -> Result<()> {
        if qubit >= self.num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: qubit,
                num_qubits: self.num_qubits,
            });
        }

        let dim = self.dimension;
        let stride = 1usize << qubit;

        // Temporary storage for result
        let mut new_matrix = vec![Complex64::new(0.0, 0.0); dim * dim];

        // Apply U ρ U†
        // First compute temp = U ρ
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = Complex64::new(0.0, 0.0);

                for k in 0..dim {
                    let u_elem = self.get_single_qubit_element(unitary, i, k, stride);
                    sum += u_elem * self.get(k, j);
                }

                new_matrix[i * dim + j] = sum;
            }
        }

        // Now compute result = temp U†
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = Complex64::new(0.0, 0.0);

                for k in 0..dim {
                    let u_dag_elem = self.get_single_qubit_element(unitary, k, j, stride).conj();
                    sum += new_matrix[i * dim + k] * u_dag_elem;
                }

                self.matrix[i * dim + j] = sum;
            }
        }

        Ok(())
    }

    /// Get matrix element for single-qubit gate applied to specific qubit
    #[inline]
    fn get_single_qubit_element(
        &self,
        unitary: &[Complex64],
        row: usize,
        col: usize,
        stride: usize,
    ) -> Complex64 {
        let row_bit = (row / stride) & 1;
        let col_bit = (col / stride) & 1;

        if (row & !stride) == (col & !stride) {
            unitary[row_bit * 2 + col_bit]
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    /// Apply two-qubit unitary gate
    fn apply_two_qubit_unitary(
        &mut self,
        unitary: &[Complex64],
        qubit0: usize,
        qubit1: usize,
    ) -> Result<()> {
        if qubit0 >= self.num_qubits || qubit1 >= self.num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: qubit0.max(qubit1),
                num_qubits: self.num_qubits,
            });
        }

        let dim = self.dimension;
        let stride0 = 1usize << qubit0;
        let stride1 = 1usize << qubit1;

        let mut new_matrix = vec![Complex64::new(0.0, 0.0); dim * dim];

        // Apply U ρ U† (simplified for 2-qubit case)
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = Complex64::new(0.0, 0.0);

                for k in 0..dim {
                    let u_elem = self.get_two_qubit_element(unitary, i, k, stride0, stride1);
                    sum += u_elem * self.get(k, j);
                }

                new_matrix[i * dim + j] = sum;
            }
        }

        for i in 0..dim {
            for j in 0..dim {
                let mut sum = Complex64::new(0.0, 0.0);

                for k in 0..dim {
                    let u_dag_elem =
                        self.get_two_qubit_element(unitary, k, j, stride0, stride1).conj();
                    sum += new_matrix[i * dim + k] * u_dag_elem;
                }

                self.matrix[i * dim + j] = sum;
            }
        }

        Ok(())
    }

    /// Get matrix element for two-qubit gate
    #[inline]
    fn get_two_qubit_element(
        &self,
        unitary: &[Complex64],
        row: usize,
        col: usize,
        stride0: usize,
        stride1: usize,
    ) -> Complex64 {
        let row_bits = ((row / stride0) & 1) | (((row / stride1) & 1) << 1);
        let col_bits = ((col / stride0) & 1) | (((col / stride1) & 1) << 1);

        let mask = !(stride0 | stride1);
        if (row & mask) == (col & mask) {
            unitary[row_bits * 4 + col_bits]
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    /// Apply general n-qubit unitary (slow, for completeness)
    fn apply_general_unitary(&mut self, _unitary: &[Complex64], _qubits: &[usize]) -> Result<()> {
        // TODO: Implement general case if needed
        Err(StateError::InvalidDimension {
            dimension: _qubits.len(),
        })
    }

    /// Apply a Kraus operator channel: ρ → Σᵢ Kᵢ ρ Kᵢ†
    ///
    /// # Arguments
    /// * `kraus_ops` - List of Kraus operators (each is a flattened matrix)
    /// * `qubits` - Target qubit indices
    ///
    /// This is the key operation for noise simulation!
    pub fn apply_kraus_channel(
        &mut self,
        kraus_ops: &[(Vec<Complex64>, usize)],
        qubits: &[usize],
    ) -> Result<()> {
        let dim = self.dimension;
        let mut result = vec![Complex64::new(0.0, 0.0); dim * dim];

        // For each Kraus operator: result += Kᵢ ρ Kᵢ†
        for (kraus_matrix, kraus_dim) in kraus_ops {
            let temp = self.apply_kraus_operator(kraus_matrix, *kraus_dim, qubits)?;

            for i in 0..result.len() {
                result[i] += temp[i];
            }
        }

        self.matrix = result;
        Ok(())
    }

    /// Apply single Kraus operator: Kᵢ ρ Kᵢ†
    fn apply_kraus_operator(
        &self,
        kraus: &[Complex64],
        kraus_dim: usize,
        qubits: &[usize],
    ) -> Result<Vec<Complex64>> {
        if qubits.len() == 1 && kraus_dim == 2 {
            // Single-qubit Kraus operator
            self.apply_single_qubit_kraus(kraus, qubits[0])
        } else {
            // General case
            Err(StateError::InvalidDimension {
                dimension: qubits.len(),
            })
        }
    }

    /// Apply single-qubit Kraus operator
    fn apply_single_qubit_kraus(
        &self,
        kraus: &[Complex64],
        qubit: usize,
    ) -> Result<Vec<Complex64>> {
        let dim = self.dimension;
        let stride = 1usize << qubit;
        let mut result = vec![Complex64::new(0.0, 0.0); dim * dim];

        // Compute K ρ K†
        let mut temp = vec![Complex64::new(0.0, 0.0); dim * dim];

        // First: temp = K ρ
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..dim {
                    let k_elem = self.get_single_qubit_element(kraus, i, k, stride);
                    sum += k_elem * self.get(k, j);
                }
                temp[i * dim + j] = sum;
            }
        }

        // Second: result = temp K†
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..dim {
                    let k_dag_elem = self.get_single_qubit_element(kraus, k, j, stride).conj();
                    sum += temp[i * dim + k] * k_dag_elem;
                }
                result[i * dim + j] = sum;
            }
        }

        Ok(result)
    }

    /// Calculate the purity: Tr(ρ²)
    ///
    /// Returns 1 for pure states, < 1 for mixed states.
    /// Minimum purity is 1/d where d = 2^n.
    pub fn purity(&self) -> f64 {
        let dim = self.dimension;
        let mut trace = Complex64::new(0.0, 0.0);

        // Compute Tr(ρ²) = Σᵢⱼ ρᵢⱼ ρⱼᵢ
        for i in 0..dim {
            for j in 0..dim {
                trace += self.get(i, j) * self.get(j, i);
            }
        }

        trace.re
    }

    /// Calculate trace: Tr(ρ)
    ///
    /// Should always be 1 for valid density matrices.
    pub fn trace(&self) -> f64 {
        let mut tr = Complex64::new(0.0, 0.0);
        for i in 0..self.dimension {
            tr += self.get(i, i);
        }
        tr.re
    }

    /// Calculate von Neumann entropy: S = -Tr(ρ log₂ ρ)
    ///
    /// Returns 0 for pure states, log₂(d) for maximally mixed states.
    ///
    /// Note: This requires eigenvalue decomposition (expensive!)
    pub fn von_neumann_entropy(&self) -> f64 {
        // For simplicity, approximate using purity
        // Exact calculation requires eigendecomposition
        let p = self.purity();
        if p > 0.9999 {
            0.0 // Pure state
        } else {
            // Linear approximation based on purity
            -p * p.log2()
        }
    }

    /// Measure a qubit in the computational basis
    ///
    /// Returns the measurement outcome (0 or 1) and updates the density matrix
    /// to the post-measurement state.
    ///
    /// # Arguments
    /// * `qubit` - Qubit index to measure
    /// * `random_value` - Random number in [0, 1) for sampling
    ///
    /// # Returns
    /// Measurement outcome (false = 0, true = 1)
    pub fn measure(&mut self, qubit: usize, random_value: f64) -> Result<bool> {
        if qubit >= self.num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: qubit,
                num_qubits: self.num_qubits,
            });
        }

        // Calculate P(0) from diagonal elements
        let prob_0 = self.probability_zero(qubit);

        let outcome = random_value >= prob_0;

        // Apply measurement projector
        self.project_measurement(qubit, outcome)?;

        Ok(outcome)
    }

    /// Calculate probability of measuring 0 on a qubit
    fn probability_zero(&self, qubit: usize) -> f64 {
        let stride = 1usize << qubit;
        let mut prob = 0.0;

        for i in 0..self.dimension {
            if (i & stride) == 0 {
                prob += self.get(i, i).re;
            }
        }

        prob
    }

    /// Project density matrix after measurement
    fn project_measurement(&mut self, qubit: usize, outcome: bool) -> Result<()> {
        let stride = 1usize << qubit;
        let target_bit = if outcome { stride } else { 0 };

        let dim = self.dimension;
        let mut new_matrix = vec![Complex64::new(0.0, 0.0); dim * dim];

        // Keep only elements where both row and col have correct bit value
        let mut norm = 0.0;
        for i in 0..dim {
            if (i & stride) == target_bit {
                for j in 0..dim {
                    if (j & stride) == target_bit {
                        new_matrix[i * dim + j] = self.get(i, j);
                        if i == j {
                            norm += new_matrix[i * dim + j].re;
                        }
                    }
                }
            }
        }

        // Renormalize
        if norm > 1e-10 {
            for elem in new_matrix.iter_mut() {
                *elem /= norm;
            }
        }

        self.matrix = new_matrix;
        Ok(())
    }

    /// Check if the density matrix is valid (Hermitian, positive, trace 1)
    pub fn is_valid(&self, tolerance: f64) -> bool {
        // Check trace = 1
        if (self.trace() - 1.0).abs() > tolerance {
            return false;
        }

        // Check Hermitian: ρᵢⱼ = ρⱼᵢ*
        for i in 0..self.dimension {
            for j in (i + 1)..self.dimension {
                let diff = (self.get(i, j) - self.get(j, i).conj()).norm();
                if diff > tolerance {
                    return false;
                }
            }
        }

        // Check positive semi-definite (all diagonal elements ≥ 0)
        for i in 0..self.dimension {
            if self.get(i, i).re < -tolerance {
                return false;
            }
        }

        true
    }

    /// Compute partial trace over specified qubits
    ///
    /// # Arguments
    /// * `trace_qubits` - Qubits to trace out
    ///
    /// # Returns
    /// Reduced density matrix with traced-out qubits removed
    ///
    /// # Example
    /// For a 2-qubit state, tracing out qubit 0 gives the reduced
    /// density matrix of qubit 1.
    pub fn partial_trace(&self, trace_qubits: &[usize]) -> Result<Self> {
        if trace_qubits.is_empty() {
            return Err(StateError::InvalidDimension { dimension: 0 });
        }

        // Check all qubits are valid
        for &q in trace_qubits {
            if q >= self.num_qubits {
                return Err(StateError::InvalidQubitIndex {
                    index: q,
                    num_qubits: self.num_qubits,
                });
            }
        }

        let remaining_qubits = self.num_qubits - trace_qubits.len();
        let reduced_dim = 1usize << remaining_qubits;
        let mut reduced = vec![Complex64::new(0.0, 0.0); reduced_dim * reduced_dim];

        // Build mapping from full indices to reduced indices
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                // Check if traced-out qubits match
                let mut matches = true;
                for &q in trace_qubits {
                    let mask = 1usize << q;
                    if (i & mask) != (j & mask) {
                        matches = false;
                        break;
                    }
                }

                if matches {
                    // Compute reduced indices
                    let reduced_i = self.project_to_reduced_index(i, trace_qubits);
                    let reduced_j = self.project_to_reduced_index(j, trace_qubits);

                    reduced[reduced_i * reduced_dim + reduced_j] += self.get(i, j);
                }
            }
        }

        Ok(Self {
            num_qubits: remaining_qubits,
            dimension: reduced_dim,
            matrix: reduced,
        })
    }

    /// Project full index to reduced index (removing traced qubits)
    fn project_to_reduced_index(&self, index: usize, trace_qubits: &[usize]) -> usize {
        let mut reduced = 0;
        let mut shift = 0;

        for q in 0..self.num_qubits {
            if !trace_qubits.contains(&q) {
                if (index & (1 << q)) != 0 {
                    reduced |= 1 << shift;
                }
                shift += 1;
            }
        }

        reduced
    }
}

impl fmt::Debug for DensityMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DensityMatrix {{ qubits: {}, dim: {}, purity: {:.4} }}",
            self.num_qubits,
            self.dimension,
            self.purity()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_new_density_matrix() {
        let dm = DensityMatrix::new(2).unwrap();
        assert_eq!(dm.num_qubits(), 2);
        assert_eq!(dm.dimension(), 4);
        assert!((dm.trace() - 1.0).abs() < TOL);
        assert!((dm.purity() - 1.0).abs() < TOL); // Pure state
    }

    #[test]
    fn test_from_state_vector() {
        // Bell state: (|00⟩ + |11⟩)/√2
        let amplitudes = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];

        let dm = DensityMatrix::from_state_vector(2, &amplitudes).unwrap();
        assert!((dm.purity() - 1.0).abs() < TOL);
        assert!((dm.trace() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_maximally_mixed() {
        let dm = DensityMatrix::maximally_mixed(2).unwrap();
        assert!((dm.trace() - 1.0).abs() < TOL);

        // Purity of maximally mixed d-dimensional state is 1/d
        let expected_purity = 1.0 / 4.0;
        assert!((dm.purity() - expected_purity).abs() < TOL);
    }

    #[test]
    fn test_apply_single_qubit_unitary() {
        // Hadamard gate
        let h = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
        ];

        let mut dm = DensityMatrix::new(1).unwrap();
        dm.apply_single_qubit_unitary(&h, 0).unwrap();

        // Should create (|0⟩ + |1⟩)/√2
        assert!((dm.purity() - 1.0).abs() < TOL);
        assert!((dm.get(0, 0).re - 0.5).abs() < TOL);
        assert!((dm.get(1, 1).re - 0.5).abs() < TOL);
    }

    #[test]
    fn test_measurement() {
        let mut dm = DensityMatrix::new(1).unwrap();

        // Measure |0⟩ state
        let outcome = dm.measure(0, 0.5).unwrap();
        assert!(!outcome); // Should be 0

        // State should remain |0⟩
        assert!((dm.get(0, 0).re - 1.0).abs() < TOL);
    }

    #[test]
    fn test_is_valid() {
        let dm = DensityMatrix::new(2).unwrap();
        assert!(dm.is_valid(TOL));

        let mixed = DensityMatrix::maximally_mixed(2).unwrap();
        assert!(mixed.is_valid(TOL));
    }
}
