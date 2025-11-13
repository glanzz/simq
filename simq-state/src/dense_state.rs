//! Dense quantum state representation with 64-byte aligned vectors
//!
//! This module provides a high-level dense state representation that wraps
//! the SIMD-optimized StateVector with additional quantum operations like
//! measurement, gate application, and state manipulation.

use crate::error::{Result, StateError};
use crate::simd::{apply_single_qubit_gate, apply_two_qubit_gate};
use crate::state_vector::StateVector;
use num_complex::Complex64;
use std::fmt;

/// Dense quantum state representation with 64-byte aligned vectors
///
/// DenseState provides a high-level interface for quantum state manipulation,
/// including gate application, measurement, and state queries. It uses
/// 64-byte aligned vectors internally for optimal SIMD performance.
///
/// # Example
///
/// ```
/// use simq_state::DenseState;
/// use num_complex::Complex64;
///
/// // Create a 2-qubit state
/// let mut state = DenseState::new(2).unwrap();
///
/// // Apply gates
/// let hadamard = [
///     [Complex64::new(0.7071067811865476, 0.0), Complex64::new(0.7071067811865476, 0.0)],
///     [Complex64::new(0.7071067811865476, 0.0), Complex64::new(-0.7071067811865476, 0.0)],
/// ];
/// state.apply_single_qubit_gate(&hadamard, 0).unwrap();
///
/// // Check properties
/// assert_eq!(state.num_qubits(), 2);
/// assert!(state.is_normalized(1e-10));
/// ```
pub struct DenseState {
    /// Underlying state vector with aligned memory
    vector: StateVector,

    /// Whether the state has been modified and needs renormalization
    needs_normalization: bool,
}

impl DenseState {
    /// Create a new dense state initialized to |0...0⟩
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    ///
    /// # Returns
    /// A new dense state in the computational basis state |0...0⟩
    ///
    /// # Errors
    /// Returns error if memory allocation fails or num_qubits is too large
    ///
    /// # Example
    /// ```
    /// use simq_state::DenseState;
    ///
    /// let state = DenseState::new(3).unwrap();
    /// assert_eq!(state.num_qubits(), 3);
    /// assert_eq!(state.dimension(), 8);
    /// ```
    pub fn new(num_qubits: usize) -> Result<Self> {
        Ok(Self {
            vector: StateVector::new(num_qubits)?,
            needs_normalization: false,
        })
    }

    /// Create a dense state from amplitude data
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    /// * `amplitudes` - Complex amplitudes (must have length 2^num_qubits)
    ///
    /// # Returns
    /// A new dense state with the given amplitudes
    ///
    /// # Errors
    /// Returns error if dimension doesn't match or allocation fails
    ///
    /// # Example
    /// ```
    /// use simq_state::DenseState;
    /// use num_complex::Complex64;
    ///
    /// let amplitudes = vec![
    ///     Complex64::new(0.5, 0.0),
    ///     Complex64::new(0.5, 0.0),
    ///     Complex64::new(0.5, 0.0),
    ///     Complex64::new(0.5, 0.0),
    /// ];
    /// let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();
    /// ```
    pub fn from_amplitudes(num_qubits: usize, amplitudes: &[Complex64]) -> Result<Self> {
        Ok(Self {
            vector: StateVector::from_amplitudes(num_qubits, amplitudes)?,
            needs_normalization: true,
        })
    }

    /// Get the number of qubits
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.vector.num_qubits()
    }

    /// Get the state dimension (2^num_qubits)
    #[inline]
    pub fn dimension(&self) -> usize {
        self.vector.dimension()
    }

    /// Get a reference to the state amplitudes
    #[inline]
    pub fn amplitudes(&self) -> &[Complex64] {
        self.vector.amplitudes()
    }

    /// Get a mutable reference to the state amplitudes
    ///
    /// Note: This marks the state as needing normalization
    #[inline]
    pub fn amplitudes_mut(&mut self) -> &mut [Complex64] {
        self.needs_normalization = true;
        self.vector.amplitudes_mut()
    }

    /// Get a reference to the underlying state vector
    #[inline]
    pub fn vector(&self) -> &StateVector {
        &self.vector
    }

    /// Check if the state is normalized (norm ≈ 1)
    ///
    /// # Arguments
    /// * `epsilon` - Tolerance for normalization check
    ///
    /// # Returns
    /// True if |norm - 1| < epsilon
    pub fn is_normalized(&self, epsilon: f64) -> bool {
        self.vector.is_normalized(epsilon)
    }

    /// Compute the norm of the state
    pub fn norm(&self) -> f64 {
        self.vector.norm()
    }

    /// Normalize the state to have unit norm
    ///
    /// This ensures that the sum of probability amplitudes equals 1.
    pub fn normalize(&mut self) {
        self.vector.normalize();
        self.needs_normalization = false;
    }

    /// Ensure the state is normalized (normalizes if needed)
    pub fn ensure_normalized(&mut self) {
        if self.needs_normalization || !self.is_normalized(1e-10) {
            self.normalize();
        }
    }

    /// Reset the state to |0...0⟩
    pub fn reset(&mut self) {
        self.vector.reset();
        self.needs_normalization = false;
    }

    /// Clone the dense state
    pub fn clone_state(&self) -> Result<Self> {
        Ok(Self {
            vector: self.vector.clone_state()?,
            needs_normalization: self.needs_normalization,
        })
    }

    /// Apply a single-qubit gate to the state
    ///
    /// # Arguments
    /// * `matrix` - 2×2 gate matrix in row-major order
    /// * `qubit` - Index of the qubit to apply the gate to (0-indexed)
    ///
    /// # Errors
    /// Returns error if qubit index is invalid
    ///
    /// # Example
    /// ```
    /// use simq_state::DenseState;
    /// use num_complex::Complex64;
    ///
    /// let mut state = DenseState::new(2).unwrap();
    ///
    /// // Pauli-X gate
    /// let x_gate = [
    ///     [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ///     [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    /// ];
    ///
    /// state.apply_single_qubit_gate(&x_gate, 0).unwrap();
    /// ```
    pub fn apply_single_qubit_gate(
        &mut self,
        matrix: &[[Complex64; 2]; 2],
        qubit: usize,
    ) -> Result<()> {
        let num_qubits = self.num_qubits();

        if qubit >= num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: qubit,
                num_qubits,
            });
        }

        apply_single_qubit_gate(
            self.vector.amplitudes_mut(),
            matrix,
            qubit,
            num_qubits,
        );

        self.needs_normalization = false; // Unitary gates preserve norm
        Ok(())
    }

    /// Apply a two-qubit gate to the state
    ///
    /// # Arguments
    /// * `matrix` - 4×4 gate matrix in row-major order
    /// * `qubit1` - Index of the first qubit (0-indexed)
    /// * `qubit2` - Index of the second qubit (0-indexed)
    ///
    /// # Errors
    /// Returns error if qubit indices are invalid or equal
    ///
    /// # Example
    /// ```
    /// use simq_state::DenseState;
    /// use num_complex::Complex64;
    ///
    /// let mut state = DenseState::new(2).unwrap();
    ///
    /// // CNOT gate matrix
    /// let cnot = [
    ///     [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
    ///     [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
    ///     [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ///     [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    /// ];
    ///
    /// state.apply_two_qubit_gate(&cnot, 0, 1).unwrap();
    /// ```
    pub fn apply_two_qubit_gate(
        &mut self,
        matrix: &[[Complex64; 4]; 4],
        qubit1: usize,
        qubit2: usize,
    ) -> Result<()> {
        let num_qubits = self.num_qubits();

        if qubit1 >= num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: qubit1,
                num_qubits,
            });
        }
        if qubit2 >= num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: qubit2,
                num_qubits,
            });
        }
        if qubit1 == qubit2 {
            return Err(StateError::InvalidQubitIndex {
                index: qubit1,
                num_qubits,
            });
        }

        apply_two_qubit_gate(
            self.vector.amplitudes_mut(),
            matrix,
            qubit1,
            qubit2,
            num_qubits,
        );

        self.needs_normalization = false; // Unitary gates preserve norm
        Ok(())
    }

    /// Get the probability of measuring a specific computational basis state
    ///
    /// # Arguments
    /// * `basis_state` - The basis state index (0 to 2^n - 1)
    ///
    /// # Returns
    /// The probability |amplitude|^2 of measuring this basis state
    ///
    /// # Errors
    /// Returns error if basis_state is out of bounds
    pub fn get_probability(&self, basis_state: usize) -> Result<f64> {
        if basis_state >= self.dimension() {
            return Err(StateError::InvalidDimension {
                dimension: basis_state,
            });
        }

        Ok(self.amplitudes()[basis_state].norm_sqr())
    }

    /// Get probabilities for all computational basis states
    ///
    /// # Returns
    /// Vector of probabilities, where probabilities[i] = |amplitude[i]|^2
    pub fn get_all_probabilities(&self) -> Vec<f64> {
        self.amplitudes()
            .iter()
            .map(|amp| amp.norm_sqr())
            .collect()
    }

    /// Measure a single qubit and collapse the state
    ///
    /// # Arguments
    /// * `qubit` - Index of the qubit to measure (0-indexed)
    /// * `random_value` - Random value in [0, 1) for outcome determination
    ///
    /// # Returns
    /// The measurement outcome (0 or 1)
    ///
    /// # Errors
    /// Returns error if qubit index is invalid
    ///
    /// # Example
    /// ```
    /// use simq_state::DenseState;
    ///
    /// let mut state = DenseState::new(2).unwrap();
    /// let outcome = state.measure_qubit(0, 0.5).unwrap();
    /// assert!(outcome == 0 || outcome == 1);
    /// ```
    pub fn measure_qubit(&mut self, qubit: usize, random_value: f64) -> Result<u8> {
        if qubit >= self.num_qubits() {
            return Err(StateError::InvalidQubitIndex {
                index: qubit,
                num_qubits: self.num_qubits(),
            });
        }

        // Calculate probability of measuring |0⟩
        let mask = 1 << qubit;
        let prob_zero: f64 = self
            .amplitudes()
            .iter()
            .enumerate()
            .filter(|(idx, _)| idx & mask == 0)
            .map(|(_, amp)| amp.norm_sqr())
            .sum();

        // Determine outcome based on random value
        let outcome = if random_value < prob_zero { 0 } else { 1 };

        // Collapse the state
        let normalization = if outcome == 0 {
            prob_zero.sqrt()
        } else {
            (1.0 - prob_zero).sqrt()
        };

        if normalization < 1e-10 {
            return Err(StateError::NotNormalized {
                norm: normalization,
            });
        }

        let inv_norm = 1.0 / normalization;

        // Zero out non-matching amplitudes and renormalize
        for (idx, amp) in self.amplitudes_mut().iter_mut().enumerate() {
            if ((idx >> qubit) & 1) != outcome as usize {
                *amp = Complex64::new(0.0, 0.0);
            } else {
                *amp *= inv_norm;
            }
        }

        self.needs_normalization = false;
        Ok(outcome)
    }

    /// Measure all qubits and collapse to a computational basis state
    ///
    /// # Arguments
    /// * `random_value` - Random value in [0, 1) for outcome determination
    ///
    /// # Returns
    /// The measurement outcome as a basis state index (0 to 2^n - 1)
    ///
    /// # Example
    /// ```
    /// use simq_state::DenseState;
    ///
    /// let mut state = DenseState::new(3).unwrap();
    /// let outcome = state.measure_all(0.5).unwrap();
    /// assert!(outcome < 8); // 2^3 = 8 possible outcomes
    /// ```
    pub fn measure_all(&mut self, random_value: f64) -> Result<usize> {
        let probabilities = self.get_all_probabilities();

        // Find outcome using cumulative probabilities
        let mut cumulative = 0.0;
        let mut outcome = 0;

        for (idx, prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_value < cumulative {
                outcome = idx;
                break;
            }
        }

        // Collapse to the measured state
        for (idx, amp) in self.amplitudes_mut().iter_mut().enumerate() {
            if idx == outcome {
                *amp = Complex64::new(1.0, 0.0);
            } else {
                *amp = Complex64::new(0.0, 0.0);
            }
        }

        self.needs_normalization = false;
        Ok(outcome)
    }

    /// Get the expectation value of a Pauli string observable
    ///
    /// # Arguments
    /// * `observable` - Diagonal observable values for each basis state
    ///
    /// # Returns
    /// The expectation value ⟨ψ|O|ψ⟩
    ///
    /// # Errors
    /// Returns error if observable dimension doesn't match state dimension
    pub fn expectation_value(&self, observable: &[f64]) -> Result<f64> {
        if observable.len() != self.dimension() {
            return Err(StateError::DimensionMismatch {
                expected: self.dimension(),
                actual: observable.len(),
            });
        }

        Ok(self
            .amplitudes()
            .iter()
            .zip(observable.iter())
            .map(|(amp, obs)| amp.norm_sqr() * obs)
            .sum())
    }

    /// Compute the inner product with another state: ⟨self|other⟩
    ///
    /// # Arguments
    /// * `other` - The other quantum state
    ///
    /// # Returns
    /// The complex inner product
    ///
    /// # Errors
    /// Returns error if states have different dimensions
    pub fn inner_product(&self, other: &DenseState) -> Result<Complex64> {
        if self.dimension() != other.dimension() {
            return Err(StateError::DimensionMismatch {
                expected: self.dimension(),
                actual: other.dimension(),
            });
        }

        Ok(self
            .amplitudes()
            .iter()
            .zip(other.amplitudes().iter())
            .map(|(a, b)| a.conj() * b)
            .sum())
    }

    /// Compute the fidelity with another state: |⟨self|other⟩|^2
    ///
    /// # Arguments
    /// * `other` - The other quantum state
    ///
    /// # Returns
    /// The fidelity between 0 and 1
    ///
    /// # Errors
    /// Returns error if states have different dimensions
    pub fn fidelity(&self, other: &DenseState) -> Result<f64> {
        Ok(self.inner_product(other)?.norm_sqr())
    }

    /// Check if the underlying memory is properly aligned for SIMD
    #[inline]
    pub fn is_simd_aligned(&self) -> bool {
        self.vector.is_simd_aligned()
    }
}

impl fmt::Debug for DenseState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DenseState")
            .field("num_qubits", &self.num_qubits())
            .field("dimension", &self.dimension())
            .field("norm", &self.norm())
            .field("is_simd_aligned", &self.is_simd_aligned())
            .finish()
    }
}

impl Clone for DenseState {
    fn clone(&self) -> Self {
        self.clone_state().expect("Failed to clone state")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_new_dense_state() {
        let state = DenseState::new(2).unwrap();
        assert_eq!(state.num_qubits(), 2);
        assert_eq!(state.dimension(), 4);
        assert!(state.is_simd_aligned());
        assert!(state.is_normalized(1e-10));
    }

    #[test]
    fn test_initial_state() {
        let state = DenseState::new(3).unwrap();
        let amps = state.amplitudes();

        assert_eq!(amps[0], Complex64::new(1.0, 0.0));
        for i in 1..amps.len() {
            assert_eq!(amps[i], Complex64::new(0.0, 0.0));
        }
    }

    #[test]
    fn test_from_amplitudes() {
        let amplitudes = vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ];

        let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();
        assert_eq!(state.amplitudes(), amplitudes.as_slice());
    }

    #[test]
    fn test_apply_single_qubit_gate() {
        let mut state = DenseState::new(1).unwrap();

        // Pauli-X gate
        let x_gate = [
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ];

        state.apply_single_qubit_gate(&x_gate, 0).unwrap();

        // Should be in |1⟩ state
        assert_relative_eq!(state.amplitudes()[0].norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(state.amplitudes()[1].norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hadamard_gate() {
        let mut state = DenseState::new(1).unwrap();

        let h = 1.0 / 2.0_f64.sqrt();
        let hadamard = [
            [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
            [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
        ];

        state.apply_single_qubit_gate(&hadamard, 0).unwrap();

        // Should be in (|0⟩ + |1⟩)/√2
        assert_relative_eq!(state.amplitudes()[0].re, h, epsilon = 1e-10);
        assert_relative_eq!(state.amplitudes()[1].re, h, epsilon = 1e-10);
        assert!(state.is_normalized(1e-10));
    }

    #[test]
    fn test_get_probability() {
        let amplitudes = vec![
            Complex64::new(0.6, 0.0),
            Complex64::new(0.8, 0.0),
        ];

        let state = DenseState::from_amplitudes(1, &amplitudes).unwrap();

        assert_relative_eq!(state.get_probability(0).unwrap(), 0.36, epsilon = 1e-10);
        assert_relative_eq!(state.get_probability(1).unwrap(), 0.64, epsilon = 1e-10);
    }

    #[test]
    fn test_get_all_probabilities() {
        let h = 0.5;
        let amplitudes = vec![
            Complex64::new(h, 0.0),
            Complex64::new(h, 0.0),
            Complex64::new(h, 0.0),
            Complex64::new(h, 0.0),
        ];

        let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();
        let probs = state.get_all_probabilities();

        for prob in probs {
            assert_relative_eq!(prob, 0.25, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_measure_qubit() {
        let h = 1.0 / 2.0_f64.sqrt();
        let amplitudes = vec![
            Complex64::new(h, 0.0),
            Complex64::new(h, 0.0),
        ];

        let mut state = DenseState::from_amplitudes(1, &amplitudes).unwrap();

        // Measure with deterministic random value
        let outcome = state.measure_qubit(0, 0.25).unwrap();
        assert_eq!(outcome, 0);

        // State should be collapsed to |0⟩
        assert_relative_eq!(state.amplitudes()[0].norm(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(state.amplitudes()[1].norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_measure_all() {
        let mut state = DenseState::new(2).unwrap();
        let outcome = state.measure_all(0.5).unwrap();

        // Should measure |00⟩ since initial state is |00⟩
        assert_eq!(outcome, 0);
        assert_relative_eq!(state.amplitudes()[0].norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expectation_value() {
        let amplitudes = vec![
            Complex64::new(0.6, 0.0),
            Complex64::new(0.8, 0.0),
        ];
        let state = DenseState::from_amplitudes(1, &amplitudes).unwrap();

        // Z observable: diag([1, -1])
        let observable = vec![1.0, -1.0];
        let expectation = state.expectation_value(&observable).unwrap();

        // ⟨Z⟩ = 0.36 * 1 + 0.64 * (-1) = -0.28
        assert_relative_eq!(expectation, -0.28, epsilon = 1e-10);
    }

    #[test]
    fn test_inner_product() {
        let state1 = DenseState::new(2).unwrap();
        let state2 = DenseState::new(2).unwrap();

        let inner = state1.inner_product(&state2).unwrap();
        assert_relative_eq!(inner.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(inner.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fidelity() {
        let state1 = DenseState::new(2).unwrap();
        let state2 = DenseState::new(2).unwrap();

        let fidelity = state1.fidelity(&state2).unwrap();
        assert_relative_eq!(fidelity, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reset() {
        let amplitudes = vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ];

        let mut state = DenseState::from_amplitudes(2, &amplitudes).unwrap();
        state.reset();

        assert_eq!(state.amplitudes()[0], Complex64::new(1.0, 0.0));
        for i in 1..state.dimension() {
            assert_eq!(state.amplitudes()[i], Complex64::new(0.0, 0.0));
        }
    }

    #[test]
    fn test_clone_state() {
        let state1 = DenseState::new(2).unwrap();
        let state2 = state1.clone_state().unwrap();

        assert_eq!(state1.num_qubits(), state2.num_qubits());
        assert_eq!(state1.amplitudes(), state2.amplitudes());
    }

    #[test]
    fn test_normalize() {
        let amplitudes = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        let mut state = DenseState::from_amplitudes(1, &amplitudes).unwrap();
        state.normalize();

        assert!(state.is_normalized(1e-10));
        assert_relative_eq!(state.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_invalid_qubit_index() {
        let mut state = DenseState::new(2).unwrap();

        let x_gate = [
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ];

        let result = state.apply_single_qubit_gate(&x_gate, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let state1 = DenseState::new(2).unwrap();
        let state2 = DenseState::new(3).unwrap();

        let result = state1.inner_product(&state2);
        assert!(result.is_err());
    }

    #[test]
    fn test_alignment() {
        let state = DenseState::new(5).unwrap();
        assert!(state.is_simd_aligned());
    }
}
