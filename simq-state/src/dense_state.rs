//! Dense quantum state representation with 64-byte aligned vectors
//!
//! This module provides a high-level dense state representation that wraps
//! the SIMD-optimized StateVector with additional quantum operations like
//! measurement, gate application, and state manipulation.

use crate::error::{Result, StateError};
use crate::simd::{apply_single_qubit_gate, apply_two_qubit_gate, apply_cnot, apply_cz, apply_controlled_u, apply_crx, apply_cry, apply_crz, apply_diagonal_gate};
use crate::sparse_state::SparseState;
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

    /// Apply a diagonal single-qubit gate to the state (optimized)
    ///
    /// This method provides a faster path for diagonal gates (gates of the form
    /// [[a, 0], [0, b]]) such as Phase, RZ, S, T, and Z gates. It's 2-3x faster
    /// than `apply_single_qubit_gate` for diagonal gates because it avoids
    /// complex matrix multiplication.
    ///
    /// # Arguments
    /// * `diagonal` - [a, b] diagonal elements of the gate matrix [[a, 0], [0, b]]
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
    /// // Apply Z gate: [[1, 0], [0, -1]]
    /// let z_diagonal = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
    /// state.apply_diagonal_gate(z_diagonal, 0).unwrap();
    ///
    /// // Apply Phase(π/4): [[1, 0], [0, e^(iπ/4)]]
    /// let theta = std::f64::consts::PI / 4.0;
    /// let phase_diagonal = [
    ///     Complex64::new(1.0, 0.0),
    ///     Complex64::new(theta.cos(), theta.sin())
    /// ];
    /// state.apply_diagonal_gate(phase_diagonal, 0).unwrap();
    /// ```
    ///
    /// # Performance
    /// Approximately 2-3x faster than `apply_single_qubit_gate` for diagonal gates.
    /// Uses SIMD optimizations (AVX2/SSE2) when available.
    pub fn apply_diagonal_gate(
        &mut self,
        diagonal: [Complex64; 2],
        qubit: usize,
    ) -> Result<()> {
        let num_qubits = self.num_qubits();

        if qubit >= num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: qubit,
                num_qubits,
            });
        }

        apply_diagonal_gate(
            self.vector.amplitudes_mut(),
            diagonal,
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

    /// Apply a CNOT (Controlled-NOT) gate optimized for the controlled structure
    ///
    /// This uses direct amplitude manipulation instead of full 4×4 matrix
    /// multiplication, providing 3-4x speedup for large state vectors.
    ///
    /// # Arguments
    /// * `control` - Index of the control qubit (0-indexed)
    /// * `target` - Index of the target qubit (0-indexed)
    ///
    /// # Errors
    /// Returns error if qubit indices are invalid or equal
    ///
    /// # Example
    /// ```
    /// use simq_state::DenseState;
    ///
    /// let mut state = DenseState::new(2).unwrap();
    /// state.apply_cnot(0, 1).unwrap();
    /// ```
    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<()> {
        let num_qubits = self.num_qubits();

        if control >= num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: control,
                num_qubits,
            });
        }
        if target >= num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: target,
                num_qubits,
            });
        }
        if control == target {
            return Err(StateError::InvalidQubitIndex {
                index: control,
                num_qubits,
            });
        }

        apply_cnot(self.vector.amplitudes_mut(), control, target, num_qubits);
        self.needs_normalization = false;
        Ok(())
    }

    /// Apply a CZ (Controlled-Z) gate optimized for the controlled structure
    ///
    /// This applies a phase of -1 only to the |11⟩ state, which is much faster
    /// than full 4×4 matrix multiplication.
    ///
    /// # Arguments
    /// * `qubit1` - Index of the first qubit (0-indexed)
    /// * `qubit2` - Index of the second qubit (0-indexed)
    ///
    /// # Errors
    /// Returns error if qubit indices are invalid or equal
    ///
    /// # Example
    /// ```
    /// use simq_state::DenseState;
    ///
    /// let mut state = DenseState::new(2).unwrap();
    /// state.apply_cz(0, 1).unwrap();
    /// ```
    pub fn apply_cz(&mut self, qubit1: usize, qubit2: usize) -> Result<()> {
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

        apply_cz(self.vector.amplitudes_mut(), qubit1, qubit2, num_qubits);
        self.needs_normalization = false;
        Ok(())
    }

    /// Apply a controlled-U gate (U gate on target if control qubit is 1)
    ///
    /// More general than CNOT but still optimized compared to full 4×4 multiplication.
    ///
    /// # Arguments
    /// * `control` - Index of the control qubit (0-indexed)
    /// * `target` - Index of the target qubit (0-indexed)
    /// * `u_matrix` - 2×2 unitary matrix to apply to target when control=1
    ///
    /// # Errors
    /// Returns error if qubit indices are invalid or equal
    pub fn apply_controlled_u(
        &mut self,
        control: usize,
        target: usize,
        u_matrix: &[[Complex64; 2]; 2],
    ) -> Result<()> {
        let num_qubits = self.num_qubits();

        if control >= num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: control,
                num_qubits,
            });
        }
        if target >= num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: target,
                num_qubits,
            });
        }
        if control == target {
            return Err(StateError::InvalidQubitIndex {
                index: control,
                num_qubits,
            });
        }

        apply_controlled_u(
            self.vector.amplitudes_mut(),
            control,
            target,
            u_matrix,
            num_qubits,
        );
        self.needs_normalization = false;
        Ok(())
    }

    /// Apply a controlled-RX(θ) gate
    ///
    /// CRX(θ) = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ RX(θ)
    /// Applies RX(θ) to the target qubit when the control qubit is 1.
    ///
    /// # Arguments
    /// * `control` - Index of the control qubit (0-indexed)
    /// * `target` - Index of the target qubit (0-indexed)
    /// * `theta` - Rotation angle in radians
    ///
    /// # Errors
    /// Returns error if qubit indices are invalid or equal
    pub fn apply_crx(&mut self, control: usize, target: usize, theta: f64) -> Result<()> {
        let num_qubits = self.num_qubits();

        if control >= num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: control,
                num_qubits,
            });
        }
        if target >= num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: target,
                num_qubits,
            });
        }
        if control == target {
            return Err(StateError::InvalidQubitIndex {
                index: control,
                num_qubits,
            });
        }

        apply_crx(
            self.vector.amplitudes_mut(),
            control,
            target,
            theta,
            num_qubits,
        );
        self.needs_normalization = false;
        Ok(())
    }

    /// Apply a controlled-RY(θ) gate
    ///
    /// CRY(θ) = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ RY(θ)
    /// Applies RY(θ) to the target qubit when the control qubit is 1.
    ///
    /// # Arguments
    /// * `control` - Index of the control qubit (0-indexed)
    /// * `target` - Index of the target qubit (0-indexed)
    /// * `theta` - Rotation angle in radians
    ///
    /// # Errors
    /// Returns error if qubit indices are invalid or equal
    pub fn apply_cry(&mut self, control: usize, target: usize, theta: f64) -> Result<()> {
        let num_qubits = self.num_qubits();

        if control >= num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: control,
                num_qubits,
            });
        }
        if target >= num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: target,
                num_qubits,
            });
        }
        if control == target {
            return Err(StateError::InvalidQubitIndex {
                index: control,
                num_qubits,
            });
        }

        apply_cry(
            self.vector.amplitudes_mut(),
            control,
            target,
            theta,
            num_qubits,
        );
        self.needs_normalization = false;
        Ok(())
    }

    /// Apply a controlled-RZ(θ) gate
    ///
    /// CRZ(θ) = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ RZ(θ)
    /// Applies RZ(θ) to the target qubit when the control qubit is 1.
    ///
    /// # Arguments
    /// * `control` - Index of the control qubit (0-indexed)
    /// * `target` - Index of the target qubit (0-indexed)
    /// * `theta` - Rotation angle in radians
    ///
    /// # Errors
    /// Returns error if qubit indices are invalid or equal
    pub fn apply_crz(&mut self, control: usize, target: usize, theta: f64) -> Result<()> {
        let num_qubits = self.num_qubits();

        if control >= num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: control,
                num_qubits,
            });
        }
        if target >= num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: target,
                num_qubits,
            });
        }
        if control == target {
            return Err(StateError::InvalidQubitIndex {
                index: control,
                num_qubits,
            });
        }

        apply_crz(
            self.vector.amplitudes_mut(),
            control,
            target,
            theta,
            num_qubits,
        );
        self.needs_normalization = false;
        Ok(())
    }    /// Get the probability of measuring a specific computational basis state
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
    ///
    /// This method uses SIMD acceleration when available for improved performance.
    pub fn get_all_probabilities(&self) -> Vec<f64> {
        let amplitudes = self.amplitudes();
        let mut probabilities = vec![0.0; amplitudes.len()];

        // Use SIMD-optimized computation
        crate::simd::kernels::compute_probabilities(amplitudes, &mut probabilities);

        probabilities
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

    /// Convert from a sparse state to dense representation
    ///
    /// # Arguments
    /// * `sparse` - The sparse state to convert
    ///
    /// # Returns
    /// A new dense state with the same quantum state
    ///
    /// # Example
    /// ```
    /// use simq_state::{SparseState, DenseState};
    ///
    /// let sparse = SparseState::new(3).unwrap();
    /// let dense = DenseState::from_sparse(&sparse).unwrap();
    /// assert_eq!(dense.num_qubits(), 3);
    /// ```
    pub fn from_sparse(sparse: &SparseState) -> Result<Self> {
        let amplitudes = sparse.to_dense();
        Self::from_amplitudes(sparse.num_qubits(), &amplitudes)
    }

    /// Convert to a sparse state representation
    ///
    /// # Returns
    /// A new sparse state with non-zero amplitudes extracted
    ///
    /// # Note
    /// Amplitudes with magnitude < 1e-14 are considered zero and discarded
    ///
    /// # Example
    /// ```
    /// use simq_state::DenseState;
    ///
    /// let dense = DenseState::new(3).unwrap();
    /// let sparse = dense.to_sparse().unwrap();
    /// assert_eq!(sparse.num_amplitudes(), 1); // Only |000⟩ is non-zero
    /// ```
    pub fn to_sparse(&self) -> Result<SparseState> {
        SparseState::from_dense_amplitudes(self.num_qubits(), self.amplitudes())
    }

    /// Calculate the sparsity of the current state
    ///
    /// # Returns
    /// The fraction of non-zero amplitudes (0.0 to 1.0)
    ///
    /// # Note
    /// Amplitudes with magnitude < 1e-14 are considered zero
    pub fn sparsity(&self) -> f32 {
        let tolerance = 1e-14;
        let non_zero_count = self
            .amplitudes()
            .iter()
            .filter(|amp| amp.norm_sqr() > tolerance)
            .count();

        non_zero_count as f32 / self.dimension() as f32
    }

    /// Check if the state would benefit from sparse representation
    ///
    /// # Arguments
    /// * `threshold` - Sparsity threshold (default 0.1 = 10%)
    ///
    /// # Returns
    /// True if sparsity < threshold (suggesting sparse representation would be more efficient)
    pub fn should_convert_to_sparse(&self, threshold: f32) -> bool {
        self.sparsity() < threshold
    }

    /// Compute the partial trace (reduced density matrix) over specified qubits
    ///
    /// The partial trace operation traces out (discards) all qubits except those
    /// specified in `qubits_to_keep`, producing a reduced density matrix for the
    /// remaining subsystem. This is essential for subsystem measurement and
    /// analyzing entanglement.
    ///
    /// Given a pure state |ψ⟩ in the full Hilbert space, the reduced density
    /// matrix ρ_A for subsystem A is: ρ_A = Tr_B(|ψ⟩⟨ψ|)
    ///
    /// For dense states, this implementation uses efficient bit manipulation to
    /// iterate over all basis states and accumulate density matrix elements.
    ///
    /// # Arguments
    /// * `qubits_to_keep` - Indices of qubits to keep in the reduced system
    ///
    /// # Returns
    /// Reduced density matrix as a flattened vector in row-major order
    /// (dimension: 2^k × 2^k where k = qubits_to_keep.len())
    ///
    /// # Errors
    /// Returns error if any qubit index is out of bounds
    ///
    /// # Example
    /// ```
    /// use simq_state::DenseState;
    /// use num_complex::Complex64;
    ///
    /// // Create Bell state (|00⟩ + |11⟩)/√2
    /// let amplitudes = vec![
    ///     Complex64::new(0.7071067811865476, 0.0), // |00⟩
    ///     Complex64::new(0.0, 0.0),                 // |01⟩
    ///     Complex64::new(0.0, 0.0),                 // |10⟩
    ///     Complex64::new(0.7071067811865476, 0.0), // |11⟩
    /// ];
    /// let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();
    ///
    /// // Trace out qubit 1, keep qubit 0
    /// let rho = state.partial_trace(&[0]).unwrap();
    ///
    /// // For Bell state, reduced density matrix is maximally mixed: [[0.5, 0], [0, 0.5]]
    /// assert!((rho[0].re - 0.5).abs() < 1e-10); // ρ_00
    /// assert!((rho[3].re - 0.5).abs() < 1e-10); // ρ_11
    /// ```
    pub fn partial_trace(&self, qubits_to_keep: &[usize]) -> Result<Vec<Complex64>> {
        // Validate qubit indices
        for &qubit in qubits_to_keep {
            if qubit >= self.num_qubits() {
                return Err(StateError::InvalidQubitIndex {
                    index: qubit,
                    num_qubits: self.num_qubits(),
                });
            }
        }

        let reduced_dim = 1 << qubits_to_keep.len();
        let mut rho = vec![Complex64::new(0.0, 0.0); reduced_dim * reduced_dim];

        // Helper to extract reduced index from full index
        let extract_reduced_idx = |full_idx: usize| -> usize {
            let mut reduced_idx = 0usize;
            for (i, &q) in qubits_to_keep.iter().enumerate() {
                let bit = (full_idx >> q) & 1;
                reduced_idx |= bit << i;
            }
            reduced_idx
        };

        // Compute reduced density matrix by tracing out unwanted qubits
        // ρ_ij = Σ_k ⟨i,k|ψ⟩⟨ψ|j,k⟩ where k ranges over traced-out qubits
        //
        // For dense states, we iterate over all basis states
        let amplitudes = self.amplitudes();
        for idx_bra in 0..self.dimension() {
            let amp_bra = amplitudes[idx_bra];
            if amp_bra.norm_sqr() < 1e-14 {
                continue; // Skip zero amplitudes for efficiency
            }

            let reduced_i = extract_reduced_idx(idx_bra);

            for idx_ket in 0..self.dimension() {
                let amp_ket = amplitudes[idx_ket];
                if amp_ket.norm_sqr() < 1e-14 {
                    continue;
                }

                let reduced_j = extract_reduced_idx(idx_ket);

                // Check if traced-out qubits match
                let mut traced_match = true;
                for q in 0..self.num_qubits() {
                    if !qubits_to_keep.contains(&q) {
                        if ((idx_bra >> q) & 1) != ((idx_ket >> q) & 1) {
                            traced_match = false;
                            break;
                        }
                    }
                }

                if traced_match {
                    // Contribution: ⟨bra|ket⟩ = amp_bra* × amp_ket
                    let matrix_element = amp_bra.conj() * amp_ket;
                    rho[reduced_i * reduced_dim + reduced_j] += matrix_element;
                }
            }
        }

        Ok(rho)
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

    #[test]
    fn test_from_sparse() {
        let sparse = SparseState::new(3).unwrap();
        let dense = DenseState::from_sparse(&sparse).unwrap();

        assert_eq!(dense.num_qubits(), 3);
        assert_eq!(dense.amplitudes()[0], Complex64::new(1.0, 0.0));
        for i in 1..dense.dimension() {
            assert_eq!(dense.amplitudes()[i], Complex64::new(0.0, 0.0));
        }
    }

    #[test]
    fn test_to_sparse() {
        let dense = DenseState::new(3).unwrap();
        let sparse = dense.to_sparse().unwrap();

        assert_eq!(sparse.num_qubits(), 3);
        assert_eq!(sparse.num_amplitudes(), 1);
        assert_eq!(sparse.get_amplitude(0), Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_sparse_dense_roundtrip() {
        let mut sparse1 = SparseState::new(2).unwrap();
        sparse1.set_amplitude(0, Complex64::new(0.6, 0.0));
        sparse1.set_amplitude(1, Complex64::new(0.8, 0.0));

        let dense = DenseState::from_sparse(&sparse1).unwrap();
        let sparse2 = dense.to_sparse().unwrap();

        assert_relative_eq!(
            sparse2.get_amplitude(0).re,
            0.6,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            sparse2.get_amplitude(1).re,
            0.8,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_sparsity() {
        let mut state = DenseState::new(3).unwrap();
        // Only |000⟩ is non-zero: 1/8 = 0.125
        assert!((state.sparsity() - 0.125).abs() < 0.01);

        // Add more non-zero amplitudes
        let amps = state.amplitudes_mut();
        amps[1] = Complex64::new(0.5, 0.0);
        amps[2] = Complex64::new(0.5, 0.0);
        // Now 3/8 = 0.375
        assert!((state.sparsity() - 0.375).abs() < 0.01);
    }

    #[test]
    fn test_should_convert_to_sparse() {
        let state = DenseState::new(4).unwrap();
        // Only 1/16 = 0.0625 sparsity
        assert!(state.should_convert_to_sparse(0.1));
        assert!(!state.should_convert_to_sparse(0.05));
    }

    #[test]
    fn test_partial_trace_product_state() {
        // Product state |0⟩|1⟩
        let amplitudes = vec![
            Complex64::new(0.0, 0.0), // |00⟩
            Complex64::new(1.0, 0.0), // |01⟩
            Complex64::new(0.0, 0.0), // |10⟩
            Complex64::new(0.0, 0.0), // |11⟩
        ];
        let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

        // Trace out qubit 1, keep qubit 0
        let rho = state.partial_trace(&[0]).unwrap();

        // Should get |1⟩⟨1| = [[0, 0], [0, 1]]
        assert_relative_eq!(rho[0].re, 0.0, epsilon = 1e-10); // |0⟩⟨0|
        assert_relative_eq!(rho[1].re, 0.0, epsilon = 1e-10); // |0⟩⟨1|
        assert_relative_eq!(rho[2].re, 0.0, epsilon = 1e-10); // |1⟩⟨0|
        assert_relative_eq!(rho[3].re, 1.0, epsilon = 1e-10); // |1⟩⟨1|
    }

    #[test]
    fn test_partial_trace_bell_state() {
        // Bell state (|00⟩ + |11⟩)/√2
        let val = 1.0 / 2.0_f64.sqrt();
        let amplitudes = vec![
            Complex64::new(val, 0.0),  // |00⟩
            Complex64::new(0.0, 0.0),  // |01⟩
            Complex64::new(0.0, 0.0),  // |10⟩
            Complex64::new(val, 0.0),  // |11⟩
        ];
        let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

        // Trace out qubit 1, keep qubit 0
        let rho = state.partial_trace(&[0]).unwrap();

        // Should get maximally mixed state: [[0.5, 0], [0, 0.5]]
        assert_relative_eq!(rho[0].re, 0.5, epsilon = 1e-10); // |0⟩⟨0|
        assert_relative_eq!(rho[1].norm(), 0.0, epsilon = 1e-10); // |0⟩⟨1|
        assert_relative_eq!(rho[2].norm(), 0.0, epsilon = 1e-10); // |1⟩⟨0|
        assert_relative_eq!(rho[3].re, 0.5, epsilon = 1e-10); // |1⟩⟨1|
    }

    #[test]
    fn test_partial_trace_three_qubits() {
        // GHZ state (|000⟩ + |111⟩)/√2
        let val = 1.0 / 2.0_f64.sqrt();
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); 8];
        amplitudes[0] = Complex64::new(val, 0.0); // |000⟩
        amplitudes[7] = Complex64::new(val, 0.0); // |111⟩

        let state = DenseState::from_amplitudes(3, &amplitudes).unwrap();

        // Trace out qubit 2, keep qubits 0 and 1
        let rho = state.partial_trace(&[0, 1]).unwrap();

        // Should get correlations between qubits 0 and 1
        assert_relative_eq!(rho[0].re, 0.5, epsilon = 1e-10);   // |00⟩⟨00|
        assert_relative_eq!(rho[15].re, 0.5, epsilon = 1e-10);  // |11⟩⟨11|
        assert_relative_eq!(rho[5].norm(), 0.0, epsilon = 1e-10);  // |01⟩⟨01|
        assert_relative_eq!(rho[10].norm(), 0.0, epsilon = 1e-10); // |10⟩⟨10|
    }

    #[test]
    fn test_partial_trace_hermiticity() {
        // Reduced density matrix must be Hermitian
        let val = 1.0 / 2.0_f64.sqrt();
        let amplitudes = vec![
            Complex64::new(val, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(val, 0.0),
        ];
        let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

        let rho = state.partial_trace(&[0]).unwrap();

        // Check ρ† = ρ
        assert_relative_eq!(rho[0].im, 0.0, epsilon = 1e-10); // Diagonal must be real
        assert_relative_eq!(rho[3].im, 0.0, epsilon = 1e-10);
        assert_relative_eq!((rho[1] - rho[2].conj()).norm(), 0.0, epsilon = 1e-10); // Off-diagonal conjugate symmetry
    }

    #[test]
    fn test_partial_trace_unit_trace() {
        // Reduced density matrix must have trace = 1
        let val = 1.0 / 2.0_f64.sqrt();
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); 8];
        amplitudes[0] = Complex64::new(val, 0.0);
        amplitudes[5] = Complex64::new(val, 0.0);

        let state = DenseState::from_amplitudes(3, &amplitudes).unwrap();
        let rho = state.partial_trace(&[0]).unwrap();

        // Trace = ρ_00 + ρ_11
        let trace = rho[0].re + rho[3].re;
        assert_relative_eq!(trace, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_partial_trace_invalid_qubit() {
        let state = DenseState::new(2).unwrap();
        let result = state.partial_trace(&[5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_partial_trace_keep_all_qubits() {
        // Keeping all qubits should give density matrix of pure state
        let val = 1.0 / 2.0_f64.sqrt();
        let amplitudes = vec![
            Complex64::new(val, 0.0),
            Complex64::new(val, 0.0),
        ];
        let state = DenseState::from_amplitudes(1, &amplitudes).unwrap();

        let rho = state.partial_trace(&[0]).unwrap();

        // ρ = |ψ⟩⟨ψ|
        assert_relative_eq!(rho[0].re, 0.5, epsilon = 1e-10); // |0⟩⟨0|
        assert_relative_eq!(rho[1].re, 0.5, epsilon = 1e-10); // |0⟩⟨1|
        assert_relative_eq!(rho[2].re, 0.5, epsilon = 1e-10); // |1⟩⟨0|
        assert_relative_eq!(rho[3].re, 0.5, epsilon = 1e-10); // |1⟩⟨1|
    }

    // ========================================================================
    // Diagonal Gate Optimization Tests
    // ========================================================================

    #[test]
    fn test_apply_diagonal_gate_z() {
        let mut state = DenseState::new(1).unwrap();

        // Apply Z gate: [[1, 0], [0, -1]]
        let z_diagonal = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
        state.apply_diagonal_gate(z_diagonal, 0).unwrap();

        // |0⟩ should remain |0⟩
        assert_relative_eq!(state.amplitudes()[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(state.amplitudes()[0].im, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state.amplitudes()[1].norm(), 0.0, epsilon = 1e-10);
        assert!(state.is_normalized(1e-10));
    }

    #[test]
    fn test_apply_diagonal_gate_phase() {
        let mut state = DenseState::new(1).unwrap();

        // First apply X gate to go to |1⟩
        let x_gate = [
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ];
        state.apply_single_qubit_gate(&x_gate, 0).unwrap();

        // Apply Phase(π/2): [[1, 0], [0, i]]
        let phase_diagonal = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        state.apply_diagonal_gate(phase_diagonal, 0).unwrap();

        // |1⟩ should become i|1⟩
        assert_relative_eq!(state.amplitudes()[0].norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(state.amplitudes()[1].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state.amplitudes()[1].im, 1.0, epsilon = 1e-10);
        assert!(state.is_normalized(1e-10));
    }

    #[test]
    fn test_apply_diagonal_gate_rz() {
        let mut state = DenseState::new(2).unwrap();

        // Create superposition on qubit 0
        let h = 1.0 / 2.0_f64.sqrt();
        let hadamard = [
            [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
            [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
        ];
        state.apply_single_qubit_gate(&hadamard, 0).unwrap();

        // Apply RZ(π) on qubit 0: [[e^(-iπ/2), 0], [0, e^(iπ/2)]] = [[-i, 0], [0, i]]
        let theta = std::f64::consts::PI;
        let half_theta = theta / 2.0;
        let rz_diagonal = [
            Complex64::new(half_theta.cos(), -half_theta.sin()),
            Complex64::new(half_theta.cos(), half_theta.sin()),
        ];
        state.apply_diagonal_gate(rz_diagonal, 0).unwrap();

        // Check normalization is preserved
        assert!(state.is_normalized(1e-10));

        // The state should be (-i|0⟩ + i|1⟩)/√2 = i(|1⟩ - |0⟩)/√2
        let amp0 = state.amplitudes()[0];
        let amp1 = state.amplitudes()[1];

        assert_relative_eq!(amp0.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(amp0.im.abs(), h, epsilon = 1e-10);
        assert_relative_eq!(amp1.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(amp1.im.abs(), h, epsilon = 1e-10);
    }

    #[test]
    fn test_diagonal_gate_vs_general_gate_z() {
        // Test that diagonal gate optimization gives same result as general gate
        let mut state1 = DenseState::new(2).unwrap();
        let mut state2 = DenseState::new(2).unwrap();

        // Create a non-trivial state
        let h = 1.0 / 2.0_f64.sqrt();
        let hadamard = [
            [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
            [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
        ];
        state1.apply_single_qubit_gate(&hadamard, 0).unwrap();
        state1.apply_single_qubit_gate(&hadamard, 1).unwrap();
        state2.apply_single_qubit_gate(&hadamard, 0).unwrap();
        state2.apply_single_qubit_gate(&hadamard, 1).unwrap();

        // Apply Z gate using general method
        let z_matrix = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
        ];
        state1.apply_single_qubit_gate(&z_matrix, 1).unwrap();

        // Apply Z gate using diagonal optimization
        let z_diagonal = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
        state2.apply_diagonal_gate(z_diagonal, 1).unwrap();

        // Compare results
        let amps1 = state1.amplitudes();
        let amps2 = state2.amplitudes();
        for i in 0..amps1.len() {
            assert_relative_eq!(amps1[i].re, amps2[i].re, epsilon = 1e-10);
            assert_relative_eq!(amps1[i].im, amps2[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_diagonal_gate_vs_general_gate_phase() {
        // Test that diagonal gate optimization gives same result as general gate for Phase
        let mut state1 = DenseState::new(3).unwrap();
        let mut state2 = DenseState::new(3).unwrap();

        // Create a non-trivial state
        let h = 1.0 / 2.0_f64.sqrt();
        let hadamard = [
            [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
            [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
        ];
        for q in 0..3 {
            state1.apply_single_qubit_gate(&hadamard, q).unwrap();
            state2.apply_single_qubit_gate(&hadamard, q).unwrap();
        }

        // Apply Phase(π/4) using general method
        let theta = std::f64::consts::PI / 4.0;
        let phase_matrix = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(theta.cos(), theta.sin())],
        ];
        state1.apply_single_qubit_gate(&phase_matrix, 1).unwrap();

        // Apply Phase(π/4) using diagonal optimization
        let phase_diagonal = [
            Complex64::new(1.0, 0.0),
            Complex64::new(theta.cos(), theta.sin()),
        ];
        state2.apply_diagonal_gate(phase_diagonal, 1).unwrap();

        // Compare results
        let amps1 = state1.amplitudes();
        let amps2 = state2.amplitudes();
        for i in 0..amps1.len() {
            assert_relative_eq!(amps1[i].re, amps2[i].re, epsilon = 1e-10);
            assert_relative_eq!(amps1[i].im, amps2[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_diagonal_gate_multi_qubit() {
        // Test diagonal gates on different qubits in a multi-qubit system
        let mut state = DenseState::new(3).unwrap();

        // Create superposition on all qubits
        let h = 1.0 / 2.0_f64.sqrt();
        let hadamard = [
            [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
            [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
        ];

        for q in 0..3 {
            state.apply_single_qubit_gate(&hadamard, q).unwrap();
        }

        // Apply Z on qubit 0, S on qubit 1, T on qubit 2
        let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
        let s_diag = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        let sqrt2_2 = std::f64::consts::FRAC_1_SQRT_2;
        let t_diag = [Complex64::new(1.0, 0.0), Complex64::new(sqrt2_2, sqrt2_2)];

        state.apply_diagonal_gate(z_diag, 0).unwrap();
        state.apply_diagonal_gate(s_diag, 1).unwrap();
        state.apply_diagonal_gate(t_diag, 2).unwrap();

        // Verify normalization
        assert!(state.is_normalized(1e-10));

        // Verify we have 8 non-zero amplitudes
        let amps = state.amplitudes();
        assert_eq!(amps.len(), 8);
        for amp in amps {
            assert!(amp.norm() > 0.0);
        }
    }

    #[test]
    fn test_diagonal_gate_invalid_qubit() {
        let mut state = DenseState::new(2).unwrap();
        let z_diagonal = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];

        // Try to apply to invalid qubit
        let result = state.apply_diagonal_gate(z_diagonal, 2);
        assert!(result.is_err());
    }
}
