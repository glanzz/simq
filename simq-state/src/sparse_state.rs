//! Sparse quantum state representation for memory-efficient simulation
//!
//! This module provides a sparse state vector implementation using AHashMap
//! for efficient representation of quantum states with few non-zero amplitudes.
//! Sparse representation is particularly efficient for circuits with:
//! - Few entangling gates
//! - Initial product states
//! - Shallow circuits (low gate depth)
//!
//! # Sparsity Tracking
//!
//! The SparseState maintains a density metric tracking the fraction of
//! non-zero amplitudes. This is used for automatic Sparse↔Dense conversion:
//! - Start simulation in Sparse mode for initial state |0...0⟩
//! - Track density after each gate
//! - Convert Sparse→Dense when density > threshold (default 10%)
//! - Never convert Dense→Sparse

use crate::error::{Result, StateError};
use ahash::AHashMap;
use num_complex::Complex64;
use std::fmt;

/// Default density threshold for automatic Sparse→Dense conversion
/// When density exceeds this, conversion to dense representation is triggered
const DEFAULT_DENSITY_THRESHOLD: f32 = 0.1;

/// Sparse quantum state representation using AHashMap
///
/// Stores non-zero amplitudes indexed by basis state (as u64).
/// This representation is memory-efficient for states with few non-zero terms.
///
/// # Example
///
/// ```
/// use simq_state::SparseState;
/// use num_complex::Complex64;
///
/// let mut state = SparseState::new(3).unwrap();
/// assert_eq!(state.num_qubits(), 3);
/// assert_eq!(state.get_amplitude(0), Complex64::new(1.0, 0.0)); // |000⟩
/// ```
#[derive(Clone)]
pub struct SparseState {
    /// Number of qubits
    num_qubits: usize,

    /// Maximum basis state index (2^num_qubits - 1)
    max_basis_idx: u64,

    /// Map from basis state index to amplitude
    amplitudes: AHashMap<u64, Complex64>,

    /// Density: fraction of non-zero amplitudes (non-zero entries / 2^num_qubits)
    density: f32,

    /// Threshold for Sparse→Dense conversion (default 0.1 = 10%)
    density_threshold: f32,
}

impl SparseState {
    /// Create a new sparse state initialized to |0...0⟩
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits (must be ≤ 30)
    ///
    /// # Returns
    /// A new sparse state with only the |0...0⟩ amplitude set to 1.0
    ///
    /// # Errors
    /// Returns error if num_qubits is too large (>30) or invalid
    ///
    /// # Example
    /// ```
    /// use simq_state::SparseState;
    ///
    /// let state = SparseState::new(4).unwrap();
    /// assert_eq!(state.num_qubits(), 4);
    /// assert_eq!(state.num_amplitudes(), 1);
    /// ```
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits > 30 {
            return Err(StateError::InvalidDimension {
                dimension: 1 << num_qubits,
            });
        }

        let max_basis_idx = ((1u64) << num_qubits) - 1;
        let mut amplitudes = AHashMap::new();

        // Initialize to |0...0⟩
        amplitudes.insert(0, Complex64::new(1.0, 0.0));

        Ok(Self {
            num_qubits,
            max_basis_idx,
            amplitudes,
            density: 1.0 / (1 << num_qubits) as f32,
            density_threshold: DEFAULT_DENSITY_THRESHOLD,
        })
    }

    /// Create a sparse state from raw amplitude data
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    /// * `amplitudes` - All amplitudes (dense format, length must be 2^num_qubits)
    ///
    /// # Returns
    /// A new sparse state with non-zero entries extracted
    ///
    /// # Errors
    /// Returns error if dimensions are invalid or amplitude count doesn't match
    pub fn from_dense_amplitudes(num_qubits: usize, amplitudes: &[Complex64]) -> Result<Self> {
        if num_qubits > 30 {
            return Err(StateError::InvalidDimension {
                dimension: 1 << num_qubits,
            });
        }

        let dimension = 1 << num_qubits;
        if amplitudes.len() != dimension {
            return Err(StateError::DimensionMismatch {
                expected: dimension,
                actual: amplitudes.len(),
            });
        }

        let mut state = Self::new(num_qubits)?;
        state.amplitudes.clear();

        // Extract non-zero amplitudes with tolerance
        let tolerance = 1e-14;
        for (idx, &amp) in amplitudes.iter().enumerate() {
            if amp.norm_sqr() > tolerance {
                state.amplitudes.insert(idx as u64, amp);
            }
        }

        state.update_density();
        Ok(state)
    }

    /// Create a sparse state from a specific basis state
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    /// * `basis_idx` - Index of the basis state
    ///
    /// # Returns
    /// A new sparse state initialized to |basis_idx⟩
    ///
    /// # Errors
    /// Returns error if basis_idx is out of range
    pub fn from_basis_state(num_qubits: usize, basis_idx: u64) -> Result<Self> {
        if num_qubits > 30 {
            return Err(StateError::InvalidDimension {
                dimension: 1 << num_qubits,
            });
        }

        let max_basis_idx = ((1u64) << num_qubits) - 1;
        if basis_idx > max_basis_idx {
            return Err(StateError::InvalidQubitIndex {
                index: basis_idx as usize,
                num_qubits,
            });
        }

        let mut amplitudes = AHashMap::new();
        amplitudes.insert(basis_idx, Complex64::new(1.0, 0.0));

        Ok(Self {
            num_qubits,
            max_basis_idx,
            amplitudes,
            density: 1.0 / (1 << num_qubits) as f32,
            density_threshold: DEFAULT_DENSITY_THRESHOLD,
        })
    }

    /// Set the density threshold for Sparse→Dense conversion
    ///
    /// # Arguments
    /// * `threshold` - Density threshold (0.0 to 1.0)
    pub fn set_density_threshold(&mut self, threshold: f32) {
        self.density_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Get the density threshold
    #[inline]
    pub fn density_threshold(&self) -> f32 {
        self.density_threshold
    }

    /// Get the current density (fraction of non-zero amplitudes)
    #[inline]
    pub fn density(&self) -> f32 {
        self.density
    }

    /// Check if conversion to dense representation is recommended
    ///
    /// Returns true if density exceeds the threshold
    #[inline]
    pub fn should_convert_to_dense(&self) -> bool {
        self.density > self.density_threshold
    }

    /// Get the number of qubits
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the total dimension (2^num_qubits)
    #[inline]
    pub fn dimension(&self) -> usize {
        1 << self.num_qubits
    }

    /// Get the number of non-zero amplitudes
    #[inline]
    pub fn num_amplitudes(&self) -> usize {
        self.amplitudes.len()
    }

    /// Get an amplitude by basis state index
    ///
    /// # Arguments
    /// * `basis_idx` - Index of the basis state
    ///
    /// # Returns
    /// The amplitude (0.0 if not stored)
    pub fn get_amplitude(&self, basis_idx: u64) -> Complex64 {
        self.amplitudes
            .get(&basis_idx)
            .copied()
            .unwrap_or_else(|| Complex64::new(0.0, 0.0))
    }

    /// Set an amplitude by basis state index
    ///
    /// # Arguments
    /// * `basis_idx` - Index of the basis state
    /// * `amplitude` - The amplitude value
    ///
    /// # Note
    /// If amplitude is very small (< 1e-14), the entry is removed
    pub fn set_amplitude(&mut self, basis_idx: u64, amplitude: Complex64) {
        let tolerance = 1e-14;
        if amplitude.norm_sqr() > tolerance {
            self.amplitudes.insert(basis_idx, amplitude);
        } else {
            self.amplitudes.remove(&basis_idx);
        }
        self.update_density();
    }

    /// Get a mutable reference to the amplitudes map
    ///
    /// # Warning
    /// Directly modifying the amplitudes map will not update density.
    /// Call `update_density()` after bulk modifications.
    #[inline]
    pub fn amplitudes_mut(&mut self) -> &mut AHashMap<u64, Complex64> {
        &mut self.amplitudes
    }

    /// Get a reference to the amplitudes map
    #[inline]
    pub fn amplitudes(&self) -> &AHashMap<u64, Complex64> {
        &self.amplitudes
    }

    /// Update the density metric
    ///
    /// Should be called after bulk amplitude modifications
    pub fn update_density(&mut self) {
        let non_zero_count = self.amplitudes.len() as f32;
        let total_dimension = (1 << self.num_qubits) as f32;
        self.density = non_zero_count / total_dimension;
    }

    /// Compute the norm of the state (should be 1.0 for normalized states)
    ///
    /// # Returns
    /// The L2 norm of the state vector
    pub fn norm(&self) -> f64 {
        self.amplitudes
            .values()
            .map(|amp| amp.norm_sqr())
            .sum::<f64>()
            .sqrt()
    }

    /// Normalize the state to unit norm
    ///
    /// # Errors
    /// Returns error if the state is zero or nearly zero
    pub fn normalize(&mut self) -> Result<()> {
        let norm = self.norm();

        if norm < 1e-14 {
            return Err(StateError::NotNormalized { norm });
        }

        if (norm - 1.0).abs() > 1e-10 {
            for amp in self.amplitudes.values_mut() {
                *amp /= norm;
            }
        }

        Ok(())
    }

    /// Check if the state is normalized (norm ≈ 1.0)
    ///
    /// # Arguments
    /// * `tolerance` - Tolerance for norm check
    pub fn is_normalized(&self, tolerance: f64) -> bool {
        let norm = self.norm();
        (norm - 1.0).abs() < tolerance
    }

    /// Convert sparse state to dense amplitude vector
    ///
    /// # Returns
    /// A vector of all 2^num_qubits amplitudes (including zeros)
    pub fn to_dense(&self) -> Vec<Complex64> {
        let mut dense = vec![Complex64::new(0.0, 0.0); 1 << self.num_qubits];
        for (&idx, &amp) in &self.amplitudes {
            dense[idx as usize] = amp;
        }
        dense
    }

    /// Apply a single-qubit unitary to a specific qubit
    ///
    /// # Arguments
    /// * `gate_matrix` - 2x2 unitary matrix as [a, b, c, d]
    /// * `qubit` - Target qubit index
    ///
    /// # Note
    /// For sparse states with few non-zero amplitudes, this is much more
    /// efficient than dense state representation.
    pub fn apply_single_qubit_gate(
        &mut self,
        gate_matrix: &[Complex64; 4],
        qubit: usize,
    ) -> Result<()> {
        if qubit >= self.num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: qubit,
                num_qubits: self.num_qubits,
            });
        }

        let mut new_amplitudes = AHashMap::new();
        let qubit_mask = 1u64 << qubit;

        // For each non-zero amplitude, compute contributions to both basis states
        for (&basis_idx, &amp) in &self.amplitudes {
            let bit = (basis_idx >> qubit) & 1;

            if bit == 0 {
                // This amplitude has qubit in |0⟩ state
                // a00 amplitude → a11 basis after gate
                let contribution_0 = gate_matrix[0] * amp; // |0⟩ ← |0⟩ (top-left)
                let contribution_1 = gate_matrix[1] * amp; // |1⟩ ← |0⟩ (top-right)

                if contribution_0.norm_sqr() > 1e-14 {
                    *new_amplitudes
                        .entry(basis_idx)
                        .or_insert(Complex64::new(0.0, 0.0)) += contribution_0;
                }
                if contribution_1.norm_sqr() > 1e-14 {
                    *new_amplitudes
                        .entry(basis_idx | qubit_mask)
                        .or_insert(Complex64::new(0.0, 0.0)) += contribution_1;
                }
            } else {
                // This amplitude has qubit in |1⟩ state
                let contribution_0 = gate_matrix[2] * amp; // |0⟩ ← |1⟩ (bottom-left)
                let contribution_1 = gate_matrix[3] * amp; // |1⟩ ← |1⟩ (bottom-right)

                if contribution_0.norm_sqr() > 1e-14 {
                    *new_amplitudes
                        .entry(basis_idx & !qubit_mask)
                        .or_insert(Complex64::new(0.0, 0.0)) += contribution_0;
                }
                if contribution_1.norm_sqr() > 1e-14 {
                    *new_amplitudes
                        .entry(basis_idx)
                        .or_insert(Complex64::new(0.0, 0.0)) += contribution_1;
                }
            }
        }

        // Remove near-zero entries
        new_amplitudes.retain(|_, amp| amp.norm_sqr() > 1e-14);
        self.amplitudes = new_amplitudes;
        self.update_density();

        Ok(())
    }

    /// Apply a two-qubit gate (4x4 unitary matrix)
    ///
    /// # Arguments
    /// * `gate_matrix` - 4x4 unitary matrix as [a00, a01, ..., a33]
    /// * `qubit0` - First target qubit
    /// * `qubit1` - Second target qubit
    ///
    /// # Note
    /// Gate matrix is indexed as: |q0 q1⟩ → [|00⟩, |01⟩, |10⟩, |11⟩]
    pub fn apply_two_qubit_gate(
        &mut self,
        gate_matrix: &[Complex64; 16],
        qubit0: usize,
        qubit1: usize,
    ) -> Result<()> {
        if qubit0 >= self.num_qubits || qubit1 >= self.num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: if qubit0 >= self.num_qubits {
                    qubit0
                } else {
                    qubit1
                },
                num_qubits: self.num_qubits,
            });
        }

        if qubit0 == qubit1 {
            return Err(StateError::InvalidQubitIndex {
                index: qubit0,
                num_qubits: self.num_qubits,
            });
        }

        let mask0 = 1u64 << qubit0;
        let mask1 = 1u64 << qubit1;

        let mut new_amplitudes = AHashMap::new();

        // For each non-zero amplitude, compute contributions
        for (&basis_idx, &amp) in &self.amplitudes {
            let bit0 = (basis_idx >> qubit0) & 1;
            let bit1 = (basis_idx >> qubit1) & 1;
            let input_state = (bit0 << 1) | bit1; // |bit0 bit1⟩

            // Iterate through output states |out0 out1⟩
            for out_state in 0..4 {
                let out_bit0 = (out_state >> 1) & 1;
                let out_bit1 = out_state & 1;

                let matrix_idx = (input_state * 4 + out_state) as usize;
                let contribution = gate_matrix[matrix_idx] * amp;

                if contribution.norm_sqr() > 1e-14 {
                    let mut new_basis_idx = basis_idx;

                    // Update qubit0
                    if out_bit0 == 1 {
                        new_basis_idx |= mask0;
                    } else {
                        new_basis_idx &= !mask0;
                    }

                    // Update qubit1
                    if out_bit1 == 1 {
                        new_basis_idx |= mask1;
                    } else {
                        new_basis_idx &= !mask1;
                    }

                    *new_amplitudes
                        .entry(new_basis_idx)
                        .or_insert(Complex64::new(0.0, 0.0)) += contribution;
                }
            }
        }

        // Remove near-zero entries
        new_amplitudes.retain(|_, amp| amp.norm_sqr() > 1e-14);
        self.amplitudes = new_amplitudes;
        self.update_density();

        Ok(())
    }

    /// Measure a single qubit and return the outcome
    ///
    /// This simulates measurement without collapsing the state (returns probabilities)
    ///
    /// # Arguments
    /// * `qubit` - Target qubit to measure
    ///
    /// # Returns
    /// Tuple of (probability_0, probability_1)
    pub fn measure_probability(&self, qubit: usize) -> Result<(f64, f64)> {
        if qubit >= self.num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: qubit,
                num_qubits: self.num_qubits,
            });
        }

        let mut prob_0 = 0.0;
        let mut prob_1 = 0.0;

        for (&basis_idx, &amp) in &self.amplitudes {
            let bit = (basis_idx >> qubit) & 1;
            let prob = amp.norm_sqr();

            if bit == 0 {
                prob_0 += prob;
            } else {
                prob_1 += prob;
            }
        }

        Ok((prob_0, prob_1))
    }

    /// Collapse the state by measurement
    ///
    /// # Arguments
    /// * `qubit` - Target qubit to measure
    /// * `outcome` - Measurement outcome (0 or 1)
    ///
    /// # Returns
    /// The probability of the measurement outcome
    pub fn measure_and_collapse(&mut self, qubit: usize, outcome: u32) -> Result<f64> {
        if qubit >= self.num_qubits {
            return Err(StateError::InvalidQubitIndex {
                index: qubit,
                num_qubits: self.num_qubits,
            });
        }

        if outcome > 1 {
            return Err(StateError::InvalidQubitIndex {
                index: outcome as usize,
                num_qubits: 1,
            });
        }

        let mut prob = 0.0;
        let mut collapsed = AHashMap::new();

        for (&basis_idx, &amp) in &self.amplitudes {
            let bit = ((basis_idx >> qubit) & 1) as u32;
            if bit == outcome {
                prob += amp.norm_sqr();
                collapsed.insert(basis_idx, amp);
            }
        }

        if prob < 1e-14 {
            return Err(StateError::NotNormalized { norm: 0.0 });
        }

        // Normalize
        let scale = Complex64::new((prob).sqrt().recip(), 0.0);
        for amp in collapsed.values_mut() {
            *amp *= scale;
        }

        self.amplitudes = collapsed;
        self.update_density();

        Ok(prob)
    }

    /// Compute expectation value of computational basis observable |i⟩⟨i|
    ///
    /// # Arguments
    /// * `basis_state` - Index of the basis state
    ///
    /// # Returns
    /// The expectation value ⟨ψ|P_i|ψ⟩
    pub fn expectation_basis(&self, basis_state: u64) -> Result<f64> {
        if basis_state > self.max_basis_idx {
            return Err(StateError::InvalidQubitIndex {
                index: basis_state as usize,
                num_qubits: self.num_qubits,
            });
        }

        let prob = self
            .amplitudes
            .get(&basis_state)
            .map(|amp| amp.norm_sqr())
            .unwrap_or(0.0);

        Ok(prob)
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
    /// use simq_state::SparseState;
    /// use num_complex::Complex64;
    ///
    /// // Create Bell state (|00⟩ + |11⟩)/√2
    /// let mut state = SparseState::new(2).unwrap();
    /// state.set_amplitude(0, Complex64::new(0.7071067811865476, 0.0));
    /// state.set_amplitude(3, Complex64::new(0.7071067811865476, 0.0));
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
            if qubit >= self.num_qubits {
                return Err(StateError::InvalidQubitIndex {
                    index: qubit,
                    num_qubits: self.num_qubits,
                });
            }
        }

        // Create masks for extracting bits
        let reduced_dim = 1 << qubits_to_keep.len();
        let mut rho = vec![Complex64::new(0.0, 0.0); reduced_dim * reduced_dim];

        // Helper to extract reduced index from full index
        let extract_reduced_idx = |full_idx: u64| -> u64 {
            let mut reduced_idx = 0u64;
            for (i, &q) in qubits_to_keep.iter().enumerate() {
                let bit = (full_idx >> q) & 1;
                reduced_idx |= bit << i;
            }
            reduced_idx
        };

        // Compute density matrix: ρ = |ψ⟩⟨ψ|, then trace out unwanted qubits
        // ρ_ij = Σ_k ⟨i,k|ψ⟩⟨ψ|j,k⟩ where k ranges over traced-out qubits
        //
        // For sparse states, we iterate over all pairs of non-zero amplitudes
        // and accumulate contributions where the traced-out qubits match
        for (&idx_bra, &amp_bra) in &self.amplitudes {
            let reduced_i = extract_reduced_idx(idx_bra);

            for (&idx_ket, &amp_ket) in &self.amplitudes {
                let reduced_j = extract_reduced_idx(idx_ket);

                // Check if traced-out qubits match
                let mut traced_match = true;
                for q in 0..self.num_qubits {
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
                    rho[(reduced_i as usize) * reduced_dim + (reduced_j as usize)] +=
                        matrix_element;
                }
            }
        }

        Ok(rho)
    }
}

impl fmt::Debug for SparseState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseState")
            .field("num_qubits", &self.num_qubits)
            .field("num_amplitudes", &self.amplitudes.len())
            .field("density", &self.density)
            .field("dimension", &self.dimension())
            .finish()
    }
}

impl fmt::Display for SparseState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparseState({} qubits, {} non-zero amplitudes, {:.2}% density)",
            self.num_qubits,
            self.num_amplitudes(),
            self.density * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_state() {
        let state = SparseState::new(3).unwrap();
        assert_eq!(state.num_qubits(), 3);
        assert_eq!(state.dimension(), 8);
        assert_eq!(state.num_amplitudes(), 1);
        assert_eq!(state.get_amplitude(0), Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_from_basis_state() {
        let state = SparseState::from_basis_state(3, 5).unwrap();
        assert_eq!(state.num_amplitudes(), 1);
        assert_eq!(state.get_amplitude(5), Complex64::new(1.0, 0.0));
        assert_eq!(state.get_amplitude(0), Complex64::new(0.0, 0.0));
    }

    #[test]
    fn test_set_amplitude() {
        let mut state = SparseState::new(2).unwrap();
        state.set_amplitude(0, Complex64::new(0.707, 0.0));
        state.set_amplitude(1, Complex64::new(0.707, 0.0));

        assert_eq!(state.num_amplitudes(), 2);
        assert!(state.is_normalized(1e-3));
    }

    #[test]
    fn test_normalize() {
        let mut state = SparseState::new(2).unwrap();
        state.set_amplitude(0, Complex64::new(2.0, 0.0));
        state.set_amplitude(1, Complex64::new(1.0, 0.0));

        state.normalize().unwrap();
        assert!(state.is_normalized(1e-10));
    }

    #[test]
    fn test_density_tracking() {
        let mut state = SparseState::new(4).unwrap();
        // 1 amplitude / 16 states = 0.0625
        assert!((state.density() - 0.0625).abs() < 1e-3);

        // Add more amplitudes
        for i in 1..3 {
            state.set_amplitude(i, Complex64::new(0.5, 0.0));
        }
        // 3 amplitudes / 16 states = 0.1875
        assert!((state.density() - 0.1875).abs() < 1e-3);
    }

    #[test]
    fn test_single_qubit_gate() {
        let mut state = SparseState::new(2).unwrap();

        // Hadamard matrix: 1/√2 * [[1, 1], [1, -1]]
        let h_gate = [
            Complex64::new(0.707, 0.0),
            Complex64::new(0.707, 0.0),
            Complex64::new(0.707, 0.0),
            Complex64::new(-0.707, 0.0),
        ];

        state.apply_single_qubit_gate(&h_gate, 0).unwrap();

        // Should create superposition: (|0⟩ + |1⟩)/√2 = |00⟩ + |01⟩
        assert_eq!(state.num_amplitudes(), 2);
        let amp0 = state.get_amplitude(0);
        let amp1 = state.get_amplitude(1);

        assert!((amp0.norm_sqr() - 0.5).abs() < 1e-3);
        assert!((amp1.norm_sqr() - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_to_dense() {
        let mut state = SparseState::new(2).unwrap();
        state.set_amplitude(0, Complex64::new(0.5, 0.0));
        state.set_amplitude(1, Complex64::new(0.5, 0.0));
        state.set_amplitude(2, Complex64::new(0.707, 0.0));

        let dense = state.to_dense();
        assert_eq!(dense.len(), 4);
        assert_eq!(dense[0], Complex64::new(0.5, 0.0));
        assert_eq!(dense[1], Complex64::new(0.5, 0.0));
        assert_eq!(dense[2], Complex64::new(0.707, 0.0));
        assert_eq!(dense[3], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn test_measure_probability() {
        let mut state = SparseState::new(2).unwrap();
        state.set_amplitude(0, Complex64::new(0.707, 0.0));
        state.set_amplitude(1, Complex64::new(0.707, 0.0));
        state.normalize().unwrap();

        let (prob_0, prob_1) = state.measure_probability(0).unwrap();
        // Basis states: 0 = |00⟩ (qubit 0 = 0), 1 = |01⟩ (qubit 0 = 1)
        // So we measure qubit 0: half the time 0, half the time 1
        assert!((prob_0 - 0.5).abs() < 1e-3);
        assert!((prob_1 - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_measure_and_collapse() {
        let mut state = SparseState::new(2).unwrap();
        state.set_amplitude(0, Complex64::new(0.707, 0.0));
        state.set_amplitude(2, Complex64::new(0.707, 0.0)); // |10⟩
        state.normalize().unwrap();

        let prob = state.measure_and_collapse(1, 0).unwrap();
        assert!((prob - 0.5).abs() < 1e-3);
        assert_eq!(state.num_amplitudes(), 1);
        assert_eq!(state.get_amplitude(0), Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_expectation_basis() {
        let mut state = SparseState::new(2).unwrap();
        state.set_amplitude(0, Complex64::new(0.6, 0.0));
        state.set_amplitude(1, Complex64::new(0.8, 0.0));

        let exp0 = state.expectation_basis(0).unwrap();
        let exp1 = state.expectation_basis(1).unwrap();

        assert!((exp0 - 0.36).abs() < 1e-3);
        assert!((exp1 - 0.64).abs() < 1e-3);
    }

    #[test]
    fn test_from_dense_amplitudes() {
        let amplitudes = vec![
            Complex64::new(0.707, 0.0),
            Complex64::new(0.707, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        let state = SparseState::from_dense_amplitudes(2, &amplitudes).unwrap();
        assert_eq!(state.num_amplitudes(), 2);
        assert_eq!(state.get_amplitude(0), amplitudes[0]);
        assert_eq!(state.get_amplitude(1), amplitudes[1]);
    }

    #[test]
    fn test_density_threshold() {
        let mut state = SparseState::new(3).unwrap();
        state.set_density_threshold(0.15);
        assert_eq!(state.density_threshold(), 0.15);
        assert!(!state.should_convert_to_dense()); // density = 1/8 = 0.125

        for i in 1..2 {
            state.set_amplitude(i, Complex64::new(0.5, 0.0));
        }
        assert!(state.should_convert_to_dense()); // density = 2/8 = 0.25
    }

    #[test]
    fn test_two_qubit_gate_cnot() {
        let mut state = SparseState::new(2).unwrap();
        // Create Bell state: (|00⟩ + |11⟩)/√2
        // First apply H to qubit 0
        let h_gate = [
            Complex64::new(0.707, 0.0),
            Complex64::new(0.707, 0.0),
            Complex64::new(0.707, 0.0),
            Complex64::new(-0.707, 0.0),
        ];
        state.apply_single_qubit_gate(&h_gate, 0).unwrap();

        // Apply CNOT (qubit 0 is control, qubit 1 is target)
        // CNOT matrix: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
        let cnot = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), // |00⟩ ← |00⟩
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), // |01⟩ ← |01⟩
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), // |11⟩ ← |10⟩
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0), // |10⟩ ← |11⟩
        ];

        state.apply_two_qubit_gate(&cnot, 0, 1).unwrap();

        // Should have |00⟩ and |11⟩ with equal probability
        let amp00 = state.get_amplitude(0);
        let amp11 = state.get_amplitude(3);

        assert!((amp00.norm_sqr() - 0.5).abs() < 1e-2);
        assert!((amp11.norm_sqr() - 0.5).abs() < 1e-2);
    }

    #[test]
    fn test_partial_trace_product_state() {
        // Product state |0⟩|1⟩ should give pure states when traced out
        let mut state = SparseState::new(2).unwrap();
        state.set_amplitude(0, Complex64::new(0.0, 0.0));
        state.set_amplitude(1, Complex64::new(1.0, 0.0)); // |01⟩ in binary: qubit0=1, qubit1=0

        // Trace out qubit 1, keep qubit 0
        let rho = state.partial_trace(&[0]).unwrap();

        // Should get |1⟩⟨1| = [[0, 0], [0, 1]]
        assert!((rho[0].re - 0.0).abs() < 1e-10); // |0⟩⟨0|
        assert!((rho[1].re - 0.0).abs() < 1e-10); // |0⟩⟨1|
        assert!((rho[2].re - 0.0).abs() < 1e-10); // |1⟩⟨0|
        assert!((rho[3].re - 1.0).abs() < 1e-10); // |1⟩⟨1|
    }

    #[test]
    fn test_partial_trace_bell_state() {
        // Bell state (|00⟩ + |11⟩)/√2
        let mut state = SparseState::new(2).unwrap();
        let val = 1.0 / 2.0_f64.sqrt();
        state.set_amplitude(0, Complex64::new(val, 0.0)); // |00⟩
        state.set_amplitude(3, Complex64::new(val, 0.0)); // |11⟩

        // Trace out qubit 1, keep qubit 0
        let rho = state.partial_trace(&[0]).unwrap();

        // Should get maximally mixed state: [[0.5, 0], [0, 0.5]]
        assert!((rho[0].re - 0.5).abs() < 1e-10); // |0⟩⟨0|
        assert!((rho[1].norm() - 0.0).abs() < 1e-10); // |0⟩⟨1|
        assert!((rho[2].norm() - 0.0).abs() < 1e-10); // |1⟩⟨0|
        assert!((rho[3].re - 0.5).abs() < 1e-10); // |1⟩⟨1|
    }

    #[test]
    fn test_partial_trace_three_qubits() {
        // GHZ state (|000⟩ + |111⟩)/√2
        let mut state = SparseState::new(3).unwrap();
        let val = 1.0 / 2.0_f64.sqrt();
        state.set_amplitude(0, Complex64::new(val, 0.0)); // |000⟩
        state.set_amplitude(7, Complex64::new(val, 0.0)); // |111⟩

        // Trace out qubit 2, keep qubits 0 and 1
        let rho = state.partial_trace(&[0, 1]).unwrap();

        // Should get maximally mixed on two qubits with correlations
        // The reduced state should have diagonal: [0.5, 0, 0, 0.5] in basis |00⟩, |01⟩, |10⟩, |11⟩
        assert!((rho[0].re - 0.5).abs() < 1e-10); // |00⟩⟨00|
        assert!((rho[15].re - 0.5).abs() < 1e-10); // |11⟩⟨11|
        assert!((rho[5].norm() - 0.0).abs() < 1e-10); // |01⟩⟨01|
        assert!((rho[10].norm() - 0.0).abs() < 1e-10); // |10⟩⟨10|
    }

    #[test]
    fn test_partial_trace_hermiticity() {
        // Reduced density matrix must be Hermitian
        let mut state = SparseState::new(2).unwrap();
        let val = 1.0 / 2.0_f64.sqrt();
        state.set_amplitude(0, Complex64::new(val, 0.0));
        state.set_amplitude(3, Complex64::new(val, 0.0));

        let rho = state.partial_trace(&[0]).unwrap();

        // Check ρ† = ρ
        assert!((rho[0].im).abs() < 1e-10); // Diagonal must be real
        assert!((rho[3].im).abs() < 1e-10);
        assert!((rho[1] - rho[2].conj()).norm() < 1e-10); // Off-diagonal conjugate symmetry
    }

    #[test]
    fn test_partial_trace_unit_trace() {
        // Reduced density matrix must have trace = 1
        let mut state = SparseState::new(3).unwrap();
        let val = 1.0 / 2.0_f64.sqrt();
        state.set_amplitude(0, Complex64::new(val, 0.0));
        state.set_amplitude(5, Complex64::new(val, 0.0));

        let rho = state.partial_trace(&[0]).unwrap();

        // Trace = ρ_00 + ρ_11
        let trace = rho[0].re + rho[3].re;
        assert!((trace - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_partial_trace_invalid_qubit() {
        let state = SparseState::new(2).unwrap();
        let result = state.partial_trace(&[5]);
        assert!(result.is_err());
    }
}
