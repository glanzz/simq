//! Copy-on-Write quantum state for efficient branching
//!
//! This module provides a CoW (Copy-on-Write) quantum state implementation
//! that enables efficient state cloning and branching. This is particularly
//! useful for:
//! - Variational quantum algorithms (VQE, QAOA)
//! - Exploring multiple measurement outcomes
//! - Checkpointing and rollback
//! - Parallel circuit evaluation

use crate::dense_state::DenseState;
use crate::error::Result;
use num_complex::Complex64;
use std::fmt;
use std::sync::Arc;

/// Copy-on-Write quantum state with efficient branching
///
/// CowState uses reference counting (Arc) to share the underlying state
/// vector between clones. When a mutation is attempted, it automatically
/// creates a private copy (copy-on-write).
///
/// # Benefits
///
/// - **Zero-cost cloning**: Cloning is O(1) - just increments reference count
/// - **Lazy copying**: Only copies when actually mutating
/// - **Memory efficient**: Multiple branches share read-only data
/// - **Cache friendly**: Shared data stays hot in cache
///
/// # Example
///
/// ```
/// use simq_state::CowState;
/// use num_complex::Complex64;
///
/// // Create initial state
/// let state1 = CowState::new(10).unwrap();
/// println!("References: {}", state1.ref_count()); // 1
///
/// // Cheap clone - just increments ref count
/// let state2 = state1.clone();
/// println!("References: {}", state1.ref_count()); // 2
///
/// // No copying yet - both share the same data
/// assert!(state1.is_shared());
/// assert!(state2.is_shared());
/// ```
pub struct CowState {
    /// Shared reference to the underlying state
    /// Uses Arc for thread-safe reference counting
    state: Arc<DenseState>,
}

impl CowState {
    /// Create a new CoW state initialized to |0...0⟩
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    ///
    /// # Returns
    /// A new CoW state with reference count 1
    ///
    /// # Example
    /// ```
    /// use simq_state::CowState;
    ///
    /// let state = CowState::new(5).unwrap();
    /// assert_eq!(state.num_qubits(), 5);
    /// assert_eq!(state.ref_count(), 1);
    /// ```
    pub fn new(num_qubits: usize) -> Result<Self> {
        Ok(Self {
            state: Arc::new(DenseState::new(num_qubits)?),
        })
    }

    /// Create a CoW state from amplitudes
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    /// * `amplitudes` - Complex amplitudes (must have length 2^num_qubits)
    ///
    /// # Returns
    /// A new CoW state with the given amplitudes
    pub fn from_amplitudes(num_qubits: usize, amplitudes: &[Complex64]) -> Result<Self> {
        Ok(Self {
            state: Arc::new(DenseState::from_amplitudes(num_qubits, amplitudes)?),
        })
    }

    /// Get the number of qubits
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.state.num_qubits()
    }

    /// Get the state dimension (2^num_qubits)
    #[inline]
    pub fn dimension(&self) -> usize {
        self.state.dimension()
    }

    /// Get the current reference count
    ///
    /// # Returns
    /// Number of CowState instances sharing this data
    ///
    /// # Example
    /// ```
    /// use simq_state::CowState;
    ///
    /// let state1 = CowState::new(5).unwrap();
    /// let state2 = state1.clone();
    /// let state3 = state1.clone();
    /// assert_eq!(state1.ref_count(), 3);
    /// ```
    #[inline]
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.state)
    }

    /// Check if this state is currently shared (ref count > 1)
    ///
    /// # Returns
    /// True if multiple CowState instances share this data
    #[inline]
    pub fn is_shared(&self) -> bool {
        self.ref_count() > 1
    }

    /// Check if this state is uniquely owned (ref count == 1)
    ///
    /// # Returns
    /// True if this is the only reference to the data
    #[inline]
    pub fn is_unique(&self) -> bool {
        self.ref_count() == 1
    }

    /// Get read-only access to amplitudes
    ///
    /// This is always cheap - no copying occurs
    #[inline]
    pub fn amplitudes(&self) -> &[Complex64] {
        self.state.amplitudes()
    }

    /// Get the norm of the state
    #[inline]
    pub fn norm(&self) -> f64 {
        self.state.norm()
    }

    /// Check if the state is normalized
    #[inline]
    pub fn is_normalized(&self, epsilon: f64) -> bool {
        self.state.is_normalized(epsilon)
    }

    /// Make this state uniquely owned (copy if shared)
    ///
    /// This is the core of copy-on-write: if the state is shared,
    /// create a private copy. Otherwise, do nothing.
    ///
    /// # Returns
    /// True if a copy was made, false if already unique
    ///
    /// # Example
    /// ```
    /// use simq_state::CowState;
    ///
    /// let state1 = CowState::new(5).unwrap();
    /// let mut state2 = state1.clone();
    ///
    /// assert!(state2.is_shared());
    /// let copied = state2.make_unique().unwrap();
    /// assert!(copied); // A copy was made
    /// assert!(state2.is_unique());
    /// ```
    pub fn make_unique(&mut self) -> Result<bool> {
        if self.is_shared() {
            // Create a new private copy
            let new_state = self.state.clone_state()?;
            self.state = Arc::new(new_state);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get mutable access to the underlying state
    ///
    /// Automatically performs copy-on-write if the state is shared.
    /// This is the key method that enables transparent CoW.
    ///
    /// # Returns
    /// Mutable reference to the underlying DenseState
    fn state_mut(&mut self) -> Result<&mut DenseState> {
        self.make_unique()?;
        // SAFETY: After make_unique(), we have exclusive ownership
        Ok(Arc::get_mut(&mut self.state).expect("Arc should be unique after make_unique"))
    }

    /// Apply a single-qubit gate with automatic CoW
    ///
    /// If the state is shared, creates a private copy first.
    ///
    /// # Arguments
    /// * `matrix` - 2×2 gate matrix in row-major order
    /// * `qubit` - Index of the qubit to apply the gate to
    ///
    /// # Returns
    /// Statistics about the operation (whether a copy was made)
    ///
    /// # Example
    /// ```
    /// use simq_state::CowState;
    /// use num_complex::Complex64;
    ///
    /// let state1 = CowState::new(5).unwrap();
    /// let mut state2 = state1.clone(); // Cheap clone
    ///
    /// // Hadamard gate
    /// let h = 1.0 / 2.0_f64.sqrt();
    /// let hadamard = [
    ///     [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
    ///     [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
    /// ];
    ///
    /// let stats = state2.apply_single_qubit_gate(&hadamard, 0).unwrap();
    /// assert!(stats.copied); // A copy was made before mutation
    /// ```
    pub fn apply_single_qubit_gate(
        &mut self,
        matrix: &[[Complex64; 2]; 2],
        qubit: usize,
    ) -> Result<CowStats> {
        let copied = self.make_unique()?;
        self.state_mut()?.apply_single_qubit_gate(matrix, qubit)?;
        Ok(CowStats { copied })
    }

    /// Apply a two-qubit gate with automatic CoW
    ///
    /// # Arguments
    /// * `matrix` - 4×4 gate matrix in row-major order
    /// * `qubit1` - Index of the first qubit
    /// * `qubit2` - Index of the second qubit
    ///
    /// # Returns
    /// Statistics about the operation
    pub fn apply_two_qubit_gate(
        &mut self,
        matrix: &[[Complex64; 4]; 4],
        qubit1: usize,
        qubit2: usize,
    ) -> Result<CowStats> {
        let copied = self.make_unique()?;
        self.state_mut()?
            .apply_two_qubit_gate(matrix, qubit1, qubit2)?;
        Ok(CowStats { copied })
    }

    /// Measure a single qubit with automatic CoW
    ///
    /// # Arguments
    /// * `qubit` - Index of the qubit to measure
    /// * `random_value` - Random value in [0, 1) for outcome determination
    ///
    /// # Returns
    /// The measurement outcome (0 or 1) and operation statistics
    pub fn measure_qubit(&mut self, qubit: usize, random_value: f64) -> Result<(u8, CowStats)> {
        let copied = self.make_unique()?;
        let outcome = self.state_mut()?.measure_qubit(qubit, random_value)?;
        Ok((outcome, CowStats { copied }))
    }

    /// Normalize the state with automatic CoW
    pub fn normalize(&mut self) -> Result<CowStats> {
        let copied = self.make_unique()?;
        self.state_mut()?.normalize();
        Ok(CowStats { copied })
    }

    /// Reset the state to |0...0⟩ with automatic CoW
    pub fn reset(&mut self) -> Result<CowStats> {
        let copied = self.make_unique()?;
        self.state_mut()?.reset();
        Ok(CowStats { copied })
    }

    /// Branch the state - create a cheap clone for exploration
    ///
    /// This is semantically identical to `clone()` but makes the intent
    /// explicit: you're creating a branch for alternative computation.
    ///
    /// # Returns
    /// A new CowState sharing the same underlying data
    ///
    /// # Example
    /// ```
    /// use simq_state::CowState;
    ///
    /// let state = CowState::new(10).unwrap();
    ///
    /// // Branch to explore measurement outcomes
    /// let branch_0 = state.branch();
    /// let branch_1 = state.branch();
    ///
    /// // All three share the same data until mutation
    /// assert_eq!(state.ref_count(), 3);
    /// ```
    pub fn branch(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }

    /// Get the probability of measuring a specific basis state
    ///
    /// This is a read-only operation, so no copying occurs.
    pub fn get_probability(&self, basis_state: usize) -> Result<f64> {
        self.state.get_probability(basis_state)
    }

    /// Get all probabilities
    ///
    /// Read-only operation - no copying.
    pub fn get_all_probabilities(&self) -> Vec<f64> {
        self.state.get_all_probabilities()
    }

    /// Compute inner product with another state
    ///
    /// Read-only operation on both states.
    pub fn inner_product(&self, other: &CowState) -> Result<Complex64> {
        self.state.inner_product(&other.state)
    }

    /// Compute fidelity with another state
    pub fn fidelity(&self, other: &CowState) -> Result<f64> {
        self.state.fidelity(&other.state)
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let shared_memory = self.dimension() * std::mem::size_of::<Complex64>();
        let overhead = std::mem::size_of::<Arc<DenseState>>();

        MemoryStats {
            shared_memory,
            overhead_per_ref: overhead,
            total_refs: self.ref_count(),
            total_overhead: overhead * self.ref_count(),
        }
    }
}

/// Statistics about a CoW operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CowStats {
    /// Whether a copy was made during this operation
    pub copied: bool,
}

/// Memory usage statistics for CoW state
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Size of shared state vector data (bytes)
    pub shared_memory: usize,
    /// Overhead per reference (Arc metadata)
    pub overhead_per_ref: usize,
    /// Total number of references
    pub total_refs: usize,
    /// Total overhead from all references
    pub total_overhead: usize,
}

impl fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MemoryStats(shared: {} bytes, refs: {}, overhead: {} bytes total)",
            self.shared_memory, self.total_refs, self.total_overhead
        )
    }
}

impl Clone for CowState {
    /// Clone creates a cheap copy that shares the underlying data
    ///
    /// This is O(1) - just increments the reference count
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

impl fmt::Debug for CowState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CowState")
            .field("num_qubits", &self.num_qubits())
            .field("dimension", &self.dimension())
            .field("ref_count", &self.ref_count())
            .field("is_shared", &self.is_shared())
            .finish()
    }
}

impl fmt::Display for CowState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CowState({} qubits, {} refs, {})",
            self.num_qubits(),
            self.ref_count(),
            if self.is_shared() { "shared" } else { "unique" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_new_cow_state() {
        let state = CowState::new(3).unwrap();
        assert_eq!(state.num_qubits(), 3);
        assert_eq!(state.dimension(), 8);
        assert_eq!(state.ref_count(), 1);
        assert!(state.is_unique());
        assert!(!state.is_shared());
    }

    #[test]
    fn test_cheap_clone() {
        let state1 = CowState::new(10).unwrap();
        let state2 = state1.clone();
        let state3 = state1.clone();

        // All share the same data
        assert_eq!(state1.ref_count(), 3);
        assert_eq!(state2.ref_count(), 3);
        assert_eq!(state3.ref_count(), 3);

        assert!(state1.is_shared());
        assert!(state2.is_shared());
        assert!(state3.is_shared());
    }

    #[test]
    fn test_copy_on_write() {
        let state1 = CowState::new(3).unwrap();
        let mut state2 = state1.clone();

        assert_eq!(state1.ref_count(), 2);

        // Mutate state2 - should trigger copy
        let h = 1.0 / 2.0_f64.sqrt();
        let hadamard = [
            [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
            [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
        ];

        let stats = state2.apply_single_qubit_gate(&hadamard, 0).unwrap();
        assert!(stats.copied); // A copy was made

        // Now they're separate
        assert_eq!(state1.ref_count(), 1);
        assert_eq!(state2.ref_count(), 1);

        // state1 is still |000⟩
        assert_eq!(state1.amplitudes()[0], Complex64::new(1.0, 0.0));

        // state2 has been modified
        assert_relative_eq!(state2.amplitudes()[0].re, h, epsilon = 1e-10);
    }

    #[test]
    fn test_branch() {
        let state = CowState::new(5).unwrap();
        let branch1 = state.branch();
        let branch2 = state.branch();

        assert_eq!(state.ref_count(), 3);
        assert_eq!(branch1.ref_count(), 3);
        assert_eq!(branch2.ref_count(), 3);
    }

    #[test]
    fn test_make_unique() {
        let state1 = CowState::new(3).unwrap();
        let mut state2 = state1.clone();

        assert!(state2.is_shared());

        let copied = state2.make_unique().unwrap();
        assert!(copied); // A copy was made

        assert!(state2.is_unique());
        assert_eq!(state1.ref_count(), 1);
        assert_eq!(state2.ref_count(), 1);
    }

    #[test]
    fn test_make_unique_idempotent() {
        let mut state = CowState::new(3).unwrap();

        assert!(state.is_unique());

        let copied = state.make_unique().unwrap();
        assert!(!copied); // No copy needed - already unique
    }

    #[test]
    fn test_multiple_mutations() {
        let state1 = CowState::new(3).unwrap();
        let mut state2 = state1.clone();

        let h = 1.0 / 2.0_f64.sqrt();
        let hadamard = [
            [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
            [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
        ];

        // First mutation triggers copy
        let stats1 = state2.apply_single_qubit_gate(&hadamard, 0).unwrap();
        assert!(stats1.copied);

        // Second mutation doesn't copy (already unique)
        let stats2 = state2.apply_single_qubit_gate(&hadamard, 1).unwrap();
        assert!(!stats2.copied);

        // Third mutation doesn't copy
        let stats3 = state2.apply_single_qubit_gate(&hadamard, 2).unwrap();
        assert!(!stats3.copied);
    }

    #[test]
    fn test_read_operations_no_copy() {
        let state1 = CowState::new(3).unwrap();
        let state2 = state1.clone();

        // Read operations shouldn't trigger copy
        let _amps = state1.amplitudes();
        let _norm = state1.norm();
        let _prob = state1.get_probability(0).unwrap();
        let _probs = state1.get_all_probabilities();

        // Still shared
        assert_eq!(state1.ref_count(), 2);
        assert_eq!(state2.ref_count(), 2);
    }

    #[test]
    fn test_normalize() {
        let amplitudes = vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        let state1 = CowState::from_amplitudes(2, &amplitudes).unwrap();
        let mut state2 = state1.clone();

        let stats = state2.normalize().unwrap();
        assert!(stats.copied);

        // state1 still has unnormalized value
        assert_relative_eq!(state1.amplitudes()[0].re, 2.0, epsilon = 1e-10);

        // state2 is normalized
        assert_relative_eq!(state2.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reset() {
        let amplitudes = vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ];

        let state1 = CowState::from_amplitudes(2, &amplitudes).unwrap();
        let mut state2 = state1.clone();

        let stats = state2.reset().unwrap();
        assert!(stats.copied);

        // state2 is now |00⟩
        assert_eq!(state2.amplitudes()[0], Complex64::new(1.0, 0.0));
        assert_eq!(state2.amplitudes()[1], Complex64::new(0.0, 0.0));

        // state1 unchanged
        assert_relative_eq!(state1.amplitudes()[0].re, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_measure_qubit() {
        let h = 1.0 / 2.0_f64.sqrt();
        let amplitudes = vec![
            Complex64::new(h, 0.0),
            Complex64::new(h, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        let state1 = CowState::from_amplitudes(2, &amplitudes).unwrap();
        let mut state2 = state1.clone();

        let (outcome, stats) = state2.measure_qubit(0, 0.25).unwrap();
        assert!(stats.copied);
        assert!(outcome == 0 || outcome == 1);

        // state1 still in superposition
        assert_relative_eq!(state1.amplitudes()[0].re, h, epsilon = 1e-10);
    }

    #[test]
    fn test_inner_product() {
        let state1 = CowState::new(2).unwrap();
        let state2 = state1.clone();

        let inner = state1.inner_product(&state2).unwrap();
        assert_relative_eq!(inner.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(inner.im, 0.0, epsilon = 1e-10);

        // Still shared - read-only operation
        assert_eq!(state1.ref_count(), 2);
    }

    #[test]
    fn test_fidelity() {
        let state1 = CowState::new(2).unwrap();
        let state2 = state1.clone();

        let fidelity = state1.fidelity(&state2).unwrap();
        assert_relative_eq!(fidelity, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_memory_stats() {
        let state = CowState::new(10).unwrap();
        let stats = state.memory_stats();

        assert_eq!(stats.shared_memory, 1024 * 16); // 2^10 * sizeof(Complex64)
        assert_eq!(stats.total_refs, 1);
        assert!(stats.overhead_per_ref > 0);
    }

    #[test]
    fn test_drop_behavior() {
        let state1 = CowState::new(5).unwrap();
        let state2 = state1.clone();
        let state3 = state1.clone();

        assert_eq!(state1.ref_count(), 3);

        drop(state2);
        assert_eq!(state1.ref_count(), 2);

        drop(state3);
        assert_eq!(state1.ref_count(), 1);
    }
}
