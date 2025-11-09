//! Type-safe circuit builder with const generics

use crate::qubit_ref::Qubit;
use crate::{Circuit, Gate, QubitId, Result};
use smallvec::SmallVec;
use std::sync::Arc;

/// A type-safe builder for quantum circuits with compile-time qubit tracking
///
/// The const generic parameter `N` represents the number of qubits.
/// This allows the type system to prevent many common errors at compile time.
///
/// # Type Safety
///
/// The builder uses const generics to enforce qubit bounds at compile time:
/// - Qubit references carry the circuit size in their type
/// - Invalid qubit indices are caught when creating `Qubit<N>` references
/// - Cross-circuit qubit usage is prevented by the type system
///
/// # Performance
///
/// The builder is a zero-cost abstraction:
/// - All methods are marked `#[inline]` for aggressive optimization
/// - Qubit references are stack-allocated with zero overhead
/// - Final circuit construction moves ownership without copying
///
/// # Example
/// ```
/// use simq_core::circuit_builder::CircuitBuilder;
/// use simq_core::Gate;
/// use std::sync::Arc;
///
/// # #[derive(Debug)]
/// # struct MockGate;
/// # impl Gate for MockGate {
/// #     fn name(&self) -> &str { "H" }
/// #     fn num_qubits(&self) -> usize { 1 }
/// # }
/// let mut builder = CircuitBuilder::<3>::new();
/// let [q0, q1, q2] = builder.qubits();
///
/// let gate = Arc::new(MockGate);
/// builder.apply_gate(gate, &[q0]).unwrap();
///
/// let circuit = builder.build();
/// assert_eq!(circuit.num_qubits(), 3);
/// ```
pub struct CircuitBuilder<const N: usize> {
    circuit: Circuit,
}

impl<const N: usize> CircuitBuilder<N> {
    /// Create a new circuit builder for N qubits
    ///
    /// # Panics
    /// Panics if N is 0
    ///
    /// # Example
    /// ```
    /// use simq_core::circuit_builder::CircuitBuilder;
    ///
    /// let builder = CircuitBuilder::<5>::new();
    /// assert_eq!(builder.num_qubits(), 5);
    /// assert_eq!(builder.num_operations(), 0);
    /// ```
    #[inline]
    pub fn new() -> Self {
        assert!(N > 0, "Circuit must have at least one qubit");
        Self {
            circuit: Circuit::new(N),
        }
    }

    /// Create a builder with pre-allocated capacity for operations
    ///
    /// Use this when you know approximately how many operations you'll add
    /// to avoid reallocations.
    ///
    /// # Example
    /// ```
    /// use simq_core::circuit_builder::CircuitBuilder;
    ///
    /// // Pre-allocate space for 100 operations
    /// let builder = CircuitBuilder::<5>::with_capacity(100);
    /// assert_eq!(builder.num_qubits(), 5);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(N > 0, "Circuit must have at least one qubit");
        Self {
            circuit: Circuit::with_capacity(N, capacity),
        }
    }

    /// Get a qubit reference by index
    ///
    /// # Errors
    /// Returns error if index >= N
    ///
    /// # Example
    /// ```
    /// use simq_core::circuit_builder::CircuitBuilder;
    ///
    /// let builder = CircuitBuilder::<3>::new();
    /// let q0 = builder.qubit(0).unwrap();
    /// assert_eq!(q0.index(), 0);
    ///
    /// // Out of bounds
    /// let invalid = builder.qubit(5);
    /// assert!(invalid.is_err());
    /// ```
    #[inline]
    pub fn qubit(&self, index: usize) -> Result<Qubit<N>> {
        Qubit::new(index)
    }

    /// Get all qubits as an array
    ///
    /// This is a convenience method for getting all qubit references at once.
    /// The array is stack-allocated and has zero runtime cost.
    ///
    /// # Example
    /// ```
    /// use simq_core::circuit_builder::CircuitBuilder;
    ///
    /// let builder = CircuitBuilder::<3>::new();
    /// let [q0, q1, q2] = builder.qubits();
    ///
    /// assert_eq!(q0.index(), 0);
    /// assert_eq!(q1.index(), 1);
    /// assert_eq!(q2.index(), 2);
    /// ```
    #[inline]
    pub fn qubits(&self) -> [Qubit<N>; N] {
        // Create array using array initialization with index
        std::array::from_fn(|i| unsafe { Qubit::new_unchecked(i) })
    }

    /// Apply a gate to the specified qubits
    ///
    /// This is the core method for building circuits. It validates the gate
    /// operation and adds it to the circuit.
    ///
    /// # Errors
    /// Returns error if:
    /// - Qubit count doesn't match gate requirements
    /// - Duplicate qubits specified
    ///
    /// # Example
    /// ```
    /// use simq_core::circuit_builder::CircuitBuilder;
    /// use simq_core::Gate;
    /// use std::sync::Arc;
    ///
    /// # #[derive(Debug)]
    /// # struct MockGate;
    /// # impl Gate for MockGate {
    /// #     fn name(&self) -> &str { "H" }
    /// #     fn num_qubits(&self) -> usize { 1 }
    /// # }
    /// let mut builder = CircuitBuilder::<2>::new();
    /// let q0 = builder.qubit(0).unwrap();
    ///
    /// let gate = Arc::new(MockGate);
    /// builder.apply_gate(gate, &[q0]).unwrap();
    ///
    /// assert_eq!(builder.num_operations(), 1);
    /// ```
    #[inline]
    pub fn apply_gate(&mut self, gate: Arc<dyn Gate>, qubits: &[Qubit<N>]) -> Result<&mut Self> {
        // Convert Qubit<N> to QubitId using SmallVec for stack allocation
        let qubit_ids: SmallVec<[QubitId; 2]> = qubits.iter().map(|q| q.to_qubit_id()).collect();

        self.circuit.add_gate(gate, &qubit_ids)?;
        Ok(self)
    }

    /// Get the number of qubits (always returns N)
    ///
    /// # Example
    /// ```
    /// use simq_core::circuit_builder::CircuitBuilder;
    ///
    /// let builder = CircuitBuilder::<5>::new();
    /// assert_eq!(builder.num_qubits(), 5);
    /// ```
    #[inline]
    pub const fn num_qubits(&self) -> usize {
        N
    }

    /// Get the current number of operations
    ///
    /// # Example
    /// ```
    /// use simq_core::circuit_builder::CircuitBuilder;
    ///
    /// let builder = CircuitBuilder::<3>::new();
    /// assert_eq!(builder.num_operations(), 0);
    /// ```
    #[inline]
    pub fn num_operations(&self) -> usize {
        self.circuit.len()
    }

    /// Check if the circuit is empty (no operations)
    ///
    /// # Example
    /// ```
    /// use simq_core::circuit_builder::CircuitBuilder;
    ///
    /// let builder = CircuitBuilder::<3>::new();
    /// assert!(builder.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.circuit.is_empty()
    }

    /// Validate the circuit
    ///
    /// Checks that all operations are valid. This is automatically
    /// called during gate application, but can be useful for debugging.
    ///
    /// # Example
    /// ```
    /// use simq_core::circuit_builder::CircuitBuilder;
    ///
    /// let builder = CircuitBuilder::<3>::new();
    /// assert!(builder.validate().is_ok());
    /// ```
    #[inline]
    pub fn validate(&self) -> Result<()> {
        self.circuit.validate()
    }

    /// Build the final circuit, consuming the builder
    ///
    /// This transfers ownership of the circuit without copying.
    ///
    /// # Example
    /// ```
    /// use simq_core::circuit_builder::CircuitBuilder;
    ///
    /// let builder = CircuitBuilder::<3>::new();
    /// let circuit = builder.build();
    /// assert_eq!(circuit.num_qubits(), 3);
    /// ```
    #[inline]
    pub fn build(self) -> Circuit {
        self.circuit
    }

    /// Get a reference to the underlying circuit (for inspection)
    ///
    /// Use this to inspect the circuit without consuming the builder.
    ///
    /// # Example
    /// ```
    /// use simq_core::circuit_builder::CircuitBuilder;
    ///
    /// let builder = CircuitBuilder::<3>::new();
    /// let circuit_ref = builder.circuit();
    /// assert_eq!(circuit_ref.num_qubits(), 3);
    /// ```
    #[inline]
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }
}

impl<const N: usize> Default for CircuitBuilder<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> std::fmt::Debug for CircuitBuilder<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitBuilder")
            .field("num_qubits", &N)
            .field("num_operations", &self.num_operations())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::Gate;

    // Mock gate for testing
    #[derive(Debug)]
    struct MockGate {
        name: String,
        num_qubits: usize,
    }

    impl Gate for MockGate {
        fn name(&self) -> &str {
            &self.name
        }

        fn num_qubits(&self) -> usize {
            self.num_qubits
        }
    }

    #[test]
    fn test_builder_creation() {
        let builder = CircuitBuilder::<5>::new();
        assert_eq!(builder.num_qubits(), 5);
        assert_eq!(builder.num_operations(), 0);
        assert!(builder.is_empty());
    }

    #[test]
    fn test_get_qubit() {
        let builder = CircuitBuilder::<3>::new();

        let q0 = builder.qubit(0).unwrap();
        assert_eq!(q0.index(), 0);

        let q2 = builder.qubit(2).unwrap();
        assert_eq!(q2.index(), 2);

        // Out of bounds
        let invalid = builder.qubit(5);
        assert!(invalid.is_err());
    }

    #[test]
    fn test_get_all_qubits() {
        let builder = CircuitBuilder::<3>::new();
        let [q0, q1, q2] = builder.qubits();

        assert_eq!(q0.index(), 0);
        assert_eq!(q1.index(), 1);
        assert_eq!(q2.index(), 2);
    }

    #[test]
    fn test_get_all_qubits_large() {
        let builder = CircuitBuilder::<10>::new();
        let qubits = builder.qubits();

        for (i, q) in qubits.iter().enumerate() {
            assert_eq!(q.index(), i);
        }
    }

    #[test]
    fn test_apply_gate() {
        let mut builder = CircuitBuilder::<2>::new();
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });

        let q0 = builder.qubit(0).unwrap();
        builder.apply_gate(gate, &[q0]).unwrap();

        assert_eq!(builder.num_operations(), 1);
        assert!(!builder.is_empty());
    }

    #[test]
    fn test_apply_two_qubit_gate() {
        let mut builder = CircuitBuilder::<3>::new();
        let gate = Arc::new(MockGate {
            name: "CNOT".to_string(),
            num_qubits: 2,
        });

        let [q0, q1, _] = builder.qubits();
        builder.apply_gate(gate, &[q0, q1]).unwrap();

        assert_eq!(builder.num_operations(), 1);
    }

    #[test]
    fn test_method_chaining() {
        let mut builder = CircuitBuilder::<2>::new();
        let h_gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });
        let x_gate = Arc::new(MockGate {
            name: "X".to_string(),
            num_qubits: 1,
        });

        let [q0, q1] = builder.qubits();

        builder
            .apply_gate(h_gate.clone(), &[q0])
            .unwrap()
            .apply_gate(x_gate, &[q1])
            .unwrap()
            .apply_gate(h_gate, &[q0])
            .unwrap();

        assert_eq!(builder.num_operations(), 3);
    }

    #[test]
    fn test_build() {
        let mut builder = CircuitBuilder::<2>::new();
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });

        let q0 = builder.qubit(0).unwrap();
        builder.apply_gate(gate, &[q0]).unwrap();

        let circuit = builder.build();
        assert_eq!(circuit.num_qubits(), 2);
        assert_eq!(circuit.len(), 1);
    }

    #[test]
    fn test_with_capacity() {
        let builder = CircuitBuilder::<5>::with_capacity(100);
        assert_eq!(builder.num_qubits(), 5);
        assert!(builder.is_empty());
    }

    #[test]
    fn test_validate() {
        let builder = CircuitBuilder::<3>::new();
        assert!(builder.validate().is_ok());
    }

    #[test]
    fn test_debug_format() {
        let builder = CircuitBuilder::<5>::new();
        let debug = format!("{:?}", builder);
        assert!(debug.contains("CircuitBuilder"));
        assert!(debug.contains("5"));
    }

    #[test]
    fn test_default() {
        let builder = CircuitBuilder::<3>::default();
        assert_eq!(builder.num_qubits(), 3);
        assert!(builder.is_empty());
    }

    #[test]
    fn test_circuit_ref() {
        let builder = CircuitBuilder::<3>::new();
        let circuit_ref = builder.circuit();
        assert_eq!(circuit_ref.num_qubits(), 3);
    }

    #[test]
    fn test_apply_gate_wrong_qubit_count() {
        let mut builder = CircuitBuilder::<3>::new();
        let gate = Arc::new(MockGate {
            name: "CNOT".to_string(),
            num_qubits: 2,
        });

        let q0 = builder.qubit(0).unwrap();

        // Try to apply 2-qubit gate to 1 qubit
        let result = builder.apply_gate(gate, &[q0]);
        assert!(result.is_err());
    }
}
