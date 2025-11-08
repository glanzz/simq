//! Dynamic circuit builder for runtime-determined sizes

use crate::{Circuit, Gate, QubitId, Result};
use smallvec::SmallVec;
use std::sync::Arc;

/// A circuit builder for circuits with runtime-determined qubit counts
///
/// Unlike `CircuitBuilder<N>`, this builder accepts the qubit count at runtime
/// and uses `usize` indices instead of typed `Qubit<N>` references.
/// This provides more flexibility at the cost of compile-time safety.
///
/// # When to Use
///
/// Use `DynamicCircuitBuilder` when:
/// - Circuit size is determined at runtime (e.g., from user input or config)
/// - Building circuits programmatically with variable sizes
/// - You need maximum flexibility over compile-time safety
///
/// Use `CircuitBuilder<N>` when:
/// - Circuit size is known at compile time
/// - You want type-safe qubit references
/// - You prefer compile-time error checking
///
/// # Performance
///
/// While slightly slower than `CircuitBuilder<N>` due to runtime validation,
/// the overhead is minimal and only occurs during circuit construction,
/// not during simulation.
///
/// # Example
/// ```
/// use simq_core::dynamic_builder::DynamicCircuitBuilder;
/// use simq_core::Gate;
/// use std::sync::Arc;
///
/// # #[derive(Debug)]
/// # struct MockGate;
/// # impl Gate for MockGate {
/// #     fn name(&self) -> &str { "H" }
/// #     fn num_qubits(&self) -> usize { 1 }
/// # }
/// // Circuit size from runtime value
/// let num_qubits = 5;
/// let mut builder = DynamicCircuitBuilder::new(num_qubits);
///
/// let gate = Arc::new(MockGate);
/// builder.apply_gate(gate, &[0]).unwrap();
///
/// let circuit = builder.build();
/// assert_eq!(circuit.num_qubits(), 5);
/// ```
pub struct DynamicCircuitBuilder {
    circuit: Circuit,
}

impl DynamicCircuitBuilder {
    /// Create a new dynamic circuit builder
    ///
    /// # Panics
    /// Panics if num_qubits is 0
    ///
    /// # Example
    /// ```
    /// use simq_core::dynamic_builder::DynamicCircuitBuilder;
    ///
    /// let builder = DynamicCircuitBuilder::new(5);
    /// assert_eq!(builder.num_qubits(), 5);
    /// assert_eq!(builder.num_operations(), 0);
    /// ```
    #[inline]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            circuit: Circuit::new(num_qubits),
        }
    }

    /// Create a builder with pre-allocated capacity
    ///
    /// Use this when you know approximately how many operations you'll add
    /// to avoid reallocations.
    ///
    /// # Example
    /// ```
    /// use simq_core::dynamic_builder::DynamicCircuitBuilder;
    ///
    /// let builder = DynamicCircuitBuilder::with_capacity(5, 100);
    /// assert_eq!(builder.num_qubits(), 5);
    /// ```
    #[inline]
    pub fn with_capacity(num_qubits: usize, capacity: usize) -> Self {
        Self {
            circuit: Circuit::with_capacity(num_qubits, capacity),
        }
    }

    /// Apply a gate to the specified qubits
    ///
    /// # Errors
    /// Returns error if:
    /// - Any qubit index is out of bounds
    /// - Qubit count doesn't match gate requirements
    /// - Duplicate qubits specified
    ///
    /// # Example
    /// ```
    /// use simq_core::dynamic_builder::DynamicCircuitBuilder;
    /// use simq_core::Gate;
    /// use std::sync::Arc;
    ///
    /// # #[derive(Debug)]
    /// # struct MockGate;
    /// # impl Gate for MockGate {
    /// #     fn name(&self) -> &str { "H" }
    /// #     fn num_qubits(&self) -> usize { 1 }
    /// # }
    /// let mut builder = DynamicCircuitBuilder::new(3);
    /// let gate = Arc::new(MockGate);
    ///
    /// builder.apply_gate(gate, &[0]).unwrap();
    /// assert_eq!(builder.num_operations(), 1);
    /// ```
    #[inline]
    pub fn apply_gate(&mut self, gate: Arc<dyn Gate>, qubits: &[usize]) -> Result<&mut Self> {
        // Convert usize indices to QubitId using SmallVec for stack allocation
        let qubit_ids: SmallVec<[QubitId; 2]> =
            qubits.iter().map(|&idx| QubitId::new(idx)).collect();

        self.circuit.add_gate(gate, &qubit_ids)?;
        Ok(self)
    }

    /// Get the number of qubits
    ///
    /// # Example
    /// ```
    /// use simq_core::dynamic_builder::DynamicCircuitBuilder;
    ///
    /// let builder = DynamicCircuitBuilder::new(10);
    /// assert_eq!(builder.num_qubits(), 10);
    /// ```
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.circuit.num_qubits()
    }

    /// Get the current number of operations
    ///
    /// # Example
    /// ```
    /// use simq_core::dynamic_builder::DynamicCircuitBuilder;
    ///
    /// let builder = DynamicCircuitBuilder::new(3);
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
    /// use simq_core::dynamic_builder::DynamicCircuitBuilder;
    ///
    /// let builder = DynamicCircuitBuilder::new(3);
    /// assert!(builder.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.circuit.is_empty()
    }

    /// Validate the circuit
    ///
    /// Checks that all operations are valid.
    ///
    /// # Example
    /// ```
    /// use simq_core::dynamic_builder::DynamicCircuitBuilder;
    ///
    /// let builder = DynamicCircuitBuilder::new(3);
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
    /// use simq_core::dynamic_builder::DynamicCircuitBuilder;
    ///
    /// let builder = DynamicCircuitBuilder::new(3);
    /// let circuit = builder.build();
    /// assert_eq!(circuit.num_qubits(), 3);
    /// ```
    #[inline]
    pub fn build(self) -> Circuit {
        self.circuit
    }

    /// Get a reference to the underlying circuit
    ///
    /// Use this to inspect the circuit without consuming the builder.
    ///
    /// # Example
    /// ```
    /// use simq_core::dynamic_builder::DynamicCircuitBuilder;
    ///
    /// let builder = DynamicCircuitBuilder::new(3);
    /// let circuit_ref = builder.circuit();
    /// assert_eq!(circuit_ref.num_qubits(), 3);
    /// ```
    #[inline]
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }
}

impl std::fmt::Debug for DynamicCircuitBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicCircuitBuilder")
            .field("num_qubits", &self.num_qubits())
            .field("num_operations", &self.num_operations())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::Gate;

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
    fn test_dynamic_builder_creation() {
        let builder = DynamicCircuitBuilder::new(5);
        assert_eq!(builder.num_qubits(), 5);
        assert_eq!(builder.num_operations(), 0);
        assert!(builder.is_empty());
    }

    #[test]
    fn test_dynamic_builder_various_sizes() {
        for size in [1, 2, 5, 10, 100, 1000] {
            let builder = DynamicCircuitBuilder::new(size);
            assert_eq!(builder.num_qubits(), size);
        }
    }

    #[test]
    fn test_apply_gate() {
        let mut builder = DynamicCircuitBuilder::new(3);
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });

        builder.apply_gate(gate, &[0]).unwrap();
        assert_eq!(builder.num_operations(), 1);
        assert!(!builder.is_empty());
    }

    #[test]
    fn test_apply_gate_out_of_bounds() {
        let mut builder = DynamicCircuitBuilder::new(2);
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });

        let result = builder.apply_gate(gate, &[5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_two_qubit_gate() {
        let mut builder = DynamicCircuitBuilder::new(3);
        let gate = Arc::new(MockGate {
            name: "CNOT".to_string(),
            num_qubits: 2,
        });

        builder.apply_gate(gate, &[0, 1]).unwrap();
        assert_eq!(builder.num_operations(), 1);
    }

    #[test]
    fn test_method_chaining() {
        let mut builder = DynamicCircuitBuilder::new(3);
        let h_gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });
        let cnot_gate = Arc::new(MockGate {
            name: "CNOT".to_string(),
            num_qubits: 2,
        });

        builder
            .apply_gate(h_gate, &[0])
            .unwrap()
            .apply_gate(cnot_gate, &[0, 1])
            .unwrap();

        assert_eq!(builder.num_operations(), 2);
    }

    #[test]
    fn test_build() {
        let mut builder = DynamicCircuitBuilder::new(3);
        let gate = Arc::new(MockGate {
            name: "X".to_string(),
            num_qubits: 1,
        });

        builder.apply_gate(gate, &[1]).unwrap();
        let circuit = builder.build();

        assert_eq!(circuit.num_qubits(), 3);
        assert_eq!(circuit.len(), 1);
    }

    #[test]
    fn test_with_capacity() {
        let builder = DynamicCircuitBuilder::with_capacity(5, 100);
        assert_eq!(builder.num_qubits(), 5);
        assert!(builder.is_empty());
    }

    #[test]
    fn test_validate() {
        let builder = DynamicCircuitBuilder::new(3);
        assert!(builder.validate().is_ok());
    }

    #[test]
    fn test_debug_format() {
        let builder = DynamicCircuitBuilder::new(7);
        let debug = format!("{:?}", builder);
        assert!(debug.contains("DynamicCircuitBuilder"));
        assert!(debug.contains("7"));
    }

    #[test]
    fn test_circuit_ref() {
        let builder = DynamicCircuitBuilder::new(3);
        let circuit_ref = builder.circuit();
        assert_eq!(circuit_ref.num_qubits(), 3);
    }

    #[test]
    fn test_apply_many_gates() {
        let mut builder = DynamicCircuitBuilder::new(10);
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });

        // Apply gate to all qubits
        for i in 0..10 {
            builder.apply_gate(gate.clone(), &[i]).unwrap();
        }

        assert_eq!(builder.num_operations(), 10);
    }

    #[test]
    fn test_apply_gate_wrong_qubit_count() {
        let mut builder = DynamicCircuitBuilder::new(3);
        let gate = Arc::new(MockGate {
            name: "CNOT".to_string(),
            num_qubits: 2,
        });

        // Try to apply 2-qubit gate to 1 qubit
        let result = builder.apply_gate(gate, &[0]);
        assert!(result.is_err());
    }
}
