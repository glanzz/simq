//! Quantum circuit representation

use crate::gate::Gate;
use crate::{GateOp, QuantumError, QubitId, Result};
use std::sync::Arc;

/// A quantum circuit
///
/// Contains a sequence of quantum gate operations applied to qubits.
///
/// # Example
/// ```
/// use simq_core::Circuit;
///
/// let circuit = Circuit::new(3);
/// assert_eq!(circuit.num_qubits(), 3);
/// assert_eq!(circuit.len(), 0);
/// ```
#[derive(Clone, Debug)]
pub struct Circuit {
    num_qubits: usize,
    operations: Vec<GateOp>,
}

impl Circuit {
    /// Create a new quantum circuit with the specified number of qubits
    ///
    /// # Panics
    /// Panics if `num_qubits` is 0
    ///
    /// # Example
    /// ```
    /// use simq_core::Circuit;
    ///
    /// let circuit = Circuit::new(3);
    /// assert_eq!(circuit.num_qubits(), 3);
    /// ```
    pub fn new(num_qubits: usize) -> Self {
        assert!(num_qubits > 0, "Circuit must have at least one qubit");
        Self {
            num_qubits,
            operations: Vec::new(),
        }
    }

    /// Create a circuit with pre-allocated capacity
    pub fn with_capacity(num_qubits: usize, capacity: usize) -> Self {
        assert!(num_qubits > 0, "Circuit must have at least one qubit");
        Self {
            num_qubits,
            operations: Vec::with_capacity(capacity),
        }
    }

    /// Get the number of qubits in the circuit
    #[inline]
    pub const fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the number of operations in the circuit
    #[inline]
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if the circuit is empty (no operations)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Add a gate operation to the circuit
    ///
    /// # Errors
    /// Returns error if any qubit index is out of bounds
    ///
    /// # Example
    /// ```ignore
    /// use simq_core::{Circuit, QubitId};
    /// use std::sync::Arc;
    ///
    /// let mut circuit = Circuit::new(2);
    /// let q0 = QubitId::new(0);
    /// circuit.add_gate(some_gate, &[q0])?;
    /// ```
    pub fn add_gate(&mut self, gate: Arc<dyn Gate>, qubits: &[QubitId]) -> Result<()> {
        // Validate qubit indices
        for &qubit in qubits {
            if qubit.index() >= self.num_qubits {
                return Err(QuantumError::invalid_qubit(qubit.index(), self.num_qubits));
            }
        }

        // Create and add gate operation
        let gate_op = GateOp::new(gate, qubits)?;
        self.operations.push(gate_op);
        Ok(())
    }

    /// Get an iterator over the operations
    pub fn operations(&self) -> impl Iterator<Item = &GateOp> {
        self.operations.iter()
    }

    /// Get a specific operation by index
    pub fn get_operation(&self, index: usize) -> Option<&GateOp> {
        self.operations.get(index)
    }

    /// Clear all operations from the circuit
    pub fn clear(&mut self) {
        self.operations.clear();
    }

    /// Get the depth of the circuit (longest path through gates)
    ///
    /// For now, returns the number of operations (sequential execution).
    /// Will be improved with parallelism analysis later.
    pub fn depth(&self) -> usize {
        self.operations.len()
    }

    /// Validate the circuit
    ///
    /// Checks that all operations are valid for this circuit.
    pub fn validate(&self) -> Result<()> {
        for (i, op) in self.operations.iter().enumerate() {
            for &qubit in op.qubits() {
                if qubit.index() >= self.num_qubits {
                    return Err(QuantumError::ValidationError(format!(
                        "Operation {} uses invalid qubit {}",
                        i, qubit
                    )));
                }
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for Circuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Circuit({} qubits, {} operations)", self.num_qubits, self.len())?;
        for (i, op) in self.operations.iter().enumerate() {
            writeln!(f, "  {}: {}", i, op)?;
        }
        Ok(())
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
    fn test_circuit_creation() {
        let circuit = Circuit::new(3);
        assert_eq!(circuit.num_qubits(), 3);
        assert_eq!(circuit.len(), 0);
        assert!(circuit.is_empty());
    }

    #[test]
    #[should_panic(expected = "at least one qubit")]
    fn test_circuit_zero_qubits() {
        Circuit::new(0);
    }

    #[test]
    fn test_add_gate() {
        let mut circuit = Circuit::new(2);
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });
        let q0 = QubitId::new(0);

        circuit.add_gate(gate, &[q0]).unwrap();
        assert_eq!(circuit.len(), 1);
        assert!(!circuit.is_empty());
    }

    #[test]
    fn test_add_gate_invalid_qubit() {
        let mut circuit = Circuit::new(2);
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });
        let q5 = QubitId::new(5);

        let result = circuit.add_gate(gate, &[q5]);
        assert!(result.is_err());

        if let Err(QuantumError::InvalidQubit(idx, num)) = result {
            assert_eq!(idx, 5);
            assert_eq!(num, 2);
        } else {
            panic!("Expected InvalidQubit error");
        }
    }

    #[test]
    fn test_operations_iter() {
        let mut circuit = Circuit::new(2);
        let gate1 = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });
        let gate2 = Arc::new(MockGate {
            name: "X".to_string(),
            num_qubits: 1,
        });

        circuit.add_gate(gate1, &[QubitId::new(0)]).unwrap();
        circuit.add_gate(gate2, &[QubitId::new(1)]).unwrap();

        let ops: Vec<_> = circuit.operations().collect();
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn test_clear() {
        let mut circuit = Circuit::new(2);
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });

        circuit.add_gate(gate.clone(), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(gate, &[QubitId::new(1)]).unwrap();
        assert_eq!(circuit.len(), 2);

        circuit.clear();
        assert_eq!(circuit.len(), 0);
        assert!(circuit.is_empty());
    }

    #[test]
    fn test_validate() {
        let circuit = Circuit::new(3);
        assert!(circuit.validate().is_ok());
    }

    #[test]
    fn test_display() {
        let mut circuit = Circuit::new(2);
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });
        circuit.add_gate(gate, &[QubitId::new(0)]).unwrap();

        let display = format!("{}", circuit);
        assert!(display.contains("2 qubits"));
        assert!(display.contains("1 operations"));
    }

    #[test]
    fn test_with_capacity() {
        let circuit = Circuit::with_capacity(3, 100);
        assert_eq!(circuit.num_qubits(), 3);
        assert!(circuit.is_empty());
    }

    #[test]
    fn test_get_operation() {
        let mut circuit = Circuit::new(2);
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });
        circuit.add_gate(gate, &[QubitId::new(0)]).unwrap();

        let op = circuit.get_operation(0);
        assert!(op.is_some());
        assert_eq!(op.unwrap().gate().name(), "H");

        let no_op = circuit.get_operation(10);
        assert!(no_op.is_none());
    }

    #[test]
    fn test_depth() {
        let mut circuit = Circuit::new(2);
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });

        assert_eq!(circuit.depth(), 0);

        circuit.add_gate(gate.clone(), &[QubitId::new(0)]).unwrap();
        assert_eq!(circuit.depth(), 1);

        circuit.add_gate(gate, &[QubitId::new(1)]).unwrap();
        assert_eq!(circuit.depth(), 2);
    }
}
