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

    /// Get a mutable reference to the operations vector
    ///
    /// This is useful for optimization passes that need to modify the circuit.
    /// Use with caution as direct modifications bypass validation.
    pub fn operations_mut(&mut self) -> &mut Vec<GateOp> {
        &mut self.operations
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
    /// This tries to compute the actual depth using DAG analysis.
    /// If DAG analysis fails, falls back to sequential depth (number of operations).
    ///
    /// For accurate depth computation, use `compute_depth()` which returns an error
    /// if the circuit cannot be analyzed.
    pub fn depth(&self) -> usize {
        // Try DAG-based depth computation
        if let Ok(dag) = crate::validation::dag::DependencyGraph::from_circuit(self) {
            if let Ok(depth) = dag.depth() {
                return depth;
            }
        }

        // Fallback to sequential depth
        self.operations.len()
    }

    /// Validate the circuit
    ///
    /// Checks that all operations are valid for this circuit.
    /// This performs basic validation (qubit bounds, gate counts).
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

    /// Validate circuit with DAG consistency checking
    ///
    /// This performs comprehensive validation including:
    /// - Basic validation (qubit bounds, gate counts)
    /// - DAG consistency (no cycles, valid dependencies)
    /// - Dependency graph validation
    ///
    /// # Errors
    /// Returns error if validation fails
    ///
    /// # Example
    /// ```ignore
    /// use simq_core::Circuit;
    ///
    /// let circuit = Circuit::new(2);
    /// circuit.validate_dag()?;
    /// ```
    pub fn validate_dag(&self) -> crate::Result<crate::ValidationReport> {
        use crate::validation::dag::DependencyGraph;
        use crate::validation::rules::{ValidationRule, CycleDetectionRule, DependencyValidationRule, QubitUsageRule};
        use crate::validation::report::ValidationReport;

        // Basic validation first
        self.validate()?;

        // Build dependency graph
        let dag = DependencyGraph::from_circuit(self)?;

        // Run validation rules
        let mut report = ValidationReport::new();
        let rules: Vec<Box<dyn ValidationRule>> = vec![
            Box::new(QubitUsageRule),
            Box::new(CycleDetectionRule),
            Box::new(DependencyValidationRule),
        ];

        for rule in rules {
            let result = rule.validate(self, &dag);
            report.add_result(rule.name(), result);
        }

        if report.has_errors() {
            return Err(QuantumError::ValidationError(report.format(self)));
        }

        Ok(report)
    }

    /// Get circuit depth using DAG analysis
    ///
    /// This computes the actual circuit depth considering parallelization,
    /// not just the number of operations.
    ///
    /// # Errors
    /// Returns error if circuit contains cycles or cannot be analyzed
    ///
    /// # Example
    /// ```ignore
    /// use simq_core::Circuit;
    ///
    /// let circuit = Circuit::new(2);
    /// let depth = circuit.compute_depth()?;
    /// ```
    pub fn compute_depth(&self) -> Result<usize> {
        use crate::validation::dag::DependencyGraph;
        let dag = DependencyGraph::from_circuit(self)?;
        dag.depth()
    }

    /// Check if circuit is acyclic
    ///
    /// # Errors
    /// Returns error if DAG cannot be constructed
    pub fn is_acyclic(&self) -> Result<bool> {
        use crate::validation::dag::DependencyGraph;
        let dag = DependencyGraph::from_circuit(self)?;
        Ok(dag.is_acyclic())
    }

    /// Analyze circuit parallelism
    ///
    /// Returns analysis of parallel execution opportunities in the circuit.
    ///
    /// # Errors
    /// Returns error if circuit contains cycles or cannot be analyzed
    ///
    /// # Example
    /// ```ignore
    /// use simq_core::Circuit;
    ///
    /// let circuit = Circuit::new(3);
    /// let analysis = circuit.analyze_parallelism()?;
    /// println!("Parallelism factor: {}", analysis.parallelism_factor);
    /// ```
    pub fn analyze_parallelism(&self) -> Result<crate::ParallelismAnalysis> {
        use crate::validation::dag::DependencyGraph;
        let dag = DependencyGraph::from_circuit(self)?;
        dag.analyze_parallelism()
    }

    /// Get dependency graph visualization in DOT format
    ///
    /// This generates a Graphviz DOT format representation of the circuit's
    /// dependency graph, which can be used for visualization.
    ///
    /// # Errors
    /// Returns error if DAG cannot be constructed
    ///
    /// # Example
    /// ```ignore
    /// use simq_core::Circuit;
    ///
    /// let circuit = Circuit::new(2);
    /// let dot = circuit.to_dot()?;
    /// // Save to file or render with Graphviz
    /// ```
    pub fn to_dot(&self) -> Result<String> {
        use crate::validation::dag::DependencyGraph;
        let dag = DependencyGraph::from_circuit(self)?;
        Ok(dag.to_dot())
    }

    #[cfg(feature = "serialization")]
    /// Serialize circuit to binary format
    ///
    /// # Errors
    /// Returns error if serialization fails
    ///
    /// # Example
    /// ```ignore
    /// use simq_core::Circuit;
    ///
    /// let circuit = Circuit::new(2);
    /// let bytes = circuit.to_bytes()?;
    /// ```
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        use crate::serialization::circuit::SerializedCircuit;
        use crate::serialization::gate::GateRegistry;
        use crate::serialization::{StandardGateRegistry, CIRCUIT_FORMAT_VERSION};

        let registry = StandardGateRegistry::default();
        let operations: Vec<_> = self
            .operations
            .iter()
            .map(|op| registry.serialize_gate_op(op))
            .collect::<Result<Vec<_>>>()
            .map_err(|e| QuantumError::SerializationError(format!("Failed to serialize gate: {}", e)))?;

        let serialized = SerializedCircuit {
            version: CIRCUIT_FORMAT_VERSION,
            num_qubits: self.num_qubits,
            operations,
            metadata: None,
        };

        bincode::serialize(&serialized)
            .map_err(|e| QuantumError::SerializationError(format!("Binary serialization failed: {}", e)))
    }

    #[cfg(feature = "serialization")]
    /// Deserialize circuit from binary format
    ///
    /// # Errors
    /// Returns error if deserialization fails or circuit is invalid
    ///
    /// # Example
    /// ```ignore
    /// use simq_core::Circuit;
    ///
    /// let bytes = vec![...];
    /// let circuit = Circuit::from_bytes(&bytes)?;
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        use crate::serialization::circuit::SerializedCircuit;
        use crate::serialization::gate::GateRegistry;
        use crate::serialization::StandardGateRegistry;

        let serialized: SerializedCircuit = bincode::deserialize(bytes)
            .map_err(|e| QuantumError::DeserializationError(format!("Binary deserialization failed: {}", e)))?;

        serialized.check_version()?;

        let registry = StandardGateRegistry::default();
        let mut circuit = Circuit::new(serialized.num_qubits);

        for op in serialized.operations {
            let gate_op = registry.create_gate_op(&op, serialized.num_qubits)
                .map_err(|e| QuantumError::DeserializationError(format!("Failed to create gate: {}", e)))?;
            circuit.operations.push(gate_op);
        }

        circuit.validate()?;
        Ok(circuit)
    }

    #[cfg(feature = "serialization")]
    /// Serialize circuit to JSON format
    ///
    /// # Errors
    /// Returns error if serialization fails
    ///
    /// # Example
    /// ```ignore
    /// use simq_core::Circuit;
    ///
    /// let circuit = Circuit::new(2);
    /// let json = circuit.to_json()?;
    /// ```
    pub fn to_json(&self) -> Result<String> {
        use crate::serialization::circuit::SerializedCircuit;
        use crate::serialization::gate::GateRegistry;
        use crate::serialization::{StandardGateRegistry, CIRCUIT_FORMAT_VERSION};

        let registry = StandardGateRegistry::default();
        let operations: Vec<_> = self
            .operations
            .iter()
            .map(|op| registry.serialize_gate_op(op))
            .collect::<Result<Vec<_>>>()
            .map_err(|e| QuantumError::SerializationError(format!("Failed to serialize gate: {}", e)))?;

        let serialized = SerializedCircuit {
            version: CIRCUIT_FORMAT_VERSION,
            num_qubits: self.num_qubits,
            operations,
            metadata: None,
        };

        serde_json::to_string(&serialized)
            .map_err(|e| QuantumError::SerializationError(format!("JSON serialization failed: {}", e)))
    }

    #[cfg(feature = "serialization")]
    /// Serialize circuit to pretty-printed JSON format
    ///
    /// # Errors
    /// Returns error if serialization fails
    pub fn to_json_pretty(&self) -> Result<String> {
        use crate::serialization::circuit::SerializedCircuit;
        use crate::serialization::gate::GateRegistry;
        use crate::serialization::{StandardGateRegistry, CIRCUIT_FORMAT_VERSION};

        let registry = StandardGateRegistry::default();
        let operations: Vec<_> = self
            .operations
            .iter()
            .map(|op| registry.serialize_gate_op(op))
            .collect::<Result<Vec<_>>>()
            .map_err(|e| QuantumError::SerializationError(format!("Failed to serialize gate: {}", e)))?;

        let serialized = SerializedCircuit {
            version: CIRCUIT_FORMAT_VERSION,
            num_qubits: self.num_qubits,
            operations,
            metadata: None,
        };

        serde_json::to_string_pretty(&serialized)
            .map_err(|e| QuantumError::SerializationError(format!("JSON serialization failed: {}", e)))
    }

    #[cfg(feature = "serialization")]
    /// Deserialize circuit from JSON format
    ///
    /// # Errors
    /// Returns error if deserialization fails or circuit is invalid
    ///
    /// # Example
    /// ```ignore
    /// use simq_core::Circuit;
    ///
    /// let json = r#"{"version": 1, "num_qubits": 2, "operations": []}"#;
    /// let circuit = Circuit::from_json(json)?;
    /// ```
    pub fn from_json(json: &str) -> Result<Self> {
        use crate::serialization::circuit::SerializedCircuit;
        use crate::serialization::gate::GateRegistry;
        use crate::serialization::StandardGateRegistry;

        let serialized: SerializedCircuit = serde_json::from_str(json)
            .map_err(|e| QuantumError::DeserializationError(format!("JSON deserialization failed: {}", e)))?;

        serialized.check_version()?;

        let registry = StandardGateRegistry::default();
        let mut circuit = Circuit::new(serialized.num_qubits);

        for op in serialized.operations {
            let gate_op = registry.create_gate_op(&op, serialized.num_qubits)
                .map_err(|e| QuantumError::DeserializationError(format!("Failed to create gate: {}", e)))?;
            circuit.operations.push(gate_op);
        }

        circuit.validate()?;
        Ok(circuit)
    }

    #[cfg(feature = "serialization")]
    /// Generate cache key from circuit structure
    ///
    /// The cache key is based on the circuit structure (gates and qubits),
    /// not parameter values, allowing parameterized circuits to share cache entries.
    ///
    /// # Example
    /// ```ignore
    /// use simq_core::Circuit;
    ///
    /// let circuit = Circuit::new(2);
    /// let key = circuit.cache_key();
    /// ```
    pub fn cache_key(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.num_qubits.hash(&mut hasher);

        // Hash gate operations (structure only, not parameters)
        for op in &self.operations {
            op.gate().name().hash(&mut hasher);
            op.gate().num_qubits().hash(&mut hasher);
            for qubit in op.qubits() {
                qubit.index().hash(&mut hasher);
            }
        }

        hasher.finish()
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

        // Empty circuit has depth 0
        assert_eq!(circuit.depth(), 0);

        // Single gate has depth 1
        circuit.add_gate(gate.clone(), &[QubitId::new(0)]).unwrap();
        assert_eq!(circuit.depth(), 1);

        // Two gates on different qubits can run in parallel, so depth is still 1
        circuit.add_gate(gate.clone(), &[QubitId::new(1)]).unwrap();
        assert_eq!(circuit.depth(), 1);

        // Gate on same qubit as first creates a dependency, depth becomes 2
        circuit.add_gate(gate, &[QubitId::new(0)]).unwrap();
        assert_eq!(circuit.depth(), 2);
    }
}
