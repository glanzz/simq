//! Error types for SimQ

use crate::QubitId;
use thiserror::Error;

/// Errors that can occur in quantum circuit operations
#[derive(Debug, Error)]
pub enum QuantumError {
    /// Invalid qubit index used
    #[error("Invalid qubit index {0}: circuit has only {1} qubits")]
    InvalidQubit(usize, usize),

    /// Gate applied to wrong number of qubits
    #[error("Gate '{gate}' requires {expected} qubits, but {actual} were provided")]
    InvalidQubitCount {
        gate: String,
        expected: usize,
        actual: usize,
    },

    /// Circuit has no qubits
    #[error("Circuit must have at least one qubit")]
    EmptyCircuit,

    /// Duplicate qubit in gate operation
    #[error("Duplicate qubit {0} in gate operation")]
    DuplicateQubit(QubitId),

    /// Generic circuit validation error
    #[error("Circuit validation failed: {0}")]
    ValidationError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Deserialization error
    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    /// Unknown gate type during deserialization
    #[error("Unknown gate type: {0}")]
    UnknownGateType(String),

    /// Version mismatch in serialized format
    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },

    /// Cache error
    #[error("Cache error: {0}")]
    CacheError(String),

    /// Cycle detected in circuit DAG
    #[error("Circuit contains cycle involving operations: {operations:?}")]
    CycleDetected { operations: Vec<usize> },

    /// Invalid dependency in circuit
    #[error("Invalid dependency from operation {from} to {to} via qubit {qubit}")]
    InvalidDependency {
        from: usize,
        to: usize,
        qubit: usize,
    },

    /// Cannot compute topological order
    #[error("Cannot compute topological order: {reason}")]
    TopologicalOrderError { reason: String },
}

impl QuantumError {
    /// Create an invalid qubit error
    pub fn invalid_qubit(qubit: usize, num_qubits: usize) -> Self {
        Self::InvalidQubit(qubit, num_qubits)
    }

    /// Create an invalid qubit count error
    pub fn invalid_qubit_count(gate: impl Into<String>, expected: usize, actual: usize) -> Self {
        Self::InvalidQubitCount {
            gate: gate.into(),
            expected,
            actual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_qubit_error() {
        let err = QuantumError::invalid_qubit(5, 3);
        let msg = format!("{}", err);
        assert!(msg.contains("5"));
        assert!(msg.contains("3"));
    }

    #[test]
    fn test_invalid_qubit_count_error() {
        let err = QuantumError::invalid_qubit_count("CNOT", 2, 1);
        let msg = format!("{}", err);
        assert!(msg.contains("CNOT"));
        assert!(msg.contains("2"));
        assert!(msg.contains("1"));
    }

    #[test]
    fn test_empty_circuit_error() {
        let err = QuantumError::EmptyCircuit;
        let msg = format!("{}", err);
        assert!(msg.contains("at least one qubit"));
    }
}
