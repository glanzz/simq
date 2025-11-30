//! Circuit serialization types

use crate::serialization::gate::SerializedGateOp;
use serde::{Deserialize, Serialize};

/// Serialized circuit representation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SerializedCircuit {
    /// Format version for compatibility checking
    pub version: u32,
    /// Number of qubits in the circuit
    pub num_qubits: usize,
    /// Gate operations in the circuit
    pub operations: Vec<SerializedGateOp>,
    /// Optional metadata
    pub metadata: Option<CircuitMetadata>,
}

/// Circuit metadata
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CircuitMetadata {
    /// Optional circuit name
    pub name: Option<String>,
    /// Optional circuit description
    pub description: Option<String>,
    /// Creation timestamp (Unix timestamp)
    pub created_at: Option<u64>,
    /// Optional tags
    pub tags: Vec<String>,
}

impl SerializedCircuit {
    /// Create a new serialized circuit
    pub fn new(num_qubits: usize, operations: Vec<SerializedGateOp>) -> Self {
        Self {
            version: crate::serialization::CIRCUIT_FORMAT_VERSION,
            num_qubits,
            operations,
            metadata: None,
        }
    }

    /// Create with metadata
    pub fn with_metadata(
        num_qubits: usize,
        operations: Vec<SerializedGateOp>,
        metadata: CircuitMetadata,
    ) -> Self {
        Self {
            version: crate::serialization::CIRCUIT_FORMAT_VERSION,
            num_qubits,
            operations,
            metadata: Some(metadata),
        }
    }

    /// Check version compatibility
    pub fn check_version(&self) -> crate::Result<()> {
        use crate::QuantumError;
        if self.version > crate::serialization::CIRCUIT_FORMAT_VERSION {
            return Err(QuantumError::VersionMismatch {
                expected: crate::serialization::CIRCUIT_FORMAT_VERSION,
                actual: self.version,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialization::gate::SerializedGate;

    #[test]
    fn test_serialized_circuit() {
        let circuit = SerializedCircuit::new(
            2,
            vec![
                SerializedGateOp {
                    gate: SerializedGate::StandardGate {
                        name: "H".to_string(),
                    },
                    qubits: vec![0],
                },
                SerializedGateOp {
                    gate: SerializedGate::StandardGate {
                        name: "CNOT".to_string(),
                    },
                    qubits: vec![0, 1],
                },
            ],
        );

        let json = serde_json::to_string(&circuit).unwrap();
        let deserialized: SerializedCircuit = serde_json::from_str(&json).unwrap();
        assert_eq!(circuit, deserialized);
    }

    #[test]
    fn test_circuit_with_metadata() {
        let metadata = CircuitMetadata {
            name: Some("Bell State".to_string()),
            description: Some("Creates a Bell state".to_string()),
            created_at: Some(1234567890),
            tags: vec!["entanglement".to_string(), "bell".to_string()],
        };

        let circuit = SerializedCircuit::with_metadata(2, vec![], metadata.clone());
        assert_eq!(circuit.metadata, Some(metadata));
    }
}
