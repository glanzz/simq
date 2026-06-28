//! Gate serialization types and registry

use crate::{Gate, GateOp, QuantumError, QubitId, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Serialized representation of a quantum gate
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SerializedGate {
    /// Standard parameterless gate (H, X, Y, Z, CNOT, etc.)
    StandardGate { name: String },
    /// Parameterized gate (RX, RY, RZ, U3, etc.)
    ParameterizedGate { name: String, parameters: Vec<f64> },
    /// Custom gate with matrix representation
    CustomGate {
        name: String,
        matrix: Vec<[f64; 2]>, // Real and imaginary parts of complex numbers
        num_qubits: usize,
    },
}

/// Serialized gate operation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SerializedGateOp {
    pub gate: SerializedGate,
    pub qubits: Vec<usize>,
}

/// Trait for gate registry that can create gates from serialized representations
pub trait GateRegistry: Send + Sync {
    /// Create a gate from serialized representation
    fn create_gate(&self, serialized: &SerializedGate) -> Result<Arc<dyn Gate>>;

    /// Serialize a gate to representation
    fn serialize_gate(&self, gate: &dyn Gate) -> Result<SerializedGate>;

    /// Serialize a gate operation
    fn serialize_gate_op(&self, gate_op: &GateOp) -> Result<SerializedGateOp> {
        let gate = self.serialize_gate(gate_op.gate().as_ref())?;
        let qubits: Vec<usize> = gate_op.qubits().iter().map(|q| q.index()).collect();
        Ok(SerializedGateOp { gate, qubits })
    }

    /// Create a gate operation from serialized representation
    fn create_gate_op(&self, serialized: &SerializedGateOp, num_qubits: usize) -> Result<GateOp> {
        // Validate qubit indices
        for &qubit in &serialized.qubits {
            if qubit >= num_qubits {
                return Err(QuantumError::invalid_qubit(qubit, num_qubits));
            }
        }

        let gate = self.create_gate(&serialized.gate)?;
        let qubit_ids: Vec<QubitId> = serialized.qubits.iter().map(|&i| QubitId::new(i)).collect();

        GateOp::new(gate, &qubit_ids)
    }
}

/// Standard gate registry for built-in gates
#[derive(Default)]
pub struct StandardGateRegistry {
    // For now, we'll use a simple name-based approach
    // In the future, this could be extended with custom gate factories
}

impl StandardGateRegistry {
    /// Create a new standard gate registry
    pub fn new() -> Self {
        Self::default()
    }
}

impl GateRegistry for StandardGateRegistry {
    fn create_gate(&self, serialized: &SerializedGate) -> Result<Arc<dyn Gate>> {
        match serialized {
            SerializedGate::StandardGate { name } => self.create_standard_gate(name, &[]),
            SerializedGate::ParameterizedGate { name, parameters } => {
                self.create_standard_gate(name, parameters)
            },
            SerializedGate::CustomGate { .. } => {
                // Custom gates not yet implemented
                Err(QuantumError::UnknownGateType("Custom gates not yet supported".to_string()))
            },
        }
    }

    fn serialize_gate(&self, gate: &dyn Gate) -> Result<SerializedGate> {
        let name = gate.name().to_string();
        let _num_qubits = gate.num_qubits();

        // For now, we'll use a simple approach: try to extract parameters from gate name
        // This is a limitation - in a real implementation, gates would need to
        // implement a trait to expose their parameters

        // Check if this is a parameterized gate by name pattern
        // This is a simplified implementation - real gates would expose parameters
        if name.starts_with("R") && name.len() > 1 {
            // Rotation gates: RX, RY, RZ (we'd need parameter info)
            // For now, return as standard gate - this will be improved when
            // we have actual gate implementations with parameter access
            Ok(SerializedGate::StandardGate { name })
        } else {
            Ok(SerializedGate::StandardGate { name })
        }
    }
}

impl StandardGateRegistry {
    fn create_standard_gate(&self, name: &str, parameters: &[f64]) -> Result<Arc<dyn Gate>> {
        let name_upper = name.to_uppercase();
        let (num_qubits, params) = match name_upper.as_str() {
            "H" | "HADAMARD" => (1, vec![]),
            "X" | "PAULIX" | "NOT" => (1, vec![]),
            "Y" | "PAULIY" => (1, vec![]),
            "Z" | "PAULIZ" => (1, vec![]),
            "S" | "SGATE" => (1, vec![]),
            "T" | "TGATE" => (1, vec![]),
            "SX" | "SQRTX" => (1, vec![]),
            "I" | "ID" | "IDENTITY" => (1, vec![]),
            "RX" | "ROTATIONX" => (1, parameters.to_vec()),
            "RY" | "ROTATIONY" => (1, parameters.to_vec()),
            "RZ" | "ROTATIONZ" => (1, parameters.to_vec()),
            "P" | "PHASE" => (1, parameters.to_vec()),
            "U1" => (1, parameters.to_vec()),
            "U2" => (1, parameters.to_vec()),
            "U3" => (1, parameters.to_vec()),
            "CNOT" | "CX" => (2, vec![]),
            "CZ" => (2, vec![]),
            "CY" => (2, vec![]),
            "SWAP" => (2, vec![]),
            "ISWAP" => (2, vec![]),
            "ECR" => (2, vec![]),
            "TOFFOLI" | "CCX" | "CCNOT" => (3, vec![]),
            "FREDKIN" | "CSWAP" => (3, vec![]),
            _ => {
                return Err(QuantumError::UnknownGateType(format!(
                    "Unknown gate type: '{}'",
                    name
                )));
            }
        };

        Ok(Arc::new(DeserializedGate {
            name: name.to_string(),
            num_qubits,
            parameters: params,
        }))
    }
}

#[derive(Debug)]
struct DeserializedGate {
    name: String,
    num_qubits: usize,
    parameters: Vec<f64>,
}

impl Gate for DeserializedGate {
    fn name(&self) -> &str {
        &self.name
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn description(&self) -> String {
        if self.parameters.is_empty() {
            format!("{}-qubit gate '{}'", self.num_qubits, self.name)
        } else {
            let params: Vec<String> = self.parameters.iter().map(|p| format!("{}", p)).collect();
            format!("{}({})", self.name, params.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialized_gate_standard() {
        let gate = SerializedGate::StandardGate {
            name: "H".to_string(),
        };
        let json = serde_json::to_string(&gate).unwrap();
        let deserialized: SerializedGate = serde_json::from_str(&json).unwrap();
        assert_eq!(gate, deserialized);
    }

    #[test]
    fn test_serialized_gate_parameterized() {
        let gate = SerializedGate::ParameterizedGate {
            name: "RX".to_string(),
            parameters: vec![1.5],
        };
        let json = serde_json::to_string(&gate).unwrap();
        let deserialized: SerializedGate = serde_json::from_str(&json).unwrap();
        assert_eq!(gate, deserialized);
    }

    #[test]
    fn test_serialized_gate_op() {
        let gate_op = SerializedGateOp {
            gate: SerializedGate::StandardGate {
                name: "CNOT".to_string(),
            },
            qubits: vec![0, 1],
        };
        let json = serde_json::to_string(&gate_op).unwrap();
        let deserialized: SerializedGateOp = serde_json::from_str(&json).unwrap();
        assert_eq!(gate_op, deserialized);
    }
}
