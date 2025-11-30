//! Tests for serialization functionality

#![cfg(test)]
#![cfg(feature = "serialization")]

use crate::circuit::Circuit;
use crate::gate::Gate;
use crate::serialization::circuit::SerializedCircuit;
use crate::serialization::gate::{SerializedGate, SerializedGateOp};
use crate::QubitId;
use std::sync::Arc;

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
fn test_serialized_circuit_json() {
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
fn test_circuit_serialization_json() {
    // Create a simple circuit
    let mut circuit = Circuit::new(2);
    let gate = Arc::new(MockGate {
        name: "H".to_string(),
        num_qubits: 1,
    });
    circuit.add_gate(gate, &[QubitId::new(0)]).unwrap();

    // Serialize to JSON
    let json = circuit.to_json().unwrap();
    assert!(!json.is_empty());
    assert!(json.contains("H"));
    assert!(json.contains("2"));
}

#[test]
fn test_circuit_serialization_binary() {
    // Create a simple circuit
    let mut circuit = Circuit::new(2);
    let gate = Arc::new(MockGate {
        name: "H".to_string(),
        num_qubits: 1,
    });
    circuit.add_gate(gate, &[QubitId::new(0)]).unwrap();

    // Serialize to binary
    let bytes = circuit.to_bytes().unwrap();
    assert!(!bytes.is_empty());

    // Note: We can't deserialize without gate implementations,
    // but we can verify the serialization works
}

#[test]
fn test_circuit_cache_key() {
    let mut circuit1 = Circuit::new(2);
    let gate1 = Arc::new(MockGate {
        name: "H".to_string(),
        num_qubits: 1,
    });
    circuit1.add_gate(gate1, &[QubitId::new(0)]).unwrap();

    let mut circuit2 = Circuit::new(2);
    let gate2 = Arc::new(MockGate {
        name: "H".to_string(),
        num_qubits: 1,
    });
    circuit2.add_gate(gate2, &[QubitId::new(0)]).unwrap();

    // Same circuit structure should have same cache key
    assert_eq!(circuit1.cache_key(), circuit2.cache_key());

    // Different circuit should have different cache key
    let mut circuit3 = Circuit::new(2);
    let gate3 = Arc::new(MockGate {
        name: "X".to_string(),
        num_qubits: 1,
    });
    circuit3.add_gate(gate3, &[QubitId::new(0)]).unwrap();

    assert_ne!(circuit1.cache_key(), circuit3.cache_key());
}

#[cfg(feature = "cache")]
#[test]
fn test_memory_cache() {
    use crate::serialization::cache::{CircuitCache, CircuitKey, MemoryCache};

    let cache = MemoryCache::new();
    let mut circuit = Circuit::new(2);
    let gate = Arc::new(MockGate {
        name: "H".to_string(),
        num_qubits: 1,
    });
    circuit.add_gate(gate, &[QubitId::new(0)]).unwrap();

    let key = CircuitKey::from_circuit(&circuit);

    // Cache miss
    assert!(cache.get(&key).is_none());

    // Put in cache
    cache.put(key.clone(), circuit.clone()).unwrap();

    // Cache hit
    let cached = cache.get(&key);
    assert!(cached.is_some());

    // Verify it's the same circuit
    let cached_circuit = cached.unwrap();
    assert_eq!(cached_circuit.num_qubits(), circuit.num_qubits());
    assert_eq!(cached_circuit.len(), circuit.len());
}
