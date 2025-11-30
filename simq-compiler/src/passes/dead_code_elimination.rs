//! Dead code elimination optimization pass
//!
//! This pass removes gates that don't affect the final output:
//! - Gates on qubits that are never measured or used later
//! - Gates that cancel out (e.g., X followed by X)
//! - Identity gates
//!
//! The pass works by analyzing which qubits are "live" (their state matters)
//! and removing gates on dead qubits.

use crate::passes::OptimizationPass;
use simq_core::{Circuit, Result};
use std::collections::HashSet;

/// Dead code elimination optimization pass
///
/// Removes gates that don't contribute to the final circuit output.
///
/// # Algorithm
/// 1. Mark all qubits as "live" by default (conservative approach)
/// 2. Scan circuit backwards to find which qubits are actually used
/// 3. Remove gates that only operate on dead qubits
/// 4. Remove self-inverse gate pairs (e.g., X-X, H-H, CNOT-CNOT)
///
/// # Example
/// ```ignore
/// use simq_compiler::passes::DeadCodeElimination;
/// use simq_core::Circuit;
///
/// let pass = DeadCodeElimination::new();
/// let mut circuit = Circuit::new(3);
/// // ... add gates ...
/// pass.apply(&mut circuit)?;
/// ```
/// 
use simq_core::QubitId;
#[derive(Debug, Clone)]
pub struct DeadCodeElimination {
    /// Remove self-inverse gate pairs (e.g., X-X → identity)
    remove_inverse_pairs: bool,
    /// Remove identity gates
    remove_identity: bool,
}

impl DeadCodeElimination {
    /// Create a new dead code elimination pass with default settings
    pub fn new() -> Self {
        Self {
            remove_inverse_pairs: true,
            remove_identity: true,
        }
    }

    /// Enable or disable removal of self-inverse gate pairs
    pub fn with_inverse_pair_removal(mut self, enable: bool) -> Self {
        self.remove_inverse_pairs = enable;
        self
    }

    /// Enable or disable removal of identity gates
    pub fn with_identity_removal(mut self, enable: bool) -> Self {
        self.remove_identity = enable;
        self
    }

    /// Check if a gate is self-inverse (applying it twice gives identity)
    fn is_self_inverse(gate_name: &str) -> bool {
        matches!(
            gate_name,
            "X" | "Y" | "Z" | "H" | "CNOT" | "CZ" | "SWAP" | "Pauli-X" | "Pauli-Y" | "Pauli-Z"
        )
    }

    /// Check if a gate is the identity gate
    fn is_identity(gate_name: &str) -> bool {
        gate_name == "I" || gate_name == "Identity" || gate_name == "ID"
    }

    /// Remove consecutive self-inverse gate pairs
    fn remove_inverse_pairs_impl(&self, circuit: &mut Circuit) -> bool {
        let operations = circuit.operations_mut();
        let mut to_remove = HashSet::new();
        let mut modified = false;

        // Scan for consecutive inverse pairs
        let mut i = 0;
        while i + 1 < operations.len() {
            let gate1 = &operations[i];
            let gate2 = &operations[i + 1];

            // Check if same gate type and qubits
            if gate1.gate().name() == gate2.gate().name()
                && gate1.qubits() == gate2.qubits()
                && Self::is_self_inverse(gate1.gate().name())
            {
                // Mark both for removal
                to_remove.insert(i);
                to_remove.insert(i + 1);
                modified = true;
                i += 2; // Skip both gates
            } else {
                i += 1;
            }
        }

        // Remove marked operations
        if !to_remove.is_empty() {
            let mut new_ops = Vec::with_capacity(operations.len() - to_remove.len());
            for (idx, op) in operations.iter().enumerate() {
                if !to_remove.contains(&idx) {
                    new_ops.push(op.clone());
                }
            }
            *operations = new_ops;
        }

        modified
    }

    /// Remove identity gates
    fn remove_identity_gates(&self, circuit: &mut Circuit) -> bool {
        let operations = circuit.operations_mut();
        let original_len = operations.len();

        operations.retain(|op| !Self::is_identity(op.gate().name()));

        operations.len() != original_len
    }

    /// Remove gates on unused qubits
    ///
    /// This is a conservative implementation that only removes gates if we can
    /// prove they're unused. For now, we assume all qubits might be measured,
    /// so this only removes gates that we know for certain don't affect anything.
    fn remove_unused_qubit_gates(&self, circuit: &mut Circuit) -> bool {
        // For now, return false as we need measurement information
        // to determine which qubits are truly unused.
        // This will be enhanced when we add measurement tracking.
        let _ = circuit; // Suppress unused warning
        false
    }
}

impl Default for DeadCodeElimination {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for DeadCodeElimination {
    fn name(&self) -> &str {
        "dead-code-elimination"
    }

    fn apply(&self, circuit: &mut Circuit) -> Result<bool> {
        let mut modified = false;

        // Remove identity gates
        if self.remove_identity {
            modified |= self.remove_identity_gates(circuit);
        }

        // Remove inverse pairs
        if self.remove_inverse_pairs {
            modified |= self.remove_inverse_pairs_impl(circuit);
        }

        // Remove gates on unused qubits (conservative for now)
        modified |= self.remove_unused_qubit_gates(circuit);

        Ok(modified)
    }

    fn description(&self) -> Option<&str> {
        Some("Removes gates that don't affect the circuit output")
    }

    fn iterative(&self) -> bool {
        true // Run multiple times as removing gates may reveal more opportunities
    }

    fn benefit_score(&self) -> f64 {
        0.9 // High benefit - directly reduces circuit size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_core::gate::Gate;
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
    fn test_remove_identity_gates() {
        let pass = DeadCodeElimination::new();
        let mut circuit = Circuit::new(2);

        // Add some gates including identity
        let x_gate = Arc::new(MockGate {
            name: "X".to_string(),
            num_qubits: 1,
        });
        let id_gate = Arc::new(MockGate {
            name: "I".to_string(),
            num_qubits: 1,
        });

        circuit.add_gate(x_gate.clone(), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(id_gate, &[QubitId::new(1)]).unwrap();
        circuit.add_gate(x_gate, &[QubitId::new(0)]).unwrap();

        assert_eq!(circuit.len(), 3);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(modified);
        // After removing identity, X gates become consecutive and cancel out
        // So the circuit becomes empty
        assert_eq!(circuit.len(), 0);
    }

    #[test]
    fn test_remove_inverse_pairs() {
        let pass = DeadCodeElimination::new();
        let mut circuit = Circuit::new(2);

        // Add X-X pair (should cancel)
        let x_gate = Arc::new(MockGate {
            name: "X".to_string(),
            num_qubits: 1,
        });

        circuit.add_gate(x_gate.clone(), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(x_gate.clone(), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(x_gate, &[QubitId::new(1)]).unwrap();

        assert_eq!(circuit.len(), 3);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(modified);
        assert_eq!(circuit.len(), 1); // X-X pair removed, one X remains
    }

    #[test]
    fn test_remove_h_h_pairs() {
        let pass = DeadCodeElimination::new();
        let mut circuit = Circuit::new(2);

        // Add H-H pair (should cancel)
        let h_gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });

        circuit.add_gate(h_gate.clone(), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(h_gate, &[QubitId::new(0)]).unwrap();

        assert_eq!(circuit.len(), 2);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(modified);
        assert_eq!(circuit.len(), 0); // H-H pair removed
    }

    #[test]
    fn test_no_modification() {
        let pass = DeadCodeElimination::new();
        let mut circuit = Circuit::new(2);

        // Add gates that shouldn't be removed
        let x_gate = Arc::new(MockGate {
            name: "X".to_string(),
            num_qubits: 1,
        });
        let h_gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });

        circuit.add_gate(x_gate, &[QubitId::new(0)]).unwrap();
        circuit.add_gate(h_gate, &[QubitId::new(1)]).unwrap();

        assert_eq!(circuit.len(), 2);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(!modified);
        assert_eq!(circuit.len(), 2);
    }

    #[test]
    fn test_cnot_pair_removal() {
        let pass = DeadCodeElimination::new();
        let mut circuit = Circuit::new(2);

        // Add CNOT-CNOT pair (should cancel)
        let cnot = Arc::new(MockGate {
            name: "CNOT".to_string(),
            num_qubits: 2,
        });

        circuit.add_gate(cnot.clone(), &[QubitId::new(0), QubitId::new(1)]).unwrap();
        circuit.add_gate(cnot, &[QubitId::new(0), QubitId::new(1)]).unwrap();

        assert_eq!(circuit.len(), 2);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(modified);
        assert_eq!(circuit.len(), 0);
    }

    #[test]
    fn test_different_qubits_no_removal() {
        let pass = DeadCodeElimination::new();
        let mut circuit = Circuit::new(3);

        // Add same gate on different qubits (shouldn't cancel)
        let x_gate = Arc::new(MockGate {
            name: "X".to_string(),
            num_qubits: 1,
        });

        circuit.add_gate(x_gate.clone(), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(x_gate, &[QubitId::new(1)]).unwrap();

        assert_eq!(circuit.len(), 2);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(!modified);
        assert_eq!(circuit.len(), 2);
    }
}
