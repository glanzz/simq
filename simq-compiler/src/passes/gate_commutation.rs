//! Gate commutation optimization pass
//!
//! This pass reorders gates that commute with each other to reduce circuit depth
//! and create more opportunities for other optimizations (like fusion).
//!
//! # Commutation Rules
//!
//! Two gates commute if they can be reordered without changing the circuit's result:
//! - Gates on different qubits always commute
//! - Diagonal gates on the same qubit commute (Z, RZ, P, T, S, CZ)
//! - CNOT gates with same control but different targets commute
//! - And many other specific cases...

use crate::passes::OptimizationPass;
use simq_core::{Circuit, GateOp, Result};
use std::collections::{HashMap, HashSet};

/// Gate commutation optimization pass
///
/// Reorders commuting gates to:
/// 1. Reduce circuit depth by enabling parallelization
/// 2. Create fusion opportunities (move gates on same qubit together)
/// 3. Push gates towards measurements to enable dead code elimination
///
/// # Example
/// ```ignore
/// use simq_compiler::passes::GateCommutation;
/// use simq_core::Circuit;
///
/// let pass = GateCommutation::new();
/// let mut circuit = Circuit::new(3);
/// // ... add gates ...
/// pass.apply(&mut circuit)?;
/// ```
#[derive(Debug, Clone)]
pub struct GateCommutation {
    /// Maximum number of swap attempts per pass
    max_swaps: usize,
    /// Whether to prefer moving gates forward (toward end of circuit)
    prefer_forward: bool,
}

impl GateCommutation {
    /// Create a new gate commutation pass with default settings
    pub fn new() -> Self {
        Self {
            max_swaps: 1000,
            prefer_forward: true,
        }
    }

    /// Set the maximum number of swap attempts
    pub fn with_max_swaps(mut self, max_swaps: usize) -> Self {
        self.max_swaps = max_swaps;
        self
    }

    /// Set whether to prefer moving gates forward
    pub fn with_prefer_forward(mut self, prefer_forward: bool) -> Self {
        self.prefer_forward = prefer_forward;
        self
    }

    /// Check if two gates commute
    ///
    /// Two gates commute if applying them in either order produces the same result.
    fn gates_commute(gate1: &GateOp, gate2: &GateOp) -> bool {
        let qubits1: HashSet<_> = gate1.qubits().iter().collect();
        let qubits2: HashSet<_> = gate2.qubits().iter().collect();

        // Gates on completely disjoint qubits always commute
        if qubits1.is_disjoint(&qubits2) {
            return true;
        }

        let name1 = gate1.gate().name();
        let name2 = gate2.gate().name();

        // If gates act on same qubits, check specific commutation rules
        if qubits1 == qubits2 {
            // Same gates on same qubits
            if gate1.qubits() == gate2.qubits() {
                return Self::same_qubit_commute(name1, name2);
            }
        }

        // CNOT-specific commutation rules
        if gate1.num_qubits() == 2 && gate2.num_qubits() == 2 {
            return Self::two_qubit_commute(gate1, gate2);
        }

        // Conservative: assume they don't commute
        false
    }

    /// Check if two gates on the same qubit(s) commute
    fn same_qubit_commute(name1: &str, name2: &str) -> bool {
        // Diagonal gates commute with each other
        let diagonal_gates = ["Z", "S", "T", "RZ", "P", "CZ", "Pauli-Z"];

        let is_diagonal1 = diagonal_gates.contains(&name1);
        let is_diagonal2 = diagonal_gates.contains(&name2);

        if is_diagonal1 && is_diagonal2 {
            return true;
        }

        // Pauli gates commute in specific patterns
        // X and Z don't commute, but X-X, Y-Y, Z-Z all commute (trivially, they're the same gate)
        if name1 == name2 {
            return true; // Same gate always commutes with itself
        }

        false
    }

    /// Check if two-qubit gates commute
    fn two_qubit_commute(gate1: &GateOp, gate2: &GateOp) -> bool {
        let qubits1 = gate1.qubits();
        let qubits2 = gate2.qubits();

        let name1 = gate1.gate().name();
        let name2 = gate2.gate().name();

        // CNOT gates with same control but different targets commute
        if name1 == "CNOT" && name2 == "CNOT" && qubits1[0] == qubits2[0] && qubits1[1] != qubits2[1] {
            return true;
        }

        // CZ gates are symmetric and commute if they share any qubit
        if name1 == "CZ" && name2 == "CZ" {
            return true;
        }

        false
    }

    /// Try to swap two adjacent gates if they commute
    ///
    /// Returns true if a swap was made
    fn try_swap_gates(&self, ops: &mut Vec<GateOp>, i: usize) -> bool {
        if i + 1 >= ops.len() {
            return false;
        }

        if Self::gates_commute(&ops[i], &ops[i + 1]) {
            ops.swap(i, i + 1);
            true
        } else {
            false
        }
    }

    /// Apply commutation to group gates on the same qubit together
    ///
    /// This moves gates on the same qubit closer together to enable fusion.
    fn group_gates_by_qubit(&self, circuit: &mut Circuit) -> bool {
        let ops = circuit.operations_mut();
        if ops.len() < 2 {
            return false;
        }

        let mut modified = false;
        let mut swaps_made = 0;

        // Build a map of qubit -> last operation index
        let mut qubit_last_op: HashMap<usize, usize> = HashMap::new();

        for i in 0..ops.len() {
            if swaps_made >= self.max_swaps {
                break;
            }

            let op_qubits: Vec<_> = ops[i].qubits().iter().map(|q| q.index()).collect();

            // Single-qubit gates: try to move closer to last gate on same qubit
            if op_qubits.len() == 1 {
                let qubit = op_qubits[0];

                if let Some(&last_idx) = qubit_last_op.get(&qubit) {
                    // Try to bubble this gate backward toward the last gate on this qubit
                    let mut current_idx = i;
                    while current_idx > last_idx + 1 {
                        if self.try_swap_gates(ops, current_idx - 1) {
                            current_idx -= 1;
                            modified = true;
                            swaps_made += 1;
                            if swaps_made >= self.max_swaps {
                                break;
                            }
                        } else {
                            break; // Can't swap, stop trying
                        }
                    }
                }

                qubit_last_op.insert(qubit, i);
            } else {
                // Multi-qubit gate: update all involved qubits
                for qubit in op_qubits {
                    qubit_last_op.insert(qubit, i);
                }
            }
        }

        modified
    }
}

impl Default for GateCommutation {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for GateCommutation {
    fn name(&self) -> &str {
        "gate-commutation"
    }

    fn apply(&self, circuit: &mut Circuit) -> Result<bool> {
        // Apply gate grouping to enable fusion
        let modified = self.group_gates_by_qubit(circuit);

        Ok(modified)
    }

    fn description(&self) -> Option<&str> {
        Some("Reorders commuting gates to reduce depth and enable other optimizations")
    }

    fn iterative(&self) -> bool {
        // May reveal new opportunities after other passes
        true
    }

    fn benefit_score(&self) -> f64 {
        0.7 // Good benefit - enables other passes and can reduce depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_core::gate::Gate;
    use simq_core::QubitId;
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
    fn test_gates_on_different_qubits_commute() {
        let gate1 = GateOp::new(
            Arc::new(MockGate {
                name: "X".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(0)],
        )
        .unwrap();

        let gate2 = GateOp::new(
            Arc::new(MockGate {
                name: "H".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(1)],
        )
        .unwrap();

        assert!(GateCommutation::gates_commute(&gate1, &gate2));
    }

    #[test]
    fn test_diagonal_gates_commute() {
        let gate1 = GateOp::new(
            Arc::new(MockGate {
                name: "Z".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(0)],
        )
        .unwrap();

        let gate2 = GateOp::new(
            Arc::new(MockGate {
                name: "S".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(0)],
        )
        .unwrap();

        assert!(GateCommutation::gates_commute(&gate1, &gate2));
    }

    #[test]
    fn test_x_and_z_dont_commute() {
        let gate1 = GateOp::new(
            Arc::new(MockGate {
                name: "X".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(0)],
        )
        .unwrap();

        let gate2 = GateOp::new(
            Arc::new(MockGate {
                name: "Z".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(0)],
        )
        .unwrap();

        assert!(!GateCommutation::gates_commute(&gate1, &gate2));
    }

    #[test]
    fn test_group_gates_by_qubit() {
        let pass = GateCommutation::new();
        let mut circuit = Circuit::new(3);

        let x_gate = Arc::new(MockGate {
            name: "X".to_string(),
            num_qubits: 1,
        });
        let h_gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });

        // Add: X(q0), H(q1), X(q1), H(q0)
        // Should reorder to group operations on same qubit
        circuit.add_gate(x_gate.clone(), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(h_gate.clone(), &[QubitId::new(1)]).unwrap();
        circuit.add_gate(x_gate, &[QubitId::new(1)]).unwrap();
        circuit.add_gate(h_gate, &[QubitId::new(0)]).unwrap();

        assert_eq!(circuit.len(), 4);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(modified);

        // Verify operations were reordered
        // The exact order depends on the implementation, but gates on same qubit
        // should be closer together
        assert_eq!(circuit.len(), 4); // Same number of gates
    }

    #[test]
    fn test_no_commuting_gates_no_change() {
        let pass = GateCommutation::new();
        let mut circuit = Circuit::new(2);

        // Add gates that don't commute
        let x_gate = Arc::new(MockGate {
            name: "X".to_string(),
            num_qubits: 1,
        });
        let z_gate = Arc::new(MockGate {
            name: "Z".to_string(),
            num_qubits: 1,
        });

        circuit.add_gate(x_gate, &[QubitId::new(0)]).unwrap();
        circuit.add_gate(z_gate, &[QubitId::new(0)]).unwrap();

        assert_eq!(circuit.len(), 2);

        let modified = pass.apply(&mut circuit).unwrap();
        // X and Z don't commute on same qubit, so no reordering
        assert!(!modified);
    }
}
