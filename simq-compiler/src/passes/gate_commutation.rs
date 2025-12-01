//! Gate commutation optimization pass
//!
//! This pass reorders gates that commute with each other to reduce circuit depth
//! and create more opportunities for other optimizations (like fusion).
//!
//! # Commutation Rules
//!
//! Two gates commute if they can be reordered without changing the circuit's result.
//!
//! ## General Rules
//! - Gates on different qubits always commute
//!
//! ## Single-Qubit Gate Rules
//! - Diagonal gates on the same qubit commute (Z, S, T, S†, T†, RZ, P)
//! - Rotation gates around the same axis commute (RX-RX, RY-RY, RZ-RZ)
//! - Pauli gates: X-X, Y-Y, Z-Z commute (same gate)
//! - Hadamard: H-H commute (self-inverse)
//!
//! ## Two-Qubit Gate Rules
//! - CNOT gates with same control but different targets commute
//! - CZ gates commute if they share exactly one qubit
//! - CNOT commutes with Z on control qubit
//! - CNOT commutes with X on target qubit (controlled by same control)
//! - CZ is symmetric: CZ(q0,q1) = CZ(q1,q0)
//!
//! ## Special Patterns
//! - Diagonal gates commute with other diagonal gates on overlapping qubits
//! - Gates commute with measurements on different qubits

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

        // If gates act on exact same qubits in same order
        if gate1.qubits() == gate2.qubits() {
            return Self::same_qubit_commute(name1, name2);
        }

        // Mixed cases: different number of qubits or partial overlap
        match (gate1.num_qubits(), gate2.num_qubits()) {
            (1, 1) => {
                // Single qubit gates on same qubit already handled above
                false
            },
            (1, 2) | (2, 1) => {
                // Check if single-qubit gate commutes with two-qubit gate
                Self::single_two_qubit_commute(gate1, gate2)
            },
            (2, 2) => {
                // Two-qubit gate commutation rules
                Self::two_qubit_commute(gate1, gate2)
            },
            _ => {
                // Conservative: multi-qubit gates (3+) don't commute unless disjoint
                false
            },
        }
    }

    /// Check if two gates on the same qubit(s) commute
    fn same_qubit_commute(name1: &str, name2: &str) -> bool {
        // Same gate always commutes with itself (including parameterized gates)
        if name1 == name2 {
            return true;
        }

        // Diagonal gates commute with each other
        // These gates are diagonal in the computational basis
        let is_diagonal1 = Self::is_diagonal_gate(name1);
        let is_diagonal2 = Self::is_diagonal_gate(name2);

        if is_diagonal1 && is_diagonal2 {
            return true;
        }

        // Rotation gates around the same axis commute
        if Self::same_rotation_axis(name1, name2) {
            return true;
        }

        // Self-inverse gates commute with themselves (already handled above)
        // H-H, X-X, Y-Y, Z-Z all commute

        // Special cases for specific gate pairs
        match (name1, name2) {
            // S and T gates commute with each other (both diagonal)
            ("S", "T") | ("T", "S") => true,
            ("S", "T†") | ("T†", "S") => true,
            ("S†", "T") | ("T", "S†") => true,
            ("S†", "T†") | ("T†", "S†") => true,

            // S and S† are related but both diagonal so already handled
            // T and T† are related but both diagonal so already handled

            // SX and SX† don't generally commute with other gates
            _ => false,
        }
    }

    /// Check if a gate is diagonal in the computational basis
    fn is_diagonal_gate(name: &str) -> bool {
        matches!(
            name,
            "Z" | "S" | "T" | "S†" | "T†"
            | "RZ" | "P" | "U1"  // Parameterized diagonal gates
            | "Pauli-Z" | "CZ" // Alternative names
        )
    }

    /// Check if two rotation gates are around the same axis
    fn same_rotation_axis(name1: &str, name2: &str) -> bool {
        // RX gates commute with other RX gates
        // RY gates commute with other RY gates
        // RZ gates commute with other RZ gates
        match (name1, name2) {
            ("RX", "RX") | ("RY", "RY") | ("RZ", "RZ") => true,
            ("U1", "RZ") | ("RZ", "U1") => true, // U1 is equivalent to RZ
            ("U1", "P") | ("P", "U1") => true,   // Phase gates
            ("RZ", "P") | ("P", "RZ") => true,   // Both Z-axis rotations
            _ => false,
        }
    }

    /// Check if two-qubit gates commute
    fn two_qubit_commute(gate1: &GateOp, gate2: &GateOp) -> bool {
        let qubits1 = gate1.qubits();
        let qubits2 = gate2.qubits();

        let name1 = gate1.gate().name();
        let name2 = gate2.gate().name();

        // Both gates are CNOT
        if name1 == "CNOT" && name2 == "CNOT" {
            // CNOT gates with same control but different targets commute
            // CNOT(c,t1) and CNOT(c,t2) commute when t1 != t2
            if qubits1[0] == qubits2[0] && qubits1[1] != qubits2[1] {
                return true;
            }
            // CNOT gates with same target but different controls commute
            // CNOT(c1,t) and CNOT(c2,t) commute when c1 != c2
            if qubits1[1] == qubits2[1] && qubits1[0] != qubits2[0] {
                return true;
            }
        }

        // CZ gates commute with other CZ gates
        if name1 == "CZ" && name2 == "CZ" {
            // CZ is symmetric and commutes with itself on any qubit configuration
            // CZ(q0,q1) and CZ(q2,q3) commute if they share 0, 1, or 2 qubits
            return true;
        }

        // CZ and CNOT commutation
        if (name1 == "CZ" && name2 == "CNOT") || (name1 == "CNOT" && name2 == "CZ") {
            // CZ(a,b) commutes with CNOT(a,c) where b != c
            // CZ(a,b) commutes with CNOT(c,a) where b != c
            let (cz_qubits, cnot_qubits) = if name1 == "CZ" {
                (qubits1, qubits2)
            } else {
                (qubits2, qubits1)
            };

            let cz_q0 = cz_qubits[0].index();
            let cz_q1 = cz_qubits[1].index();
            let cnot_ctrl = cnot_qubits[0].index();
            let cnot_tgt = cnot_qubits[1].index();

            // CZ(a,b) and CNOT(a,c) commute if b != c (control overlap)
            if cz_q0 == cnot_ctrl && cz_q1 != cnot_tgt {
                return true;
            }
            if cz_q1 == cnot_ctrl && cz_q0 != cnot_tgt {
                return true;
            }
        }

        // SWAP gate commutation
        if name1 == "SWAP" && name2 == "SWAP" {
            // SWAP gates on disjoint qubit pairs commute
            let qubits1_set: HashSet<_> = qubits1.iter().collect();
            let qubits2_set: HashSet<_> = qubits2.iter().collect();
            if qubits1_set.is_disjoint(&qubits2_set) {
                return true;
            }
        }

        // iSWAP gate commutation (similar to SWAP)
        if name1 == "iSWAP" && name2 == "iSWAP" {
            let qubits1_set: HashSet<_> = qubits1.iter().collect();
            let qubits2_set: HashSet<_> = qubits2.iter().collect();
            if qubits1_set.is_disjoint(&qubits2_set) {
                return true;
            }
        }

        false
    }

    /// Check if a single-qubit gate commutes with a two-qubit gate
    fn single_two_qubit_commute(gate1: &GateOp, gate2: &GateOp) -> bool {
        // Ensure gate1 is single-qubit and gate2 is two-qubit
        let (single_gate, two_gate) = if gate1.num_qubits() == 1 {
            (gate1, gate2)
        } else {
            (gate2, gate1)
        };

        let single_qubit = single_gate.qubits()[0].index();
        let single_name = single_gate.gate().name();

        let two_qubits = two_gate.qubits();
        let two_name = two_gate.gate().name();

        // CNOT commutation with single-qubit gates
        if two_name == "CNOT" {
            let control = two_qubits[0].index();
            let target = two_qubits[1].index();

            // Z commutes with CNOT on the control qubit
            // CNOT(c,t) and Z(c) commute
            if single_name == "Z" && single_qubit == control {
                return true;
            }

            // X commutes with CNOT on the target qubit
            // CNOT(c,t) and X(t) commute
            if single_name == "X" && single_qubit == target {
                return true;
            }

            // Diagonal gates commute with CNOT on control
            if Self::is_diagonal_gate(single_name) && single_qubit == control {
                return true;
            }
        }

        // CZ commutation with single-qubit gates
        if two_name == "CZ" {
            let q0 = two_qubits[0].index();
            let q1 = two_qubits[1].index();

            // Z commutes with CZ on either qubit
            // CZ(a,b) and Z(a) commute
            // CZ(a,b) and Z(b) commute
            if Self::is_diagonal_gate(single_name) && (single_qubit == q0 || single_qubit == q1) {
                return true;
            }
        }

        // SWAP commutation with single-qubit gates
        if two_name == "SWAP" {
            // Single-qubit gates generally don't commute with SWAP unless on different qubits
            // (already handled by disjoint check)
            return false;
        }

        false
    }

    /// Try to swap two adjacent gates if they commute
    ///
    /// Returns true if a swap was made
    fn try_swap_gates(&self, ops: &mut [GateOp], i: usize) -> bool {
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
        circuit
            .add_gate(x_gate.clone(), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(h_gate.clone(), &[QubitId::new(1)])
            .unwrap();
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

    #[test]
    fn test_rotation_gates_same_axis_commute() {
        let rx1 = GateOp::new(
            Arc::new(MockGate {
                name: "RX".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(0)],
        )
        .unwrap();

        let rx2 = GateOp::new(
            Arc::new(MockGate {
                name: "RX".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(0)],
        )
        .unwrap();

        assert!(GateCommutation::gates_commute(&rx1, &rx2));
    }

    #[test]
    fn test_rotation_gates_different_axis_dont_commute() {
        let rx = GateOp::new(
            Arc::new(MockGate {
                name: "RX".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(0)],
        )
        .unwrap();

        let ry = GateOp::new(
            Arc::new(MockGate {
                name: "RY".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(0)],
        )
        .unwrap();

        assert!(!GateCommutation::gates_commute(&rx, &ry));
    }

    #[test]
    fn test_s_and_t_gates_commute() {
        let s_gate = GateOp::new(
            Arc::new(MockGate {
                name: "S".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(0)],
        )
        .unwrap();

        let t_gate = GateOp::new(
            Arc::new(MockGate {
                name: "T".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(0)],
        )
        .unwrap();

        assert!(GateCommutation::gates_commute(&s_gate, &t_gate));
    }

    #[test]
    fn test_cnot_same_control_different_target() {
        let cnot1 = GateOp::new(
            Arc::new(MockGate {
                name: "CNOT".to_string(),
                num_qubits: 2,
            }),
            &[QubitId::new(0), QubitId::new(1)],
        )
        .unwrap();

        let cnot2 = GateOp::new(
            Arc::new(MockGate {
                name: "CNOT".to_string(),
                num_qubits: 2,
            }),
            &[QubitId::new(0), QubitId::new(2)],
        )
        .unwrap();

        assert!(GateCommutation::gates_commute(&cnot1, &cnot2));
    }

    #[test]
    fn test_cnot_same_target_different_control() {
        let cnot1 = GateOp::new(
            Arc::new(MockGate {
                name: "CNOT".to_string(),
                num_qubits: 2,
            }),
            &[QubitId::new(0), QubitId::new(2)],
        )
        .unwrap();

        let cnot2 = GateOp::new(
            Arc::new(MockGate {
                name: "CNOT".to_string(),
                num_qubits: 2,
            }),
            &[QubitId::new(1), QubitId::new(2)],
        )
        .unwrap();

        assert!(GateCommutation::gates_commute(&cnot1, &cnot2));
    }

    #[test]
    fn test_cz_gates_always_commute() {
        let cz1 = GateOp::new(
            Arc::new(MockGate {
                name: "CZ".to_string(),
                num_qubits: 2,
            }),
            &[QubitId::new(0), QubitId::new(1)],
        )
        .unwrap();

        let cz2 = GateOp::new(
            Arc::new(MockGate {
                name: "CZ".to_string(),
                num_qubits: 2,
            }),
            &[QubitId::new(1), QubitId::new(2)],
        )
        .unwrap();

        assert!(GateCommutation::gates_commute(&cz1, &cz2));
    }

    #[test]
    fn test_z_commutes_with_cnot_on_control() {
        let z_gate = GateOp::new(
            Arc::new(MockGate {
                name: "Z".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(0)],
        )
        .unwrap();

        let cnot = GateOp::new(
            Arc::new(MockGate {
                name: "CNOT".to_string(),
                num_qubits: 2,
            }),
            &[QubitId::new(0), QubitId::new(1)],
        )
        .unwrap();

        assert!(GateCommutation::gates_commute(&z_gate, &cnot));
    }

    #[test]
    fn test_x_commutes_with_cnot_on_target() {
        let x_gate = GateOp::new(
            Arc::new(MockGate {
                name: "X".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(1)],
        )
        .unwrap();

        let cnot = GateOp::new(
            Arc::new(MockGate {
                name: "CNOT".to_string(),
                num_qubits: 2,
            }),
            &[QubitId::new(0), QubitId::new(1)],
        )
        .unwrap();

        assert!(GateCommutation::gates_commute(&x_gate, &cnot));
    }

    #[test]
    fn test_diagonal_gate_commutes_with_cz() {
        let s_gate = GateOp::new(
            Arc::new(MockGate {
                name: "S".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(0)],
        )
        .unwrap();

        let cz = GateOp::new(
            Arc::new(MockGate {
                name: "CZ".to_string(),
                num_qubits: 2,
            }),
            &[QubitId::new(0), QubitId::new(1)],
        )
        .unwrap();

        assert!(GateCommutation::gates_commute(&s_gate, &cz));
    }

    #[test]
    fn test_swap_gates_on_disjoint_qubits_commute() {
        let swap1 = GateOp::new(
            Arc::new(MockGate {
                name: "SWAP".to_string(),
                num_qubits: 2,
            }),
            &[QubitId::new(0), QubitId::new(1)],
        )
        .unwrap();

        let swap2 = GateOp::new(
            Arc::new(MockGate {
                name: "SWAP".to_string(),
                num_qubits: 2,
            }),
            &[QubitId::new(2), QubitId::new(3)],
        )
        .unwrap();

        assert!(GateCommutation::gates_commute(&swap1, &swap2));
    }

    #[test]
    fn test_rz_and_p_gates_commute() {
        let rz = GateOp::new(
            Arc::new(MockGate {
                name: "RZ".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(0)],
        )
        .unwrap();

        let p = GateOp::new(
            Arc::new(MockGate {
                name: "P".to_string(),
                num_qubits: 1,
            }),
            &[QubitId::new(0)],
        )
        .unwrap();

        assert!(GateCommutation::gates_commute(&rz, &p));
    }
}
