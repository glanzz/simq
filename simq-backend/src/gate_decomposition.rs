//! Gate decomposition implementations
//!
//! This module provides concrete implementations of gate decompositions
//! to various target gate sets (IBM, Rigetti, etc.)

use crate::{BackendError, GateSet, Result};
use simq_core::{Circuit, GateOp, QubitId};
use simq_gates::{
    CNot, CZ, PauliX, RotationX, RotationZ, SXGate,
};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;

/// Gate decomposer that converts gates to a target gate set
pub struct GateDecomposer {
    /// Target gate set
    target_gates: GateSet,

    /// Decomposition rules
    rules: HashMap<String, Box<dyn Fn(&GateOp) -> Result<Vec<GateOp>>>>,
}

impl GateDecomposer {
    /// Create a new gate decomposer for the target gate set
    pub fn new(target_gates: GateSet) -> Self {
        Self {
            target_gates,
            rules: HashMap::new(),
        }
    }

    /// Create decomposer for IBM native gates (RZ, SX, X, CNOT)
    pub fn ibm_native() -> Self {
        let mut decomposer = Self::new(Self::ibm_gate_set());
        decomposer.register_ibm_rules();
        decomposer
    }

    /// Create decomposer for Rigetti native gates (RZ, RX, CZ)
    pub fn rigetti_native() -> Self {
        let mut decomposer = Self::new(Self::rigetti_gate_set());
        decomposer.register_rigetti_rules();
        decomposer
    }

    /// IBM native gate set
    fn ibm_gate_set() -> GateSet {
        let mut gates = GateSet::new();
        gates.insert("RZ".to_string());
        gates.insert("SX".to_string());
        gates.insert("X".to_string());
        gates.insert("CNOT".to_string());
        gates.insert("Measure".to_string());
        gates.insert("Barrier".to_string());
        gates
    }

    /// Rigetti native gate set
    fn rigetti_gate_set() -> GateSet {
        let mut gates = GateSet::new();
        gates.insert("RZ".to_string());
        gates.insert("RX".to_string());
        gates.insert("CZ".to_string());
        gates.insert("Measure".to_string());
        gates
    }

    /// Register IBM decomposition rules
    fn register_ibm_rules(&mut self) {
        // H → RZ(π/2) SX RZ(π/2)
        self.add_rule("H", |op| {
            decompose_h_to_ibm(op.qubits()[0])
        });

        // Y → RZ(π) X
        self.add_rule("Y", |op| {
            decompose_y_to_ibm(op.qubits()[0])
        });

        // Z → RZ(π)
        self.add_rule("Z", |op| {
            decompose_z_to_ibm(op.qubits()[0])
        });

        // T → RZ(π/4)
        self.add_rule("T", |op| {
            decompose_t_to_ibm(op.qubits()[0])
        });

        // S → RZ(π/2)
        self.add_rule("S", |op| {
            decompose_s_to_ibm(op.qubits()[0])
        });

        // CZ → H CNOT H
        self.add_rule("CZ", |op| {
            decompose_cz_to_ibm(op.qubits()[0], op.qubits()[1])
        });

        // SWAP → CNOT CNOT CNOT
        self.add_rule("SWAP", |op| {
            decompose_swap_to_ibm(op.qubits()[0], op.qubits()[1])
        });
    }

    /// Register Rigetti decomposition rules
    fn register_rigetti_rules(&mut self) {
        // H → RZ(π/2) RX(π/2) RZ(π/2)
        self.add_rule("H", |op| {
            decompose_h_to_rigetti(op.qubits()[0])
        });

        // CNOT → RZ(π/2) RX(π/2) CZ RX(π/2) RZ(π/2)
        self.add_rule("CNOT", |op| {
            decompose_cnot_to_rigetti(op.qubits()[0], op.qubits()[1])
        });
    }

    /// Add a decomposition rule
    pub fn add_rule<F>(&mut self, gate_name: &str, rule: F)
    where
        F: Fn(&GateOp) -> Result<Vec<GateOp>> + 'static,
    {
        self.rules.insert(gate_name.to_string(), Box::new(rule));
    }

    /// Check if a gate needs decomposition
    pub fn needs_decomposition(&self, gate_name: &str) -> bool {
        !self.target_gates.contains(&gate_name.to_string())
    }

    /// Decompose a single gate operation
    pub fn decompose_gate(&self, op: &GateOp) -> Result<Vec<GateOp>> {
        let gate_name = op.gate().name();

        // If gate is in target set, no decomposition needed
        if !self.needs_decomposition(gate_name) {
            return Ok(vec![op.clone()]);
        }

        // Try to apply decomposition rule
        if let Some(rule) = self.rules.get(gate_name) {
            rule(op)
        } else {
            Err(BackendError::TranspilationFailed(format!(
                "No decomposition rule for gate '{}' to target gate set",
                gate_name
            )))
        }
    }

    /// Decompose an entire circuit
    pub fn decompose_circuit(&self, circuit: &Circuit) -> Result<Circuit> {
        let mut decomposed = Circuit::with_capacity(
            circuit.num_qubits(),
            circuit.len() * 3, // Estimate: some gates expand
        );

        for op in circuit.operations() {
            let decomposed_ops = self.decompose_gate(op)?;
            for new_op in decomposed_ops {
                decomposed.operations_mut().push(new_op);
            }
        }

        Ok(decomposed)
    }
}

// IBM Decomposition Functions

fn decompose_h_to_ibm(qubit: QubitId) -> Result<Vec<GateOp>> {
    // H = RZ(π/2) SX RZ(π/2)
    Ok(vec![
        GateOp::new(Arc::new(RotationZ::new(PI / 2.0)), &[qubit])?,
        GateOp::new(Arc::new(SXGate), &[qubit])?,
        GateOp::new(Arc::new(RotationZ::new(PI / 2.0)), &[qubit])?,
    ])
}

fn decompose_y_to_ibm(qubit: QubitId) -> Result<Vec<GateOp>> {
    // Y = RZ(π) X
    Ok(vec![
        GateOp::new(Arc::new(RotationZ::new(PI)), &[qubit])?,
        GateOp::new(Arc::new(PauliX), &[qubit])?,
    ])
}

fn decompose_z_to_ibm(qubit: QubitId) -> Result<Vec<GateOp>> {
    // Z = RZ(π)
    Ok(vec![GateOp::new(Arc::new(RotationZ::new(PI)), &[qubit])?])
}

fn decompose_t_to_ibm(qubit: QubitId) -> Result<Vec<GateOp>> {
    // T = RZ(π/4)
    Ok(vec![GateOp::new(Arc::new(RotationZ::new(PI / 4.0)), &[qubit])?])
}

fn decompose_s_to_ibm(qubit: QubitId) -> Result<Vec<GateOp>> {
    // S = RZ(π/2)
    Ok(vec![GateOp::new(Arc::new(RotationZ::new(PI / 2.0)), &[qubit])?])
}

fn decompose_cz_to_ibm(control: QubitId, target: QubitId) -> Result<Vec<GateOp>> {
    // CZ = H(target) CNOT(control, target) H(target)
    // But H needs to be decomposed too: H = RZ(π/2) SX RZ(π/2)
    // So: CZ = RZ(π/2) SX RZ(π/2) CNOT RZ(π/2) SX RZ(π/2) on target
    Ok(vec![
        GateOp::new(Arc::new(RotationZ::new(PI / 2.0)), &[target])?,
        GateOp::new(Arc::new(SXGate), &[target])?,
        GateOp::new(Arc::new(RotationZ::new(PI / 2.0)), &[target])?,
        GateOp::new(Arc::new(CNot), &[control, target])?,
        GateOp::new(Arc::new(RotationZ::new(PI / 2.0)), &[target])?,
        GateOp::new(Arc::new(SXGate), &[target])?,
        GateOp::new(Arc::new(RotationZ::new(PI / 2.0)), &[target])?,
    ])
}

fn decompose_swap_to_ibm(qubit1: QubitId, qubit2: QubitId) -> Result<Vec<GateOp>> {
    // SWAP = CNOT(q1,q2) CNOT(q2,q1) CNOT(q1,q2)
    Ok(vec![
        GateOp::new(Arc::new(CNot), &[qubit1, qubit2])?,
        GateOp::new(Arc::new(CNot), &[qubit2, qubit1])?,
        GateOp::new(Arc::new(CNot), &[qubit1, qubit2])?,
    ])
}

// Rigetti Decomposition Functions

fn decompose_h_to_rigetti(qubit: QubitId) -> Result<Vec<GateOp>> {
    // H = RZ(π/2) RX(π/2) RZ(π/2)
    Ok(vec![
        GateOp::new(Arc::new(RotationZ::new(PI / 2.0)), &[qubit])?,
        GateOp::new(Arc::new(RotationX::new(PI / 2.0)), &[qubit])?,
        GateOp::new(Arc::new(RotationZ::new(PI / 2.0)), &[qubit])?,
    ])
}

fn decompose_cnot_to_rigetti(control: QubitId, target: QubitId) -> Result<Vec<GateOp>> {
    // CNOT = RZ(π/2)(t) RX(π/2)(t) CZ(c,t) RX(-π/2)(t) RZ(-π/2)(t)
    // This is the standard decomposition of CNOT to CZ + single-qubit rotations
    Ok(vec![
        GateOp::new(Arc::new(RotationZ::new(PI / 2.0)), &[target])?,
        GateOp::new(Arc::new(RotationX::new(PI / 2.0)), &[target])?,
        GateOp::new(Arc::new(CZ), &[control, target])?,
        GateOp::new(Arc::new(RotationX::new(-PI / 2.0)), &[target])?,
        GateOp::new(Arc::new(RotationZ::new(-PI / 2.0)), &[target])?,
    ])
}

/// Optimization pass to remove adjacent inverse gates
pub fn optimize_inverse_gates(circuit: &Circuit) -> Result<Circuit> {
    let mut optimized = Circuit::new(circuit.num_qubits());
    let ops: Vec<_> = circuit.operations().collect();

    let mut i = 0;
    while i < ops.len() {
        let should_skip = if i + 1 < ops.len() {
            is_inverse_pair(ops[i], ops[i + 1])
        } else {
            false
        };

        if should_skip {
            // Skip both operations (they cancel)
            i += 2;
        } else {
            optimized.operations_mut().push(ops[i].clone());
            i += 1;
        }
    }

    Ok(optimized)
}

/// Check if two operations are inverses that cancel
fn is_inverse_pair(op1: &GateOp, op2: &GateOp) -> bool {
    // Check if same qubits
    if op1.qubits() != op2.qubits() {
        return false;
    }

    let name1 = op1.gate().name();
    let name2 = op2.gate().name();

    // Hermitian gates (self-inverse)
    let hermitian = ["H", "X", "Y", "Z", "CNOT", "CZ", "SWAP"];
    if hermitian.contains(&name1) && name1 == name2 {
        return true;
    }

    // T and T† cancel
    if (name1 == "T" && name2 == "Tdg") || (name1 == "Tdg" && name2 == "T") {
        return true;
    }

    // S and S† cancel
    if (name1 == "S" && name2 == "Sdg") || (name1 == "Sdg" && name2 == "S") {
        return true;
    }

    false
}

/// Optimization pass to merge single-qubit rotations
pub fn optimize_merge_rotations(circuit: &Circuit) -> Result<Circuit> {
    // This would merge adjacent RZ, RX, RY gates on the same qubit
    // For now, return as-is
    // Full implementation requires parameter extraction and gate construction
    Ok(circuit.clone())
}

/// Count gate types in circuit
pub fn analyze_gate_distribution(circuit: &Circuit) -> HashMap<String, usize> {
    circuit.gate_counts()
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_core::QubitId;

    #[test]
    fn test_ibm_gate_set() {
        let gates = GateDecomposer::ibm_gate_set();
        assert!(gates.contains(&"RZ".to_string()));
        assert!(gates.contains(&"SX".to_string()));
        assert!(gates.contains(&"X".to_string()));
        assert!(gates.contains(&"CNOT".to_string()));
    }

    #[test]
    fn test_rigetti_gate_set() {
        let gates = GateDecomposer::rigetti_gate_set();
        assert!(gates.contains(&"RZ".to_string()));
        assert!(gates.contains(&"RX".to_string()));
        assert!(gates.contains(&"CZ".to_string()));
    }

    #[test]
    fn test_decomposer_creation() {
        let ibm = GateDecomposer::ibm_native();
        assert!(!ibm.needs_decomposition("RZ"));
        assert!(!ibm.needs_decomposition("CNOT"));
        assert!(ibm.needs_decomposition("H"));
        assert!(ibm.needs_decomposition("CZ"));
    }

    #[test]
    fn test_needs_decomposition() {
        let decomposer = GateDecomposer::ibm_native();

        // Native gates don't need decomposition
        assert!(!decomposer.needs_decomposition("X"));
        assert!(!decomposer.needs_decomposition("CNOT"));

        // Non-native gates need decomposition
        assert!(decomposer.needs_decomposition("H"));
        assert!(decomposer.needs_decomposition("T"));
    }

    #[test]
    fn test_decompose_h_to_ibm() {
        let q0 = QubitId::new(0);
        let ops = decompose_h_to_ibm(q0).unwrap();

        // H decomposes to RZ(π/2) SX RZ(π/2)
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0].gate().name(), "RZ");
        assert_eq!(ops[1].gate().name(), "SX");
        assert_eq!(ops[2].gate().name(), "RZ");

        for op in &ops {
            assert_eq!(op.qubits().len(), 1);
            assert_eq!(op.qubits()[0], q0);
        }
    }

    #[test]
    fn test_decompose_t_to_ibm() {
        let q0 = QubitId::new(0);
        let ops = decompose_t_to_ibm(q0).unwrap();

        // T decomposes to RZ(π/4)
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].gate().name(), "RZ");
        assert_eq!(ops[0].qubits()[0], q0);
    }

    #[test]
    fn test_decompose_swap_to_ibm() {
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        let ops = decompose_swap_to_ibm(q0, q1).unwrap();

        // SWAP decomposes to 3 CNOTs
        assert_eq!(ops.len(), 3);
        for op in &ops {
            assert_eq!(op.gate().name(), "CNOT");
            assert_eq!(op.qubits().len(), 2);
        }

        // Pattern: CNOT(q0,q1) CNOT(q1,q0) CNOT(q0,q1)
        assert_eq!(ops[0].qubits()[0], q0);
        assert_eq!(ops[0].qubits()[1], q1);
        assert_eq!(ops[1].qubits()[0], q1);
        assert_eq!(ops[1].qubits()[1], q0);
        assert_eq!(ops[2].qubits()[0], q0);
        assert_eq!(ops[2].qubits()[1], q1);
    }

    #[test]
    fn test_decompose_cnot_to_rigetti() {
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        let ops = decompose_cnot_to_rigetti(q0, q1).unwrap();

        // CNOT decomposes to RZ, RX, CZ, RX, RZ on target
        assert_eq!(ops.len(), 5);
        assert_eq!(ops[0].gate().name(), "RZ");
        assert_eq!(ops[1].gate().name(), "RX");
        assert_eq!(ops[2].gate().name(), "CZ");
        assert_eq!(ops[3].gate().name(), "RX");
        assert_eq!(ops[4].gate().name(), "RZ");

        // Single-qubit gates on target
        assert_eq!(ops[0].qubits()[0], q1);
        assert_eq!(ops[1].qubits()[0], q1);

        // CZ on control and target
        assert_eq!(ops[2].qubits()[0], q0);
        assert_eq!(ops[2].qubits()[1], q1);

        // More single-qubit gates on target
        assert_eq!(ops[3].qubits()[0], q1);
        assert_eq!(ops[4].qubits()[0], q1);
    }

    #[test]
    fn test_decompose_gate_operation() {
        let decomposer = GateDecomposer::ibm_native();
        let q0 = QubitId::new(0);

        // Create a Hadamard gate operation
        let h_gate = Arc::new(Hadamard);
        let h_op = GateOp::new(h_gate, &[q0]).unwrap();

        // Decompose it
        let decomposed = decomposer.decompose_gate(&h_op).unwrap();

        // Should decompose to 3 gates: RZ, SX, RZ
        assert_eq!(decomposed.len(), 3);
        assert_eq!(decomposed[0].gate().name(), "RZ");
        assert_eq!(decomposed[1].gate().name(), "SX");
        assert_eq!(decomposed[2].gate().name(), "RZ");
    }

    #[test]
    fn test_decompose_circuit() {
        let decomposer = GateDecomposer::ibm_native();
        let mut circuit = Circuit::new(2);

        // Add H and T gates which need decomposition
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);

        circuit
            .add_gate(Arc::new(Hadamard), &[q0])
            .unwrap();
        circuit
            .add_gate(Arc::new(TGate), &[q1])
            .unwrap();

        // Original circuit has 2 gates
        assert_eq!(circuit.len(), 2);

        // Decompose circuit
        let decomposed = decomposer.decompose_circuit(&circuit).unwrap();

        // H → 3 gates, T → 1 gate = 4 total
        assert_eq!(decomposed.len(), 4);

        // Verify all gates are in target set
        for op in decomposed.operations() {
            assert!(
                !decomposer.needs_decomposition(op.gate().name()),
                "Gate {} should not need decomposition",
                op.gate().name()
            );
        }
    }

    #[test]
    fn test_is_inverse_pair_hermitian() {
        let q0 = QubitId::new(0);

        // Hermitian gates (self-inverse)
        let h1 = GateOp::new(Arc::new(Hadamard), &[q0]).unwrap();
        let h2 = GateOp::new(Arc::new(Hadamard), &[q0]).unwrap();
        assert!(is_inverse_pair(&h1, &h2));

        let x1 = GateOp::new(Arc::new(PauliX), &[q0]).unwrap();
        let x2 = GateOp::new(Arc::new(PauliX), &[q0]).unwrap();
        assert!(is_inverse_pair(&x1, &x2));

        // Non-inverse pairs
        let t = GateOp::new(Arc::new(TGate), &[q0]).unwrap();
        let s = GateOp::new(Arc::new(SGate), &[q0]).unwrap();
        assert!(!is_inverse_pair(&t, &s));
    }

    #[test]
    fn test_optimize_inverse_gates() {
        let mut circuit = Circuit::new(1);
        let q0 = QubitId::new(0);

        // Add H H (should cancel)
        circuit.add_gate(Arc::new(Hadamard), &[q0]).unwrap();
        circuit.add_gate(Arc::new(Hadamard), &[q0]).unwrap();

        // Add X X (should cancel)
        circuit.add_gate(Arc::new(PauliX), &[q0]).unwrap();
        circuit.add_gate(Arc::new(PauliX), &[q0]).unwrap();

        // Add Z (should remain)
        circuit.add_gate(Arc::new(PauliZ), &[q0]).unwrap();

        assert_eq!(circuit.len(), 5);

        let optimized = optimize_inverse_gates(&circuit).unwrap();

        // Should only have Z remaining
        assert_eq!(optimized.len(), 1);
        assert_eq!(optimized.operations().next().unwrap().gate().name(), "Z");
    }

    #[test]
    fn test_analyze_gate_distribution() {
        let mut circuit = Circuit::new(2);
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);

        circuit.add_gate(Arc::new(Hadamard), &[q0]).unwrap();
        circuit.add_gate(Arc::new(Hadamard), &[q1]).unwrap();
        circuit.add_gate(Arc::new(CNot), &[q0, q1]).unwrap();
        circuit.add_gate(Arc::new(TGate), &[q0]).unwrap();

        let counts = analyze_gate_distribution(&circuit);

        assert_eq!(counts.get("H"), Some(&2));
        assert_eq!(counts.get("CNOT"), Some(&1));
        assert_eq!(counts.get("T"), Some(&1));
    }
}
