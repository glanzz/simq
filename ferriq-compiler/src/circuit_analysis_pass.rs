//! Circuit pattern analysis for adaptive pass selection
//!
//! This module provides analysis of circuit characteristics to enable
//! adaptive pass selection and optimization strategies.

use ferriq_core::{gate::Gate, Circuit, QubitId};
use std::collections::{HashMap, HashSet};

/// Characteristics of a circuit used for adaptive optimization
#[derive(Debug, Clone)]
pub struct CircuitCharacteristics {
    /// Total number of gates
    pub gate_count: usize,
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit depth
    pub depth: usize,
    /// Ratio of single-qubit to two-qubit gates
    pub single_to_two_qubit_ratio: f64,
    /// Density of commutation opportunities
    pub commutation_density: f64,
    /// Density of fuseable gates
    pub fusion_density: f64,
    /// Density of template pattern matches
    pub template_density: f64,
    /// Density of dead code (inverse pairs)
    pub dead_code_density: f64,
    /// Average gates per qubit
    pub gates_per_qubit: f64,
    /// Parallelism factor (average gates per layer)
    pub parallelism_factor: f64,
}

impl CircuitCharacteristics {
    /// Analyze a circuit to extract characteristics
    pub fn analyze(circuit: &Circuit) -> Self {
        let gate_count = circuit.len();
        let num_qubits = circuit.num_qubits();

        if gate_count == 0 {
            return Self::empty(num_qubits);
        }

        let depth = Self::calculate_depth(circuit);
        let (single_qubit, two_qubit) = Self::count_gate_types(circuit);
        let single_to_two_qubit_ratio = if two_qubit > 0 {
            single_qubit as f64 / two_qubit as f64
        } else {
            single_qubit as f64
        };

        let commutation_density = Self::estimate_commutation_density(circuit);
        let fusion_density = Self::estimate_fusion_density(circuit);
        let template_density = Self::estimate_template_density(circuit);
        let dead_code_density = Self::estimate_dead_code_density(circuit);

        let gates_per_qubit = gate_count as f64 / num_qubits as f64;
        let parallelism_factor = if depth > 0 {
            gate_count as f64 / depth as f64
        } else {
            0.0
        };

        Self {
            gate_count,
            num_qubits,
            depth,
            single_to_two_qubit_ratio,
            commutation_density,
            fusion_density,
            template_density,
            dead_code_density,
            gates_per_qubit,
            parallelism_factor,
        }
    }

    fn empty(num_qubits: usize) -> Self {
        Self {
            gate_count: 0,
            num_qubits,
            depth: 0,
            single_to_two_qubit_ratio: 0.0,
            commutation_density: 0.0,
            fusion_density: 0.0,
            template_density: 0.0,
            dead_code_density: 0.0,
            gates_per_qubit: 0.0,
            parallelism_factor: 0.0,
        }
    }

    /// Calculate circuit depth
    fn calculate_depth(circuit: &Circuit) -> usize {
        // Use the built-in depth computation if available
        circuit.compute_depth().unwrap_or_else(|_| {
            // Fallback: simple depth calculation
            let mut qubit_depths: HashMap<QubitId, usize> = HashMap::new();
            let mut max_depth = 0;

            for op in circuit.operations() {
                let qubits = op.qubits();
                let current_depth = qubits
                    .iter()
                    .map(|q| *qubit_depths.get(q).unwrap_or(&0))
                    .max()
                    .unwrap_or(0);

                let new_depth = current_depth + 1;
                for qubit in qubits {
                    qubit_depths.insert(*qubit, new_depth);
                }
                max_depth = max_depth.max(new_depth);
            }

            max_depth
        })
    }

    /// Count single-qubit and two-qubit gates
    fn count_gate_types(circuit: &Circuit) -> (usize, usize) {
        let mut single_qubit = 0;
        let mut two_qubit = 0;

        for op in circuit.operations() {
            match op.num_qubits() {
                1 => single_qubit += 1,
                2 => two_qubit += 1,
                _ => {},
            }
        }

        (single_qubit, two_qubit)
    }

    /// Estimate density of commutation opportunities
    fn estimate_commutation_density(circuit: &Circuit) -> f64 {
        let mut opportunities = 0;
        let total_gates = circuit.len();

        if total_gates < 2 {
            return 0.0;
        }

        let operations: Vec<_> = circuit.operations().collect();

        for i in 0..operations.len().saturating_sub(1) {
            let op1 = operations[i];
            let gate1 = op1.gate();
            let qubits1: HashSet<_> = op1.qubits().iter().cloned().collect();
            let op2 = operations[i + 1];
            let gate2 = op2.gate();
            let qubits2: HashSet<_> = op2.qubits().iter().cloned().collect();

            // Check if gates might commute
            if Self::might_commute(&**gate1, &qubits1, &**gate2, &qubits2) {
                opportunities += 1;
            }
        }

        opportunities as f64 / (total_gates - 1) as f64
    }

    /// Check if two gates might commute (conservative estimate)
    fn might_commute(
        gate1: &dyn Gate,
        qubits1: &HashSet<QubitId>,
        gate2: &dyn Gate,
        qubits2: &HashSet<QubitId>,
    ) -> bool {
        let name1 = gate1.name();
        let name2 = gate2.name();

        // Disjoint qubits always commute
        if qubits1.is_disjoint(qubits2) {
            return true;
        }

        // Diagonal gates commute
        if Self::is_diagonal(name1) && Self::is_diagonal(name2) {
            return true;
        }

        // Same-axis rotations commute
        if Self::same_rotation_axis(name1, name2) {
            return true;
        }

        false
    }

    fn is_diagonal(name: &str) -> bool {
        matches!(name, "Z" | "S" | "T" | "S†" | "T†" | "RZ" | "P" | "U1" | "CZ")
    }

    fn same_rotation_axis(name1: &str, name2: &str) -> bool {
        matches!((name1, name2), ("RX", "RX") | ("RY", "RY") | ("RZ", "RZ"))
    }

    /// Estimate density of fuseable gate sequences
    fn estimate_fusion_density(circuit: &Circuit) -> f64 {
        let mut fuseable_sequences = 0;
        let total_gates = circuit.len();

        if total_gates < 2 {
            return 0.0;
        }

        let operations: Vec<_> = circuit.operations().collect();

        for i in 0..operations.len().saturating_sub(1) {
            let op1 = operations[i];
            let op2 = operations[i + 1];

            // Check if consecutive gates on same qubit (potential fusion)
            if op1.num_qubits() == 1 && op2.num_qubits() == 1 && op1.qubits()[0] == op2.qubits()[0]
            {
                fuseable_sequences += 1;
            }
        }

        fuseable_sequences as f64 / (total_gates - 1) as f64
    }

    /// Estimate density of template pattern matches
    fn estimate_template_density(circuit: &Circuit) -> f64 {
        let mut patterns = 0;
        let total_gates = circuit.len();

        if total_gates < 2 {
            return 0.0;
        }

        let operations: Vec<_> = circuit.operations().collect();

        for i in 0..operations.len().saturating_sub(1) {
            let gate1 = operations[i].gate().name();
            let gate2 = operations[i + 1].gate().name();

            // Check for common patterns
            if Self::is_known_pattern(gate1, gate2) {
                patterns += 1;
            }
        }

        patterns as f64 / (total_gates - 1) as f64
    }

    fn is_known_pattern(gate1: &str, gate2: &str) -> bool {
        matches!(
            (gate1, gate2),
            ("H", "Z") | ("H", "X") | ("X", "X") | ("Y", "Y") | ("Z", "Z") | ("H", "H")
        )
    }

    /// Estimate density of dead code (inverse pairs)
    fn estimate_dead_code_density(circuit: &Circuit) -> f64 {
        let mut inverse_pairs = 0;
        let total_gates = circuit.len();

        if total_gates < 2 {
            return 0.0;
        }

        let operations: Vec<_> = circuit.operations().collect();

        for i in 0..operations.len().saturating_sub(1) {
            let op1 = operations[i];
            let op2 = operations[i + 1];
            let gate1 = op1.gate().name();
            let gate2 = op2.gate().name();
            let qubits1 = op1.qubits();
            let qubits2 = op2.qubits();

            // Check for self-inverse gates on same qubits
            if qubits1 == qubits2 && Self::is_self_inverse_pair(gate1, gate2) {
                inverse_pairs += 1;
            }
        }

        inverse_pairs as f64 / (total_gates - 1) as f64
    }

    fn is_self_inverse_pair(gate1: &str, gate2: &str) -> bool {
        gate1 == gate2 && matches!(gate1, "X" | "Y" | "Z" | "H" | "CNOT" | "CZ" | "SWAP")
    }

    /// Determine if commutation pass would be beneficial
    pub fn should_use_commutation(&self) -> bool {
        self.commutation_density > 0.1 || self.gate_count > 50
    }

    /// Determine if fusion pass would be beneficial
    pub fn should_use_fusion(&self) -> bool {
        self.fusion_density > 0.05 || self.single_to_two_qubit_ratio > 1.0
    }

    /// Determine if template matching would be beneficial
    pub fn should_use_templates(&self) -> bool {
        self.template_density > 0.05 || self.gate_count > 20
    }

    /// Determine if dead code elimination would be beneficial
    pub fn should_use_dce(&self) -> bool {
        self.dead_code_density > 0.01 || self.gate_count > 10
    }

    /// Suggest optimal number of iterations based on circuit size
    pub fn suggest_iterations(&self) -> usize {
        if self.gate_count < 20 {
            3
        } else if self.gate_count < 100 {
            5
        } else if self.gate_count < 500 {
            7
        } else {
            10
        }
    }

    /// Categorize circuit size
    pub fn size_category(&self) -> CircuitSize {
        if self.gate_count < 50 {
            CircuitSize::Small
        } else if self.gate_count < 200 {
            CircuitSize::Medium
        } else if self.gate_count < 1000 {
            CircuitSize::Large
        } else {
            CircuitSize::VeryLarge
        }
    }
}

/// Circuit size category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitSize {
    Small,
    Medium,
    Large,
    VeryLarge,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferriq_core::gate::Gate as GateTrait;
    use ferriq_core::Circuit;
    use ferriq_gates::standard::{CNot, Hadamard, PauliX, PauliY, PauliZ, RotationX, RotationZ};
    use std::sync::Arc;

    #[test]
    fn test_empty_circuit() {
        let circuit = Circuit::new(3);
        let chars = CircuitCharacteristics::analyze(&circuit);

        assert_eq!(chars.gate_count, 0);
        assert_eq!(chars.num_qubits, 3);
        assert_eq!(chars.depth, 0);
    }

    #[test]
    fn test_size_categories() {
        let circuit = Circuit::new(3);
        let chars = CircuitCharacteristics::analyze(&circuit);
        assert_eq!(chars.size_category(), CircuitSize::Small);
    }

    #[test]
    fn test_suggest_iterations() {
        let circuit = Circuit::new(3);
        let chars = CircuitCharacteristics::analyze(&circuit);
        assert_eq!(chars.suggest_iterations(), 3);
    }

    /// Covers the single-gate circuit path: total_gates < 2 short-circuit
    /// (lines 141-142, 205-206, 230-231, 261-262) plus the `_ =>` arm in
    /// `count_gate_types` (line 129) via a three-qubit CNOT-like gate is not
    /// applicable here, so we cover 129 separately below.
    #[test]
    fn test_single_gate_circuit_zero_densities() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(Hadamard) as Arc<dyn GateTrait>, &[QubitId::new(0)])
            .unwrap();

        let chars = CircuitCharacteristics::analyze(&circuit);
        assert_eq!(chars.gate_count, 1);
        // With only one gate, all "adjacent pair" densities must be 0.0
        assert_eq!(chars.commutation_density, 0.0);
        assert_eq!(chars.fusion_density, 0.0);
        assert_eq!(chars.template_density, 0.0);
        assert_eq!(chars.dead_code_density, 0.0);
    }

    /// Covers the `_ => {}` arm (line 129) in `count_gate_types` for gates
    /// acting on more than two qubits (e.g. a 3-qubit Toffoli-style gate).
    #[derive(Debug)]
    struct ThreeQubitMockGate;

    impl GateTrait for ThreeQubitMockGate {
        fn name(&self) -> &str {
            "CCX"
        }
        fn num_qubits(&self) -> usize {
            3
        }
    }

    #[test]
    fn test_count_gate_types_ignores_multi_qubit_gates() {
        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(
                Arc::new(ThreeQubitMockGate) as Arc<dyn GateTrait>,
                &[QubitId::new(0), QubitId::new(1), QubitId::new(2)],
            )
            .unwrap();
        circuit
            .add_gate(Arc::new(Hadamard) as Arc<dyn GateTrait>, &[QubitId::new(0)])
            .unwrap();

        let chars = CircuitCharacteristics::analyze(&circuit);
        // single_to_two_qubit_ratio uses (single_qubit as f64) when two_qubit == 0
        // Only the Hadamard counts as single-qubit; the 3-qubit gate is ignored.
        assert_eq!(chars.single_to_two_qubit_ratio, 1.0);
    }

    /// Covers the diagonal-gates-commute branch (line 181) in `might_commute`.
    #[test]
    fn test_diagonal_gates_commute() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(PauliZ) as Arc<dyn GateTrait>, &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(RotationZ::new(0.5)) as Arc<dyn GateTrait>, &[QubitId::new(0)])
            .unwrap();

        let chars = CircuitCharacteristics::analyze(&circuit);
        // Z and RZ are both diagonal and act on the same qubit -> should commute
        assert_eq!(chars.commutation_density, 1.0);
    }

    /// Covers the same-rotation-axis-commutes branch (line 186) in `might_commute`.
    #[test]
    fn test_same_axis_rotations_commute() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(RotationX::new(0.3)) as Arc<dyn GateTrait>, &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(RotationX::new(0.7)) as Arc<dyn GateTrait>, &[QubitId::new(0)])
            .unwrap();

        let chars = CircuitCharacteristics::analyze(&circuit);
        // RX, RX share the same rotation axis -> should commute
        assert_eq!(chars.commutation_density, 1.0);
    }

    /// Sanity check that non-commuting, non-diagonal, non-same-axis gates on
    /// the same qubit are correctly identified as not commuting (covers the
    /// `false` fallthrough at the end of `might_commute`).
    #[test]
    fn test_non_commuting_gates() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(PauliX) as Arc<dyn GateTrait>, &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliY) as Arc<dyn GateTrait>, &[QubitId::new(0)])
            .unwrap();

        let chars = CircuitCharacteristics::analyze(&circuit);
        assert_eq!(chars.commutation_density, 0.0);
    }

    /// Covers `suggest_iterations` returning 10 for large circuits (line 317)
    /// and `size_category` Medium/Large/VeryLarge branches (lines 325-330).
    fn build_circuit_with_gate_count(n: usize) -> Circuit {
        let mut circuit = Circuit::new(1);
        let h = Arc::new(Hadamard) as Arc<dyn GateTrait>;
        for _ in 0..n {
            circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();
        }
        circuit
    }

    #[test]
    fn test_suggest_iterations_large_circuit() {
        let circuit = build_circuit_with_gate_count(600);
        let chars = CircuitCharacteristics::analyze(&circuit);
        assert_eq!(chars.gate_count, 600);
        assert_eq!(chars.suggest_iterations(), 10);
    }

    #[test]
    fn test_size_category_medium() {
        let circuit = build_circuit_with_gate_count(100);
        let chars = CircuitCharacteristics::analyze(&circuit);
        assert_eq!(chars.size_category(), CircuitSize::Medium);
    }

    #[test]
    fn test_size_category_large() {
        let circuit = build_circuit_with_gate_count(500);
        let chars = CircuitCharacteristics::analyze(&circuit);
        assert_eq!(chars.size_category(), CircuitSize::Large);
    }

    #[test]
    fn test_size_category_very_large() {
        let circuit = build_circuit_with_gate_count(1000);
        let chars = CircuitCharacteristics::analyze(&circuit);
        assert_eq!(chars.size_category(), CircuitSize::VeryLarge);
    }

    /// Sanity check for a two-qubit gate, exercising the `two_qubit += 1` arm
    /// in `count_gate_types` and the `two_qubit > 0` branch of the ratio calc
    /// (already covered by other tests indirectly, but explicit here).
    #[test]
    fn test_two_qubit_gate_ratio() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(Hadamard) as Arc<dyn GateTrait>, &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(CNot) as Arc<dyn GateTrait>, &[QubitId::new(0), QubitId::new(1)])
            .unwrap();

        let chars = CircuitCharacteristics::analyze(&circuit);
        assert_eq!(chars.single_to_two_qubit_ratio, 1.0);
    }
}
