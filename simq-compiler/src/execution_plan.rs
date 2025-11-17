//! Execution Plan Generation
//!
//! This module generates optimized execution plans for quantum circuits,
//! including gate scheduling, parallelization opportunities, and resource
//! estimation.

use simq_core::{Circuit, QubitId};
use std::collections::{HashMap, HashSet};

/// Represents a layer of gates that can be executed in parallel
#[derive(Debug, Clone)]
pub struct ExecutionLayer {
    /// Gates in this layer (by instruction index)
    pub gates: Vec<usize>,
    /// Qubits involved in this layer
    pub qubits: HashSet<QubitId>,
    /// Estimated execution time for this layer
    pub estimated_time: f64,
}

impl ExecutionLayer {
    /// Create a new execution layer
    pub fn new() -> Self {
        Self {
            gates: Vec::new(),
            qubits: HashSet::new(),
            estimated_time: 0.0,
        }
    }

    /// Add a gate to this layer
    pub fn add_gate(&mut self, gate_index: usize, qubits: &[QubitId], time: f64) {
        self.gates.push(gate_index);
        self.qubits.extend(qubits.iter().cloned());
        self.estimated_time = self.estimated_time.max(time);
    }

    /// Check if a gate can be added to this layer (no qubit conflicts)
    pub fn can_add(&self, qubits: &[QubitId]) -> bool {
        qubits.iter().all(|q| !self.qubits.contains(q))
    }

    /// Number of gates in this layer
    pub fn len(&self) -> usize {
        self.gates.len()
    }

    /// Check if layer is empty
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }
}

impl Default for ExecutionLayer {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution plan for a quantum circuit
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Execution layers (gates grouped by parallelization)
    pub layers: Vec<ExecutionLayer>,
    /// Total estimated execution time
    pub total_time: f64,
    /// Circuit depth (number of layers)
    pub depth: usize,
    /// Total number of gates
    pub gate_count: usize,
    /// Parallelism factor (average gates per layer)
    pub parallelism_factor: f64,
    /// Critical path (longest dependency chain)
    pub critical_path_length: usize,
    /// Resource requirements
    pub resources: ResourceRequirements,
}

impl ExecutionPlan {
    /// Create a new execution plan
    fn new(layers: Vec<ExecutionLayer>, gate_count: usize) -> Self {
        let depth = layers.len();
        let total_time: f64 = layers.iter().map(|l| l.estimated_time).sum();
        let parallelism_factor = if depth > 0 {
            gate_count as f64 / depth as f64
        } else {
            0.0
        };

        // Find critical path
        let critical_path_length = layers.iter().map(|l| l.len()).max().unwrap_or(0);

        Self {
            layers,
            total_time,
            depth,
            gate_count,
            parallelism_factor,
            critical_path_length,
            resources: ResourceRequirements::default(),
        }
    }

    /// Get the efficiency of parallelization (0.0 to 1.0)
    ///
    /// 1.0 means perfect parallelization, 0.0 means purely sequential
    pub fn parallelization_efficiency(&self) -> f64 {
        if self.gate_count == 0 {
            return 0.0;
        }

        // Efficiency = actual parallelism / theoretical maximum
        let sequential_depth = self.gate_count;
        let actual_depth = self.depth;

        if sequential_depth == 0 {
            0.0
        } else {
            1.0 - (actual_depth as f64 / sequential_depth as f64)
        }
    }

    /// Get the average layer size
    pub fn average_layer_size(&self) -> f64 {
        self.parallelism_factor
    }

    /// Find the bottleneck layer (largest layer)
    pub fn bottleneck_layer(&self) -> Option<usize> {
        self.layers
            .iter()
            .enumerate()
            .max_by_key(|(_, layer)| layer.len())
            .map(|(idx, _)| idx)
    }
}

impl std::fmt::Display for ExecutionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Execution Plan:")?;
        writeln!(f, "  Circuit depth: {} layers", self.depth)?;
        writeln!(f, "  Total gates: {}", self.gate_count)?;
        writeln!(f, "  Parallelism factor: {:.2}", self.parallelism_factor)?;
        writeln!(f, "  Parallelization efficiency: {:.1}%",
            self.parallelization_efficiency() * 100.0)?;
        writeln!(f, "  Estimated execution time: {:.2} units", self.total_time)?;
        writeln!(f, "  Critical path length: {}", self.critical_path_length)?;
        writeln!(f, "\nLayer breakdown:")?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "  Layer {}: {} gates, {:.2} time units",
                i, layer.len(), layer.estimated_time)?;
        }
        Ok(())
    }
}

/// Resource requirements for circuit execution
#[derive(Debug, Clone, Default)]
pub struct ResourceRequirements {
    /// Peak qubit count
    pub peak_qubits: usize,
    /// Peak classical memory (bytes)
    pub peak_memory: usize,
    /// Number of measurements
    pub measurement_count: usize,
    /// Two-qubit gate count
    pub two_qubit_gates: usize,
}

/// Execution plan generator
pub struct ExecutionPlanner {
    /// Gate time estimates (gate name -> time in arbitrary units)
    gate_times: HashMap<String, f64>,
}

impl ExecutionPlanner {
    /// Create a new execution planner with default gate times
    pub fn new() -> Self {
        let mut gate_times = HashMap::new();

        // Default gate times (arbitrary units)
        gate_times.insert("I".to_string(), 0.0);
        gate_times.insert("H".to_string(), 1.0);
        gate_times.insert("X".to_string(), 1.0);
        gate_times.insert("Y".to_string(), 1.0);
        gate_times.insert("Z".to_string(), 0.1); // Virtual gate
        gate_times.insert("S".to_string(), 0.1);
        gate_times.insert("T".to_string(), 0.1);
        gate_times.insert("RX".to_string(), 1.0);
        gate_times.insert("RY".to_string(), 1.0);
        gate_times.insert("RZ".to_string(), 0.1);
        gate_times.insert("CNOT".to_string(), 10.0); // Two-qubit gates are expensive
        gate_times.insert("CZ".to_string(), 10.0);
        gate_times.insert("SWAP".to_string(), 30.0);

        Self { gate_times }
    }

    /// Create a planner with custom gate times
    pub fn with_gate_times(gate_times: HashMap<String, f64>) -> Self {
        Self { gate_times }
    }

    /// Set the execution time for a gate
    pub fn set_gate_time(&mut self, gate_name: &str, time: f64) {
        self.gate_times.insert(gate_name.to_string(), time);
    }

    /// Get the execution time for a gate
    fn get_gate_time(&self, gate_name: &str) -> f64 {
        self.gate_times.get(gate_name).copied().unwrap_or(1.0)
    }

    /// Generate an execution plan for a circuit
    ///
    /// This analyzes the circuit and creates layers of gates that can be
    /// executed in parallel, optimizing for minimum execution time.
    pub fn generate_plan(&self, circuit: &Circuit) -> ExecutionPlan {
        let mut layers = Vec::new();
        let mut current_layer = ExecutionLayer::new();

        let operations: Vec<_> = circuit.operations().collect();

        for (idx, op) in operations.iter().enumerate() {
            let qubits = op.qubits();
            let gate_time = self.get_gate_time(op.gate().name());

            // Try to add to current layer
            if current_layer.can_add(qubits) {
                current_layer.add_gate(idx, qubits, gate_time);
            } else {
                // Start new layer
                if !current_layer.is_empty() {
                    layers.push(current_layer);
                }
                current_layer = ExecutionLayer::new();
                current_layer.add_gate(idx, qubits, gate_time);
            }
        }

        // Add final layer
        if !current_layer.is_empty() {
            layers.push(current_layer);
        }

        let mut plan = ExecutionPlan::new(layers, circuit.len());
        plan.resources = self.estimate_resources(circuit);
        plan
    }

    /// Estimate resource requirements for a circuit
    fn estimate_resources(&self, circuit: &Circuit) -> ResourceRequirements {
        let peak_qubits = circuit.num_qubits();

        // Estimate memory: 2^n complex numbers × 16 bytes
        let peak_memory = if peak_qubits < 30 {
            (1usize << peak_qubits) * 16
        } else {
            usize::MAX // Too large to represent
        };

        let mut two_qubit_gates = 0;
        let measurement_count = 0; // Would need to scan for measurement ops

        for op in circuit.operations() {
            if op.num_qubits() == 2 {
                two_qubit_gates += 1;
            }
        }

        ResourceRequirements {
            peak_qubits,
            peak_memory,
            measurement_count,
            two_qubit_gates,
        }
    }

    /// Generate an optimized plan that reorders gates for better parallelization
    ///
    /// This uses a greedy algorithm to maximize parallelism while preserving
    /// circuit semantics (respecting dependencies).
    pub fn generate_optimized_plan(&self, circuit: &Circuit) -> ExecutionPlan {
        // Build dependency graph
        let deps = self.build_dependency_graph(circuit);

        // Topological sort with parallelization
        let layers = self.schedule_with_dependencies(circuit, &deps);

        let mut plan = ExecutionPlan::new(layers, circuit.len());
        plan.resources = self.estimate_resources(circuit);
        plan
    }

    /// Build dependency graph for circuit gates
    fn build_dependency_graph(&self, circuit: &Circuit) -> HashMap<usize, Vec<usize>> {
        let mut deps: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut last_gate_on_qubit: HashMap<QubitId, usize> = HashMap::new();

        for (idx, op) in circuit.operations().enumerate() {
            let mut gate_deps = Vec::new();

            // Find dependencies: gates on the same qubits
            for qubit in op.qubits() {
                if let Some(&prev_idx) = last_gate_on_qubit.get(qubit) {
                    gate_deps.push(prev_idx);
                }
                last_gate_on_qubit.insert(*qubit, idx);
            }

            deps.insert(idx, gate_deps);
        }

        deps
    }

    /// Schedule gates into layers respecting dependencies
    fn schedule_with_dependencies(
        &self,
        circuit: &Circuit,
        deps: &HashMap<usize, Vec<usize>>,
    ) -> Vec<ExecutionLayer> {
        let operations: Vec<_> = circuit.operations().collect();
        let mut layers = Vec::new();
        let mut scheduled = HashSet::new();
        let mut ready_queue: Vec<usize> = Vec::new();

        // Find gates with no dependencies (ready to execute)
        for (idx, gate_deps) in deps.iter() {
            if gate_deps.is_empty() {
                ready_queue.push(*idx);
            }
        }

        while !ready_queue.is_empty() {
            let mut layer = ExecutionLayer::new();
            let mut used_in_layer = Vec::new();

            // Greedily add gates to current layer
            for &idx in &ready_queue {
                let op = operations[idx];
                let qubits = op.qubits();
                let gate_time = self.get_gate_time(op.gate().name());

                if layer.can_add(qubits) {
                    layer.add_gate(idx, qubits, gate_time);
                    used_in_layer.push(idx);
                }
            }

            // Remove scheduled gates from ready queue
            for &idx in &used_in_layer {
                ready_queue.retain(|&x| x != idx);
                scheduled.insert(idx);
            }

            // Add newly ready gates
            for (idx, gate_deps) in deps.iter() {
                if !scheduled.contains(idx) && !ready_queue.contains(idx) {
                    if gate_deps.iter().all(|&dep| scheduled.contains(&dep)) {
                        ready_queue.push(*idx);
                    }
                }
            }

            if !layer.is_empty() {
                layers.push(layer);
            }
        }

        layers
    }
}

impl Default for ExecutionPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_core::gate::Gate;
    use std::sync::Arc;

    #[derive(Debug)]
    struct MockGate {
        name: String,
    }

    impl Gate for MockGate {
        fn name(&self) -> &str {
            &self.name
        }
        fn num_qubits(&self) -> usize {
            match self.name.as_str() {
                "H" | "X" | "Y" | "Z" => 1,
                "CNOT" | "CZ" => 2,
                _ => 1,
            }
        }
    }

    #[test]
    fn test_execution_layer() {
        let mut layer = ExecutionLayer::new();

        // Add first gate
        layer.add_gate(0, &[QubitId::new(0)], 1.0);
        assert_eq!(layer.len(), 1);

        // Can add gate on different qubit
        assert!(layer.can_add(&[QubitId::new(1)]));

        // Cannot add gate on same qubit
        assert!(!layer.can_add(&[QubitId::new(0)]));
    }

    #[test]
    fn test_simple_plan() {
        let mut circuit = Circuit::new(2);

        let h = Arc::new(MockGate { name: "H".to_string() });
        circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(h.clone(), &[QubitId::new(1)]).unwrap();

        let planner = ExecutionPlanner::new();
        let plan = planner.generate_plan(&circuit);

        // Two gates on different qubits should be in same layer
        assert_eq!(plan.depth, 1);
        assert_eq!(plan.gate_count, 2);
        assert_eq!(plan.parallelism_factor, 2.0);
    }

    #[test]
    fn test_sequential_plan() {
        let mut circuit = Circuit::new(1);

        let h = Arc::new(MockGate { name: "H".to_string() });
        circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();

        let planner = ExecutionPlanner::new();
        let plan = planner.generate_plan(&circuit);

        // Three gates on same qubit should be in separate layers
        assert_eq!(plan.depth, 3);
        assert_eq!(plan.gate_count, 3);
        assert_eq!(plan.parallelism_factor, 1.0);
    }

    #[test]
    fn test_mixed_parallelism() {
        let mut circuit = Circuit::new(3);

        let h = Arc::new(MockGate { name: "H".to_string() });
        let cnot = Arc::new(MockGate { name: "CNOT".to_string() });

        // Layer 1: H on all qubits (parallel)
        circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(h.clone(), &[QubitId::new(1)]).unwrap();
        circuit.add_gate(h.clone(), &[QubitId::new(2)]).unwrap();

        // Layer 2: CNOT (sequential with layer 1)
        circuit.add_gate(cnot, &[QubitId::new(0), QubitId::new(1)]).unwrap();

        let planner = ExecutionPlanner::new();
        let plan = planner.generate_plan(&circuit);

        assert_eq!(plan.depth, 2);
        assert_eq!(plan.gate_count, 4);
    }

    #[test]
    fn test_parallelization_efficiency() {
        let mut circuit = Circuit::new(4);
        let h = Arc::new(MockGate { name: "H".to_string() });

        // Perfect parallelization: all gates in one layer
        for i in 0..4 {
            circuit.add_gate(h.clone(), &[QubitId::new(i)]).unwrap();
        }

        let planner = ExecutionPlanner::new();
        let plan = planner.generate_plan(&circuit);

        // Efficiency should be high (close to 1.0)
        assert!(plan.parallelization_efficiency() > 0.7);
    }

    #[test]
    fn test_resource_estimation() {
        let mut circuit = Circuit::new(5);

        let h = Arc::new(MockGate { name: "H".to_string() });
        let cnot = Arc::new(MockGate { name: "CNOT".to_string() });

        circuit.add_gate(h, &[QubitId::new(0)]).unwrap();
        circuit.add_gate(cnot.clone(), &[QubitId::new(0), QubitId::new(1)]).unwrap();
        circuit.add_gate(cnot, &[QubitId::new(2), QubitId::new(3)]).unwrap();

        let planner = ExecutionPlanner::new();
        let plan = planner.generate_plan(&circuit);

        assert_eq!(plan.resources.peak_qubits, 5);
        assert_eq!(plan.resources.two_qubit_gates, 2);
    }
}
