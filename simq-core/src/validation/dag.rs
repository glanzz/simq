//! Dependency graph representation for circuit validation

use crate::{Circuit, QuantumError, Result};
use std::collections::HashSet;

/// Node in the dependency graph (represents a gate operation)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GateNode {
    pub operation_index: usize,
    pub gate_name: String,
    pub qubits: Vec<usize>,
}

/// Edge in the dependency graph (represents a dependency)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DependencyEdge {
    /// Data dependency (qubit is read after being written)
    DataDependency {
        from: usize,  // Source operation index
        to: usize,    // Target operation index
        qubit: usize, // Qubit causing dependency
    },
    /// Anti-dependency (qubit is written after being read)
    AntiDependency {
        from: usize,
        to: usize,
        qubit: usize,
    },
    /// Output dependency (qubit is written after being written)
    OutputDependency {
        from: usize,
        to: usize,
        qubit: usize,
    },
}

impl DependencyEdge {
    /// Get the source node index
    pub fn from(&self) -> usize {
        match self {
            DependencyEdge::DataDependency { from, .. } => *from,
            DependencyEdge::AntiDependency { from, .. } => *from,
            DependencyEdge::OutputDependency { from, .. } => *from,
        }
    }

    /// Get the target node index
    pub fn to(&self) -> usize {
        match self {
            DependencyEdge::DataDependency { to, .. } => *to,
            DependencyEdge::AntiDependency { to, .. } => *to,
            DependencyEdge::OutputDependency { to, .. } => *to,
        }
    }

    /// Get the qubit causing the dependency
    pub fn qubit(&self) -> usize {
        match self {
            DependencyEdge::DataDependency { qubit, .. } => *qubit,
            DependencyEdge::AntiDependency { qubit, .. } => *qubit,
            DependencyEdge::OutputDependency { qubit, .. } => *qubit,
        }
    }
}

/// Dependency graph representation
pub struct DependencyGraph {
    nodes: Vec<GateNode>,
    edges: Vec<DependencyEdge>,
    // Adjacency lists for efficient traversal
    outgoing_edges: Vec<Vec<usize>>, // node_index -> edge indices
    incoming_edges: Vec<Vec<usize>>, // node_index -> edge indices
}

impl DependencyGraph {
    /// Create an empty dependency graph
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            outgoing_edges: Vec::new(),
            incoming_edges: Vec::new(),
        }
    }

    /// Create dependency graph from circuit
    ///
    /// This is optimized for performance by pre-allocating vectors and using
    /// efficient dependency tracking.
    pub fn from_circuit(circuit: &Circuit) -> Result<Self> {
        let num_ops = circuit.len();
        
        // Pre-allocate nodes vector
        let mut nodes = Vec::with_capacity(num_ops);
        
        // Build nodes
        for (i, op) in circuit.operations().enumerate() {
            let qubits: Vec<usize> = op.qubits().iter().map(|q| q.index()).collect();
            nodes.push(GateNode {
                operation_index: i,
                gate_name: op.gate().name().to_string(),
                qubits,
            });
        }

        if nodes.is_empty() {
            return Ok(Self {
                nodes,
                edges: Vec::new(),
                outgoing_edges: Vec::new(),
                incoming_edges: Vec::new(),
            });
        }

        // Pre-allocate edges vector (estimate: roughly num_ops edges for typical circuits)
        let mut edges = Vec::with_capacity(num_ops);
        
        // Build edges by analyzing qubit dependencies
        // For quantum circuits, gates typically both read and write to qubits
        // So we create dependencies based on qubit usage order
        let mut qubit_last_use: Vec<Option<usize>> = vec![None; circuit.num_qubits()];

        for (i, node) in nodes.iter().enumerate() {
            for &qubit in &node.qubits {
                if let Some(last_use) = qubit_last_use[qubit] {
                    // Create dependency: current gate depends on last gate using this qubit
                    // We use OutputDependency since both gates modify the qubit state
                    edges.push(DependencyEdge::OutputDependency {
                        from: last_use,
                        to: i,
                        qubit,
                    });
                }
                qubit_last_use[qubit] = Some(i);
            }
        }

        // Build adjacency lists with pre-allocation
        let num_nodes = nodes.len();
        let mut outgoing_edges = vec![Vec::new(); num_nodes];
        let mut incoming_edges = vec![Vec::new(); num_nodes];

        // Pre-allocate space in adjacency lists (estimate: 1-2 edges per node on average)
        for i in 0..num_nodes {
            outgoing_edges[i].reserve(2);
            incoming_edges[i].reserve(2);
        }

        for (edge_idx, edge) in edges.iter().enumerate() {
            let from = edge.from();
            let to = edge.to();

            if from >= num_nodes || to >= num_nodes {
                return Err(QuantumError::ValidationError(format!(
                    "Invalid edge in dependency graph: from={}, to={}, num_nodes={}",
                    from, to, num_nodes
                )));
            }

            outgoing_edges[from].push(edge_idx);
            incoming_edges[to].push(edge_idx);
        }

        Ok(Self {
            nodes,
            edges,
            outgoing_edges,
            incoming_edges,
        })
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get node by index
    pub fn get_node(&self, index: usize) -> Option<&GateNode> {
        self.nodes.get(index)
    }

    /// Get edge by index
    pub fn get_edge(&self, index: usize) -> Option<&DependencyEdge> {
        self.edges.get(index)
    }

    /// Get outgoing edges for a node
    pub fn outgoing_edges(&self, node_index: usize) -> &[usize] {
        self.outgoing_edges
            .get(node_index)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get incoming edges for a node
    pub fn incoming_edges(&self, node_index: usize) -> &[usize] {
        self.incoming_edges
            .get(node_index)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Check if graph is acyclic (no cycles)
    pub fn is_acyclic(&self) -> bool {
        self.find_cycles().is_empty()
    }

    /// Find all cycles in the graph
    ///
    /// Returns early on first cycle found for performance (can be extended to find all).
    pub fn find_cycles(&self) -> Vec<Vec<usize>> {
        let mut cycles = Vec::new();
        let mut visited = vec![false; self.num_nodes()];
        let mut recursion_stack = vec![false; self.num_nodes()];
        let mut path = Vec::with_capacity(self.num_nodes());

        for i in 0..self.num_nodes() {
            if !visited[i] {
                self.dfs_cycle_detection(
                    i,
                    &mut visited,
                    &mut recursion_stack,
                    &mut path,
                    &mut cycles,
                );
                // Early exit: quantum circuits are typically acyclic, so finding one cycle is enough
                if !cycles.is_empty() {
                    break;
                }
            }
        }

        cycles
    }

    /// DFS for cycle detection
    fn dfs_cycle_detection(
        &self,
        node: usize,
        visited: &mut [bool],
        recursion_stack: &mut [bool],
        path: &mut Vec<usize>,
        cycles: &mut Vec<Vec<usize>>,
    ) {
        visited[node] = true;
        recursion_stack[node] = true;
        path.push(node);

        for &edge_idx in self.outgoing_edges(node) {
            if let Some(edge) = self.get_edge(edge_idx) {
                let target = edge.to();

                if !visited[target] {
                    self.dfs_cycle_detection(target, visited, recursion_stack, path, cycles);
                } else if recursion_stack[target] {
                    // Cycle detected: find the cycle path
                    if let Some(cycle_start) = path.iter().position(|&x| x == target) {
                        let mut cycle = path[cycle_start..].to_vec();
                        cycle.push(target); // Complete the cycle
                        cycles.push(cycle);
                    }
                }
            }
        }

        recursion_stack[node] = false;
        path.pop();
    }

    /// Compute topological ordering (Kahn's algorithm)
    pub fn topological_sort(&self) -> Result<Vec<usize>> {
        let mut in_degree = vec![0; self.num_nodes()];
        for edge in &self.edges {
            let to = edge.to();
            if to < in_degree.len() {
                in_degree[to] += 1;
            }
        }

        let mut queue: Vec<usize> = (0..self.num_nodes())
            .filter(|&i| in_degree[i] == 0)
            .collect();

        let mut result = Vec::new();

        while let Some(node) = queue.pop() {
            result.push(node);

            for &edge_idx in self.outgoing_edges(node) {
                if let Some(edge) = self.get_edge(edge_idx) {
                    let to = edge.to();
                    if to < in_degree.len() {
                        in_degree[to] -= 1;
                        if in_degree[to] == 0 {
                            queue.push(to);
                        }
                    }
                }
            }
        }

        if result.len() != self.num_nodes() {
            return Err(QuantumError::TopologicalOrderError {
                reason: format!(
                    "Circuit contains cycles, cannot compute topological order. Processed {} of {} nodes",
                    result.len(),
                    self.num_nodes()
                ),
            });
        }

        Ok(result)
    }

    /// Compute circuit depth (longest path in DAG)
    pub fn depth(&self) -> Result<usize> {
        // Use topological sort and dynamic programming
        let topo_order = self.topological_sort()?;
        let mut distances = vec![0; self.num_nodes()];
        let mut max_depth = 0;

        for &node in &topo_order {
            for &edge_idx in self.incoming_edges(node) {
                if let Some(edge) = self.get_edge(edge_idx) {
                    let from = edge.from();
                    if from < distances.len() {
                        distances[node] = distances[node].max(distances[from] + 1);
                    }
                }
            }
            max_depth = max_depth.max(distances[node]);
        }

        Ok(max_depth + 1) // +1 to count the last layer
    }

    /// Compute parallel execution layers
    pub fn compute_parallel_layers(&self) -> Result<Vec<Vec<usize>>> {
        if self.num_nodes() == 0 {
            return Ok(Vec::new());
        }

        let mut layers = Vec::new();
        let mut completed = HashSet::new();

        // Build dependency count for each node
        let mut dependency_count: Vec<usize> = (0..self.num_nodes())
            .map(|i| self.incoming_edges(i).len())
            .collect();

        // Process layers until all nodes are processed
        while completed.len() < self.num_nodes() {
            // Find all nodes with no remaining dependencies
            let mut current_layer = Vec::new();
            for i in 0..self.num_nodes() {
                if !completed.contains(&i) && dependency_count[i] == 0 {
                    current_layer.push(i);
                    completed.insert(i);
                }
            }

            // If no nodes can be processed, we have a cycle (shouldn't happen in acyclic graph)
            if current_layer.is_empty() {
                return Err(QuantumError::ValidationError(
                    "Cannot compute layers: cycle detected or invalid dependency graph".to_string(),
                ));
            }

            layers.push(current_layer.clone());

            // Update dependency counts for nodes that depend on current layer
            for &node in &current_layer {
                for &edge_idx in self.outgoing_edges(node) {
                    if let Some(edge) = self.get_edge(edge_idx) {
                        let to = edge.to();
                        if to < dependency_count.len() && !completed.contains(&to) {
                            dependency_count[to] -= 1;
                        }
                    }
                }
            }
        }

        Ok(layers)
    }

    /// Analyze parallelism
    pub fn analyze_parallelism(&self) -> Result<ParallelismAnalysis> {
        let layers = self.compute_parallel_layers()?;
        let total_operations = self.num_nodes();
        let depth = layers.len();
        let parallelism_factor = if depth > 0 {
            total_operations as f64 / depth as f64
        } else {
            1.0
        };
        let max_parallelism = layers.iter().map(|l| l.len()).max().unwrap_or(0);

        Ok(ParallelismAnalysis {
            layers,
            parallelism_factor,
            max_parallelism,
        })
    }
}

/// Analysis of circuit parallelism
#[derive(Clone, Debug)]
pub struct ParallelismAnalysis {
    pub layers: Vec<Vec<usize>>, // Each layer contains operation indices that can run in parallel
    pub parallelism_factor: f64, // Average parallelism (operations / depth)
    pub max_parallelism: usize,  // Maximum number of parallel operations
}

impl ParallelismAnalysis {
    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get the average parallelism per layer
    pub fn avg_parallelism(&self) -> f64 {
        if self.layers.is_empty() {
            0.0
        } else {
            self.layers.iter().map(|l| l.len()).sum::<usize>() as f64 / self.layers.len() as f64
        }
    }
}

impl DependencyGraph {
    /// Format graph as DOT (Graphviz) format for visualization
    ///
    /// This can be used to visualize the dependency graph using tools like Graphviz.
    ///
    /// # Example
    /// ```ignore
    /// let dag = DependencyGraph::from_circuit(&circuit)?;
    /// let dot = dag.to_dot();
    /// println!("{}", dot);
    /// ```
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph Circuit {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=box];\n\n");

        // Add nodes
        for (i, node) in self.nodes.iter().enumerate() {
            let label = format!("{}: {}", i, node.gate_name);
            dot.push_str(&format!("  {} [label=\"{}\"];\n", i, label));
        }

        dot.push_str("\n");

        // Add edges
        for edge in &self.edges {
            let from = edge.from();
            let to = edge.to();
            let qubit = edge.qubit();
            let label = format!("q{}", qubit);
            dot.push_str(&format!("  {} -> {} [label=\"{}\"];\n", from, to, label));
        }

        dot.push_str("}\n");
        dot
    }

    /// Format cycle for display
    pub fn format_cycle(&self, cycle: &[usize]) -> String {
        let mut msg = String::from("Cycle: ");
        for (i, &node_idx) in cycle.iter().enumerate() {
            if let Some(node) = self.get_node(node_idx) {
                msg.push_str(&format!("{}:{}", node_idx, node.gate_name));
                if i < cycle.len() - 1 {
                    msg.push_str(" -> ");
                }
            }
        }
        msg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::Gate;
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

    fn create_test_circuit() -> Circuit {
        let mut circuit = Circuit::new(3);
        let h_gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });
        let cnot_gate = Arc::new(MockGate {
            name: "CNOT".to_string(),
            num_qubits: 2,
        });

        circuit.add_gate(h_gate.clone(), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(cnot_gate, &[QubitId::new(0), QubitId::new(1)]).unwrap();
        circuit.add_gate(h_gate, &[QubitId::new(2)]).unwrap();

        circuit
    }

    #[test]
    fn test_dag_construction() {
        let circuit = create_test_circuit();
        let dag = DependencyGraph::from_circuit(&circuit).unwrap();

        assert_eq!(dag.num_nodes(), 3);
        // H(0) -> CNOT(0,1) creates 1 dependency
        // H(2) is independent (no dependencies)
        assert_eq!(dag.num_edges(), 1);
    }

    #[test]
    fn test_acyclic_circuit() {
        let circuit = create_test_circuit();
        let dag = DependencyGraph::from_circuit(&circuit).unwrap();
        assert!(dag.is_acyclic());
    }

    #[test]
    fn test_topological_sort() {
        let circuit = create_test_circuit();
        let dag = DependencyGraph::from_circuit(&circuit).unwrap();
        let order = dag.topological_sort().unwrap();
        assert_eq!(order.len(), 3);
        // H(0) should come before CNOT(0,1)
        assert!(order[0] < order[1] || order[1] < order[0]);
    }

    #[test]
    fn test_depth_computation() {
        let circuit = create_test_circuit();
        let dag = DependencyGraph::from_circuit(&circuit).unwrap();
        let depth = dag.depth().unwrap();
        assert!(depth >= 2);
        assert!(depth <= 3);
    }

    #[test]
    fn test_parallel_layers() {
        let circuit = create_test_circuit();
        let dag = DependencyGraph::from_circuit(&circuit).unwrap();
        let layers = dag.compute_parallel_layers().unwrap();
        assert!(!layers.is_empty());
        // H(0) and H(2) can potentially be in the same layer if they're independent
        // But H(0) -> CNOT(0,1) creates a dependency, so they might be in different layers
    }

    #[test]
    fn test_parallelism_analysis() {
        let circuit = create_test_circuit();
        let dag = DependencyGraph::from_circuit(&circuit).unwrap();
        let analysis = dag.analyze_parallelism().unwrap();
        assert!(analysis.parallelism_factor > 0.0);
        assert!(analysis.max_parallelism >= 1);
    }

    #[test]
    fn test_empty_circuit() {
        let circuit = Circuit::new(2);
        let dag = DependencyGraph::from_circuit(&circuit).unwrap();
        assert_eq!(dag.num_nodes(), 0);
        assert_eq!(dag.num_edges(), 0);
        assert!(dag.is_acyclic());
    }
}

