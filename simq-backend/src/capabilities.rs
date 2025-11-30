//! Backend capabilities and constraints

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Capabilities of a quantum backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// Maximum number of qubits supported
    pub max_qubits: usize,

    /// Maximum circuit depth supported (None = unlimited)
    pub max_circuit_depth: Option<usize>,

    /// Maximum number of shots per job (None = unlimited)
    pub max_shots: Option<usize>,

    /// Supported gate set
    pub supported_gates: GateSet,

    /// Native gate set (gates that don't need decomposition)
    pub native_gates: GateSet,

    /// Qubit connectivity graph (None = all-to-all)
    pub connectivity: Option<ConnectivityGraph>,

    /// Whether the backend supports mid-circuit measurement
    pub supports_mid_circuit_measurement: bool,

    /// Whether the backend supports conditional operations
    pub supports_conditional: bool,

    /// Whether the backend supports reset operations
    pub supports_reset: bool,

    /// Whether the backend supports parametric circuits
    pub supports_parametric: bool,

    /// Execution cost per shot (in credits, if applicable)
    pub cost_per_shot: Option<f64>,

    /// Average queue time in seconds (if applicable)
    pub average_queue_time: Option<u64>,

    /// Backend-specific metadata
    pub metadata: HashMap<String, String>,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            max_qubits: 64,
            max_circuit_depth: None,
            max_shots: None,
            supported_gates: GateSet::universal(),
            native_gates: GateSet::universal(),
            connectivity: None, // All-to-all
            supports_mid_circuit_measurement: true,
            supports_conditional: true,
            supports_reset: true,
            supports_parametric: true,
            cost_per_shot: None,
            average_queue_time: None,
            metadata: HashMap::new(),
        }
    }
}

impl BackendCapabilities {
    /// Create capabilities for a perfect simulator
    pub fn simulator() -> Self {
        Self::default()
    }

    /// Create capabilities for IBM Quantum hardware
    pub fn ibm_quantum(num_qubits: usize, connectivity: ConnectivityGraph) -> Self {
        let mut native_gates = GateSet::new();
        native_gates.insert("RZ".to_string());
        native_gates.insert("SX".to_string());
        native_gates.insert("X".to_string());
        native_gates.insert("CNOT".to_string());
        native_gates.insert("Measure".to_string());

        Self {
            max_qubits: num_qubits,
            max_circuit_depth: Some(10000),
            max_shots: Some(100000),
            supported_gates: GateSet::universal(),
            native_gates,
            connectivity: Some(connectivity),
            supports_mid_circuit_measurement: true,
            supports_conditional: true,
            supports_reset: true,
            supports_parametric: false,
            cost_per_shot: Some(0.00003),  // Example cost
            average_queue_time: Some(120), // 2 minutes
            metadata: HashMap::new(),
        }
    }

    /// Check if a gate type is supported
    pub fn supports_gate(&self, gate_type: String) -> bool {
        self.supported_gates.contains(&gate_type)
    }

    /// Check if a gate type is native (doesn't need decomposition)
    pub fn is_native_gate(&self, gate_type: String) -> bool {
        self.native_gates.contains(&gate_type)
    }

    /// Check if two qubits are connected
    pub fn are_qubits_connected(&self, q1: usize, q2: usize) -> bool {
        match &self.connectivity {
            None => true, // All-to-all connectivity
            Some(graph) => graph.are_connected(q1, q2),
        }
    }
}

/// Set of supported gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateSet {
    gates: HashSet<String>,
}

impl GateSet {
    /// Create an empty gate set
    pub fn new() -> Self {
        Self {
            gates: HashSet::new(),
        }
    }

    /// Create a universal gate set (all standard gates)
    pub fn universal() -> Self {
        let mut set = Self::new();
        // Single-qubit gates
        set.insert("I".to_string());
        set.insert("X".to_string());
        set.insert("Y".to_string());
        set.insert("Z".to_string());
        set.insert("H".to_string());
        set.insert("S".to_string());
        set.insert("T".to_string());
        set.insert("SX".to_string());
        set.insert("RX".to_string());
        set.insert("RY".to_string());
        set.insert("RZ".to_string());
        set.insert("Phase".to_string());
        set.insert("U".to_string());

        // Two-qubit gates
        set.insert("CNOT".to_string());
        set.insert("CZ".to_string());
        set.insert("CY".to_string());
        set.insert("SWAP".to_string());
        set.insert("ISWAP".to_string());
        set.insert("CRX".to_string());
        set.insert("CRY".to_string());
        set.insert("CRZ".to_string());
        set.insert("CPhase".to_string());

        // Three-qubit gates
        set.insert("Toffoli".to_string());
        set.insert("Fredkin".to_string());

        // Measurement
        set.insert("Measure".to_string());

        set
    }

    /// Insert a gate type
    pub fn insert(&mut self, gate_type: String) {
        self.gates.insert(gate_type);
    }

    /// Check if a gate type is in the set
    pub fn contains(&self, gate_type: &String) -> bool {
        self.gates.contains(gate_type)
    }

    /// Get all gate types in the set
    pub fn gates(&self) -> &HashSet<String> {
        &self.gates
    }

    /// Get the number of gates in the set
    pub fn len(&self) -> usize {
        self.gates.len()
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }
}

impl Default for GateSet {
    fn default() -> Self {
        Self::universal()
    }
}

/// Qubit connectivity graph for hardware with limited connectivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityGraph {
    /// Number of physical qubits
    num_qubits: usize,

    /// Adjacency list: qubit -> connected qubits
    edges: HashMap<usize, HashSet<usize>>,

    /// Whether the graph is directed (for asymmetric gates)
    directed: bool,
}

impl ConnectivityGraph {
    /// Create a new connectivity graph
    pub fn new(num_qubits: usize, directed: bool) -> Self {
        Self {
            num_qubits,
            edges: HashMap::new(),
            directed,
        }
    }

    /// Create an all-to-all connectivity graph
    pub fn all_to_all(num_qubits: usize) -> Self {
        let mut graph = Self::new(num_qubits, false);
        for i in 0..num_qubits {
            for j in 0..num_qubits {
                if i != j {
                    graph.add_edge(i, j);
                }
            }
        }
        graph
    }

    /// Create a linear chain connectivity (nearest-neighbor only)
    pub fn linear_chain(num_qubits: usize) -> Self {
        let mut graph = Self::new(num_qubits, false);
        for i in 0..num_qubits - 1 {
            graph.add_edge(i, i + 1);
        }
        graph
    }

    /// Create a grid connectivity
    pub fn grid(rows: usize, cols: usize) -> Self {
        let num_qubits = rows * cols;
        let mut graph = Self::new(num_qubits, false);

        for row in 0..rows {
            for col in 0..cols {
                let qubit = row * cols + col;

                // Connect to right neighbor
                if col < cols - 1 {
                    graph.add_edge(qubit, qubit + 1);
                }

                // Connect to bottom neighbor
                if row < rows - 1 {
                    graph.add_edge(qubit, qubit + cols);
                }
            }
        }

        graph
    }

    /// Add an edge between two qubits
    pub fn add_edge(&mut self, q1: usize, q2: usize) {
        self.edges.entry(q1).or_insert_with(HashSet::new).insert(q2);

        if !self.directed {
            self.edges.entry(q2).or_insert_with(HashSet::new).insert(q1);
        }
    }

    /// Check if two qubits are connected
    pub fn are_connected(&self, q1: usize, q2: usize) -> bool {
        self.edges
            .get(&q1)
            .map(|neighbors| neighbors.contains(&q2))
            .unwrap_or(false)
    }

    /// Get all neighbors of a qubit
    pub fn neighbors(&self, qubit: usize) -> Option<&HashSet<usize>> {
        self.edges.get(&qubit)
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Check if the graph is directed
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// Get the degree (number of connections) of a qubit
    pub fn degree(&self, qubit: usize) -> usize {
        self.edges.get(&qubit).map(|n| n.len()).unwrap_or(0)
    }

    /// Find the shortest path between two qubits (for SWAP chain calculation)
    pub fn shortest_path(&self, start: usize, end: usize) -> Option<Vec<usize>> {
        if start == end {
            return Some(vec![start]);
        }

        let mut visited = HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut parent = HashMap::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(current) = queue.pop_front() {
            if current == end {
                // Reconstruct path
                let mut path = vec![end];
                let mut node = end;
                while let Some(&prev) = parent.get(&node) {
                    path.push(prev);
                    node = prev;
                }
                path.reverse();
                return Some(path);
            }

            if let Some(neighbors) = self.neighbors(current) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        parent.insert(neighbor, current);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        None // No path found
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_to_all_connectivity() {
        let graph = ConnectivityGraph::all_to_all(5);
        assert!(graph.are_connected(0, 1));
        assert!(graph.are_connected(0, 4));
        assert!(graph.are_connected(2, 3));
    }

    #[test]
    fn test_linear_chain() {
        let graph = ConnectivityGraph::linear_chain(5);
        assert!(graph.are_connected(0, 1));
        assert!(graph.are_connected(1, 2));
        assert!(!graph.are_connected(0, 2));
        assert!(!graph.are_connected(0, 4));
    }

    #[test]
    fn test_grid_connectivity() {
        let graph = ConnectivityGraph::grid(3, 3);
        // 0-1-2
        // | | |
        // 3-4-5
        // | | |
        // 6-7-8

        assert!(graph.are_connected(0, 1));
        assert!(graph.are_connected(0, 3));
        assert!(!graph.are_connected(0, 2));
        assert!(!graph.are_connected(0, 4));
    }

    #[test]
    fn test_shortest_path() {
        let graph = ConnectivityGraph::linear_chain(5);
        let path = graph.shortest_path(0, 4).unwrap();
        assert_eq!(path, vec![0, 1, 2, 3, 4]);
    }
}
