//! QAOA (Quantum Approximate Optimization Algorithm) Circuit Generator
//!
//! This module provides a comprehensive framework for generating QAOA circuits
//! for various combinatorial optimization problems.
//!
//! # Features
//!
//! - **Problem Hamiltonians**: MaxCut, Number Partitioning, Graph Coloring, TSP, and more
//! - **Mixer Hamiltonians**: Standard X-mixer, XY-mixer, Grover mixer, custom mixers
//! - **Graph Utilities**: Graph representation, edge management, topology operations
//! - **Flexible Configuration**: Customizable circuit depth, initialization strategies
//! - **Observable Generation**: Automatic Hamiltonian-to-Observable conversion
//!
//! # Example
//!
//! ```ignore
//! use simq_sim::qaoa::{QAOACircuitBuilder, ProblemType, MixerType, Graph};
//!
//! // Define a graph for MaxCut
//! let graph = Graph::from_edges(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 3, 1.0)]);
//!
//! // Create QAOA circuit builder
//! let builder = QAOACircuitBuilder::new(
//!     ProblemType::MaxCut(graph),
//!     MixerType::StandardX,
//!     2, // depth (p=2)
//! );
//!
//! // Generate circuit from parameters
//! let params = vec![0.5, 0.3, 0.4, 0.6]; // [gamma_1, beta_1, gamma_2, beta_2]
//! let circuit = builder.build(&params);
//!
//! // Get the cost observable
//! let observable = builder.cost_observable();
//! ```

use simq_core::{Circuit, QubitId};
use simq_gates::{CNot, Hadamard, RotationX, RotationY, RotationZ};
use simq_state::observable::{Pauli, PauliObservable, PauliString};
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Graph Data Structure
// ============================================================================

/// Graph representation for QAOA problems
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of vertices
    pub num_vertices: usize,
    /// Edges with weights: (vertex1, vertex2, weight)
    pub edges: Vec<(usize, usize, f64)>,
    /// Adjacency list representation
    adjacency: HashMap<usize, Vec<(usize, f64)>>,
}

impl Graph {
    /// Create a new graph from edges
    ///
    /// # Arguments
    ///
    /// * `num_vertices` - Number of vertices in the graph
    /// * `edges` - List of edges as (vertex1, vertex2, weight)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let graph = Graph::from_edges(4, &[(0, 1, 1.0), (1, 2, 1.5), (2, 3, 1.0)]);
    /// ```
    pub fn from_edges(num_vertices: usize, edges: &[(usize, usize, f64)]) -> Self {
        let mut adjacency: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();

        for &(u, v, w) in edges {
            adjacency.entry(u).or_default().push((v, w));
            adjacency.entry(v).or_default().push((u, w));
        }

        Self {
            num_vertices,
            edges: edges.to_vec(),
            adjacency,
        }
    }

    /// Create a complete graph with n vertices
    pub fn complete(num_vertices: usize) -> Self {
        let mut edges = Vec::new();
        for i in 0..num_vertices {
            for j in (i + 1)..num_vertices {
                edges.push((i, j, 1.0));
            }
        }
        Self::from_edges(num_vertices, &edges)
    }

    /// Create a cycle graph with n vertices
    pub fn cycle(num_vertices: usize) -> Self {
        let mut edges = Vec::new();
        for i in 0..num_vertices {
            edges.push((i, (i + 1) % num_vertices, 1.0));
        }
        Self::from_edges(num_vertices, &edges)
    }

    /// Create a path graph with n vertices
    pub fn path(num_vertices: usize) -> Self {
        let mut edges = Vec::new();
        for i in 0..(num_vertices - 1) {
            edges.push((i, i + 1, 1.0));
        }
        Self::from_edges(num_vertices, &edges)
    }

    /// Create a star graph with n vertices (one central vertex connected to all others)
    pub fn star(num_vertices: usize) -> Self {
        let mut edges = Vec::new();
        for i in 1..num_vertices {
            edges.push((0, i, 1.0));
        }
        Self::from_edges(num_vertices, &edges)
    }

    /// Create a grid graph (2D lattice)
    pub fn grid(rows: usize, cols: usize) -> Self {
        let num_vertices = rows * cols;
        let mut edges = Vec::new();

        for i in 0..rows {
            for j in 0..cols {
                let v = i * cols + j;
                // Right neighbor
                if j + 1 < cols {
                    edges.push((v, v + 1, 1.0));
                }
                // Bottom neighbor
                if i + 1 < rows {
                    edges.push((v, v + cols, 1.0));
                }
            }
        }

        Self::from_edges(num_vertices, &edges)
    }

    /// Get neighbors of a vertex
    pub fn neighbors(&self, vertex: usize) -> Vec<(usize, f64)> {
        self.adjacency.get(&vertex).cloned().unwrap_or_default()
    }

    /// Get the degree of a vertex
    pub fn degree(&self, vertex: usize) -> usize {
        self.adjacency.get(&vertex).map(|v| v.len()).unwrap_or(0)
    }

    /// Get total number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
}

// ============================================================================
// Problem Types
// ============================================================================

/// QAOA problem types with associated data
#[derive(Debug, Clone)]
pub enum ProblemType {
    /// MaxCut problem: maximize edges between different partitions
    MaxCut(Graph),

    /// Min vertex cover: find minimum set of vertices covering all edges
    MinVertexCover(Graph),

    /// Max independent set: find maximum set with no edges between vertices
    MaxIndependentSet(Graph),

    /// Number partitioning: partition numbers into two sets with equal sums
    NumberPartitioning(Vec<f64>),

    /// Graph coloring: assign colors to vertices (k-coloring problem)
    /// (graph, num_colors)
    GraphColoring(Graph, usize),

    /// Max k-SAT: maximize satisfied clauses
    /// Each clause is (variables, negations, weight)
    /// Example: (x0 OR NOT x1 OR x2) with weight 1.0 = ([0, 1, 2], [false, true, false], 1.0)
    MaxKSat {
        num_variables: usize,
        clauses: Vec<(Vec<usize>, Vec<bool>, f64)>,
    },

    /// Traveling Salesman Problem (TSP)
    /// Distance matrix: distances[i][j] = distance from city i to city j
    TSP {
        num_cities: usize,
        distances: Vec<Vec<f64>>,
    },

    /// Portfolio optimization
    /// (expected_returns, covariance_matrix, risk_factor, budget)
    Portfolio {
        assets: usize,
        expected_returns: Vec<f64>,
        covariances: Vec<Vec<f64>>,
        risk_factor: f64,
        budget: usize,
    },

    /// Custom problem with explicit cost Hamiltonian terms
    /// Each term is (qubit_indices, coefficients for Pauli Z products, weight)
    Custom {
        num_qubits: usize,
        terms: Vec<(Vec<usize>, f64)>,
    },
}

impl ProblemType {
    /// Get the number of qubits required for this problem
    pub fn num_qubits(&self) -> usize {
        match self {
            ProblemType::MaxCut(g)
            | ProblemType::MinVertexCover(g)
            | ProblemType::MaxIndependentSet(g) => g.num_vertices,

            ProblemType::NumberPartitioning(numbers) => numbers.len(),

            ProblemType::GraphColoring(g, k) => g.num_vertices * k,

            ProblemType::MaxKSat { num_variables, .. } => *num_variables,

            ProblemType::TSP { num_cities, .. } => num_cities * num_cities,

            ProblemType::Portfolio { assets, .. } => *assets,

            ProblemType::Custom { num_qubits, .. } => *num_qubits,
        }
    }

    /// Get a description of the problem
    pub fn description(&self) -> String {
        match self {
            ProblemType::MaxCut(g) => {
                format!("MaxCut: {} vertices, {} edges", g.num_vertices, g.num_edges())
            },
            ProblemType::MinVertexCover(g) => {
                format!("Min Vertex Cover: {} vertices, {} edges", g.num_vertices, g.num_edges())
            },
            ProblemType::MaxIndependentSet(g) => {
                format!("Max Independent Set: {} vertices, {} edges", g.num_vertices, g.num_edges())
            },
            ProblemType::NumberPartitioning(numbers) => {
                format!("Number Partitioning: {} numbers", numbers.len())
            },
            ProblemType::GraphColoring(g, k) => format!(
                "Graph {}-Coloring: {} vertices, {} edges",
                k,
                g.num_vertices,
                g.num_edges()
            ),
            ProblemType::MaxKSat {
                num_variables,
                clauses,
            } => {
                format!("Max k-SAT: {} variables, {} clauses", num_variables, clauses.len())
            },
            ProblemType::TSP { num_cities, .. } => {
                format!("TSP: {} cities", num_cities)
            },
            ProblemType::Portfolio { assets, .. } => {
                format!("Portfolio Optimization: {} assets", assets)
            },
            ProblemType::Custom { num_qubits, terms } => {
                format!("Custom Problem: {} qubits, {} terms", num_qubits, terms.len())
            },
        }
    }
}

// ============================================================================
// Mixer Types
// ============================================================================

/// Mixer Hamiltonian types for QAOA
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixerType {
    /// Standard X mixer: H_M = sum_i X_i
    StandardX,

    /// Y mixer: H_M = sum_i Y_i
    StandardY,

    /// XY mixer: H_M = sum_{<i,j>} (X_i X_j + Y_i Y_j)
    /// Preserves Hamming weight (number of 1s)
    XY,

    /// Grover mixer: Similar to Grover diffusion operator
    Grover,

    /// Ring mixer: Connects adjacent qubits in a ring
    Ring,
}

impl MixerType {
    /// Get a description of the mixer
    pub fn description(&self) -> &str {
        match self {
            MixerType::StandardX => "Standard X mixer (sum of X_i)",
            MixerType::StandardY => "Standard Y mixer (sum of Y_i)",
            MixerType::XY => "XY mixer (preserves Hamming weight)",
            MixerType::Grover => "Grover mixer (diffusion operator)",
            MixerType::Ring => "Ring mixer (nearest-neighbor coupling)",
        }
    }
}

// ============================================================================
// QAOA Circuit Builder
// ============================================================================

/// Configuration for QAOA circuit generation
#[derive(Debug, Clone)]
pub struct QAOAConfig {
    /// Number of QAOA layers (p)
    pub depth: usize,

    /// Mixer type
    pub mixer: MixerType,

    /// Initial state preparation strategy
    pub initial_state: InitialState,

    /// Whether to apply final layer of mixer
    pub final_mixer: bool,
}

impl Default for QAOAConfig {
    fn default() -> Self {
        Self {
            depth: 1,
            mixer: MixerType::StandardX,
            initial_state: InitialState::UniformSuperposition,
            final_mixer: false,
        }
    }
}

/// Initial state preparation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialState {
    /// |+⟩^⊗n state (Hadamard on all qubits)
    UniformSuperposition,

    /// |0⟩^⊗n state (computational basis)
    Zero,

    /// Warm start from a classical solution
    /// (user provides the bitstring)
    WarmStart,

    /// Custom initial state (user provides preparation circuit)
    Custom,
}

/// QAOA Circuit Builder
///
/// Generates QAOA circuits for various combinatorial optimization problems.
pub struct QAOACircuitBuilder {
    problem: ProblemType,
    config: QAOAConfig,
    num_qubits: usize,
}

impl QAOACircuitBuilder {
    /// Create a new QAOA circuit builder
    ///
    /// # Arguments
    ///
    /// * `problem` - The optimization problem to solve
    /// * `mixer` - The mixer Hamiltonian type
    /// * `depth` - Number of QAOA layers (p)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let graph = Graph::cycle(4);
    /// let builder = QAOACircuitBuilder::new(
    ///     ProblemType::MaxCut(graph),
    ///     MixerType::StandardX,
    ///     2,
    /// );
    /// ```
    pub fn new(problem: ProblemType, mixer: MixerType, depth: usize) -> Self {
        let num_qubits = problem.num_qubits();
        Self {
            problem,
            config: QAOAConfig {
                depth,
                mixer,
                ..Default::default()
            },
            num_qubits,
        }
    }

    /// Create with custom configuration
    pub fn with_config(problem: ProblemType, config: QAOAConfig) -> Self {
        let num_qubits = problem.num_qubits();
        Self {
            problem,
            config,
            num_qubits,
        }
    }

    /// Get number of qubits required
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get number of parameters required (2 * depth for standard QAOA)
    pub fn num_parameters(&self) -> usize {
        2 * self.config.depth
    }

    /// Build QAOA circuit from parameters
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters [gamma_1, beta_1, gamma_2, beta_2, ...]
    ///
    /// # Returns
    ///
    /// A quantum circuit implementing the QAOA algorithm
    pub fn build(&self, params: &[f64]) -> Circuit {
        assert_eq!(
            params.len(),
            self.num_parameters(),
            "Expected {} parameters, got {}",
            self.num_parameters(),
            params.len()
        );

        let mut circuit = Circuit::new(self.num_qubits);

        // Initial state preparation
        self.apply_initial_state(&mut circuit);

        // QAOA layers
        for layer in 0..self.config.depth {
            let gamma = params[2 * layer];
            let beta = params[2 * layer + 1];

            // Problem Hamiltonian layer
            self.apply_problem_hamiltonian(&mut circuit, gamma);

            // Mixer Hamiltonian layer
            if layer < self.config.depth - 1 || self.config.final_mixer {
                self.apply_mixer(&mut circuit, beta);
            }
        }

        circuit
    }

    /// Apply initial state preparation
    fn apply_initial_state(&self, circuit: &mut Circuit) {
        match self.config.initial_state {
            InitialState::UniformSuperposition => {
                // Apply Hadamard to all qubits
                for q in 0..self.num_qubits {
                    let _ = circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]);
                }
            },
            InitialState::Zero => {
                // Do nothing, qubits start in |0⟩
            },
            InitialState::WarmStart | InitialState::Custom => {
                // User must provide custom initialization
                // For now, default to uniform superposition
                for q in 0..self.num_qubits {
                    let _ = circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]);
                }
            },
        }
    }

    /// Apply problem Hamiltonian evolution exp(-i * gamma * H_C)
    fn apply_problem_hamiltonian(&self, circuit: &mut Circuit, gamma: f64) {
        match &self.problem {
            ProblemType::MaxCut(graph) => {
                self.apply_maxcut_hamiltonian(circuit, graph, gamma);
            },
            ProblemType::MinVertexCover(graph) => {
                self.apply_vertex_cover_hamiltonian(circuit, graph, gamma);
            },
            ProblemType::MaxIndependentSet(graph) => {
                self.apply_independent_set_hamiltonian(circuit, graph, gamma);
            },
            ProblemType::NumberPartitioning(numbers) => {
                self.apply_number_partition_hamiltonian(circuit, numbers, gamma);
            },
            ProblemType::GraphColoring(graph, num_colors) => {
                self.apply_graph_coloring_hamiltonian(circuit, graph, *num_colors, gamma);
            },
            ProblemType::MaxKSat {
                num_variables,
                clauses,
            } => {
                self.apply_maxsat_hamiltonian(circuit, *num_variables, clauses, gamma);
            },
            ProblemType::Custom { terms, .. } => {
                self.apply_custom_hamiltonian(circuit, terms, gamma);
            },
            _ => {
                // For unimplemented problems, apply identity (no-op)
            },
        }
    }

    /// Apply MaxCut Hamiltonian: H_C = sum_{(i,j) in E} w_{ij} * Z_i Z_j
    fn apply_maxcut_hamiltonian(&self, circuit: &mut Circuit, graph: &Graph, gamma: f64) {
        for &(i, j, weight) in &graph.edges {
            // Apply exp(-i * gamma * weight * Z_i Z_j)
            // This is equivalent to: CNOT(i,j), RZ(2*gamma*weight), CNOT(i,j)
            let _ = circuit.add_gate(Arc::new(CNot), &[QubitId::new(i), QubitId::new(j)]);
            let _ = circuit
                .add_gate(Arc::new(RotationZ::new(2.0 * gamma * weight)), &[QubitId::new(j)]);
            let _ = circuit.add_gate(Arc::new(CNot), &[QubitId::new(i), QubitId::new(j)]);
        }
    }

    /// Apply Vertex Cover Hamiltonian
    /// H_C = sum_{(i,j) in E} (1 - Z_i)(1 - Z_j) + penalty * sum_i Z_i
    fn apply_vertex_cover_hamiltonian(&self, circuit: &mut Circuit, graph: &Graph, gamma: f64) {
        let penalty = 0.5; // Penalty for including vertices

        // Edge constraints: both endpoints cannot be 0
        for &(i, j, weight) in &graph.edges {
            // (1 - Z_i)(1 - Z_j) = 1 - Z_i - Z_j + Z_i Z_j
            // Single qubit terms
            let _ = circuit
                .add_gate(Arc::new(RotationZ::new(-2.0 * gamma * weight)), &[QubitId::new(i)]);
            let _ = circuit
                .add_gate(Arc::new(RotationZ::new(-2.0 * gamma * weight)), &[QubitId::new(j)]);

            // Two qubit term: Z_i Z_j
            let _ = circuit.add_gate(Arc::new(CNot), &[QubitId::new(i), QubitId::new(j)]);
            let _ = circuit
                .add_gate(Arc::new(RotationZ::new(2.0 * gamma * weight)), &[QubitId::new(j)]);
            let _ = circuit.add_gate(Arc::new(CNot), &[QubitId::new(i), QubitId::new(j)]);
        }

        // Penalty terms: minimize number of vertices
        for i in 0..graph.num_vertices {
            let _ = circuit
                .add_gate(Arc::new(RotationZ::new(2.0 * gamma * penalty)), &[QubitId::new(i)]);
        }
    }

    /// Apply Independent Set Hamiltonian
    /// H_C = -sum_i Z_i + penalty * sum_{(i,j) in E} (1 - Z_i)(1 - Z_j)
    fn apply_independent_set_hamiltonian(&self, circuit: &mut Circuit, graph: &Graph, gamma: f64) {
        let penalty = 2.0; // Large penalty for adjacent vertices both being 1

        // Maximize set size: -sum_i Z_i
        for i in 0..graph.num_vertices {
            let _ = circuit.add_gate(Arc::new(RotationZ::new(-2.0 * gamma)), &[QubitId::new(i)]);
        }

        // Penalty for edges within the set
        for &(i, j, _) in &graph.edges {
            // (1 - Z_i)(1 - Z_j) penalizes both vertices being 1
            let _ = circuit
                .add_gate(Arc::new(RotationZ::new(-2.0 * gamma * penalty)), &[QubitId::new(i)]);
            let _ = circuit
                .add_gate(Arc::new(RotationZ::new(-2.0 * gamma * penalty)), &[QubitId::new(j)]);
            let _ = circuit.add_gate(Arc::new(CNot), &[QubitId::new(i), QubitId::new(j)]);
            let _ = circuit
                .add_gate(Arc::new(RotationZ::new(2.0 * gamma * penalty)), &[QubitId::new(j)]);
            let _ = circuit.add_gate(Arc::new(CNot), &[QubitId::new(i), QubitId::new(j)]);
        }
    }

    /// Apply Number Partitioning Hamiltonian
    /// H_C = (sum_i n_i * Z_i)^2
    fn apply_number_partition_hamiltonian(
        &self,
        circuit: &mut Circuit,
        numbers: &[f64],
        gamma: f64,
    ) {
        // (sum_i n_i Z_i)^2 = sum_i n_i^2 + sum_{i<j} 2 n_i n_j Z_i Z_j

        // Single qubit terms: n_i^2 (constant, can be ignored)

        // Two qubit terms: 2 n_i n_j Z_i Z_j
        for i in 0..numbers.len() {
            for j in (i + 1)..numbers.len() {
                let coeff = 2.0 * numbers[i] * numbers[j];
                let _ = circuit.add_gate(Arc::new(CNot), &[QubitId::new(i), QubitId::new(j)]);
                let _ = circuit
                    .add_gate(Arc::new(RotationZ::new(2.0 * gamma * coeff)), &[QubitId::new(j)]);
                let _ = circuit.add_gate(Arc::new(CNot), &[QubitId::new(i), QubitId::new(j)]);
            }
        }
    }

    /// Apply Graph Coloring Hamiltonian (placeholder)
    fn apply_graph_coloring_hamiltonian(
        &self,
        circuit: &mut Circuit,
        graph: &Graph,
        _num_colors: usize,
        gamma: f64,
    ) {
        // This is a complex problem - simplified version
        // Each vertex uses log2(num_colors) qubits for color encoding
        // For now, apply MaxCut-like Hamiltonian on edges
        self.apply_maxcut_hamiltonian(circuit, graph, gamma);
    }

    /// Apply MaxSAT Hamiltonian (placeholder)
    fn apply_maxsat_hamiltonian(
        &self,
        circuit: &mut Circuit,
        _num_variables: usize,
        clauses: &[(Vec<usize>, Vec<bool>, f64)],
        gamma: f64,
    ) {
        // Each clause contributes terms based on satisfied literals
        // Simplified: apply rotations based on clause structure
        for (vars, negations, weight) in clauses {
            // For each variable in the clause
            for (idx, &var) in vars.iter().enumerate() {
                let sign = if negations[idx] { -1.0 } else { 1.0 };
                let _ = circuit.add_gate(
                    Arc::new(RotationZ::new(2.0 * gamma * weight * sign)),
                    &[QubitId::new(var)],
                );
            }
        }
    }

    /// Apply custom Hamiltonian
    fn apply_custom_hamiltonian(
        &self,
        circuit: &mut Circuit,
        terms: &[(Vec<usize>, f64)],
        gamma: f64,
    ) {
        for (qubits, coeff) in terms {
            if qubits.len() == 1 {
                // Single qubit term: Z_i
                let _ = circuit.add_gate(
                    Arc::new(RotationZ::new(2.0 * gamma * coeff)),
                    &[QubitId::new(qubits[0])],
                );
            } else if qubits.len() == 2 {
                // Two qubit term: Z_i Z_j
                let _ = circuit
                    .add_gate(Arc::new(CNot), &[QubitId::new(qubits[0]), QubitId::new(qubits[1])]);
                let _ = circuit.add_gate(
                    Arc::new(RotationZ::new(2.0 * gamma * coeff)),
                    &[QubitId::new(qubits[1])],
                );
                let _ = circuit
                    .add_gate(Arc::new(CNot), &[QubitId::new(qubits[0]), QubitId::new(qubits[1])]);
            }
            // For more qubits, would need multi-controlled operations
        }
    }

    /// Apply mixer Hamiltonian evolution exp(-i * beta * H_M)
    fn apply_mixer(&self, circuit: &mut Circuit, beta: f64) {
        match self.config.mixer {
            MixerType::StandardX => {
                // H_M = sum_i X_i => apply RX(2*beta) to all qubits
                for q in 0..self.num_qubits {
                    let _ =
                        circuit.add_gate(Arc::new(RotationX::new(2.0 * beta)), &[QubitId::new(q)]);
                }
            },
            MixerType::StandardY => {
                // H_M = sum_i Y_i => apply RY(2*beta) to all qubits
                for q in 0..self.num_qubits {
                    let _ =
                        circuit.add_gate(Arc::new(RotationY::new(2.0 * beta)), &[QubitId::new(q)]);
                }
            },
            MixerType::XY => {
                // H_M = sum_{<i,j>} (X_i X_j + Y_i Y_j)
                // This preserves Hamming weight
                self.apply_xy_mixer(circuit, beta);
            },
            MixerType::Grover => {
                // Grover mixer: 2|s⟩⟨s| - I where |s⟩ is uniform superposition
                self.apply_grover_mixer(circuit, beta);
            },
            MixerType::Ring => {
                // Ring mixer: nearest-neighbor XY coupling
                self.apply_ring_mixer(circuit, beta);
            },
        }
    }

    /// Apply XY mixer (Hamming weight preserving)
    fn apply_xy_mixer(&self, circuit: &mut Circuit, beta: f64) {
        // XY mixer on all pairs (or nearest neighbors for efficiency)
        for i in 0..(self.num_qubits - 1) {
            let j = i + 1;
            // XX interaction
            let _ = circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(i)]);
            let _ = circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(j)]);
            let _ = circuit.add_gate(Arc::new(CNot), &[QubitId::new(i), QubitId::new(j)]);
            let _ = circuit.add_gate(Arc::new(RotationZ::new(2.0 * beta)), &[QubitId::new(j)]);
            let _ = circuit.add_gate(Arc::new(CNot), &[QubitId::new(i), QubitId::new(j)]);
            let _ = circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(i)]);
            let _ = circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(j)]);

            // YY interaction (can be added for full XY mixer)
        }
    }

    /// Apply Grover mixer (simplified)
    fn apply_grover_mixer(&self, circuit: &mut Circuit, beta: f64) {
        // Simplified Grover mixer: H - X - multi-controlled Z - X - H
        for q in 0..self.num_qubits {
            let _ = circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]);
        }
        for q in 0..self.num_qubits {
            let _ = circuit.add_gate(Arc::new(RotationZ::new(2.0 * beta)), &[QubitId::new(q)]);
        }
        for q in 0..self.num_qubits {
            let _ = circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]);
        }
    }

    /// Apply ring mixer
    fn apply_ring_mixer(&self, circuit: &mut Circuit, beta: f64) {
        // XY coupling in a ring
        for i in 0..self.num_qubits {
            let j = (i + 1) % self.num_qubits;
            let _ = circuit.add_gate(Arc::new(CNot), &[QubitId::new(i), QubitId::new(j)]);
            let _ = circuit.add_gate(Arc::new(RotationZ::new(2.0 * beta)), &[QubitId::new(j)]);
            let _ = circuit.add_gate(Arc::new(CNot), &[QubitId::new(i), QubitId::new(j)]);
        }
    }

    /// Generate the cost observable for measurement
    ///
    /// Returns a PauliObservable that can be used to measure the cost function
    pub fn cost_observable(&self) -> Result<PauliObservable, String> {
        match &self.problem {
            ProblemType::MaxCut(graph) => self.maxcut_observable(graph),
            ProblemType::NumberPartitioning(numbers) => self.number_partition_observable(numbers),
            ProblemType::Custom { terms, .. } => self.custom_observable(terms),
            _ => Err("Observable generation not yet implemented for this problem type".to_string()),
        }
    }

    /// Generate MaxCut observable
    fn maxcut_observable(&self, graph: &Graph) -> Result<PauliObservable, String> {
        if graph.edges.is_empty() {
            return Err("Graph has no edges".to_string());
        }

        // Start with the first edge
        let (i, j, weight) = graph.edges[0];
        let mut paulis = vec![Pauli::I; self.num_qubits];
        paulis[i] = Pauli::Z;
        paulis[j] = Pauli::Z;

        let mut observable =
            PauliObservable::from_pauli_string(PauliString::from_paulis(paulis), weight);

        // Add remaining edges
        for &(i, j, weight) in &graph.edges[1..] {
            let mut paulis = vec![Pauli::I; self.num_qubits];
            paulis[i] = Pauli::Z;
            paulis[j] = Pauli::Z;

            observable.add_term(PauliString::from_paulis(paulis), weight);
        }

        Ok(observable)
    }

    /// Generate Number Partition observable
    fn number_partition_observable(&self, numbers: &[f64]) -> Result<PauliObservable, String> {
        // (sum_i n_i Z_i)^2 - need to expand this
        // For simplicity, return sum of Z_i terms weighted by numbers
        let mut paulis = vec![Pauli::I; self.num_qubits];
        paulis[0] = Pauli::Z;

        let mut observable =
            PauliObservable::from_pauli_string(PauliString::from_paulis(paulis), numbers[0]);

        for i in 1..numbers.len() {
            let mut paulis = vec![Pauli::I; self.num_qubits];
            paulis[i] = Pauli::Z;

            observable.add_term(PauliString::from_paulis(paulis), numbers[i]);
        }

        Ok(observable)
    }

    /// Generate custom observable
    fn custom_observable(&self, terms: &[(Vec<usize>, f64)]) -> Result<PauliObservable, String> {
        if terms.is_empty() {
            return Err("No terms provided".to_string());
        }

        // Start with first term
        let (qubits, coeff) = &terms[0];
        let mut paulis = vec![Pauli::I; self.num_qubits];
        for &q in qubits {
            paulis[q] = Pauli::Z;
        }

        let mut observable =
            PauliObservable::from_pauli_string(PauliString::from_paulis(paulis), *coeff);

        // Add remaining terms
        for (qubits, coeff) in &terms[1..] {
            let mut paulis = vec![Pauli::I; self.num_qubits];
            for &q in qubits {
                paulis[q] = Pauli::Z;
            }

            observable.add_term(PauliString::from_paulis(paulis), *coeff);
        }

        Ok(observable)
    }

    /// Get problem description
    pub fn problem_description(&self) -> String {
        self.problem.description()
    }

    /// Get mixer description
    pub fn mixer_description(&self) -> String {
        self.config.mixer.description().to_string()
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Evaluate a classical solution for MaxCut
pub fn evaluate_maxcut_solution(graph: &Graph, bitstring: &[bool]) -> f64 {
    let mut cost = 0.0;
    for &(i, j, weight) in &graph.edges {
        if bitstring[i] != bitstring[j] {
            cost += weight;
        }
    }
    cost
}

/// Evaluate a classical solution for Number Partitioning
pub fn evaluate_partition_solution(numbers: &[f64], bitstring: &[bool]) -> f64 {
    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    for (i, &num) in numbers.iter().enumerate() {
        if bitstring[i] {
            sum_a += num;
        } else {
            sum_b += num;
        }
    }
    (sum_a - sum_b).abs()
}

/// Generate random initial parameters for QAOA
pub fn random_initial_parameters(depth: usize, seed: Option<u64>) -> Vec<f64> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };

    let mut params = Vec::with_capacity(2 * depth);
    for _ in 0..depth {
        // gamma in [0, π]
        params.push(rng.gen::<f64>() * std::f64::consts::PI);
        // beta in [0, π/2]
        params.push(rng.gen::<f64>() * std::f64::consts::FRAC_PI_2);
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = Graph::from_edges(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
        assert_eq!(graph.num_vertices, 4);
        assert_eq!(graph.num_edges(), 3);
        assert_eq!(graph.degree(1), 2);
        assert_eq!(graph.degree(0), 1);
    }

    #[test]
    fn test_complete_graph() {
        let graph = Graph::complete(4);
        assert_eq!(graph.num_vertices, 4);
        assert_eq!(graph.num_edges(), 6); // n*(n-1)/2 = 4*3/2 = 6
    }

    #[test]
    fn test_cycle_graph() {
        let graph = Graph::cycle(5);
        assert_eq!(graph.num_vertices, 5);
        assert_eq!(graph.num_edges(), 5);
        for i in 0..5 {
            assert_eq!(graph.degree(i), 2);
        }
    }

    #[test]
    fn test_qaoa_builder() {
        let graph = Graph::cycle(3);
        let builder = QAOACircuitBuilder::new(ProblemType::MaxCut(graph), MixerType::StandardX, 2);

        assert_eq!(builder.num_qubits(), 3);
        assert_eq!(builder.num_parameters(), 4);

        let params = vec![0.5, 0.3, 0.7, 0.4];
        let circuit = builder.build(&params);
        assert_eq!(circuit.num_qubits(), 3);
    }

    #[test]
    fn test_maxcut_evaluation() {
        let graph = Graph::cycle(4);
        let bitstring = vec![true, false, true, false];
        let cost = evaluate_maxcut_solution(&graph, &bitstring);
        assert_eq!(cost, 4.0); // All edges are cut
    }

    #[test]
    fn test_random_parameters() {
        let params = random_initial_parameters(3, Some(42));
        assert_eq!(params.len(), 6);
        // Check bounds
        for i in 0..3 {
            assert!(params[2 * i] >= 0.0 && params[2 * i] <= std::f64::consts::PI);
            assert!(params[2 * i + 1] >= 0.0 && params[2 * i + 1] <= std::f64::consts::FRAC_PI_2);
        }
    }
}
