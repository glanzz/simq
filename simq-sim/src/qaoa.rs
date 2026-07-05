//! QAOA (Quantum Approximate Optimization Algorithm) Circuit Generator
//!
//! This module provides a comprehensive framework for generating QAOA circuits
//! for various combinatorial optimization problems.
//!
//! # Features
//!
//! - **Problem Hamiltonians**: MaxCut, Number Partitioning, Vertex Cover,
//!   Independent Set, Max k-SAT, and custom Ising-type Hamiltonians
//! - **Mixer Hamiltonians**: Standard X/Y mixers, XY chain mixer, ring mixer
//! - **Graph Utilities**: Graph representation, edge management, topology operations
//! - **Flexible Configuration**: Customizable circuit depth, initialization strategies
//! - **Observable Generation**: Automatic Hamiltonian-to-Observable conversion
//!
//! # Failure semantics
//!
//! Construction fails loudly: problem types and mixers whose Hamiltonians are
//! not implemented ([`ProblemType::TSP`], [`ProblemType::Portfolio`],
//! [`ProblemType::GraphColoring`], [`MixerType::Grover`]) return
//! [`QAOAError::Unsupported`] from [`QAOACircuitBuilder::build`] instead of
//! silently producing a circuit that evolves under the wrong (or no)
//! Hamiltonian.
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
//! let circuit = builder.build(&params)?;
//!
//! // Get the cost observable
//! let observable = builder.cost_observable()?;
//! ```

use simq_core::{Circuit, QubitId};
use simq_gates::{CNot, Hadamard, PauliX, RotationX, RotationY, RotationZ, SGate, SGateDagger};
use simq_state::observable::{Pauli, PauliObservable, PauliString};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use thiserror::Error;

/// Result type for QAOA circuit construction
pub type QAOAResult<T> = std::result::Result<T, QAOAError>;

/// Errors raised during QAOA circuit or observable construction
///
/// Following the fail-loudly discipline, anything the builder cannot do
/// *correctly* is an error — there are no silent approximations or no-op
/// fallbacks.
#[derive(Error, Debug, Clone)]
pub enum QAOAError {
    /// Wrong number of variational parameters for the configured depth
    #[error("expected {expected} parameters for {depth} QAOA layer(s), got {actual}")]
    ParameterCountMismatch {
        expected: usize,
        actual: usize,
        depth: usize,
    },

    /// The requested problem type or mixer has no implementation yet
    #[error("{feature} is not implemented: {detail}")]
    Unsupported { feature: String, detail: String },

    /// The problem definition itself is malformed
    #[error("invalid problem definition: {0}")]
    InvalidProblem(String),

    /// The configured initial state needs data that was not provided
    #[error("initial state {state:?} requires {detail}")]
    MissingInitialStateData {
        state: InitialState,
        detail: String,
    },

    /// Underlying circuit construction failed
    #[error("circuit construction failed: {0}")]
    Circuit(String),
}

impl From<simq_core::QuantumError> for QAOAError {
    fn from(err: simq_core::QuantumError) -> Self {
        QAOAError::Circuit(err.to_string())
    }
}

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

    /// Check that all edge endpoints are valid vertex indices
    fn validate(&self) -> QAOAResult<()> {
        for &(u, v, _) in &self.edges {
            if u >= self.num_vertices || v >= self.num_vertices {
                return Err(QAOAError::InvalidProblem(format!(
                    "edge ({}, {}) references a vertex outside 0..{}",
                    u, v, self.num_vertices
                )));
            }
        }
        Ok(())
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
    ///
    /// **Not implemented**: building a circuit or observable for this problem
    /// returns [`QAOAError::Unsupported`]. The one-hot encoding it requires
    /// (vertex × color qubits with validity constraints) has not been written
    /// yet.
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
    ///
    /// **Not implemented**: building a circuit or observable for this problem
    /// returns [`QAOAError::Unsupported`].
    TSP {
        num_cities: usize,
        distances: Vec<Vec<f64>>,
    },

    /// Portfolio optimization
    /// (expected_returns, covariance_matrix, risk_factor, budget)
    ///
    /// **Not implemented**: building a circuit or observable for this problem
    /// returns [`QAOAError::Unsupported`].
    Portfolio {
        assets: usize,
        expected_returns: Vec<f64>,
        covariances: Vec<Vec<f64>>,
        risk_factor: f64,
        budget: usize,
    },

    /// Custom problem with explicit cost Hamiltonian terms
    /// Each term is (qubit_indices, weight) representing weight · Z_{i1} Z_{i2} ⋯
    /// Terms may involve any number of qubits.
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

    /// Validate the problem definition, failing loudly on malformed input
    fn validate(&self) -> QAOAResult<()> {
        match self {
            ProblemType::MaxCut(g)
            | ProblemType::MinVertexCover(g)
            | ProblemType::MaxIndependentSet(g) => g.validate(),

            ProblemType::NumberPartitioning(numbers) => {
                if numbers.is_empty() {
                    return Err(QAOAError::InvalidProblem(
                        "number partitioning requires at least one number".to_string(),
                    ));
                }
                Ok(())
            },

            ProblemType::MaxKSat {
                num_variables,
                clauses,
            } => validate_maxksat(*num_variables, clauses),

            ProblemType::Custom { num_qubits, terms } => {
                if terms.is_empty() {
                    return Err(QAOAError::InvalidProblem(
                        "custom problem has no Hamiltonian terms".to_string(),
                    ));
                }
                for (qubits, _) in terms {
                    let mut seen = qubits.clone();
                    seen.sort_unstable();
                    seen.dedup();
                    if seen.len() != qubits.len() {
                        return Err(QAOAError::InvalidProblem(format!(
                            "custom term {:?} repeats a qubit index",
                            qubits
                        )));
                    }
                    if let Some(&q) = qubits.iter().find(|&&q| q >= *num_qubits) {
                        return Err(QAOAError::InvalidProblem(format!(
                            "custom term references qubit {} but the problem has {} qubits",
                            q, num_qubits
                        )));
                    }
                }
                Ok(())
            },

            // Unsupported problems are rejected at build/observable time with a
            // more specific error; nothing to validate structurally here.
            ProblemType::GraphColoring(..) | ProblemType::TSP { .. } | ProblemType::Portfolio { .. } => {
                Ok(())
            },
        }
    }

    /// Error describing why this problem type cannot be compiled, if it can't
    fn unsupported_error(&self) -> Option<QAOAError> {
        let feature = match self {
            ProblemType::GraphColoring(..) => "GraphColoring cost Hamiltonian",
            ProblemType::TSP { .. } => "TSP cost Hamiltonian",
            ProblemType::Portfolio { .. } => "Portfolio cost Hamiltonian",
            _ => return None,
        };
        Some(QAOAError::Unsupported {
            feature: feature.to_string(),
            detail: "use ProblemType::Custom with explicit Ising terms, or contribute an encoding"
                .to_string(),
        })
    }
}

/// Validate a Max k-SAT instance
fn validate_maxksat(
    num_variables: usize,
    clauses: &[(Vec<usize>, Vec<bool>, f64)],
) -> QAOAResult<()> {
    /// Each clause of k literals expands into 2^k Pauli-Z product terms
    const MAX_CLAUSE_SIZE: usize = 16;

    if clauses.is_empty() {
        return Err(QAOAError::InvalidProblem(
            "Max k-SAT instance has no clauses".to_string(),
        ));
    }
    for (idx, (vars, negations, _)) in clauses.iter().enumerate() {
        if vars.is_empty() {
            return Err(QAOAError::InvalidProblem(format!("clause {} is empty", idx)));
        }
        if vars.len() != negations.len() {
            return Err(QAOAError::InvalidProblem(format!(
                "clause {} has {} variables but {} negation flags",
                idx,
                vars.len(),
                negations.len()
            )));
        }
        if vars.len() > MAX_CLAUSE_SIZE {
            return Err(QAOAError::InvalidProblem(format!(
                "clause {} has {} literals; clauses larger than {} are not supported \
                 (the Pauli expansion is exponential in clause size)",
                idx,
                vars.len(),
                MAX_CLAUSE_SIZE
            )));
        }
        let mut seen = vars.clone();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != vars.len() {
            return Err(QAOAError::InvalidProblem(format!(
                "clause {} repeats a variable",
                idx
            )));
        }
        if let Some(&v) = vars.iter().find(|&&v| v >= num_variables) {
            return Err(QAOAError::InvalidProblem(format!(
                "clause {} references variable {} but the instance has {} variables",
                idx, v, num_variables
            )));
        }
    }
    Ok(())
}

/// Expand a Max k-SAT instance into Pauli-Z product terms.
///
/// With the convention x_i = (1 − Z_i)/2 (bit value 1 ⇔ Z eigenvalue −1), a
/// clause is *unsatisfied* exactly when every literal is false, so its
/// indicator is Π_j (1 + s_j Z_j)/2 with s_j = +1 for a positive literal and
/// s_j = −1 for a negated one. The cost Hamiltonian counts the weighted
/// number of unsatisfied clauses:
///
/// H_C = Σ_c w_c · Π_{j∈c} (1 + s_j Z_j)/2
///
/// Returns terms keyed by their (sorted) qubit support; the empty support is
/// the constant (identity) contribution.
fn maxksat_z_terms(
    clauses: &[(Vec<usize>, Vec<bool>, f64)],
) -> BTreeMap<Vec<usize>, f64> {
    let mut terms: BTreeMap<Vec<usize>, f64> = BTreeMap::new();

    for (vars, negations, weight) in clauses {
        let k = vars.len();
        let scale = weight / (1u64 << k) as f64;

        for subset in 0..(1u64 << k) {
            let mut support = Vec::new();
            let mut sign = 1.0;
            for (j, &var) in vars.iter().enumerate() {
                if subset & (1 << j) != 0 {
                    support.push(var);
                    // positive literal → +Z, negated literal → −Z
                    if negations[j] {
                        sign = -sign;
                    }
                }
            }
            support.sort_unstable();
            *terms.entry(support).or_insert(0.0) += sign * scale;
        }
    }

    terms.retain(|_, coeff| coeff.abs() > 1e-12);
    terms
}

/// Classical evaluation of a Max k-SAT cost (weighted unsatisfied clauses)
pub fn evaluate_maxksat_solution(
    clauses: &[(Vec<usize>, Vec<bool>, f64)],
    bitstring: &[bool],
) -> f64 {
    let mut cost = 0.0;
    for (vars, negations, weight) in clauses {
        let satisfied = vars
            .iter()
            .zip(negations)
            .any(|(&v, &neg)| bitstring[v] != neg);
        if !satisfied {
            cost += weight;
        }
    }
    cost
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

    /// XY chain mixer: H_M = sum_{i} (X_i X_{i+1} + Y_i Y_{i+1})
    /// Preserves Hamming weight (number of 1s)
    XY,

    /// Grover mixer: e^{-iβ|s⟩⟨s|} where |s⟩ is the uniform superposition.
    ///
    /// **Not implemented**: this mixer requires a multi-controlled phase
    /// decomposition that has not been written; selecting it returns
    /// [`QAOAError::Unsupported`] from [`QAOACircuitBuilder::build`].
    Grover,

    /// Ring mixer: XY coupling (X_i X_j + Y_i Y_j) around a closed ring
    Ring,
}

impl MixerType {
    /// Get a description of the mixer
    pub fn description(&self) -> &str {
        match self {
            MixerType::StandardX => "Standard X mixer (sum of X_i)",
            MixerType::StandardY => "Standard Y mixer (sum of Y_i)",
            MixerType::XY => "XY chain mixer (preserves Hamming weight)",
            MixerType::Grover => "Grover mixer (diffusion operator; not implemented)",
            MixerType::Ring => "Ring mixer (nearest-neighbor XY coupling on a ring)",
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

    /// Whether to apply the mixer on the last layer.
    ///
    /// Standard QAOA (Farhi et al.) applies the mixer in *every* layer,
    /// including the last: U(β_p, γ_p)···U(β_1, γ_1)|+⟩^⊗n with
    /// U(β, γ) = e^{-iβB} e^{-iγC}. Disabling this is a niche variation;
    /// with it off, a depth-1 circuit contains no mixer at all and β has no
    /// effect on the output distribution.
    pub final_mixer: bool,
}

impl Default for QAOAConfig {
    fn default() -> Self {
        Self {
            depth: 1,
            mixer: MixerType::StandardX,
            initial_state: InitialState::UniformSuperposition,
            final_mixer: true,
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

    /// Warm start from a classical solution bitstring.
    ///
    /// The bitstring must be supplied via
    /// [`QAOACircuitBuilder::with_warm_start`]; building without it is an
    /// error (the builder never silently substitutes another initial state).
    WarmStart,

    /// Custom initial state prepared by a user-supplied circuit.
    ///
    /// The circuit must be supplied via
    /// [`QAOACircuitBuilder::with_initial_circuit`]; building without it is
    /// an error.
    Custom,
}

/// QAOA Circuit Builder
///
/// Generates QAOA circuits for various combinatorial optimization problems.
pub struct QAOACircuitBuilder {
    problem: ProblemType,
    config: QAOAConfig,
    num_qubits: usize,
    warm_start: Option<Vec<bool>>,
    initial_circuit: Option<Circuit>,
}

impl QAOACircuitBuilder {
    /// Penalty coefficient used by the Min Vertex Cover Hamiltonian
    pub const VERTEX_COVER_PENALTY: f64 = 0.5;
    /// Penalty coefficient used by the Max Independent Set Hamiltonian
    pub const INDEPENDENT_SET_PENALTY: f64 = 2.0;

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
            warm_start: None,
            initial_circuit: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(problem: ProblemType, config: QAOAConfig) -> Self {
        let num_qubits = problem.num_qubits();
        Self {
            problem,
            config,
            num_qubits,
            warm_start: None,
            initial_circuit: None,
        }
    }

    /// Provide the classical bitstring for [`InitialState::WarmStart`].
    ///
    /// The bitstring length must equal the number of qubits; this is checked
    /// at [`build`](Self::build) time.
    pub fn with_warm_start(mut self, bitstring: Vec<bool>) -> Self {
        self.warm_start = Some(bitstring);
        self
    }

    /// Provide the preparation circuit for [`InitialState::Custom`].
    ///
    /// The circuit's qubit count must equal the problem's qubit count; this
    /// is checked at [`build`](Self::build) time.
    pub fn with_initial_circuit(mut self, circuit: Circuit) -> Self {
        self.initial_circuit = Some(circuit);
        self
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
    /// A quantum circuit implementing the QAOA algorithm.
    ///
    /// # Errors
    ///
    /// Fails loudly instead of degrading silently:
    /// * [`QAOAError::ParameterCountMismatch`] if `params.len() != 2·depth`
    /// * [`QAOAError::Unsupported`] for problem types or mixers whose
    ///   Hamiltonians are not implemented (TSP, Portfolio, GraphColoring,
    ///   Grover mixer)
    /// * [`QAOAError::InvalidProblem`] for malformed problem definitions
    /// * [`QAOAError::MissingInitialStateData`] if `WarmStart`/`Custom`
    ///   initial states were selected without supplying their data
    pub fn build(&self, params: &[f64]) -> QAOAResult<Circuit> {
        if params.len() != self.num_parameters() {
            return Err(QAOAError::ParameterCountMismatch {
                expected: self.num_parameters(),
                actual: params.len(),
                depth: self.config.depth,
            });
        }

        self.problem.validate()?;
        if let Some(err) = self.problem.unsupported_error() {
            return Err(err);
        }

        let mut circuit = Circuit::new(self.num_qubits);

        // Initial state preparation
        self.apply_initial_state(&mut circuit)?;

        // QAOA layers
        for layer in 0..self.config.depth {
            let gamma = params[2 * layer];
            let beta = params[2 * layer + 1];

            // Problem Hamiltonian layer
            self.apply_problem_hamiltonian(&mut circuit, gamma)?;

            // Mixer Hamiltonian layer
            if layer < self.config.depth - 1 || self.config.final_mixer {
                self.apply_mixer(&mut circuit, beta)?;
            }
        }

        Ok(circuit)
    }

    /// Apply initial state preparation
    fn apply_initial_state(&self, circuit: &mut Circuit) -> QAOAResult<()> {
        match self.config.initial_state {
            InitialState::UniformSuperposition => {
                for q in 0..self.num_qubits {
                    circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(q)])?;
                }
            },
            InitialState::Zero => {
                // Qubits start in |0⟩
            },
            InitialState::WarmStart => {
                let bits = self.warm_start.as_ref().ok_or_else(|| {
                    QAOAError::MissingInitialStateData {
                        state: InitialState::WarmStart,
                        detail: "a bitstring; supply one with QAOACircuitBuilder::with_warm_start"
                            .to_string(),
                    }
                })?;
                if bits.len() != self.num_qubits {
                    return Err(QAOAError::MissingInitialStateData {
                        state: InitialState::WarmStart,
                        detail: format!(
                            "a bitstring of length {} (got {})",
                            self.num_qubits,
                            bits.len()
                        ),
                    });
                }
                for (q, &bit) in bits.iter().enumerate() {
                    if bit {
                        circuit.add_gate(Arc::new(PauliX), &[QubitId::new(q)])?;
                    }
                }
            },
            InitialState::Custom => {
                let prep = self.initial_circuit.as_ref().ok_or_else(|| {
                    QAOAError::MissingInitialStateData {
                        state: InitialState::Custom,
                        detail:
                            "a preparation circuit; supply one with QAOACircuitBuilder::with_initial_circuit"
                                .to_string(),
                    }
                })?;
                if prep.num_qubits() != self.num_qubits {
                    return Err(QAOAError::MissingInitialStateData {
                        state: InitialState::Custom,
                        detail: format!(
                            "a preparation circuit on {} qubits (got {})",
                            self.num_qubits,
                            prep.num_qubits()
                        ),
                    });
                }
                circuit.append(prep)?;
            },
        }
        Ok(())
    }

    /// Expand the cost Hamiltonian into Pauli-Z product terms.
    ///
    /// This is the **single source of truth** for every problem's Hamiltonian:
    /// both the evolution circuit ([`build`](Self::build)) and the measurement
    /// observable ([`cost_observable`](Self::cost_observable)) are generated
    /// from this term list, so they cannot drift apart.
    ///
    /// Bit convention: bit x_i = 1 (vertex selected / variable true)
    /// corresponds to the Z_i eigenvalue −1, i.e. x_i = (1 − Z_i)/2. This
    /// matches computational-basis measurement of |1⟩.
    ///
    /// Terms with empty support are constants: they contribute to observable
    /// eigenvalues but generate no gates (global phase).
    fn cost_z_terms(&self) -> QAOAResult<Vec<(Vec<usize>, f64)>> {
        match &self.problem {
            // H = Σ_edges w Z_i Z_j (eigenvalue w·(+1) on uncut, −w on cut edges)
            ProblemType::MaxCut(graph) => Ok(graph
                .edges
                .iter()
                .map(|&(i, j, w)| (vec![i, j], w))
                .collect()),

            // H = Σ_edges 4w·(1−x_i)(1−x_j) + penalty·Σ_i x_i
            //   = Σ_edges w·(1 + Z_i + Z_j + Z_iZ_j)
            //     + penalty·(n − Σ_i Z_i)/2
            // Eigenvalue: 4w per uncovered edge + penalty·|cover|.
            ProblemType::MinVertexCover(graph) => {
                let mut terms: Vec<(Vec<usize>, f64)> = Vec::new();
                let mut constant = 0.0;
                for &(i, j, w) in &graph.edges {
                    terms.push((vec![i], w));
                    terms.push((vec![j], w));
                    terms.push((vec![i, j], w));
                    constant += w;
                }
                let penalty = Self::VERTEX_COVER_PENALTY;
                for i in 0..graph.num_vertices {
                    terms.push((vec![i], -penalty / 2.0));
                }
                constant += penalty * graph.num_vertices as f64 / 2.0;
                terms.push((vec![], constant));
                Ok(terms)
            },

            // H = −Σ_i x_i + penalty·Σ_edges x_i x_j
            //   = (Σ_i Z_i − n)/2 + penalty·Σ_edges (1 − Z_i − Z_j + Z_iZ_j)/4
            // Eigenvalue: −|S| + penalty·(#edges inside S).
            ProblemType::MaxIndependentSet(graph) => {
                let penalty = Self::INDEPENDENT_SET_PENALTY;
                let mut terms: Vec<(Vec<usize>, f64)> = Vec::new();
                let mut constant = -(graph.num_vertices as f64) / 2.0;
                for i in 0..graph.num_vertices {
                    terms.push((vec![i], 0.5));
                }
                for &(i, j, _) in &graph.edges {
                    terms.push((vec![i], -penalty / 4.0));
                    terms.push((vec![j], -penalty / 4.0));
                    terms.push((vec![i, j], penalty / 4.0));
                    constant += penalty / 4.0;
                }
                terms.push((vec![], constant));
                Ok(terms)
            },

            // H = (Σ n_i Z_i)² = Σ n_i² + Σ_{i<j} 2 n_i n_j Z_i Z_j
            // Eigenvalue: (S_A − S_B)².
            ProblemType::NumberPartitioning(numbers) => {
                let mut terms: Vec<(Vec<usize>, f64)> = Vec::new();
                terms.push((vec![], numbers.iter().map(|n| n * n).sum()));
                for i in 0..numbers.len() {
                    for j in (i + 1)..numbers.len() {
                        terms.push((vec![i, j], 2.0 * numbers[i] * numbers[j]));
                    }
                }
                Ok(terms)
            },

            // Exact clause expansion; eigenvalue = weighted unsatisfied clauses.
            ProblemType::MaxKSat { clauses, .. } => {
                Ok(maxksat_z_terms(clauses).into_iter().collect())
            },

            ProblemType::Custom { terms, .. } => Ok(terms.clone()),

            ProblemType::GraphColoring(..) | ProblemType::TSP { .. } | ProblemType::Portfolio { .. } => {
                // Unreachable from build()/cost_observable(), which reject these
                // up front. Guard anyway so a future direct call cannot silently
                // produce a no-op Hamiltonian.
                Err(self.problem.unsupported_error().expect("unsupported problem"))
            },
        }
    }

    /// Apply problem Hamiltonian evolution exp(-i * gamma * H_C)
    fn apply_problem_hamiltonian(&self, circuit: &mut Circuit, gamma: f64) -> QAOAResult<()> {
        for (support, coeff) in self.cost_z_terms()? {
            Self::apply_z_product(circuit, &support, gamma, coeff)?;
        }
        Ok(())
    }

    /// Apply exp(-i·γ·coeff·Z_{q1}Z_{q2}⋯Z_{qk}) for an arbitrary Z-product
    /// term using a CNOT parity ladder around an RZ rotation.
    fn apply_z_product(
        circuit: &mut Circuit,
        qubits: &[usize],
        gamma: f64,
        coeff: f64,
    ) -> QAOAResult<()> {
        match qubits {
            [] => Ok(()), // constant term: global phase, no gate needed
            [q] => {
                circuit
                    .add_gate(Arc::new(RotationZ::new(2.0 * gamma * coeff)), &[QubitId::new(*q)])?;
                Ok(())
            },
            _ => {
                for pair in qubits.windows(2) {
                    circuit.add_gate(
                        Arc::new(CNot),
                        &[QubitId::new(pair[0]), QubitId::new(pair[1])],
                    )?;
                }
                let last = *qubits.last().expect("non-empty");
                circuit.add_gate(
                    Arc::new(RotationZ::new(2.0 * gamma * coeff)),
                    &[QubitId::new(last)],
                )?;
                for pair in qubits.windows(2).rev() {
                    circuit.add_gate(
                        Arc::new(CNot),
                        &[QubitId::new(pair[0]), QubitId::new(pair[1])],
                    )?;
                }
                Ok(())
            },
        }
    }

    /// Apply mixer Hamiltonian evolution exp(-i * beta * H_M)
    fn apply_mixer(&self, circuit: &mut Circuit, beta: f64) -> QAOAResult<()> {
        match self.config.mixer {
            MixerType::StandardX => {
                // H_M = sum_i X_i => apply RX(2*beta) to all qubits
                for q in 0..self.num_qubits {
                    circuit.add_gate(Arc::new(RotationX::new(2.0 * beta)), &[QubitId::new(q)])?;
                }
                Ok(())
            },
            MixerType::StandardY => {
                // H_M = sum_i Y_i => apply RY(2*beta) to all qubits
                for q in 0..self.num_qubits {
                    circuit.add_gate(Arc::new(RotationY::new(2.0 * beta)), &[QubitId::new(q)])?;
                }
                Ok(())
            },
            MixerType::XY => {
                // Nearest-neighbour chain: sum_i (X_i X_{i+1} + Y_i Y_{i+1})
                for i in 0..self.num_qubits.saturating_sub(1) {
                    Self::apply_xy_interaction(circuit, i, i + 1, beta)?;
                }
                Ok(())
            },
            MixerType::Grover => Err(QAOAError::Unsupported {
                feature: "Grover mixer".to_string(),
                detail: "e^{-iβ|s⟩⟨s|} needs a multi-controlled phase decomposition that has \
                         not been implemented; choose StandardX, StandardY, XY, or Ring"
                    .to_string(),
            }),
            MixerType::Ring => {
                // XY coupling around a closed ring (chain plus wrap-around edge)
                match self.num_qubits {
                    0 | 1 => Ok(()), // no pairs to couple
                    2 => Self::apply_xy_interaction(circuit, 0, 1, beta),
                    n => {
                        for i in 0..n {
                            Self::apply_xy_interaction(circuit, i, (i + 1) % n, beta)?;
                        }
                        Ok(())
                    },
                }
            },
        }
    }

    /// Apply exp(-iβ(X_i X_j + Y_i Y_j)) exactly.
    ///
    /// X_iX_j and Y_iY_j commute, so the evolution factors into
    /// e^{-iβ X_iX_j} · e^{-iβ Y_iY_j}. Each factor is a basis-changed ZZ
    /// rotation: X = H Z H and Y = S X S† = (SH) Z (SH)†.
    fn apply_xy_interaction(
        circuit: &mut Circuit,
        i: usize,
        j: usize,
        beta: f64,
    ) -> QAOAResult<()> {
        let (qi, qj) = (QubitId::new(i), QubitId::new(j));

        // XX part: (H⊗H) e^{-iβ ZZ} (H⊗H)
        circuit.add_gate(Arc::new(Hadamard), &[qi])?;
        circuit.add_gate(Arc::new(Hadamard), &[qj])?;
        Self::apply_z_product(circuit, &[i, j], beta, 1.0)?;
        circuit.add_gate(Arc::new(Hadamard), &[qi])?;
        circuit.add_gate(Arc::new(Hadamard), &[qj])?;

        // YY part: (S⊗S)(H⊗H) e^{-iβ ZZ} (H⊗H)(S†⊗S†)
        circuit.add_gate(Arc::new(SGateDagger), &[qi])?;
        circuit.add_gate(Arc::new(SGateDagger), &[qj])?;
        circuit.add_gate(Arc::new(Hadamard), &[qi])?;
        circuit.add_gate(Arc::new(Hadamard), &[qj])?;
        Self::apply_z_product(circuit, &[i, j], beta, 1.0)?;
        circuit.add_gate(Arc::new(Hadamard), &[qi])?;
        circuit.add_gate(Arc::new(Hadamard), &[qj])?;
        circuit.add_gate(Arc::new(SGate), &[qi])?;
        circuit.add_gate(Arc::new(SGate), &[qj])?;

        Ok(())
    }

    /// Generate the cost observable for measurement
    ///
    /// Returns a `PauliObservable` matching the exact Hamiltonian the circuit
    /// evolves under (including constant/identity terms so eigenvalues equal
    /// the classical costs).
    ///
    /// # Errors
    ///
    /// Returns [`QAOAError::Unsupported`] for the problem types whose
    /// Hamiltonians are not implemented (GraphColoring, TSP, Portfolio) and
    /// [`QAOAError::InvalidProblem`] for malformed definitions.
    pub fn cost_observable(&self) -> QAOAResult<PauliObservable> {
        self.problem.validate()?;
        if let Some(err) = self.problem.unsupported_error() {
            return Err(err);
        }

        self.observable_from_z_terms(self.cost_z_terms()?)
    }

    /// Build an observable from a list of (support, coefficient) Z-product terms
    fn observable_from_z_terms<I>(&self, terms: I) -> QAOAResult<PauliObservable>
    where
        I: IntoIterator<Item = (Vec<usize>, f64)>,
    {
        let mut iter = terms.into_iter();
        let make_string = |support: &[usize]| {
            let mut paulis = vec![Pauli::I; self.num_qubits];
            for &q in support {
                paulis[q] = Pauli::Z;
            }
            PauliString::from_paulis(paulis)
        };

        let (first_support, first_coeff) = iter.next().ok_or_else(|| {
            QAOAError::InvalidProblem("cost Hamiltonian has no terms".to_string())
        })?;
        let mut observable =
            PauliObservable::from_pauli_string(make_string(&first_support), first_coeff);
        for (support, coeff) in iter {
            observable.add_term(make_string(&support), coeff);
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

/// Evaluate a classical solution for Min Vertex Cover.
///
/// Matches the QAOA cost Hamiltonian: 4w per uncovered edge plus
/// penalty·|cover| (lower is better; valid minimum covers minimize it).
pub fn evaluate_vertex_cover_solution(graph: &Graph, bitstring: &[bool]) -> f64 {
    let mut cost = 0.0;
    for &(i, j, w) in &graph.edges {
        if !bitstring[i] && !bitstring[j] {
            cost += 4.0 * w;
        }
    }
    let cover_size = bitstring.iter().filter(|&&b| b).count() as f64;
    cost + QAOACircuitBuilder::VERTEX_COVER_PENALTY * cover_size
}

/// Evaluate a classical solution for Max Independent Set.
///
/// Matches the QAOA cost Hamiltonian: −|S| + penalty·(edges inside S)
/// (lower is better; maximum independent sets minimize it).
pub fn evaluate_independent_set_solution(graph: &Graph, bitstring: &[bool]) -> f64 {
    let mut cost = -(bitstring.iter().filter(|&&b| b).count() as f64);
    for &(i, j, _) in &graph.edges {
        if bitstring[i] && bitstring[j] {
            cost += QAOACircuitBuilder::INDEPENDENT_SET_PENALTY;
        }
    }
    cost
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
    use crate::{Simulator, SimulatorConfig};
    use simq_state::DenseState;

    fn make_sim() -> Simulator {
        Simulator::new(SimulatorConfig::default().with_optimization(false))
    }

    /// Eigenvalue of a diagonal (Z-only) observable on a computational basis state
    fn diagonal_eigenvalue(obs: &PauliObservable, num_qubits: usize, bits: &[bool]) -> f64 {
        use num_complex::Complex64;
        let dim = 1usize << num_qubits;
        // Little-endian: bit q of the index is qubit q
        let mut index = 0usize;
        for (q, &b) in bits.iter().enumerate() {
            if b {
                index |= 1 << q;
            }
        }
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); dim];
        amplitudes[index] = Complex64::new(1.0, 0.0);
        let state = DenseState::from_amplitudes(num_qubits, &amplitudes).unwrap();
        obs.expectation_value(&state).unwrap()
    }

    /// Iterate all bitstrings of the given length
    fn all_bitstrings(n: usize) -> impl Iterator<Item = Vec<bool>> {
        (0usize..(1 << n)).map(move |x| (0..n).map(|q| x & (1 << q) != 0).collect())
    }

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
        let circuit = builder.build(&params).unwrap();
        assert_eq!(circuit.num_qubits(), 3);
    }

    #[test]
    fn test_wrong_parameter_count_is_error() {
        let builder = QAOACircuitBuilder::new(
            ProblemType::MaxCut(Graph::cycle(3)),
            MixerType::StandardX,
            2,
        );
        let err = builder.build(&[0.5, 0.3]).unwrap_err();
        assert!(matches!(err, QAOAError::ParameterCountMismatch { expected: 4, actual: 2, .. }));
    }

    /// Regression test for issue #40: with the default config, a depth-1 QAOA
    /// circuit must contain a mixer layer, so its output distribution depends
    /// on β and is not uniform.
    #[test]
    fn test_depth1_distribution_depends_on_beta() {
        let simulator = make_sim();
        let probabilities = |gamma: f64, beta: f64| -> Vec<f64> {
            let builder = QAOACircuitBuilder::new(
                ProblemType::MaxCut(Graph::cycle(5)),
                MixerType::StandardX,
                1,
            );
            let circuit = builder.build(&[gamma, beta]).unwrap();
            let result = simulator.run(&circuit).unwrap();
            result
                .state
                .to_dense_vec()
                .iter()
                .map(|a| a.norm_sqr())
                .collect()
        };

        let probs = probabilities(0.4, 0.6);
        let min = probs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max - min > 1e-3,
            "p=1 QAOA distribution is uniform (min={}, max={}): mixer missing",
            min,
            max
        );

        // And β must actually change the distribution
        let probs_other_beta = probabilities(0.4, 0.9);
        let diff: f64 = probs
            .iter()
            .zip(&probs_other_beta)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-3, "β has no effect on the p=1 distribution");
    }

    #[test]
    fn test_final_mixer_opt_out_still_available() {
        let config = QAOAConfig {
            depth: 1,
            final_mixer: false,
            ..QAOAConfig::default()
        };
        let builder = QAOACircuitBuilder::with_config(ProblemType::MaxCut(Graph::cycle(3)), config);
        let circuit = builder.build(&[0.4, 0.6]).unwrap();
        // 3 Hadamards + 3 edges × (CNOT, RZ, CNOT) and no RX mixer gates
        assert!(!circuit
            .operations()
            .any(|op| op.gate().name().starts_with("RX")));
    }

    // ------------------------------------------------------------------
    // Fail-loudly behavior
    // ------------------------------------------------------------------

    #[test]
    fn test_tsp_build_fails_loudly() {
        let builder = QAOACircuitBuilder::new(
            ProblemType::TSP {
                num_cities: 2,
                distances: vec![vec![0.0, 1.0], vec![1.0, 0.0]],
            },
            MixerType::StandardX,
            1,
        );
        let err = builder.build(&[0.5, 0.3]).unwrap_err();
        assert!(matches!(err, QAOAError::Unsupported { .. }), "got {:?}", err);
        assert!(builder.cost_observable().is_err());
    }

    #[test]
    fn test_portfolio_build_fails_loudly() {
        let builder = QAOACircuitBuilder::new(
            ProblemType::Portfolio {
                assets: 2,
                expected_returns: vec![0.1, 0.2],
                covariances: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                risk_factor: 0.5,
                budget: 1,
            },
            MixerType::StandardX,
            1,
        );
        assert!(matches!(
            builder.build(&[0.5, 0.3]).unwrap_err(),
            QAOAError::Unsupported { .. }
        ));
    }

    #[test]
    fn test_graph_coloring_build_fails_loudly() {
        let builder = QAOACircuitBuilder::new(
            ProblemType::GraphColoring(Graph::from_edges(2, &[(0, 1, 1.0)]), 2),
            MixerType::StandardX,
            1,
        );
        assert!(matches!(
            builder.build(&[0.5, 0.3]).unwrap_err(),
            QAOAError::Unsupported { .. }
        ));
        assert!(builder.cost_observable().is_err());
    }

    #[test]
    fn test_grover_mixer_fails_loudly() {
        let config = QAOAConfig {
            mixer: MixerType::Grover,
            ..QAOAConfig::default()
        };
        let builder =
            QAOACircuitBuilder::with_config(ProblemType::MaxCut(Graph::cycle(3)), config);
        assert!(matches!(
            builder.build(&[0.5, 0.3]).unwrap_err(),
            QAOAError::Unsupported { .. }
        ));
    }

    #[test]
    fn test_warm_start_without_bitstring_fails() {
        let config = QAOAConfig {
            initial_state: InitialState::WarmStart,
            ..QAOAConfig::default()
        };
        let builder =
            QAOACircuitBuilder::with_config(ProblemType::MaxCut(Graph::cycle(3)), config);
        assert!(matches!(
            builder.build(&[0.5, 0.3]).unwrap_err(),
            QAOAError::MissingInitialStateData { .. }
        ));
    }

    #[test]
    fn test_warm_start_wrong_length_fails() {
        let config = QAOAConfig {
            initial_state: InitialState::WarmStart,
            ..QAOAConfig::default()
        };
        let builder = QAOACircuitBuilder::with_config(ProblemType::MaxCut(Graph::cycle(3)), config)
            .with_warm_start(vec![true, false]); // 2 bits for a 3-qubit problem
        assert!(matches!(
            builder.build(&[0.5, 0.3]).unwrap_err(),
            QAOAError::MissingInitialStateData { .. }
        ));
    }

    #[test]
    fn test_warm_start_prepares_basis_state() {
        let config = QAOAConfig {
            initial_state: InitialState::WarmStart,
            final_mixer: false, // no mixer so the basis state only picks up phases
            ..QAOAConfig::default()
        };
        let builder = QAOACircuitBuilder::with_config(ProblemType::MaxCut(Graph::cycle(3)), config)
            .with_warm_start(vec![true, false, true]);
        let circuit = builder.build(&[0.5, 0.3]).unwrap();
        let result = make_sim().run(&circuit).unwrap();
        let amps = result.state.to_dense_vec();
        // |101⟩ little-endian = index 5; cost layer is diagonal so probability stays 1
        assert!((amps[5].norm_sqr() - 1.0).abs() < 1e-10, "amps = {:?}", amps);
    }

    #[test]
    fn test_custom_initial_state_without_circuit_fails() {
        let config = QAOAConfig {
            initial_state: InitialState::Custom,
            ..QAOAConfig::default()
        };
        let builder =
            QAOACircuitBuilder::with_config(ProblemType::MaxCut(Graph::cycle(3)), config);
        assert!(matches!(
            builder.build(&[0.5, 0.3]).unwrap_err(),
            QAOAError::MissingInitialStateData { .. }
        ));
    }

    #[test]
    fn test_custom_initial_state_wrong_size_fails() {
        let config = QAOAConfig {
            initial_state: InitialState::Custom,
            ..QAOAConfig::default()
        };
        let builder = QAOACircuitBuilder::with_config(ProblemType::MaxCut(Graph::cycle(3)), config)
            .with_initial_circuit(Circuit::new(2));
        assert!(matches!(
            builder.build(&[0.5, 0.3]).unwrap_err(),
            QAOAError::MissingInitialStateData { .. }
        ));
    }

    #[test]
    fn test_custom_initial_state_applied() {
        let mut prep = Circuit::new(3);
        prep.add_gate(Arc::new(PauliX), &[QubitId::new(1)]).unwrap();
        let config = QAOAConfig {
            initial_state: InitialState::Custom,
            final_mixer: false,
            ..QAOAConfig::default()
        };
        let builder = QAOACircuitBuilder::with_config(ProblemType::MaxCut(Graph::cycle(3)), config)
            .with_initial_circuit(prep);
        let circuit = builder.build(&[0.5, 0.3]).unwrap();
        let result = make_sim().run(&circuit).unwrap();
        let amps = result.state.to_dense_vec();
        // |010⟩ little-endian = index 2
        assert!((amps[2].norm_sqr() - 1.0).abs() < 1e-10, "amps = {:?}", amps);
    }

    #[test]
    fn test_invalid_graph_edge_fails() {
        let builder = QAOACircuitBuilder::new(
            ProblemType::MaxCut(Graph::from_edges(2, &[(0, 5, 1.0)])),
            MixerType::StandardX,
            1,
        );
        assert!(matches!(
            builder.build(&[0.5, 0.3]).unwrap_err(),
            QAOAError::InvalidProblem(_)
        ));
    }

    #[test]
    fn test_empty_custom_problem_fails() {
        let builder = QAOACircuitBuilder::new(
            ProblemType::Custom {
                num_qubits: 2,
                terms: vec![],
            },
            MixerType::StandardX,
            1,
        );
        assert!(matches!(
            builder.build(&[0.5, 0.3]).unwrap_err(),
            QAOAError::InvalidProblem(_)
        ));
    }

    // ------------------------------------------------------------------
    // Hamiltonian / observable correctness
    // ------------------------------------------------------------------

    /// The Max k-SAT Pauli expansion must reproduce the classical cost
    /// (weighted unsatisfied clauses) as the observable's eigenvalue on every
    /// computational basis state.
    #[test]
    fn test_maxksat_observable_matches_classical_cost() {
        let clauses = vec![
            // (x0 OR ¬x1 OR x2)
            (vec![0, 1, 2], vec![false, true, false], 1.0),
            // (¬x0 OR x1)
            (vec![0, 1], vec![true, false], 2.5),
            // (x2)
            (vec![2], vec![false], 0.7),
        ];
        let builder = QAOACircuitBuilder::new(
            ProblemType::MaxKSat {
                num_variables: 3,
                clauses: clauses.clone(),
            },
            MixerType::StandardX,
            1,
        );
        let obs = builder.cost_observable().unwrap();
        for bits in all_bitstrings(3) {
            let expected = evaluate_maxksat_solution(&clauses, &bits);
            let eigenvalue = diagonal_eigenvalue(&obs, 3, &bits);
            assert!(
                (eigenvalue - expected).abs() < 1e-10,
                "bits {:?}: eigenvalue {} != classical cost {}",
                bits,
                eigenvalue,
                expected
            );
        }
    }

    /// The number-partition observable eigenvalue must equal the squared
    /// imbalance (S_A − S_B)² for every partition.
    #[test]
    fn test_number_partition_observable_matches_classical_cost() {
        let numbers = vec![3.0, 1.0, 2.0];
        let builder = QAOACircuitBuilder::new(
            ProblemType::NumberPartitioning(numbers.clone()),
            MixerType::StandardX,
            1,
        );
        let obs = builder.cost_observable().unwrap();
        for bits in all_bitstrings(3) {
            let imbalance = evaluate_partition_solution(&numbers, &bits);
            let eigenvalue = diagonal_eigenvalue(&obs, 3, &bits);
            assert!(
                (eigenvalue - imbalance * imbalance).abs() < 1e-10,
                "bits {:?}: eigenvalue {} != imbalance² {}",
                bits,
                eigenvalue,
                imbalance * imbalance
            );
        }
    }

    /// Vertex-cover observable eigenvalue must equal the classical cost:
    /// 4w per uncovered edge plus penalty·|cover|. In particular, uncovered
    /// edges must be *penalized* under the standard x=(1−Z)/2 measurement
    /// convention (the old Hamiltonian had the convention backwards and
    /// rewarded selecting every vertex).
    #[test]
    fn test_vertex_cover_observable_matches_classical_cost() {
        let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.5)]);
        let builder =
            QAOACircuitBuilder::new(ProblemType::MinVertexCover(g.clone()), MixerType::StandardX, 1);
        let obs = builder.cost_observable().unwrap();
        for bits in all_bitstrings(3) {
            let expected = evaluate_vertex_cover_solution(&g, &bits);
            let eigenvalue = diagonal_eigenvalue(&obs, 3, &bits);
            assert!(
                (eigenvalue - expected).abs() < 1e-10,
                "bits {:?}: eigenvalue {} != expected {}",
                bits,
                eigenvalue,
                expected
            );
        }
        // Sanity: a minimum cover ({1}) must beat both the empty set and the
        // full set.
        let cover = evaluate_vertex_cover_solution(&g, &[false, true, false]);
        let empty = evaluate_vertex_cover_solution(&g, &[false, false, false]);
        let full = evaluate_vertex_cover_solution(&g, &[true, true, true]);
        assert!(cover < empty && cover < full);
    }

    /// Independent-set observable eigenvalue must equal −|S| + penalty·(edges
    /// inside S) for every subset.
    #[test]
    fn test_independent_set_observable_matches_classical_cost() {
        let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
        let builder = QAOACircuitBuilder::new(
            ProblemType::MaxIndependentSet(g.clone()),
            MixerType::StandardX,
            1,
        );
        let obs = builder.cost_observable().unwrap();
        for bits in all_bitstrings(3) {
            let expected = evaluate_independent_set_solution(&g, &bits);
            let eigenvalue = diagonal_eigenvalue(&obs, 3, &bits);
            assert!(
                (eigenvalue - expected).abs() < 1e-10,
                "bits {:?}: eigenvalue {} != expected {}",
                bits,
                eigenvalue,
                expected
            );
        }
        // The maximum independent set {0, 2} must have the lowest cost.
        let best = evaluate_independent_set_solution(&g, &[true, false, true]);
        for bits in all_bitstrings(3) {
            assert!(evaluate_independent_set_solution(&g, &bits) >= best - 1e-12);
        }
    }

    /// Multi-qubit custom terms must generate real evolution: applying the
    /// cost layer to a basis state imprints the phase e^{-iγ·eigenvalue}.
    #[test]
    fn test_custom_three_qubit_term_evolves_correct_phase() {
        let gamma = 0.37;
        let coeff = 0.8;
        let config = QAOAConfig {
            depth: 1,
            initial_state: InitialState::WarmStart,
            final_mixer: false,
            ..QAOAConfig::default()
        };
        // Z0 Z1 Z2 on |110⟩ (little-endian: x0=1, x1=1, x2=0) → (−1)(−1)(+1) = +1
        let builder = QAOACircuitBuilder::with_config(
            ProblemType::Custom {
                num_qubits: 3,
                terms: vec![(vec![0, 1, 2], coeff)],
            },
            config,
        )
        .with_warm_start(vec![true, true, false]);
        let circuit = builder.build(&[gamma, 0.0]).unwrap();
        let result = make_sim().run(&circuit).unwrap();
        let amps = result.state.to_dense_vec();
        let index = 0b011; // x0=1, x1=1, x2=0
        let amp = amps[index];
        assert!((amp.norm() - 1.0).abs() < 1e-10);
        // Phase must be e^{-iγ·coeff·(+1)}
        let expected = num_complex::Complex64::from_polar(1.0, -gamma * coeff);
        assert!(
            (amp - expected).norm() < 1e-10,
            "amp = {:?}, expected {:?}",
            amp,
            expected
        );
    }

    /// The XY mixer must preserve Hamming weight: starting from |0011⟩ the
    /// full distribution stays inside the weight-2 subspace.
    #[test]
    fn test_xy_mixer_preserves_hamming_weight() {
        let config = QAOAConfig {
            depth: 1,
            mixer: MixerType::XY,
            initial_state: InitialState::WarmStart,
            final_mixer: true,
        };
        let builder = QAOACircuitBuilder::with_config(
            ProblemType::MaxCut(Graph::cycle(4)),
            config,
        )
        .with_warm_start(vec![true, true, false, false]);
        let circuit = builder.build(&[0.4, 0.7]).unwrap();
        let result = make_sim().run(&circuit).unwrap();
        let amps = result.state.to_dense_vec();

        let mut outside_weight2 = 0.0;
        let mut moved = 0.0;
        for (idx, amp) in amps.iter().enumerate() {
            let weight = (idx as u32).count_ones();
            if weight != 2 {
                outside_weight2 += amp.norm_sqr();
            } else if idx != 0b0011 {
                moved += amp.norm_sqr();
            }
        }
        assert!(
            outside_weight2 < 1e-10,
            "XY mixer leaked {} probability outside the weight-2 subspace",
            outside_weight2
        );
        // And it must actually mix (this failed when the YY half was missing)
        assert!(moved > 1e-3, "XY mixer did not move any amplitude: {}", moved);
    }

    /// The ring mixer must be an actual mixer: it has to move probability
    /// between basis states (a diagonal ZZ 'mixer' cannot).
    #[test]
    fn test_ring_mixer_actually_mixes() {
        let config = QAOAConfig {
            depth: 1,
            mixer: MixerType::Ring,
            initial_state: InitialState::WarmStart,
            final_mixer: true,
        };
        let builder = QAOACircuitBuilder::with_config(
            ProblemType::MaxCut(Graph::cycle(3)),
            config,
        )
        .with_warm_start(vec![true, false, false]);
        let circuit = builder.build(&[0.0, 0.6]).unwrap();
        let result = make_sim().run(&circuit).unwrap();
        let amps = result.state.to_dense_vec();
        let stayed = amps[0b001].norm_sqr();
        assert!(
            stayed < 1.0 - 1e-3,
            "ring mixer left the initial basis state untouched (p = {})",
            stayed
        );
        // Ring mixer is XY-based, so it also preserves Hamming weight
        let outside: f64 = amps
            .iter()
            .enumerate()
            .filter(|(idx, _)| (*idx as u32).count_ones() != 1)
            .map(|(_, a)| a.norm_sqr())
            .sum();
        assert!(outside < 1e-10, "ring mixer leaked weight: {}", outside);
    }

    /// Malformed Max k-SAT clauses must be rejected, not silently mangled.
    #[test]
    fn test_maxksat_validation() {
        let cases = vec![
            // mismatched negations
            vec![(vec![0, 1], vec![false], 1.0)],
            // out-of-range variable
            vec![(vec![0, 7], vec![false, false], 1.0)],
            // duplicate variable
            vec![(vec![1, 1], vec![false, true], 1.0)],
            // empty clause
            vec![(vec![], vec![], 1.0)],
        ];
        for clauses in cases {
            let builder = QAOACircuitBuilder::new(
                ProblemType::MaxKSat {
                    num_variables: 3,
                    clauses: clauses.clone(),
                },
                MixerType::StandardX,
                1,
            );
            assert!(
                matches!(builder.build(&[0.5, 0.3]).unwrap_err(), QAOAError::InvalidProblem(_)),
                "clauses {:?} were not rejected",
                clauses
            );
        }
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

    #[test]
    fn test_graph_path_star_grid() {
        let path = Graph::path(4);
        assert_eq!(path.num_vertices, 4);
        assert_eq!(path.num_edges(), 3);

        let star = Graph::star(4);
        assert_eq!(star.num_vertices, 4);
        assert_eq!(star.num_edges(), 3);

        let grid = Graph::grid(2, 3);
        assert_eq!(grid.num_vertices, 6);
        assert_eq!(grid.num_edges(), 7);
    }

    #[test]
    fn test_graph_neighbors_empty() {
        let graph = Graph::from_edges(3, &[(0, 1, 1.0)]);
        // Vertex 2 has no neighbors
        assert!(graph.neighbors(2).is_empty());
    }

    #[test]
    fn test_problem_type_descriptions() {
        let g = Graph::cycle(3);
        assert!(ProblemType::MaxCut(g.clone())
            .description()
            .contains("MaxCut"));
        assert!(ProblemType::MinVertexCover(g.clone())
            .description()
            .contains("Vertex Cover"));
        assert!(ProblemType::MaxIndependentSet(g.clone())
            .description()
            .contains("Independent Set"));
        assert!(ProblemType::NumberPartitioning(vec![1.0, 2.0])
            .description()
            .contains("Number Partitioning"));
        assert!(ProblemType::GraphColoring(g.clone(), 3)
            .description()
            .contains("Coloring"));
        assert!(ProblemType::MaxKSat {
            num_variables: 3,
            clauses: vec![(vec![0, 1], vec![false, true], 1.0)],
        }
        .description()
        .contains("SAT"));
        assert!(ProblemType::TSP {
            num_cities: 3,
            distances: vec![]
        }
        .description()
        .contains("TSP"));
        assert!(ProblemType::Portfolio {
            assets: 2,
            expected_returns: vec![],
            covariances: vec![],
            risk_factor: 0.5,
            budget: 1,
        }
        .description()
        .contains("Portfolio"));
        assert!(ProblemType::Custom {
            num_qubits: 2,
            terms: vec![]
        }
        .description()
        .contains("Custom"));
    }

    #[test]
    fn test_problem_type_num_qubits() {
        let g = Graph::cycle(3);
        assert_eq!(ProblemType::MaxCut(g.clone()).num_qubits(), 3);
        assert_eq!(ProblemType::MinVertexCover(g.clone()).num_qubits(), 3);
        assert_eq!(ProblemType::MaxIndependentSet(g.clone()).num_qubits(), 3);
        assert_eq!(ProblemType::NumberPartitioning(vec![1.0, 2.0, 3.0]).num_qubits(), 3);
        assert_eq!(ProblemType::GraphColoring(g.clone(), 2).num_qubits(), 6);
        assert_eq!(
            ProblemType::MaxKSat {
                num_variables: 4,
                clauses: vec![]
            }
            .num_qubits(),
            4
        );
        assert_eq!(
            ProblemType::TSP {
                num_cities: 2,
                distances: vec![]
            }
            .num_qubits(),
            4
        );
        assert_eq!(
            ProblemType::Portfolio {
                assets: 3,
                expected_returns: vec![],
                covariances: vec![],
                risk_factor: 0.0,
                budget: 1,
            }
            .num_qubits(),
            3
        );
        assert_eq!(
            ProblemType::Custom {
                num_qubits: 5,
                terms: vec![]
            }
            .num_qubits(),
            5
        );
    }
}
