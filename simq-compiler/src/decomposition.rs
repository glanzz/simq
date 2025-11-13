//! Gate decomposition for quantum circuits
//!
//! This module provides comprehensive gate decomposition functionality for translating
//! quantum gates into different basis gate sets. Decomposition is essential for:
//!
//! - **Hardware compilation**: Mapping gates to native gate sets of quantum processors
//! - **Circuit optimization**: Expressing gates in fault-tolerant bases (e.g., Clifford+T)
//! - **Gate synthesis**: Breaking down complex gates into elementary operations
//! - **Cross-platform compatibility**: Translating between different quantum frameworks
//!
//! # Architecture
//!
//! The decomposition system is organized into several specialized modules:
//!
//! - [`basis`]: Defines basis gate sets (IBM, Google, IonQ, Clifford+T, etc.)
//! - [`single_qubit`]: Single-qubit gate decompositions (ZYZ, ZXZ, U3, etc.)
//! - [`two_qubit`]: Two-qubit gate decompositions (CNOT synthesis, CZ, SWAP)
//! - [`multi_qubit`]: Multi-controlled gate decompositions (Toffoli, Fredkin)
//! - [`clifford_t`]: Clifford+T decomposition for fault-tolerant quantum computing
//!
//! # Decomposition Strategies
//!
//! ## Single-Qubit Gates
//!
//! Any single-qubit unitary U can be decomposed using Euler angle decompositions:
//!
//! - **ZYZ**: U = e^(iα) Rz(β) Ry(γ) Rz(δ)
//! - **ZXZ**: U = e^(iα) Rz(β) Rx(γ) Rz(δ)
//! - **XYX**: U = e^(iα) Rx(β) Ry(γ) Rx(δ)
//! - **U3**: U = Rz(φ) Ry(θ) Rz(λ) (IBM's native gate)
//!
//! For specific basis gates like {H, T}:
//! - Solovay-Kitaev algorithm for approximation
//! - Gridsynth/Ross-Selinger for exact synthesis
//!
//! ## Two-Qubit Gates
//!
//! Two-qubit gates can be decomposed into CNOT + single-qubit gates:
//!
//! - **Canonical decomposition**: Any 2-qubit unitary requires at most 3 CNOTs
//! - **CZ decomposition**: CZ = H₁ CNOT H₁
//! - **SWAP decomposition**: SWAP = CNOT₀₁ CNOT₁₀ CNOT₀₁
//! - **iSWAP decomposition**: Uses CNOT + rotations
//!
//! ## Multi-Qubit Gates
//!
//! - **Toffoli (CCX)**: 6 CNOTs + T gates (Clifford+T)
//! - **Fredkin (CSWAP)**: Decomposed into Toffoli + CNOTs
//! - **Multi-controlled gates**: Linear or logarithmic decomposition using ancillas
//!
//! # Example
//!
//! ```ignore
//! use simq_compiler::decomposition::{DecompositionConfig, BasisGateSet, decompose_circuit};
//!
//! // Configure decomposition to IBM basis
//! let config = DecompositionConfig {
//!     basis: BasisGateSet::IBM,
//!     optimization_level: 2,
//!     fidelity_threshold: 0.9999,
//!     ..Default::default()
//! };
//!
//! // Decompose circuit
//! let decomposed = decompose_circuit(&circuit, &config)?;
//! ```
//!
//! # Performance
//!
//! - **Compile-time optimization**: Common angles use cached matrices
//! - **High fidelity**: Numerical precision < 10^-10 for exact decompositions
//! - **Optimized sequences**: Minimizes gate count and circuit depth
//!
//! # References
//!
//! - Nielsen & Chuang, "Quantum Computation and Quantum Information"
//! - Barenco et al., "Elementary gates for quantum computation" (1995)
//! - Shende, Bullock, Markov, "Synthesis of quantum-logic circuits" (2006)
//! - Solovay-Kitaev theorem for universal approximation
//! - Ross & Selinger, "Optimal ancilla-free Clifford+T approximation" (2016)

pub mod basis;
pub mod single_qubit;
pub mod two_qubit;
pub mod multi_qubit;
pub mod clifford_t;

use num_complex::Complex64;
use simq_core::{Gate, Result, QuantumError};
use std::sync::Arc;

// Re-export main types
pub use basis::{BasisGateSet, BasisGate};
pub use single_qubit::{SingleQubitDecomposer, EulerBasis};
pub use two_qubit::{TwoQubitDecomposer, EntanglementGate};
pub use multi_qubit::MultiQubitDecomposer;
pub use clifford_t::{CliffordTDecomposer, GridSynthConfig};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for gate decomposition
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    /// Target basis gate set
    pub basis: BasisGateSet,

    /// Optimization level (0-3)
    /// - 0: No optimization, direct decomposition
    /// - 1: Basic optimization (merge adjacent rotations)
    /// - 2: Advanced optimization (circuit identities, commutation)
    /// - 3: Aggressive optimization (numerical search, Solovay-Kitaev)
    pub optimization_level: u8,

    /// Maximum circuit depth (None = unlimited)
    pub max_depth: Option<usize>,

    /// Minimum fidelity threshold (0.0 to 1.0)
    /// Decomposition fails if fidelity < threshold
    pub fidelity_threshold: f64,

    /// Maximum number of gates in decomposition (None = unlimited)
    pub max_gates: Option<usize>,

    /// Use ancilla qubits for optimization
    pub allow_ancillas: bool,

    /// Number of ancilla qubits available
    pub num_ancillas: usize,

    /// Precision for Clifford+T synthesis (epsilon)
    pub clifford_t_epsilon: f64,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            basis: BasisGateSet::Universal,
            optimization_level: 1,
            max_depth: None,
            fidelity_threshold: 0.9999,
            max_gates: None,
            allow_ancillas: false,
            num_ancillas: 0,
            clifford_t_epsilon: 1e-10,
        }
    }
}

// ============================================================================
// Decomposition Result
// ============================================================================

/// Result of gate decomposition
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// Sequence of gates in the decomposition
    pub gates: Vec<Arc<dyn Gate>>,

    /// Fidelity of the decomposition (1.0 = perfect)
    pub fidelity: f64,

    /// Circuit depth of the decomposition
    pub depth: usize,

    /// Total gate count
    pub gate_count: usize,

    /// Number of two-qubit gates (important metric)
    pub two_qubit_count: usize,

    /// Metadata about the decomposition
    pub metadata: DecompositionMetadata,
}

/// Metadata about the decomposition process
#[derive(Debug, Clone, Default)]
pub struct DecompositionMetadata {
    /// Which decomposition strategy was used
    pub strategy: String,

    /// Whether optimization was applied
    pub optimized: bool,

    /// Number of optimization passes applied
    pub optimization_passes: usize,

    /// Original gate count before optimization
    pub original_gate_count: usize,
}

// ============================================================================
// Main Decomposer Trait
// ============================================================================

/// Trait for gate decomposition strategies
pub trait Decomposer: Send + Sync {
    /// Decompose a gate into a sequence of basis gates
    fn decompose(&self, gate: &dyn Gate, config: &DecompositionConfig) -> Result<DecompositionResult>;

    /// Check if this decomposer can handle the given gate
    fn can_decompose(&self, gate: &dyn Gate) -> bool;

    /// Get the name of this decomposition strategy
    fn name(&self) -> &str;

    /// Estimate the cost of decomposition (gate count)
    fn estimate_cost(&self, gate: &dyn Gate) -> Option<usize> {
        None
    }
}

// ============================================================================
// Universal Decomposer
// ============================================================================

/// Universal decomposer that dispatches to specialized decomposers
pub struct UniversalDecomposer {
    config: DecompositionConfig,
}

impl UniversalDecomposer {
    /// Create a new universal decomposer with given configuration
    pub fn new(config: DecompositionConfig) -> Self {
        Self { config }
    }

    /// Decompose a gate using the appropriate strategy
    pub fn decompose_gate(&self, gate: &dyn Gate) -> Result<DecompositionResult> {
        let num_qubits = gate.num_qubits();

        match num_qubits {
            1 => {
                let decomposer = SingleQubitDecomposer::new(EulerBasis::ZYZ);
                decomposer.decompose(gate, &self.config)
            }
            2 => {
                let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
                decomposer.decompose(gate, &self.config)
            }
            _ => {
                let decomposer = MultiQubitDecomposer::new();
                decomposer.decompose(gate, &self.config)
            }
        }
    }

    /// Decompose a sequence of gates
    pub fn decompose_gates(&self, gates: &[Arc<dyn Gate>]) -> Result<Vec<DecompositionResult>> {
        gates.iter()
            .map(|g| self.decompose_gate(g.as_ref()))
            .collect()
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute fidelity between two unitary matrices
///
/// Fidelity F = |Tr(U†V)| / d, where d is the matrix dimension
pub fn compute_fidelity(u: &[Vec<Complex64>], v: &[Vec<Complex64>]) -> Result<f64> {
    if u.len() != v.len() || u.is_empty() {
        return Err(QuantumError::ValidationError("Matrix dimensions mismatch".to_string()));
    }

    let dim = u.len();
    let mut trace = Complex64::new(0.0, 0.0);

    for i in 0..dim {
        for j in 0..dim {
            if j >= u[i].len() || j >= v[i].len() {
                return Err(QuantumError::ValidationError("Invalid matrix structure".to_string()));
            }
            trace += u[j][i].conj() * v[j][i];
        }
    }

    Ok((trace.norm() / dim as f64).min(1.0))
}

/// Validate that a decomposition meets the fidelity threshold
pub fn validate_decomposition(
    original: &[Vec<Complex64>],
    decomposed: &[Vec<Complex64>],
    threshold: f64,
) -> Result<f64> {
    let fidelity = compute_fidelity(original, decomposed)?;

    if fidelity < threshold {
        return Err(QuantumError::ValidationError(
            format!("Decomposition fidelity {} below threshold {}", fidelity, threshold)
        ));
    }

    Ok(fidelity)
}

/// Optimize a gate sequence by merging adjacent rotations
pub fn optimize_gate_sequence(gates: Vec<Arc<dyn Gate>>, level: u8) -> Vec<Arc<dyn Gate>> {
    if level == 0 {
        return gates;
    }

    // TODO: Implement gate fusion and optimization
    // For now, return gates as-is
    gates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DecompositionConfig::default();
        assert_eq!(config.optimization_level, 1);
        assert!(config.fidelity_threshold > 0.999);
    }

    #[test]
    fn test_fidelity_identical() {
        let m1 = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let m2 = m1.clone();

        let fidelity = compute_fidelity(&m1, &m2).unwrap();
        assert!((fidelity - 1.0).abs() < 1e-10);
    }
}
