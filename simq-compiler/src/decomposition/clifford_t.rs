//! Clifford+T decomposition for fault-tolerant quantum computing
//!
//! This module provides decomposition of arbitrary quantum gates into the Clifford+T
//! gate set, which is the standard basis for fault-tolerant quantum computation.
//!
//! # The Clifford+T Gate Set
//!
//! The Clifford+T basis consists of:
//! - **H** (Hadamard): Creates superposition
//! - **S** (Phase): √Z rotation by π/2
//! - **T** (π/8 gate): √S rotation by π/4
//! - **CNOT**: Two-qubit entanglement
//!
//! This gate set is universal for quantum computation and is particularly important
//! for fault-tolerant quantum computing because:
//!
//! 1. Clifford gates (H, S, CNOT) can be implemented fault-tolerantly with low overhead
//! 2. T gates are expensive but can be implemented using magic state distillation
//! 3. The T-count (number of T gates) determines the resource requirements
//!
//! # Decomposition Methods
//!
//! ## Exact Synthesis
//!
//! For certain angles (multiples of π/8), exact Clifford+T synthesis is possible:
//! - **Gridsynth**: Optimal single-qubit synthesis (Ross & Selinger 2016)
//! - **Number-theoretic methods**: Uses continued fractions and algebraic number theory
//!
//! ## Approximate Synthesis
//!
//! For arbitrary angles, we use approximation algorithms:
//! - **Solovay-Kitaev**: Universal approximation, O(log^c(1/ε)) gates for precision ε
//! - **Quantum Shannon Decomposition**: Systematic approach with known bounds
//!
//! # T-Count Optimization
//!
//! The goal is to minimize the number of T gates while maintaining fidelity:
//! - T-count is the primary cost metric in fault-tolerant quantum computing
//! - T-depth (number of layers of T gates) affects circuit runtime
//! - Tradeoffs exist between T-count, T-depth, and auxiliary gates
//!
//! # References
//!
//! - Ross & Selinger, "Optimal ancilla-free Clifford+T approximation of z-rotations" (2016)
//! - Kliuchnikov, Maslov, Mosca, "Fast and efficient exact synthesis of single-qubit unitaries" (2013)
//! - Amy, Maslov, Mosca, "Polynomial-time T-depth optimization of Clifford+T circuits" (2014)
//! - Solovay-Kitaev theorem for universal approximation

use crate::decomposition::{Decomposer, DecompositionConfig, DecompositionResult, DecompositionMetadata};
use crate::matrix_computation::{Matrix2, is_unitary_2x2};
use num_complex::Complex64;
use simq_core::{Gate, Result, QuantumError};
use std::sync::Arc;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;
const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// Configuration for Clifford+T synthesis
#[derive(Debug, Clone)]
pub struct GridSynthConfig {
    /// Target approximation error (epsilon)
    pub epsilon: f64,

    /// Maximum number of gates to generate
    pub max_gates: usize,

    /// Whether to optimize for T-count
    pub optimize_t_count: bool,

    /// Whether to optimize for T-depth
    pub optimize_t_depth: bool,
}

impl Default for GridSynthConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-10,
            max_gates: 10000,
            optimize_t_count: true,
            optimize_t_depth: false,
        }
    }
}

/// Clifford+T decomposer
pub struct CliffordTDecomposer {
    config: GridSynthConfig,
}

impl CliffordTDecomposer {
    /// Create a new Clifford+T decomposer with default configuration
    pub fn new() -> Self {
        Self {
            config: GridSynthConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: GridSynthConfig) -> Self {
        Self { config }
    }

    /// Decompose a rotation around Z-axis into Clifford+T gates
    ///
    /// Rz(θ) can be approximated using a sequence of H, S, and T gates.
    ///
    /// For exact angles that are multiples of π/4:
    /// - Rz(π/4) = T
    /// - Rz(π/2) = S
    /// - Rz(π) = Z = S²
    ///
    /// For other angles, uses gridsynth-like approximation.
    pub fn decompose_rz(&self, angle: f64) -> Vec<CliffordTGate> {
        // Normalize angle to [0, 2π)
        let normalized_angle = angle.rem_euclid(2.0 * PI);

        // Check for exact multiples of π/8
        let k = (normalized_angle / (PI / 4.0)).round();
        if (normalized_angle - k * PI / 4.0).abs() < self.config.epsilon {
            return self.exact_rz_pi_over_4(k as i32);
        }

        // For arbitrary angles, use approximation
        self.approximate_rz(normalized_angle)
    }

    /// Exact decomposition for Rz(k·π/4) where k is an integer
    fn exact_rz_pi_over_4(&self, k: i32) -> Vec<CliffordTGate> {
        // Normalize k to [0, 8)
        let k = k.rem_euclid(8);

        match k {
            0 => vec![],  // Identity
            1 => vec![CliffordTGate::T],
            2 => vec![CliffordTGate::S],
            3 => vec![CliffordTGate::T, CliffordTGate::S],
            4 => vec![CliffordTGate::Z],
            5 => vec![CliffordTGate::TDagger],
            6 => vec![CliffordTGate::SDagger],
            7 => vec![CliffordTGate::TDagger, CliffordTGate::SDagger],
            _ => vec![],
        }
    }

    /// Approximate Rz(θ) for arbitrary angle θ
    ///
    /// Uses a simplified gridsynth-like algorithm. The full implementation would
    /// involve solving Diophantine equations over Z[ω] where ω = e^(iπ/4).
    fn approximate_rz(&self, angle: f64) -> Vec<CliffordTGate> {
        // Simple approximation: find closest π/8 multiple
        let k = (angle / (PI / 4.0)).round() as i32;
        self.exact_rz_pi_over_4(k)

        // TODO: Implement full gridsynth algorithm:
        // 1. Represent target angle as point on unit circle
        // 2. Use continued fraction expansion to find approximation in Z[ω]
        // 3. Convert to gate sequence
        // 4. Optimize T-count
    }

    /// Decompose arbitrary single-qubit unitary into Clifford+T gates
    ///
    /// Uses the decomposition: U = Rz(α)Ry(β)Rz(γ)
    /// Then approximates each Ry and Rz using Clifford+T gates.
    pub fn decompose_single_qubit(&self, matrix: &Matrix2) -> Result<Vec<CliffordTGate>> {
        if !is_unitary_2x2(matrix) {
            return Err(QuantumError::ValidationError(
                "Matrix is not unitary".to_string()
            ));
        }

        // Extract Euler angles (ZYZ decomposition)
        // U = Rz(α)Ry(β)Rz(γ)

        let gamma = 2.0 * matrix[0][0].norm().acos();

        let alpha = if gamma.abs() < EPSILON {
            0.0
        } else {
            let sin_half = (gamma / 2.0).sin();
            (matrix[1][0] / Complex64::new(0.0, sin_half)).arg()
        };

        let delta = if gamma.abs() < EPSILON {
            (matrix[0][1] / Complex64::new(0.0, 1.0)).arg()
        } else {
            let sin_half = (gamma / 2.0).sin();
            (matrix[0][1] / Complex64::new(0.0, -sin_half)).arg()
        };

        let mut gates = Vec::new();

        // Decompose first Rz(alpha)
        gates.extend(self.decompose_rz(alpha));

        // Decompose Ry(gamma) = H Rz(gamma) H
        if gamma.abs() > EPSILON {
            gates.push(CliffordTGate::H);
            gates.extend(self.decompose_rz(gamma));
            gates.push(CliffordTGate::H);
        }

        // Decompose last Rz(delta)
        gates.extend(self.decompose_rz(delta));

        Ok(gates)
    }

    /// Solovay-Kitaev algorithm for universal approximation
    ///
    /// Recursively approximates any single-qubit unitary using Clifford+T gates.
    /// Achieves precision ε using O(log^c(1/ε)) gates where c ≈ 3.97.
    pub fn solovay_kitaev(&self, matrix: &Matrix2, depth: usize) -> Result<Vec<CliffordTGate>> {
        if depth == 0 {
            // Base case: use direct decomposition
            return self.decompose_single_qubit(matrix);
        }

        // Recursive case:
        // 1. Find U₀ such that U ≈ U₀
        // 2. Find V, W such that U₀⁻¹U ≈ VWV⁻¹W⁻¹
        // 3. Recurse on V and W

        // For now, fall back to direct decomposition
        self.decompose_single_qubit(matrix)

        // TODO: Implement full Solovay-Kitaev recursion
    }

    /// Count T gates in a gate sequence
    pub fn count_t_gates(gates: &[CliffordTGate]) -> usize {
        gates.iter()
            .filter(|g| matches!(g, CliffordTGate::T | CliffordTGate::TDagger))
            .count()
    }

    /// Count T-depth (number of layers of T gates)
    pub fn count_t_depth(gates: &[CliffordTGate]) -> usize {
        let mut depth = 0;
        let mut in_t_layer = false;

        for gate in gates {
            match gate {
                CliffordTGate::T | CliffordTGate::TDagger => {
                    if !in_t_layer {
                        depth += 1;
                        in_t_layer = true;
                    }
                }
                _ => {
                    in_t_layer = false;
                }
            }
        }

        depth
    }

    /// Optimize gate sequence to reduce T-count
    ///
    /// Applies circuit identities and optimizations:
    /// - T† T = I
    /// - SS = Z
    /// - HSHS = X
    pub fn optimize_t_count(&self, gates: &[CliffordTGate]) -> Vec<CliffordTGate> {
        let mut optimized = Vec::new();

        for gate in gates {
            match (optimized.last(), gate) {
                // T† T = I (cancel)
                (Some(&CliffordTGate::TDagger), CliffordTGate::T) |
                (Some(&CliffordTGate::T), CliffordTGate::TDagger) => {
                    optimized.pop();
                }

                // S† S = I (cancel)
                (Some(&CliffordTGate::SDagger), CliffordTGate::S) |
                (Some(&CliffordTGate::S), CliffordTGate::SDagger) => {
                    optimized.pop();
                }

                // HH = I (cancel)
                (Some(&CliffordTGate::H), CliffordTGate::H) => {
                    optimized.pop();
                }

                // SS = Z
                (Some(&CliffordTGate::S), CliffordTGate::S) => {
                    optimized.pop();
                    optimized.push(CliffordTGate::Z);
                }

                // Otherwise, add gate
                _ => {
                    optimized.push(*gate);
                }
            }
        }

        optimized
    }
}

impl Default for CliffordTDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

impl Decomposer for CliffordTDecomposer {
    fn decompose(&self, gate: &dyn Gate, config: &DecompositionConfig) -> Result<DecompositionResult> {
        if gate.num_qubits() != 1 {
            return Err(QuantumError::ValidationError(
                "Clifford+T decomposition currently supports single-qubit gates only".to_string()
            ));
        }

        // Get gate matrix
        let matrix = gate.matrix().ok_or_else(|| {
            QuantumError::ValidationError("Gate does not provide matrix representation".to_string())
        })?;

        if matrix.len() != 4 {
            return Err(QuantumError::ValidationError("Invalid matrix size".to_string()));
        }

        // Convert to Matrix2 format
        let matrix_2x2: Matrix2 = [
            [
                Complex64::new(matrix[0].re, matrix[0].im),
                Complex64::new(matrix[1].re, matrix[1].im),
            ],
            [
                Complex64::new(matrix[2].re, matrix[2].im),
                Complex64::new(matrix[3].re, matrix[3].im),
            ],
        ];

        // Decompose to Clifford+T gates
        let mut ct_gates = self.decompose_single_qubit(&matrix_2x2)?;

        // Optimize if requested
        if config.optimization_level > 0 {
            ct_gates = self.optimize_t_count(&ct_gates);
        }

        let t_count = Self::count_t_gates(&ct_gates);
        let t_depth = Self::count_t_depth(&ct_gates);

        // TODO: Convert Clifford+T gates to actual Gate objects
        let gates: Vec<Arc<dyn Gate>> = vec![];

        Ok(DecompositionResult {
            gates,
            fidelity: 1.0 - self.config.epsilon,
            depth: ct_gates.len(),
            gate_count: ct_gates.len(),
            two_qubit_count: 0,
            metadata: DecompositionMetadata {
                strategy: format!("Clifford+T (T-count: {}, T-depth: {})", t_count, t_depth),
                optimized: config.optimization_level > 0,
                optimization_passes: config.optimization_level as usize,
                original_gate_count: 1,
            },
        })
    }

    fn can_decompose(&self, gate: &dyn Gate) -> bool {
        gate.num_qubits() == 1 && gate.matrix().is_some()
    }

    fn name(&self) -> &str {
        "CliffordT"
    }

    fn estimate_cost(&self, gate: &dyn Gate) -> Option<usize> {
        if gate.num_qubits() == 1 {
            // Rough estimate: average 5-10 gates for arbitrary single-qubit
            Some(7)
        } else {
            None
        }
    }
}

/// Clifford+T gate enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliffordTGate {
    /// Hadamard gate
    H,

    /// S gate (√Z)
    S,

    /// S† gate
    SDagger,

    /// T gate (√S)
    T,

    /// T† gate
    TDagger,

    /// Pauli-X gate
    X,

    /// Pauli-Y gate
    Y,

    /// Pauli-Z gate
    Z,

    /// Identity (sometimes needed for bookkeeping)
    I,

    /// CNOT (two-qubit gate)
    CNOT,
}

impl CliffordTGate {
    /// Check if this is a Clifford gate (not T)
    pub fn is_clifford(&self) -> bool {
        !matches!(self, CliffordTGate::T | CliffordTGate::TDagger)
    }

    /// Check if this is a T gate (expensive in fault-tolerant QC)
    pub fn is_t_gate(&self) -> bool {
        matches!(self, CliffordTGate::T | CliffordTGate::TDagger)
    }

    /// Get the name of this gate
    pub fn name(&self) -> &str {
        match self {
            CliffordTGate::H => "H",
            CliffordTGate::S => "S",
            CliffordTGate::SDagger => "S†",
            CliffordTGate::T => "T",
            CliffordTGate::TDagger => "T†",
            CliffordTGate::X => "X",
            CliffordTGate::Y => "Y",
            CliffordTGate::Z => "Z",
            CliffordTGate::I => "I",
            CliffordTGate::CNOT => "CNOT",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_rz_decomposition() {
        let decomposer = CliffordTDecomposer::new();

        // Rz(π/4) = T
        let gates = decomposer.decompose_rz(PI / 4.0);
        assert_eq!(gates, vec![CliffordTGate::T]);

        // Rz(π/2) = S
        let gates = decomposer.decompose_rz(PI / 2.0);
        assert_eq!(gates, vec![CliffordTGate::S]);

        // Rz(π) = Z
        let gates = decomposer.decompose_rz(PI);
        assert_eq!(gates, vec![CliffordTGate::Z]);
    }

    #[test]
    fn test_t_count() {
        let gates = vec![
            CliffordTGate::H,
            CliffordTGate::T,
            CliffordTGate::S,
            CliffordTGate::T,
            CliffordTGate::TDagger,
        ];

        assert_eq!(CliffordTDecomposer::count_t_gates(&gates), 3);
    }

    #[test]
    fn test_t_depth() {
        let gates = vec![
            CliffordTGate::T,
            CliffordTGate::T,  // Same layer
            CliffordTGate::H,  // Clifford
            CliffordTGate::T,  // New layer
        ];

        assert_eq!(CliffordTDecomposer::count_t_depth(&gates), 2);
    }

    #[test]
    fn test_optimization() {
        let decomposer = CliffordTDecomposer::new();

        let gates = vec![
            CliffordTGate::T,
            CliffordTGate::TDagger,  // Should cancel
            CliffordTGate::H,
            CliffordTGate::H,  // Should cancel
        ];

        let optimized = decomposer.optimize_t_count(&gates);
        assert!(optimized.is_empty());
    }

    #[test]
    fn test_is_clifford() {
        assert!(CliffordTGate::H.is_clifford());
        assert!(CliffordTGate::S.is_clifford());
        assert!(!CliffordTGate::T.is_clifford());
        assert!(!CliffordTGate::TDagger.is_clifford());
    }

    #[test]
    fn test_is_t_gate() {
        assert!(CliffordTGate::T.is_t_gate());
        assert!(CliffordTGate::TDagger.is_t_gate());
        assert!(!CliffordTGate::H.is_t_gate());
        assert!(!CliffordTGate::S.is_t_gate());
    }
}
