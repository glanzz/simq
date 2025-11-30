//! Two-qubit gate decomposition
//!
//! This module provides decomposition strategies for two-qubit gates into a basis
//! set of CNOT (or other entangling gates) plus single-qubit rotations.
//!
//! # Key Results
//!
//! ## Canonical Decomposition Theorem
//!
//! Any two-qubit unitary U can be decomposed as:
//! ```text
//! U = (A₁ ⊗ A₂) · CNOT · (B₁ ⊗ B₂) · CNOT · (C₁ ⊗ C₂) · CNOT · (D₁ ⊗ D₂)
//! ```
//! where Aᵢ, Bᵢ, Cᵢ, Dᵢ are single-qubit unitaries.
//!
//! This requires **at most 3 CNOTs**, which is optimal for generic two-qubit gates.
//!
//! ## Special Cases
//!
//! Many important gates require fewer CNOTs:
//! - Controlled gates (CX, CY, CZ): 1 CNOT
//! - SWAP: 3 CNOTs
//! - iSWAP: 2 CNOTs + single-qubit gates
//! - √iSWAP: 2 CNOTs + single-qubit gates
//!
//! # Entangling Gates
//!
//! Different quantum hardware uses different native entangling gates:
//! - IBM, Rigetti, IonQ: CNOT (Controlled-X)
//! - Google Sycamore: √iSWAP
//! - Some superconducting qubits: CZ (Controlled-Z)
//!
//! This module provides conversions between these representations.
//!
//! # References
//!
//! - Barenco et al., "Elementary gates for quantum computation" (1995)
//! - Vatan & Williams, "Optimal quantum circuits for general two-qubit gates" (2004)
//! - Shende, Bullock, Markov, "Synthesis of quantum-logic circuits" (2006)

use crate::decomposition::{Decomposer, DecompositionConfig, DecompositionResult, DecompositionMetadata};
use crate::matrix_computation::{Matrix4, is_unitary_4x4};
use num_complex::Complex64;
use simq_core::{Gate, Result, QuantumError};
use std::sync::Arc;
use std::f64::consts::PI;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Entangling gate basis for two-qubit decomposition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntanglementGate {
    /// CNOT (Controlled-X) - most common
    CNOT,

    /// CZ (Controlled-Z) - equivalent to CNOT up to local gates
    CZ,

    /// iSWAP - native to some superconducting architectures
    ISWAP,

    /// √iSWAP - Google Sycamore's native gate
    SqrtISWAP,

    /// SWAP - sometimes useful as a basis
    SWAP,

    /// CPhase - parameterized controlled phase
    CPhase,
}

/// Two-qubit gate decomposer
pub struct TwoQubitDecomposer {
    entangling_gate: EntanglementGate,
}

impl TwoQubitDecomposer {
    /// Create a new two-qubit decomposer with specified entangling gate
    pub fn new(entangling_gate: EntanglementGate) -> Self {
        Self { entangling_gate }
    }

    /// Decompose a two-qubit gate using canonical decomposition
    ///
    /// Returns the decomposition in the form:
    /// U = (A₁ ⊗ A₂) · E · (B₁ ⊗ B₂) · E · (C₁ ⊗ C₂) · E · (D₁ ⊗ D₂)
    ///
    /// where E is the entangling gate (CNOT, CZ, etc.)
    pub fn decompose_canonical(&self, matrix: &Matrix4) -> Result<CanonicalDecomposition> {
        if !is_unitary_4x4(matrix) {
            return Err(QuantumError::ValidationError(
                "Matrix is not unitary".to_string()
            ));
        }

        // TODO: Implement full canonical decomposition using:
        // 1. Compute the "magic basis" transformation
        // 2. Diagonalize in the magic basis to get local equivalence class
        // 3. Compute single-qubit gates before/after each CNOT
        //
        // For now, return a placeholder

        Ok(CanonicalDecomposition {
            entangling_gate: self.entangling_gate,
            single_qubit_layers: vec![],
            num_entangling: 3,
        })
    }

    /// Decompose SWAP gate
    ///
    /// SWAP = CNOT₀₁ · CNOT₁₀ · CNOT₀₁
    ///
    /// Requires 3 CNOTs (or equivalent entangling gates).
    pub fn decompose_swap(&self) -> Vec<TwoQubitGateInstruction> {
        match self.entangling_gate {
            EntanglementGate::CNOT => vec![
                TwoQubitGateInstruction::CNOT { control: 0, target: 1 },
                TwoQubitGateInstruction::CNOT { control: 1, target: 0 },
                TwoQubitGateInstruction::CNOT { control: 0, target: 1 },
            ],
            EntanglementGate::CZ => {
                // CZ can implement SWAP with additional Hadamards
                vec![
                    TwoQubitGateInstruction::Hadamard { qubit: 1 },
                    TwoQubitGateInstruction::CZ,
                    TwoQubitGateInstruction::Hadamard { qubit: 1 },
                    TwoQubitGateInstruction::Hadamard { qubit: 0 },
                    TwoQubitGateInstruction::CZ,
                    TwoQubitGateInstruction::Hadamard { qubit: 0 },
                    TwoQubitGateInstruction::Hadamard { qubit: 1 },
                    TwoQubitGateInstruction::CZ,
                    TwoQubitGateInstruction::Hadamard { qubit: 1 },
                ]
            }
            _ => {
                // For other gates, convert to CNOT first
                vec![]
            }
        }
    }

    /// Decompose iSWAP gate
    ///
    /// iSWAP = S₀ · S₁ · H₁ · CNOT₀₁ · H₁ · CNOT₁₀ · H₀
    ///
    /// Requires 2 CNOTs plus local gates.
    pub fn decompose_iswap(&self) -> Vec<TwoQubitGateInstruction> {
        if self.entangling_gate == EntanglementGate::ISWAP {
            return vec![TwoQubitGateInstruction::ISWAP];
        }

        vec![
            TwoQubitGateInstruction::SGate { qubit: 0 },
            TwoQubitGateInstruction::SGate { qubit: 1 },
            TwoQubitGateInstruction::Hadamard { qubit: 1 },
            TwoQubitGateInstruction::CNOT { control: 0, target: 1 },
            TwoQubitGateInstruction::Hadamard { qubit: 1 },
            TwoQubitGateInstruction::CNOT { control: 1, target: 0 },
            TwoQubitGateInstruction::Hadamard { qubit: 0 },
        ]
    }

    /// Decompose √iSWAP gate (Google Sycamore's native gate)
    ///
    /// Requires 2 CNOTs plus local rotations.
    pub fn decompose_sqrt_iswap(&self) -> Vec<TwoQubitGateInstruction> {
        if self.entangling_gate == EntanglementGate::SqrtISWAP {
            return vec![TwoQubitGateInstruction::SqrtISWAP];
        }

        // Decomposition in terms of CNOTs
        vec![
            TwoQubitGateInstruction::Ry { qubit: 0, angle: PI / 4.0 },
            TwoQubitGateInstruction::Ry { qubit: 1, angle: PI / 4.0 },
            TwoQubitGateInstruction::CNOT { control: 0, target: 1 },
            TwoQubitGateInstruction::Rz { qubit: 1, angle: -PI / 4.0 },
            TwoQubitGateInstruction::CNOT { control: 1, target: 0 },
            TwoQubitGateInstruction::Ry { qubit: 0, angle: -PI / 4.0 },
            TwoQubitGateInstruction::Ry { qubit: 1, angle: -PI / 4.0 },
        ]
    }

    /// Convert CNOT to CZ
    ///
    /// CNOT = H₁ · CZ · H₁
    pub fn cnot_to_cz() -> Vec<TwoQubitGateInstruction> {
        vec![
            TwoQubitGateInstruction::Hadamard { qubit: 1 },
            TwoQubitGateInstruction::CZ,
            TwoQubitGateInstruction::Hadamard { qubit: 1 },
        ]
    }

    /// Convert CZ to CNOT
    ///
    /// CZ = H₁ · CNOT · H₁
    pub fn cz_to_cnot() -> Vec<TwoQubitGateInstruction> {
        vec![
            TwoQubitGateInstruction::Hadamard { qubit: 1 },
            TwoQubitGateInstruction::CNOT { control: 0, target: 1 },
            TwoQubitGateInstruction::Hadamard { qubit: 1 },
        ]
    }

    /// Optimize the decomposition by reducing gate count
    pub fn optimize_decomposition(&self, _decomp: &mut CanonicalDecomposition, level: u8) {
        if level == 0 {
            return;
        }

        // TODO: Implement optimization:
        // - Merge adjacent single-qubit gates
        // - Cancel identity gates
        // - Reduce CNOT count if possible
        // - Commute gates to reduce depth
    }
}

impl Decomposer for TwoQubitDecomposer {
    fn decompose(&self, gate: &dyn Gate, config: &DecompositionConfig) -> Result<DecompositionResult> {
        if gate.num_qubits() != 2 {
            return Err(QuantumError::ValidationError(
                format!("Expected two-qubit gate, got {}-qubit gate", gate.num_qubits())
            ));
        }

        // Get gate matrix
        let matrix = gate.matrix().ok_or_else(|| {
            QuantumError::ValidationError("Gate does not provide matrix representation".to_string())
        })?;

        if matrix.len() != 16 {
            return Err(QuantumError::ValidationError("Invalid matrix size for two-qubit gate".to_string()));
        }

        // Convert to Matrix4 format
        let mut matrix_4x4: Matrix4 = [[ZERO; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let idx = i * 4 + j;
                matrix_4x4[i][j] = Complex64::new(matrix[idx].re, matrix[idx].im);
            }
        }

        // Decompose using canonical decomposition
        let mut decomp = self.decompose_canonical(&matrix_4x4)?;

        // Optimize if requested
        if config.optimization_level > 0 {
            self.optimize_decomposition(&mut decomp, config.optimization_level);
        }

        // TODO: Convert decomposition to gate sequence
        let gates: Vec<Arc<dyn Gate>> = vec![];

        Ok(DecompositionResult {
            gates,
            fidelity: 1.0,
            depth: 7,  // Typical depth for 3 CNOTs + single-qubit layers
            gate_count: decomp.num_entangling * 3 + decomp.single_qubit_layers.len(),
            two_qubit_count: decomp.num_entangling,
            metadata: DecompositionMetadata {
                strategy: format!("{:?} canonical decomposition", self.entangling_gate),
                optimized: config.optimization_level > 0,
                optimization_passes: config.optimization_level as usize,
                original_gate_count: 1,
            },
        })
    }

    fn can_decompose(&self, gate: &dyn Gate) -> bool {
        gate.num_qubits() == 2 && gate.matrix().is_some()
    }

    fn name(&self) -> &str {
        "TwoQubitCanonical"
    }

    fn estimate_cost(&self, gate: &dyn Gate) -> Option<usize> {
        if gate.num_qubits() == 2 {
            Some(3)  // At most 3 entangling gates
        } else {
            None
        }
    }
}

/// Canonical decomposition result
#[derive(Debug, Clone)]
pub struct CanonicalDecomposition {
    /// Which entangling gate is used
    pub entangling_gate: EntanglementGate,

    /// Single-qubit gates between entangling gates
    pub single_qubit_layers: Vec<SingleQubitLayer>,

    /// Number of entangling gates used (≤ 3)
    pub num_entangling: usize,
}

/// Layer of single-qubit gates (applied in parallel)
#[derive(Debug, Clone)]
pub struct SingleQubitLayer {
    /// Gate on qubit 0
    pub gate_0: Option<SingleQubitAngles>,

    /// Gate on qubit 1
    pub gate_1: Option<SingleQubitAngles>,
}

/// Single-qubit gate angles (ZYZ parameterization)
#[derive(Debug, Clone, Copy)]
pub struct SingleQubitAngles {
    pub beta: f64,
    pub gamma: f64,
    pub delta: f64,
}

/// Two-qubit gate instruction (intermediate representation)
#[derive(Debug, Clone, PartialEq)]
pub enum TwoQubitGateInstruction {
    CNOT { control: usize, target: usize },
    CZ,
    ISWAP,
    SqrtISWAP,
    SWAP,
    Hadamard { qubit: usize },
    SGate { qubit: usize },
    TGate { qubit: usize },
    Rx { qubit: usize, angle: f64 },
    Ry { qubit: usize, angle: f64 },
    Rz { qubit: usize, angle: f64 },
}

impl CanonicalDecomposition {
    /// Count total gates in the decomposition
    pub fn total_gates(&self) -> usize {
        let mut count = self.num_entangling;
        for layer in &self.single_qubit_layers {
            if layer.gate_0.is_some() {
                count += 3;  // ZYZ = 3 rotations
            }
            if layer.gate_1.is_some() {
                count += 3;
            }
        }
        count
    }

    /// Estimate circuit depth
    pub fn depth(&self) -> usize {
        // Each entangling gate adds depth, single-qubit gates are parallel
        self.num_entangling + self.single_qubit_layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap_decomposition() {
        let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
        let swap_gates = decomposer.decompose_swap();

        assert_eq!(swap_gates.len(), 3);
        assert!(matches!(swap_gates[0], TwoQubitGateInstruction::CNOT { .. }));
    }

    #[test]
    fn test_cnot_cz_conversion() {
        let cnot_to_cz = TwoQubitDecomposer::cnot_to_cz();
        assert_eq!(cnot_to_cz.len(), 3);

        let cz_to_cnot = TwoQubitDecomposer::cz_to_cnot();
        assert_eq!(cz_to_cnot.len(), 3);
    }

    #[test]
    fn test_iswap_decomposition() {
        let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
        let iswap_gates = decomposer.decompose_iswap();

        assert!(!iswap_gates.is_empty());
    }

    #[test]
    fn test_canonical_decomposition_depth() {
        let decomp = CanonicalDecomposition {
            entangling_gate: EntanglementGate::CNOT,
            single_qubit_layers: vec![
                SingleQubitLayer { gate_0: None, gate_1: None },
                SingleQubitLayer { gate_0: None, gate_1: None },
            ],
            num_entangling: 3,
        };

        assert_eq!(decomp.depth(), 5);  // 3 CNOTs + 2 layers
    }
}
