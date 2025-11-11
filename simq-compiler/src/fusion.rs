//! Gate fusion optimization pass
//!
//! This module implements gate fusion, which combines adjacent single-qubit gates
//! operating on the same qubit into a single composite gate. This reduces the
//! circuit depth and can improve simulation performance.
//!
//! # Theory
//!
//! When multiple single-qubit gates act on the same qubit sequentially, they can
//! be combined by multiplying their unitary matrices:
//!
//! ```text
//! |ψ⟩ → U₃(U₂(U₁|ψ⟩)) = (U₃·U₂·U₁)|ψ⟩
//! ```
//!
//! The fused gate's matrix is the product of the individual gate matrices,
//! applied in reverse order (rightmost gate is applied first).
//!
//! # Example
//!
//! ```ignore
//! use simq_core::Circuit;
//! use simq_gates::standard::{Hadamard, PauliX};
//! use simq_compiler::fusion::fuse_single_qubit_gates;
//!
//! // Create circuit with adjacent single-qubit gates
//! let mut circuit = Circuit::new(2);
//! circuit.add_gate(Hadamard, &[q0]);
//! circuit.add_gate(PauliX, &[q0]);
//! circuit.add_gate(Hadamard, &[q0]);
//!
//! // Apply fusion optimization
//! let optimized = fuse_single_qubit_gates(&circuit, None).unwrap();
//! // Three gates on q0 are fused into one FusedGate
//! assert!(optimized.len() < circuit.len());
//! ```

use crate::matrix_utils::{is_identity, matrix_to_vec, multiply_2x2};
use ahash::{AHashMap, AHashSet};
use num_complex::Complex64;
use simq_core::{gate::Gate, Circuit, GateOp, QubitId, Result};
use smallvec::SmallVec;
use std::fmt;
use std::sync::Arc;

/// Configuration for gate fusion optimization
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Minimum number of gates to fuse (default: 2)
    /// Gates sequences shorter than this won't be fused
    pub min_fusion_size: usize,

    /// Whether to eliminate identity gates (default: true)
    /// If a fused gate results in identity, it will be removed
    pub eliminate_identity: bool,

    /// Epsilon for identity detection (default: 1e-10)
    pub identity_epsilon: f64,

    /// Maximum fusion chain length (default: None = unlimited)
    /// Limits how many consecutive gates can be fused together
    pub max_fusion_size: Option<usize>,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            min_fusion_size: 2,
            eliminate_identity: true,
            identity_epsilon: 1e-10,
            max_fusion_size: None,
        }
    }
}

/// A fused quantum gate representing the composition of multiple single-qubit gates
///
/// This gate stores the product of multiple gate matrices and maintains
/// information about the original gates for debugging and visualization.
#[derive(Clone)]
pub struct FusedGate {
    /// The composed unitary matrix (2x2 for single-qubit gates)
    matrix: [[Complex64; 2]; 2],

    /// Names of the original gates in application order
    /// (first gate in vector is applied first)
    component_gates: SmallVec<[String; 4]>,
}

impl FusedGate {
    /// Create a new fused gate from a matrix and component gate names
    ///
    /// # Arguments
    /// * `matrix` - The composed 2x2 unitary matrix
    /// * `component_gates` - Names of the original gates (in application order)
    pub fn new(matrix: [[Complex64; 2]; 2], component_gates: SmallVec<[String; 4]>) -> Self {
        Self {
            matrix,
            component_gates,
        }
    }

    /// Get the unitary matrix
    pub fn matrix(&self) -> &[[Complex64; 2]; 2] {
        &self.matrix
    }

    /// Get the names of component gates
    pub fn component_gates(&self) -> &[String] {
        &self.component_gates
    }

    /// Get the number of gates that were fused
    pub fn num_components(&self) -> usize {
        self.component_gates.len()
    }
}

impl Gate for FusedGate {
    fn name(&self) -> &str {
        "FUSED"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn is_unitary(&self) -> bool {
        true
    }

    fn description(&self) -> String {
        format!(
            "Fused gate: {}",
            self.component_gates.join(" → ")
        )
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(matrix_to_vec(&self.matrix))
    }
}

impl fmt::Debug for FusedGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FusedGate")
            .field("components", &self.component_gates)
            .field("num_gates", &self.num_components())
            .finish()
    }
}

/// A fusion opportunity represents a sequence of gates that can be fused
#[derive(Debug)]
struct FusionChain {
    /// Indices of operations in the circuit
    operation_indices: Vec<usize>,
    /// The qubit this chain operates on
    qubit: QubitId,
}

/// Analyze circuit to find fusion opportunities
///
/// Identifies sequences of adjacent single-qubit gates operating on the same qubit
/// that have no intervening operations on that qubit.
///
/// # Arguments
/// * `circuit` - The circuit to analyze
/// * `config` - Fusion configuration
///
/// # Returns
/// Vector of fusion chains, where each chain is a sequence of operation indices
fn find_fusion_chains(circuit: &Circuit, config: &FusionConfig) -> Vec<FusionChain> {
    let num_qubits = circuit.num_qubits();
    let operations: Vec<_> = circuit.operations().collect();

    // Track the last operation index for each qubit
    let mut qubit_last_op: Vec<Option<usize>> = vec![None; num_qubits];

    // Track current fusion chains for each qubit
    let mut current_chains: Vec<Option<Vec<usize>>> = vec![None; num_qubits];

    // Completed fusion chains
    let mut fusion_chains = Vec::new();

    for (op_idx, op) in operations.iter().enumerate() {
        let qubits = op.qubits();

        // Multi-qubit gates break fusion chains on all their qubits
        if qubits.len() > 1 {
            for &qubit in qubits {
                let qubit_idx = qubit.index();
                if let Some(chain) = current_chains[qubit_idx].take() {
                    if chain.len() >= config.min_fusion_size {
                        fusion_chains.push(FusionChain {
                            operation_indices: chain,
                            qubit,
                        });
                    }
                }
                qubit_last_op[qubit_idx] = Some(op_idx);
            }
            continue;
        }

        // Single-qubit gate
        let qubit = qubits[0];
        let qubit_idx = qubit.index();

        // Check if this gate can be fused (must have a matrix representation)
        if op.gate().matrix().is_none() {
            // Non-unitary or unsupported gate, break the chain
            if let Some(chain) = current_chains[qubit_idx].take() {
                if chain.len() >= config.min_fusion_size {
                    fusion_chains.push(FusionChain {
                        operation_indices: chain,
                        qubit,
                    });
                }
            }
            qubit_last_op[qubit_idx] = Some(op_idx);
            continue;
        }

        // Add to or start fusion chain
        match current_chains[qubit_idx].as_mut() {
            Some(chain) => {
                // Check max fusion size
                if let Some(max_size) = config.max_fusion_size {
                    if chain.len() >= max_size {
                        // Finalize current chain and start new one
                        let old_chain = current_chains[qubit_idx].take().unwrap();
                        if old_chain.len() >= config.min_fusion_size {
                            fusion_chains.push(FusionChain {
                                operation_indices: old_chain,
                                qubit,
                            });
                        }
                        current_chains[qubit_idx] = Some(vec![op_idx]);
                    } else {
                        chain.push(op_idx);
                    }
                } else {
                    chain.push(op_idx);
                }
            }
            None => {
                current_chains[qubit_idx] = Some(vec![op_idx]);
            }
        }

        qubit_last_op[qubit_idx] = Some(op_idx);
    }

    // Finalize remaining chains
    for (qubit_idx, chain) in current_chains.into_iter().enumerate() {
        if let Some(chain) = chain {
            if chain.len() >= config.min_fusion_size {
                fusion_chains.push(FusionChain {
                    operation_indices: chain,
                    qubit: QubitId::new(qubit_idx),
                });
            }
        }
    }

    fusion_chains
}

/// Fuse a sequence of single-qubit gates into a single FusedGate
///
/// # Arguments
/// * `gates` - Sequence of gate operations to fuse (in application order)
/// * `config` - Fusion configuration
///
/// # Returns
/// A FusedGate representing the composition, or None if the result is identity
/// and identity elimination is enabled
fn fuse_gates(gates: &[&GateOp], config: &FusionConfig) -> Option<Arc<dyn Gate>> {
    assert!(!gates.is_empty(), "Cannot fuse empty gate sequence");
    assert!(
        gates.iter().all(|g| g.num_qubits() == 1),
        "All gates must be single-qubit"
    );

    // Extract matrices from gates
    let matrices: Vec<[[Complex64; 2]; 2]> = gates
        .iter()
        .map(|op| {
            let matrix_vec = op
                .gate()
                .matrix()
                .expect("Gate must have matrix representation");
            // Convert from Vec<Complex64> to [[Complex64; 2]; 2]
            [
                [matrix_vec[0], matrix_vec[1]],
                [matrix_vec[2], matrix_vec[3]],
            ]
        })
        .collect();

    // Compose matrices: rightmost (first applied) gate is multiplied first
    // If gates are [G1, G2, G3], the result is G3 * G2 * G1
    let mut result = matrices[0];
    for matrix in &matrices[1..] {
        result = multiply_2x2(matrix, &result);
    }

    // Check if result is identity
    if config.eliminate_identity && is_identity(&result, config.identity_epsilon) {
        return None;
    }

    // Collect gate names for debugging
    let component_gates: SmallVec<[String; 4]> = gates
        .iter()
        .map(|op| op.gate().name().to_string())
        .collect();

    Some(Arc::new(FusedGate::new(result, component_gates)))
}

/// Apply gate fusion optimization to a quantum circuit
///
/// This function analyzes the circuit to find sequences of adjacent single-qubit
/// gates on the same qubit and combines them into fused gates.
///
/// # Arguments
/// * `circuit` - The input circuit to optimize
/// * `config` - Optional fusion configuration (uses default if None)
///
/// # Returns
/// A new optimized circuit with fused gates
///
/// # Errors
/// Returns error if circuit construction fails (should be rare)
///
/// # Example
/// ```ignore
/// use simq_core::Circuit;
/// use simq_compiler::fusion::fuse_single_qubit_gates;
///
/// let circuit = /* ... build circuit ... */;
/// let optimized = fuse_single_qubit_gates(&circuit, None).unwrap();
/// ```
pub fn fuse_single_qubit_gates(
    circuit: &Circuit,
    config: Option<FusionConfig>,
) -> Result<Circuit> {
    let config = config.unwrap_or_default();

    // Find all fusion opportunities
    let fusion_chains = find_fusion_chains(circuit, &config);

    // If no fusion opportunities, return original circuit
    if fusion_chains.is_empty() {
        return Ok(circuit.clone());
    }

    // Build set of operation indices that will be replaced by fused gates
    let fused_indices: AHashSet<usize> = fusion_chains
        .iter()
        .flat_map(|chain| chain.operation_indices.iter().copied())
        .collect();

    // Build the optimized circuit
    let mut optimized = Circuit::with_capacity(circuit.num_qubits(), circuit.len());
    let operations: Vec<_> = circuit.operations().collect();

    // Map from fusion chain start index to fused gate
    let mut fused_gates: AHashMap<usize, (Arc<dyn Gate>, QubitId)> = AHashMap::new();

    for chain in &fusion_chains {
        let chain_ops: Vec<&GateOp> = chain
            .operation_indices
            .iter()
            .map(|&idx| operations[idx])
            .collect();

        if let Some(fused_gate) = fuse_gates(&chain_ops, &config) {
            let start_idx = chain.operation_indices[0];
            fused_gates.insert(start_idx, (fused_gate, chain.qubit));
        }
    }

    // Reconstruct circuit
    for (idx, op) in operations.iter().enumerate() {
        if let Some((fused_gate, qubit)) = fused_gates.get(&idx) {
            // Insert fused gate at the position of the first gate in the chain
            optimized.add_gate(Arc::clone(fused_gate), &[*qubit])?;
        } else if !fused_indices.contains(&idx) {
            // Keep original operation if it's not part of any fusion
            optimized.add_gate(Arc::clone(op.gate()), op.qubits())?;
        }
        // Skip operations that were fused (and are not the first in their chain)
    }

    Ok(optimized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_gates::standard::{Hadamard, PauliX, PauliY, PauliZ, SGate, TGate};

    fn create_test_circuit_simple() -> Circuit {
        let mut circuit = Circuit::new(2);
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);

        // Three gates on q0 that should be fused
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();
        circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[q0]).unwrap();
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();

        // One gate on q1 (not fused, too small)
        circuit.add_gate(Arc::new(PauliZ) as Arc<dyn Gate>, &[q1]).unwrap();

        circuit
    }

    fn create_test_circuit_multiple_qubits() -> Circuit {
        let mut circuit = Circuit::new(3);
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        let q2 = QubitId::new(2);

        // Chain on q0
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();
        circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[q0]).unwrap();

        // Chain on q1
        circuit.add_gate(Arc::new(PauliY) as Arc<dyn Gate>, &[q1]).unwrap();
        circuit.add_gate(Arc::new(PauliZ) as Arc<dyn Gate>, &[q1]).unwrap();

        // Single gate on q2
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q2]).unwrap();

        // More on q0
        circuit.add_gate(Arc::new(PauliZ) as Arc<dyn Gate>, &[q0]).unwrap();

        circuit
    }

    #[test]
    fn test_find_fusion_chains_simple() {
        let circuit = create_test_circuit_simple();
        let config = FusionConfig::default();
        let chains = find_fusion_chains(&circuit, &config);

        // Should find one chain on q0 with 3 gates
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].operation_indices.len(), 3);
        assert_eq!(chains[0].qubit, QubitId::new(0));
    }

    #[test]
    fn test_find_fusion_chains_multiple() {
        let circuit = create_test_circuit_multiple_qubits();
        let config = FusionConfig::default();
        let chains = find_fusion_chains(&circuit, &config);

        // Should find chains on q0 and q1 (2 gates each)
        // Single gates on q2 and later q0 are too small to fuse
        assert_eq!(chains.len(), 2);
    }

    #[test]
    fn test_fuse_gates_hadamard_x_hadamard() {
        let q0 = QubitId::new(0);
        let h: Arc<dyn Gate> = Arc::new(Hadamard);
        let x: Arc<dyn Gate> = Arc::new(PauliX);

        let op1 = GateOp::new(Arc::clone(&h), &[q0]).unwrap();
        let op2 = GateOp::new(Arc::clone(&x), &[q0]).unwrap();
        let op3 = GateOp::new(Arc::clone(&h), &[q0]).unwrap();

        let gates = vec![&op1, &op2, &op3];
        let config = FusionConfig::default();

        let fused = fuse_gates(&gates, &config);
        assert!(fused.is_some());

        let fused_gate = fused.unwrap();
        assert_eq!(fused_gate.name(), "FUSED");
        assert_eq!(fused_gate.num_qubits(), 1);
    }

    #[test]
    fn test_fuse_gates_self_inverse() {
        // X * X = I, should be eliminated
        let q0 = QubitId::new(0);
        let x: Arc<dyn Gate> = Arc::new(PauliX);

        let op1 = GateOp::new(Arc::clone(&x), &[q0]).unwrap();
        let op2 = GateOp::new(Arc::clone(&x), &[q0]).unwrap();

        let gates = vec![&op1, &op2];
        let config = FusionConfig::default();

        let fused = fuse_gates(&gates, &config);
        // Should return None because result is identity
        assert!(fused.is_none());
    }

    #[test]
    fn test_fuse_gates_s_s_equals_z() {
        // S * S = Z (not identity)
        let q0 = QubitId::new(0);
        let s: Arc<dyn Gate> = Arc::new(SGate);

        let op1 = GateOp::new(Arc::clone(&s), &[q0]).unwrap();
        let op2 = GateOp::new(Arc::clone(&s), &[q0]).unwrap();

        let gates = vec![&op1, &op2];
        let config = FusionConfig::default();

        let fused = fuse_gates(&gates, &config);
        assert!(fused.is_some());

        // Verify the result is approximately Z
        let fused_gate = fused.unwrap();
        let matrix = fused_gate.matrix().unwrap();

        // Z matrix in flattened form: [1, 0, 0, -1]
        assert!((matrix[0].re - 1.0).abs() < 1e-10);
        assert!(matrix[0].im.abs() < 1e-10);
        assert!(matrix[1].norm() < 1e-10);
        assert!(matrix[2].norm() < 1e-10);
        assert!((matrix[3].re - (-1.0)).abs() < 1e-10);
        assert!(matrix[3].im.abs() < 1e-10);
    }

    #[test]
    fn test_fuse_single_qubit_gates_simple() {
        let circuit = create_test_circuit_simple();
        let original_len = circuit.len();

        let optimized = fuse_single_qubit_gates(&circuit, None).unwrap();

        // Should have fewer operations (3 gates fused into 1, plus 1 unfused)
        assert!(optimized.len() < original_len);
        assert_eq!(optimized.len(), 2); // 1 fused gate on q0, 1 gate on q1
    }

    #[test]
    fn test_fuse_single_qubit_gates_preserves_qubits() {
        let circuit = create_test_circuit_simple();
        let optimized = fuse_single_qubit_gates(&circuit, None).unwrap();

        assert_eq!(optimized.num_qubits(), circuit.num_qubits());
    }

    #[test]
    fn test_fusion_config_min_size() {
        let circuit = create_test_circuit_simple();

        let config = FusionConfig {
            min_fusion_size: 4, // Require at least 4 gates to fuse
            ..Default::default()
        };

        let optimized = fuse_single_qubit_gates(&circuit, Some(config)).unwrap();

        // No fusion should occur (chain has only 3 gates)
        assert_eq!(optimized.len(), circuit.len());
    }

    #[test]
    fn test_fusion_config_no_identity_elimination() {
        // X * X = I
        let mut circuit = Circuit::new(1);
        let q0 = QubitId::new(0);
        let x: Arc<dyn Gate> = Arc::new(PauliX);
        circuit.add_gate(Arc::clone(&x), &[q0]).unwrap();
        circuit.add_gate(Arc::clone(&x), &[q0]).unwrap();

        let config = FusionConfig {
            eliminate_identity: false,
            ..Default::default()
        };

        let optimized = fuse_single_qubit_gates(&circuit, Some(config)).unwrap();

        // Should have 1 fused gate (identity not eliminated)
        assert_eq!(optimized.len(), 1);
    }

    #[test]
    fn test_fusion_config_max_size() {
        let mut circuit = Circuit::new(1);
        let q0 = QubitId::new(0);

        // Add 5 T gates
        let t: Arc<dyn Gate> = Arc::new(TGate);
        for _ in 0..5 {
            circuit.add_gate(Arc::clone(&t), &[q0]).unwrap();
        }

        let config = FusionConfig {
            max_fusion_size: Some(3),
            ..Default::default()
        };

        let optimized = fuse_single_qubit_gates(&circuit, Some(config)).unwrap();

        // Should create 2 fused gates: one with 3 gates, one with 2 gates
        assert_eq!(optimized.len(), 2);
    }

    #[test]
    fn test_fused_gate_description() {
        let q0 = QubitId::new(0);
        let h: Arc<dyn Gate> = Arc::new(Hadamard);
        let x: Arc<dyn Gate> = Arc::new(PauliX);

        let op1 = GateOp::new(Arc::clone(&h), &[q0]).unwrap();
        let op2 = GateOp::new(Arc::clone(&x), &[q0]).unwrap();

        let gates = vec![&op1, &op2];
        let config = FusionConfig::default();

        let fused = fuse_gates(&gates, &config).unwrap();
        let desc = fused.description();

        assert!(desc.contains("H"));
        assert!(desc.contains("X"));
        assert!(desc.contains("→")); // Arrow separator
    }
}
