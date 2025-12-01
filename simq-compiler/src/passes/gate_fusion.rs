//! Gate fusion optimization pass
//!
//! This pass combines adjacent single-qubit gates operating on the same qubit
//! into a single composite gate. This reduces circuit depth and gate count.

use crate::fusion::{fuse_single_qubit_gates, FusionConfig};
use crate::passes::OptimizationPass;
use simq_core::{Circuit, Result};

/// Gate fusion optimization pass
///
/// Combines sequences of adjacent single-qubit gates on the same qubit into
/// single fused gates. This can significantly reduce circuit size and depth.
///
/// # Example
/// ```ignore
/// use simq_compiler::passes::GateFusion;
/// use simq_core::Circuit;
///
/// let pass = GateFusion::new();
/// let mut circuit = Circuit::new(3);
/// // ... add gates ...
/// pass.apply(&mut circuit)?;
/// ```
#[derive(Debug, Clone)]
pub struct GateFusion {
    config: FusionConfig,
}

impl GateFusion {
    /// Create a new gate fusion pass with default configuration
    pub fn new() -> Self {
        Self {
            config: FusionConfig::default(),
        }
    }

    /// Create a gate fusion pass with custom configuration
    pub fn with_config(config: FusionConfig) -> Self {
        Self { config }
    }

    /// Set the minimum number of gates required for fusion
    pub fn with_min_fusion_size(mut self, min_size: usize) -> Self {
        self.config.min_fusion_size = min_size;
        self
    }

    /// Set the maximum number of gates that can be fused together
    pub fn with_max_fusion_size(mut self, max_size: Option<usize>) -> Self {
        self.config.max_fusion_size = max_size;
        self
    }

    /// Enable or disable identity elimination
    pub fn with_identity_elimination(mut self, enable: bool) -> Self {
        self.config.eliminate_identity = enable;
        self
    }

    /// Set the epsilon threshold for identity detection
    pub fn with_identity_epsilon(mut self, epsilon: f64) -> Self {
        self.config.identity_epsilon = epsilon;
        self
    }
}

impl Default for GateFusion {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for GateFusion {
    fn name(&self) -> &str {
        "gate-fusion"
    }

    fn apply(&self, circuit: &mut Circuit) -> Result<bool> {
        let original_len = circuit.len();

        // Apply fusion using existing implementation
        let optimized = fuse_single_qubit_gates(circuit, Some(self.config.clone()))?;

        // Check if circuit was modified
        let modified = optimized.len() != original_len;

        // Replace circuit contents if modified
        if modified {
            *circuit = optimized;
        }

        Ok(modified)
    }

    fn description(&self) -> Option<&str> {
        Some("Combines adjacent single-qubit gates on the same qubit into fused gates")
    }

    fn iterative(&self) -> bool {
        // Fusion can reveal new opportunities after other passes
        true
    }

    fn benefit_score(&self) -> f64 {
        0.95 // Very high benefit - significantly reduces circuit size and depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use simq_core::gate::Gate;
    use simq_core::QubitId;
    use std::sync::Arc;

    // Mock gate for testing
    #[derive(Debug)]
    struct MockGate {
        name: String,
        matrix: Option<Vec<Complex64>>,
    }

    impl Gate for MockGate {
        fn name(&self) -> &str {
            &self.name
        }

        fn num_qubits(&self) -> usize {
            1
        }

        fn matrix(&self) -> Option<Vec<Complex64>> {
            self.matrix.clone()
        }
    }

    fn pauli_x_matrix() -> Vec<Complex64> {
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]
    }

    fn hadamard_matrix() -> Vec<Complex64> {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(-inv_sqrt2, 0.0),
        ]
    }

    #[test]
    fn test_fusion_reduces_gate_count() {
        let pass = GateFusion::new();
        let mut circuit = Circuit::new(2);

        // Add two gates on the same qubit
        let x_gate = Arc::new(MockGate {
            name: "X".to_string(),
            matrix: Some(pauli_x_matrix()),
        });

        circuit
            .add_gate(x_gate.clone(), &[QubitId::new(0)])
            .unwrap();
        circuit.add_gate(x_gate, &[QubitId::new(0)]).unwrap();

        assert_eq!(circuit.len(), 2);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(modified);
        // X-X should fuse to identity and be removed
        assert_eq!(circuit.len(), 0);
    }

    #[test]
    fn test_fusion_different_qubits_no_change() {
        let pass = GateFusion::new();
        let mut circuit = Circuit::new(3);

        let h_gate = Arc::new(MockGate {
            name: "H".to_string(),
            matrix: Some(hadamard_matrix()),
        });

        circuit
            .add_gate(h_gate.clone(), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(h_gate.clone(), &[QubitId::new(1)])
            .unwrap();
        circuit.add_gate(h_gate, &[QubitId::new(2)]).unwrap();

        assert_eq!(circuit.len(), 3);

        let modified = pass.apply(&mut circuit).unwrap();
        // Gates on different qubits shouldn't be fused
        assert!(!modified);
        assert_eq!(circuit.len(), 3);
    }

    #[test]
    fn test_fusion_with_min_size() {
        let pass = GateFusion::new().with_min_fusion_size(3);
        let mut circuit = Circuit::new(2);

        let h_gate = Arc::new(MockGate {
            name: "H".to_string(),
            matrix: Some(hadamard_matrix()),
        });

        // Add only 2 gates (less than min_size)
        circuit
            .add_gate(h_gate.clone(), &[QubitId::new(0)])
            .unwrap();
        circuit.add_gate(h_gate, &[QubitId::new(0)]).unwrap();

        assert_eq!(circuit.len(), 2);

        let modified = pass.apply(&mut circuit).unwrap();
        // Should not fuse because below minimum size
        assert!(!modified);
        assert_eq!(circuit.len(), 2);
    }

    #[test]
    fn test_no_matrix_gates_not_fused() {
        let pass = GateFusion::new();
        let mut circuit = Circuit::new(2);

        // Gate without matrix representation
        let no_matrix_gate = Arc::new(MockGate {
            name: "Custom".to_string(),
            matrix: None,
        });

        circuit
            .add_gate(no_matrix_gate.clone(), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(no_matrix_gate, &[QubitId::new(0)])
            .unwrap();

        assert_eq!(circuit.len(), 2);

        let modified = pass.apply(&mut circuit).unwrap();
        // Gates without matrices shouldn't be fused
        assert!(!modified);
        assert_eq!(circuit.len(), 2);
    }
}
