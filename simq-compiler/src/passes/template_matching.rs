//! Advanced template pattern matching for circuit optimization
//!
//! This module provides a more sophisticated template matching system that can:
//! - Match complex gate patterns
//! - Generate replacement gates dynamically
//! - Support parameterized templates
//! - Handle gate rewriting with proper gate instances

use crate::passes::OptimizationPass;
use simq_core::{gate::Gate, Circuit, GateOp, QubitId, Result};
use simq_gates::standard::{PauliX, PauliY, PauliZ};
use std::sync::Arc;

/// A pattern matcher function that checks if a sequence matches
type PatternMatcher = fn(&[&GateOp]) -> bool;

/// A replacement generator function that creates replacement gates
type ReplacementGenerator = fn(QubitId) -> Vec<(Arc<dyn Gate>, QubitId)>;

/// Type alias for match result
type MatchResult = Option<(usize, Vec<(Arc<dyn Gate>, QubitId)>)>;

/// Type alias for best match result
type BestMatch = Option<(usize, usize, Vec<(Arc<dyn Gate>, QubitId)>)>;

/// An advanced template with pattern matching and replacement generation
struct AdvancedTemplate {
    /// Name of this template
    #[allow(dead_code)]
    name: &'static str,
    /// Minimum pattern length
    min_length: usize,
    /// Maximum pattern length
    max_length: usize,
    /// Pattern matching function
    matcher: PatternMatcher,
    /// Replacement generator
    generator: ReplacementGenerator,
    /// Description
    #[allow(dead_code)]
    description: &'static str,
}

/// Advanced template pattern matching optimization pass
///
/// This pass uses sophisticated pattern matching to recognize and replace
/// gate sequences with optimized equivalents. Unlike simple string matching,
/// this supports:
/// - Dynamic pattern matching based on gate properties
/// - Proper gate instance creation for replacements
/// - Parameterized gate handling
///
/// # Example
/// ```ignore
/// use simq_compiler::passes::AdvancedTemplateMatching;
/// use simq_core::Circuit;
///
/// let pass = AdvancedTemplateMatching::new();
/// let mut circuit = Circuit::new(3);
/// // ... add gates ...
/// pass.apply(&mut circuit)?;
/// ```
#[derive(Debug, Clone)]
pub struct AdvancedTemplateMatching {
    /// Whether to enable the pass
    enabled: bool,
}

impl AdvancedTemplateMatching {
    /// Create a new advanced template matching pass
    pub fn new() -> Self {
        Self { enabled: true }
    }

    /// Get all available templates
    fn templates() -> Vec<AdvancedTemplate> {
        vec![
            // H-Z-H = X (Hadamard conjugation)
            AdvancedTemplate {
                name: "h-z-h",
                min_length: 3,
                max_length: 3,
                matcher: |ops| {
                    ops.len() == 3
                        && ops[0].gate().name() == "H"
                        && ops[1].gate().name() == "Z"
                        && ops[2].gate().name() == "H"
                },
                generator: |qubit| vec![(Arc::new(PauliX) as Arc<dyn Gate>, qubit)],
                description: "H-Z-H conjugation gives X",
            },
            // H-X-H = Z (Hadamard conjugation)
            AdvancedTemplate {
                name: "h-x-h",
                min_length: 3,
                max_length: 3,
                matcher: |ops| {
                    ops.len() == 3
                        && ops[0].gate().name() == "H"
                        && ops[1].gate().name() == "X"
                        && ops[2].gate().name() == "H"
                },
                generator: |qubit| vec![(Arc::new(PauliZ) as Arc<dyn Gate>, qubit)],
                description: "H-X-H conjugation gives Z",
            },
            // X-X = I (self-inverse)
            AdvancedTemplate {
                name: "x-x",
                min_length: 2,
                max_length: 2,
                matcher: |ops| {
                    ops.len() == 2 && ops[0].gate().name() == "X" && ops[1].gate().name() == "X"
                },
                generator: |_| vec![], // Identity - remove both gates
                description: "X is self-inverse",
            },
            // Y-Y = I (self-inverse)
            AdvancedTemplate {
                name: "y-y",
                min_length: 2,
                max_length: 2,
                matcher: |ops| {
                    ops.len() == 2 && ops[0].gate().name() == "Y" && ops[1].gate().name() == "Y"
                },
                generator: |_| vec![],
                description: "Y is self-inverse",
            },
            // Z-Z = I (self-inverse)
            AdvancedTemplate {
                name: "z-z",
                min_length: 2,
                max_length: 2,
                matcher: |ops| {
                    ops.len() == 2 && ops[0].gate().name() == "Z" && ops[1].gate().name() == "Z"
                },
                generator: |_| vec![],
                description: "Z is self-inverse",
            },
            // H-H = I (self-inverse)
            AdvancedTemplate {
                name: "h-h",
                min_length: 2,
                max_length: 2,
                matcher: |ops| {
                    ops.len() == 2 && ops[0].gate().name() == "H" && ops[1].gate().name() == "H"
                },
                generator: |_| vec![],
                description: "Hadamard is self-inverse",
            },
            // X-Y-X = -Y (conjugation, ignoring global phase)
            AdvancedTemplate {
                name: "x-y-x",
                min_length: 3,
                max_length: 3,
                matcher: |ops| {
                    ops.len() == 3
                        && ops[0].gate().name() == "X"
                        && ops[1].gate().name() == "Y"
                        && ops[2].gate().name() == "X"
                },
                generator: |qubit| vec![(Arc::new(PauliY) as Arc<dyn Gate>, qubit)],
                description: "X-Y-X gives -Y (global phase ignored)",
            },
            // Z-X-Z = -X (conjugation, ignoring global phase)
            AdvancedTemplate {
                name: "z-x-z",
                min_length: 3,
                max_length: 3,
                matcher: |ops| {
                    ops.len() == 3
                        && ops[0].gate().name() == "Z"
                        && ops[1].gate().name() == "X"
                        && ops[2].gate().name() == "Z"
                },
                generator: |qubit| vec![(Arc::new(PauliX) as Arc<dyn Gate>, qubit)],
                description: "Z-X-Z gives -X (global phase ignored)",
            },
            // Z-Y-Z = -Y (conjugation, ignoring global phase)
            AdvancedTemplate {
                name: "z-y-z",
                min_length: 3,
                max_length: 3,
                matcher: |ops| {
                    ops.len() == 3
                        && ops[0].gate().name() == "Z"
                        && ops[1].gate().name() == "Y"
                        && ops[2].gate().name() == "Z"
                },
                generator: |qubit| vec![(Arc::new(PauliY) as Arc<dyn Gate>, qubit)],
                description: "Z-Y-Z gives -Y (global phase ignored)",
            },
        ]
    }

    /// Try to match and apply a template at a given position
    fn try_match_template(
        &self,
        ops: &[GateOp],
        start: usize,
        template: &AdvancedTemplate,
    ) -> MatchResult {
        // Check if we have enough gates remaining
        if start + template.min_length > ops.len() {
            return None;
        }

        // Try different pattern lengths
        for len in template.min_length..=template.max_length.min(ops.len() - start) {
            // Check all gates are single-qubit on the same qubit
            let first_qubit = ops[start].qubits()[0];
            let mut valid = true;

            for i in 0..len {
                let op = &ops[start + i];
                if op.num_qubits() != 1 || op.qubits()[0] != first_qubit {
                    valid = false;
                    break;
                }
            }

            if !valid {
                continue;
            }

            // Collect operation references for matching
            let op_refs: Vec<&GateOp> = (start..start + len).map(|i| &ops[i]).collect();

            // Try to match the pattern
            if (template.matcher)(&op_refs) {
                // Generate replacement gates
                let replacements = (template.generator)(first_qubit);
                return Some((len, replacements));
            }
        }

        None
    }

    /// Apply template matching to a circuit
    fn apply_matching(&self, circuit: &mut Circuit) -> bool {
        let templates = Self::templates();
        let mut modified = false;

        // Keep applying templates until no more matches found
        loop {
            let ops = circuit.operations_mut();
            if ops.len() < 2 {
                break;
            }

            let mut found_match = false;
            let mut i = 0;

            while i < ops.len() {
                let mut best_match: BestMatch = None;

                // Try all templates, prefer longer matches
                for (tidx, template) in templates.iter().enumerate() {
                    if let Some((match_len, replacements)) =
                        self.try_match_template(ops, i, template)
                    {
                        if best_match.is_none() || match_len > best_match.as_ref().unwrap().0 {
                            best_match = Some((match_len, tidx, replacements));
                        }
                    }
                }

                if let Some((match_len, _tidx, replacements)) = best_match {
                    // Remove matched gates
                    let _ = ops.drain(i..i + match_len).collect::<Vec<_>>();

                    // Insert replacements at the same position
                    for (idx, (gate, qubit)) in replacements.into_iter().enumerate() {
                        let gate_op = GateOp::new(gate, &[qubit]).unwrap();
                        ops.insert(i + idx, gate_op);
                    }

                    found_match = true;
                    modified = true;
                    // Don't increment i, check same position again
                } else {
                    i += 1;
                }
            }

            if !found_match {
                break; // No more matches found
            }
        }

        modified
    }
}

impl Default for AdvancedTemplateMatching {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for AdvancedTemplateMatching {
    fn name(&self) -> &str {
        "advanced-template-matching"
    }

    fn apply(&self, circuit: &mut Circuit) -> Result<bool> {
        if !self.enabled {
            return Ok(false);
        }
        Ok(self.apply_matching(circuit))
    }

    fn description(&self) -> Option<&str> {
        Some("Advanced pattern matching with dynamic gate replacement")
    }

    fn iterative(&self) -> bool {
        true // May reveal new patterns after substitution
    }

    fn benefit_score(&self) -> f64 {
        0.85 // High benefit - creates proper gate instances
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_gates::Hadamard;

    #[test]
    fn test_h_z_h_to_x() {
        let pass = AdvancedTemplateMatching::new();
        let mut circuit = Circuit::new(2);

        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliZ), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();

        assert_eq!(circuit.len(), 3);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(modified);
        assert_eq!(circuit.len(), 1);

        // Should have X gate
        let op = circuit.get_operation(0).unwrap();
        assert_eq!(op.gate().name(), "X");
    }

    #[test]
    fn test_h_x_h_to_z() {
        let pass = AdvancedTemplateMatching::new();
        let mut circuit = Circuit::new(2);

        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();

        assert_eq!(circuit.len(), 3);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(modified);
        assert_eq!(circuit.len(), 1);

        // Should have Z gate
        let op = circuit.get_operation(0).unwrap();
        assert_eq!(op.gate().name(), "Z");
    }

    #[test]
    fn test_y_y_removal() {
        // Covers the y-y template's generator arm (line 125).
        let pass = AdvancedTemplateMatching::new();
        let mut circuit = Circuit::new(2);

        circuit
            .add_gate(Arc::new(PauliY), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliY), &[QubitId::new(0)])
            .unwrap();

        assert_eq!(circuit.len(), 2);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(modified);
        assert_eq!(circuit.len(), 0); // Y-Y removed
    }

    #[test]
    fn test_z_z_removal() {
        // Covers the z-z template's generator arm (line 136).
        let pass = AdvancedTemplateMatching::new();
        let mut circuit = Circuit::new(2);

        circuit
            .add_gate(Arc::new(PauliZ), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliZ), &[QubitId::new(0)])
            .unwrap();

        assert_eq!(circuit.len(), 2);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(modified);
        assert_eq!(circuit.len(), 0); // Z-Z removed
    }

    #[test]
    fn test_h_h_removal() {
        // Covers the h-h template's generator arm (line 147).
        let pass = AdvancedTemplateMatching::new();
        let mut circuit = Circuit::new(2);

        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();

        assert_eq!(circuit.len(), 2);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(modified);
        assert_eq!(circuit.len(), 0); // H-H removed
    }

    #[test]
    fn test_x_y_x_to_y() {
        // Covers the x-y-x template's matcher (lines 158-159) and generator
        // (line 161).
        let pass = AdvancedTemplateMatching::new();
        let mut circuit = Circuit::new(2);

        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliY), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();

        assert_eq!(circuit.len(), 3);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(modified);
        assert_eq!(circuit.len(), 1);

        let op = circuit.get_operation(0).unwrap();
        assert_eq!(op.gate().name(), "Y");
    }

    #[test]
    fn test_z_y_z_to_y() {
        // Covers the z-y-z template's matcher (line 187) and generator
        // (line 189).
        let pass = AdvancedTemplateMatching::new();
        let mut circuit = Circuit::new(2);

        circuit
            .add_gate(Arc::new(PauliZ), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliY), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliZ), &[QubitId::new(0)])
            .unwrap();

        assert_eq!(circuit.len(), 3);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(modified);
        assert_eq!(circuit.len(), 1);

        let op = circuit.get_operation(0).unwrap();
        assert_eq!(op.gate().name(), "Y");
    }

    #[test]
    fn test_default_trait_matches_new() {
        // Covers `impl Default for AdvancedTemplateMatching` (lines 296-297),
        // which delegates to `Self::new()`.
        let default_pass = AdvancedTemplateMatching::default();
        let new_pass = AdvancedTemplateMatching::new();
        assert_eq!(default_pass.enabled, new_pass.enabled);
    }

    #[test]
    fn test_disabled_pass_returns_false_without_modifying() {
        // Covers the early-return branch in `apply` (line 308) when the pass
        // is disabled.
        let pass = AdvancedTemplateMatching { enabled: false };
        let mut circuit = Circuit::new(2);

        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();

        assert_eq!(circuit.len(), 2);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(!modified);
        assert_eq!(circuit.len(), 2); // Unchanged since pass is disabled
    }

    #[test]
    fn test_self_inverse_removal() {
        let pass = AdvancedTemplateMatching::new();
        let mut circuit = Circuit::new(2);

        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();

        assert_eq!(circuit.len(), 2);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(modified);
        assert_eq!(circuit.len(), 0); // X-X removed
    }

    #[test]
    fn test_multiple_patterns() {
        let pass = AdvancedTemplateMatching::new();
        let mut circuit = Circuit::new(2);

        // H-Z-H (→ X) + X-X (→ I)
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliZ), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();

        assert_eq!(circuit.len(), 4);

        let modified = pass.apply(&mut circuit).unwrap();
        assert!(modified);
        // H-Z-H → X, then X-X → I
        assert_eq!(circuit.len(), 0);
    }
}
