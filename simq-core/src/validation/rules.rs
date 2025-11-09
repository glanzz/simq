//! Validation rules for circuit validation

use crate::Circuit;
use crate::validation::dag::DependencyGraph;

/// Validation rule that can check circuit properties
pub trait ValidationRule: Send + Sync {
    /// Name of the validation rule
    fn name(&self) -> &str;

    /// Description of what this rule checks
    fn description(&self) -> &str;

    /// Validate the circuit
    fn validate(&self, circuit: &Circuit, dag: &DependencyGraph) -> ValidationResult;
}

/// Result of validation
#[derive(Clone, Debug)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
}

impl ValidationResult {
    /// Create a valid result
    pub fn ok() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create an error result
    pub fn error(error: ValidationError) -> Self {
        Self {
            is_valid: false,
            errors: vec![error],
            warnings: Vec::new(),
        }
    }

    /// Add a warning to the result
    pub fn with_warning(mut self, warning: ValidationWarning) -> Self {
        self.warnings.push(warning);
        self
    }

    /// Check if result is valid
    pub fn is_valid(&self) -> bool {
        self.is_valid && self.errors.is_empty()
    }
}

/// Validation error with location information
#[derive(Clone, Debug)]
pub struct ValidationError {
    pub rule_name: String,
    pub message: String,
    pub operation_indices: Vec<usize>,
    pub qubits: Vec<usize>,
    pub suggestion: Option<String>,
}

impl ValidationError {
    /// Format error for display
    pub fn format(&self, circuit: &Circuit) -> String {
        let mut msg = format!("Validation error [{}]: {}\n", self.rule_name, self.message);

        if !self.operation_indices.is_empty() {
            msg.push_str("  Involved operations:\n");
            for &idx in &self.operation_indices {
                if let Some(op) = circuit.get_operation(idx) {
                    msg.push_str(&format!("    {}: {}\n", idx, op));
                }
            }
        }

        if !self.qubits.is_empty() {
            msg.push_str(&format!("  Involved qubits: {:?}\n", self.qubits));
        }

        if let Some(suggestion) = &self.suggestion {
            msg.push_str(&format!("  Suggestion: {}\n", suggestion));
        }

        msg
    }
}

/// Validation warning
#[derive(Clone, Debug)]
pub struct ValidationWarning {
    pub rule_name: String,
    pub message: String,
    pub operation_indices: Vec<usize>,
}

impl ValidationWarning {
    /// Format warning for display
    pub fn format(&self) -> String {
        format!("Warning [{}]: {}", self.rule_name, self.message)
    }
}

/// Cycle detection rule
pub struct CycleDetectionRule;

impl ValidationRule for CycleDetectionRule {
    fn name(&self) -> &str {
        "cycle_detection"
    }

    fn description(&self) -> &str {
        "Detects cycles in the circuit dependency graph"
    }

    fn validate(&self, _circuit: &Circuit, dag: &DependencyGraph) -> ValidationResult {
        let cycles = dag.find_cycles();
        if cycles.is_empty() {
            ValidationResult::ok()
        } else {
            // Use the first cycle found for error reporting
            let cycle = &cycles[0];
            let error = ValidationError {
                rule_name: self.name().to_string(),
                message: format!("Circuit contains {} cycle(s)", cycles.len()),
                operation_indices: cycle.clone(),
                qubits: Vec::new(),
                suggestion: Some(
                    "Circuits must be acyclic. Check for gates that create circular dependencies."
                        .to_string(),
                ),
            };
            ValidationResult::error(error)
        }
    }
}

/// Dependency validation rule
pub struct DependencyValidationRule;

impl ValidationRule for DependencyValidationRule {
    fn name(&self) -> &str {
        "dependency_validation"
    }

    fn description(&self) -> &str {
        "Validates that all gate dependencies are properly ordered"
    }

    fn validate(&self, _circuit: &Circuit, dag: &DependencyGraph) -> ValidationResult {
        // Check that topological sort succeeds
        match dag.topological_sort() {
            Ok(_) => ValidationResult::ok(),
            Err(e) => ValidationResult::error(ValidationError {
                rule_name: self.name().to_string(),
                message: format!("Dependency validation failed: {}", e),
                operation_indices: Vec::new(),
                qubits: Vec::new(),
                suggestion: Some("Circuit dependencies are invalid".to_string()),
            }),
        }
    }
}

/// Qubit usage validation rule
pub struct QubitUsageRule;

impl ValidationRule for QubitUsageRule {
    fn name(&self) -> &str {
        "qubit_usage"
    }

    fn description(&self) -> &str {
        "Validates that all qubits are used within circuit bounds"
    }

    fn validate(&self, circuit: &Circuit, _dag: &DependencyGraph) -> ValidationResult {
        let mut errors = Vec::new();

        for (i, op) in circuit.operations().enumerate() {
            for &qubit in op.qubits() {
                if qubit.index() >= circuit.num_qubits() {
                    errors.push(ValidationError {
                        rule_name: self.name().to_string(),
                        message: format!(
                            "Operation {} uses qubit {} which is out of bounds (circuit has {} qubits)",
                            i,
                            qubit.index(),
                            circuit.num_qubits()
                        ),
                        operation_indices: vec![i],
                        qubits: vec![qubit.index()],
                        suggestion: Some(format!(
                            "Ensure qubit index is between 0 and {}",
                            circuit.num_qubits() - 1
                        )),
                    });
                }
            }
        }

        if errors.is_empty() {
            ValidationResult::ok()
        } else {
            ValidationResult {
                is_valid: false,
                errors,
                warnings: Vec::new(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::Gate;
    use crate::QubitId;
    use std::sync::Arc;

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

    #[test]
    fn test_cycle_detection_rule() {
        let mut circuit = Circuit::new(2);
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });
        circuit.add_gate(gate, &[QubitId::new(0)]).unwrap();

        let dag = DependencyGraph::from_circuit(&circuit).unwrap();
        let rule = CycleDetectionRule;
        let result = rule.validate(&circuit, &dag);
        assert!(result.is_valid());
    }

    #[test]
    fn test_dependency_validation_rule() {
        let mut circuit = Circuit::new(2);
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });
        circuit.add_gate(gate, &[QubitId::new(0)]).unwrap();

        let dag = DependencyGraph::from_circuit(&circuit).unwrap();
        let rule = DependencyValidationRule;
        let result = rule.validate(&circuit, &dag);
        assert!(result.is_valid());
    }

    #[test]
    fn test_qubit_usage_rule() {
        let mut circuit = Circuit::new(2);
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });
        circuit.add_gate(gate, &[QubitId::new(0)]).unwrap();

        let dag = DependencyGraph::from_circuit(&circuit).unwrap();
        let rule = QubitUsageRule;
        let result = rule.validate(&circuit, &dag);
        assert!(result.is_valid());
    }
}

