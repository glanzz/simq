//! Validation report aggregation

use crate::validation::rules::{ValidationError, ValidationResult, ValidationWarning};
use crate::Circuit;
use std::collections::HashMap;

/// Comprehensive validation report
#[derive(Clone, Debug)]
pub struct ValidationReport {
    results: HashMap<String, ValidationResult>,
}

impl ValidationReport {
    /// Create a new validation report
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    /// Add a validation result for a rule
    pub fn add_result(&mut self, rule_name: &str, result: ValidationResult) {
        self.results.insert(rule_name.to_string(), result);
    }

    /// Check if report has any errors
    pub fn has_errors(&self) -> bool {
        self.results
            .values()
            .any(|r| !r.is_valid || !r.errors.is_empty())
    }

    /// Check if report has any warnings
    pub fn has_warnings(&self) -> bool {
        self.results.values().any(|r| !r.warnings.is_empty())
    }

    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        !self.has_errors()
    }

    /// Format the validation report
    pub fn format(&self, circuit: &Circuit) -> String {
        let mut msg = String::new();

        if self.is_valid() {
            msg.push_str("✓ Circuit validation passed\n");
        } else {
            msg.push_str("✗ Circuit validation failed\n");
        }

        for (rule_name, result) in &self.results {
            if !result.is_valid || !result.errors.is_empty() {
                msg.push_str(&format!("\nRule '{}':\n", rule_name));
                for error in &result.errors {
                    msg.push_str(&error.format(circuit));
                }
            }
        }

        if self.has_warnings() {
            msg.push_str("\nWarnings:\n");
            for (rule_name, result) in &self.results {
                for warning in &result.warnings {
                    msg.push_str(&format!("  [{}] {}\n", rule_name, warning.format()));
                }
            }
        }

        msg
    }

    /// Get all errors
    pub fn errors(&self) -> Vec<&ValidationError> {
        self.results
            .values()
            .flat_map(|r| r.errors.iter())
            .collect()
    }

    /// Get all warnings
    pub fn warnings(&self) -> Vec<&ValidationWarning> {
        self.results
            .values()
            .flat_map(|r| r.warnings.iter())
            .collect()
    }

    /// Get result for a specific rule
    pub fn get_result(&self, rule_name: &str) -> Option<&ValidationResult> {
        self.results.get(rule_name)
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_report() {
        let report = ValidationReport::new();
        assert!(report.is_valid());
        assert!(!report.has_errors());
        assert!(!report.has_warnings());
    }

    #[test]
    fn test_validation_report_with_errors() {
        let mut report = ValidationReport::new();
        let error = ValidationError {
            rule_name: "test".to_string(),
            message: "Test error".to_string(),
            operation_indices: vec![0],
            qubits: vec![0],
            suggestion: None,
        };
        let result = ValidationResult::error(error);
        report.add_result("test_rule", result);

        assert!(!report.is_valid());
        assert!(report.has_errors());
        assert_eq!(report.errors().len(), 1);
    }

    // Tests for previously uncovered lines

    #[test]
    fn test_validation_report_format_invalid() {
        // Covers lines 50, 55-57: invalid report format
        use crate::Circuit;

        let mut report = ValidationReport::new();
        let error = ValidationError {
            rule_name: "rule_a".to_string(),
            message: "something failed".to_string(),
            operation_indices: vec![],
            qubits: vec![],
            suggestion: None,
        };
        report.add_result("rule_a", ValidationResult::error(error));

        let circuit = Circuit::new(2);
        let formatted = report.format(&circuit);
        assert!(formatted.contains("✗ Circuit validation failed"));
        assert!(formatted.contains("rule_a"));
    }

    #[test]
    fn test_validation_report_format_valid() {
        // Covers line 50: valid report format (✓ passed)
        use crate::Circuit;

        let report = ValidationReport::new();
        let circuit = Circuit::new(2);
        let formatted = report.format(&circuit);
        assert!(formatted.contains("✓ Circuit validation passed"));
    }

    #[test]
    fn test_validation_report_format_with_warnings() {
        // Covers lines 63-66: warnings block in format
        use crate::validation::rules::ValidationWarning;
        use crate::Circuit;

        let mut report = ValidationReport::new();
        let warning = ValidationWarning {
            rule_name: "warn_rule".to_string(),
            message: "be careful".to_string(),
            operation_indices: vec![],
        };
        let result = ValidationResult::ok().with_warning(warning);
        report.add_result("warn_rule", result);

        let circuit = Circuit::new(2);
        let formatted = report.format(&circuit);
        assert!(formatted.contains("Warnings:"));
        assert!(formatted.contains("warn_rule"));
        assert!(formatted.contains("be careful"));
    }

    #[test]
    fn test_validation_report_has_warnings() {
        // Covers lines 83-84, 86: has_warnings
        use crate::validation::rules::ValidationWarning;

        let mut report = ValidationReport::new();
        assert!(!report.has_warnings());

        let warning = ValidationWarning {
            rule_name: "w".to_string(),
            message: "watch out".to_string(),
            operation_indices: vec![],
        };
        let result = ValidationResult::ok().with_warning(warning);
        report.add_result("w", result);

        assert!(report.has_warnings());
        assert_eq!(report.warnings().len(), 1);
    }

    #[test]
    fn test_validation_report_get_result() {
        // Covers lines 91-92, 97-98: get_result
        let mut report = ValidationReport::new();
        assert!(report.get_result("nonexistent").is_none());

        let error = ValidationError {
            rule_name: "r".to_string(),
            message: "e".to_string(),
            operation_indices: vec![],
            qubits: vec![],
            suggestion: None,
        };
        report.add_result("r", ValidationResult::error(error));

        let result = report.get_result("r");
        assert!(result.is_some());
        assert!(!result.unwrap().is_valid());
    }

    #[test]
    fn test_validation_report_default() {
        let report = ValidationReport::default();
        assert!(report.is_valid());
        assert!(!report.has_errors());
    }
}
