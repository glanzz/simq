//! Validation report aggregation

use crate::Circuit;
use crate::validation::rules::{ValidationResult, ValidationError, ValidationWarning};
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
        self.results.values().any(|r| !r.is_valid || !r.errors.is_empty())
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
}

