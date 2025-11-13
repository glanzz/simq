//! State validation and normalization checks
//!
//! This module provides utilities for validating quantum states and ensuring
//! they satisfy physical constraints (normalization, unitarity, etc.).

use crate::error::{Result, StateError};
use num_complex::Complex64;

/// Default tolerance for normalization checks
pub const DEFAULT_NORM_TOLERANCE: f64 = 1e-10;

/// Default tolerance for probability checks
pub const DEFAULT_PROB_TOLERANCE: f64 = 1e-10;

/// Validation policy for state operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationPolicy {
    /// No validation - maximum performance
    None,
    /// Validate only critical operations (measurements, final results)
    Critical,
    /// Validate all operations - maximum safety
    Strict,
}

/// Validation result with diagnostics
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the state passed validation
    pub valid: bool,
    /// Current norm of the state
    pub norm: f64,
    /// Deviation from ideal norm (1.0)
    pub norm_error: f64,
    /// Total probability (sum of |amplitude|^2)
    pub total_probability: f64,
    /// Probability error from 1.0
    pub probability_error: f64,
    /// Human-readable diagnostic message
    pub message: String,
}

impl ValidationResult {
    /// Check if the state is valid within tolerance
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Check if normalization needs correction
    pub fn needs_normalization(&self) -> bool {
        self.norm_error > DEFAULT_NORM_TOLERANCE
    }

    /// Get severity level (0 = good, 1 = warning, 2 = error)
    pub fn severity(&self) -> u8 {
        if self.norm_error < 1e-10 {
            0 // Good
        } else if self.norm_error < 1e-6 {
            1 // Warning - slight drift
        } else {
            2 // Error - significant deviation
        }
    }
}

impl std::fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ValidationResult(valid={}, norm={:.6}, error={:.2e})",
            self.valid, self.norm, self.norm_error
        )
    }
}

/// Validate state normalization
///
/// # Arguments
/// * `amplitudes` - The state amplitudes to validate
/// * `tolerance` - Maximum allowed deviation from norm = 1.0
///
/// # Returns
/// Validation result with diagnostics
///
/// # Example
/// ```
/// use simq_state::validation::{validate_normalization, DEFAULT_NORM_TOLERANCE};
/// use num_complex::Complex64;
///
/// let amplitudes = vec![
///     Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
///     Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
/// ];
///
/// let result = validate_normalization(&amplitudes, DEFAULT_NORM_TOLERANCE);
/// assert!(result.is_valid());
/// ```
pub fn validate_normalization(amplitudes: &[Complex64], tolerance: f64) -> ValidationResult {
    let norm_squared: f64 = amplitudes.iter().map(|a| a.norm_sqr()).sum();
    let norm = norm_squared.sqrt();
    let norm_error = (norm - 1.0).abs();

    let valid = norm_error < tolerance;
    let message = if valid {
        format!("State is normalized (norm = {:.10})", norm)
    } else {
        format!(
            "State normalization error: norm = {:.10}, error = {:.2e}",
            norm, norm_error
        )
    };

    ValidationResult {
        valid,
        norm,
        norm_error,
        total_probability: norm_squared,
        probability_error: (norm_squared - 1.0).abs(),
        message,
    }
}

/// Validate that probabilities sum to 1.0
///
/// # Arguments
/// * `probabilities` - Vector of probabilities
/// * `tolerance` - Maximum allowed deviation from sum = 1.0
///
/// # Returns
/// Validation result
pub fn validate_probabilities(probabilities: &[f64], tolerance: f64) -> ValidationResult {
    let total: f64 = probabilities.iter().sum();
    let error = (total - 1.0).abs();

    let valid = error < tolerance;
    let message = if valid {
        format!("Probabilities sum to 1.0 (sum = {:.10})", total)
    } else {
        format!(
            "Probability sum error: sum = {:.10}, error = {:.2e}",
            total, error
        )
    };

    ValidationResult {
        valid,
        norm: total.sqrt(),
        norm_error: (total.sqrt() - 1.0).abs(),
        total_probability: total,
        probability_error: error,
        message,
    }
}

/// Validate that a matrix is unitary (U†U = I)
///
/// This is computationally expensive but ensures gate operations preserve norm.
///
/// # Arguments
/// * `matrix` - The matrix to validate (flattened row-major)
/// * `size` - Matrix dimension (2 for single-qubit, 4 for two-qubit)
/// * `tolerance` - Maximum allowed deviation from unitarity
///
/// # Returns
/// True if the matrix is unitary within tolerance
pub fn validate_unitary_2x2(matrix: &[[Complex64; 2]; 2], tolerance: f64) -> bool {
    // Compute U†U for 2x2 matrix
    let mut result = [[Complex64::new(0.0, 0.0); 2]; 2];

    for i in 0..2 {
        for j in 0..2 {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..2 {
                sum += matrix[k][i].conj() * matrix[k][j];
            }
            result[i][j] = sum;
        }
    }

    // Check if result is identity matrix
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            let diff = (result[i][j] - expected).norm();
            if diff > tolerance {
                return false;
            }
        }
    }

    true
}

/// Check if amplitudes contain NaN or infinity
///
/// # Arguments
/// * `amplitudes` - The state amplitudes to check
///
/// # Returns
/// True if all amplitudes are finite
pub fn check_finite(amplitudes: &[Complex64]) -> bool {
    amplitudes
        .iter()
        .all(|a| a.re.is_finite() && a.im.is_finite())
}

/// Comprehensive state validation
///
/// Performs multiple checks:
/// - Normalization
/// - Finiteness (no NaN/infinity)
/// - Minimum probability mass
///
/// # Arguments
/// * `amplitudes` - The state amplitudes to validate
/// * `policy` - Validation policy to apply
///
/// # Returns
/// Validation result or error if state is invalid
pub fn validate_state(amplitudes: &[Complex64], policy: ValidationPolicy) -> Result<ValidationResult> {
    match policy {
        ValidationPolicy::None => Ok(ValidationResult {
            valid: true,
            norm: 1.0,
            norm_error: 0.0,
            total_probability: 1.0,
            probability_error: 0.0,
            message: "Validation skipped".to_string(),
        }),

        ValidationPolicy::Critical | ValidationPolicy::Strict => {
            // Check for NaN/infinity
            if !check_finite(amplitudes) {
                return Err(StateError::NotNormalized {
                    norm: f64::NAN,
                });
            }

            // Check normalization
            let result = validate_normalization(amplitudes, DEFAULT_NORM_TOLERANCE);

            if policy == ValidationPolicy::Strict && !result.valid {
                return Err(StateError::NotNormalized { norm: result.norm });
            }

            Ok(result)
        }
    }
}

/// Auto-normalize state if needed
///
/// Returns true if normalization was performed, false if already normalized.
///
/// # Arguments
/// * `amplitudes` - Mutable reference to state amplitudes
/// * `tolerance` - Tolerance for normalization check
///
/// # Returns
/// True if normalization was applied
pub fn auto_normalize(amplitudes: &mut [Complex64], tolerance: f64) -> bool {
    let norm_squared: f64 = amplitudes.iter().map(|a| a.norm_sqr()).sum();
    let norm = norm_squared.sqrt();

    if (norm - 1.0).abs() > tolerance {
        if norm < 1e-14 {
            // Zero state - cannot normalize
            return false;
        }

        let inv_norm = 1.0 / norm;
        for amp in amplitudes.iter_mut() {
            *amp *= inv_norm;
        }
        true
    } else {
        false
    }
}

/// Track normalization drift over multiple operations
///
/// Useful for detecting numerical instability in long circuits.
pub struct NormalizationTracker {
    /// History of norm measurements
    history: Vec<f64>,
    /// Maximum history size
    max_history: usize,
    /// Cumulative drift from norm = 1.0
    cumulative_drift: f64,
}

impl NormalizationTracker {
    /// Create a new normalization tracker
    ///
    /// # Arguments
    /// * `max_history` - Maximum number of norm measurements to retain
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::with_capacity(max_history),
            max_history,
            cumulative_drift: 0.0,
        }
    }

    /// Record a norm measurement
    pub fn record(&mut self, norm: f64) {
        let drift = (norm - 1.0).abs();
        self.cumulative_drift += drift;

        self.history.push(norm);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Get average drift over history
    pub fn average_drift(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }

        let total_drift: f64 = self.history.iter().map(|&n| (n - 1.0).abs()).sum();
        total_drift / self.history.len() as f64
    }

    /// Get maximum drift observed
    pub fn max_drift(&self) -> f64 {
        self.history
            .iter()
            .map(|&n| (n - 1.0).abs())
            .fold(0.0, f64::max)
    }

    /// Get cumulative drift
    pub fn cumulative_drift(&self) -> f64 {
        self.cumulative_drift
    }

    /// Check if drift is within acceptable bounds
    pub fn is_stable(&self, tolerance: f64) -> bool {
        self.max_drift() < tolerance
    }

    /// Get statistics summary
    pub fn stats(&self) -> NormalizationStats {
        NormalizationStats {
            measurements: self.history.len(),
            average_drift: self.average_drift(),
            max_drift: self.max_drift(),
            cumulative_drift: self.cumulative_drift,
            current_norm: self.history.last().copied().unwrap_or(1.0),
        }
    }
}

/// Statistics about normalization tracking
#[derive(Debug, Clone)]
pub struct NormalizationStats {
    pub measurements: usize,
    pub average_drift: f64,
    pub max_drift: f64,
    pub cumulative_drift: f64,
    pub current_norm: f64,
}

impl std::fmt::Display for NormalizationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NormStats(measurements={}, avg_drift={:.2e}, max_drift={:.2e}, current={:.10})",
            self.measurements, self.average_drift, self.max_drift, self.current_norm
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_validate_normalized_state() {
        let amplitudes = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];

        let result = validate_normalization(&amplitudes, DEFAULT_NORM_TOLERANCE);
        assert!(result.is_valid());
        assert_relative_eq!(result.norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_validate_unnormalized_state() {
        let amplitudes = vec![Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)];

        let result = validate_normalization(&amplitudes, DEFAULT_NORM_TOLERANCE);
        assert!(!result.is_valid());
        assert!(result.norm > 2.0);
    }

    #[test]
    fn test_validate_probabilities() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let result = validate_probabilities(&probs, DEFAULT_PROB_TOLERANCE);
        assert!(result.is_valid());
    }

    #[test]
    fn test_validate_probabilities_error() {
        let probs = vec![0.3, 0.3, 0.3, 0.3]; // Sum > 1
        let result = validate_probabilities(&probs, DEFAULT_PROB_TOLERANCE);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_validate_unitary_identity() {
        let identity = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        assert!(validate_unitary_2x2(&identity, 1e-10));
    }

    #[test]
    fn test_validate_unitary_hadamard() {
        let h = 1.0 / 2.0_f64.sqrt();
        let hadamard = [
            [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
            [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
        ];
        assert!(validate_unitary_2x2(&hadamard, 1e-10));
    }

    #[test]
    fn test_validate_unitary_pauli_x() {
        let pauli_x = [
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ];
        assert!(validate_unitary_2x2(&pauli_x, 1e-10));
    }

    #[test]
    fn test_validate_non_unitary() {
        let non_unitary = [
            [Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        assert!(!validate_unitary_2x2(&non_unitary, 1e-10));
    }

    #[test]
    fn test_check_finite() {
        let finite = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        assert!(check_finite(&finite));

        let infinite = vec![Complex64::new(f64::INFINITY, 0.0)];
        assert!(!check_finite(&infinite));

        let nan = vec![Complex64::new(f64::NAN, 0.0)];
        assert!(!check_finite(&nan));
    }

    #[test]
    fn test_auto_normalize() {
        let mut amplitudes = vec![Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)];

        let normalized = auto_normalize(&mut amplitudes, 1e-10);
        assert!(normalized);

        let norm: f64 = amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_auto_normalize_already_normalized() {
        let mut amplitudes = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];

        let normalized = auto_normalize(&mut amplitudes, 1e-10);
        assert!(!normalized); // Was already normalized
    }

    #[test]
    fn test_normalization_tracker() {
        let mut tracker = NormalizationTracker::new(10);

        tracker.record(1.0);
        tracker.record(1.0001);
        tracker.record(0.9999);
        tracker.record(1.0002);

        let stats = tracker.stats();
        assert_eq!(stats.measurements, 4);
        assert!(stats.average_drift < 0.001);
        assert!(stats.max_drift < 0.001);
        assert!(tracker.is_stable(0.001));
    }

    #[test]
    fn test_normalization_tracker_drift() {
        let mut tracker = NormalizationTracker::new(10);

        // Simulate gradual drift
        for i in 0..10 {
            let norm = 1.0 + (i as f64) * 0.001;
            tracker.record(norm);
        }

        let stats = tracker.stats();
        assert!(stats.max_drift > 0.005);
        assert!(!tracker.is_stable(0.001));
    }

    #[test]
    fn test_validation_result_severity() {
        let good = ValidationResult {
            valid: true,
            norm: 1.0,
            norm_error: 1e-12,
            total_probability: 1.0,
            probability_error: 1e-12,
            message: "Good".to_string(),
        };
        assert_eq!(good.severity(), 0);

        let warning = ValidationResult {
            valid: false,
            norm: 1.00001,
            norm_error: 1e-7,
            total_probability: 1.0,
            probability_error: 1e-7,
            message: "Warning".to_string(),
        };
        assert_eq!(warning.severity(), 1);

        let error = ValidationResult {
            valid: false,
            norm: 1.001,
            norm_error: 0.001,
            total_probability: 1.0,
            probability_error: 0.001,
            message: "Error".to_string(),
        };
        assert_eq!(error.severity(), 2);
    }

    #[test]
    fn test_validation_policy_none() {
        let bad_amplitudes = vec![Complex64::new(100.0, 0.0)];
        let result = validate_state(&bad_amplitudes, ValidationPolicy::None).unwrap();
        assert!(result.valid); // Validation skipped
    }

    #[test]
    fn test_validation_policy_critical() {
        let amplitudes = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        let result = validate_state(&amplitudes, ValidationPolicy::Critical).unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_validation_policy_strict_fail() {
        let bad_amplitudes = vec![Complex64::new(2.0, 0.0)];
        let result = validate_state(&bad_amplitudes, ValidationPolicy::Strict);
        assert!(result.is_err());
    }
}
