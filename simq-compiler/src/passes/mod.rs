//! Circuit optimization pass trait and implementations

use simq_core::{Circuit, Result};

/// Trait for circuit optimization passes
///
/// An optimization pass transforms a circuit to improve its properties
/// (e.g., reduce gate count, depth, or improve execution efficiency).
///
/// # Example
/// ```ignore
/// use simq_compiler::passes::OptimizationPass;
/// use simq_core::Circuit;
///
/// struct MyPass;
///
/// impl OptimizationPass for MyPass {
///     fn name(&self) -> &str {
///         "my-pass"
///     }
///
///     fn apply(&self, circuit: &mut Circuit) -> Result<bool> {
///         // Transform circuit
///         Ok(true) // Return true if modified
///     }
/// }
/// ```
pub trait OptimizationPass: Send + Sync {
    /// The name of this optimization pass
    fn name(&self) -> &str;

    /// Apply the optimization pass to a circuit
    ///
    /// # Arguments
    /// * `circuit` - The circuit to optimize (modified in-place)
    ///
    /// # Returns
    /// * `Ok(true)` if the circuit was modified
    /// * `Ok(false)` if the circuit was not modified
    /// * `Err(_)` if an error occurred
    ///
    /// # Errors
    /// Returns an error if the pass fails to apply or creates an invalid circuit
    fn apply(&self, circuit: &mut Circuit) -> Result<bool>;

    /// Optional description of what this pass does
    fn description(&self) -> Option<&str> {
        None
    }

    /// Whether this pass should run multiple times until no changes occur
    ///
    /// If true, the compiler will repeatedly apply this pass until it returns false
    /// (indicating no more changes can be made).
    fn iterative(&self) -> bool {
        false
    }

    /// Estimated benefit of this pass (0.0 to 1.0)
    ///
    /// Higher values indicate passes that typically provide more optimization.
    /// Used for pass ordering.
    fn benefit_score(&self) -> f64 {
        0.5
    }
}

/// Statistics about a pass execution
#[derive(Debug, Clone)]
pub struct PassStatistics {
    /// The name of the pass
    pub pass_name: String,
    /// Number of times the pass was applied
    pub applications: usize,
    /// Total time spent in this pass (microseconds)
    pub time_us: u64,
    /// Whether the pass modified the circuit
    pub modified: bool,
}

impl PassStatistics {
    /// Create new pass statistics
    pub fn new(pass_name: String) -> Self {
        Self {
            pass_name,
            applications: 0,
            time_us: 0,
            modified: false,
        }
    }
}

/// Result of applying optimization passes
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Whether any pass modified the circuit
    pub modified: bool,
    /// Statistics for each pass
    pub pass_stats: Vec<PassStatistics>,
    /// Total optimization time (microseconds)
    pub total_time_us: u64,
}

impl OptimizationResult {
    /// Create a new optimization result
    pub fn new() -> Self {
        Self {
            modified: false,
            pass_stats: Vec::new(),
            total_time_us: 0,
        }
    }

    /// Add statistics for a pass
    pub fn add_pass_stats(&mut self, stats: PassStatistics) {
        self.modified |= stats.modified;
        self.total_time_us += stats.time_us;
        self.pass_stats.push(stats);
    }
}

impl Default for OptimizationResult {
    fn default() -> Self {
        Self::new()
    }
}

// Pass implementations
mod dead_code_elimination;
mod gate_commutation;
mod gate_fusion;
mod template_matching;
mod template_substitution;

pub use dead_code_elimination::DeadCodeElimination;
pub use gate_commutation::GateCommutation;
pub use gate_fusion::GateFusion;
pub use template_matching::AdvancedTemplateMatching;
pub use template_substitution::TemplateSubstitution;

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal pass that only implements the required trait methods, relying
    // on the default implementations for `description`, `iterative`, and
    // `benefit_score` (lines 47-48, 55-56).
    struct MinimalPass;

    impl OptimizationPass for MinimalPass {
        fn name(&self) -> &str {
            "minimal-pass"
        }

        fn apply(&self, _circuit: &mut Circuit) -> Result<bool> {
            Ok(false)
        }
    }

    #[test]
    fn test_optimization_pass_default_description_is_none() {
        let pass = MinimalPass;
        assert_eq!(pass.description(), None);
    }

    #[test]
    fn test_optimization_pass_default_iterative_is_false() {
        let pass = MinimalPass;
        assert!(!pass.iterative());
    }

    #[test]
    fn test_optimization_pass_default_benefit_score() {
        let pass = MinimalPass;
        assert_eq!(pass.benefit_score(), 0.5);
    }

    #[test]
    fn test_optimization_result_default_matches_new() {
        // Covers `impl Default for OptimizationResult` (lines 123-124), which
        // delegates to `Self::new()`.
        let default_result = OptimizationResult::default();
        let new_result = OptimizationResult::new();

        assert_eq!(default_result.modified, new_result.modified);
        assert_eq!(default_result.pass_stats.len(), new_result.pass_stats.len());
        assert_eq!(default_result.total_time_us, new_result.total_time_us);

        assert!(!default_result.modified);
        assert!(default_result.pass_stats.is_empty());
        assert_eq!(default_result.total_time_us, 0);
    }

    #[test]
    fn test_pass_statistics_new_and_add_pass_stats() {
        let mut result = OptimizationResult::new();
        let mut stats = PassStatistics::new("minimal-pass".to_string());
        stats.applications = 1;
        stats.time_us = 42;
        stats.modified = true;

        result.add_pass_stats(stats);

        assert!(result.modified);
        assert_eq!(result.total_time_us, 42);
        assert_eq!(result.pass_stats.len(), 1);
        assert_eq!(result.pass_stats[0].pass_name, "minimal-pass");
    }
}
