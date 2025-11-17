//! Circuit compilation pipeline
//!
//! The compiler orchestrates multiple optimization passes to transform
//! quantum circuits into optimized forms.

use crate::passes::{OptimizationPass, OptimizationResult, PassStatistics};
use simq_core::{Circuit, Result};
use std::sync::Arc;
use std::time::Instant;

/// Configuration for the circuit compiler
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Maximum number of fixed-point iterations
    pub max_iterations: usize,
    /// Whether to enable pass timing
    pub enable_timing: bool,
    /// Minimum benefit score for passes to run (0.0 to 1.0)
    pub min_benefit_score: f64,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            enable_timing: true,
            min_benefit_score: 0.0,
        }
    }
}

/// Circuit compiler that applies optimization passes
///
/// The compiler applies a sequence of optimization passes to a circuit,
/// optionally iterating until a fixed point is reached (no more changes).
///
/// # Example
/// ```ignore
/// use simq_compiler::{Compiler, CompilerConfig};
/// use simq_core::Circuit;
///
/// let config = CompilerConfig::default();
/// let mut compiler = Compiler::new(config);
///
/// // Add optimization passes
/// compiler.add_pass(Box::new(DeadCodeEliminationPass));
/// compiler.add_pass(Box::new(GateFusionPass));
///
/// // Compile a circuit
/// let mut circuit = Circuit::new(3);
/// let result = compiler.compile(&mut circuit)?;
/// ```
#[derive(Clone)]
pub struct Compiler {
    config: CompilerConfig,
    passes: Vec<Arc<dyn OptimizationPass>>,
}

impl Compiler {
    /// Create a new compiler with the given configuration
    pub fn new(config: CompilerConfig) -> Self {
        Self {
            config,
            passes: Vec::new(),
        }
    }

    /// Create a compiler with default configuration
    pub fn default() -> Self {
        Self::new(CompilerConfig::default())
    }

    /// Add an optimization pass to the compiler
    ///
    /// Passes are applied in the order they are added.
    pub fn add_pass(&mut self, pass: Arc<dyn OptimizationPass>) {
        self.passes.push(pass);
    }

    /// Add multiple optimization passes
    pub fn add_passes(&mut self, passes: Vec<Arc<dyn OptimizationPass>>) {
        self.passes.extend(passes);
    }

    /// Get the number of registered passes
    pub fn num_passes(&self) -> usize {
        self.passes.len()
    }

    /// Compile a circuit by applying all optimization passes
    ///
    /// This applies passes in order, optionally iterating until no changes occur.
    ///
    /// # Arguments
    /// * `circuit` - The circuit to compile (modified in-place)
    ///
    /// # Returns
    /// * `Ok(OptimizationResult)` with statistics about the compilation
    /// * `Err(_)` if a pass fails
    pub fn compile(&self, circuit: &mut Circuit) -> Result<OptimizationResult> {
        let start_time = Instant::now();
        let mut result = OptimizationResult::new();

        // Filter passes by benefit score
        let active_passes: Vec<_> = self
            .passes
            .iter()
            .filter(|p| p.benefit_score() >= self.config.min_benefit_score)
            .collect();

        if active_passes.is_empty() {
            return Ok(result);
        }

        // Fixed-point iteration
        for iteration in 0..self.config.max_iterations {
            let mut changed = false;

            for pass in &active_passes {
                let pass_start = if self.config.enable_timing {
                    Some(Instant::now())
                } else {
                    None
                };

                let modified = pass.apply(circuit)?;
                changed |= modified;

                if self.config.enable_timing {
                    let elapsed = pass_start.unwrap().elapsed();
                    let mut stats = PassStatistics::new(pass.name().to_string());
                    stats.applications = iteration + 1;
                    stats.time_us = elapsed.as_micros() as u64;
                    stats.modified = modified;
                    result.add_pass_stats(stats);
                }
            }

            // If no pass made changes, we've reached a fixed point
            if !changed {
                break;
            }
        }

        result.total_time_us = start_time.elapsed().as_micros() as u64;
        Ok(result)
    }

    /// Run a single pass on a circuit
    ///
    /// This is useful for testing individual passes or applying
    /// specific optimizations without running the full pipeline.
    pub fn run_pass(
        &self,
        pass: &dyn OptimizationPass,
        circuit: &mut Circuit,
    ) -> Result<bool> {
        pass.apply(circuit)
    }
}

/// Builder for constructing a compiler with common pass configurations
pub struct CompilerBuilder {
    config: CompilerConfig,
    passes: Vec<Arc<dyn OptimizationPass>>,
}

impl CompilerBuilder {
    /// Create a new compiler builder
    pub fn new() -> Self {
        Self {
            config: CompilerConfig::default(),
            passes: Vec::new(),
        }
    }

    /// Set the compiler configuration
    pub fn config(mut self, config: CompilerConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.config.max_iterations = max_iterations;
        self
    }

    /// Enable or disable pass timing
    pub fn enable_timing(mut self, enable: bool) -> Self {
        self.config.enable_timing = enable;
        self
    }

    /// Set the minimum benefit score for passes
    pub fn min_benefit_score(mut self, score: f64) -> Self {
        self.config.min_benefit_score = score;
        self
    }

    /// Add a pass to the compiler
    pub fn add_pass(mut self, pass: Arc<dyn OptimizationPass>) -> Self {
        self.passes.push(pass);
        self
    }

    /// Build the compiler
    pub fn build(self) -> Compiler {
        let mut compiler = Compiler::new(self.config);
        compiler.add_passes(self.passes);
        compiler
    }
}

impl Default for CompilerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_core::Circuit;

    struct TestPass {
        name: String,
        should_modify: bool,
        apply_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl OptimizationPass for TestPass {
        fn name(&self) -> &str {
            &self.name
        }

        fn apply(&self, _circuit: &mut Circuit) -> Result<bool> {
            self.apply_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(self.should_modify)
        }
    }

    #[test]
    fn test_compiler_creation() {
        let compiler = Compiler::default();
        assert_eq!(compiler.num_passes(), 0);
    }

    #[test]
    fn test_add_pass() {
        let mut compiler = Compiler::default();
        let apply_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pass = Arc::new(TestPass {
            name: "test".to_string(),
            should_modify: false,
            apply_count: apply_count.clone(),
        });

        compiler.add_pass(pass);
        assert_eq!(compiler.num_passes(), 1);

        let mut circuit = Circuit::new(2);
        compiler.compile(&mut circuit).unwrap();

        assert_eq!(
            apply_count.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[test]
    fn test_fixed_point_iteration() {
        let mut compiler = Compiler::default();
        let apply_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Pass that modifies on first call, then stops
        struct IterativePass {
            apply_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        }

        impl OptimizationPass for IterativePass {
            fn name(&self) -> &str {
                "iterative"
            }

            fn apply(&self, _circuit: &mut Circuit) -> Result<bool> {
                let count = self
                    .apply_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(count < 3) // Modify for first 3 calls
            }
        }

        let pass = Arc::new(IterativePass {
            apply_count: apply_count.clone(),
        });
        compiler.add_pass(pass);

        let mut circuit = Circuit::new(2);
        compiler.compile(&mut circuit).unwrap();

        // Should stop after 4 iterations (3 with changes + 1 with no changes)
        assert_eq!(
            apply_count.load(std::sync::atomic::Ordering::SeqCst),
            4
        );
    }

    #[test]
    fn test_compiler_builder() {
        let apply_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pass = Arc::new(TestPass {
            name: "test".to_string(),
            should_modify: false,
            apply_count: apply_count.clone(),
        });

        let compiler = CompilerBuilder::new()
            .max_iterations(5)
            .enable_timing(false)
            .add_pass(pass)
            .build();

        assert_eq!(compiler.num_passes(), 1);
        assert_eq!(compiler.config.max_iterations, 5);
        assert!(!compiler.config.enable_timing);
    }

    #[test]
    fn test_min_benefit_score() {
        let mut compiler = Compiler::new(CompilerConfig {
            min_benefit_score: 0.7,
            ..Default::default()
        });

        struct LowBenefitPass;
        impl OptimizationPass for LowBenefitPass {
            fn name(&self) -> &str {
                "low-benefit"
            }
            fn apply(&self, _circuit: &mut Circuit) -> Result<bool> {
                Ok(true)
            }
            fn benefit_score(&self) -> f64 {
                0.3
            }
        }

        compiler.add_pass(Arc::new(LowBenefitPass));

        let mut circuit = Circuit::new(2);
        let result = compiler.compile(&mut circuit).unwrap();

        // Pass should not run because benefit score is too low
        assert!(!result.modified);
    }
}
