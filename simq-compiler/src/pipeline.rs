//! Pre-configured optimization pipelines
//!
//! This module provides factory functions for creating common optimization pipelines
//! with sensible defaults for different use cases.

use crate::compiler::{Compiler, CompilerBuilder};
use crate::passes::{
    AdvancedTemplateMatching, DeadCodeElimination, GateCommutation, GateFusion,
    TemplateSubstitution,
};
use std::sync::Arc;

/// Optimization level presets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationLevel {
    /// No optimization
    O0,
    /// Basic optimization (dead code elimination only)
    O1,
    /// Standard optimization (dead code + fusion + templates)
    #[default]
    O2,
    /// Aggressive optimization (all passes with commutation)
    O3,
}

/// Create a compiler with the specified optimization level
///
/// # Optimization Levels
///
/// - **O0**: No optimization (useful for debugging)
/// - **O1**: Dead code elimination only (fast, minimal changes)
/// - **O2**: Dead code + fusion + templates (balanced, recommended)
/// - **O3**: All passes including commutation (aggressive, max optimization)
///
/// # Example
/// ```ignore
/// use simq_compiler::pipeline::{create_compiler, OptimizationLevel};
/// use simq_core::Circuit;
///
/// let compiler = create_compiler(OptimizationLevel::O2);
/// let mut circuit = Circuit::new(3);
/// // ... add gates ...
/// compiler.compile(&mut circuit)?;
/// ```
pub fn create_compiler(level: OptimizationLevel) -> Compiler {
    match level {
        OptimizationLevel::O0 => create_o0_compiler(),
        OptimizationLevel::O1 => create_o1_compiler(),
        OptimizationLevel::O2 => create_o2_compiler(),
        OptimizationLevel::O3 => create_o3_compiler(),
    }
}

/// Create O0 compiler (no optimization)
fn create_o0_compiler() -> Compiler {
    CompilerBuilder::new()
        .max_iterations(1)
        .enable_timing(false)
        .build()
}

/// Create O1 compiler (dead code elimination only)
fn create_o1_compiler() -> Compiler {
    CompilerBuilder::new()
        .add_pass(Arc::new(DeadCodeElimination::new()))
        .max_iterations(3)
        .enable_timing(true)
        .build()
}

/// Create O2 compiler (standard optimization)
///
/// This is the recommended default configuration, providing a good balance
/// between optimization quality and compilation time.
fn create_o2_compiler() -> Compiler {
    CompilerBuilder::new()
        .add_pass(Arc::new(DeadCodeElimination::new()))
        .add_pass(Arc::new(AdvancedTemplateMatching::new()))
        .add_pass(Arc::new(GateFusion::new()))
        .max_iterations(5)
        .enable_timing(true)
        .build()
}

/// Create O3 compiler (aggressive optimization)
///
/// Applies all available optimization passes including commutation.
/// This may take longer but produces the most optimized circuits.
fn create_o3_compiler() -> Compiler {
    CompilerBuilder::new()
        .add_pass(Arc::new(DeadCodeElimination::new()))
        .add_pass(Arc::new(GateCommutation::new()))
        .add_pass(Arc::new(AdvancedTemplateMatching::new()))
        .add_pass(Arc::new(GateFusion::new()))
        .max_iterations(10)
        .enable_timing(true)
        .build()
}

/// Create a custom compiler with specific passes
///
/// This builder provides a fluent API for creating custom optimization pipelines.
///
/// # Example
/// ```ignore
/// use simq_compiler::pipeline::PipelineBuilder;
///
/// let compiler = PipelineBuilder::new()
///     .with_dead_code_elimination()
///     .with_gate_fusion()
///     .max_iterations(5)
///     .build();
/// ```
pub struct PipelineBuilder {
    builder: CompilerBuilder,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    pub fn new() -> Self {
        Self {
            builder: CompilerBuilder::new(),
        }
    }

    /// Add dead code elimination pass
    pub fn with_dead_code_elimination(mut self) -> Self {
        self.builder = self.builder.add_pass(Arc::new(DeadCodeElimination::new()));
        self
    }

    /// Add gate fusion pass
    pub fn with_gate_fusion(mut self) -> Self {
        self.builder = self.builder.add_pass(Arc::new(GateFusion::new()));
        self
    }

    /// Add gate commutation pass
    pub fn with_gate_commutation(mut self) -> Self {
        self.builder = self.builder.add_pass(Arc::new(GateCommutation::new()));
        self
    }

    /// Add template substitution pass
    pub fn with_template_substitution(mut self) -> Self {
        self.builder = self.builder.add_pass(Arc::new(TemplateSubstitution::new()));
        self
    }

    /// Add advanced template matching pass (recommended over basic template substitution)
    pub fn with_advanced_template_matching(mut self) -> Self {
        self.builder = self
            .builder
            .add_pass(Arc::new(AdvancedTemplateMatching::new()));
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.builder = self.builder.max_iterations(max_iterations);
        self
    }

    /// Enable or disable timing statistics
    pub fn enable_timing(mut self, enable: bool) -> Self {
        self.builder = self.builder.enable_timing(enable);
        self
    }

    /// Set the minimum benefit score for passes to run
    pub fn min_benefit_score(mut self, score: f64) -> Self {
        self.builder = self.builder.min_benefit_score(score);
        self
    }

    /// Build the compiler
    pub fn build(self) -> Compiler {
        self.builder.build()
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_core::{gate::Gate, Circuit, QubitId};
    use std::sync::Arc;

    #[derive(Debug)]
    struct MockGate {
        name: String,
    }

    impl Gate for MockGate {
        fn name(&self) -> &str {
            &self.name
        }
        fn num_qubits(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_o0_no_optimization() {
        let compiler = create_compiler(OptimizationLevel::O0);
        let mut circuit = Circuit::new(2);

        let x = Arc::new(MockGate {
            name: "X".to_string(),
        });
        circuit.add_gate(x.clone(), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(x, &[QubitId::new(0)]).unwrap();

        let result = compiler.compile(&mut circuit).unwrap();
        assert!(!result.modified); // O0 doesn't optimize
        assert_eq!(circuit.len(), 2);
    }

    #[test]
    fn test_o1_basic_optimization() {
        let compiler = create_compiler(OptimizationLevel::O1);
        assert_eq!(compiler.num_passes(), 1);
    }

    #[test]
    fn test_o2_standard_optimization() {
        let compiler = create_compiler(OptimizationLevel::O2);
        assert_eq!(compiler.num_passes(), 3);
    }

    #[test]
    fn test_o3_aggressive_optimization() {
        let compiler = create_compiler(OptimizationLevel::O3);
        assert_eq!(compiler.num_passes(), 4);
    }

    #[test]
    fn test_pipeline_builder() {
        let compiler = PipelineBuilder::new()
            .with_dead_code_elimination()
            .with_gate_fusion()
            .max_iterations(5)
            .enable_timing(false)
            .build();

        assert_eq!(compiler.num_passes(), 2);
    }

    #[test]
    fn test_pipeline_builder_all_passes() {
        let compiler = PipelineBuilder::new()
            .with_dead_code_elimination()
            .with_gate_commutation()
            .with_gate_fusion()
            .with_template_substitution()
            .build();

        assert_eq!(compiler.num_passes(), 4);
    }
}
