//! Adaptive compilation pipeline
//!
//! This module provides adaptive pass selection based on circuit characteristics.
//! Instead of always running all passes, it analyzes the circuit and selects
//! only the passes that are likely to provide significant benefit.

use crate::circuit_analysis_pass::CircuitCharacteristics;
use crate::compiler::{Compiler, CompilerBuilder};
use crate::passes::{
    AdvancedTemplateMatching, DeadCodeElimination, GateCommutation, GateFusion,
};
use simq_core::Circuit;
use std::sync::Arc;

/// Adaptive compiler that selects passes based on circuit characteristics
pub struct AdaptiveCompiler {
    /// Whether to enable verbose logging
    pub verbose: bool,
}

impl AdaptiveCompiler {
    /// Create a new adaptive compiler
    pub fn new() -> Self {
        Self { verbose: false }
    }

    /// Create a new adaptive compiler with verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Create a compiler optimized for the given circuit
    ///
    /// This analyzes the circuit characteristics and builds a custom
    /// compiler with only the passes that are likely to be beneficial.
    ///
    /// # Example
    /// ```ignore
    /// use simq_compiler::adaptive_pipeline::AdaptiveCompiler;
    /// use simq_core::Circuit;
    ///
    /// let adaptive = AdaptiveCompiler::new();
    /// let circuit = Circuit::new(5);
    /// let compiler = adaptive.create_for_circuit(&circuit);
    /// ```
    pub fn create_for_circuit(&self, circuit: &Circuit) -> Compiler {
        let chars = CircuitCharacteristics::analyze(circuit);

        if self.verbose {
            println!("Circuit characteristics:");
            println!("  Gates: {}", chars.gate_count);
            println!("  Qubits: {}", chars.num_qubits);
            println!("  Depth: {}", chars.depth);
            println!("  Commutation density: {:.2}%", chars.commutation_density * 100.0);
            println!("  Fusion density: {:.2}%", chars.fusion_density * 100.0);
            println!("  Template density: {:.2}%", chars.template_density * 100.0);
            println!("  Dead code density: {:.2}%", chars.dead_code_density * 100.0);
        }

        self.create_from_characteristics(&chars)
    }

    /// Create a compiler from circuit characteristics
    pub fn create_from_characteristics(&self, chars: &CircuitCharacteristics) -> Compiler {
        let mut builder = CompilerBuilder::new();

        // Always run DCE first if beneficial
        if chars.should_use_dce() {
            if self.verbose {
                println!("✓ Adding Dead Code Elimination");
            }
            builder = builder.add_pass(Arc::new(DeadCodeElimination::new()));
        }

        // Add commutation pass if beneficial
        if chars.should_use_commutation() {
            if self.verbose {
                println!("✓ Adding Gate Commutation");
            }
            builder = builder.add_pass(Arc::new(GateCommutation::new()));
        }

        // Add template matching if beneficial
        if chars.should_use_templates() {
            if self.verbose {
                println!("✓ Adding Advanced Template Matching");
            }
            builder = builder.add_pass(Arc::new(AdvancedTemplateMatching::new()));
        }

        // Add fusion pass if beneficial
        if chars.should_use_fusion() {
            if self.verbose {
                println!("✓ Adding Gate Fusion");
            }
            builder = builder.add_pass(Arc::new(GateFusion::new()));
        }

        // Set iterations based on circuit size
        let iterations = chars.suggest_iterations();
        if self.verbose {
            println!("Setting max iterations: {}", iterations);
        }

        builder
            .max_iterations(iterations)
            .enable_timing(true)
            .build()
    }
}

impl Default for AdaptiveCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-level optimization strategy
///
/// Applies optimization in multiple levels:
/// 1. Coarse optimization: Quick cleanup (O1-level)
/// 2. Medium optimization: Standard passes (O2-level)
/// 3. Fine optimization: Aggressive optimization (O3-level)
pub struct MultiLevelOptimizer {
    /// Whether to enable verbose logging
    pub verbose: bool,
}

impl MultiLevelOptimizer {
    /// Create a new multi-level optimizer
    pub fn new() -> Self {
        Self { verbose: false }
    }

    /// Enable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Apply multi-level optimization to a circuit
    ///
    /// This applies optimization in three stages:
    /// 1. Coarse: Quick DCE pass to reduce circuit size
    /// 2. Medium: Standard template matching and fusion
    /// 3. Fine: Aggressive optimization with commutation
    ///
    /// Returns the total number of gates removed.
    pub fn optimize(&self, circuit: &mut Circuit) -> crate::passes::OptimizationResult {
        use crate::pipeline::{create_compiler, OptimizationLevel};

        let original_size = circuit.len();

        // Level 1: Coarse optimization (quick cleanup)
        if self.verbose {
            println!("\n=== Level 1: Coarse Optimization ===");
            println!("Initial gates: {}", circuit.len());
        }

        let compiler_o1 = create_compiler(OptimizationLevel::O1);
        let result_o1 = compiler_o1.compile(circuit).unwrap();

        if self.verbose {
            println!("After O1: {} gates (removed: {})",
                circuit.len(),
                original_size - circuit.len()
            );
        }

        // Level 2: Medium optimization (standard passes)
        if self.verbose {
            println!("\n=== Level 2: Medium Optimization ===");
        }

        let after_o1 = circuit.len();
        let compiler_o2 = create_compiler(OptimizationLevel::O2);
        let result_o2 = compiler_o2.compile(circuit).unwrap();

        if self.verbose {
            println!("After O2: {} gates (removed: {})",
                circuit.len(),
                after_o1 - circuit.len()
            );
        }

        // Level 3: Fine optimization (aggressive)
        // Only apply if circuit is small enough (to avoid long compilation times)
        let chars = CircuitCharacteristics::analyze(circuit);
        if matches!(chars.size_category(), crate::circuit_analysis_pass::CircuitSize::Small | crate::circuit_analysis_pass::CircuitSize::Medium) {
            if self.verbose {
                println!("\n=== Level 3: Fine Optimization ===");
            }

            let after_o2 = circuit.len();
            let compiler_o3 = create_compiler(OptimizationLevel::O3);
            let _result_o3 = compiler_o3.compile(circuit).unwrap();

            if self.verbose {
                println!("After O3: {} gates (removed: {})",
                    circuit.len(),
                    after_o2 - circuit.len()
                );
            }
        } else if self.verbose {
            println!("\n=== Skipping Level 3 (circuit too large) ===");
        }

        if self.verbose {
            println!("\n=== Summary ===");
            println!("Total gates removed: {} ({:.1}%)",
                original_size - circuit.len(),
                (original_size - circuit.len()) as f64 / original_size as f64 * 100.0
            );
        }

        // Combine results
        let mut combined = result_o1;
        combined.pass_stats.extend(result_o2.pass_stats);
        combined.total_time_us += result_o2.total_time_us;
        combined.modified |= result_o2.modified;
        combined
    }
}

impl Default for MultiLevelOptimizer {
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
    fn test_adaptive_compiler_empty_circuit() {
        let circuit = Circuit::new(3);
        let adaptive = AdaptiveCompiler::new();
        let compiler = adaptive.create_for_circuit(&circuit);

        // Empty circuit should get minimal passes
        assert!(compiler.num_passes() <= 1);
    }

    #[test]
    fn test_adaptive_compiler_small_circuit() {
        let mut circuit = Circuit::new(2);
        let x = Arc::new(MockGate {
            name: "X".to_string(),
        });

        // Add a few gates
        for _ in 0..5 {
            circuit.add_gate(x.clone(), &[QubitId::new(0)]).unwrap();
        }

        let adaptive = AdaptiveCompiler::new();
        let compiler = adaptive.create_for_circuit(&circuit);

        // Should add some passes
        assert!(compiler.num_passes() > 0);
    }

    #[test]
    fn test_multi_level_optimizer() {
        let mut circuit = Circuit::new(2);
        let x = Arc::new(MockGate {
            name: "X".to_string(),
        });

        // Add inverse pairs (should be removed)
        circuit.add_gate(x.clone(), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(x.clone(), &[QubitId::new(0)]).unwrap();

        let optimizer = MultiLevelOptimizer::new();
        let result = optimizer.optimize(&mut circuit);

        // Circuit should be optimized
        assert!(result.modified);
        assert_eq!(circuit.len(), 0);
    }

    #[test]
    fn test_adaptive_with_verbose() {
        let circuit = Circuit::new(3);
        let adaptive = AdaptiveCompiler::new().with_verbose(true);
        assert!(adaptive.verbose);

        let _compiler = adaptive.create_for_circuit(&circuit);
    }
}
