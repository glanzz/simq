//! Circuit optimization and compilation for SimQ
//!
//! This crate provides optimization passes for quantum circuits, including:
//! - **Gate fusion**: Combining adjacent single-qubit gates
//! - **Dead code elimination**: Removing gates that don't affect the output
//! - **Gate commutation**: Reordering commuting gates to reduce depth
//! - **Template substitution**: Pattern matching and replacement of gate sequences
//! - **Lazy evaluation**: Deferring gate matrix computation until needed
//! - **Circuit simplification**: Reducing gate count and depth
//! - **Gate decomposition**: Translating gates to different basis sets
//! - **Matrix computation**: Advanced matrix operations for quantum gates
//! - **Circuit analysis**: Statistics, resource estimation, and parallelism analysis
//!
//! # Optimization Pipeline
//!
//! The compiler module provides a flexible optimization pipeline that can apply
//! multiple passes with fixed-point iteration:
//!
//! ```ignore
//! use simq_compiler::{CompilerBuilder, passes::*};
//! use simq_core::Circuit;
//! use std::sync::Arc;
//!
//! // Create an optimization pipeline
//! let compiler = CompilerBuilder::new()
//!     .add_pass(Arc::new(DeadCodeElimination::new()))
//!     .add_pass(Arc::new(GateCommutation::new()))
//!     .add_pass(Arc::new(GateFusion::new()))
//!     .add_pass(Arc::new(TemplateSubstitution::new()))
//!     .max_iterations(10)
//!     .build();
//!
//! // Optimize a circuit
//! let mut circuit = Circuit::new(3);
//! // ... add gates ...
//! let result = compiler.compile(&mut circuit)?;
//! println!("Optimized: {} gates removed", result.pass_stats.len());
//! ```
//!
//! # Gate Decomposition
//!
//! The decomposition module provides comprehensive gate decomposition functionality
//! for translating quantum circuits into different basis gate sets:
//!
//! ```ignore
//! use simq_compiler::decomposition::{DecompositionConfig, BasisGateSet, UniversalDecomposer};
//!
//! // Configure decomposition to IBM basis
//! let config = DecompositionConfig {
//!     basis: BasisGateSet::IBM,
//!     optimization_level: 2,
//!     fidelity_threshold: 0.9999,
//!     ..Default::default()
//! };
//!
//! // Create decomposer and decompose gates
//! let decomposer = UniversalDecomposer::new(config);
//! let result = decomposer.decompose_gate(&gate)?;
//! ```
//!
//! # Circuit Analysis
//!
//! The analysis module provides comprehensive circuit analysis including gate statistics,
//! resource estimation, and parallelism analysis:
//!
//! ```ignore
//! use simq_compiler::CircuitAnalysis;
//! use simq_core::Circuit;
//!
//! // Analyze a circuit
//! let circuit = Circuit::new(3);
//! // ... add gates ...
//! let analysis = CircuitAnalysis::analyze(&circuit)?;
//!
//! println!("{}", analysis);
//! // Prints:
//! // - Gate statistics (counts by type, depth)
//! // - Resource estimates (memory, time)
//! // - Parallelism analysis (layers, parallelism factor)
//! ```
//!
//! # Adaptive Compilation
//!
//! The adaptive pipeline automatically selects optimization passes based on circuit
//! characteristics:
//!
//! ```ignore
//! use simq_compiler::adaptive_pipeline::AdaptiveCompiler;
//! use simq_core::Circuit;
//!
//! let mut circuit = Circuit::new(5);
//! // ... add gates ...
//!
//! // Adaptive compiler analyzes the circuit and selects optimal passes
//! let adaptive = AdaptiveCompiler::new().with_verbose(true);
//! let compiler = adaptive.create_for_circuit(&circuit);
//! compiler.compile(&mut circuit)?;
//! ```
//!
//! # Hardware-Aware Compilation
//!
//! Optimize circuits for specific quantum hardware platforms:
//!
//! ```ignore
//! use simq_compiler::hardware_aware::{HardwareType, CostModel};
//! use simq_core::Circuit;
//!
//! let circuit = Circuit::new(3);
//! let cost_model = CostModel::new(HardwareType::IBM);
//! let cost = cost_model.circuit_cost(&circuit);
//! println!("Circuit cost on IBM hardware: {}", cost);
//! ```
//!
//! # Compilation Caching
//!
//! Cache compilation results to avoid redundant optimization:
//!
//! ```ignore
//! use simq_compiler::{CachedCompiler, pipeline::{create_compiler, OptimizationLevel}};
//! use simq_core::Circuit;
//!
//! let compiler = create_compiler(OptimizationLevel::O2);
//! let mut cached_compiler = CachedCompiler::new(compiler, 100);
//!
//! let mut circuit = Circuit::new(5);
//! // ... add gates ...
//!
//! // First compilation - cache miss
//! let result1 = cached_compiler.compile(&mut circuit)?;
//! assert!(!result1.is_cached());
//!
//! // Second compilation of same circuit - cache hit!
//! let result2 = cached_compiler.compile(&mut circuit)?;
//! assert!(result2.is_cached());
//!
//! // View cache statistics
//! let stats = cached_compiler.cache().statistics();
//! println!("Hit rate: {:.1}%", stats.hit_rate());
//! ```
//!
//! # Execution Planning
//!
//! Generate optimized execution plans with gate scheduling and parallelization:
//!
//! ```ignore
//! use simq_compiler::execution_plan::ExecutionPlanner;
//! use simq_core::Circuit;
//!
//! let circuit = Circuit::new(5);
//! // ... add gates ...
//!
//! // Generate execution plan
//! let planner = ExecutionPlanner::new();
//! let plan = planner.generate_plan(&circuit);
//!
//! println!("Circuit depth: {} layers", plan.depth);
//! println!("Parallelism factor: {:.2}x", plan.parallelism_factor);
//! println!("Estimated time: {:.2} µs", plan.total_time);
//!
//! // Visualize execution layers
//! for (i, layer) in plan.layers.iter().enumerate() {
//!     println!("Layer {}: {} gates on {} qubits",
//!         i, layer.gates.len(), layer.qubits.len());
//! }
//! ```

pub mod adaptive_pipeline;
pub mod analysis;
pub mod cache;
pub mod cached_compiler;
pub mod circuit_analysis_pass;
pub mod compiler;
pub mod decomposition;
pub mod execution_plan;
pub mod fusion;
pub mod hardware_aware;
pub mod lazy;
pub mod matrix_computation;
pub mod matrix_utils;
pub mod passes;
pub mod pipeline;

pub use adaptive_pipeline::{AdaptiveCompiler, MultiLevelOptimizer};
pub use analysis::{CircuitAnalysis, GateStatistics, ResourceEstimate};
pub use cache::{CacheStatistics, CircuitFingerprint, CompilationCache, SharedCompilationCache};
pub use cached_compiler::{CachedCompiler, CachedOptimizationResult, SharedCachedCompiler};
pub use circuit_analysis_pass::{CircuitCharacteristics, CircuitSize};
pub use compiler::{Compiler, CompilerBuilder, CompilerConfig};
pub use decomposition::{
    BasisGate, BasisGateSet, Decomposer, DecompositionConfig, DecompositionResult,
    UniversalDecomposer,
};
pub use execution_plan::{ExecutionLayer, ExecutionPlan, ExecutionPlanner, ResourceRequirements};
pub use fusion::{fuse_single_qubit_gates, FusedGate};
pub use hardware_aware::{
    CostModel, GoogleHardware, HardwareModel, HardwareType, IBMHardware, IonQHardware,
};
pub use lazy::{LazyConfig, LazyExecutor, LazyGate};
pub use matrix_computation::{
    controlled_gate_2x2, controlled_gate_4x4, decompose_zyz, doubly_controlled_gate_2x2,
    gate_fidelity_2x2, tensor_product_2x2, tensor_product_4x4, DynamicMatrix, Matrix2, Matrix4,
    Matrix8,
};
pub use passes::{OptimizationPass, OptimizationResult, PassStatistics};
pub use pipeline::{create_compiler, OptimizationLevel, PipelineBuilder};
