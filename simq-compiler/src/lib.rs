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

pub mod fusion;
pub mod lazy;
pub mod matrix_utils;
pub mod matrix_computation;
pub mod decomposition;
pub mod analysis;
pub mod passes;
pub mod compiler;

pub use fusion::{fuse_single_qubit_gates, FusedGate};
pub use lazy::{LazyConfig, LazyExecutor, LazyGate};
pub use decomposition::{
    DecompositionConfig,
    DecompositionResult,
    UniversalDecomposer,
    BasisGateSet,
    BasisGate,
    Decomposer,
};
pub use matrix_computation::{
    Matrix2,
    Matrix4,
    Matrix8,
    DynamicMatrix,
    tensor_product_2x2,
    tensor_product_4x4,
    controlled_gate_2x2,
    controlled_gate_4x4,
    doubly_controlled_gate_2x2,
    decompose_zyz,
    gate_fidelity_2x2,
};
pub use analysis::{
    GateStatistics,
    ResourceEstimate,
    CircuitAnalysis,
};
pub use passes::{
    OptimizationPass,
    PassStatistics,
    OptimizationResult,
};
pub use compiler::{
    Compiler,
    CompilerConfig,
    CompilerBuilder,
};
