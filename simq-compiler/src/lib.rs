//! Circuit optimization and compilation for SimQ
//!
//! This crate provides optimization passes for quantum circuits, including:
//! - **Gate fusion**: Combining adjacent single-qubit gates
//! - **Lazy evaluation**: Deferring gate matrix computation until needed
//! - **Circuit simplification**: Reducing gate count and depth
//! - **Gate decomposition**: Translating gates to different basis sets
//! - **Matrix computation**: Advanced matrix operations for quantum gates
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

pub mod fusion;
pub mod lazy;
pub mod matrix_utils;
pub mod matrix_computation;
pub mod decomposition;

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
