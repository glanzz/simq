//! Circuit optimization and compilation for SimQ
//!
//! This crate provides optimization passes for quantum circuits, including:
//! - Gate fusion: Combining adjacent single-qubit gates
//! - Lazy evaluation: Deferring gate matrix computation until needed
//! - Circuit simplification
//! - Gate decomposition

pub mod fusion;
pub mod lazy;
pub mod matrix_utils;

pub use fusion::{fuse_single_qubit_gates, FusedGate};
pub use lazy::{LazyConfig, LazyExecutor, LazyGate};
