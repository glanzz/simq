//! Circuit optimization and compilation for SimQ
//!
//! This crate provides optimization passes for quantum circuits, including:
//! - Gate fusion: Combining adjacent single-qubit gates
//! - Circuit simplification
//! - Gate decomposition

pub mod fusion;
pub mod matrix_utils;

pub use fusion::{fuse_single_qubit_gates, FusedGate};
