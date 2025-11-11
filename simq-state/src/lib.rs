//! Quantum state representations with SIMD-optimized operations
//!
//! This crate provides high-performance quantum state vector implementations
//! with SIMD-optimized matrix-vector multiplication for efficient simulation.

pub mod state_vector;
pub mod simd;
pub mod error;

pub use state_vector::StateVector;
pub use error::{StateError, Result};
