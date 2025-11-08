//! Core types and traits for SimQ quantum computing SDK
//!
//! This crate provides the fundamental types for building quantum circuits:
//! - [`QubitId`]: Type-safe qubit addressing
//! - [`Gate`]: Trait for quantum operations
//! - [`Circuit`]: Quantum circuit container
//!
//! # Example
//! ```
//! use simq_core::{Circuit, QubitId};
//!
//! let mut circuit = Circuit::new(2);
//! let q0 = QubitId::new(0);
//! // Add gates...
//! ```

pub mod circuit;
pub mod error;
pub mod gate;
pub mod qubit;

// Re-exports for convenience
pub use circuit::Circuit;
pub use error::QuantumError;
pub use gate::{Gate, GateOp};
pub use num_complex::Complex64;
pub use qubit::QubitId;

/// Type alias for results in SimQ
pub type Result<T> = std::result::Result<T, QuantumError>;
