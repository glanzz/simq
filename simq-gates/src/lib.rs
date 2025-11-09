//! Quantum gate library for SimQ
//!
//! This crate provides standard quantum gate implementations with pre-computed
//! matrices for optimal performance. All common gate matrices are computed at
//! compile time using const evaluation.
//!
//! # Features
//!
//! - **Compile-time matrix computation**: Common gates (H, X, Y, Z, CNOT, etc.)
//!   have their matrices computed at compile time for zero runtime overhead
//! - **Type-safe gate interface**: All gates implement the `Gate` trait from `simq-core`
//! - **Parameterized gates**: Support for rotation gates (RX, RY, RZ) and phase gates
//! - **Standard gate library**: Comprehensive set of commonly used quantum gates
//!
//! # Examples
//!
//! ```
//! use simq_gates::standard::{Hadamard, CNot, RotationX};
//! use simq_gates::matrices;
//! use std::f64::consts::PI;
//!
//! // Access pre-computed gate matrices
//! let h_matrix = Hadamard::matrix();
//! let cnot_matrix = CNot::matrix();
//!
//! // Create parameterized gates
//! let rx_gate = RotationX::new(PI / 2.0);
//! let rx_matrix = rx_gate.matrix();
//!
//! // Direct access to raw matrices
//! let pauli_x = &matrices::PAULI_X;
//! let swap = &matrices::SWAP;
//! ```

pub mod matrices;
pub mod standard;

// Re-export commonly used items
pub use standard::*;
