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
//! - **Compile-time caching**: Pre-computed rotation matrices for common angles embedded in binary
//! - **Type-safe gate interface**: All gates implement the `Gate` trait from `simq-core`
//! - **Parameterized gates**: Support for rotation gates (RX, RY, RZ) and phase gates
//! - **Standard gate library**: Comprehensive set of commonly used quantum gates
//! - **Lookup tables**: High-performance lookup tables for small-angle rotations
//! - **Optimized implementations**: Drop-in optimized versions of rotation gates
//! - **Multi-level caching**: Universal cache with fallback strategies for maximum performance
//!
//! # Examples
//!
//! ## Basic Gate Usage
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
//!
//! ## Optimized Small-Angle Rotations
//!
//! For circuits with many small-angle rotations (common in VQE and QAOA),
//! use lookup tables for significant performance improvements:
//!
//! ```
//! use simq_gates::optimized::{create_global_lookup_table, OptimizedRotationX};
//!
//! // Create a shared lookup table (do once at program start)
//! let table = create_global_lookup_table();
//!
//! // Use optimized rotation gates
//! let rx = OptimizedRotationX::new(0.01, &table);
//! let matrix = rx.compute_matrix(); // Fast lookup
//! ```
//!
//! See the `lookup` and `optimized` modules for more details.

pub mod compile_time_cache;
pub mod custom;
pub mod gate_registry;
pub mod generated;
pub mod lookup;
pub mod matrices;
pub mod matrix_ops;
pub mod optimized;
pub mod standard;

// Re-export commonly used items
pub use standard::*;

// Re-export compile-time cache
pub use compile_time_cache::{CommonAngles, UniversalCache, VQEAngles};

// Re-export enhanced cache with generated matrices
pub use generated::{EnhancedUniversalCache, GeneratedAngleCache};

// Re-export matrix operations
pub use matrix_ops::{
    circuit_matrix, circuit_matrix_with_provider, embed_gate_matrix, embed_gate_matrix_vec,
    is_hermitian, is_unitary, matrix_adjoint, matrix_multiply, matrix_to_vec, matrix_trace,
    tensor_product, vec_to_matrix_2x2,
};

// Re-export custom gates
pub use custom::{
    CustomGate, CustomGateBuilder, CustomGateError, ParametricCustomGate,
    ParametricCustomGateBuilder,
};
pub use gate_registry::GateRegistry;
