//! Optimized gate application kernels
//!
//! This module provides highly optimized implementations for applying
//! various types of quantum gates to state vectors.

pub mod controlled;
pub mod diagonal;
pub mod matrix;
pub mod single_qubit;
pub mod sparse;
pub mod two_qubit;

pub use controlled::{apply_controlled_gate, apply_multi_controlled};
pub use diagonal::{apply_diagonal_gate, apply_phase_gate};
pub use matrix::GateMatrix;
pub use single_qubit::{apply_single_qubit_dense, apply_single_qubit_dense_simd};
pub use sparse::{apply_single_qubit_sparse, apply_two_qubit_sparse};
pub use two_qubit::{apply_cnot, apply_cz, apply_swap, apply_two_qubit_dense};

use num_complex::Complex64;

/// Standard gate matrix type
pub type Matrix2x2 = [[Complex64; 2]; 2];
pub type Matrix4x4 = [[Complex64; 4]; 4];
