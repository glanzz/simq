//! Quantum state representations with SIMD-optimized operations
//!
//! This crate provides high-performance quantum state vector implementations
//! with both dense (StateVector) and sparse (SparseState) representations,
//! SIMD-optimized matrix-vector multiplication, and automatic Sparse↔Dense conversion.
//!
//! # State Representations
//!
//! - **Dense**: Full 2^n amplitude vector with SIMD alignment (use for highly entangled states)
//! - **Sparse**: AHashMap-based storage for efficient representation of sparse states
//!
//! # Automatic Conversion
//!
//! Sparse states automatically recommend conversion to dense when the density
//! exceeds a configurable threshold (default 10%).
//!
//! # Example
//!
//! ```
//! use simq_state::{SparseState, StateVector};
//! use num_complex::Complex64;
//!
//! // Start with sparse representation
//! let sparse = SparseState::new(10).unwrap();
//! println!("Initial density: {:.2}%", sparse.density() * 100.0);
//!
//! // Can convert to dense when needed
//! let dense_amplitudes = sparse.to_dense();
//! ```

pub mod state_vector;
pub mod sparse_state;
pub mod dense_state;
pub mod adaptive_state;
pub mod cow_state;
pub mod validation;
pub mod simd;
pub mod error;
pub mod measurement;
pub mod observable;

pub use state_vector::StateVector;
pub use sparse_state::SparseState;
pub use dense_state::DenseState;
pub use adaptive_state::{AdaptiveState, StateStats};
pub use cow_state::{CowState, CowStats, MemoryStats};
pub use error::{StateError, Result};
pub use measurement::{Measurement, MeasurementResult, SamplingResult, ComputationalBasis};
pub use observable::{Pauli, PauliString, PauliObservable};
