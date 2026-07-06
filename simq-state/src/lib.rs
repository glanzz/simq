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

pub mod adaptive_state;
pub mod cow_state;
pub mod dense_state;
pub mod density_matrix;
pub mod density_matrix_simulator;
pub mod error;
pub mod measurement;
pub mod monte_carlo_simulator;
pub mod observable;
pub mod simd;
pub mod sparse_state;
pub mod state_vector;
pub mod validation;

pub use adaptive_state::{AdaptiveState, StateStats};
pub use cow_state::{CowState, CowStats, MemoryStats};
pub use dense_state::DenseState;
pub use density_matrix::DensityMatrix;
pub use density_matrix_simulator::{DensityMatrixConfig, DensityMatrixSimulator, SimulationStats};
pub use error::{Result, StateError};
pub use measurement::{
    ComputationalBasis, Measurement, MeasurementResult, MidCircuitMeasurement, SamplingResult,
};
pub use monte_carlo_simulator::{MonteCarloConfig, MonteCarloSimulator, MonteCarloStats};
pub use observable::{Pauli, PauliObservable, PauliString};
pub use sparse_state::SparseState;
pub use state_vector::StateVector;
