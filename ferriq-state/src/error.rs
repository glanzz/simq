//! Error types for state vector operations

use thiserror::Error;

/// Errors that can occur during state vector operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum StateError {
    /// Invalid qubit index
    #[error("Invalid qubit index {index} for {num_qubits}-qubit state")]
    InvalidQubitIndex { index: usize, num_qubits: usize },

    /// Invalid state dimension
    #[error("Invalid state dimension {dimension}, expected power of 2")]
    InvalidDimension { dimension: usize },

    /// State not normalized
    #[error("State vector not normalized, norm = {norm}")]
    NotNormalized { norm: f64 },

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Memory allocation error
    #[error("Failed to allocate {size} bytes for state vector")]
    AllocationError { size: usize },
}

/// Result type for state vector operations
pub type Result<T> = std::result::Result<T, StateError>;
