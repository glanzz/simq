//! Error types for execution engine

use simq_core::QubitId;
use thiserror::Error;

/// Result type for execution engine operations
pub type Result<T> = std::result::Result<T, ExecutionError>;

/// Errors that can occur during circuit execution
#[derive(Error, Debug, Clone)]
pub enum ExecutionError {
    /// Gate application failed
    #[error("Gate application failed: {gate} on qubits {qubits:?}: {reason}")]
    GateApplicationFailed {
        gate: String,
        qubits: Vec<QubitId>,
        reason: String,
    },

    /// Invalid gate matrix
    #[error("Invalid gate matrix for {gate}: {reason}")]
    InvalidGateMatrix { gate: String, reason: String },

    /// Qubit index out of bounds
    #[error("Qubit index {qubit:?} out of bounds (max: {max})")]
    QubitOutOfBounds { qubit: QubitId, max: usize },

    /// State validation failed
    #[error("State validation failed: {reason}")]
    ValidationFailed { reason: String },

    /// Checkpoint error
    #[error("Checkpoint operation failed: {reason}")]
    CheckpointFailed { reason: String },

    /// GPU execution error
    #[error("GPU execution failed: {reason}")]
    GpuError { reason: String },

    /// Parallel execution error
    #[error("Parallel execution failed on layer {layer}: {reason}")]
    ParallelExecutionFailed { layer: usize, reason: String },

    /// Resource exhaustion
    #[error("Resource exhausted: {resource} (limit: {limit}, requested: {requested})")]
    ResourceExhausted {
        resource: String,
        limit: usize,
        requested: usize,
    },

    /// State error
    #[error("State error: {0}")]
    StateError(#[from] simq_state::StateError),

    /// Execution halted by recovery policy
    #[error("Execution halted after {attempts} failed attempts")]
    ExecutionHalted { attempts: usize },

    /// Timeout error
    #[error("Execution timeout after {elapsed:?} (limit: {limit:?})")]
    ExecutionTimeout {
        elapsed: std::time::Duration,
        limit: std::time::Duration,
    },

    /// Circuit error
    #[error("Circuit error: {0}")]
    CircuitError(String),

    /// Internal error
    #[error("Internal execution engine error: {0}")]
    Internal(String),
}

impl ExecutionError {
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            ExecutionError::GateApplicationFailed { .. }
                | ExecutionError::ParallelExecutionFailed { .. }
                | ExecutionError::GpuError { .. }
        )
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            ExecutionError::QubitOutOfBounds { .. } | ExecutionError::InvalidGateMatrix { .. } => {
                ErrorSeverity::Critical
            },
            ExecutionError::ResourceExhausted { .. } | ExecutionError::ExecutionTimeout { .. } => {
                ErrorSeverity::High
            },
            ExecutionError::GateApplicationFailed { .. }
            | ExecutionError::ParallelExecutionFailed { .. } => ErrorSeverity::Medium,
            ExecutionError::ValidationFailed { .. } | ExecutionError::GpuError { .. } => {
                ErrorSeverity::Low
            },
            _ => ErrorSeverity::Medium,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}
