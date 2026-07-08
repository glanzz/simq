//! Error types for execution engine

use ferriq_core::QubitId;
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
    StateError(#[from] ferriq_state::StateError),

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

#[cfg(test)]
mod tests {
    use super::*;
    use ferriq_core::QubitId;
    use std::time::Duration;

    #[test]
    fn test_is_recoverable() {
        let e = ExecutionError::GateApplicationFailed {
            gate: "H".to_string(),
            qubits: vec![QubitId::new(0)],
            reason: "test".to_string(),
        };
        assert!(e.is_recoverable());

        let e = ExecutionError::ParallelExecutionFailed {
            layer: 0,
            reason: "test".to_string(),
        };
        assert!(e.is_recoverable());

        let e = ExecutionError::GpuError {
            reason: "test".to_string(),
        };
        assert!(e.is_recoverable());

        // Non-recoverable
        let e = ExecutionError::InvalidGateMatrix {
            gate: "H".to_string(),
            reason: "bad".to_string(),
        };
        assert!(!e.is_recoverable());
    }

    #[test]
    fn test_severity_critical() {
        let e = ExecutionError::QubitOutOfBounds {
            qubit: QubitId::new(5),
            max: 2,
        };
        assert_eq!(e.severity(), ErrorSeverity::Critical);

        let e = ExecutionError::InvalidGateMatrix {
            gate: "H".to_string(),
            reason: "bad".to_string(),
        };
        assert_eq!(e.severity(), ErrorSeverity::Critical);
    }

    #[test]
    fn test_severity_high() {
        let e = ExecutionError::ResourceExhausted {
            resource: "memory".to_string(),
            limit: 100,
            requested: 200,
        };
        assert_eq!(e.severity(), ErrorSeverity::High);

        let e = ExecutionError::ExecutionTimeout {
            elapsed: Duration::from_secs(10),
            limit: Duration::from_secs(5),
        };
        assert_eq!(e.severity(), ErrorSeverity::High);
    }

    #[test]
    fn test_severity_medium() {
        let e = ExecutionError::GateApplicationFailed {
            gate: "H".to_string(),
            qubits: vec![QubitId::new(0)],
            reason: "test".to_string(),
        };
        assert_eq!(e.severity(), ErrorSeverity::Medium);

        let e = ExecutionError::ParallelExecutionFailed {
            layer: 0,
            reason: "test".to_string(),
        };
        assert_eq!(e.severity(), ErrorSeverity::Medium);

        // Fallback branch (e.g., ExecutionHalted)
        let e = ExecutionError::ExecutionHalted { attempts: 3 };
        assert_eq!(e.severity(), ErrorSeverity::Medium);
    }

    #[test]
    fn test_severity_low() {
        let e = ExecutionError::ValidationFailed {
            reason: "bad norm".to_string(),
        };
        assert_eq!(e.severity(), ErrorSeverity::Low);

        let e = ExecutionError::GpuError {
            reason: "no gpu".to_string(),
        };
        assert_eq!(e.severity(), ErrorSeverity::Low);
    }

    #[test]
    fn test_display_impl() {
        let e = ExecutionError::CheckpointFailed {
            reason: "io error".to_string(),
        };
        let s = format!("{}", e);
        assert!(s.contains("Checkpoint"));

        let e = ExecutionError::CircuitError("bad circuit".to_string());
        let s = format!("{}", e);
        assert!(s.contains("bad circuit"));

        let e = ExecutionError::Internal("internal".to_string());
        let s = format!("{}", e);
        assert!(s.contains("internal"));
    }
}
