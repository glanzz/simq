//! Error types for backend operations

use thiserror::Error;

/// Result type for backend operations
pub type Result<T> = std::result::Result<T, BackendError>;

/// Errors that can occur during backend operations
#[derive(Error, Debug)]
pub enum BackendError {
    /// Circuit is not compatible with this backend
    #[error("Circuit incompatible with backend: {0}")]
    CircuitIncompatible(String),

    /// Backend capabilities exceeded
    #[error("Backend capability exceeded: {0}")]
    CapabilityExceeded(String),

    /// Backend communication error
    #[error("Backend communication error: {0}")]
    CommunicationError(String),

    /// Authentication failed
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    /// Job submission failed
    #[error("Job submission failed: {0}")]
    JobSubmissionFailed(String),

    /// Job execution failed
    #[error("Job execution failed: {0}")]
    JobExecutionFailed(String),

    /// Job not found
    #[error("Job not found: {job_id}")]
    JobNotFound { job_id: String },

    /// Job timeout
    #[error("Job timeout after {timeout_seconds}s")]
    JobTimeout { timeout_seconds: u64 },

    /// Transpilation failed
    #[error("Transpilation failed: {0}")]
    TranspilationFailed(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Backend not available
    #[error("Backend not available: {0}")]
    BackendUnavailable(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    /// Insufficient credits/quota
    #[error("Insufficient credits or quota: {0}")]
    InsufficientQuota(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Deserialization error
    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Other error
    #[error("{0}")]
    Other(String),
}

impl From<serde_json::Error> for BackendError {
    fn from(err: serde_json::Error) -> Self {
        BackendError::SerializationError(err.to_string())
    }
}

impl From<simq_core::QuantumError> for BackendError {
    fn from(err: simq_core::QuantumError) -> Self {
        BackendError::Other(format!("Core error: {}", err))
    }
}
