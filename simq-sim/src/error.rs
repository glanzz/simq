//! Error types for the simulator

use std::fmt;

/// Result type for simulator operations
pub type Result<T> = std::result::Result<T, SimulatorError>;

/// Errors that can occur during simulation
#[derive(Debug, Clone)]
pub enum SimulatorError {
    /// Invalid configuration
    InvalidConfig(String),

    /// Circuit is invalid or malformed
    InvalidCircuit(String),

    /// Memory limit exceeded
    OutOfMemory {
        requested: usize,
        limit: usize,
    },

    /// Too many qubits for available memory
    TooManyQubits {
        num_qubits: usize,
        max_qubits: usize,
    },

    /// Compilation failed
    CompilationFailed(String),

    /// State initialization failed
    StateInitializationFailed(String),

    /// Gate application failed
    GateApplicationFailed {
        gate_index: usize,
        reason: String,
    },

    /// Measurement failed
    MeasurementFailed(String),

    /// Invalid qubit index
    InvalidQubit {
        qubit: usize,
        num_qubits: usize,
    },

    /// Execution engine failure
    ExecutionFailed {
        message: String,
    },

    /// State error from state operations
    StateError {
        message: String,
    },

    /// Operation timeout
    Timeout {
        message: String,
    },

    /// Other error
    Other(String),
}

impl fmt::Display for SimulatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimulatorError::InvalidConfig(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            SimulatorError::InvalidCircuit(msg) => {
                write!(f, "Invalid circuit: {}", msg)
            }
            SimulatorError::OutOfMemory { requested, limit } => {
                write!(
                    f,
                    "Out of memory: requested {} bytes, limit {} bytes",
                    requested, limit
                )
            }
            SimulatorError::TooManyQubits {
                num_qubits,
                max_qubits,
            } => {
                write!(
                    f,
                    "Too many qubits: circuit has {}, max supported is {}",
                    num_qubits, max_qubits
                )
            }
            SimulatorError::CompilationFailed(msg) => {
                write!(f, "Compilation failed: {}", msg)
            }
            SimulatorError::StateInitializationFailed(msg) => {
                write!(f, "State initialization failed: {}", msg)
            }
            SimulatorError::GateApplicationFailed { gate_index, reason } => {
                write!(f, "Gate {} application failed: {}", gate_index, reason)
            }
            SimulatorError::MeasurementFailed(msg) => {
                write!(f, "Measurement failed: {}", msg)
            }
            SimulatorError::InvalidQubit {
                qubit,
                num_qubits,
            } => {
                write!(
                    f,
                    "Invalid qubit index {}: circuit has {} qubits",
                    qubit, num_qubits
                )
            }
            SimulatorError::ExecutionFailed { message } => {
                write!(f, "Execution failed: {}", message)
            }
            SimulatorError::StateError { message } => {
                write!(f, "State error: {}", message)
            }
            SimulatorError::Timeout { message } => {
                write!(f, "Operation timeout: {}", message)
            }
            SimulatorError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for SimulatorError {}

impl From<simq_core::QuantumError> for SimulatorError {
    fn from(err: simq_core::QuantumError) -> Self {
        SimulatorError::InvalidCircuit(err.to_string())
    }
}

impl From<simq_state::error::StateError> for SimulatorError {
    fn from(err: simq_state::error::StateError) -> Self {
        SimulatorError::StateInitializationFailed(err.to_string())
    }
}
