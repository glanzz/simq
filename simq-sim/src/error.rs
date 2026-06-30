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
    OutOfMemory { requested: usize, limit: usize },

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
    GateApplicationFailed { gate_index: usize, reason: String },

    /// Measurement failed
    MeasurementFailed(String),

    /// Invalid qubit index
    InvalidQubit { qubit: usize, num_qubits: usize },

    /// Other error
    Other(String),
}

impl fmt::Display for SimulatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimulatorError::InvalidConfig(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            },
            SimulatorError::InvalidCircuit(msg) => {
                write!(f, "Invalid circuit: {}", msg)
            },
            SimulatorError::OutOfMemory { requested, limit } => {
                write!(f, "Out of memory: requested {} bytes, limit {} bytes", requested, limit)
            },
            SimulatorError::TooManyQubits {
                num_qubits,
                max_qubits,
            } => {
                write!(
                    f,
                    "Too many qubits: circuit has {}, max supported is {}",
                    num_qubits, max_qubits
                )
            },
            SimulatorError::CompilationFailed(msg) => {
                write!(f, "Compilation failed: {}", msg)
            },
            SimulatorError::StateInitializationFailed(msg) => {
                write!(f, "State initialization failed: {}", msg)
            },
            SimulatorError::GateApplicationFailed { gate_index, reason } => {
                write!(f, "Gate {} application failed: {}", gate_index, reason)
            },
            SimulatorError::MeasurementFailed(msg) => {
                write!(f, "Measurement failed: {}", msg)
            },
            SimulatorError::InvalidQubit { qubit, num_qubits } => {
                write!(f, "Invalid qubit index {}: circuit has {} qubits", qubit, num_qubits)
            },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_config_display() {
        let e = SimulatorError::InvalidConfig("bad param".to_string());
        let msg = e.to_string();
        assert!(msg.contains("Invalid configuration"));
        assert!(msg.contains("bad param"));
    }

    #[test]
    fn test_invalid_circuit_display() {
        let e = SimulatorError::InvalidCircuit("missing qubits".to_string());
        let msg = e.to_string();
        assert!(msg.contains("Invalid circuit"));
        assert!(msg.contains("missing qubits"));
    }

    #[test]
    fn test_out_of_memory_display() {
        let e = SimulatorError::OutOfMemory { requested: 1024, limit: 512 };
        let msg = e.to_string();
        assert!(msg.contains("Out of memory"));
        assert!(msg.contains("1024"));
        assert!(msg.contains("512"));
    }

    #[test]
    fn test_too_many_qubits_display() {
        let e = SimulatorError::TooManyQubits { num_qubits: 100, max_qubits: 30 };
        let msg = e.to_string();
        assert!(msg.contains("Too many qubits"));
        assert!(msg.contains("100"));
        assert!(msg.contains("30"));
    }

    #[test]
    fn test_compilation_failed_display() {
        let e = SimulatorError::CompilationFailed("linker error".to_string());
        let msg = e.to_string();
        assert!(msg.contains("Compilation failed"));
        assert!(msg.contains("linker error"));
    }

    #[test]
    fn test_state_initialization_failed_display() {
        let e = SimulatorError::StateInitializationFailed("alloc failed".to_string());
        let msg = e.to_string();
        assert!(msg.contains("State initialization failed"));
        assert!(msg.contains("alloc failed"));
    }

    #[test]
    fn test_gate_application_failed_display() {
        let e = SimulatorError::GateApplicationFailed {
            gate_index: 3,
            reason: "unitary check failed".to_string(),
        };
        let msg = e.to_string();
        assert!(msg.contains("Gate 3"));
        assert!(msg.contains("unitary check failed"));
    }

    #[test]
    fn test_measurement_failed_display() {
        let e = SimulatorError::MeasurementFailed("no state".to_string());
        let msg = e.to_string();
        assert!(msg.contains("Measurement failed"));
        assert!(msg.contains("no state"));
    }

    #[test]
    fn test_invalid_qubit_display() {
        let e = SimulatorError::InvalidQubit { qubit: 5, num_qubits: 3 };
        let msg = e.to_string();
        assert!(msg.contains("Invalid qubit index 5"));
        assert!(msg.contains("3 qubits"));
    }

    #[test]
    fn test_other_display() {
        let e = SimulatorError::Other("generic error".to_string());
        let msg = e.to_string();
        assert!(msg.contains("generic error"));
    }

    #[test]
    fn test_error_is_std_error() {
        let e = SimulatorError::Other("test".to_string());
        // Can be used as a std::error::Error
        let _: &dyn std::error::Error = &e;
    }

    #[test]
    fn test_error_clone() {
        let e = SimulatorError::InvalidConfig("x".to_string());
        let e2 = e.clone();
        assert!(e2.to_string().contains("x"));
    }

    #[test]
    fn test_error_debug() {
        let e = SimulatorError::TooManyQubits { num_qubits: 10, max_qubits: 5 };
        let dbg = format!("{:?}", e);
        assert!(dbg.contains("TooManyQubits"));
    }
}
