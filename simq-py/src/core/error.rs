//! Error handling for Python bindings
//!
//! Converts Rust QuantumError types to Python exceptions

use pyo3::prelude::*;
use pyo3::exceptions::PyException;
use simq_core::QuantumError;

// Create Python exception types
pyo3::create_exception!(simq, QuantumException, PyException, "Base exception for SimQ quantum operations");
pyo3::create_exception!(simq, InvalidQubitError, QuantumException, "Invalid qubit index or operation");
pyo3::create_exception!(simq, InvalidGateError, QuantumException, "Invalid gate or gate operation");
pyo3::create_exception!(simq, InvalidParameterError, QuantumException, "Invalid parameter value or operation");
pyo3::create_exception!(simq, CircuitError, QuantumException, "Circuit construction or validation error");

/// Helper trait to convert QuantumError to PyErr
pub trait IntoPyErr {
    fn into_pyerr(self) -> PyErr;
}

impl IntoPyErr for QuantumError {
    fn into_pyerr(self) -> PyErr {
        let error_msg = self.to_string();

        // Match on error type and create appropriate Python exception
        match self {
            QuantumError::InvalidQubit(..) | QuantumError::DuplicateQubit(_) => {
                InvalidQubitError::new_err(error_msg)
            }
            QuantumError::InvalidQubitCount { .. } | QuantumError::UnknownGateType(_) => {
                InvalidGateError::new_err(error_msg)
            }
            QuantumError::ValidationError(_) => {
                InvalidParameterError::new_err(error_msg)
            }
            QuantumError::EmptyCircuit
            | QuantumError::CycleDetected { .. }
            | QuantumError::InvalidDependency { .. }
            | QuantumError::TopologicalOrderError { .. } => {
                CircuitError::new_err(error_msg)
            }
            _ => {
                // Generic quantum exception for other error types
                QuantumException::new_err(error_msg)
            }
        }
    }
}

/// Register exception types with the Python module
pub fn register_exceptions(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("QuantumException", py.get_type_bound::<QuantumException>())?;
    m.add("InvalidQubitError", py.get_type_bound::<InvalidQubitError>())?;
    m.add("InvalidGateError", py.get_type_bound::<InvalidGateError>())?;
    m.add("InvalidParameterError", py.get_type_bound::<InvalidParameterError>())?;
    m.add("CircuitError", py.get_type_bound::<CircuitError>())?;
    Ok(())
}
