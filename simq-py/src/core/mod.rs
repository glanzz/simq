//! Core type bindings for SimQ
//!
//! This module contains Python bindings for fundamental types:
//! - Circuit and CircuitBuilder
//! - QubitId
//! - Parameter
//! - Error types

use pyo3::prelude::*;

// Submodules
pub mod circuit;
pub mod error;
pub mod parameter;
pub mod qubit;

// Re-exports for convenience
pub use circuit::{PyCircuit, PyCircuitBuilder};
pub use parameter::PyParameter;
pub use qubit::PyQubitId;

/// Register core types with the Python module
pub fn register(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register exception types
    error::register_exceptions(py, m)?;

    // Register core classes
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyCircuitBuilder>()?;
    m.add_class::<PyQubitId>()?;
    m.add_class::<PyParameter>()?;

    Ok(())
}
