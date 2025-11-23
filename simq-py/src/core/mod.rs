//! Core type bindings for SimQ
//!
//! This module contains Python bindings for fundamental types:
//! - Circuit and CircuitBuilder
//! - QubitId
//! - Parameter and ParameterRegistry
//! - Error types

use pyo3::prelude::*;

// Submodules (to be implemented in Phase 1)
// pub mod circuit;
// pub mod error;
// pub mod parameter;
// pub mod qubit;

/// Register core types with the Python module
pub fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Will register types in Phase 1:
    // m.add_class::<circuit::PyCircuit>()?;
    // m.add_class::<circuit::PyCircuitBuilder>()?;
    // error::register_exceptions(m)?;
    Ok(())
}
