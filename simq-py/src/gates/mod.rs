//! Gate bindings for SimQ
//!
//! This module contains Python bindings for quantum gates:
//! - Standard gates (H, X, Y, Z, etc.)
//! - Parameterized gates (RX, RY, RZ, etc.)
//! - Custom gates
//! 
//! Submodules:
//! - standard: Standard single and multi-qubit gates
//! - parameterized: Gates with tunable parameters
//! - custom: User-defined gates

use pyo3::prelude::*;

pub mod custom;
pub mod parameterized;
pub mod standard;

/// Register gate types with the Python module
pub fn register(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let gates_module = PyModule::new_bound(py, "gates")?;
    
    // Register submodules
    standard::register(py, &gates_module)?;
    parameterized::register(py, &gates_module)?;
    custom::register(py, &gates_module)?;
    
    m.add_submodule(&gates_module)?;
    
    Ok(())
}
