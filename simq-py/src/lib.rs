//! Python bindings for SimQ quantum computing SDK
//!
//! This crate provides Python bindings for the SimQ quantum computing framework,
//! enabling high-performance quantum circuit simulation from Python.

use pyo3::prelude::*;

// Module declarations
pub mod backend;
pub mod compiler;
pub mod core;
pub mod gates;
pub mod noise;
pub mod simulation;

/// SimQ: High-Performance Quantum Computing SDK
///
/// SimQ is a Rust-based quantum computing framework with Python bindings,
/// designed for high-performance quantum circuit simulation and execution.
///
/// # Quick Start
///
/// ```python
/// import simq
///
/// # Create a 3-qubit circuit
/// builder = simq.CircuitBuilder(3)
/// builder.h(0)
/// builder.cx(0, 1)
/// builder.cx(1, 2)
///
/// # Build the circuit
/// circuit = builder.build()
/// print(f"Circuit has {circuit.num_qubits} qubits and {circuit.gate_count} gates")
///
/// # Simulate
/// simulator = simq.Simulator()
/// result = simulator.run(circuit)
/// print(f"State vector: {result.state_vector}")
/// ```
///
/// # Features
///
/// - **High Performance**: Rust-powered quantum simulation
/// - **Easy to Use**: Pythonic API for circuit building
/// - **Flexible**: Support for parameterized circuits
/// - **Realistic**: Noise models for accurate simulation
/// - **Extensible**: Custom gates and backends
#[pymodule]
fn _simq(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "SimQ Contributors")?;
    m.add(
        "__doc__",
        "High-performance quantum computing SDK with Python bindings",
    )?;

    // Register core types
    core::register(_py, m)?;

    // Register gate types
    gates::register(_py, m)?;
    noise::register(_py, m)?;
    simulation::register(_py, m)?;

    // Phase 5: Register advanced features
    backend::register(_py, m)?;
    compiler::register(_py, m)?;

    Ok(())
}
