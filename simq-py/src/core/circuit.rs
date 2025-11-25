//! Circuit and CircuitBuilder bindings for Python

use pyo3::prelude::*;
use simq_core::{Circuit as RustCircuit, DynamicCircuitBuilder};
use simq_gates::*;
use parking_lot::Mutex;
use std::sync::Arc;

use crate::core::error::IntoPyErr;

/// Python wrapper for Circuit (read-only)
///
/// Represents a quantum circuit with qubits and gates.
/// This is a read-only view of a circuit after it has been built.
///
/// # Examples
///
/// ```python
/// from simq import CircuitBuilder
///
/// builder = CircuitBuilder(3)
/// builder.h(0)
/// builder.cx(0, 1)
///
/// circuit = builder.build()
/// print(f"Qubits: {circuit.num_qubits}")
/// print(f"Gates: {circuit.gate_count}")
/// print(f"Depth: {circuit.depth}")
/// ```
#[pyclass(name = "Circuit", module = "simq")]
pub struct PyCircuit {
    pub(crate) inner: Arc<RustCircuit>,
}

#[pymethods]
impl PyCircuit {
    /// Number of qubits in the circuit
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Number of gates in the circuit
    #[getter]
    fn gate_count(&self) -> usize {
        self.inner.len()
    }

    /// Depth of the circuit (longest path)
    #[getter]
    fn depth(&self) -> usize {
        self.inner.depth()
    }

    /// String representation for debugging
    fn __repr__(&self) -> String {
        format!(
            "Circuit(qubits={}, gates={}, depth={})",
            self.num_qubits(),
            self.gate_count(),
            self.depth()
        )
    }

    /// ASCII diagram of the circuit
    fn __str__(&self) -> String {
        simq_core::render_ascii(&self.inner)
    }

    /// Get ASCII representation of the circuit
    ///
    /// Returns:
    ///     String containing ASCII circuit diagram
    fn to_ascii(&self) -> String {
        simq_core::render_ascii(&self.inner)
    }
}

impl PyCircuit {
    /// Access the inner Rust circuit (internal use)
    pub(crate) fn inner(&self) -> &RustCircuit {
        &self.inner
    }

    /// Create PyCircuit from Rust circuit (internal use)
    pub(crate) fn from_rust(circuit: Arc<RustCircuit>) -> Self {
        Self { inner: circuit }
    }
}

/// Python wrapper for CircuitBuilder
///
/// Builder for constructing quantum circuits with a dynamic number of qubits.
///
/// # Examples
///
/// ```python
/// from simq import CircuitBuilder
///
/// # Create a 3-qubit circuit
/// builder = CircuitBuilder(3)
///
/// # Add gates
/// builder.h(0)           # Hadamard on qubit 0
/// builder.x(1)           # Pauli-X on qubit 1
/// builder.cx(0, 1)       # CNOT with control=0, target=1
///
/// # Build the circuit
/// circuit = builder.build()
/// ```
#[pyclass(name = "CircuitBuilder", module = "simq")]
pub struct PyCircuitBuilder {
    inner: Arc<Mutex<DynamicCircuitBuilder>>,
}

#[pymethods]
impl PyCircuitBuilder {
    /// Create a new circuit builder
    ///
    /// Args:
    ///     num_qubits: Number of qubits in the circuit
    ///
    /// Returns:
    ///     A new CircuitBuilder instance
    #[new]
    fn new(num_qubits: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(DynamicCircuitBuilder::new(num_qubits))),
        }
    }

    /// Get the number of qubits
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.lock().num_qubits()
    }

    /// Apply Hadamard gate
    ///
    /// Args:
    ///     qubit: Qubit index to apply gate to
    ///
    /// Raises:
    ///     InvalidQubitError: If qubit index is out of range
    fn h(&mut self, qubit: usize) -> PyResult<()> {
        let mut builder = self.inner.lock();
        builder
            .apply_gate(Arc::new(Hadamard), &[qubit])
            .map(|_| ())
            .map(|_| ()).map(|_| ()).map_err(|e| e.into_pyerr())
    }

    /// Apply Pauli-X (NOT) gate
    ///
    /// Args:
    ///     qubit: Qubit index to apply gate to
    ///
    /// Raises:
    ///     InvalidQubitError: If qubit index is out of range
    fn x(&mut self, qubit: usize) -> PyResult<()> {
        let mut builder = self.inner.lock();
        builder
            .apply_gate(Arc::new(PauliX), &[qubit])
            .map(|_| ()).map(|_| ()).map_err(|e| e.into_pyerr())
    }

    /// Apply Pauli-Y gate
    ///
    /// Args:
    ///     qubit: Qubit index to apply gate to
    ///
    /// Raises:
    ///     InvalidQubitError: If qubit index is out of range
    fn y(&mut self, qubit: usize) -> PyResult<()> {
        let mut builder = self.inner.lock();
        builder
            .apply_gate(Arc::new(PauliY), &[qubit])
            .map(|_| ()).map(|_| ()).map_err(|e| e.into_pyerr())
    }

    /// Apply Pauli-Z gate
    ///
    /// Args:
    ///     qubit: Qubit index to apply gate to
    ///
    /// Raises:
    ///     InvalidQubitError: If qubit index is out of range
    fn z(&mut self, qubit: usize) -> PyResult<()> {
        let mut builder = self.inner.lock();
        builder
            .apply_gate(Arc::new(PauliZ), &[qubit])
            .map(|_| ()).map(|_| ()).map_err(|e| e.into_pyerr())
    }

    /// Apply S gate (Phase gate, √Z)
    ///
    /// Args:
    ///     qubit: Qubit index to apply gate to
    ///
    /// Raises:
    ///     InvalidQubitError: If qubit index is out of range
    fn s(&mut self, qubit: usize) -> PyResult<()> {
        let mut builder = self.inner.lock();
        builder
            .apply_gate(Arc::new(SGate), &[qubit])
            .map(|_| ()).map(|_| ()).map_err(|e| e.into_pyerr())
    }

    /// Apply T gate (π/8 gate)
    ///
    /// Args:
    ///     qubit: Qubit index to apply gate to
    ///
    /// Raises:
    ///     InvalidQubitError: If qubit index is out of range
    fn t(&mut self, qubit: usize) -> PyResult<()> {
        let mut builder = self.inner.lock();
        builder
            .apply_gate(Arc::new(TGate), &[qubit])
            .map(|_| ()).map(|_| ()).map_err(|e| e.into_pyerr())
    }

    /// Apply CNOT (Controlled-NOT) gate
    ///
    /// Args:
    ///     control: Control qubit index
    ///     target: Target qubit index
    ///
    /// Raises:
    ///     InvalidQubitError: If qubit indices are out of range or equal
    fn cx(&mut self, control: usize, target: usize) -> PyResult<()> {
        let mut builder = self.inner.lock();
        builder
            .apply_gate(Arc::new(CNot), &[control, target])
            .map(|_| ()).map(|_| ()).map_err(|e| e.into_pyerr())
    }

    /// Apply CZ (Controlled-Z) gate
    ///
    /// Args:
    ///     control: Control qubit index
    ///     target: Target qubit index
    ///
    /// Raises:
    ///     InvalidQubitError: If qubit indices are out of range or equal
    fn cz(&mut self, control: usize, target: usize) -> PyResult<()> {
        let mut builder = self.inner.lock();
        builder
            .apply_gate(Arc::new(CZ), &[control, target])
            .map(|_| ()).map(|_| ()).map_err(|e| e.into_pyerr())
    }

    /// Apply SWAP gate
    ///
    /// Args:
    ///     qubit1: First qubit index
    ///     qubit2: Second qubit index
    ///
    /// Raises:
    ///     InvalidQubitError: If qubit indices are out of range or equal
    fn swap(&mut self, qubit1: usize, qubit2: usize) -> PyResult<()> {
        let mut builder = self.inner.lock();
        builder
            .apply_gate(Arc::new(Swap), &[qubit1, qubit2])
            .map(|_| ()).map(|_| ()).map_err(|e| e.into_pyerr())
    }

    /// Apply rotation around X-axis
    ///
    /// Args:
    ///     qubit: Qubit index to apply gate to
    ///     theta: Rotation angle in radians
    ///
    /// Raises:
    ///     InvalidQubitError: If qubit index is out of range
    fn rx(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        let mut builder = self.inner.lock();
        builder
            .apply_gate(Arc::new(RotationX::new(theta)), &[qubit])
            .map(|_| ()).map(|_| ()).map_err(|e| e.into_pyerr())
    }

    /// Apply rotation around Y-axis
    ///
    /// Args:
    ///     qubit: Qubit index to apply gate to
    ///     theta: Rotation angle in radians
    ///
    /// Raises:
    ///     InvalidQubitError: If qubit index is out of range
    fn ry(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        let mut builder = self.inner.lock();
        builder
            .apply_gate(Arc::new(RotationY::new(theta)), &[qubit])
            .map(|_| ()).map(|_| ()).map_err(|e| e.into_pyerr())
    }

    /// Apply rotation around Z-axis
    ///
    /// Args:
    ///     qubit: Qubit index to apply gate to
    ///     theta: Rotation angle in radians
    ///
    /// Raises:
    ///     InvalidQubitError: If qubit index is out of range
    fn rz(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        let mut builder = self.inner.lock();
        builder
            .apply_gate(Arc::new(RotationZ::new(theta)), &[qubit])
            .map(|_| ()).map(|_| ()).map_err(|e| e.into_pyerr())
    }

    /// Build the final circuit
    ///
    /// Returns:
    ///     The constructed Circuit
    fn build(&self) -> PyCircuit {
        let builder = self.inner.lock();
        // Clone the inner circuit to return it
        // DynamicCircuitBuilder doesn't impl Clone, so we clone the circuit
        let circuit = builder.circuit().clone();
        PyCircuit {
            inner: Arc::new(circuit),
        }
    }

    fn __repr__(&self) -> String {
        format!("CircuitBuilder(num_qubits={})", self.num_qubits())
    }
}

/// Convert from Rust Circuit to Python Circuit
impl From<RustCircuit> for PyCircuit {
    fn from(circuit: RustCircuit) -> Self {
        Self {
            inner: Arc::new(circuit),
        }
    }
}
