//! QubitId bindings for Python

use pyo3::prelude::*;
use simq_core::QubitId as RustQubitId;
use std::hash::{Hash, Hasher};

/// Python wrapper for QubitId
///
/// Represents a quantum qubit identifier with a unique index.
///
/// # Examples
///
/// ```python
/// from simq import QubitId
///
/// q0 = QubitId(0)
/// q1 = QubitId(1)
///
/// print(int(q0))  # 0
/// print(repr(q0))  # QubitId(0)
/// print(q0 == q1)  # False
/// ```
#[pyclass(name = "QubitId", module = "simq")]
#[derive(Clone, Copy, Debug)]
pub struct PyQubitId {
    pub(crate) inner: RustQubitId,
}

#[pymethods]
impl PyQubitId {
    /// Create a new QubitId
    ///
    /// Args:
    ///     index: The qubit index (must be non-negative)
    ///
    /// Returns:
    ///     A new QubitId instance
    #[new]
    fn new(index: usize) -> Self {
        Self {
            inner: RustQubitId::from(index),
        }
    }

    /// Get the index as an integer
    fn __int__(&self) -> usize {
        self.inner.into()
    }

    /// String representation for debugging
    fn __repr__(&self) -> String {
        format!("QubitId({})", usize::from(self.inner))
    }

    /// User-friendly string representation
    fn __str__(&self) -> String {
        format!("q{}", usize::from(self.inner))
    }

    /// Equality comparison
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Inequality comparison
    fn __ne__(&self, other: &Self) -> bool {
        self.inner != other.inner
    }

    /// Hash for use in sets and dicts
    fn __hash__(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    /// Get the qubit index
    ///
    /// Returns:
    ///     The zero-based index of this qubit
    #[getter]
    fn index(&self) -> usize {
        self.inner.into()
    }
}

/// Convert from Rust QubitId to Python QubitId
impl From<RustQubitId> for PyQubitId {
    fn from(qubit: RustQubitId) -> Self {
        Self { inner: qubit }
    }
}

/// Convert from Python QubitId to Rust QubitId
impl From<PyQubitId> for RustQubitId {
    fn from(py_qubit: PyQubitId) -> Self {
        py_qubit.inner
    }
}

/// Convert from reference to Python QubitId to Rust QubitId
impl From<&PyQubitId> for RustQubitId {
    fn from(py_qubit: &PyQubitId) -> Self {
        py_qubit.inner
    }
}
