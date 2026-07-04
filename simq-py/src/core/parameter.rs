//! Parameter bindings for Python

use pyo3::prelude::*;
use simq_core::Parameter as RustParameter;

use crate::core::error::IntoPyErr;

/// Python wrapper for Parameter
///
/// Represents a parameterized value that can be bound to a concrete value later.
/// Useful for variational quantum algorithms and parameterized circuits.
///
/// # Examples
///
/// ```python
/// from simq import Parameter
///
/// # Create a symbolic parameter
/// theta = Parameter("theta")
///
/// # Bind it to a concrete value
/// theta_bound = theta.bind(3.14159)
/// print(theta_bound.value)  # 3.14159
/// ```
#[pyclass(name = "Parameter", module = "simq")]
#[derive(Clone)]
pub struct PyParameter {
    pub(crate) inner: RustParameter,
}

#[pymethods]
impl PyParameter {
    /// Create a new parameter with a value
    ///
    /// Args:
    ///     value: The parameter value
    ///     name: Optional name for the parameter (default: None)
    ///
    /// Returns:
    ///     A new Parameter
    #[new]
    #[pyo3(signature = (value, name=None))]
    fn new(value: f64, name: Option<String>) -> Self {
        Self {
            inner: match name {
                Some(n) => RustParameter::named(n, value),
                None => RustParameter::new(value),
            },
        }
    }

    /// Set the parameter value
    ///
    /// Args:
    ///     value: The new value
    ///
    /// Raises:
    ///     QuantumException: If parameter is frozen or value violates bounds
    fn set_value(&mut self, value: f64) -> PyResult<()> {
        self.inner.set_value(value).map_err(|e| e.into_pyerr())
    }

    /// Get the parameter's name
    #[getter]
    fn name(&self) -> Option<String> {
        self.inner.name().map(|s| s.to_string())
    }

    /// Get the parameter's value
    #[getter]
    fn value(&self) -> f64 {
        self.inner.value()
    }

    /// String representation
    fn __repr__(&self) -> String {
        let val = self.inner.value();
        if let Some(name) = self.inner.name() {
            format!("Parameter('{}', value={})", name, val)
        } else {
            format!("Parameter(value={})", val)
        }
    }

    fn __str__(&self) -> String {
        let val = self.inner.value();
        if let Some(name) = self.inner.name() {
            format!("{}={}", name, val)
        } else {
            val.to_string()
        }
    }
}

/// Convert from Rust Parameter to Python Parameter
impl From<RustParameter> for PyParameter {
    fn from(param: RustParameter) -> Self {
        Self { inner: param }
    }
}

/// Convert from Python Parameter to Rust Parameter
impl From<PyParameter> for RustParameter {
    fn from(py_param: PyParameter) -> Self {
        py_param.inner
    }
}

/// Convert from reference to Python Parameter to Rust Parameter
impl From<&PyParameter> for RustParameter {
    fn from(py_param: &PyParameter) -> Self {
        py_param.inner.clone()
    }
}
