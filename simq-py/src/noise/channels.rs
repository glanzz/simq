use pyo3::prelude::*;
use simq_core::noise::channels::{
    DepolarizingChannel as RustDepolarizing,
    AmplitudeDamping as RustAmplitudeDamping,
    PhaseDamping as RustPhaseDamping,
    ReadoutError as RustReadoutError,
};
use simq_core::noise::types::NoiseChannel;

#[pyclass(name = "DepolarizingChannel")]
#[derive(Clone)]
pub struct PyDepolarizingChannel {
    pub inner: RustDepolarizing,
}

#[pymethods]
impl PyDepolarizingChannel {
    #[new]
    fn new(error_probability: f64) -> PyResult<Self> {
        let inner = RustDepolarizing::new(error_probability)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn error_probability(&self) -> f64 {
        self.inner.error_probability()
    }

    fn __repr__(&self) -> String {
        format!("DepolarizingChannel(p={})", self.inner.error_probability())
    }
}

#[pyclass(name = "AmplitudeDamping")]
#[derive(Clone)]
pub struct PyAmplitudeDamping {
    pub inner: RustAmplitudeDamping,
}

#[pymethods]
impl PyAmplitudeDamping {
    #[new]
    fn new(gamma: f64) -> PyResult<Self> {
        let inner = RustAmplitudeDamping::new(gamma)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_t1(t1: f64, gate_time: f64) -> PyResult<Self> {
        let inner = RustAmplitudeDamping::from_t1(t1, gate_time)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    fn __repr__(&self) -> String {
        format!("AmplitudeDamping(gamma={})", self.inner.gamma())
    }
}

#[pyclass(name = "PhaseDamping")]
#[derive(Clone)]
pub struct PyPhaseDamping {
    pub inner: RustPhaseDamping,
}

#[pymethods]
impl PyPhaseDamping {
    #[new]
    fn new(lambda: f64) -> PyResult<Self> {
        let inner = RustPhaseDamping::new(lambda)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_t2(t2: f64, gate_time: f64) -> PyResult<Self> {
        let inner = RustPhaseDamping::from_t2(t2, gate_time)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    #[pyo3(name = "lambda_value")]
    fn lambda(&self) -> f64 {
        self.inner.lambda()
    }

    fn __repr__(&self) -> String {
        format!("PhaseDamping(lambda={})", self.inner.lambda())
    }
}

#[pyclass(name = "ReadoutError")]
#[derive(Clone)]
pub struct PyReadoutError {
    pub inner: RustReadoutError,
}

#[pymethods]
impl PyReadoutError {
    #[new]
    fn new(p01: f64, p10: f64) -> PyResult<Self> {
        let inner = RustReadoutError::new(p01, p10)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn symmetric(error_rate: f64) -> PyResult<Self> {
        let inner = RustReadoutError::symmetric(error_rate)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn p01(&self) -> f64 {
        self.inner.p01()
    }

    #[getter]
    fn p10(&self) -> f64 {
        self.inner.p10()
    }

    #[getter]
    fn average_error(&self) -> f64 {
        self.inner.average_error()
    }

    fn __repr__(&self) -> String {
        format!("ReadoutError(p01={}, p10={})", self.inner.p01(), self.inner.p10())
    }
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDepolarizingChannel>()?;
    m.add_class::<PyAmplitudeDamping>()?;
    m.add_class::<PyPhaseDamping>()?;
    m.add_class::<PyReadoutError>()?;
    Ok(())
}
