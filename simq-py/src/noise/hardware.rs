use pyo3::prelude::*;
use simq_core::noise::hardware::{
    HardwareNoiseModel as RustHardwareNoiseModel,
};
use crate::noise::channels::{
    PyAmplitudeDamping, PyPhaseDamping, PyDepolarizingChannel, PyReadoutError
};

#[pyclass(name = "HardwareNoiseModel")]
#[derive(Clone)]
pub struct PyHardwareNoiseModel {
    pub inner: RustHardwareNoiseModel,
}

#[pymethods]
impl PyHardwareNoiseModel {
    #[new]
    fn new(num_qubits: usize) -> Self {
        Self {
            inner: RustHardwareNoiseModel::new(num_qubits),
        }
    }

    #[staticmethod]
    fn ibm_washington() -> Self {
        Self {
            inner: RustHardwareNoiseModel::ibm_washington(),
        }
    }

    #[staticmethod]
    fn google_sycamore() -> Self {
        Self {
            inner: RustHardwareNoiseModel::google_sycamore(),
        }
    }

    #[staticmethod]
    fn ionq_aria() -> Self {
        Self {
            inner: RustHardwareNoiseModel::ionq_aria(),
        }
    }

    #[staticmethod]
    fn ibm_falcon_5q() -> Self {
        Self {
            inner: RustHardwareNoiseModel::ibm_falcon_5q(),
        }
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn set_qubit_t1(&mut self, qubit: usize, t1: f64) {
        self.inner.set_qubit_t1(qubit, t1);
    }

    fn set_qubit_t2(&mut self, qubit: usize, t2: f64) {
        self.inner.set_qubit_t2(qubit, t2);
    }

    fn set_readout_error(&mut self, qubit: usize, p01: f64, p10: f64) {
        self.inner.set_readout_error(qubit, p01, p10);
    }

    fn set_single_qubit_fidelity(&mut self, qubit: usize, fidelity: f64) {
        self.inner.set_single_qubit_fidelity(qubit, fidelity);
    }

    fn set_two_qubit_gate(&mut self, qubit1: usize, qubit2: usize, fidelity: f64, duration_us: f64) {
        self.inner.set_two_qubit_gate(qubit1, qubit2, fidelity, duration_us);
    }

    fn set_idle_noise_enabled(&mut self, enabled: bool) {
        self.inner.set_idle_noise_enabled(enabled);
    }

    fn is_idle_noise_enabled(&self) -> bool {
        self.inner.is_idle_noise_enabled()
    }

    fn set_crosstalk_enabled(&mut self, enabled: bool) {
        self.inner.set_crosstalk_enabled(enabled);
    }

    fn is_crosstalk_enabled(&self) -> bool {
        self.inner.is_crosstalk_enabled()
    }

    // Helper methods to get noise channels for a qubit (useful for debugging/inspection)
    
    fn amplitude_damping_single_gate(&self, qubit: usize) -> PyResult<PyAmplitudeDamping> {
        let inner = self.inner.amplitude_damping_single_gate(qubit)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyAmplitudeDamping { inner })
    }

    fn phase_damping_single_gate(&self, qubit: usize) -> PyResult<PyPhaseDamping> {
        let inner = self.inner.phase_damping_single_gate(qubit)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyPhaseDamping { inner })
    }

    fn depolarizing_single_gate(&self, qubit: usize) -> PyResult<PyDepolarizingChannel> {
        let inner = self.inner.depolarizing_single_gate(qubit)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyDepolarizingChannel { inner })
    }

    fn readout_error(&self, qubit: usize) -> PyResult<PyReadoutError> {
        let inner = self.inner.readout_error(qubit)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyReadoutError { inner })
    }
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHardwareNoiseModel>()?;
    Ok(())
}
