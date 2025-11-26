//! Python bindings for quantum backends
//!
//! This module provides Python bindings for SimQ's backend system, enabling
//! execution on local simulators and cloud quantum hardware.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use simq_backend::{
    BackendCapabilities, BackendResult, BackendType, JobStatus, LocalSimulatorBackend,
    LocalSimulatorConfig, QuantumBackend as RustQuantumBackend,
};
use simq_core::Circuit;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "ibm-quantum")]
use simq_backend::ibm_quantum::{IBMConfig, IBMQuantumBackend};

/// Backend execution result containing measurement counts and metadata
#[pyclass(name = "BackendResult")]
#[derive(Clone)]
pub struct PyBackendResult {
    inner: BackendResult,
}

#[pymethods]
impl PyBackendResult {
    /// Get measurement counts as a dictionary
    #[getter]
    fn counts(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (state, count) in &self.inner.counts {
            dict.set_item(state, count)?;
        }
        Ok(dict.into())
    }

    /// Total number of shots
    #[getter]
    fn shots(&self) -> usize {
        self.inner.shots
    }

    /// Job ID (if applicable)
    #[getter]
    fn job_id(&self) -> Option<String> {
        self.inner.job_id.clone()
    }

    /// Get probabilities (normalized counts)
    fn probabilities(&self, py: Python) -> PyResult<PyObject> {
        let probs = self.inner.probabilities();
        let dict = PyDict::new_bound(py);
        for (state, prob) in probs {
            dict.set_item(state, prob)?;
        }
        Ok(dict.into())
    }

    /// Get the most frequent measurement outcome
    fn most_frequent(&self) -> Option<String> {
        self.inner.most_frequent().map(|(s, _)| s.clone())
    }

    /// Get count for a specific bitstring
    fn get_count(&self, bitstring: &str) -> usize {
        self.inner.get_count(bitstring)
    }

    /// Backend name
    #[getter]
    fn backend_name(&self) -> Option<String> {
        self.inner.metadata.backend_name.clone()
    }

    /// Execution time in seconds
    #[getter]
    fn execution_time(&self) -> Option<f64> {
        self.inner
            .metadata
            .execution_time
            .map(|d| d.as_secs_f64())
    }

    /// Total time (including queue wait) in seconds
    #[getter]
    fn total_time(&self) -> Option<f64> {
        self.inner.metadata.total_time.map(|d| d.as_secs_f64())
    }

    /// Cost in dollars
    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.metadata.cost
    }

    fn __repr__(&self) -> String {
        format!(
            "BackendResult(shots={}, outcomes={}, backend={})",
            self.shots(),
            self.inner.counts.len(),
            self.backend_name().unwrap_or_else(|| "unknown".to_string())
        )
    }
}

/// Job execution status
#[pyclass(name = "JobStatus")]
#[derive(Clone)]
pub struct PyJobStatus {
    inner: JobStatus,
}

#[pymethods]
impl PyJobStatus {
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __str__(&self) -> String {
        match self.inner {
            JobStatus::Queued => "Queued".to_string(),
            JobStatus::Validating => "Validating".to_string(),
            JobStatus::Running => "Running".to_string(),
            JobStatus::Completed => "Completed".to_string(),
            JobStatus::Failed { .. } => "Failed".to_string(),
            JobStatus::Cancelled => "Cancelled".to_string(),
        }
    }

    /// Check if job is completed
    fn is_completed(&self) -> bool {
        matches!(self.inner, JobStatus::Completed)
    }

    /// Check if job failed
    fn is_failed(&self) -> bool {
        matches!(self.inner, JobStatus::Failed { .. })
    }

    /// Check if job is running
    fn is_running(&self) -> bool {
        matches!(self.inner, JobStatus::Running)
    }
}

/// Backend type enum
#[pyclass(name = "BackendType")]
#[derive(Clone)]
pub struct PyBackendType {
    inner: BackendType,
}

#[pymethods]
impl PyBackendType {
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __str__(&self) -> String {
        match self.inner {
            BackendType::Simulator => "Simulator".to_string(),
            BackendType::CloudSimulator => "CloudSimulator".to_string(),
            BackendType::Hardware => "Hardware".to_string(),
            BackendType::HybridSimulator => "HybridSimulator".to_string(),
            BackendType::Emulator => "Emulator".to_string(),
        }
    }
}

/// Local simulator backend configuration
#[pyclass(name = "LocalSimulatorConfig")]
#[derive(Clone)]
pub struct PyLocalSimulatorConfig {
    inner: LocalSimulatorConfig,
}

#[pymethods]
impl PyLocalSimulatorConfig {
    #[new]
    #[pyo3(signature = (seed=None, max_qubits=30, sparse_threshold=0.1, parallel=true, num_threads=None))]
    fn new(
        seed: Option<u64>,
        max_qubits: usize,
        sparse_threshold: f64,
        parallel: bool,
        num_threads: Option<usize>,
    ) -> Self {
        Self {
            inner: LocalSimulatorConfig {
                seed,
                max_qubits,
                sparse_threshold,
                parallel,
                num_threads,
            },
        }
    }

    #[getter]
    fn seed(&self) -> Option<u64> {
        self.inner.seed
    }

    #[getter]
    fn max_qubits(&self) -> usize {
        self.inner.max_qubits
    }

    #[getter]
    fn sparse_threshold(&self) -> f64 {
        self.inner.sparse_threshold
    }

    #[getter]
    fn parallel(&self) -> bool {
        self.inner.parallel
    }

    #[getter]
    fn num_threads(&self) -> Option<usize> {
        self.inner.num_threads
    }

    fn __repr__(&self) -> String {
        format!(
            "LocalSimulatorConfig(seed={:?}, max_qubits={}, sparse_threshold={}, parallel={}, num_threads={:?})",
            self.seed(),
            self.max_qubits(),
            self.sparse_threshold(),
            self.parallel(),
            self.num_threads()
        )
    }
}

/// Local quantum simulator backend
#[pyclass(name = "LocalSimulatorBackend")]
pub struct PyLocalSimulatorBackend {
    inner: Arc<LocalSimulatorBackend>,
}

#[pymethods]
impl PyLocalSimulatorBackend {
    /// Create a new local simulator backend with default configuration
    #[new]
    #[pyo3(signature = (config=None, name=None))]
    fn new(config: Option<PyLocalSimulatorConfig>, name: Option<String>) -> Self {
        let backend = match (config, name) {
            (Some(cfg), Some(n)) => {
                LocalSimulatorBackend::with_config(cfg.inner).with_name(n)
            }
            (Some(cfg), None) => LocalSimulatorBackend::with_config(cfg.inner),
            (None, Some(n)) => LocalSimulatorBackend::new().with_name(n),
            (None, None) => LocalSimulatorBackend::new(),
        };
        Self {
            inner: Arc::new(backend),
        }
    }

    /// Get backend name
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Get backend type
    fn backend_type(&self) -> PyBackendType {
        PyBackendType {
            inner: self.inner.backend_type(),
        }
    }

    /// Execute a circuit and return results
    #[pyo3(signature = (circuit, shots=1024))]
    fn execute(
        &self,
        py: Python,
        circuit: &crate::core::circuit::PyCircuit,
        shots: usize,
    ) -> PyResult<PyBackendResult> {
        // Release GIL during computation
        let result = py.allow_threads(|| {
            self.inner
                .execute(circuit.inner(), shots)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;

        Ok(PyBackendResult { inner: result })
    }

    /// Submit a job (for compatibility, executes immediately for local backend)
    #[pyo3(signature = (circuit, shots=1024))]
    fn submit_job(&self, circuit: &crate::core::circuit::PyCircuit, shots: usize) -> PyResult<String> {
        self.inner
            .submit_job(circuit.inner(), shots)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get job status
    fn job_status(&self, job_id: &str) -> PyResult<PyJobStatus> {
        let status = self
            .inner
            .job_status(job_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyJobStatus { inner: status })
    }

    /// Get job result
    fn get_result(&self, job_id: &str) -> PyResult<PyBackendResult> {
        let result = self
            .inner
            .get_result(job_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyBackendResult { inner: result })
    }

    /// Check if backend is available
    fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    /// Estimate cost for executing circuit
    #[pyo3(signature = (circuit, shots=1024))]
    fn estimate_cost(&self, circuit: &crate::core::circuit::PyCircuit, shots: usize) -> Option<f64> {
        self.inner.estimate_cost(circuit.inner(), shots)
    }

    /// Get maximum number of qubits
    fn max_qubits(&self) -> usize {
        self.inner.capabilities().max_qubits
    }

    fn __repr__(&self) -> String {
        format!(
            "LocalSimulatorBackend(name='{}', max_qubits={})",
            self.name(),
            self.max_qubits()
        )
    }
}

/// IBM Quantum backend configuration
#[cfg(feature = "ibm-quantum")]
#[pyclass(name = "IBMConfig")]
#[derive(Clone)]
pub struct PyIBMConfig {
    inner: IBMConfig,
}

#[cfg(feature = "ibm-quantum")]
#[pymethods]
impl PyIBMConfig {
    #[new]
    #[pyo3(signature = (api_token, instance=None, api_url=None, max_polling_attempts=300, polling_interval_seconds=2, request_timeout_seconds=30))]
    fn new(
        api_token: String,
        instance: Option<String>,
        api_url: Option<String>,
        max_polling_attempts: usize,
        polling_interval_seconds: u64,
        request_timeout_seconds: u64,
    ) -> Self {
        let mut config = IBMConfig::new(api_token);
        if let Some(inst) = instance {
            config = config.with_instance(&inst);
        }
        if let Some(url) = api_url {
            config = config.with_api_url(&url);
        }
        config = config.with_polling(max_polling_attempts, polling_interval_seconds);
        config = config.with_timeout(request_timeout_seconds);
        Self { inner: config }
    }

    fn __repr__(&self) -> String {
        format!(
            "IBMConfig(instance={:?}, max_polling_attempts={}, polling_interval={}s)",
            self.inner.instance, self.inner.max_polling_attempts, self.inner.polling_interval_seconds
        )
    }
}

/// IBM Quantum backend for real quantum hardware and cloud simulators
#[cfg(feature = "ibm-quantum")]
#[pyclass(name = "IBMQuantumBackend")]
pub struct PyIBMQuantumBackend {
    inner: Arc<IBMQuantumBackend>,
}

#[cfg(feature = "ibm-quantum")]
#[pymethods]
impl PyIBMQuantumBackend {
    /// Create a new IBM Quantum backend
    ///
    /// Args:
    ///     config: IBM Quantum configuration with API token
    ///     backend_name: Name of the backend (e.g., 'ibm_brisbane', 'ibmq_qasm_simulator')
    #[new]
    fn new(config: PyIBMConfig, backend_name: String) -> PyResult<Self> {
        let backend = IBMQuantumBackend::new(config.inner, &backend_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(backend),
        })
    }

    /// Get backend name
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Get backend type
    fn backend_type(&self) -> PyBackendType {
        PyBackendType {
            inner: self.inner.backend_type(),
        }
    }

    /// Execute a circuit on IBM Quantum hardware/simulator
    ///
    /// This will submit the job, wait for completion, and return results.
    /// For long-running jobs, use submit_job() and poll status instead.
    #[pyo3(signature = (circuit, shots=1024))]
    fn execute(
        &self,
        py: Python,
        circuit: &crate::core::circuit::PyCircuit,
        shots: usize,
    ) -> PyResult<PyBackendResult> {
        // Release GIL during execution (includes waiting)
        let result = py.allow_threads(|| {
            self.inner
                .execute(circuit.inner(), shots)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;

        Ok(PyBackendResult { inner: result })
    }

    /// Submit a job without waiting for completion
    #[pyo3(signature = (circuit, shots=1024))]
    fn submit_job(&self, circuit: &crate::core::circuit::PyCircuit, shots: usize) -> PyResult<String> {
        self.inner
            .submit_job(circuit.inner(), shots)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get job status
    fn job_status(&self, job_id: &str) -> PyResult<PyJobStatus> {
        let status = self
            .inner
            .job_status(job_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyJobStatus { inner: status })
    }

    /// Get job result
    fn get_result(&self, py: Python, job_id: &str) -> PyResult<PyBackendResult> {
        // Release GIL during potentially long wait
        let result = py.allow_threads(|| {
            self.inner
                .get_result(job_id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(PyBackendResult { inner: result })
    }

    /// Cancel a running job
    fn cancel_job(&self, job_id: &str) -> PyResult<()> {
        self.inner
            .cancel_job(job_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Check if backend is available
    fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    /// Estimate cost for executing circuit
    #[pyo3(signature = (circuit, shots=1024))]
    fn estimate_cost(&self, circuit: &crate::core::circuit::PyCircuit, shots: usize) -> Option<f64> {
        self.inner.estimate_cost(circuit.inner(), shots)
    }

    /// Get maximum number of qubits
    fn max_qubits(&self) -> usize {
        self.inner.capabilities().max_qubits
    }

    /// Get average queue time in seconds
    fn average_queue_time(&self) -> Option<u64> {
        self.inner.capabilities().average_queue_time
    }

    fn __repr__(&self) -> String {
        format!(
            "IBMQuantumBackend(name='{}', type={}, max_qubits={})",
            self.name(),
            self.backend_type().__str__(),
            self.max_qubits()
        )
    }
}

/// Register backend module with Python
pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Result types
    m.add_class::<PyBackendResult>()?;
    m.add_class::<PyJobStatus>()?;
    m.add_class::<PyBackendType>()?;

    // Local simulator
    m.add_class::<PyLocalSimulatorConfig>()?;
    m.add_class::<PyLocalSimulatorBackend>()?;

    // IBM Quantum (feature-gated)
    #[cfg(feature = "ibm-quantum")]
    {
        m.add_class::<PyIBMConfig>()?;
        m.add_class::<PyIBMQuantumBackend>()?;
    }

    Ok(())
}
