//! Core backend trait and types

use crate::{BackendCapabilities, BackendResult, Result};
use simq_core::Circuit;
use std::fmt;

/// Trait for quantum execution backends
///
/// This trait provides a unified interface for executing quantum circuits
/// on different backends, including simulators and real quantum hardware.
///
/// # Example
///
/// ```no_run
/// use simq_backend::{QuantumBackend, BackendResult};
/// use simq_core::Circuit;
///
/// fn execute_circuit<B: QuantumBackend>(backend: &B, circuit: &Circuit, shots: usize) -> BackendResult {
///     backend.execute(circuit, shots).expect("Execution failed")
/// }
/// ```
pub trait QuantumBackend: Send + Sync {
    /// Get the backend name
    fn name(&self) -> &str;

    /// Get backend type
    fn backend_type(&self) -> BackendType;

    /// Execute a circuit synchronously
    ///
    /// # Arguments
    ///
    /// * `circuit` - The quantum circuit to execute
    /// * `shots` - Number of measurement shots
    ///
    /// # Returns
    ///
    /// A `BackendResult` containing measurement counts and metadata
    fn execute(&self, circuit: &Circuit, shots: usize) -> Result<BackendResult>;

    /// Get backend capabilities
    fn capabilities(&self) -> &BackendCapabilities;

    /// Submit a job asynchronously (optional, for cloud backends)
    ///
    /// Returns a job ID that can be used to query status and retrieve results
    fn submit_job(&self, circuit: &Circuit, shots: usize) -> Result<String> {
        // Default implementation: execute synchronously and return fake job ID
        let _result = self.execute(circuit, shots)?;
        Ok(format!("sync-{}", uuid::Uuid::new_v4()))
    }

    /// Get job status (optional, for cloud backends)
    fn job_status(&self, _job_id: &str) -> Result<crate::JobStatus> {
        Ok(crate::JobStatus::Completed)
    }

    /// Retrieve job results (optional, for cloud backends)
    fn get_result(&self, _job_id: &str) -> Result<BackendResult> {
        Err(crate::BackendError::Other("Async job retrieval not supported".to_string()))
    }

    /// Cancel a running job (optional, for cloud backends)
    fn cancel_job(&self, _job_id: &str) -> Result<()> {
        Ok(())
    }

    /// Check if the backend is available
    fn is_available(&self) -> bool {
        true
    }

    /// Estimate cost for execution (in credits, if applicable)
    fn estimate_cost(&self, _circuit: &Circuit, shots: usize) -> Option<f64> {
        self.capabilities()
            .cost_per_shot
            .map(|cost_per_shot| cost_per_shot * shots as f64)
    }

    /// Validate circuit compatibility with backend
    fn validate_circuit(&self, circuit: &Circuit) -> Result<()> {
        let caps = self.capabilities();

        // Check qubit count
        if circuit.num_qubits() > caps.max_qubits {
            return Err(crate::BackendError::CapabilityExceeded(format!(
                "Circuit requires {} qubits, backend supports max {}",
                circuit.num_qubits(),
                caps.max_qubits
            )));
        }

        // TODO: Add circuit depth and gate support checks once Circuit API is stable

        Ok(())
    }

    /// Get backend description/status
    fn description(&self) -> String {
        format!(
            "{} ({}) - {} qubits",
            self.name(),
            self.backend_type(),
            self.capabilities().max_qubits
        )
    }
}

/// Backend type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Local simulator
    Simulator,

    /// Cloud simulator
    CloudSimulator,

    /// Real quantum hardware
    Hardware,

    /// Hybrid simulator (uses both CPU and GPU)
    HybridSimulator,

    /// Emulator (simulates hardware noise)
    Emulator,
}

impl fmt::Display for BackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendType::Simulator => write!(f, "Simulator"),
            BackendType::CloudSimulator => write!(f, "Cloud Simulator"),
            BackendType::Hardware => write!(f, "Hardware"),
            BackendType::HybridSimulator => write!(f, "Hybrid Simulator"),
            BackendType::Emulator => write!(f, "Emulator"),
        }
    }
}

/// Helper trait for async backends (requires "async" feature)
#[cfg(feature = "async")]
#[async_trait::async_trait]
pub trait AsyncQuantumBackend: QuantumBackend {
    /// Execute a circuit asynchronously
    async fn execute_async(&self, circuit: &Circuit, shots: usize) -> Result<BackendResult>;

    /// Submit a job and wait for completion
    async fn execute_and_wait(
        &self,
        circuit: &Circuit,
        shots: usize,
        timeout_seconds: Option<u64>,
    ) -> Result<BackendResult> {
        let job_id = self.submit_job(circuit, shots)?;

        let start = std::time::Instant::now();
        loop {
            let status = self.job_status(&job_id)?;

            match status {
                crate::JobStatus::Completed => {
                    return self.get_result(&job_id);
                },
                crate::JobStatus::Failed | crate::JobStatus::Cancelled => {
                    return Err(crate::BackendError::JobExecutionFailed(format!(
                        "Job {} ended with status: {}",
                        job_id, status
                    )));
                },
                _ => {
                    // Still running, check timeout
                    if let Some(timeout) = timeout_seconds {
                        if start.elapsed().as_secs() > timeout {
                            self.cancel_job(&job_id)?;
                            return Err(crate::BackendError::JobTimeout {
                                timeout_seconds: timeout,
                            });
                        }
                    }

                    // Wait before polling again
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock backend for testing
    struct MockBackend {
        name: String,
        capabilities: BackendCapabilities,
    }

    impl QuantumBackend for MockBackend {
        fn name(&self) -> &str {
            &self.name
        }

        fn backend_type(&self) -> BackendType {
            BackendType::Simulator
        }

        fn execute(&self, _circuit: &Circuit, _shots: usize) -> Result<BackendResult> {
            use std::collections::HashMap;
            let mut counts = HashMap::new();
            counts.insert("000".to_string(), 100);
            Ok(BackendResult::new(counts, 100))
        }

        fn capabilities(&self) -> &BackendCapabilities {
            &self.capabilities
        }
    }

    #[test]
    fn test_backend_trait() {
        let backend = MockBackend {
            name: "test_backend".to_string(),
            capabilities: BackendCapabilities::simulator(),
        };

        assert_eq!(backend.name(), "test_backend");
        assert_eq!(backend.backend_type(), BackendType::Simulator);
        assert!(backend.is_available());
    }

    #[test]
    fn test_backend_types() {
        assert_eq!(format!("{}", BackendType::Simulator), "Simulator");
        assert_eq!(format!("{}", BackendType::Hardware), "Hardware");
    }
}
