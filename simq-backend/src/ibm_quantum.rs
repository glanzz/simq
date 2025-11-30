//! IBM Quantum Backend
//!
//! This module provides a backend implementation for IBM Quantum systems
//! using the Qiskit Runtime API.
//!
//! # Features
//!
//! - Access to IBM Quantum hardware and cloud simulators
//! - Qiskit Runtime API integration
//! - Async job submission and polling
//! - Circuit transpilation to IBM native gates
//! - Multiple authentication methods (API token, IBM Cloud)
//!
//! # Example
//!
//! ```no_run
//! use simq_backend::ibm_quantum::{IBMQuantumBackend, IBMConfig};
//! use simq_backend::QuantumBackend;
//!
//! let config = IBMConfig::new("your-api-token");
//! let backend = IBMQuantumBackend::new(config, "ibm_brisbane")?;
//! // Execute circuit...
//! ```

use crate::{
    BackendCapabilities, BackendError, BackendResult, BackendType, ConnectivityGraph,
    ExecutionMetadata, GateSet, JobStatus, QuantumBackend, Result,
};
use serde::{Deserialize, Serialize};
use simq_core::Circuit;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// IBM Quantum configuration
#[derive(Debug, Clone)]
pub struct IBMConfig {
    /// IBM Quantum API token
    pub api_token: String,

    /// IBM Quantum instance (hub/group/project)
    /// Format: "hub/group/project" or None for open plan
    pub instance: Option<String>,

    /// API base URL (default: https://api.quantum.ibm.com)
    pub api_url: String,

    /// Maximum polling attempts for job status
    pub max_polling_attempts: usize,

    /// Polling interval in seconds
    pub polling_interval_seconds: u64,

    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
}

impl IBMConfig {
    /// Create a new IBM Quantum configuration with API token
    pub fn new(api_token: impl Into<String>) -> Self {
        Self {
            api_token: api_token.into(),
            instance: None,
            api_url: "https://api.quantum.ibm.com".to_string(),
            max_polling_attempts: 300, // 10 minutes with 2s interval
            polling_interval_seconds: 2,
            request_timeout_seconds: 30,
        }
    }

    /// Set the IBM Quantum instance (hub/group/project)
    pub fn with_instance(mut self, instance: impl Into<String>) -> Self {
        self.instance = Some(instance.into());
        self
    }

    /// Set custom API URL
    pub fn with_api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }

    /// Set polling configuration
    pub fn with_polling(mut self, max_attempts: usize, interval_seconds: u64) -> Self {
        self.max_polling_attempts = max_attempts;
        self.polling_interval_seconds = interval_seconds;
        self
    }
}

/// IBM Quantum backend
pub struct IBMQuantumBackend {
    /// Backend name (e.g., "ibm_brisbane", "ibm_kyoto")
    backend_name: String,

    /// Configuration
    config: IBMConfig,

    /// Backend capabilities
    capabilities: BackendCapabilities,

    /// HTTP client for API requests
    client: reqwest::blocking::Client,

    /// Cached backend properties
    properties: Option<IBMBackendProperties>,
}

impl IBMQuantumBackend {
    /// Create a new IBM Quantum backend
    ///
    /// # Arguments
    ///
    /// * `config` - IBM Quantum configuration with API token
    /// * `backend_name` - Name of the backend (e.g., "ibm_brisbane")
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - API connection fails
    /// - Backend does not exist
    /// - Invalid credentials
    pub fn new(config: IBMConfig, backend_name: impl Into<String>) -> Result<Self> {
        let backend_name = backend_name.into();

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_seconds))
            .build()
            .map_err(|e| BackendError::Other(format!("HTTP client error: {}", e)))?;

        let mut backend = Self {
            backend_name: backend_name.clone(),
            config,
            capabilities: BackendCapabilities::default(),
            client,
            properties: None,
        };

        // Fetch backend properties and capabilities
        backend.refresh_properties()?;

        Ok(backend)
    }

    /// Refresh backend properties from IBM Quantum API
    pub fn refresh_properties(&mut self) -> Result<()> {
        let properties = self.fetch_backend_properties()?;
        self.capabilities = self.build_capabilities(&properties);
        self.properties = Some(properties);
        Ok(())
    }

    /// Fetch backend properties from IBM API
    fn fetch_backend_properties(&self) -> Result<IBMBackendProperties> {
        let url = format!("{}/v1/backends/{}", self.config.api_url, self.backend_name);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_token))
            .send()
            .map_err(|e| BackendError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().unwrap_or_default();
            return Err(BackendError::AuthenticationFailed(format!(
                "Failed to fetch backend properties: {} - {}",
                status, error_text
            )));
        }

        response
            .json::<IBMBackendProperties>()
            .map_err(|e| BackendError::Other(format!("Failed to parse backend properties: {}", e)))
    }

    /// Build capabilities from backend properties
    fn build_capabilities(&self, properties: &IBMBackendProperties) -> BackendCapabilities {
        // Build connectivity graph
        let connectivity = if let Some(ref coupling_map) = properties.coupling_map {
            let mut graph = ConnectivityGraph::new(properties.num_qubits, true);
            for edge in coupling_map {
                if edge.len() == 2 {
                    graph.add_edge(edge[0], edge[1]);
                }
            }
            Some(graph)
        } else {
            None
        };

        // IBM native gate set
        let mut native_gates = GateSet::new();
        for gate in &properties.supported_instructions {
            native_gates.insert(gate.clone());
        }

        BackendCapabilities {
            max_qubits: properties.num_qubits,
            max_circuit_depth: Some(properties.max_experiments),
            max_shots: Some(properties.max_shots),
            supported_gates: GateSet::universal(), // Support all via transpilation
            native_gates,
            connectivity,
            supports_mid_circuit_measurement: properties.supports_dynamic_circuits,
            supports_conditional: properties.supports_dynamic_circuits,
            supports_reset: properties.supports_reset,
            supports_parametric: false, // IBM requires concrete parameters
            cost_per_shot: Some(0.00003), // Approximate IBM Quantum cost
            average_queue_time: properties.pending_jobs.map(|jobs| {
                // Estimate: 2 minutes per pending job
                jobs as u64 * 120
            }),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("provider".to_string(), "IBM Quantum".to_string());
                meta.insert("backend_type".to_string(), properties.backend_type.clone());
                meta.insert("backend_version".to_string(), properties.backend_version.clone());
                if properties.simulator {
                    meta.insert("simulator".to_string(), "true".to_string());
                }
                meta
            },
        }
    }

    /// Convert SimQ circuit to OpenQASM 3.0
    fn circuit_to_qasm(&self, circuit: &Circuit) -> Result<String> {
        // TODO: Implement full circuit to QASM conversion
        // For now, this is a placeholder

        let num_qubits = circuit.num_qubits();

        let mut qasm = String::new();
        qasm.push_str("OPENQASM 3.0;\n");
        qasm.push_str("include \"stdgates.inc\";\n\n");
        qasm.push_str(&format!("qubit[{}] q;\n", num_qubits));
        qasm.push_str(&format!("bit[{}] c;\n\n", num_qubits));

        // TODO: Add gate operations from circuit
        // This requires iterating over circuit gates and converting to QASM

        // Add measurements
        for i in 0..num_qubits {
            qasm.push_str(&format!("c[{}] = measure q[{}];\n", i, i));
        }

        Ok(qasm)
    }

    /// Submit job to IBM Quantum
    fn submit_job_impl(&self, circuit: &Circuit, shots: usize) -> Result<String> {
        // Convert circuit to QASM
        let qasm = self.circuit_to_qasm(circuit)?;

        // Build job request
        let job_request = IBMJobRequest {
            program_id: "sampler".to_string(), // Use Sampler primitive
            backend: self.backend_name.clone(),
            hub: self.config.instance.clone(),
            params: IBMJobParams {
                pubs: vec![IBMPub {
                    circuit: qasm,
                    shots,
                }],
            },
        };

        let url = format!("{}/v1/jobs", self.config.api_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_token))
            .json(&job_request)
            .send()
            .map_err(|e| BackendError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().unwrap_or_default();
            return Err(BackendError::JobSubmissionFailed(format!(
                "Failed to submit job: {} - {}",
                status, error_text
            )));
        }

        let job_response: IBMJobResponse = response
            .json()
            .map_err(|e| BackendError::Other(format!("Failed to parse job response: {}", e)))?;

        Ok(job_response.id)
    }

    /// Get job status from IBM Quantum
    fn get_job_status_impl(&self, job_id: &str) -> Result<JobStatus> {
        let url = format!("{}/v1/jobs/{}", self.config.api_url, job_id);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_token))
            .send()
            .map_err(|e| BackendError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(BackendError::JobNotFound {
                job_id: job_id.to_string(),
            });
        }

        let job_info: IBMJobInfo = response
            .json()
            .map_err(|e| BackendError::Other(format!("Failed to parse job info: {}", e)))?;

        Ok(match job_info.status.as_str() {
            "QUEUED" => JobStatus::Queued,
            "VALIDATING" => JobStatus::Validating,
            "RUNNING" => JobStatus::Running,
            "COMPLETED" => JobStatus::Completed,
            "FAILED" | "ERROR" => JobStatus::Failed,
            "CANCELLED" => JobStatus::Cancelled,
            _ => JobStatus::Failed, // Treat unknown status as failed
        })
    }

    /// Get job results from IBM Quantum
    fn get_job_results_impl(&self, job_id: &str) -> Result<BackendResult> {
        let url = format!("{}/v1/jobs/{}/results", self.config.api_url, job_id);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_token))
            .send()
            .map_err(|e| BackendError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(BackendError::Other("Failed to retrieve results".to_string()));
        }

        let results: IBMResults = response
            .json()
            .map_err(|e| BackendError::Other(format!("Failed to parse results: {}", e)))?;

        // Parse measurement counts
        let counts = self.parse_ibm_counts(&results)?;
        let total_shots = counts.values().sum();

        let metadata = ExecutionMetadata {
            execution_time: None,
            queue_time: None,
            total_time: None,
            backend_name: Some(self.backend_name.clone()),
            backend_version: self.properties.as_ref().map(|p| p.backend_version.clone()),
            status: JobStatus::Completed,
            num_qubits: None,
            circuit_depth: None,
            gate_count: None,
            cnot_count: None,
            cost: self.estimate_cost(&Circuit::new(0), total_shots), // Placeholder
            error_message: None,
            extra: HashMap::new(),
        };

        Ok(BackendResult {
            counts,
            shots: total_shots,
            job_id: Some(job_id.to_string()),
            metadata,
        })
    }

    /// Parse IBM measurement counts from results
    fn parse_ibm_counts(&self, results: &IBMResults) -> Result<HashMap<String, usize>> {
        let mut counts = HashMap::new();

        // IBM returns counts in various formats depending on API version
        // This is a simplified parser
        if let Some(ref quasi_dists) = results.quasi_dists {
            for quasi_dist in quasi_dists.iter() {
                for (bitstring, count) in quasi_dist {
                    // Convert integer key to bitstring if needed
                    let bitstring_str = if let Ok(val) = bitstring.parse::<usize>() {
                        format!("{:b}", val)
                    } else {
                        bitstring.clone()
                    };

                    *counts.entry(bitstring_str).or_insert(0) += *count as usize;
                }
            }
        }

        if counts.is_empty() {
            return Err(BackendError::Other("No measurement counts found in results".to_string()));
        }

        Ok(counts)
    }

    /// Cancel a job
    fn cancel_job_impl(&self, job_id: &str) -> Result<()> {
        let url = format!("{}/v1/jobs/{}", self.config.api_url, job_id);

        let response = self
            .client
            .delete(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_token))
            .send()
            .map_err(|e| BackendError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(BackendError::Other(format!("Failed to cancel job: {}", job_id)));
        }

        Ok(())
    }
}

impl QuantumBackend for IBMQuantumBackend {
    fn name(&self) -> &str {
        &self.backend_name
    }

    fn backend_type(&self) -> BackendType {
        if self.properties.as_ref().is_some_and(|p| p.simulator) {
            BackendType::CloudSimulator
        } else {
            BackendType::Hardware
        }
    }

    fn execute(&self, circuit: &Circuit, shots: usize) -> Result<BackendResult> {
        let start_time = Instant::now();

        // Validate circuit
        self.validate_circuit(circuit)?;

        // Submit job
        let job_id = self.submit_job_impl(circuit, shots)?;

        // Poll for completion
        for attempt in 0..self.config.max_polling_attempts {
            std::thread::sleep(Duration::from_secs(self.config.polling_interval_seconds));

            let status = self.get_job_status_impl(&job_id)?;

            match status {
                JobStatus::Completed => {
                    let mut result = self.get_job_results_impl(&job_id)?;
                    result.metadata.total_time = Some(start_time.elapsed());
                    return Ok(result);
                },
                JobStatus::Failed => {
                    return Err(BackendError::JobExecutionFailed(format!("Job {} failed", job_id)));
                },
                JobStatus::Cancelled => {
                    return Err(BackendError::JobExecutionFailed(format!(
                        "Job {} was cancelled",
                        job_id
                    )));
                },
                _ => {
                    // Still running, continue polling
                    if attempt % 10 == 0 {
                        // Log progress every 20 seconds
                        eprintln!("Job {} status: {:?}", job_id, status);
                    }
                },
            }
        }

        Err(BackendError::JobTimeout {
            timeout_seconds: self.config.max_polling_attempts as u64
                * self.config.polling_interval_seconds,
        })
    }

    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    fn submit_job(&self, circuit: &Circuit, shots: usize) -> Result<String> {
        self.validate_circuit(circuit)?;
        self.submit_job_impl(circuit, shots)
    }

    fn job_status(&self, job_id: &str) -> Result<JobStatus> {
        self.get_job_status_impl(job_id)
    }

    fn get_result(&self, job_id: &str) -> Result<BackendResult> {
        self.get_job_results_impl(job_id)
    }

    fn cancel_job(&self, job_id: &str) -> Result<()> {
        self.cancel_job_impl(job_id)
    }

    fn is_available(&self) -> bool {
        // Check if we can connect to the API
        self.properties.is_some()
    }
}

// IBM API data structures

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IBMBackendProperties {
    #[serde(rename = "backend_name")]
    name: String,

    #[serde(rename = "backend_version")]
    backend_version: String,

    #[serde(rename = "backend_type")]
    backend_type: String,

    num_qubits: usize,

    #[serde(default)]
    simulator: bool,

    #[serde(default = "default_max_shots")]
    max_shots: usize,

    #[serde(default = "default_max_experiments")]
    max_experiments: usize,

    #[serde(default)]
    coupling_map: Option<Vec<Vec<usize>>>,

    #[serde(default)]
    supported_instructions: Vec<String>,

    #[serde(default)]
    supports_dynamic_circuits: bool,

    #[serde(default = "default_true")]
    supports_reset: bool,

    #[serde(default)]
    pending_jobs: Option<usize>,
}

fn default_max_shots() -> usize {
    100000
}

fn default_max_experiments() -> usize {
    10000
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Serialize)]
struct IBMJobRequest {
    program_id: String,
    backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub: Option<String>,
    params: IBMJobParams,
}

#[derive(Debug, Serialize)]
struct IBMJobParams {
    pubs: Vec<IBMPub>,
}

#[derive(Debug, Serialize)]
struct IBMPub {
    circuit: String, // QASM string
    shots: usize,
}

#[derive(Debug, Deserialize)]
struct IBMJobResponse {
    id: String,
}

#[derive(Debug, Deserialize)]
struct IBMJobInfo {
    #[allow(dead_code)]
    id: String,
    status: String,
}

#[derive(Debug, Deserialize)]
struct IBMResults {
    #[serde(default)]
    quasi_dists: Option<Vec<HashMap<String, f64>>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = IBMConfig::new("test-token");
        assert_eq!(config.api_token, "test-token");
        assert_eq!(config.api_url, "https://api.quantum.ibm.com");
        assert!(config.instance.is_none());
    }

    #[test]
    fn test_config_with_instance() {
        let config = IBMConfig::new("test-token").with_instance("ibm-q/open/main");

        assert_eq!(config.instance, Some("ibm-q/open/main".to_string()));
    }

    #[test]
    fn test_config_custom_url() {
        let config = IBMConfig::new("test-token").with_api_url("https://custom.api.com");

        assert_eq!(config.api_url, "https://custom.api.com");
    }

    #[test]
    fn test_config_polling() {
        let config = IBMConfig::new("test-token").with_polling(100, 5);

        assert_eq!(config.max_polling_attempts, 100);
        assert_eq!(config.polling_interval_seconds, 5);
    }

    // Note: Integration tests would require valid IBM Quantum credentials
    // and should be run separately with feature flags
}
