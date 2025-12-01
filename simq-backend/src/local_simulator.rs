//! Local Simulator Backend
//!
//! This module provides a backend implementation that wraps the SimQ local
//! simulator (simq-sim) to provide execution through the unified backend API.
//!
//! # Features
//!
//! - Zero network overhead (local execution)
//! - Adaptive sparse/dense state representation
//! - Parallel gate execution
//! - Deterministic and sampling modes
//! - Full measurement support
//!
//! # Example
//!
//! ```no_run
//! use simq_backend::local_simulator::LocalSimulatorBackend;
//! use simq_backend::QuantumBackend;
//! use simq_core::Circuit;
//!
//! let backend = LocalSimulatorBackend::new();
//! // Execute circuit...
//! ```

use crate::{
    BackendCapabilities, BackendError, BackendResult, BackendType, ExecutionMetadata, JobStatus,
    QuantumBackend, Result,
};
use rand::SeedableRng;
use simq_core::Circuit;
use simq_sim::Simulator;
use simq_state::AdaptiveState;
use std::collections::HashMap;
use std::time::Instant;

/// Configuration for the local simulator backend
#[derive(Debug, Clone)]
pub struct LocalSimulatorConfig {
    /// Use deterministic seed for reproducibility
    pub seed: Option<u64>,

    /// Maximum number of qubits (default: 30 for practical memory limits)
    pub max_qubits: usize,

    /// Threshold for sparse-to-dense conversion
    pub sparse_threshold: f64,

    /// Enable parallel execution
    pub parallel: bool,

    /// Number of threads (None = use all available)
    pub num_threads: Option<usize>,
}

impl Default for LocalSimulatorConfig {
    fn default() -> Self {
        Self {
            seed: None,
            max_qubits: 30, // ~8GB for dense state
            sparse_threshold: 0.1,
            parallel: true,
            num_threads: None,
        }
    }
}

/// Local simulator backend using simq-sim
pub struct LocalSimulatorBackend {
    name: String,
    config: LocalSimulatorConfig,
    capabilities: BackendCapabilities,
}

impl LocalSimulatorBackend {
    /// Create a new local simulator backend with default configuration
    pub fn new() -> Self {
        Self::with_config(LocalSimulatorConfig::default())
    }

    /// Create a new local simulator backend with custom configuration
    pub fn with_config(config: LocalSimulatorConfig) -> Self {
        let capabilities = BackendCapabilities {
            max_qubits: config.max_qubits,
            max_circuit_depth: None, // No depth limit for simulator
            max_shots: None,         // No shot limit
            supported_gates: crate::GateSet::universal(),
            native_gates: crate::GateSet::universal(), // All gates are native
            connectivity: None,                        // All-to-all connectivity
            supports_mid_circuit_measurement: true,
            supports_conditional: true,
            supports_reset: true,
            supports_parametric: true,
            cost_per_shot: None, // Free!
            average_queue_time: None,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("type".to_string(), "local_simulator".to_string());
                meta.insert("parallel".to_string(), config.parallel.to_string());
                meta.insert("sparse_threshold".to_string(), config.sparse_threshold.to_string());
                meta
            },
        };

        Self {
            name: "LocalSimulator".to_string(),
            config,
            capabilities,
        }
    }

    /// Set the backend name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    /// Create a simulator instance for this backend
    fn create_simulator(&self, _num_qubits: usize) -> Simulator {
        Simulator::new(simq_sim::SimulatorConfig::default())
    }

    /// Convert simulator state to measurement counts
    fn sample_measurements(
        &self,
        state: &AdaptiveState,
        shots: usize,
    ) -> Result<HashMap<String, usize>> {
        let mut counts = HashMap::new();

        // Get probability distribution
        let probabilities = self.compute_probabilities(state)?;

        // Sample from the distribution
        let mut rng = if let Some(seed) = self.config.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        for _ in 0..shots {
            let sample = self.sample_once(&probabilities, &mut rng);
            *counts.entry(sample).or_insert(0) += 1;
        }

        Ok(counts)
    }

    /// Compute probability distribution from state
    fn compute_probabilities(&self, state: &AdaptiveState) -> Result<Vec<(String, f64)>> {
        match state {
            AdaptiveState::Dense(dense) => {
                let amplitudes = dense.amplitudes();
                let num_qubits = dense.num_qubits();

                let mut probs = Vec::new();
                for (i, amp) in amplitudes.iter().enumerate() {
                    let prob = amp.norm_sqr();
                    if prob > 1e-10 {
                        // Only include non-negligible probabilities
                        let bitstring = format!("{:0width$b}", i, width = num_qubits);
                        probs.push((bitstring, prob));
                    }
                }

                Ok(probs)
            },
            AdaptiveState::Sparse { state, .. } => {
                let num_qubits = state.num_qubits();
                let mut probs = Vec::new();

                for (basis_state, amp) in state.amplitudes().iter() {
                    let prob = amp.norm_sqr();
                    if prob > 1e-10 {
                        let bitstring = format!("{:0width$b}", basis_state, width = num_qubits);
                        probs.push((bitstring, prob));
                    }
                }

                Ok(probs)
            },
        }
    }

    /// Sample once from probability distribution
    fn sample_once<R: rand::Rng>(&self, probabilities: &[(String, f64)], rng: &mut R) -> String {
        let mut cumulative = 0.0;
        let random_value: f64 = rng.gen();

        for (bitstring, prob) in probabilities {
            cumulative += prob;
            if random_value <= cumulative {
                return bitstring.clone();
            }
        }

        // Fallback (shouldn't happen if probabilities sum to 1)
        probabilities.last().unwrap().0.clone()
    }
}

impl Default for LocalSimulatorBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumBackend for LocalSimulatorBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Simulator
    }

    fn execute(&self, circuit: &Circuit, shots: usize) -> Result<BackendResult> {
        let start_time = Instant::now();

        // Validate circuit
        self.validate_circuit(circuit)?;

        // Create simulator
        let simulator = self.create_simulator(circuit.num_qubits());

        // Run simulation
        let sim_result = simulator
            .run(circuit)
            .map_err(|e| BackendError::JobExecutionFailed(e.to_string()))?;

        // Sample measurements
        let counts = self.sample_measurements(&sim_result.state, shots)?;

        // Create metadata
        let execution_time = start_time.elapsed();
        let metadata = ExecutionMetadata {
            execution_time: Some(execution_time),
            total_time: Some(execution_time),
            backend_name: Some(self.name.clone()),
            backend_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            status: JobStatus::Completed,
            num_qubits: Some(circuit.num_qubits()),
            circuit_depth: None, // TODO: Add when Circuit supports depth()
            gate_count: None,    // TODO: Add when Circuit supports gate count
            cnot_count: None,
            cost: None,
            error_message: None,
            queue_time: None,
            extra: HashMap::new(),
        };

        Ok(BackendResult {
            counts,
            shots,
            job_id: None,
            metadata,
        })
    }

    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    fn is_available(&self) -> bool {
        true // Local simulator is always available
    }

    fn estimate_cost(&self, _circuit: &Circuit, _shots: usize) -> Option<f64> {
        Some(0.0) // Free!
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_core::CircuitBuilder;
    use simq_gates::standard::PauliX;
    use std::sync::Arc;

    #[test]
    fn test_backend_creation() {
        let backend = LocalSimulatorBackend::new();
        assert_eq!(backend.name(), "LocalSimulator");
        assert_eq!(backend.backend_type(), BackendType::Simulator);
        assert!(backend.is_available());
    }

    #[test]
    fn test_backend_with_custom_name() {
        let backend = LocalSimulatorBackend::new().with_name("MySimulator".to_string());
        assert_eq!(backend.name(), "MySimulator");
    }

    #[test]
    fn test_backend_capabilities() {
        let backend = LocalSimulatorBackend::new();
        let caps = backend.capabilities();

        assert_eq!(caps.max_qubits, 30);
        assert!(caps.supports_mid_circuit_measurement);
        assert!(caps.supports_conditional);
        assert!(caps.supports_parametric);
        assert_eq!(caps.cost_per_shot, None);
    }

    #[test]
    fn test_cost_estimation() {
        let backend = LocalSimulatorBackend::new();
        let circuit = CircuitBuilder::<3>::new().build();

        let cost = backend.estimate_cost(&circuit, 1000);
        assert_eq!(cost, Some(0.0));
    }

    #[test]
    fn test_circuit_validation() {
        let backend = LocalSimulatorBackend::with_config(LocalSimulatorConfig {
            max_qubits: 5,
            ..Default::default()
        });

        // Valid circuit
        let small_circuit = CircuitBuilder::<3>::new().build();
        assert!(backend.validate_circuit(&small_circuit).is_ok());

        // Too many qubits
        let large_circuit = CircuitBuilder::<10>::new().build();
        assert!(backend.validate_circuit(&large_circuit).is_err());
    }

    #[test]
    fn test_simple_execution() {
        let backend = LocalSimulatorBackend::new();

        // Create a simple 2-qubit circuit with a gate
        let mut circuit = CircuitBuilder::<2>::new();
        circuit
            .apply_gate(Arc::new(PauliX), &[circuit.qubits()[0]])
            .unwrap();
        let circuit = circuit.build();

        // Execute with 100 shots
        let result = backend.execute(&circuit, 100);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.shots, 100);
        assert!(result.metadata.is_success());
        assert_eq!(result.metadata.status, JobStatus::Completed);

        // Should measure some deterministic outcome with high probability (X gate applied)
        let most_frequent = result.most_frequent();
        assert!(most_frequent.is_some());
        let (_, count) = most_frequent.unwrap();
        assert!(*count > 80); // At least 80% probability for the most likely outcome
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let config = LocalSimulatorConfig {
            seed: Some(42),
            ..Default::default()
        };

        let backend1 = LocalSimulatorBackend::with_config(config.clone());
        let backend2 = LocalSimulatorBackend::with_config(config);

        let mut circuit = CircuitBuilder::<2>::new();
        circuit
            .apply_gate(Arc::new(PauliX), &[circuit.qubits()[0]])
            .unwrap();
        let circuit = circuit.build();

        let result1 = backend1.execute(&circuit, 100).unwrap();
        let result2 = backend2.execute(&circuit, 100).unwrap();

        // Results should be identical with same seed
        assert_eq!(result1.counts, result2.counts);
    }
}
