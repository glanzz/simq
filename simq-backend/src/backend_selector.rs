//! Backend selection logic
//!
//! Automatically selects the best backend for a given circuit based on
//! requirements, capabilities, and available backends.

use crate::{BackendError, QuantumBackend, Result};
use simq_core::Circuit;
use std::sync::Arc;

/// Criteria for backend selection
#[derive(Debug, Clone, Default)]
pub struct SelectionCriteria {
    /// Maximum allowed circuit depth
    pub max_depth: Option<usize>,

    /// Maximum number of qubits needed
    pub max_qubits: Option<usize>,

    /// Required native gates (if any)
    pub required_gates: Vec<String>,

    /// Prefer backends with specific features
    pub prefer_features: Vec<BackendFeature>,

    /// Maximum cost per shot (if applicable)
    pub max_cost_per_shot: Option<f64>,

    /// Prefer simulation over real hardware
    pub prefer_simulation: bool,
}

/// Backend features that can be preferred
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendFeature {
    /// Supports mid-circuit measurement
    MidCircuitMeasurement,

    /// Supports dynamic circuits
    DynamicCircuits,

    /// Has low error rates
    LowErrorRate,

    /// Provides fast execution
    FastExecution,

    /// Free to use
    Free,
}

impl SelectionCriteria {
    /// Create criteria from a circuit
    pub fn from_circuit(circuit: &Circuit) -> Self {
        Self {
            max_depth: Some(circuit.depth()),
            max_qubits: Some(circuit.num_qubits()),
            required_gates: Vec::new(),
            prefer_features: Vec::new(),
            max_cost_per_shot: None,
            prefer_simulation: false,
        }
    }

    /// Prefer simulation backends
    pub fn prefer_simulation(mut self) -> Self {
        self.prefer_simulation = true;
        self
    }

    /// Set maximum cost per shot
    pub fn max_cost(mut self, cost: f64) -> Self {
        self.max_cost_per_shot = Some(cost);
        self
    }

    /// Require specific gates
    pub fn require_gates(mut self, gates: Vec<String>) -> Self {
        self.required_gates = gates;
        self
    }

    /// Prefer specific features
    pub fn prefer_features(mut self, features: Vec<BackendFeature>) -> Self {
        self.prefer_features = features;
        self
    }
}

/// Backend selector that chooses the best backend for a circuit
pub struct BackendSelector {
    /// Available backends
    backends: Vec<Arc<dyn QuantumBackend>>,
}

impl BackendSelector {
    /// Create a new backend selector
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    /// Register a backend
    pub fn register(&mut self, backend: Arc<dyn QuantumBackend>) {
        self.backends.push(backend);
    }

    /// Select the best backend for a circuit
    pub fn select(
        &self,
        circuit: &Circuit,
        criteria: &SelectionCriteria,
    ) -> Result<Arc<dyn QuantumBackend>> {
        if self.backends.is_empty() {
            return Err(BackendError::Other("No backends available".to_string()));
        }

        // Filter compatible backends
        let mut compatible: Vec<_> = self
            .backends
            .iter()
            .filter(|b| self.is_compatible(b, circuit, criteria))
            .collect();

        if compatible.is_empty() {
            return Err(BackendError::Other(
                "No compatible backends found for circuit".to_string(),
            ));
        }

        // Score and sort backends
        compatible.sort_by_key(|b| std::cmp::Reverse(self.score_backend(b, circuit, criteria)));

        Ok(Arc::clone(compatible[0]))
    }

    /// Select backend by name
    pub fn select_by_name(&self, name: &str) -> Result<Arc<dyn QuantumBackend>> {
        self.backends
            .iter()
            .find(|b| b.name() == name)
            .cloned()
            .ok_or_else(|| BackendError::Other(format!("Backend '{}' not found", name)))
    }

    /// Get all available backends
    pub fn available_backends(&self) -> Vec<&str> {
        self.backends.iter().map(|b| b.name()).collect()
    }

    /// Check if backend is compatible with circuit and criteria
    fn is_compatible(
        &self,
        backend: &Arc<dyn QuantumBackend>,
        circuit: &Circuit,
        criteria: &SelectionCriteria,
    ) -> bool {
        let caps = backend.capabilities();

        // Check availability
        if !backend.is_available() {
            return false;
        }

        // Check qubit count
        if circuit.num_qubits() > caps.max_qubits {
            return false;
        }

        // Check max qubits criteria
        if let Some(max_qubits) = criteria.max_qubits {
            if max_qubits > caps.max_qubits {
                return false;
            }
        }

        // Check cost criteria
        if let Some(max_cost) = criteria.max_cost_per_shot {
            if let Some(cost) = backend.estimate_cost(circuit, 1000) {
                let cost_per_shot = cost / 1000.0;
                if cost_per_shot > max_cost {
                    return false;
                }
            }
        }

        // Check circuit validation
        backend.validate_circuit(circuit).is_ok()
    }

    /// Score a backend (higher is better)
    fn score_backend(
        &self,
        backend: &Arc<dyn QuantumBackend>,
        circuit: &Circuit,
        criteria: &SelectionCriteria,
    ) -> i32 {
        let mut score = 0;
        let caps = backend.capabilities();

        // Prefer simulation if requested
        if criteria.prefer_simulation {
            if backend.backend_type().to_string().contains("Simulator") {
                score += 1000;
            }
        } else {
            // Prefer real hardware
            if !backend.backend_type().to_string().contains("Simulator") {
                score += 500;
            }
        }

        // Prefer backends with more qubits (but not too many more)
        let qubit_diff = caps.max_qubits as i32 - circuit.num_qubits() as i32;
        if (0..10).contains(&qubit_diff) {
            score += 100 - qubit_diff * 5;
        }

        // Prefer lower cost
        if let Some(cost) = backend.estimate_cost(circuit, 1000) {
            let cost_per_shot = cost / 1000.0;
            if cost_per_shot == 0.0 {
                score += 200; // Free backends are great
            } else {
                score -= (cost_per_shot * 100.0) as i32;
            }
        } else {
            score += 150; // No cost = likely free
        }

        // Check feature preferences
        for feature in &criteria.prefer_features {
            match feature {
                BackendFeature::Free => {
                    if backend.estimate_cost(circuit, 1000).unwrap_or(0.0) == 0.0 {
                        score += 100;
                    }
                },
                BackendFeature::FastExecution => {
                    if backend.backend_type().to_string().contains("Simulator") {
                        score += 50;
                    }
                },
                _ => {},
            }
        }

        score
    }
}

impl Default for BackendSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackendCapabilities, BackendResult, BackendType};
    use std::collections::HashMap;

    // Mock backend for testing
    struct MockBackend {
        name: String,
        backend_type: BackendType,
        capabilities: BackendCapabilities,
        available: bool,
    }

    impl MockBackend {
        fn new(name: &str, max_qubits: usize, backend_type: BackendType) -> Self {
            Self {
                name: name.to_string(),
                backend_type,
                capabilities: BackendCapabilities {
                    max_qubits,
                    max_circuit_depth: None,
                    max_shots: Some(100000),
                    supported_gates: Default::default(),
                    native_gates: Default::default(),
                    connectivity: None,
                    supports_mid_circuit_measurement: false,
                    supports_conditional: false,
                    supports_reset: false,
                    supports_parametric: false,
                    cost_per_shot: None,
                    average_queue_time: None,
                    metadata: HashMap::new(),
                },
                available: true,
            }
        }
    }

    impl QuantumBackend for MockBackend {
        fn name(&self) -> &str {
            &self.name
        }

        fn backend_type(&self) -> BackendType {
            self.backend_type
        }

        fn execute(&self, _circuit: &Circuit, shots: usize) -> Result<BackendResult> {
            Ok(BackendResult {
                counts: HashMap::new(),
                shots,
                job_id: None,
                metadata: Default::default(),
            })
        }

        fn capabilities(&self) -> &BackendCapabilities {
            &self.capabilities
        }

        fn is_available(&self) -> bool {
            self.available
        }

        fn estimate_cost(&self, _circuit: &Circuit, shots: usize) -> Option<f64> {
            match self.backend_type {
                BackendType::Simulator => Some(0.0),
                _ => Some(shots as f64 * 0.001), // $0.001 per shot for hardware
            }
        }
    }

    #[test]
    fn test_selector_creation() {
        let selector = BackendSelector::new();
        assert_eq!(selector.available_backends().len(), 0);
    }

    #[test]
    fn test_register_backends() {
        let mut selector = BackendSelector::new();

        let backend1 = Arc::new(MockBackend::new("sim1", 10, BackendType::Simulator));
        let backend2 = Arc::new(MockBackend::new("hw1", 5, BackendType::Hardware));

        selector.register(backend1);
        selector.register(backend2);

        assert_eq!(selector.available_backends().len(), 2);
    }

    #[test]
    fn test_select_by_name() {
        let mut selector = BackendSelector::new();

        let backend = Arc::new(MockBackend::new("test_backend", 10, BackendType::Simulator));
        selector.register(backend);

        let selected = selector.select_by_name("test_backend").unwrap();
        assert_eq!(selected.name(), "test_backend");

        assert!(selector.select_by_name("nonexistent").is_err());
    }

    #[test]
    fn test_select_prefer_simulation() {
        let mut selector = BackendSelector::new();

        let sim = Arc::new(MockBackend::new("simulator", 10, BackendType::Simulator));
        let hw = Arc::new(MockBackend::new("hardware", 10, BackendType::Hardware));

        selector.register(sim);
        selector.register(hw);

        let circuit = Circuit::new(5);
        let criteria = SelectionCriteria::from_circuit(&circuit).prefer_simulation();

        let selected = selector.select(&circuit, &criteria).unwrap();
        assert_eq!(selected.name(), "simulator");
    }

    #[test]
    fn test_select_qubit_compatibility() {
        let mut selector = BackendSelector::new();

        let small = Arc::new(MockBackend::new("small", 5, BackendType::Simulator));
        let large = Arc::new(MockBackend::new("large", 20, BackendType::Simulator));

        selector.register(small);
        selector.register(large);

        // Circuit with 10 qubits should select large backend
        let circuit = Circuit::new(10);
        let criteria = SelectionCriteria::from_circuit(&circuit);

        let selected = selector.select(&circuit, &criteria).unwrap();
        assert_eq!(selected.name(), "large");
    }

    #[test]
    fn test_no_compatible_backends() {
        let mut selector = BackendSelector::new();

        let backend = Arc::new(MockBackend::new("small", 5, BackendType::Simulator));
        selector.register(backend);

        // Circuit needs 10 qubits, but backend only has 5
        let circuit = Circuit::new(10);
        let criteria = SelectionCriteria::from_circuit(&circuit);

        assert!(selector.select(&circuit, &criteria).is_err());
    }

    #[test]
    fn test_criteria_from_circuit() {
        let circuit = Circuit::new(5);
        let criteria = SelectionCriteria::from_circuit(&circuit);

        assert_eq!(criteria.max_qubits, Some(5));
        assert_eq!(criteria.max_depth, Some(circuit.depth()));
    }
}
