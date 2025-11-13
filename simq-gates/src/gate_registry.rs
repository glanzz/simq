//! Custom gate registry for managing and retrieving custom gates
//!
//! The registry provides a centralized way to store, retrieve, and manage
//! custom quantum gates across an application.
//!
//! # Example
//!
//! ```rust
//! use simq_gates::gate_registry::GateRegistry;
//! use simq_gates::custom::CustomGateBuilder;
//! use num_complex::Complex64;
//! use std::f64::consts::SQRT_2;
//!
//! let mut registry = GateRegistry::new();
//!
//! // Create and register a custom gate
//! let inv_sqrt2 = 1.0 / SQRT_2;
//! let hadamard = CustomGateBuilder::new("MyH")
//!     .matrix_2x2([
//!         [Complex64::new(inv_sqrt2, 0.0), Complex64::new(inv_sqrt2, 0.0)],
//!         [Complex64::new(inv_sqrt2, 0.0), Complex64::new(-inv_sqrt2, 0.0)],
//!     ])
//!     .build()
//!     .unwrap();
//!
//! registry.register("myh_1q", hadamard);
//!
//! // Retrieve the gate
//! if let Some(gate) = registry.get("myh_1q") {
//!     println!("Found gate: {}", gate.name());
//! }
//! ```

use super::custom::CustomGate;
use simq_core::gate::Gate;
use std::collections::HashMap;
use std::sync::Arc;

/// A registry for managing custom quantum gates
///
/// Provides centralized storage and retrieval of custom gates,
/// making it easy to share gates across an application.
#[derive(Debug)]
pub struct GateRegistry {
    gates: HashMap<String, Arc<CustomGate>>,
}

impl GateRegistry {
    /// Create a new empty gate registry
    pub fn new() -> Self {
        Self {
            gates: HashMap::new(),
        }
    }

    /// Create a registry with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            gates: HashMap::with_capacity(capacity),
        }
    }

    /// Register a custom gate
    ///
    /// If a gate with the same name already exists, it will be replaced.
    pub fn register(&mut self, name: impl Into<String>, gate: CustomGate) {
        self.gates.insert(name.into(), Arc::new(gate));
    }

    /// Register a custom gate as an Arc
    pub fn register_arc(&mut self, name: impl Into<String>, gate: Arc<CustomGate>) {
        self.gates.insert(name.into(), gate);
    }

    /// Retrieve a registered gate
    ///
    /// Returns a clone of the Arc pointer, allowing multiple concurrent accesses.
    pub fn get(&self, name: &str) -> Option<Arc<CustomGate>> {
        self.gates.get(name).cloned()
    }

    /// Check if a gate is registered
    pub fn contains(&self, name: &str) -> bool {
        self.gates.contains_key(name)
    }

    /// Get all registered gate names
    pub fn gate_names(&self) -> Vec<&str> {
        self.gates.keys().map(|s| s.as_str()).collect()
    }

    /// Remove a registered gate
    pub fn unregister(&mut self, name: &str) -> Option<Arc<CustomGate>> {
        self.gates.remove(name)
    }

    /// Clear all registered gates
    pub fn clear(&mut self) {
        self.gates.clear();
    }

    /// Get the number of registered gates
    pub fn len(&self) -> usize {
        self.gates.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    /// Get gates for a specific qubit count
    ///
    /// Returns all registered gates that operate on exactly `num_qubits` qubits.
    pub fn gates_for_qubits(&self, num_qubits: usize) -> Vec<(&str, Arc<CustomGate>)> {
        self.gates
            .iter()
            .filter(|(_, gate)| gate.num_qubits() == num_qubits)
            .map(|(name, gate)| (name.as_str(), gate.clone()))
            .collect()
    }

    /// List gates with detailed information
    pub fn list_gates(&self) -> Vec<GateInfo> {
        self.gates
            .iter()
            .map(|(name, gate)| GateInfo {
                name: name.clone(),
                num_qubits: gate.num_qubits(),
                is_hermitian: gate.is_hermitian(),
                description: gate.description(),
            })
            .collect()
    }

    /// Print formatted gate list
    pub fn print_gates(&self) {
        if self.gates.is_empty() {
            println!("Gate registry is empty");
            return;
        }

        println!("Registered Custom Gates:");
        println!("{:<20} {:<10} {:<12} {}", "Name", "Qubits", "Hermitian", "Description");
        println!("{}", "-".repeat(70));

        for info in self.list_gates() {
            println!(
                "{:<20} {:<10} {:<12} {}",
                info.name,
                info.num_qubits,
                if info.is_hermitian { "Yes" } else { "No" },
                info.description
            );
        }
    }
}

impl Default for GateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a registered gate
#[derive(Debug, Clone)]
pub struct GateInfo {
    pub name: String,
    pub num_qubits: usize,
    pub is_hermitian: bool,
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::CustomGateBuilder;
    use num_complex::Complex64;
    use std::f64::consts::SQRT_2;

    #[test]
    fn test_registry_creation() {
        let registry = GateRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_register_and_retrieve() {
        let mut registry = GateRegistry::new();

        let inv_sqrt2 = 1.0 / SQRT_2;
        let gate = CustomGateBuilder::new("TestH")
            .matrix_2x2([
                [Complex64::new(inv_sqrt2, 0.0), Complex64::new(inv_sqrt2, 0.0)],
                [Complex64::new(inv_sqrt2, 0.0), Complex64::new(-inv_sqrt2, 0.0)],
            ])
            .build()
            .unwrap();

        registry.register("my_h", gate);
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("my_h"));
        assert!(registry.get("my_h").is_some());
    }

    #[test]
    fn test_unregister() {
        let mut registry = GateRegistry::new();

        let gate = CustomGateBuilder::new("X")
            .matrix_2x2([
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            ])
            .build()
            .unwrap();

        registry.register("pauli_x", gate);
        assert_eq!(registry.len(), 1);

        let removed = registry.unregister("pauli_x");
        assert!(removed.is_some());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_gates_for_qubits() {
        let mut registry = GateRegistry::new();

        // Register 1-qubit gates
        for i in 0..3 {
            let gate = CustomGateBuilder::new(format!("1Q_{}", i))
                .matrix_2x2([
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
                ])
                .build()
                .unwrap();
            registry.register(format!("gate_{}", i), gate);
        }

        let one_qubit_gates = registry.gates_for_qubits(1);
        assert_eq!(one_qubit_gates.len(), 3);
    }

    #[test]
    fn test_clear() {
        let mut registry = GateRegistry::with_capacity(10);

        for i in 0..5 {
            let gate = CustomGateBuilder::new(format!("Gate{}", i))
                .matrix_2x2([
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
                ])
                .build()
                .unwrap();
            registry.register(format!("gate_{}", i), gate);
        }

        assert_eq!(registry.len(), 5);
        registry.clear();
        assert!(registry.is_empty());
    }
}
