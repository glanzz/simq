//! Hardware-aware compilation
//!
//! This module provides traits and implementations for compiling circuits
//! optimized for specific quantum hardware platforms.

use simq_core::{gate::Gate, Circuit};
use std::collections::HashSet;

/// Hardware model trait
///
/// Represents a quantum hardware platform with specific characteristics
/// like gate costs, connectivity, and native gate sets.
pub trait HardwareModel: Send + Sync {
    /// Name of the hardware platform
    fn name(&self) -> &str;

    /// Cost of executing a gate (in arbitrary units)
    ///
    /// Higher cost indicates more expensive operations (time, error rate, etc.)
    fn gate_cost(&self, gate: &dyn Gate) -> f64;

    /// Native gates supported by this hardware
    ///
    /// Gates in this set can be executed directly without decomposition.
    fn native_gates(&self) -> &HashSet<String>;

    /// Whether a gate is natively supported
    fn is_native(&self, gate_name: &str) -> bool {
        self.native_gates().contains(gate_name)
    }

    /// Estimate total circuit cost
    fn circuit_cost(&self, circuit: &Circuit) -> f64 {
        circuit
            .operations()
            .map(|op| {
                let gate = op.gate();
                self.gate_cost(&**gate)
            })
            .sum()
    }

    /// Relative benefit score for using this hardware model
    ///
    /// Used to prioritize passes that reduce the most expensive operations.
    fn optimization_benefit(&self, gate_name: &str) -> f64 {
        // Higher benefit for reducing more expensive gates
        let cost = self.gate_cost_by_name(gate_name);
        cost / 10.0 // Normalize to 0.0-1.0 range
    }

    /// Get gate cost by name (helper method)
    fn gate_cost_by_name(&self, gate_name: &str) -> f64;
}

/// IBM Quantum hardware model
///
/// Represents IBM's quantum processors with their characteristic
/// gate set and cost model.
pub struct IBMHardware {
    native: HashSet<String>,
}

impl IBMHardware {
    pub fn new() -> Self {
        let mut native = HashSet::new();
        // IBM native gates
        native.insert("ID".to_string());
        native.insert("RZ".to_string());
        native.insert("SX".to_string()); // sqrt(X)
        native.insert("X".to_string());
        native.insert("CX".to_string()); // CNOT
        native.insert("CNOT".to_string());

        Self { native }
    }
}

impl Default for IBMHardware {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareModel for IBMHardware {
    fn name(&self) -> &str {
        "IBM Quantum"
    }

    fn gate_cost(&self, gate: &dyn Gate) -> f64 {
        self.gate_cost_by_name(gate.name())
    }

    fn gate_cost_by_name(&self, gate_name: &str) -> f64 {
        match gate_name {
            // Single-qubit gates (cheap)
            "ID" | "I" => 0.0,
            "RZ" | "P" | "U1" => 0.1, // Virtual Z rotations (very cheap)
            "SX" | "X" => 1.0,        // Physical single-qubit gates
            "H" | "Y" | "Z" => 1.0,
            "S" | "T" | "S†" | "T†" => 1.0,
            "RX" | "RY" => 1.5, // Require decomposition

            // Two-qubit gates (expensive)
            "CX" | "CNOT" => 10.0, // Most expensive, high error rate
            "CZ" => 11.0,          // Requires decomposition on IBM
            "SWAP" => 30.0,        // 3 CNOTs
            "iSWAP" => 30.0,

            // Three-qubit gates (very expensive)
            "CCX" | "Toffoli" => 50.0,
            "CSWAP" | "Fredkin" => 60.0,

            // Default
            _ => 5.0,
        }
    }

    fn native_gates(&self) -> &HashSet<String> {
        &self.native
    }
}

/// Google Sycamore hardware model
pub struct GoogleHardware {
    native: HashSet<String>,
}

impl GoogleHardware {
    pub fn new() -> Self {
        let mut native = HashSet::new();
        // Google native gates
        native.insert("PhasedXZ".to_string());
        native.insert("RZ".to_string());
        native.insert("RX".to_string());
        native.insert("RY".to_string());
        native.insert("CZ".to_string());
        native.insert("iSWAP".to_string());
        native.insert("SWAP".to_string());

        Self { native }
    }
}

impl Default for GoogleHardware {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareModel for GoogleHardware {
    fn name(&self) -> &str {
        "Google Sycamore"
    }

    fn gate_cost(&self, gate: &dyn Gate) -> f64 {
        self.gate_cost_by_name(gate.name())
    }

    fn gate_cost_by_name(&self, gate_name: &str) -> f64 {
        match gate_name {
            // Single-qubit gates
            "ID" | "I" => 0.0,
            "RZ" | "P" | "U1" => 0.1,
            "RX" | "RY" => 1.0,
            "PhasedXZ" => 1.0,
            "X" | "Y" | "Z" | "H" => 1.0,
            "S" | "T" | "S†" | "T†" => 1.0,

            // Two-qubit gates
            "CZ" => 8.0,      // Native, cheaper than CNOT
            "iSWAP" => 8.0,   // Native
            "SWAP" => 9.0,    // Native or low-cost
            "CNOT" | "CX" => 10.0, // Requires decomposition

            // Three-qubit gates
            "CCX" | "Toffoli" => 45.0,
            "CSWAP" | "Fredkin" => 55.0,

            // Default
            _ => 5.0,
        }
    }

    fn native_gates(&self) -> &HashSet<String> {
        &self.native
    }
}

/// IonQ trapped-ion hardware model
pub struct IonQHardware {
    native: HashSet<String>,
}

impl IonQHardware {
    pub fn new() -> Self {
        let mut native = HashSet::new();
        // IonQ has all-to-all connectivity and arbitrary single-qubit rotations
        native.insert("RX".to_string());
        native.insert("RY".to_string());
        native.insert("RZ".to_string());
        native.insert("GPI".to_string());
        native.insert("GPI2".to_string());
        native.insert("MS".to_string()); // Mølmer-Sørensen gate
        native.insert("XX".to_string());

        Self { native }
    }
}

impl Default for IonQHardware {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareModel for IonQHardware {
    fn name(&self) -> &str {
        "IonQ Trapped-Ion"
    }

    fn gate_cost(&self, gate: &dyn Gate) -> f64 {
        self.gate_cost_by_name(gate.name())
    }

    fn gate_cost_by_name(&self, gate_name: &str) -> f64 {
        match gate_name {
            // Single-qubit gates (very cheap)
            "ID" | "I" => 0.0,
            "RX" | "RY" | "RZ" => 0.5,
            "GPI" | "GPI2" => 0.5,
            "X" | "Y" | "Z" | "H" => 0.5,
            "S" | "T" | "S†" | "T†" => 0.5,
            "P" | "U1" => 0.1,

            // Two-qubit gates (all-to-all connectivity)
            "MS" => 5.0,          // Native
            "XX" => 5.0,          // Native
            "CNOT" | "CX" => 6.0, // Decomposed from XX
            "CZ" => 6.0,
            "SWAP" => 5.0, // Can be done efficiently
            "iSWAP" => 6.0,

            // Three-qubit gates
            "CCX" | "Toffoli" => 25.0,
            "CSWAP" | "Fredkin" => 30.0,

            // Default
            _ => 3.0,
        }
    }

    fn native_gates(&self) -> &HashSet<String> {
        &self.native
    }
}

/// Cost-based optimization configuration
#[derive(Clone)]
pub struct CostModel {
    hardware: HardwareType,
    /// Penalty factor for non-native gates
    pub decomposition_penalty: f64,
}

impl CostModel {
    /// Create a new cost model for the specified hardware
    pub fn new(hardware: HardwareType) -> Self {
        Self {
            hardware,
            decomposition_penalty: 2.0,
        }
    }

    /// Get the hardware model
    pub fn hardware_model(&self) -> Box<dyn HardwareModel> {
        match self.hardware {
            HardwareType::IBM => Box::new(IBMHardware::new()),
            HardwareType::Google => Box::new(GoogleHardware::new()),
            HardwareType::IonQ => Box::new(IonQHardware::new()),
        }
    }

    /// Calculate the cost of a circuit for this hardware
    pub fn circuit_cost(&self, circuit: &Circuit) -> f64 {
        let model = self.hardware_model();
        let mut total_cost = 0.0;

        for op in circuit.operations() {
            let gate = op.gate();
            let base_cost = model.gate_cost(&**gate);

            // Apply decomposition penalty for non-native gates
            let cost = if model.is_native(gate.name()) {
                base_cost
            } else {
                base_cost * self.decomposition_penalty
            };

            total_cost += cost;
        }

        total_cost
    }
}

/// Hardware type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareType {
    IBM,
    Google,
    IonQ,
}

impl HardwareType {
    /// Get the name of the hardware platform
    pub fn name(&self) -> &str {
        match self {
            HardwareType::IBM => "IBM Quantum",
            HardwareType::Google => "Google Sycamore",
            HardwareType::IonQ => "IonQ Trapped-Ion",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ibm_hardware_costs() {
        let ibm = IBMHardware::new();

        // Single-qubit gates should be cheap
        assert!(ibm.gate_cost_by_name("RZ") < 1.0);
        assert!(ibm.gate_cost_by_name("X") < 2.0);

        // Two-qubit gates should be expensive
        assert!(ibm.gate_cost_by_name("CNOT") > 5.0);
        assert!(ibm.gate_cost_by_name("CZ") > ibm.gate_cost_by_name("CNOT"));
    }

    #[test]
    fn test_google_hardware_costs() {
        let google = GoogleHardware::new();

        // CZ should be cheaper than CNOT on Google hardware
        assert!(google.gate_cost_by_name("CZ") < google.gate_cost_by_name("CNOT"));
    }

    #[test]
    fn test_ionq_hardware_costs() {
        let ionq = IonQHardware::new();

        // IonQ should have cheap single-qubit gates
        assert!(ionq.gate_cost_by_name("RX") < 1.0);

        // All two-qubit gates should be similar cost (all-to-all)
        let cnot_cost = ionq.gate_cost_by_name("CNOT");
        let cz_cost = ionq.gate_cost_by_name("CZ");
        assert!((cnot_cost - cz_cost).abs() < 2.0);
    }

    #[test]
    fn test_native_gates() {
        let ibm = IBMHardware::new();
        assert!(ibm.is_native("CNOT"));
        assert!(ibm.is_native("RZ"));
        assert!(!ibm.is_native("CZ"));

        let google = GoogleHardware::new();
        assert!(google.is_native("CZ"));
        assert!(!google.is_native("CNOT"));
    }

    #[test]
    fn test_cost_model() {
        let cost_model = CostModel::new(HardwareType::IBM);
        let circuit = Circuit::new(2);

        let cost = cost_model.circuit_cost(&circuit);
        assert_eq!(cost, 0.0); // Empty circuit
    }

    #[test]
    fn test_hardware_type_names() {
        assert_eq!(HardwareType::IBM.name(), "IBM Quantum");
        assert_eq!(HardwareType::Google.name(), "Google Sycamore");
        assert_eq!(HardwareType::IonQ.name(), "IonQ Trapped-Ion");
    }
}
