//! Basis gate set definitions
//!
//! This module defines common basis gate sets used by quantum hardware vendors
//! and quantum computing frameworks. Each platform has a native set of gates
//! that can be executed directly, and all other gates must be decomposed.
//!
//! # Common Basis Sets
//!
//! - **IBM**: {U3, U2, U1, CNOT} or {RZ, SX, X, CNOT}
//! - **Google/Sycamore**: {√iSWAP, SYC, PhasedXZ}
//! - **IonQ**: {GPI, GPI2, MS} (trapped ion gates)
//! - **Rigetti**: {RZ, RX, CZ}
//! - **Clifford+T**: {H, S, CNOT, T} (fault-tolerant)
//! - **Universal**: {H, T, CNOT} (theoretically universal)
//!
//! # References
//!
//! - IBM Quantum Backend Specifications
//! - Google Cirq Documentation
//! - IonQ Native Gates API
//! - Rigetti Quil Specification

use std::collections::HashSet;

/// Enumeration of standard quantum gate basis sets
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BasisGateSet {
    /// IBM Quantum basis: {U3, U2, U1, CNOT}
    IBM,

    /// IBM Quantum newer basis: {RZ, SX, X, CNOT}
    IBMQiskit,

    /// Google Sycamore basis: {√iSWAP, SYC, PhasedXZ}
    Google,

    /// Google Cirq basis: {PhasedXZ, FSIM, CZ}
    GoogleCirq,

    /// IonQ trapped ion basis: {GPI, GPI2, MS}
    IonQ,

    /// Rigetti basis: {RZ, RX, CZ}
    Rigetti,

    /// Amazon Braket basis: {Rx, Ry, Rz, CNOT, CZ}
    Braket,

    /// Clifford+T basis: {H, S, T, CNOT} (fault-tolerant)
    CliffordT,

    /// Strict Clifford+T: {H, S, T, CNOT} with exact synthesis
    StrictCliffordT,

    /// Universal basis: {H, T, CNOT} (theoretically minimal)
    Universal,

    /// Rotation basis: {RX, RY, RZ, CNOT}
    Rotation,

    /// Pauli basis: {X, Y, Z, H, S, CNOT}
    Pauli,

    /// All gates (no decomposition)
    All,

    /// Custom basis (user-defined)
    Custom(CustomBasis),
}

/// Custom basis gate set
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomBasis {
    /// Single-qubit gates in the basis
    pub single_qubit: HashSet<BasisGate>,

    /// Two-qubit gates in the basis
    pub two_qubit: HashSet<BasisGate>,

    /// Multi-qubit gates in the basis
    pub multi_qubit: HashSet<BasisGate>,
}

impl CustomBasis {
    /// Create a new custom basis
    pub fn new() -> Self {
        Self {
            single_qubit: HashSet::new(),
            two_qubit: HashSet::new(),
            multi_qubit: HashSet::new(),
        }
    }

    /// Add a single-qubit gate to the basis
    pub fn add_single_qubit(&mut self, gate: BasisGate) {
        self.single_qubit.insert(gate);
    }

    /// Add a two-qubit gate to the basis
    pub fn add_two_qubit(&mut self, gate: BasisGate) {
        self.two_qubit.insert(gate);
    }

    /// Add a multi-qubit gate to the basis
    pub fn add_multi_qubit(&mut self, gate: BasisGate) {
        self.multi_qubit.insert(gate);
    }

    /// Check if a gate is in the basis
    pub fn contains(&self, gate: &BasisGate) -> bool {
        self.single_qubit.contains(gate)
            || self.two_qubit.contains(gate)
            || self.multi_qubit.contains(gate)
    }
}

impl Default for CustomBasis {
    fn default() -> Self {
        Self::new()
    }
}

/// Individual basis gates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasisGate {
    // ========================================
    // Single-Qubit Gates
    // ========================================
    /// Identity gate
    I,

    /// Pauli-X (NOT)
    X,

    /// Pauli-Y
    Y,

    /// Pauli-Z
    Z,

    /// Hadamard
    H,

    /// S gate (√Z)
    S,

    /// S† gate
    SDagger,

    /// T gate (√S)
    T,

    /// T† gate
    TDagger,

    /// √X gate
    SX,

    /// √X† gate
    SXDagger,

    /// √Y gate
    SY,

    /// √Y† gate
    SYDagger,

    /// Rotation around X-axis
    RX,

    /// Rotation around Y-axis
    RY,

    /// Rotation around Z-axis
    RZ,

    /// Phase gate
    Phase,

    /// Universal single-qubit gate (IBM)
    U1,

    /// Universal single-qubit gate (IBM)
    U2,

    /// Universal single-qubit gate (IBM)
    U3,

    /// Phased X-Z gate (Google)
    PhasedXZ,

    /// GPI gate (IonQ)
    GPI,

    /// GPI2 gate (IonQ)
    GPI2,

    // ========================================
    // Two-Qubit Gates
    // ========================================
    /// Controlled-NOT
    CNOT,

    /// Controlled-Y
    CY,

    /// Controlled-Z
    CZ,

    /// Controlled-Phase
    CPhase,

    /// SWAP
    SWAP,

    /// iSWAP
    ISWAP,

    /// √iSWAP
    SqrtISWAP,

    /// fSim gate (Google)
    FSIM,

    /// Sycamore gate (Google)
    SYC,

    /// Mølmer-Sørensen gate (IonQ)
    MS,

    /// RXX gate
    RXX,

    /// RYY gate
    RYY,

    /// RZZ gate
    RZZ,

    /// ECR (Echoed Cross-Resonance)
    ECR,

    // ========================================
    // Multi-Qubit Gates
    // ========================================
    /// Toffoli (CCNOT)
    Toffoli,

    /// Fredkin (CSWAP)
    Fredkin,

    /// Multi-controlled X
    MCX,

    /// Multi-controlled Z
    MCZ,
}

impl BasisGateSet {
    /// Get the gates in this basis set
    pub fn gates(&self) -> Vec<BasisGate> {
        match self {
            BasisGateSet::IBM => vec![
                BasisGate::U1,
                BasisGate::U2,
                BasisGate::U3,
                BasisGate::CNOT,
            ],

            BasisGateSet::IBMQiskit => vec![
                BasisGate::RZ,
                BasisGate::SX,
                BasisGate::X,
                BasisGate::CNOT,
            ],

            BasisGateSet::Google => vec![
                BasisGate::PhasedXZ,
                BasisGate::SqrtISWAP,
                BasisGate::SYC,
            ],

            BasisGateSet::GoogleCirq => vec![
                BasisGate::PhasedXZ,
                BasisGate::FSIM,
                BasisGate::CZ,
            ],

            BasisGateSet::IonQ => vec![
                BasisGate::GPI,
                BasisGate::GPI2,
                BasisGate::MS,
            ],

            BasisGateSet::Rigetti => vec![
                BasisGate::RZ,
                BasisGate::RX,
                BasisGate::CZ,
            ],

            BasisGateSet::Braket => vec![
                BasisGate::RX,
                BasisGate::RY,
                BasisGate::RZ,
                BasisGate::CNOT,
                BasisGate::CZ,
            ],

            BasisGateSet::CliffordT | BasisGateSet::StrictCliffordT => vec![
                BasisGate::H,
                BasisGate::S,
                BasisGate::T,
                BasisGate::CNOT,
            ],

            BasisGateSet::Universal => vec![
                BasisGate::H,
                BasisGate::T,
                BasisGate::CNOT,
            ],

            BasisGateSet::Rotation => vec![
                BasisGate::RX,
                BasisGate::RY,
                BasisGate::RZ,
                BasisGate::CNOT,
            ],

            BasisGateSet::Pauli => vec![
                BasisGate::X,
                BasisGate::Y,
                BasisGate::Z,
                BasisGate::H,
                BasisGate::S,
                BasisGate::CNOT,
            ],

            BasisGateSet::All => vec![],  // All gates allowed

            BasisGateSet::Custom(custom) => {
                let mut gates = Vec::new();
                gates.extend(custom.single_qubit.iter().copied());
                gates.extend(custom.two_qubit.iter().copied());
                gates.extend(custom.multi_qubit.iter().copied());
                gates
            }
        }
    }

    /// Check if a specific gate is in this basis
    pub fn contains(&self, gate: BasisGate) -> bool {
        if matches!(self, BasisGateSet::All) {
            return true;
        }

        if let BasisGateSet::Custom(custom) = self {
            return custom.contains(&gate);
        }

        self.gates().contains(&gate)
    }

    /// Get the single-qubit gates in this basis
    pub fn single_qubit_gates(&self) -> Vec<BasisGate> {
        self.gates()
            .into_iter()
            .filter(|g| matches!(
                g,
                BasisGate::I | BasisGate::X | BasisGate::Y | BasisGate::Z |
                BasisGate::H | BasisGate::S | BasisGate::SDagger |
                BasisGate::T | BasisGate::TDagger |
                BasisGate::SX | BasisGate::SXDagger |
                BasisGate::SY | BasisGate::SYDagger |
                BasisGate::RX | BasisGate::RY | BasisGate::RZ |
                BasisGate::Phase | BasisGate::U1 | BasisGate::U2 | BasisGate::U3 |
                BasisGate::PhasedXZ | BasisGate::GPI | BasisGate::GPI2
            ))
            .collect()
    }

    /// Get the two-qubit gates in this basis
    pub fn two_qubit_gates(&self) -> Vec<BasisGate> {
        self.gates()
            .into_iter()
            .filter(|g| matches!(
                g,
                BasisGate::CNOT | BasisGate::CY | BasisGate::CZ |
                BasisGate::CPhase | BasisGate::SWAP | BasisGate::ISWAP |
                BasisGate::SqrtISWAP | BasisGate::FSIM | BasisGate::SYC |
                BasisGate::MS | BasisGate::RXX | BasisGate::RYY | BasisGate::RZZ |
                BasisGate::ECR
            ))
            .collect()
    }

    /// Get the primary entangling gate for this basis
    pub fn entangling_gate(&self) -> Option<BasisGate> {
        match self {
            BasisGateSet::IBM | BasisGateSet::IBMQiskit => Some(BasisGate::CNOT),
            BasisGateSet::Google => Some(BasisGate::SqrtISWAP),
            BasisGateSet::GoogleCirq => Some(BasisGate::CZ),
            BasisGateSet::IonQ => Some(BasisGate::MS),
            BasisGateSet::Rigetti => Some(BasisGate::CZ),
            BasisGateSet::Braket => Some(BasisGate::CNOT),
            BasisGateSet::CliffordT | BasisGateSet::StrictCliffordT => Some(BasisGate::CNOT),
            BasisGateSet::Universal => Some(BasisGate::CNOT),
            BasisGateSet::Rotation => Some(BasisGate::CNOT),
            BasisGateSet::Pauli => Some(BasisGate::CNOT),
            BasisGateSet::All => None,
            BasisGateSet::Custom(custom) => {
                // Return first two-qubit gate
                custom.two_qubit.iter().next().copied()
            }
        }
    }

    /// Check if this basis supports native rotation gates
    pub fn has_rotation_gates(&self) -> bool {
        matches!(
            self,
            BasisGateSet::IBMQiskit
                | BasisGateSet::Rigetti
                | BasisGateSet::Braket
                | BasisGateSet::Rotation
        )
    }

    /// Check if this is a discrete gate basis (Clifford+T)
    pub fn is_discrete(&self) -> bool {
        matches!(
            self,
            BasisGateSet::CliffordT | BasisGateSet::StrictCliffordT | BasisGateSet::Universal
        )
    }

    /// Get a human-readable description of this basis
    pub fn description(&self) -> &str {
        match self {
            BasisGateSet::IBM => "IBM Quantum (U1, U2, U3, CNOT)",
            BasisGateSet::IBMQiskit => "IBM Qiskit (RZ, SX, X, CNOT)",
            BasisGateSet::Google => "Google Sycamore (PhasedXZ, √iSWAP, SYC)",
            BasisGateSet::GoogleCirq => "Google Cirq (PhasedXZ, FSIM, CZ)",
            BasisGateSet::IonQ => "IonQ Trapped Ion (GPI, GPI2, MS)",
            BasisGateSet::Rigetti => "Rigetti (RZ, RX, CZ)",
            BasisGateSet::Braket => "Amazon Braket (Rx, Ry, Rz, CNOT, CZ)",
            BasisGateSet::CliffordT => "Clifford+T (H, S, T, CNOT)",
            BasisGateSet::StrictCliffordT => "Strict Clifford+T (exact synthesis)",
            BasisGateSet::Universal => "Universal (H, T, CNOT)",
            BasisGateSet::Rotation => "Rotation basis (RX, RY, RZ, CNOT)",
            BasisGateSet::Pauli => "Pauli basis (X, Y, Z, H, S, CNOT)",
            BasisGateSet::All => "All gates (no decomposition)",
            BasisGateSet::Custom(_) => "Custom basis",
        }
    }
}

impl BasisGate {
    /// Get the number of qubits this gate acts on
    pub fn num_qubits(&self) -> usize {
        match self {
            // Single-qubit
            BasisGate::I | BasisGate::X | BasisGate::Y | BasisGate::Z |
            BasisGate::H | BasisGate::S | BasisGate::SDagger |
            BasisGate::T | BasisGate::TDagger |
            BasisGate::SX | BasisGate::SXDagger |
            BasisGate::SY | BasisGate::SYDagger |
            BasisGate::RX | BasisGate::RY | BasisGate::RZ |
            BasisGate::Phase | BasisGate::U1 | BasisGate::U2 | BasisGate::U3 |
            BasisGate::PhasedXZ | BasisGate::GPI | BasisGate::GPI2 => 1,

            // Two-qubit
            BasisGate::CNOT | BasisGate::CY | BasisGate::CZ |
            BasisGate::CPhase | BasisGate::SWAP | BasisGate::ISWAP |
            BasisGate::SqrtISWAP | BasisGate::FSIM | BasisGate::SYC |
            BasisGate::MS | BasisGate::RXX | BasisGate::RYY | BasisGate::RZZ |
            BasisGate::ECR => 2,

            // Multi-qubit
            BasisGate::Toffoli => 3,
            BasisGate::Fredkin => 3,
            BasisGate::MCX | BasisGate::MCZ => 2,  // Minimum, can be more
        }
    }

    /// Check if this is a parameterized gate
    pub fn is_parameterized(&self) -> bool {
        matches!(
            self,
            BasisGate::RX | BasisGate::RY | BasisGate::RZ |
            BasisGate::Phase | BasisGate::U1 | BasisGate::U2 | BasisGate::U3 |
            BasisGate::CPhase | BasisGate::RXX | BasisGate::RYY | BasisGate::RZZ |
            BasisGate::PhasedXZ | BasisGate::FSIM | BasisGate::GPI | BasisGate::GPI2
        )
    }

    /// Check if this is a Clifford gate
    pub fn is_clifford(&self) -> bool {
        matches!(
            self,
            BasisGate::I | BasisGate::X | BasisGate::Y | BasisGate::Z |
            BasisGate::H | BasisGate::S | BasisGate::SDagger |
            BasisGate::SX | BasisGate::SXDagger |
            BasisGate::CNOT | BasisGate::CY | BasisGate::CZ | BasisGate::SWAP
        )
    }

    /// Get the name of this gate
    pub fn name(&self) -> &str {
        match self {
            BasisGate::I => "I",
            BasisGate::X => "X",
            BasisGate::Y => "Y",
            BasisGate::Z => "Z",
            BasisGate::H => "H",
            BasisGate::S => "S",
            BasisGate::SDagger => "S†",
            BasisGate::T => "T",
            BasisGate::TDagger => "T†",
            BasisGate::SX => "SX",
            BasisGate::SXDagger => "SX†",
            BasisGate::SY => "SY",
            BasisGate::SYDagger => "SY†",
            BasisGate::RX => "RX",
            BasisGate::RY => "RY",
            BasisGate::RZ => "RZ",
            BasisGate::Phase => "P",
            BasisGate::U1 => "U1",
            BasisGate::U2 => "U2",
            BasisGate::U3 => "U3",
            BasisGate::PhasedXZ => "PhasedXZ",
            BasisGate::GPI => "GPI",
            BasisGate::GPI2 => "GPI2",
            BasisGate::CNOT => "CNOT",
            BasisGate::CY => "CY",
            BasisGate::CZ => "CZ",
            BasisGate::CPhase => "CPhase",
            BasisGate::SWAP => "SWAP",
            BasisGate::ISWAP => "iSWAP",
            BasisGate::SqrtISWAP => "√iSWAP",
            BasisGate::FSIM => "FSIM",
            BasisGate::SYC => "SYC",
            BasisGate::MS => "MS",
            BasisGate::RXX => "RXX",
            BasisGate::RYY => "RYY",
            BasisGate::RZZ => "RZZ",
            BasisGate::ECR => "ECR",
            BasisGate::Toffoli => "Toffoli",
            BasisGate::Fredkin => "Fredkin",
            BasisGate::MCX => "MCX",
            BasisGate::MCZ => "MCZ",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ibm_basis() {
        let basis = BasisGateSet::IBM;
        assert!(basis.contains(BasisGate::U3));
        assert!(basis.contains(BasisGate::CNOT));
        assert!(!basis.contains(BasisGate::T));
    }

    #[test]
    fn test_clifford_t_basis() {
        let basis = BasisGateSet::CliffordT;
        assert!(basis.contains(BasisGate::H));
        assert!(basis.contains(BasisGate::S));
        assert!(basis.contains(BasisGate::T));
        assert!(basis.contains(BasisGate::CNOT));
        assert!(!basis.contains(BasisGate::RZ));
    }

    #[test]
    fn test_custom_basis() {
        let mut custom = CustomBasis::new();
        custom.add_single_qubit(BasisGate::H);
        custom.add_single_qubit(BasisGate::T);
        custom.add_two_qubit(BasisGate::CNOT);

        let basis = BasisGateSet::Custom(custom);
        assert!(basis.contains(BasisGate::H));
        assert!(basis.contains(BasisGate::T));
        assert!(basis.contains(BasisGate::CNOT));
        assert!(!basis.contains(BasisGate::RX));
    }

    #[test]
    fn test_entangling_gates() {
        assert_eq!(BasisGateSet::IBM.entangling_gate(), Some(BasisGate::CNOT));
        assert_eq!(BasisGateSet::Rigetti.entangling_gate(), Some(BasisGate::CZ));
        assert_eq!(BasisGateSet::IonQ.entangling_gate(), Some(BasisGate::MS));
    }

    #[test]
    fn test_gate_properties() {
        assert_eq!(BasisGate::CNOT.num_qubits(), 2);
        assert_eq!(BasisGate::H.num_qubits(), 1);
        assert_eq!(BasisGate::Toffoli.num_qubits(), 3);

        assert!(BasisGate::RX.is_parameterized());
        assert!(!BasisGate::H.is_parameterized());

        assert!(BasisGate::H.is_clifford());
        assert!(!BasisGate::T.is_clifford());
    }

    #[test]
    fn test_discrete_basis() {
        assert!(BasisGateSet::CliffordT.is_discrete());
        assert!(!BasisGateSet::IBM.is_discrete());
    }
}
