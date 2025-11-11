//! Standard quantum gate implementations with pre-computed matrices

use num_complex::Complex64;
use simq_core::gate::Gate;
use crate::matrices;

/// Helper macro to implement the matrix() method for gates
macro_rules! impl_matrix_method {
    ($gate_type:ty, $matrix_fn:expr, $size:expr) => {
        impl $gate_type {
            /// Returns the pre-computed gate matrix
            #[inline]
            pub const fn matrix() -> &'static [[Complex64; $size]; $size] {
                $matrix_fn
            }

            /// Returns the matrix as a flattened vector (for Gate trait)
            #[inline]
            fn matrix_vec() -> Vec<Complex64> {
                Self::matrix().iter().flatten().copied().collect()
            }
        }
    };
}

// ============================================================================
// Single-Qubit Gates
// ============================================================================

/// Hadamard gate
///
/// Creates superposition: H|0⟩ = (|0⟩ + |1⟩)/√2
#[derive(Debug, Clone, Copy)]
pub struct Hadamard;

impl Gate for Hadamard {
    fn name(&self) -> &str {
        "H"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn is_hermitian(&self) -> bool {
        true
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(Hadamard, &matrices::HADAMARD, 2);

/// Pauli-X gate (NOT gate)
///
/// Bit flip: X|0⟩ = |1⟩, X|1⟩ = |0⟩
#[derive(Debug, Clone, Copy)]
pub struct PauliX;

impl Gate for PauliX {
    fn name(&self) -> &str {
        "X"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn is_hermitian(&self) -> bool {
        true
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(PauliX, &matrices::PAULI_X, 2);

/// Pauli-Y gate
///
/// Combined bit and phase flip
#[derive(Debug, Clone, Copy)]
pub struct PauliY;

impl Gate for PauliY {
    fn name(&self) -> &str {
        "Y"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn is_hermitian(&self) -> bool {
        true
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(PauliY, &matrices::PAULI_Y, 2);

/// Pauli-Z gate
///
/// Phase flip: Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
#[derive(Debug, Clone, Copy)]
pub struct PauliZ;

impl Gate for PauliZ {
    fn name(&self) -> &str {
        "Z"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn is_hermitian(&self) -> bool {
        true
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(PauliZ, &matrices::PAULI_Z, 2);

/// S gate (Phase gate, √Z)
///
/// Applies a 90° phase rotation
#[derive(Debug, Clone, Copy)]
pub struct SGate;

impl Gate for SGate {
    fn name(&self) -> &str {
        "S"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(SGate, &matrices::S_GATE, 2);

/// S† gate (adjoint of S gate)
///
/// Applies a -90° phase rotation
#[derive(Debug, Clone, Copy)]
pub struct SGateDagger;

impl Gate for SGateDagger {
    fn name(&self) -> &str {
        "S†"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(SGateDagger, &matrices::S_GATE_DAGGER, 2);

/// T gate (π/8 gate, √S)
///
/// Applies a 45° phase rotation
#[derive(Debug, Clone, Copy)]
pub struct TGate;

impl Gate for TGate {
    fn name(&self) -> &str {
        "T"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(TGate, &matrices::T_GATE, 2);

/// T† gate (adjoint of T gate)
///
/// Applies a -45° phase rotation
#[derive(Debug, Clone, Copy)]
pub struct TGateDagger;

impl Gate for TGateDagger {
    fn name(&self) -> &str {
        "T†"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(TGateDagger, &matrices::T_GATE_DAGGER, 2);

/// Identity gate
///
/// No-op gate, useful for circuit padding
#[derive(Debug, Clone, Copy)]
pub struct Identity;

impl Gate for Identity {
    fn name(&self) -> &str {
        "I"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn is_hermitian(&self) -> bool {
        true
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(Identity, &matrices::IDENTITY, 2);

// ============================================================================
// Two-Qubit Gates
// ============================================================================

/// CNOT gate (Controlled-NOT)
///
/// Flips target qubit if control qubit is |1⟩
#[derive(Debug, Clone, Copy)]
pub struct CNot;

impl Gate for CNot {
    fn name(&self) -> &str {
        "CNOT"
    }

    fn num_qubits(&self) -> usize {
        2
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(CNot, &matrices::CNOT, 4);

/// CZ gate (Controlled-Z)
///
/// Applies Z gate to target if control qubit is |1⟩
#[derive(Debug, Clone, Copy)]
pub struct CZ;

impl Gate for CZ {
    fn name(&self) -> &str {
        "CZ"
    }

    fn num_qubits(&self) -> usize {
        2
    }

    fn is_hermitian(&self) -> bool {
        true
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(CZ, &matrices::CZ, 4);

/// SWAP gate
///
/// Swaps the states of two qubits
#[derive(Debug, Clone, Copy)]
pub struct Swap;

impl Gate for Swap {
    fn name(&self) -> &str {
        "SWAP"
    }

    fn num_qubits(&self) -> usize {
        2
    }

    fn is_hermitian(&self) -> bool {
        true
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(Swap, &matrices::SWAP, 4);

/// iSWAP gate
///
/// Swaps two qubits and applies a phase
#[derive(Debug, Clone, Copy)]
pub struct ISwap;

impl Gate for ISwap {
    fn name(&self) -> &str {
        "iSWAP"
    }

    fn num_qubits(&self) -> usize {
        2
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(ISwap, &matrices::ISWAP, 4);

// ============================================================================
// Parameterized Gates
// ============================================================================

/// Rotation-X gate
///
/// Rotates around X-axis by angle θ
#[derive(Debug, Clone, Copy)]
pub struct RotationX {
    theta: f64,
}

impl RotationX {
    /// Creates a new RX gate with the given angle
    pub const fn new(theta: f64) -> Self {
        Self { theta }
    }

    /// Returns the rotation angle
    pub const fn angle(&self) -> f64 {
        self.theta
    }

    /// Computes the RX matrix for this angle
    #[inline]
    pub fn matrix(&self) -> [[Complex64; 2]; 2] {
        matrices::rotation_x(self.theta)
    }
}

impl Gate for RotationX {
    fn name(&self) -> &str {
        "RX"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn description(&self) -> String {
        format!("RX({:.4})", self.theta)
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.matrix().iter().flatten().copied().collect())
    }
}

/// Rotation-Y gate
///
/// Rotates around Y-axis by angle θ
#[derive(Debug, Clone, Copy)]
pub struct RotationY {
    theta: f64,
}

impl RotationY {
    /// Creates a new RY gate with the given angle
    pub const fn new(theta: f64) -> Self {
        Self { theta }
    }

    /// Returns the rotation angle
    pub const fn angle(&self) -> f64 {
        self.theta
    }

    /// Computes the RY matrix for this angle
    #[inline]
    pub fn matrix(&self) -> [[Complex64; 2]; 2] {
        matrices::rotation_y(self.theta)
    }
}

impl Gate for RotationY {
    fn name(&self) -> &str {
        "RY"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn description(&self) -> String {
        format!("RY({:.4})", self.theta)
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.matrix().iter().flatten().copied().collect())
    }
}

/// Rotation-Z gate
///
/// Rotates around Z-axis by angle θ
#[derive(Debug, Clone, Copy)]
pub struct RotationZ {
    theta: f64,
}

impl RotationZ {
    /// Creates a new RZ gate with the given angle
    pub const fn new(theta: f64) -> Self {
        Self { theta }
    }

    /// Returns the rotation angle
    pub const fn angle(&self) -> f64 {
        self.theta
    }

    /// Computes the RZ matrix for this angle
    #[inline]
    pub fn matrix(&self) -> [[Complex64; 2]; 2] {
        matrices::rotation_z(self.theta)
    }
}

impl Gate for RotationZ {
    fn name(&self) -> &str {
        "RZ"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn description(&self) -> String {
        format!("RZ({:.4})", self.theta)
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.matrix().iter().flatten().copied().collect())
    }
}

/// Phase gate
///
/// Applies a phase rotation by angle θ
#[derive(Debug, Clone, Copy)]
pub struct Phase {
    theta: f64,
}

impl Phase {
    /// Creates a new Phase gate with the given angle
    pub const fn new(theta: f64) -> Self {
        Self { theta }
    }

    /// Returns the phase angle
    pub const fn angle(&self) -> f64 {
        self.theta
    }

    /// Computes the Phase matrix for this angle
    #[inline]
    pub fn matrix(&self) -> [[Complex64; 2]; 2] {
        matrices::phase(self.theta)
    }
}

impl Gate for Phase {
    fn name(&self) -> &str {
        "P"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn description(&self) -> String {
        format!("P({:.4})", self.theta)
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.matrix().iter().flatten().copied().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_properties() {
        assert_eq!(Hadamard.name(), "H");
        assert_eq!(Hadamard.num_qubits(), 1);
        assert!(Hadamard.is_hermitian());

        assert_eq!(PauliX.name(), "X");
        assert!(PauliX.is_hermitian());

        assert_eq!(CNot.name(), "CNOT");
        assert_eq!(CNot.num_qubits(), 2);
    }

    #[test]
    fn test_matrix_access() {
        let h_matrix = Hadamard::matrix();
        assert_eq!(h_matrix.len(), 2);
        assert_eq!(h_matrix[0].len(), 2);

        let cnot_matrix = CNot::matrix();
        assert_eq!(cnot_matrix.len(), 4);
        assert_eq!(cnot_matrix[0].len(), 4);
    }

    #[test]
    fn test_parameterized_gates() {
        use std::f64::consts::PI;

        let rx = RotationX::new(PI / 2.0);
        assert_eq!(rx.name(), "RX");
        assert_eq!(rx.angle(), PI / 2.0);
        assert!(rx.description().contains("RX"));

        let ry = RotationY::new(PI);
        assert_eq!(ry.angle(), PI);

        let rz = RotationZ::new(PI / 4.0);
        assert_eq!(rz.angle(), PI / 4.0);

        let phase = Phase::new(PI / 3.0);
        assert_eq!(phase.angle(), PI / 3.0);
    }
}
