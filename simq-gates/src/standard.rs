//! Standard quantum gate implementations with pre-computed matrices

use crate::matrices;
use num_complex::Complex64;
use simq_core::gate::{DiagonalGate, Gate};

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

    fn is_diagonal(&self) -> bool {
        true
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl DiagonalGate for PauliZ {
    #[inline]
    fn diagonal_elements(&self) -> [Complex64; 2] {
        // Z = [[1, 0], [0, -1]]
        [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)]
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

    fn is_diagonal(&self) -> bool {
        true
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl DiagonalGate for SGate {
    #[inline]
    fn diagonal_elements(&self) -> [Complex64; 2] {
        // S = [[1, 0], [0, i]]
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]
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

    fn is_diagonal(&self) -> bool {
        true
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl DiagonalGate for SGateDagger {
    #[inline]
    fn diagonal_elements(&self) -> [Complex64; 2] {
        // S† = [[1, 0], [0, -i]]
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, -1.0)]
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

    fn is_diagonal(&self) -> bool {
        true
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl DiagonalGate for TGate {
    #[inline]
    fn diagonal_elements(&self) -> [Complex64; 2] {
        // T = [[1, 0], [0, e^(iπ/4)]]
        const SQRT2_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
        [Complex64::new(1.0, 0.0), Complex64::new(SQRT2_2, SQRT2_2)]
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

    fn is_diagonal(&self) -> bool {
        true
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl DiagonalGate for TGateDagger {
    #[inline]
    fn diagonal_elements(&self) -> [Complex64; 2] {
        // T† = [[1, 0], [0, e^(-iπ/4)]]
        const SQRT2_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
        [Complex64::new(1.0, 0.0), Complex64::new(SQRT2_2, -SQRT2_2)]
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

/// SX gate (√X gate)
///
/// Square root of X gate: SX·SX = X
#[derive(Debug, Clone, Copy)]
pub struct SXGate;

impl Gate for SXGate {
    fn name(&self) -> &str {
        "SX"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(SXGate, &matrices::SX_GATE, 2);

/// SX† gate (adjoint of √X gate)
///
/// Adjoint of square root of X gate
#[derive(Debug, Clone, Copy)]
pub struct SXGateDagger;

impl Gate for SXGateDagger {
    fn name(&self) -> &str {
        "SX†"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(SXGateDagger, &matrices::SX_GATE_DAGGER, 2);

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

/// CY gate (Controlled-Y)
///
/// Applies Y gate to target if control qubit is |1⟩
#[derive(Debug, Clone, Copy)]
pub struct CY;

impl Gate for CY {
    fn name(&self) -> &str {
        "CY"
    }

    fn num_qubits(&self) -> usize {
        2
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(CY, &matrices::CY, 4);

/// ECR gate (Echoed Cross-Resonance)
///
/// Native gate for IBM quantum hardware
#[derive(Debug, Clone, Copy)]
pub struct ECR;

impl Gate for ECR {
    fn name(&self) -> &str {
        "ECR"
    }

    fn num_qubits(&self) -> usize {
        2
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl_matrix_method!(ECR, &matrices::ECR, 4);

// ============================================================================
// Three-Qubit Gates
// ============================================================================

/// Toffoli gate (CCNOT - Controlled-Controlled-NOT)
///
/// Flips target qubit only when both control qubits are |1⟩
#[derive(Debug, Clone, Copy)]
pub struct Toffoli;

impl Gate for Toffoli {
    fn name(&self) -> &str {
        "CCNOT"
    }

    fn num_qubits(&self) -> usize {
        3
    }

    fn description(&self) -> String {
        "Toffoli (CCNOT)".to_string()
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl Toffoli {
    /// Returns the pre-computed gate matrix
    #[inline]
    pub const fn matrix() -> &'static [[Complex64; 8]; 8] {
        &matrices::TOFFOLI
    }

    /// Returns the matrix as a flattened vector (for Gate trait)
    #[inline]
    fn matrix_vec() -> Vec<Complex64> {
        Self::matrix().iter().flatten().copied().collect()
    }
}

/// Fredkin gate (CSWAP - Controlled-SWAP)
///
/// Swaps target qubits only when control qubit is |1⟩
#[derive(Debug, Clone, Copy)]
pub struct Fredkin;

impl Gate for Fredkin {
    fn name(&self) -> &str {
        "CSWAP"
    }

    fn num_qubits(&self) -> usize {
        3
    }

    fn description(&self) -> String {
        "Fredkin (CSWAP)".to_string()
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(Self::matrix_vec())
    }
}

impl Fredkin {
    /// Returns the pre-computed gate matrix
    #[inline]
    pub const fn matrix() -> &'static [[Complex64; 8]; 8] {
        &matrices::FREDKIN
    }

    /// Returns the matrix as a flattened vector (for Gate trait)
    #[inline]
    fn matrix_vec() -> Vec<Complex64> {
        Self::matrix().iter().flatten().copied().collect()
    }
}

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
    ///
    /// Uses compile-time caching when available for optimal performance:
    /// - Common angles (π/4, π/2, etc.): ~0 ns (compile-time constant)
    /// - VQE range (0 to π/4): ~2-5 ns (array lookup)
    /// - Other angles: ~20-50 ns (trigonometric computation)
    #[inline]
    pub fn matrix(&self) -> [[Complex64; 2]; 2] {
        crate::generated::EnhancedUniversalCache::rx(self.theta)
    }

    /// Computes the RX matrix without caching (for testing/benchmarking)
    #[inline]
    pub fn matrix_uncached(&self) -> [[Complex64; 2]; 2] {
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
    ///
    /// Uses compile-time caching when available for optimal performance.
    #[inline]
    pub fn matrix(&self) -> [[Complex64; 2]; 2] {
        crate::generated::EnhancedUniversalCache::ry(self.theta)
    }

    /// Computes the RY matrix without caching (for testing/benchmarking)
    #[inline]
    pub fn matrix_uncached(&self) -> [[Complex64; 2]; 2] {
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
    ///
    /// Uses compile-time caching when available for optimal performance.
    #[inline]
    pub fn matrix(&self) -> [[Complex64; 2]; 2] {
        crate::generated::EnhancedUniversalCache::rz(self.theta)
    }

    /// Computes the RZ matrix without caching (for testing/benchmarking)
    #[inline]
    pub fn matrix_uncached(&self) -> [[Complex64; 2]; 2] {
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

    fn is_diagonal(&self) -> bool {
        true
    }

    fn description(&self) -> String {
        format!("RZ({:.4})", self.theta)
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.matrix().iter().flatten().copied().collect())
    }
}

impl DiagonalGate for RotationZ {
    #[inline]
    fn diagonal_elements(&self) -> [Complex64; 2] {
        // RZ(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
        let half_theta = self.theta / 2.0;
        [
            Complex64::new(half_theta.cos(), -half_theta.sin()),
            Complex64::new(half_theta.cos(), half_theta.sin()),
        ]
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

    fn is_diagonal(&self) -> bool {
        true
    }

    fn description(&self) -> String {
        format!("P({:.4})", self.theta)
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.matrix().iter().flatten().copied().collect())
    }
}

impl DiagonalGate for Phase {
    #[inline]
    fn diagonal_elements(&self) -> [Complex64; 2] {
        // P(θ) = [[1, 0], [0, e^(iθ)]]
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(self.theta.cos(), self.theta.sin()),
        ]
    }
}

/// U1 gate (Universal 1-parameter gate)
///
/// Applies a phase rotation U1(λ) = P(λ)
#[derive(Debug, Clone, Copy)]
pub struct U1 {
    lambda: f64,
}

impl U1 {
    /// Creates a new U1 gate with the given λ parameter
    pub const fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Returns the λ parameter
    pub const fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Computes the U1 matrix for these parameters
    #[inline]
    pub fn matrix(&self) -> [[Complex64; 2]; 2] {
        matrices::u1(self.lambda)
    }
}

impl Gate for U1 {
    fn name(&self) -> &str {
        "U1"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn is_diagonal(&self) -> bool {
        true
    }

    fn description(&self) -> String {
        format!("U1({:.4})", self.lambda)
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.matrix().iter().flatten().copied().collect())
    }
}

impl DiagonalGate for U1 {
    #[inline]
    fn diagonal_elements(&self) -> [Complex64; 2] {
        // U1(λ) = [[1, 0], [0, e^(iλ)]]
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(self.lambda.cos(), self.lambda.sin()),
        ]
    }
}

/// U2 gate (Universal 2-parameter gate)
///
/// Applies Hadamard-like rotation U2(φ, λ)
#[derive(Debug, Clone, Copy)]
pub struct U2 {
    phi: f64,
    lambda: f64,
}

impl U2 {
    /// Creates a new U2 gate with the given parameters
    pub const fn new(phi: f64, lambda: f64) -> Self {
        Self { phi, lambda }
    }

    /// Returns the φ parameter
    pub const fn phi(&self) -> f64 {
        self.phi
    }

    /// Returns the λ parameter
    pub const fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Computes the U2 matrix for these parameters
    #[inline]
    pub fn matrix(&self) -> [[Complex64; 2]; 2] {
        matrices::u2(self.phi, self.lambda)
    }
}

impl Gate for U2 {
    fn name(&self) -> &str {
        "U2"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn description(&self) -> String {
        format!("U2({:.4}, {:.4})", self.phi, self.lambda)
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.matrix().iter().flatten().copied().collect())
    }
}

/// U3 gate (Universal 3-parameter gate)
///
/// Most general single-qubit unitary gate U3(θ, φ, λ)
#[derive(Debug, Clone, Copy)]
pub struct U3 {
    theta: f64,
    phi: f64,
    lambda: f64,
}

impl U3 {
    /// Creates a new U3 gate with the given parameters
    pub const fn new(theta: f64, phi: f64, lambda: f64) -> Self {
        Self { theta, phi, lambda }
    }

    /// Returns the θ parameter
    pub const fn theta(&self) -> f64 {
        self.theta
    }

    /// Returns the φ parameter
    pub const fn phi(&self) -> f64 {
        self.phi
    }

    /// Returns the λ parameter
    pub const fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Computes the U3 matrix for these parameters
    #[inline]
    pub fn matrix(&self) -> [[Complex64; 2]; 2] {
        matrices::u3(self.theta, self.phi, self.lambda)
    }
}

impl Gate for U3 {
    fn name(&self) -> &str {
        "U3"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn description(&self) -> String {
        format!("U3({:.4}, {:.4}, {:.4})", self.theta, self.phi, self.lambda)
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.matrix().iter().flatten().copied().collect())
    }
}

/// Controlled-Phase gate
///
/// Applies a phase to the |11⟩ state
#[derive(Debug, Clone, Copy)]
pub struct CPhase {
    theta: f64,
}

impl CPhase {
    /// Creates a new Controlled-Phase gate with the given angle
    pub const fn new(theta: f64) -> Self {
        Self { theta }
    }

    /// Returns the phase angle
    pub const fn angle(&self) -> f64 {
        self.theta
    }

    /// Computes the CP matrix for this angle
    #[inline]
    pub fn matrix(&self) -> [[Complex64; 4]; 4] {
        matrices::controlled_phase(self.theta)
    }
}

impl Gate for CPhase {
    fn name(&self) -> &str {
        "CP"
    }

    fn num_qubits(&self) -> usize {
        2
    }

    fn description(&self) -> String {
        format!("CP({:.4})", self.theta)
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.matrix().iter().flatten().copied().collect())
    }
}

/// RXX gate (XX rotation)
///
/// Two-qubit rotation around X⊗X axis
#[derive(Debug, Clone, Copy)]
pub struct RXX {
    theta: f64,
}

impl RXX {
    /// Creates a new RXX gate with the given angle
    pub const fn new(theta: f64) -> Self {
        Self { theta }
    }

    /// Returns the rotation angle
    pub const fn angle(&self) -> f64 {
        self.theta
    }

    /// Computes the RXX matrix for this angle
    #[inline]
    pub fn matrix(&self) -> [[Complex64; 4]; 4] {
        matrices::rxx(self.theta)
    }
}

impl Gate for RXX {
    fn name(&self) -> &str {
        "RXX"
    }

    fn num_qubits(&self) -> usize {
        2
    }

    fn description(&self) -> String {
        format!("RXX({:.4})", self.theta)
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.matrix().iter().flatten().copied().collect())
    }
}

/// RYY gate (YY rotation)
///
/// Two-qubit rotation around Y⊗Y axis
#[derive(Debug, Clone, Copy)]
pub struct RYY {
    theta: f64,
}

impl RYY {
    /// Creates a new RYY gate with the given angle
    pub const fn new(theta: f64) -> Self {
        Self { theta }
    }

    /// Returns the rotation angle
    pub const fn angle(&self) -> f64 {
        self.theta
    }

    /// Computes the RYY matrix for this angle
    #[inline]
    pub fn matrix(&self) -> [[Complex64; 4]; 4] {
        matrices::ryy(self.theta)
    }
}

impl Gate for RYY {
    fn name(&self) -> &str {
        "RYY"
    }

    fn num_qubits(&self) -> usize {
        2
    }

    fn description(&self) -> String {
        format!("RYY({:.4})", self.theta)
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.matrix().iter().flatten().copied().collect())
    }
}

/// RZZ gate (ZZ rotation)
///
/// Two-qubit rotation around Z⊗Z axis
#[derive(Debug, Clone, Copy)]
pub struct RZZ {
    theta: f64,
}

impl RZZ {
    /// Creates a new RZZ gate with the given angle
    pub const fn new(theta: f64) -> Self {
        Self { theta }
    }

    /// Returns the rotation angle
    pub const fn angle(&self) -> f64 {
        self.theta
    }

    /// Computes the RZZ matrix for this angle
    #[inline]
    pub fn matrix(&self) -> [[Complex64; 4]; 4] {
        matrices::rzz(self.theta)
    }
}

impl Gate for RZZ {
    fn name(&self) -> &str {
        "RZZ"
    }

    fn num_qubits(&self) -> usize {
        2
    }

    fn description(&self) -> String {
        format!("RZZ({:.4})", self.theta)
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
