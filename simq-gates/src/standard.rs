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

    fn is_diagonal(&self) -> bool {
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

    fn is_diagonal(&self) -> bool {
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

    // --- Coverage for lines 804-805, 1112-1113, 1156-1157 ---

    #[test]
    fn test_phase_gate_description() {
        use simq_core::gate::Gate;
        use std::f64::consts::PI;
        // Lines 804-805: Phase::description() returns format!("P({:.4})", theta)
        let phase = Phase::new(PI / 4.0);
        let desc = Gate::description(&phase);
        assert!(desc.contains("P("), "desc was: {desc}");
    }

    #[test]
    fn test_ryy_gate_description() {
        use simq_core::gate::Gate;
        use std::f64::consts::PI;
        // Lines 1112-1113: RYY::description()
        let ryy = RYY::new(PI / 2.0);
        let desc = Gate::description(&ryy);
        assert!(desc.contains("RYY("), "desc was: {desc}");
    }

    #[test]
    fn test_rzz_gate_description() {
        use simq_core::gate::Gate;
        use std::f64::consts::PI;
        // Lines 1156-1157: RZZ::description()
        let rzz = RZZ::new(PI / 3.0);
        let desc = Gate::description(&rzz);
        assert!(desc.contains("RZZ("), "desc was: {desc}");
    }

    #[test]
    fn test_single_qubit_gates_comprehensive() {
        use simq_core::gate::Gate;
        use std::f64::consts::PI;

        // Hadamard
        assert_eq!(Hadamard.name(), "H");
        assert_eq!(Hadamard.num_qubits(), 1);
        assert!(Hadamard.is_hermitian());
        assert!(Hadamard.matrix().is_some());
        let h_vec = Hadamard.matrix().unwrap();
        assert_eq!(h_vec.len(), 4); // 2x2 flattened

        // PauliX
        assert_eq!(PauliX.name(), "X");
        assert_eq!(PauliX.num_qubits(), 1);
        assert!(PauliX.is_hermitian());
        assert!(PauliX.matrix().is_some());

        // PauliY
        assert_eq!(PauliY.name(), "Y");
        assert_eq!(PauliY.num_qubits(), 1);
        assert!(PauliY.is_hermitian());
        assert!(PauliY.matrix().is_some());

        // PauliZ
        assert_eq!(PauliZ.name(), "Z");
        assert_eq!(PauliZ.num_qubits(), 1);
        assert!(PauliZ.is_hermitian());
        assert!(PauliZ.is_diagonal());
        assert!(PauliZ.matrix().is_some());

        // SGate
        assert_eq!(SGate.name(), "S");
        assert_eq!(SGate.num_qubits(), 1);
        assert!(SGate.is_diagonal());
        assert!(SGate.matrix().is_some());

        // SGateDagger
        assert_eq!(SGateDagger.name(), "S†");
        assert_eq!(SGateDagger.num_qubits(), 1);
        assert!(SGateDagger.is_diagonal());
        assert!(SGateDagger.matrix().is_some());

        // TGate
        assert_eq!(TGate.name(), "T");
        assert_eq!(TGate.num_qubits(), 1);
        assert!(TGate.is_diagonal());
        assert!(TGate.matrix().is_some());

        // TGateDagger
        assert_eq!(TGateDagger.name(), "T†");
        assert_eq!(TGateDagger.num_qubits(), 1);
        assert!(TGateDagger.is_diagonal());
        assert!(TGateDagger.matrix().is_some());

        // Identity
        assert_eq!(Identity.name(), "I");
        assert_eq!(Identity.num_qubits(), 1);
        assert!(Identity.is_hermitian());
        assert!(Identity.is_diagonal());
        assert!(Identity.matrix().is_some());

        // SXGate
        assert_eq!(SXGate.name(), "SX");
        assert_eq!(SXGate.num_qubits(), 1);
        assert!(SXGate.matrix().is_some());

        // SXGateDagger
        assert_eq!(SXGateDagger.name(), "SX†");
        assert_eq!(SXGateDagger.num_qubits(), 1);
        assert!(SXGateDagger.matrix().is_some());

        // RotationX - struct's matrix() returns [[Complex64;2];2], use Gate trait for Option
        let rx = RotationX::new(PI / 2.0);
        assert_eq!(rx.name(), "RX");
        assert_eq!(rx.num_qubits(), 1);
        assert!(Gate::matrix(&rx).is_some());
        assert_eq!(Gate::matrix(&rx).unwrap().len(), 4);

        // RotationY - struct's matrix() returns [[Complex64;2];2], use Gate trait for Option
        let ry = RotationY::new(PI / 4.0);
        assert_eq!(ry.name(), "RY");
        assert_eq!(ry.num_qubits(), 1);
        assert!(Gate::matrix(&ry).is_some());

        // RotationZ
        let rz = RotationZ::new(PI / 4.0);
        assert_eq!(rz.name(), "RZ");
        assert_eq!(rz.num_qubits(), 1);
        assert!(rz.is_diagonal());
        assert!(Gate::matrix(&rz).is_some());

        // Phase
        let phase = Phase::new(PI / 3.0);
        assert_eq!(phase.name(), "P");
        assert_eq!(phase.num_qubits(), 1);
        assert!(phase.is_diagonal());
        assert!(Gate::matrix(&phase).is_some());

        // U1 - struct's matrix() returns [[Complex64;2];2], use Gate trait for Option
        let u1 = U1::new(PI / 4.0);
        assert_eq!(u1.name(), "U1");
        assert_eq!(u1.num_qubits(), 1);
        assert!(u1.is_diagonal());
        assert!(Gate::matrix(&u1).is_some());

        // U2
        let u2 = U2::new(PI / 4.0, PI / 4.0);
        assert_eq!(u2.name(), "U2");
        assert_eq!(u2.num_qubits(), 1);
        assert!(Gate::matrix(&u2).is_some());

        // U3
        let u3 = U3::new(PI / 4.0, PI / 4.0, PI / 4.0);
        assert_eq!(u3.name(), "U3");
        assert_eq!(u3.num_qubits(), 1);
        assert!(Gate::matrix(&u3).is_some());
    }

    #[test]
    fn test_two_qubit_gates_comprehensive() {
        use simq_core::gate::Gate;

        // CNot
        assert_eq!(CNot.name(), "CNOT");
        assert_eq!(CNot.num_qubits(), 2);
        assert!(CNot.matrix().is_some());
        assert_eq!(CNot.matrix().unwrap().len(), 16); // 4x4 flattened

        // CZ
        assert_eq!(CZ.name(), "CZ");
        assert_eq!(CZ.num_qubits(), 2);
        assert!(CZ.is_hermitian());
        assert!(CZ.is_diagonal());
        assert!(CZ.matrix().is_some());

        // Swap
        assert_eq!(Swap.name(), "SWAP");
        assert_eq!(Swap.num_qubits(), 2);
        assert!(Swap.is_hermitian());
        assert!(Swap.matrix().is_some());

        // ISwap
        assert_eq!(ISwap.name(), "iSWAP");
        assert_eq!(ISwap.num_qubits(), 2);
        assert!(ISwap.matrix().is_some());

        // CY
        assert_eq!(CY.name(), "CY");
        assert_eq!(CY.num_qubits(), 2);
        assert!(CY.matrix().is_some());

        // ECR
        assert_eq!(ECR.name(), "ECR");
        assert_eq!(ECR.num_qubits(), 2);
        assert!(ECR.matrix().is_some());
    }

    #[test]
    fn test_three_qubit_gates_comprehensive() {
        use simq_core::gate::Gate;

        // Toffoli
        let t = Toffoli;
        assert_eq!(t.name(), "CCNOT");
        assert_eq!(t.num_qubits(), 3);
        assert!(t.matrix().is_some());
        assert_eq!(t.matrix().unwrap().len(), 64); // 8x8 flattened
        assert!(t.description().contains("Toffoli"));

        // Fredkin
        let f = Fredkin;
        assert_eq!(f.name(), "CSWAP");
        assert_eq!(f.num_qubits(), 3);
        assert!(f.matrix().is_some());
        assert_eq!(f.matrix().unwrap().len(), 64);
        assert!(f.description().contains("Fredkin"));
    }

    #[test]
    fn test_parameterized_two_qubit_gates() {
        use simq_core::gate::Gate;
        use std::f64::consts::PI;

        // CPhase - struct method returns [[Complex64;4];4], Gate trait method returns Option<Vec>
        let cp = CPhase::new(PI / 4.0);
        assert_eq!(cp.name(), "CP");
        assert_eq!(cp.num_qubits(), 2);
        assert_eq!(cp.angle(), PI / 4.0);
        // The Gate trait method
        let cp_gate_matrix = Gate::matrix(&cp);
        assert!(cp_gate_matrix.is_some());
        assert_eq!(cp_gate_matrix.unwrap().len(), 16);
        assert!(cp.description().contains("CP"));

        // RXX
        let rxx = RXX::new(PI / 4.0);
        assert_eq!(rxx.name(), "RXX");
        assert_eq!(rxx.num_qubits(), 2);
        assert_eq!(rxx.angle(), PI / 4.0);
        let rxx_gate_matrix = Gate::matrix(&rxx);
        assert!(rxx_gate_matrix.is_some());
        assert!(rxx.description().contains("RXX"));

        // RYY
        let ryy = RYY::new(PI / 4.0);
        assert_eq!(ryy.name(), "RYY");
        assert_eq!(ryy.num_qubits(), 2);
        assert!(Gate::matrix(&ryy).is_some());

        // RZZ
        let rzz = RZZ::new(PI / 4.0);
        assert_eq!(rzz.name(), "RZZ");
        assert_eq!(rzz.num_qubits(), 2);
        assert!(Gate::matrix(&rzz).is_some());
    }

    #[test]
    fn test_diagonal_gates_diagonal_elements() {
        use simq_core::gate::DiagonalGate;
        use std::f64::consts::{FRAC_1_SQRT_2, PI};

        // PauliZ diagonal: [1, -1]
        let z_diag = PauliZ.diagonal_elements();
        assert!((z_diag[0].re - 1.0).abs() < 1e-12);
        assert!((z_diag[1].re - (-1.0)).abs() < 1e-12);

        // SGate diagonal: [1, i]
        let s_diag = SGate.diagonal_elements();
        assert!((s_diag[0].re - 1.0).abs() < 1e-12);
        assert!((s_diag[1].im - 1.0).abs() < 1e-12);

        // SGateDagger diagonal: [1, -i]
        let sd_diag = SGateDagger.diagonal_elements();
        assert!((sd_diag[1].im - (-1.0)).abs() < 1e-12);

        // TGate diagonal: [1, e^(iπ/4)]
        let t_diag = TGate.diagonal_elements();
        assert!((t_diag[0].re - 1.0).abs() < 1e-12);
        assert!((t_diag[1].re - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((t_diag[1].im - FRAC_1_SQRT_2).abs() < 1e-10);

        // TGateDagger diagonal: [1, e^(-iπ/4)]
        let td_diag = TGateDagger.diagonal_elements();
        assert!((td_diag[1].re - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((td_diag[1].im - (-FRAC_1_SQRT_2)).abs() < 1e-10);

        // RotationZ diagonal
        let rz = RotationZ::new(PI / 2.0);
        let rz_diag = rz.diagonal_elements();
        assert!((rz_diag[0].norm_sqr() - 1.0).abs() < 1e-10);
        assert!((rz_diag[1].norm_sqr() - 1.0).abs() < 1e-10);

        // Phase diagonal
        let phase = Phase::new(PI / 4.0);
        let p_diag = phase.diagonal_elements();
        assert!((p_diag[0].re - 1.0).abs() < 1e-12);
        assert!((p_diag[1].norm_sqr() - 1.0).abs() < 1e-10);

        // U1 diagonal
        let u1 = U1::new(PI);
        let u1_diag = u1.diagonal_elements();
        assert!((u1_diag[0].re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_rotation_gate_angles_and_descriptions() {
        use simq_core::gate::Gate;
        use std::f64::consts::PI;

        let rx = RotationX::new(1.23);
        assert!((rx.angle() - 1.23).abs() < 1e-12);
        assert!(rx.description().contains("1.23"));

        let ry = RotationY::new(PI);
        assert!(ry.description().contains("RY"));

        let rz = RotationZ::new(PI / 2.0);
        assert!(rz.description().contains("RZ"));

        // U-gate descriptions
        let u2 = U2::new(0.1, 0.2);
        assert_eq!(u2.phi(), 0.1);
        assert_eq!(u2.lambda(), 0.2);
        assert!(u2.description().contains("U2"));

        let u3 = U3::new(0.1, 0.2, 0.3);
        assert_eq!(u3.theta(), 0.1);
        assert_eq!(u3.phi(), 0.2);
        assert_eq!(u3.lambda(), 0.3);
        assert!(u3.description().contains("U3"));

        let u1 = U1::new(0.5);
        assert_eq!(u1.lambda(), 0.5);
        assert!(u1.description().contains("U1"));
    }

    #[test]
    fn test_rotation_gate_matrix_uncached() {
        use std::f64::consts::PI;

        let rx = RotationX::new(PI / 3.0);
        let cached = rx.matrix();
        let uncached = rx.matrix_uncached();
        // Both should give same result
        for i in 0..2 {
            for j in 0..2 {
                let diff_re = (cached[i][j].re - uncached[i][j].re).abs();
                let diff_im = (cached[i][j].im - uncached[i][j].im).abs();
                assert!(diff_re < 1e-10, "re diff at [{i}][{j}] = {diff_re}");
                assert!(diff_im < 1e-10, "im diff at [{i}][{j}] = {diff_im}");
            }
        }

        let ry = RotationY::new(PI / 3.0);
        let ry_cached = ry.matrix();
        let ry_uncached = ry.matrix_uncached();
        for i in 0..2 {
            for j in 0..2 {
                assert!((ry_cached[i][j].re - ry_uncached[i][j].re).abs() < 1e-10);
            }
        }

        let rz = RotationZ::new(PI / 3.0);
        let rz_cached = rz.matrix();
        let rz_uncached = rz.matrix_uncached();
        for i in 0..2 {
            for j in 0..2 {
                assert!((rz_cached[i][j].re - rz_uncached[i][j].re).abs() < 1e-10);
            }
        }
    }
}
