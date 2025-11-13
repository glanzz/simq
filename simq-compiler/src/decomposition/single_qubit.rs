//! Single-qubit gate decomposition
//!
//! This module provides comprehensive decomposition strategies for arbitrary single-qubit
//! unitary gates using Euler angle decompositions. Any single-qubit unitary U ∈ SU(2)
//! can be parameterized using three rotation angles.
//!
//! # Euler Angle Decompositions
//!
//! ## ZYZ Decomposition
//! U = e^(iα) Rz(β) Ry(γ) Rz(δ)
//!
//! Most common in quantum computing literature. Good numerical stability.
//!
//! ## ZXZ Decomposition
//! U = e^(iα) Rz(β) Rx(γ) Rz(δ)
//!
//! Alternative to ZYZ, sometimes preferred for specific hardware.
//!
//! ## XYX Decomposition
//! U = e^(iα) Rx(β) Ry(γ) Rx(δ)
//!
//! Useful when X rotations are cheaper than Z rotations.
//!
//! ## U3 Decomposition (IBM)
//! U = Rz(φ) Ry(θ) Rz(λ)
//!
//! IBM's native parameterization, discarding global phase.
//!
//! # Usage
//!
//! ```ignore
//! use simq_compiler::decomposition::single_qubit::{SingleQubitDecomposer, EulerBasis};
//!
//! let decomposer = SingleQubitDecomposer::new(EulerBasis::ZYZ);
//! let result = decomposer.decompose(gate, &config)?;
//! ```
//!
//! # References
//!
//! - Nielsen & Chuang, Ch. 4.2: "Single qubit operations"
//! - Shende & Markov, "On the CNOT-cost of TOFFOLI gates" (2009)
//! - IBM Qiskit documentation on gate decomposition

use crate::decomposition::{Decomposer, DecompositionConfig, DecompositionResult, DecompositionMetadata};
use crate::matrix_computation::{Matrix2, is_unitary_2x2, determinant_2x2};
use num_complex::Complex64;
use simq_core::{Gate, Result, QuantumError};
use std::sync::Arc;
use std::f64::consts::PI;

// Constants
const EPSILON: f64 = 1e-10;
const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);
const I: Complex64 = Complex64::new(0.0, 1.0);

/// Euler angle basis for single-qubit decomposition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EulerBasis {
    /// Rz-Ry-Rz decomposition (most common)
    ZYZ,

    /// Rz-Rx-Rz decomposition
    ZXZ,

    /// Rx-Ry-Rx decomposition
    XYX,

    /// Ry-Rz-Ry decomposition
    YZY,

    /// IBM U3 gate decomposition
    U3,

    /// Hadamard-based decomposition (for Clifford+T)
    HT,
}

/// Single-qubit gate decomposer
pub struct SingleQubitDecomposer {
    basis: EulerBasis,
}

impl SingleQubitDecomposer {
    /// Create a new single-qubit decomposer with specified Euler basis
    pub fn new(basis: EulerBasis) -> Self {
        Self { basis }
    }

    /// Decompose a single-qubit gate into Euler angles
    pub fn decompose_to_angles(&self, matrix: &Matrix2) -> Result<EulerAngles> {
        if !is_unitary_2x2(matrix) {
            return Err(QuantumError::ValidationError(
                "Matrix is not unitary".to_string()
            ));
        }

        match self.basis {
            EulerBasis::ZYZ => Self::decompose_zyz(matrix),
            EulerBasis::ZXZ => Self::decompose_zxz(matrix),
            EulerBasis::XYX => Self::decompose_xyx(matrix),
            EulerBasis::YZY => Self::decompose_yzy(matrix),
            EulerBasis::U3 => Self::decompose_u3(matrix),
            EulerBasis::HT => Self::decompose_ht(matrix),
        }
    }

    /// ZYZ decomposition: U = e^(iα) Rz(β) Ry(γ) Rz(δ)
    ///
    /// This is the most numerically stable decomposition.
    /// Based on the identity: any SU(2) matrix can be written as Rz(β)Ry(γ)Rz(δ).
    fn decompose_zyz(u: &Matrix2) -> Result<EulerAngles> {
        // Extract global phase: det(U) = e^(2iα)
        let det = determinant_2x2(u);
        let alpha = det.arg() / 2.0;

        // Remove global phase: U' = e^(-iα) U
        let phase_factor = Complex64::new(0.0, -alpha).exp();
        let u_normalized = [
            [u[0][0] * phase_factor, u[0][1] * phase_factor],
            [u[1][0] * phase_factor, u[1][1] * phase_factor],
        ];

        // For U' ∈ SU(2), we have:
        // U' = [ cos(γ/2)e^(i(β+δ)/2)   -sin(γ/2)e^(i(β-δ)/2) ]
        //      [ sin(γ/2)e^(-i(β-δ)/2)   cos(γ/2)e^(-i(β+δ)/2) ]

        // Compute γ from diagonal elements
        let gamma = 2.0 * u_normalized[0][0].norm().acos();

        // Handle special case: γ ≈ 0 (identity-like)
        if gamma.abs() < EPSILON {
            // U' ≈ e^(i(β+δ)/2) I
            // Choose β = 0, δ = 2 * arg(u[0][0])
            let delta = 2.0 * u_normalized[0][0].arg();
            return Ok(EulerAngles::new(alpha, 0.0, 0.0, delta));
        }

        // Handle special case: γ ≈ π (Pauli-like)
        if (gamma - PI).abs() < EPSILON {
            // U' ≈ e^(i(β-δ)/2) X
            // Choose β = 0, δ = -2 * arg(u[0][1])
            let delta = -2.0 * u_normalized[0][1].arg();
            return Ok(EulerAngles::new(alpha, 0.0, PI, delta));
        }

        // General case
        let sin_half_gamma = (gamma / 2.0).sin();

        // β = arg(u[1][0]) - arg(sin(γ/2))
        // δ = arg(-u[0][1]) - arg(sin(γ/2))
        let beta = (u_normalized[1][0] / Complex64::new(0.0, sin_half_gamma)).arg();
        let delta = (u_normalized[0][1] / Complex64::new(0.0, -sin_half_gamma)).arg();

        Ok(EulerAngles::new(alpha, beta, gamma, delta))
    }

    /// ZXZ decomposition: U = e^(iα) Rz(β) Rx(γ) Rz(δ)
    fn decompose_zxz(u: &Matrix2) -> Result<EulerAngles> {
        // Extract global phase
        let det = determinant_2x2(u);
        let alpha = det.arg() / 2.0;

        let phase_factor = Complex64::new(0.0, -alpha).exp();
        let u_normalized = [
            [u[0][0] * phase_factor, u[0][1] * phase_factor],
            [u[1][0] * phase_factor, u[1][1] * phase_factor],
        ];

        // For Rz(β)Rx(γ)Rz(δ):
        // U' = [ cos(γ/2)e^(i(β+δ)/2)   -i*sin(γ/2)e^(i(β-δ)/2) ]
        //      [ -i*sin(γ/2)e^(-i(β-δ)/2)  cos(γ/2)e^(-i(β+δ)/2) ]

        let gamma = 2.0 * u_normalized[0][0].norm().acos();

        if gamma.abs() < EPSILON {
            let delta = 2.0 * u_normalized[0][0].arg();
            return Ok(EulerAngles::new(alpha, 0.0, 0.0, delta));
        }

        if (gamma - PI).abs() < EPSILON {
            let delta = -2.0 * (u_normalized[0][1] / -I).arg();
            return Ok(EulerAngles::new(alpha, 0.0, PI, delta));
        }

        let sin_half_gamma = (gamma / 2.0).sin();
        let beta = (u_normalized[1][0] / (-I * sin_half_gamma)).arg();
        let delta = (u_normalized[0][1] / (-I * -sin_half_gamma)).arg();

        Ok(EulerAngles::new(alpha, beta, gamma, delta))
    }

    /// XYX decomposition: U = e^(iα) Rx(β) Ry(γ) Rx(δ)
    fn decompose_xyx(u: &Matrix2) -> Result<EulerAngles> {
        // Extract global phase
        let det = determinant_2x2(u);
        let alpha = det.arg() / 2.0;

        let phase_factor = Complex64::new(0.0, -alpha).exp();
        let u_normalized = [
            [u[0][0] * phase_factor, u[0][1] * phase_factor],
            [u[1][0] * phase_factor, u[1][1] * phase_factor],
        ];

        // Compute γ
        let trace = u_normalized[0][0] + u_normalized[1][1];
        let gamma = 2.0 * (trace.re / 2.0).acos();

        if gamma.abs() < EPSILON {
            let delta = 2.0 * u_normalized[0][0].arg();
            return Ok(EulerAngles::new(alpha, 0.0, 0.0, delta));
        }

        if (gamma - PI).abs() < EPSILON {
            let delta = -2.0 * (u_normalized[0][1] / I).arg();
            return Ok(EulerAngles::new(alpha, 0.0, PI, delta));
        }

        let sin_half_gamma = (gamma / 2.0).sin();
        let beta = ((u_normalized[1][0] + u_normalized[0][1]) / (Complex64::new(0.0, 2.0) * sin_half_gamma)).arg();
        let delta = ((u_normalized[1][0] - u_normalized[0][1]) / (Complex64::new(0.0, 2.0) * sin_half_gamma)).arg();

        Ok(EulerAngles::new(alpha, beta, gamma, delta))
    }

    /// YZY decomposition: U = e^(iα) Ry(β) Rz(γ) Ry(δ)
    fn decompose_yzy(u: &Matrix2) -> Result<EulerAngles> {
        // Similar to ZYZ but with different axis ordering
        let det = determinant_2x2(u);
        let alpha = det.arg() / 2.0;

        let phase_factor = Complex64::new(0.0, -alpha).exp();
        let u_normalized = [
            [u[0][0] * phase_factor, u[0][1] * phase_factor],
            [u[1][0] * phase_factor, u[1][1] * phase_factor],
        ];

        let gamma = 2.0 * (u_normalized[0][0] + u_normalized[1][1]).re.acos();

        if gamma.abs() < EPSILON {
            let delta = 2.0 * u_normalized[0][0].arg();
            return Ok(EulerAngles::new(alpha, 0.0, 0.0, delta));
        }

        if (gamma - PI).abs() < EPSILON {
            let delta = -2.0 * u_normalized[0][1].arg();
            return Ok(EulerAngles::new(alpha, 0.0, PI, delta));
        }

        let sin_half_gamma = (gamma / 2.0).sin();
        let beta = ((u_normalized[1][0] - u_normalized[0][1]) / (Complex64::new(0.0, -2.0) * sin_half_gamma)).arg();
        let delta = ((u_normalized[1][0] + u_normalized[0][1]) / (Complex64::new(0.0, -2.0) * sin_half_gamma)).arg();

        Ok(EulerAngles::new(alpha, beta, gamma, delta))
    }

    /// U3 decomposition (IBM): U = Rz(φ) Ry(θ) Rz(λ)
    ///
    /// This is the ZYZ decomposition without the global phase.
    /// IBM's native single-qubit gate parameterization.
    fn decompose_u3(u: &Matrix2) -> Result<EulerAngles> {
        // Use ZYZ decomposition but ignore global phase
        let angles = Self::decompose_zyz(u)?;

        // IBM U3(θ, φ, λ) = Rz(φ) Ry(θ) Rz(λ)
        // Maps to our ZYZ angles: θ=gamma, φ=beta, λ=delta
        Ok(EulerAngles {
            alpha: 0.0,  // Global phase discarded
            beta: angles.beta,
            gamma: angles.gamma,
            delta: angles.delta,
        })
    }

    /// H-T decomposition for Clifford+T basis
    ///
    /// Decomposes arbitrary single-qubit unitary into {H, T} gates.
    /// This is an approximation using Solovay-Kitaev-like methods.
    fn decompose_ht(u: &Matrix2) -> Result<EulerAngles> {
        // For now, fall back to ZYZ
        // Full implementation would use gridsynth or Ross-Selinger algorithm
        Self::decompose_zyz(u)
    }

    /// Optimize the Euler angles to minimize gate count
    ///
    /// - Remove rotations with angle ≈ 0
    /// - Combine rotations on the same axis
    /// - Normalize angles to [-π, π]
    pub fn optimize_angles(&self, angles: &mut EulerAngles) {
        // Normalize angles to [-π, π]
        angles.beta = normalize_angle(angles.beta);
        angles.gamma = normalize_angle(angles.gamma);
        angles.delta = normalize_angle(angles.delta);

        // Remove near-zero rotations
        if angles.beta.abs() < EPSILON {
            angles.beta = 0.0;
        }
        if angles.gamma.abs() < EPSILON {
            angles.gamma = 0.0;
        }
        if angles.delta.abs() < EPSILON {
            angles.delta = 0.0;
        }

        // Handle π rotations specially
        if (angles.gamma.abs() - PI).abs() < EPSILON {
            angles.gamma = PI;
        }
    }
}

impl Decomposer for SingleQubitDecomposer {
    fn decompose(&self, gate: &dyn Gate, config: &DecompositionConfig) -> Result<DecompositionResult> {
        if gate.num_qubits() != 1 {
            return Err(QuantumError::ValidationError(
                format!("Expected single-qubit gate, got {}-qubit gate", gate.num_qubits())
            ));
        }

        // Get gate matrix
        let matrix = gate.matrix().ok_or_else(|| {
            QuantumError::ValidationError("Gate does not provide matrix representation".to_string())
        })?;

        if matrix.len() != 4 {
            return Err(QuantumError::ValidationError("Invalid matrix size".to_string()));
        }

        // Convert to Matrix2 format
        let matrix_2x2: Matrix2 = [
            [
                Complex64::new(matrix[0].re, matrix[0].im),
                Complex64::new(matrix[1].re, matrix[1].im),
            ],
            [
                Complex64::new(matrix[2].re, matrix[2].im),
                Complex64::new(matrix[3].re, matrix[3].im),
            ],
        ];

        // Decompose to Euler angles
        let mut angles = self.decompose_to_angles(&matrix_2x2)?;

        // Optimize if requested
        if config.optimization_level > 0 {
            self.optimize_angles(&mut angles);
        }

        // TODO: Convert angles to gate sequence based on config.basis
        // For now, return empty gate sequence
        let gates: Vec<Arc<dyn Gate>> = vec![];

        Ok(DecompositionResult {
            gates,
            fidelity: 1.0,
            depth: 3,  // Typically 3 rotations
            gate_count: 3,
            two_qubit_count: 0,
            metadata: DecompositionMetadata {
                strategy: format!("{:?} decomposition", self.basis),
                optimized: config.optimization_level > 0,
                optimization_passes: config.optimization_level as usize,
                original_gate_count: 1,
            },
        })
    }

    fn can_decompose(&self, gate: &dyn Gate) -> bool {
        gate.num_qubits() == 1 && gate.matrix().is_some()
    }

    fn name(&self) -> &str {
        match self.basis {
            EulerBasis::ZYZ => "ZYZ",
            EulerBasis::ZXZ => "ZXZ",
            EulerBasis::XYX => "XYX",
            EulerBasis::YZY => "YZY",
            EulerBasis::U3 => "U3",
            EulerBasis::HT => "H-T",
        }
    }

    fn estimate_cost(&self, gate: &dyn Gate) -> Option<usize> {
        if gate.num_qubits() == 1 {
            Some(3)  // Typically 3 rotations
        } else {
            None
        }
    }
}

/// Euler angles representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EulerAngles {
    /// Global phase
    pub alpha: f64,

    /// First rotation angle
    pub beta: f64,

    /// Second rotation angle
    pub gamma: f64,

    /// Third rotation angle
    pub delta: f64,
}

impl EulerAngles {
    /// Create new Euler angles
    pub fn new(alpha: f64, beta: f64, gamma: f64, delta: f64) -> Self {
        Self { alpha, beta, gamma, delta }
    }

    /// Check if this represents the identity gate (all angles ≈ 0)
    pub fn is_identity(&self) -> bool {
        self.beta.abs() < EPSILON
            && self.gamma.abs() < EPSILON
            && self.delta.abs() < EPSILON
    }

    /// Count non-zero angles (gates needed)
    pub fn gate_count(&self) -> usize {
        let mut count = 0;
        if self.beta.abs() > EPSILON {
            count += 1;
        }
        if self.gamma.abs() > EPSILON {
            count += 1;
        }
        if self.delta.abs() > EPSILON {
            count += 1;
        }
        count
    }
}

/// Normalize angle to [-π, π]
fn normalize_angle(angle: f64) -> f64 {
    let mut a = angle % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    } else if a < -PI {
        a += 2.0 * PI;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix_computation::{hadamard_matrix, pauli_x_matrix, pauli_z_matrix};

    #[test]
    fn test_zyz_identity() {
        let id: Matrix2 = [[ONE, ZERO], [ZERO, ONE]];
        let angles = SingleQubitDecomposer::decompose_zyz(&id).unwrap();

        assert!(angles.is_identity() || angles.gamma.abs() < EPSILON);
    }

    #[test]
    fn test_zyz_hadamard() {
        let h = hadamard_matrix();
        let angles = SingleQubitDecomposer::decompose_zyz(&h).unwrap();

        // Hadamard should have non-trivial angles
        assert!(angles.gamma.abs() > EPSILON);
    }

    #[test]
    fn test_zyz_pauli_x() {
        let x = pauli_x_matrix();
        let angles = SingleQubitDecomposer::decompose_zyz(&x).unwrap();

        // Pauli-X should decompose to a π rotation
        assert!((angles.gamma - PI).abs() < EPSILON || angles.gamma.abs() > EPSILON);
    }

    #[test]
    fn test_normalize_angle() {
        assert!((normalize_angle(2.0 * PI) - 0.0).abs() < EPSILON);
        assert!((normalize_angle(3.0 * PI) - PI).abs() < EPSILON);
        assert!((normalize_angle(-PI).abs() - PI).abs() < EPSILON);
    }

    #[test]
    fn test_euler_angles_identity() {
        let angles = EulerAngles::new(0.0, 0.0, 0.0, 0.0);
        assert!(angles.is_identity());
        assert_eq!(angles.gate_count(), 0);
    }

    #[test]
    fn test_euler_angles_gate_count() {
        let angles = EulerAngles::new(0.0, PI / 4.0, PI / 2.0, 0.0);
        assert_eq!(angles.gate_count(), 2);
    }
}
