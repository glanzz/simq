//! Implementations of common quantum noise channels

use super::types::{KrausOperator, NoiseChannel};
use crate::Result;
use num_complex::Complex64;

/// Depolarizing noise channel
///
/// Models random Pauli errors with probability p:
/// - With probability (1-p): no error (identity)
/// - With probability p/3 each: X, Y, or Z error
///
/// This is the most common symmetric noise model.
///
/// # Physical Interpretation
/// Represents uniform random errors from all sources.
/// After a gate, the qubit has probability p of experiencing
/// a random Pauli error.
///
/// # Kraus Operators
/// ```text
/// K₀ = √(1-p) I
/// K₁ = √(p/3) X
/// K₂ = √(p/3) Y
/// K₃ = √(p/3) Z
/// ```
///
/// # Example
/// ```
/// # use simq_core::noise::DepolarizingChannel;
/// // 1% error rate
/// let channel = DepolarizingChannel::new(0.01).unwrap();
/// assert_eq!(channel.error_probability(), 0.01);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DepolarizingChannel {
    /// Error probability p ∈ [0, 1]
    error_probability: f64,
}

impl DepolarizingChannel {
    /// Create a new depolarizing channel
    ///
    /// # Arguments
    /// * `error_probability` - Probability of error p ∈ [0, 1]
    ///
    /// # Errors
    /// Returns error if probability is not in [0, 1]
    pub fn new(error_probability: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&error_probability) {
            return Err(crate::QuantumError::ValidationError(format!(
                "Error probability must be in [0,1], got {}",
                error_probability
            )));
        }

        Ok(Self { error_probability })
    }

    /// Get the error probability
    pub fn error_probability(&self) -> f64 {
        self.error_probability
    }

    /// Pauli X matrix
    fn pauli_x() -> Vec<Complex64> {
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]
    }

    /// Pauli Y matrix
    fn pauli_y() -> Vec<Complex64> {
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ]
    }

    /// Pauli Z matrix
    fn pauli_z() -> Vec<Complex64> {
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ]
    }

    /// Identity matrix
    fn identity() -> Vec<Complex64> {
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]
    }
}

impl NoiseChannel for DepolarizingChannel {
    fn kraus_operators(&self) -> Vec<KrausOperator> {
        let p = self.error_probability;
        let sqrt_1_minus_p = (1.0 - p).sqrt();
        let sqrt_p_over_3 = (p / 3.0).sqrt();

        vec![
            // K₀ = √(1-p) I
            KrausOperator::new(
                Self::identity()
                    .iter()
                    .map(|&x| x * sqrt_1_minus_p)
                    .collect(),
                2,
            )
            .unwrap(),
            // K₁ = √(p/3) X
            KrausOperator::new(
                Self::pauli_x()
                    .iter()
                    .map(|&x| x * sqrt_p_over_3)
                    .collect(),
                2,
            )
            .unwrap(),
            // K₂ = √(p/3) Y
            KrausOperator::new(
                Self::pauli_y()
                    .iter()
                    .map(|&x| x * sqrt_p_over_3)
                    .collect(),
                2,
            )
            .unwrap(),
            // K₃ = √(p/3) Z
            KrausOperator::new(
                Self::pauli_z()
                    .iter()
                    .map(|&x| x * sqrt_p_over_3)
                    .collect(),
                2,
            )
            .unwrap(),
        ]
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "depolarizing"
    }
}

/// Amplitude damping channel
///
/// Models energy relaxation (T1 decay) where the qubit loses
/// energy to the environment.
///
/// # Physical Interpretation
/// Represents spontaneous emission: |1⟩ → |0⟩ with probability γ.
/// The excited state |1⟩ decays to ground state |0⟩.
///
/// For a qubit with T1 relaxation time, after time t:
/// γ = 1 - exp(-t/T1)
///
/// # Kraus Operators
/// ```text
/// K₀ = [[1, 0], [0, √(1-γ)]]
/// K₁ = [[0, √γ], [0, 0]]
/// ```
///
/// # Example
/// ```
/// # use simq_core::noise::AmplitudeDamping;
/// // 2% decay probability
/// let channel = AmplitudeDamping::new(0.02).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AmplitudeDamping {
    /// Decay probability γ ∈ [0, 1]
    gamma: f64,
}

impl AmplitudeDamping {
    /// Create a new amplitude damping channel
    ///
    /// # Arguments
    /// * `gamma` - Decay probability γ ∈ [0, 1]
    ///
    /// # Errors
    /// Returns error if gamma is not in [0, 1]
    pub fn new(gamma: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&gamma) {
            return Err(crate::QuantumError::ValidationError(format!(
                "Gamma must be in [0,1], got {}",
                gamma
            )));
        }

        Ok(Self { gamma })
    }

    /// Create from T1 relaxation time and gate duration
    ///
    /// # Arguments
    /// * `t1` - T1 relaxation time (same units as gate_time)
    /// * `gate_time` - Duration of the gate operation
    ///
    /// Computes γ = 1 - exp(-gate_time/T1)
    pub fn from_t1(t1: f64, gate_time: f64) -> Result<Self> {
        if t1 <= 0.0 {
            return Err(crate::QuantumError::ValidationError(
                "T1 must be positive".to_string(),
            ));
        }
        if gate_time < 0.0 {
            return Err(crate::QuantumError::ValidationError(
                "Gate time must be non-negative".to_string(),
            ));
        }

        let gamma = 1.0 - (-gate_time / t1).exp();
        Self::new(gamma)
    }

    /// Get the decay probability
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

impl NoiseChannel for AmplitudeDamping {
    fn kraus_operators(&self) -> Vec<KrausOperator> {
        let gamma = self.gamma;
        let sqrt_gamma = gamma.sqrt();
        let sqrt_1_minus_gamma = (1.0 - gamma).sqrt();

        vec![
            // K₀ = [[1, 0], [0, √(1-γ)]]
            KrausOperator::new(
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(sqrt_1_minus_gamma, 0.0),
                ],
                2,
            )
            .unwrap(),
            // K₁ = [[0, √γ], [0, 0]]
            KrausOperator::new(
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(sqrt_gamma, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
                2,
            )
            .unwrap(),
        ]
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "amplitude_damping"
    }
}

/// Phase damping channel
///
/// Models pure dephasing (T2 decay) where the qubit loses phase
/// information without energy loss.
///
/// # Physical Interpretation
/// Represents random Z rotations that destroy quantum coherence.
/// The qubit randomly acquires a phase flip with probability λ.
///
/// For a qubit with T2 dephasing time (pure dephasing T2*), after time t:
/// λ = (1 - exp(-t/T2))/2
///
/// Note: Total dephasing T2 relates to T1 and pure dephasing T2*:
/// 1/T2 = 1/(2T1) + 1/T2*
///
/// # Kraus Operators
/// ```text
/// K₀ = √(1-λ) I
/// K₁ = √λ Z
/// ```
///
/// # Example
/// ```
/// # use simq_core::noise::PhaseDamping;
/// // 1.5% dephasing probability
/// let channel = PhaseDamping::new(0.015).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PhaseDamping {
    /// Dephasing probability λ ∈ [0, 0.5]
    lambda: f64,
}

impl PhaseDamping {
    /// Create a new phase damping channel
    ///
    /// # Arguments
    /// * `lambda` - Dephasing probability λ ∈ [0, 0.5]
    ///
    /// # Errors
    /// Returns error if lambda is not in [0, 0.5]
    pub fn new(lambda: f64) -> Result<Self> {
        if !(0.0..=0.5).contains(&lambda) {
            return Err(crate::QuantumError::ValidationError(format!(
                "Lambda must be in [0, 0.5], got {}",
                lambda
            )));
        }

        Ok(Self { lambda })
    }

    /// Create from T2 dephasing time and gate duration
    ///
    /// # Arguments
    /// * `t2` - T2 dephasing time (same units as gate_time)
    /// * `gate_time` - Duration of the gate operation
    ///
    /// Computes λ = (1 - exp(-gate_time/T2))/2
    pub fn from_t2(t2: f64, gate_time: f64) -> Result<Self> {
        if t2 <= 0.0 {
            return Err(crate::QuantumError::ValidationError(
                "T2 must be positive".to_string(),
            ));
        }
        if gate_time < 0.0 {
            return Err(crate::QuantumError::ValidationError(
                "Gate time must be non-negative".to_string(),
            ));
        }

        let lambda = (1.0 - (-gate_time / t2).exp()) / 2.0;
        Self::new(lambda)
    }

    /// Get the dephasing probability
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl NoiseChannel for PhaseDamping {
    fn kraus_operators(&self) -> Vec<KrausOperator> {
        let lambda = self.lambda;
        let sqrt_1_minus_lambda = (1.0 - lambda).sqrt();
        let sqrt_lambda = lambda.sqrt();

        vec![
            // K₀ = √(1-λ) I
            KrausOperator::new(
                vec![
                    Complex64::new(sqrt_1_minus_lambda, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(sqrt_1_minus_lambda, 0.0),
                ],
                2,
            )
            .unwrap(),
            // K₁ = √λ Z
            KrausOperator::new(
                vec![
                    Complex64::new(sqrt_lambda, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-sqrt_lambda, 0.0),
                ],
                2,
            )
            .unwrap(),
        ]
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "phase_damping"
    }
}

/// Readout error channel
///
/// Models measurement bit-flip errors where:
/// - |0⟩ is measured as |1⟩ with probability p01
/// - |1⟩ is measured as |0⟩ with probability p10
///
/// # Physical Interpretation
/// Real measurement devices have finite fidelity. They may
/// incorrectly report the measured bit value.
///
/// Common values:
/// - High-fidelity superconducting qubits: p01, p10 ≈ 0.01-0.03
/// - Trapped ions: p01, p10 ≈ 0.001
///
/// # Kraus Operators
/// This channel is applied classically to measurement outcomes,
/// not via Kraus operators during state evolution.
///
/// # Example
/// ```
/// # use simq_core::noise::ReadoutError;
/// // 2% false positive, 3% false negative
/// let channel = ReadoutError::new(0.02, 0.03).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ReadoutError {
    /// Probability of measuring |1⟩ when state is |0⟩
    p01: f64,
    /// Probability of measuring |0⟩ when state is |1⟩
    p10: f64,
}

impl ReadoutError {
    /// Create a new readout error channel
    ///
    /// # Arguments
    /// * `p01` - P(measure 1 | state is 0) ∈ [0, 1]
    /// * `p10` - P(measure 0 | state is 1) ∈ [0, 1]
    ///
    /// # Errors
    /// Returns error if probabilities are not in [0, 1]
    pub fn new(p01: f64, p10: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&p01) {
            return Err(crate::QuantumError::ValidationError(format!(
                "p01 must be in [0,1], got {}",
                p01
            )));
        }
        if !(0.0..=1.0).contains(&p10) {
            return Err(crate::QuantumError::ValidationError(format!(
                "p10 must be in [0,1], got {}",
                p10
            )));
        }

        Ok(Self { p01, p10 })
    }

    /// Create a symmetric readout error channel
    ///
    /// # Arguments
    /// * `error_rate` - Symmetric error probability ∈ [0, 1]
    ///
    /// Sets p01 = p10 = error_rate
    pub fn symmetric(error_rate: f64) -> Result<Self> {
        Self::new(error_rate, error_rate)
    }

    /// Get the 0→1 error probability
    pub fn p01(&self) -> f64 {
        self.p01
    }

    /// Get the 1→0 error probability
    pub fn p10(&self) -> f64 {
        self.p10
    }

    /// Get the average error rate
    pub fn average_error(&self) -> f64 {
        (self.p01 + self.p10) / 2.0
    }

    /// Apply readout error to a classical bit
    ///
    /// Returns true if the bit should be flipped
    pub fn should_flip(&self, measured_bit: bool, random_value: f64) -> bool {
        if measured_bit {
            // Measured 1, flip to 0 with probability p10
            random_value < self.p10
        } else {
            // Measured 0, flip to 1 with probability p01
            random_value < self.p01
        }
    }
}

impl NoiseChannel for ReadoutError {
    fn kraus_operators(&self) -> Vec<KrausOperator> {
        // Readout errors are applied classically to measurement outcomes
        // For compatibility with the NoiseChannel trait, we return
        // identity-like operators. The actual error is applied during
        // measurement post-processing.
        vec![KrausOperator::new(
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            2,
        )
        .unwrap()]
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "readout_error"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;

    #[test]
    fn test_depolarizing_channel() {
        let channel = DepolarizingChannel::new(0.1).unwrap();
        assert_eq!(channel.error_probability(), 0.1);
        assert_eq!(channel.num_qubits(), 1);
        assert_eq!(channel.name(), "depolarizing");

        let ops = channel.kraus_operators();
        assert_eq!(ops.len(), 4);

        // Verify completeness
        assert!(channel.verify_completeness(TOLERANCE));
    }

    #[test]
    fn test_depolarizing_invalid_probability() {
        assert!(DepolarizingChannel::new(-0.1).is_err());
        assert!(DepolarizingChannel::new(1.1).is_err());
    }

    #[test]
    fn test_amplitude_damping_channel() {
        let channel = AmplitudeDamping::new(0.05).unwrap();
        assert_eq!(channel.gamma(), 0.05);
        assert_eq!(channel.num_qubits(), 1);
        assert_eq!(channel.name(), "amplitude_damping");

        let ops = channel.kraus_operators();
        assert_eq!(ops.len(), 2);

        // Verify completeness
        assert!(channel.verify_completeness(TOLERANCE));
    }

    #[test]
    fn test_amplitude_damping_from_t1() {
        let t1 = 50.0; // μs
        let gate_time = 0.1; // μs
        let channel = AmplitudeDamping::from_t1(t1, gate_time).unwrap();

        let expected_gamma = 1.0 - (-gate_time / t1).exp();
        assert!((channel.gamma() - expected_gamma).abs() < TOLERANCE);
    }

    #[test]
    fn test_phase_damping_channel() {
        let channel = PhaseDamping::new(0.03).unwrap();
        assert_eq!(channel.lambda(), 0.03);
        assert_eq!(channel.num_qubits(), 1);
        assert_eq!(channel.name(), "phase_damping");

        let ops = channel.kraus_operators();
        assert_eq!(ops.len(), 2);

        // Verify completeness
        assert!(channel.verify_completeness(TOLERANCE));
    }

    #[test]
    fn test_phase_damping_from_t2() {
        let t2 = 100.0; // μs
        let gate_time = 0.2; // μs
        let channel = PhaseDamping::from_t2(t2, gate_time).unwrap();

        let expected_lambda = (1.0 - (-gate_time / t2).exp()) / 2.0;
        assert!((channel.lambda() - expected_lambda).abs() < TOLERANCE);
    }

    #[test]
    fn test_phase_damping_invalid_lambda() {
        assert!(PhaseDamping::new(-0.1).is_err());
        assert!(PhaseDamping::new(0.6).is_err()); // > 0.5
    }

    #[test]
    fn test_readout_error_channel() {
        let channel = ReadoutError::new(0.02, 0.03).unwrap();
        assert_eq!(channel.p01(), 0.02);
        assert_eq!(channel.p10(), 0.03);
        assert_eq!(channel.average_error(), 0.025);
        assert_eq!(channel.num_qubits(), 1);
        assert_eq!(channel.name(), "readout_error");
    }

    #[test]
    fn test_readout_error_symmetric() {
        let channel = ReadoutError::symmetric(0.02).unwrap();
        assert_eq!(channel.p01(), 0.02);
        assert_eq!(channel.p10(), 0.02);
    }

    #[test]
    fn test_readout_error_should_flip() {
        let channel = ReadoutError::new(0.1, 0.2).unwrap();

        // Measured 0, random < p01 → flip
        assert!(channel.should_flip(false, 0.05));
        // Measured 0, random >= p01 → don't flip
        assert!(!channel.should_flip(false, 0.15));

        // Measured 1, random < p10 → flip
        assert!(channel.should_flip(true, 0.15));
        // Measured 1, random >= p10 → don't flip
        assert!(!channel.should_flip(true, 0.25));
    }

    #[test]
    fn test_readout_error_invalid_probabilities() {
        assert!(ReadoutError::new(-0.1, 0.1).is_err());
        assert!(ReadoutError::new(0.1, 1.1).is_err());
    }
}
