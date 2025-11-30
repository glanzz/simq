//! Monte Carlo noise sampling for efficient trajectory-based simulation
//!
//! Instead of evolving full density matrices, Monte Carlo methods sample
//! noise trajectories and average over many runs. This is much more
//! memory-efficient for large systems.
//!
//! # Approach
//!
//! For a Kraus channel {K_i}, instead of computing ρ' = Σ K_i ρ K_i†,
//! we:
//! 1. Sample an operator K_i with probability p_i = Tr(K_i† K_i ρ)
//! 2. Apply it to the state: |ψ'⟩ = K_i|ψ⟩ / ||K_i|ψ⟩||
//! 3. Repeat for many trajectories and average results
//!
//! # Example
//!
//! ```ignore
//! use simq_core::noise::{DepolarizingChannel, MonteCarlo};
//!
//! let depol = DepolarizingChannel::new(0.01)?;
//! let sampler = MonteCarlo::new(depol);
//!
//! // Sample which error occurred
//! let error = sampler.sample(0.5); // random value ∈ [0,1)
//! ```

use super::channels::{AmplitudeDamping, DepolarizingChannel, PhaseDamping, ReadoutError};
use crate::Result;

/// Monte Carlo sampler for noise channels
///
/// Converts Kraus operators into probabilistic single-trajectory operations.
pub trait MonteCarloSampler: Send + Sync {
    /// Sample which Kraus operator to apply
    ///
    /// # Arguments
    /// * `random_value` - Random number in [0, 1)
    ///
    /// # Returns
    /// Index of Kraus operator to apply
    fn sample(&self, random_value: f64) -> usize;

    /// Get the Pauli operation to apply for the sampled index
    ///
    /// Returns (pauli_type, normalization_factor) where:
    /// - pauli_type: 0=I, 1=X, 2=Y, 3=Z
    /// - normalization_factor: Scaling factor for the operation
    fn get_operation(&self, index: usize) -> PauliOperation;

    /// Number of possible outcomes
    fn num_outcomes(&self) -> usize;
}

/// Pauli operation for Monte Carlo sampling
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PauliOperation {
    /// Identity (no error)
    Identity,
    /// Pauli X (bit flip)
    X,
    /// Pauli Y
    Y,
    /// Pauli Z (phase flip)
    Z,
    /// Jump to ground state |0⟩ (amplitude damping)
    JumpToZero { sqrt_gamma: f64 },
    /// No jump, but rescale (amplitude damping)
    NoJump { sqrt_1_minus_gamma: f64 },
}

/// Monte Carlo sampler for depolarizing channel
///
/// Samples uniformly from {I, X, Y, Z} with appropriate probabilities.
pub struct DepolarizingMC {
    /// Error probability p
    error_prob: f64,
}

impl DepolarizingMC {
    /// Create new depolarizing Monte Carlo sampler
    pub fn new(channel: &DepolarizingChannel) -> Self {
        Self {
            error_prob: channel.error_probability(),
        }
    }

    /// Create from error probability directly
    pub fn from_probability(error_prob: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&error_prob) {
            return Err(crate::QuantumError::ValidationError(format!(
                "Error probability must be in [0,1], got {}",
                error_prob
            )));
        }
        Ok(Self { error_prob })
    }
}

impl MonteCarloSampler for DepolarizingMC {
    fn sample(&self, random_value: f64) -> usize {
        let p = self.error_prob;

        if random_value < 1.0 - p {
            0 // Identity
        } else if random_value < 1.0 - p + p / 3.0 {
            1 // X
        } else if random_value < 1.0 - p + 2.0 * p / 3.0 {
            2 // Y
        } else {
            3 // Z
        }
    }

    fn get_operation(&self, index: usize) -> PauliOperation {
        match index {
            0 => PauliOperation::Identity,
            1 => PauliOperation::X,
            2 => PauliOperation::Y,
            3 => PauliOperation::Z,
            _ => PauliOperation::Identity,
        }
    }

    fn num_outcomes(&self) -> usize {
        4
    }
}

/// Monte Carlo sampler for amplitude damping
///
/// Samples between "jump to |0⟩" and "no jump" outcomes.
pub struct AmplitudeDampingMC {
    /// Decay probability γ
    gamma: f64,
}

impl AmplitudeDampingMC {
    /// Create new amplitude damping Monte Carlo sampler
    pub fn new(channel: &AmplitudeDamping) -> Self {
        Self {
            gamma: channel.gamma(),
        }
    }

    /// Create from gamma directly
    pub fn from_gamma(gamma: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&gamma) {
            return Err(crate::QuantumError::ValidationError(format!(
                "Gamma must be in [0,1], got {}",
                gamma
            )));
        }
        Ok(Self { gamma })
    }

    /// Get the decay probability
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

impl MonteCarloSampler for AmplitudeDampingMC {
    fn sample(&self, random_value: f64) -> usize {
        // For amplitude damping, the sampling depends on the current state
        // We'll handle this specially in the simulator
        // For now, return which Kraus operator to try
        if random_value < self.gamma {
            1 // Jump operator
        } else {
            0 // No-jump operator
        }
    }

    fn get_operation(&self, index: usize) -> PauliOperation {
        match index {
            0 => PauliOperation::NoJump {
                sqrt_1_minus_gamma: (1.0 - self.gamma).sqrt(),
            },
            1 => PauliOperation::JumpToZero {
                sqrt_gamma: self.gamma.sqrt(),
            },
            _ => PauliOperation::Identity,
        }
    }

    fn num_outcomes(&self) -> usize {
        2
    }
}

/// Monte Carlo sampler for phase damping
///
/// Samples between "no error" and "Z error".
pub struct PhaseDampingMC {
    /// Dephasing probability λ
    lambda: f64,
}

impl PhaseDampingMC {
    /// Create new phase damping Monte Carlo sampler
    pub fn new(channel: &PhaseDamping) -> Self {
        Self {
            lambda: channel.lambda(),
        }
    }

    /// Create from lambda directly
    pub fn from_lambda(lambda: f64) -> Result<Self> {
        if !(0.0..=0.5).contains(&lambda) {
            return Err(crate::QuantumError::ValidationError(format!(
                "Lambda must be in [0, 0.5], got {}",
                lambda
            )));
        }
        Ok(Self { lambda })
    }
}

impl MonteCarloSampler for PhaseDampingMC {
    fn sample(&self, random_value: f64) -> usize {
        if random_value < 1.0 - self.lambda {
            0 // Identity
        } else {
            1 // Z
        }
    }

    fn get_operation(&self, index: usize) -> PauliOperation {
        match index {
            0 => PauliOperation::Identity,
            1 => PauliOperation::Z,
            _ => PauliOperation::Identity,
        }
    }

    fn num_outcomes(&self) -> usize {
        2
    }
}

/// Monte Carlo sampler for readout errors
///
/// Flips measurement outcomes probabilistically.
pub struct ReadoutErrorMC {
    /// P(measure 1 | state is 0)
    p01: f64,
    /// P(measure 0 | state is 1)
    p10: f64,
}

impl ReadoutErrorMC {
    /// Create new readout error Monte Carlo sampler
    pub fn new(channel: &ReadoutError) -> Self {
        Self {
            p01: channel.p01(),
            p10: channel.p10(),
        }
    }

    /// Create from error probabilities directly
    pub fn from_probabilities(p01: f64, p10: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&p01) || !(0.0..=1.0).contains(&p10) {
            return Err(crate::QuantumError::ValidationError(
                "Readout error probabilities must be in [0,1]".to_string(),
            ));
        }
        Ok(Self { p01, p10 })
    }

    /// Apply readout error to a measurement outcome
    ///
    /// # Arguments
    /// * `measured_bit` - The ideal measurement outcome
    /// * `random_value` - Random number in [0, 1)
    ///
    /// # Returns
    /// The noisy measurement outcome (possibly flipped)
    pub fn apply_to_measurement(&self, measured_bit: bool, random_value: f64) -> bool {
        if measured_bit {
            // Measured 1, flip to 0 with probability p10
            random_value >= self.p10
        } else {
            // Measured 0, flip to 1 with probability p01
            random_value < self.p01
        }
    }

    /// Get error probabilities
    pub fn probabilities(&self) -> (f64, f64) {
        (self.p01, self.p10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depolarizing_mc_sampling() {
        let mc = DepolarizingMC::from_probability(0.3).unwrap();

        // With p=0.3:
        // 0.0-0.7: Identity (70%)
        // 0.7-0.8: X (10%)
        // 0.8-0.9: Y (10%)
        // 0.9-1.0: Z (10%)

        assert_eq!(mc.sample(0.0), 0); // Identity
        assert_eq!(mc.sample(0.69), 0); // Still identity
        assert_eq!(mc.sample(0.75), 1); // X error
        assert_eq!(mc.sample(0.85), 2); // Y error
        assert_eq!(mc.sample(0.95), 3); // Z error

        assert_eq!(mc.num_outcomes(), 4);
    }

    #[test]
    fn test_depolarizing_mc_operations() {
        let mc = DepolarizingMC::from_probability(0.1).unwrap();

        assert_eq!(mc.get_operation(0), PauliOperation::Identity);
        assert_eq!(mc.get_operation(1), PauliOperation::X);
        assert_eq!(mc.get_operation(2), PauliOperation::Y);
        assert_eq!(mc.get_operation(3), PauliOperation::Z);
    }

    #[test]
    fn test_amplitude_damping_mc() {
        let mc = AmplitudeDampingMC::from_gamma(0.1).unwrap();

        assert_eq!(mc.sample(0.05), 1); // Jump
        assert_eq!(mc.sample(0.15), 0); // No jump

        assert_eq!(mc.num_outcomes(), 2);

        match mc.get_operation(0) {
            PauliOperation::NoJump { sqrt_1_minus_gamma } => {
                assert!((sqrt_1_minus_gamma - (0.9_f64).sqrt()).abs() < 1e-10);
            },
            _ => panic!("Expected NoJump"),
        }

        match mc.get_operation(1) {
            PauliOperation::JumpToZero { sqrt_gamma } => {
                assert!((sqrt_gamma - (0.1_f64).sqrt()).abs() < 1e-10);
            },
            _ => panic!("Expected JumpToZero"),
        }
    }

    #[test]
    fn test_phase_damping_mc() {
        let mc = PhaseDampingMC::from_lambda(0.2).unwrap();

        assert_eq!(mc.sample(0.5), 0); // Identity
        assert_eq!(mc.sample(0.85), 1); // Z error

        assert_eq!(mc.num_outcomes(), 2);
    }

    #[test]
    fn test_readout_error_mc() {
        let mc = ReadoutErrorMC::from_probabilities(0.02, 0.03).unwrap();

        // Measure 0, flip with p=0.02
        assert!(mc.apply_to_measurement(false, 0.01)); // Flipped
        assert!(!mc.apply_to_measurement(false, 0.03)); // Not flipped

        // Measure 1, flip with p=0.03
        assert!(!mc.apply_to_measurement(true, 0.02)); // Flipped
        assert!(mc.apply_to_measurement(true, 0.04)); // Not flipped

        assert_eq!(mc.probabilities(), (0.02, 0.03));
    }

    #[test]
    fn test_invalid_probabilities() {
        assert!(DepolarizingMC::from_probability(-0.1).is_err());
        assert!(DepolarizingMC::from_probability(1.1).is_err());
        assert!(AmplitudeDampingMC::from_gamma(1.5).is_err());
        assert!(PhaseDampingMC::from_lambda(0.6).is_err());
    }
}
