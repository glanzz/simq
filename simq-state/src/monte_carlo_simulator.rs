//! Monte Carlo quantum simulator with trajectory-based noise
//!
//! This simulator uses pure state vectors and stochastically samples
//! noise trajectories, providing memory-efficient noisy simulation.
//!
//! # Comparison: Monte Carlo vs Density Matrix
//!
//! | Aspect | Monte Carlo | Density Matrix |
//! |--------|-------------|----------------|
//! | Memory | O(2^n) | O(4^n) |
//! | State | Pure (each trajectory) | Mixed |
//! | Noise | Sampled randomly | Exact average |
//! | Convergence | √N for N trajectories | Exact |
//! | Max qubits | ~20 | ~10 |
//!
//! # Example
//!
//! ```ignore
//! use simq_state::{MonteCarloSimulator, MonteCarloConfig};
//! use simq_core::noise::DepolarizingMC;
//!
//! let config = MonteCarloConfig::new()
//!     .with_trajectories(1000)
//!     .with_seed(42);
//!
//! let mut sim = MonteCarloSimulator::new(2, config)?;
//!
//! // Apply gate
//! sim.apply_gate(&hadamard, &[0])?;
//!
//! // Apply stochastic noise
//! let noise = DepolarizingMC::from_probability(0.01)?;
//! sim.apply_stochastic_noise(&noise, 0)?;
//!
//! // Run many trajectories and average
//! let results = sim.run_trajectories()?;
//! ```

use crate::dense_state::DenseState;
use crate::error::{Result, StateError};
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use simq_core::noise::{MonteCarloSampler, PauliOperation};

/// Configuration for Monte Carlo simulation
#[derive(Debug, Clone)]
pub struct MonteCarloConfig {
    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// Number of trajectories to average over
    pub trajectories: usize,

    /// Number of measurement shots per trajectory
    pub shots_per_trajectory: usize,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            seed: None,
            trajectories: 1000,
            shots_per_trajectory: 1,
        }
    }
}

impl MonteCarloConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_trajectories(mut self, trajectories: usize) -> Self {
        self.trajectories = trajectories;
        self
    }

    pub fn with_shots_per_trajectory(mut self, shots: usize) -> Self {
        self.shots_per_trajectory = shots;
        self
    }
}

/// Monte Carlo quantum simulator
///
/// Uses stochastic sampling of noise channels with pure state vectors.
pub struct MonteCarloSimulator {
    /// Current state vector (pure state)
    state: DenseState,

    /// Initial state (for resetting between trajectories)
    initial_state: Vec<Complex64>,

    /// Configuration
    config: MonteCarloConfig,

    /// Random number generator
    rng: StdRng,

    /// Statistics
    gate_count: usize,
    noise_ops_applied: usize,
    current_trajectory: usize,
}

impl MonteCarloSimulator {
    /// Create a new Monte Carlo simulator
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    /// * `config` - Simulation configuration
    pub fn new(num_qubits: usize, config: MonteCarloConfig) -> Result<Self> {
        let state = DenseState::new(num_qubits)?;
        let initial_state = state.amplitudes().to_vec();

        let rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        Ok(Self {
            state,
            initial_state,
            config,
            rng,
            gate_count: 0,
            noise_ops_applied: 0,
            current_trajectory: 0,
        })
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.state.num_qubits()
    }

    /// Get reference to current state
    pub fn state(&self) -> &DenseState {
        &self.state
    }

    /// Apply a unitary gate
    ///
    /// # Arguments
    /// * `unitary` - Gate matrix (2×2 for 1-qubit)
    /// * `qubits` - Target qubit indices
    pub fn apply_gate(&mut self, unitary: &[[Complex64; 2]; 2], qubits: &[usize]) -> Result<()> {
        if qubits.len() == 1 {
            self.state.apply_single_qubit_gate(unitary, qubits[0])?;
            self.gate_count += 1;
            Ok(())
        } else {
            Err(StateError::InvalidDimension {
                dimension: qubits.len(),
            })
        }
    }

    /// Apply stochastic noise using Monte Carlo sampling
    ///
    /// Randomly samples which error occurs and applies it.
    ///
    /// # Arguments
    /// * `sampler` - Monte Carlo noise sampler
    /// * `qubit` - Target qubit
    pub fn apply_stochastic_noise(
        &mut self,
        sampler: &dyn MonteCarloSampler,
        qubit: usize,
    ) -> Result<()> {
        let random_value = self.rng.gen::<f64>();
        let index = sampler.sample(random_value);
        let operation = sampler.get_operation(index);

        self.apply_pauli_operation(operation, qubit)?;
        self.noise_ops_applied += 1;

        Ok(())
    }

    /// Apply a Pauli operation from Monte Carlo sampling
    fn apply_pauli_operation(&mut self, operation: PauliOperation, qubit: usize) -> Result<()> {
        match operation {
            PauliOperation::Identity => {
                // No-op
                Ok(())
            },
            PauliOperation::X => {
                // Pauli X (bit flip)
                let x_gate = [
                    [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                ];
                self.state.apply_single_qubit_gate(&x_gate, qubit)
            },
            PauliOperation::Y => {
                // Pauli Y
                let y_gate = [
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
                    [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
                ];
                self.state.apply_single_qubit_gate(&y_gate, qubit)
            },
            PauliOperation::Z => {
                // Pauli Z (phase flip)
                let z_gate = [
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
                ];
                self.state.apply_single_qubit_gate(&z_gate, qubit)
            },
            PauliOperation::JumpToZero { sqrt_gamma: _ } => {
                // For amplitude damping: project to |0⟩ and renormalize
                // This is state-dependent, handled specially
                self.apply_amplitude_damping_jump(qubit)
            },
            PauliOperation::NoJump { sqrt_1_minus_gamma } => {
                // For amplitude damping: apply decay to |1⟩ component
                self.apply_amplitude_damping_no_jump(qubit, sqrt_1_minus_gamma)
            },
        }
    }

    /// Apply amplitude damping jump operation
    fn apply_amplitude_damping_jump(&mut self, qubit: usize) -> Result<()> {
        // Project to |0⟩ on target qubit and renormalize
        let amplitudes = self.state.amplitudes_mut();
        let stride = 1 << qubit;

        let mut norm_sq = 0.0;

        // Zero out |1⟩ components and accumulate norm
        for (i, amp) in amplitudes.iter_mut().enumerate() {
            if (i & stride) != 0 {
                // This is a |1⟩ component - zero it
                *amp = Complex64::new(0.0, 0.0);
            } else {
                norm_sq += amp.norm_sqr();
            }
        }

        // Renormalize
        if norm_sq > 1e-15 {
            let norm = norm_sq.sqrt();
            for amp in amplitudes.iter_mut() {
                *amp /= norm;
            }
        }

        Ok(())
    }

    /// Apply amplitude damping no-jump operation
    fn apply_amplitude_damping_no_jump(&mut self, qubit: usize, factor: f64) -> Result<()> {
        // Scale the |1⟩ components and renormalize
        let amplitudes = self.state.amplitudes_mut();
        let stride = 1 << qubit;

        let mut norm_sq = 0.0;

        for (i, amp) in amplitudes.iter_mut().enumerate() {
            if (i & stride) != 0 {
                // This is a |1⟩ component - scale it
                *amp *= factor;
            }
            norm_sq += amp.norm_sqr();
        }

        // Renormalize
        if norm_sq > 1e-15 {
            let norm = norm_sq.sqrt();
            for amp in amplitudes.iter_mut() {
                *amp /= norm;
            }
        }

        Ok(())
    }

    /// Measure all qubits
    pub fn measure_all(&mut self) -> Result<Vec<bool>> {
        let mut results = Vec::with_capacity(self.num_qubits());

        for qubit in 0..self.num_qubits() {
            let random_value = self.rng.gen::<f64>();
            let outcome = self.state.measure_qubit(qubit, random_value)?;
            results.push(outcome != 0);
        }

        Ok(results)
    }

    /// Reset to initial state
    pub fn reset_to_initial(&mut self) -> Result<()> {
        self.state = DenseState::from_amplitudes(self.num_qubits(), &self.initial_state)?;
        self.gate_count = 0;
        self.noise_ops_applied = 0;
        Ok(())
    }

    /// Get simulation statistics
    pub fn stats(&self) -> MonteCarloStats {
        MonteCarloStats {
            gate_count: self.gate_count,
            noise_ops_applied: self.noise_ops_applied,
            current_trajectory: self.current_trajectory,
            total_trajectories: self.config.trajectories,
        }
    }
}

/// Monte Carlo simulation statistics
#[derive(Debug, Clone)]
pub struct MonteCarloStats {
    pub gate_count: usize,
    pub noise_ops_applied: usize,
    pub current_trajectory: usize,
    pub total_trajectories: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_core::noise::{DepolarizingMC, PhaseDampingMC};

    #[test]
    fn test_simulator_creation() {
        let config = MonteCarloConfig::new().with_seed(42);
        let sim = MonteCarloSimulator::new(2, config).unwrap();
        assert_eq!(sim.num_qubits(), 2);
    }

    #[test]
    fn test_apply_gate() {
        let config = MonteCarloConfig::new().with_seed(42);
        let mut sim = MonteCarloSimulator::new(1, config).unwrap();

        // Hadamard gate
        let h = [
            [
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            ],
            [
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
            ],
        ];

        sim.apply_gate(&h, &[0]).unwrap();
        assert_eq!(sim.stats().gate_count, 1);
    }

    #[test]
    fn test_stochastic_noise_depolarizing() {
        let config = MonteCarloConfig::new().with_seed(42);
        let mut sim = MonteCarloSimulator::new(1, config).unwrap();

        let noise = DepolarizingMC::from_probability(0.5).unwrap();
        sim.apply_stochastic_noise(&noise, 0).unwrap();

        assert_eq!(sim.stats().noise_ops_applied, 1);
    }

    #[test]
    fn test_stochastic_noise_phase_damping() {
        let config = MonteCarloConfig::new().with_seed(42);
        let mut sim = MonteCarloSimulator::new(1, config).unwrap();

        let noise = PhaseDampingMC::from_lambda(0.1).unwrap();
        sim.apply_stochastic_noise(&noise, 0).unwrap();

        assert_eq!(sim.stats().noise_ops_applied, 1);
    }

    #[test]
    fn test_measurement() {
        let config = MonteCarloConfig::new().with_seed(42).with_trajectories(100);
        let mut sim = MonteCarloSimulator::new(2, config).unwrap();

        // Measure |00⟩ state
        let results = sim.measure_all().unwrap();
        assert_eq!(results.len(), 2);
        assert!(!results[0]);
        assert!(!results[1]);
    }

    #[test]
    fn test_reset() {
        let config = MonteCarloConfig::new().with_seed(42);
        let mut sim = MonteCarloSimulator::new(1, config).unwrap();

        // Apply gate
        let h = [
            [
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            ],
            [
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
            ],
        ];
        sim.apply_gate(&h, &[0]).unwrap();

        // Reset
        sim.reset_to_initial().unwrap();
        assert_eq!(sim.stats().gate_count, 0);
    }
}
