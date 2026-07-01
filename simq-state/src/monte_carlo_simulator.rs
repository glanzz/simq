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

    // ---- New coverage tests ----

    #[test]
    fn test_config_without_seed() {
        // MonteCarloConfig::new() without with_seed — constructs from entropy
        let config = MonteCarloConfig::new();
        assert!(config.seed.is_none());
        assert_eq!(config.trajectories, 1000);
        // Simulator creation should succeed
        let sim = MonteCarloSimulator::new(2, config).unwrap();
        assert_eq!(sim.num_qubits(), 2);
    }

    #[test]
    fn test_config_with_shots_per_trajectory() {
        let config = MonteCarloConfig::new().with_shots_per_trajectory(5);
        assert_eq!(config.shots_per_trajectory, 5);
    }

    #[test]
    fn test_apply_gate_wrong_qubit_count_returns_error() {
        // Passing a 2-element qubits array to apply_gate (which only supports 1) → Err
        let config = MonteCarloConfig::new().with_seed(42);
        let mut sim = MonteCarloSimulator::new(2, config).unwrap();

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

        let result = sim.apply_gate(&h, &[0, 1]);
        assert!(result.is_err());
        match result.unwrap_err() {
            StateError::InvalidDimension { dimension } => assert_eq!(dimension, 2),
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn test_noise_ops_increments_each_call() {
        let config = MonteCarloConfig::new().with_seed(99);
        let mut sim = MonteCarloSimulator::new(1, config).unwrap();

        let noise = DepolarizingMC::from_probability(0.1).unwrap();
        sim.apply_stochastic_noise(&noise, 0).unwrap();
        assert_eq!(sim.stats().noise_ops_applied, 1);
        sim.apply_stochastic_noise(&noise, 0).unwrap();
        assert_eq!(sim.stats().noise_ops_applied, 2);
    }

    #[test]
    fn test_measure_all_returns_num_qubits_results() {
        // Hadamard-prepared state: measure_all length == num_qubits
        let config = MonteCarloConfig::new().with_seed(42);
        let mut sim = MonteCarloSimulator::new(3, config).unwrap();

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
        let results = sim.measure_all().unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_stats_trajectory_count() {
        let config = MonteCarloConfig::new().with_seed(42).with_trajectories(500);
        let sim = MonteCarloSimulator::new(2, config).unwrap();
        let stats = sim.stats();
        assert_eq!(stats.total_trajectories, 500);
        assert_eq!(stats.current_trajectory, 0);
    }

    #[test]
    fn test_reset_preserves_initial_state() {
        // reset_to_initial restores state across multiple calls
        let config = MonteCarloConfig::new().with_seed(42);
        let mut sim = MonteCarloSimulator::new(2, config).unwrap();

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
        sim.apply_gate(&h, &[1]).unwrap();
        assert_eq!(sim.stats().gate_count, 2);

        sim.reset_to_initial().unwrap();
        assert_eq!(sim.stats().gate_count, 0);
        assert_eq!(sim.stats().noise_ops_applied, 0);

        // Second reset after no changes — still OK
        sim.reset_to_initial().unwrap();
        assert_eq!(sim.stats().gate_count, 0);
    }

    #[test]
    fn test_amplitude_damping_mc_noise_increments_counter() {
        use simq_core::noise::AmplitudeDampingMC;
        let config = MonteCarloConfig::new().with_seed(7);
        let mut sim = MonteCarloSimulator::new(1, config).unwrap();

        let noise = AmplitudeDampingMC::from_gamma(0.1).unwrap();
        sim.apply_stochastic_noise(&noise, 0).unwrap();
        assert_eq!(sim.stats().noise_ops_applied, 1);
    }

    // ---- Deterministic branch coverage for apply_pauli_operation and
    // amplitude-damping helpers. We call the private methods directly
    // (accessible via `use super::*` in the same crate) to avoid relying on
    // RNG draws landing in a specific sampler range. ----

    #[test]
    fn test_apply_pauli_operation_y_directly() {
        // Forces PauliOperation::Y branch (lines applying the Y gate).
        let config = MonteCarloConfig::new().with_seed(1);
        let mut sim = MonteCarloSimulator::new(1, config).unwrap();

        sim.apply_pauli_operation(PauliOperation::Y, 0).unwrap();

        // Y|0> = i|1>, so amplitude at index 1 should have norm ~1 and index 0 ~0.
        let amps = sim.state().amplitudes();
        assert!(amps[0].norm() < 1e-10);
        assert!((amps[1].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_pauli_operation_z_directly() {
        // Forces PauliOperation::Z branch (lines applying the Z gate).
        let config = MonteCarloConfig::new().with_seed(1);
        let mut sim = MonteCarloSimulator::new(1, config).unwrap();

        // Put qubit in |1> first so Z has an observable effect on phase.
        sim.apply_pauli_operation(PauliOperation::X, 0).unwrap();
        sim.apply_pauli_operation(PauliOperation::Z, 0).unwrap();

        let amps = sim.state().amplitudes();
        // Z|1> = -|1>, so amplitude at index 1 should be -1 (real part).
        assert!((amps[1].re - (-1.0)).abs() < 1e-10);
        assert!(amps[0].norm() < 1e-10);
    }

    #[test]
    fn test_apply_pauli_operation_nojump_directly() {
        // Forces PauliOperation::NoJump branch, dispatching into
        // apply_amplitude_damping_no_jump.
        let config = MonteCarloConfig::new().with_seed(1);
        let mut sim = MonteCarloSimulator::new(1, config).unwrap();

        // Put the qubit into an equal superposition so both the |0> and |1>
        // components are non-zero, exercising the scaling branch and the
        // renormalization branch inside apply_amplitude_damping_no_jump.
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

        let factor = 0.5_f64;
        sim.apply_pauli_operation(PauliOperation::NoJump { sqrt_1_minus_gamma: factor }, 0)
            .unwrap();

        let amps = sim.state().amplitudes();
        // After scaling |1> by 0.5 and renormalizing, |1> component should
        // shrink relative to |0>, and the state should remain normalized.
        let norm_sq: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm_sq - 1.0).abs() < 1e-9);
        assert!(amps[1].norm() < amps[0].norm());
    }

    #[test]
    fn test_apply_amplitude_damping_no_jump_zero_norm_skips_renormalization() {
        // If the resulting state has ~zero norm (e.g. factor=0 applied to a
        // pure |1> state, and no |0> component exists), the norm_sq <= 1e-15
        // branch must be taken, skipping renormalization without panicking
        // (e.g. divide by zero).
        let config = MonteCarloConfig::new().with_seed(1);
        let mut sim = MonteCarloSimulator::new(1, config).unwrap();

        // Prepare |1> state via X gate.
        sim.apply_pauli_operation(PauliOperation::X, 0).unwrap();

        // factor = 0.0 zeroes out the only (|1>) component entirely, leaving
        // norm_sq == 0, which must not trigger a division by zero.
        sim.apply_amplitude_damping_no_jump(0, 0.0).unwrap();

        let amps = sim.state().amplitudes();
        assert!(amps[0].norm() < 1e-10);
        assert!(amps[1].norm() < 1e-10);
    }

    #[test]
    fn test_apply_stochastic_noise_depolarizing_y_branch_via_full_range() {
        // Exercise the full apply_stochastic_noise -> apply_pauli_operation
        // dispatch path (not just the direct helper call) for the Y branch,
        // using error_prob=1.0 so Identity is never sampled, and scanning a
        // few seeds to find one landing in the Y third of the range.
        let noise = DepolarizingMC::from_probability(1.0).unwrap();
        let mut found_y = false;
        for seed in 0..50u64 {
            let config = MonteCarloConfig::new().with_seed(seed);
            let mut sim = MonteCarloSimulator::new(1, config).unwrap();
            sim.apply_stochastic_noise(&noise, 0).unwrap();
            let amps = sim.state().amplitudes();
            // Y|0> = i|1>: amplitude at index 1 is purely imaginary with norm 1.
            if amps[1].norm() > 1e-9 && amps[1].re.abs() < 1e-9 && amps[1].im.abs() > 1e-9 {
                found_y = true;
                break;
            }
        }
        assert!(found_y, "expected at least one seed to sample the Y branch");
    }
}
