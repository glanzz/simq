//! Density matrix simulator with comprehensive noise support
//!
//! This simulator uses density matrices to represent quantum states,
//! enabling simulation of noisy quantum circuits with decoherence,
//! gate errors, and measurement errors.

use crate::density_matrix::DensityMatrix;
use crate::error::{Result, StateError};
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Configuration for density matrix simulation
#[derive(Debug, Clone)]
pub struct DensityMatrixConfig {
    /// Random seed for reproducible results
    pub seed: Option<u64>,

    /// Number of measurement shots
    pub shots: usize,

    /// Whether to validate density matrix after each operation
    pub validate_state: bool,

    /// Tolerance for validation checks
    pub tolerance: f64,
}

impl Default for DensityMatrixConfig {
    fn default() -> Self {
        Self {
            seed: None,
            shots: 1024,
            validate_state: false,
            tolerance: 1e-10,
        }
    }
}

impl DensityMatrixConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_shots(mut self, shots: usize) -> Self {
        self.shots = shots;
        self
    }

    pub fn with_validation(mut self, enabled: bool) -> Self {
        self.validate_state = enabled;
        self
    }
}

/// Density matrix simulator with noise support
///
/// # Example
///
/// ```ignore
/// use simq_state::DensityMatrixSimulator;
/// use simq_core::noise::{DepolarizingChannel, NoiseChannel};
///
/// let mut sim = DensityMatrixSimulator::new(2, DensityMatrixConfig::default())?;
///
/// // Apply gates with noise
/// let hadamard = [...]; // 2x2 matrix
/// let noise = DepolarizingChannel::new(0.01)?;
///
/// sim.apply_gate(&hadamard, &[0])?;
/// sim.apply_noise_channel(&noise, 0)?;
///
/// // Measure with readout errors
/// let results = sim.measure_all_shots()?;
/// ```
pub struct DensityMatrixSimulator {
    /// Current density matrix state
    state: DensityMatrix,

    /// Configuration
    config: DensityMatrixConfig,

    /// Random number generator
    rng: StdRng,

    /// Statistics
    gate_count: usize,
    noise_ops_applied: usize,
}

impl DensityMatrixSimulator {
    /// Create a new density matrix simulator
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    /// * `config` - Simulation configuration
    pub fn new(num_qubits: usize, config: DensityMatrixConfig) -> Result<Self> {
        let state = DensityMatrix::new(num_qubits)?;

        let rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        Ok(Self {
            state,
            config,
            rng,
            gate_count: 0,
            noise_ops_applied: 0,
        })
    }

    /// Get reference to current density matrix
    pub fn state(&self) -> &DensityMatrix {
        &self.state
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.state.num_qubits()
    }

    /// Apply a unitary gate
    ///
    /// # Arguments
    /// * `unitary` - Gate matrix (flattened, row-major)
    /// * `qubits` - Target qubit indices
    pub fn apply_gate(&mut self, unitary: &[Complex64], qubits: &[usize]) -> Result<()> {
        self.state.apply_unitary(unitary, qubits)?;
        self.gate_count += 1;

        if self.config.validate_state && !self.state.is_valid(self.config.tolerance) {
            return Err(StateError::NotNormalized {
                norm: self.state.trace(),
            });
        }

        Ok(())
    }

    /// Apply a noise channel using Kraus operators
    ///
    /// This is the key method for noise simulation!
    ///
    /// # Arguments
    /// * `kraus_ops` - Kraus operators as (matrix, dimension) pairs
    /// * `qubits` - Target qubit indices
    pub fn apply_kraus_channel(
        &mut self,
        kraus_ops: &[(Vec<Complex64>, usize)],
        qubits: &[usize],
    ) -> Result<()> {
        self.state.apply_kraus_channel(kraus_ops, qubits)?;
        self.noise_ops_applied += 1;

        if self.config.validate_state && !self.state.is_valid(self.config.tolerance) {
            return Err(StateError::NotNormalized {
                norm: self.state.trace(),
            });
        }

        Ok(())
    }

    /// Measure a single qubit
    ///
    /// # Arguments
    /// * `qubit` - Qubit index to measure
    ///
    /// # Returns
    /// Measurement outcome (false = 0, true = 1)
    pub fn measure_qubit(&mut self, qubit: usize) -> Result<bool> {
        let random_value = self.rng.gen::<f64>();
        self.state.measure(qubit, random_value)
    }

    /// Measure all qubits and return a single bitstring
    ///
    /// # Returns
    /// Vector of measurement outcomes (false = 0, true = 1)
    pub fn measure_all(&mut self) -> Result<Vec<bool>> {
        let mut results = Vec::with_capacity(self.num_qubits());

        for qubit in 0..self.num_qubits() {
            results.push(self.measure_qubit(qubit)?);
        }

        Ok(results)
    }

    /// Perform multiple measurement shots
    ///
    /// # Returns
    /// Vector of bitstrings, one per shot
    pub fn measure_all_shots(&mut self) -> Result<Vec<Vec<bool>>> {
        let mut results = Vec::with_capacity(self.config.shots);

        // Save initial state
        let initial_state = self.state.matrix().to_vec();

        for _ in 0..self.config.shots {
            // Restore state
            self.state.matrix_mut().copy_from_slice(&initial_state);

            // Measure
            results.push(self.measure_all()?);
        }

        Ok(results)
    }

    /// Get current purity: Tr(ρ²)
    ///
    /// Returns 1.0 for pure states, < 1.0 for mixed states
    pub fn purity(&self) -> f64 {
        self.state.purity()
    }

    /// Get von Neumann entropy: -Tr(ρ log₂ ρ)
    pub fn entropy(&self) -> f64 {
        self.state.von_neumann_entropy()
    }

    /// Compute partial trace to get reduced density matrix
    ///
    /// # Arguments
    /// * `trace_qubits` - Qubits to trace out
    ///
    /// # Returns
    /// New simulator with reduced state
    pub fn partial_trace(&self, trace_qubits: &[usize]) -> Result<Self> {
        let reduced_state = self.state.partial_trace(trace_qubits)?;

        let rng = if let Some(seed) = self.config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        Ok(Self {
            state: reduced_state,
            config: self.config.clone(),
            rng,
            gate_count: 0,
            noise_ops_applied: 0,
        })
    }

    /// Get simulation statistics
    pub fn stats(&self) -> SimulationStats {
        SimulationStats {
            gate_count: self.gate_count,
            noise_ops_applied: self.noise_ops_applied,
            current_purity: self.purity(),
            current_entropy: self.entropy(),
        }
    }

    /// Reset to initial |0...0⟩ state
    pub fn reset(&mut self) -> Result<()> {
        self.state = DensityMatrix::new(self.num_qubits())?;
        self.gate_count = 0;
        self.noise_ops_applied = 0;
        Ok(())
    }
}

/// Simulation statistics
#[derive(Debug, Clone)]
pub struct SimulationStats {
    pub gate_count: usize,
    pub noise_ops_applied: usize,
    pub current_purity: f64,
    pub current_entropy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulator_creation() {
        let sim = DensityMatrixSimulator::new(2, DensityMatrixConfig::default()).unwrap();
        assert_eq!(sim.num_qubits(), 2);
        assert!((sim.purity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_hadamard() {
        let config = DensityMatrixConfig::new().with_seed(42);
        let mut sim = DensityMatrixSimulator::new(1, config).unwrap();

        let h = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
        ];

        sim.apply_gate(&h, &[0]).unwrap();
        assert!((sim.purity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_measurement() {
        let config = DensityMatrixConfig::new().with_seed(42).with_shots(100);
        let mut sim = DensityMatrixSimulator::new(1, config).unwrap();

        // Measure |0⟩ state - should always get 0
        let outcome = sim.measure_qubit(0).unwrap();
        assert!(!outcome);
    }

    #[test]
    fn test_stats() {
        let mut sim = DensityMatrixSimulator::new(2, DensityMatrixConfig::default()).unwrap();

        let h = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
        ];

        sim.apply_gate(&h, &[0]).unwrap();

        let stats = sim.stats();
        assert_eq!(stats.gate_count, 1);
        assert_eq!(stats.noise_ops_applied, 0);
    }

    // ---- New coverage tests ----

    fn hadamard_flat() -> Vec<Complex64> {
        let h = 1.0 / 2.0_f64.sqrt();
        vec![
            Complex64::new(h, 0.0),
            Complex64::new(h, 0.0),
            Complex64::new(h, 0.0),
            Complex64::new(-h, 0.0),
        ]
    }

    #[test]
    fn test_config_without_seed() {
        // Constructor without seed — uses entropy
        let config = DensityMatrixConfig::new();
        assert!(config.seed.is_none());
        let sim = DensityMatrixSimulator::new(2, config).unwrap();
        assert_eq!(sim.num_qubits(), 2);
    }

    #[test]
    fn test_config_with_shots() {
        let config = DensityMatrixConfig::new().with_shots(50);
        assert_eq!(config.shots, 50);
    }

    #[test]
    fn test_apply_kraus_channel_increments_noise_ops() {
        let config = DensityMatrixConfig::new().with_seed(42);
        let mut sim = DensityMatrixSimulator::new(1, config).unwrap();

        // Depolarizing Kraus channel: identity Kraus op scaled by sqrt(1)
        // Just use a single identity Kraus operator — trivial but valid channel
        let identity_kraus: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let kraus_ops = vec![(identity_kraus, 2usize)];
        sim.apply_kraus_channel(&kraus_ops, &[0]).unwrap();
        assert_eq!(sim.stats().noise_ops_applied, 1);
    }

    #[test]
    fn test_purity_decreases_after_depolarizing_kraus() {
        // Depolarizing Kraus channel lowers purity below 1.0
        let config = DensityMatrixConfig::new().with_seed(42);
        let mut sim = DensityMatrixSimulator::new(1, config).unwrap();

        // Apply Hadamard to get pure |+⟩ state (purity = 1.0)
        sim.apply_gate(&hadamard_flat(), &[0]).unwrap();
        let purity_before = sim.purity();
        assert!((purity_before - 1.0).abs() < 1e-10);

        // Apply a depolarizing Kraus channel: { sqrt(1-p)*I, sqrt(p/3)*X, sqrt(p/3)*Y, sqrt(p/3)*Z }
        let p = 0.3_f64;
        let a = (1.0 - p).sqrt();
        let b = (p / 3.0).sqrt();

        let zero = Complex64::new(0.0, 0.0);
        let k0: Vec<Complex64> = vec![Complex64::new(a, 0.0), zero, zero, Complex64::new(a, 0.0)];
        let k1: Vec<Complex64> = vec![zero, Complex64::new(b, 0.0), Complex64::new(b, 0.0), zero]; // b*X
        let k2: Vec<Complex64> = vec![zero, Complex64::new(0.0, -b), Complex64::new(0.0, b), zero]; // b*Y
        let k3: Vec<Complex64> = vec![Complex64::new(b, 0.0), zero, zero, Complex64::new(-b, 0.0)]; // b*Z

        let kraus_ops = vec![(k0, 2usize), (k1, 2usize), (k2, 2usize), (k3, 2usize)];
        sim.apply_kraus_channel(&kraus_ops, &[0]).unwrap();

        let purity_after = sim.purity();
        assert!(purity_after < 1.0, "purity should decrease after noise, got {purity_after}");
        assert_eq!(sim.stats().noise_ops_applied, 1);
    }

    #[test]
    fn test_entropy_positive_after_depolarizing_kraus() {
        let config = DensityMatrixConfig::new().with_seed(42);
        let mut sim = DensityMatrixSimulator::new(1, config).unwrap();
        sim.apply_gate(&hadamard_flat(), &[0]).unwrap();

        let p = 0.3_f64;
        let a = (1.0 - p).sqrt();
        let b = (p / 3.0).sqrt();
        let zero = Complex64::new(0.0, 0.0);

        let k0: Vec<Complex64> = vec![Complex64::new(a, 0.0), zero, zero, Complex64::new(a, 0.0)];
        let k1: Vec<Complex64> = vec![zero, Complex64::new(b, 0.0), Complex64::new(b, 0.0), zero];
        let k2: Vec<Complex64> = vec![zero, Complex64::new(0.0, -b), Complex64::new(0.0, b), zero];
        let k3: Vec<Complex64> = vec![Complex64::new(b, 0.0), zero, zero, Complex64::new(-b, 0.0)];

        let kraus_ops = vec![(k0, 2usize), (k1, 2usize), (k2, 2usize), (k3, 2usize)];
        sim.apply_kraus_channel(&kraus_ops, &[0]).unwrap();

        let entropy = sim.entropy();
        assert!(entropy > 0.0, "entropy should be positive for mixed state, got {entropy}");
    }

    #[test]
    fn test_measure_all_2qubit() {
        let config = DensityMatrixConfig::new().with_seed(42);
        let mut sim = DensityMatrixSimulator::new(2, config).unwrap();

        let results = sim.measure_all().unwrap();
        assert_eq!(results.len(), 2);
        // |00⟩ state → both outcomes must be false
        assert!(!results[0]);
        assert!(!results[1]);
    }

    #[test]
    fn test_measure_all_shots_pure_zero_state() {
        let config = DensityMatrixConfig::new().with_seed(42).with_shots(10);
        let mut sim = DensityMatrixSimulator::new(1, config).unwrap();

        let all_shots = sim.measure_all_shots().unwrap();
        assert_eq!(all_shots.len(), 10);
        for shot in &all_shots {
            assert_eq!(shot.len(), 1);
            // |0⟩ state always measures 0
            assert!(!shot[0]);
        }
    }

    #[test]
    fn test_partial_trace_2qubit() {
        // Apply H on qubit 0, then partial_trace out qubit 0 → 1-qubit reduced simulator
        let config = DensityMatrixConfig::new().with_seed(42);
        let mut sim = DensityMatrixSimulator::new(2, config).unwrap();
        sim.apply_gate(&hadamard_flat(), &[0]).unwrap();

        let reduced = sim.partial_trace(&[0]).unwrap();
        assert_eq!(reduced.num_qubits(), 1);
    }

    #[test]
    fn test_reset_clears_counts() {
        let config = DensityMatrixConfig::new().with_seed(42);
        let mut sim = DensityMatrixSimulator::new(2, config).unwrap();

        sim.apply_gate(&hadamard_flat(), &[0]).unwrap();
        let identity_kraus: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        sim.apply_kraus_channel(&[(identity_kraus, 2usize)], &[0])
            .unwrap();

        assert_eq!(sim.stats().gate_count, 1);
        assert_eq!(sim.stats().noise_ops_applied, 1);

        sim.reset().unwrap();
        assert_eq!(sim.stats().gate_count, 0);
        assert_eq!(sim.stats().noise_ops_applied, 0);
        assert!((sim.purity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_with_validation_valid_gate_succeeds() {
        let config = DensityMatrixConfig::new()
            .with_seed(42)
            .with_validation(true);
        let mut sim = DensityMatrixSimulator::new(1, config).unwrap();
        // Valid Hadamard gate — should not trigger validation error
        sim.apply_gate(&hadamard_flat(), &[0]).unwrap();
        assert!((sim.purity() - 1.0).abs() < 1e-10);
    }
}
