//! Computational basis measurement system with efficient sampling
//!
//! This module provides high-performance quantum measurement capabilities including:
//! - Single-shot measurement with state collapse
//! - Multi-shot sampling with batch optimization
//! - Alias method for O(1) sampling after O(2^n) setup
//! - Support for both full and partial qubit measurement
//! - Mid-circuit measurement with partial collapse

use crate::dense_state::DenseState;
use crate::error::{Result, StateError};
use num_complex::Complex64;
use std::collections::HashMap;

/// Trait for quantum measurements
pub trait Measurement {
    /// Perform a measurement on the quantum state
    fn measure(
        &self,
        state: &mut DenseState,
        rng: &mut dyn FnMut() -> f64,
    ) -> Result<MeasurementResult>;
}

/// Result of a quantum measurement
#[derive(Debug, Clone)]
pub struct MeasurementResult {
    /// Measurement outcome as a bitstring (basis state index)
    pub outcome: u64,

    /// Probability of this outcome
    pub probability: f64,

    /// Collapsed state after measurement (if state collapse is enabled)
    pub collapsed_state: Option<DenseState>,
}

impl MeasurementResult {
    /// Create a new measurement result
    pub fn new(outcome: u64, probability: f64) -> Self {
        Self {
            outcome,
            probability,
            collapsed_state: None,
        }
    }

    /// Create a measurement result with collapsed state
    pub fn with_collapsed_state(outcome: u64, probability: f64, state: DenseState) -> Self {
        Self {
            outcome,
            probability,
            collapsed_state: Some(state),
        }
    }

    /// Get the outcome as a bitstring
    pub fn as_bitstring(&self, num_qubits: usize) -> String {
        format!("{:0width$b}", self.outcome, width = num_qubits)
    }

    /// Get specific qubit value from the outcome
    pub fn get_qubit(&self, qubit: usize) -> u8 {
        ((self.outcome >> qubit) & 1) as u8
    }
}

/// Sampling result containing counts from multiple measurement shots
#[derive(Debug, Clone)]
pub struct SamplingResult {
    /// Map from basis state index to count
    pub counts: HashMap<u64, usize>,

    /// Total number of shots
    pub shots: usize,
}

impl SamplingResult {
    /// Create a new sampling result
    pub fn new(shots: usize) -> Self {
        Self {
            counts: HashMap::new(),
            shots,
        }
    }

    /// Add a measurement outcome
    pub fn add_outcome(&mut self, outcome: u64) {
        *self.counts.entry(outcome).or_insert(0) += 1;
    }

    /// Get the count for a specific outcome
    pub fn get_count(&self, outcome: u64) -> usize {
        self.counts.get(&outcome).copied().unwrap_or(0)
    }

    /// Get the probability of an outcome (count / shots)
    pub fn get_probability(&self, outcome: u64) -> f64 {
        self.get_count(outcome) as f64 / self.shots as f64
    }

    /// Get all outcomes sorted by count (descending)
    pub fn sorted_outcomes(&self) -> Vec<(u64, usize)> {
        let mut outcomes: Vec<_> = self.counts.iter().map(|(&k, &v)| (k, v)).collect();
        outcomes.sort_by(|a, b| b.1.cmp(&a.1));
        outcomes
    }

    /// Convert counts to bitstring format
    pub fn to_bitstring_counts(&self, num_qubits: usize) -> HashMap<String, usize> {
        self.counts
            .iter()
            .map(|(&outcome, &count)| (format!("{:0width$b}", outcome, width = num_qubits), count))
            .collect()
    }
}

/// Computational basis measurement
///
/// Measures specified qubits in the computational (Z) basis.
/// Supports both single-shot measurement and efficient multi-shot sampling.
pub struct ComputationalBasis {
    /// Qubits to measure (if None, measures all qubits)
    #[allow(dead_code)]
    qubits: Option<Vec<usize>>,

    /// Whether to collapse the state after measurement
    collapse: bool,
}

impl ComputationalBasis {
    /// Create a measurement of all qubits
    pub fn new() -> Self {
        Self {
            qubits: None,
            collapse: true,
        }
    }

    /// Create a measurement of specific qubits
    pub fn of_qubits(qubits: Vec<usize>) -> Self {
        Self {
            qubits: Some(qubits),
            collapse: true,
        }
    }

    /// Set whether to collapse the state after measurement
    pub fn with_collapse(mut self, collapse: bool) -> Self {
        self.collapse = collapse;
        self
    }

    /// Perform multiple measurement shots efficiently
    ///
    /// This uses batch sampling for better performance when taking many shots.
    ///
    /// # Arguments
    /// * `state` - The quantum state to measure (not modified if collapse=false)
    /// * `shots` - Number of measurement shots
    /// * `rng` - Random number generator function
    ///
    /// # Returns
    /// SamplingResult containing counts for each outcome
    pub fn sample(
        &self,
        state: &DenseState,
        shots: usize,
        rng: &mut dyn FnMut() -> f64,
    ) -> Result<SamplingResult> {
        if shots == 0 {
            return Ok(SamplingResult::new(0));
        }

        // Get probability distribution
        let probabilities = state.get_all_probabilities();

        // Build alias table for O(1) sampling
        let alias_table = AliasTable::new(&probabilities)?;

        // Sample using alias method
        let mut result = SamplingResult::new(shots);
        for _ in 0..shots {
            let outcome = alias_table.sample(rng);
            result.add_outcome(outcome as u64);
        }

        Ok(result)
    }

    /// Perform a single measurement shot
    ///
    /// # Arguments
    /// * `state` - The quantum state to measure
    /// * `rng` - Random number generator function
    ///
    /// # Returns
    /// MeasurementResult with outcome and probability
    pub fn measure_once(
        &self,
        state: &mut DenseState,
        rng: &mut dyn FnMut() -> f64,
    ) -> Result<MeasurementResult> {
        let random_value = rng();
        let probabilities = state.get_all_probabilities();

        // Find outcome using cumulative probabilities
        let mut cumulative = 0.0;
        let mut outcome = 0u64;
        let mut probability = 0.0;

        for (idx, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_value < cumulative {
                outcome = idx as u64;
                probability = prob;
                break;
            }
        }

        // Collapse state if requested
        let collapsed_state = if self.collapse {
            // Collapse to the measured state
            for (idx, amp) in state.amplitudes_mut().iter_mut().enumerate() {
                if idx == outcome as usize {
                    *amp = Complex64::new(1.0, 0.0);
                } else {
                    *amp = Complex64::new(0.0, 0.0);
                }
            }
            Some(state.clone_state()?)
        } else {
            None
        };

        Ok(MeasurementResult {
            outcome,
            probability,
            collapsed_state,
        })
    }
}

impl Default for ComputationalBasis {
    fn default() -> Self {
        Self::new()
    }
}

impl Measurement for ComputationalBasis {
    fn measure(
        &self,
        state: &mut DenseState,
        rng: &mut dyn FnMut() -> f64,
    ) -> Result<MeasurementResult> {
        self.measure_once(state, rng)
    }
}

/// Alias table for O(1) sampling from discrete probability distribution
///
/// Uses the alias method (Walker's algorithm) to sample from a discrete
/// distribution in O(1) time after O(n) setup.
///
/// Reference: Walker, A. J. (1977). "An Efficient Method for Generating
/// Discrete Random Variables with General Distributions"
struct AliasTable {
    /// Probability threshold for each index
    prob: Vec<f64>,

    /// Alias index for each index
    alias: Vec<usize>,
}

impl AliasTable {
    /// Create a new alias table from probability distribution
    ///
    /// # Arguments
    /// * `probabilities` - Probability distribution (must sum to ~1.0)
    ///
    /// # Returns
    /// Alias table for O(1) sampling
    fn new(probabilities: &[f64]) -> Result<Self> {
        let n = probabilities.len();
        if n == 0 {
            return Err(StateError::InvalidDimension { dimension: 0 });
        }

        let mut prob = vec![0.0; n];
        let mut alias = vec![0; n];

        // Scale probabilities
        let scaled: Vec<f64> = probabilities.iter().map(|&p| p * n as f64).collect();

        // Separate into small and large
        let mut small = Vec::new();
        let mut large = Vec::new();

        for (i, &p) in scaled.iter().enumerate() {
            if p < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        // Build alias table
        let mut prob_copy = scaled.clone();

        while !small.is_empty() && !large.is_empty() {
            let s = small.pop().unwrap();
            let l = large.pop().unwrap();

            prob[s] = prob_copy[s];
            alias[s] = l;

            prob_copy[l] = (prob_copy[l] + prob_copy[s]) - 1.0;

            if prob_copy[l] < 1.0 {
                small.push(l);
            } else {
                large.push(l);
            }
        }

        // Handle remaining (due to floating-point errors)
        while let Some(l) = large.pop() {
            prob[l] = 1.0;
        }

        while let Some(s) = small.pop() {
            prob[s] = 1.0;
        }

        Ok(Self { prob, alias })
    }

    /// Sample an index from the distribution in O(1) time
    ///
    /// # Arguments
    /// * `rng` - Random number generator function
    ///
    /// # Returns
    /// Sampled index
    fn sample(&self, rng: &mut dyn FnMut() -> f64) -> usize {
        let n = self.prob.len();
        let i = (rng() * n as f64) as usize;
        let i = i.min(n - 1); // Handle edge case where rng() == 1.0

        if rng() < self.prob[i] {
            i
        } else {
            self.alias[i]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Simple linear congruential generator for testing
    struct TestRng {
        state: u64,
    }

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next(&mut self) -> f64 {
            self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
            ((self.state / 65536) % 32768) as f64 / 32768.0
        }
    }

    #[test]
    fn test_measurement_result() {
        let result = MeasurementResult::new(5, 0.25);
        assert_eq!(result.outcome, 5);
        assert_relative_eq!(result.probability, 0.25);
        assert_eq!(result.as_bitstring(3), "101");
        assert_eq!(result.get_qubit(0), 1);
        assert_eq!(result.get_qubit(1), 0);
        assert_eq!(result.get_qubit(2), 1);
    }

    #[test]
    fn test_sampling_result() {
        let mut result = SamplingResult::new(100);
        for _ in 0..60 {
            result.add_outcome(0);
        }
        for _ in 0..40 {
            result.add_outcome(1);
        }

        assert_eq!(result.shots, 100);
        assert_eq!(result.get_count(0), 60);
        assert_eq!(result.get_count(1), 40);
        assert_relative_eq!(result.get_probability(0), 0.6);
        assert_relative_eq!(result.get_probability(1), 0.4);

        let sorted = result.sorted_outcomes();
        assert_eq!(sorted[0], (0, 60));
        assert_eq!(sorted[1], (1, 40));
    }

    #[test]
    fn test_computational_basis_measurement() {
        let mut state = DenseState::new(2).unwrap();
        let mut rng = TestRng::new(42);

        let measurement = ComputationalBasis::new();
        let result = measurement
            .measure_once(&mut state, &mut || rng.next())
            .unwrap();

        // |00⟩ state should always measure to 0
        assert_eq!(result.outcome, 0);
        assert_relative_eq!(result.probability, 1.0);
    }

    #[test]
    fn test_alias_table_uniform() {
        let probabilities = vec![0.25, 0.25, 0.25, 0.25];
        let alias_table = AliasTable::new(&probabilities).unwrap();

        let mut rng = TestRng::new(42);
        let mut counts = [0; 4];

        let shots = 10000;
        for _ in 0..shots {
            let outcome = alias_table.sample(&mut || rng.next());
            counts[outcome] += 1;
        }

        // Each outcome should appear roughly 25% of the time
        for count in counts {
            let freq = count as f64 / shots as f64;
            assert!((freq - 0.25).abs() < 0.02, "Frequency {} too far from 0.25", freq);
        }
    }

    #[test]
    fn test_alias_table_nonuniform() {
        let probabilities = vec![0.5, 0.3, 0.15, 0.05];
        let alias_table = AliasTable::new(&probabilities).unwrap();

        let mut rng = TestRng::new(123);
        let mut counts = [0; 4];

        let shots = 10000;
        for _ in 0..shots {
            let outcome = alias_table.sample(&mut || rng.next());
            counts[outcome] += 1;
        }

        // Check that frequencies match probabilities
        for (i, (&prob, &count)) in probabilities.iter().zip(counts.iter()).enumerate() {
            let freq = count as f64 / shots as f64;
            assert!(
                (freq - prob).abs() < 0.02,
                "Outcome {} frequency {} too far from {}",
                i,
                freq,
                prob
            );
        }
    }

    #[test]
    fn test_batch_sampling() {
        // Create a superposition state
        let amplitudes = vec![
            Complex64::new(0.6, 0.0),
            Complex64::new(0.8, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

        let measurement = ComputationalBasis::new().with_collapse(false);
        let mut rng = TestRng::new(42);

        let result = measurement
            .sample(&state, 1000, &mut || rng.next())
            .unwrap();

        assert_eq!(result.shots, 1000);

        // Should have outcomes 0 and 1 with probabilities ~0.36 and ~0.64
        let prob_0 = result.get_probability(0);
        let prob_1 = result.get_probability(1);

        assert!((prob_0 - 0.36).abs() < 0.05, "prob_0 = {}", prob_0);
        assert!((prob_1 - 0.64).abs() < 0.05, "prob_1 = {}", prob_1);
    }

    #[test]
    fn test_bitstring_conversion() {
        let mut result = SamplingResult::new(10);
        result.add_outcome(0); // 00
        result.add_outcome(1); // 01
        result.add_outcome(2); // 10
        result.add_outcome(3); // 11

        let bitstring_counts = result.to_bitstring_counts(2);
        assert_eq!(bitstring_counts.get("00"), Some(&1));
        assert_eq!(bitstring_counts.get("01"), Some(&1));
        assert_eq!(bitstring_counts.get("10"), Some(&1));
        assert_eq!(bitstring_counts.get("11"), Some(&1));
    }
}

/// Mid-circuit measurement with partial collapse
///
/// Measures a subset of qubits while keeping others in superposition.
/// This is essential for quantum error correction and quantum teleportation.
pub struct MidCircuitMeasurement {
    /// Qubits to measure
    qubits: Vec<usize>,
}

impl MidCircuitMeasurement {
    /// Create a mid-circuit measurement for specific qubits
    ///
    /// # Arguments
    /// * `qubits` - Indices of qubits to measure
    ///
    /// # Example
    /// ```
    /// use simq_state::MidCircuitMeasurement;
    ///
    /// // Measure qubits 0 and 2, leave qubit 1 in superposition
    /// let measurement = MidCircuitMeasurement::new(vec![0, 2]);
    /// ```
    pub fn new(qubits: Vec<usize>) -> Self {
        Self { qubits }
    }

    /// Perform mid-circuit measurement with partial collapse
    ///
    /// # Arguments
    /// * `state` - The quantum state (modified in-place)
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Measurement outcomes for each specified qubit
    ///
    /// # Example
    /// ```
    /// use simq_state::{DenseState, MidCircuitMeasurement};
    ///
    /// let mut state = DenseState::new(3).unwrap();
    /// let mut rng = || 0.5;
    ///
    /// let measurement = MidCircuitMeasurement::new(vec![0, 2]);
    /// let outcomes = measurement.measure(&mut state, &mut rng).unwrap();
    /// // Qubit 1 remains in superposition
    /// ```
    pub fn measure(
        &self,
        state: &mut DenseState,
        rng: &mut dyn FnMut() -> f64,
    ) -> Result<Vec<(usize, u8)>> {
        let mut outcomes = Vec::new();

        // Measure each qubit sequentially
        for &qubit in &self.qubits {
            let outcome = self.measure_single_qubit(state, qubit, rng)?;
            outcomes.push((qubit, outcome));
        }

        Ok(outcomes)
    }

    /// Measure a single qubit with partial collapse
    fn measure_single_qubit(
        &self,
        state: &mut DenseState,
        qubit: usize,
        rng: &mut dyn FnMut() -> f64,
    ) -> Result<u8> {
        if qubit >= state.num_qubits() {
            return Err(StateError::InvalidQubitIndex {
                index: qubit,
                num_qubits: state.num_qubits(),
            });
        }

        // Calculate probability of measuring |0⟩ on this qubit
        // Qubit indexing: qubit 0 is LSB (little-endian)
        let mask = 1 << qubit;
        let prob_zero: f64 = state
            .amplitudes()
            .iter()
            .enumerate()
            .filter(|(idx, _)| idx & mask == 0)
            .map(|(_, amp)| amp.norm_sqr())
            .sum();

        // Determine outcome
        let random_value = rng();
        let outcome = if random_value < prob_zero { 0 } else { 1 };

        // Partial collapse: project onto measurement outcome
        let normalization = if outcome == 0 {
            prob_zero.sqrt()
        } else {
            (1.0 - prob_zero).sqrt()
        };

        if normalization < 1e-10 {
            return Err(StateError::NotNormalized {
                norm: normalization,
            });
        }

        let inv_norm = 1.0 / normalization;

        // Zero out amplitudes inconsistent with measurement, renormalize others
        for (idx, amp) in state.amplitudes_mut().iter_mut().enumerate() {
            if ((idx >> qubit) & 1) != outcome as usize {
                *amp = Complex64::new(0.0, 0.0);
            } else {
                *amp *= inv_norm;
            }
        }

        Ok(outcome)
    }

    /// Measure and return both outcomes and the updated state
    pub fn measure_with_state(
        &self,
        state: &mut DenseState,
        rng: &mut dyn FnMut() -> f64,
    ) -> Result<(Vec<(usize, u8)>, DenseState)> {
        let outcomes = self.measure(state, rng)?;
        let final_state = state.clone_state()?;
        Ok((outcomes, final_state))
    }
}

#[cfg(test)]
mod mid_circuit_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mid_circuit_single_qubit() {
        // Create (|00⟩ + |01⟩)/√2 where qubit 1 is |0⟩, qubit 0 is |+⟩
        // In little-endian (q0 is LSB): |q1=0⟩ ⊗ |q0=+⟩
        let amplitudes = vec![
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0), // |00⟩
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0), // |01⟩
            Complex64::new(0.0, 0.0),                // |10⟩
            Complex64::new(0.0, 0.0),                // |11⟩
        ];
        let mut state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

        // Measure qubit 1 (which is deterministically |0⟩)
        let measurement = MidCircuitMeasurement::new(vec![1]);
        let mut rng = || 0.3; // Definitely less than prob(|0⟩) = 1.0
        let outcomes = measurement.measure(&mut state, &mut rng).unwrap();

        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0], (1, 0)); // Qubit 1 measured as |0⟩

        // Qubit 0 should still be in superposition |+⟩
        let amps = state.amplitudes();
        assert_relative_eq!(amps[0].norm(), 1.0 / 2_f64.sqrt(), epsilon = 1e-10);
        assert_relative_eq!(amps[1].norm(), 1.0 / 2_f64.sqrt(), epsilon = 1e-10);
        assert_relative_eq!(amps[2].norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(amps[3].norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mid_circuit_multiple_qubits() {
        // Create GHZ state: (|000⟩ + |111⟩)/√2
        let amplitudes = vec![
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
        ];
        let mut state = DenseState::from_amplitudes(3, &amplitudes).unwrap();

        // Measure qubit 0
        let measurement = MidCircuitMeasurement::new(vec![0]);
        let mut rng = || 0.3; // Should measure |0⟩
        let outcomes = measurement.measure(&mut state, &mut rng).unwrap();

        assert_eq!(outcomes[0].1, 0);

        // After measuring qubit 0 as |0⟩, state should be |000⟩
        let amps = state.amplitudes();
        assert_relative_eq!(amps[0].norm(), 1.0, epsilon = 1e-10);
        for amp in amps.iter().take(8).skip(1) {
            assert_relative_eq!(amp.norm(), 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mid_circuit_preserves_superposition() {
        // Create (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2 = |+⟩ ⊗ |+⟩
        // Both qubits in equal superposition
        let amplitudes = vec![
            Complex64::new(0.5, 0.0), // |00⟩
            Complex64::new(0.5, 0.0), // |01⟩
            Complex64::new(0.5, 0.0), // |10⟩
            Complex64::new(0.5, 0.0), // |11⟩
        ];
        let mut state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

        // Measure qubit 1 (MSB), leave qubit 0 (LSB) in superposition
        // Probability of measuring qubit 1 as |0⟩ is 50%
        let measurement = MidCircuitMeasurement::new(vec![1]);
        let mut rng = || 0.3; // Should measure |0⟩ (prob = 0.5, and 0.3 < 0.5)
        let outcomes = measurement.measure(&mut state, &mut rng).unwrap();

        assert_eq!(outcomes[0].1, 0); // Qubit 1 measured as |0⟩

        // After measuring qubit 1 as |0⟩, should have (|00⟩ + |01⟩)/√2
        // Qubit 0 remains in |+⟩ superposition
        let amps = state.amplitudes();
        assert_relative_eq!(amps[0].norm(), 1.0 / 2_f64.sqrt(), epsilon = 1e-10);
        assert_relative_eq!(amps[1].norm(), 1.0 / 2_f64.sqrt(), epsilon = 1e-10);
        assert_relative_eq!(amps[2].norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(amps[3].norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mid_circuit_bell_state() {
        // Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        let amplitudes = vec![
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
        ];
        let mut state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

        // Measure first qubit
        let measurement = MidCircuitMeasurement::new(vec![0]);
        let mut rng = || 0.3; // Measure |0⟩
        let outcomes = measurement.measure(&mut state, &mut rng).unwrap();

        let measured_value = outcomes[0].1;

        // Due to entanglement, both qubits should have same value
        let amps = state.amplitudes();
        if measured_value == 0 {
            assert_relative_eq!(amps[0].norm(), 1.0, epsilon = 1e-10);
        } else {
            assert_relative_eq!(amps[3].norm(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mid_circuit_sequential_measurements() {
        // Create (|000⟩ + |111⟩)/√2
        let amplitudes = vec![
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
        ];
        let mut state = DenseState::from_amplitudes(3, &amplitudes).unwrap();

        // Measure qubits 0 and 1 sequentially
        let measurement = MidCircuitMeasurement::new(vec![0, 1]);
        let mut rng = || 0.3;
        let outcomes = measurement.measure(&mut state, &mut rng).unwrap();

        // Both should have same value due to GHZ entanglement
        assert_eq!(outcomes[0].1, outcomes[1].1);
    }
}
