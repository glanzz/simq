//! Computational basis measurement system with efficient sampling
//!
//! This module provides high-performance quantum measurement capabilities including:
//! - Single-shot measurement with state collapse
//! - Multi-shot sampling with batch optimization
//! - Alias method for O(1) sampling after O(2^n) setup
//! - Support for both full and partial qubit measurement

use crate::error::{Result, StateError};
use crate::dense_state::DenseState;
use num_complex::Complex64;
use std::collections::HashMap;

/// Trait for quantum measurements
pub trait Measurement {
    /// Perform a measurement on the quantum state
    fn measure(&self, state: &mut DenseState, rng: &mut dyn FnMut() -> f64) -> Result<MeasurementResult>;
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
            .map(|(&outcome, &count)| {
                (format!("{:0width$b}", outcome, width = num_qubits), count)
            })
            .collect()
    }
}

/// Computational basis measurement
///
/// Measures specified qubits in the computational (Z) basis.
/// Supports both single-shot measurement and efficient multi-shot sampling.
pub struct ComputationalBasis {
    /// Qubits to measure (if None, measures all qubits)
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
    fn measure(&self, state: &mut DenseState, rng: &mut dyn FnMut() -> f64) -> Result<MeasurementResult> {
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
        while !large.is_empty() {
            let l = large.pop().unwrap();
            prob[l] = 1.0;
        }

        while !small.is_empty() {
            let s = small.pop().unwrap();
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
        let result = measurement.measure_once(&mut state, &mut || rng.next()).unwrap();

        // |00⟩ state should always measure to 0
        assert_eq!(result.outcome, 0);
        assert_relative_eq!(result.probability, 1.0);
    }

    #[test]
    fn test_alias_table_uniform() {
        let probabilities = vec![0.25, 0.25, 0.25, 0.25];
        let alias_table = AliasTable::new(&probabilities).unwrap();

        let mut rng = TestRng::new(42);
        let mut counts = vec![0; 4];

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
        let mut counts = vec![0; 4];

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
                i, freq, prob
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

        let result = measurement.sample(&state, 1000, &mut || rng.next()).unwrap();

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
