//! Simulation result types

use simq_state::AdaptiveState;
use std::collections::HashMap;
use std::fmt;

use crate::statistics::ExecutionStatistics;

/// Result of a quantum circuit simulation
#[derive(Debug)]
pub struct SimulationResult {
    /// Final quantum state after all gates have been applied
    pub state: AdaptiveState,

    /// Measurement counts (if measurements were performed)
    pub measurements: Option<MeasurementCounts>,

    /// Execution statistics (if statistics collection was enabled)
    pub statistics: Option<ExecutionStatistics>,
}

impl SimulationResult {
    /// Create a new simulation result
    pub fn new(state: AdaptiveState) -> Self {
        Self {
            state,
            measurements: None,
            statistics: None,
        }
    }

    /// Add measurement counts to the result
    pub fn with_measurements(mut self, counts: MeasurementCounts) -> Self {
        self.measurements = Some(counts);
        self
    }

    /// Add execution statistics to the result
    pub fn with_statistics(mut self, stats: ExecutionStatistics) -> Self {
        self.statistics = Some(stats);
        self
    }

    /// Get the number of qubits in the final state
    pub fn num_qubits(&self) -> usize {
        self.state.num_qubits()
    }

    /// Get the total number of measurement shots (if measurements were performed)
    pub fn total_shots(&self) -> Option<usize> {
        self.measurements.as_ref().map(|m| m.total_shots())
    }

    /// Check if the state is in sparse representation
    pub fn is_sparse(&self) -> bool {
        matches!(self.state, AdaptiveState::Sparse { .. })
    }

    /// Check if the state is in dense representation
    pub fn is_dense(&self) -> bool {
        matches!(self.state, AdaptiveState::Dense(_))
    }
}

/// Measurement outcome counts
///
/// Maps bitstrings (measurement outcomes) to the number of times they were observed.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementCounts {
    /// Map from bitstring to count
    counts: HashMap<String, usize>,
    /// Total number of shots
    total_shots: usize,
}

impl MeasurementCounts {
    /// Create a new measurement counts object
    pub fn new(total_shots: usize) -> Self {
        Self {
            counts: HashMap::new(),
            total_shots,
        }
    }

    /// Create from a counts map
    pub fn from_counts(counts: HashMap<String, usize>) -> Self {
        let total_shots = counts.values().sum();
        Self {
            counts,
            total_shots,
        }
    }

    /// Add a measurement outcome
    pub fn add(&mut self, bitstring: String, count: usize) {
        *self.counts.entry(bitstring).or_insert(0) += count;
    }

    /// Get the count for a specific bitstring
    pub fn get(&self, bitstring: &str) -> usize {
        self.counts.get(bitstring).copied().unwrap_or(0)
    }

    /// Get the probability of a specific bitstring
    pub fn probability(&self, bitstring: &str) -> f64 {
        if self.total_shots == 0 {
            0.0
        } else {
            self.get(bitstring) as f64 / self.total_shots as f64
        }
    }

    /// Get all bitstrings that were observed
    pub fn bitstrings(&self) -> impl Iterator<Item = &String> {
        self.counts.keys()
    }

    /// Get all counts
    pub fn counts(&self) -> &HashMap<String, usize> {
        &self.counts
    }

    /// Get total number of shots
    pub fn total_shots(&self) -> usize {
        self.total_shots
    }

    /// Get number of unique outcomes observed
    pub fn num_outcomes(&self) -> usize {
        self.counts.len()
    }

    /// Get the most common outcome
    pub fn most_common(&self) -> Option<(&String, usize)> {
        self.counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(bs, &count)| (bs, count))
    }

    /// Get outcomes sorted by frequency (descending)
    pub fn sorted(&self) -> Vec<(&String, usize)> {
        let mut sorted: Vec<_> = self.counts.iter().map(|(bs, &count)| (bs, count)).collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted
    }

    /// Convert to a probability distribution
    pub fn to_probabilities(&self) -> HashMap<String, f64> {
        self.counts
            .iter()
            .map(|(bs, &count)| {
                let prob = count as f64 / self.total_shots as f64;
                (bs.clone(), prob)
            })
            .collect()
    }
}

impl fmt::Display for MeasurementCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Measurement Counts ({} shots):", self.total_shots)?;

        let sorted = self.sorted();
        for (bitstring, count) in sorted.iter().take(10) {
            let prob = *count as f64 / self.total_shots as f64;
            writeln!(f, "  {}: {} ({:.2}%)", bitstring, count, prob * 100.0)?;
        }

        if sorted.len() > 10 {
            writeln!(f, "  ... and {} more outcomes", sorted.len() - 10)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measurement_counts_basic() {
        let mut counts = MeasurementCounts::new(100);
        counts.add("00".to_string(), 50);
        counts.add("01".to_string(), 30);
        counts.add("11".to_string(), 20);

        assert_eq!(counts.get("00"), 50);
        assert_eq!(counts.get("01"), 30);
        assert_eq!(counts.get("11"), 20);
        assert_eq!(counts.get("10"), 0);

        assert_eq!(counts.total_shots(), 100);
        assert_eq!(counts.num_outcomes(), 3);
    }

    #[test]
    fn test_measurement_probabilities() {
        let mut counts = MeasurementCounts::new(1000);
        counts.add("00".to_string(), 500);
        counts.add("11".to_string(), 500);

        assert!((counts.probability("00") - 0.5).abs() < 1e-10);
        assert!((counts.probability("11") - 0.5).abs() < 1e-10);
        assert_eq!(counts.probability("01"), 0.0);
    }

    #[test]
    fn test_most_common() {
        let mut counts = MeasurementCounts::new(100);
        counts.add("00".to_string(), 60);
        counts.add("01".to_string(), 30);
        counts.add("11".to_string(), 10);

        let (bitstring, count) = counts.most_common().unwrap();
        assert_eq!(bitstring, "00");
        assert_eq!(count, 60);
    }

    #[test]
    fn test_sorted() {
        let mut counts = MeasurementCounts::new(100);
        counts.add("00".to_string(), 10);
        counts.add("01".to_string(), 60);
        counts.add("11".to_string(), 30);

        let sorted = counts.sorted();
        assert_eq!(sorted[0].0, "01");
        assert_eq!(sorted[1].0, "11");
        assert_eq!(sorted[2].0, "00");
    }

    #[test]
    fn test_from_counts() {
        let mut map = HashMap::new();
        map.insert("00".to_string(), 50);
        map.insert("11".to_string(), 50);

        let counts = MeasurementCounts::from_counts(map);
        assert_eq!(counts.total_shots(), 100);
        assert_eq!(counts.num_outcomes(), 2);
    }
}
