//! Backend execution results

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Result of executing a circuit on a backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendResult {
    /// Measurement counts: bitstring -> count
    pub counts: HashMap<String, usize>,

    /// Total number of shots executed
    pub shots: usize,

    /// Job ID (for async backends)
    pub job_id: Option<String>,

    /// Execution metadata
    pub metadata: ExecutionMetadata,
}

impl BackendResult {
    /// Create a new backend result
    pub fn new(counts: HashMap<String, usize>, shots: usize) -> Self {
        Self {
            counts,
            shots,
            job_id: None,
            metadata: ExecutionMetadata::default(),
        }
    }

    /// Get the most frequent measurement outcome
    pub fn most_frequent(&self) -> Option<(&String, &usize)> {
        self.counts.iter().max_by_key(|(_, &count)| count)
    }

    /// Get probability distribution (counts normalized by total shots)
    pub fn probabilities(&self) -> HashMap<String, f64> {
        self.counts
            .iter()
            .map(|(bitstring, &count)| {
                (bitstring.clone(), count as f64 / self.shots as f64)
            })
            .collect()
    }

    /// Get the count for a specific bitstring
    pub fn get_count(&self, bitstring: &str) -> usize {
        self.counts.get(bitstring).copied().unwrap_or(0)
    }

    /// Get all unique bitstrings measured
    pub fn bitstrings(&self) -> Vec<&String> {
        self.counts.keys().collect()
    }

    /// Calculate expectation value for a diagonal observable
    ///
    /// The observable is specified as a function mapping bitstring -> eigenvalue
    pub fn expectation_value<F>(&self, observable: F) -> f64
    where
        F: Fn(&str) -> f64,
    {
        let total: f64 = self
            .counts
            .iter()
            .map(|(bitstring, &count)| {
                let eigenvalue = observable(bitstring);
                eigenvalue * (count as f64)
            })
            .sum();

        total / self.shots as f64
    }

    /// Merge results from multiple backends (e.g., for ensemble methods)
    pub fn merge(&mut self, other: &BackendResult) {
        for (bitstring, &count) in &other.counts {
            *self.counts.entry(bitstring.clone()).or_insert(0) += count;
        }
        self.shots += other.shots;
    }
}

/// Execution metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    /// Execution time (wall clock)
    pub execution_time: Option<Duration>,

    /// Queue time (time spent waiting)
    pub queue_time: Option<Duration>,

    /// Total time (queue + execution)
    pub total_time: Option<Duration>,

    /// Backend name
    pub backend_name: Option<String>,

    /// Backend version
    pub backend_version: Option<String>,

    /// Job status
    pub status: JobStatus,

    /// Number of qubits used
    pub num_qubits: Option<usize>,

    /// Circuit depth after compilation
    pub circuit_depth: Option<usize>,

    /// Total gate count
    pub gate_count: Option<usize>,

    /// Number of CNOT gates (typically the most expensive)
    pub cnot_count: Option<usize>,

    /// Cost in credits (if applicable)
    pub cost: Option<f64>,

    /// Error message (if failed)
    pub error_message: Option<String>,

    /// Additional backend-specific data
    pub extra: HashMap<String, String>,
}

impl ExecutionMetadata {
    /// Create metadata for a successful execution
    pub fn success(backend_name: String, execution_time: Duration) -> Self {
        Self {
            execution_time: Some(execution_time),
            backend_name: Some(backend_name),
            status: JobStatus::Completed,
            ..Default::default()
        }
    }

    /// Create metadata for a failed execution
    pub fn failed(error_message: String) -> Self {
        Self {
            status: JobStatus::Failed,
            error_message: Some(error_message),
            ..Default::default()
        }
    }

    /// Check if the job succeeded
    pub fn is_success(&self) -> bool {
        self.status == JobStatus::Completed
    }

    /// Check if the job failed
    pub fn is_failed(&self) -> bool {
        matches!(self.status, JobStatus::Failed | JobStatus::Cancelled)
    }
}

/// Job status for async backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    /// Job is queued
    Queued,

    /// Job is validating
    Validating,

    /// Job is running
    Running,

    /// Job completed successfully
    Completed,

    /// Job failed
    Failed,

    /// Job was cancelled
    Cancelled,
}

impl Default for JobStatus {
    fn default() -> Self {
        JobStatus::Queued
    }
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobStatus::Queued => write!(f, "Queued"),
            JobStatus::Validating => write!(f, "Validating"),
            JobStatus::Running => write!(f, "Running"),
            JobStatus::Completed => write!(f, "Completed"),
            JobStatus::Failed => write!(f, "Failed"),
            JobStatus::Cancelled => write!(f, "Cancelled"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probabilities() {
        let mut counts = HashMap::new();
        counts.insert("00".to_string(), 40);
        counts.insert("11".to_string(), 60);

        let result = BackendResult::new(counts, 100);
        let probs = result.probabilities();

        assert_eq!(probs.get("00"), Some(&0.4));
        assert_eq!(probs.get("11"), Some(&0.6));
    }

    #[test]
    fn test_expectation_value() {
        let mut counts = HashMap::new();
        counts.insert("00".to_string(), 50);
        counts.insert("11".to_string(), 50);

        let result = BackendResult::new(counts, 100);

        // Z ⊗ Z observable: |00⟩ → +1, |11⟩ → +1, |01⟩ → -1, |10⟩ → -1
        let expectation = result.expectation_value(|bitstring| match bitstring {
            "00" | "11" => 1.0,
            "01" | "10" => -1.0,
            _ => 0.0,
        });

        assert_eq!(expectation, 1.0);
    }

    #[test]
    fn test_merge() {
        let mut counts1 = HashMap::new();
        counts1.insert("00".to_string(), 30);

        let mut counts2 = HashMap::new();
        counts2.insert("00".to_string(), 20);
        counts2.insert("11".to_string(), 50);

        let mut result1 = BackendResult::new(counts1, 30);
        let result2 = BackendResult::new(counts2, 70);

        result1.merge(&result2);

        assert_eq!(result1.get_count("00"), 50);
        assert_eq!(result1.get_count("11"), 50);
        assert_eq!(result1.shots, 100);
    }
}
