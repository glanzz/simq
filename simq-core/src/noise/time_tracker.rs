//! Time tracking for noise-aware circuit execution
//!
//! This module provides utilities for tracking qubit idle times during
//! circuit execution, which is essential for realistic time-dependent
//! noise modeling.
//!
//! # Overview
//!
//! Real quantum computers experience decoherence continuously, not just
//! during gate operations. A qubit that sits idle while other qubits are
//! operated will accumulate T1 and T2 errors. This module helps track:
//!
//! - Per-qubit accumulated time
//! - Idle time between operations
//! - Total circuit execution time
//!
//! # Example
//!
//! ```ignore
//! use simq_core::noise::{QubitTimeTracker, GateTiming};
//!
//! let timing = GateTiming::default();
//! let mut tracker = QubitTimeTracker::new(3, timing);
//!
//! // Apply single-qubit gate on qubit 0
//! tracker.apply_single_qubit_gate(0);
//!
//! // Apply two-qubit gate on qubits 0 and 1
//! // Qubit 2 remains idle during this time
//! tracker.apply_two_qubit_gate(0, 1);
//!
//! // Get idle time for qubit 2
//! let idle_time = tracker.idle_time_since_last_operation(2);
//! ```

use super::GateTiming;

/// Tracks execution time for each qubit in a circuit
///
/// This tracker maintains the current time for each qubit, allowing
/// calculation of idle periods for realistic noise simulation.
#[derive(Debug, Clone)]
pub struct QubitTimeTracker {
    /// Number of qubits being tracked
    num_qubits: usize,

    /// Current time for each qubit (μs)
    ///
    /// This represents the time at which the qubit was last operated
    qubit_times: Vec<f64>,

    /// Gate timing information
    timing: GateTiming,

    /// Total elapsed time (μs)
    total_time: f64,
}

impl QubitTimeTracker {
    /// Create a new time tracker
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits to track
    /// * `timing` - Gate timing information
    pub fn new(num_qubits: usize, timing: GateTiming) -> Self {
        Self {
            num_qubits,
            qubit_times: vec![0.0; num_qubits],
            timing,
            total_time: 0.0,
        }
    }

    /// Apply a single-qubit gate to the specified qubit
    ///
    /// Updates the qubit's time by the single-qubit gate duration
    ///
    /// # Arguments
    /// * `qubit` - Index of the qubit being operated
    pub fn apply_single_qubit_gate(&mut self, qubit: usize) {
        if qubit < self.num_qubits {
            self.qubit_times[qubit] += self.timing.single_qubit_gate_time;
            self.update_total_time();
        }
    }

    /// Apply a two-qubit gate to the specified qubits
    ///
    /// Updates both qubits' time by the two-qubit gate duration
    ///
    /// # Arguments
    /// * `qubit1` - Index of the first qubit
    /// * `qubit2` - Index of the second qubit
    pub fn apply_two_qubit_gate(&mut self, qubit1: usize, qubit2: usize) {
        if qubit1 < self.num_qubits && qubit2 < self.num_qubits {
            let gate_time = self.timing.two_qubit_gate_time;
            self.qubit_times[qubit1] += gate_time;
            self.qubit_times[qubit2] += gate_time;
            self.update_total_time();
        }
    }

    /// Apply a two-qubit gate with custom duration
    ///
    /// Useful for gates with non-standard timing
    ///
    /// # Arguments
    /// * `qubit1` - Index of the first qubit
    /// * `qubit2` - Index of the second qubit
    /// * `duration_us` - Gate duration in microseconds
    pub fn apply_two_qubit_gate_custom(&mut self, qubit1: usize, qubit2: usize, duration_us: f64) {
        if qubit1 < self.num_qubits && qubit2 < self.num_qubits {
            self.qubit_times[qubit1] += duration_us;
            self.qubit_times[qubit2] += duration_us;
            self.update_total_time();
        }
    }

    /// Apply measurement to a qubit
    ///
    /// Updates the qubit's time by the measurement duration
    ///
    /// # Arguments
    /// * `qubit` - Index of the qubit being measured
    pub fn apply_measurement(&mut self, qubit: usize) {
        if qubit < self.num_qubits {
            self.qubit_times[qubit] += self.timing.measurement_time;
            self.update_total_time();
        }
    }

    /// Advance a qubit's time by a specific duration
    ///
    /// Useful for custom operations or explicit idle periods
    ///
    /// # Arguments
    /// * `qubit` - Index of the qubit
    /// * `duration_us` - Time to advance in microseconds
    pub fn advance_qubit_time(&mut self, qubit: usize, duration_us: f64) {
        if qubit < self.num_qubits {
            self.qubit_times[qubit] += duration_us;
            self.update_total_time();
        }
    }

    /// Get the current time for a specific qubit
    ///
    /// # Arguments
    /// * `qubit` - Index of the qubit
    ///
    /// # Returns
    /// Current time in microseconds, or None if qubit index is invalid
    pub fn qubit_time(&self, qubit: usize) -> Option<f64> {
        self.qubit_times.get(qubit).copied()
    }

    /// Get idle time since last operation for a qubit
    ///
    /// Computes how long the qubit has been idle relative to the
    /// most recently operated qubit. This is the time during which
    /// idle noise should be applied.
    ///
    /// # Arguments
    /// * `qubit` - Index of the qubit
    ///
    /// # Returns
    /// Idle time in microseconds
    pub fn idle_time_since_last_operation(&self, qubit: usize) -> f64 {
        if qubit >= self.num_qubits {
            return 0.0;
        }

        let current_time = self.qubit_times[qubit];
        self.total_time - current_time
    }

    /// Get all idle times for all qubits
    ///
    /// Returns a vector where index i contains the idle time for qubit i
    ///
    /// # Returns
    /// Vector of idle times in microseconds
    pub fn all_idle_times(&self) -> Vec<f64> {
        (0..self.num_qubits)
            .map(|q| self.idle_time_since_last_operation(q))
            .collect()
    }

    /// Synchronize all qubits to the current total time
    ///
    /// After calling this, all qubits will have zero idle time.
    /// Useful after applying idle noise to bring all qubits up to date.
    pub fn synchronize_all_qubits(&mut self) {
        for qubit_time in &mut self.qubit_times {
            *qubit_time = self.total_time;
        }
    }

    /// Synchronize specific qubits to the current total time
    ///
    /// # Arguments
    /// * `qubits` - Indices of qubits to synchronize
    pub fn synchronize_qubits(&mut self, qubits: &[usize]) {
        for &qubit in qubits {
            if qubit < self.num_qubits {
                self.qubit_times[qubit] = self.total_time;
            }
        }
    }

    /// Get the total elapsed time
    ///
    /// Returns the maximum time across all qubits
    ///
    /// # Returns
    /// Total time in microseconds
    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    /// Reset all times to zero
    ///
    /// Useful for starting a new circuit execution
    pub fn reset(&mut self) {
        self.qubit_times.fill(0.0);
        self.total_time = 0.0;
    }

    /// Get the number of qubits being tracked
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Update the total time to the maximum qubit time
    fn update_total_time(&mut self) {
        self.total_time = self
            .qubit_times
            .iter()
            .copied()
            .fold(0.0, f64::max);
    }

    /// Get timing information
    pub fn timing(&self) -> &GateTiming {
        &self.timing
    }

    /// Update timing information
    pub fn set_timing(&mut self, timing: GateTiming) {
        self.timing = timing;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_tracker_creation() {
        let timing = GateTiming::default();
        let tracker = QubitTimeTracker::new(3, timing);

        assert_eq!(tracker.num_qubits(), 3);
        assert_eq!(tracker.total_time(), 0.0);
        assert_eq!(tracker.qubit_time(0), Some(0.0));
    }

    #[test]
    fn test_single_qubit_gate() {
        let timing = GateTiming::default();
        let mut tracker = QubitTimeTracker::new(3, timing);

        tracker.apply_single_qubit_gate(0);

        assert_eq!(tracker.qubit_time(0), Some(0.02)); // default single-qubit time
        assert_eq!(tracker.qubit_time(1), Some(0.0));
        assert_eq!(tracker.total_time(), 0.02);
    }

    #[test]
    fn test_two_qubit_gate() {
        let timing = GateTiming::default();
        let mut tracker = QubitTimeTracker::new(3, timing);

        tracker.apply_two_qubit_gate(0, 1);

        assert_eq!(tracker.qubit_time(0), Some(0.1)); // default two-qubit time
        assert_eq!(tracker.qubit_time(1), Some(0.1));
        assert_eq!(tracker.qubit_time(2), Some(0.0)); // idle
        assert_eq!(tracker.total_time(), 0.1);
    }

    #[test]
    fn test_idle_time_calculation() {
        let timing = GateTiming::default();
        let mut tracker = QubitTimeTracker::new(3, timing);

        // Operate on qubit 0
        tracker.apply_single_qubit_gate(0); // t=0.02
        // Operate on qubit 1
        tracker.apply_single_qubit_gate(1); // t=0.02
        // Operate on qubit 0 again
        tracker.apply_single_qubit_gate(0); // t=0.04

        // Qubits 1 and 2 are now idle relative to the total time
        assert_eq!(tracker.idle_time_since_last_operation(0), 0.0); // just operated
        assert_eq!(tracker.idle_time_since_last_operation(1), 0.02); // idle for 0.02μs
        assert_eq!(tracker.idle_time_since_last_operation(2), 0.04); // idle entire time
    }

    #[test]
    fn test_all_idle_times() {
        let timing = GateTiming::default();
        let mut tracker = QubitTimeTracker::new(3, timing);

        tracker.apply_single_qubit_gate(0);
        tracker.apply_single_qubit_gate(1);

        let idle_times = tracker.all_idle_times();
        assert_eq!(idle_times.len(), 3);
        assert_eq!(idle_times[2], 0.02); // qubit 2 idle for entire time
    }

    #[test]
    fn test_synchronize_all() {
        let timing = GateTiming::default();
        let mut tracker = QubitTimeTracker::new(3, timing);

        tracker.apply_single_qubit_gate(0);
        tracker.synchronize_all_qubits();

        // All qubits should now be at the same time
        assert_eq!(tracker.qubit_time(0), Some(0.02));
        assert_eq!(tracker.qubit_time(1), Some(0.02));
        assert_eq!(tracker.qubit_time(2), Some(0.02));
        assert_eq!(tracker.idle_time_since_last_operation(0), 0.0);
        assert_eq!(tracker.idle_time_since_last_operation(1), 0.0);
    }

    #[test]
    fn test_synchronize_specific() {
        let timing = GateTiming::default();
        let mut tracker = QubitTimeTracker::new(3, timing);

        tracker.apply_single_qubit_gate(0);
        tracker.synchronize_qubits(&[1]);

        assert_eq!(tracker.qubit_time(1), Some(0.02));
        assert_eq!(tracker.qubit_time(2), Some(0.0)); // not synchronized
    }

    #[test]
    fn test_measurement() {
        let timing = GateTiming::default();
        let mut tracker = QubitTimeTracker::new(2, timing);

        tracker.apply_measurement(0);

        assert_eq!(tracker.qubit_time(0), Some(1.0)); // default measurement time
        assert_eq!(tracker.total_time(), 1.0);
    }

    #[test]
    fn test_custom_duration() {
        let timing = GateTiming::default();
        let mut tracker = QubitTimeTracker::new(3, timing);

        tracker.apply_two_qubit_gate_custom(0, 1, 0.5);

        assert_eq!(tracker.qubit_time(0), Some(0.5));
        assert_eq!(tracker.qubit_time(1), Some(0.5));
    }

    #[test]
    fn test_advance_qubit_time() {
        let timing = GateTiming::default();
        let mut tracker = QubitTimeTracker::new(2, timing);

        tracker.advance_qubit_time(0, 2.5);

        assert_eq!(tracker.qubit_time(0), Some(2.5));
        assert_eq!(tracker.total_time(), 2.5);
    }

    #[test]
    fn test_reset() {
        let timing = GateTiming::default();
        let mut tracker = QubitTimeTracker::new(3, timing);

        tracker.apply_single_qubit_gate(0);
        tracker.apply_two_qubit_gate(1, 2);

        tracker.reset();

        assert_eq!(tracker.total_time(), 0.0);
        assert_eq!(tracker.qubit_time(0), Some(0.0));
        assert_eq!(tracker.qubit_time(1), Some(0.0));
    }

    #[test]
    fn test_timing_update() {
        let timing = GateTiming::default();
        let mut tracker = QubitTimeTracker::new(1, timing);

        let new_timing = GateTiming {
            single_qubit_gate_time: 0.05,
            two_qubit_gate_time: 0.2,
            measurement_time: 2.0,
        };

        tracker.set_timing(new_timing);
        tracker.apply_single_qubit_gate(0);

        assert_eq!(tracker.qubit_time(0), Some(0.05));
    }

    #[test]
    fn test_complex_sequence() {
        let timing = GateTiming::default();
        let mut tracker = QubitTimeTracker::new(4, timing);

        // H on q0
        tracker.apply_single_qubit_gate(0); // t=0.02

        // CNOT q0, q1
        tracker.apply_two_qubit_gate(0, 1); // both qubits += 0.1

        // H on q2
        tracker.apply_single_qubit_gate(2); // t=0.02 for q2

        // At this point:
        // q0: 0.02 + 0.1 = 0.12, q1: 0.1, q2: 0.02, q3: 0.0
        // Total: max = 0.12

        const TOLERANCE: f64 = 1e-6;

        let t0 = tracker.qubit_time(0).unwrap();
        let t1 = tracker.qubit_time(1).unwrap();
        let t2 = tracker.qubit_time(2).unwrap();
        let t3 = tracker.qubit_time(3).unwrap();
        let total = tracker.total_time();

        // q0 should be 0.02 (single) + 0.1 (two-qubit) = 0.12
        assert!((t0 - 0.12).abs() < TOLERANCE, "q0 time: expected 0.12, got {}", t0);
        // q1 should be 0.1 (two-qubit)
        assert!((t1 - 0.1).abs() < TOLERANCE, "q1 time: expected 0.1, got {}", t1);
        // q2 should be 0.02 (single)
        assert!((t2 - 0.02).abs() < TOLERANCE, "q2 time: expected 0.02, got {}", t2);
        // q3 should be 0
        assert_eq!(t3, 0.0);
        // Total should be max = 0.12
        assert!((total - 0.12).abs() < TOLERANCE, "total time: expected 0.12, got {}", total);

        let idle0 = tracker.idle_time_since_last_operation(0);
        let idle1 = tracker.idle_time_since_last_operation(1);
        let idle2 = tracker.idle_time_since_last_operation(2);
        let idle3 = tracker.idle_time_since_last_operation(3);

        assert!(idle0.abs() < TOLERANCE); // Should be 0
        assert!((idle1 - 0.02).abs() < TOLERANCE); // total(0.12) - t1(0.1) = 0.02
        assert!((idle2 - 0.1).abs() < TOLERANCE);  // total(0.12) - t2(0.02) = 0.1
        assert!((idle3 - 0.12).abs() < TOLERANCE); // total(0.12) - t3(0) = 0.12
    }
}
