//! Execution statistics tracking

use std::time::Duration;

/// Execution statistics for a simulation run
#[derive(Debug, Clone, Default)]
pub struct ExecutionStatistics {
    /// Total execution time
    pub total_time: Duration,

    /// Time spent on circuit compilation
    pub compilation_time: Duration,

    /// Time spent initializing state
    pub initialization_time: Duration,

    /// Time spent applying gates
    pub gate_application_time: Duration,

    /// Time spent on measurements
    pub measurement_time: Duration,

    /// Number of gates executed
    pub gates_executed: usize,

    /// Number of gates after optimization
    pub optimized_gates: usize,

    /// Number of sparse-to-dense conversions
    pub sparse_to_dense_conversions: usize,

    /// Peak memory usage (estimated, in bytes)
    pub peak_memory_bytes: usize,

    /// Final state density (fraction of non-zero amplitudes)
    pub final_density: f64,

    /// Whether the final state is sparse
    pub final_is_sparse: bool,
}

impl ExecutionStatistics {
    /// Create a new statistics object
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the gate optimization ratio
    ///
    /// Returns the fraction of gates eliminated by optimization.
    pub fn optimization_ratio(&self) -> f64 {
        if self.gates_executed == 0 {
            0.0
        } else {
            1.0 - (self.optimized_gates as f64 / self.gates_executed as f64)
        }
    }

    /// Get the gate execution rate (gates per second)
    pub fn gates_per_second(&self) -> f64 {
        let secs = self.gate_application_time.as_secs_f64();
        if secs == 0.0 {
            0.0
        } else {
            self.optimized_gates as f64 / secs
        }
    }

    /// Get the compilation overhead as a percentage of total time
    pub fn compilation_overhead_percent(&self) -> f64 {
        let total_secs = self.total_time.as_secs_f64();
        if total_secs == 0.0 {
            0.0
        } else {
            100.0 * self.compilation_time.as_secs_f64() / total_secs
        }
    }

    /// Get peak memory usage in MB
    pub fn peak_memory_mb(&self) -> f64 {
        self.peak_memory_bytes as f64 / 1_000_000.0
    }
}

impl std::fmt::Display for ExecutionStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Execution Statistics:")?;
        writeln!(f, "  Total time: {:?}", self.total_time)?;
        writeln!(f, "    Compilation: {:?} ({:.1}%)",
            self.compilation_time,
            self.compilation_overhead_percent())?;
        writeln!(f, "    Initialization: {:?}", self.initialization_time)?;
        writeln!(f, "    Gate application: {:?}", self.gate_application_time)?;
        writeln!(f, "    Measurement: {:?}", self.measurement_time)?;

        writeln!(f, "\n  Gates:")?;
        writeln!(f, "    Original: {}", self.gates_executed)?;
        writeln!(f, "    Optimized: {} ({:.1}% reduction)",
            self.optimized_gates,
            self.optimization_ratio() * 100.0)?;
        writeln!(f, "    Execution rate: {:.0} gates/sec", self.gates_per_second())?;

        writeln!(f, "\n  Memory:")?;
        writeln!(f, "    Peak usage: {:.2} MB", self.peak_memory_mb())?;

        writeln!(f, "\n  State:")?;
        writeln!(f, "    Final representation: {}",
            if self.final_is_sparse { "sparse" } else { "dense" })?;
        writeln!(f, "    Final density: {:.2}%", self.final_density * 100.0)?;
        writeln!(f, "    Sparse→Dense conversions: {}", self.sparse_to_dense_conversions)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_ratio() {
        let stats = ExecutionStatistics {
            gates_executed: 100,
            optimized_gates: 60,
            ..Default::default()
        };

        assert!((stats.optimization_ratio() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_gates_per_second() {
        let stats = ExecutionStatistics {
            optimized_gates: 1000,
            gate_application_time: Duration::from_millis(100),
            ..Default::default()
        };

        assert!((stats.gates_per_second() - 10000.0).abs() < 1.0);
    }

    #[test]
    fn test_compilation_overhead() {
        let stats = ExecutionStatistics {
            total_time: Duration::from_secs(1),
            compilation_time: Duration::from_millis(100),
            ..Default::default()
        };

        assert!((stats.compilation_overhead_percent() - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_peak_memory_mb() {
        let stats = ExecutionStatistics {
            peak_memory_bytes: 5_000_000,
            ..Default::default()
        };

        assert!((stats.peak_memory_mb() - 5.0).abs() < 0.01);
    }
}
