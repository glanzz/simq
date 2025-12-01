//! Circuit analysis and statistics
//!
//! This module provides comprehensive analysis of quantum circuits including:
//! - Gate statistics (counts by type, depth)
//! - Resource estimation (memory, execution time)
//! - Circuit properties (entanglement, parallelism)

use simq_core::{Circuit, Result};
use std::collections::HashMap;

/// Statistics about gate usage in a circuit
#[derive(Clone, Debug)]
pub struct GateStatistics {
    /// Total number of gates in the circuit
    pub total_gates: usize,
    /// Number of gates by type (gate name -> count)
    pub gate_counts: HashMap<String, usize>,
    /// Number of single-qubit gates
    pub single_qubit_gates: usize,
    /// Number of two-qubit gates
    pub two_qubit_gates: usize,
    /// Number of multi-qubit gates (3+ qubits)
    pub multi_qubit_gates: usize,
    /// Circuit depth (longest path through gates)
    pub depth: usize,
    /// Number of qubits in the circuit
    pub num_qubits: usize,
}

impl GateStatistics {
    /// Compute statistics for a circuit
    pub fn from_circuit(circuit: &Circuit) -> Result<Self> {
        let mut gate_counts = HashMap::new();
        let mut single_qubit_gates = 0;
        let mut two_qubit_gates = 0;
        let mut multi_qubit_gates = 0;

        // Count gates by type and qubit count
        for op in circuit.operations() {
            let gate_name = op.gate().name().to_string();
            *gate_counts.entry(gate_name).or_insert(0) += 1;

            match op.num_qubits() {
                1 => single_qubit_gates += 1,
                2 => two_qubit_gates += 1,
                _ => multi_qubit_gates += 1,
            }
        }

        // Compute circuit depth using DAG analysis
        let depth = circuit.compute_depth().unwrap_or(circuit.len());

        Ok(Self {
            total_gates: circuit.len(),
            gate_counts,
            single_qubit_gates,
            two_qubit_gates,
            multi_qubit_gates,
            depth,
            num_qubits: circuit.num_qubits(),
        })
    }

    /// Get the count for a specific gate type
    pub fn gate_count(&self, gate_name: &str) -> usize {
        self.gate_counts.get(gate_name).copied().unwrap_or(0)
    }

    /// Get the most common gate type
    pub fn most_common_gate(&self) -> Option<(&str, usize)> {
        self.gate_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(name, count)| (name.as_str(), *count))
    }

    /// Calculate gate density (gates per qubit)
    pub fn gate_density(&self) -> f64 {
        if self.num_qubits == 0 {
            0.0
        } else {
            self.total_gates as f64 / self.num_qubits as f64
        }
    }

    /// Calculate two-qubit gate fraction
    pub fn two_qubit_fraction(&self) -> f64 {
        if self.total_gates == 0 {
            0.0
        } else {
            self.two_qubit_gates as f64 / self.total_gates as f64
        }
    }
}

impl std::fmt::Display for GateStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Circuit Statistics:")?;
        writeln!(f, "  Qubits: {}", self.num_qubits)?;
        writeln!(f, "  Total gates: {}", self.total_gates)?;
        writeln!(f, "  Circuit depth: {}", self.depth)?;
        writeln!(f, "  Gate breakdown:")?;
        writeln!(f, "    Single-qubit: {}", self.single_qubit_gates)?;
        writeln!(f, "    Two-qubit: {}", self.two_qubit_gates)?;
        writeln!(f, "    Multi-qubit: {}", self.multi_qubit_gates)?;
        writeln!(f, "  Gate counts by type:")?;

        let mut gate_types: Vec<_> = self.gate_counts.iter().collect();
        gate_types.sort_by_key(|(name, _)| *name);

        for (name, count) in gate_types {
            writeln!(f, "    {}: {}", name, count)?;
        }

        writeln!(f, "  Metrics:")?;
        writeln!(f, "    Gate density: {:.2}", self.gate_density())?;
        writeln!(f, "    Two-qubit fraction: {:.2}%", self.two_qubit_fraction() * 100.0)?;

        Ok(())
    }
}

/// Estimated resource requirements for circuit execution
#[derive(Clone, Debug)]
pub struct ResourceEstimate {
    /// Estimated memory in bytes for dense state vector
    pub dense_memory_bytes: usize,
    /// Estimated memory in bytes for sparse state vector (best case)
    pub sparse_memory_bytes_min: usize,
    /// Estimated execution time in microseconds (rough estimate)
    pub estimated_time_us: f64,
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of gates
    pub num_gates: usize,
    /// Circuit depth
    pub depth: usize,
}

impl ResourceEstimate {
    /// Estimate resources for a circuit
    ///
    /// This provides rough estimates based on circuit size and structure.
    /// Actual resource usage will vary based on:
    /// - Gate types
    /// - State sparsity
    /// - Hardware capabilities
    /// - Parallelization
    pub fn from_circuit(circuit: &Circuit) -> Result<Self> {
        let num_qubits = circuit.num_qubits();
        let num_gates = circuit.len();
        let depth = circuit.compute_depth().unwrap_or(num_gates);

        // Memory estimates
        let state_size = 1usize << num_qubits; // 2^n states
        let complex_size = 16; // Complex<f64> is 16 bytes
        let dense_memory_bytes = state_size * complex_size;

        // Sparse memory: assume best case of 1% density after initialization
        // Each sparse entry needs: 8 bytes (key) + 16 bytes (value) + overhead
        let sparse_entry_size = 32;
        let sparse_density = 0.01f64.max(1.0 / state_size as f64);
        let sparse_memory_bytes_min =
            (state_size as f64 * sparse_density * sparse_entry_size as f64) as usize;

        // Time estimates (very rough)
        let stats = GateStatistics::from_circuit(circuit)?;

        // Base time per gate (microseconds, highly hardware dependent)
        // Single-qubit: ~0.1 us
        // Two-qubit: ~1.0 us
        // Multi-qubit: ~10.0 us
        let single_qubit_time = stats.single_qubit_gates as f64 * 0.1;
        let two_qubit_time = stats.two_qubit_gates as f64 * 1.0;
        let multi_qubit_time = stats.multi_qubit_gates as f64 * 10.0;

        // Scale by state size (larger states take longer)
        let state_scale = (num_qubits as f64).exp2() / 1024.0; // normalize to 10 qubits
        let estimated_time_us =
            (single_qubit_time + two_qubit_time + multi_qubit_time) * state_scale.max(1.0);

        Ok(Self {
            dense_memory_bytes,
            sparse_memory_bytes_min,
            estimated_time_us,
            num_qubits,
            num_gates,
            depth,
        })
    }

    /// Check if circuit can fit in given memory (bytes)
    pub fn fits_in_memory(&self, available_bytes: usize) -> bool {
        self.dense_memory_bytes <= available_bytes
    }

    /// Get maximum qubits that fit in given memory
    pub fn max_qubits_for_memory(memory_bytes: usize) -> usize {
        let complex_size = 16;
        let mut qubits: usize = 0;
        while (1usize << qubits) * complex_size <= memory_bytes {
            qubits += 1;
        }
        qubits.saturating_sub(1)
    }

    /// Format memory size as human-readable string
    pub fn format_memory(bytes: usize) -> String {
        const KB: usize = 1024;
        const MB: usize = KB * 1024;
        const GB: usize = MB * 1024;

        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} bytes", bytes)
        }
    }

    /// Format time as human-readable string
    pub fn format_time(microseconds: f64) -> String {
        const MS: f64 = 1000.0;
        const SEC: f64 = MS * 1000.0;

        if microseconds >= SEC {
            format!("{:.2} s", microseconds / SEC)
        } else if microseconds >= MS {
            format!("{:.2} ms", microseconds / MS)
        } else {
            format!("{:.2} µs", microseconds)
        }
    }
}

impl std::fmt::Display for ResourceEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Resource Estimate:")?;
        writeln!(f, "  Circuit properties:")?;
        writeln!(f, "    Qubits: {}", self.num_qubits)?;
        writeln!(f, "    Gates: {}", self.num_gates)?;
        writeln!(f, "    Depth: {}", self.depth)?;
        writeln!(f, "  Memory requirements:")?;
        writeln!(f, "    Dense state: {}", Self::format_memory(self.dense_memory_bytes))?;
        writeln!(
            f,
            "    Sparse state (min): {}",
            Self::format_memory(self.sparse_memory_bytes_min)
        )?;
        writeln!(f, "  Estimated execution time: {}", Self::format_time(self.estimated_time_us))?;

        // Memory feasibility check
        let mem_32gb = 32usize * 1024 * 1024 * 1024;
        if self.dense_memory_bytes > mem_32gb {
            writeln!(f, "  ⚠️  Warning: Circuit requires more than 32GB RAM")?;
        }

        Ok(())
    }
}

/// Complete circuit analysis combining all metrics
#[derive(Clone, Debug)]
pub struct CircuitAnalysis {
    /// Gate statistics
    pub statistics: GateStatistics,
    /// Resource estimates
    pub resources: ResourceEstimate,
    /// Parallelism analysis (from DAG)
    pub parallelism: Option<simq_core::ParallelismAnalysis>,
}

impl CircuitAnalysis {
    /// Perform complete analysis of a circuit
    pub fn analyze(circuit: &Circuit) -> Result<Self> {
        let statistics = GateStatistics::from_circuit(circuit)?;
        let resources = ResourceEstimate::from_circuit(circuit)?;
        let parallelism = circuit.analyze_parallelism().ok();

        Ok(Self {
            statistics,
            resources,
            parallelism,
        })
    }

    /// Get parallelism factor (average gates per layer)
    pub fn parallelism_factor(&self) -> f64 {
        self.parallelism
            .as_ref()
            .map(|p| p.parallelism_factor)
            .unwrap_or(1.0)
    }

    /// Get maximum parallelism (gates that can run simultaneously)
    pub fn max_parallelism(&self) -> usize {
        self.parallelism
            .as_ref()
            .map(|p| p.max_parallelism)
            .unwrap_or(1)
    }
}

impl std::fmt::Display for CircuitAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Circuit Analysis ===")?;
        writeln!(f)?;
        write!(f, "{}", self.statistics)?;
        writeln!(f)?;
        write!(f, "{}", self.resources)?;

        if let Some(ref p) = self.parallelism {
            writeln!(f)?;
            writeln!(f, "Parallelism Analysis:")?;
            writeln!(f, "  Number of layers: {}", p.num_layers())?;
            writeln!(f, "  Parallelism factor: {:.2}", p.parallelism_factor)?;
            writeln!(f, "  Max parallel gates: {}", p.max_parallelism)?;
            writeln!(f, "  Average parallelism: {:.2}", p.avg_parallelism())?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_core::{gate::Gate, Circuit, QubitId};
    use std::sync::Arc;

    #[derive(Debug)]
    struct MockGate {
        name: String,
        num_qubits: usize,
    }

    impl Gate for MockGate {
        fn name(&self) -> &str {
            &self.name
        }

        fn num_qubits(&self) -> usize {
            self.num_qubits
        }
    }

    fn create_test_circuit() -> Circuit {
        let mut circuit = Circuit::new(3);
        let h_gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });
        let cnot_gate = Arc::new(MockGate {
            name: "CNOT".to_string(),
            num_qubits: 2,
        });
        let x_gate = Arc::new(MockGate {
            name: "X".to_string(),
            num_qubits: 1,
        });

        circuit
            .add_gate(h_gate.clone(), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(cnot_gate, &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        circuit.add_gate(h_gate, &[QubitId::new(2)]).unwrap();
        circuit.add_gate(x_gate, &[QubitId::new(1)]).unwrap();

        circuit
    }

    #[test]
    fn test_gate_statistics() {
        let circuit = create_test_circuit();
        let stats = GateStatistics::from_circuit(&circuit).unwrap();

        assert_eq!(stats.total_gates, 4);
        assert_eq!(stats.single_qubit_gates, 3);
        assert_eq!(stats.two_qubit_gates, 1);
        assert_eq!(stats.multi_qubit_gates, 0);
        assert_eq!(stats.num_qubits, 3);

        assert_eq!(stats.gate_count("H"), 2);
        assert_eq!(stats.gate_count("CNOT"), 1);
        assert_eq!(stats.gate_count("X"), 1);
    }

    #[test]
    fn test_gate_statistics_most_common() {
        let circuit = create_test_circuit();
        let stats = GateStatistics::from_circuit(&circuit).unwrap();

        let (name, count) = stats.most_common_gate().unwrap();
        assert_eq!(name, "H");
        assert_eq!(count, 2);
    }

    #[test]
    fn test_gate_density() {
        let circuit = create_test_circuit();
        let stats = GateStatistics::from_circuit(&circuit).unwrap();

        let density = stats.gate_density();
        assert!((density - 4.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_two_qubit_fraction() {
        let circuit = create_test_circuit();
        let stats = GateStatistics::from_circuit(&circuit).unwrap();

        let fraction = stats.two_qubit_fraction();
        assert!((fraction - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_resource_estimate() {
        let circuit = create_test_circuit();
        let estimate = ResourceEstimate::from_circuit(&circuit).unwrap();

        assert_eq!(estimate.num_qubits, 3);
        assert_eq!(estimate.num_gates, 4);

        // 3 qubits = 2^3 = 8 states * 16 bytes = 128 bytes
        assert_eq!(estimate.dense_memory_bytes, 128);

        // Sparse should be less than dense
        assert!(estimate.sparse_memory_bytes_min < estimate.dense_memory_bytes);
    }

    #[test]
    fn test_resource_estimate_large_circuit() {
        let circuit = Circuit::new(30);
        let estimate = ResourceEstimate::from_circuit(&circuit).unwrap();

        // 30 qubits = 2^30 states * 16 bytes = 16 GB
        let expected_bytes = (1usize << 30) * 16;
        assert_eq!(estimate.dense_memory_bytes, expected_bytes);
    }

    #[test]
    fn test_max_qubits_for_memory() {
        // 1 GB = 2^30 bytes
        let one_gb = 1024 * 1024 * 1024;
        let max_qubits = ResourceEstimate::max_qubits_for_memory(one_gb);

        // With 1 GB, we can store 2^26 states (26 qubits)
        // 2^26 * 16 bytes = 1 GB
        assert_eq!(max_qubits, 26);
    }

    #[test]
    fn test_format_memory() {
        assert_eq!(ResourceEstimate::format_memory(512), "512 bytes");
        assert_eq!(ResourceEstimate::format_memory(1024), "1.00 KB");
        assert_eq!(ResourceEstimate::format_memory(1024 * 1024), "1.00 MB");
        assert_eq!(ResourceEstimate::format_memory(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_time() {
        assert_eq!(ResourceEstimate::format_time(100.0), "100.00 µs");
        assert_eq!(ResourceEstimate::format_time(1500.0), "1.50 ms");
        assert_eq!(ResourceEstimate::format_time(2_000_000.0), "2.00 s");
    }

    #[test]
    fn test_circuit_analysis() {
        let circuit = create_test_circuit();
        let analysis = CircuitAnalysis::analyze(&circuit).unwrap();

        assert_eq!(analysis.statistics.total_gates, 4);
        assert_eq!(analysis.resources.num_qubits, 3);
        assert!(analysis.parallelism_factor() >= 1.0);
    }

    #[test]
    fn test_empty_circuit_analysis() {
        let circuit = Circuit::new(2);
        let stats = GateStatistics::from_circuit(&circuit).unwrap();

        assert_eq!(stats.total_gates, 0);
        assert_eq!(stats.single_qubit_gates, 0);
        assert_eq!(stats.two_qubit_gates, 0);
        assert_eq!(stats.gate_density(), 0.0);
    }
}
