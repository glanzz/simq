//! Core simulator implementation

use simq_compiler::pipeline::{create_compiler, OptimizationLevel};
use simq_core::Circuit;
use simq_state::AdaptiveState;
use std::time::Instant;

use crate::{
    config::SimulatorConfig,
    error::{Result, SimulatorError},
    result::SimulationResult,
    statistics::ExecutionStatistics,
};

/// High-performance quantum circuit simulator
///
/// The simulator provides efficient quantum state evolution with automatic
/// optimization, sparse/dense representation switching, and parallel execution.
///
/// # Example
///
/// ```ignore
/// use simq_sim::{Simulator, SimulatorConfig};
/// use simq_core::Circuit;
/// use simq_gates::standard::Hadamard;
/// use std::sync::Arc;
///
/// let simulator = Simulator::new(SimulatorConfig::default());
///
/// let mut circuit = Circuit::new(2);
/// circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
///
/// let result = simulator.run(&circuit).unwrap();
/// ```
pub struct Simulator {
    config: SimulatorConfig,
}

impl Simulator {
    /// Create a new simulator with the given configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Simulator configuration
    ///
    /// # Panics
    ///
    /// Panics if the configuration is invalid.
    pub fn new(config: SimulatorConfig) -> Self {
        config.validate().expect("Invalid simulator configuration");

        Self { config }
    }

    /// Create a simulator with default configuration
    pub fn default() -> Self {
        Self::new(SimulatorConfig::default())
    }

    /// Get the simulator configuration
    pub fn config(&self) -> &SimulatorConfig {
        &self.config
    }

    /// Run a quantum circuit simulation
    ///
    /// This is the main entry point for circuit execution. It:
    /// 1. Compiles/optimizes the circuit (if enabled)
    /// 2. Initializes the quantum state
    /// 3. Executes all gates
    /// 4. Collects statistics (if enabled)
    ///
    /// # Arguments
    ///
    /// * `circuit` - The quantum circuit to simulate
    ///
    /// # Returns
    ///
    /// A `SimulationResult` containing the final state and statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The circuit is invalid
    /// - Too many qubits for available memory
    /// - Compilation fails
    pub fn run(&self, circuit: &Circuit) -> Result<SimulationResult> {
        let total_start = Instant::now();

        // Validate circuit
        if circuit.len() == 0 {
            return Err(SimulatorError::InvalidCircuit(
                "Empty circuit".to_string(),
            ));
        }

        // Check qubit count
        let num_qubits = circuit.num_qubits();
        let max_qubits = self.estimate_max_qubits();
        if num_qubits > max_qubits {
            return Err(SimulatorError::TooManyQubits {
                num_qubits,
                max_qubits,
            });
        }

        let mut stats = if self.config.collect_statistics {
            Some(ExecutionStatistics::new())
        } else {
            None
        };

        // 1. Compile circuit
        let (compiled_circuit, compilation_time) = if self.config.optimize_circuit {
            self.compile_circuit(circuit)?
        } else {
            (circuit.clone(), std::time::Duration::ZERO)
        };

        if let Some(ref mut s) = stats {
            s.compilation_time = compilation_time;
            s.gates_executed = circuit.len();
            s.optimized_gates = compiled_circuit.len();
        }

        // 2. Initialize state
        let init_start = Instant::now();
        let state = AdaptiveState::new(num_qubits)?;
        let init_time = init_start.elapsed();

        if let Some(ref mut s) = stats {
            s.initialization_time = init_time;
        }


        // 3. Execute circuit using execution engine
        let gate_start = Instant::now();
        {
            use crate::execution_engine::{ExecutionEngine, ExecutionConfig};
            let exec_config = ExecutionConfig {
                use_parallel: self.config.parallel_threshold > 0,
                use_simd: true,
            };
            let engine = ExecutionEngine::new(exec_config);
            // AdaptiveState exposes as_mut() for mutation
            engine.execute(&compiled_circuit, state.as_mut());
        }
        let gate_time = gate_start.elapsed();

        if let Some(ref mut s) = stats {
            s.gate_application_time = gate_time;
            s.final_density = state.density() as f64;
            s.final_is_sparse = matches!(state, AdaptiveState::Sparse { .. });
            // Calculate memory usage: 2^n amplitudes × 16 bytes each
            s.peak_memory_bytes = (1 << num_qubits) * 16;
            s.total_time = total_start.elapsed();
        }

        // 4. Build result
        let mut result = SimulationResult::new(state);

        if let Some(s) = stats {
            result = result.with_statistics(s);
        }

        Ok(result)
    }

    /// Compile and optimize a circuit
    fn compile_circuit(&self, circuit: &Circuit) -> Result<(Circuit, std::time::Duration)> {
        let start = Instant::now();

        let opt_level = match self.config.optimization_level {
            0 => OptimizationLevel::O0,
            1 => OptimizationLevel::O1,
            2 => OptimizationLevel::O2,
            _ => OptimizationLevel::O3,
        };

        let compiler = create_compiler(opt_level);
        let mut compiled = circuit.clone();

        compiler
            .compile(&mut compiled)
            .map_err(|e| SimulatorError::CompilationFailed(e.to_string()))?;

        let compilation_time = start.elapsed();

        Ok((compiled, compilation_time))
    }

    /// Estimate maximum number of qubits that can be simulated
    ///
    /// Based on memory limit and state vector size.
    fn estimate_max_qubits(&self) -> usize {
        if self.config.memory_limit == 0 {
            // No limit - use practical maximum (35-40 qubits)
            40
        } else {
            // Each amplitude is 16 bytes (Complex<f64>)
            // State vector size = 2^n * 16 bytes
            let max_amplitudes = self.config.memory_limit / 16;
            (max_amplitudes as f64).log2().floor() as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_core::QubitId;
    use simq_gates::standard::{CNot, Hadamard, PauliX};
    use std::sync::Arc;

    #[test]
    fn test_simulator_creation() {
        let sim = Simulator::default();
        assert!(sim.config().optimize_circuit);
        assert_eq!(sim.config().shots, 1024);
    }

    #[test]
    fn test_simple_circuit() {
        let sim = Simulator::new(SimulatorConfig::default());

        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();

        let result = sim.run(&circuit);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.num_qubits(), 2);
    }

    #[test]
    fn test_empty_circuit() {
        let sim = Simulator::default();
        let circuit = Circuit::new(2);

        let result = sim.run(&circuit);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_statistics() {
        let sim = Simulator::new(SimulatorConfig::default().with_statistics(true));

        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();

        let result = sim.run(&circuit).unwrap();
        assert!(result.statistics.is_some());

        let stats = result.statistics.unwrap();
        assert_eq!(stats.gates_executed, 1);
    }

    #[test]
    fn test_optimization() {
        let sim_opt = Simulator::new(SimulatorConfig::default().with_statistics(true));
        let sim_no_opt = Simulator::new(
            SimulatorConfig::default()
                .with_optimization(false)
                .with_statistics(true),
        );

        // Circuit with redundant gates
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap(); // Cancels

        let result_opt = sim_opt.run(&circuit).unwrap();
        let result_no_opt = sim_no_opt.run(&circuit).unwrap();

        let stats_opt = result_opt.statistics.unwrap();
        let stats_no_opt = result_no_opt.statistics.unwrap();

        // Optimized should have fewer gates
        assert!(stats_opt.optimized_gates < stats_no_opt.optimized_gates);
    }

    #[test]
    fn test_max_qubits_estimation() {
        let sim = Simulator::new(
            SimulatorConfig::default().with_memory_limit(1024 * 1024) // 1 MB
        );

        let max_qubits = sim.estimate_max_qubits();
        // 1 MB / 16 bytes = 65536 amplitudes = 2^16, so 16 qubits
        assert_eq!(max_qubits, 16);
    }
}
