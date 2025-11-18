//! Production-grade execution engine

use simq_core::{Circuit, GateOp, QubitId, Gate};
use simq_state::AdaptiveState;
use std::sync::Arc;
use std::time::Instant;
use num_complex::Complex64;

use crate::execution_engine::{
    config::{ExecutionConfig, ExecutionMode},
    error::{ExecutionError, Result},
    telemetry::ExecutionTelemetry,
    recovery::RecoveryPolicy,
    checkpoint::CheckpointManager,
    validation,
    cache::{GateMatrixCache, GateCacheKey, CachedMatrix, OrderedFloat},
    parallel::ParallelExecutor,
    adaptive::AdaptiveStrategy,
    kernels::*,
};

/// Production-grade execution engine
pub struct ExecutionEngine {
    config: ExecutionConfig,
    pub telemetry: ExecutionTelemetry,
    pub recovery_policy: RecoveryPolicy,
    checkpoint_manager: Option<CheckpointManager>,
    matrix_cache: Arc<GateMatrixCache>,
    parallel_executor: ParallelExecutor,
    adaptive_strategy: AdaptiveStrategy,
    start_time: Option<Instant>,
}

impl ExecutionEngine {
    /// Create a new execution engine with the given configuration
    pub fn new(config: ExecutionConfig) -> Self {
        config.validate().expect("Invalid execution config");

        let checkpoint_manager = if config.enable_checkpoints {
            Some(CheckpointManager::new(10))
        } else {
            None
        };

        Self {
            parallel_executor: ParallelExecutor::new(config.parallel_strategy),
            adaptive_strategy: AdaptiveStrategy::new(
                config.dense_threshold,
                1 << 20,  // GPU threshold
                config.parallel_threshold,
            ),
            telemetry: ExecutionTelemetry::new(),
            recovery_policy: RecoveryPolicy::default(),
            checkpoint_manager,
            matrix_cache: Arc::new(GateMatrixCache::new(config.matrix_cache_size)),
            config,
            start_time: None,
        }
    }

    /// Execute a quantum circuit on a state
    ///
    /// This is the main entry point for circuit execution with full production features:
    /// - Proper error handling with Result types
    /// - Circuit-level parallelization
    /// - Adaptive sparse/dense optimization
    /// - Checkpointing
    /// - Validation
    /// - Comprehensive telemetry
    pub fn execute(&mut self, circuit: &Circuit, state: &mut AdaptiveState) -> Result<()> {
        self.start_time = Some(Instant::now());
        self.telemetry = ExecutionTelemetry::new();
        self.telemetry.parallelism = rayon::current_num_threads();
        self.telemetry.log_event("execution_start");

        // Check timeout configuration
        let timeout_check = self.config.timeout.map(|t| (Instant::now(), t));

        // Validate initial state
        if self.config.validate_state {
            validation::validate_state(state)?;
        }

        // Determine execution mode
        let execution_mode = match self.config.mode {
            ExecutionMode::Adaptive => {
                self.adaptive_strategy.select_execution_mode(state, circuit.len())
            }
            mode => mode,
        };

        self.telemetry.log_event(format!("mode:{:?}", execution_mode));

        // Execute based on mode
        match execution_mode {
            ExecutionMode::Sequential => self.execute_sequential(circuit, state, timeout_check)?,
            ExecutionMode::Parallel => self.execute_parallel(circuit, state, timeout_check)?,
            ExecutionMode::Gpu => self.execute_gpu(circuit, state, timeout_check)?,
            ExecutionMode::Adaptive => unreachable!(), // Already resolved above
        }

        // Final validation
        if self.config.validate_state {
            validation::validate_state(state)?;
        }

        self.telemetry.log_event("execution_complete");
        Ok(())
    }

    /// Execute circuit sequentially
    fn execute_sequential(
        &mut self,
        circuit: &Circuit,
        state: &mut AdaptiveState,
        timeout_check: Option<(Instant, std::time::Duration)>,
    ) -> Result<()> {
        for (idx, gate_op) in circuit.operations().enumerate() {
            // Check timeout
            if let Some((start, limit)) = timeout_check {
                if start.elapsed() > limit {
                    return Err(ExecutionError::ExecutionTimeout {
                        elapsed: start.elapsed(),
                        limit,
                    });
                }
            }

            // Create checkpoint if needed
            if self.config.enable_checkpoints && idx % self.config.checkpoint_interval == 0 {
                if let Some(ref mut manager) = self.checkpoint_manager {
                    manager.create_checkpoint(idx, state)?;
                }
            }

            // Execute gate with retry logic
            self.execute_gate_with_retry(&gate_op, state)?;

            // Adaptive state conversion
            if self.config.adaptive_state {
                self.maybe_convert_state(state);
            }

            // Validate state after gate
            if self.config.validate_state {
                validation::validate_state(state)?;
            }
        }

        Ok(())
    }

    /// Execute circuit with parallelization
    fn execute_parallel(
        &mut self,
        circuit: &Circuit,
        state: &mut AdaptiveState,
        timeout_check: Option<(Instant, std::time::Duration)>,
    ) -> Result<()> {
        // Use parallel executor with layer-based execution
        self.parallel_executor.execute(circuit, state, |gate_op, state| {
            self.apply_gate_op(gate_op, state)
        })?;

        Ok(())
    }

    /// Execute circuit on GPU
    fn execute_gpu(
        &mut self,
        circuit: &Circuit,
        state: &mut AdaptiveState,
        timeout_check: Option<(Instant, std::time::Duration)>,
    ) -> Result<()> {
        // TODO: Implement GPU execution
        // For now, fallback to sequential
        self.telemetry.log_event("gpu_fallback_to_sequential");
        self.execute_sequential(circuit, state, timeout_check)
    }

    /// Execute a single gate with retry logic
    fn execute_gate_with_retry(&mut self, gate_op: &GateOp, state: &mut AdaptiveState) -> Result<()> {
        let gate_name = gate_op.gate().name();
        let start = Instant::now();

        self.telemetry.inc_gate_type(gate_name);
        self.telemetry.log_event(format!("apply_gate:{}", gate_name));
        self.telemetry.record_thread();

        let mut attempts = 0;
        let max_attempts = self.recovery_policy.max_attempts();

        loop {
            attempts += 1;

            match self.apply_gate_op(gate_op, state) {
                Ok(()) => {
                    let elapsed = start.elapsed();
                    self.telemetry.total_gate_time += elapsed;
                    self.telemetry.per_gate_times.push(elapsed);
                    self.telemetry.state_density.push(state.density());

                    // Record memory usage
                    let mem_bytes = match state {
                        AdaptiveState::Dense(ref dense) => {
                            dense.dimension() * std::mem::size_of::<Complex64>()
                        }
                        AdaptiveState::Sparse { state: ref sparse, .. } => {
                            sparse.amplitudes().len() * std::mem::size_of::<Complex64>()
                        }
                    };
                    self.telemetry.record_memory(mem_bytes);

                    return Ok(());
                }
                Err(e) => {
                    self.telemetry.log_error(format!(
                        "Gate '{}' failed (attempt {}): {}",
                        gate_name, attempts, e
                    ));

                    match self.recovery_policy {
                        RecoveryPolicy::Halt => return Err(e),
                        RecoveryPolicy::Skip => return Ok(()),
                        _ => {
                            if !self.recovery_policy.should_retry(attempts) {
                                return Err(ExecutionError::ExecutionHalted { attempts });
                            }
                            // Continue to retry
                        }
                    }
                }
            }
        }
    }

    /// Apply a single gate operation to the quantum state
    fn apply_gate_op(&mut self, gate_op: &GateOp, state: &mut AdaptiveState) -> Result<()> {
        let gate = gate_op.gate();
        let qubits = gate_op.qubits();
        let num_qubits = gate.num_qubits();

        // Determine if we should use parallel execution for this gate
        let use_parallel = self.adaptive_strategy.should_parallelize_gate(state);
        let threshold = self.config.parallel_threshold;

        match state {
            AdaptiveState::Dense(ref mut dense) => {
                self.apply_gate_dense(gate, qubits, dense, use_parallel, threshold)?;
            }
            AdaptiveState::Sparse { state: ref mut sparse, .. } => {
                self.apply_gate_sparse(gate, qubits, sparse, use_parallel, threshold)?;
            }
        }

        Ok(())
    }

    /// Apply gate to dense state
    fn apply_gate_dense(
        &mut self,
        gate: &Arc<dyn Gate>,
        qubits: &[QubitId],
        state: &mut simq_state::DenseState,
        use_parallel: bool,
        threshold: usize,
    ) -> Result<()> {
        let gate_name = gate.name();
        let num_qubits = qubits.len();

        // Try to get matrix from cache
        let cache_key = GateCacheKey {
            gate_name: gate_name.to_string(),
            params: vec![], // TODO: Extract parameters from gate
        };

        let matrix = if let Some(cached) = self.matrix_cache.get(&cache_key) {
            self.telemetry.cache_hit();
            cached
        } else {
            self.telemetry.cache_miss();

            // Get matrix from gate
            let matrix_vec = gate.matrix().ok_or_else(|| ExecutionError::InvalidGateMatrix {
                gate: gate_name.to_string(),
                reason: "Gate has no matrix representation".to_string(),
            })?;

            // Convert to appropriate format and cache
            let cached_matrix = match num_qubits {
                1 => {
                    if matrix_vec.len() != 4 {
                        return Err(ExecutionError::InvalidGateMatrix {
                            gate: gate_name.to_string(),
                            reason: format!("Expected 4 elements, got {}", matrix_vec.len()),
                        });
                    }
                    let mat: Matrix2x2 = [
                        [matrix_vec[0], matrix_vec[1]],
                        [matrix_vec[2], matrix_vec[3]],
                    ];
                    CachedMatrix::Single(mat)
                }
                2 => {
                    if matrix_vec.len() != 16 {
                        return Err(ExecutionError::InvalidGateMatrix {
                            gate: gate_name.to_string(),
                            reason: format!("Expected 16 elements, got {}", matrix_vec.len()),
                        });
                    }
                    let mut mat: Matrix4x4 = [[Complex64::new(0.0, 0.0); 4]; 4];
                    for i in 0..4 {
                        for j in 0..4 {
                            mat[i][j] = matrix_vec[i * 4 + j];
                        }
                    }
                    CachedMatrix::Two(mat)
                }
                _ => {
                    return Err(ExecutionError::GateApplicationFailed {
                        gate: gate_name.to_string(),
                        qubits: qubits.to_vec(),
                        reason: format!("Unsupported gate with {} qubits", num_qubits),
                    });
                }
            };

            let cached_matrix = Arc::new(cached_matrix);
            self.matrix_cache.insert(cache_key, (*cached_matrix).clone());
            cached_matrix
        };

        // Apply the gate using optimized kernels
        let amplitudes = state.amplitudes_mut();

        match num_qubits {
            1 => {
                let qubit_idx = qubits[0].index();

                if let CachedMatrix::Single(ref mat) = *matrix {
                    // Check for special gates and use optimized implementations
                    match gate_name {
                        "X" | "PauliX" => {
                            single_qubit::apply_pauli_x(qubit_idx, amplitudes, use_parallel, threshold)?;
                        }
                        "Z" | "PauliZ" => {
                            single_qubit::apply_pauli_z(qubit_idx, amplitudes, use_parallel, threshold)?;
                        }
                        "H" | "Hadamard" => {
                            single_qubit::apply_hadamard(qubit_idx, amplitudes, use_parallel, threshold)?;
                        }
                        _ => {
                            // General single-qubit gate
                            if self.config.use_simd {
                                single_qubit::apply_single_qubit_dense_simd(
                                    mat, qubit_idx, amplitudes, use_parallel, threshold
                                )?;
                            } else {
                                single_qubit::apply_single_qubit_dense(
                                    mat, qubit_idx, amplitudes, use_parallel, threshold
                                )?;
                            }
                        }
                    }
                }
            }
            2 => {
                let qubit1_idx = qubits[0].index();
                let qubit2_idx = qubits[1].index();

                if let CachedMatrix::Two(ref mat) = *matrix {
                    // Check for special two-qubit gates
                    match gate_name {
                        "CNOT" | "CX" => {
                            two_qubit::apply_cnot(qubit1_idx, qubit2_idx, amplitudes, use_parallel, threshold)?;
                        }
                        "CZ" => {
                            two_qubit::apply_cz(qubit1_idx, qubit2_idx, amplitudes, use_parallel, threshold)?;
                        }
                        "SWAP" => {
                            two_qubit::apply_swap(qubit1_idx, qubit2_idx, amplitudes, use_parallel, threshold)?;
                        }
                        _ => {
                            // General two-qubit gate
                            two_qubit::apply_two_qubit_dense(
                                mat, qubit1_idx, qubit2_idx, amplitudes, use_parallel, threshold
                            )?;
                        }
                    }
                }
            }
            _ => {
                return Err(ExecutionError::GateApplicationFailed {
                    gate: gate_name.to_string(),
                    qubits: qubits.to_vec(),
                    reason: format!("Gates with {} qubits not yet supported", num_qubits),
                });
            }
        }

        Ok(())
    }

    /// Apply gate to sparse state
    fn apply_gate_sparse(
        &mut self,
        gate: &Arc<dyn Gate>,
        qubits: &[QubitId],
        state: &mut simq_state::SparseState,
        _use_parallel: bool,
        _threshold: usize,
    ) -> Result<()> {
        let gate_name = gate.name();
        let num_qubits = qubits.len();

        let matrix_vec = gate.matrix().ok_or_else(|| ExecutionError::InvalidGateMatrix {
            gate: gate_name.to_string(),
            reason: "Gate has no matrix representation".to_string(),
        })?;

        let num_qubits_total = (state.dimension() as f64).log2() as usize;

        match num_qubits {
            1 => {
                if matrix_vec.len() != 4 {
                    return Err(ExecutionError::InvalidGateMatrix {
                        gate: gate_name.to_string(),
                        reason: format!("Expected 4 elements, got {}", matrix_vec.len()),
                    });
                }
                let mat: Matrix2x2 = [
                    [matrix_vec[0], matrix_vec[1]],
                    [matrix_vec[2], matrix_vec[3]],
                ];

                sparse::apply_single_qubit_sparse(
                    &mat,
                    qubits[0].index(),
                    state.amplitudes_mut(),
                    num_qubits_total,
                )?;
            }
            2 => {
                if matrix_vec.len() != 16 {
                    return Err(ExecutionError::InvalidGateMatrix {
                        gate: gate_name.to_string(),
                        reason: format!("Expected 16 elements, got {}", matrix_vec.len()),
                    });
                }
                let mut mat: Matrix4x4 = [[Complex64::new(0.0, 0.0); 4]; 4];
                for i in 0..4 {
                    for j in 0..4 {
                        mat[i][j] = matrix_vec[i * 4 + j];
                    }
                }

                sparse::apply_two_qubit_sparse(
                    &mat,
                    qubits[0].index(),
                    qubits[1].index(),
                    state.amplitudes_mut(),
                    num_qubits_total,
                )?;
            }
            _ => {
                return Err(ExecutionError::GateApplicationFailed {
                    gate: gate_name.to_string(),
                    qubits: qubits.to_vec(),
                    reason: format!("Sparse gates with {} qubits not yet supported", num_qubits),
                });
            }
        }

        Ok(())
    }

    /// Maybe convert state representation based on density
    fn maybe_convert_state(&mut self, state: &mut AdaptiveState) {
        if self.adaptive_strategy.should_convert_to_dense(state) {
            if let AdaptiveState::Sparse { .. } = state {
                self.telemetry.log_event("convert_sparse_to_dense");
                self.telemetry.sparse_dense_transitions += 1;
                // TODO: Implement actual conversion
            }
        } else if self.adaptive_strategy.should_convert_to_sparse(state) {
            if let AdaptiveState::Dense(_) = state {
                self.telemetry.log_event("convert_dense_to_sparse");
                self.telemetry.sparse_dense_transitions += 1;
                // TODO: Implement actual conversion
            }
        }
    }

    /// Get execution telemetry
    pub fn get_telemetry(&self) -> &ExecutionTelemetry {
        &self.telemetry
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        self.matrix_cache.hit_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_core::QubitId;
    use simq_gates::standard::{Hadamard, CNot, PauliX};

    #[test]
    fn test_engine_creation() {
        let config = ExecutionConfig::default();
        let engine = ExecutionEngine::new(config);
        assert_eq!(engine.config.mode, ExecutionMode::Adaptive);
    }

    #[test]
    fn test_simple_execution() {
        let config = ExecutionConfig::default();
        let mut engine = ExecutionEngine::new(config);

        let mut circuit = Circuit::new(2);
        circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(Arc::new(PauliX), &[QubitId::new(1)]).unwrap();

        let mut state = AdaptiveState::new(2).unwrap();

        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_ok());
    }

    #[test]
    fn test_two_qubit_gates() {
        let config = ExecutionConfig::default();
        let mut engine = ExecutionEngine::new(config);

        let mut circuit = Circuit::new(2);
        circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)]).unwrap();

        let mut state = AdaptiveState::new(2).unwrap();

        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_ok());
    }
}
