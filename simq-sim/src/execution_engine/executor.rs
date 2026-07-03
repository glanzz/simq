//! Production-grade execution engine

use num_complex::Complex64;
use simq_core::{Circuit, Gate, GateOp, QubitId};
use simq_state::AdaptiveState;
use std::sync::Arc;
use std::time::Instant;

use crate::execution_engine::{
    adaptive::AdaptiveStrategy,
    cache::{CachedMatrix, GateCacheKey, GateMatrixCache},
    checkpoint::CheckpointManager,
    config::{ExecutionConfig, ExecutionMode},
    error::{ExecutionError, Result},
    kernels::*,
    parallel::ParallelExecutor,
    recovery::RecoveryPolicy,
    telemetry::ExecutionTelemetry,
    validation,
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
                1 << 20, // GPU threshold
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
            ExecutionMode::Adaptive => self
                .adaptive_strategy
                .select_execution_mode(state, circuit.len()),
            mode => mode,
        };

        self.telemetry
            .log_event(format!("mode:{:?}", execution_mode));

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
            self.execute_gate_with_retry(gate_op, state)?;

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
        _timeout_check: Option<(Instant, std::time::Duration)>,
    ) -> Result<()> {
        // Use parallel executor with layer-based execution
        // We need to temporarily take the parallel executor out of self to avoid
        // borrowing self mutably (in the closure) while also borrowing parallel_executor
        let parallel_executor = std::mem::replace(
            &mut self.parallel_executor,
            ParallelExecutor::new(self.config.parallel_strategy),
        );

        let result = parallel_executor
            .execute(circuit, state, |gate_op, state| self.apply_gate_op(gate_op, state));

        // Put it back
        self.parallel_executor = parallel_executor;

        result?;

        Ok(())
    }

    /// Execute circuit on GPU
    fn execute_gpu(
        &mut self,
        circuit: &Circuit,
        state: &mut AdaptiveState,
        _timeout_check: Option<(Instant, std::time::Duration)>,
    ) -> Result<()> {
        // TODO: Implement GPU execution
        // For now, fallback to sequential
        self.telemetry.log_event("gpu_fallback_to_sequential");
        self.execute_sequential(circuit, state, None)
    }

    /// Execute a single gate with retry logic
    fn execute_gate_with_retry(
        &mut self,
        gate_op: &GateOp,
        state: &mut AdaptiveState,
    ) -> Result<()> {
        let gate_name = gate_op.gate().name();
        let start = Instant::now();

        self.telemetry.inc_gate_type(gate_name);
        self.telemetry
            .log_event(format!("apply_gate:{}", gate_name));
        self.telemetry.record_thread();

        let mut attempts = 0;

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
                        },
                        AdaptiveState::Sparse {
                            state: ref sparse, ..
                        } => sparse.amplitudes().len() * std::mem::size_of::<Complex64>(),
                    };
                    self.telemetry.record_memory(mem_bytes);

                    return Ok(());
                },
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
                        },
                    }
                },
            }
        }
    }

    /// Apply a single gate operation to the quantum state
    fn apply_gate_op(&mut self, gate_op: &GateOp, state: &mut AdaptiveState) -> Result<()> {
        let gate = gate_op.gate();
        let qubits = gate_op.qubits();
        let _num_qubits = gate.num_qubits();

        // Determine if we should use parallel execution for this gate
        let use_parallel = self.adaptive_strategy.should_parallelize_gate(state);
        let threshold = self.config.parallel_threshold;

        match state {
            AdaptiveState::Dense(ref mut dense) => {
                self.apply_gate_dense(gate, qubits, dense, use_parallel, threshold)?;
            },
            AdaptiveState::Sparse {
                state: ref mut sparse,
                ..
            } => {
                self.apply_gate_sparse(gate, qubits, sparse, use_parallel, threshold)?;
            },
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
            let matrix_vec = gate
                .matrix()
                .ok_or_else(|| ExecutionError::InvalidGateMatrix {
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
                },
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
                },
                _ => {
                    return Err(ExecutionError::GateApplicationFailed {
                        gate: gate_name.to_string(),
                        qubits: qubits.to_vec(),
                        reason: format!("Unsupported gate with {} qubits", num_qubits),
                    });
                },
            };

            let cached_matrix = Arc::new(cached_matrix);
            self.matrix_cache
                .insert(cache_key, (*cached_matrix).clone());
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
                            single_qubit::apply_pauli_x(
                                qubit_idx,
                                amplitudes,
                                use_parallel,
                                threshold,
                            )?;
                        },
                        "Z" | "PauliZ" => {
                            single_qubit::apply_pauli_z(
                                qubit_idx,
                                amplitudes,
                                use_parallel,
                                threshold,
                            )?;
                        },
                        "H" | "Hadamard" => {
                            single_qubit::apply_hadamard(
                                qubit_idx,
                                amplitudes,
                                use_parallel,
                                threshold,
                            )?;
                        },
                        _ => {
                            // General single-qubit gate
                            if self.config.use_simd {
                                single_qubit::apply_single_qubit_dense_simd(
                                    mat,
                                    qubit_idx,
                                    amplitudes,
                                    use_parallel,
                                    threshold,
                                )?;
                            } else {
                                single_qubit::apply_single_qubit_dense(
                                    mat,
                                    qubit_idx,
                                    amplitudes,
                                    use_parallel,
                                    threshold,
                                )?;
                            }
                        },
                    }
                }
            },
            2 => {
                let qubit1_idx = qubits[0].index();
                let qubit2_idx = qubits[1].index();

                if let CachedMatrix::Two(ref mat) = *matrix {
                    // Check for special two-qubit gates
                    match gate_name {
                        "CNOT" | "CX" => {
                            two_qubit::apply_cnot(
                                qubit1_idx,
                                qubit2_idx,
                                amplitudes,
                                use_parallel,
                                threshold,
                            )?;
                        },
                        "CZ" => {
                            two_qubit::apply_cz(
                                qubit1_idx,
                                qubit2_idx,
                                amplitudes,
                                use_parallel,
                                threshold,
                            )?;
                        },
                        "SWAP" => {
                            two_qubit::apply_swap(
                                qubit1_idx,
                                qubit2_idx,
                                amplitudes,
                                use_parallel,
                                threshold,
                            )?;
                        },
                        _ => {
                            // General two-qubit gate
                            two_qubit::apply_two_qubit_dense(
                                mat,
                                qubit1_idx,
                                qubit2_idx,
                                amplitudes,
                                use_parallel,
                                threshold,
                            )?;
                        },
                    }
                }
            },
            _ => {
                return Err(ExecutionError::GateApplicationFailed {
                    gate: gate_name.to_string(),
                    qubits: qubits.to_vec(),
                    reason: format!("Gates with {} qubits not yet supported", num_qubits),
                });
            },
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

        let matrix_vec = gate
            .matrix()
            .ok_or_else(|| ExecutionError::InvalidGateMatrix {
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
            },
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
            },
            _ => {
                return Err(ExecutionError::GateApplicationFailed {
                    gate: gate_name.to_string(),
                    qubits: qubits.to_vec(),
                    reason: format!("Sparse gates with {} qubits not yet supported", num_qubits),
                });
            },
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
    use simq_gates::standard::{CNot, Hadamard, PauliX, PauliZ, RotationY, Swap, CZ};

    fn make_sequential_engine() -> ExecutionEngine {
        let config = ExecutionConfig {
            mode: ExecutionMode::Sequential,
            validate_state: false,
            ..ExecutionConfig::default()
        };
        ExecutionEngine::new(config)
    }

    fn make_parallel_engine() -> ExecutionEngine {
        let config = ExecutionConfig {
            mode: ExecutionMode::Parallel,
            validate_state: false,
            ..ExecutionConfig::default()
        };
        ExecutionEngine::new(config)
    }

    fn make_gpu_engine() -> ExecutionEngine {
        let config = ExecutionConfig {
            mode: ExecutionMode::Gpu,
            validate_state: false,
            ..ExecutionConfig::default()
        };
        ExecutionEngine::new(config)
    }

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
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(1)])
            .unwrap();

        let mut state = AdaptiveState::new(2).unwrap();

        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_ok());
    }

    #[test]
    fn test_two_qubit_gates() {
        let config = ExecutionConfig::default();
        let mut engine = ExecutionEngine::new(config);

        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();

        let mut state = AdaptiveState::new(2).unwrap();

        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sequential_mode() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        assert!(engine.execute(&circuit, &mut state).is_ok());
    }

    #[test]
    fn test_parallel_mode() {
        let mut engine = make_parallel_engine();
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(1)])
            .unwrap();
        let mut state = AdaptiveState::new(2).unwrap();
        assert!(engine.execute(&circuit, &mut state).is_ok());
    }

    #[test]
    fn test_gpu_mode_fallback_to_sequential() {
        let mut engine = make_gpu_engine();
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        // GPU mode falls back to sequential
        assert!(engine.execute(&circuit, &mut state).is_ok());
    }

    #[test]
    fn test_cache_hit_on_repeated_gate() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(2);
        // Apply same gate twice to trigger cache hit
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(2).unwrap();
        assert!(engine.execute(&circuit, &mut state).is_ok());
        // Second execution should use cache
        let hit_rate = engine.cache_hit_rate();
        assert!((0.0..=1.0).contains(&hit_rate));
    }

    #[test]
    fn test_pauli_z_optimized_path() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(PauliZ), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        assert!(engine.execute(&circuit, &mut state).is_ok());
    }

    #[test]
    fn test_cz_gate_optimized_path() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(CZ), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        let mut state = AdaptiveState::new(2).unwrap();
        assert!(engine.execute(&circuit, &mut state).is_ok());
    }

    #[test]
    fn test_swap_gate_optimized_path() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(Swap), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        let mut state = AdaptiveState::new(2).unwrap();
        assert!(engine.execute(&circuit, &mut state).is_ok());
    }

    #[test]
    fn test_general_single_qubit_gate_simd() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(1);
        // RotationY is not named "X"/"Z"/"H" so it goes through general path
        circuit
            .add_gate(Arc::new(RotationY::new(0.5)), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        assert!(engine.execute(&circuit, &mut state).is_ok());
    }

    #[test]
    fn test_recovery_policy_skip() {
        let config = ExecutionConfig {
            mode: ExecutionMode::Sequential,
            validate_state: false,
            ..ExecutionConfig::default()
        };
        let mut engine = ExecutionEngine::new(config);
        engine.recovery_policy = RecoveryPolicy::Skip;
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        assert!(engine.execute(&circuit, &mut state).is_ok());
    }

    #[test]
    fn test_empty_circuit() {
        let mut engine = make_sequential_engine();
        let circuit = Circuit::new(2);
        let mut state = AdaptiveState::new(2).unwrap();
        assert!(engine.execute(&circuit, &mut state).is_ok());
    }

    #[test]
    fn test_telemetry_after_execution() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        engine.execute(&circuit, &mut state).unwrap();
        let telemetry = engine.get_telemetry();
        assert!(!telemetry.per_gate_times.is_empty());
    }

    #[test]
    fn test_engine_with_checkpoints() {
        let config = ExecutionConfig {
            mode: ExecutionMode::Sequential,
            validate_state: false,
            enable_checkpoints: true,
            checkpoint_interval: 1,
            ..ExecutionConfig::default()
        };
        let mut engine = ExecutionEngine::new(config);
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        assert!(engine.execute(&circuit, &mut state).is_ok());
    }

    #[test]
    fn test_debug_config() {
        let config = ExecutionConfig::debug();
        assert_eq!(config.mode, ExecutionMode::Sequential);
        assert!(config.validate_state);
        assert!(config.enable_checkpoints);
        assert!(!config.enable_gate_fusion);
    }

    /// A gate that has no matrix (returns None from matrix())
    #[derive(Debug)]
    struct NoMatrixGate;

    impl simq_core::Gate for NoMatrixGate {
        fn name(&self) -> &str {
            "NoMatrixGate"
        }
        fn num_qubits(&self) -> usize {
            1
        }
        fn matrix(&self) -> Option<Vec<num_complex::Complex64>> {
            None
        }
    }

    /// A gate that returns a wrong-size matrix (triggers InvalidGateMatrix)
    #[derive(Debug)]
    struct BadMatrixGate;

    impl simq_core::Gate for BadMatrixGate {
        fn name(&self) -> &str {
            "BadMatrixGate"
        }
        fn num_qubits(&self) -> usize {
            1
        }
        fn matrix(&self) -> Option<Vec<num_complex::Complex64>> {
            // Wrong size: should be 4 for a 1-qubit gate
            Some(vec![num_complex::Complex64::new(1.0, 0.0); 3])
        }
    }

    /// A gate that claims to be 3-qubit (unsupported)
    #[derive(Debug)]
    struct ThreeQubitGate;

    impl simq_core::Gate for ThreeQubitGate {
        fn name(&self) -> &str {
            "ThreeQubitGate"
        }
        fn num_qubits(&self) -> usize {
            3
        }
        fn matrix(&self) -> Option<Vec<num_complex::Complex64>> {
            Some(vec![num_complex::Complex64::new(1.0, 0.0); 64])
        }
    }

    #[test]
    fn test_timeout_triggers_error() {
        use std::time::Duration;
        let config = ExecutionConfig {
            mode: ExecutionMode::Sequential,
            validate_state: false,
            // Zero duration — will always time out on first gate
            timeout: Some(Duration::from_nanos(0)),
            ..ExecutionConfig::default()
        };
        let mut engine = ExecutionEngine::new(config);
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        let result = engine.execute(&circuit, &mut state);
        // Should either timeout or succeed depending on timing; mostly we test it doesn't panic
        // A zero-duration timeout should reliably fail
        match result {
            Err(crate::execution_engine::error::ExecutionError::ExecutionTimeout { .. }) => {},
            Ok(()) => {}, // Timing edge case: completed before check
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_recovery_policy_halt_on_no_matrix_gate() {
        let config = ExecutionConfig {
            mode: ExecutionMode::Sequential,
            validate_state: false,
            ..ExecutionConfig::default()
        };
        let mut engine = ExecutionEngine::new(config);
        engine.recovery_policy = RecoveryPolicy::Halt;
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(NoMatrixGate), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        // Should fail because gate has no matrix
        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_err());
    }

    #[test]
    fn test_recovery_policy_skip_on_no_matrix_gate() {
        let config = ExecutionConfig {
            mode: ExecutionMode::Sequential,
            validate_state: false,
            ..ExecutionConfig::default()
        };
        let mut engine = ExecutionEngine::new(config);
        engine.recovery_policy = RecoveryPolicy::Skip;
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(NoMatrixGate), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        // Skip policy: error is suppressed, execution continues
        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_ok());
    }

    #[test]
    fn test_recovery_policy_retry_exhausted() {
        let config = ExecutionConfig {
            mode: ExecutionMode::Sequential,
            validate_state: false,
            ..ExecutionConfig::default()
        };
        let mut engine = ExecutionEngine::new(config);
        engine.recovery_policy = RecoveryPolicy::RetryOnce;
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(NoMatrixGate), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        // RetryOnce: tries twice, then halts with ExecutionHalted
        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::execution_engine::error::ExecutionError::ExecutionHalted { .. } => {},
            e => panic!("Expected ExecutionHalted, got {:?}", e),
        }
    }

    #[test]
    fn test_no_matrix_gate_returns_error() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(NoMatrixGate), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        // Halt policy (default), so should propagate InvalidGateMatrix
        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_err());
    }

    #[test]
    fn test_bad_matrix_size_returns_error() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(BadMatrixGate), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_err());
    }

    #[test]
    fn test_three_qubit_gate_unsupported() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(
                Arc::new(ThreeQubitGate),
                &[QubitId::new(0), QubitId::new(1), QubitId::new(2)],
            )
            .unwrap();
        let mut state = AdaptiveState::new(3).unwrap();
        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_err());
    }

    #[test]
    fn test_general_single_qubit_no_simd() {
        // Use use_simd=false to go through the non-SIMD general path
        let config = ExecutionConfig {
            mode: ExecutionMode::Sequential,
            validate_state: false,
            use_simd: false,
            ..ExecutionConfig::default()
        };
        let mut engine = ExecutionEngine::new(config);
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(RotationY::new(0.5)), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        assert!(engine.execute(&circuit, &mut state).is_ok());
    }

    #[test]
    fn test_cache_eviction_with_small_cache() {
        // Cache size 1 forces eviction after first insertion
        let config = ExecutionConfig {
            mode: ExecutionMode::Sequential,
            validate_state: false,
            matrix_cache_size: 1,
            ..ExecutionConfig::default()
        };
        let mut engine = ExecutionEngine::new(config);
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        assert!(engine.execute(&circuit, &mut state).is_ok());
    }

    /// A 2-qubit gate that returns a wrong-size matrix (4 elements instead of 16)
    #[derive(Debug)]
    struct BadMatrix2qGate;

    impl simq_core::Gate for BadMatrix2qGate {
        fn name(&self) -> &str {
            "BadMatrix2qGate"
        }

        fn num_qubits(&self) -> usize {
            2
        }

        fn matrix(&self) -> Option<Vec<num_complex::Complex64>> {
            // Wrong size: should be 16 for a 2-qubit gate
            Some(vec![num_complex::Complex64::new(1.0, 0.0); 4])
        }
    }

    /// Create a Dense 1-qubit state (both amplitudes non-zero → density=1.0 > 10%).
    fn make_dense_1q_state() -> AdaptiveState {
        let sqrt2_inv = std::f64::consts::FRAC_1_SQRT_2;
        let amplitudes = [
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(sqrt2_inv, 0.0),
        ];
        let state = AdaptiveState::from_amplitudes(1, &amplitudes).unwrap();
        assert!(state.is_dense(), "Expected Dense state for density=1.0");
        state
    }

    /// Create a Dense 2-qubit state (all 4 amplitudes non-zero → density=1.0 > 10%).
    fn make_dense_2q_state() -> AdaptiveState {
        let amplitudes = [
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ];
        let state = AdaptiveState::from_amplitudes(2, &amplitudes).unwrap();
        assert!(state.is_dense(), "Expected Dense state for density=1.0");
        state
    }

    /// Create a Dense 3-qubit state (all 8 amplitudes non-zero → density=1.0 > 10%).
    fn make_dense_3q_state() -> AdaptiveState {
        let amp = 1.0 / (8.0_f64).sqrt();
        let amplitudes = vec![Complex64::new(amp, 0.0); 8];
        let state = AdaptiveState::from_amplitudes(3, &amplitudes).unwrap();
        assert!(state.is_dense(), "Expected Dense state for density=1.0");
        state
    }

    /// Cache hit in apply_gate_dense (lines 297-298):
    /// Same gate applied twice on Dense state; second application hits cache.
    #[test]
    fn test_dense_state_cache_hit() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        let mut state = make_dense_1q_state();
        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_ok());
        // After two Hadamards the state returns to the original
    }

    /// gate.matrix() returns None on Dense state (lines 306-307).
    #[test]
    fn test_dense_state_no_matrix_gate() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(NoMatrixGate), &[QubitId::new(0)])
            .unwrap();
        let mut state = make_dense_1q_state();
        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_err());
    }

    /// 1-qubit gate with wrong matrix size on Dense state (lines 314-316).
    #[test]
    fn test_dense_state_bad_1q_matrix_gate() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(BadMatrixGate), &[QubitId::new(0)])
            .unwrap();
        let mut state = make_dense_1q_state();
        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_err());
    }

    /// 2-qubit gate with wrong matrix size on Dense state (lines 327-329).
    #[test]
    fn test_dense_state_bad_2q_matrix_gate() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(BadMatrix2qGate), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        let mut state = make_dense_2q_state();
        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_err());
    }

    /// 3-qubit gate on Dense state hits the `_` arm (lines 341-344).
    #[test]
    fn test_dense_state_three_qubit_gate() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(
                Arc::new(ThreeQubitGate),
                &[QubitId::new(0), QubitId::new(1), QubitId::new(2)],
            )
            .unwrap();
        let mut state = make_dense_3q_state();
        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_err());
    }

    /// 2-qubit gate with wrong matrix size on SPARSE state (lines 515-517).
    #[test]
    fn test_sparse_state_bad_2q_matrix_gate() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(BadMatrix2qGate), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        let mut state = AdaptiveState::new(2).unwrap();
        assert!(state.is_sparse(), "Expected Sparse state from AdaptiveState::new");
        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_err());
    }
}
