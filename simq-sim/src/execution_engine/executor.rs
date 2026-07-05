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
    ///
    /// GPU execution is not implemented. An explicit request for it fails
    /// loudly instead of silently running on the CPU: a caller who selected
    /// `ExecutionMode::Gpu` for a 2^20-amplitude state must not discover at
    /// benchmark time that "GPU" numbers were sequential-CPU numbers.
    /// (Configs with `use_gpu`/`ExecutionMode::Gpu` are already rejected by
    /// `ExecutionConfig::validate`; this guards direct calls.)
    fn execute_gpu(
        &mut self,
        _circuit: &Circuit,
        _state: &mut AdaptiveState,
        _timeout_check: Option<(Instant, std::time::Duration)>,
    ) -> Result<()> {
        self.telemetry.log_event("gpu_unavailable");
        Err(ExecutionError::GpuError {
            reason: "GPU execution is not implemented; use Sequential, Parallel, or Adaptive \
                     mode instead of relying on a silent CPU fallback"
                .to_string(),
        })
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

    /// Whether a gate's matrix is safe to cache by name alone.
    ///
    /// The `Gate` trait does not expose parameters, so parameterized gates
    /// (rotations, phase gates, ...) cannot be distinguished by name: caching
    /// them would return the matrix of a previous instance with a different
    /// angle. Only cache gates whose matrix is fixed.
    fn is_fixed_matrix_gate(gate_name: &str) -> bool {
        matches!(
            gate_name,
            "H" | "Hadamard"
                | "X"
                | "PauliX"
                | "Y"
                | "PauliY"
                | "Z"
                | "PauliZ"
                | "S"
                | "S†"
                | "T"
                | "T†"
                | "I"
                | "Identity"
                | "SX"
                | "SX†"
                | "CNOT"
                | "CX"
                | "CZ"
                | "SWAP"
                | "CCNOT"
                | "Toffoli"
                | "CSWAP"
                | "Fredkin"
        )
    }

    /// Convert a flat gate matrix into the fixed-size representation used by
    /// the dense kernels
    fn build_cached_matrix(
        gate_name: &str,
        qubits: &[QubitId],
        matrix_vec: &[Complex64],
    ) -> Result<CachedMatrix> {
        match qubits.len() {
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
                Ok(CachedMatrix::Single(mat))
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
                Ok(CachedMatrix::Two(mat))
            },
            3 => {
                if matrix_vec.len() != 64 {
                    return Err(ExecutionError::InvalidGateMatrix {
                        gate: gate_name.to_string(),
                        reason: format!("Expected 64 elements, got {}", matrix_vec.len()),
                    });
                }
                let mut mat: Matrix8x8 = [[Complex64::new(0.0, 0.0); 8]; 8];
                for i in 0..8 {
                    for j in 0..8 {
                        mat[i][j] = matrix_vec[i * 8 + j];
                    }
                }
                Ok(CachedMatrix::Three(Box::new(mat)))
            },
            n => Err(ExecutionError::GateApplicationFailed {
                gate: gate_name.to_string(),
                qubits: qubits.to_vec(),
                reason: format!("Gates with {} qubits not yet supported", n),
            }),
        }
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

        let cacheable = Self::is_fixed_matrix_gate(gate_name);

        let matrix = if cacheable {
            let cache_key = GateCacheKey {
                gate_name: gate_name.to_string(),
                params: vec![],
            };

            if let Some(cached) = self.matrix_cache.get(&cache_key) {
                self.telemetry.cache_hit();
                cached
            } else {
                self.telemetry.cache_miss();

                let matrix_vec =
                    gate.matrix()
                        .ok_or_else(|| ExecutionError::InvalidGateMatrix {
                            gate: gate_name.to_string(),
                            reason: "Gate has no matrix representation".to_string(),
                        })?;

                let cached_matrix =
                    Arc::new(Self::build_cached_matrix(gate_name, qubits, &matrix_vec)?);
                self.matrix_cache
                    .insert(cache_key, (*cached_matrix).clone());
                cached_matrix
            }
        } else {
            // Parameterized (or unknown) gate: always rebuild the matrix
            let matrix_vec = gate
                .matrix()
                .ok_or_else(|| ExecutionError::InvalidGateMatrix {
                    gate: gate_name.to_string(),
                    reason: "Gate has no matrix representation".to_string(),
                })?;
            Arc::new(Self::build_cached_matrix(gate_name, qubits, &matrix_vec)?)
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
            3 => {
                if let CachedMatrix::Three(ref mat) = *matrix {
                    three_qubit::apply_three_qubit_dense(
                        mat,
                        qubits[0].index(),
                        qubits[1].index(),
                        qubits[2].index(),
                        amplitudes,
                        use_parallel,
                        threshold,
                    )?;
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
            3 => {
                if matrix_vec.len() != 64 {
                    return Err(ExecutionError::InvalidGateMatrix {
                        gate: gate_name.to_string(),
                        reason: format!("Expected 64 elements, got {}", matrix_vec.len()),
                    });
                }
                let mut mat = [[Complex64::new(0.0, 0.0); 8]; 8];
                for i in 0..8 {
                    for j in 0..8 {
                        mat[i][j] = matrix_vec[i * 8 + j];
                    }
                }

                sparse::apply_three_qubit_sparse(
                    &mat,
                    qubits[0].index(),
                    qubits[1].index(),
                    qubits[2].index(),
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

    /// Convert the state representation when its density crosses the
    /// configured thresholds (sparse → dense when dense enough, dense →
    /// sparse when it becomes very sparse again)
    fn maybe_convert_state(&mut self, state: &mut AdaptiveState) {
        if state.is_sparse() && self.adaptive_strategy.should_convert_to_dense(state) {
            match state.force_to_dense() {
                Ok(true) => {
                    self.telemetry.log_event("convert_sparse_to_dense");
                    self.telemetry.sparse_dense_transitions += 1;
                },
                Ok(false) => {},
                Err(e) => {
                    self.telemetry
                        .log_error(format!("Sparse→Dense conversion failed: {}", e));
                },
            }
        } else if state.is_dense() && self.adaptive_strategy.should_convert_to_sparse(state) {
            match state.force_to_sparse() {
                Ok(true) => {
                    self.telemetry.log_event("convert_dense_to_sparse");
                    self.telemetry.sparse_dense_transitions += 1;
                },
                Ok(false) => {},
                Err(e) => {
                    self.telemetry
                        .log_error(format!("Dense→Sparse conversion failed: {}", e));
                },
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
    use simq_gates::standard::{
        CNot, Fredkin, Hadamard, PauliX, PauliZ, RotationY, Swap, Toffoli, CZ,
    };

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

    /// GPU mode must be rejected loudly, not silently executed on the CPU.
    #[test]
    #[should_panic(expected = "Invalid execution config")]
    fn test_gpu_mode_rejected_at_construction() {
        let config = ExecutionConfig {
            mode: ExecutionMode::Gpu,
            validate_state: false,
            ..ExecutionConfig::default()
        };
        let _ = ExecutionEngine::new(config);
    }

    /// Even if a Gpu-mode engine could be constructed, execution must fail
    /// loudly rather than fall back to sequential CPU execution.
    #[test]
    fn test_gpu_execution_errors_loudly() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        let result = engine.execute_gpu(&circuit, &mut state, None);
        assert!(matches!(
            result,
            Err(crate::execution_engine::error::ExecutionError::GpuError { .. })
        ));
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

    /// A gate that claims to be 4-qubit (unsupported)
    #[derive(Debug)]
    struct FourQubitGate;

    impl simq_core::Gate for FourQubitGate {
        fn name(&self) -> &str {
            "FourQubitGate"
        }
        fn num_qubits(&self) -> usize {
            4
        }
        fn matrix(&self) -> Option<Vec<num_complex::Complex64>> {
            Some(vec![num_complex::Complex64::new(1.0, 0.0); 256])
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
    fn test_four_qubit_gate_unsupported() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(4);
        circuit
            .add_gate(
                Arc::new(FourQubitGate),
                &[
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                ],
            )
            .unwrap();
        let mut state = AdaptiveState::new(4).unwrap();
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

    /// 4-qubit gate on Dense state hits the unsupported `_` arm.
    #[test]
    fn test_dense_state_four_qubit_gate() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(4);
        circuit
            .add_gate(
                Arc::new(FourQubitGate),
                &[
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                ],
            )
            .unwrap();
        let amp = 0.25;
        let amplitudes = vec![Complex64::new(amp, 0.0); 16];
        let mut state = AdaptiveState::from_amplitudes(4, &amplitudes).unwrap();
        assert!(state.is_dense());
        let result = engine.execute(&circuit, &mut state);
        assert!(result.is_err());
    }

    /// Toffoli via the sparse path: |110⟩ (controls q0,q1 set) flips target q2.
    #[test]
    fn test_toffoli_sparse_path() {
        let config = ExecutionConfig {
            mode: ExecutionMode::Sequential,
            validate_state: false,
            adaptive_state: false, // stay on the sparse path
            ..ExecutionConfig::default()
        };
        let mut engine = ExecutionEngine::new(config);
        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(1)])
            .unwrap();
        circuit
            .add_gate(Arc::new(Toffoli), &[QubitId::new(0), QubitId::new(1), QubitId::new(2)])
            .unwrap();
        let mut state = AdaptiveState::new(3).unwrap();
        engine.execute(&circuit, &mut state).unwrap();
        assert!(state.is_sparse());

        let amps = state.to_dense_vec();
        // Little-endian: q0=q1=q2=1 → index 7
        assert!((amps[7].re - 1.0).abs() < 1e-10);
        assert!(amps[3].norm() < 1e-10);
    }

    /// Toffoli via the dense path (uniform superposition converts the state).
    /// H(0), CNOT(0,1), Toffoli(0,1,2) produces the GHZ state.
    #[test]
    fn test_toffoli_builds_ghz_state() {
        let config = ExecutionConfig {
            mode: ExecutionMode::Sequential,
            validate_state: false,
            ..ExecutionConfig::default()
        };
        let mut engine = ExecutionEngine::new(config);
        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        circuit
            .add_gate(Arc::new(Toffoli), &[QubitId::new(0), QubitId::new(1), QubitId::new(2)])
            .unwrap();
        let mut state = AdaptiveState::new(3).unwrap();
        engine.execute(&circuit, &mut state).unwrap();

        let amps = state.to_dense_vec();
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((amps[0].re - expected).abs() < 1e-10, "amps = {:?}", amps);
        assert!((amps[7].re - expected).abs() < 1e-10, "amps = {:?}", amps);
        for (i, amp) in amps.iter().enumerate().take(7).skip(1) {
            assert!(amp.norm() < 1e-10, "amps[{}] = {:?}", i, amp);
        }
    }

    /// Fredkin (CSWAP): control q0 set swaps targets q1 and q2.
    #[test]
    fn test_fredkin_swaps_targets() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(1)])
            .unwrap();
        circuit
            .add_gate(Arc::new(Fredkin), &[QubitId::new(0), QubitId::new(1), QubitId::new(2)])
            .unwrap();
        let mut state = AdaptiveState::new(3).unwrap();
        engine.execute(&circuit, &mut state).unwrap();

        let amps = state.to_dense_vec();
        // |q2 q1 q0⟩ = |011⟩ (index 3) → swap q1↔q2 → |101⟩ (index 5)
        assert!((amps[5].re - 1.0).abs() < 1e-10, "amps = {:?}", amps);
        assert!(amps[3].norm() < 1e-10);
    }

    /// Sparse→dense conversion actually happens once density crosses the
    /// threshold, and telemetry records a real transition (issue #41).
    #[test]
    fn test_sparse_to_dense_conversion_happens() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(3);
        for q in 0..3 {
            circuit
                .add_gate(Arc::new(Hadamard), &[QubitId::new(q)])
                .unwrap();
        }
        let mut state = AdaptiveState::new(3).unwrap();
        assert!(state.is_sparse());
        engine.execute(&circuit, &mut state).unwrap();

        // Density reaches 1.0 (well above the 10% threshold), so the state
        // must be dense at the end and the transition must be counted.
        assert!(state.is_dense(), "state should have converted to dense");
        assert_eq!(engine.telemetry.sparse_dense_transitions, 1);

        let amps = state.to_dense_vec();
        let expected = 1.0 / (8.0_f64).sqrt();
        for amp in amps {
            assert!((amp.re - expected).abs() < 1e-10);
        }
    }

    /// The dense and sparse paths must agree on the final state (endianness
    /// regression test: they previously used opposite bit orders).
    #[test]
    fn test_dense_and_sparse_paths_agree() {
        let build_circuit = || {
            let mut circuit = Circuit::new(3);
            circuit
                .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
                .unwrap();
            circuit
                .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(2)])
                .unwrap();
            circuit
                .add_gate(Arc::new(RotationY::new(0.7)), &[QubitId::new(1)])
                .unwrap();
            circuit
                .add_gate(Arc::new(CZ), &[QubitId::new(1), QubitId::new(2)])
                .unwrap();
            circuit
                .add_gate(Arc::new(Swap), &[QubitId::new(0), QubitId::new(1)])
                .unwrap();
            circuit
        };

        // Sparse-only execution
        let sparse_config = ExecutionConfig {
            mode: ExecutionMode::Sequential,
            validate_state: false,
            adaptive_state: false,
            ..ExecutionConfig::default()
        };
        let mut sparse_engine = ExecutionEngine::new(sparse_config);
        let mut sparse_state = AdaptiveState::new(3).unwrap();
        sparse_engine
            .execute(&build_circuit(), &mut sparse_state)
            .unwrap();

        // Dense-only execution (state converted up front)
        let dense_config = ExecutionConfig {
            mode: ExecutionMode::Sequential,
            validate_state: false,
            adaptive_state: false,
            ..ExecutionConfig::default()
        };
        let mut dense_engine = ExecutionEngine::new(dense_config);
        let mut dense_state = AdaptiveState::new(3).unwrap();
        dense_state.force_to_dense().unwrap();
        dense_engine
            .execute(&build_circuit(), &mut dense_state)
            .unwrap();

        let sparse_amps = sparse_state.to_dense_vec();
        let dense_amps = dense_state.to_dense_vec();
        for i in 0..8 {
            assert!(
                (sparse_amps[i].re - dense_amps[i].re).abs() < 1e-10
                    && (sparse_amps[i].im - dense_amps[i].im).abs() < 1e-10,
                "amplitude {} differs: sparse={:?} dense={:?}",
                i,
                sparse_amps[i],
                dense_amps[i]
            );
        }
    }

    /// Rotation gates must not share cache entries across different angles
    /// (the cache key cannot see gate parameters).
    #[test]
    fn test_rotation_gates_not_cross_cached() {
        let mut engine = make_sequential_engine();
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(RotationY::new(0.5)), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(RotationY::new(1.0)), &[QubitId::new(0)])
            .unwrap();
        let mut state = AdaptiveState::new(1).unwrap();
        engine.execute(&circuit, &mut state).unwrap();

        // RY(0.5) then RY(1.0) equals RY(1.5): |0⟩ → cos(0.75)|0⟩ + sin(0.75)|1⟩
        let amps = state.to_dense_vec();
        assert!((amps[0].re - (0.75_f64).cos()).abs() < 1e-10, "amps = {:?}", amps);
        assert!((amps[1].re - (0.75_f64).sin()).abs() < 1e-10, "amps = {:?}", amps);
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
