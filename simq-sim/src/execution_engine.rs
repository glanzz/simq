/// Policy for error recovery during execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryPolicy {
    /// Stop execution on first error
    Halt,
    /// Skip failed gate and continue
    Skip,
    /// Attempt to retry failed gate once
    RetryOnce,
}
use std::time::{Duration, Instant};
use std::collections::HashMap;
use thread_id;
/// Telemetry data for execution profiling
#[derive(Debug, Default)]
pub struct ExecutionTelemetry {
    pub total_gate_time: Duration,
    pub per_gate_times: Vec<Duration>,
    pub state_density: Vec<f32>,
    pub memory_usage: Vec<usize>,
    pub gate_type_counts: HashMap<String, usize>,
    pub thread_ids: Vec<u64>,
    pub parallelism: usize,
    pub error_events: Vec<String>,
    pub custom_events: Vec<(String, Instant)>,
}

impl ExecutionTelemetry {
    pub fn log_error(&mut self, msg: impl Into<String>) {
        self.error_events.push(msg.into());
    }
    pub fn log_event(&mut self, label: impl Into<String>) {
        self.custom_events.push((label.into(), Instant::now()));
    }
    pub fn inc_gate_type(&mut self, gate_name: &str) {
        *self.gate_type_counts.entry(gate_name.to_string()).or_insert(0) += 1;
    }
    pub fn record_memory(&mut self, bytes: usize) {
        self.memory_usage.push(bytes);
    }
    pub fn record_thread(&mut self) {
        self.thread_ids.push(thread_id::get() as u64);
    }
}
/// Execution engine for SimQ simulator

use simq_core::{Circuit, QubitId, Gate, GateOp};
use std::sync::Arc;
use simq_state::{AdaptiveState, SparseState, DenseState};
// use rayon::prelude::*; // Only needed in kernel

/// Configuration for execution engine
pub struct ExecutionConfig {
    pub use_parallel: bool,
    pub use_simd: bool,
    pub parallel_threshold: usize,
}

/// Execution engine for quantum circuits
pub struct ExecutionEngine {
    config: ExecutionConfig,
    pub telemetry: ExecutionTelemetry,
    pub recovery_policy: RecoveryPolicy,
}

impl ExecutionEngine {
    pub fn new(config: ExecutionConfig) -> Self {
        Self {
            config,
            telemetry: ExecutionTelemetry::default(),
            recovery_policy: RecoveryPolicy::Halt,
        }
    }

    /// Execute a compiled circuit on a quantum state
    pub fn execute(&mut self, circuit: &Circuit, state: &mut AdaptiveState) {
        let mut total_gate_time = Duration::ZERO;
        self.telemetry.per_gate_times.clear();
        self.telemetry.state_density.clear();
        self.telemetry.memory_usage.clear();
        self.telemetry.gate_type_counts.clear();
        self.telemetry.thread_ids.clear();
        self.telemetry.error_events.clear();
        self.telemetry.custom_events.clear();
        self.telemetry.parallelism = rayon::current_num_threads();

        for gate_op in circuit.operations() {
            let start = Instant::now();
            let gate_name = gate_op.gate().name();
            self.telemetry.inc_gate_type(gate_name);
            self.telemetry.log_event(format!("apply_gate:{}", gate_name));
            self.telemetry.record_thread();
            // Estimate memory usage (dense: 2^n * 16, sparse: hashmap size)
            let mem_bytes = match state {
                AdaptiveState::Dense(ref dense) => {
                    dense.dimension() * std::mem::size_of::<num_complex::Complex64>()
                }
                AdaptiveState::Sparse { state: ref sparse, .. } => {
                    sparse.amplitudes().len() * std::mem::size_of::<num_complex::Complex64>()
                }
            };
            self.telemetry.record_memory(mem_bytes);

            let mut gate_attempts = 0;
            let mut gate_success = false;
            loop {
                gate_attempts += 1;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.apply_gate_op(gate_op, state)
                }));
                match result {
                    Ok(_) => {
                        gate_success = true;
                        break;
                    }
                    Err(_) => {
                        self.telemetry.log_error(format!("Gate '{}' panicked (attempt {})", gate_name, gate_attempts));
                        match self.recovery_policy {
                            RecoveryPolicy::Halt => break,
                            RecoveryPolicy::Skip => break,
                            RecoveryPolicy::RetryOnce if gate_attempts < 2 => continue,
                            _ => break,
                        }
                    }
                }
            }
            let elapsed = start.elapsed();
            total_gate_time += elapsed;
            self.telemetry.per_gate_times.push(elapsed);
            self.telemetry.state_density.push(state.density());
            if !gate_success && self.recovery_policy == RecoveryPolicy::Halt {
                self.telemetry.log_event("execution_halted_on_error");
                break;
            }
        }
        self.telemetry.total_gate_time = total_gate_time;
        self.telemetry.log_event("execution_complete");
    }

    /// Apply a single gate operation to the quantum state
    fn apply_gate_op(&self, gate_op: &GateOp, state: &mut AdaptiveState) {
        match state {
            AdaptiveState::Dense(ref mut dense) => {
                // Use SIMD if enabled and single-qubit gate
                if self.config.use_simd && gate_op.gate().num_qubits() == 1 {
                    if let Some(matrix) = gate_op.gate().matrix() {
                        // matrix: Vec<Complex<f64>> expected to be 4 elements for 2x2
                        if matrix.len() == 4 {
                            let mat: [[Complex64; 2]; 2] = [
                                [matrix[0], matrix[1]],
                                [matrix[2], matrix[3]],
                            ];
                            let qubit_idx = gate_op.qubits()[0].index();
                            let threshold = self.config.parallel_threshold;
                            apply_single_qubit_dense_simd(&mat, qubit_idx, dense.amplitudes_mut(), threshold);
                        } else {
                            self.apply_gate_dense(gate_op.gate(), gate_op.qubits(), dense);
                        }
                    } else {
                        self.apply_gate_dense(gate_op.gate(), gate_op.qubits(), dense);
                    }
                } else {
                    self.apply_gate_dense(gate_op.gate(), gate_op.qubits(), dense);
                }
            }
            AdaptiveState::Sparse { state: ref mut sparse, .. } => {
                self.apply_gate_sparse(gate_op.gate(), gate_op.qubits(), sparse);
            }
        }
    }

    /// Apply gate to dense state (SIMD + parallelism)
    fn apply_gate_dense(&self, gate: &Arc<dyn Gate>, qubits: &[QubitId], state: &mut DenseState) {
        // Example: single-qubit gate
        if qubits.len() == 1 {
            let qubit = qubits[0].index();
            let _matrix = gate.matrix();
            let stride = 1 << qubit;
            let amplitudes = state.amplitudes_mut();
            if self.config.use_parallel {
                amplitudes.par_chunks_mut(stride * 2).for_each(|_chunk| {
                    // Apply 2x2 matrix to pairs of amplitudes
                    // TODO: SIMD optimization
                });
            } else {
                amplitudes.chunks_mut(stride * 2).for_each(|_chunk| {
                    // TODO: SIMD optimization
                });
            }
        }
        // TODO: two-qubit, controlled, diagonal gates
    }

    /// Apply gate to sparse state
    fn apply_gate_sparse(&self, _gate: &Arc<dyn Gate>, _qubits: &[QubitId], _state: &mut SparseState) {
        // For each non-zero amplitude, compute new amplitudes
        // TODO: Efficient sparse update
    }
}

/// SIMD support
use num_complex::Complex64;
use rayon::prelude::*;
/// SIMD-optimized single-qubit gate application for dense state with automatic parallelism selection
pub fn apply_single_qubit_dense_simd(
    gate: &[[Complex64; 2]; 2],
    qubit: usize,
    state: &mut [Complex64],
    parallel_threshold: usize,
) {
    let stride = 1 << qubit;
    let n = state.len();
    // If state size exceeds threshold, use parallel execution
    if n >= parallel_threshold {
        state.par_chunks_mut(stride * 2).for_each(|chunk| {
            for j in 0..stride {
                let idx0 = j;
                let idx1 = j + stride;
                let a = chunk[idx0];
                let b = chunk[idx1];
                let out0 = gate[0][0] * a + gate[0][1] * b;
                let out1 = gate[1][0] * a + gate[1][1] * b;
                chunk[idx0] = out0;
                chunk[idx1] = out1;
            }
        });
    } else {
        let mut i = 0;
        while i < n {
            for j in 0..stride {
                let idx0 = i + j;
                let idx1 = idx0 + stride;
                let a = state[idx0];
                let b = state[idx1];
                let out0 = gate[0][0] * a + gate[0][1] * b;
                let out1 = gate[1][0] * a + gate[1][1] * b;
                state[idx0] = out0;
                state[idx1] = out1;
            }
            i += stride * 2;
        }
    }
}
