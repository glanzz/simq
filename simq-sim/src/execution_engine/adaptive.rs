//! Adaptive execution strategies

use crate::execution_engine::config::ExecutionMode;
use simq_state::AdaptiveState;

/// Adaptive strategy selector
pub struct AdaptiveStrategy {
    sparse_threshold: f32,
    /// Retained for when a real GPU backend lands; `select_execution_mode`
    /// deliberately ignores it until then.
    #[allow(dead_code)]
    gpu_threshold: usize,
    parallel_threshold: usize,
}

/// Density at which hash-map gate application costs as much as a dense pass.
///
/// A sparse gate costs O(entries) with a large per-entry constant (hash
/// lookups, rebuild allocations — measured ~300× the per-amplitude cost of
/// the dense kernels), so the cost crossover sits near 1/300 density, not at
/// the 10% default threshold — and the per-entry constant grows further once
/// the map outgrows cache (DRAM-latency-bound probing at ≥0.5M entries).
/// Waiting for 10% let the map grow to millions of entries on the way to a
/// dense state: at 24 qubits the sparse warm-up alone was ~55% of total
/// runtime, and at 26 qubits a 1/128 threshold still left ~2.5 s of hashing
/// that 1/1024 cuts to a handful of extra (cheap, sequential) dense passes.
/// The dense→sparse direction uses half the threshold as hysteresis.
const COST_CROSSOVER_DENSITY: f32 = 1.0 / 1024.0;

impl AdaptiveStrategy {
    pub fn new(sparse_threshold: f32, gpu_threshold: usize, parallel_threshold: usize) -> Self {
        Self {
            sparse_threshold,
            gpu_threshold,
            parallel_threshold,
        }
    }

    /// The density at which representation conversion actually happens:
    /// the configured threshold, capped at the measured cost crossover.
    fn effective_threshold(&self) -> f32 {
        self.sparse_threshold.min(COST_CROSSOVER_DENSITY)
    }

    /// Decide whether to convert sparse to dense
    pub fn should_convert_to_dense(&self, state: &AdaptiveState) -> bool {
        state.density() > self.effective_threshold()
    }

    /// Decide whether to convert dense to sparse
    pub fn should_convert_to_sparse(&self, state: &AdaptiveState) -> bool {
        state.density() < (self.effective_threshold() / 2.0)
    }

    /// Decide which execution mode to use
    ///
    /// Never selects [`ExecutionMode::Gpu`]: the GPU backend is not
    /// implemented, and routing large states there would either fail every
    /// big-circuit run or (previously) silently execute sequentially. The
    /// `gpu_threshold` field is kept so the policy can be re-enabled once a
    /// real backend exists.
    pub fn select_execution_mode(&self, state: &AdaptiveState, num_gates: usize) -> ExecutionMode {
        let state_size = match state {
            AdaptiveState::Dense(dense) => dense.dimension(),
            AdaptiveState::Sparse { state, .. } => state.amplitudes().len(),
        };

        if state_size >= self.parallel_threshold || num_gates > 100 {
            ExecutionMode::Parallel
        } else {
            ExecutionMode::Sequential
        }
    }

    /// Decide whether to use parallel execution for a gate
    pub fn should_parallelize_gate(&self, state: &AdaptiveState) -> bool {
        match state {
            AdaptiveState::Dense(dense) => dense.dimension() >= self.parallel_threshold,
            AdaptiveState::Sparse { .. } => false, // Sparse doesn't benefit as much from parallelism
        }
    }
}

impl Default for AdaptiveStrategy {
    fn default() -> Self {
        Self::new(0.1, 1 << 20, 1 << 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_strategy() {
        let strategy = AdaptiveStrategy::default();
        assert_eq!(strategy.sparse_threshold, 0.1);
    }

    #[test]
    fn test_never_selects_unimplemented_gpu_mode() {
        // Even when the state size exceeds the GPU threshold, the strategy
        // must not route execution to the unimplemented GPU backend.
        let strategy = AdaptiveStrategy::new(0.1, 1, 2);
        let state = AdaptiveState::new(4).unwrap();
        let mode = strategy.select_execution_mode(&state, 1);
        assert_ne!(mode, ExecutionMode::Gpu);
    }

    #[test]
    fn test_select_parallel_mode_when_num_gates_exceeds_100() {
        // Thresholds set high so state size alone doesn't trigger Parallel/Gpu
        let strategy = AdaptiveStrategy::new(0.1, 1 << 30, 1 << 30);
        let state = AdaptiveState::new(1).unwrap(); // 2 amplitudes, well below thresholds
        let mode = strategy.select_execution_mode(&state, 101);
        assert_eq!(mode, ExecutionMode::Parallel);
    }

    #[test]
    fn test_select_sequential_mode_for_small_state_and_few_gates() {
        let strategy = AdaptiveStrategy::new(0.1, 1 << 30, 1 << 30);
        let state = AdaptiveState::new(1).unwrap();
        let mode = strategy.select_execution_mode(&state, 10);
        assert_eq!(mode, ExecutionMode::Sequential);
    }

    #[test]
    fn test_should_convert_to_dense_and_sparse() {
        // AdaptiveState::new creates Sparse |00⟩ with 1 non-zero amplitude out of 4 → density≈0.0625.
        // Use a threshold of 0.05 so that density 0.0625 > 0.05 → should_convert_to_dense = true.
        let strategy = AdaptiveStrategy::new(0.05, 1 << 20, 1 << 10);
        let state = AdaptiveState::new(2).unwrap();
        assert!(strategy.should_convert_to_dense(&state));
        // density 0.0625 is not < 0.025 → should NOT convert to sparse
        assert!(!strategy.should_convert_to_sparse(&state));
    }

    #[test]
    fn test_should_parallelize_gate_sparse_is_false() {
        let strategy = AdaptiveStrategy::default();
        // Build a sparse state
        let sparse_state = simq_state::AdaptiveState::Sparse {
            state: simq_state::SparseState::new(2).unwrap(),
            threshold: 0.1,
        };
        assert!(!strategy.should_parallelize_gate(&sparse_state));
    }
}
