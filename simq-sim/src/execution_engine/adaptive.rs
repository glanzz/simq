//! Adaptive execution strategies

use crate::execution_engine::config::ExecutionMode;
use simq_state::AdaptiveState;

/// Adaptive strategy selector
pub struct AdaptiveStrategy {
    sparse_threshold: f32,
    gpu_threshold: usize,
    parallel_threshold: usize,
}

impl AdaptiveStrategy {
    pub fn new(sparse_threshold: f32, gpu_threshold: usize, parallel_threshold: usize) -> Self {
        Self {
            sparse_threshold,
            gpu_threshold,
            parallel_threshold,
        }
    }

    /// Decide whether to convert sparse to dense
    pub fn should_convert_to_dense(&self, state: &AdaptiveState) -> bool {
        state.density() > self.sparse_threshold
    }

    /// Decide whether to convert dense to sparse
    pub fn should_convert_to_sparse(&self, state: &AdaptiveState) -> bool {
        state.density() < (self.sparse_threshold / 2.0)
    }

    /// Decide which execution mode to use
    pub fn select_execution_mode(&self, state: &AdaptiveState, num_gates: usize) -> ExecutionMode {
        let state_size = match state {
            AdaptiveState::Dense(dense) => dense.dimension(),
            AdaptiveState::Sparse { state, .. } => state.amplitudes().len(),
        };

        if state_size >= self.gpu_threshold {
            ExecutionMode::Gpu
        } else if state_size >= self.parallel_threshold || num_gates > 100 {
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
}
