//! Parallel execution strategies

use crate::execution_engine::config::ParallelStrategy;
use crate::execution_engine::error::{ExecutionError, Result};
use simq_compiler::execution_plan::ExecutionPlanner;
use simq_core::{Circuit, GateOp};
use simq_state::AdaptiveState;

/// Parallel executor for quantum circuits
pub struct ParallelExecutor {
    strategy: ParallelStrategy,
    planner: ExecutionPlanner,
}

impl ParallelExecutor {
    pub fn new(strategy: ParallelStrategy) -> Self {
        Self {
            strategy,
            planner: ExecutionPlanner::new(),
        }
    }

    /// Execute circuit with parallel gate execution
    pub fn execute<F>(
        &self,
        circuit: &Circuit,
        state: &mut AdaptiveState,
        apply_gate: F,
    ) -> Result<()>
    where
        F: FnMut(&GateOp, &mut AdaptiveState) -> Result<()> + Send + Sync,
    {
        match self.strategy {
            ParallelStrategy::LayerBased => self.execute_layer_based(circuit, state, apply_gate),
            ParallelStrategy::DataParallel => {
                self.execute_data_parallel(circuit, state, apply_gate)
            },
            ParallelStrategy::Hybrid => self.execute_hybrid(circuit, state, apply_gate),
            ParallelStrategy::TaskBased => self.execute_task_based(circuit, state, apply_gate),
        }
    }

    /// Execute gates in parallel layers
    fn execute_layer_based<F>(
        &self,
        circuit: &Circuit,
        state: &mut AdaptiveState,
        mut apply_gate: F,
    ) -> Result<()>
    where
        F: FnMut(&GateOp, &mut AdaptiveState) -> Result<()>,
    {
        let plan = self.planner.generate_optimized_plan(circuit);
        let operations: Vec<_> = circuit.operations().collect();

        for (layer_idx, layer) in plan.layers.iter().enumerate() {
            // For now, execute sequentially within layer
            // TODO: Implement true parallel execution with state partitioning
            for &gate_idx in &layer.gates {
                if let Some(op) = operations.get(gate_idx) {
                    apply_gate(op, state).map_err(|e| ExecutionError::ParallelExecutionFailed {
                        layer: layer_idx,
                        reason: format!("{}", e),
                    })?;
                }
            }
        }

        Ok(())
    }

    /// Execute with data parallelism (parallelize within gate application)
    fn execute_data_parallel<F>(
        &self,
        circuit: &Circuit,
        state: &mut AdaptiveState,
        mut apply_gate: F,
    ) -> Result<()>
    where
        F: FnMut(&GateOp, &mut AdaptiveState) -> Result<()>,
    {
        // Sequential gate execution, parallelism happens inside gate application
        for op in circuit.operations() {
            apply_gate(&op, state)?;
        }
        Ok(())
    }

    /// Hybrid: combine layer and data parallelism
    fn execute_hybrid<F>(
        &self,
        circuit: &Circuit,
        state: &mut AdaptiveState,
        apply_gate: F,
    ) -> Result<()>
    where
        F: FnMut(&GateOp, &mut AdaptiveState) -> Result<()>,
    {
        // For now, same as layer-based
        // TODO: Add data parallelism within layers
        self.execute_layer_based(circuit, state, apply_gate)
    }

    /// Task-based parallelism with work stealing
    fn execute_task_based<F>(
        &self,
        circuit: &Circuit,
        state: &mut AdaptiveState,
        apply_gate: F,
    ) -> Result<()>
    where
        F: FnMut(&GateOp, &mut AdaptiveState) -> Result<()>,
    {
        // For now, same as layer-based
        // TODO: Implement work-stealing task queue
        self.execute_layer_based(circuit, state, apply_gate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_executor_creation() {
        let executor = ParallelExecutor::new(ParallelStrategy::LayerBased);
        assert_eq!(executor.strategy, ParallelStrategy::LayerBased);
    }
}
