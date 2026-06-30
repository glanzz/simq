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
            apply_gate(op, state)?;
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
    use simq_core::QubitId;
    use simq_gates::standard::{Hadamard, PauliX};
    use std::sync::Arc;

    fn make_circuit() -> Circuit {
        let mut c = Circuit::new(2);
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
        c.add_gate(Arc::new(PauliX), &[QubitId::new(1)]).unwrap();
        c
    }

    fn no_op_gate_fn(
        _gate_op: &simq_core::GateOp,
        _state: &mut AdaptiveState,
    ) -> Result<()> {
        Ok(())
    }

    #[test]
    fn test_parallel_executor_creation() {
        let executor = ParallelExecutor::new(ParallelStrategy::LayerBased);
        assert_eq!(executor.strategy, ParallelStrategy::LayerBased);
    }

    #[test]
    fn test_data_parallel_strategy() {
        let executor = ParallelExecutor::new(ParallelStrategy::DataParallel);
        let circuit = make_circuit();
        let mut state = AdaptiveState::new(2).unwrap();
        let result = executor.execute(&circuit, &mut state, no_op_gate_fn);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hybrid_strategy() {
        let executor = ParallelExecutor::new(ParallelStrategy::Hybrid);
        let circuit = make_circuit();
        let mut state = AdaptiveState::new(2).unwrap();
        let result = executor.execute(&circuit, &mut state, no_op_gate_fn);
        assert!(result.is_ok());
    }

    #[test]
    fn test_task_based_strategy() {
        let executor = ParallelExecutor::new(ParallelStrategy::TaskBased);
        let circuit = make_circuit();
        let mut state = AdaptiveState::new(2).unwrap();
        let result = executor.execute(&circuit, &mut state, no_op_gate_fn);
        assert!(result.is_ok());
    }

    #[test]
    fn test_layer_based_strategy() {
        let executor = ParallelExecutor::new(ParallelStrategy::LayerBased);
        let circuit = make_circuit();
        let mut state = AdaptiveState::new(2).unwrap();
        let result = executor.execute(&circuit, &mut state, no_op_gate_fn);
        assert!(result.is_ok());
    }

    #[test]
    fn test_empty_circuit_all_strategies() {
        let circuit = Circuit::new(1);
        let mut state = AdaptiveState::new(1).unwrap();
        for strategy in &[
            ParallelStrategy::LayerBased,
            ParallelStrategy::DataParallel,
            ParallelStrategy::Hybrid,
            ParallelStrategy::TaskBased,
        ] {
            let executor = ParallelExecutor::new(*strategy);
            assert!(executor.execute(&circuit, &mut state, no_op_gate_fn).is_ok());
        }
    }
}
