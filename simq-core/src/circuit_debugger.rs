//! Circuit debugger for step-by-step circuit execution and visualization
//!
//! This module provides tools to debug quantum circuits by:
//! - Stepping through gates one at a time
//! - Inspecting circuit state at each step
//! - Visualizing the current execution position
//! - Tracking gate application order
//!
//! # Example
//!
//! ```ignore
//! use simq_core::{Circuit, CircuitDebugger};
//!
//! let mut circuit = Circuit::new(2);
//! // ... add gates ...
//!
//! let mut debugger = CircuitDebugger::new(&circuit);
//! while debugger.has_next() {
//!     debugger.step();
//!     println!("Step {}: {}", debugger.step_number(), debugger.current_gate_description());
//!     println!("{}", debugger.visualize_current_position());
//! }
//! ```

use crate::{Circuit, GateOp};
use std::fmt;

/// Circuit debugger for step-by-step execution
///
/// Provides mechanisms to walk through a circuit gate by gate,
/// inspecting operations and visualizing execution progress.
#[derive(Clone, Debug)]
pub struct CircuitDebugger<'a> {
    circuit: &'a Circuit,
    current_step: usize,
    breakpoints: Vec<usize>,
    step_history: Vec<StepInfo>,
}

/// Information about a single execution step
#[derive(Clone, Debug)]
pub struct StepInfo {
    /// Step number (0-indexed)
    pub step: usize,
    /// Gate operation applied
    pub gate_name: String,
    /// Qubits affected
    pub qubits: Vec<usize>,
    /// Gate description
    pub description: String,
}

impl<'a> CircuitDebugger<'a> {
    /// Create a new debugger for the given circuit
    ///
    /// # Arguments
    /// * `circuit` - The circuit to debug
    ///
    /// # Example
    /// ```ignore
    /// use simq_core::{Circuit, CircuitDebugger};
    ///
    /// let circuit = Circuit::new(2);
    /// let debugger = CircuitDebugger::new(&circuit);
    /// ```
    pub fn new(circuit: &'a Circuit) -> Self {
        Self {
            circuit,
            current_step: 0,
            breakpoints: Vec::new(),
            step_history: Vec::new(),
        }
    }

    /// Get the total number of gates in the circuit
    #[inline]
    pub fn total_gates(&self) -> usize {
        self.circuit.len()
    }

    /// Get the current step number (0-indexed)
    #[inline]
    pub fn step_number(&self) -> usize {
        self.current_step
    }

    /// Check if there are more gates to execute
    #[inline]
    pub fn has_next(&self) -> bool {
        self.current_step < self.total_gates()
    }

    /// Check if we're at the beginning
    #[inline]
    pub fn is_at_start(&self) -> bool {
        self.current_step == 0
    }

    /// Check if we're at the end
    #[inline]
    pub fn is_at_end(&self) -> bool {
        self.current_step >= self.total_gates()
    }

    /// Get the current gate operation (if not at end)
    pub fn current_gate(&self) -> Option<&GateOp> {
        self.circuit.get_operation(self.current_step)
    }

    /// Get the current gate's name
    pub fn current_gate_name(&self) -> Option<&str> {
        self.current_gate().map(|op| op.gate().name())
    }

    /// Get the current gate's description
    pub fn current_gate_description(&self) -> String {
        self.current_gate()
            .map(|op| op.gate().description())
            .unwrap_or_else(|| "(end of circuit)".to_string())
    }

    /// Get the qubits affected by the current gate
    pub fn current_qubits(&self) -> Vec<usize> {
        self.current_gate()
            .map(|op| op.qubits().iter().map(|q| q.index()).collect())
            .unwrap_or_default()
    }

    /// Step forward one gate
    ///
    /// # Returns
    /// `true` if stepped successfully, `false` if already at end
    ///
    /// # Example
    /// ```ignore
    /// let mut debugger = CircuitDebugger::new(&circuit);
    /// while debugger.step() {
    ///     println!("Executed: {}", debugger.current_gate_name().unwrap());
    /// }
    /// ```
    pub fn step(&mut self) -> bool {
        if !self.has_next() {
            return false;
        }

        // Record step info before advancing
        if let Some(gate_op) = self.current_gate() {
            let step_info = StepInfo {
                step: self.current_step,
                gate_name: gate_op.gate().name().to_string(),
                qubits: gate_op.qubits().iter().map(|q| q.index()).collect(),
                description: gate_op.gate().description(),
            };
            self.step_history.push(step_info);
        }

        self.current_step += 1;
        true
    }

    /// Step backward one gate
    ///
    /// # Returns
    /// `true` if stepped back successfully, `false` if already at start
    pub fn step_back(&mut self) -> bool {
        if self.is_at_start() {
            return false;
        }

        self.current_step -= 1;
        if let Some(last_idx) = self.step_history.len().checked_sub(1) {
            self.step_history.truncate(last_idx);
        }
        true
    }

    /// Reset to the beginning of the circuit
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.step_history.clear();
    }

    /// Jump to a specific step
    ///
    /// # Arguments
    /// * `step` - The step number to jump to (0-indexed)
    ///
    /// # Returns
    /// `true` if jump was successful, `false` if step is out of bounds
    pub fn jump_to(&mut self, step: usize) -> bool {
        if step > self.total_gates() {
            return false;
        }

        // If jumping forward, need to record history
        if step > self.current_step {
            while self.current_step < step {
                if !self.step() {
                    return false;
                }
            }
        } else if step < self.current_step {
            // If jumping backward, reset and step forward
            self.reset();
            while self.current_step < step {
                if !self.step() {
                    return false;
                }
            }
        }

        true
    }

    /// Run until the next breakpoint or end
    ///
    /// # Returns
    /// `true` if hit a breakpoint, `false` if reached end
    pub fn continue_execution(&mut self) -> bool {
        while self.has_next() {
            let next_step = self.current_step + 1;
            if self.breakpoints.contains(&next_step) {
                self.step();
                return true;
            }
            self.step();
        }
        false
    }

    /// Add a breakpoint at the specified step
    ///
    /// # Arguments
    /// * `step` - The step number where execution should pause
    pub fn add_breakpoint(&mut self, step: usize) {
        if step < self.total_gates() && !self.breakpoints.contains(&step) {
            self.breakpoints.push(step);
            self.breakpoints.sort_unstable();
        }
    }

    /// Remove a breakpoint
    ///
    /// # Arguments
    /// * `step` - The step number to remove breakpoint from
    pub fn remove_breakpoint(&mut self, step: usize) {
        self.breakpoints.retain(|&s| s != step);
    }

    /// Clear all breakpoints
    pub fn clear_breakpoints(&mut self) {
        self.breakpoints.clear();
    }

    /// Get all breakpoints
    pub fn breakpoints(&self) -> &[usize] {
        &self.breakpoints
    }

    /// Get the step history
    pub fn history(&self) -> &[StepInfo] {
        &self.step_history
    }

    /// Visualize the current execution position in the circuit
    ///
    /// Returns an ASCII representation with a marker showing where we are.
    ///
    /// # Example output
    /// ```text
    /// q0: ───[H]────●─────   ← HERE
    /// q1: ───────────⊕────
    /// ```
    pub fn visualize_current_position(&self) -> String {
        use crate::ascii_renderer::AsciiConfig;

        let mut output = String::new();

        // Render the circuit with a marker for current position
        output.push_str(&format!("Step {}/{}\n", self.current_step, self.total_gates()));

        if let Some(gate_op) = self.current_gate() {
            output.push_str(&format!(
                "Current: {} on qubits {:?}\n",
                gate_op.gate().name(),
                self.current_qubits()
            ));
        } else {
            output.push_str("Current: (end of circuit)\n");
        }

        output.push('\n');

        // Render full circuit
        let config = AsciiConfig {
            max_width: 80,
            compact: true,
            ..Default::default()
        };
        output.push_str(&crate::ascii_renderer::render_with_config(self.circuit, &config));

        output
    }

    /// Get a compact summary of the current state
    pub fn status(&self) -> DebuggerStatus {
        DebuggerStatus {
            current_step: self.current_step,
            total_gates: self.total_gates(),
            current_gate: self.current_gate_name().map(|s| s.to_string()),
            current_qubits: self.current_qubits(),
            breakpoints: self.breakpoints.clone(),
            history_length: self.step_history.len(),
        }
    }

    /// Print a detailed execution trace
    pub fn print_trace(&self) {
        println!("=== Execution Trace ===");
        println!("Total gates: {}", self.total_gates());
        println!("Current step: {}", self.current_step);
        println!();

        for step_info in &self.step_history {
            println!(
                "[{}] {} on qubits {:?} - {}",
                step_info.step, step_info.gate_name, step_info.qubits, step_info.description
            );
        }

        if self.has_next() {
            if let Some(gate) = self.current_gate() {
                println!(
                    ">>> [{}] {} on qubits {:?} (NEXT)",
                    self.current_step,
                    gate.gate().name(),
                    self.current_qubits()
                );
            }
        } else {
            println!(">>> [END OF CIRCUIT]");
        }
    }

    /// Get a slice of the circuit from start to current position
    ///
    /// Useful for partial execution or replay.
    pub fn executed_operations(&self) -> &[GateOp] {
        &self.circuit.operations_slice()[..self.current_step.min(self.circuit.len())]
    }

    /// Get a slice of remaining operations
    pub fn remaining_operations(&self) -> &[GateOp] {
        if self.current_step < self.circuit.len() {
            &self.circuit.operations_slice()[self.current_step..]
        } else {
            &[]
        }
    }

    /// Create a sub-circuit containing only executed operations
    pub fn to_executed_circuit(&self) -> Circuit {
        let mut new_circuit = Circuit::new(self.circuit.num_qubits());
        for op in self.executed_operations() {
            new_circuit
                .add_gate(op.gate().clone(), op.qubits())
                .unwrap();
        }
        new_circuit
    }
}

/// Summary of debugger status
#[derive(Clone, Debug)]
pub struct DebuggerStatus {
    pub current_step: usize,
    pub total_gates: usize,
    pub current_gate: Option<String>,
    pub current_qubits: Vec<usize>,
    pub breakpoints: Vec<usize>,
    pub history_length: usize,
}

impl fmt::Display for DebuggerStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Debugger Status:")?;
        writeln!(f, "  Step: {}/{}", self.current_step, self.total_gates)?;
        if let Some(ref gate) = self.current_gate {
            writeln!(f, "  Current: {} on {:?}", gate, self.current_qubits)?;
        } else {
            writeln!(f, "  Current: (end)")?;
        }
        writeln!(f, "  Breakpoints: {:?}", self.breakpoints)?;
        writeln!(f, "  History length: {}", self.history_length)?;
        Ok(())
    }
}

impl fmt::Display for StepInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Step {}: {} on {:?} - {}",
            self.step, self.gate_name, self.qubits, self.description
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Mock gate for testing
    #[derive(Debug)]
    struct MockGate(&'static str);
    impl crate::Gate for MockGate {
        fn name(&self) -> &str {
            self.0
        }
        fn num_qubits(&self) -> usize {
            1
        }
    }

    fn create_test_circuit() -> Circuit {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(MockGate("H")), &[crate::QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(MockGate("X")), &[crate::QubitId::new(1)])
            .unwrap();
        circuit
            .add_gate(Arc::new(MockGate("CNOT")), &[crate::QubitId::new(0)])
            .unwrap();
        circuit
    }

    #[test]
    fn test_basic_stepping() {
        let circuit = create_test_circuit();
        let mut debugger = CircuitDebugger::new(&circuit);

        assert_eq!(debugger.total_gates(), 3);
        assert!(debugger.is_at_start());
        assert!(debugger.has_next());

        // Step 1
        assert!(debugger.step());
        assert_eq!(debugger.step_number(), 1);
        assert_eq!(debugger.current_gate_name(), Some("X"));

        // Step 2
        assert!(debugger.step());
        assert_eq!(debugger.step_number(), 2);

        // Step 3
        assert!(debugger.step());
        assert!(debugger.is_at_end());
        assert!(!debugger.has_next());

        // Can't step past end
        assert!(!debugger.step());
    }

    #[test]
    fn test_step_back() {
        let circuit = create_test_circuit();
        let mut debugger = CircuitDebugger::new(&circuit);

        debugger.step();
        debugger.step();
        assert_eq!(debugger.step_number(), 2);

        assert!(debugger.step_back());
        assert_eq!(debugger.step_number(), 1);

        assert!(debugger.step_back());
        assert_eq!(debugger.step_number(), 0);

        assert!(!debugger.step_back());
    }

    #[test]
    fn test_reset() {
        let circuit = create_test_circuit();
        let mut debugger = CircuitDebugger::new(&circuit);

        debugger.step();
        debugger.step();
        debugger.reset();

        assert_eq!(debugger.step_number(), 0);
        assert!(debugger.is_at_start());
        assert_eq!(debugger.history().len(), 0);
    }

    #[test]
    fn test_jump_to() {
        let circuit = create_test_circuit();
        let mut debugger = CircuitDebugger::new(&circuit);

        assert!(debugger.jump_to(2));
        assert_eq!(debugger.step_number(), 2);

        assert!(debugger.jump_to(1));
        assert_eq!(debugger.step_number(), 1);

        assert!(!debugger.jump_to(100));
    }

    #[test]
    fn test_breakpoints() {
        let circuit = create_test_circuit();
        let mut debugger = CircuitDebugger::new(&circuit);

        debugger.add_breakpoint(1);
        debugger.add_breakpoint(2);

        assert_eq!(debugger.breakpoints(), &[1, 2]);

        debugger.remove_breakpoint(1);
        assert_eq!(debugger.breakpoints(), &[2]);

        debugger.clear_breakpoints();
        assert_eq!(debugger.breakpoints().len(), 0);
    }

    #[test]
    fn test_continue_execution() {
        let circuit = create_test_circuit();
        let mut debugger = CircuitDebugger::new(&circuit);

        debugger.add_breakpoint(2);

        assert!(debugger.continue_execution());
        assert_eq!(debugger.step_number(), 2);

        assert!(!debugger.continue_execution());
        assert!(debugger.is_at_end());
    }

    #[test]
    fn test_history() {
        let circuit = create_test_circuit();
        let mut debugger = CircuitDebugger::new(&circuit);

        debugger.step();
        debugger.step();

        let history = debugger.history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].gate_name, "H");
        assert_eq!(history[1].gate_name, "X");
    }

    #[test]
    fn test_executed_operations() {
        let circuit = create_test_circuit();
        let mut debugger = CircuitDebugger::new(&circuit);

        debugger.step();
        debugger.step();

        let executed = debugger.executed_operations();
        assert_eq!(executed.len(), 2);

        let remaining = debugger.remaining_operations();
        assert_eq!(remaining.len(), 1);
    }

    #[test]
    fn test_to_executed_circuit() {
        let circuit = create_test_circuit();
        let mut debugger = CircuitDebugger::new(&circuit);

        debugger.step();
        debugger.step();

        let partial = debugger.to_executed_circuit();
        assert_eq!(partial.len(), 2);
        assert_eq!(partial.num_qubits(), 2);
    }

    #[test]
    fn test_status() {
        let circuit = create_test_circuit();
        let mut debugger = CircuitDebugger::new(&circuit);

        debugger.step();
        let status = debugger.status();

        assert_eq!(status.current_step, 1);
        assert_eq!(status.total_gates, 3);
        assert_eq!(status.current_gate, Some("X".to_string()));
    }
}
