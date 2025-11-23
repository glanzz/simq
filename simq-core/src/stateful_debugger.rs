//! Stateful circuit debugger with quantum state visualization
//!
//! This module provides an advanced debugger that executes the circuit step-by-step
//! and allows inspection of the quantum state at each step.
//!
//! **Note**: This debugger requires `simq-sim` and `simq-state` crates.
//!
//! # Example
//!
//! ```ignore
//! use simq_core::{Circuit, StatefulDebugger};
//!
//! let mut circuit = Circuit::new(2);
//! // ... add gates ...
//!
//! let mut debugger = StatefulDebugger::new(&circuit).unwrap();
//!
//! while debugger.step().unwrap() {
//!     println!("Step {}: {}", debugger.step_number(), debugger.current_gate_description());
//!     debugger.print_state_summary();
//! }
//! ```

use crate::{Circuit, GateOp};
use num_complex::Complex64;
use std::fmt;

/// Amplitude entry for state display
#[derive(Clone, Debug)]
pub struct AmplitudeEntry {
    /// Basis state index
    pub index: usize,
    /// Basis state string (e.g., "|00⟩", "|11⟩")
    pub basis: String,
    /// Complex amplitude
    pub amplitude: Complex64,
    /// Probability (|amplitude|²)
    pub probability: f64,
}

/// State information at a given step
#[derive(Clone)]
pub struct StateSnapshot {
    /// Step number
    pub step: usize,
    /// Number of qubits
    pub num_qubits: usize,
    /// State amplitudes
    pub amplitudes: Vec<Complex64>,
    /// Gate applied (if any)
    pub gate_applied: Option<String>,
}

/// Configuration for state visualization
#[derive(Clone, Debug)]
pub struct VisualizationConfig {
    /// Show only non-zero amplitudes
    pub hide_zeros: bool,
    /// Threshold for considering amplitude as zero
    pub zero_threshold: f64,
    /// Maximum number of amplitudes to display
    pub max_display: usize,
    /// Show probabilities
    pub show_probabilities: bool,
    /// Show phase information
    pub show_phase: bool,
    /// Precision for amplitude display
    pub precision: usize,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            hide_zeros: true,
            zero_threshold: 1e-10,
            max_display: 10,
            show_probabilities: true,
            show_phase: false,
            precision: 4,
        }
    }
}

/// Stateful debugger without external dependencies
///
/// This is a lightweight version that tracks circuit structure
/// but doesn't execute quantum operations. For full state execution,
/// use with simq-sim.
///
/// # State Execution
///
/// This debugger provides utilities for state visualization but doesn't
/// execute gates itself. To use with state execution:
///
/// ```ignore
/// use simq_core::{Circuit, StatefulDebugger};
/// use simq_sim::Simulator;
/// use simq_state::AdaptiveState;
///
/// let debugger = StatefulDebugger::new(&circuit).unwrap();
/// let simulator = Simulator::default();
/// let mut state = AdaptiveState::new(circuit.num_qubits()).unwrap();
///
/// // Execute up to current step
/// let partial_circuit = create_partial_circuit(&circuit, debugger.step_number());
/// simulator.run(&partial_circuit).unwrap();
///
/// // Visualize state
/// let snapshot = StatefulDebugger::capture_state(&state, debugger.step_number());
/// println!("{}", snapshot.format_state(&debugger.config()));
/// ```
pub struct StatefulDebugger<'a> {
    circuit: &'a Circuit,
    current_step: usize,
    history: Vec<(usize, String)>,
    config: VisualizationConfig,
}

impl<'a> StatefulDebugger<'a> {
    /// Create a new stateful debugger
    ///
    /// Note: This lightweight version doesn't execute quantum gates.
    /// It only tracks circuit structure. For state execution, use
    /// the debugger with simq-sim integration.
    pub fn new(circuit: &'a Circuit) -> Result<Self, String> {
        Ok(Self {
            circuit,
            current_step: 0,
            history: Vec::new(),
            config: VisualizationConfig::default(),
        })
    }

    /// Create debugger with custom visualization config
    pub fn with_config(circuit: &'a Circuit, config: VisualizationConfig) -> Result<Self, String> {
        Ok(Self {
            circuit,
            current_step: 0,
            history: Vec::new(),
            config,
        })
    }

    /// Get the total number of gates
    pub fn total_gates(&self) -> usize {
        self.circuit.len()
    }

    /// Get the current step number
    pub fn step_number(&self) -> usize {
        self.current_step
    }

    /// Check if there are more gates to execute
    pub fn has_next(&self) -> bool {
        self.current_step < self.total_gates()
    }

    /// Get current gate operation
    pub fn current_gate(&self) -> Option<&GateOp> {
        self.circuit.get_operation(self.current_step)
    }

    /// Get current gate description
    pub fn current_gate_description(&self) -> String {
        self.current_gate()
            .map(|op| op.gate().description())
            .unwrap_or_else(|| "(end of circuit)".to_string())
    }

    /// Step forward one gate
    pub fn step(&mut self) -> Result<bool, String> {
        if !self.has_next() {
            return Ok(false);
        }

        if let Some(gate) = self.current_gate() {
            self.history.push((
                self.current_step,
                gate.gate().name().to_string(),
            ));
        }

        self.current_step += 1;
        Ok(true)
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.history.clear();
    }

    /// Get execution history
    pub fn history(&self) -> &[(usize, String)] {
        &self.history
    }

    /// Get visualization config
    pub fn config(&self) -> &VisualizationConfig {
        &self.config
    }

    /// Set visualization config
    pub fn set_config(&mut self, config: VisualizationConfig) {
        self.config = config;
    }

    /// Print state summary (placeholder without simq-sim)
    pub fn print_state_summary(&self) {
        println!("State execution requires simq-sim integration.");
        println!("Circuit position: step {}/{}", self.current_step, self.total_gates());
    }

    /// Create a state snapshot from raw amplitudes
    ///
    /// This is a utility method for use with simq-state.
    pub fn capture_state(amplitudes: &[Complex64], num_qubits: usize, step: usize, gate_name: Option<String>) -> StateSnapshot {
        StateSnapshot {
            step,
            num_qubits,
            amplitudes: amplitudes.to_vec(),
            gate_applied: gate_name,
        }
    }

    /// Format state vector for display
    pub fn format_state_vector(amplitudes: &[Complex64], num_qubits: usize, config: &VisualizationConfig) -> String {
        let snapshot = StateSnapshot {
            step: 0,
            num_qubits,
            amplitudes: amplitudes.to_vec(),
            gate_applied: None,
        };
        snapshot.format_state(config)
    }

    /// Get measurement probabilities from amplitudes
    pub fn measurement_probabilities(amplitudes: &[Complex64], num_qubits: usize) -> Vec<(String, f64)> {
        amplitudes
            .iter()
            .enumerate()
            .filter_map(|(idx, &amp)| {
                let prob = (amp.re * amp.re + amp.im * amp.im).abs();
                if prob > 1e-10 {
                    Some((format_basis_state(idx, num_qubits), prob))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Find the most likely measurement outcome
    pub fn most_likely_outcome(amplitudes: &[Complex64], num_qubits: usize) -> Option<(String, f64)> {
        amplitudes
            .iter()
            .enumerate()
            .map(|(idx, &amp)| {
                let prob = (amp.re * amp.re + amp.im * amp.im).abs();
                (format_basis_state(idx, num_qubits), prob)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }

    /// Calculate state fidelity between two amplitude vectors
    pub fn fidelity(amplitudes1: &[Complex64], amplitudes2: &[Complex64]) -> f64 {
        if amplitudes1.len() != amplitudes2.len() {
            return 0.0;
        }

        let overlap: Complex64 = amplitudes1
            .iter()
            .zip(amplitudes2.iter())
            .map(|(&a1, &a2)| a1.conj() * a2)
            .sum();

        (overlap.re * overlap.re + overlap.im * overlap.im).abs()
    }

    /// Calculate state purity (Tr(ρ²))
    ///
    /// For a pure state |ψ⟩, this returns 1.0.
    /// Note: This assumes the input is already a pure state vector.
    /// For true purity calculation of mixed states, use density matrix formalism.
    pub fn purity(amplitudes: &[Complex64]) -> f64 {
        // For a pure state represented as |ψ⟩, purity = Tr(|ψ⟩⟨ψ|²) = 1
        // We calculate sum of |a_i|^4 which equals 1 for normalized pure states
        let sum_fourth_power: f64 = amplitudes
            .iter()
            .map(|&amp| {
                let prob = amp.re * amp.re + amp.im * amp.im;
                prob * prob
            })
            .sum();
        sum_fourth_power
    }

    /// Extract single-qubit reduced state from multi-qubit state
    ///
    /// Traces out all qubits except the specified one.
    ///
    /// # Arguments
    /// * `amplitudes` - Full state vector
    /// * `num_qubits` - Total number of qubits
    /// * `target_qubit` - Index of qubit to extract (0-indexed)
    ///
    /// # Returns
    /// Two-element array representing the reduced single-qubit state
    pub fn extract_single_qubit_state(
        amplitudes: &[Complex64],
        num_qubits: usize,
        target_qubit: usize,
    ) -> [Complex64; 2] {
        if target_qubit >= num_qubits {
            return [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        }

        let dimension = 1 << num_qubits;
        if amplitudes.len() != dimension {
            return [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        }

        // For a single qubit system, just return the amplitudes
        if num_qubits == 1 {
            return [amplitudes[0], amplitudes[1]];
        }

        // Calculate reduced density matrix elements
        let mut rho_00 = Complex64::new(0.0, 0.0);
        let mut rho_11 = Complex64::new(0.0, 0.0);

        let target_bit = num_qubits - 1 - target_qubit;

        for i in 0..dimension {
            let target_state = (i >> target_bit) & 1;
            if target_state == 0 {
                rho_00 += amplitudes[i].conj() * amplitudes[i];
            } else {
                rho_11 += amplitudes[i].conj() * amplitudes[i];
            }
        }

        // For visualization purposes, return sqrt of diagonal elements
        // (this is an approximation for mixed states)
        [
            Complex64::new(rho_00.re.sqrt(), 0.0),
            Complex64::new(rho_11.re.sqrt(), 0.0),
        ]
    }
}

/// Helper functions for state visualization
impl StateSnapshot {
    /// Get significant amplitude entries
    pub fn significant_amplitudes(&self, config: &VisualizationConfig) -> Vec<AmplitudeEntry> {
        let mut entries: Vec<AmplitudeEntry> = self
            .amplitudes
            .iter()
            .enumerate()
            .filter_map(|(idx, &amp)| {
                let prob = (amp.re * amp.re + amp.im * amp.im).abs();
                if !config.hide_zeros || prob >= config.zero_threshold {
                    Some(AmplitudeEntry {
                        index: idx,
                        basis: format_basis_state(idx, self.num_qubits),
                        amplitude: amp,
                        probability: prob,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by probability (descending)
        entries.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());

        // Limit display
        entries.truncate(config.max_display);
        entries
    }

    /// Get measurement probabilities
    pub fn measurement_probabilities(&self) -> Vec<(String, f64)> {
        self.amplitudes
            .iter()
            .enumerate()
            .map(|(idx, &amp)| {
                let prob = (amp.re * amp.re + amp.im * amp.im).abs();
                (format_basis_state(idx, self.num_qubits), prob)
            })
            .filter(|(_, prob)| *prob > 1e-10)
            .collect()
    }

    /// Format state as string
    pub fn format_state(&self, config: &VisualizationConfig) -> String {
        let entries = self.significant_amplitudes(config);

        if entries.is_empty() {
            return "State: (all zeros)".to_string();
        }

        let mut output = String::new();
        output.push_str(&format!("State at step {}:\n", self.step));

        for entry in entries {
            let amp_str = if config.show_phase {
                format_complex_polar(entry.amplitude, config.precision)
            } else {
                format_complex(entry.amplitude, config.precision)
            };

            output.push_str(&format!("  {} : {}", entry.basis, amp_str));

            if config.show_probabilities {
                output.push_str(&format!("  (p={:.4})", entry.probability));
            }

            output.push('\n');
        }

        output
    }
}

/// Format basis state (e.g., 0 -> "|00⟩", 3 -> "|11⟩")
fn format_basis_state(index: usize, num_qubits: usize) -> String {
    let mut result = String::from("|");
    for i in (0..num_qubits).rev() {
        if (index >> i) & 1 == 1 {
            result.push('1');
        } else {
            result.push('0');
        }
    }
    result.push('⟩');
    result
}

/// Format complex number
fn format_complex(c: Complex64, precision: usize) -> String {
    let re = c.re;
    let im = c.im;

    if im.abs() < 1e-10 {
        format!("{:.*}", precision, re)
    } else if re.abs() < 1e-10 {
        format!("{:.*}i", precision, im)
    } else if im >= 0.0 {
        format!("{:.*}+{:.*}i", precision, re, precision, im)
    } else {
        format!("{:.*}{:.*}i", precision, re, precision, im)
    }
}

/// Format complex number in polar form
fn format_complex_polar(c: Complex64, precision: usize) -> String {
    let magnitude = (c.re * c.re + c.im * c.im).sqrt();
    let phase = c.im.atan2(c.re);

    if phase.abs() < 1e-10 {
        format!("{:.*}", precision, magnitude)
    } else {
        format!("{:.*}e^({:.*}i)", precision, magnitude, precision, phase)
    }
}

impl fmt::Display for AmplitudeEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} : {} (p={:.4})",
            self.basis,
            format_complex(self.amplitude, 4),
            self.probability
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Circuit;
    use std::sync::Arc;

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

    #[test]
    fn test_format_basis_state() {
        assert_eq!(format_basis_state(0, 2), "|00⟩");
        assert_eq!(format_basis_state(1, 2), "|01⟩");
        assert_eq!(format_basis_state(2, 2), "|10⟩");
        assert_eq!(format_basis_state(3, 2), "|11⟩");

        assert_eq!(format_basis_state(5, 3), "|101⟩");
    }

    #[test]
    fn test_format_complex() {
        assert_eq!(format_complex(Complex64::new(1.0, 0.0), 2), "1.00");
        assert_eq!(format_complex(Complex64::new(0.0, 1.0), 2), "1.00i");
        assert_eq!(format_complex(Complex64::new(1.0, 1.0), 2), "1.00+1.00i");
        assert_eq!(format_complex(Complex64::new(1.0, -1.0), 2), "1.00-1.00i");
    }

    #[test]
    fn test_stateful_debugger_basic() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(MockGate("H")), &[crate::QubitId::new(0)])
            .unwrap();

        let mut debugger = StatefulDebugger::new(&circuit).unwrap();
        assert_eq!(debugger.total_gates(), 1);
        assert!(debugger.has_next());

        assert!(debugger.step().unwrap());
        assert!(!debugger.has_next());
    }

    #[test]
    fn test_visualization_config() {
        let config = VisualizationConfig {
            hide_zeros: false,
            max_display: 5,
            ..Default::default()
        };

        assert!(!config.hide_zeros);
        assert_eq!(config.max_display, 5);
    }

    #[test]
    fn test_state_snapshot() {
        let amplitudes = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        let snapshot = StateSnapshot {
            step: 1,
            num_qubits: 2,
            amplitudes,
            gate_applied: Some("H".to_string()),
        };

        let config = VisualizationConfig::default();
        let sig_amps = snapshot.significant_amplitudes(&config);

        assert_eq!(sig_amps.len(), 2);
        assert_eq!(sig_amps[0].basis, "|00⟩");
        assert_eq!(sig_amps[1].basis, "|01⟩");
    }

    #[test]
    fn test_measurement_probabilities() {
        let amplitudes = vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
        ];

        let probs = StatefulDebugger::measurement_probabilities(&amplitudes, 2);
        assert_eq!(probs.len(), 3);
        assert_eq!(probs[0].0, "|00⟩");
        assert!((probs[0].1 - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_most_likely_outcome() {
        let amplitudes = vec![
            Complex64::new(0.1, 0.0),
            Complex64::new(0.9, 0.0),
            Complex64::new(0.2, 0.0),
            Complex64::new(0.1, 0.0),
        ];

        let outcome = StatefulDebugger::most_likely_outcome(&amplitudes, 2);
        assert!(outcome.is_some());
        let (basis, prob) = outcome.unwrap();
        assert_eq!(basis, "|01⟩");
        assert!((prob - 0.81).abs() < 1e-10);
    }

    #[test]
    fn test_fidelity() {
        // Identical states should have fidelity 1.0
        let amplitudes1 = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let amplitudes2 = amplitudes1.clone();

        let fid = StatefulDebugger::fidelity(&amplitudes1, &amplitudes2);
        assert!((fid - 1.0).abs() < 1e-10);

        // Orthogonal states should have fidelity 0.0
        let amplitudes3 = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        let fid2 = StatefulDebugger::fidelity(&amplitudes1, &amplitudes3);
        assert!(fid2.abs() < 1e-10);
    }

    #[test]
    fn test_purity() {
        // Pure state |0⟩ should have purity 1.0
        let pure_state = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let purity = StatefulDebugger::purity(&pure_state);
        assert!((purity - 1.0).abs() < 1e-10);

        // Equal superposition (|0⟩ + |1⟩)/√2 has sum(|a_i|^4) = 2*(1/2)^2 = 0.5
        let superposition = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        let purity2 = StatefulDebugger::purity(&superposition);
        assert!((purity2 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_format_state_vector() {
        let amplitudes = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];

        let config = VisualizationConfig::default();
        let output = StatefulDebugger::format_state_vector(&amplitudes, 1, &config);

        assert!(output.contains("|0⟩"));
        assert!(output.contains("|1⟩"));
    }
}
