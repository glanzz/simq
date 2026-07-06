//! LaTeX circuit renderer using quantikz package
//!
//! Generates LaTeX code for quantum circuit visualization using the
//! `quantikz` package, which provides high-quality typeset circuit diagrams.
//!
//! # Example
//!
//! ```ignore
//! use simq_core::{Circuit, LatexConfig, render_latex};
//!
//! let circuit = build_circuit();
//! let latex = circuit.to_latex();
//! // Output can be included in LaTeX documents with \usepackage{quantikz}
//! ```
//!
//! # Required LaTeX packages
//!
//! ```latex
//! \usepackage{tikz}
//! \usepackage{quantikz}
//! ```

use crate::{Circuit, GateOp};
use std::collections::HashMap;
use std::fmt::Write;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for LaTeX rendering
#[derive(Debug, Clone)]
pub struct LatexConfig {
    /// Include document preamble (for standalone compilation)
    pub standalone: bool,
    /// Show qubit labels (|q_0\rangle, etc.)
    pub show_labels: bool,
    /// Use ket notation for labels (|0\rangle vs q_0)
    pub ket_labels: bool,
    /// Scale factor for the circuit
    pub scale: f64,
    /// Row separation in cm
    pub row_sep: f64,
    /// Column separation in cm
    pub col_sep: f64,
    /// Precision for floating-point parameters
    pub float_precision: usize,
    /// Use thin lines for wires
    pub thin_lines: bool,
    /// Add slice indicators for circuit depth
    pub show_slices: bool,
    /// Custom gate styles (gate_name -> latex_style)
    pub gate_styles: HashMap<String, String>,
}

impl Default for LatexConfig {
    fn default() -> Self {
        Self {
            standalone: false,
            show_labels: true,
            ket_labels: true,
            scale: 1.0,
            row_sep: 0.5,
            col_sep: 0.5,
            float_precision: 4,
            thin_lines: false,
            show_slices: false,
            gate_styles: HashMap::new(),
        }
    }
}

impl LatexConfig {
    /// Create a standalone document configuration
    pub fn standalone() -> Self {
        Self {
            standalone: true,
            ..Default::default()
        }
    }

    /// Builder-style method to set standalone mode
    pub fn with_standalone(mut self, standalone: bool) -> Self {
        self.standalone = standalone;
        self
    }

    /// Builder-style method to set scale
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Builder-style method to set labels
    pub fn with_labels(mut self, show: bool) -> Self {
        self.show_labels = show;
        self
    }

    /// Builder-style method to use ket notation
    pub fn with_ket_labels(mut self, ket: bool) -> Self {
        self.ket_labels = ket;
        self
    }

    /// Builder-style method to add custom gate style
    pub fn with_gate_style(mut self, gate: impl Into<String>, style: impl Into<String>) -> Self {
        self.gate_styles.insert(gate.into(), style.into());
        self
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Render a circuit to LaTeX with default configuration
pub fn render(circuit: &Circuit) -> String {
    render_with_config(circuit, &LatexConfig::default())
}

/// Render a circuit to LaTeX with custom configuration
pub fn render_with_config(circuit: &Circuit, config: &LatexConfig) -> String {
    LatexRenderer::new(circuit, config).render()
}

// ============================================================================
// Core Renderer
// ============================================================================

struct LatexRenderer<'a> {
    circuit: &'a Circuit,
    config: &'a LatexConfig,
}

impl<'a> LatexRenderer<'a> {
    fn new(circuit: &'a Circuit, config: &'a LatexConfig) -> Self {
        Self { circuit, config }
    }

    fn render(&self) -> String {
        let mut output = String::new();

        if self.config.standalone {
            self.write_preamble(&mut output);
        }

        self.write_circuit(&mut output);

        if self.config.standalone {
            self.write_postamble(&mut output);
        }

        output
    }

    fn write_preamble(&self, output: &mut String) {
        output.push_str("\\documentclass[border=2pt]{standalone}\n");
        output.push_str("\\usepackage{tikz}\n");
        output.push_str("\\usepackage{quantikz}\n");
        output.push_str("\\begin{document}\n");
    }

    fn write_postamble(&self, output: &mut String) {
        output.push_str("\\end{document}\n");
    }

    fn write_circuit(&self, output: &mut String) {
        let num_qubits = self.circuit.num_qubits();

        // Build column structure
        let columns = self.build_columns();

        // Start quantikz environment
        if self.config.scale != 1.0 {
            let _ = writeln!(
                output,
                "\\begin{{quantikz}}[row sep={{{:.2}cm}}, column sep={{{:.2}cm}}, scale={:.2}]",
                self.config.row_sep, self.config.col_sep, self.config.scale
            );
        } else {
            let _ = writeln!(
                output,
                "\\begin{{quantikz}}[row sep={{{:.2}cm}}, column sep={{{:.2}cm}}]",
                self.config.row_sep, self.config.col_sep
            );
        }

        // Render each qubit row
        for q in 0..num_qubits {
            self.write_qubit_row(output, q, &columns);
            if q < num_qubits - 1 {
                output.push_str(" \\\\\n");
            } else {
                output.push('\n');
            }
        }

        output.push_str("\\end{quantikz}\n");
    }

    fn build_columns(&self) -> Vec<Vec<(usize, &GateOp)>> {
        let mut columns: Vec<Vec<(usize, &GateOp)>> = Vec::new();
        let mut qubit_column: HashMap<usize, usize> = HashMap::new();

        for (op_idx, op) in self.circuit.operations().enumerate() {
            let target_col = self.find_target_column(op, &qubit_column);

            while columns.len() <= target_col {
                columns.push(Vec::new());
            }

            columns[target_col].push((op_idx, op));

            // Update qubit positions
            for &qubit in op.qubits() {
                qubit_column.insert(qubit.index(), target_col);
            }

            // Mark intermediate qubits for multi-qubit gates
            if op.qubits().len() > 1 {
                let indices: Vec<usize> = op.qubits().iter().map(|q| q.index()).collect();
                if let (Some(&min_q), Some(&max_q)) = (indices.iter().min(), indices.iter().max()) {
                    for q in min_q..=max_q {
                        qubit_column.insert(q, target_col);
                    }
                }
            }
        }

        columns
    }

    fn find_target_column(&self, op: &GateOp, qubit_column: &HashMap<usize, usize>) -> usize {
        let mut target_col = 0;

        for &qubit in op.qubits() {
            if let Some(&col) = qubit_column.get(&qubit.index()) {
                target_col = target_col.max(col + 1);
            }
        }

        if op.qubits().len() > 1 {
            let indices: Vec<usize> = op.qubits().iter().map(|q| q.index()).collect();
            if let (Some(&min_q), Some(&max_q)) = (indices.iter().min(), indices.iter().max()) {
                for q in min_q..=max_q {
                    if let Some(&col) = qubit_column.get(&q) {
                        target_col = target_col.max(col + 1);
                    }
                }
            }
        }

        target_col
    }

    fn write_qubit_row(
        &self,
        output: &mut String,
        qubit: usize,
        columns: &[Vec<(usize, &GateOp)>],
    ) {
        // Write label
        if self.config.show_labels {
            if self.config.ket_labels {
                let _ = write!(output, "\\lstick{{\\ket{{q_{}}}}}", qubit);
            } else {
                let _ = write!(output, "\\lstick{{q_{}}}", qubit);
            }
        }

        // Write gates for each column
        for (col_idx, col) in columns.iter().enumerate() {
            output.push_str(" & ");

            let gate_for_qubit = self.find_gate_for_qubit(col, qubit);

            match gate_for_qubit {
                Some((op, role)) => {
                    self.write_gate(output, op, qubit, role);
                },
                None => {
                    // Check if we're a wire passing through a multi-qubit gate
                    output.push_str("\\qw");
                },
            }

            // Add slice marker if enabled
            if self.config.show_slices && col_idx < columns.len() - 1 {
                output.push_str(" \\slice{}");
            }
        }

        // End with wire
        output.push_str(" & \\qw");
    }

    fn find_gate_for_qubit<'b>(
        &self,
        col: &'b [(usize, &'b GateOp)],
        qubit: usize,
    ) -> Option<(&'b GateOp, GateRole)> {
        for (_, op) in col {
            let qubits = op.qubits();
            let indices: Vec<usize> = qubits.iter().map(|q| q.index()).collect();

            if indices.contains(&qubit) {
                let role = self.determine_role(op, qubit, &indices);
                return Some((op, role));
            }
        }
        None
    }

    #[allow(dead_code)]
    fn is_wire_through(&self, col: &[(usize, &GateOp)], qubit: usize) -> bool {
        for (_, op) in col {
            let indices: Vec<usize> = op.qubits().iter().map(|q| q.index()).collect();
            if let (Some(&min_q), Some(&max_q)) = (indices.iter().min(), indices.iter().max()) {
                if qubit > min_q && qubit < max_q && !indices.contains(&qubit) {
                    return true;
                }
            }
        }
        false
    }

    fn determine_role(&self, op: &GateOp, qubit: usize, indices: &[usize]) -> GateRole {
        if indices.len() == 1 {
            return GateRole::Single;
        }

        let name = op.gate().name().to_uppercase();
        let position = indices.iter().position(|&q| q == qubit).unwrap();

        // SWAP gates
        if name == "SWAP" || name == "ISWAP" {
            return if position == 0 {
                GateRole::SwapFirst
            } else {
                GateRole::SwapSecond
            };
        }

        // Controlled SWAP (Fredkin)
        if name == "CSWAP" || name == "FREDKIN" {
            return match position {
                0 => GateRole::Control,
                1 => GateRole::SwapFirst,
                _ => GateRole::SwapSecond,
            };
        }

        // Controlled gates
        if name.starts_with('C') || name == "CNOT" || name == "CCNOT" || name == "TOFFOLI" {
            return if position < indices.len() - 1 {
                GateRole::Control
            } else {
                GateRole::Target
            };
        }

        // Symmetric multi-qubit gates
        GateRole::Multi(position)
    }

    fn write_gate(&self, output: &mut String, op: &GateOp, qubit: usize, role: GateRole) {
        let name = op.gate().name();
        let name_upper = name.to_uppercase();
        let indices: Vec<usize> = op.qubits().iter().map(|q| q.index()).collect();

        // Check for custom style
        if let Some(style) = self.config.gate_styles.get(name) {
            let _ = write!(output, "{}", style);
            return;
        }

        match role {
            GateRole::Single => {
                self.write_single_gate(output, op);
            },
            GateRole::Control => {
                // Find target qubit offset
                let target_idx = indices.last().unwrap();
                let offset = (*target_idx as isize) - (qubit as isize);
                let _ = write!(output, "\\ctrl{{{}}}", offset);
            },
            GateRole::Target => {
                match name_upper.as_str() {
                    "CNOT" | "CX" | "CCNOT" | "CCX" | "TOFFOLI" => {
                        output.push_str("\\targ{}");
                    },
                    "CZ" => {
                        // CZ shows control on both ends
                        let first_idx = indices.first().unwrap();
                        let offset = (*first_idx as isize) - (qubit as isize);
                        let _ = write!(output, "\\ctrl{{{}}}", offset);
                    },
                    "CY" => {
                        output.push_str("\\gate{Y}");
                    },
                    _ => {
                        // Generic controlled gate - show base gate
                        let base = name_upper.trim_start_matches('C');
                        let _ = write!(output, "\\gate{{{}}}", base);
                    },
                }
            },
            GateRole::SwapFirst => {
                let other_idx = indices.last().unwrap();
                let offset = (*other_idx as isize) - (qubit as isize);
                let _ = write!(output, "\\swap{{{}}}", offset);
            },
            GateRole::SwapSecond => {
                output.push_str("\\targX{}");
            },
            GateRole::Multi(pos) => {
                // Multi-qubit gate spanning multiple qubits
                if pos == 0 {
                    let span = indices.len();
                    let gate_text = self.format_gate_text(op);
                    let _ = write!(output, "\\gate[{}]{{{}}}", span, gate_text);
                } else {
                    // Ghost entry for multi-span gate
                    output.push_str("\\ghost{}");
                }
            },
        }
    }

    fn write_single_gate(&self, output: &mut String, op: &GateOp) {
        let name = op.gate().name();
        let name_upper = name.to_uppercase();

        match name_upper.as_str() {
            "H" => output.push_str("\\gate{H}"),
            "X" => output.push_str("\\gate{X}"),
            "Y" => output.push_str("\\gate{Y}"),
            "Z" => output.push_str("\\gate{Z}"),
            "S" => output.push_str("\\gate{S}"),
            "S†" | "SDG" | "SDAGGER" => output.push_str("\\gate{S^\\dagger}"),
            "T" => output.push_str("\\gate{T}"),
            "T†" | "TDG" | "TDAGGER" => output.push_str("\\gate{T^\\dagger}"),
            "I" | "ID" | "IDENTITY" => output.push_str("\\qw"), // Identity is just wire
            "SX" => output.push_str("\\gate{\\sqrt{X}}"),
            "SX†" | "SXDG" => output.push_str("\\gate{\\sqrt{X}^\\dagger}"),
            _ => {
                // Parametric or custom gate
                let gate_text = self.format_gate_text(op);
                let _ = write!(output, "\\gate{{{}}}", gate_text);
            },
        }
    }

    fn format_gate_text(&self, op: &GateOp) -> String {
        let name = op.gate().name();
        let desc = op.gate().description();
        let default_desc = format!("{}-qubit gate '{}'", op.gate().num_qubits(), name);

        // Check if parametric
        if desc.contains('(') && desc.contains(')') {
            // Parse and format parameters
            self.format_parametric_gate(&desc)
        } else if desc != default_desc && desc != name {
            // Custom description
            escape_latex(&desc)
        } else {
            escape_latex(name)
        }
    }

    fn format_parametric_gate(&self, desc: &str) -> String {
        // Extract base name and parameters
        if let Some(paren_idx) = desc.find('(') {
            let base = &desc[..paren_idx];
            let params = &desc[paren_idx..];

            // Format common rotation gates nicely
            let base_upper = base.to_uppercase();
            let formatted_base = match base_upper.as_str() {
                "RX" => "R_x".to_string(),
                "RY" => "R_y".to_string(),
                "RZ" => "R_z".to_string(),
                "RXX" => "R_{xx}".to_string(),
                "RYY" => "R_{yy}".to_string(),
                "RZZ" => "R_{zz}".to_string(),
                "P" | "PHASE" => "P".to_string(),
                "U1" => "U_1".to_string(),
                "U2" => "U_2".to_string(),
                "U3" => "U_3".to_string(),
                _ => escape_latex(base),
            };

            // Format parameters - try to use nice fractions for common angles
            let formatted_params = self.format_parameters(params);

            format!("{}({})", formatted_base, formatted_params)
        } else {
            escape_latex(desc)
        }
    }

    fn format_parameters(&self, params: &str) -> String {
        // Remove outer parentheses
        let inner = params.trim_start_matches('(').trim_end_matches(')');

        // Try to format each parameter
        let formatted: Vec<String> = inner
            .split(',')
            .map(|p| self.format_single_param(p.trim()))
            .collect();

        formatted.join(", ")
    }

    fn format_single_param(&self, param: &str) -> String {
        // Try to parse as float
        if let Ok(val) = param.parse::<f64>() {
            // Check for common angles
            let pi = std::f64::consts::PI;

            // Common fractions of pi
            let fractions = [
                (pi, "\\pi"),
                (pi / 2.0, "\\pi/2"),
                (pi / 4.0, "\\pi/4"),
                (pi / 3.0, "\\pi/3"),
                (pi / 6.0, "\\pi/6"),
                (pi / 8.0, "\\pi/8"),
                (2.0 * pi, "2\\pi"),
                (3.0 * pi / 2.0, "3\\pi/2"),
                (-pi, "-\\pi"),
                (-pi / 2.0, "-\\pi/2"),
                (-pi / 4.0, "-\\pi/4"),
            ];

            for (angle, latex) in fractions {
                if (val - angle).abs() < 1e-6 {
                    return latex.to_string();
                }
            }

            // Format with configured precision
            format!("{:.prec$}", val, prec = self.config.float_precision)
        } else {
            escape_latex(param)
        }
    }
}

// ============================================================================
// Helper Types
// ============================================================================

#[derive(Debug, Clone, Copy)]
enum GateRole {
    Single,
    Control,
    Target,
    SwapFirst,
    SwapSecond,
    Multi(usize),
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Escape special LaTeX characters
fn escape_latex(s: &str) -> String {
    s.replace('\\', "\\textbackslash{}")
        .replace('{', "\\{")
        .replace('}', "\\}")
        .replace('%', "\\%")
        .replace('&', "\\&")
        .replace('#', "\\#")
        .replace('_', "\\_")
        .replace('^', "\\^{}")
        .replace('~', "\\~{}")
        .replace('$', "\\$")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::Gate;
    use crate::QubitId;
    use std::sync::Arc;

    #[derive(Debug)]
    struct MockGate {
        name: String,
        num_qubits: usize,
        description: Option<String>,
    }

    impl MockGate {
        fn new(name: &str, num_qubits: usize) -> Self {
            Self {
                name: name.to_string(),
                num_qubits,
                description: None,
            }
        }

        fn with_description(name: &str, num_qubits: usize, desc: &str) -> Self {
            Self {
                name: name.to_string(),
                num_qubits,
                description: Some(desc.to_string()),
            }
        }
    }

    impl Gate for MockGate {
        fn name(&self) -> &str {
            &self.name
        }

        fn num_qubits(&self) -> usize {
            self.num_qubits
        }

        fn description(&self) -> String {
            self.description
                .clone()
                .unwrap_or_else(|| format!("{}-qubit gate '{}'", self.num_qubits, self.name))
        }
    }

    #[test]
    fn test_empty_circuit() {
        let circuit = Circuit::new(2);
        let latex = render(&circuit);
        assert!(latex.contains("\\begin{quantikz}"));
        assert!(latex.contains("\\end{quantikz}"));
        assert!(latex.contains("\\ket{q_0}"));
        assert!(latex.contains("\\ket{q_1}"));
    }

    #[test]
    fn test_single_gate() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(MockGate::new("H", 1)), &[QubitId::new(0)])
            .unwrap();

        let latex = render(&circuit);
        assert!(latex.contains("\\gate{H}"));
    }

    #[test]
    fn test_cnot_gate() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(MockGate::new("CNOT", 2)), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();

        let latex = render(&circuit);
        assert!(latex.contains("\\ctrl{1}"));
        assert!(latex.contains("\\targ{}"));
    }

    #[test]
    fn test_swap_gate() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(MockGate::new("SWAP", 2)), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();

        let latex = render(&circuit);
        assert!(latex.contains("\\swap{1}"));
        assert!(latex.contains("\\targX{}"));
    }

    #[test]
    fn test_parametric_gate() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(
                Arc::new(MockGate::with_description(
                    "RX",
                    1,
                    &format!("RX({})", std::f64::consts::PI / 2.0),
                )),
                &[QubitId::new(0)],
            )
            .unwrap();

        let latex = render(&circuit);
        assert!(latex.contains("R_x"));
        assert!(latex.contains("\\pi/2"));
    }

    #[test]
    fn test_standalone_document() {
        let circuit = Circuit::new(1);
        let config = LatexConfig::standalone();
        let latex = render_with_config(&circuit, &config);

        assert!(latex.contains("\\documentclass"));
        assert!(latex.contains("\\usepackage{quantikz}"));
        assert!(latex.contains("\\begin{document}"));
        assert!(latex.contains("\\end{document}"));
    }

    #[test]
    fn test_no_labels() {
        let circuit = Circuit::new(2);
        let config = LatexConfig::default().with_labels(false);
        let latex = render_with_config(&circuit, &config);

        assert!(!latex.contains("\\lstick"));
    }

    #[test]
    fn test_escape_latex() {
        assert_eq!(escape_latex("a_b"), "a\\_b");
        assert_eq!(escape_latex("x^2"), "x\\^{}2");
        assert_eq!(escape_latex("50%"), "50\\%");
    }

    #[test]
    fn test_multi_qubit_gate() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(
                Arc::new(MockGate::with_description("XX", 2, "XX(0.785)")),
                &[QubitId::new(0), QubitId::new(1)],
            )
            .unwrap();

        let latex = render(&circuit);
        assert!(latex.contains("\\gate[2]"));
    }

    // ---- New tests for uncovered branches ----

    #[test]
    fn test_config_default_values() {
        let config = LatexConfig::default();
        assert!(!config.standalone);
        assert!(config.show_labels);
        assert!(config.ket_labels);
        assert_eq!(config.scale, 1.0);
        assert_eq!(config.row_sep, 0.5);
        assert_eq!(config.col_sep, 0.5);
        assert_eq!(config.float_precision, 4);
        assert!(!config.thin_lines);
        assert!(!config.show_slices);
        assert!(config.gate_styles.is_empty());
    }

    #[test]
    fn test_config_standalone_constructor() {
        let config = LatexConfig::standalone();
        assert!(config.standalone);
        assert!(config.show_labels);
    }

    #[test]
    fn test_config_with_standalone_builder() {
        let config = LatexConfig::default().with_standalone(true);
        assert!(config.standalone);
        let config2 = LatexConfig::default().with_standalone(false);
        assert!(!config2.standalone);
    }

    #[test]
    fn test_config_with_scale_builder() {
        let config = LatexConfig::default().with_scale(2.0);
        assert_eq!(config.scale, 2.0);
    }

    #[test]
    fn test_config_with_ket_labels_false() {
        let circuit = Circuit::new(2);
        let config = LatexConfig::default().with_ket_labels(false);
        let latex = render_with_config(&circuit, &config);
        // Should use q_0 style (not ket notation)
        assert!(latex.contains("\\lstick{q_0}"));
        assert!(!latex.contains("\\ket{q_0}"));
    }

    #[test]
    fn test_config_with_ket_labels_true() {
        let circuit = Circuit::new(1);
        let config = LatexConfig::default().with_ket_labels(true);
        let latex = render_with_config(&circuit, &config);
        assert!(latex.contains("\\ket{q_0}"));
    }

    #[test]
    fn test_config_with_gate_style() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(MockGate::new("MY_GATE", 1)), &[QubitId::new(0)])
            .unwrap();
        let config = LatexConfig::default().with_gate_style("MY_GATE", "\\gate{Custom}");
        let latex = render_with_config(&circuit, &config);
        assert!(latex.contains("\\gate{Custom}"));
    }

    #[test]
    fn test_scale_not_one_includes_scale_in_quantikz() {
        let circuit = Circuit::new(1);
        let config = LatexConfig::default().with_scale(1.5);
        let latex = render_with_config(&circuit, &config);
        assert!(latex.contains("scale=1.50"));
    }

    #[test]
    fn test_show_slices() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(MockGate::new("H", 1)), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(MockGate::new("X", 1)), &[QubitId::new(0)])
            .unwrap();
        let config = LatexConfig {
            show_slices: true,
            ..Default::default()
        };
        let latex = render_with_config(&circuit, &config);
        assert!(latex.contains("\\slice{}"));
    }

    #[test]
    fn test_x_y_z_s_t_gates() {
        let mut circuit = Circuit::new(1);
        for name in &["X", "Y", "Z", "S", "T"] {
            circuit
                .add_gate(Arc::new(MockGate::new(name, 1)), &[QubitId::new(0)])
                .unwrap();
        }
        let latex = render(&circuit);
        assert!(latex.contains("\\gate{X}"));
        assert!(latex.contains("\\gate{Y}"));
        assert!(latex.contains("\\gate{Z}"));
        assert!(latex.contains("\\gate{S}"));
        assert!(latex.contains("\\gate{T}"));
    }

    #[test]
    fn test_sdagger_gate() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(MockGate::new("SDG", 1)), &[QubitId::new(0)])
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("S^\\dagger"));
    }

    #[test]
    fn test_tdagger_gate() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(MockGate::new("TDG", 1)), &[QubitId::new(0)])
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("T^\\dagger"));
    }

    #[test]
    fn test_identity_gate_renders_as_wire() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(MockGate::new("I", 1)), &[QubitId::new(0)])
            .unwrap();
        let latex = render(&circuit);
        // Identity should render as \\qw
        assert!(latex.contains("\\qw"));
    }

    #[test]
    fn test_sx_gate() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(MockGate::new("SX", 1)), &[QubitId::new(0)])
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("\\sqrt{X}"));
    }

    #[test]
    fn test_sxdg_gate() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(MockGate::new("SXDG", 1)), &[QubitId::new(0)])
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("\\sqrt{X}^\\dagger"));
    }

    #[test]
    fn test_cx_gate_target_is_targ() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(MockGate::new("CX", 2)), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("\\targ{}"));
        assert!(latex.contains("\\ctrl{1}"));
    }

    #[test]
    fn test_ccx_toffoli_gate() {
        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(
                Arc::new(MockGate::new("CCNOT", 3)),
                &[QubitId::new(0), QubitId::new(1), QubitId::new(2)],
            )
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("\\targ{}"));
        // Two control qubits
        assert!(latex.contains("\\ctrl{"));
    }

    #[test]
    fn test_cz_gate_target_shows_ctrl_back() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(MockGate::new("CZ", 2)), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        let latex = render(&circuit);
        // CZ shows ctrl on both qubits
        assert!(latex.contains("\\ctrl{1}"));
        assert!(latex.contains("\\ctrl{-1}"));
    }

    #[test]
    fn test_cy_gate_target_shows_y() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(MockGate::new("CY", 2)), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("\\gate{Y}"));
    }

    #[test]
    fn test_generic_controlled_gate_shows_base() {
        // A gate starting with C but not a specific known one - shows base gate name
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(MockGate::new("CPHASE", 2)), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        let latex = render(&circuit);
        // Target qubit should show the base gate name (PHASE)
        assert!(latex.contains("\\gate{PHASE}"));
    }

    #[test]
    fn test_cswap_fredkin_gate() {
        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(
                Arc::new(MockGate::new("CSWAP", 3)),
                &[QubitId::new(0), QubitId::new(1), QubitId::new(2)],
            )
            .unwrap();
        let latex = render(&circuit);
        // Control qubit
        assert!(latex.contains("\\ctrl{"));
        // Swap qubits
        assert!(latex.contains("\\swap{"));
        assert!(latex.contains("\\targX{}"));
    }

    #[test]
    fn test_iswap_gate() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(MockGate::new("ISWAP", 2)), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("\\swap{1}"));
        assert!(latex.contains("\\targX{}"));
    }

    #[test]
    fn test_parametric_ry_gate() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(
                Arc::new(MockGate::with_description(
                    "RY",
                    1,
                    &format!("RY({})", std::f64::consts::PI),
                )),
                &[QubitId::new(0)],
            )
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("R_y"));
        assert!(latex.contains("\\pi"));
    }

    #[test]
    fn test_parametric_rz_gate() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(
                Arc::new(MockGate::with_description(
                    "RZ",
                    1,
                    &format!("RZ({})", std::f64::consts::PI / 4.0),
                )),
                &[QubitId::new(0)],
            )
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("R_z"));
        assert!(latex.contains("\\pi/4"));
    }

    #[test]
    fn test_parametric_phase_gate() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(
                Arc::new(MockGate::with_description(
                    "PHASE",
                    1,
                    &format!("PHASE({})", std::f64::consts::PI / 2.0),
                )),
                &[QubitId::new(0)],
            )
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("P("));
    }

    #[test]
    fn test_parametric_u1_u2_u3_gates() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(MockGate::with_description("U1", 1, "U1(0.5)")), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(
                Arc::new(MockGate::with_description("U2", 1, "U2(0.5, 1.0)")),
                &[QubitId::new(0)],
            )
            .unwrap();
        circuit
            .add_gate(
                Arc::new(MockGate::with_description("U3", 1, "U3(0.5, 1.0, 1.5)")),
                &[QubitId::new(0)],
            )
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("U_1"));
        assert!(latex.contains("U_2"));
        assert!(latex.contains("U_3"));
    }

    #[test]
    fn test_parametric_rxx_ryy_rzz_gates() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(
                Arc::new(MockGate::with_description("RXX", 2, "RXX(0.5)")),
                &[QubitId::new(0), QubitId::new(1)],
            )
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("R_{xx}"));

        let mut circuit2 = Circuit::new(2);
        circuit2
            .add_gate(
                Arc::new(MockGate::with_description("RYY", 2, "RYY(0.5)")),
                &[QubitId::new(0), QubitId::new(1)],
            )
            .unwrap();
        let latex2 = render(&circuit2);
        assert!(latex2.contains("R_{yy}"));

        let mut circuit3 = Circuit::new(2);
        circuit3
            .add_gate(
                Arc::new(MockGate::with_description("RZZ", 2, "RZZ(0.5)")),
                &[QubitId::new(0), QubitId::new(1)],
            )
            .unwrap();
        let latex3 = render(&circuit3);
        assert!(latex3.contains("R_{zz}"));
    }

    #[test]
    fn test_parametric_common_angles() {
        // Test all the common angle fractions
        let angles = [
            (std::f64::consts::PI, "\\pi"),
            (std::f64::consts::PI / 2.0, "\\pi/2"),
            (std::f64::consts::PI / 4.0, "\\pi/4"),
            (std::f64::consts::PI / 3.0, "\\pi/3"),
            (std::f64::consts::PI / 6.0, "\\pi/6"),
            (std::f64::consts::PI / 8.0, "\\pi/8"),
            (2.0 * std::f64::consts::PI, "2\\pi"),
            (3.0 * std::f64::consts::PI / 2.0, "3\\pi/2"),
            (-std::f64::consts::PI, "-\\pi"),
            (-std::f64::consts::PI / 2.0, "-\\pi/2"),
            (-std::f64::consts::PI / 4.0, "-\\pi/4"),
        ];
        for (angle, expected_latex) in &angles {
            let mut circuit = Circuit::new(1);
            circuit
                .add_gate(
                    Arc::new(MockGate::with_description("RX", 1, &format!("RX({})", angle))),
                    &[QubitId::new(0)],
                )
                .unwrap();
            let latex = render(&circuit);
            assert!(
                latex.contains(expected_latex),
                "Expected {} for angle {}, got:\n{}",
                expected_latex,
                angle,
                latex
            );
        }
    }

    #[test]
    fn test_parametric_non_standard_angle() {
        // A non-standard angle should be formatted with float_precision digits
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(
                Arc::new(MockGate::with_description("RX", 1, "RX(1.23456789)")),
                &[QubitId::new(0)],
            )
            .unwrap();
        let latex = render(&circuit);
        // Should contain a decimal representation (not a pi fraction)
        assert!(latex.contains("R_x"));
        assert!(latex.contains("1.2346")); // default float_precision=4
    }

    #[test]
    fn test_custom_description_gate() {
        // Gate with a custom description (not matching default "N-qubit gate 'name'" pattern)
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(
                Arc::new(MockGate::with_description("MGATE", 1, "My Custom Gate")),
                &[QubitId::new(0)],
            )
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("My Custom Gate"));
    }

    #[test]
    fn test_escape_latex_all_chars() {
        // Backslash gets replaced with \textbackslash{}, but then the {} get escaped too
        // So a\b -> a\textbackslash{}b -> a\textbackslash\{\}b
        assert!(escape_latex("a\\b").contains("textbackslash"));
        // Curly braces are escaped
        assert_eq!(escape_latex("a{b}"), "a\\{b\\}");
        assert_eq!(escape_latex("a&b"), "a\\&b");
        assert_eq!(escape_latex("a#b"), "a\\#b");
        assert_eq!(escape_latex("a~b"), "a\\~{}b");
        assert_eq!(escape_latex("a$b"), "a\\$b");
    }

    #[test]
    fn test_sequential_gates_same_qubit() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(MockGate::new("H", 1)), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(MockGate::new("X", 1)), &[QubitId::new(0)])
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("\\gate{H}"));
        assert!(latex.contains("\\gate{X}"));
    }

    #[test]
    fn test_multi_qubit_gate_ghost_entry() {
        // A 3-qubit multi gate: position 0 gets gate[3], positions 1 and 2 get ghost
        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(
                Arc::new(MockGate::with_description("BIGGATE", 3, "BigGate(0.5)")),
                &[QubitId::new(0), QubitId::new(1), QubitId::new(2)],
            )
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("\\gate[3]"));
        assert!(latex.contains("\\ghost{}"));
    }

    #[test]
    fn test_row_sep_col_sep_in_output() {
        let circuit = Circuit::new(1);
        let config = LatexConfig {
            row_sep: 0.75,
            col_sep: 0.25,
            ..Default::default()
        };
        let latex = render_with_config(&circuit, &config);
        assert!(latex.contains("row sep={0.75cm}"));
        assert!(latex.contains("column sep={0.25cm}"));
    }

    #[test]
    fn test_cnot_upper_case_variations() {
        // Test CCX and TOFFOLI also produce \\targ{}
        for name in &["CCX", "TOFFOLI"] {
            let mut circuit = Circuit::new(3);
            circuit
                .add_gate(
                    Arc::new(MockGate::new(name, 3)),
                    &[QubitId::new(0), QubitId::new(1), QubitId::new(2)],
                )
                .unwrap();
            let latex = render(&circuit);
            assert!(latex.contains("\\targ{}"), "Expected \\targ{{}} for {}", name);
        }
    }

    #[test]
    fn test_fredkin_cswap_variant() {
        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(
                Arc::new(MockGate::new("FREDKIN", 3)),
                &[QubitId::new(0), QubitId::new(1), QubitId::new(2)],
            )
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("\\ctrl{"));
        assert!(latex.contains("\\swap{"));
        assert!(latex.contains("\\targX{}"));
    }

    #[test]
    fn test_two_qubit_circuit_parallel_gates() {
        // Gates on different qubits can appear in same column
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(MockGate::new("H", 1)), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(MockGate::new("X", 1)), &[QubitId::new(1)])
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("\\gate{H}"));
        assert!(latex.contains("\\gate{X}"));
    }

    #[test]
    fn test_format_parametric_gate_unknown_base() {
        // A gate with params but unknown base name gets escaped
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(
                Arc::new(MockGate::with_description("MYROT", 1, "MYROT(0.5)")),
                &[QubitId::new(0)],
            )
            .unwrap();
        let latex = render(&circuit);
        // Should still have the gate rendered (escaped base name)
        assert!(latex.contains("MYROT"));
    }

    // ---- Additional coverage tests for private helpers / edge branches ----

    #[test]
    fn test_is_wire_through_true_for_intermediate_qubit() {
        // Covers is_wire_through returning true (lines 316-321): a qubit strictly
        // between the min and max of a multi-qubit gate's indices, and not itself
        // one of the gate's qubits, counts as a "wire passing through".
        let circuit = Circuit::new(4);
        let config = LatexConfig::default();
        let renderer = LatexRenderer::new(&circuit, &config);

        let op =
            GateOp::new(Arc::new(MockGate::new("CNOT", 2)), &[QubitId::new(0), QubitId::new(3)])
                .unwrap();
        let col: Vec<(usize, &GateOp)> = vec![(0, &op)];

        assert!(renderer.is_wire_through(&col, 1));
        assert!(renderer.is_wire_through(&col, 2));
    }

    #[test]
    fn test_is_wire_through_false_when_not_intermediate() {
        // Covers is_wire_through returning false (line 325): either the qubit is
        // one of the gate's own qubits, or it's outside the [min, max] span, or
        // the column has no gates at all.
        let circuit = Circuit::new(4);
        let config = LatexConfig::default();
        let renderer = LatexRenderer::new(&circuit, &config);

        let op =
            GateOp::new(Arc::new(MockGate::new("CNOT", 2)), &[QubitId::new(0), QubitId::new(3)])
                .unwrap();
        let col: Vec<(usize, &GateOp)> = vec![(0, &op)];

        // qubit 0 and 3 are the gate's own qubits, not "passing through"
        assert!(!renderer.is_wire_through(&col, 0));
        assert!(!renderer.is_wire_through(&col, 3));

        // empty column: loop body never executes, falls through to `false`
        let empty_col: Vec<(usize, &GateOp)> = Vec::new();
        assert!(!renderer.is_wire_through(&empty_col, 1));
    }

    #[test]
    fn test_format_gate_text_falls_back_to_escaped_name() {
        // Covers the final `else` branch of format_gate_text (line 468):
        // a single-qubit gate whose description matches the auto-generated
        // default description (no parens, and desc == default_desc), and whose
        // name isn't one of the specially-handled cases in write_single_gate,
        // so write_single_gate's `_` arm calls format_gate_text -> escape_latex(name).
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(MockGate::new("ORACLE", 1)), &[QubitId::new(0)])
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("\\gate{ORACLE}"));
    }

    #[test]
    fn test_format_parametric_gate_no_paren_directly() {
        // Covers the `else` branch of format_parametric_gate (line 499): reached
        // only when the description has no '(' at all. format_gate_text itself
        // only calls format_parametric_gate when desc contains both '(' and ')',
        // so this branch is unreachable via the public rendering path; we call
        // the private helper directly to exercise it.
        let circuit = Circuit::new(1);
        let config = LatexConfig::default();
        let renderer = LatexRenderer::new(&circuit, &config);

        let result = renderer.format_parametric_gate("NoParensHere");
        assert_eq!(result, escape_latex("NoParensHere"));
    }

    #[test]
    fn test_format_single_param_non_numeric_is_escaped() {
        // Covers the `else` branch of format_single_param (line 546): a
        // parameter that fails to parse as f64 gets passed through escape_latex.
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(
                Arc::new(MockGate::with_description("RX", 1, "RX(theta)")),
                &[QubitId::new(0)],
            )
            .unwrap();
        let latex = render(&circuit);
        assert!(latex.contains("theta"));
    }
}
