//! ASCII circuit renderer for terminal visualization
//!
//! Renders quantum circuits as ASCII art, adapting to terminal width.

use crate::{Circuit, GateOp};
use std::collections::HashMap;

/// Configuration for ASCII rendering
#[derive(Debug, Clone)]
pub struct AsciiConfig {
    /// Maximum width (0 = auto-detect terminal width)
    pub max_width: usize,
    /// Minimum gate box width
    pub min_gate_width: usize,
    /// Show qubit labels
    pub show_labels: bool,
    /// Compact mode for narrow terminals
    pub compact: bool,
}

impl Default for AsciiConfig {
    fn default() -> Self {
        Self {
            max_width: 0, // Auto-detect
            min_gate_width: 3,
            show_labels: true,
            compact: false,
        }
    }
}

impl AsciiConfig {
    /// Get effective terminal width
    fn effective_width(&self) -> usize {
        if self.max_width > 0 {
            self.max_width
        } else {
            terminal_width().unwrap_or(80)
        }
    }
}

/// Get terminal width (cross-platform)
fn terminal_width() -> Option<usize> {
    // Try environment variable first
    if let Ok(cols) = std::env::var("COLUMNS") {
        if let Ok(w) = cols.parse::<usize>() {
            return Some(w);
        }
    }

    // Try termsize if available, otherwise default
    #[cfg(unix)]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("tput").arg("cols").output() {
            if let Ok(s) = String::from_utf8(output.stdout) {
                if let Ok(w) = s.trim().parse::<usize>() {
                    return Some(w);
                }
            }
        }
    }

    None
}

/// Render a circuit to ASCII art
pub fn render(circuit: &Circuit) -> String {
    render_with_config(circuit, &AsciiConfig::default())
}

/// Render with custom configuration
pub fn render_with_config(circuit: &Circuit, config: &AsciiConfig) -> String {
    let renderer = AsciiRenderer::new(circuit, config);
    renderer.render()
}

struct AsciiRenderer<'a> {
    circuit: &'a Circuit,
    config: &'a AsciiConfig,
    width: usize,
    label_width: usize,
}

impl<'a> AsciiRenderer<'a> {
    fn new(circuit: &'a Circuit, config: &'a AsciiConfig) -> Self {
        let width = config.effective_width();
        let label_width = if config.show_labels {
            // "q0: " format, accommodate larger qubit numbers
            format!("q{}: ", circuit.num_qubits().saturating_sub(1)).len()
        } else {
            0
        };

        Self {
            circuit,
            config,
            width,
            label_width,
        }
    }

    fn render(&self) -> String {
        if self.circuit.is_empty() {
            return self.render_empty();
        }

        // Build time slices (columns) - gates that can be parallel
        let columns = self.build_columns();

        // Calculate column widths
        let col_widths = self.calculate_column_widths(&columns);

        // Render the circuit
        self.render_circuit(&columns, &col_widths)
    }

    fn render_empty(&self) -> String {
        let mut result = String::new();
        let wire_width = self.width.saturating_sub(self.label_width + 1).min(40);

        for q in 0..self.circuit.num_qubits() {
            if self.config.show_labels {
                result.push_str(&format!("q{}: ", q));
            }
            result.push_str(&"─".repeat(wire_width));
            result.push('\n');
        }
        result
    }

    /// Group operations into columns based on qubit dependencies
    fn build_columns(&self) -> Vec<Vec<(usize, &GateOp)>> {
        let mut columns: Vec<Vec<(usize, &GateOp)>> = Vec::new();
        let mut qubit_column: HashMap<usize, usize> = HashMap::new();

        for (op_idx, op) in self.circuit.operations().enumerate() {
            // Find the earliest column this gate can go in
            let mut target_col = 0;
            for &qubit in op.qubits() {
                let q = qubit.index();
                if let Some(&col) = qubit_column.get(&q) {
                    target_col = target_col.max(col + 1);
                }
            }

            // Also check for multi-qubit gates that span across qubits
            if op.qubits().len() > 1 {
                let min_q = op.qubits().iter().map(|q| q.index()).min().unwrap();
                let max_q = op.qubits().iter().map(|q| q.index()).max().unwrap();
                // Check if any qubit in between has a gate at target_col
                for q in min_q..=max_q {
                    if let Some(&col) = qubit_column.get(&q) {
                        target_col = target_col.max(col + 1);
                    }
                }
            }

            // Ensure column exists
            while columns.len() <= target_col {
                columns.push(Vec::new());
            }

            // Add gate to column
            columns[target_col].push((op_idx, op));

            // Update qubit positions
            for &qubit in op.qubits() {
                qubit_column.insert(qubit.index(), target_col);
            }

            // For multi-qubit gates, mark all qubits in range
            if op.qubits().len() > 1 {
                let min_q = op.qubits().iter().map(|q| q.index()).min().unwrap();
                let max_q = op.qubits().iter().map(|q| q.index()).max().unwrap();
                for q in min_q..=max_q {
                    qubit_column.insert(q, target_col);
                }
            }
        }

        columns
    }

    /// Calculate width for each column
    fn calculate_column_widths(&self, columns: &[Vec<(usize, &GateOp)>]) -> Vec<usize> {
        let available = self.width.saturating_sub(self.label_width + 4); // +4 for trailing wire
        let num_cols = columns.len();

        // Calculate natural widths (minimum needed for each gate)
        let natural_widths: Vec<usize> = columns
            .iter()
            .map(|col| {
                col.iter()
                    .map(|(_, op)| self.gate_display_width(op))
                    .max()
                    .unwrap_or(self.config.min_gate_width)
            })
            .collect();

        let total_natural: usize = natural_widths.iter().sum();
        let separators = num_cols.saturating_sub(1);

        if total_natural + separators <= available {
            // Everything fits - use natural widths
            natural_widths
        } else {
            // Need to compress - distribute available space
            let usable = available.saturating_sub(separators);
            let per_col = usable / num_cols.max(1);
            let min_w = self.config.min_gate_width;

            // Try to fit gates, truncating if necessary
            natural_widths
                .iter()
                .map(|&w| {
                    if w <= per_col {
                        w
                    } else {
                        per_col.max(min_w)
                    }
                })
                .collect()
        }
    }

    fn gate_display_width(&self, op: &GateOp) -> usize {
        let name = self.gate_symbol(op);
        // Add 2 for box borders [ ]
        name.len() + 2
    }

    fn gate_symbol(&self, op: &GateOp) -> String {
        let name = op.gate().name();
        let desc = op.gate().description();

        // Check if description has useful info (parameters or custom details)
        let default_desc = format!("{}-qubit gate '{}'", op.gate().num_qubits(), name);
        let has_params = desc.contains('(') && desc.contains(')');
        let is_custom_desc = desc != default_desc && desc != name;

        if has_params {
            // Parametric gate - use description with params
            if desc.len() > 12 && self.config.compact {
                // Truncate long params in compact mode
                if let Some(paren_idx) = desc.find('(') {
                    let base = &desc[..paren_idx];
                    format!("{}(..)", base)
                } else {
                    format!("{}(..)", name)
                }
            } else {
                desc.to_string()
            }
        } else if is_custom_desc {
            // Custom gate with custom description - show description
            if desc.len() > 12 && self.config.compact {
                format!("{}...", &desc[..9])
            } else {
                desc.to_string()
            }
        } else {
            // Simple gate - just use name
            name.to_string()
        }
    }

    fn render_circuit(
        &self,
        columns: &[Vec<(usize, &GateOp)>],
        col_widths: &[usize],
    ) -> String {
        let num_qubits = self.circuit.num_qubits();
        let mut lines: Vec<String> = vec![String::new(); num_qubits];

        // Add labels
        if self.config.show_labels {
            for (q, line) in lines.iter_mut().enumerate() {
                let label = format!("q{}: ", q);
                let padded = format!("{:>width$}", label, width = self.label_width);
                line.push_str(&padded);
            }
        }

        // Render each column
        for (col_idx, col) in columns.iter().enumerate() {
            let width = col_widths[col_idx];
            self.render_column(&mut lines, col, width);

            // Add wire between columns (except last)
            if col_idx < columns.len() - 1 {
                for line in lines.iter_mut() {
                    line.push('─');
                }
            }
        }

        // Add trailing wire
        for line in &mut lines {
            line.push_str("──");
        }

        lines.join("\n")
    }

    fn render_column(&self, lines: &mut [String], col: &[(usize, &GateOp)], width: usize) {
        let num_qubits = self.circuit.num_qubits();

        // Track which qubits have gates in this column
        let mut qubit_gate: HashMap<usize, &GateOp> = HashMap::new();
        let mut qubit_role: HashMap<usize, QubitRole> = HashMap::new();

        for (_, op) in col {
            let qubits = op.qubits();
            if qubits.len() == 1 {
                let q = qubits[0].index();
                qubit_gate.insert(q, *op);
                qubit_role.insert(q, QubitRole::Single);
            } else if qubits.len() >= 2 {
                // Multi-qubit gate
                let indices: Vec<usize> = qubits.iter().map(|q| q.index()).collect();
                let min_q = *indices.iter().min().unwrap();
                let max_q = *indices.iter().max().unwrap();

                let name = op.gate().name().to_uppercase();
                let is_controlled = name.starts_with('C') || name == "CNOT" || name == "CCNOT" || name == "TOFFOLI";
                let is_swap = name == "SWAP" || name == "ISWAP";
                let is_cswap = name == "CSWAP" || name == "FREDKIN";

                for (i, &q) in indices.iter().enumerate() {
                    qubit_gate.insert(q, *op);
                    if is_swap {
                        // SWAP: both qubits are targets (×)
                        qubit_role.insert(q, QubitRole::Target);
                    } else if is_cswap {
                        // CSWAP/Fredkin: first is control, rest are targets
                        if i == 0 {
                            qubit_role.insert(q, QubitRole::Control);
                        } else {
                            qubit_role.insert(q, QubitRole::Target);
                        }
                    } else if is_controlled && i < indices.len() - 1 {
                        qubit_role.insert(q, QubitRole::Control);
                    } else if is_controlled {
                        qubit_role.insert(q, QubitRole::Target);
                    } else {
                        qubit_role.insert(q, QubitRole::Multi);
                    }
                }

                // Mark intermediate qubits as wires
                for q in (min_q + 1)..max_q {
                    if !indices.contains(&q) {
                        qubit_role.insert(q, QubitRole::Wire);
                    }
                }
            }
        }

        // Render each qubit line
        for q in 0..num_qubits {
            let line = &mut lines[q];

            match qubit_role.get(&q) {
                Some(QubitRole::Single) => {
                    let op = qubit_gate[&q];
                    let sym = self.gate_symbol(op);
                    let boxed = format!("[{}]", sym);
                    let padded = center_str(&boxed, width);
                    line.push_str(&padded);
                }
                Some(QubitRole::Control) => {
                    let ctrl = center_str("●", width);
                    line.push_str(&ctrl);
                }
                Some(QubitRole::Target) => {
                    let op = qubit_gate[&q];
                    let name = op.gate().name().to_uppercase();
                    let sym = if name == "CNOT" || name == "CX" {
                        "⊕".to_string()
                    } else if name == "CCNOT" || name == "CCX" || name == "TOFFOLI" {
                        "⊕".to_string()
                    } else if name == "CZ" {
                        "●".to_string()
                    } else if name == "CY" {
                        "[Y]".to_string()
                    } else if name == "SWAP" || name == "ISWAP" {
                        "×".to_string()
                    } else if name == "CSWAP" || name == "FREDKIN" {
                        "×".to_string()
                    } else {
                        // For other controlled gates, show the base gate
                        let base = name.trim_start_matches('C');
                        format!("[{}]", base)
                    };
                    line.push_str(&center_str(&sym, width));
                }
                Some(QubitRole::Multi) => {
                    let op = qubit_gate[&q];
                    let sym = self.gate_symbol(op);
                    let boxed = format!("[{}]", sym);
                    line.push_str(&center_str(&boxed, width));
                }
                Some(QubitRole::Wire) => {
                    // Vertical wire for multi-qubit gates
                    line.push_str(&center_str("│", width));
                }
                None => {
                    // Just wire
                    line.push_str(&"─".repeat(width));
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum QubitRole {
    Single,
    Control,
    Target,
    Multi,
    Wire,
}

/// Center a string within a given width, padding with wire characters
fn center_str(s: &str, width: usize) -> String {
    let s_width = unicode_width(s);
    if s_width >= width {
        return s.to_string();
    }

    let total_pad = width - s_width;
    let left_pad = total_pad / 2;
    let right_pad = total_pad - left_pad;

    format!("{}{}{}", "─".repeat(left_pad), s, "─".repeat(right_pad))
}

/// Get display width of string (accounting for unicode)
fn unicode_width(s: &str) -> usize {
    // Simple approximation: count chars, unicode symbols often width 1-2
    s.chars()
        .map(|c| {
            if c.is_ascii() {
                1
            } else {
                // Most unicode box drawing and symbols are width 1
                1
            }
        })
        .sum()
}

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
    }

    impl Gate for MockGate {
        fn name(&self) -> &str {
            &self.name
        }
        fn num_qubits(&self) -> usize {
            self.num_qubits
        }
    }

    #[test]
    fn test_empty_circuit() {
        let circuit = Circuit::new(2);
        let ascii = render(&circuit);
        assert!(ascii.contains("q0:"));
        assert!(ascii.contains("q1:"));
        assert!(ascii.contains("─"));
    }

    #[test]
    fn test_single_gate() {
        let mut circuit = Circuit::new(2);
        let h = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });
        circuit.add_gate(h, &[QubitId::new(0)]).unwrap();

        let ascii = render(&circuit);
        assert!(ascii.contains("[H]"));
    }

    #[test]
    fn test_custom_width() {
        let circuit = Circuit::new(2);
        let config = AsciiConfig {
            max_width: 40,
            ..Default::default()
        };
        let ascii = render_with_config(&circuit, &config);
        // Verify output is generated and has expected structure
        assert!(ascii.contains("q0:"));
        assert!(ascii.contains("q1:"));
        // Each line should have reasonable visual width (unicode chars take more bytes)
        for line in ascii.lines() {
            assert!(line.chars().count() <= 50);
        }
    }

    #[test]
    fn test_custom_gate_with_description() {
        // Custom gate with custom description
        #[derive(Debug)]
        struct MyOracle;
        impl Gate for MyOracle {
            fn name(&self) -> &str { "Oracle" }
            fn num_qubits(&self) -> usize { 1 }
            fn description(&self) -> String { "Grover Oracle".to_string() }
        }

        let mut circuit = Circuit::new(2);
        circuit.add_gate(Arc::new(MyOracle), &[QubitId::new(0)]).unwrap();
        let ascii = render(&circuit);
        assert!(ascii.contains("Grover Oracle"));
    }

    #[test]
    fn test_parametric_gate() {
        #[derive(Debug)]
        struct ParamGate(f64, f64);
        impl Gate for ParamGate {
            fn name(&self) -> &str { "U" }
            fn num_qubits(&self) -> usize { 1 }
            fn description(&self) -> String { format!("U({:.2}, {:.2})", self.0, self.1) }
        }

        let mut circuit = Circuit::new(1);
        circuit.add_gate(Arc::new(ParamGate(1.57, 3.14)), &[QubitId::new(0)]).unwrap();
        let ascii = render(&circuit);
        assert!(ascii.contains("U(1.57, 3.14)"));
    }

    #[test]
    fn test_multi_qubit_custom_gate() {
        #[derive(Debug)]
        struct CustomTwoQubit;
        impl Gate for CustomTwoQubit {
            fn name(&self) -> &str { "XX" }
            fn num_qubits(&self) -> usize { 2 }
            fn description(&self) -> String { "XX(π/2)".to_string() }
        }

        let mut circuit = Circuit::new(2);
        circuit.add_gate(Arc::new(CustomTwoQubit), &[QubitId::new(0), QubitId::new(1)]).unwrap();
        let ascii = render(&circuit);
        // Multi-qubit non-controlled gates show [description] on each qubit
        assert!(ascii.contains("[XX(π/2)]"));
    }
}
