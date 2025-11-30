//! Production-grade ASCII circuit renderer for terminal visualization
//!
//! This module provides a comprehensive ASCII art renderer for quantum circuits
//! with support for all standard gates, custom gates, terminal width adaptation,
//! and multiple rendering styles.
//!
//! # Features
//!
//! - **Automatic terminal width detection** with cross-platform support
//! - **Smart gate placement** using dependency analysis for optimal layout
//! - **Full gate coverage** including single, two, and three-qubit gates
//! - **Custom gate support** with parametric display and descriptions
//! - **Multiple output styles** (Unicode box drawing, ASCII-only, compact)
//! - **Configurable appearance** (labels, wire style, gate padding)
//!
//! # Example
//!
//! ```ignore
//! use simq_core::{Circuit, AsciiConfig, render_ascii};
//!
//! let circuit = build_circuit();
//!
//! // Default rendering
//! println!("{}", circuit.to_ascii());
//!
//! // Custom configuration
//! let config = AsciiConfig::builder()
//!     .max_width(80)
//!     .style(RenderStyle::Unicode)
//!     .show_labels(true)
//!     .build();
//! println!("{}", circuit.to_ascii_with_config(&config));
//! ```

use crate::{Circuit, GateOp};
use std::collections::HashMap;
use std::fmt;

// ============================================================================
// Configuration Types
// ============================================================================

/// Rendering style for the circuit
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderStyle {
    /// Unicode box-drawing characters (default, best appearance)
    #[default]
    Unicode,
    /// ASCII-only characters for maximum compatibility
    Ascii,
    /// Minimal style with reduced spacing
    Compact,
}

/// Wire style for qubit lines
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WireStyle {
    /// Single horizontal line (─)
    #[default]
    Single,
    /// Double horizontal line (═)
    Double,
    /// Dashed line (╌)
    Dashed,
}

/// Configuration for ASCII rendering
///
/// Use `AsciiConfig::default()` for sensible defaults or `AsciiConfig::builder()`
/// for fine-grained control.
#[derive(Debug, Clone)]
pub struct AsciiConfig {
    /// Maximum width in characters (0 = auto-detect terminal width)
    pub max_width: usize,
    /// Minimum gate box width
    pub min_gate_width: usize,
    /// Show qubit labels (q0:, q1:, ...)
    pub show_labels: bool,
    /// Compact mode for narrow terminals (truncates long gate names)
    pub compact: bool,
    /// Rendering style
    pub style: RenderStyle,
    /// Wire style
    pub wire_style: WireStyle,
    /// Show measurement symbols distinctly
    pub show_measurements: bool,
    /// Precision for floating-point parameters
    pub float_precision: usize,
    /// Maximum parameter string length before truncation
    pub max_param_length: usize,
    /// Vertical spacing between qubit wires (1 = no extra space)
    pub vertical_spacing: usize,
    /// Show circuit depth annotation
    pub show_depth: bool,
    /// Custom label format (None = default "q{n}: ")
    pub label_format: Option<String>,
}

impl Default for AsciiConfig {
    fn default() -> Self {
        Self {
            max_width: 0,
            min_gate_width: 3,
            show_labels: true,
            compact: false,
            style: RenderStyle::Unicode,
            wire_style: WireStyle::Single,
            show_measurements: true,
            float_precision: 4,
            max_param_length: 20,
            vertical_spacing: 1,
            show_depth: false,
            label_format: None,
        }
    }
}

impl AsciiConfig {
    /// Create a new builder for AsciiConfig
    pub fn builder() -> AsciiConfigBuilder {
        AsciiConfigBuilder::default()
    }

    /// Create a compact configuration suitable for narrow terminals
    pub fn compact() -> Self {
        Self {
            compact: true,
            max_param_length: 8,
            min_gate_width: 3,
            ..Default::default()
        }
    }

    /// Create an ASCII-only configuration for maximum compatibility
    pub fn ascii_only() -> Self {
        Self {
            style: RenderStyle::Ascii,
            ..Default::default()
        }
    }

    /// Get effective terminal width
    fn effective_width(&self) -> usize {
        if self.max_width > 0 {
            self.max_width
        } else {
            terminal_width().unwrap_or(80)
        }
    }

    /// Get wire character based on style
    fn wire_char(&self) -> char {
        match (self.style, self.wire_style) {
            (RenderStyle::Ascii, _) => '-',
            (_, WireStyle::Single) => '─',
            (_, WireStyle::Double) => '═',
            (_, WireStyle::Dashed) => '╌',
        }
    }

    /// Get vertical wire character
    #[allow(dead_code)]
    fn vertical_wire(&self) -> char {
        match self.style {
            RenderStyle::Ascii => '|',
            _ => '│',
        }
    }
}

/// Builder for AsciiConfig with fluent API
#[derive(Debug, Clone, Default)]
pub struct AsciiConfigBuilder {
    config: AsciiConfig,
}

impl AsciiConfigBuilder {
    pub fn max_width(mut self, width: usize) -> Self {
        self.config.max_width = width;
        self
    }

    pub fn min_gate_width(mut self, width: usize) -> Self {
        self.config.min_gate_width = width;
        self
    }

    pub fn show_labels(mut self, show: bool) -> Self {
        self.config.show_labels = show;
        self
    }

    pub fn compact(mut self, compact: bool) -> Self {
        self.config.compact = compact;
        self
    }

    pub fn style(mut self, style: RenderStyle) -> Self {
        self.config.style = style;
        self
    }

    pub fn wire_style(mut self, style: WireStyle) -> Self {
        self.config.wire_style = style;
        self
    }

    pub fn float_precision(mut self, precision: usize) -> Self {
        self.config.float_precision = precision;
        self
    }

    pub fn max_param_length(mut self, length: usize) -> Self {
        self.config.max_param_length = length;
        self
    }

    pub fn show_depth(mut self, show: bool) -> Self {
        self.config.show_depth = show;
        self
    }

    pub fn label_format(mut self, format: impl Into<String>) -> Self {
        self.config.label_format = Some(format.into());
        self
    }

    pub fn build(self) -> AsciiConfig {
        self.config
    }
}

// ============================================================================
// Symbol Definitions
// ============================================================================

/// Gate symbols used in rendering
#[allow(dead_code)]
struct GateSymbols {
    control: &'static str,
    target_x: &'static str,
    swap: &'static str,
    measure: &'static str,
    barrier: &'static str,
    wire_h: char,
    wire_v: char,
    corner_tl: char,
    corner_tr: char,
    corner_bl: char,
    corner_br: char,
    bracket_l: char,
    bracket_r: char,
}

impl GateSymbols {
    fn unicode() -> Self {
        Self {
            control: "●",
            target_x: "⊕",
            swap: "×",
            measure: "M",
            barrier: "░",
            wire_h: '─',
            wire_v: '│',
            corner_tl: '┌',
            corner_tr: '┐',
            corner_bl: '└',
            corner_br: '┘',
            bracket_l: '[',
            bracket_r: ']',
        }
    }

    fn ascii() -> Self {
        Self {
            control: "@",
            target_x: "(+)",
            swap: "x",
            measure: "M",
            barrier: "|",
            wire_h: '-',
            wire_v: '|',
            corner_tl: '+',
            corner_tr: '+',
            corner_bl: '+',
            corner_br: '+',
            bracket_l: '[',
            bracket_r: ']',
        }
    }

    fn from_style(style: RenderStyle) -> Self {
        match style {
            RenderStyle::Ascii => Self::ascii(),
            _ => Self::unicode(),
        }
    }
}

// ============================================================================
// Terminal Width Detection
// ============================================================================

/// Get terminal width with cross-platform support
///
/// Detection priority:
/// 1. COLUMNS environment variable
/// 2. Platform-specific methods (tput on Unix, stty on Unix fallback)
/// 3. Default to None (caller should use fallback)
fn terminal_width() -> Option<usize> {
    // Priority 1: Environment variable (most reliable when set)
    if let Ok(cols) = std::env::var("COLUMNS") {
        if let Ok(w) = cols.parse::<usize>() {
            if w > 0 {
                return Some(w);
            }
        }
    }

    // Priority 2: Platform-specific detection
    #[cfg(unix)]
    {
        if let Some(w) = get_terminal_size_tput() {
            return Some(w);
        }
        if let Some(w) = get_terminal_size_stty() {
            return Some(w);
        }
    }

    None
}

#[cfg(unix)]
fn get_terminal_size_tput() -> Option<usize> {
    use std::process::Command;

    Command::new("tput")
        .arg("cols")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
            } else {
                None
            }
        })
}

#[cfg(unix)]
fn get_terminal_size_stty() -> Option<usize> {
    use std::process::Command;

    // stty size returns "rows cols"
    Command::new("stty")
        .arg("size")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok().and_then(|s| {
                    let parts: Vec<&str> = s.trim().split_whitespace().collect();
                    if parts.len() >= 2 {
                        parts[1].parse().ok()
                    } else {
                        None
                    }
                })
            } else {
                None
            }
        })
}

// ============================================================================
// Public API
// ============================================================================

/// Render a circuit to ASCII art with default configuration
pub fn render(circuit: &Circuit) -> String {
    render_with_config(circuit, &AsciiConfig::default())
}

/// Render a circuit with custom configuration
pub fn render_with_config(circuit: &Circuit, config: &AsciiConfig) -> String {
    AsciiRenderer::new(circuit, config).render()
}

/// Rendered circuit with metadata
#[derive(Debug, Clone)]
pub struct RenderedCircuit {
    /// The ASCII art representation
    pub ascii: String,
    /// Number of visual columns (gate slots)
    pub columns: usize,
    /// Maximum visual width in characters
    pub width: usize,
    /// Number of qubit lines
    pub qubits: usize,
}

impl fmt::Display for RenderedCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ascii)
    }
}

/// Render with full metadata
pub fn render_detailed(circuit: &Circuit, config: &AsciiConfig) -> RenderedCircuit {
    let renderer = AsciiRenderer::new(circuit, config);
    let columns = renderer.build_columns();
    let ascii = renderer.render();
    let width = ascii.lines().map(|l| l.chars().count()).max().unwrap_or(0);

    RenderedCircuit {
        ascii,
        columns: columns.len(),
        width,
        qubits: circuit.num_qubits(),
    }
}

// ============================================================================
// Core Renderer
// ============================================================================

struct AsciiRenderer<'a> {
    circuit: &'a Circuit,
    config: &'a AsciiConfig,
    symbols: GateSymbols,
    width: usize,
    label_width: usize,
}

impl<'a> AsciiRenderer<'a> {
    fn new(circuit: &'a Circuit, config: &'a AsciiConfig) -> Self {
        let width = config.effective_width();
        let symbols = GateSymbols::from_style(config.style);

        let label_width = if config.show_labels {
            if let Some(ref fmt) = config.label_format {
                // Estimate width from format string
                fmt.len() + digit_count(circuit.num_qubits().saturating_sub(1))
            } else {
                // Default "q{n}: " format
                4 + digit_count(circuit.num_qubits().saturating_sub(1))
            }
        } else {
            0
        };

        Self {
            circuit,
            config,
            symbols,
            width,
            label_width,
        }
    }

    fn render(&self) -> String {
        let mut result = String::new();

        // Add header if showing depth
        if self.config.show_depth && !self.circuit.is_empty() {
            let depth = self.circuit.depth();
            result.push_str(&format!("// Depth: {}\n", depth));
        }

        if self.circuit.is_empty() {
            result.push_str(&self.render_empty());
        } else {
            let columns = self.build_columns();
            let col_widths = self.calculate_column_widths(&columns);
            result.push_str(&self.render_circuit(&columns, &col_widths));
        }

        result
    }

    fn render_empty(&self) -> String {
        let mut result = String::new();
        let wire_width = self.width.saturating_sub(self.label_width + 1).min(40);
        let wire_char = self.config.wire_char();

        for q in 0..self.circuit.num_qubits() {
            if self.config.show_labels {
                result.push_str(&self.format_label(q));
            }
            for _ in 0..wire_width {
                result.push(wire_char);
            }
            result.push('\n');
        }
        result
    }

    fn format_label(&self, qubit: usize) -> String {
        if let Some(ref fmt) = self.config.label_format {
            format!("{:>width$}", fmt.replace("{n}", &qubit.to_string()), width = self.label_width)
        } else {
            format!("{:>width$}", format!("q{}: ", qubit), width = self.label_width)
        }
    }

    /// Group operations into columns based on qubit dependencies
    fn build_columns(&self) -> Vec<Vec<(usize, &GateOp)>> {
        let mut columns: Vec<Vec<(usize, &GateOp)>> = Vec::new();
        let mut qubit_column: HashMap<usize, usize> = HashMap::new();

        for (op_idx, op) in self.circuit.operations().enumerate() {
            let target_col = self.find_target_column(op, &qubit_column);

            // Ensure column exists
            while columns.len() <= target_col {
                columns.push(Vec::new());
            }

            // Add gate to column
            columns[target_col].push((op_idx, op));

            // Update qubit positions
            self.update_qubit_positions(op, target_col, &mut qubit_column);
        }

        columns
    }

    fn find_target_column(&self, op: &GateOp, qubit_column: &HashMap<usize, usize>) -> usize {
        let mut target_col = 0;

        // Find earliest column based on direct qubit dependencies
        for &qubit in op.qubits() {
            if let Some(&col) = qubit_column.get(&qubit.index()) {
                target_col = target_col.max(col + 1);
            }
        }

        // For multi-qubit gates, also check qubits in between (for crossing wires)
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

    fn update_qubit_positions(
        &self,
        op: &GateOp,
        col: usize,
        qubit_column: &mut HashMap<usize, usize>,
    ) {
        // Update all qubits the gate acts on
        for &qubit in op.qubits() {
            qubit_column.insert(qubit.index(), col);
        }

        // For multi-qubit gates, also mark intermediate qubits
        if op.qubits().len() > 1 {
            let indices: Vec<usize> = op.qubits().iter().map(|q| q.index()).collect();
            if let (Some(&min_q), Some(&max_q)) = (indices.iter().min(), indices.iter().max()) {
                for q in min_q..=max_q {
                    qubit_column.insert(q, col);
                }
            }
        }
    }

    /// Calculate optimal width for each column
    fn calculate_column_widths(&self, columns: &[Vec<(usize, &GateOp)>]) -> Vec<usize> {
        if columns.is_empty() {
            return Vec::new();
        }

        let available = self.width.saturating_sub(self.label_width + 4);
        let num_cols = columns.len();

        // Calculate natural (ideal) widths
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
            natural_widths
        } else {
            // Need to compress - use proportional scaling
            let usable = available.saturating_sub(separators);
            let min_w = self.config.min_gate_width;

            if total_natural == 0 {
                return vec![min_w; num_cols];
            }

            natural_widths
                .iter()
                .map(|&w| {
                    let scaled = (w * usable) / total_natural;
                    scaled.max(min_w).min(w)
                })
                .collect()
        }
    }

    fn gate_display_width(&self, op: &GateOp) -> usize {
        let symbol = self.gate_symbol(op);
        // Account for brackets and padding
        display_width(&symbol) + 2
    }

    fn gate_symbol(&self, op: &GateOp) -> String {
        let name = op.gate().name();
        let desc = op.gate().description();
        let default_desc = format!("{}-qubit gate '{}'", op.gate().num_qubits(), name);

        // Determine what to display
        let display = if desc.contains('(') && desc.contains(')') {
            // Parametric gate
            self.format_parametric(&desc)
        } else if desc != default_desc && desc != name {
            // Custom description
            self.format_custom_desc(&desc)
        } else {
            name.to_string()
        };

        // Apply length limits
        self.truncate_if_needed(&display)
    }

    fn format_parametric(&self, desc: &str) -> String {
        if self.config.compact && desc.len() > self.config.max_param_length {
            if let Some(paren_idx) = desc.find('(') {
                let base = &desc[..paren_idx];
                return format!("{}(..)", base);
            }
        }
        desc.to_string()
    }

    fn format_custom_desc(&self, desc: &str) -> String {
        if self.config.compact && desc.len() > self.config.max_param_length {
            let truncate_at = self.config.max_param_length.saturating_sub(3);
            if truncate_at > 0 && truncate_at < desc.len() {
                // Truncate at char boundary
                let truncated: String = desc.chars().take(truncate_at).collect();
                return format!("{}...", truncated);
            }
        }
        desc.to_string()
    }

    fn truncate_if_needed(&self, s: &str) -> String {
        if s.len() > self.config.max_param_length {
            let truncate_at = self.config.max_param_length.saturating_sub(3);
            let truncated: String = s.chars().take(truncate_at).collect();
            format!("{}...", truncated)
        } else {
            s.to_string()
        }
    }

    fn render_circuit(&self, columns: &[Vec<(usize, &GateOp)>], col_widths: &[usize]) -> String {
        let num_qubits = self.circuit.num_qubits();
        let mut lines: Vec<String> = vec![String::new(); num_qubits];

        // Add labels
        if self.config.show_labels {
            for (q, line) in lines.iter_mut().enumerate() {
                line.push_str(&self.format_label(q));
            }
        }

        // Render each column
        for (col_idx, col) in columns.iter().enumerate() {
            let width = col_widths
                .get(col_idx)
                .copied()
                .unwrap_or(self.config.min_gate_width);
            self.render_column(&mut lines, col, width);

            // Add wire between columns
            if col_idx < columns.len() - 1 {
                let wire = self.config.wire_char();
                for line in lines.iter_mut() {
                    line.push(wire);
                }
            }
        }

        // Add trailing wire
        let wire = self.config.wire_char();
        for line in &mut lines {
            line.push(wire);
            line.push(wire);
        }

        lines.join("\n")
    }

    fn render_column(&self, lines: &mut [String], col: &[(usize, &GateOp)], width: usize) {
        let num_qubits = self.circuit.num_qubits();

        // Build qubit role map for this column
        let (qubit_gate, qubit_role) = self.analyze_column(col);

        // Render each qubit line
        for q in 0..num_qubits {
            let line = &mut lines[q];
            let content = self.render_qubit_slot(q, &qubit_gate, &qubit_role, width);
            line.push_str(&content);
        }
    }

    fn analyze_column<'b>(
        &self,
        col: &[(usize, &'b GateOp)],
    ) -> (HashMap<usize, &'b GateOp>, HashMap<usize, QubitRole>) {
        let mut qubit_gate: HashMap<usize, &GateOp> = HashMap::new();
        let mut qubit_role: HashMap<usize, QubitRole> = HashMap::new();

        for (_, op) in col {
            let qubits = op.qubits();

            if qubits.len() == 1 {
                let q = qubits[0].index();
                qubit_gate.insert(q, *op);
                qubit_role.insert(q, QubitRole::Single);
            } else {
                self.analyze_multi_qubit_gate(*op, &mut qubit_gate, &mut qubit_role);
            }
        }

        (qubit_gate, qubit_role)
    }

    fn analyze_multi_qubit_gate<'b>(
        &self,
        op: &'b GateOp,
        qubit_gate: &mut HashMap<usize, &'b GateOp>,
        qubit_role: &mut HashMap<usize, QubitRole>,
    ) {
        let qubits = op.qubits();
        let indices: Vec<usize> = qubits.iter().map(|q| q.index()).collect();
        let (min_q, max_q) = (*indices.iter().min().unwrap(), *indices.iter().max().unwrap());

        let gate_type = self.classify_gate(op);

        for (i, &q) in indices.iter().enumerate() {
            qubit_gate.insert(q, op);
            qubit_role.insert(q, gate_type.role_for_index(i, indices.len()));
        }

        // Mark intermediate qubits as wires
        for q in (min_q + 1)..max_q {
            if !indices.contains(&q) {
                qubit_role.insert(q, QubitRole::Wire);
            }
        }
    }

    fn classify_gate(&self, op: &GateOp) -> GateType {
        let name = op.gate().name().to_uppercase();

        // Check for specific gate patterns
        if name == "SWAP" || name == "ISWAP" {
            return GateType::Swap;
        }

        if name == "CSWAP" || name == "FREDKIN" {
            return GateType::ControlledSwap;
        }

        if name.starts_with('C') || name == "CNOT" || name == "CCNOT" || name == "TOFFOLI" {
            return GateType::Controlled;
        }

        // Check for rotation gates (RXX, RYY, RZZ, etc.)
        if name.starts_with("R") && name.len() == 3 {
            let suffix = &name[1..];
            if suffix == "XX" || suffix == "YY" || suffix == "ZZ" {
                return GateType::Symmetric;
            }
        }

        GateType::Symmetric
    }

    fn render_qubit_slot(
        &self,
        qubit: usize,
        qubit_gate: &HashMap<usize, &GateOp>,
        qubit_role: &HashMap<usize, QubitRole>,
        width: usize,
    ) -> String {
        match qubit_role.get(&qubit) {
            Some(QubitRole::Single) => {
                let op = qubit_gate[&qubit];
                let sym = self.gate_symbol(op);
                let boxed = format!("{}{}{}", self.symbols.bracket_l, sym, self.symbols.bracket_r);
                center_with_wire(&boxed, width, self.config.wire_char())
            },

            Some(QubitRole::Control) => {
                center_with_wire(self.symbols.control, width, self.config.wire_char())
            },

            Some(QubitRole::Target) => {
                let op = qubit_gate[&qubit];
                let sym = self.get_target_symbol(op);
                center_with_wire(&sym, width, self.config.wire_char())
            },

            Some(QubitRole::SwapEnd) => {
                center_with_wire(self.symbols.swap, width, self.config.wire_char())
            },

            Some(QubitRole::Multi) => {
                let op = qubit_gate[&qubit];
                let sym = self.gate_symbol(op);
                let boxed = format!("{}{}{}", self.symbols.bracket_l, sym, self.symbols.bracket_r);
                center_with_wire(&boxed, width, self.config.wire_char())
            },

            Some(QubitRole::Wire) => {
                center_with_wire(&self.symbols.wire_v.to_string(), width, self.config.wire_char())
            },

            None => {
                // Empty slot - just wire
                let wire = self.config.wire_char();
                (0..width).map(|_| wire).collect()
            },
        }
    }

    fn get_target_symbol(&self, op: &GateOp) -> String {
        let name = op.gate().name().to_uppercase();

        match name.as_str() {
            "CNOT" | "CX" | "CCNOT" | "CCX" | "TOFFOLI" => self.symbols.target_x.to_string(),
            "CZ" => self.symbols.control.to_string(),
            "CY" => format!("{}Y{}", self.symbols.bracket_l, self.symbols.bracket_r),
            "SWAP" | "ISWAP" | "CSWAP" | "FREDKIN" => self.symbols.swap.to_string(),
            _ => {
                // For other controlled gates, show the base gate
                let base = name.trim_start_matches('C');
                format!("{}{}{}", self.symbols.bracket_l, base, self.symbols.bracket_r)
            },
        }
    }
}

// ============================================================================
// Helper Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QubitRole {
    /// Single-qubit gate
    Single,
    /// Control qubit of controlled gate
    Control,
    /// Target qubit of controlled gate
    Target,
    /// End of a SWAP gate
    SwapEnd,
    /// Part of a symmetric multi-qubit gate
    Multi,
    /// Wire passing through (for non-adjacent multi-qubit gates)
    Wire,
}

#[derive(Debug, Clone, Copy)]
enum GateType {
    /// Controlled gate (CNOT, CZ, CCX, etc.)
    Controlled,
    /// SWAP-type gate
    Swap,
    /// Controlled SWAP (Fredkin)
    ControlledSwap,
    /// Symmetric multi-qubit gate (RXX, RZZ, etc.)
    Symmetric,
}

impl GateType {
    fn role_for_index(self, index: usize, total: usize) -> QubitRole {
        match self {
            GateType::Controlled => {
                if index < total - 1 {
                    QubitRole::Control
                } else {
                    QubitRole::Target
                }
            },
            GateType::Swap => QubitRole::SwapEnd,
            GateType::ControlledSwap => {
                if index == 0 {
                    QubitRole::Control
                } else {
                    QubitRole::SwapEnd
                }
            },
            GateType::Symmetric => QubitRole::Multi,
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Count digits in a number
fn digit_count(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        (n as f64).log10().floor() as usize + 1
    }
}

/// Get display width of a string (handles Unicode)
fn display_width(s: &str) -> usize {
    s.chars()
        .map(|c| {
            if c.is_ascii() {
                1
            } else {
                // Most box-drawing and mathematical symbols are single-width
                // CJK characters would be double-width, but we don't use them
                1
            }
        })
        .sum()
}

/// Center a string within width, padding with wire characters
fn center_with_wire(s: &str, width: usize, wire: char) -> String {
    let s_width = display_width(s);
    if s_width >= width {
        return s.to_string();
    }

    let total_pad = width - s_width;
    let left_pad = total_pad / 2;
    let right_pad = total_pad - left_pad;

    let mut result = String::with_capacity(width);
    for _ in 0..left_pad {
        result.push(wire);
    }
    result.push_str(s);
    for _ in 0..right_pad {
        result.push(wire);
    }
    result
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
        let ascii = render(&circuit);
        assert!(ascii.contains("q0:"));
        assert!(ascii.contains("q1:"));
        assert!(ascii.contains("─") || ascii.contains("-"));
    }

    #[test]
    fn test_single_gate() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(MockGate::new("H", 1)), &[QubitId::new(0)])
            .unwrap();

        let ascii = render(&circuit);
        assert!(ascii.contains("[H]"));
    }

    #[test]
    fn test_config_builder() {
        let config = AsciiConfig::builder()
            .max_width(80)
            .show_labels(false)
            .style(RenderStyle::Ascii)
            .build();

        assert_eq!(config.max_width, 80);
        assert!(!config.show_labels);
        assert_eq!(config.style, RenderStyle::Ascii);
    }

    #[test]
    fn test_ascii_style() {
        let circuit = Circuit::new(2);
        let config = AsciiConfig::ascii_only();
        let ascii = render_with_config(&circuit, &config);

        // Should not contain Unicode characters
        assert!(ascii.chars().all(|c| c.is_ascii() || c == '\n'));
    }

    #[test]
    fn test_custom_gate_description() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(
                Arc::new(MockGate::with_description("Oracle", 1, "Grover Oracle")),
                &[QubitId::new(0)],
            )
            .unwrap();

        let ascii = render(&circuit);
        assert!(ascii.contains("Grover Oracle"));
    }

    #[test]
    fn test_parametric_gate() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(
                Arc::new(MockGate::with_description("U", 1, "U(1.57, 3.14)")),
                &[QubitId::new(0)],
            )
            .unwrap();

        let ascii = render(&circuit);
        assert!(ascii.contains("U(1.57, 3.14)"));
    }

    #[test]
    fn test_compact_truncation() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(
                Arc::new(MockGate::with_description(
                    "VeryLongGate",
                    1,
                    "VeryLongGateName(1.0, 2.0, 3.0)",
                )),
                &[QubitId::new(0)],
            )
            .unwrap();

        let config = AsciiConfig::compact();
        let ascii = render_with_config(&circuit, &config);

        // Should be truncated
        assert!(ascii.contains(".."));
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

        let ascii = render(&circuit);
        assert!(ascii.contains("[XX(0.785)]"));
    }

    #[test]
    fn test_controlled_gate() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(MockGate::new("CNOT", 2)), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();

        let ascii = render(&circuit);
        // Should have control dot and target
        assert!(ascii.contains("●") || ascii.contains("@"));
        assert!(ascii.contains("⊕") || ascii.contains("(+)"));
    }

    #[test]
    fn test_swap_gate() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(MockGate::new("SWAP", 2)), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();

        let ascii = render(&circuit);
        // Should have swap symbols on both lines
        assert!(ascii.contains("×") || ascii.contains("x"));
    }

    #[test]
    fn test_render_detailed() {
        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(Arc::new(MockGate::new("H", 1)), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(MockGate::new("X", 1)), &[QubitId::new(1)])
            .unwrap();

        let result = render_detailed(&circuit, &AsciiConfig::default());
        assert_eq!(result.qubits, 3);
        assert!(result.columns > 0);
        assert!(result.width > 0);
    }

    #[test]
    fn test_non_adjacent_multi_qubit() {
        let mut circuit = Circuit::new(4);
        circuit
            .add_gate(Arc::new(MockGate::new("CNOT", 2)), &[QubitId::new(0), QubitId::new(3)])
            .unwrap();

        let ascii = render(&circuit);
        // Should have vertical wire for intermediate qubits
        assert!(ascii.contains("│") || ascii.contains("|"));
    }

    #[test]
    fn test_digit_count() {
        assert_eq!(digit_count(0), 1);
        assert_eq!(digit_count(9), 1);
        assert_eq!(digit_count(10), 2);
        assert_eq!(digit_count(99), 2);
        assert_eq!(digit_count(100), 3);
    }

    #[test]
    fn test_display_width() {
        assert_eq!(display_width("abc"), 3);
        assert_eq!(display_width("[H]"), 3);
        assert_eq!(display_width("●"), 1);
        assert_eq!(display_width("⊕"), 1);
    }

    #[test]
    fn test_center_with_wire() {
        let result = center_with_wire("[H]", 7, '─');
        // Visual width should be 7: 2 wire chars + [H] (3 chars) + 2 wire chars
        assert_eq!(display_width(&result), 7);
        assert!(result.contains("[H]"));
        assert!(result.starts_with("──"));
        assert!(result.ends_with("──"));
    }
}
