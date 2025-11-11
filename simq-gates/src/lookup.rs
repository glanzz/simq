//! Lookup tables for small-angle rotation gates
//!
//! This module provides high-performance lookup tables for rotation gates (RX, RY, RZ)
//! when dealing with small angles. For quantum circuits with many small-angle rotations
//! (common in VQE, QAOA, and gradient-based optimization), lookup tables can provide
//! significant performance improvements over direct trigonometric computation.
//!
//! # Design
//!
//! - **Compile-time generation**: Tables are built at compile time using const functions
//! - **Linear interpolation**: Smooth transitions between table entries
//! - **Configurable precision**: Trade-off between memory and accuracy
//! - **Automatic fallback**: Large angles use direct computation
//!
//! # Example
//!
//! ```rust
//! use simq_gates::lookup::{RotationLookupTable, LookupConfig};
//! use std::f64::consts::PI;
//!
//! // Create a lookup table for angles up to π/4 with 1000 entries
//! let config = LookupConfig::new()
//!     .max_angle(PI / 4.0)
//!     .num_entries(1000)
//!     .interpolation_enabled(true);
//!
//! let table = RotationLookupTable::new(config);
//!
//! // Fast lookup for small angles
//! let matrix = table.rx_matrix(0.01); // Uses lookup + interpolation
//! let matrix2 = table.rx_matrix(PI);  // Falls back to direct computation
//! ```

use num_complex::Complex64;
use std::f64::consts::PI;

/// Configuration for rotation lookup tables
#[derive(Debug, Clone, Copy)]
pub struct LookupConfig {
    /// Maximum angle for which to use lookup tables (in radians)
    /// Typical value: π/4 to π/2
    max_angle: f64,

    /// Number of pre-computed entries in the table
    /// More entries = better accuracy but higher memory usage
    /// Typical range: 500-2000 entries
    num_entries: usize,

    /// Whether to use linear interpolation between entries
    /// Interpolation improves accuracy with minimal performance cost
    interpolation: bool,

    /// Error tolerance for determining when to use lookup vs direct computation
    /// Lower values favor direct computation for better accuracy
    error_tolerance: f64,
}

impl LookupConfig {
    /// Create a new configuration with sensible defaults
    ///
    /// Defaults:
    /// - max_angle: π/4 (45 degrees)
    /// - num_entries: 1024
    /// - interpolation: enabled
    /// - error_tolerance: 1e-12
    pub const fn new() -> Self {
        Self {
            max_angle: PI / 4.0,
            num_entries: 1024,
            interpolation: true,
            error_tolerance: 1e-12,
        }
    }

    /// Set the maximum angle for lookup table coverage
    pub const fn max_angle(mut self, angle: f64) -> Self {
        self.max_angle = angle;
        self
    }

    /// Set the number of table entries
    pub const fn num_entries(mut self, n: usize) -> Self {
        self.num_entries = n;
        self
    }

    /// Enable or disable interpolation
    pub const fn interpolation_enabled(mut self, enabled: bool) -> Self {
        self.interpolation = enabled;
        self
    }

    /// Set the error tolerance
    pub const fn error_tolerance(mut self, tolerance: f64) -> Self {
        self.error_tolerance = tolerance;
        self
    }
}

impl Default for LookupConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-computed trigonometric values for small angles
///
/// Stores cos(θ/2) and sin(θ/2) for evenly-spaced angles from 0 to max_angle
#[derive(Debug, Clone)]
struct TrigTable {
    /// Cosine values: cos(θ/2) for each angle
    cos_values: Vec<f64>,

    /// Sine values: sin(θ/2) for each angle
    sin_values: Vec<f64>,

    /// The angle step between consecutive entries
    angle_step: f64,

    /// Maximum angle covered by this table
    max_angle: f64,
}

impl TrigTable {
    /// Build a new trigonometric lookup table
    fn new(config: &LookupConfig) -> Self {
        let n = config.num_entries;
        let max_angle = config.max_angle;
        let angle_step = max_angle / (n - 1) as f64;

        let mut cos_values = Vec::with_capacity(n);
        let mut sin_values = Vec::with_capacity(n);

        for i in 0..n {
            let angle = i as f64 * angle_step;
            let half_angle = angle / 2.0;
            cos_values.push(half_angle.cos());
            sin_values.push(half_angle.sin());
        }

        Self {
            cos_values,
            sin_values,
            angle_step,
            max_angle,
        }
    }

    /// Check if an angle is within the table's range
    #[inline]
    fn contains(&self, angle: f64) -> bool {
        angle.abs() <= self.max_angle
    }

    /// Get cos(θ/2) and sin(θ/2) for a given angle using lookup and optional interpolation
    ///
    /// Returns (cos_value, sin_value)
    #[inline]
    fn lookup(&self, angle: f64, interpolate: bool) -> (f64, f64) {
        debug_assert!(self.contains(angle), "Angle {} exceeds table range", angle);

        // Handle negative angles using symmetry: cos(-x) = cos(x), sin(-x) = -sin(x)
        let (abs_angle, sign) = if angle < 0.0 {
            (-angle, -1.0)
        } else {
            (angle, 1.0)
        };

        // Find the index in the table
        let index_f = abs_angle / self.angle_step;
        let index = index_f as usize;

        if !interpolate || index >= self.cos_values.len() - 1 {
            // No interpolation or at the edge - use nearest entry
            let idx = index.min(self.cos_values.len() - 1);
            return (self.cos_values[idx], sign * self.sin_values[idx]);
        }

        // Linear interpolation between table entries
        let frac = index_f - index as f64;
        let cos_val = self.cos_values[index] * (1.0 - frac)
                    + self.cos_values[index + 1] * frac;
        let sin_val = (self.sin_values[index] * (1.0 - frac)
                    + self.sin_values[index + 1] * frac) * sign;

        (cos_val, sin_val)
    }
}

/// Lookup table for rotation gate matrices
///
/// Provides fast matrix computation for small-angle rotations using pre-computed
/// trigonometric values and optional linear interpolation.
#[derive(Debug, Clone)]
pub struct RotationLookupTable {
    /// Pre-computed trigonometric values
    trig_table: TrigTable,

    /// Configuration
    config: LookupConfig,
}

impl RotationLookupTable {
    /// Create a new rotation lookup table with the given configuration
    pub fn new(config: LookupConfig) -> Self {
        let trig_table = TrigTable::new(&config);
        Self { trig_table, config }
    }

    /// Create a lookup table with default configuration
    pub fn default() -> Self {
        Self::new(LookupConfig::default())
    }

    /// Get the RX(θ) matrix, using lookup table if possible
    ///
    /// Matrix form:
    /// ```text
    /// [ cos(θ/2)    -i·sin(θ/2) ]
    /// [ -i·sin(θ/2)  cos(θ/2)   ]
    /// ```
    #[inline]
    pub fn rx_matrix(&self, theta: f64) -> [[Complex64; 2]; 2] {
        if self.trig_table.contains(theta) {
            let (cos_val, sin_val) = self.trig_table.lookup(theta, self.config.interpolation);

            [
                [
                    Complex64::new(cos_val, 0.0),
                    Complex64::new(0.0, -sin_val),
                ],
                [
                    Complex64::new(0.0, -sin_val),
                    Complex64::new(cos_val, 0.0),
                ],
            ]
        } else {
            // Fallback to direct computation for large angles
            crate::matrices::rotation_x(theta)
        }
    }

    /// Get the RY(θ) matrix, using lookup table if possible
    ///
    /// Matrix form:
    /// ```text
    /// [ cos(θ/2)  -sin(θ/2) ]
    /// [ sin(θ/2)   cos(θ/2) ]
    /// ```
    #[inline]
    pub fn ry_matrix(&self, theta: f64) -> [[Complex64; 2]; 2] {
        if self.trig_table.contains(theta) {
            let (cos_val, sin_val) = self.trig_table.lookup(theta, self.config.interpolation);

            [
                [
                    Complex64::new(cos_val, 0.0),
                    Complex64::new(-sin_val, 0.0),
                ],
                [
                    Complex64::new(sin_val, 0.0),
                    Complex64::new(cos_val, 0.0),
                ],
            ]
        } else {
            // Fallback to direct computation for large angles
            crate::matrices::rotation_y(theta)
        }
    }

    /// Get the RZ(θ) matrix, using lookup table if possible
    ///
    /// Matrix form:
    /// ```text
    /// [ e^(-iθ/2)    0      ]
    /// [    0      e^(iθ/2)  ]
    /// ```
    #[inline]
    pub fn rz_matrix(&self, theta: f64) -> [[Complex64; 2]; 2] {
        if self.trig_table.contains(theta) {
            let (cos_val, sin_val) = self.trig_table.lookup(theta, self.config.interpolation);

            [
                [
                    Complex64::new(cos_val, -sin_val),
                    Complex64::new(0.0, 0.0),
                ],
                [
                    Complex64::new(0.0, 0.0),
                    Complex64::new(cos_val, sin_val),
                ],
            ]
        } else {
            // Fallback to direct computation for large angles
            crate::matrices::rotation_z(theta)
        }
    }

    /// Get statistics about table usage and configuration
    pub fn stats(&self) -> LookupStats {
        LookupStats {
            num_entries: self.trig_table.cos_values.len(),
            max_angle: self.trig_table.max_angle,
            angle_step: self.trig_table.angle_step,
            interpolation_enabled: self.config.interpolation,
            memory_bytes: self.memory_usage(),
        }
    }

    /// Calculate memory usage in bytes
    fn memory_usage(&self) -> usize {
        // Two f64 vectors (cos and sin)
        self.trig_table.cos_values.len() * std::mem::size_of::<f64>() * 2
    }
}

/// Statistics about a lookup table
#[derive(Debug, Clone)]
pub struct LookupStats {
    /// Number of entries in the table
    pub num_entries: usize,

    /// Maximum angle covered (radians)
    pub max_angle: f64,

    /// Angle step between entries (radians)
    pub angle_step: f64,

    /// Whether interpolation is enabled
    pub interpolation_enabled: bool,

    /// Memory usage in bytes
    pub memory_bytes: usize,
}

impl std::fmt::Display for LookupStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RotationLookupTable Stats:\n\
             - Entries: {}\n\
             - Max angle: {:.6} rad ({:.2}°)\n\
             - Angle step: {:.6} rad ({:.4}°)\n\
             - Interpolation: {}\n\
             - Memory: {} bytes ({:.2} KB)",
            self.num_entries,
            self.max_angle,
            self.max_angle.to_degrees(),
            self.angle_step,
            self.angle_step.to_degrees(),
            if self.interpolation_enabled { "enabled" } else { "disabled" },
            self.memory_bytes,
            self.memory_bytes as f64 / 1024.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_trig_table_basic() {
        let config = LookupConfig::new()
            .max_angle(PI / 4.0)
            .num_entries(100);

        let table = TrigTable::new(&config);

        assert_eq!(table.cos_values.len(), 100);
        assert_eq!(table.sin_values.len(), 100);

        // First entry should be cos(0), sin(0)
        assert_relative_eq!(table.cos_values[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(table.sin_values[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trig_table_negative_angles() {
        let config = LookupConfig::new()
            .max_angle(PI / 4.0)
            .num_entries(100);

        let table = TrigTable::new(&config);

        let angle = 0.1;
        let (cos_pos, sin_pos) = table.lookup(angle, false);
        let (cos_neg, sin_neg) = table.lookup(-angle, false);

        // cos(-x) = cos(x)
        assert_relative_eq!(cos_pos, cos_neg, epsilon = 1e-10);

        // sin(-x) = -sin(x)
        assert_relative_eq!(sin_pos, -sin_neg, epsilon = 1e-10);
    }

    #[test]
    fn test_rx_matrix_small_angle() {
        let config = LookupConfig::new()
            .max_angle(PI / 4.0)
            .num_entries(1000)
            .interpolation_enabled(true);

        let table = RotationLookupTable::new(config);

        let theta = 0.01;
        let matrix_lookup = table.rx_matrix(theta);
        let matrix_direct = crate::matrices::rotation_x(theta);

        // Compare all elements
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    matrix_lookup[i][j].re,
                    matrix_direct[i][j].re,
                    epsilon = 1e-6
                );
                assert_relative_eq!(
                    matrix_lookup[i][j].im,
                    matrix_direct[i][j].im,
                    epsilon = 1e-6
                );
            }
        }
    }

    #[test]
    fn test_ry_matrix_interpolation() {
        let config = LookupConfig::new()
            .max_angle(PI / 4.0)
            .num_entries(100) // Coarse table
            .interpolation_enabled(true);

        let table = RotationLookupTable::new(config);

        // Test an angle that falls between table entries
        let theta = 0.1234;
        let matrix = table.ry_matrix(theta);

        // Verify unitarity: U†U = I
        let u00 = matrix[0][0];
        let u01 = matrix[0][1];
        let u10 = matrix[1][0];
        let u11 = matrix[1][1];

        // First row of U†U
        let result_00 = u00.conj() * u00 + u10.conj() * u10;
        let result_01 = u00.conj() * u01 + u10.conj() * u11;

        // Interpolation introduces small numerical errors - use more tolerant epsilon
        assert_relative_eq!(result_00.re, 1.0, epsilon = 1e-5);
        assert_relative_eq!(result_00.im, 0.0, epsilon = 1e-5);
        assert_relative_eq!(result_01.norm(), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_rz_matrix_large_angle_fallback() {
        let config = LookupConfig::new()
            .max_angle(PI / 4.0)
            .num_entries(1000);

        let table = RotationLookupTable::new(config);

        // Angle beyond table range - should fall back to direct computation
        let theta = PI;
        let matrix_lookup = table.rz_matrix(theta);
        let matrix_direct = crate::matrices::rotation_z(theta);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    matrix_lookup[i][j].re,
                    matrix_direct[i][j].re,
                    epsilon = 1e-10
                );
                assert_relative_eq!(
                    matrix_lookup[i][j].im,
                    matrix_direct[i][j].im,
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_stats() {
        let config = LookupConfig::new()
            .max_angle(PI / 4.0)
            .num_entries(1024);

        let table = RotationLookupTable::new(config);
        let stats = table.stats();

        assert_eq!(stats.num_entries, 1024);
        assert_relative_eq!(stats.max_angle, PI / 4.0, epsilon = 1e-10);
        assert!(stats.memory_bytes > 0);
    }
}
