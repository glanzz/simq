//! Optimized rotation gates using lookup tables
//!
//! This module provides drop-in replacements for rotation gates that use
//! pre-computed lookup tables for small angles, offering significant performance
//! improvements in circuits with many small-angle rotations.
//!
//! # When to Use
//!
//! Use these optimized gates when:
//! - Your circuit has many rotation gates with small angles (< π/4)
//! - You're running VQE, QAOA, or other variational algorithms
//! - You're performing gradient-based optimization with small parameter updates
//! - Performance is critical and you can afford the small memory overhead
//!
//! # Example
//!
//! ```rust
//! use simq_gates::optimized::{OptimizedRotationX, create_global_lookup_table};
//! use std::f64::consts::PI;
//!
//! // Create a global lookup table (do this once at program start)
//! let table = create_global_lookup_table();
//!
//! // Use optimized rotation gates
//! let rx = OptimizedRotationX::new(0.01, &table);
//! let matrix = rx.compute_matrix(); // Uses fast lookup
//!
//! // Large angles automatically fall back to direct computation
//! let rx_large = OptimizedRotationX::new(PI, &table);
//! let matrix_large = rx_large.compute_matrix(); // Uses direct computation
//! ```

use crate::lookup::{LookupConfig, RotationLookupTable};
use num_complex::Complex64;
use simq_core::gate::Gate;
use std::f64::consts::PI;
use std::sync::Arc;

/// Thread-safe shared lookup table
pub type SharedLookupTable = Arc<RotationLookupTable>;

/// Create a global lookup table with sensible defaults for general use
///
/// This creates a table optimized for angles up to π/2 with 2048 entries,
/// providing a good balance between accuracy and memory usage.
///
/// Memory usage: ~32 KB
pub fn create_global_lookup_table() -> SharedLookupTable {
    let config = LookupConfig::new()
        .max_angle(PI / 2.0)
        .num_entries(2048)
        .interpolation_enabled(true);

    Arc::new(RotationLookupTable::new(config))
}

/// Create a high-precision lookup table for demanding applications
///
/// This creates a table with 4096 entries covering angles up to π/2,
/// providing higher accuracy at the cost of more memory.
///
/// Memory usage: ~64 KB
pub fn create_high_precision_table() -> SharedLookupTable {
    let config = LookupConfig::new()
        .max_angle(PI / 2.0)
        .num_entries(4096)
        .interpolation_enabled(true);

    Arc::new(RotationLookupTable::new(config))
}

/// Create a compact lookup table for memory-constrained environments
///
/// This creates a table with 512 entries covering angles up to π/4.
/// Suitable for embedded systems or when memory is limited.
///
/// Memory usage: ~8 KB
pub fn create_compact_table() -> SharedLookupTable {
    let config = LookupConfig::new()
        .max_angle(PI / 4.0)
        .num_entries(512)
        .interpolation_enabled(true);

    Arc::new(RotationLookupTable::new(config))
}

/// Optimized RX gate using lookup tables
#[derive(Debug, Clone)]
pub struct OptimizedRotationX {
    theta: f64,
    table: SharedLookupTable,
}

impl OptimizedRotationX {
    /// Create a new optimized RX gate
    pub fn new(theta: f64, table: &SharedLookupTable) -> Self {
        Self {
            theta,
            table: Arc::clone(table),
        }
    }

    /// Get the rotation angle
    pub fn angle(&self) -> f64 {
        self.theta
    }

    /// Compute the matrix using lookup table when possible
    #[inline]
    pub fn compute_matrix(&self) -> [[Complex64; 2]; 2] {
        self.table.rx_matrix(self.theta)
    }
}

impl Gate for OptimizedRotationX {
    fn name(&self) -> &str {
        "RX"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn description(&self) -> String {
        format!("RX({:.4}) [optimized]", self.theta)
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.compute_matrix().iter().flatten().copied().collect())
    }

    fn is_unitary(&self) -> bool {
        true
    }

    fn is_hermitian(&self) -> bool {
        false
    }
}

/// Optimized RY gate using lookup tables
#[derive(Debug, Clone)]
pub struct OptimizedRotationY {
    theta: f64,
    table: SharedLookupTable,
}

impl OptimizedRotationY {
    /// Create a new optimized RY gate
    pub fn new(theta: f64, table: &SharedLookupTable) -> Self {
        Self {
            theta,
            table: Arc::clone(table),
        }
    }

    /// Get the rotation angle
    pub fn angle(&self) -> f64 {
        self.theta
    }

    /// Compute the matrix using lookup table when possible
    #[inline]
    pub fn compute_matrix(&self) -> [[Complex64; 2]; 2] {
        self.table.ry_matrix(self.theta)
    }
}

impl Gate for OptimizedRotationY {
    fn name(&self) -> &str {
        "RY"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn description(&self) -> String {
        format!("RY({:.4}) [optimized]", self.theta)
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.compute_matrix().iter().flatten().copied().collect())
    }

    fn is_unitary(&self) -> bool {
        true
    }

    fn is_hermitian(&self) -> bool {
        false
    }
}

/// Optimized RZ gate using lookup tables
#[derive(Debug, Clone)]
pub struct OptimizedRotationZ {
    theta: f64,
    table: SharedLookupTable,
}

impl OptimizedRotationZ {
    /// Create a new optimized RZ gate
    pub fn new(theta: f64, table: &SharedLookupTable) -> Self {
        Self {
            theta,
            table: Arc::clone(table),
        }
    }

    /// Get the rotation angle
    pub fn angle(&self) -> f64 {
        self.theta
    }

    /// Compute the matrix using lookup table when possible
    #[inline]
    pub fn compute_matrix(&self) -> [[Complex64; 2]; 2] {
        self.table.rz_matrix(self.theta)
    }
}

impl Gate for OptimizedRotationZ {
    fn name(&self) -> &str {
        "RZ"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn description(&self) -> String {
        format!("RZ({:.4}) [optimized]", self.theta)
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.compute_matrix().iter().flatten().copied().collect())
    }

    fn is_unitary(&self) -> bool {
        true
    }

    fn is_hermitian(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrices;
    use approx::assert_relative_eq;

    #[test]
    fn test_optimized_rx_small_angle() {
        let table = create_global_lookup_table();
        let theta = 0.05;

        let gate = OptimizedRotationX::new(theta, &table);
        let matrix_opt = gate.compute_matrix();
        let matrix_ref = matrices::rotation_x(theta);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(matrix_opt[i][j].re, matrix_ref[i][j].re, epsilon = 1e-6);
                assert_relative_eq!(matrix_opt[i][j].im, matrix_ref[i][j].im, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_optimized_ry_large_angle() {
        let table = create_global_lookup_table();
        let theta = PI;

        let gate = OptimizedRotationY::new(theta, &table);
        let matrix_opt = gate.compute_matrix();
        let matrix_ref = matrices::rotation_y(theta);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(matrix_opt[i][j].re, matrix_ref[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(matrix_opt[i][j].im, matrix_ref[i][j].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_optimized_rz_negative_angle() {
        let table = create_global_lookup_table();
        let theta = -0.1;

        let gate = OptimizedRotationZ::new(theta, &table);
        let matrix_opt = gate.compute_matrix();
        let matrix_ref = matrices::rotation_z(theta);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(matrix_opt[i][j].re, matrix_ref[i][j].re, epsilon = 1e-6);
                assert_relative_eq!(matrix_opt[i][j].im, matrix_ref[i][j].im, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_table_sharing() {
        let table = create_global_lookup_table();

        let rx1 = OptimizedRotationX::new(0.1, &table);
        let _rx2 = OptimizedRotationX::new(0.2, &table);
        let _ry1 = OptimizedRotationY::new(0.3, &table);

        // All gates share the same table (reference counting)
        assert_eq!(Arc::strong_count(&rx1.table), 4); // rx1, rx2, ry1, and original table
    }

    #[test]
    fn test_compact_table_memory() {
        let table = create_compact_table();
        let stats = table.stats();

        // Compact table should use less than 10 KB
        assert!(stats.memory_bytes < 10 * 1024);
    }
}
