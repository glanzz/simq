//! Auto-generated compile-time gate matrices
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
//!
//! This module contains matrices generated at build time by build.rs.
//! All matrices are embedded in the binary for zero-cost runtime access.

// Include the build-time generated code
include!(concat!(env!("OUT_DIR"), "/generated_gates.rs"));

use num_complex::Complex64;

/// Unified interface for generated angle lookups
pub struct GeneratedAngleCache;

impl GeneratedAngleCache {
    /// Lookup RX matrix for Clifford+T angles
    pub fn rx_clifford_t(theta: f64) -> Option<[[Complex64; 2]; 2]> {
        use std::f64::consts::PI;
        const EPSILON: f64 = 1e-10;

        if (theta - PI / 2.0).abs() < EPSILON {
            Some(clifford_t::RX_PI_OVER_2)
        } else if (theta - PI / 4.0).abs() < EPSILON {
            Some(clifford_t::RX_PI_OVER_4)
        } else if (theta - PI / 8.0).abs() < EPSILON {
            Some(clifford_t::RX_PI_OVER_8)
        } else if (theta - PI / 16.0).abs() < EPSILON {
            Some(clifford_t::RX_PI_OVER_16)
        } else if (theta - PI / 32.0).abs() < EPSILON {
            Some(clifford_t::RX_PI_OVER_32)
        } else {
            None
        }
    }

    /// Lookup RY matrix for Clifford+T angles
    pub fn ry_clifford_t(theta: f64) -> Option<[[Complex64; 2]; 2]> {
        use std::f64::consts::PI;
        const EPSILON: f64 = 1e-10;

        if (theta - PI / 2.0).abs() < EPSILON {
            Some(clifford_t::RY_PI_OVER_2)
        } else if (theta - PI / 4.0).abs() < EPSILON {
            Some(clifford_t::RY_PI_OVER_4)
        } else if (theta - PI / 8.0).abs() < EPSILON {
            Some(clifford_t::RY_PI_OVER_8)
        } else if (theta - PI / 16.0).abs() < EPSILON {
            Some(clifford_t::RY_PI_OVER_16)
        } else if (theta - PI / 32.0).abs() < EPSILON {
            Some(clifford_t::RY_PI_OVER_32)
        } else {
            None
        }
    }

    /// Lookup RZ matrix for Clifford+T angles
    pub fn rz_clifford_t(theta: f64) -> Option<[[Complex64; 2]; 2]> {
        use std::f64::consts::PI;
        const EPSILON: f64 = 1e-10;

        if (theta - PI / 2.0).abs() < EPSILON {
            Some(clifford_t::RZ_PI_OVER_2)
        } else if (theta - PI / 4.0).abs() < EPSILON {
            Some(clifford_t::RZ_PI_OVER_4)
        } else if (theta - PI / 8.0).abs() < EPSILON {
            Some(clifford_t::RZ_PI_OVER_8)
        } else if (theta - PI / 16.0).abs() < EPSILON {
            Some(clifford_t::RZ_PI_OVER_16)
        } else if (theta - PI / 32.0).abs() < EPSILON {
            Some(clifford_t::RZ_PI_OVER_32)
        } else {
            None
        }
    }

    /// Lookup RX matrix for QAOA mixer angles
    ///
    /// The cache is only used when `theta` lands exactly (within 1e-12) on a
    /// grid point; any other angle is computed at runtime. Nearest-neighbor
    /// snapping is never acceptable for gate matrices — it silently returns
    /// the matrix of a different angle (see issue #37).
    #[inline]
    pub fn rx_qaoa(theta: f64) -> [[Complex64; 2]; 2] {
        use std::f64::consts::PI;

        if (0.0..=PI).contains(&theta) {
            let index = (theta / qaoa::ANGLE_STEP).round() as usize;
            let index = index.min(qaoa::NUM_QAOA_ANGLES - 1);
            let cached_theta = index as f64 * qaoa::ANGLE_STEP;
            if (theta - cached_theta).abs() < 1e-12 {
                return qaoa::RX_MIXER[index];
            }
        }

        crate::matrices::rotation_x(theta)
    }

    /// Lookup RZ matrix for QAOA cost angles
    ///
    /// The cache is only used when `theta` lands exactly (within 1e-12) on a
    /// grid point; any other angle is computed at runtime. Nearest-neighbor
    /// snapping is never acceptable for gate matrices — it silently returns
    /// the matrix of a different angle (see issue #37).
    #[inline]
    pub fn rz_qaoa(theta: f64) -> [[Complex64; 2]; 2] {
        use std::f64::consts::PI;

        if (0.0..=PI).contains(&theta) {
            let index = (theta / qaoa::ANGLE_STEP).round() as usize;
            let index = index.min(qaoa::NUM_QAOA_ANGLES - 1);
            let cached_theta = index as f64 * qaoa::ANGLE_STEP;
            if (theta - cached_theta).abs() < 1e-12 {
                return qaoa::RZ_COST[index];
            }
        }

        crate::matrices::rotation_z(theta)
    }

    /// Lookup RX matrix for common π fractions
    pub fn rx_pi_fraction(theta: f64) -> Option<[[Complex64; 2]; 2]> {
        use std::f64::consts::PI;
        const EPSILON: f64 = 1e-10;

        if (theta - PI / 2.0).abs() < EPSILON {
            Some(pi_fractions::RX_PI_OVER_2)
        } else if (theta - PI / 3.0).abs() < EPSILON {
            Some(pi_fractions::RX_PI_OVER_3)
        } else if (theta - PI / 4.0).abs() < EPSILON {
            Some(pi_fractions::RX_PI_OVER_4)
        } else if (theta - PI / 5.0).abs() < EPSILON {
            Some(pi_fractions::RX_PI_OVER_5)
        } else if (theta - PI / 6.0).abs() < EPSILON {
            Some(pi_fractions::RX_PI_OVER_6)
        } else if (theta - PI / 8.0).abs() < EPSILON {
            Some(pi_fractions::RX_PI_OVER_8)
        } else if (theta - PI / 10.0).abs() < EPSILON {
            Some(pi_fractions::RX_PI_OVER_10)
        } else if (theta - PI / 12.0).abs() < EPSILON {
            Some(pi_fractions::RX_PI_OVER_12)
        } else {
            None
        }
    }
}

/// Enhanced universal cache that includes generated matrices
pub struct EnhancedUniversalCache;

impl EnhancedUniversalCache {
    /// Lookup RX matrix with all caching strategies
    ///
    /// Priority order:
    /// 1. Common angles (compile-time constants)
    /// 2. Clifford+T angles (build-time generated)
    /// 3. π fractions (build-time generated)
    /// 4. VQE range cache
    /// 5. QAOA range (build-time generated)
    /// 6. Runtime computation
    #[inline]
    pub fn rx(theta: f64) -> [[Complex64; 2]; 2] {
        // Level 1: Common angles
        if let Some(matrix) = crate::compile_time_cache::CommonAngles::rx_lookup(theta) {
            return matrix;
        }

        // Level 2: Clifford+T angles
        if let Some(matrix) = GeneratedAngleCache::rx_clifford_t(theta) {
            return matrix;
        }

        // Level 3: π fractions
        if let Some(matrix) = GeneratedAngleCache::rx_pi_fraction(theta) {
            return matrix;
        }

        // Level 4: VQE range cache
        if theta.abs() <= crate::compile_time_cache::VQEAngles::MAX_ANGLE {
            return crate::compile_time_cache::VQEAngles::rx_cached(theta);
        }

        // Level 5: QAOA range
        use std::f64::consts::PI;
        if (0.0..=PI).contains(&theta) {
            return GeneratedAngleCache::rx_qaoa(theta);
        }

        // Level 6: Runtime computation
        crate::matrices::rotation_x(theta)
    }

    /// Lookup RY matrix with all caching strategies
    #[inline]
    pub fn ry(theta: f64) -> [[Complex64; 2]; 2] {
        if let Some(matrix) = crate::compile_time_cache::CommonAngles::ry_lookup(theta) {
            return matrix;
        }

        if let Some(matrix) = GeneratedAngleCache::ry_clifford_t(theta) {
            return matrix;
        }

        if theta.abs() <= crate::compile_time_cache::VQEAngles::MAX_ANGLE {
            return crate::compile_time_cache::VQEAngles::ry_cached(theta);
        }

        crate::matrices::rotation_y(theta)
    }

    /// Lookup RZ matrix with all caching strategies
    #[inline]
    pub fn rz(theta: f64) -> [[Complex64; 2]; 2] {
        if let Some(matrix) = crate::compile_time_cache::CommonAngles::rz_lookup(theta) {
            return matrix;
        }

        if let Some(matrix) = GeneratedAngleCache::rz_clifford_t(theta) {
            return matrix;
        }

        if theta.abs() <= crate::compile_time_cache::VQEAngles::MAX_ANGLE {
            return crate::compile_time_cache::VQEAngles::rz_cached(theta);
        }

        use std::f64::consts::PI;
        if (0.0..=PI).contains(&theta) {
            return GeneratedAngleCache::rz_qaoa(theta);
        }

        crate::matrices::rotation_z(theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_clifford_t_angles() {
        // Test π/8
        let cached = GeneratedAngleCache::rx_clifford_t(PI / 8.0).unwrap();
        let computed = crate::matrices::rotation_x(PI / 8.0);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(cached[i][j].re, computed[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(cached[i][j].im, computed[i][j].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_qaoa_cache() {
        let angle = PI / 2.0;
        let cached = GeneratedAngleCache::rx_qaoa(angle);
        let computed = crate::matrices::rotation_x(angle);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(cached[i][j].re, computed[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(cached[i][j].im, computed[i][j].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_enhanced_universal_cache() {
        // Should find π/4 in common angles
        let matrix1 = EnhancedUniversalCache::rx(PI / 4.0);
        let expected1 = crate::matrices::rotation_x(PI / 4.0);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(matrix1[i][j].re, expected1[i][j].re, epsilon = 1e-10);
            }
        }

        // Should find π/8 in Clifford+T
        let matrix2 = EnhancedUniversalCache::rx(PI / 8.0);
        let expected2 = crate::matrices::rotation_x(PI / 8.0);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(matrix2[i][j].re, expected2[i][j].re, epsilon = 1e-10);
            }
        }

        // π/2 resolves through the common-angle level and must be exact
        let matrix3 = EnhancedUniversalCache::rx(PI / 2.0);
        let expected3 = crate::matrices::rotation_x(PI / 2.0);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(matrix3[i][j].re, expected3[i][j].re, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_pi_fractions() {
        let angle = PI / 3.0;
        let cached = GeneratedAngleCache::rx_pi_fraction(angle).unwrap();
        let computed = crate::matrices::rotation_x(angle);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(cached[i][j].re, computed[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(cached[i][j].im, computed[i][j].im, epsilon = 1e-10);
            }
        }
    }

    // --- Additional coverage for uncovered branches ---

    #[test]
    fn test_rx_clifford_t_all_branches() {
        // line 23: PI/2
        assert!(GeneratedAngleCache::rx_clifford_t(PI / 2.0).is_some());
        // existing test covers PI/4 and PI/8
        // line 29: PI/16
        assert!(GeneratedAngleCache::rx_clifford_t(PI / 16.0).is_some());
        // line 31: PI/32
        assert!(GeneratedAngleCache::rx_clifford_t(PI / 32.0).is_some());
        // None branch
        assert!(GeneratedAngleCache::rx_clifford_t(1.23456).is_none());
    }

    #[test]
    fn test_ry_clifford_t_all_branches() {
        // line 43: PI/2
        assert!(GeneratedAngleCache::ry_clifford_t(PI / 2.0).is_some());
        // line 45: PI/4
        assert!(GeneratedAngleCache::ry_clifford_t(PI / 4.0).is_some());
        // line 47: PI/8 (already similar tests elsewhere, include for safety)
        assert!(GeneratedAngleCache::ry_clifford_t(PI / 8.0).is_some());
        // line 49: PI/16
        assert!(GeneratedAngleCache::ry_clifford_t(PI / 16.0).is_some());
        // line 51: PI/32
        assert!(GeneratedAngleCache::ry_clifford_t(PI / 32.0).is_some());
        // None
        assert!(GeneratedAngleCache::ry_clifford_t(1.23456).is_none());
    }

    #[test]
    fn test_rz_clifford_t_all_branches() {
        // line 63: PI/2
        assert!(GeneratedAngleCache::rz_clifford_t(PI / 2.0).is_some());
        // line 65: PI/4
        assert!(GeneratedAngleCache::rz_clifford_t(PI / 4.0).is_some());
        // line 67: PI/8
        assert!(GeneratedAngleCache::rz_clifford_t(PI / 8.0).is_some());
        // line 69: PI/16
        assert!(GeneratedAngleCache::rz_clifford_t(PI / 16.0).is_some());
        // line 71: PI/32
        assert!(GeneratedAngleCache::rz_clifford_t(PI / 32.0).is_some());
        // None
        assert!(GeneratedAngleCache::rz_clifford_t(1.23456).is_none());
    }

    #[test]
    fn test_rx_qaoa_out_of_range() {
        // line 84: theta outside [0, PI] → direct computation
        let neg_angle = -0.5;
        let cached = GeneratedAngleCache::rx_qaoa(neg_angle);
        let computed = crate::matrices::rotation_x(neg_angle);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(cached[i][j].re, computed[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(cached[i][j].im, computed[i][j].im, epsilon = 1e-10);
            }
        }
        // Also test > PI
        let big_angle = PI + 0.5;
        let cached2 = GeneratedAngleCache::rx_qaoa(big_angle);
        let computed2 = crate::matrices::rotation_x(big_angle);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(cached2[i][j].re, computed2[i][j].re, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_rz_qaoa_out_of_range() {
        // line 99: theta outside [0, PI] → direct computation
        let neg_angle = -0.5;
        let cached = GeneratedAngleCache::rz_qaoa(neg_angle);
        let computed = crate::matrices::rotation_z(neg_angle);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(cached[i][j].re, computed[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(cached[i][j].im, computed[i][j].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_rx_pi_fraction_more_branches() {
        // line 114: PI/2
        assert!(GeneratedAngleCache::rx_pi_fraction(PI / 2.0).is_some());
        // line 118: PI/4
        assert!(GeneratedAngleCache::rx_pi_fraction(PI / 4.0).is_some());
        // line 124: PI/8
        assert!(GeneratedAngleCache::rx_pi_fraction(PI / 8.0).is_some());
        // PI/5
        assert!(GeneratedAngleCache::rx_pi_fraction(PI / 5.0).is_some());
        // PI/6
        assert!(GeneratedAngleCache::rx_pi_fraction(PI / 6.0).is_some());
        // PI/10
        assert!(GeneratedAngleCache::rx_pi_fraction(PI / 10.0).is_some());
        // PI/12
        assert!(GeneratedAngleCache::rx_pi_fraction(PI / 12.0).is_some());
        // None branch
        assert!(GeneratedAngleCache::rx_pi_fraction(1.23456).is_none());
    }

    #[test]
    fn test_enhanced_cache_ry_clifford_t() {
        // line 157: ry uses clifford_t cache for PI/8
        let matrix = EnhancedUniversalCache::ry(PI / 8.0);
        let expected = crate::matrices::rotation_y(PI / 8.0);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(matrix[i][j].re, expected[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(matrix[i][j].im, expected[i][j].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_enhanced_cache_ry_runtime() {
        // line 188: ry falls through to runtime computation (large angle outside VQE range)
        let large_angle = 100.0;
        let matrix = EnhancedUniversalCache::ry(large_angle);
        let expected = crate::matrices::rotation_y(large_angle);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(matrix[i][j].re, expected[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(matrix[i][j].im, expected[i][j].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_enhanced_cache_rz_qaoa_branch() {
        // rz in QAOA range (not in common angles or clifford_t, not in VQE, in [0,PI])
        // must now be exact: the QAOA level either hits a grid point exactly or
        // falls back to runtime computation.
        let angle = PI * 0.7; // 0.7π — not a standard Clifford+T angle
        let matrix = EnhancedUniversalCache::rz(angle);
        let expected = crate::matrices::rotation_z(angle);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(matrix[i][j].re, expected[i][j].re, epsilon = 1e-12);
                assert_relative_eq!(matrix[i][j].im, expected[i][j].im, epsilon = 1e-12);
            }
        }
    }

    // --- Regression tests for issue #37: QAOA cache must never snap angles ---

    /// Max element-wise absolute difference between two 2x2 complex matrices
    fn max_diff(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2]) -> f64 {
        let mut max = 0.0_f64;
        for i in 0..2 {
            for j in 0..2 {
                max = max.max((a[i][j] - b[i][j]).norm());
            }
        }
        max
    }

    #[test]
    fn test_issue_37_generic_angles_are_exact() {
        // Angles in (π/4, π] that miss every exact cache level used to be
        // snapped to a π/99 grid (up to ~1.6e-2 rad away). They must now be
        // computed exactly.
        for &theta in &[0.8, 1.0, 1.3, 2.0, 2.71, 3.0] {
            let rz = GeneratedAngleCache::rz_qaoa(theta);
            let rz_exact = crate::matrices::rotation_z(theta);
            assert!(
                max_diff(&rz, &rz_exact) < 1e-14,
                "rz_qaoa({theta}) deviates from exact matrix by {}",
                max_diff(&rz, &rz_exact)
            );

            let rx = GeneratedAngleCache::rx_qaoa(theta);
            let rx_exact = crate::matrices::rotation_x(theta);
            assert!(
                max_diff(&rx, &rx_exact) < 1e-14,
                "rx_qaoa({theta}) deviates from exact matrix by {}",
                max_diff(&rx, &rx_exact)
            );

            // Same guarantee through the full dispatch chain
            assert!(max_diff(&EnhancedUniversalCache::rz(theta), &rz_exact) < 1e-14);
            assert!(max_diff(&EnhancedUniversalCache::rx(theta), &rx_exact) < 1e-14);
        }
    }

    #[test]
    fn test_issue_37_epsilon_perturbation_changes_matrix() {
        // Finite-difference gradients perturb parameters by ~1e-7. Before the
        // fix, θ and θ+ε mapped to the same grid entry and returned identical
        // matrices, making FD gradients identically zero.
        let theta = 0.8;
        let eps = 1e-7;

        let rz_a = EnhancedUniversalCache::rz(theta);
        let rz_b = EnhancedUniversalCache::rz(theta + eps);
        let diff = max_diff(&rz_a, &rz_b);
        assert!(
            diff > 1e-9,
            "RZ({theta}) and RZ({theta} + {eps}) returned (nearly) identical matrices (diff = {diff}); \
             finite-difference gradients would vanish"
        );

        let rx_a = EnhancedUniversalCache::rx(theta);
        let rx_b = EnhancedUniversalCache::rx(theta + eps);
        assert!(max_diff(&rx_a, &rx_b) > 1e-9);
    }

    #[test]
    fn test_issue_37_exact_grid_points_still_hit_cache() {
        // Angles exactly on the π/99 grid should agree with runtime
        // computation (build-time entries are computed with full f64 trig).
        for index in [1_usize, 25, 50, 98] {
            let theta = index as f64 * qaoa::ANGLE_STEP;
            let rz = GeneratedAngleCache::rz_qaoa(theta);
            let rz_exact = crate::matrices::rotation_z(theta);
            assert!(max_diff(&rz, &rz_exact) < 1e-12);

            let rx = GeneratedAngleCache::rx_qaoa(theta);
            let rx_exact = crate::matrices::rotation_x(theta);
            assert!(max_diff(&rx, &rx_exact) < 1e-12);
        }
    }
}
