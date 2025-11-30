//! Auto-generated compile-time gate matrices
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

    /// Lookup RX matrix for QAOA mixer angles using nearest neighbor
    #[inline]
    pub fn rx_qaoa(theta: f64) -> [[Complex64; 2]; 2] {
        use std::f64::consts::PI;

        if !(0.0..=PI).contains(&theta) {
            // Out of range - compute directly
            return crate::matrices::rotation_x(theta);
        }

        let index = (theta / qaoa::ANGLE_STEP).round() as usize;
        let index = index.min(qaoa::NUM_QAOA_ANGLES - 1);

        qaoa::RX_MIXER[index]
    }

    /// Lookup RZ matrix for QAOA cost angles using nearest neighbor
    #[inline]
    pub fn rz_qaoa(theta: f64) -> [[Complex64; 2]; 2] {
        use std::f64::consts::PI;

        if !(0.0..=PI).contains(&theta) {
            return crate::matrices::rotation_z(theta);
        }

        let index = (theta / qaoa::ANGLE_STEP).round() as usize;
        let index = index.min(qaoa::NUM_QAOA_ANGLES - 1);

        qaoa::RZ_COST[index]
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
                // QAOA uses nearest neighbor, so allow some error
                assert_relative_eq!(cached[i][j].re, computed[i][j].re, epsilon = 1e-2);
                assert_relative_eq!(cached[i][j].im, computed[i][j].im, epsilon = 1e-2);
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

        // Should use QAOA cache for π/2
        let matrix3 = EnhancedUniversalCache::rx(PI / 2.0);
        let expected3 = crate::matrices::rotation_x(PI / 2.0);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(matrix3[i][j].re, expected3[i][j].re, epsilon = 1e-2);
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
}
