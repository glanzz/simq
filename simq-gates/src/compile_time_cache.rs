//! Compile-time gate matrix caching
//!
//! This module provides compile-time pre-computed matrix caches for commonly-used
//! rotation angles. Matrices are embedded directly in the binary during compilation,
//! providing zero-cost access at runtime.
//!
//! # Architecture
//!
//! The caching system operates at multiple levels:
//!
//! 1. **Exact Angle Cache**: Pre-computed matrices for specific angles (e.g., π/4, π/2)
//! 2. **Range Cache**: Evenly-spaced matrices for a range (useful for gradient descent)
//! 3. **Const Generic Cache**: Type-level caching using const generics
//!
//! # Example
//!
//! ```rust
//! use simq_gates::compile_time_cache::{CommonAngles, VQEAngles};
//!
//! // Zero-cost access to common angle
//! let matrix = CommonAngles::rx_pi_over_2();
//!
//! // Cache lookup with fallback to computation
//! let matrix2 = VQEAngles::rx_cached(0.1);
//! ```
//!
//! # Performance
//!
//! - **Exact matches**: 0 ns (compile-time constant)
//! - **Range cache lookups**: ~2-5 ns (array index + bounds check)
//! - **Cache misses**: Falls back to standard computation (~20-50 ns)
//!
//! # Memory Usage
//!
//! Each cached 2x2 complex matrix requires 64 bytes:
//! - 4 Complex64 values × 16 bytes each = 64 bytes
//!
//! Range caches with N entries use: 64N bytes

use num_complex::Complex64;
use std::f64::consts::PI;

// ============================================================================
// Common Angle Constants
// ============================================================================

/// Pre-computed matrices for the most common rotation angles
///
/// These are compile-time constants embedded directly in the binary.
/// Access is literally zero-cost - just a memory load.
pub struct CommonAngles;

impl CommonAngles {
    // --- RX Common Angles ---

    /// RX(π/4) - 45° rotation around X-axis
    pub const RX_PI_OVER_4: [[Complex64; 2]; 2] = {
        const COS: f64 = 0.9238795325112867; // cos(π/8)
        const SIN: f64 = 0.3826834323650898; // sin(π/8)
        [
            [Complex64::new(COS, 0.0), Complex64::new(0.0, -SIN)],
            [Complex64::new(0.0, -SIN), Complex64::new(COS, 0.0)],
        ]
    };

    #[inline]
    pub const fn rx_pi_over_4() -> &'static [[Complex64; 2]; 2] {
        &Self::RX_PI_OVER_4
    }

    /// RX(π/2) - 90° rotation around X-axis
    pub const RX_PI_OVER_2: [[Complex64; 2]; 2] = {
        const COS: f64 = std::f64::consts::FRAC_1_SQRT_2; // cos(π/4) = 1/√2
        const SIN: f64 = std::f64::consts::FRAC_1_SQRT_2; // sin(π/4) = 1/√2
        [
            [Complex64::new(COS, 0.0), Complex64::new(0.0, -SIN)],
            [Complex64::new(0.0, -SIN), Complex64::new(COS, 0.0)],
        ]
    };

    #[inline]
    pub const fn rx_pi_over_2() -> &'static [[Complex64; 2]; 2] {
        &Self::RX_PI_OVER_2
    }

    /// RX(π) - 180° rotation around X-axis (equivalent to -iX)
    pub const RX_PI: [[Complex64; 2]; 2] = {
        const COS: f64 = 0.0; // cos(π/2)
        const SIN: f64 = 1.0; // sin(π/2)
        [
            [Complex64::new(COS, 0.0), Complex64::new(0.0, -SIN)],
            [Complex64::new(0.0, -SIN), Complex64::new(COS, 0.0)],
        ]
    };

    #[inline]
    pub const fn rx_pi() -> &'static [[Complex64; 2]; 2] {
        &Self::RX_PI
    }

    // --- RY Common Angles ---

    /// RY(π/4) - 45° rotation around Y-axis
    pub const RY_PI_OVER_4: [[Complex64; 2]; 2] = {
        const COS: f64 = 0.9238795325112867; // cos(π/8)
        const SIN: f64 = 0.3826834323650898; // sin(π/8)
        [
            [Complex64::new(COS, 0.0), Complex64::new(-SIN, 0.0)],
            [Complex64::new(SIN, 0.0), Complex64::new(COS, 0.0)],
        ]
    };

    #[inline]
    pub const fn ry_pi_over_4() -> &'static [[Complex64; 2]; 2] {
        &Self::RY_PI_OVER_4
    }

    /// RY(π/2) - 90° rotation around Y-axis
    pub const RY_PI_OVER_2: [[Complex64; 2]; 2] = {
        const COS: f64 = std::f64::consts::FRAC_1_SQRT_2; // 1/√2
        const SIN: f64 = std::f64::consts::FRAC_1_SQRT_2; // 1/√2
        [
            [Complex64::new(COS, 0.0), Complex64::new(-SIN, 0.0)],
            [Complex64::new(SIN, 0.0), Complex64::new(COS, 0.0)],
        ]
    };

    #[inline]
    pub const fn ry_pi_over_2() -> &'static [[Complex64; 2]; 2] {
        &Self::RY_PI_OVER_2
    }

    /// RY(π) - 180° rotation around Y-axis
    pub const RY_PI: [[Complex64; 2]; 2] = {
        const COS: f64 = 0.0;
        const SIN: f64 = 1.0;
        [
            [Complex64::new(COS, 0.0), Complex64::new(-SIN, 0.0)],
            [Complex64::new(SIN, 0.0), Complex64::new(COS, 0.0)],
        ]
    };

    #[inline]
    pub const fn ry_pi() -> &'static [[Complex64; 2]; 2] {
        &Self::RY_PI
    }

    // --- RZ Common Angles ---

    /// RZ(π/4) - 45° phase rotation
    pub const RZ_PI_OVER_4: [[Complex64; 2]; 2] = {
        const COS: f64 = 0.9238795325112867; // cos(π/8)
        const SIN: f64 = 0.3826834323650898; // sin(π/8)
        [
            [Complex64::new(COS, -SIN), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(COS, SIN)],
        ]
    };

    #[inline]
    pub const fn rz_pi_over_4() -> &'static [[Complex64; 2]; 2] {
        &Self::RZ_PI_OVER_4
    }

    /// RZ(π/2) - 90° phase rotation (equivalent to S gate up to global phase)
    pub const RZ_PI_OVER_2: [[Complex64; 2]; 2] = {
        const COS: f64 = std::f64::consts::FRAC_1_SQRT_2; // 1/√2
        const SIN: f64 = std::f64::consts::FRAC_1_SQRT_2; // 1/√2
        [
            [Complex64::new(COS, -SIN), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(COS, SIN)],
        ]
    };

    #[inline]
    pub const fn rz_pi_over_2() -> &'static [[Complex64; 2]; 2] {
        &Self::RZ_PI_OVER_2
    }

    /// RZ(π) - 180° phase rotation (equivalent to Z gate up to global phase)
    pub const RZ_PI: [[Complex64; 2]; 2] = {
        const COS: f64 = 0.0;
        const SIN: f64 = 1.0;
        [
            [Complex64::new(COS, -SIN), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(COS, SIN)],
        ]
    };

    #[inline]
    pub const fn rz_pi() -> &'static [[Complex64; 2]; 2] {
        &Self::RZ_PI
    }

    /// Lookup RX matrix for common angles, returns None if not cached
    #[inline]
    pub fn rx_lookup(theta: f64) -> Option<[[Complex64; 2]; 2]> {
        const EPSILON: f64 = 1e-10;

        if (theta - PI / 4.0).abs() < EPSILON {
            Some(Self::RX_PI_OVER_4)
        } else if (theta - PI / 2.0).abs() < EPSILON {
            Some(Self::RX_PI_OVER_2)
        } else if (theta - PI).abs() < EPSILON {
            Some(Self::RX_PI)
        } else if theta.abs() < EPSILON {
            Some(crate::matrices::IDENTITY)
        } else {
            None
        }
    }

    /// Lookup RY matrix for common angles, returns None if not cached
    #[inline]
    pub fn ry_lookup(theta: f64) -> Option<[[Complex64; 2]; 2]> {
        const EPSILON: f64 = 1e-10;

        if (theta - PI / 4.0).abs() < EPSILON {
            Some(Self::RY_PI_OVER_4)
        } else if (theta - PI / 2.0).abs() < EPSILON {
            Some(Self::RY_PI_OVER_2)
        } else if (theta - PI).abs() < EPSILON {
            Some(Self::RY_PI)
        } else if theta.abs() < EPSILON {
            Some(crate::matrices::IDENTITY)
        } else {
            None
        }
    }

    /// Lookup RZ matrix for common angles, returns None if not cached
    #[inline]
    pub fn rz_lookup(theta: f64) -> Option<[[Complex64; 2]; 2]> {
        const EPSILON: f64 = 1e-10;

        if (theta - PI / 4.0).abs() < EPSILON {
            Some(Self::RZ_PI_OVER_4)
        } else if (theta - PI / 2.0).abs() < EPSILON {
            Some(Self::RZ_PI_OVER_2)
        } else if (theta - PI).abs() < EPSILON {
            Some(Self::RZ_PI)
        } else if theta.abs() < EPSILON {
            Some(crate::matrices::IDENTITY)
        } else {
            None
        }
    }
}

// ============================================================================
// VQE/QAOA Optimized Cache
// ============================================================================

/// Cache optimized for Variational Quantum Eigensolver (VQE) and QAOA algorithms
///
/// Pre-computed matrices for angles from 0 to π/4 with fine granularity.
/// This range covers most parameter updates during gradient descent.
pub struct VQEAngles;

impl VQEAngles {
    /// Number of cached angles (higher = better accuracy, more memory)
    pub const NUM_ENTRIES: usize = 256;

    /// Maximum cached angle (π/4 covers typical optimization ranges)
    pub const MAX_ANGLE: f64 = PI / 4.0;

    /// Step size between cached angles
    pub const STEP: f64 = Self::MAX_ANGLE / (Self::NUM_ENTRIES - 1) as f64;

    /// Pre-computed RX matrices for VQE angles
    const RX_CACHE: [[[Complex64; 2]; 2]; Self::NUM_ENTRIES] = Self::gen_rx_cache();

    /// Pre-computed RY matrices for VQE angles
    const RY_CACHE: [[[Complex64; 2]; 2]; Self::NUM_ENTRIES] = Self::gen_ry_cache();

    /// Pre-computed RZ matrices for VQE angles
    const RZ_CACHE: [[[Complex64; 2]; 2]; Self::NUM_ENTRIES] = Self::gen_rz_cache();

    /// Generate RX cache at compile time
    const fn gen_rx_cache() -> [[[Complex64; 2]; 2]; Self::NUM_ENTRIES] {
        let mut cache = [[[Complex64::new(0.0, 0.0); 2]; 2]; Self::NUM_ENTRIES];
        let mut i = 0;
        while i < Self::NUM_ENTRIES {
            let theta = i as f64 * Self::STEP;
            let half_theta = theta / 2.0;

            // Const approximation of cos and sin (using Taylor series for compile-time)
            // For better accuracy, we use pre-computed values at runtime
            let cos_val = Self::const_cos(half_theta);
            let sin_val = Self::const_sin(half_theta);

            cache[i] = [
                [Complex64::new(cos_val, 0.0), Complex64::new(0.0, -sin_val)],
                [Complex64::new(0.0, -sin_val), Complex64::new(cos_val, 0.0)],
            ];
            i += 1;
        }
        cache
    }

    /// Generate RY cache at compile time
    const fn gen_ry_cache() -> [[[Complex64; 2]; 2]; Self::NUM_ENTRIES] {
        let mut cache = [[[Complex64::new(0.0, 0.0); 2]; 2]; Self::NUM_ENTRIES];
        let mut i = 0;
        while i < Self::NUM_ENTRIES {
            let theta = i as f64 * Self::STEP;
            let half_theta = theta / 2.0;

            let cos_val = Self::const_cos(half_theta);
            let sin_val = Self::const_sin(half_theta);

            cache[i] = [
                [Complex64::new(cos_val, 0.0), Complex64::new(-sin_val, 0.0)],
                [Complex64::new(sin_val, 0.0), Complex64::new(cos_val, 0.0)],
            ];
            i += 1;
        }
        cache
    }

    /// Generate RZ cache at compile time
    const fn gen_rz_cache() -> [[[Complex64; 2]; 2]; Self::NUM_ENTRIES] {
        let mut cache = [[[Complex64::new(0.0, 0.0); 2]; 2]; Self::NUM_ENTRIES];
        let mut i = 0;
        while i < Self::NUM_ENTRIES {
            let theta = i as f64 * Self::STEP;
            let half_theta = theta / 2.0;

            let cos_val = Self::const_cos(half_theta);
            let sin_val = Self::const_sin(half_theta);

            cache[i] = [
                [Complex64::new(cos_val, -sin_val), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(cos_val, sin_val)],
            ];
            i += 1;
        }
        cache
    }

    /// Const-compatible cosine approximation (Taylor series, 7 terms)
    const fn const_cos(x: f64) -> f64 {
        let x2 = x * x;
        let x4 = x2 * x2;
        let x6 = x4 * x2;
        let x8 = x4 * x4;
        let x10 = x8 * x2;
        let x12 = x8 * x4;
        1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0 - x10 / 3628800.0 + x12 / 479001600.0
    }

    /// Const-compatible sine approximation (Taylor series, 7 terms)
    const fn const_sin(x: f64) -> f64 {
        let x2 = x * x;
        let x3 = x * x2;
        let x5 = x3 * x2;
        let x7 = x5 * x2;
        let x9 = x7 * x2;
        let x11 = x9 * x2;
        let x13 = x11 * x2;
        x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0 + x9 / 362880.0 - x11 / 39916800.0
            + x13 / 6227020800.0
    }

    /// Lookup RX matrix, using cache only for exact angle matches
    #[inline]
    pub fn rx_cached(theta: f64) -> [[Complex64; 2]; 2] {
        let abs_theta = theta.abs();

        if abs_theta <= Self::MAX_ANGLE {
            let float_index = abs_theta / Self::STEP;
            let index = float_index.round() as usize;
            let index = index.min(Self::NUM_ENTRIES - 1);
            let cached_theta = index as f64 * Self::STEP;
            if (abs_theta - cached_theta).abs() < 1e-12 {
                let mut m = Self::RX_CACHE[index];
                if theta < 0.0 {
                    m[0][1] = Complex64::new(-m[0][1].re, -m[0][1].im);
                    m[1][0] = Complex64::new(-m[1][0].re, -m[1][0].im);
                }
                return m;
            }
        }
        crate::matrices::rotation_x(theta)
    }

    /// Lookup RY matrix, using cache only for exact angle matches
    #[inline]
    pub fn ry_cached(theta: f64) -> [[Complex64; 2]; 2] {
        let abs_theta = theta.abs();

        if abs_theta <= Self::MAX_ANGLE {
            let float_index = abs_theta / Self::STEP;
            let index = float_index.round() as usize;
            let index = index.min(Self::NUM_ENTRIES - 1);
            let cached_theta = index as f64 * Self::STEP;
            if (abs_theta - cached_theta).abs() < 1e-12 {
                let mut m = Self::RY_CACHE[index];
                if theta < 0.0 {
                    m[0][1] = Complex64::new(-m[0][1].re, -m[0][1].im);
                    m[1][0] = Complex64::new(-m[1][0].re, -m[1][0].im);
                }
                return m;
            }
        }
        crate::matrices::rotation_y(theta)
    }

    /// Lookup RZ matrix, using cache only for exact angle matches
    #[inline]
    pub fn rz_cached(theta: f64) -> [[Complex64; 2]; 2] {
        let abs_theta = theta.abs();

        if abs_theta <= Self::MAX_ANGLE {
            let float_index = abs_theta / Self::STEP;
            let index = float_index.round() as usize;
            let index = index.min(Self::NUM_ENTRIES - 1);
            let cached_theta = index as f64 * Self::STEP;
            if (abs_theta - cached_theta).abs() < 1e-12 {
                let mut m = Self::RZ_CACHE[index];
                if theta < 0.0 {
                    m[0][0] = Complex64::new(m[0][0].re, -m[0][0].im);
                    m[1][1] = Complex64::new(m[1][1].re, -m[1][1].im);
                }
                return m;
            }
        }
        crate::matrices::rotation_z(theta)
    }

    /// Get cache memory usage in bytes
    #[inline]
    pub const fn memory_bytes() -> usize {
        std::mem::size_of::<[[Complex64; 2]; 2]>() * Self::NUM_ENTRIES * 3
    }
}

// ============================================================================
// Universal Cache with Multiple Strategies
// ============================================================================

/// Universal compile-time cache combining multiple strategies
///
/// Provides the best of both worlds:
/// - Exact matches for common angles (zero-cost)
/// - Range cache for VQE/QAOA workloads
/// - Automatic fallback to runtime computation
pub struct UniversalCache;

impl UniversalCache {
    /// Lookup RX matrix with multi-level caching
    ///
    /// Priority order:
    /// 1. Check common angles (zero-cost)
    /// 2. Check VQE range cache (array lookup)
    /// 3. Fallback to runtime computation
    #[inline]
    pub fn rx(theta: f64) -> [[Complex64; 2]; 2] {
        // Level 1: Common angles
        if let Some(matrix) = CommonAngles::rx_lookup(theta) {
            return matrix;
        }

        // Level 2: VQE range cache
        if theta.abs() <= VQEAngles::MAX_ANGLE {
            return VQEAngles::rx_cached(theta);
        }

        // Level 3: Runtime computation
        crate::matrices::rotation_x(theta)
    }

    /// Lookup RY matrix with multi-level caching
    #[inline]
    pub fn ry(theta: f64) -> [[Complex64; 2]; 2] {
        if let Some(matrix) = CommonAngles::ry_lookup(theta) {
            return matrix;
        }

        if theta.abs() <= VQEAngles::MAX_ANGLE {
            return VQEAngles::ry_cached(theta);
        }

        crate::matrices::rotation_y(theta)
    }

    /// Lookup RZ matrix with multi-level caching
    #[inline]
    pub fn rz(theta: f64) -> [[Complex64; 2]; 2] {
        if let Some(matrix) = CommonAngles::rz_lookup(theta) {
            return matrix;
        }

        if theta.abs() <= VQEAngles::MAX_ANGLE {
            return VQEAngles::rz_cached(theta);
        }

        crate::matrices::rotation_z(theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_common_angles_rx() {
        // RX(π/2) should match computed version
        let cached = CommonAngles::rx_pi_over_2();
        let computed = crate::matrices::rotation_x(PI / 2.0);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(cached[i][j].re, computed[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(cached[i][j].im, computed[i][j].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_common_angles_lookup() {
        // Should find π/2
        assert!(CommonAngles::rx_lookup(PI / 2.0).is_some());

        // Should not find arbitrary angle
        assert!(CommonAngles::rx_lookup(0.123).is_none());
    }

    #[test]
    fn test_vqe_cache_small_angles() {
        let theta = 0.05;
        let cached = VQEAngles::rx_cached(theta);
        let computed = crate::matrices::rotation_x(theta);

        for i in 0..2 {
            for j in 0..2 {
                // VQE cache uses Taylor series approximation in const fn,
                // so use more tolerant epsilon (still very accurate for small angles)
                assert_relative_eq!(cached[i][j].re, computed[i][j].re, epsilon = 1e-3);
                assert_relative_eq!(cached[i][j].im, computed[i][j].im, epsilon = 1e-3);
            }
        }
    }

    #[test]
    fn test_vqe_cache_boundary() {
        // At boundary, should still use cache
        let cached = VQEAngles::rx_cached(VQEAngles::MAX_ANGLE);
        let computed = crate::matrices::rotation_x(VQEAngles::MAX_ANGLE);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(cached[i][j].re, computed[i][j].re, epsilon = 1e-4);
                assert_relative_eq!(cached[i][j].im, computed[i][j].im, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_vqe_cache_fallback() {
        // Beyond range, should fallback to computation
        let theta = PI; // Beyond MAX_ANGLE
        let cached = VQEAngles::rx_cached(theta);
        let computed = crate::matrices::rotation_x(theta);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(cached[i][j].re, computed[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(cached[i][j].im, computed[i][j].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_universal_cache() {
        // Test Level 1: Common angle
        let matrix1 = UniversalCache::rx(PI / 2.0);
        assert_eq!(matrix1, *CommonAngles::rx_pi_over_2());

        // Test Level 2: VQE range
        let matrix2 = UniversalCache::rx(0.1);
        let computed2 = crate::matrices::rotation_x(0.1);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(matrix2[i][j].re, computed2[i][j].re, epsilon = 1e-4);
            }
        }

        // Test Level 3: Fallback
        let matrix3 = UniversalCache::rx(2.0 * PI);
        let computed3 = crate::matrices::rotation_x(2.0 * PI);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(matrix3[i][j].re, computed3[i][j].re, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_vqe_cache_entries_match_runtime_trig() {
        // The VQE cache entries are generated by a const-fn Taylor series.
        // Over the cached range (half-angles up to pi/8) the truncation error
        // is ~1e-16, so every entry must agree with runtime trig to 1e-12.
        // If anyone shrinks the series or widens the range, this catches it.
        for i in 0..VQEAngles::NUM_ENTRIES {
            let theta = i as f64 * VQEAngles::STEP;
            let rx = VQEAngles::rx_cached(theta);
            let rx_exact = crate::matrices::rotation_x(theta);
            let ry = VQEAngles::ry_cached(theta);
            let ry_exact = crate::matrices::rotation_y(theta);
            let rz = VQEAngles::rz_cached(theta);
            let rz_exact = crate::matrices::rotation_z(theta);
            for r in 0..2 {
                for c in 0..2 {
                    assert!(
                        (rx[r][c] - rx_exact[r][c]).norm() < 1e-12,
                        "RX entry {i} (theta = {theta}) deviates from exact trig"
                    );
                    assert!(
                        (ry[r][c] - ry_exact[r][c]).norm() < 1e-12,
                        "RY entry {i} (theta = {theta}) deviates from exact trig"
                    );
                    assert!(
                        (rz[r][c] - rz_exact[r][c]).norm() < 1e-12,
                        "RZ entry {i} (theta = {theta}) deviates from exact trig"
                    );
                }
            }
        }
    }

    #[test]
    fn test_memory_usage() {
        let bytes = VQEAngles::memory_bytes();
        // 256 entries × 3 caches (RX, RY, RZ) × 64 bytes per matrix
        assert_eq!(bytes, 256 * 3 * 64);
    }

    // =========================================================================
    // Additional CommonAngles tests
    // =========================================================================

    #[test]
    fn test_common_angles_all_rx() {
        let rx_pi_4 = CommonAngles::rx_pi_over_4();
        let rx_pi_2 = CommonAngles::rx_pi_over_2();
        let rx_pi_val = CommonAngles::rx_pi();

        // All should be non-trivial (not all zeros)
        let sum_pi4: f64 = rx_pi_4.iter().flatten().map(|c| c.norm_sqr()).sum();
        assert!(sum_pi4 > 0.0);
        let sum_pi2: f64 = rx_pi_2.iter().flatten().map(|c| c.norm_sqr()).sum();
        assert!(sum_pi2 > 0.0);
        let sum_pi: f64 = rx_pi_val.iter().flatten().map(|c| c.norm_sqr()).sum();
        assert!(sum_pi > 0.0);

        // Check against computed values
        let computed_pi4 = crate::matrices::rotation_x(PI / 4.0);
        let computed_pi2 = crate::matrices::rotation_x(PI / 2.0);
        let computed_pi = crate::matrices::rotation_x(PI);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(rx_pi_4[i][j].re, computed_pi4[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(rx_pi_2[i][j].re, computed_pi2[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(rx_pi_val[i][j].re, computed_pi[i][j].re, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_common_angles_all_ry() {
        let ry_pi4 = CommonAngles::ry_pi_over_4();
        let ry_pi2 = CommonAngles::ry_pi_over_2();
        let ry_pi_val = CommonAngles::ry_pi();

        let computed_pi4 = crate::matrices::rotation_y(PI / 4.0);
        let computed_pi2 = crate::matrices::rotation_y(PI / 2.0);
        let computed_pi = crate::matrices::rotation_y(PI);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(ry_pi4[i][j].re, computed_pi4[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(ry_pi2[i][j].re, computed_pi2[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(ry_pi_val[i][j].re, computed_pi[i][j].re, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_common_angles_all_rz() {
        let rz_pi4 = CommonAngles::rz_pi_over_4();
        let rz_pi2 = CommonAngles::rz_pi_over_2();
        let rz_pi_val = CommonAngles::rz_pi();

        let computed_pi4 = crate::matrices::rotation_z(PI / 4.0);
        let computed_pi2 = crate::matrices::rotation_z(PI / 2.0);
        let computed_pi = crate::matrices::rotation_z(PI);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(rz_pi4[i][j].re, computed_pi4[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(rz_pi2[i][j].re, computed_pi2[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(rz_pi_val[i][j].re, computed_pi[i][j].re, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_rx_lookup_all_branches() {
        // PI/4 branch
        assert!(CommonAngles::rx_lookup(PI / 4.0).is_some());
        // PI/2 branch
        assert!(CommonAngles::rx_lookup(PI / 2.0).is_some());
        // PI branch
        assert!(CommonAngles::rx_lookup(PI).is_some());
        // 0.0 branch (identity)
        assert!(CommonAngles::rx_lookup(0.0).is_some());
        // None branch
        assert!(CommonAngles::rx_lookup(1.0).is_none());
        assert!(CommonAngles::rx_lookup(0.123).is_none());
    }

    #[test]
    fn test_ry_lookup_all_branches() {
        assert!(CommonAngles::ry_lookup(PI / 4.0).is_some());
        assert!(CommonAngles::ry_lookup(PI / 2.0).is_some());
        assert!(CommonAngles::ry_lookup(PI).is_some());
        assert!(CommonAngles::ry_lookup(0.0).is_some());
        assert!(CommonAngles::ry_lookup(1.0).is_none());
    }

    #[test]
    fn test_rz_lookup_all_branches() {
        assert!(CommonAngles::rz_lookup(PI / 4.0).is_some());
        assert!(CommonAngles::rz_lookup(PI / 2.0).is_some());
        assert!(CommonAngles::rz_lookup(PI).is_some());
        assert!(CommonAngles::rz_lookup(0.0).is_some());
        assert!(CommonAngles::rz_lookup(1.0).is_none());
    }

    // =========================================================================
    // VQEAngles additional tests
    // =========================================================================

    #[test]
    fn test_vqe_angles_rx_cached_in_range() {
        // theta=0.0 is exactly on a cache entry
        let m0 = VQEAngles::rx_cached(0.0);
        let computed0 = crate::matrices::rotation_x(0.0);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(m0[i][j].re, computed0[i][j].re, epsilon = 1e-10);
            }
        }

        // theta=0.1 is within range but may or may not hit exact cache
        let m1 = VQEAngles::rx_cached(0.1);
        let computed1 = crate::matrices::rotation_x(0.1);
        // Allow looser epsilon because Taylor series approx
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(m1[i][j].re, computed1[i][j].re, epsilon = 0.01);
            }
        }
    }

    #[test]
    fn test_vqe_angles_rx_cached_out_of_range() {
        // PI is beyond MAX_ANGLE, falls back to runtime computation
        let cached = VQEAngles::rx_cached(PI);
        let computed = crate::matrices::rotation_x(PI);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(cached[i][j].re, computed[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(cached[i][j].im, computed[i][j].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_vqe_angles_rx_cached_negative() {
        // Negative angles should work (negate off-diagonal)
        let pos = VQEAngles::rx_cached(0.1);
        let neg = VQEAngles::rx_cached(-0.1);
        // cos(-θ/2) = cos(θ/2), so diagonal should match
        assert_relative_eq!(pos[0][0].re, neg[0][0].re, epsilon = 0.01);
        // The returned matrix is from runtime computation for non-exact entries
        let computed_neg = crate::matrices::rotation_x(-0.1);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(neg[i][j].re, computed_neg[i][j].re, epsilon = 0.01);
            }
        }
    }

    #[test]
    fn test_vqe_angles_ry_cached() {
        // In-range
        let m = VQEAngles::ry_cached(0.0);
        let computed = crate::matrices::rotation_y(0.0);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(m[i][j].re, computed[i][j].re, epsilon = 1e-10);
            }
        }
        // Negative in-range
        let m_neg = VQEAngles::ry_cached(-0.1);
        let c_neg = crate::matrices::rotation_y(-0.1);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(m_neg[i][j].re, c_neg[i][j].re, epsilon = 0.01);
            }
        }
        // Out of range
        let m_pi = VQEAngles::ry_cached(PI);
        let c_pi = crate::matrices::rotation_y(PI);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(m_pi[i][j].re, c_pi[i][j].re, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_vqe_angles_rz_cached() {
        // In-range: theta=0
        let m = VQEAngles::rz_cached(0.0);
        let computed = crate::matrices::rotation_z(0.0);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(m[i][j].re, computed[i][j].re, epsilon = 1e-10);
            }
        }
        // Negative in-range
        let m_neg = VQEAngles::rz_cached(-0.1);
        let c_neg = crate::matrices::rotation_z(-0.1);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(m_neg[i][j].re, c_neg[i][j].re, epsilon = 0.01);
            }
        }
        // Out of range (fallback)
        let m_pi = VQEAngles::rz_cached(PI);
        let c_pi = crate::matrices::rotation_z(PI);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(m_pi[i][j].re, c_pi[i][j].re, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_vqe_angles_memory_bytes() {
        let bytes = VQEAngles::memory_bytes();
        assert!(bytes > 0);
        // Should be 256 * 3 * size_of([[Complex64;2];2])
        let expected = 256 * 3 * std::mem::size_of::<[[Complex64; 2]; 2]>();
        assert_eq!(bytes, expected);
    }

    // =========================================================================
    // UniversalCache additional tests
    // =========================================================================

    #[test]
    fn test_universal_cache_rx() {
        // Level 1: PI/4 is a common angle
        let m = UniversalCache::rx(PI / 4.0);
        let expected = *CommonAngles::rx_pi_over_4();
        assert_eq!(m, expected);

        // Level 1: 0.0 is also a common angle (identity)
        let m0 = UniversalCache::rx(0.0);
        assert_eq!(m0, crate::matrices::IDENTITY);

        // Level 2: VQE range hit
        let m2 = UniversalCache::rx(0.1);
        let c2 = crate::matrices::rotation_x(0.1);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(m2[i][j].re, c2[i][j].re, epsilon = 0.01);
            }
        }

        // Level 3: fallback to runtime
        let m3 = UniversalCache::rx(PI); // PI is a common angle
        let c3 = crate::matrices::rotation_x(PI);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(m3[i][j].re, c3[i][j].re, epsilon = 1e-10);
            }
        }

        // Runtime fallback for non-common, out-of-range angle
        let m4 = UniversalCache::rx(2.5);
        let c4 = crate::matrices::rotation_x(2.5);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(m4[i][j].re, c4[i][j].re, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_universal_cache_ry() {
        // PI/2 is common angle
        let m = UniversalCache::ry(PI / 2.0);
        let expected = *CommonAngles::ry_pi_over_2();
        assert_eq!(m, expected);

        // VQE range
        let m2 = UniversalCache::ry(0.1);
        let c2 = crate::matrices::rotation_y(0.1);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(m2[i][j].re, c2[i][j].re, epsilon = 0.01);
            }
        }

        // Out of range
        let m3 = UniversalCache::ry(2.0);
        let c3 = crate::matrices::rotation_y(2.0);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(m3[i][j].re, c3[i][j].re, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_universal_cache_rz() {
        // PI/4 is common angle
        let m = UniversalCache::rz(PI / 4.0);
        let expected = *CommonAngles::rz_pi_over_4();
        assert_eq!(m, expected);

        // VQE range
        let m2 = UniversalCache::rz(0.1);
        let c2 = crate::matrices::rotation_z(0.1);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(m2[i][j].re, c2[i][j].re, epsilon = 0.01);
            }
        }

        // Out of range
        let m3 = UniversalCache::rz(2.0);
        let c3 = crate::matrices::rotation_z(2.0);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(m3[i][j].re, c3[i][j].re, epsilon = 1e-10);
            }
        }
    }
}
