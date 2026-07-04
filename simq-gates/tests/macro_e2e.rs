mod rx_cache {
    simq_macros::cached_rotations!(RX, 0.0, 0.5, 1.0, 1.5707963267948966);
}

mod ry_cache {
    simq_macros::cached_rotations!(RY, 0.0, 1.0, 3.141592653589793);
}

mod rz_cache {
    simq_macros::cached_rotations!(RZ, 0.0, 0.5, 1.5707963267948966);
}

mod rx_range {
    simq_macros::cache_rotation_range!(RX, 0.0, 1.5707963267948966, 10);
}

mod ry_range {
    simq_macros::cache_rotation_range!(RY, 0.0, 3.141592653589793, 5);
}

mod rz_range {
    simq_macros::cache_rotation_range!(RZ, 0.0, 1.5707963267948966, 8);
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    const TOLERANCE: f64 = 1e-10;

    fn assert_matrix_close(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2], tol: f64) {
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (a[i][j] - b[i][j]).norm() < tol,
                    "Matrix mismatch at [{i}][{j}]: {:?} vs {:?}",
                    a[i][j],
                    b[i][j]
                );
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn is_unitary(m: &[[Complex64; 2]; 2]) -> bool {
        let identity = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let mut product = [[Complex64::new(0.0, 0.0); 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    product[i][j] += m[i][k] * m[j][k].conj();
                }
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                if (product[i][j] - identity[i][j]).norm() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    // ===== cached_rotations! tests =====

    #[test]
    fn test_rx_cache_num_cached() {
        assert_eq!(rx_cache::RXCache::num_cached(), 4);
    }

    #[test]
    fn test_ry_cache_num_cached() {
        assert_eq!(ry_cache::RYCache::num_cached(), 3);
    }

    #[test]
    fn test_rz_cache_num_cached() {
        assert_eq!(rz_cache::RZCache::num_cached(), 3);
    }

    #[test]
    fn test_rx_cache_is_cached() {
        assert!(rx_cache::RXCache::is_cached(0.0));
        assert!(rx_cache::RXCache::is_cached(0.5));
        assert!(rx_cache::RXCache::is_cached(1.0));
        assert!(!rx_cache::RXCache::is_cached(0.75));
    }

    #[test]
    fn test_rx_zero_angle_is_identity() {
        let m = rx_cache::RXCache::lookup(0.0);
        let identity = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        assert_matrix_close(&m, &identity, TOLERANCE);
    }

    #[test]
    fn test_ry_zero_angle_is_identity() {
        let m = ry_cache::RYCache::lookup(0.0);
        let identity = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        assert_matrix_close(&m, &identity, TOLERANCE);
    }

    #[test]
    fn test_rz_zero_angle_is_identity() {
        let m = rz_cache::RZCache::lookup(0.0);
        let identity = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        assert_matrix_close(&m, &identity, TOLERANCE);
    }

    #[test]
    fn test_rx_pi_half_cached() {
        let m = rx_cache::RXCache::lookup(std::f64::consts::FRAC_PI_2);
        assert!(is_unitary(&m));
        let cos_val = (std::f64::consts::FRAC_PI_4).cos();
        let sin_val = (std::f64::consts::FRAC_PI_4).sin();
        assert!((m[0][0].re - cos_val).abs() < TOLERANCE);
        assert!((m[0][1].im - (-sin_val)).abs() < TOLERANCE);
        assert!((m[1][0].im - (-sin_val)).abs() < TOLERANCE);
        assert!((m[1][1].re - cos_val).abs() < TOLERANCE);
    }

    #[test]
    fn test_ry_pi_cached() {
        let m = ry_cache::RYCache::lookup(std::f64::consts::PI);
        assert!(is_unitary(&m));
        assert!((m[0][0].re).abs() < TOLERANCE);
        assert!((m[0][1].re - (-1.0)).abs() < TOLERANCE);
        assert!((m[1][0].re - 1.0).abs() < TOLERANCE);
        assert!((m[1][1].re).abs() < TOLERANCE);
    }

    #[test]
    fn test_rx_uncached_angle_computes_correctly() {
        let theta: f64 = 0.75;
        let m = rx_cache::RXCache::lookup(theta);
        assert!(is_unitary(&m));
        let half: f64 = theta / 2.0;
        let expected = [
            [
                Complex64::new(half.cos(), 0.0),
                Complex64::new(0.0, -half.sin()),
            ],
            [
                Complex64::new(0.0, -half.sin()),
                Complex64::new(half.cos(), 0.0),
            ],
        ];
        assert_matrix_close(&m, &expected, TOLERANCE);
    }

    #[test]
    fn test_ry_uncached_angle_computes_correctly() {
        let theta: f64 = 2.0;
        let m = ry_cache::RYCache::lookup(theta);
        assert!(is_unitary(&m));
        let half: f64 = theta / 2.0;
        let expected = [
            [
                Complex64::new(half.cos(), 0.0),
                Complex64::new(-half.sin(), 0.0),
            ],
            [
                Complex64::new(half.sin(), 0.0),
                Complex64::new(half.cos(), 0.0),
            ],
        ];
        assert_matrix_close(&m, &expected, TOLERANCE);
    }

    #[test]
    fn test_rz_uncached_angle_computes_correctly() {
        let theta: f64 = 1.0;
        let m = rz_cache::RZCache::lookup(theta);
        assert!(is_unitary(&m));
        let half: f64 = theta / 2.0;
        let expected = [
            [
                Complex64::new(half.cos(), -half.sin()),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(half.cos(), half.sin()),
            ],
        ];
        assert_matrix_close(&m, &expected, TOLERANCE);
    }

    #[test]
    fn test_rx_all_cached_are_unitary() {
        for &angle in &[0.0, 0.5, 1.0, std::f64::consts::FRAC_PI_2] {
            let m = rx_cache::RXCache::lookup(angle);
            assert!(is_unitary(&m), "RX({angle}) not unitary");
        }
    }

    #[test]
    fn test_convenience_function_rx() {
        let m1 = rx_cache::RXCache::lookup(0.5);
        let m2 = rx_cache::rx_cached(0.5);
        assert_matrix_close(&m1, &m2, TOLERANCE);
    }

    #[test]
    fn test_convenience_function_ry() {
        let m1 = ry_cache::RYCache::lookup(1.0);
        let m2 = ry_cache::ry_cached(1.0);
        assert_matrix_close(&m1, &m2, TOLERANCE);
    }

    #[test]
    fn test_convenience_function_rz() {
        let m1 = rz_cache::RZCache::lookup(0.5);
        let m2 = rz_cache::rz_cached(0.5);
        assert_matrix_close(&m1, &m2, TOLERANCE);
    }

    // ===== cache_rotation_range! tests =====

    #[test]
    fn test_rx_range_num_cached() {
        assert_eq!(rx_range::RXRangeCache::num_cached(), 10);
    }

    #[test]
    fn test_ry_range_num_cached() {
        assert_eq!(ry_range::RYRangeCache::num_cached(), 5);
    }

    #[test]
    fn test_rz_range_num_cached() {
        assert_eq!(rz_range::RZRangeCache::num_cached(), 8);
    }

    #[test]
    fn test_rx_range_memory_bytes() {
        let bytes = rx_range::RXRangeCache::memory_bytes();
        assert_eq!(bytes, std::mem::size_of::<[[Complex64; 2]; 2]>() * 10);
    }

    #[test]
    fn test_rx_range_start_is_identity() {
        let m = rx_range::RXRangeCache::lookup(0.0);
        let identity = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        assert_matrix_close(&m, &identity, TOLERANCE);
    }

    #[test]
    fn test_rx_range_end_angle() {
        let m = rx_range::RXRangeCache::lookup(std::f64::consts::FRAC_PI_2);
        assert!(is_unitary(&m));
    }

    #[test]
    fn test_rx_range_all_unitary() {
        let step = std::f64::consts::FRAC_PI_2 / 9.0;
        for i in 0..10 {
            let angle = step * i as f64;
            let m = rx_range::RXRangeCache::lookup(angle);
            assert!(is_unitary(&m), "RX range({angle}) not unitary");
        }
    }

    #[test]
    fn test_rx_range_outside_range_computes() {
        let m = rx_range::RXRangeCache::lookup(3.0);
        assert!(is_unitary(&m));
        let half: f64 = 1.5;
        let expected = [
            [
                Complex64::new(half.cos(), 0.0),
                Complex64::new(0.0, -half.sin()),
            ],
            [
                Complex64::new(0.0, -half.sin()),
                Complex64::new(half.cos(), 0.0),
            ],
        ];
        assert_matrix_close(&m, &expected, TOLERANCE);
    }

    #[test]
    fn test_rx_range_negative_outside_computes() {
        let m = rx_range::RXRangeCache::lookup(-1.0);
        assert!(is_unitary(&m));
    }

    #[test]
    fn test_rx_range_interpolated_start() {
        let m = rx_range::RXRangeCache::lookup_interpolated(0.0);
        let identity = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        assert_matrix_close(&m, &identity, TOLERANCE);
    }

    #[test]
    fn test_rx_range_interpolated_outside() {
        let m = rx_range::RXRangeCache::lookup_interpolated(5.0);
        assert!(is_unitary(&m));
    }

    #[test]
    fn test_rx_range_interpolated_end() {
        let m = rx_range::RXRangeCache::lookup_interpolated(std::f64::consts::FRAC_PI_2);
        assert!(is_unitary(&m));
    }

    #[test]
    fn test_ry_range_lookup() {
        let m = ry_range::RYRangeCache::lookup(1.0);
        assert!(is_unitary(&m));
    }

    #[test]
    fn test_rz_range_lookup() {
        let m = rz_range::RZRangeCache::lookup(0.5);
        assert!(is_unitary(&m));
    }

    #[test]
    fn test_range_convenience_function_rx() {
        let m1 = rx_range::RXRangeCache::lookup(0.5);
        let m2 = rx_range::rx_range_cached(0.5);
        assert_matrix_close(&m1, &m2, TOLERANCE);
    }

    #[test]
    fn test_range_convenience_function_ry() {
        let m1 = ry_range::RYRangeCache::lookup(1.0);
        let m2 = ry_range::ry_range_cached(1.0);
        assert_matrix_close(&m1, &m2, TOLERANCE);
    }

    #[test]
    fn test_range_convenience_function_rz() {
        let m1 = rz_range::RZRangeCache::lookup(0.5);
        let m2 = rz_range::rz_range_cached(0.5);
        assert_matrix_close(&m1, &m2, TOLERANCE);
    }

    // ===== Cross-validation with simq-gates =====

    #[test]
    fn test_rx_cache_matches_simq_gates() {
        use simq_gates::matrices::{rotation_x, rotation_y, rotation_z};

        for &angle in &[0.0, 0.5, 1.0, std::f64::consts::FRAC_PI_2] {
            let cached = rx_cache::RXCache::lookup(angle);
            let reference = rotation_x(angle);
            assert_matrix_close(&cached, &reference, TOLERANCE);
        }

        for &angle in &[0.0, 1.0, std::f64::consts::PI] {
            let cached = ry_cache::RYCache::lookup(angle);
            let reference = rotation_y(angle);
            assert_matrix_close(&cached, &reference, TOLERANCE);
        }

        for &angle in &[0.0, 0.5, std::f64::consts::FRAC_PI_2] {
            let cached = rz_cache::RZCache::lookup(angle);
            let reference = rotation_z(angle);
            assert_matrix_close(&cached, &reference, TOLERANCE);
        }
    }

    #[test]
    fn test_rx_uncached_matches_simq_gates() {
        use simq_gates::matrices::rotation_x;

        for angle in [0.123, 0.789, 2.345, -1.0, 5.5] {
            let cached = rx_cache::RXCache::lookup(angle);
            let reference = rotation_x(angle);
            assert_matrix_close(&cached, &reference, TOLERANCE);
        }
    }

    #[test]
    fn test_ry_uncached_matches_simq_gates() {
        use simq_gates::matrices::rotation_y;

        for angle in [0.456, 2.0, -0.5] {
            let cached = ry_cache::RYCache::lookup(angle);
            let reference = rotation_y(angle);
            assert_matrix_close(&cached, &reference, TOLERANCE);
        }
    }

    #[test]
    fn test_rz_uncached_matches_simq_gates() {
        use simq_gates::matrices::rotation_z;

        for angle in [0.25, 1.0, 3.0] {
            let cached = rz_cache::RZCache::lookup(angle);
            let reference = rotation_z(angle);
            assert_matrix_close(&cached, &reference, TOLERANCE);
        }
    }

    #[test]
    fn test_range_rx_matches_simq_gates() {
        use simq_gates::matrices::rotation_x;

        // lookup() must be exact for EVERY angle: in-range angles that miss
        // the grid fall back to on-demand computation instead of being
        // snapped to the nearest cached entry (same bug class as issue #37).
        for angle in [0.0, 0.3, 0.7, 1.2, std::f64::consts::FRAC_PI_2, 3.0, -1.0] {
            let range_result = rx_range::RXRangeCache::lookup(angle);
            let reference = rotation_x(angle);
            assert_matrix_close(&range_result, &reference, TOLERANCE);
        }
    }

    #[test]
    fn test_range_lookup_never_snaps_in_range_angles() {
        use simq_gates::matrices::{rotation_x, rotation_y, rotation_z};

        // Angles inside the cached range but off the grid: exact to 1e-12
        let m = rx_range::RXRangeCache::lookup(0.7);
        assert_matrix_close(&m, &rotation_x(0.7), 1e-12);

        let m = ry_range::RYRangeCache::lookup(1.9);
        assert_matrix_close(&m, &rotation_y(1.9), 1e-12);

        let m = rz_range::RZRangeCache::lookup(0.6);
        assert_matrix_close(&m, &rotation_z(0.6), 1e-12);

        // Angles exactly on the grid still hit the cache and agree with
        // runtime computation
        let step = std::f64::consts::FRAC_PI_2 / 9.0;
        for i in 0..10 {
            let angle = step * i as f64;
            let m = rx_range::RXRangeCache::lookup(angle);
            assert_matrix_close(&m, &rotation_x(angle), 1e-12);
        }
    }
}
