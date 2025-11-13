//! Low-level SIMD kernels for common operations

#[cfg(target_arch = "x86_64")]
use num_complex::Complex64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Compute norm using SSE2
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn norm_sse2(vec: &[Complex64]) -> f64 {
    let mut sum = _mm_setzero_pd();
    let len = vec.len();
    let ptr = vec.as_ptr() as *const f64;

    let mut i = 0;
    while i + 1 <= len {
        // Load complex number [re, im]
        let z = _mm_loadu_pd(ptr.add(i * 2));

        // Square: [re*re, im*im]
        let z_sq = _mm_mul_pd(z, z);

        // Accumulate
        sum = _mm_add_pd(sum, z_sq);

        i += 1;
    }

    // Horizontal add
    let sum_array = [0.0f64; 2];
    _mm_storeu_pd(sum_array.as_ptr() as *mut f64, sum);

    (sum_array[0] + sum_array[1]).sqrt()
}

/// Compute norm using AVX2
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn norm_avx2(vec: &[Complex64]) -> f64 {
    let mut sum = _mm256_setzero_pd();
    let len = vec.len();
    let ptr = vec.as_ptr() as *const f64;

    // Process 2 complex numbers at a time (4 f64 values)
    let mut i = 0;
    while i + 2 <= len {
        // Load 2 complex numbers: [re0, im0, re1, im1]
        let z = _mm256_loadu_pd(ptr.add(i * 2));

        // Square: [re0*re0, im0*im0, re1*re1, im1*im1]
        let z_sq = _mm256_mul_pd(z, z);

        // Accumulate
        sum = _mm256_add_pd(sum, z_sq);

        i += 2;
    }

    // Handle remaining element
    let mut scalar_sum = 0.0;
    while i < len {
        scalar_sum += vec[i].norm_sqr();
        i += 1;
    }

    // Horizontal add
    let sum_array = [0.0f64; 4];
    _mm256_storeu_pd(sum_array.as_ptr() as *mut f64, sum);

    (sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + scalar_sum).sqrt()
}

/// Scale vector by scalar using AVX2
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn scale_avx2(vec: &mut [Complex64], scalar: f64) {
    let scale = _mm256_set1_pd(scalar);
    let len = vec.len();
    let ptr = vec.as_mut_ptr() as *mut f64;

    // Process 2 complex numbers at a time
    let mut i = 0;
    while i + 2 <= len {
        let z = _mm256_loadu_pd(ptr.add(i * 2));
        let scaled = _mm256_mul_pd(z, scale);
        _mm256_storeu_pd(ptr.add(i * 2), scaled);
        i += 2;
    }

    // Handle remaining element
    while i < len {
        vec[i] *= scalar;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_norm_sse2() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        let vec = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 1.0),
        ];

        unsafe {
            let norm = norm_sse2(&vec);
            assert_relative_eq!(norm, 2.0, epsilon = 1e-10);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_norm_avx2() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let vec = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
        ];

        unsafe {
            let norm = norm_avx2(&vec);
            assert_relative_eq!(norm, 2.0, epsilon = 1e-10);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_scale_avx2() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let mut vec = vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(2.0, 2.0),
        ];

        unsafe {
            scale_avx2(&mut vec, 0.5);
        }

        assert_relative_eq!(vec[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(vec[1].im, 1.0, epsilon = 1e-10);
        assert_relative_eq!(vec[2].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(vec[2].im, 1.0, epsilon = 1e-10);
    }
}
