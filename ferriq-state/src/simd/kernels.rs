//! Low-level SIMD kernels for common operations

use num_complex::Complex64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Compute norm using SSE2
///
/// # Safety
/// Requires SSE2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn norm_sse2(vec: &[Complex64]) -> f64 {
    let mut sum = _mm_setzero_pd();
    let len = vec.len();
    let ptr = vec.as_ptr() as *const f64;

    let mut i = 0;
    while i < len {
        // Load complex number [re, im]
        let z = _mm_loadu_pd(ptr.add(i * 2));

        // Square: [re*re, im*im]
        let z_sq = _mm_mul_pd(z, z);

        // Accumulate
        sum = _mm_add_pd(sum, z_sq);

        i += 1;
    }

    // Horizontal add
    let mut sum_array = [0.0f64; 2];
    _mm_storeu_pd(sum_array.as_mut_ptr(), sum);

    (sum_array[0] + sum_array[1]).sqrt()
}

/// Compute norm using AVX2
///
/// # Safety
/// Requires AVX2 CPU support.
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
    let mut sum_array = [0.0f64; 4];
    _mm256_storeu_pd(sum_array.as_mut_ptr(), sum);

    (sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + scalar_sum).sqrt()
}

/// Scale vector by scalar using AVX2
///
/// # Safety
/// Requires AVX2 CPU support.
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

/// Compute probability distribution (|amplitude|^2 for each amplitude) using AVX2
///
/// This is optimized for computing all probabilities at once, which is more efficient
/// than computing them individually when sampling.
///
/// # Safety
/// Requires AVX2 support
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn compute_probabilities_avx2(amplitudes: &[Complex64], output: &mut [f64]) {
    let len = amplitudes.len();
    assert_eq!(len, output.len(), "Output buffer must match input length");

    let amp_ptr = amplitudes.as_ptr() as *const f64;
    let out_ptr = output.as_mut_ptr();

    let mut i = 0;

    // Process 2 complex numbers at a time (4 f64 values)
    while i + 2 <= len {
        // Load 2 complex numbers: [re0, im0, re1, im1]
        let z = _mm256_loadu_pd(amp_ptr.add(i * 2));

        // Square: [re0*re0, im0*im0, re1*re1, im1*im1]
        let z_sq = _mm256_mul_pd(z, z);

        // Horizontal add pairs: [re0^2 + im0^2, re1^2 + im1^2, _, _]
        // We need to add adjacent pairs: (0,1) and (2,3)
        let shuffled = _mm256_permute_pd(z_sq, 0b0101); // Swap pairs within 128-bit lanes
        let summed_128 = _mm256_hadd_pd(z_sq, shuffled);

        // Extract results
        let mut temp = [0.0f64; 4];
        _mm256_storeu_pd(temp.as_mut_ptr(), summed_128);

        // Store norm squared values
        *out_ptr.add(i) = temp[0];
        *out_ptr.add(i + 1) = temp[2];

        i += 2;
    }

    // Handle remaining elements
    while i < len {
        output[i] = amplitudes[i].norm_sqr();
        i += 1;
    }
}

/// Compute probability distribution using SSE2 (fallback for older CPUs)
///
/// # Safety
/// Requires SSE2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn compute_probabilities_sse2(amplitudes: &[Complex64], output: &mut [f64]) {
    let len = amplitudes.len();
    assert_eq!(len, output.len(), "Output buffer must match input length");

    let amp_ptr = amplitudes.as_ptr() as *const f64;
    let out_ptr = output.as_mut_ptr();

    let mut i = 0;

    // Process 1 complex number at a time (2 f64 values)
    while i < len {
        // Load complex number [re, im]
        let z = _mm_loadu_pd(amp_ptr.add(i * 2));

        // Square: [re*re, im*im]
        let z_sq = _mm_mul_pd(z, z);

        // Horizontal add: re^2 + im^2
        let mut temp = [0.0f64; 2];
        _mm_storeu_pd(temp.as_mut_ptr(), z_sq);

        *out_ptr.add(i) = temp[0] + temp[1];

        i += 1;
    }
}

/// Portable fallback for computing probability distribution (no SIMD)
pub fn compute_probabilities_scalar(amplitudes: &[Complex64], output: &mut [f64]) {
    for (i, amp) in amplitudes.iter().enumerate() {
        output[i] = amp.norm_sqr();
    }
}

/// Dispatch to best available SIMD implementation for probability computation
pub fn compute_probabilities(amplitudes: &[Complex64], output: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { compute_probabilities_avx2(amplitudes, output) }
        } else if is_x86_feature_detected!("sse2") {
            unsafe { compute_probabilities_sse2(amplitudes, output) }
        } else {
            compute_probabilities_scalar(amplitudes, output)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        compute_probabilities_scalar(amplitudes, output)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    use super::*;
    #[cfg(target_arch = "x86_64")]
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

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_compute_probabilities_avx2_with_remainder() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // 3 elements: processes one pair via SIMD, then hits the scalar
        // "remaining element" loop (odd tail) inside compute_probabilities_avx2.
        let amplitudes = vec![
            Complex64::new(3.0, 4.0), // norm_sqr = 25
            Complex64::new(1.0, 0.0), // norm_sqr = 1
            Complex64::new(0.0, 2.0), // norm_sqr = 4 (tail element)
        ];
        let mut output = vec![0.0; 3];

        unsafe {
            compute_probabilities_avx2(&amplitudes, &mut output);
        }

        assert_relative_eq!(output[0], 25.0, epsilon = 1e-10);
        assert_relative_eq!(output[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(output[2], 4.0, epsilon = 1e-10);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_compute_probabilities_sse2() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        let amplitudes = vec![
            Complex64::new(3.0, 4.0), // norm_sqr = 25
            Complex64::new(0.0, 1.0), // norm_sqr = 1
            Complex64::new(2.0, 0.0), // norm_sqr = 4
        ];
        let mut output = vec![0.0; 3];

        unsafe {
            compute_probabilities_sse2(&amplitudes, &mut output);
        }

        assert_relative_eq!(output[0], 25.0, epsilon = 1e-10);
        assert_relative_eq!(output[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(output[2], 4.0, epsilon = 1e-10);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_compute_probabilities_scalar_direct() {
        let amplitudes = vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 1.0)];
        let mut output = vec![0.0; 2];
        compute_probabilities_scalar(&amplitudes, &mut output);
        assert_relative_eq!(output[0], 25.0, epsilon = 1e-10);
        assert_relative_eq!(output[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_compute_probabilities_dispatch() {
        // Exercises the public dispatcher; on this CI runner AVX2 is always
        // available so this runs the avx2 path end-to-end (the sse2/scalar
        // arms are runtime-unreachable on an AVX2-capable CPU).
        let amplitudes = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 1.0),
        ];
        let mut output = vec![0.0; 3];
        compute_probabilities(&amplitudes, &mut output);
        assert_relative_eq!(output[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(output[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(output[2], 2.0, epsilon = 1e-10);
    }
}
