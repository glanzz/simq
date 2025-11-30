//! State vector representation with aligned memory for SIMD operations

use crate::error::{Result, StateError};
use num_complex::Complex64;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

/// Alignment requirement for SIMD operations (64 bytes for AVX-512)
const SIMD_ALIGNMENT: usize = 64;

/// Quantum state vector with SIMD-aligned memory
///
/// This structure represents a quantum state as a complex-valued vector
/// with proper alignment for SIMD operations. The memory is allocated
/// with 64-byte alignment to support AVX-512 instructions.
///
/// # Example
///
/// ```
/// use simq_state::StateVector;
///
/// // Create a 2-qubit state (4 amplitudes)
/// let state = StateVector::new(2).unwrap();
/// assert_eq!(state.num_qubits(), 2);
/// assert_eq!(state.dimension(), 4);
/// ```
pub struct StateVector {
    /// Number of qubits
    num_qubits: usize,

    /// State dimension (2^num_qubits)
    dimension: usize,

    /// Pointer to aligned state data
    data: NonNull<Complex64>,

    /// Memory layout for deallocation
    layout: Layout,
}

impl StateVector {
    /// Create a new state vector initialized to |0...0⟩
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    ///
    /// # Returns
    /// A new state vector in the computational basis state |0...0⟩
    ///
    /// # Errors
    /// Returns error if memory allocation fails or num_qubits is too large
    ///
    /// # Example
    /// ```
    /// use simq_state::StateVector;
    ///
    /// let state = StateVector::new(3).unwrap();
    /// assert_eq!(state.num_qubits(), 3);
    /// ```
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits > 30 {
            return Err(StateError::InvalidDimension {
                dimension: 1 << num_qubits,
            });
        }

        let dimension = 1 << num_qubits;

        // Create aligned layout
        let layout =
            Layout::from_size_align(dimension * std::mem::size_of::<Complex64>(), SIMD_ALIGNMENT)
                .map_err(|_| StateError::AllocationError {
                size: dimension * std::mem::size_of::<Complex64>(),
            })?;

        // Allocate aligned memory
        let data = unsafe {
            let ptr = alloc(layout) as *mut Complex64;
            if ptr.is_null() {
                return Err(StateError::AllocationError {
                    size: layout.size(),
                });
            }

            // Initialize to |0...0⟩ state
            std::ptr::write_bytes(ptr, 0, dimension);
            (*ptr) = Complex64::new(1.0, 0.0);

            NonNull::new_unchecked(ptr)
        };

        Ok(Self {
            num_qubits,
            dimension,
            data,
            layout,
        })
    }

    /// Create a state vector from raw amplitude data
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    /// * `amplitudes` - Complex amplitudes (must have length 2^num_qubits)
    ///
    /// # Returns
    /// A new state vector with the given amplitudes
    ///
    /// # Errors
    /// Returns error if dimension doesn't match or allocation fails
    pub fn from_amplitudes(num_qubits: usize, amplitudes: &[Complex64]) -> Result<Self> {
        let dimension = 1 << num_qubits;

        if amplitudes.len() != dimension {
            return Err(StateError::DimensionMismatch {
                expected: dimension,
                actual: amplitudes.len(),
            });
        }

        let state = Self::new(num_qubits)?;

        // Copy amplitudes
        unsafe {
            std::ptr::copy_nonoverlapping(amplitudes.as_ptr(), state.data.as_ptr(), dimension);
        }

        Ok(state)
    }

    /// Get the number of qubits
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the state dimension (2^num_qubits)
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get a reference to the state amplitudes
    #[inline]
    pub fn amplitudes(&self) -> &[Complex64] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.dimension) }
    }

    /// Get a mutable reference to the state amplitudes
    #[inline]
    pub fn amplitudes_mut(&mut self) -> &mut [Complex64] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.dimension) }
    }

    /// Get a raw pointer to the state data (for SIMD operations)
    #[inline]
    pub fn as_ptr(&self) -> *const Complex64 {
        self.data.as_ptr()
    }

    /// Get a mutable raw pointer to the state data (for SIMD operations)
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut Complex64 {
        self.data.as_ptr()
    }

    /// Check if the pointer is properly aligned for SIMD
    #[inline]
    pub fn is_simd_aligned(&self) -> bool {
        (self.data.as_ptr() as usize) % SIMD_ALIGNMENT == 0
    }

    /// Compute the norm of the state vector
    ///
    /// # Returns
    /// The L2 norm of the state vector
    pub fn norm(&self) -> f64 {
        let amplitudes = self.amplitudes();
        amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt()
    }

    /// Normalize the state vector
    ///
    /// Scales all amplitudes so that the norm equals 1.
    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > 1e-10 {
            let inv_norm = 1.0 / norm;
            for amplitude in self.amplitudes_mut() {
                *amplitude *= inv_norm;
            }
        }
    }

    /// Check if the state is normalized (norm ≈ 1)
    ///
    /// # Arguments
    /// * `epsilon` - Tolerance for normalization check
    ///
    /// # Returns
    /// True if |norm - 1| < epsilon
    pub fn is_normalized(&self, epsilon: f64) -> bool {
        (self.norm() - 1.0).abs() < epsilon
    }

    /// Reset the state to |0...0⟩
    pub fn reset(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.data.as_ptr(), 0, self.dimension);
            (*self.data.as_ptr()) = Complex64::new(1.0, 0.0);
        }
    }

    /// Clone the state vector
    pub fn clone_state(&self) -> Result<Self> {
        Self::from_amplitudes(self.num_qubits, self.amplitudes())
    }
}

impl Drop for StateVector {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.data.as_ptr() as *mut u8, self.layout);
        }
    }
}

// Safety: StateVector owns its data and ensures exclusive access
unsafe impl Send for StateVector {}
unsafe impl Sync for StateVector {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_new_state_vector() {
        let state = StateVector::new(2).unwrap();
        assert_eq!(state.num_qubits(), 2);
        assert_eq!(state.dimension(), 4);
        assert!(state.is_simd_aligned());
    }

    #[test]
    fn test_initial_state() {
        let state = StateVector::new(3).unwrap();
        let amplitudes = state.amplitudes();

        // Should be |000⟩
        assert_eq!(amplitudes[0], Complex64::new(1.0, 0.0));
        for i in 1..amplitudes.len() {
            assert_eq!(amplitudes[i], Complex64::new(0.0, 0.0));
        }
    }

    #[test]
    fn test_from_amplitudes() {
        let amplitudes = vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ];

        let state = StateVector::from_amplitudes(2, &amplitudes).unwrap();
        assert_eq!(state.amplitudes(), amplitudes.as_slice());
    }

    #[test]
    fn test_norm() {
        let state = StateVector::new(2).unwrap();
        assert_relative_eq!(state.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize() {
        let amplitudes = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        let mut state = StateVector::from_amplitudes(2, &amplitudes).unwrap();
        state.normalize();

        assert_relative_eq!(state.norm(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(state.amplitudes()[0].norm(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_reset() {
        let amplitudes = vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ];

        let mut state = StateVector::from_amplitudes(2, &amplitudes).unwrap();
        state.reset();

        assert_eq!(state.amplitudes()[0], Complex64::new(1.0, 0.0));
        for i in 1..state.dimension() {
            assert_eq!(state.amplitudes()[i], Complex64::new(0.0, 0.0));
        }
    }

    #[test]
    fn test_is_normalized() {
        let state = StateVector::new(2).unwrap();
        assert!(state.is_normalized(1e-10));
    }

    #[test]
    fn test_alignment() {
        let state = StateVector::new(5).unwrap();
        let ptr = state.as_ptr() as usize;
        assert_eq!(ptr % SIMD_ALIGNMENT, 0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let amplitudes = vec![Complex64::new(1.0, 0.0)];
        let result = StateVector::from_amplitudes(2, &amplitudes);
        assert!(result.is_err());
    }
}
