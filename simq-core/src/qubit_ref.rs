//! Type-safe qubit references for circuit builders

use crate::{QuantumError, QubitId, Result};
use std::marker::PhantomData;

/// A type-safe qubit reference with compile-time circuit size tracking
///
/// The const generic parameter `N` represents the total number of qubits
/// in the circuit. This allows the type system to prevent invalid qubit
/// references at compile time.
///
/// # Example
/// ```
/// use simq_core::qubit_ref::Qubit;
///
/// // Qubit 0 in a 3-qubit circuit
/// let q0: Qubit<3> = Qubit::new(0).unwrap();
/// assert_eq!(q0.index(), 0);
///
/// // This would fail - qubit 5 doesn't exist in a 3-qubit circuit
/// let invalid: Result<Qubit<3>, _> = Qubit::new(5);
/// assert!(invalid.is_err());
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Qubit<const N: usize> {
    index: usize,
    _phantom: PhantomData<[(); N]>,
}

impl<const N: usize> Qubit<N> {
    /// Create a new qubit reference
    ///
    /// # Errors
    /// Returns error if index >= N
    ///
    /// # Example
    /// ```
    /// use simq_core::qubit_ref::Qubit;
    ///
    /// let q0: Qubit<5> = Qubit::new(0).unwrap();
    /// assert_eq!(q0.index(), 0);
    ///
    /// let invalid: Result<Qubit<5>, _> = Qubit::new(10);
    /// assert!(invalid.is_err());
    /// ```
    pub fn new(index: usize) -> Result<Self> {
        if index >= N {
            Err(QuantumError::invalid_qubit(index, N))
        } else {
            Ok(Self {
                index,
                _phantom: PhantomData,
            })
        }
    }

    /// Create a qubit reference without bounds checking
    ///
    /// # Safety
    /// Caller must ensure index < N
    ///
    /// This method is pub(crate) and used internally for optimization
    #[inline]
    pub(crate) unsafe fn new_unchecked(index: usize) -> Self {
        debug_assert!(index < N, "Qubit index out of bounds: {} >= {}", index, N);
        Self {
            index,
            _phantom: PhantomData,
        }
    }

    /// Get the qubit index
    ///
    /// # Example
    /// ```
    /// use simq_core::qubit_ref::Qubit;
    ///
    /// let q3: Qubit<10> = Qubit::new(3).unwrap();
    /// assert_eq!(q3.index(), 3);
    /// ```
    #[inline]
    pub fn index(self) -> usize {
        self.index
    }

    /// Convert to QubitId
    ///
    /// # Example
    /// ```
    /// use simq_core::qubit_ref::Qubit;
    ///
    /// let q2: Qubit<5> = Qubit::new(2).unwrap();
    /// let id = q2.to_qubit_id();
    /// assert_eq!(id.index(), 2);
    /// ```
    #[inline]
    pub fn to_qubit_id(self) -> QubitId {
        QubitId::new(self.index)
    }

    /// Get the circuit size this qubit belongs to
    ///
    /// # Example
    /// ```
    /// use simq_core::qubit_ref::Qubit;
    ///
    /// assert_eq!(Qubit::<5>::circuit_size(), 5);
    /// assert_eq!(Qubit::<10>::circuit_size(), 10);
    /// ```
    #[inline]
    pub const fn circuit_size() -> usize {
        N
    }
}

impl<const N: usize> std::fmt::Display for Qubit<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "q{}", self.index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qubit_creation() {
        let q0: Qubit<5> = Qubit::new(0).unwrap();
        assert_eq!(q0.index(), 0);

        let q4: Qubit<5> = Qubit::new(4).unwrap();
        assert_eq!(q4.index(), 4);
    }

    #[test]
    fn test_qubit_out_of_bounds() {
        let result: Result<Qubit<3>> = Qubit::new(5);
        assert!(result.is_err());

        if let Err(QuantumError::InvalidQubit(idx, size)) = result {
            assert_eq!(idx, 5);
            assert_eq!(size, 3);
        } else {
            panic!("Expected InvalidQubit error");
        }
    }

    #[test]
    fn test_qubit_exact_boundary() {
        // N-1 should succeed
        let q4: Result<Qubit<5>> = Qubit::new(4);
        assert!(q4.is_ok());

        // N should fail
        let q5: Result<Qubit<5>> = Qubit::new(5);
        assert!(q5.is_err());
    }

    #[test]
    fn test_qubit_display() {
        let q2: Qubit<10> = Qubit::new(2).unwrap();
        assert_eq!(format!("{}", q2), "q2");

        let q7: Qubit<10> = Qubit::new(7).unwrap();
        assert_eq!(format!("{}", q7), "q7");
    }

    #[test]
    fn test_qubit_equality() {
        let q0a: Qubit<5> = Qubit::new(0).unwrap();
        let q0b: Qubit<5> = Qubit::new(0).unwrap();
        let q1: Qubit<5> = Qubit::new(1).unwrap();

        assert_eq!(q0a, q0b);
        assert_ne!(q0a, q1);
    }

    #[test]
    fn test_qubit_ordering() {
        let q0: Qubit<5> = Qubit::new(0).unwrap();
        let q1: Qubit<5> = Qubit::new(1).unwrap();
        let q2: Qubit<5> = Qubit::new(2).unwrap();

        assert!(q0 < q1);
        assert!(q1 < q2);
        assert!(q0 < q2);
    }

    #[test]
    fn test_qubit_to_qubit_id() {
        let q3: Qubit<10> = Qubit::new(3).unwrap();
        let id = q3.to_qubit_id();
        assert_eq!(id.index(), 3);
    }

    #[test]
    fn test_circuit_size() {
        assert_eq!(Qubit::<5>::circuit_size(), 5);
        assert_eq!(Qubit::<10>::circuit_size(), 10);
        assert_eq!(Qubit::<100>::circuit_size(), 100);
    }

    #[test]
    fn test_qubit_copy_clone() {
        let q1: Qubit<5> = Qubit::new(1).unwrap();
        let q1_copy = q1;
        let q1_clone = q1; // Copy, not clone

        assert_eq!(q1, q1_copy);
        assert_eq!(q1, q1_clone);
        assert_eq!(q1_copy, q1_clone);
    }

    #[test]
    fn test_qubit_hash() {
        use std::collections::HashSet;

        let q0: Qubit<5> = Qubit::new(0).unwrap();
        let q1: Qubit<5> = Qubit::new(1).unwrap();
        let q0_dup: Qubit<5> = Qubit::new(0).unwrap();

        let mut set = HashSet::new();
        set.insert(q0);
        set.insert(q1);
        set.insert(q0_dup);

        // Should only have 2 unique qubits
        assert_eq!(set.len(), 2);
        assert!(set.contains(&q0));
        assert!(set.contains(&q1));
    }
}
