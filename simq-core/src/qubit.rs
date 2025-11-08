//! Qubit addressing and identification

use std::fmt;

/// Type-safe identifier for a qubit
///
/// Provides compile-time type safety to prevent accidentally using
/// raw integers where qubit indices are expected.
///
/// # Example
/// ```
/// use simq_core::QubitId;
///
/// let q0 = QubitId::new(0);
/// let q1 = QubitId::new(1);
/// assert!(q0 < q1);
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct QubitId(usize);

impl QubitId {
    /// Create a new qubit identifier
    ///
    /// # Example
    /// ```
    /// use simq_core::QubitId;
    /// let q = QubitId::new(5);
    /// ```
    #[inline]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Get the underlying index
    ///
    /// # Example
    /// ```
    /// use simq_core::QubitId;
    /// let q = QubitId::new(5);
    /// assert_eq!(q.index(), 5);
    /// ```
    #[inline]
    pub const fn index(&self) -> usize {
        self.0
    }
}

impl fmt::Display for QubitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "q{}", self.0)
    }
}

impl From<usize> for QubitId {
    #[inline]
    fn from(id: usize) -> Self {
        Self::new(id)
    }
}

impl From<QubitId> for usize {
    #[inline]
    fn from(qid: QubitId) -> Self {
        qid.index()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qubit_creation() {
        let q = QubitId::new(5);
        assert_eq!(q.index(), 5);
    }

    #[test]
    fn test_qubit_equality() {
        let q1 = QubitId::new(0);
        let q2 = QubitId::new(0);
        let q3 = QubitId::new(1);

        assert_eq!(q1, q2);
        assert_ne!(q1, q3);
    }

    #[test]
    fn test_qubit_ordering() {
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        let q2 = QubitId::new(2);

        assert!(q0 < q1);
        assert!(q1 < q2);
        assert!(q2 > q0);
    }

    #[test]
    fn test_qubit_display() {
        let q = QubitId::new(5);
        assert_eq!(format!("{}", q), "q5");
    }

    #[test]
    fn test_qubit_from_usize() {
        let q: QubitId = 5.into();
        assert_eq!(q.index(), 5);
    }

    #[test]
    fn test_usize_from_qubit() {
        let q = QubitId::new(5);
        let i: usize = q.into();
        assert_eq!(i, 5);
    }
}
