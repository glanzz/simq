//! Parameter identification and tracking

use std::fmt;

/// A unique identifier for a parameter in a parameter registry
///
/// `ParameterId` is a lightweight, copyable handle that references a parameter
/// stored in a `ParameterRegistry`. It uses interior indices for fast lookup
/// with zero runtime overhead.
///
/// # Design
///
/// - **Copy**: ParameterId is cheap to copy (just a usize)
/// - **Type-safe**: Prevents accidental use of raw indices
/// - **Compact**: 8 bytes on 64-bit systems
///
/// # Example
/// ```
/// use simq_core::parameter_id::ParameterId;
///
/// // Typically created by ParameterRegistry
/// let id = ParameterId::new(0);
/// assert_eq!(id.index(), 0);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParameterId {
    index: usize,
}

impl ParameterId {
    /// Create a new parameter ID
    ///
    /// This is typically called by `ParameterRegistry`, not by users directly.
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_id::ParameterId;
    ///
    /// let id = ParameterId::new(42);
    /// assert_eq!(id.index(), 42);
    /// ```
    #[inline]
    pub(crate) fn new(index: usize) -> Self {
        Self { index }
    }

    /// Get the internal index
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_id::ParameterId;
    ///
    /// let id = ParameterId::new(5);
    /// assert_eq!(id.index(), 5);
    /// ```
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }
}

impl fmt::Display for ParameterId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "param_{}", self.index)
    }
}

impl From<usize> for ParameterId {
    fn from(index: usize) -> Self {
        Self { index }
    }
}

impl From<ParameterId> for usize {
    fn from(id: ParameterId) -> Self {
        id.index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_id_creation() {
        let id = ParameterId::new(0);
        assert_eq!(id.index(), 0);

        let id = ParameterId::new(42);
        assert_eq!(id.index(), 42);
    }

    #[test]
    fn test_parameter_id_equality() {
        let id1 = ParameterId::new(5);
        let id2 = ParameterId::new(5);
        let id3 = ParameterId::new(10);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_parameter_id_ordering() {
        let id1 = ParameterId::new(1);
        let id2 = ParameterId::new(2);
        let id3 = ParameterId::new(3);

        assert!(id1 < id2);
        assert!(id2 < id3);
        assert!(id1 < id3);
    }

    #[test]
    fn test_parameter_id_display() {
        let id = ParameterId::new(42);
        assert_eq!(format!("{}", id), "param_42");
    }

    #[test]
    fn test_parameter_id_copy() {
        let id1 = ParameterId::new(10);
        let id2 = id1;

        assert_eq!(id1, id2);
        assert_eq!(id1.index(), id2.index());
    }

    #[test]
    fn test_parameter_id_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(ParameterId::new(0));
        set.insert(ParameterId::new(1));
        set.insert(ParameterId::new(0)); // Duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&ParameterId::new(0)));
        assert!(set.contains(&ParameterId::new(1)));
    }

    #[test]
    fn test_parameter_id_from_usize() {
        let id: ParameterId = 42.into();
        assert_eq!(id.index(), 42);
    }

    #[test]
    fn test_usize_from_parameter_id() {
        let id = ParameterId::new(42);
        let index: usize = id.into();
        assert_eq!(index, 42);
    }
}
