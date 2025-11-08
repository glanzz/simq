//! Parameter type with metadata and constraints for variational quantum algorithms

use crate::{QuantumError, Result};

/// A quantum circuit parameter with optional metadata and constraints
///
/// Parameters are used in variational quantum algorithms (VQE, QAOA) to
/// represent classical values that control quantum gates. They support:
/// - Optional naming for debugging and serialization
/// - Value bounds/constraints
/// - Freezing to exclude from optimization
/// - Metadata for gradients (future autodiff support)
///
/// # Example
/// ```
/// use simq_core::parameter::Parameter;
///
/// // Simple parameter
/// let theta = Parameter::new(0.5);
/// assert_eq!(theta.value(), 0.5);
///
/// // Named parameter with bounds
/// let mut beta = Parameter::named("beta_0", 1.0)
///     .with_bounds(0.0, 2.0 * std::f64::consts::PI)
///     .unwrap();
///
/// beta.set_value(3.0).unwrap();
/// assert_eq!(beta.value(), 3.0);
/// ```
#[derive(Clone, Debug)]
pub struct Parameter {
    name: Option<String>,
    value: f64,
    bounds: Option<(f64, f64)>,
    frozen: bool,
}

impl Parameter {
    /// Create a new parameter with a value
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter::Parameter;
    ///
    /// let param = Parameter::new(1.5);
    /// assert_eq!(param.value(), 1.5);
    /// assert!(param.name().is_none());
    /// assert!(!param.is_frozen());
    /// ```
    pub fn new(value: f64) -> Self {
        Self {
            name: None,
            value,
            bounds: None,
            frozen: false,
        }
    }

    /// Create a named parameter
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter::Parameter;
    ///
    /// let param = Parameter::named("theta_0", 0.5);
    /// assert_eq!(param.name(), Some("theta_0"));
    /// assert_eq!(param.value(), 0.5);
    /// ```
    pub fn named(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: Some(name.into()),
            value,
            bounds: None,
            frozen: false,
        }
    }

    /// Get parameter name
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter::Parameter;
    ///
    /// let param = Parameter::named("alpha", 1.0);
    /// assert_eq!(param.name(), Some("alpha"));
    ///
    /// let param2 = Parameter::new(1.0);
    /// assert_eq!(param2.name(), None);
    /// ```
    #[inline]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get parameter value
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter::Parameter;
    ///
    /// let param = Parameter::new(2.5);
    /// assert_eq!(param.value(), 2.5);
    /// ```
    #[inline]
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Set parameter value
    ///
    /// # Errors
    /// Returns error if:
    /// - Parameter is frozen
    /// - Value violates bounds
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter::Parameter;
    ///
    /// let mut param = Parameter::new(1.0);
    /// param.set_value(2.0).unwrap();
    /// assert_eq!(param.value(), 2.0);
    ///
    /// // Frozen parameters cannot be modified
    /// param.freeze();
    /// assert!(param.set_value(3.0).is_err());
    /// ```
    pub fn set_value(&mut self, value: f64) -> Result<()> {
        if self.frozen {
            return Err(QuantumError::ValidationError(format!(
                "Cannot modify frozen parameter{}",
                self.name
                    .as_ref()
                    .map(|n| format!(" '{}'", n))
                    .unwrap_or_default()
            )));
        }

        if let Some((min, max)) = self.bounds {
            if value < min || value > max {
                return Err(QuantumError::ValidationError(format!(
                    "Value {} outside bounds [{}, {}]{}",
                    value,
                    min,
                    max,
                    self.name
                        .as_ref()
                        .map(|n| format!(" for parameter '{}'", n))
                        .unwrap_or_default()
                )));
            }
        }

        self.value = value;
        Ok(())
    }

    /// Get parameter bounds
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter::Parameter;
    ///
    /// let param = Parameter::new(1.0).with_bounds(0.0, 2.0).unwrap();
    /// assert_eq!(param.bounds(), Some((0.0, 2.0)));
    /// ```
    #[inline]
    pub fn bounds(&self) -> Option<(f64, f64)> {
        self.bounds
    }

    /// Set parameter bounds (builder pattern)
    ///
    /// # Errors
    /// Returns error if:
    /// - min > max
    /// - Current value is outside new bounds
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter::Parameter;
    ///
    /// let param = Parameter::new(1.0).with_bounds(0.0, 2.0).unwrap();
    /// assert_eq!(param.bounds(), Some((0.0, 2.0)));
    ///
    /// // Invalid bounds
    /// let result = Parameter::new(1.0).with_bounds(2.0, 0.0);
    /// assert!(result.is_err());
    ///
    /// // Value outside bounds
    /// let result = Parameter::new(5.0).with_bounds(0.0, 2.0);
    /// assert!(result.is_err());
    /// ```
    pub fn with_bounds(mut self, min: f64, max: f64) -> Result<Self> {
        if min > max {
            return Err(QuantumError::ValidationError(format!(
                "Invalid bounds: min ({}) > max ({})",
                min, max
            )));
        }
        if self.value < min || self.value > max {
            return Err(QuantumError::ValidationError(format!(
                "Value {} outside bounds [{}, {}]",
                self.value, min, max
            )));
        }
        self.bounds = Some((min, max));
        Ok(self)
    }

    /// Set parameter bounds (mutable)
    ///
    /// # Errors
    /// Same as `with_bounds`
    pub fn set_bounds(&mut self, min: f64, max: f64) -> Result<()> {
        if min > max {
            return Err(QuantumError::ValidationError(format!(
                "Invalid bounds: min ({}) > max ({})",
                min, max
            )));
        }
        if self.value < min || self.value > max {
            return Err(QuantumError::ValidationError(format!(
                "Current value {} outside new bounds [{}, {}]",
                self.value, min, max
            )));
        }
        self.bounds = Some((min, max));
        Ok(())
    }

    /// Remove bounds
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter::Parameter;
    ///
    /// let mut param = Parameter::new(1.0).with_bounds(0.0, 2.0).unwrap();
    /// assert!(param.bounds().is_some());
    ///
    /// param.clear_bounds();
    /// assert!(param.bounds().is_none());
    /// ```
    pub fn clear_bounds(&mut self) {
        self.bounds = None;
    }

    /// Freeze parameter (prevent modifications)
    ///
    /// Frozen parameters are excluded from optimization in VQE/QAOA.
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter::Parameter;
    ///
    /// let mut param = Parameter::new(1.0);
    /// assert!(!param.is_frozen());
    ///
    /// param.freeze();
    /// assert!(param.is_frozen());
    /// assert!(param.set_value(2.0).is_err());
    /// ```
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Unfreeze parameter (allow modifications)
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter::Parameter;
    ///
    /// let mut param = Parameter::new(1.0);
    /// param.freeze();
    /// assert!(param.is_frozen());
    ///
    /// param.unfreeze();
    /// assert!(!param.is_frozen());
    /// param.set_value(2.0).unwrap();
    /// ```
    pub fn unfreeze(&mut self) {
        self.frozen = false;
    }

    /// Check if parameter is frozen
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter::Parameter;
    ///
    /// let param = Parameter::new(1.0).as_frozen();
    /// assert!(param.is_frozen());
    /// ```
    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Builder pattern: set name
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter::Parameter;
    ///
    /// let param = Parameter::new(1.0).with_name("theta");
    /// assert_eq!(param.name(), Some("theta"));
    /// ```
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Builder pattern: set as frozen
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter::Parameter;
    ///
    /// let param = Parameter::new(1.0).as_frozen();
    /// assert!(param.is_frozen());
    /// ```
    pub fn as_frozen(mut self) -> Self {
        self.frozen = true;
        self
    }
}

impl std::fmt::Display for Parameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "{}={}", name, self.value)?;
        } else {
            write!(f, "{}", self.value)?;
        }
        if let Some((min, max)) = self.bounds {
            write!(f, " ∈ [{}, {}]", min, max)?;
        }
        if self.frozen {
            write!(f, " (frozen)")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_creation() {
        let param = Parameter::new(1.5);
        assert_eq!(param.value(), 1.5);
        assert!(param.name().is_none());
        assert!(param.bounds().is_none());
        assert!(!param.is_frozen());
    }

    #[test]
    fn test_named_parameter() {
        let param = Parameter::named("theta", 2.0);
        assert_eq!(param.name(), Some("theta"));
        assert_eq!(param.value(), 2.0);
    }

    #[test]
    fn test_set_value() {
        let mut param = Parameter::new(1.0);
        param.set_value(2.0).unwrap();
        assert_eq!(param.value(), 2.0);
    }

    #[test]
    fn test_frozen_parameter() {
        let mut param = Parameter::new(1.0);
        param.freeze();
        assert!(param.is_frozen());

        let result = param.set_value(2.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_unfreeze_parameter() {
        let mut param = Parameter::new(1.0);
        param.freeze();
        param.unfreeze();
        assert!(!param.is_frozen());

        param.set_value(2.0).unwrap();
        assert_eq!(param.value(), 2.0);
    }

    #[test]
    fn test_bounds() {
        let param = Parameter::new(1.0).with_bounds(0.0, 2.0).unwrap();
        assert_eq!(param.bounds(), Some((0.0, 2.0)));
    }

    #[test]
    fn test_invalid_bounds() {
        let result = Parameter::new(1.0).with_bounds(2.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_value_outside_bounds() {
        let result = Parameter::new(5.0).with_bounds(0.0, 2.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_value_with_bounds() {
        let mut param = Parameter::new(1.0).with_bounds(0.0, 2.0).unwrap();

        param.set_value(1.5).unwrap();
        assert_eq!(param.value(), 1.5);

        let result = param.set_value(3.0);
        assert!(result.is_err());

        let result = param.set_value(-1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_clear_bounds() {
        let mut param = Parameter::new(1.0).with_bounds(0.0, 2.0).unwrap();
        assert!(param.bounds().is_some());

        param.clear_bounds();
        assert!(param.bounds().is_none());

        // Can now set value outside old bounds
        param.set_value(10.0).unwrap();
        assert_eq!(param.value(), 10.0);
    }

    #[test]
    fn test_builder_pattern() {
        let param = Parameter::new(1.0)
            .with_name("alpha")
            .with_bounds(0.0, PI * 2.0)
            .unwrap()
            .as_frozen();

        assert_eq!(param.name(), Some("alpha"));
        assert_eq!(param.bounds(), Some((0.0, PI * 2.0)));
        assert!(param.is_frozen());
    }

    #[test]
    fn test_display() {
        let param1 = Parameter::new(1.5);
        assert_eq!(format!("{}", param1), "1.5");

        let param2 = Parameter::named("theta", 2.0);
        assert_eq!(format!("{}", param2), "theta=2");

        let param3 = Parameter::new(1.0).with_bounds(0.0, 2.0).unwrap();
        assert!(format!("{}", param3).contains("[0, 2]"));

        let param4 = Parameter::new(1.0).as_frozen();
        assert!(format!("{}", param4).contains("frozen"));
    }

    #[test]
    fn test_set_bounds_mutable() {
        let mut param = Parameter::new(1.0);
        param.set_bounds(0.0, 2.0).unwrap();
        assert_eq!(param.bounds(), Some((0.0, 2.0)));
    }
}
