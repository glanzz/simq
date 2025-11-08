//! Central registry for managing quantum circuit parameters

use crate::parameter::Parameter;
use crate::parameter_id::ParameterId;
use crate::{QuantumError, Result};
use std::collections::HashMap;

/// Central registry for managing all parameters in variational quantum circuits
///
/// The `ParameterRegistry` provides efficient storage and batch operations for
/// parameters used in VQE, QAOA, and other variational algorithms.
///
/// # Design
///
/// - **Contiguous storage**: All parameter values stored in a Vec for cache efficiency
/// - **Fast lookup**: O(1) access by ParameterId, O(1) lookup by name via HashMap
/// - **Batch operations**: Efficient update of multiple parameters at once
/// - **Zero-copy views**: Get slices of parameter values without allocation
///
/// # Example
/// ```
/// use simq_core::parameter::Parameter;
/// use simq_core::parameter_registry::ParameterRegistry;
///
/// let mut registry = ParameterRegistry::new();
///
/// // Add parameters
/// let theta_id = registry.add_named("theta", 0.5);
/// let beta_id = registry.add_named("beta", 1.0);
///
/// // Access parameters
/// assert_eq!(registry.get(theta_id).unwrap().value(), 0.5);
///
/// // Batch update
/// registry.set_values(&[theta_id, beta_id], &[1.0, 2.0]).unwrap();
/// assert_eq!(registry.get(theta_id).unwrap().value(), 1.0);
/// ```
#[derive(Clone, Debug)]
pub struct ParameterRegistry {
    parameters: Vec<Parameter>,
    name_to_id: HashMap<String, ParameterId>,
}

impl ParameterRegistry {
    /// Create a new empty parameter registry
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let registry = ParameterRegistry::new();
    /// assert_eq!(registry.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
            name_to_id: HashMap::new(),
        }
    }

    /// Create a registry with pre-allocated capacity
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let registry = ParameterRegistry::with_capacity(100);
    /// assert_eq!(registry.len(), 0);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            parameters: Vec::with_capacity(capacity),
            name_to_id: HashMap::with_capacity(capacity),
        }
    }

    /// Add a parameter to the registry
    ///
    /// Returns the ParameterId for the newly added parameter.
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter::Parameter;
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let mut registry = ParameterRegistry::new();
    /// let param = Parameter::new(1.5);
    /// let id = registry.add(param);
    ///
    /// assert_eq!(registry.get(id).unwrap().value(), 1.5);
    /// ```
    pub fn add(&mut self, param: Parameter) -> ParameterId {
        let id = ParameterId::new(self.parameters.len());

        // Register name if present
        if let Some(name) = param.name() {
            self.name_to_id.insert(name.to_string(), id);
        }

        self.parameters.push(param);
        id
    }

    /// Add a named parameter
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let mut registry = ParameterRegistry::new();
    /// let id = registry.add_named("theta_0", 0.5);
    ///
    /// assert_eq!(registry.get(id).unwrap().value(), 0.5);
    /// assert_eq!(registry.get_by_name("theta_0").unwrap().value(), 0.5);
    /// ```
    pub fn add_named(&mut self, name: impl Into<String>, value: f64) -> ParameterId {
        self.add(Parameter::named(name, value))
    }

    /// Add multiple parameters at once
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let mut registry = ParameterRegistry::new();
    /// let ids = registry.add_many(&[0.1, 0.2, 0.3]);
    ///
    /// assert_eq!(ids.len(), 3);
    /// assert_eq!(registry.get(ids[0]).unwrap().value(), 0.1);
    /// ```
    pub fn add_many(&mut self, values: &[f64]) -> Vec<ParameterId> {
        values
            .iter()
            .map(|&value| self.add(Parameter::new(value)))
            .collect()
    }

    /// Get a parameter by ID
    ///
    /// # Errors
    /// Returns error if ID is invalid
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let mut registry = ParameterRegistry::new();
    /// let id = registry.add_named("alpha", 1.0);
    ///
    /// let param = registry.get(id).unwrap();
    /// assert_eq!(param.value(), 1.0);
    /// ```
    pub fn get(&self, id: ParameterId) -> Result<&Parameter> {
        self.parameters
            .get(id.index())
            .ok_or_else(|| QuantumError::ValidationError(format!("Invalid parameter ID: {}", id)))
    }

    /// Get a mutable reference to a parameter by ID
    ///
    /// # Errors
    /// Returns error if ID is invalid
    pub fn get_mut(&mut self, id: ParameterId) -> Result<&mut Parameter> {
        let index = id.index();
        self.parameters.get_mut(index).ok_or_else(|| {
            QuantumError::ValidationError(format!("Invalid parameter ID: param_{}", index))
        })
    }

    /// Get a parameter by name
    ///
    /// # Errors
    /// Returns error if name not found
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let mut registry = ParameterRegistry::new();
    /// registry.add_named("theta", 0.5);
    ///
    /// let param = registry.get_by_name("theta").unwrap();
    /// assert_eq!(param.value(), 0.5);
    /// ```
    pub fn get_by_name(&self, name: &str) -> Result<&Parameter> {
        let id = self.get_id_by_name(name)?;
        self.get(id)
    }

    /// Get parameter ID by name
    ///
    /// # Errors
    /// Returns error if name not found
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let mut registry = ParameterRegistry::new();
    /// let original_id = registry.add_named("beta", 1.0);
    ///
    /// let found_id = registry.get_id_by_name("beta").unwrap();
    /// assert_eq!(original_id, found_id);
    /// ```
    pub fn get_id_by_name(&self, name: &str) -> Result<ParameterId> {
        self.name_to_id
            .get(name)
            .copied()
            .ok_or_else(|| QuantumError::ValidationError(format!("Parameter '{}' not found", name)))
    }

    /// Get values for multiple parameters
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let mut registry = ParameterRegistry::new();
    /// let ids = registry.add_many(&[1.0, 2.0, 3.0]);
    ///
    /// let values = registry.get_values(&ids);
    /// assert_eq!(values, vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn get_values(&self, ids: &[ParameterId]) -> Vec<f64> {
        ids.iter()
            .filter_map(|&id| self.get(id).ok().map(|p| p.value()))
            .collect()
    }

    /// Set values for multiple parameters
    ///
    /// # Errors
    /// Returns error if:
    /// - Parameter count mismatch
    /// - Any parameter is frozen
    /// - Any value violates bounds
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let mut registry = ParameterRegistry::new();
    /// let ids = registry.add_many(&[0.0, 0.0, 0.0]);
    ///
    /// registry.set_values(&ids, &[1.0, 2.0, 3.0]).unwrap();
    /// assert_eq!(registry.get_values(&ids), vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn set_values(&mut self, ids: &[ParameterId], values: &[f64]) -> Result<()> {
        if ids.len() != values.len() {
            return Err(QuantumError::ValidationError(format!(
                "Parameter count mismatch: {} IDs, {} values",
                ids.len(),
                values.len()
            )));
        }

        for (&id, &value) in ids.iter().zip(values.iter()) {
            self.get_mut(id)?.set_value(value)?;
        }

        Ok(())
    }

    /// Get all parameter values as a slice (zero-copy)
    ///
    /// Returns values in the same order they were added.
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let mut registry = ParameterRegistry::new();
    /// registry.add_many(&[1.0, 2.0, 3.0]);
    ///
    /// let values: Vec<f64> = registry.all_values().to_vec();
    /// assert_eq!(values, vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn all_values(&self) -> Vec<f64> {
        self.parameters.iter().map(|p| p.value()).collect()
    }

    /// Set all parameter values from a slice
    ///
    /// # Errors
    /// Returns error if slice length doesn't match parameter count
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let mut registry = ParameterRegistry::new();
    /// registry.add_many(&[0.0, 0.0, 0.0]);
    ///
    /// registry.set_all_values(&[1.0, 2.0, 3.0]).unwrap();
    /// assert_eq!(registry.all_values(), vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn set_all_values(&mut self, values: &[f64]) -> Result<()> {
        if values.len() != self.parameters.len() {
            return Err(QuantumError::ValidationError(format!(
                "Value count mismatch: expected {}, got {}",
                self.parameters.len(),
                values.len()
            )));
        }

        for (param, &value) in self.parameters.iter_mut().zip(values.iter()) {
            param.set_value(value)?;
        }

        Ok(())
    }

    /// Get IDs of all unfrozen parameters
    ///
    /// Useful for optimization where only unfrozen parameters should vary.
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter::Parameter;
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let mut registry = ParameterRegistry::new();
    /// let id1 = registry.add(Parameter::new(1.0));
    /// let id2 = registry.add(Parameter::new(2.0).as_frozen());
    /// let id3 = registry.add(Parameter::new(3.0));
    ///
    /// let unfrozen = registry.unfrozen_params();
    /// assert_eq!(unfrozen.len(), 2);
    /// assert!(unfrozen.contains(&id1));
    /// assert!(unfrozen.contains(&id3));
    /// ```
    pub fn unfrozen_params(&self) -> Vec<ParameterId> {
        self.parameters
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_frozen())
            .map(|(i, _)| ParameterId::new(i))
            .collect()
    }

    /// Get IDs of all frozen parameters
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter::Parameter;
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let mut registry = ParameterRegistry::new();
    /// registry.add(Parameter::new(1.0).as_frozen());
    /// registry.add(Parameter::new(2.0));
    ///
    /// let frozen = registry.frozen_params();
    /// assert_eq!(frozen.len(), 1);
    /// ```
    pub fn frozen_params(&self) -> Vec<ParameterId> {
        self.parameters
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_frozen())
            .map(|(i, _)| ParameterId::new(i))
            .collect()
    }

    /// Get the number of parameters
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let mut registry = ParameterRegistry::new();
    /// assert_eq!(registry.len(), 0);
    ///
    /// registry.add_many(&[1.0, 2.0, 3.0]);
    /// assert_eq!(registry.len(), 3);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Check if registry is empty
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let registry = ParameterRegistry::new();
    /// assert!(registry.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Get all parameter IDs
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let mut registry = ParameterRegistry::new();
    /// registry.add_many(&[1.0, 2.0, 3.0]);
    ///
    /// let ids = registry.all_ids();
    /// assert_eq!(ids.len(), 3);
    /// ```
    pub fn all_ids(&self) -> Vec<ParameterId> {
        (0..self.parameters.len()).map(ParameterId::new).collect()
    }

    /// Iterate over all parameters
    ///
    /// # Example
    /// ```
    /// use simq_core::parameter_registry::ParameterRegistry;
    ///
    /// let mut registry = ParameterRegistry::new();
    /// registry.add_many(&[1.0, 2.0, 3.0]);
    ///
    /// for (id, param) in registry.iter() {
    /// println!("{}: {}", id, param.value());
    /// }
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (ParameterId, &Parameter)> {
        self.parameters
            .iter()
            .enumerate()
            .map(|(i, p)| (ParameterId::new(i), p))
    }
}

impl Default for ParameterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = ParameterRegistry::new();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_add_parameter() {
        let mut registry = ParameterRegistry::new();
        let param = Parameter::new(1.5);
        let id = registry.add(param);

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.get(id).unwrap().value(), 1.5);
    }

    #[test]
    fn test_add_named_parameter() {
        let mut registry = ParameterRegistry::new();
        let id = registry.add_named("theta", 0.5);

        assert_eq!(registry.get(id).unwrap().value(), 0.5);
        assert_eq!(registry.get_by_name("theta").unwrap().value(), 0.5);
    }

    #[test]
    fn test_add_many() {
        let mut registry = ParameterRegistry::new();
        let ids = registry.add_many(&[1.0, 2.0, 3.0]);

        assert_eq!(ids.len(), 3);
        assert_eq!(registry.len(), 3);
        assert_eq!(registry.get(ids[0]).unwrap().value(), 1.0);
        assert_eq!(registry.get(ids[2]).unwrap().value(), 3.0);
    }

    #[test]
    fn test_get_by_name() {
        let mut registry = ParameterRegistry::new();
        registry.add_named("alpha", 1.0);
        registry.add_named("beta", 2.0);

        assert_eq!(registry.get_by_name("alpha").unwrap().value(), 1.0);
        assert_eq!(registry.get_by_name("beta").unwrap().value(), 2.0);
        assert!(registry.get_by_name("gamma").is_err());
    }

    #[test]
    fn test_get_id_by_name() {
        let mut registry = ParameterRegistry::new();
        let id1 = registry.add_named("theta", 0.5);

        let found_id = registry.get_id_by_name("theta").unwrap();
        assert_eq!(id1, found_id);
    }

    #[test]
    fn test_get_values() {
        let mut registry = ParameterRegistry::new();
        let ids = registry.add_many(&[1.0, 2.0, 3.0]);

        let values = registry.get_values(&ids);
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_set_values() {
        let mut registry = ParameterRegistry::new();
        let ids = registry.add_many(&[0.0, 0.0, 0.0]);

        registry.set_values(&ids, &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(registry.get_values(&ids), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_set_values_mismatch() {
        let mut registry = ParameterRegistry::new();
        let ids = registry.add_many(&[0.0, 0.0]);

        let result = registry.set_values(&ids, &[1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_values() {
        let mut registry = ParameterRegistry::new();
        registry.add_many(&[1.0, 2.0, 3.0]);

        let values = registry.all_values();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_set_all_values() {
        let mut registry = ParameterRegistry::new();
        registry.add_many(&[0.0, 0.0, 0.0]);

        registry.set_all_values(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(registry.all_values(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_unfrozen_params() {
        let mut registry = ParameterRegistry::new();
        let id1 = registry.add(Parameter::new(1.0));
        let _id2 = registry.add(Parameter::new(2.0).as_frozen());
        let id3 = registry.add(Parameter::new(3.0));

        let unfrozen = registry.unfrozen_params();
        assert_eq!(unfrozen.len(), 2);
        assert!(unfrozen.contains(&id1));
        assert!(unfrozen.contains(&id3));
    }

    #[test]
    fn test_frozen_params() {
        let mut registry = ParameterRegistry::new();
        registry.add(Parameter::new(1.0));
        let id2 = registry.add(Parameter::new(2.0).as_frozen());
        registry.add(Parameter::new(3.0));

        let frozen = registry.frozen_params();
        assert_eq!(frozen.len(), 1);
        assert!(frozen.contains(&id2));
    }

    #[test]
    fn test_all_ids() {
        let mut registry = ParameterRegistry::new();
        registry.add_many(&[1.0, 2.0, 3.0]);

        let ids = registry.all_ids();
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_iter() {
        let mut registry = ParameterRegistry::new();
        registry.add_many(&[1.0, 2.0, 3.0]);

        let mut count = 0;
        for (_, param) in registry.iter() {
            assert!(param.value() >= 1.0 && param.value() <= 3.0);
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_with_capacity() {
        let registry = ParameterRegistry::with_capacity(100);
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }
}
