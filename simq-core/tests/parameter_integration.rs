//! Integration tests for parameter tracking system

use simq_core::{Parameter, ParameterRegistry};
use std::f64::consts::PI;

#[test]
fn test_vqe_parameter_workflow() {
    // Simulate VQE with multiple ansatz layers
    let mut registry = ParameterRegistry::new();

    // Layer 1: Rotation angles for each qubit
    let theta_0 = registry.add_named("theta_0", 0.1);
    let _theta_1 = registry.add_named("theta_1", 0.2);
    let _theta_2 = registry.add_named("theta_2", 0.3);

    // Layer 2: Entangling gates with parameters
    let _phi_01 = registry.add_named("phi_01", 0.5);
    let phi_12 = registry.add_named("phi_12", 0.6);

    assert_eq!(registry.len(), 5);

    // Optimize: update all parameters
    let new_values = vec![0.15, 0.25, 0.35, 0.55, 0.65];
    let all_ids = registry.all_ids();
    registry.set_values(&all_ids, &new_values).unwrap();

    // Verify updates
    assert_eq!(registry.get(theta_0).unwrap().value(), 0.15);
    assert_eq!(registry.get(phi_12).unwrap().value(), 0.65);
}

#[test]
fn test_qaoa_parameter_structure() {
    // QAOA with p=3 rounds
    let mut registry = ParameterRegistry::new();
    let num_rounds = 3;

    // Beta parameters (mixer Hamiltonian)
    let mut beta_params = Vec::new();
    for i in 0..num_rounds {
        let beta = registry.add_named(format!("beta_{}", i), PI / 4.0);
        beta_params.push(beta);
    }

    // Gamma parameters (problem Hamiltonian)
    let mut gamma_params = Vec::new();
    for i in 0..num_rounds {
        let gamma = registry.add_named(format!("gamma_{}", i), PI / 2.0);
        gamma_params.push(gamma);
    }

    assert_eq!(registry.len(), 6);

    // Update only beta parameters
    let new_betas = vec![0.1, 0.2, 0.3];
    registry.set_values(&beta_params, &new_betas).unwrap();

    // Gamma parameters should remain unchanged
    for gamma_id in &gamma_params {
        assert_eq!(registry.get(*gamma_id).unwrap().value(), PI / 2.0);
    }

    // Beta parameters should be updated
    for (i, beta_id) in beta_params.iter().enumerate() {
        assert_eq!(registry.get(*beta_id).unwrap().value(), new_betas[i]);
    }
}

#[test]
fn test_parameter_freezing_for_optimization() {
    let mut registry = ParameterRegistry::new();

    // Some parameters are optimized, others are fixed
    let opt_param1 = registry.add_named("optimizable_1", 0.5);
    let fixed_param = registry.add(Parameter::named("fixed", 1.0).as_frozen());
    let opt_param2 = registry.add_named("optimizable_2", 0.75);

    // Get only unfrozen parameters for optimization
    let unfrozen = registry.unfrozen_params();
    assert_eq!(unfrozen.len(), 2);
    assert!(unfrozen.contains(&opt_param1));
    assert!(unfrozen.contains(&opt_param2));
    assert!(!unfrozen.contains(&fixed_param));

    // Optimizer only updates unfrozen parameters
    registry.set_values(&unfrozen, &[0.6, 0.85]).unwrap();

    assert_eq!(registry.get(opt_param1).unwrap().value(), 0.6);
    assert_eq!(registry.get(opt_param2).unwrap().value(), 0.85);
    assert_eq!(registry.get(fixed_param).unwrap().value(), 1.0); // Unchanged
}

#[test]
fn test_parameter_bounds_enforcement() {
    let mut registry = ParameterRegistry::new();

    // Add parameter with bounds [0, 2π]
    let param = Parameter::new(PI).with_bounds(0.0, 2.0 * PI).unwrap();
    let id = registry.add(param);

    // Valid update
    registry.get_mut(id).unwrap().set_value(PI / 2.0).unwrap();
    assert_eq!(registry.get(id).unwrap().value(), PI / 2.0);

    // Invalid update (out of bounds)
    let result = registry.get_mut(id).unwrap().set_value(3.0 * PI);
    assert!(result.is_err());

    // Value should remain unchanged
    assert_eq!(registry.get(id).unwrap().value(), PI / 2.0);
}

#[test]
fn test_batch_parameter_update_performance() {
    // Simulate optimization loop with many parameters
    let mut registry = ParameterRegistry::with_capacity(100);

    // Add 100 parameters
    let ids: Vec<_> = (0..100)
        .map(|i| registry.add_named(format!("param_{}", i), i as f64 * 0.01))
        .collect();

    // Batch update all parameters (as in optimization iteration)
    let new_values: Vec<f64> = (0..100).map(|i| (i as f64 * 0.01) + 0.1).collect();

    registry.set_values(&ids, &new_values).unwrap();

    // Verify all updated
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(registry.get(*id).unwrap().value(), new_values[i]);
    }
}

#[test]
fn test_parameter_name_lookup() {
    let mut registry = ParameterRegistry::new();

    // Add parameters with meaningful names
    registry.add_named("input_angle", 0.5);
    registry.add_named("entanglement_strength", 1.2);
    registry.add_named("measurement_basis", PI / 4.0);

    // Look up by name
    assert_eq!(registry.get_by_name("input_angle").unwrap().value(), 0.5);
    assert_eq!(
        registry
            .get_by_name("entanglement_strength")
            .unwrap()
            .value(),
        1.2
    );

    // Get ID by name
    let id = registry.get_id_by_name("measurement_basis").unwrap();
    assert_eq!(registry.get(id).unwrap().value(), PI / 4.0);
}

#[test]
fn test_zero_copy_value_access() {
    let mut registry = ParameterRegistry::new();
    registry.add_many(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    // Get all values efficiently (no intermediate allocations in hot path)
    let values = registry.all_values();
    assert_eq!(values.len(), 5);
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    // Set all values efficiently
    registry.set_all_values(&[1.1, 2.2, 3.3, 4.4, 5.5]).unwrap();
    let updated = registry.all_values();
    assert_eq!(updated, vec![1.1, 2.2, 3.3, 4.4, 5.5]);
}

#[test]
fn test_hardware_efficient_ansatz_parameters() {
    // Hardware-efficient ansatz: alternating single-qubit rotations and entangling gates
    let mut registry = ParameterRegistry::new();
    let num_qubits = 4;
    let num_layers = 3;

    for layer in 0..num_layers {
        // Single-qubit rotations for each qubit
        for qubit in 0..num_qubits {
            registry.add_named(format!("layer_{}_qubit_{}_rx", layer, qubit), 0.1);
            registry.add_named(format!("layer_{}_qubit_{}_rz", layer, qubit), 0.2);
        }
    }

    // Total parameters: 3 layers * 4 qubits * 2 rotations = 24
    assert_eq!(registry.len(), 24);

    // Access specific parameter
    let param_name = "layer_1_qubit_2_rx";
    assert!(registry.get_by_name(param_name).is_ok());
}

#[test]
fn test_parameter_iteration() {
    let mut registry = ParameterRegistry::new();

    registry.add_named("alpha", 1.0);
    registry.add_named("beta", 2.0);
    registry.add_named("gamma", 3.0);

    // Iterate and collect values
    let mut values = Vec::new();
    for (_, param) in registry.iter() {
        values.push(param.value());
    }

    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_empty_registry() {
    let registry = ParameterRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
    assert_eq!(registry.all_ids().len(), 0);
    assert_eq!(registry.all_values().len(), 0);
}

#[test]
fn test_parameter_metadata_preservation() {
    let mut registry = ParameterRegistry::new();

    // Add parameter with rich metadata
    let param = Parameter::named("complex_param", 1.5)
        .with_bounds(0.0, 2.0 * PI)
        .unwrap()
        .as_frozen();

    let id = registry.add(param);

    // Verify all metadata preserved
    let retrieved = registry.get(id).unwrap();
    assert_eq!(retrieved.name(), Some("complex_param"));
    assert_eq!(retrieved.value(), 1.5);
    assert_eq!(retrieved.bounds(), Some((0.0, 2.0 * PI)));
    assert!(retrieved.is_frozen());
}
