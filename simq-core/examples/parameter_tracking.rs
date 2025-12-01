//! Parameter tracking examples for VQE and QAOA

use simq_core::{Parameter, ParameterRegistry};
use std::f64::consts::PI;

fn main() {
    println!("=== SimQ Parameter Tracking Examples ===\n");

    example_vqe_parameters();
    println!();

    example_qaoa_parameters();
    println!();

    example_parameter_optimization();
    println!();

    example_parameter_constraints();
}

fn example_vqe_parameters() {
    println!("Example 1: VQE Parameter Management");
    println!("------------------------------------");
    println!("Variational Quantum Eigensolver with parameterized ansatz\n");

    let mut registry = ParameterRegistry::new();

    // Create parameters for a 3-qubit, 2-layer VQE ansatz
    println!("Setting up VQE ansatz parameters:");

    // Layer 1
    let theta_0_l1 = registry.add_named("theta_0_layer1", 0.1);
    let _theta_1_l1 = registry.add_named("theta_1_layer1", 0.2);
    let _theta_2_l1 = registry.add_named("theta_2_layer1", 0.3);

    // Layer 2
    let _theta_0_l2 = registry.add_named("theta_0_layer2", 0.4);
    let _theta_1_l2 = registry.add_named("theta_1_layer2", 0.5);
    let theta_2_l2 = registry.add_named("theta_2_layer2", 0.6);

    println!("  Created {} parameters", registry.len());
    println!("  Parameter names:");
    for (id, param) in registry.iter() {
        println!("    {}: {} = {:.3}", id, param.name().unwrap(), param.value());
    }

    // Simulate optimization iteration
    println!("\nSimulating optimization iteration...");
    let new_values = vec![0.15, 0.25, 0.35, 0.45, 0.55, 0.65];
    let all_ids = registry.all_ids();
    registry.set_values(&all_ids, &new_values).unwrap();

    println!("Updated all parameters:");
    println!("  theta_0_layer1 = {:.3}", registry.get(theta_0_l1).unwrap().value());
    println!("  theta_2_layer2 = {:.3}", registry.get(theta_2_l2).unwrap().value());

    println!("\n✓ VQE parameter management complete!");
}

fn example_qaoa_parameters() {
    println!("Example 2: QAOA Parameter Structure");
    println!("------------------------------------");
    println!("Quantum Approximate Optimization Algorithm with p=3 rounds\n");

    let mut registry = ParameterRegistry::new();
    let num_rounds = 3;

    // QAOA uses two types of parameters: beta and gamma
    let mut beta_params = Vec::new();
    let mut gamma_params = Vec::new();

    println!("Creating QAOA parameters for {} rounds:", num_rounds);

    for i in 0..num_rounds {
        // Beta parameters (mixer Hamiltonian angles)
        let beta = registry.add_named(format!("beta_{}", i), PI / (4.0 * (i + 1) as f64));
        beta_params.push(beta);

        // Gamma parameters (problem Hamiltonian angles)
        let gamma = registry.add_named(format!("gamma_{}", i), PI / (2.0 * (i + 1) as f64));
        gamma_params.push(gamma);

        println!(
            "  Round {}: beta_{} = {:.4}, gamma_{} = {:.4}",
            i,
            i,
            registry.get(beta).unwrap().value(),
            i,
            registry.get(gamma).unwrap().value()
        );
    }

    println!("\nTotal parameters: {}", registry.len());

    // Update only beta parameters (typical in layerwise optimization)
    println!("\nOptimizing beta parameters while keeping gamma fixed:");
    let new_betas = vec![0.2, 0.3, 0.4];
    registry.set_values(&beta_params, &new_betas).unwrap();

    for (i, beta_id) in beta_params.iter().enumerate() {
        println!("  beta_{} updated to {:.3}", i, registry.get(*beta_id).unwrap().value());
    }

    println!("\n✓ QAOA parameter structure demonstration complete!");
}

fn example_parameter_optimization() {
    println!("Example 3: Parameter Optimization with Freezing");
    println!("-----------------------------------------------");
    println!("Selective parameter optimization\n");

    let mut registry = ParameterRegistry::new();

    // Some parameters should be optimized, others fixed
    let opt1 = registry.add_named("optimizable_theta", 0.5);
    let fixed1 = registry.add(Parameter::named("fixed_bias", 1.0).as_frozen());
    let opt2 = registry.add_named("optimizable_phi", 0.75);
    let fixed2 = registry.add(Parameter::named("fixed_scale", 2.0).as_frozen());

    println!("Parameter setup:");
    println!(
        "  Optimizable: theta={:.2}, phi={:.2}",
        registry.get(opt1).unwrap().value(),
        registry.get(opt2).unwrap().value()
    );
    println!(
        "  Fixed: bias={:.2}, scale={:.2}",
        registry.get(fixed1).unwrap().value(),
        registry.get(fixed2).unwrap().value()
    );

    // Get only unfrozen parameters for optimizer
    let unfrozen_ids = registry.unfrozen_params();
    println!("\nParameters available for optimization: {}", unfrozen_ids.len());

    // Optimizer updates only unfrozen parameters
    println!("\nApplying gradient descent step...");
    let updated_values = vec![0.6, 0.85]; // Optimized values
    registry.set_values(&unfrozen_ids, &updated_values).unwrap();

    println!("\nAfter optimization:");
    println!(
        "  Optimizable: theta={:.2}, phi={:.2}",
        registry.get(opt1).unwrap().value(),
        registry.get(opt2).unwrap().value()
    );
    println!(
        "  Fixed: bias={:.2}, scale={:.2} (unchanged)",
        registry.get(fixed1).unwrap().value(),
        registry.get(fixed2).unwrap().value()
    );

    println!("\n✓ Selective optimization demonstration complete!");
}

fn example_parameter_constraints() {
    println!("Example 4: Parameter Bounds and Constraints");
    println!("-------------------------------------------");
    println!("Enforcing physical constraints on parameters\n");

    let mut registry = ParameterRegistry::new();

    // Add parameters with physical bounds
    let angle = Parameter::named("rotation_angle", PI)
        .with_bounds(0.0, 2.0 * PI)
        .unwrap();
    let angle_id = registry.add(angle);

    let phase = Parameter::named("phase_shift", 0.0)
        .with_bounds(-PI, PI)
        .unwrap();
    let phase_id = registry.add(phase);

    println!("Parameters with bounds:");
    for (_id, param) in registry.iter() {
        if let Some((min, max)) = param.bounds() {
            println!(
                "  {}: {:.3} ∈ [{:.3}, {:.3}]",
                param.name().unwrap(),
                param.value(),
                min,
                max
            );
        }
    }

    // Try valid update
    println!("\nAttempting valid update:");
    registry
        .get_mut(angle_id)
        .unwrap()
        .set_value(PI / 2.0)
        .unwrap();
    println!("  ✓ rotation_angle updated to {:.3}", registry.get(angle_id).unwrap().value());

    // Try invalid update
    println!("\nAttempting invalid update (out of bounds):");
    let result = registry.get_mut(phase_id).unwrap().set_value(4.0 * PI);
    match result {
        Ok(_) => println!("  Unexpected success"),
        Err(e) => println!("  ✗ Correctly rejected: {}", e),
    }

    println!("\n✓ Constraint enforcement demonstration complete!");
}
