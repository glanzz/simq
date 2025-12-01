//! Example: Automatic Gradient Computation with Fallback
//!
//! This example demonstrates the automatic gradient computation with fallback
//! mechanism. It shows how the system automatically tries parameter shift rule
//! first and falls back to finite differences if needed.

use simq_core::Circuit;
use simq_sim::gradient::{compute_gradient, compute_gradient_auto, GradientConfig, GradientMethod};
use simq_sim::Simulator;
use simq_state::observable::PauliObservable;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Gradient Computation with Automatic Fallback ===\n");

    // Create simulator
    let simulator = Simulator::new(Default::default());

    // Example 1: Automatic method selection (simplest interface)
    println!("Example 1: Automatic gradient computation");
    println!("{}", "-".repeat(50));

    let params = vec![0.5, 1.0, 1.5];

    // Build a simple parameterized circuit
    let circuit_builder = |_p: &[f64]| {
        // Add parameterized gates here
        Circuit::new(2)
    };

    // Create observable (Z⊗Z)
    let observable = PauliObservable::new();

    // Compute gradient with automatic fallback
    let result = compute_gradient_auto(&simulator, circuit_builder, &observable, &params)?;

    println!("Gradient: {:?}", result.gradients);
    println!("Method used: {:?}", result.method_used);
    println!("Evaluations: {}", result.num_evaluations);
    println!("Time: {:?}\n", result.computation_time);

    // Example 2: Explicit configuration
    println!("Example 2: Explicit configuration");
    println!("{}", "-".repeat(50));

    let config = GradientConfig {
        method: GradientMethod::Auto,
        shift: std::f64::consts::FRAC_PI_2,
        epsilon: 1e-7,
        parallel: true,
        cache_circuits: true,
    };

    let result = compute_gradient(&simulator, circuit_builder, &observable, &params, &config)?;

    println!("Gradient: {:?}", result.gradients);
    println!("Method used: {:?}", result.method_used);
    println!("Gradient norm: {:.6}", result.norm());
    println!();

    // Example 3: Force specific method
    println!("Example 3: Force parameter shift rule");
    println!("{}", "-".repeat(50));

    let ps_config = GradientConfig {
        method: GradientMethod::ParameterShift,
        ..Default::default()
    };

    let result = compute_gradient(&simulator, circuit_builder, &observable, &params, &ps_config)?;

    println!("Gradient: {:?}", result.gradients);
    println!("Method used: {:?}", result.method_used);
    println!();

    // Example 4: Force finite differences
    println!("Example 4: Force finite differences");
    println!("{}", "-".repeat(50));

    let fd_config = GradientConfig {
        method: GradientMethod::FiniteDifference,
        epsilon: 1e-6,
        ..Default::default()
    };

    let result = compute_gradient(&simulator, circuit_builder, &observable, &params, &fd_config)?;

    println!("Gradient: {:?}", result.gradients);
    println!("Method used: {:?}", result.method_used);
    println!();

    // Example 5: VQE-style optimization loop
    println!("Example 5: VQE optimization loop with gradient descent");
    println!("{}", "-".repeat(50));

    let mut params = vec![0.1, 0.2, 0.3];
    let learning_rate = 0.01;
    let num_iterations = 5;

    for iteration in 0..num_iterations {
        // Compute gradient
        let grad_result = compute_gradient_auto(&simulator, circuit_builder, &observable, &params)?;

        // Gradient descent update
        for (p, g) in params.iter_mut().zip(grad_result.gradients.iter()) {
            *p -= learning_rate * g;
        }

        println!(
            "Iteration {}: params = {:?}, |grad| = {:.6}",
            iteration,
            params,
            grad_result.norm()
        );
    }

    println!("\n=== All examples completed successfully! ===");
    Ok(())
}
