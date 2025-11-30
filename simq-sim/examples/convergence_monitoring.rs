//! Convergence Monitoring Example
//!
//! This example demonstrates the convergence monitoring capabilities
//! for VQE/QAOA optimization, including:
//! - Real-time progress tracking
//! - Early stopping strategies
//! - Convergence diagnostics
//! - Custom callbacks
//!
//! Run with: cargo run --example convergence_monitoring

use simq_core::{Circuit, QubitId};
use simq_gates::{CNot, Hadamard, RotationY};
use simq_sim::gradient::{
    compute_gradient_auto, progress_callback, target_energy_callback, ConvergenceMonitor,
    MonitorConfig, StoppingCriterion,
};
use simq_sim::Simulator;
use simq_state::observable::{Pauli, PauliObservable, PauliString};
use simq_state::AdaptiveState;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{:=<80}", "");
    println!("  CONVERGENCE MONITORING DEMONSTRATION");
    println!("{:=<80}\n", "");

    // Setup: Simple VQE problem
    let num_qubits = 2;
    let simulator = Simulator::new(Default::default());

    // Observable: Z0 ⊗ Z1
    let mut paulis = vec![Pauli::I; num_qubits];
    paulis[0] = Pauli::Z;
    paulis[1] = Pauli::Z;
    let observable = PauliObservable::from_pauli_string(PauliString::from_paulis(paulis), 1.0);

    // Ansatz circuit builder
    let circuit_builder = |params: &[f64]| {
        let mut circuit = Circuit::new(num_qubits);
        let _ = circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]);
        let _ = circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(1)]);
        let _ = circuit.add_gate(Arc::new(RotationY::new(params[0])), &[QubitId::new(0)]);
        let _ = circuit.add_gate(Arc::new(RotationY::new(params[1])), &[QubitId::new(1)]);
        let _ = circuit.add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)]);
        circuit
    };

    // ========================================================================
    // Example 1: Basic Convergence Monitoring
    // ========================================================================

    println!("{:-<80}", "");
    println!("EXAMPLE 1: Basic Convergence Monitoring");
    println!("{:-<80}\n", "");

    let config = MonitorConfig::default()
        .with_energy_tolerance(1e-4)
        .with_gradient_tolerance(1e-4)
        .with_max_iterations(50);

    let mut monitor = ConvergenceMonitor::new(config);
    let mut params = vec![0.5, 0.3];
    let learning_rate = 0.1;

    println!("Running gradient descent with monitoring...\n");

    for iteration in 0..50 {
        // Compute energy
        let circuit = circuit_builder(&params);
        let result = simulator.run(&circuit)?;
        let energy = compute_expectation(&result, &observable)?;

        // Compute gradient
        let grad_result = compute_gradient_auto(&simulator, circuit_builder, &observable, &params)?;

        // Record metrics
        monitor.record(iteration, energy, &grad_result.gradients, &params);

        // Check stopping condition
        if monitor.should_stop() {
            println!("\nStopping: {}", monitor.stopping_criterion().description());
            break;
        }

        // Update parameters
        for (param, grad) in params.iter_mut().zip(grad_result.gradients.iter()) {
            *param -= learning_rate * grad;
        }
    }

    // Print convergence report
    let report = monitor.convergence_report();
    report.print();

    // ========================================================================
    // Example 2: Monitoring with Callbacks
    // ========================================================================

    println!("\n{:-<80}", "");
    println!("EXAMPLE 2: Monitoring with Custom Callbacks");
    println!("{:-<80}\n", "");

    // Create monitor with progress callback
    let config_with_callback = MonitorConfig::default()
        .with_energy_tolerance(1e-4)
        .with_max_iterations(30)
        .with_callback(progress_callback(5)); // Print every 5 iterations

    let mut monitor2 = ConvergenceMonitor::new(config_with_callback);
    let mut params2 = vec![0.8, -0.2];

    println!("Running with progress callback (every 5 iterations):\n");

    for iteration in 0..30 {
        let circuit = circuit_builder(&params2);
        let result = simulator.run(&circuit)?;
        let energy = compute_expectation(&result, &observable)?;
        let grad_result =
            compute_gradient_auto(&simulator, circuit_builder, &observable, &params2)?;

        monitor2.record(iteration, energy, &grad_result.gradients, &params2);

        if monitor2.should_stop() {
            break;
        }

        for (param, grad) in params2.iter_mut().zip(grad_result.gradients.iter()) {
            *param -= learning_rate * grad;
        }
    }

    println!("\nFinal result: {}", monitor2.convergence_report().summary());

    // ========================================================================
    // Example 3: Early Stopping with Patience
    // ========================================================================

    println!("\n{:-<80}", "");
    println!("EXAMPLE 3: Early Stopping with Patience");
    println!("{:-<80}\n", "");

    // Start near a local minimum to trigger patience
    let config_patience = MonitorConfig::default()
        .with_energy_tolerance(1e-8) // Very strict
        .with_patience(5) // Stop after 5 iterations without improvement
        .with_max_iterations(100);

    let mut monitor3 = ConvergenceMonitor::new(config_patience);
    let mut params3 = vec![0.0, 0.0]; // Start at a suboptimal point

    println!("Running with patience=5 (will stop if no improvement):\n");

    for iteration in 0..100 {
        let circuit = circuit_builder(&params3);
        let result = simulator.run(&circuit)?;
        let energy = compute_expectation(&result, &observable)?;
        let grad_result =
            compute_gradient_auto(&simulator, circuit_builder, &observable, &params3)?;

        monitor3.record(iteration, energy, &grad_result.gradients, &params3);

        if iteration % 10 == 0 {
            println!(
                "Iteration {:3}: energy = {:.6}, best = {:.6}",
                iteration,
                energy,
                monitor3.best_energy()
            );
        }

        if monitor3.should_stop() {
            println!(
                "\nStopped at iteration {}: {}",
                iteration,
                monitor3.stopping_criterion().description()
            );
            break;
        }

        // Very small updates to simulate slow progress
        for (param, grad) in params3.iter_mut().zip(grad_result.gradients.iter()) {
            *param -= 0.01 * grad;
        }
    }

    println!(
        "\nBest energy found: {:.8} at iteration {}",
        monitor3.best_energy(),
        monitor3.best_iteration()
    );

    // ========================================================================
    // Example 4: Target Energy Callback
    // ========================================================================

    println!("\n{:-<80}", "");
    println!("EXAMPLE 4: Target Energy Callback");
    println!("{:-<80}\n", "");

    let target_energy = -0.5;

    let config_target = MonitorConfig::default()
        .with_max_iterations(100)
        .with_callback(target_energy_callback(target_energy));

    let mut monitor4 = ConvergenceMonitor::new(config_target);
    let mut params4 = vec![1.0, 1.0];

    println!("Running until energy reaches {} ...\n", target_energy);

    for iteration in 0..100 {
        let circuit = circuit_builder(&params4);
        let result = simulator.run(&circuit)?;
        let energy = compute_expectation(&result, &observable)?;
        let grad_result =
            compute_gradient_auto(&simulator, circuit_builder, &observable, &params4)?;

        monitor4.record(iteration, energy, &grad_result.gradients, &params4);

        if monitor4.should_stop() {
            if monitor4.stopping_criterion() == StoppingCriterion::UserStop {
                println!("Target energy {} reached at iteration {}!", target_energy, iteration);
            }
            break;
        }

        for (param, grad) in params4.iter_mut().zip(grad_result.gradients.iter()) {
            *param -= 0.2 * grad;
        }
    }

    println!("Final energy: {:.6}", monitor4.best_energy());

    // ========================================================================
    // Example 5: Verbose Mode with Full Diagnostics
    // ========================================================================

    println!("\n{:-<80}", "");
    println!("EXAMPLE 5: Verbose Mode with Full Diagnostics");
    println!("{:-<80}\n", "");

    let config_verbose = MonitorConfig::default()
        .with_energy_tolerance(1e-4)
        .with_gradient_tolerance(1e-4)
        .with_max_iterations(15)
        .with_verbose(true);

    let mut monitor5 = ConvergenceMonitor::new(config_verbose);
    let mut params5 = vec![0.7, -0.3];

    println!("Running in verbose mode (automatic logging):\n");

    for iteration in 0..15 {
        let circuit = circuit_builder(&params5);
        let result = simulator.run(&circuit)?;
        let energy = compute_expectation(&result, &observable)?;
        let grad_result =
            compute_gradient_auto(&simulator, circuit_builder, &observable, &params5)?;

        monitor5.record(iteration, energy, &grad_result.gradients, &params5);

        if monitor5.should_stop() {
            break;
        }

        for (param, grad) in params5.iter_mut().zip(grad_result.gradients.iter()) {
            *param -= 0.15 * grad;
        }
    }

    // Full diagnostic report
    monitor5.convergence_report().print();

    // ========================================================================
    // Summary
    // ========================================================================

    println!("\n{:=<80}", "");
    println!("  MONITORING FEATURES DEMONSTRATED");
    println!("{:=<80}\n", "");

    println!("1. Basic Monitoring     - Track energy, gradients, convergence");
    println!("2. Progress Callbacks   - Custom actions during optimization");
    println!("3. Early Stopping       - Patience-based termination");
    println!("4. Target Energy        - Stop when reaching a goal");
    println!("5. Verbose Diagnostics  - Detailed logging and reports");

    println!("\nKey Features:");
    println!("  - Multiple stopping criteria (energy, gradient, patience, time)");
    println!("  - Best solution tracking");
    println!("  - Oscillation detection");
    println!("  - Barren plateau warnings");
    println!("  - Convergence reports with diagnostics");
    println!();

    Ok(())
}

/// Helper to compute expectation value
fn compute_expectation(
    result: &simq_sim::SimulationResult,
    observable: &PauliObservable,
) -> Result<f64, Box<dyn std::error::Error>> {
    let expectation = match &result.state {
        AdaptiveState::Dense(dense) => observable.expectation_value(dense)?,
        AdaptiveState::Sparse { state: sparse, .. } => {
            use simq_state::DenseState;
            let dense = DenseState::from_sparse(sparse)?;
            observable.expectation_value(&dense)?
        },
    };
    Ok(expectation)
}
