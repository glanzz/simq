//! Optimizer Comparison Example
//!
//! This example compares all available optimizers on a simple VQE problem
//! to demonstrate their different convergence characteristics.
//!
//! Run with: cargo run --example optimizer_comparison

use simq_sim::Simulator;
use simq_sim::gradient::{
    VQEOptimizer, VQEConfig,
    AdamOptimizer, AdamConfig,
    MomentumOptimizer, MomentumConfig,
    gradient_descent,
};
use simq_core::{Circuit, QubitId};
use simq_state::observable::{PauliObservable, PauliString};
use simq_gates::standard::{RotationY, RotationZ, CNot};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Optimizer Comparison on VQE Problem\n");
    println!("====================================\n");

    // Simple VQE problem: 2-qubit system
    let num_qubits = 2;
    let simulator = Simulator::new(Default::default());

    // Observable: Z0Z1 (maximize correlation)
    let observable = PauliObservable::from_pauli_string(PauliString::from_str("ZZ")?, -1.0);

    // Ansatz circuit
    let circuit_builder = |params: &[f64]| {
        let mut circuit = Circuit::new(num_qubits);
        circuit.add_gate(Arc::new(RotationY::new(params[0])), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(Arc::new(RotationY::new(params[1])), &[QubitId::new(1)]).unwrap();
        circuit.add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)]).unwrap();
        circuit.add_gate(Arc::new(RotationZ::new(params[2])), &[QubitId::new(0)]).unwrap();
        circuit.add_gate(Arc::new(RotationZ::new(params[3])), &[QubitId::new(1)]).unwrap();
        circuit
    };

    // Same initial parameters for all optimizers
    let initial_params = vec![0.5, 0.5, 0.5, 0.5];

    // Compute initial energy
    let initial_circuit = circuit_builder(&initial_params);
    let initial_result = simulator.run(&initial_circuit)?;
    let initial_energy = match &initial_result.state {
        simq_state::AdaptiveState::Dense(dense) => {
            observable.expectation_value(dense)?
        }
        simq_state::AdaptiveState::Sparse { state: sparse, .. } => {
            use simq_state::DenseState;
            let dense = DenseState::from_sparse(sparse)?;
            observable.expectation_value(&dense)?
        }
    };
    println!("Initial energy: {:.8}\n", initial_energy);

    // Storage for results
    let mut results = Vec::new();

    // 1. Vanilla Gradient Descent
    println!("1. Vanilla Gradient Descent");
    println!("{:-<60}", "");
    let start = Instant::now();
    let gd_result = gradient_descent(
        &simulator,
        &circuit_builder,
        &observable,
        &initial_params,
        0.1,  // learning rate
        100,  // max iterations
    )?;
    let gd_time = start.elapsed();

    println!("Final energy:    {:.8}", gd_result.energy);
    println!("Iterations:      {}", gd_result.num_iterations);
    println!("Gradient norm:   {:.8}", gd_result.gradient.iter().map(|g| g*g).sum::<f64>().sqrt());
    println!("Time:            {:?}", gd_time);
    println!("Status:          {:?}\n", gd_result.status);

    results.push(("Gradient Descent", gd_result.energy, gd_result.num_iterations, gd_time));

    // 2. VQE with Adaptive Learning Rate
    println!("2. VQE with Adaptive Learning Rate");
    println!("{:-<60}", "");
    let vqe_config = VQEConfig {
        max_iterations: 100,
        learning_rate: 0.1,
        adaptive_learning_rate: true,
        energy_tolerance: 1e-8,
        gradient_tolerance: 1e-8,
        ..Default::default()
    };

    let mut vqe_optimizer = VQEOptimizer::new(&circuit_builder, vqe_config);
    let vqe_result = vqe_optimizer.optimize(&simulator, &observable, &initial_params)?;

    println!("Final energy:    {:.8}", vqe_result.energy);
    println!("Iterations:      {}", vqe_result.num_iterations);
    println!("Gradient norm:   {:.8}", vqe_result.gradient.iter().map(|g| g*g).sum::<f64>().sqrt());
    println!("Time:            {:?}", vqe_result.total_time);
    println!("Status:          {:?}\n", vqe_result.status);

    results.push(("VQE Adaptive", vqe_result.energy, vqe_result.num_iterations, vqe_result.total_time));

    // 3. Momentum Optimizer
    println!("3. Momentum Optimizer");
    println!("{:-<60}", "");
    let momentum_config = MomentumConfig {
        learning_rate: 0.05,
        momentum: 0.9,
        max_iterations: 100,
        energy_tolerance: 1e-8,
        gradient_tolerance: 1e-8,
    };

    let mut momentum_optimizer = MomentumOptimizer::new(&circuit_builder, momentum_config);
    let momentum_result = momentum_optimizer.optimize(&simulator, &observable, &initial_params)?;

    println!("Final energy:    {:.8}", momentum_result.energy);
    println!("Iterations:      {}", momentum_result.num_iterations);
    println!("Gradient norm:   {:.8}", momentum_result.gradient.iter().map(|g| g*g).sum::<f64>().sqrt());
    println!("Time:            {:?}", momentum_result.total_time);
    println!("Status:          {:?}\n", momentum_result.status);

    results.push(("Momentum", momentum_result.energy, momentum_result.num_iterations, momentum_result.total_time));

    // 4. Adam Optimizer
    println!("4. Adam Optimizer");
    println!("{:-<60}", "");
    let adam_config = AdamConfig {
        learning_rate: 0.1,
        max_iterations: 100,
        energy_tolerance: 1e-8,
        gradient_tolerance: 1e-8,
        ..Default::default()
    };

    let mut adam_optimizer = AdamOptimizer::new(&circuit_builder, adam_config);
    let adam_result = adam_optimizer.optimize(&simulator, &observable, &initial_params)?;

    println!("Final energy:    {:.8}", adam_result.energy);
    println!("Iterations:      {}", adam_result.num_iterations);
    println!("Gradient norm:   {:.8}", adam_result.gradient.iter().map(|g| g*g).sum::<f64>().sqrt());
    println!("Time:            {:?}", adam_result.total_time);
    println!("Status:          {:?}\n", adam_result.status);

    results.push(("Adam", adam_result.energy, adam_result.num_iterations, adam_result.total_time));

    // Summary comparison
    println!("\nSummary Comparison");
    println!("{:=<80}", "");
    println!("{:25} {:>15} {:>12} {:>12} {:>12}",
        "Optimizer", "Final Energy", "Iterations", "Time (μs)", "Energy Gain");
    println!("{:-<80}", "");

    for (name, energy, iters, time) in &results {
        let energy_gain = initial_energy - energy;
        println!("{:25} {:>15.8} {:>12} {:>12} {:>12.8}",
            name, energy, iters, time.as_micros(), energy_gain);
    }

    // Find best result
    let best = results.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    println!("{:=<80}", "");
    println!("Best optimizer: {} (energy = {:.8})", best.0, best.1);

    // Convergence analysis
    println!("\n\nConvergence Analysis");
    println!("{:=<80}", "");

    // Show first 10 iterations for each optimizer
    println!("\nGradient Descent trajectory:");
    print_trajectory(&gd_result.history, 10);

    println!("\nVQE Adaptive trajectory:");
    print_trajectory(&vqe_result.history, 10);

    println!("\nMomentum trajectory:");
    print_trajectory(&momentum_result.history, 10);

    println!("\nAdam trajectory:");
    print_trajectory(&adam_result.history, 10);

    println!("\n✓ Comparison complete!");
    println!("\nKey Insights:");
    println!("- Adam typically converges fastest with adaptive per-parameter learning rates");
    println!("- Momentum helps escape local minima and accelerate in consistent directions");
    println!("- Adaptive VQE adjusts learning rate based on progress");
    println!("- Vanilla gradient descent is simplest but may be slower");

    Ok(())
}

fn print_trajectory(history: &[simq_sim::gradient::OptimizationStep], n: usize) {
    println!("{:>4} {:>15} {:>15} {:>15}",
        "Iter", "Energy", "Grad Norm", "Energy Change");
    for step in history.iter().take(n) {
        println!("{:>4} {:>15.8} {:>15.8} {:>15.8}",
            step.iteration, step.energy, step.gradient_norm, step.energy_change);
    }
}
