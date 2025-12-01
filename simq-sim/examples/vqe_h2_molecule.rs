//! VQE Example: H2 Molecule Ground State Energy
//!
//! This example demonstrates using VQE (Variational Quantum Eigensolver)
//! to find the ground state energy of the H2 molecule.
//!
//! Run with: cargo run --example vqe_h2_molecule

use simq_core::{Circuit, QubitId};
use simq_gates::standard::{CNot, RotationY};
use simq_sim::gradient::{AdamConfig, AdamOptimizer, VQEConfig, VQEOptimizer};
use simq_sim::Simulator;
use simq_state::observable::{PauliObservable, PauliString};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("VQE: H2 Molecule Ground State Energy\n");
    println!("=====================================\n");

    // Create simulator
    let num_qubits = 2;
    let simulator = Simulator::new(Default::default());

    // H2 molecule Hamiltonian (simplified, at equilibrium bond length)
    // H = -1.0523 * I - 0.3979 * Z0 - 0.3979 * Z1 - 0.0112 * Z0Z1 + 0.1809 * X0X1
    // We'll optimize the X0X1 term component
    let observable = PauliObservable::from_pauli_string(PauliString::from_str("XX")?, -0.1809);

    // VQE ansatz: Hardware-efficient ansatz
    // Circuit structure: RY(θ0) ⊗ RY(θ1) - CNOT(0,1) - RY(θ2) ⊗ RY(θ3)
    let circuit_builder = |params: &[f64]| {
        let mut circuit = Circuit::new(num_qubits);

        // Layer 1: Single-qubit rotations
        circuit
            .add_gate(Arc::new(RotationY::new(params[0])), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(RotationY::new(params[1])), &[QubitId::new(1)])
            .unwrap();

        // Entangling layer
        circuit
            .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();

        // Layer 2: Single-qubit rotations
        circuit
            .add_gate(Arc::new(RotationY::new(params[2])), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(RotationY::new(params[3])), &[QubitId::new(1)])
            .unwrap();

        circuit
    };

    // Initial parameters (random initialization)
    let initial_params = vec![0.5, 0.3, -0.2, 0.7];

    println!("Method 1: VQE with adaptive gradient descent\n");
    println!("{:-<60}", "");

    let config = VQEConfig {
        max_iterations: 100,
        learning_rate: 0.1,
        adaptive_learning_rate: true,
        energy_tolerance: 1e-6,
        gradient_tolerance: 1e-6,
        ..Default::default()
    };

    let mut optimizer = VQEOptimizer::new(circuit_builder, config);
    let result = optimizer.optimize(&simulator, &observable, &initial_params)?;

    println!("Initial energy:  {:.8}", result.history[0].energy);
    println!("Final energy:    {:.8}", result.energy);
    println!("Iterations:      {}", result.num_iterations);
    println!("Time:            {:?}", result.total_time);
    println!("Status:          {:?}", result.status);
    println!("Converged:       {}", result.converged());
    println!("\nOptimal parameters:");
    for (i, &p) in result.parameters.iter().enumerate() {
        println!("  θ{} = {:.6}", i, p);
    }
    println!(
        "\nGradient norm:   {:.8}",
        result.gradient.iter().map(|g| g * g).sum::<f64>().sqrt()
    );

    // Show optimization trajectory
    println!("\nOptimization trajectory (first 10 steps):");
    println!("{:-<60}", "");
    println!("{:>4} {:>15} {:>15} {:>15}", "Iter", "Energy", "Grad Norm", "Energy Change");
    for step in result.history.iter().take(10) {
        println!(
            "{:>4} {:>15.8} {:>15.8} {:>15.8}",
            step.iteration, step.energy, step.gradient_norm, step.energy_change
        );
    }

    println!("\n\nMethod 2: VQE with Adam optimizer\n");
    println!("{:-<60}", "");

    let adam_config = AdamConfig {
        learning_rate: 0.1,
        max_iterations: 100,
        energy_tolerance: 1e-6,
        gradient_tolerance: 1e-6,
        ..Default::default()
    };

    let mut adam_optimizer = AdamOptimizer::new(circuit_builder, adam_config);
    let adam_result = adam_optimizer.optimize(&simulator, &observable, &initial_params)?;

    println!("Initial energy:  {:.8}", adam_result.history[0].energy);
    println!("Final energy:    {:.8}", adam_result.energy);
    println!("Iterations:      {}", adam_result.num_iterations);
    println!("Time:            {:?}", adam_result.total_time);
    println!("Status:          {:?}", adam_result.status);
    println!("Converged:       {}", adam_result.converged());
    println!("\nOptimal parameters:");
    for (i, &p) in adam_result.parameters.iter().enumerate() {
        println!("  θ{} = {:.6}", i, p);
    }

    // Compare methods
    println!("\n\nComparison\n");
    println!("{:-<60}", "");
    println!(
        "{:20} {:>15} {:>10} {:>12}",
        "Method", "Final Energy", "Iterations", "Time (ms)"
    );
    println!("{:-<60}", "");
    println!(
        "{:20} {:>15.8} {:>10} {:>12}",
        "Adaptive GD",
        result.energy,
        result.num_iterations,
        result.total_time.as_millis()
    );
    println!(
        "{:20} {:>15.8} {:>10} {:>12}",
        "Adam",
        adam_result.energy,
        adam_result.num_iterations,
        adam_result.total_time.as_millis()
    );

    println!("\n✓ VQE optimization complete!");

    Ok(())
}
