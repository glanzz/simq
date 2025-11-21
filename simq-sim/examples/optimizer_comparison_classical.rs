//! Classical Optimizers Comparison for VQE
//!
//! This example compares different classical optimization algorithms
//! (L-BFGS, Nelder-Mead, Adam, Gradient Descent) for a VQE problem.
//!
//! Run with: cargo run --example optimizer_comparison_classical

use simq_sim::Simulator;
use simq_sim::gradient::{
    LBFGSOptimizer, LBFGSConfig,
    NelderMeadOptimizer, NelderMeadConfig,
    AdamOptimizer, AdamConfig,
    VQEOptimizer, VQEConfig,
};
use simq_core::{Circuit, QubitId};
use simq_state::observable::{PauliObservable, PauliString, Pauli};
use simq_gates::{Hadamard, RotationY, CNot};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{:=<80}", "");
    println!("  CLASSICAL OPTIMIZER COMPARISON FOR VQE");
    println!("{:=<80}\n", "");

    // Problem: H2 molecule Hamiltonian (simplified)
    // H = -1.0523 * I - 0.3979 * Z0 - 0.3979 * Z1 - 0.0112 * Z0Z1 + 0.1809 * X0X1
    //
    // We'll focus on the Z0Z1 term for this comparison

    let num_qubits = 2;
    let simulator = Simulator::new(num_qubits);

    // Create observable: Z0 ⊗ Z1
    let mut paulis = vec![Pauli::I; num_qubits];
    paulis[0] = Pauli::Z;
    paulis[1] = Pauli::Z;
    let observable = PauliObservable::from_pauli_string(PauliString::from_paulis(paulis), -0.5);

    println!("Problem: VQE for H2 molecule (simplified)");
    println!("Observable: -0.5 * Z0⊗Z1");
    println!("Ansatz: Hardware-efficient (4 parameters)");
    println!("Circuit: H⊗H - RY(θ0)⊗RY(θ1) - CNOT(0,1) - RY(θ2)⊗RY(θ3)\n");

    // Hardware-efficient ansatz
    let circuit_builder = |params: &[f64]| {
        let mut circuit = Circuit::new(num_qubits);

        // Initial Hadamard layer
        let _ = circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]);
        let _ = circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(1)]);

        // Parameterized rotations
        let _ = circuit.add_gate(Arc::new(RotationY::new(params[0])), &[QubitId::new(0)]);
        let _ = circuit.add_gate(Arc::new(RotationY::new(params[1])), &[QubitId::new(1)]);

        // Entangling layer
        let _ = circuit.add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)]);

        // Second rotation layer
        let _ = circuit.add_gate(Arc::new(RotationY::new(params[2])), &[QubitId::new(0)]);
        let _ = circuit.add_gate(Arc::new(RotationY::new(params[3])), &[QubitId::new(1)]);

        circuit
    };

    // Initial parameters (same for all optimizers for fair comparison)
    let initial_params = vec![0.5, 0.3, -0.2, 0.7];

    println!("Initial parameters: {:?}", initial_params);
    let init_circuit = circuit_builder(&initial_params);
    let init_result = simulator.run(&init_circuit)?;
    let init_energy = match &init_result.state {
        simq_state::AdaptiveState::Dense(dense) => {
            observable.expectation_value(dense)?
        }
        simq_state::AdaptiveState::Sparse { state: sparse, .. } => {
            use simq_state::DenseState;
            let dense = DenseState::from_sparse(sparse);
            observable.expectation_value(&dense)?
        }
    };
    println!("Initial energy: {:.8}\n", init_energy);

    println!("{:=<80}", "");
    println!("  OPTIMIZATION RESULTS");
    println!("{:=<80}\n", "");

    // ========================================================================
    // 1. L-BFGS Optimizer
    // ========================================================================

    println!("1. L-BFGS (Limited-memory BFGS)");
    println!("{:-<60}", "");
    println!("   Properties: Quasi-Newton, uses numerical gradients");
    println!("   Best for: Smooth functions, many parameters");
    println!();

    let lbfgs_config = LBFGSConfig {
        max_iterations: 50,
        tolerance: 1e-6,
        memory_size: 10,
        ..Default::default()
    };

    let mut lbfgs_optimizer = LBFGSOptimizer::new(circuit_builder, lbfgs_config);
    let lbfgs_result = lbfgs_optimizer.optimize(&simulator, &observable, &initial_params)?;

    println!("   Final energy:    {:.8}", lbfgs_result.energy);
    println!("   Iterations:      {}", lbfgs_result.num_iterations);
    println!("   Time:            {:?}", lbfgs_result.total_time);
    println!("   Status:          {:?}", lbfgs_result.status);
    println!("   Converged:       {}", lbfgs_result.converged());
    println!("   Optimal params:  [{:.4}, {:.4}, {:.4}, {:.4}]",
        lbfgs_result.parameters[0],
        lbfgs_result.parameters[1],
        lbfgs_result.parameters[2],
        lbfgs_result.parameters[3]);
    println!();

    // ========================================================================
    // 2. Nelder-Mead Optimizer
    // ========================================================================

    println!("2. Nelder-Mead (Simplex Method)");
    println!("{:-<60}", "");
    println!("   Properties: Gradient-free, robust to noise");
    println!("   Best for: Noisy functions, few parameters, NISQ devices");
    println!();

    let nm_config = NelderMeadConfig {
        max_iterations: 100,
        tolerance: 1e-6,
        ..Default::default()
    };

    let mut nm_optimizer = NelderMeadOptimizer::new(circuit_builder, nm_config);
    let nm_result = nm_optimizer.optimize(&simulator, &observable, &initial_params)?;

    println!("   Final energy:    {:.8}", nm_result.energy);
    println!("   Iterations:      {}", nm_result.num_iterations);
    println!("   Time:            {:?}", nm_result.total_time);
    println!("   Status:          {:?}", nm_result.status);
    println!("   Converged:       {}", nm_result.converged());
    println!("   Optimal params:  [{:.4}, {:.4}, {:.4}, {:.4}]",
        nm_result.parameters[0],
        nm_result.parameters[1],
        nm_result.parameters[2],
        nm_result.parameters[3]);
    println!();

    // ========================================================================
    // 3. Adam Optimizer
    // ========================================================================

    println!("3. Adam (Adaptive Moment Estimation)");
    println!("{:-<60}", "");
    println!("   Properties: Adaptive learning rate, momentum");
    println!("   Best for: Fast convergence, automatic tuning");
    println!();

    let adam_config = AdamConfig {
        learning_rate: 0.1,
        max_iterations: 100,
        energy_tolerance: 1e-6,
        gradient_tolerance: 1e-6,
        ..Default::default()
    };

    let mut adam_optimizer = AdamOptimizer::new(circuit_builder, adam_config);
    let adam_result = adam_optimizer.optimize(&simulator, &observable, &initial_params)?;

    println!("   Final energy:    {:.8}", adam_result.energy);
    println!("   Iterations:      {}", adam_result.num_iterations);
    println!("   Time:            {:?}", adam_result.total_time);
    println!("   Status:          {:?}", adam_result.status);
    println!("   Converged:       {}", adam_result.converged());
    println!("   Optimal params:  [{:.4}, {:.4}, {:.4}, {:.4}]",
        adam_result.parameters[0],
        adam_result.parameters[1],
        adam_result.parameters[2],
        adam_result.parameters[3]);
    println!();

    // ========================================================================
    // 4. Standard Gradient Descent
    // ========================================================================

    println!("4. Gradient Descent (with adaptive learning rate)");
    println!("{:-<60}", "");
    println!("   Properties: Simple, adaptive learning rate");
    println!("   Best for: Baseline comparison");
    println!();

    let vqe_config = VQEConfig {
        max_iterations: 100,
        learning_rate: 0.1,
        adaptive_learning_rate: true,
        energy_tolerance: 1e-6,
        gradient_tolerance: 1e-6,
        ..Default::default()
    };

    let mut vqe_optimizer = VQEOptimizer::new(circuit_builder, vqe_config);
    let vqe_result = vqe_optimizer.optimize(&simulator, &observable, &initial_params)?;

    println!("   Final energy:    {:.8}", vqe_result.energy);
    println!("   Iterations:      {}", vqe_result.num_iterations);
    println!("   Time:            {:?}", vqe_result.total_time);
    println!("   Status:          {:?}", vqe_result.status);
    println!("   Converged:       {}", vqe_result.converged());
    println!("   Optimal params:  [{:.4}, {:.4}, {:.4}, {:.4}]",
        vqe_result.parameters[0],
        vqe_result.parameters[1],
        vqe_result.parameters[2],
        vqe_result.parameters[3]);
    println!();

    // ========================================================================
    // Summary Comparison
    // ========================================================================

    println!("\n{:=<80}", "");
    println!("  SUMMARY COMPARISON");
    println!("{:=<80}\n", "");

    println!("{:<25} {:>15} {:>10} {:>12} {:>12}",
        "Optimizer", "Final Energy", "Iterations", "Time (ms)", "Converged");
    println!("{:-<80}", "");

    println!("{:<25} {:>15.8} {:>10} {:>12} {:>12}",
        "L-BFGS",
        lbfgs_result.energy,
        lbfgs_result.num_iterations,
        lbfgs_result.total_time.as_millis(),
        if lbfgs_result.converged() { "Yes" } else { "No" });

    println!("{:<25} {:>15.8} {:>10} {:>12} {:>12}",
        "Nelder-Mead",
        nm_result.energy,
        nm_result.num_iterations,
        nm_result.total_time.as_millis(),
        if nm_result.converged() { "Yes" } else { "No" });

    println!("{:<25} {:>15.8} {:>10} {:>12} {:>12}",
        "Adam",
        adam_result.energy,
        adam_result.num_iterations,
        adam_result.total_time.as_millis(),
        if adam_result.converged() { "Yes" } else { "No" });

    println!("{:<25} {:>15.8} {:>10} {:>12} {:>12}",
        "Gradient Descent",
        vqe_result.energy,
        vqe_result.num_iterations,
        vqe_result.total_time.as_millis(),
        if vqe_result.converged() { "Yes" } else { "No" });

    println!("\n{:=<80}", "");
    println!("  RECOMMENDATIONS");
    println!("{:=<80}\n", "");

    println!("📊 **L-BFGS**");
    println!("   ✓ Use for noiseless simulations");
    println!("   ✓ Best for smooth, well-behaved objective functions");
    println!("   ✓ Excellent convergence with many parameters");
    println!();

    println!("🎯 **Nelder-Mead**");
    println!("   ✓ Use for noisy quantum hardware (NISQ devices)");
    println!("   ✓ Best for few parameters (<10)");
    println!("   ✓ Most robust to measurement noise");
    println!("   ✓ Doesn't require gradients");
    println!();

    println!("⚡ **Adam**");
    println!("   ✓ Good general-purpose optimizer");
    println!("   ✓ Fast convergence in most cases");
    println!("   ✓ Self-adaptive, minimal tuning needed");
    println!();

    println!("📈 **Gradient Descent**");
    println!("   ✓ Simple baseline");
    println!("   ✓ Use when gradients are cheap to compute");
    println!("   ✓ Requires careful learning rate tuning");
    println!();

    Ok(())
}
