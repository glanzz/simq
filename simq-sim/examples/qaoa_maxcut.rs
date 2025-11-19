//! QAOA Example: MaxCut Problem
//!
//! This example demonstrates using QAOA (Quantum Approximate Optimization Algorithm)
//! to solve the MaxCut problem on a simple graph.
//!
//! Run with: cargo run --example qaoa_maxcut

use simq_sim::Simulator;
use simq_sim::gradient::{QAOAOptimizer, QAOAConfig, AdamOptimizer, AdamConfig};
use simq_core::{Circuit, Gate};
use simq_state::observable::PauliObservable;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("QAOA: MaxCut Problem\n");
    println!("====================\n");

    // Graph: Triangle (3 nodes, 3 edges)
    // Edges: (0,1), (1,2), (0,2)
    // Goal: Maximize the number of edges between different partitions
    let num_qubits = 3;
    let simulator = Simulator::new(num_qubits);

    // MaxCut Hamiltonian: H_C = -0.5 * sum_{(i,j) in E} (I - Z_i Z_j)
    // For triangle: H_C = -0.5 * ((I - Z0Z1) + (I - Z1Z2) + (I - Z0Z2))
    // We'll use the cost observable: Z0Z1 + Z1Z2 + Z0Z2 (minimize to solve MaxCut)
    let observable = {
        let mut obs = PauliObservable::from_pauli_string("ZZI", 0.5)?; // Z0Z1
        obs = obs.add(&PauliObservable::from_pauli_string("IZZ", 0.5)?)?; // Z1Z2
        obs.add(&PauliObservable::from_pauli_string("ZIZ", 0.5)?)?  // Z0Z2
    };

    // QAOA circuit builder
    // Parameters: [gamma_1, beta_1, gamma_2, beta_2, ...]
    // gamma: problem Hamiltonian (cost function)
    // beta: mixer Hamiltonian (X rotations)
    let num_layers = 2;
    let circuit_builder = move |params: &[f64]| {
        let mut circuit = Circuit::new(num_qubits);

        // Initial state: uniform superposition
        for i in 0..num_qubits {
            circuit.add_gate(Gate::H(i));
        }

        // QAOA layers
        for layer in 0..num_layers {
            let gamma = params[2 * layer];
            let beta = params[2 * layer + 1];

            // Problem Hamiltonian: exp(-i * gamma * H_C)
            // For each edge (i,j), apply: RZZ(2*gamma) on qubits i,j
            // RZZ(theta) = exp(-i * theta/2 * Z_i Z_j)

            // Edge (0,1)
            circuit.add_gate(Gate::CNOT(0, 1));
            circuit.add_gate(Gate::RZ(1, 2.0 * gamma));
            circuit.add_gate(Gate::CNOT(0, 1));

            // Edge (1,2)
            circuit.add_gate(Gate::CNOT(1, 2));
            circuit.add_gate(Gate::RZ(2, 2.0 * gamma));
            circuit.add_gate(Gate::CNOT(1, 2));

            // Edge (0,2)
            circuit.add_gate(Gate::CNOT(0, 2));
            circuit.add_gate(Gate::RZ(2, 2.0 * gamma));
            circuit.add_gate(Gate::CNOT(0, 2));

            // Mixer Hamiltonian: exp(-i * beta * sum_i X_i)
            for i in 0..num_qubits {
                circuit.add_gate(Gate::RX(i, 2.0 * beta));
            }
        }

        circuit
    };

    // Initial parameters (small random values)
    let initial_params = vec![0.1, 0.3, 0.2, 0.4]; // [gamma_1, beta_1, gamma_2, beta_2]

    println!("Graph: Triangle with 3 nodes and 3 edges");
    println!("Edges: (0,1), (1,2), (0,2)");
    println!("QAOA layers (p): {}", num_layers);
    println!("Parameters: {} (2 per layer)\n", initial_params.len());

    println!("Method 1: QAOA with simultaneous optimization\n");
    println!("{:-<70}", "");

    let config = QAOAConfig {
        num_layers,
        max_iterations: 100,
        energy_tolerance: 1e-6,
        gradient_tolerance: 1e-6,
        gamma_learning_rate: 0.1,
        beta_learning_rate: 0.1,
        layer_wise: false,
        ..Default::default()
    };

    let mut optimizer = QAOAOptimizer::new(circuit_builder, config.clone());
    let result = optimizer.optimize(&simulator, &observable, &initial_params)?;

    println!("Initial cost:    {:.8}", result.history[0].energy);
    println!("Final cost:      {:.8}", result.energy);
    println!("Iterations:      {}", result.num_iterations);
    println!("Time:            {:?}", result.total_time);
    println!("Status:          {:?}", result.status);
    println!("\nOptimal parameters:");
    for layer in 0..num_layers {
        println!("  Layer {}: gamma = {:.6}, beta = {:.6}",
            layer + 1,
            result.parameters[2 * layer],
            result.parameters[2 * layer + 1]);
    }

    // Interpret result
    let final_circuit = circuit_builder(&result.parameters);
    let final_result = simulator.run(&final_circuit)?;
    println!("\nTop measurement outcomes (probability > 5%):");
    let amplitudes = final_result.state_vector();
    let mut probs: Vec<(usize, f64)> = amplitudes
        .iter()
        .enumerate()
        .map(|(i, amp)| (i, amp.norm_sqr()))
        .collect();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (state, prob) in probs.iter().take(5) {
        if *prob > 0.05 {
            let bitstring = format!("{:0width$b}", state, width = num_qubits);
            // Calculate cut size
            let bits: Vec<u8> = bitstring.chars().map(|c| c.to_digit(2).unwrap() as u8).collect();
            let mut cut_size = 0;
            // Edge (0,1)
            if bits[0] != bits[1] { cut_size += 1; }
            // Edge (1,2)
            if bits[1] != bits[2] { cut_size += 1; }
            // Edge (0,2)
            if bits[0] != bits[2] { cut_size += 1; }

            println!("  |{}⟩: {:.1}% (cut size: {})", bitstring, prob * 100.0, cut_size);
        }
    }

    println!("\n\nMethod 2: QAOA with layer-wise optimization\n");
    println!("{:-<70}", "");

    let mut config_layerwise = config.clone();
    config_layerwise.layer_wise = true;

    let mut optimizer_layerwise = QAOAOptimizer::new(circuit_builder, config_layerwise);
    let result_layerwise = optimizer_layerwise.optimize(&simulator, &observable, &initial_params)?;

    println!("Initial cost:    {:.8}", initial_params[0]);
    println!("Final cost:      {:.8}", result_layerwise.energy);
    println!("Iterations:      {}", result_layerwise.num_iterations);
    println!("Time:            {:?}", result_layerwise.total_time);
    println!("Status:          {:?}", result_layerwise.status);
    println!("\nOptimal parameters:");
    for layer in 0..num_layers {
        println!("  Layer {}: gamma = {:.6}, beta = {:.6}",
            layer + 1,
            result_layerwise.parameters[2 * layer],
            result_layerwise.parameters[2 * layer + 1]);
    }

    println!("\n\nMethod 3: QAOA with Adam optimizer\n");
    println!("{:-<70}", "");

    let adam_config = AdamConfig {
        learning_rate: 0.1,
        max_iterations: 100,
        energy_tolerance: 1e-6,
        gradient_tolerance: 1e-6,
        ..Default::default()
    };

    let mut adam_optimizer = AdamOptimizer::new(circuit_builder, adam_config);
    let adam_result = adam_optimizer.optimize(&simulator, &observable, &initial_params)?;

    println!("Initial cost:    {:.8}", adam_result.history[0].energy);
    println!("Final cost:      {:.8}", adam_result.energy);
    println!("Iterations:      {}", adam_result.num_iterations);
    println!("Time:            {:?}", adam_result.total_time);
    println!("Status:          {:?}", adam_result.status);

    // Comparison
    println!("\n\nComparison\n");
    println!("{:-<70}", "");
    println!("{:25} {:>15} {:>10} {:>12}", "Method", "Final Cost", "Iterations", "Time (ms)");
    println!("{:-<70}", "");
    println!("{:25} {:>15.8} {:>10} {:>12}",
        "Simultaneous", result.energy, result.num_iterations, result.total_time.as_millis());
    println!("{:25} {:>15.8} {:>10} {:>12}",
        "Layer-wise", result_layerwise.energy, result_layerwise.num_iterations, result_layerwise.total_time.as_millis());
    println!("{:25} {:>15.8} {:>10} {:>12}",
        "Adam", adam_result.energy, adam_result.num_iterations, adam_result.total_time.as_millis());

    println!("\n✓ QAOA optimization complete!");
    println!("\nNote: For this MaxCut problem, the optimal solution cuts 2 edges.");
    println!("States |001⟩, |010⟩, |100⟩, |011⟩, |101⟩, |110⟩ all achieve this (cut size = 2).");

    Ok(())
}
