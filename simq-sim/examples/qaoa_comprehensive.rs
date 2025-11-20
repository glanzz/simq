//! Comprehensive QAOA Example
//!
//! This example demonstrates the new QAOA circuit generator with various
//! problem types, mixer strategies, and optimization approaches.
//!
//! Run with: cargo run --example qaoa_comprehensive

use simq_sim::Simulator;
use simq_sim::qaoa::{
    QAOACircuitBuilder, ProblemType, MixerType, Graph,
    evaluate_maxcut_solution, random_initial_parameters,
};
use simq_sim::gradient::{AdamOptimizer, AdamConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{:=<80}", "");
    println!("  COMPREHENSIVE QAOA DEMONSTRATION");
    println!("{:=<80}\n", "");

    // ========================================================================
    // Example 1: MaxCut on Different Graph Topologies
    // ========================================================================

    println!("\n{:-<80}", "");
    println!("EXAMPLE 1: MaxCut on Different Graph Topologies");
    println!("{:-<80}\n", "");

    // 1a. Triangle Graph
    println!("1a. Triangle Graph (Complete graph with 3 vertices)");
    println!("{:.<40}", "");
    let triangle = Graph::complete(3);
    run_maxcut_example(&triangle, "Triangle", 2)?;

    // 1b. Cycle Graph
    println!("\n1b. Cycle Graph (4 vertices in a ring)");
    println!("{:.<40}", "");
    let cycle = Graph::cycle(4);
    run_maxcut_example(&cycle, "Cycle-4", 2)?;

    // 1c. Star Graph
    println!("\n1c. Star Graph (1 center + 4 peripheral vertices)");
    println!("{:.<40}", "");
    let star = Graph::star(5);
    run_maxcut_example(&star, "Star-5", 2)?;

    // 1d. Grid Graph
    println!("\n1d. Grid Graph (2x3 lattice)");
    println!("{:.<40}", "");
    let grid = Graph::grid(2, 3);
    run_maxcut_example(&grid, "Grid-2x3", 3)?;

    // ========================================================================
    // Example 2: Different Mixer Strategies
    // ========================================================================

    println!("\n\n{:-<80}", "");
    println!("EXAMPLE 2: Comparing Different Mixer Strategies");
    println!("{:-<80}\n", "");

    let graph = Graph::cycle(4);
    let mixers = vec![
        (MixerType::StandardX, "Standard X Mixer"),
        (MixerType::StandardY, "Standard Y Mixer"),
        (MixerType::XY, "XY Mixer (Hamming weight preserving)"),
        (MixerType::Ring, "Ring Mixer (nearest-neighbor)"),
    ];

    for (mixer_type, name) in mixers {
        println!("\nMixer: {}", name);
        println!("{:.<40}", "");
        run_maxcut_with_mixer(&graph, mixer_type, 2)?;
    }

    // ========================================================================
    // Example 3: Number Partitioning Problem
    // ========================================================================

    println!("\n\n{:-<80}", "");
    println!("EXAMPLE 3: Number Partitioning Problem");
    println!("{:-<80}\n", "");

    let numbers = vec![3.0, 1.0, 1.0, 2.0, 2.0, 1.0];
    println!("Numbers to partition: {:?}", numbers);
    println!("Goal: Find two subsets with equal (or close) sums\n");

    let builder = QAOACircuitBuilder::new(
        ProblemType::NumberPartitioning(numbers.clone()),
        MixerType::StandardX,
        3, // depth
    );

    let simulator = Simulator::new(builder.num_qubits());
    let observable = builder.cost_observable()?;

    println!("Problem size: {} numbers → {} qubits", numbers.len(), builder.num_qubits());
    println!("QAOA depth: {}", 3);
    println!("Parameters: {}", builder.num_parameters());

    // Initial parameters
    let initial_params = random_initial_parameters(3, Some(42));

    // Optimize
    let adam_config = AdamConfig {
        learning_rate: 0.05,
        max_iterations: 50,
        energy_tolerance: 1e-6,
        gradient_tolerance: 1e-6,
        ..Default::default()
    };

    let mut optimizer = AdamOptimizer::new(|p| builder.build(p), adam_config);
    let result = optimizer.optimize(&simulator, &observable, &initial_params)?;

    println!("\nOptimization Results:");
    println!("  Initial cost:  {:.6}", result.history[0].energy);
    println!("  Final cost:    {:.6}", result.energy);
    println!("  Iterations:    {}", result.num_iterations);
    println!("  Converged:     {}", result.converged());

    // Analyze solution
    let final_circuit = builder.build(&result.parameters);
    let final_result = simulator.run(&final_circuit)?;

    println!("\nTop solutions (probability > 5%):");
    let amplitudes = final_result.state_vector();
    let mut probs: Vec<(usize, f64)> = amplitudes
        .iter()
        .enumerate()
        .map(|(i, amp)| (i, amp.norm_sqr()))
        .collect();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (state, prob) in probs.iter().take(5) {
        if *prob > 0.05 {
            let bitstring = format!("{:0width$b}", state, width = numbers.len());
            let bits: Vec<bool> = bitstring.chars().map(|c| c == '1').collect();

            let mut sum_a = 0.0;
            let mut sum_b = 0.0;
            for (i, &num) in numbers.iter().enumerate() {
                if bits[i] {
                    sum_a += num;
                } else {
                    sum_b += num;
                }
            }

            println!("  |{}⟩: {:.1}% → A={:.1}, B={:.1}, |diff|={:.1}",
                bitstring, prob * 100.0, sum_a, sum_b, (sum_a - sum_b).abs());
        }
    }

    // ========================================================================
    // Example 4: Varying QAOA Depth
    // ========================================================================

    println!("\n\n{:-<80}", "");
    println!("EXAMPLE 4: Impact of QAOA Depth (p)");
    println!("{:-<80}\n", "");

    let graph = Graph::complete(4);
    println!("Problem: MaxCut on K4 (complete graph, 4 vertices)");
    println!("Comparing QAOA performance for different depths\n");

    println!("{:<10} {:<15} {:<15} {:<12}", "Depth (p)", "Final Cost", "Iterations", "Time (ms)");
    println!("{:-<60}", "");

    for depth in [1, 2, 3, 4] {
        let builder = QAOACircuitBuilder::new(
            ProblemType::MaxCut(graph.clone()),
            MixerType::StandardX,
            depth,
        );

        let simulator = Simulator::new(builder.num_qubits());
        let observable = builder.cost_observable()?;
        let initial_params = random_initial_parameters(depth, Some(42 + depth as u64));

        let adam_config = AdamConfig {
            learning_rate: 0.1,
            max_iterations: 30,
            ..Default::default()
        };

        let mut optimizer = AdamOptimizer::new(|p| builder.build(p), adam_config);
        let result = optimizer.optimize(&simulator, &observable, &initial_params)?;

        println!("{:<10} {:<15.8} {:<15} {:<12}",
            depth,
            result.energy,
            result.num_iterations,
            result.total_time.as_millis());
    }

    // ========================================================================
    // Example 5: Custom Problem Hamiltonian
    // ========================================================================

    println!("\n\n{:-<80}", "");
    println!("EXAMPLE 5: Custom Problem Hamiltonian");
    println!("{:-<80}\n", "");

    println!("Custom Hamiltonian: H = Z0 + 0.5*Z0*Z1 + 0.3*Z1*Z2 + 0.2*Z2");

    let custom_terms = vec![
        (vec![0], 1.0),           // Z0
        (vec![0, 1], 0.5),        // Z0*Z1
        (vec![1, 2], 0.3),        // Z1*Z2
        (vec![2], 0.2),           // Z2
    ];

    let builder = QAOACircuitBuilder::new(
        ProblemType::Custom {
            num_qubits: 3,
            terms: custom_terms,
        },
        MixerType::StandardX,
        2,
    );

    let simulator = Simulator::new(3);
    let observable = builder.cost_observable()?;
    let initial_params = vec![0.5, 0.3, 0.7, 0.4];

    let adam_config = AdamConfig {
        learning_rate: 0.1,
        max_iterations: 40,
        ..Default::default()
    };

    let mut optimizer = AdamOptimizer::new(|p| builder.build(p), adam_config);
    let result = optimizer.optimize(&simulator, &observable, &initial_params)?;

    println!("Optimization completed:");
    println!("  Final cost:    {:.6}", result.energy);
    println!("  Iterations:    {}", result.num_iterations);
    println!("  Status:        {:?}", result.status);

    // ========================================================================
    // Summary
    // ========================================================================

    println!("\n\n{:=<80}", "");
    println!("  DEMONSTRATION COMPLETE");
    println!("{:=<80}\n", "");

    println!("Key Features Demonstrated:");
    println!("  ✓ Multiple graph topologies (complete, cycle, star, grid)");
    println!("  ✓ Various mixer strategies (X, Y, XY, Ring)");
    println!("  ✓ Different problem types (MaxCut, Number Partitioning)");
    println!("  ✓ QAOA depth analysis (p=1 to p=4)");
    println!("  ✓ Custom Hamiltonian construction");
    println!("  ✓ Adam optimizer integration");
    println!("\n");

    Ok(())
}

// Helper function to run MaxCut example
fn run_maxcut_example(graph: &Graph, name: &str, depth: usize) -> Result<(), Box<dyn std::error::Error>> {
    let builder = QAOACircuitBuilder::new(
        ProblemType::MaxCut(graph.clone()),
        MixerType::StandardX,
        depth,
    );

    let simulator = Simulator::new(builder.num_qubits());
    let observable = builder.cost_observable()?;

    println!("Graph: {}", name);
    println!("  Vertices: {}, Edges: {}", graph.num_vertices, graph.num_edges());
    println!("  QAOA depth: p={}", depth);

    // Quick optimization
    let initial_params = random_initial_parameters(depth, Some(42));
    let adam_config = AdamConfig {
        learning_rate: 0.1,
        max_iterations: 30,
        energy_tolerance: 1e-6,
        ..Default::default()
    };

    let mut optimizer = AdamOptimizer::new(|p| builder.build(p), adam_config);
    let result = optimizer.optimize(&simulator, &observable, &initial_params)?;

    println!("  Final cost: {:.6} (in {} iterations)", result.energy, result.num_iterations);

    // Find best cut
    let final_circuit = builder.build(&result.parameters);
    let final_result = simulator.run(&final_circuit)?;
    let amplitudes = final_result.state_vector();

    let (best_state, best_prob) = amplitudes
        .iter()
        .enumerate()
        .map(|(i, amp)| (i, amp.norm_sqr()))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    let bitstring = format!("{:0width$b}", best_state, width = graph.num_vertices);
    let bits: Vec<bool> = bitstring.chars().map(|c| c == '1').collect();
    let cut_value = evaluate_maxcut_solution(graph, &bits);

    println!("  Best solution: |{}⟩ ({:.1}%) → cut value = {:.1}",
        bitstring, best_prob * 100.0, cut_value);

    Ok(())
}

// Helper function to run MaxCut with different mixers
fn run_maxcut_with_mixer(
    graph: &Graph,
    mixer: MixerType,
    depth: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let builder = QAOACircuitBuilder::new(
        ProblemType::MaxCut(graph.clone()),
        mixer,
        depth,
    );

    let simulator = Simulator::new(builder.num_qubits());
    let observable = builder.cost_observable()?;

    let initial_params = random_initial_parameters(depth, Some(42));
    let adam_config = AdamConfig {
        learning_rate: 0.1,
        max_iterations: 25,
        ..Default::default()
    };

    let mut optimizer = AdamOptimizer::new(|p| builder.build(p), adam_config);
    let result = optimizer.optimize(&simulator, &observable, &initial_params)?;

    println!("  Cost: {:.6}, Iterations: {}, Time: {:?}",
        result.energy, result.num_iterations, result.total_time);

    Ok(())
}
