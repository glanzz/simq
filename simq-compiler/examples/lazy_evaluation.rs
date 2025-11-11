//! Example demonstrating lazy gate evaluation
//!
//! This example shows how to use the LazyExecutor to execute circuits
//! with deferred matrix computation and caching.
//!
//! Run with: `cargo run --example lazy_evaluation`

use simq_compiler::lazy::{LazyConfig, LazyExecutor};
use simq_core::circuit_builder::CircuitBuilder;
use simq_gates::standard::{CNot, Hadamard, RotationX};
use simq_state::StateVector;
use std::sync::Arc;

fn main() {
    println!("=== Lazy Gate Evaluation Example ===\n");

    // Example 1: Basic Bell state creation with default config
    example_bell_state();

    // Example 2: Parameterized circuit with caching (VQE-like)
    example_parameterized_circuit();

    // Example 3: Large circuit with fusion
    example_large_circuit_with_fusion();

    // Example 4: Cache performance demonstration
    example_cache_performance();
}

fn example_bell_state() {
    println!("Example 1: Creating Bell State");
    println!("--------------------------------");

    // Build circuit: H(q0) → CNOT(q0, q1)
    let mut builder = CircuitBuilder::<2>::new();
    let [q0, q1] = builder.qubits();
    builder.apply_gate(Arc::new(Hadamard), &[q0]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[q0, q1]).unwrap();
    let circuit = builder.build();

    println!("Circuit: H(q0) → CNOT(q0, q1)");
    println!("Expected result: (|00⟩ + |11⟩)/√2\n");

    // Execute with lazy evaluation
    let mut state = StateVector::new(2).unwrap();
    let mut executor = LazyExecutor::new(LazyConfig::default());

    executor
        .execute(&circuit, state.amplitudes_mut())
        .unwrap();

    // Show results
    println!("State amplitudes:");
    for (i, amp) in state.amplitudes().iter().enumerate() {
        if amp.norm() > 1e-10 {
            println!("  |{:02b}⟩: {:.4} + {:.4}i", i, amp.re, amp.im);
        }
    }

    println!("\nExecution statistics:");
    println!("{}", executor.stats());
    println!();
}

fn example_parameterized_circuit() {
    println!("Example 2: Parameterized Circuit (VQE-like)");
    println!("-------------------------------------------");

    // Build a parameterized circuit
    let angle = std::f64::consts::PI / 4.0;
    let mut builder = CircuitBuilder::<2>::new();
    let [q0, q1] = builder.qubits();

    builder
        .apply_gate(Arc::new(RotationX::new(angle)), &[q0])
        .unwrap();
    builder
        .apply_gate(Arc::new(RotationX::new(angle)), &[q1])
        .unwrap();
    builder.apply_gate(Arc::new(CNot), &[q0, q1]).unwrap();
    builder
        .apply_gate(Arc::new(RotationX::new(angle)), &[q0])
        .unwrap();

    let circuit = builder.build();

    println!("Circuit with repeated RX(π/4) gates");
    println!("Matrix caching should improve performance\n");

    let mut state = StateVector::new(2).unwrap();
    let mut executor = LazyExecutor::new(LazyConfig::default());

    // Execute multiple times (simulating VQE iterations)
    println!("Executing circuit 5 times:");
    for i in 1..=5 {
        let mut fresh_state = StateVector::new(2).unwrap(); // Reset state
        executor
            .execute(&circuit, fresh_state.amplitudes_mut())
            .unwrap();
        println!("  Iteration {}: {} matrices computed", i, executor.stats().matrices_computed);
        state = fresh_state; // Keep last state
    }

    println!("\nFinal statistics:");
    println!("{}", executor.stats());
    println!("{}", executor.cache_stats());
    println!();
}

fn example_large_circuit_with_fusion() {
    println!("Example 3: Large Circuit with Gate Fusion");
    println!("-----------------------------------------");

    // Build a circuit with many fusible gates
    let mut builder = CircuitBuilder::<3>::new();
    let [q0, q1, q2] = builder.qubits();

    // Add many single-qubit gates that can be fused
    for _ in 0..20 {
        builder.apply_gate(Arc::new(Hadamard), &[q0]).unwrap();
        builder.apply_gate(Arc::new(Hadamard), &[q1]).unwrap();
        builder.apply_gate(Arc::new(Hadamard), &[q2]).unwrap();
    }

    let circuit = builder.build();
    println!("Circuit with 60 Hadamard gates (20 per qubit)");

    let mut state = StateVector::new(3).unwrap();

    // Execute with fusion enabled
    println!("\nWith fusion enabled:");
    let config_fusion = LazyConfig {
        enable_fusion: true,
        ..Default::default()
    };
    let mut executor_fusion = LazyExecutor::new(config_fusion);
    executor_fusion
        .execute(&circuit, state.amplitudes_mut())
        .unwrap();
    println!("  Gates fused: {}", executor_fusion.stats().gates_fused);
    println!("  Matrices computed: {}", executor_fusion.stats().matrices_computed);

    // Execute without fusion
    state = StateVector::new(3).unwrap(); // Reset state
    println!("\nWithout fusion:");
    let config_no_fusion = LazyConfig {
        enable_fusion: false,
        ..Default::default()
    };
    let mut executor_no_fusion = LazyExecutor::new(config_no_fusion);
    executor_no_fusion
        .execute(&circuit, state.amplitudes_mut())
        .unwrap();
    println!("  Gates fused: {}", executor_no_fusion.stats().gates_fused);
    println!("  Matrices computed: {}", executor_no_fusion.stats().matrices_computed);

    println!("\nFusion reduced matrix computations significantly!\n");
}

fn example_cache_performance() {
    println!("Example 4: Cache Performance");
    println!("----------------------------");

    // Build a circuit with repeated parameterized gates
    let angle1 = std::f64::consts::PI / 3.0;
    let angle2 = std::f64::consts::PI / 4.0;

    let mut builder = CircuitBuilder::<2>::new();
    let [q0, _q1] = builder.qubits();

    // Alternate between two angles
    for i in 0..10 {
        let angle = if i % 2 == 0 { angle1 } else { angle2 };
        builder
            .apply_gate(Arc::new(RotationX::new(angle)), &[q0])
            .unwrap();
    }

    let circuit = builder.build();
    println!("Circuit with 10 RX gates alternating between two angles");

    let mut state = StateVector::new(2).unwrap();

    // With caching
    println!("\nWith caching enabled:");
    let config_cache = LazyConfig {
        enable_caching: true,
        enable_fusion: false, // Disable to see pure caching effect
        ..Default::default()
    };
    let mut executor_cache = LazyExecutor::new(config_cache);
    executor_cache
        .execute(&circuit, state.amplitudes_mut())
        .unwrap();
    let stats = executor_cache.stats();
    println!("  Matrices computed: {}", stats.matrices_computed);
    println!("  Cache hits: {}", stats.cache_hits);
    println!("  Cache misses: {}", stats.cache_misses);
    println!("  Hit rate: {:.1}%", stats.cache_hit_rate() * 100.0);

    // Without caching
    state = StateVector::new(2).unwrap(); // Reset state
    println!("\nWithout caching:");
    let config_no_cache = LazyConfig {
        enable_caching: false,
        enable_fusion: false,
        ..Default::default()
    };
    let mut executor_no_cache = LazyExecutor::new(config_no_cache);
    executor_no_cache
        .execute(&circuit, state.amplitudes_mut())
        .unwrap();
    let stats_no_cache = executor_no_cache.stats();
    println!("  Matrices computed: {}", stats_no_cache.matrices_computed);
    println!("  Cache hits: {}", stats_no_cache.cache_hits);
    println!("  Cache misses: {}", stats_no_cache.cache_misses);

    println!("\nCaching reduced computations from {} to {}!",
        stats_no_cache.matrices_computed,
        stats.matrices_computed);
    println!();
}
