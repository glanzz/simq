//! Demonstrates automatic Sparse↔Dense conversion in AdaptiveState
//!
//! Run with: cargo run --example adaptive_conversion

use num_complex::Complex64;
use simq_state::{AdaptiveState, DenseState, SparseState};

fn main() {
    println!("=== Automatic Sparse↔Dense Conversion Demo ===\n");

    // Example 1: Start sparse, automatically convert to dense
    println!("1. Automatic Conversion on Entangling Gates:");
    println!("   Creating 10-qubit state (starts sparse)...");

    let mut state = AdaptiveState::new(10).unwrap();
    println!(
        "   Initial: {} | Density: {:.2}%",
        state.representation(),
        state.density() * 100.0
    );

    // Hadamard gate matrix
    let h = 1.0 / 2.0_f64.sqrt();
    let hadamard = [
        [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
        [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
    ];

    // Apply Hadamards to create superposition
    println!("   Applying Hadamard gates...");
    for qubit in 0..10 {
        let converted = state.apply_single_qubit_gate(&hadamard, qubit).unwrap();
        let stats = state.stats();

        println!(
            "   After H on qubit {}: {} | Density: {:.2}% | Memory: {} entries",
            qubit,
            stats.representation,
            stats.density * 100.0,
            stats.memory_entries
        );

        if converted {
            println!("   ✓ Automatic conversion to Dense occurred!");
        }
    }
    println!();

    // Example 2: Manual conversions between representations
    println!("2. Manual Sparse↔Dense Conversion:");

    // Create a sparse state
    let sparse = SparseState::new(5).unwrap();
    println!("   Sparse state: {} amplitudes stored", sparse.num_amplitudes());

    // Convert to dense
    let dense = DenseState::from_sparse(&sparse).unwrap();
    println!("   Dense state: {} amplitudes (2^5 = 32)", dense.dimension());

    // Convert back to sparse
    let sparse_again = dense.to_sparse().unwrap();
    println!("   Back to sparse: {} amplitudes stored", sparse_again.num_amplitudes());
    println!();

    // Example 3: Custom threshold
    println!("3. Custom Conversion Threshold:");

    let mut custom_state = AdaptiveState::with_threshold(5, 0.5).unwrap();
    println!("   Created with 50% threshold");
    println!("   Initial density: {:.2}%", custom_state.density() * 100.0);

    // Apply gates
    custom_state.apply_single_qubit_gate(&hadamard, 0).unwrap();
    custom_state.apply_single_qubit_gate(&hadamard, 1).unwrap();
    custom_state.apply_single_qubit_gate(&hadamard, 2).unwrap();

    println!(
        "   After 3 Hadamards: {} | Density: {:.2}%",
        custom_state.representation(),
        custom_state.density() * 100.0
    );
    println!("   Still sparse because density < 50% threshold");
    println!();

    // Example 4: Memory efficiency comparison
    println!("4. Memory Efficiency Comparison:");

    let num_qubits = 20;
    println!("   For {} qubits:", num_qubits);

    let sparse_state = SparseState::new(num_qubits).unwrap();
    let sparse_memory = sparse_state.num_amplitudes() * std::mem::size_of::<Complex64>();
    println!(
        "   Sparse: {} bytes ({} amplitudes)",
        sparse_memory,
        sparse_state.num_amplitudes()
    );

    let dense_state = DenseState::new(num_qubits).unwrap();
    let dense_memory = dense_state.dimension() * std::mem::size_of::<Complex64>();
    println!("   Dense:  {} bytes ({} amplitudes)", dense_memory, dense_state.dimension());

    let ratio = dense_memory as f64 / sparse_memory as f64;
    println!("   Memory savings: {:.0}x for sparse |0...0⟩ state", ratio);
    println!();

    // Example 5: Tracking density during simulation
    println!("5. Density Tracking During Circuit:");

    let mut adaptive = AdaptiveState::new(6).unwrap();
    println!("   Starting 6-qubit simulation...");

    // Track density over multiple gates
    let gates = vec![("H@0", 0), ("H@1", 1), ("H@2", 2), ("H@3", 3)];

    for (name, qubit) in gates {
        adaptive.apply_single_qubit_gate(&hadamard, qubit).unwrap();
        println!(
            "   After {}: {:.2}% density ({})",
            name,
            adaptive.density() * 100.0,
            adaptive.representation()
        );
    }

    println!();

    // Example 6: Best practices
    println!("6. Best Practices:");
    println!("   ✓ Start with AdaptiveState for most simulations");
    println!("   ✓ Default 10% threshold works well for most circuits");
    println!("   ✓ Sparse is great for: shallow circuits, few gates, product states");
    println!("   ✓ Dense is great for: highly entangled states, deep circuits");
    println!("   ✓ AdaptiveState handles the transition automatically!");
    println!();

    println!("=== Demo Complete ===");
}
