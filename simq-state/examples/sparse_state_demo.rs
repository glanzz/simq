//! SparseState implementation example showcasing Phase 2.1 of SimQ development plan
//!
//! This example demonstrates:
//! - Creating sparse states for various quantum circuits
//! - Tracking density and automatic conversion thresholds
//! - Applying single-qubit and two-qubit gates efficiently
//! - Measuring qubits and computing expectation values
//! - Memory efficiency compared to dense representations

use num_complex::Complex64;
use simq_state::SparseState;

fn main() {
    println!("=== SimQ SparseState Implementation (Phase 2.1) ===\n");

    example_1_basic_sparse_state();
    example_2_bell_state_creation();
    example_3_density_tracking();
    example_4_gate_operations();
    example_5_measurement_and_collapse();
    example_6_memory_efficiency();
}

/// Example 1: Create and inspect basic sparse states
fn example_1_basic_sparse_state() {
    println!("### Example 1: Basic Sparse State Creation ###");

    // Create a 10-qubit state (normally would require 2^10 = 1024 amplitudes in dense form)
    let state = SparseState::new(10).unwrap();
    println!("Created 10-qubit state initialized to |0...0⟩");
    println!("  - Dimension: 2^10 = {}", state.dimension());
    println!("  - Non-zero amplitudes: {}", state.num_amplitudes());
    println!(
        "  - Density: {:.4}% (only {} bytes to store 1 amplitude!)",
        state.density() * 100.0,
        std::mem::size_of::<Complex64>()
    );

    // Create a specific basis state
    let state_5 = SparseState::from_basis_state(5, 5).unwrap();
    println!("\nCreated 5-qubit state |00101⟩:");
    println!("  - Amplitude at index 5: {}", state_5.get_amplitude(5));
    println!("  - Amplitude at index 0: {}", state_5.get_amplitude(0));
    println!();
}

/// Example 2: Create a Bell state using single-qubit gates
fn example_2_bell_state_creation() {
    println!("### Example 2: Bell State Creation (|00⟩ + |11⟩)/√2 ###");

    let mut state = SparseState::new(2).unwrap();
    println!("Initial state: |00⟩");
    println!("  Density: {:.4}%\n", state.density() * 100.0);

    // Apply Hadamard to first qubit
    let h_gate = [
        Complex64::new(0.707106781, 0.0),
        Complex64::new(0.707106781, 0.0),
        Complex64::new(0.707106781, 0.0),
        Complex64::new(-0.707106781, 0.0),
    ];

    state.apply_single_qubit_gate(&h_gate, 0).unwrap();
    println!("After H on qubit 0: (|0⟩ + |1⟩)|0⟩/√2 = (|00⟩ + |10⟩)/√2");
    println!("  Non-zero amplitudes: {}", state.num_amplitudes());
    println!("  Density: {:.4}%\n", state.density() * 100.0);

    // Apply CNOT (qubit 0 control, qubit 1 target)
    let cnot = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];

    state.apply_two_qubit_gate(&cnot, 0, 1).unwrap();
    println!("After CNOT: (|00⟩ + |11⟩)/√2 (maximally entangled Bell state!)");
    println!("  Non-zero amplitudes: {}", state.num_amplitudes());
    println!("  Density: {:.4}%\n", state.density() * 100.0);

    // Verify the Bell state
    let amp00 = state.get_amplitude(0);
    let amp11 = state.get_amplitude(3);
    println!("Verification:");
    println!("  Amplitude |00⟩: {:.4}", amp00.norm());
    println!("  Amplitude |11⟩: {:.4}", amp11.norm());
    println!(
        "  Both equal to 1/√2 ≈ 0.707? {}\n",
        (amp00.norm() - 0.707).abs() < 0.01 && (amp11.norm() - 0.707).abs() < 0.01
    );
}

/// Example 3: Track density and conversion thresholds
fn example_3_density_tracking() {
    println!("### Example 3: Density Tracking and Conversion Thresholds ###");

    let mut state = SparseState::new(4).unwrap();
    println!("4-qubit state (16 possible basis states)");
    println!("Initial density: {:.4}%\n", state.density() * 100.0);

    state.set_density_threshold(0.25);
    println!("Set density threshold to 25%");
    println!("  Should convert to dense? {}\n", state.should_convert_to_dense());

    // Add amplitudes gradually
    for i in 1..4 {
        state.set_amplitude(i, Complex64::new(0.5, 0.0));
        println!("Added amplitude at index {}", i);
        println!("  Current density: {:.4}%", state.density() * 100.0);
        println!("  Should convert to dense? {}", state.should_convert_to_dense());

        if state.should_convert_to_dense() {
            println!("  → Density exceeded threshold! Would convert to dense representation");
            break;
        }
        println!();
    }
}

/// Example 4: Efficient gate operations
fn example_4_gate_operations() {
    println!("\n### Example 4: Efficient Gate Operations on Sparse States ###");

    let mut state = SparseState::new(8).unwrap();

    // Create a product state by applying gates
    let rx_90 = [
        Complex64::new(0.707, 0.0),
        Complex64::new(0.0, -0.707),
        Complex64::new(0.0, -0.707),
        Complex64::new(0.707, 0.0),
    ];

    println!("8-qubit state, applying RX(π/2) to qubits 0, 2, 4, 6");
    println!("Initial amplitudes: 1 (state |00000000⟩)");

    state.apply_single_qubit_gate(&rx_90, 0).unwrap();
    state.apply_single_qubit_gate(&rx_90, 2).unwrap();
    state.apply_single_qubit_gate(&rx_90, 4).unwrap();
    state.apply_single_qubit_gate(&rx_90, 6).unwrap();

    println!("After operations: {} non-zero amplitudes", state.num_amplitudes());
    println!("Density: {:.4}%", state.density() * 100.0);
    println!(
        "For dense representation, would need {} amplitudes (space wasted!)",
        state.dimension()
    );
    println!("With sparse: only storing {} non-zero values\n", state.num_amplitudes());
}

/// Example 5: Measurement and collapse
fn example_5_measurement_and_collapse() {
    println!("### Example 5: Measurement and State Collapse ###");

    let mut state = SparseState::new(2).unwrap();

    // Create equal superposition: (|0⟩ + |1⟩ + |2⟩ + |3⟩)/2
    state.set_amplitude(0, Complex64::new(0.5, 0.0));
    state.set_amplitude(1, Complex64::new(0.5, 0.0));
    state.set_amplitude(2, Complex64::new(0.5, 0.0));
    state.set_amplitude(3, Complex64::new(0.5, 0.0));

    println!("Created superposition state: (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2");
    println!("Non-zero amplitudes before measurement: {}\n", state.num_amplitudes());

    // Measure qubit 0
    let (prob_0, prob_1) = state.measure_probability(0).unwrap();
    println!("Measurement probabilities for qubit 0:");
    println!("  P(0) = {:.4}", prob_0);
    println!("  P(1) = {:.4}\n", prob_1);

    // Collapse to outcome 0
    let mut state_collapsed = state.clone();
    let collapse_prob = state_collapsed.measure_and_collapse(0, 0).unwrap();

    println!("Collapsed to qubit 0 = 0 with probability {:.4}", collapse_prob);
    println!("Non-zero amplitudes after collapse: {}", state_collapsed.num_amplitudes());
    println!("Remaining states: |00⟩, |01⟩ (equal superposition)\n");
}

/// Example 6: Memory efficiency comparison
fn example_6_memory_efficiency() {
    println!("### Example 6: Memory Efficiency Comparison ###");

    let num_qubits = 20;
    let dimension = 1u64 << num_qubits;

    let sparse = SparseState::new(num_qubits).unwrap();

    let sparse_memory = sparse.num_amplitudes() * std::mem::size_of::<Complex64>();
    let dense_memory = dimension as usize * std::mem::size_of::<Complex64>();

    println!("For a {}-qubit system:", num_qubits);
    println!("  Dense representation: 2^{} = {} amplitudes", num_qubits, dimension);
    println!("  Memory required (dense): {} MB", (dense_memory as f64) / (1024.0 * 1024.0));
    println!("\n  Sparse representation (initial state |0...0⟩):");
    println!("    Non-zero amplitudes: {}", sparse.num_amplitudes());
    println!("    Memory required (sparse): {} bytes", sparse_memory);
    println!(
        "\n  Memory savings: {:.1}x smaller!",
        (dense_memory as f64) / (sparse_memory as f64)
    );
    println!("\nThis is why sparse states are crucial for efficient simulation!");
    println!("As gates are applied, density increases gradually.\n");
}
