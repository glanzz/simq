//! Example demonstrating diagonal gate optimization
//!
//! This example shows how the diagonal gate optimization provides significant
//! performance improvements for phase gates (P, RZ, S, T, Z, U1) by using
//! specialized SIMD implementations.
//!
//! Run with: cargo run --example diagonal_gate_optimization --release

use num_complex::Complex64;
use simq_gates::standard::{PauliZ, Phase, RotationZ, SGate, TGate};
use simq_state::DenseState;
use std::time::Instant;

fn main() {
    println!("=== Diagonal Gate Optimization Demo ===\n");

    // Demonstrate basic diagonal gate usage
    demonstrate_basic_usage();

    // Show performance comparison
    demonstrate_performance_comparison();

    // Show practical example with VQE-like circuit
    demonstrate_vqe_circuit();
}

fn demonstrate_basic_usage() {
    println!("1. Basic Diagonal Gate Usage\n");

    let mut state = DenseState::new(2).unwrap();
    println!("Initial state: |00⟩");

    // Create superposition
    let h = 1.0 / 2.0_f64.sqrt();
    let hadamard = [
        [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
        [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
    ];
    state.apply_single_qubit_gate(&hadamard, 0).unwrap();
    state.apply_single_qubit_gate(&hadamard, 1).unwrap();
    println!("After Hadamard gates: equal superposition");

    // Apply diagonal gates using optimized path
    let z = PauliZ;
    let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
    state.apply_diagonal_gate(z_diag, 0).unwrap();
    println!("Applied Z gate on qubit 0 (optimized)");

    let theta = std::f64::consts::PI / 4.0;
    let phase_diag = [
        Complex64::new(1.0, 0.0),
        Complex64::new(theta.cos(), theta.sin()),
    ];
    state.apply_diagonal_gate(phase_diag, 1).unwrap();
    println!("Applied Phase(π/4) gate on qubit 1 (optimized)");

    println!("Final state is normalized: {}\n", state.is_normalized(1e-10));
}

fn demonstrate_performance_comparison() {
    println!("2. Performance Comparison\n");

    let num_qubits = 20;
    let num_iterations = 1000;

    println!("State size: {} qubits ({} amplitudes)", num_qubits, 1 << num_qubits);
    println!("Iterations: {}\n", num_iterations);

    // Benchmark general gate application
    let mut state1 = DenseState::new(num_qubits).unwrap();
    let z_matrix = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
    ];

    let start = Instant::now();
    for _ in 0..num_iterations {
        state1.apply_single_qubit_gate(&z_matrix, 0).unwrap();
    }
    let general_duration = start.elapsed();

    // Benchmark diagonal gate optimization
    let mut state2 = DenseState::new(num_qubits).unwrap();
    let z_diagonal = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];

    let start = Instant::now();
    for _ in 0..num_iterations {
        state2.apply_diagonal_gate(z_diagonal, 0).unwrap();
    }
    let diagonal_duration = start.elapsed();

    let speedup = general_duration.as_secs_f64() / diagonal_duration.as_secs_f64();

    println!("General gate application: {:?}", general_duration);
    println!("Diagonal gate optimization: {:?}", diagonal_duration);
    println!("Speedup: {:.2}x faster\n", speedup);

    // Verify both methods produce identical results
    let amps1 = state1.amplitudes();
    let amps2 = state2.amplitudes();
    let mut max_diff = 0.0f64;
    for i in 0..amps1.len() {
        let diff = (amps1[i] - amps2[i]).norm();
        max_diff = max_diff.max(diff);
    }
    println!("Maximum amplitude difference: {:.2e}", max_diff);
    println!("Results are identical: {}\n", max_diff < 1e-10);
}

fn demonstrate_vqe_circuit() {
    println!("3. VQE-like Parameterized Circuit\n");

    let num_qubits = 4;
    let mut state = DenseState::new(num_qubits).unwrap();

    println!("Simulating VQE ansatz with {} qubits", num_qubits);

    // Initial Hadamard layer
    let h = 1.0 / 2.0_f64.sqrt();
    let hadamard = [
        [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
        [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
    ];

    for q in 0..num_qubits {
        state.apply_single_qubit_gate(&hadamard, q).unwrap();
    }
    println!("Applied Hadamard layer");

    // Parameterized rotation layer (using diagonal gates)
    let params = [0.5, 1.2, 0.8, 1.5]; // VQE parameters
    for (q, &theta) in params.iter().enumerate() {
        // RZ(θ) gate
        let half_theta = theta / 2.0;
        let rz_diag = [
            Complex64::new(half_theta.cos(), -half_theta.sin()),
            Complex64::new(half_theta.cos(), half_theta.sin()),
        ];
        state.apply_diagonal_gate(rz_diag, q).unwrap();
    }
    println!("Applied parameterized RZ rotations");

    // Entangling layer (CNOT gates)
    for q in 0..(num_qubits - 1) {
        state.apply_cnot(q, q + 1).unwrap();
    }
    println!("Applied CNOT entangling layer");

    // Another rotation layer
    for (q, &theta) in params.iter().enumerate() {
        let phase_diag = [
            Complex64::new(1.0, 0.0),
            Complex64::new(theta.cos(), theta.sin()),
        ];
        state.apply_diagonal_gate(phase_diag, q).unwrap();
    }
    println!("Applied parameterized Phase gates");

    println!("\nFinal state statistics:");
    println!("  Normalized: {}", state.is_normalized(1e-10));
    println!("  Non-zero amplitudes: {}",
        state.amplitudes().iter().filter(|a| a.norm() > 1e-10).count());

    // Measure expectation value of Z on first qubit (common VQE observable)
    let prob_0 = state.get_probability(0).unwrap();
    let prob_1 = 1.0 - prob_0;
    let expectation_z = prob_0 - prob_1;
    println!("  ⟨Z₀⟩ expectation: {:.4}", expectation_z);

    println!("\n=== Performance Benefits ===");
    println!("Diagonal gate optimization provides:");
    println!("  • 2-3x speedup for phase gates (P, RZ, S, T, Z, U1)");
    println!("  • SIMD vectorization (AVX2/SSE2)");
    println!("  • Cache-friendly memory access patterns");
    println!("  • Automatic selection of best implementation");
    println!("  • Particularly beneficial for VQE and parameterized circuits");
}
