//! Demonstrates state normalization validation and tracking
//!
//! Run with: cargo run --example validation_demo

use num_complex::Complex64;
use simq_state::validation::{
    auto_normalize, check_finite, validate_normalization, validate_probabilities,
    validate_unitary_2x2, NormalizationTracker, ValidationPolicy, DEFAULT_NORM_TOLERANCE,
};
use simq_state::DenseState;

fn main() {
    println!("=== State Normalization Validation Demo ===\n");

    // Example 1: Basic normalization validation
    println!("1. Basic Normalization Validation:");

    let good_amplitudes = vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ];

    let result = validate_normalization(&good_amplitudes, DEFAULT_NORM_TOLERANCE);
    println!("   Good state: {}", result);
    println!("   Valid: {}", result.is_valid());
    println!("   Norm: {:.10}", result.norm);
    println!("   Error: {:.2e}", result.norm_error);
    println!();

    let bad_amplitudes = vec![Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)];

    let result = validate_normalization(&bad_amplitudes, DEFAULT_NORM_TOLERANCE);
    println!("   Bad state: {}", result);
    println!("   Valid: {}", result.is_valid());
    println!("   Norm: {:.10}", result.norm);
    println!("   Error: {:.2e}", result.norm_error);
    println!("   Severity: {}", result.severity());
    println!();

    // Example 2: Auto-normalization
    println!("2. Automatic Normalization:");

    let mut amplitudes = vec![Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)];

    println!(
        "   Before: norm = {:.6}",
        amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt()
    );

    let normalized = auto_normalize(&mut amplitudes, DEFAULT_NORM_TOLERANCE);
    println!("   Normalization applied: {}", normalized);

    println!(
        "   After: norm = {:.10}",
        amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt()
    );
    println!("   Amplitudes: {:?}", amplitudes);
    println!();

    // Example 3: Probability validation
    println!("3. Probability Validation:");

    let valid_probs = vec![0.25, 0.25, 0.25, 0.25];
    let result = validate_probabilities(&valid_probs, DEFAULT_NORM_TOLERANCE);
    println!("   Valid probabilities: {}", result.is_valid());
    println!("   Sum: {:.10}", result.total_probability);

    let invalid_probs = vec![0.3, 0.3, 0.3, 0.3];
    let result = validate_probabilities(&invalid_probs, DEFAULT_NORM_TOLERANCE);
    println!("   Invalid probabilities: {}", result.is_valid());
    println!("   Sum: {:.10} (should be 1.0)", result.total_probability);
    println!("   Error: {:.2e}", result.probability_error);
    println!();

    // Example 4: Unitary gate validation
    println!("4. Unitary Gate Validation:");

    // Hadamard gate
    let h = 1.0 / 2.0_f64.sqrt();
    let hadamard = [
        [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
        [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
    ];
    println!("   Hadamard gate is unitary: {}", validate_unitary_2x2(&hadamard, 1e-10));

    // Pauli-X gate
    let pauli_x = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    println!("   Pauli-X gate is unitary: {}", validate_unitary_2x2(&pauli_x, 1e-10));

    // Non-unitary matrix
    let non_unitary = [
        [Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    println!("   Non-unitary matrix: {}", validate_unitary_2x2(&non_unitary, 1e-10));
    println!();

    // Example 5: Checking for NaN/infinity
    println!("5. Finiteness Checks:");

    let finite = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
    println!("   Finite amplitudes: {}", check_finite(&finite));

    let infinite = vec![Complex64::new(f64::INFINITY, 0.0)];
    println!("   Infinite amplitudes: {}", check_finite(&infinite));

    let nan = vec![Complex64::new(f64::NAN, 0.0)];
    println!("   NaN amplitudes: {}", check_finite(&nan));
    println!();

    // Example 6: Normalization tracking over circuit
    println!("6. Normalization Tracking:");

    let mut tracker = NormalizationTracker::new(100);
    let mut state = DenseState::new(3).unwrap();

    // Apply sequence of gates
    let gates_to_apply = 20;
    for i in 0..gates_to_apply {
        // Apply Hadamard to alternating qubits
        state.apply_single_qubit_gate(&hadamard, i % 3).unwrap();

        // Track normalization
        let norm = state.norm();
        tracker.record(norm);

        if i % 5 == 4 {
            let stats = tracker.stats();
            println!("   After {} gates: {}", i + 1, stats);
        }
    }

    let final_stats = tracker.stats();
    println!("\n   Final statistics:");
    println!("   - Measurements: {}", final_stats.measurements);
    println!("   - Average drift: {:.2e}", final_stats.average_drift);
    println!("   - Max drift: {:.2e}", final_stats.max_drift);
    println!("   - Cumulative drift: {:.2e}", final_stats.cumulative_drift);
    println!("   - Current norm: {:.10}", final_stats.current_norm);
    println!("   - Stable (< 1e-6): {}", tracker.is_stable(1e-6));
    println!();

    // Example 7: Validation policies
    println!("7. Validation Policies:");

    use simq_state::validation::validate_state;

    let state_amps = DenseState::new(2).unwrap().amplitudes().to_vec();

    // No validation
    println!("   ValidationPolicy::None - always passes");

    // Critical validation
    match validate_state(&state_amps, ValidationPolicy::Critical) {
        Ok(result) => println!("   ValidationPolicy::Critical - passed: {}", result.is_valid()),
        Err(e) => println!("   ValidationPolicy::Critical - error: {}", e),
    }

    // Strict validation
    match validate_state(&state_amps, ValidationPolicy::Strict) {
        Ok(result) => println!("   ValidationPolicy::Strict - passed: {}", result.is_valid()),
        Err(e) => println!("   ValidationPolicy::Strict - error: {}", e),
    }

    // Strict validation on bad state
    let bad_state = vec![Complex64::new(10.0, 0.0)];
    match validate_state(&bad_state, ValidationPolicy::Strict) {
        Ok(_) => println!("   Bad state - passed (unexpected!)"),
        Err(e) => println!("   Bad state - correctly rejected: {}", e),
    }
    println!();

    // Example 8: Tracking numerical drift in long circuits
    println!("8. Numerical Drift Detection:");

    let mut long_tracker = NormalizationTracker::new(1000);
    let mut long_state = DenseState::new(5).unwrap();

    // Simulate a long circuit with many gates
    for gate_num in 0..100 {
        long_state
            .apply_single_qubit_gate(&hadamard, gate_num % 5)
            .unwrap();
        long_tracker.record(long_state.norm());
    }

    let drift_stats = long_tracker.stats();
    println!("   After 100 gates:");
    println!("   - Average drift: {:.2e}", drift_stats.average_drift);
    println!("   - Max drift: {:.2e}", drift_stats.max_drift);

    if drift_stats.max_drift > 1e-6 {
        println!("   ⚠ Warning: Significant drift detected!");
        println!("   Consider renormalizing the state.");
    } else {
        println!("   ✓ Numerical stability maintained!");
    }
    println!();

    // Example 9: Best practices
    println!("9. Best Practices:");
    println!("   ✓ Use ValidationPolicy::Critical for production");
    println!("   ✓ Use ValidationPolicy::Strict for debugging");
    println!("   ✓ Track normalization in long circuits (>50 gates)");
    println!("   ✓ Auto-normalize if drift exceeds 1e-6");
    println!("   ✓ Validate unitary gates during development");
    println!("   ✓ Check for NaN/infinity after complex operations");
    println!("   ✓ Monitor cumulative drift in variational algorithms");
    println!();

    println!("=== Demo Complete ===");
}
