//! Example demonstrating computational basis measurement
//!
//! This example shows how to use the ComputationalBasis measurement
//! system for both single-shot and multi-shot sampling.

use num_complex::Complex64;
use simq_state::{ComputationalBasis, DenseState};

// Simple random number generator for examples
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f64 {
        // Linear congruential generator
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.state / 65536) % 32768) as f64 / 32768.0
    }
}

fn main() {
    println!("=== Computational Basis Measurement Examples ===\n");

    // Example 1: Measuring a computational basis state
    example_basis_state();

    // Example 2: Measuring a superposition state
    example_superposition();

    // Example 3: Multi-shot sampling
    example_sampling();

    // Example 4: Bell state measurement
    example_bell_state();

    // Example 5: Performance comparison
    example_performance();
}

fn example_basis_state() {
    println!("Example 1: Measuring a computational basis state");
    println!("------------------------------------------------");

    let mut state = DenseState::new(2).unwrap();
    println!("Initial state: |00⟩");
    println!("Amplitudes: {:?}", state.amplitudes());

    let measurement = ComputationalBasis::new();
    let mut rng = SimpleRng::new(42);

    let result = measurement.measure_once(&mut state, &mut || rng.next()).unwrap();

    println!("Measurement outcome: {}", result.as_bitstring(2));
    println!("Probability: {:.4}", result.probability);
    println!();
}

fn example_superposition() {
    println!("Example 2: Measuring a superposition state");
    println!("------------------------------------------");

    // Create |+⟩ state: (|0⟩ + |1⟩) / √2
    let amplitudes = vec![
        Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
    ];
    let mut state = DenseState::from_amplitudes(1, &amplitudes).unwrap();

    println!("Initial state: |+⟩ = (|0⟩ + |1⟩) / √2");
    println!("Amplitudes: {:?}", state.amplitudes());

    let measurement = ComputationalBasis::new();
    let mut rng = SimpleRng::new(42);

    let result = measurement.measure_once(&mut state, &mut || rng.next()).unwrap();

    println!("Measurement outcome: {}", result.as_bitstring(1));
    println!("Probability: {:.4}", result.probability);
    println!("State after measurement: {:?}", state.amplitudes());
    println!();
}

fn example_sampling() {
    println!("Example 3: Multi-shot sampling");
    println!("-------------------------------");

    // Create a 2-qubit state with unequal superposition
    let amplitudes = vec![
        Complex64::new(0.6, 0.0),  // |00⟩
        Complex64::new(0.8, 0.0),  // |01⟩
        Complex64::new(0.0, 0.0),  // |10⟩
        Complex64::new(0.0, 0.0),  // |11⟩
    ];
    let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

    println!("Initial state: 0.6|00⟩ + 0.8|01⟩");
    println!("Expected probabilities: |00⟩ = 0.36, |01⟩ = 0.64");

    let measurement = ComputationalBasis::new().with_collapse(false);
    let mut rng = SimpleRng::new(42);

    let shots = 10000;
    let result = measurement.sample(&state, shots, &mut || rng.next()).unwrap();

    println!("\nSampling with {} shots:", shots);
    println!("Outcome counts:");

    let bitstring_counts = result.to_bitstring_counts(2);
    let mut sorted: Vec<_> = bitstring_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    for (bitstring, count) in sorted {
        let probability = *count as f64 / shots as f64;
        println!("  |{}⟩: {} ({:.4})", bitstring, count, probability);
    }
    println!();
}

fn example_bell_state() {
    println!("Example 4: Bell state measurement");
    println!("----------------------------------");

    // Create Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
    let amplitudes = vec![
        Complex64::new(1.0 / 2_f64.sqrt(), 0.0),  // |00⟩
        Complex64::new(0.0, 0.0),                   // |01⟩
        Complex64::new(0.0, 0.0),                   // |10⟩
        Complex64::new(1.0 / 2_f64.sqrt(), 0.0),  // |11⟩
    ];
    let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

    println!("Initial state: |Φ+⟩ = (|00⟩ + |11⟩) / √2");

    let measurement = ComputationalBasis::new().with_collapse(false);
    let mut rng = SimpleRng::new(42);

    let shots = 1000;
    let result = measurement.sample(&state, shots, &mut || rng.next()).unwrap();

    println!("\nSampling with {} shots:", shots);
    println!("Expected: 50% |00⟩, 50% |11⟩");
    println!("Actual outcomes:");

    let sorted = result.sorted_outcomes();
    for (outcome, count) in sorted {
        let bitstring = format!("{:02b}", outcome);
        let probability = count as f64 / shots as f64;
        println!("  |{}⟩: {} ({:.2}%)", bitstring, count, probability * 100.0);
    }
    println!();
}

fn example_performance() {
    println!("Example 5: Performance comparison");
    println!("----------------------------------");

    let num_qubits = 15;
    let shots = 10000;

    // Create a random state
    let mut rng = SimpleRng::new(42);
    let dimension = 1 << num_qubits;
    let amplitudes: Vec<Complex64> = (0..dimension)
        .map(|_| Complex64::new(rng.next() - 0.5, rng.next() - 0.5))
        .collect();

    let mut state = DenseState::from_amplitudes(num_qubits, &amplitudes).unwrap();
    state.normalize();

    println!("State: {} qubits ({} amplitudes)", num_qubits, dimension);
    println!("Shots: {}", shots);

    // Batch sampling (uses alias method)
    let measurement = ComputationalBasis::new().with_collapse(false);
    let mut rng = SimpleRng::new(123);

    let start = std::time::Instant::now();
    let result = measurement.sample(&state, shots, &mut || rng.next()).unwrap();
    let elapsed = start.elapsed();

    println!("\nBatch sampling:");
    println!("  Time: {:?}", elapsed);
    println!("  Throughput: {:.2} shots/ms", shots as f64 / elapsed.as_micros() as f64 * 1000.0);
    println!("  Unique outcomes: {}", result.counts.len());

    // Show top 5 outcomes
    let sorted = result.sorted_outcomes();
    println!("\nTop 5 outcomes:");
    for (outcome, count) in sorted.iter().take(5) {
        let probability = *count as f64 / shots as f64;
        println!("  State {}: {} ({:.2}%)", outcome, count, probability * 100.0);
    }
}
