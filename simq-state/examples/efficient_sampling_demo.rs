//! Demonstration of efficient shot sampling capabilities
//!
//! This example showcases the performance of Walker's alias method
//! for O(1) per-shot sampling from quantum states.

use num_complex::Complex64;
use simq_state::{ComputationalBasis, DenseState};
use std::time::Instant;

// Simple RNG for reproducible results
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.state / 65536) % 32768) as f64 / 32768.0
    }
}

fn main() {
    println!("=== Efficient Shot Sampling Demonstration ===\n");

    demo_throughput();
    demo_scaling_with_shots();
    demo_scaling_with_qubits();
    demo_statistical_accuracy();
    demo_vs_individual_measurements();
}

fn demo_throughput() {
    println!("Demo 1: Sampling Throughput");
    println!("----------------------------");

    let num_qubits = 10;
    let shots = 100000;

    // Create uniform superposition
    let dimension = 1 << num_qubits;
    let amplitude = Complex64::new(1.0 / (dimension as f64).sqrt(), 0.0);
    let state = DenseState::from_amplitudes(num_qubits, &vec![amplitude; dimension]).unwrap();

    let measurement = ComputationalBasis::new().with_collapse(false);
    let mut rng = SimpleRng::new(42);

    println!("State: {} qubits (uniform superposition)", num_qubits);
    println!("Shots: {}", shots);

    let start = Instant::now();
    let result = measurement.sample(&state, shots, &mut || rng.next()).unwrap();
    let elapsed = start.elapsed();

    let throughput = shots as f64 / elapsed.as_secs_f64();

    println!("\nResults:");
    println!("  Time: {:?}", elapsed);
    println!("  Throughput: {:.2} M shots/sec", throughput / 1_000_000.0);
    println!("  Unique outcomes: {} / {}", result.counts.len(), dimension);
    println!();
}

fn demo_scaling_with_shots() {
    println!("Demo 2: Scaling with Number of Shots");
    println!("-------------------------------------");

    let num_qubits = 12;
    let dimension = 1 << num_qubits;
    let amplitude = Complex64::new(1.0 / (dimension as f64).sqrt(), 0.0);
    let state = DenseState::from_amplitudes(num_qubits, &vec![amplitude; dimension]).unwrap();

    let measurement = ComputationalBasis::new().with_collapse(false);

    println!("State: {} qubits\n", num_qubits);
    println!("{:>10} {:>12} {:>18}", "Shots", "Time", "Throughput");
    println!("{:-<42}", "");

    for &shots in &[10, 100, 1000, 10000, 100000] {
        let mut rng = SimpleRng::new(42);

        let start = Instant::now();
        let _ = measurement.sample(&state, shots, &mut || rng.next()).unwrap();
        let elapsed = start.elapsed();

        let throughput = shots as f64 / elapsed.as_secs_f64() / 1_000_000.0;

        println!(
            "{:>10} {:>10.3} ms {:>12.2} M/s",
            shots,
            elapsed.as_secs_f64() * 1000.0,
            throughput
        );
    }

    println!("\nNote: Throughput increases with shots (amortized setup cost)\n");
}

fn demo_scaling_with_qubits() {
    println!("Demo 3: Scaling with Number of Qubits");
    println!("--------------------------------------");

    let shots = 10000;

    println!("Shots: {}\n", shots);
    println!("{:>8} {:>12} {:>18} {:>12}", "Qubits", "Time", "Throughput", "Setup Cost");
    println!("{:-<52}", "");

    for &num_qubits in &[5, 8, 10, 12, 15] {
        let dimension = 1 << num_qubits;
        let amplitude = Complex64::new(1.0 / (dimension as f64).sqrt(), 0.0);
        let state = DenseState::from_amplitudes(num_qubits, &vec![amplitude; dimension]).unwrap();

        let measurement = ComputationalBasis::new().with_collapse(false);
        let mut rng = SimpleRng::new(42);

        let start = Instant::now();
        let _ = measurement.sample(&state, shots, &mut || rng.next()).unwrap();
        let elapsed = start.elapsed();

        let throughput = shots as f64 / elapsed.as_secs_f64() / 1_000_000.0;
        let setup_fraction = dimension as f64 * 16.0 / (elapsed.as_micros() as f64);

        println!(
            "{:>8} {:>10.3} ms {:>12.2} M/s {:>10.1}%",
            num_qubits,
            elapsed.as_secs_f64() * 1000.0,
            throughput,
            setup_fraction * 100.0
        );
    }

    println!("\nNote: Setup cost (O(2^n)) dominates for large states\n");
}

fn demo_statistical_accuracy() {
    println!("Demo 4: Statistical Accuracy");
    println!("-----------------------------");

    // Create a 3-qubit state with known probabilities
    let amplitudes = vec![
        Complex64::new(0.5, 0.0),   // |000⟩: P = 0.25
        Complex64::new(0.5, 0.0),   // |001⟩: P = 0.25
        Complex64::new(0.5, 0.0),   // |010⟩: P = 0.25
        Complex64::new(0.5, 0.0),   // |011⟩: P = 0.25
        Complex64::new(0.0, 0.0),   // |100⟩: P = 0.00
        Complex64::new(0.0, 0.0),   // |101⟩: P = 0.00
        Complex64::new(0.0, 0.0),   // |110⟩: P = 0.00
        Complex64::new(0.0, 0.0),   // |111⟩: P = 0.00
    ];

    let state = DenseState::from_amplitudes(3, &amplitudes).unwrap();
    let measurement = ComputationalBasis::new().with_collapse(false);

    println!("State: Equal superposition of |000⟩, |001⟩, |010⟩, |011⟩");
    println!("Expected: Each outcome has 25% probability\n");

    let mut rng = SimpleRng::new(42);
    let result = measurement.sample(&state, 100000, &mut || rng.next()).unwrap();

    println!("Results from 100,000 shots:");
    println!("{:>8} {:>8} {:>12} {:>10}", "Outcome", "Count", "Frequency", "Error");
    println!("{:-<40}", "");

    for outcome in 0..8 {
        let count = result.get_count(outcome);
        let freq = result.get_probability(outcome);
        let expected = if outcome < 4 { 0.25 } else { 0.0 };
        let error = (freq - expected).abs();

        println!(
            "|{:03b}⟩   {:>8} {:>10.2}%   {:>8.4}%",
            outcome,
            count,
            freq * 100.0,
            error * 100.0
        );
    }

    println!("\nNote: Errors decrease as O(1/√N) with shot count\n");
}

fn demo_vs_individual_measurements() {
    println!("Demo 5: Batch vs Individual Measurements");
    println!("-----------------------------------------");

    let num_qubits = 10;
    let shots = 1000;

    let dimension = 1 << num_qubits;
    let amplitude = Complex64::new(1.0 / (dimension as f64).sqrt(), 0.0);
    let state = DenseState::from_amplitudes(num_qubits, &vec![amplitude; dimension]).unwrap();

    println!("State: {} qubits", num_qubits);
    println!("Shots: {}\n", shots);

    // Method 1: Batch sampling
    let measurement = ComputationalBasis::new().with_collapse(false);
    let mut rng = SimpleRng::new(42);

    let start = Instant::now();
    let _ = measurement.sample(&state, shots, &mut || rng.next()).unwrap();
    let batch_time = start.elapsed();

    println!("Batch sampling (alias method):");
    println!("  Time: {:?}", batch_time);

    // Method 2: Individual measurements (simulated)
    let measurement2 = ComputationalBasis::new();
    let mut rng2 = SimpleRng::new(42);

    let start = Instant::now();
    for _ in 0..shots {
        let mut state_copy = state.clone_state().unwrap();
        let _ = measurement2.measure_once(&mut state_copy, &mut || rng2.next()).unwrap();
    }
    let individual_time = start.elapsed();

    println!("\nIndividual measurements (with state copy):");
    println!("  Time: {:?}", individual_time);

    let speedup = individual_time.as_secs_f64() / batch_time.as_secs_f64();
    println!("\nSpeedup: {:.1}x faster with batch sampling", speedup);
    println!();
}
