//! Quantum Teleportation using Mid-Circuit Measurements
//!
//! This example demonstrates quantum teleportation - transferring a quantum state
//! from one qubit to another using entanglement and classical communication,
//! without physically moving the quantum information.
//!
//! The protocol:
//! 1. Alice has a qubit in unknown state |ψ⟩ she wants to send to Bob
//! 2. Alice and Bob share an entangled Bell pair (|00⟩ + |11⟩)/√2
//! 3. Alice performs Bell basis measurement on her two qubits (mid-circuit!)
//! 4. Based on measurement outcomes, Bob applies corrections to his qubit
//! 5. Bob's qubit is now in state |ψ⟩ (teleportation complete!)
//!
//! This demonstrates mid-circuit measurement because Alice's measurements
//! happen while Bob's qubit is still in superposition.

use num_complex::Complex64;
use simq_state::{DenseState, MidCircuitMeasurement};

fn main() {
    println!("=== Quantum Teleportation Protocol ===\n");

    // Example 1: Teleport |0⟩ state
    println!("Example 1: Teleporting |0⟩ state");
    println!("----------------------------------");
    teleport_state("0", (0.0, 0.0));
    println!();

    // Example 2: Teleport |1⟩ state
    println!("Example 2: Teleporting |1⟩ state");
    println!("----------------------------------");
    teleport_state("1", (std::f64::consts::PI, 0.0));
    println!();

    // Example 3: Teleport |+⟩ state
    println!("Example 3: Teleporting |+⟩ = (|0⟩ + |1⟩)/√2");
    println!("---------------------------------------------");
    teleport_state("+", (std::f64::consts::PI / 2.0, 0.0));
    println!();

    // Example 4: Teleport |−⟩ state
    println!("Example 4: Teleporting |−⟩ = (|0⟩ - |1⟩)/√2");
    println!("---------------------------------------------");
    teleport_state("−", (std::f64::consts::PI / 2.0, std::f64::consts::PI));
    println!();

    // Example 5: Teleport arbitrary state
    println!("Example 5: Teleporting arbitrary state");
    println!("---------------------------------------");
    let theta = std::f64::consts::PI / 3.0; // 60 degrees
    let phi = std::f64::consts::PI / 4.0; // 45 degrees
    println!("θ = π/3, φ = π/4");
    teleport_state("arbitrary", (theta, phi));
    println!();

    // Example 6: Multiple teleportations
    println!("Example 6: Teleporting 100 random states");
    println!("-----------------------------------------");
    test_teleportation_fidelity(100);
}

/// Teleport a single-qubit state parameterized by Bloch sphere angles
/// |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
fn teleport_state(name: &str, angles: (f64, f64)) {
    let (theta, phi) = angles;

    // Step 1: Prepare initial state with 3 qubits
    // Qubit layout (little-endian, q0 is LSB):
    //   q2 (MSB): Alice's state to teleport |ψ⟩
    //   q1: Alice's half of Bell pair
    //   q0 (LSB): Bob's half of Bell pair
    //
    // Initial: |ψ⟩ ⊗ (|00⟩ + |11⟩)/√2
    //        = (α|0⟩ + β|1⟩) ⊗ (|00⟩ + |11⟩)/√2
    //        = (α|000⟩ + α|011⟩ + β|100⟩ + β|111⟩)/√2

    let alpha = Complex64::new((theta / 2.0).cos(), 0.0);
    let beta = Complex64::new((theta / 2.0).sin() * phi.cos(), (theta / 2.0).sin() * phi.sin());

    let sqrt2_inv = 1.0 / 2_f64.sqrt();

    let mut amplitudes = vec![Complex64::new(0.0, 0.0); 8];
    amplitudes[0b000] = alpha * sqrt2_inv; // |ψ=0⟩|Bell=00⟩
    amplitudes[0b011] = alpha * sqrt2_inv; // |ψ=0⟩|Bell=11⟩
    amplitudes[0b100] = beta * sqrt2_inv; // |ψ=1⟩|Bell=00⟩
    amplitudes[0b111] = beta * sqrt2_inv; // |ψ=1⟩|Bell=11⟩

    let mut state = DenseState::from_amplitudes(3, &amplitudes).unwrap();

    println!("Initial state |ψ⟩ coefficients:");
    println!("  α (for |0⟩) = {:.4} + {:.4}i", alpha.re, alpha.im);
    println!("  β (for |1⟩) = {:.4} + {:.4}i", beta.re, beta.im);
    println!("  |α|² = {:.4}, |β|² = {:.4}", alpha.norm_sqr(), beta.norm_sqr());

    // Step 2: Alice applies CNOT gate (q2 control, q1 target)
    // This would normally be done with a gate application function
    // For now, we'll manually construct the post-CNOT state
    // After CNOT: (α|000⟩ + α|011⟩ + β|110⟩ + β|101⟩)/√2
    amplitudes[0b000] = alpha * sqrt2_inv;
    amplitudes[0b011] = alpha * sqrt2_inv;
    amplitudes[0b101] = beta * sqrt2_inv;
    amplitudes[0b110] = beta * sqrt2_inv;
    amplitudes[0b100] = Complex64::new(0.0, 0.0);
    amplitudes[0b111] = Complex64::new(0.0, 0.0);

    state = DenseState::from_amplitudes(3, &amplitudes).unwrap();

    // Step 3: Alice applies Hadamard to q2
    // H|0⟩ = (|0⟩ + |1⟩)/√2, H|1⟩ = (|0⟩ - |1⟩)/√2
    // After H on q2:
    //   = (α/2)|000⟩ + (α/2)|100⟩ + (α/2)|011⟩ + (α/2)|111⟩ +
    //     (β/2)|010⟩ - (β/2)|110⟩ + (β/2)|001⟩ - (β/2)|101⟩
    let half = 0.5;
    amplitudes[0b000] = alpha * half;
    amplitudes[0b001] = beta * half;
    amplitudes[0b010] = beta * half;
    amplitudes[0b011] = alpha * half;
    amplitudes[0b100] = alpha * half;
    amplitudes[0b101] = -beta * half;
    amplitudes[0b110] = -beta * half;
    amplitudes[0b111] = alpha * half;

    state = DenseState::from_amplitudes(3, &amplitudes).unwrap();

    // Step 4: Alice measures her two qubits (q2 and q1) - MID-CIRCUIT MEASUREMENT!
    // This is the key feature we're demonstrating
    let measurement = MidCircuitMeasurement::new(vec![2, 1]);

    // Use deterministic RNG for reproducibility in this example
    let mut rng_state = 0.0;
    let mut rng = || {
        // Simple deterministic sequence
        rng_state = (rng_state + 0.7) % 1.0;
        rng_state
    };

    let outcomes = measurement.measure(&mut state, &mut rng).unwrap();

    let m2 = outcomes[0].1; // Measurement of q2
    let m1 = outcomes[1].1; // Measurement of q1

    println!("\nAlice's measurement outcomes:");
    println!("  Qubit 2 (ψ): {} ", m2);
    println!("  Qubit 1 (Bell): {}", m1);
    println!("  Classical bits sent to Bob: {}{}", m2, m1);

    // Step 5: Bob applies corrections based on Alice's classical bits
    // If m1 = 1: apply X gate to q0
    // If m2 = 1: apply Z gate to q0
    // After corrections, Bob's qubit (q0) will be in state |ψ⟩

    println!("\nBob's corrections:");
    if m1 == 1 {
        println!("  Apply X gate (m1 = 1)");
        // X gate swaps amplitudes between |q0=0⟩ and |q0=1⟩
        apply_x_gate(&mut state, 0);
    }
    if m2 == 1 {
        println!("  Apply Z gate (m2 = 1)");
        // Z gate applies phase flip to |q0=1⟩
        apply_z_gate(&mut state, 0);
    }
    if m1 == 0 && m2 == 0 {
        println!("  No corrections needed (m1=m2=0)");
    }

    // Step 6: Verify teleportation by checking Bob's qubit state
    // After measurement and corrections, qubits q2 and q1 are in definite states
    // Bob's qubit (q0) should be in state |ψ⟩ = α|0⟩ + β|1⟩

    let amps = state.amplitudes();

    // Find Bob's qubit amplitudes (q0)
    // Bob's state is determined by the amplitudes at indices where q2=m2, q1=m1
    let base_idx = (m2 as usize) << 2 | (m1 as usize) << 1;
    let bob_amp_0 = amps[base_idx]; // |q0=0⟩
    let bob_amp_1 = amps[base_idx | 1]; // |q0=1⟩

    println!("\nBob's final qubit state:");
    println!("  Coefficient for |0⟩: {:.4} + {:.4}i", bob_amp_0.re, bob_amp_0.im);
    println!("  Coefficient for |1⟩: {:.4} + {:.4}i", bob_amp_1.re, bob_amp_1.im);

    // Calculate fidelity: F = |⟨ψ_original|ψ_Bob⟩|²
    let fidelity = (alpha.conj() * bob_amp_0 + beta.conj() * bob_amp_1).norm_sqr();
    println!("\nTeleportation fidelity: {:.6}", fidelity);

    if fidelity > 0.99 {
        println!("✓ Teleportation successful!");
    } else {
        println!("✗ Teleportation failed (fidelity too low)");
    }
}

/// Apply X gate to a specific qubit (flips |0⟩ ↔ |1⟩)
fn apply_x_gate(state: &mut DenseState, qubit: usize) {
    let mask = 1 << qubit;
    let amplitudes = state.amplitudes_mut();
    let dim = amplitudes.len();

    for idx in 0..dim {
        if idx & mask == 0 {
            // Swap amplitude at idx with idx | mask
            let partner_idx = idx | mask;
            let temp = amplitudes[idx];
            amplitudes[idx] = amplitudes[partner_idx];
            amplitudes[partner_idx] = temp;
        }
    }
}

/// Apply Z gate to a specific qubit (phase flip: |1⟩ → -|1⟩)
fn apply_z_gate(state: &mut DenseState, qubit: usize) {
    let mask = 1 << qubit;
    let amplitudes = state.amplitudes_mut();

    for idx in 0..amplitudes.len() {
        if idx & mask != 0 {
            amplitudes[idx] = -amplitudes[idx];
        }
    }
}

/// Test teleportation fidelity over many random states
fn test_teleportation_fidelity(num_trials: usize) {
    let mut total_fidelity = 0.0;
    let mut perfect_count = 0;

    // Use a simple pseudo-random number generator
    let mut seed = 42u64;
    let mut rand = || {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        ((seed / 65536) % 32768) as f64 / 32768.0
    };

    for trial in 0..num_trials {
        // Generate random Bloch sphere angles
        let theta = rand() * std::f64::consts::PI;
        let phi = rand() * 2.0 * std::f64::consts::PI;

        let alpha = Complex64::new((theta / 2.0).cos(), 0.0);
        let beta = Complex64::new(
            (theta / 2.0).sin() * phi.cos(),
            (theta / 2.0).sin() * phi.sin(),
        );

        // Prepare and teleport
        let sqrt2_inv = 1.0 / 2_f64.sqrt();
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); 8];
        amplitudes[0b000] = alpha * sqrt2_inv;
        amplitudes[0b011] = alpha * sqrt2_inv;
        amplitudes[0b100] = beta * sqrt2_inv;
        amplitudes[0b111] = beta * sqrt2_inv;

        let mut state = DenseState::from_amplitudes(3, &amplitudes).unwrap();

        // Apply CNOT (q2 control, q1 target)
        amplitudes[0b000] = alpha * sqrt2_inv;
        amplitudes[0b011] = alpha * sqrt2_inv;
        amplitudes[0b101] = beta * sqrt2_inv;
        amplitudes[0b110] = beta * sqrt2_inv;
        amplitudes[0b100] = Complex64::new(0.0, 0.0);
        amplitudes[0b111] = Complex64::new(0.0, 0.0);
        state = DenseState::from_amplitudes(3, &amplitudes).unwrap();

        // Apply Hadamard to q2
        let half = 0.5;
        amplitudes[0b000] = alpha * half;
        amplitudes[0b001] = beta * half;
        amplitudes[0b010] = beta * half;
        amplitudes[0b011] = alpha * half;
        amplitudes[0b100] = alpha * half;
        amplitudes[0b101] = -beta * half;
        amplitudes[0b110] = -beta * half;
        amplitudes[0b111] = alpha * half;
        state = DenseState::from_amplitudes(3, &amplitudes).unwrap();

        // Measure Alice's qubits
        let measurement = MidCircuitMeasurement::new(vec![2, 1]);
        let mut rng = || rand();
        let outcomes = measurement.measure(&mut state, &mut rng).unwrap();

        let m2 = outcomes[0].1;
        let m1 = outcomes[1].1;

        // Apply corrections
        if m1 == 1 {
            apply_x_gate(&mut state, 0);
        }
        if m2 == 1 {
            apply_z_gate(&mut state, 0);
        }

        // Calculate fidelity
        let amps = state.amplitudes();
        let base_idx = (m2 as usize) << 2 | (m1 as usize) << 1;
        let bob_amp_0 = amps[base_idx];
        let bob_amp_1 = amps[base_idx | 1];

        let fidelity = (alpha.conj() * bob_amp_0 + beta.conj() * bob_amp_1).norm_sqr();
        total_fidelity += fidelity;

        if fidelity > 0.99 {
            perfect_count += 1;
        }

        if trial < 5 {
            println!(
                "Trial {}: θ={:.3}, φ={:.3}, m2m1={}{}, F={:.6}",
                trial + 1,
                theta,
                phi,
                m2,
                m1,
                fidelity
            );
        }
    }

    let avg_fidelity = total_fidelity / num_trials as f64;
    println!("\n...\n");
    println!("Statistics over {} trials:", num_trials);
    println!("  Average fidelity: {:.6}", avg_fidelity);
    println!("  Perfect teleportations (F > 0.99): {}/{}", perfect_count, num_trials);
    println!("  Success rate: {:.1}%", (perfect_count as f64 / num_trials as f64) * 100.0);

    if avg_fidelity > 0.99 {
        println!("\n✓ Quantum teleportation protocol working correctly!");
    } else {
        println!("\n✗ Warning: Average fidelity below expected threshold");
    }
}
