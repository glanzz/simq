//! Bloch Sphere Visualization Demo
//!
//! Demonstrates visualization of single-qubit states on the Bloch sphere.
//!
//! Run with: cargo run --example bloch_sphere -p simq-core

use num_complex::Complex64;
use simq_core::BlochVector;

fn main() {
    println!("=== Bloch Sphere Visualization Demo ===\n");

    // Demo 1: Computational basis states
    println!("1. Computational Basis States:");
    println!("\n|0⟩ state (north pole):");
    let zero_state = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let bloch = BlochVector::from_state(&zero_state);
    println!("{}", bloch.describe());
    println!("{}", bloch.render_ascii());

    println!("\n|1⟩ state (south pole):");
    let one_state = [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
    let bloch = BlochVector::from_state(&one_state);
    println!("{}", bloch.describe());

    // Demo 2: Superposition states
    println!("\n2. Superposition States:");
    println!("\n|+⟩ = (|0⟩ + |1⟩)/√2 (east pole, +x axis):");
    let plus_state = [
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ];
    let bloch = BlochVector::from_state(&plus_state);
    println!("{}", bloch.describe());
    println!("{}", bloch.render_ascii());

    println!("\n|−⟩ = (|0⟩ − |1⟩)/√2 (west pole, -x axis):");
    let minus_state = [
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
    ];
    let bloch = BlochVector::from_state(&minus_state);
    println!("{}", bloch.describe());

    // Demo 3: Complex phase states
    println!("\n3. Complex Phase States:");
    println!("\n|+i⟩ = (|0⟩ + i|1⟩)/√2 (front pole, +y axis):");
    let plus_i_state = [
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(0.0, 1.0 / 2.0_f64.sqrt()),
    ];
    let bloch = BlochVector::from_state(&plus_i_state);
    println!("{}", bloch.describe());
    println!("{}", bloch.render_ascii());

    println!("\n|−i⟩ = (|0⟩ − i|1⟩)/√2 (back pole, -y axis):");
    let minus_i_state = [
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(0.0, -1.0 / 2.0_f64.sqrt()),
    ];
    let bloch = BlochVector::from_state(&minus_i_state);
    println!("{}", bloch.describe());

    // Demo 4: Arbitrary state
    println!("\n4. Arbitrary Pure State:");
    let theta = std::f64::consts::PI / 3.0; // 60 degrees from z-axis
    let phi = std::f64::consts::PI / 4.0; // 45 degrees in xy-plane
    let arbitrary_state = [
        Complex64::new((theta / 2.0).cos(), 0.0),
        Complex64::new((theta / 2.0).sin() * phi.cos(), (theta / 2.0).sin() * phi.sin()),
    ];
    let bloch = BlochVector::from_state(&arbitrary_state);
    println!("{}", bloch.describe());
    println!("{}", bloch.render_ascii());

    // Demo 5: State evolution visualization
    println!("\n5. State Evolution (rotation around Z-axis):");
    let steps = 4;
    for i in 0..=steps {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (steps as f64);
        let state = [
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(
                (1.0 / 2.0_f64.sqrt()) * angle.cos(),
                (1.0 / 2.0_f64.sqrt()) * angle.sin(),
            ),
        ];
        let bloch = BlochVector::from_state(&state);
        println!("\nStep {} (φ = {:.2}π):", i, angle / std::f64::consts::PI);
        println!("  Position: ({:.3}, {:.3}, {:.3})", bloch.x, bloch.y, bloch.z);
    }
}
