//! Minimal VQE-style workflow using the fluent API
//!
//! Run with: cargo run -p ferriq --example vqe_fluent
//!
//! Optimizes a one-parameter ansatz RY(θ)|0⟩ against H = Z by gradient
//! descent using exact expectation values from `QuantumCircuit`. The ground
//! state of Z is |1⟩ with energy −1, reached at θ = π.

use ferriq::{PauliObservable, PauliString, QuantumCircuit};
use std::f64::consts::PI;

fn energy(theta: f64, hamiltonian: &PauliObservable) -> f64 {
    let mut qc = QuantumCircuit::new(1);
    qc.ry(theta, 0);
    qc.expectation_value(hamiltonian)
        .expect("simulation failed")
}

fn main() {
    let hamiltonian = PauliObservable::from_pauli_string(PauliString::from_str("Z").unwrap(), 1.0);

    // Analytic energy: ⟨Z⟩ = cos(θ); minimum −1 at θ = π
    let mut theta = 0.5;
    let learning_rate = 0.4;
    let eps = 1e-6;

    println!("{:>4}  {:>10}  {:>10}", "iter", "theta", "energy");
    for iter in 0..40 {
        let e = energy(theta, &hamiltonian);
        if iter % 5 == 0 {
            println!("{iter:>4}  {theta:>10.6}  {e:>10.6}");
        }

        // Central finite-difference gradient. Exact gate matrices make this
        // reliable at any angle (see issue #37 for what happens otherwise).
        let grad =
            (energy(theta + eps, &hamiltonian) - energy(theta - eps, &hamiltonian)) / (2.0 * eps);
        theta -= learning_rate * grad;
    }

    let final_energy = energy(theta, &hamiltonian);
    println!("\nConverged: theta = {theta:.6} (target {:.6})", PI);
    println!("Energy:    {final_energy:.8} (target -1)");
    assert!((final_energy + 1.0).abs() < 1e-6, "VQE failed to converge");
}
