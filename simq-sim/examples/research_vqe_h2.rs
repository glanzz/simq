//! Research Workflow 4: VQE Ground-State Energy of the H2 Molecule
//!
//! We minimize the full 2-qubit H2 Hamiltonian (BK-encoded, STO-3G basis,
//! R = 0.75 Angstrom, coefficients from O'Malley et al., PRX 6, 031007 (2016)):
//!
//!   H = g0*II + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*Y0Y1 + g5*X0X1
//!
//! with a hardware-efficient RY ansatz, and benchmark two optimizers
//! (adaptive gradient descent vs Adam) against the exact ground-state
//! energy from direct diagonalization. The target for quantum chemistry is
//! "chemical accuracy": |E - E_exact| < 1.6 mHa.
//!
//! Run with: cargo run --release -p simq-sim --example research_vqe_h2

use simq_core::{ascii_renderer, Circuit, QubitId};
use simq_gates::standard::{CNot, RotationY};
use simq_sim::gradient::{AdamConfig, AdamOptimizer, VQEConfig, VQEOptimizer};
use simq_sim::Simulator;
use simq_state::observable::{PauliObservable, PauliString};
use std::sync::Arc;

// Hamiltonian coefficients at R = 0.75 A (Hartree)
const G0: f64 = -0.4804;
const G1: f64 = 0.3435;
const G2: f64 = -0.4347;
const G3: f64 = 0.5716;
const G4: f64 = 0.0910; // Y0Y1
const G5: f64 = 0.0910; // X0X1

fn h2_hamiltonian() -> Result<PauliObservable, Box<dyn std::error::Error>> {
    let mut obs = PauliObservable::from_pauli_string(PauliString::from_str("II")?, G0);
    obs.add_term(PauliString::from_str("ZI")?, G1); // Z on qubit 0
    obs.add_term(PauliString::from_str("IZ")?, G2); // Z on qubit 1
    obs.add_term(PauliString::from_str("ZZ")?, G3);
    obs.add_term(PauliString::from_str("YY")?, G4);
    obs.add_term(PauliString::from_str("XX")?, G5);
    Ok(obs)
}

/// Exact ground energy by diagonalizing the 4x4 Hamiltonian.
/// XX+YY (equal weights) couples only |01> <-> |10> with element g4+g5;
/// |00> <-> |11> couples with g5-g4 = 0. The matrix is block diagonal.
fn exact_ground_energy() -> f64 {
    let diag = |b0: f64, b1: f64| G0 + G1 * b0 + G2 * b1 + G3 * b0 * b1;
    let e00 = diag(1.0, 1.0); // |q1 q0> = |00>
    let e01 = diag(-1.0, 1.0); // q0=1
    let e10 = diag(1.0, -1.0); // q1=1
    let e11 = diag(-1.0, -1.0);

    let avg = 0.5 * (e01 + e10);
    let det = (0.5 * (e01 - e10)).hypot(G4 + G5);
    let e_block = avg - det;

    e00.min(e11).min(e_block)
}

/// Hardware-efficient ansatz: RY layer - CNOT - RY layer (4 parameters)
fn ansatz(params: &[f64]) -> Circuit {
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(RotationY::new(params[0])), &[QubitId::new(0)])
        .unwrap();
    c.add_gate(Arc::new(RotationY::new(params[1])), &[QubitId::new(1)])
        .unwrap();
    c.add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
        .unwrap();
    c.add_gate(Arc::new(RotationY::new(params[2])), &[QubitId::new(0)])
        .unwrap();
    c.add_gate(Arc::new(RotationY::new(params[3])), &[QubitId::new(1)])
        .unwrap();
    c
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=========================================================");
    println!(" Workflow 4: VQE — H2 Ground State (full Hamiltonian)");
    println!("=========================================================\n");

    let simulator = Simulator::new(Default::default());
    let observable = h2_hamiltonian()?;
    let e_exact = exact_ground_energy();

    println!("Hamiltonian: {} Pauli terms", observable.num_terms());
    println!(
        "  H = {:+.4} II {:+.4} Z0 {:+.4} Z1 {:+.4} Z0Z1 {:+.4} Y0Y1 {:+.4} X0X1",
        G0, G1, G2, G3, G4, G5
    );
    println!("\nExact ground-state energy (diagonalization): {:.6} Ha", e_exact);

    println!("\nAnsatz (hardware-efficient, 4 parameters):\n");
    println!("{}", ascii_renderer::render(&ansatz(&[0.1, 0.2, 0.3, 0.4])));

    let initial = vec![0.1, -0.1, 0.05, 0.05];

    // ------------------------------------------------------------------
    // Optimizer 1: adaptive gradient descent
    // ------------------------------------------------------------------
    println!("--- Optimizer 1: adaptive gradient descent ---\n");
    let config = VQEConfig {
        max_iterations: 300,
        learning_rate: 0.2,
        adaptive_learning_rate: true,
        energy_tolerance: 1e-9,
        gradient_tolerance: 1e-7,
        ..Default::default()
    };
    let mut vqe = VQEOptimizer::new(ansatz, config);
    let gd = vqe.optimize(&simulator, &observable, &initial)?;

    println!("  Iter    Energy (Ha)      |grad|");
    for step in gd
        .history
        .iter()
        .step_by((gd.history.len() / 10).max(1))
    {
        println!(
            "  {:>4}   {:>12.8}   {:>10.2e}",
            step.iteration, step.energy, step.gradient_norm
        );
    }
    println!(
        "\n  Final: E = {:.8} Ha in {} iterations ({:?}), status {:?}",
        gd.energy, gd.num_iterations, gd.total_time, gd.status
    );

    // ------------------------------------------------------------------
    // Optimizer 2: Adam
    // ------------------------------------------------------------------
    println!("\n--- Optimizer 2: Adam ---\n");
    let adam_config = AdamConfig {
        learning_rate: 0.1,
        max_iterations: 300,
        energy_tolerance: 1e-9,
        gradient_tolerance: 1e-7,
        ..Default::default()
    };
    let mut adam = AdamOptimizer::new(ansatz, adam_config);
    let ad = adam.optimize(&simulator, &observable, &initial)?;
    println!(
        "  Final: E = {:.8} Ha in {} iterations ({:?}), status {:?}",
        ad.energy, ad.num_iterations, ad.total_time, ad.status
    );

    // ------------------------------------------------------------------
    // Research summary
    // ------------------------------------------------------------------
    println!("\n--- Summary ---\n");
    println!("  {:<22} {:>14} {:>12} {:>10}", "Method", "Energy (Ha)", "Error (mHa)", "Iters");
    println!("  {:-<62}", "");
    println!(
        "  {:<22} {:>14.8} {:>12.4} {:>10}",
        "Exact diagonalization", e_exact, 0.0, "-"
    );
    println!(
        "  {:<22} {:>14.8} {:>12.4} {:>10}",
        "VQE (adaptive GD)",
        gd.energy,
        (gd.energy - e_exact) * 1000.0,
        gd.num_iterations
    );
    println!(
        "  {:<22} {:>14.8} {:>12.4} {:>10}",
        "VQE (Adam)",
        ad.energy,
        (ad.energy - e_exact) * 1000.0,
        ad.num_iterations
    );

    let best_err = (gd.energy - e_exact).abs().min((ad.energy - e_exact).abs());
    println!(
        "\n  Chemical accuracy (<1.6 mHa): {}",
        if best_err < 1.6e-3 { "ACHIEVED" } else { "NOT reached" }
    );
    println!("\n  Optimal parameters (best run):");
    let best = if (gd.energy - e_exact).abs() <= (ad.energy - e_exact).abs() { &gd } else { &ad };
    for (i, p) in best.parameters.iter().enumerate() {
        println!("    theta_{} = {:+.6} rad", i, p);
    }

    Ok(())
}
