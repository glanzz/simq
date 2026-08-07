//! Research Workflow 1: Entanglement Verification & CHSH Bell Inequality
//!
//! A standard first experiment when validating a quantum simulator or device:
//! 1. Prepare Bell and GHZ states
//! 2. Verify measurement correlations from sampled shots
//! 3. Run a CHSH experiment — a classical device is bounded by |S| <= 2,
//!    quantum mechanics reaches |S| = 2*sqrt(2) ~ 2.828 (Tsirelson's bound)
//!
//! Run with: cargo run --release -p simq-sim --example research_entanglement

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use simq_core::{ascii_renderer, Circuit, QubitId};
use simq_gates::standard::{CNot, Hadamard, RotationY};
use simq_sim::Simulator;
use simq_state::measurement::ComputationalBasis;
use simq_state::observable::{PauliObservable, PauliString};
use simq_state::DenseState;
use std::sync::Arc;

const SHOTS: usize = 8192;

/// Sample measurement counts from the final state of a circuit
fn sample_counts(
    sim: &Simulator,
    circuit: &Circuit,
    shots: usize,
    seed: u64,
) -> Result<Vec<(String, usize)>, Box<dyn std::error::Error>> {
    let result = sim.run(circuit)?;
    let amps = result.state.to_dense_vec();
    let dense = DenseState::from_amplitudes(result.state.num_qubits(), &amps)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let sampling = ComputationalBasis::new().sample(&dense, shots, &mut || rng.gen::<f64>())?;
    let mut counts: Vec<(String, usize)> = sampling
        .to_bitstring_counts(result.state.num_qubits())
        .into_iter()
        .collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(counts)
}

fn print_histogram(counts: &[(String, usize)], shots: usize) {
    for (bits, count) in counts {
        let p = *count as f64 / shots as f64;
        let bar = "#".repeat((p * 50.0).round() as usize);
        println!("  |{}>  {:>6} shots  ({:>6.3})  {}", bits, count, p, bar);
    }
}

/// Expectation value <O> computed from the exact final state
fn expectation(
    sim: &Simulator,
    circuit: &Circuit,
    obs: &PauliObservable,
) -> Result<f64, Box<dyn std::error::Error>> {
    let result = sim.run(circuit)?;
    let amps = result.state.to_dense_vec();
    let dense = DenseState::from_amplitudes(result.state.num_qubits(), &amps)?;
    Ok(obs.expectation_value(&dense)?)
}

/// Bell state circuit with measurement-basis rotations for CHSH.
/// Measuring qubit i in the basis rotated by angle theta around Y is
/// equivalent to applying RY(-theta) before a Z measurement.
fn chsh_circuit(theta_a: f64, theta_b: f64) -> Circuit {
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
        .unwrap();
    c.add_gate(Arc::new(RotationY::new(-theta_a)), &[QubitId::new(0)])
        .unwrap();
    c.add_gate(Arc::new(RotationY::new(-theta_b)), &[QubitId::new(1)])
        .unwrap();
    c
}

/// Correlation E(a,b) = <Z Z> estimated from sampled shots
fn correlation_from_shots(
    sim: &Simulator,
    circuit: &Circuit,
    shots: usize,
    seed: u64,
) -> Result<f64, Box<dyn std::error::Error>> {
    let counts = sample_counts(sim, circuit, shots, seed)?;
    let mut e = 0.0;
    for (bits, count) in &counts {
        let parity = bits.chars().filter(|&c| c == '1').count() % 2;
        let sign = if parity == 0 { 1.0 } else { -1.0 };
        e += sign * (*count as f64) / shots as f64;
    }
    Ok(e)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==============================================================");
    println!(" Workflow 1: Entanglement Verification & CHSH Bell Inequality");
    println!("==============================================================\n");

    let sim = Simulator::new(Default::default());

    // ---------------------------------------------------------------
    // Part A: Bell state |Phi+> = (|00> + |11>)/sqrt(2)
    // ---------------------------------------------------------------
    println!("--- Part A: Bell state |Phi+> ---\n");
    let mut bell = Circuit::new(2);
    bell.add_gate(Arc::new(Hadamard), &[QubitId::new(0)])?;
    bell.add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])?;

    println!("{}", ascii_renderer::render(&bell));

    let counts = sample_counts(&sim, &bell, SHOTS, 7)?;
    println!("Measurement histogram ({} shots):", SHOTS);
    print_histogram(&counts, SHOTS);

    let zz = expectation(&sim, &bell, &PauliObservable::from_pauli_string(PauliString::from_str("ZZ")?, 1.0))?;
    let xx = expectation(&sim, &bell, &PauliObservable::from_pauli_string(PauliString::from_str("XX")?, 1.0))?;
    println!("\nExact correlations: <ZZ> = {:+.6}   <XX> = {:+.6}", zz, xx);
    println!("(both +1 -> maximally entangled, phase-coherent Bell pair)\n");

    // ---------------------------------------------------------------
    // Part B: 4-qubit GHZ state
    // ---------------------------------------------------------------
    println!("--- Part B: 4-qubit GHZ state ---\n");
    let n = 4;
    let mut ghz = Circuit::new(n);
    ghz.add_gate(Arc::new(Hadamard), &[QubitId::new(0)])?;
    for i in 0..n - 1 {
        ghz.add_gate(Arc::new(CNot), &[QubitId::new(i), QubitId::new(i + 1)])?;
    }
    println!("{}", ascii_renderer::render(&ghz));

    let counts = sample_counts(&sim, &ghz, SHOTS, 11)?;
    println!("Measurement histogram ({} shots):", SHOTS);
    print_histogram(&counts, SHOTS);

    let zzzz = expectation(&sim, &ghz, &PauliObservable::from_pauli_string(PauliString::from_str("ZZZZ")?, 1.0))?;
    let xxxx = expectation(&sim, &ghz, &PauliObservable::from_pauli_string(PauliString::from_str("XXXX")?, 1.0))?;
    println!("\nGHZ witnesses: <ZZZZ> = {:+.6}   <XXXX> = {:+.6}", zzzz, xxxx);

    // ---------------------------------------------------------------
    // Part C: CHSH inequality
    // For |Phi+> the correlation is E(a,b) = cos(theta_a - theta_b).
    // Alice angles: a = 0, a' = pi/2; Bob angles: b = pi/4, b' = 3*pi/4
    // S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
    // ---------------------------------------------------------------
    println!("\n--- Part C: CHSH Bell test ({} shots per setting) ---\n", SHOTS);
    let (a, ap) = (0.0, std::f64::consts::FRAC_PI_2);
    let (b, bp) = (std::f64::consts::FRAC_PI_4, 3.0 * std::f64::consts::FRAC_PI_4);

    let e_ab = correlation_from_shots(&sim, &chsh_circuit(a, b), SHOTS, 21)?;
    let e_abp = correlation_from_shots(&sim, &chsh_circuit(a, bp), SHOTS, 22)?;
    let e_apb = correlation_from_shots(&sim, &chsh_circuit(ap, b), SHOTS, 23)?;
    let e_apbp = correlation_from_shots(&sim, &chsh_circuit(ap, bp), SHOTS, 24)?;

    println!("  E(a , b ) = {:+.4}", e_ab);
    println!("  E(a , b') = {:+.4}", e_abp);
    println!("  E(a', b ) = {:+.4}", e_apb);
    println!("  E(a', b') = {:+.4}", e_apbp);

    let s = e_ab - e_abp + e_apb + e_apbp;
    println!("\n  CHSH S = {:.4}", s);
    println!("  Classical bound:  |S| <= 2");
    println!("  Tsirelson bound:  |S| <= 2*sqrt(2) = {:.4}", 2.0 * (2.0_f64).sqrt());
    if s.abs() > 2.0 {
        println!("\n  ==> Bell inequality VIOLATED: correlations are genuinely quantum.");
    } else {
        println!("\n  ==> No violation observed (unexpected for an ideal simulator!)");
    }

    Ok(())
}
