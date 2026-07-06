//! Research Workflow 2: Quantum Phase Estimation (QPE) via the inverse QFT
//!
//! QPE is the workhorse behind Shor's algorithm and quantum chemistry
//! eigenvalue estimation. We estimate the eigenphase phi of U = P(2*pi*phi)
//! acting on its eigenstate |1>:
//!
//!   Case 1: phi = 1/8  -> exactly representable with 3 counting bits,
//!                         the readout must be deterministic.
//!   Case 2: phi = 1/3  -> NOT representable, we observe the characteristic
//!                         sinc-squared leakage distribution around 0.333.
//!
//! Run with: cargo run --release -p simq-sim --example research_qpe

use simq_core::{ascii_renderer, Circuit, QubitId};
use simq_gates::standard::{CPhase, Hadamard, PauliX, Swap};
use simq_sim::Simulator;
use simq_state::DenseState;
use std::f64::consts::PI;
use std::sync::Arc;

/// Append an inverse QFT on `qubits` (given MSB-first) to the circuit.
fn inverse_qft(circuit: &mut Circuit, qubits: &[usize]) {
    let n = qubits.len();
    // Undo the bit-reversal swaps first
    for i in 0..n / 2 {
        circuit
            .add_gate(
                Arc::new(Swap),
                &[QubitId::new(qubits[i]), QubitId::new(qubits[n - 1 - i])],
            )
            .unwrap();
    }
    // Reverse of the QFT gate sequence, with negated angles
    for i in (0..n).rev() {
        for j in (i + 1..n).rev() {
            let angle = -PI / (1 << (j - i)) as f64;
            circuit
                .add_gate(
                    Arc::new(CPhase::new(angle)),
                    &[QubitId::new(qubits[j]), QubitId::new(qubits[i])],
                )
                .unwrap();
        }
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(qubits[i])])
            .unwrap();
    }
}

/// Build a QPE circuit with `t` counting qubits (0..t) and one eigenstate
/// qubit (index t) for the unitary U = P(2*pi*phi).
/// Counting qubit k controls U^(2^k).
fn qpe_circuit(t: usize, phi: f64) -> Circuit {
    let mut c = Circuit::new(t + 1);

    // Prepare eigenstate |1> of the phase gate on the target qubit
    c.add_gate(Arc::new(PauliX), &[QubitId::new(t)]).unwrap();

    // Superposition over the counting register
    for k in 0..t {
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(k)]).unwrap();
    }

    // Controlled-U^(2^k): phase kickback e^(2*pi*i*phi*2^k) onto qubit k
    for k in 0..t {
        let angle = 2.0 * PI * phi * (1 << k) as f64;
        c.add_gate(
            Arc::new(CPhase::new(angle)),
            &[QubitId::new(k), QubitId::new(t)],
        )
        .unwrap();
    }

    // Inverse QFT on the counting register (qubit t-1 is the MSB)
    let msb_first: Vec<usize> = (0..t).rev().collect();
    inverse_qft(&mut c, &msb_first);

    c
}

/// Return P(m) for each counting-register value m, marginalized over the
/// eigenstate qubit.
fn counting_distribution(
    sim: &Simulator,
    circuit: &Circuit,
    t: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let result = sim.run(circuit)?;
    let amps = result.state.to_dense_vec();
    let dense = DenseState::from_amplitudes(result.state.num_qubits(), &amps)?;
    let probs = dense.get_all_probabilities();

    let mut dist = vec![0.0; 1 << t];
    for (idx, p) in probs.iter().enumerate() {
        let m = idx & ((1 << t) - 1); // counting qubits are the low bits
        dist[m] += p;
    }
    Ok(dist)
}

fn print_distribution(dist: &[f64], t: usize, true_phi: f64) {
    println!("  m   bin    phi_est    P(m)");
    println!("  ---------------------------------------------");
    for (m, p) in dist.iter().enumerate() {
        let phi_est = m as f64 / (1 << t) as f64;
        let bar = "#".repeat((p * 60.0).round() as usize);
        println!(
            "  {}  {:0width$b}   {:.4}    {:.4}  {}",
            m,
            m,
            phi_est,
            p,
            bar,
            width = t
        );
    }
    let best = dist
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!(
        "\n  Best estimate: phi ~ {}/{} = {:.4}   (true phi = {:.4}, error = {:.4})",
        best.0,
        1 << t,
        best.0 as f64 / (1 << t) as f64,
        true_phi,
        (best.0 as f64 / (1 << t) as f64 - true_phi).abs()
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=========================================================");
    println!(" Workflow 2: Quantum Phase Estimation with inverse QFT");
    println!("=========================================================\n");

    let sim = Simulator::new(Default::default());
    let t = 3; // counting bits -> resolution 1/8

    // -----------------------------------------------------------
    // Case 1: phi = 1/8 (exactly representable -> deterministic)
    // -----------------------------------------------------------
    let phi = 1.0 / 8.0;
    println!("--- Case 1: U = P(2*pi/8), eigenphase phi = 0.125 ---\n");
    let circuit = qpe_circuit(t, phi);
    println!("{}", ascii_renderer::render(&circuit));
    println!(
        "Circuit: {} gates, {} qubits ({} counting + 1 eigenstate)\n",
        circuit.len(),
        circuit.num_qubits(),
        t
    );

    let dist = counting_distribution(&sim, &circuit, t)?;
    print_distribution(&dist, t, phi);

    // -----------------------------------------------------------
    // Case 2: phi = 1/3 (not representable -> leakage distribution)
    // -----------------------------------------------------------
    let phi = 1.0 / 3.0;
    println!("\n--- Case 2: U = P(2*pi/3), eigenphase phi = 0.3333 ---\n");
    let circuit = qpe_circuit(t, phi);
    let dist = counting_distribution(&sim, &circuit, t)?;
    print_distribution(&dist, t, phi);

    // Theoretical check: P(m) = |sin(pi*(2^t*phi - m)) / (2^t * sin(pi*(phi - m/2^t)))|^2
    println!("\n  Theory comparison (Fejer kernel):");
    let n = 1 << t;
    for (m, p) in dist.iter().enumerate() {
        let delta = phi - m as f64 / n as f64;
        let theory = if delta.abs() < 1e-12 {
            1.0
        } else {
            let num = (PI * n as f64 * delta).sin();
            let den = n as f64 * (PI * delta).sin();
            (num / den).powi(2)
        };
        println!(
            "    m={}: simulated {:.4} vs theory {:.4}  (diff {:.2e})",
            m,
            p,
            theory,
            (p - theory).abs()
        );
    }

    println!("\n==> QPE resolves exact phases perfectly and reproduces the");
    println!("    analytic leakage profile for non-representable phases.");
    Ok(())
}
