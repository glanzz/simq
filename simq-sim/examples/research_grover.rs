//! Research Workflow 3: Grover's Search — Amplitude Amplification Study
//!
//! We search an unstructured database of N = 8 items (3 qubits) for the
//! marked item |101> (index 5). A classical search needs ~N/2 = 4 queries
//! on average; Grover needs O(sqrt(N)) ~ 2.
//!
//! As researchers we don't just run it — we sweep the iteration count and
//! check the success probability against the theoretical prediction
//! P(k) = sin^2((2k+1) * asin(1/sqrt(N))), including the characteristic
//! "overshoot" when iterating past the optimum.
//!
//! Run with: cargo run --release -p simq-sim --example research_grover

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use simq_core::{ascii_renderer, Circuit, QubitId};
use simq_gates::standard::{CNot, Hadamard, PauliX, TGate, TGateDagger, Toffoli};
use simq_sim::Simulator;
use simq_state::measurement::ComputationalBasis;
use simq_state::DenseState;
use std::sync::Arc;

const N_QUBITS: usize = 3;
const MARKED: usize = 0b101; // qubit0=1, qubit1=0, qubit2=1

/// Toffoli decomposed into the standard 6-CNOT Clifford+T network.
///
/// NOTE: simq ships a native `Toffoli` gate, but the execution engine
/// rejects all 3-qubit gates ("Gates with 3 qubits not yet supported",
/// in both the dense and sparse paths) — see repo issue. We therefore
/// decompose it manually, exactly as one would for hardware.
fn toffoli(c: &mut Circuit, a: usize, b: usize, t: usize) {
    let cx = |c: &mut Circuit, x: usize, y: usize| {
        c.add_gate(Arc::new(CNot), &[QubitId::new(x), QubitId::new(y)])
            .unwrap()
    };
    let g1 = |c: &mut Circuit, g: Arc<dyn simq_core::Gate>, q: usize| {
        c.add_gate(g, &[QubitId::new(q)]).unwrap()
    };
    g1(c, Arc::new(Hadamard), t);
    cx(c, b, t);
    g1(c, Arc::new(TGateDagger), t);
    cx(c, a, t);
    g1(c, Arc::new(TGate), t);
    cx(c, b, t);
    g1(c, Arc::new(TGateDagger), t);
    cx(c, a, t);
    g1(c, Arc::new(TGate), b);
    g1(c, Arc::new(TGate), t);
    g1(c, Arc::new(Hadamard), t);
    cx(c, a, b);
    g1(c, Arc::new(TGate), a);
    g1(c, Arc::new(TGateDagger), b);
    cx(c, a, b);
}

/// CCZ via H-Toffoli-H conjugation on the target
fn ccz(c: &mut Circuit, q0: usize, q1: usize, q2: usize) {
    c.add_gate(Arc::new(Hadamard), &[QubitId::new(q2)]).unwrap();
    toffoli(c, q0, q1, q2);
    c.add_gate(Arc::new(Hadamard), &[QubitId::new(q2)]).unwrap();
}

/// Oracle: phase-flip the marked state |101>
/// (X on the 0-bits maps |101> -> |111>, CCZ flips |111>, X restores)
fn oracle(c: &mut Circuit) {
    for q in 0..N_QUBITS {
        if (MARKED >> q) & 1 == 0 {
            c.add_gate(Arc::new(PauliX), &[QubitId::new(q)]).unwrap();
        }
    }
    ccz(c, 0, 1, 2);
    for q in 0..N_QUBITS {
        if (MARKED >> q) & 1 == 0 {
            c.add_gate(Arc::new(PauliX), &[QubitId::new(q)]).unwrap();
        }
    }
}

/// Diffusion operator: reflection about the uniform superposition
fn diffusion(c: &mut Circuit) {
    for q in 0..N_QUBITS {
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
    }
    for q in 0..N_QUBITS {
        c.add_gate(Arc::new(PauliX), &[QubitId::new(q)]).unwrap();
    }
    ccz(c, 0, 1, 2);
    for q in 0..N_QUBITS {
        c.add_gate(Arc::new(PauliX), &[QubitId::new(q)]).unwrap();
    }
    for q in 0..N_QUBITS {
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
    }
}

fn grover_circuit(iterations: usize) -> Circuit {
    let mut c = Circuit::new(N_QUBITS);
    for q in 0..N_QUBITS {
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
    }
    for _ in 0..iterations {
        oracle(&mut c);
        diffusion(&mut c);
    }
    c
}

fn success_probability(
    sim: &Simulator,
    circuit: &Circuit,
) -> Result<f64, Box<dyn std::error::Error>> {
    let result = sim.run(circuit)?;
    Ok(result.state.get_probability(MARKED)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================================");
    println!(" Workflow 3: Grover Search for |101> in N=8 (3 qubits)");
    println!("========================================================\n");

    let sim = Simulator::new(Default::default());

    let n = 1usize << N_QUBITS;
    let theta = (1.0 / (n as f64).sqrt()).asin();
    let optimal = ((std::f64::consts::FRAC_PI_4) * (n as f64).sqrt()).floor() as usize;
    println!(
        "Marked item: |{:03b}>   Optimal iterations: pi/4*sqrt(8) ~ {}\n",
        MARKED, optimal
    );

    // Demonstrate the native-Toffoli limitation before working around it
    println!("--- Sanity check: native Toffoli gate ---");
    let mut native = Circuit::new(3);
    for q in 0..3 {
        native.add_gate(Arc::new(Hadamard), &[QubitId::new(q)])?;
    }
    native.add_gate(
        Arc::new(Toffoli),
        &[QubitId::new(0), QubitId::new(1), QubitId::new(2)],
    )?;
    match sim.run(&native) {
        Ok(_) => println!("  Native Toffoli executed fine.\n"),
        Err(e) => println!(
            "  Native Toffoli FAILED (using 6-CNOT Clifford+T decomposition instead):\n  {}\n",
            e
        ),
    }

    // One Grover iteration, rendered
    println!("Single Grover iteration (oracle + diffusion, decomposed Toffolis):\n");
    let one_iter = grover_circuit(1);
    println!("{}", ascii_renderer::render(&one_iter));

    // Sweep iterations and compare with theory
    println!("Amplitude amplification sweep:\n");
    println!("  k   P(success) sim   P(success) theory   ");
    println!("  ------------------------------------------------");
    for k in 0..=4 {
        let p_sim = success_probability(&sim, &grover_circuit(k))?;
        let p_theory = ((2 * k + 1) as f64 * theta).sin().powi(2);
        let bar = "#".repeat((p_sim * 40.0).round() as usize);
        println!(
            "  {}      {:.6}          {:.6}       {}",
            k, p_sim, p_theory, bar
        );
    }

    // Sample the optimal-depth circuit like a real experiment
    let shots = 4096;
    println!("\nSampling the k={} circuit with {} shots:\n", optimal, shots);
    let circuit = grover_circuit(optimal);
    let result = sim.run(&circuit)?;
    let amps = result.state.to_dense_vec();
    let dense = DenseState::from_amplitudes(N_QUBITS, &amps)?;
    let mut rng = StdRng::seed_from_u64(42);
    let sampling = ComputationalBasis::new().sample(&dense, shots, &mut || rng.gen::<f64>())?;

    let mut counts: Vec<(u64, usize)> = sampling.counts.iter().map(|(&k, &v)| (k, v)).collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1));
    for (outcome, count) in &counts {
        let p = *count as f64 / shots as f64;
        let marker = if *outcome as usize == MARKED { "  <-- marked" } else { "" };
        let bar = "#".repeat((p * 50.0).round() as usize);
        println!("  |{:03b}>  {:>5} shots ({:.4})  {}{}", outcome, count, p, bar, marker);
    }

    let p_marked = sampling.get_probability(MARKED as u64);
    println!(
        "\n==> Found the marked item with P = {:.3} after only {} oracle calls",
        p_marked, optimal
    );
    println!("    (classical exhaustive search: expected ~4 queries, P=1/8 per random guess)");

    Ok(())
}
