//! Gate-application performance probe for issue #76.
//!
//! Times `Simulator::run` on layered VQE/QAOA-style circuits at 4-16 qubits
//! and reports per-gate cost so kernel regressions are visible without a full
//! criterion run: `cargo run --release -p simq-sim --example perf_probe`

use simq_core::{Circuit, QubitId};
use simq_gates::standard::{CNot, Hadamard, RotationY, RotationZ};
use simq_sim::{Simulator, SimulatorConfig};
use std::sync::Arc;
use std::time::Instant;

fn vqe_circuit(num_qubits: usize, layers: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits);
    for q in 0..num_qubits {
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
    }
    for l in 0..layers {
        for q in 0..num_qubits {
            let theta = 0.1 + 0.37 * (l * num_qubits + q) as f64;
            c.add_gate(Arc::new(RotationY::new(theta)), &[QubitId::new(q)])
                .unwrap();
        }
        for q in 0..num_qubits - 1 {
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
                .unwrap();
        }
        for q in 0..num_qubits {
            let phi = 0.05 + 0.21 * (l * num_qubits + q) as f64;
            c.add_gate(Arc::new(RotationZ::new(phi)), &[QubitId::new(q)])
                .unwrap();
        }
    }
    c
}

fn time_run(sim: &Simulator, circuit: &Circuit, reps: usize) -> f64 {
    // Warmup
    sim.run(circuit).unwrap();
    let start = Instant::now();
    for _ in 0..reps {
        let r = sim.run(circuit).unwrap();
        std::hint::black_box(r.state.num_qubits());
    }
    start.elapsed().as_secs_f64() * 1e3 / reps as f64
}

fn main() {
    println!(
        "{:>6} {:>7} {:>12} {:>12} {:>14}",
        "qubits", "gates", "default(ms)", "no-opt(ms)", "ns/amp/gate"
    );
    for &n in &[4usize, 8, 12, 16] {
        let circuit = vqe_circuit(n, 3);
        let gates = circuit.len();
        let reps = if n >= 16 { 5 } else { 20 };

        let sim = Simulator::new(SimulatorConfig::default());
        let default_ms = time_run(&sim, &circuit, reps);

        let sim_noopt = Simulator::new(SimulatorConfig::default().with_optimization(false));
        let noopt_ms = time_run(&sim_noopt, &circuit, reps);

        let ns_per_amp_gate = noopt_ms * 1e6 / (gates as f64) / (1u64 << n) as f64;
        println!(
            "{:>6} {:>7} {:>12.3} {:>12.3} {:>14.2}",
            n, gates, default_ms, noopt_ms, ns_per_amp_gate
        );
    }
}
