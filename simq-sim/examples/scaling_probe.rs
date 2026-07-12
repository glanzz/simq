//! High-qubit scaling probe: where does SimQ fall over above 16 qubits?
//!
//! Times a 1-layer VQE-style circuit at increasing qubit counts and reports
//! the split between the sparse warm-up phase (before adaptive conversion)
//! and dense gate application. Run individual sizes to bound memory:
//! `cargo run --release -p simq-sim --example scaling_probe -- 20 22 24`

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

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("qubit count"))
        .collect();
    let sizes = if args.is_empty() {
        vec![16, 18, 20, 22]
    } else {
        args
    };

    println!(
        "{:>6} {:>7} {:>12} {:>12} {:>10}",
        "qubits", "gates", "total(ms)", "ms/gate", "GiB state"
    );
    for &n in &sizes {
        let circuit = vqe_circuit(n, 1);
        let gates = circuit.len();
        let sim = Simulator::new(SimulatorConfig::default());
        let reps = if n >= 24 { 1 } else { 3 };
        // Warmup only for small sizes (avoid double peak memory at large n)
        if n < 22 {
            sim.run(&circuit).unwrap();
        }
        let mut best = f64::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            match sim.run(&circuit) {
                Ok(r) => {
                    std::hint::black_box(r.state.num_qubits());
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                },
                Err(e) => {
                    println!("{:>6}  FAILED: {}", n, e);
                    best = f64::NAN;
                    break;
                },
            }
        }
        let gib = (1u64 << n) as f64 * 16.0 / (1u64 << 30) as f64;
        println!(
            "{:>6} {:>7} {:>12.1} {:>12.3} {:>10.3}",
            n,
            gates,
            best,
            best / gates as f64,
            gib
        );
    }
}
