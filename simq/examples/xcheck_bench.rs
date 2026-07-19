//! Cross-validation harness: prints the benchmark workloads' observable
//! values as JSON so `benchmarks/compare.py` can verify them against the
//! Qiskit implementation (to 1e-12) before publishing any timing comparison.
//!
//! Uses the exact same workload builders as `benches/end_to_end.rs`
//! (`simq::bench_workloads`), so a passing cross-check certifies the timed
//! circuits themselves.
//!
//! Usage: `cargo run --release -p simq --example xcheck_bench`

use simq::bench_workloads as wl;
use simq::state::AdaptiveState;

/// Qubit count for the multi-instance cross-check entries -- must match
/// `end_to_end.rs`'s `MULTI_INSTANCE_SIZE` and the Python baselines'
/// `MULTI_INSTANCE_SIZE`.
const MULTI_INSTANCE_SIZE: usize = 8;

fn ghz_p0(sim: &simq::sim::Simulator, n: usize) -> f64 {
    let circuit = wl::ghz_circuit(n);
    let result = sim.run(&circuit).unwrap();
    let amps = match result.state {
        AdaptiveState::Dense(d) => d.amplitudes().to_vec(),
        s => s.to_dense_vec(),
    };
    amps[0].norm_sqr()
}

fn main() {
    let sim = wl::default_simulator();
    let sizes = [4usize, 8, 12, 16];

    let mut entries = Vec::new();
    for &n in &sizes {
        entries.push(format!("    \"vqe_energy/{}q\": {:.15e}", n, wl::vqe_energy(&sim, n)));
        entries.push(format!("    \"qaoa_maxcut/{}q\": {:.15e}", n, wl::qaoa_cost(&sim, n)));
        entries.push(format!("    \"ghz_p0/{}q\": {:.15e}", n, ghz_p0(&sim, n)));
        entries.push(format!("    \"qft_probe/{}q\": {:.15e}", n, wl::qft_probe(&sim, n)));
        entries.push(format!(
            "    \"random_circuit/{}q\": {:.15e}",
            n,
            wl::random_circuit_p0(&sim, n)
        ));
    }

    let n = MULTI_INSTANCE_SIZE;
    for (i, energy) in wl::vqe_energy_instances(&sim, n).into_iter().enumerate() {
        entries.push(format!("    \"vqe_energy_multi/{n}q_i{i}\": {:.15e}", energy));
    }
    for (i, cost) in wl::qaoa_cost_instances(&sim, n).into_iter().enumerate() {
        entries.push(format!("    \"qaoa_maxcut_multi/{n}q_i{i}\": {:.15e}", cost));
    }

    println!("{{\n{}\n}}", entries.join(",\n"));
}
