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
        entries.push(format!(
            "    \"vqe_energy/{}q\": {:.15e}",
            n,
            wl::vqe_energy(&sim, n)
        ));
        entries.push(format!(
            "    \"qaoa_maxcut/{}q\": {:.15e}",
            n,
            wl::qaoa_cost(&sim, n)
        ));
        entries.push(format!("    \"ghz_p0/{}q\": {:.15e}", n, ghz_p0(&sim, n)));
    }

    println!("{{\n{}\n}}", entries.join(",\n"));
}
