//! Fluent circuit-building API demo
//!
//! Run with: cargo run -p ferriq --example fluent_api
//!
//! Shows the `QuantumCircuit` facade: named gate methods that chain without
//! `Arc`/`QubitId`/`unwrap` boilerplate, with errors deferred until
//! `build()`/`simulate()`.

use ferriq::QuantumCircuit;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Bell state ---
    let mut bell = QuantumCircuit::new(2);
    bell.h(0).cnot(0, 1);

    let result = bell.simulate_with_shots(1024)?;
    let counts = result.measurements.expect("shots > 0 always yields counts");
    println!("Bell state counts ({} shots):", counts.total_shots());
    for (bitstring, count) in counts.sorted() {
        println!("  |{bitstring}⟩: {count}");
    }

    // --- Parameterized ansatz: the nine-line Arc-based version is now three lines ---
    let params = [0.3, 0.8, 1.2];
    let mut ansatz = QuantumCircuit::new(3);
    ansatz.ry(params[0], 0).ry(params[1], 1).ry(params[2], 2);
    ansatz.cnot(0, 1).cnot(1, 2).rz(PI / 3.0, 2);

    println!("\nAnsatz ({} gates, depth {}):", ansatz.len(), ansatz.depth());
    println!("{}", ansatz.to_ascii());

    // --- Error handling: no panics, errors surface at the end ---
    let mut bad = QuantumCircuit::new(2);
    bad.h(0).cnot(0, 7); // qubit 7 doesn't exist
    match bad.build() {
        Ok(_) => unreachable!(),
        Err(e) => println!("Deferred build error, as expected: {e}"),
    }

    Ok(())
}
