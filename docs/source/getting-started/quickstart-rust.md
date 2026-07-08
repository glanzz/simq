---
myst:
  html_meta:
    description: "SimQ Rust quickstart — build and simulate your first quantum circuit with the fluent QuantumCircuit API in five minutes."
---

# Quickstart (Rust)

This walkthrough takes you from an empty project to a simulated quantum
circuit with measurement statistics.

## 1. Create a project

```bash
cargo new hello-quantum
cd hello-quantum
cargo add simq --git https://github.com/glanzz/simq
```

```{warning}
The `--git` flag matters: a plain `cargo add simq` installs an unrelated
job-queue crate that happens to own the `simq` name on crates.io.
```

## 2. Build and simulate a circuit

The fluent `QuantumCircuit` builder is the fastest
way to get going. Replace `src/main.rs` with:

```rust
use simq::QuantumCircuit;

fn main() {
    // Create a 3-qubit GHZ circuit
    let mut qc = QuantumCircuit::new(3);
    qc.h(0)          // Hadamard on qubit 0
        .cnot(0, 1)  // CNOT: control=0, target=1
        .cnot(1, 2); // CNOT: control=1, target=2

    // Simulate with 1024 measurement shots
    let result = qc.simulate_with_shots(1024).unwrap();
    let counts = result.measurements.unwrap();
    println!("Results: {:?}", counts.sorted());
    // e.g. [("000", 517), ("111", 507)]
}
```

Run it:

```bash
cargo run
```

Only `000` and `111` appear — the three qubits are entangled in a GHZ state.

### Error handling without panics

Gate methods chain fluently and never panic: the first invalid operation
(for example, an out-of-range qubit index) is recorded and returned as an
error from `build()` or `simulate()`:

```rust
use simq::QuantumCircuit;

let mut qc = QuantumCircuit::new(2);
qc.h(0).cnot(0, 5); // qubit 5 doesn't exist
assert!(qc.build().is_err());
```

### The standard gate set

All common gates are available as chainable methods:

| Category | Methods |
|----------|---------|
| Single-qubit | `h`, `x`, `y`, `z`, `id`, `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg` |
| Rotations | `rx(θ, q)`, `ry(θ, q)`, `rz(θ, q)`, `p(θ, q)`, `u1`, `u2`, `u3` |
| Two-qubit | `cnot`/`cx`, `cy`, `cz`, `cp`, `swap`, `iswap`, `ecr`, `rxx`, `ryy`, `rzz` |
| Three-qubit | `toffoli`/`ccx`, `cswap` |
| Escape hatch | `gate(Arc<dyn Gate>, &[qubits])` for custom gates |

## 3. Exact quantities, not just shots

You don't need sampling to inspect a state — probabilities and expectation
values are exact:

```rust
use simq::{PauliObservable, PauliString, QuantumCircuit};

let mut qc = QuantumCircuit::new(1);
qc.ry(0.8, 0);

// Exact basis-state probabilities
let probs = qc.probabilities().unwrap();

// Exact ⟨Z⟩ expectation value: cos(0.8)
let z = PauliObservable::from_pauli_string(
    PauliString::from_str("Z").unwrap(), 1.0);
let energy = qc.expectation_value(&z).unwrap();
assert!((energy - 0.8f64.cos()).abs() < 1e-10);
```

## 4. Drop down when you need control

The subcrate APIs stay fully accessible through the same dependency:

```rust
use simq::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new(2);
    circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)])?;
    circuit.add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])?;

    let result = Simulator::new(SimulatorConfig::default()).run(&circuit)?;
    println!("Final state has {} qubits", result.num_qubits());
    Ok(())
}
```

For compile-time qubit-bounds checking, use the const-generic
`CircuitBuilder`:

```rust
use simq::prelude::*;

let mut builder = CircuitBuilder::<4>::new(); // 4-qubit circuit
let [q0, q1, q2, q3] = builder.qubits();      // typed qubit references
builder.apply_gate(Arc::new(Hadamard), &[q0]).unwrap();
builder.qubit(5).unwrap_err(); // qubit 5 doesn't exist — caught immediately
let circuit = builder.build();
```

## Where to next?

- [Circuits in depth](../guide/circuits.md) — builders, validation, visualization
- [Simulation](../guide/simulation.md) — configuration, sparse vs dense, statistics
- [Observables & VQE](../guide/observables-vqe.md) — variational workflows
- [Examples](../examples/index.md) — full runnable programs
