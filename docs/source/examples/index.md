---
myst:
  html_meta:
    description: "Runnable Ferriq examples — Bell states, GHZ, quantum teleportation, VQE for the H2 molecule, QAOA MaxCut, compiler pipelines, and Python scripts."
---

# Examples

Every example below ships in the repository and runs out of the box. Rust
examples run with `cargo run -p <crate> --example <name>`; Python examples
run with `python <script>` after building the bindings.

## Start here

### Bell state / GHZ (Rust)

```rust
use ferriq::QuantumCircuit;

fn main() {
    let mut qc = QuantumCircuit::new(3);
    qc.h(0).cnot(0, 1).cnot(1, 2);

    let result = qc.simulate_with_shots(1024).unwrap();
    println!("{:?}", result.measurements.unwrap().sorted());
    // [("000", ~512), ("111", ~512)]
}
```

### Fluent API tour

```bash
cargo run -p ferriq --example fluent_api
```

Covers the whole builder surface: gates, inspection, ASCII rendering,
error handling, exact probabilities.

## Variational algorithms

### Minimal VQE with gradient descent

```bash
cargo run -p ferriq --example vqe_fluent
```

Optimizes the one-parameter ansatz RY(θ)|0⟩ against H = Z and converges to
energy −1 at θ = π. The whole loop:

```rust
use ferriq::{PauliObservable, PauliString, QuantumCircuit};

fn energy(theta: f64, hamiltonian: &PauliObservable) -> f64 {
    let mut qc = QuantumCircuit::new(1);
    qc.ry(theta, 0);
    qc.expectation_value(hamiltonian).expect("simulation failed")
}

fn main() {
    let hamiltonian = PauliObservable::from_pauli_string(
        PauliString::from_str("Z").unwrap(), 1.0);

    let (mut theta, lr, eps) = (0.5, 0.4, 1e-6);
    for _ in 0..40 {
        let grad = (energy(theta + eps, &hamiltonian)
                  - energy(theta - eps, &hamiltonian)) / (2.0 * eps);
        theta -= lr * grad;
    }
    println!("theta = {theta:.6}, energy = {:.8}", energy(theta, &hamiltonian));
}
```

### Full-scale VQE and QAOA

| Example | Command |
|---------|---------|
| H₂ molecule ground state | `cargo run -p ferriq-sim --example vqe_h2_molecule` |
| MaxCut with QAOA | `cargo run -p ferriq-sim --example qaoa_maxcut` |
| Comprehensive QAOA | `cargo run -p ferriq-sim --example qaoa_comprehensive` |
| Optimizer comparison (L-BFGS vs Nelder–Mead vs GD) | `cargo run -p ferriq-sim --example optimizer_comparison` |
| Convergence monitoring | `cargo run -p ferriq-sim --example convergence_monitoring` |
| Gradient method fallback | `cargo run -p ferriq-sim --example gradient_fallback` |

## States, measurement, and observables

| Example | Command |
|---------|---------|
| Quantum teleportation | `cargo run -p ferriq-state --example quantum_teleportation` |
| Pauli observables | `cargo run -p ferriq-state --example pauli_observables` |
| Computational-basis measurement | `cargo run -p ferriq-state --example computational_basis_measurement` |
| Sparse states | `cargo run -p ferriq-state --example sparse_state_demo` |
| Adaptive sparse↔dense conversion | `cargo run -p ferriq-state --example adaptive_conversion` |
| Copy-on-write state branching | `cargo run -p ferriq-state --example cow_branching` |
| Efficient sampling | `cargo run -p ferriq-state --example efficient_sampling_demo` |

## Circuit tooling

| Example | Command |
|---------|---------|
| Const-generic circuit builder | `cargo run -p ferriq-core --example circuit_builder` |
| ASCII circuit rendering | `cargo run -p ferriq-core --example ascii_circuit` |
| Bloch sphere | `cargo run -p ferriq-core --example bloch_sphere` |
| Circuit debugger | `cargo run -p ferriq-core --example circuit_debugger` |
| Validation | `cargo run -p ferriq-core --example circuit_validation` |
| Serialization | `cargo run -p ferriq-core --example circuit_serialization` |
| Parameter tracking | `cargo run -p ferriq-core --example parameter_tracking` |

## Gates and compiler

| Example | Command |
|---------|---------|
| Custom gates tutorial | `cargo run -p ferriq-gates --example custom_gates_tutorial` |
| Gate matrices | `cargo run -p ferriq-gates --example gate_matrices` |
| Compile-time lookup tables | `cargo run -p ferriq-gates --example lookup_table_demo` |
| Optimization pipeline | `cargo run -p ferriq-compiler --example pipeline_demo` |
| Gate fusion | `cargo run -p ferriq-compiler --example gate_fusion` |
| Diagonal gate optimization | `cargo run --example diagonal_gate_optimization` |

(See [the compiler guide](../guide/compiler.md) for the full list of
compiler demos.)

## Python examples

Located in [`ferriq-py/examples/`](https://github.com/glanzz/ferriq/tree/main/ferriq-py/examples):

| Script | What it shows |
|--------|---------------|
| `00_getting_started.py` | End-to-end tour of the Python API |
| `basic_circuit.py` | Circuit construction and simulation |
| `parameterized_circuit.py` | Parameterized gates and sweeps |
| `noise_simulation.py` | Noise channels and hardware models |
| `vqe_example.py` | Complete VQE optimization loop |

```bash
cd ferriq-py
maturin develop --release
python examples/vqe_example.py
```
