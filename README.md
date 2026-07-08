# Ferriq - High-Performance Quantum Computing SDK

[![Coverage Status](https://coveralls.io/repos/github/glanzz/ferriq/badge.svg?branch=main)](https://coveralls.io/github/glanzz/ferriq?branch=main)
[![Docs Site](https://github.com/glanzz/ferriq/actions/workflows/docs.yml/badge.svg)](https://glanzz.github.io/ferriq/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

**Ferriq** is a fast quantum computing SDK written in Rust, built for variational-algorithm loops: measured **4-45× faster than Qiskit** (Statevector *and* Aer) for VQE/QAOA energy evaluations up to ~10 qubits, with type safety and ergonomic APIs. Every performance number we quote is reproducible with one command — see [Performance](#performance) below and [BENCHMARKS.md](BENCHMARKS.md) for the full, warts-and-all comparison.

## Documentation

The full documentation lives at **[glanzz.github.io/ferriq](https://glanzz.github.io/ferriq/)**:

| Section | What's there |
|---------|--------------|
| [Installation](https://glanzz.github.io/ferriq/getting-started/installation.html) | Rust crate setup and Python bindings via maturin |
| [Quickstart (Rust)](https://glanzz.github.io/ferriq/getting-started/quickstart-rust.html) | Your first circuit in five minutes |
| [Quickstart (Python)](https://glanzz.github.io/ferriq/getting-started/quickstart-python.html) | The Python API tour |
| [User guide](https://glanzz.github.io/ferriq/guide/circuits.html) | Circuits, simulation, VQE/QAOA, compiler, noise, backends |
| [Examples](https://glanzz.github.io/ferriq/examples/) | Every runnable example in the workspace, catalogued |
| [Architecture](https://glanzz.github.io/ferriq/architecture/) | How the eight crates fit together |
| [Contributing](https://glanzz.github.io/ferriq/contributing/) | Dev setup, testing, and the PR workflow |

Rust API reference: `cargo doc --workspace --exclude ferriq-py --no-deps --open` locally (docs.rs will host it once the crate is published — see the note in [Quick Start](#quick-start)).

## Features

- **Fast where variational algorithms live**: 4-45× faster than Qiskit for VQE/QAOA evaluation loops up to ~10 qubits ([measured](BENCHMARKS.md))
- **⚡ Compile-Time Gate Matrix Caching (NEW!)**: Revolutionary multi-level caching system with ~0-5ns matrix access (see [details](#compile-time-caching))
- **Type-Safe**: Compile-time verification of quantum operations
- **Memory Efficient**: Hybrid sparse/dense state representation, simulate up to 35-40 qubits on 32GB RAM
- **Hardware Ready**: Same code runs on simulators and real quantum computers
- **Zero-Cost Abstractions**: No runtime overhead from high-level APIs
- **Built-in Gradients**: Automatic gradient computation for variational algorithms
- **Multi-Backend**: Support for IBM Quantum, AWS Braket, and more

## Performance

Measured on the workloads that matter for variational algorithms — one full
build-simulate-measure iteration, cross-validated so Ferriq and Qiskit provably
run identical circuits:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benchmarks/results/2026-07-08/chart-dark.svg">
  <img alt="Ferriq speedup over Qiskit per workload, log scale" src="benchmarks/results/2026-07-08/chart-light.svg">
</picture>

Ferriq wins big in the small-circuit regime where variational loops spend
thousands of iterations; Qiskit Aer is currently faster above ~12 qubits
(that gap is tracked in [#76](https://github.com/glanzz/ferriq/issues/76) —
we publish both sides). Exact circuits, qubit counts, hardware, versions,
and methodology: **[BENCHMARKS.md](BENCHMARKS.md)**. Reproduce on your
machine with one command:

```bash
./benchmarks/run.sh
```

## Quick Start

Add Ferriq to your `Cargo.toml` as a git dependency:

```toml
[dependencies]
ferriq = { git = "https://github.com/glanzz/ferriq" }
```

> **Why the git dependency?** Ferriq isn't on crates.io quite yet — the
> git dependency is the supported install path until `cargo add ferriq`
> lands. (Ferriq was briefly known as SimQ; it was renamed because the
> `simq` name on crates.io and PyPI belongs to unrelated projects.)

Create your first quantum circuit:

```rust
use ferriq::QuantumCircuit;

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

Gate methods chain fluently and never panic: the first invalid operation
(e.g. an out-of-range qubit) is recorded and returned as an error from
`build()` or `simulate()`. The full standard gate set is available —
`h`, `x`, `y`, `z`, `s`, `t`, `sx` (and daggers), `rx`, `ry`, `rz`, `p`,
`u1`/`u2`/`u3`, `cnot`/`cx`, `cy`, `cz`, `cp`, `swap`, `iswap`, `ecr`,
`rxx`/`ryy`/`rzz`, `toffoli`/`ccx`, `cswap` — plus a `gate(...)` escape
hatch for custom gates.

If you need lower-level control, the subcrate APIs remain fully accessible
through the same dependency:

```rust
use ferriq::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new(2);
    circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)])?;
    circuit.add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])?;

    let result = Simulator::new(SimulatorConfig::default()).run(&circuit)?;
    println!("Final state has {} qubits", result.num_qubits());
    Ok(())
}
```

## Why Ferriq?

### Performance First

Ferriq is designed from the ground up for speed:

- **Sparse state vectors** for memory-efficient simulation
- **SIMD-optimized** gate operations
- **Compile-time circuit optimization** with zero runtime overhead
- **Parallel execution** with automatic work distribution
- **Zero-copy operations** wherever possible

### Type Safety Without Compromise

For compile-time qubit bounds, use the const-generic `CircuitBuilder`:

```rust
use ferriq::prelude::*;

let mut builder = CircuitBuilder::<4>::new(); // 4-qubit circuit
let [q0, q1, q2, q3] = builder.qubits();      // typed qubit references
builder.apply_gate(Arc::new(Hadamard), &[q0]).unwrap();
builder.qubit(5).unwrap_err(); // qubit 5 doesn't exist — caught immediately
let circuit = builder.build();
```

### Built for Variational Algorithms

Exact expectation values are one call away, so an energy function for VQE is
a few lines:

```rust
use ferriq::{PauliObservable, PauliString, QuantumCircuit};

let hamiltonian = PauliObservable::from_pauli_string(
    PauliString::from_str("Z").unwrap(), 1.0);

let energy = |theta: f64| {
    let mut qc = QuantumCircuit::new(1);
    qc.ry(theta, 0);
    qc.expectation_value(&hamiltonian).unwrap()
};
// energy(θ) = cos(θ); minimize with your favourite optimizer
```

Run `cargo run -p ferriq --example vqe_fluent` for a complete gradient-descent
VQE loop, and see `ferriq-sim/examples/` for full workflows with the built-in
optimizers and gradient methods (`vqe_h2_molecule`, `qaoa_maxcut`, ...).

## Compile-Time Caching

Ferriq features a revolutionary **multi-level compile-time gate matrix caching system** that dramatically improves performance for rotation gates (RX, RY, RZ):

| Cache Level | Access Time | Speedup | Memory |
|-------------|-------------|---------|--------|
| Level 1: Common Angles (π/4, π/2, π) | ~0 ns | ∞× | 0 bytes |
| Level 2: Clifford+T (π/8, π/16, π/32) | ~1 ns | ~50× | ~1 KB |
| Level 3: π Fractions (π/3, π/5, π/6, ...) | ~1 ns | ~50× | ~2 KB |
| Level 4: VQE Range (0 to π/4, 256 steps) | ~2-5 ns | ~10× | 48 KB |
| Level 5: QAOA Range (0 to π, 100 steps) | ~2-5 ns | ~10× | 19 KB |
| Level 6: Runtime Compute (any angle) | ~20-50 ns | 1× | 0 bytes |

**Total static memory: ~70 KB** (embedded in binary)

**Accuracy guarantee**: every cache level is exact-match only — a cached
matrix is returned only when the requested angle equals the cached angle to
within 1e-12. Any other angle falls through to full-precision runtime
computation. Gate matrices are never approximated or snapped to a grid.

```rust
use ferriq_gates::RotationX;
use std::f64::consts::PI;

// Automatically uses optimal caching strategy
let rx1 = RotationX::new(PI / 4.0);   // ~0 ns (common angle cache)
let rx2 = RotationX::new(0.1);         // ~2-5 ns (VQE range cache)
let rx3 = RotationX::new(10.0);        // ~20-50 ns (runtime fallback)
```

**For complete documentation**, see [`ferriq-gates/COMPILE_TIME_CACHING.md`](ferriq-gates/COMPILE_TIME_CACHING.md)

## Architecture

Ferriq consists of several optimized crates:

- **ferriq**: Umbrella crate — the fluent `QuantumCircuit` builder plus re-exports of everything below
- **ferriq-core**: Core types and traits
- **ferriq-state**: Quantum state representations (sparse/dense)
- **ferriq-gates**: Gate library with SIMD optimizations and compile-time caching
- **ferriq-macros**: Procedural macros for compile-time code generation
- **ferriq-compiler**: Circuit optimization passes
- **ferriq-sim**: High-performance simulator
- **ferriq-backend**: Hardware backend abstraction





## Community

- **Questions, ideas, show & tell**: [GitHub Discussions](https://github.com/glanzz/ferriq/discussions)
- **Bugs and feature requests**: [Issues](https://github.com/glanzz/ferriq/issues)
- **First contribution?** Grab a [`good first issue`](https://github.com/glanzz/ferriq/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) — each one has file-level pointers and acceptance criteria, and we're happy to mentor you through it.

## Contributing

We welcome contributions! Please see the [contributing guide](https://glanzz.github.io/ferriq/contributing/) for development setup, coding standards, and the PR workflow.

### Development Setup

```bash
# Clone the repository
cd ferriq

# Build the project
cargo build

# Run tests
cargo test

# Run benchmarks (see BENCHMARKS.md; ./benchmarks/run.sh adds the Qiskit comparison)
cargo bench

# Format code
cargo fmt

# Run linter
cargo clippy

# Build the documentation site (output in docs/build/html)
pip install -r docs/requirements.txt
make -C docs html
```


## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Acknowledgments

Ferriq is inspired by:

- [Qiskit](https://qiskit.org/) - Python quantum computing framework
- [Cirq](https://quantumai.google/cirq) - Google's quantum framework
- [Q#](https://azure.microsoft.com/en-us/products/quantum/) - Microsoft's quantum language
- [QuEST](https://quest.qtechtheory.org/) - Quantum Exact Simulation Toolkit

## Citation

If you use Ferriq in your research, please cite:

```bibtex
@software{ferriq2024,
  title = {Ferriq: High-Performance Quantum Computing SDK in Rust},
  author = {Ferriq Contributors},
  year = {2024},
  url = {https://github.com/glanzz/ferriq}
}
```
