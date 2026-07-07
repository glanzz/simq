# SimQ - High-Performance Quantum Computing SDK

[![Coverage Status](https://coveralls.io/repos/github/glanzz/simq/badge.svg?branch=main)](https://coveralls.io/github/glanzz/simq?branch=main)
[![Crates.io](https://img.shields.io/crates/v/simq.svg)](https://crates.io/crates/simq)
[![Documentation](https://docs.rs/simq/badge.svg)](https://docs.rs/simq)
[![Docs Site](https://github.com/glanzz/simq/actions/workflows/docs.yml/badge.svg)](https://glanzz.github.io/simq/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

**SimQ** is a blazingly fast quantum computing SDK written in Rust, designed to be 8-10x faster than Qiskit for variational algorithms while maintaining type safety and ergonomic APIs.

## Documentation

The full documentation lives at **[glanzz.github.io/simq](https://glanzz.github.io/simq/)**:

| Section | What's there |
|---------|--------------|
| [Installation](https://glanzz.github.io/simq/getting-started/installation.html) | Rust crate setup and Python bindings via maturin |
| [Quickstart (Rust)](https://glanzz.github.io/simq/getting-started/quickstart-rust.html) | Your first circuit in five minutes |
| [Quickstart (Python)](https://glanzz.github.io/simq/getting-started/quickstart-python.html) | The Python API tour |
| [User guide](https://glanzz.github.io/simq/guide/circuits.html) | Circuits, simulation, VQE/QAOA, compiler, noise, backends |
| [Examples](https://glanzz.github.io/simq/examples/) | Every runnable example in the workspace, catalogued |
| [Architecture](https://glanzz.github.io/simq/architecture/) | How the eight crates fit together |
| [Contributing](https://glanzz.github.io/simq/contributing/) | Dev setup, testing, and the PR workflow |

Rust API reference: [docs.rs/simq](https://docs.rs/simq) (or `cargo doc --workspace --exclude simq-py --no-deps --open` locally).

## Features

- **Extreme Performance**: 8-10x faster than Qiskit for VQE/QAOA workloads
- **⚡ Compile-Time Gate Matrix Caching (NEW!)**: Revolutionary multi-level caching system with ~0-5ns matrix access (see [details](#compile-time-caching))
- **Type-Safe**: Compile-time verification of quantum operations
- **Memory Efficient**: Hybrid sparse/dense state representation, simulate up to 35-40 qubits on 32GB RAM
- **Hardware Ready**: Same code runs on simulators and real quantum computers
- **Zero-Cost Abstractions**: No runtime overhead from high-level APIs
- **Built-in Gradients**: Automatic gradient computation for variational algorithms
- **Multi-Backend**: Support for IBM Quantum, AWS Braket, and more

## Quick Start

Add SimQ to your `Cargo.toml`:

```toml
[dependencies]
simq = "0.1"
```

Create your first quantum circuit:

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

## Why SimQ?

### Performance First

SimQ is designed from the ground up for speed:

- **Sparse state vectors** for memory-efficient simulation
- **SIMD-optimized** gate operations
- **Compile-time circuit optimization** with zero runtime overhead
- **Parallel execution** with automatic work distribution
- **Zero-copy operations** wherever possible

### Type Safety Without Compromise

For compile-time qubit bounds, use the const-generic `CircuitBuilder`:

```rust
use simq::prelude::*;

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
use simq::{PauliObservable, PauliString, QuantumCircuit};

let hamiltonian = PauliObservable::from_pauli_string(
    PauliString::from_str("Z").unwrap(), 1.0);

let energy = |theta: f64| {
    let mut qc = QuantumCircuit::new(1);
    qc.ry(theta, 0);
    qc.expectation_value(&hamiltonian).unwrap()
};
// energy(θ) = cos(θ); minimize with your favourite optimizer
```

Run `cargo run -p simq --example vqe_fluent` for a complete gradient-descent
VQE loop, and see `simq-sim/examples/` for full workflows with the built-in
optimizers and gradient methods (`vqe_h2_molecule`, `qaoa_maxcut`, ...).

## Compile-Time Caching

SimQ features a revolutionary **multi-level compile-time gate matrix caching system** that dramatically improves performance for rotation gates (RX, RY, RZ):

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
use simq_gates::RotationX;
use std::f64::consts::PI;

// Automatically uses optimal caching strategy
let rx1 = RotationX::new(PI / 4.0);   // ~0 ns (common angle cache)
let rx2 = RotationX::new(0.1);         // ~2-5 ns (VQE range cache)
let rx3 = RotationX::new(10.0);        // ~20-50 ns (runtime fallback)
```

**For complete documentation**, see [`simq-gates/COMPILE_TIME_CACHING.md`](simq-gates/COMPILE_TIME_CACHING.md)

## Architecture

SimQ consists of several optimized crates:

- **simq**: Umbrella crate — the fluent `QuantumCircuit` builder plus re-exports of everything below
- **simq-core**: Core types and traits
- **simq-state**: Quantum state representations (sparse/dense)
- **simq-gates**: Gate library with SIMD optimizations and compile-time caching
- **simq-macros**: Procedural macros for compile-time code generation
- **simq-compiler**: Circuit optimization passes
- **simq-sim**: High-performance simulator
- **simq-backend**: Hardware backend abstraction





## Contributing

We welcome contributions! Please see the [contributing guide](https://glanzz.github.io/simq/contributing/) for development setup, coding standards, and the PR workflow.

### Development Setup

```bash
# Clone the repository
cd simq

# Build the project
cargo build

# Run tests
cargo test

# Run benchmarks
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

SimQ is inspired by:

- [Qiskit](https://qiskit.org/) - Python quantum computing framework
- [Cirq](https://quantumai.google/cirq) - Google's quantum framework
- [Q#](https://azure.microsoft.com/en-us/products/quantum/) - Microsoft's quantum language
- [QuEST](https://quest.qtechtheory.org/) - Quantum Exact Simulation Toolkit

## Citation

If you use SimQ in your research, please cite:

```bibtex
@software{simq2024,
  title = {SimQ: High-Performance Quantum Computing SDK in Rust},
  author = {SimQ Contributors},
  year = {2024},
  url = {https://github.com/glanzz/simq}
}
```
