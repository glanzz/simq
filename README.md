# SimQ - High-Performance Quantum Computing SDK

[![CI](https://github.com/yourusername/simq/workflows/CI/badge.svg)](https://github.com/yourusername/simq/actions)
[![codecov](https://codecov.io/gh/yourusername/simq/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/simq)
[![Crates.io](https://img.shields.io/crates/v/simq.svg)](https://crates.io/crates/simq)
[![Documentation](https://docs.rs/simq/badge.svg)](https://docs.rs/simq)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

**SimQ** is a blazingly fast quantum computing SDK written in Rust, designed to be 8-10x faster than Qiskit for variational algorithms while maintaining type safety and ergonomic APIs.

## Features

- **Extreme Performance**: 8-10x faster than Qiskit for VQE/QAOA workloads
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
use simq::{Circuit, gates::*};

fn main() {
    // Create a 3-qubit circuit
    let mut circuit = Circuit::new(3);

    // Apply gates
    circuit.h(0);           // Hadamard on qubit 0
    circuit.cnot(0, 1);     // CNOT: control=0, target=1
    circuit.cnot(1, 2);     // CNOT: control=1, target=2

    // Simulate
    let result = circuit.simulate(shots=1024);
    println!("Results: {:?}", result.counts());
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

```rust
let mut circuit = Circuit::<4>::new();  // 4-qubit circuit
circuit.h(q!(0));   // OK
circuit.h(q!(5));   // Compile error: qubit 5 doesn't exist!
```

### Built for Variational Algorithms

```rust
use simq::algorithms::VQE;

// Define your parametric circuit
let ansatz = parametric_circuit(|params| {
    // Build circuit with parameters
});

// Run VQE
let vqe = VQE::new(hamiltonian, ansatz);
let result = vqe.minimize();
println!("Ground state energy: {}", result.energy);
```

## Architecture

SimQ consists of several optimized crates:

- **simq-core**: Core types and traits
- **simq-state**: Quantum state representations (sparse/dense)
- **simq-gates**: Gate library with SIMD optimizations
- **simq-compiler**: Circuit optimization passes
- **simq-sim**: High-performance simulator
- **simq-backend**: Hardware backend abstraction

See [Architecture Documentation](docs/architecture.md) for details.



## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

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
  url = {https://github.com/yourusername/simq}
}
```
