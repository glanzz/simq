---
myst:
  html_meta:
    description: "SimQ is a high-performance quantum computing SDK written in Rust with Python bindings — measured 1.6-90x faster than Qiskit Aer on a cross-validated VQE/QAOA/sampling suite at 4-16 qubits, type-safe, and memory efficient."
    keywords: "quantum computing, quantum simulator, Rust, Python, VQE, QAOA, SDK, quantum circuits, SimQ"
---

# SimQ

```{raw} html
<div class="simq-hero">
  <h1>Quantum computing at Rust speed</h1>
  <p class="simq-tagline">
    SimQ is a high-performance quantum computing SDK written in Rust —
    measured 1.6–90× faster than Qiskit Aer on a cross-validated 4–16 qubit suite,
    with type-safe circuit construction and first-class Python bindings.
  </p>
  <div class="simq-buttons">
    <a class="simq-btn simq-btn-primary" href="getting-started/quickstart-rust.html">Get started&nbsp;→</a>
    <a class="simq-btn simq-btn-secondary" href="https://github.com/glanzz/simq">View on GitHub</a>
  </div>
</div>
```

::::{tab-set}

:::{tab-item} Rust
```rust
use simq::QuantumCircuit;

fn main() {
    // 3-qubit GHZ state
    let mut qc = QuantumCircuit::new(3);
    qc.h(0).cnot(0, 1).cnot(1, 2);

    let result = qc.simulate_with_shots(1024).unwrap();
    println!("{:?}", result.measurements.unwrap().sorted());
    // [("000", 517), ("111", 507)]
}
```
:::

:::{tab-item} Python
```python
import simq

builder = simq.CircuitBuilder(2)
builder.h(0)
builder.cx(0, 1)
circuit = builder.build()

simulator = simq.Simulator(simq.SimulatorConfig(shots=1024))
result = simulator.run(circuit)
print(result.state_vector)
```
:::

::::

## Why SimQ?

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} ⚡ Extreme performance
Sparse state vectors, SIMD-optimized gate kernels, parallel execution, and
compile-time gate matrix caching with ~0–5 ns matrix access.
:::

:::{grid-item-card} 🛡️ Type-safe by construction
Compile-time verification of quantum operations. Invalid circuits are caught
before they ever run — often before they even compile.
:::

:::{grid-item-card} 🧠 Memory efficient
Hybrid sparse/dense state representation lets you simulate up to 35–40
qubits on 32 GB of RAM.
:::

:::{grid-item-card} 📉 Built for variational algorithms
Exact expectation values, automatic gradients, and ready-made VQE/QAOA
helpers and optimizers.
:::

:::{grid-item-card} 🔌 Hardware ready
The same circuit runs on the local simulator or real quantum hardware via
the backend abstraction (IBM Quantum and more).
:::

:::{grid-item-card} 🐍 First-class Python bindings
A familiar, Qiskit-like Python API backed by the full-speed Rust core —
including noise models and visualization.
:::

::::

## Explore the documentation

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 🚀 Getting started
:link: getting-started/installation
:link-type: doc
Install SimQ and run your first circuit in Rust or Python in under five
minutes.
:::

:::{grid-item-card} 📖 User guide
:link: guide/circuits
:link-type: doc
Circuits, simulation, observables, VQE/QAOA, the compiler, noise models,
and hardware backends.
:::

:::{grid-item-card} 🧪 Examples
:link: examples/index
:link-type: doc
Runnable, end-to-end examples: Bell states, teleportation, H₂ ground-state
VQE, MaxCut QAOA, and more.
:::

:::{grid-item-card} 🏗️ Architecture
:link: architecture/index
:link-type: doc
How the eight workspace crates fit together — for contributors and the
curious.
:::

:::{grid-item-card} 🤝 Contributing
:link: contributing/index
:link-type: doc
Development setup, coding standards, testing, and how to send your first
pull request.
:::

:::{grid-item-card} 🔍 API reference
:link: api/rust
:link-type: doc
Rust API docs (rustdoc) and the Python binding reference.
:::

::::

```{toctree}
:hidden:
:caption: Getting started

getting-started/installation
getting-started/quickstart-rust
getting-started/quickstart-python
```

```{toctree}
:hidden:
:caption: User guide

guide/circuits
guide/simulation
guide/observables-vqe
guide/gates-caching
guide/compiler
guide/noise
guide/backends
```

```{toctree}
:hidden:
:caption: Examples

examples/index
```

```{toctree}
:hidden:
:caption: Reference

api/rust
api/python
```

```{toctree}
:hidden:
:caption: Development

architecture/index
contributing/index
contributing/testing
contributing/documentation
```
