---
myst:
  html_meta:
    description: "Ferriq Python quickstart — build and simulate quantum circuits from Python using the Rust-powered ferriq package."
---

# Quickstart (Python)

The `ferriq` Python package wraps the Rust core with a familiar,
Qiskit-flavoured API. Install it first (see
[Installation](installation.md#python)), then dive in.

## Your first circuit

```python
import ferriq

# Build a Bell-state circuit
builder = ferriq.CircuitBuilder(2)
builder.h(0)
builder.cx(0, 1)
circuit = builder.build()

# Simulate
config = ferriq.SimulatorConfig(shots=1000)
simulator = ferriq.Simulator(config)
result = simulator.run(circuit)

print(f"State vector: {result.state_vector}")

counts = simulator.run_with_shots(circuit, shots=1024)
print(f"Measurement counts: {counts}")
```

## Parameterized circuits

Rotation gates take angles directly, so parameter sweeps are ordinary
Python loops:

```python
import numpy as np
import ferriq

builder = ferriq.CircuitBuilder(2)
builder.rx(0, theta=np.pi / 4)
builder.ry(1, theta=np.pi / 2)
builder.cx(0, 1)
circuit = builder.build()
```

## Noise simulation

Attach a hardware noise model to make simulations realistic:

```python
import ferriq

noise_model = ferriq.HardwareNoiseModel()
noise_model.add_gate_error("cx", ferriq.DepolarizingChannel(0.01))

config = ferriq.SimulatorConfig(noise_model=noise_model, shots=1000)
simulator = ferriq.Simulator(config)
result = simulator.run(circuit)
```

See the [noise guide](../guide/noise.md) for the full set of channels
(depolarizing, amplitude damping, phase damping, ...).

## Visualization

The `ferriq.visualization` module provides plotting helpers for measurement
histograms and states, and circuits can be rendered as ASCII or LaTeX
through the core bindings.

## Complete example programs

The repository ships runnable Python examples in
[`ferriq-py/examples/`](https://github.com/glanzz/ferriq/tree/main/ferriq-py/examples):

| Script | What it shows |
|--------|---------------|
| `00_getting_started.py` | Minimal end-to-end tour |
| `basic_circuit.py` | Circuit construction and simulation basics |
| `parameterized_circuit.py` | Parameter binding and sweeps |
| `noise_simulation.py` | Noise channels and hardware noise models |
| `vqe_example.py` | A complete VQE optimization loop |

Run any of them (after `maturin develop`):

```bash
cd ferriq-py
python examples/00_getting_started.py
```

## Where to next?

- [Python API reference](../api/python.md)
- [Noise models](../guide/noise.md)
- [Examples](../examples/index.md)
