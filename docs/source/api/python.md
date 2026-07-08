---
myst:
  html_meta:
    description: "Ferriq Python API reference — circuits, gates, simulator, noise models, compiler, backends, and visualization from the ferriq package."
---

# Python API reference

The `ferriq` package is a PyO3 extension built from the Rust core, plus thin
pure-Python layers for gates, noise, simulation, and visualization. This
page maps the public surface; docstrings on each class/function carry the
details (`help(ferriq.CircuitBuilder)`).

## Core types (`ferriq`)

| Name | Description |
|------|-------------|
| `CircuitBuilder(num_qubits)` | Mutable builder with gate methods (`h`, `x`, `cx`, `rx`, ...); `build()` returns a `Circuit` |
| `Circuit` | Immutable circuit representation |
| `QubitId` | Typed qubit index |
| `Parameter` | Symbolic circuit parameter |

### Exceptions

`QuantumException` (base), `InvalidQubitError`, `InvalidGateError`,
`InvalidParameterError`.

## Gates (`ferriq.gates`)

Gate classes and lowercase factory functions, mirroring the Rust standard
gate set:

- **Single-qubit**: `HGate`, `XGate`, `YGate`, `ZGate`, `SGate`, `SdgGate`,
  `TGate`, `TdgGate`, `SXGate`, `SXdgGate` — factories `h(q)`, `x(q)`, ...
- **Parameterized single-qubit**: `RXGate`, `RYGate`, `RZGate`,
  `PhaseGate`, `U3Gate` — factories `rx(q, theta)`, `u3(q, θ, φ, λ)`, ...
- **Two-qubit**: `CXGate`, `CZGate`, `SwapGate`, `iSwapGate`, `ECRGate` —
  factories `cx(c, t)`, `swap(a, b)`, ...
- **Parameterized two-qubit**: `RXXGate`, `RYYGate`, `RZZGate`,
  `CPhaseGate` — factories `rxx(a, b, theta)`, `cphase(c, t, theta)`, ...
- **Three-qubit**: `ToffoliGate`, `FredkinGate` — factories
  `toffoli(c1, c2, t)`, `fredkin(c, t1, t2)`
- **Custom**: `CustomGate` / `custom(qubits, matrix, name=None)` — any
  unitary from a NumPy matrix

## Simulation (`ferriq.simulation`, re-exported at top level)

| Name | Description |
|------|-------------|
| `Simulator(config=None)` | Runs circuits: `run(circuit)`, `run_with_shots(circuit, shots)` |
| `SimulatorConfig(shots=..., noise_model=..., seed=..., ...)` | Simulation options |
| `SimulationResult` | `state_vector`, probabilities, measurement counts |

## Noise (`ferriq.noise`, re-exported at top level)

`DepolarizingChannel`, `AmplitudeDamping`, `PhaseDamping`, `ReadoutError`,
and `HardwareNoiseModel` (with `add_gate_error(gate_name, channel)`). See
the [noise guide](../guide/noise.md).

## Compiler

`Compiler`, `OptimizationLevel`, `CircuitAnalysis` — run optimization
passes and inspect circuit metrics from Python.

## Backends

`LocalSimulatorBackend` / `LocalSimulatorConfig` for local execution;
`IBMConfig` / `IBMQuantumBackend` for IBM Quantum, plus `BackendResult`,
`JobStatus`, and `BackendType`.

## Visualization (`ferriq.visualization`)

| Function | Description |
|----------|-------------|
| `plot_histogram(counts, ...)` | Bar chart of measurement counts (requires matplotlib) |
| `plot_bloch_vector(vec, title="")` | Bloch-sphere rendering of a single-qubit state |

## Typing

The package ships a `py.typed` marker, so type checkers pick up the
annotations automatically.
