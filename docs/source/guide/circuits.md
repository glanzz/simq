---
myst:
  html_meta:
    description: "Building quantum circuits in SimQ — the fluent QuantumCircuit API, const-generic and dynamic builders, validation, and circuit visualization."
---

# Building circuits

SimQ offers three ways to build circuits, from ergonomic to maximally
type-safe. All of them produce the same `Circuit` value that the simulator,
compiler, and backends consume.

## The fluent builder: `QuantumCircuit`

The recommended entry point for most code. Gate methods chain, and errors
are deferred — nothing panics mid-chain:

```rust
use simq::QuantumCircuit;

let mut qc = QuantumCircuit::new(3);
qc.h(0)
  .cnot(0, 1)
  .rz(0.35, 1)
  .toffoli(0, 1, 2);

let circuit = qc.build().unwrap(); // errors surface here
assert_eq!(circuit.len(), 4);
```

Key inspection methods before/without building:

| Method | Returns |
|--------|---------|
| `num_qubits()` | Number of qubits |
| `len()` / `is_empty()` | Gate count |
| `depth()` | Circuit depth |
| `error()` | The first recorded builder error, if any |
| `to_ascii()` | ASCII-art rendering of the circuit |
| `circuit()` | Borrow of the underlying `Circuit` |

`QuantumCircuit` can also simulate directly — see
[Simulation](simulation.md).

## The const-generic builder: `CircuitBuilder<N>`

When the number of qubits is known at compile time, `CircuitBuilder<N>`
gives you typed qubit references, so out-of-range qubits are impossible to
even name:

```rust
use simq::prelude::*;

let mut builder = CircuitBuilder::<4>::new();
let [q0, q1, q2, q3] = builder.qubits();
builder.apply_gate(Arc::new(Hadamard), &[q0]).unwrap();
builder.qubit(5).unwrap_err(); // caught immediately, not at run time
let circuit = builder.build();
```

## The dynamic builder and raw `Circuit`

`DynamicCircuitBuilder` covers the case where the qubit count is only known
at run time, and `Circuit` itself exposes the lowest-level API:

```rust
use simq::prelude::*;

let mut circuit = Circuit::new(2);
circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)])?;
circuit.add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])?;
# Ok::<(), simq::QuantumError>(())
```

## Validation

`simq-core` ships a circuit validation module that checks structural
invariants (qubit bounds, gate arity, connectivity constraints where
applicable) before execution. The builders run these checks for you; if you
construct circuits by hand, run validation explicitly — see
`simq-core/examples/circuit_validation.rs`.

## Visualization and debugging

- **ASCII rendering** — `qc.to_ascii()` or the `ascii_renderer` module
  (`simq-core/examples/ascii_circuit.rs`)
- **LaTeX rendering** — `latex_renderer` produces publication-quality
  circuit diagrams
- **Bloch sphere** — `bloch_sphere` visualizes single-qubit states
  (`simq-core/examples/bloch_sphere.rs`)
- **Circuit debugger** — step through a circuit gate by gate and inspect
  intermediate states (`simq-core/examples/circuit_debugger.rs`)

## Serialization

Circuits serialize to/from JSON via `serde` — see the `serialization`
module and `simq-core/examples/circuit_serialization.rs`.

## Parameterized circuits

Rotation gates accept `f64` angles directly. For symbolic parameters that
you bind later (parameter sweeps, VQE), `simq-core` provides `Parameter`,
`ParameterId`, and a `ParameterRegistry`; see
`simq-core/examples/parameter_tracking.rs`.
