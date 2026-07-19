---
myst:
  html_meta:
    description: "Running SimQ circuits on real hardware: the QuantumBackend trait, IBM Quantum backend, transpiler, routing, and backend selection."
---

# Hardware backends

`simq-backend` abstracts over execution targets so the same circuit runs on
the local simulator or a real quantum computer.

## The `QuantumBackend` trait

Every target implements `QuantumBackend` (with an `AsyncQuantumBackend`
extension for job-queue style providers). A backend advertises its
`capabilities()` (native gate set, qubit count, connectivity) and
executes circuits, returning device-format results.

Implementations in-tree:

| Backend | Module | Notes |
|---------|--------|-------|
| Local simulator | `local_simulator` | Wraps `simq-sim`; the default |
| IBM Quantum | `ibm_quantum` | Submits jobs to IBM Quantum services |

## IBM Quantum

```rust
use simq_backend::ibm_quantum::{IBMConfig, IBMQuantumBackend};

let config = IBMConfig::new(std::env::var("IBM_API_TOKEN")?);
let backend = IBMQuantumBackend::new(config, "ibm_brisbane")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

```{warning}
Keep API tokens out of source code: read them from the environment or a
credentials file.
```

## Transpilation

Real devices support a limited native gate set and connectivity. The
`Transpiler` rewrites a logical circuit to fit:

1. **Gate decomposition** (`gate_decomposition`, `DecompositionRules`):
   breaks unsupported gates into native ones
2. **Qubit mapping & routing** (`routing`, `QubitMapping`, `SwapStrategy`):
   places logical qubits on physical ones and inserts SWAPs where the
   coupling map requires
3. **Optimization** (`OptimizationLevel`): re-runs
   [compiler passes](compiler.md) on the transpiled circuit

`TranspilationCost` reports the overhead (added gates/depth) so you can
compare strategies.

## Backend selection

`backend_selector` picks the best available backend for a circuit given its
capability requirements, useful when the same program should use a
simulator locally and hardware in production.

## From Python

The Python bindings expose backend support through `simq`'s compiled core
(see `simq-py/src/backend/`), including IBM Quantum access. The interface
mirrors the Rust API.
