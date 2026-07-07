---
myst:
  html_meta:
    description: "Simulating quantum circuits with SimQ — SimulatorConfig presets, sparse vs dense state vectors, shots, seeding, statistics, and memory limits."
---

# Simulation

`simq-sim` is SimQ's high-performance simulator. It picks sparse or dense
state representations adaptively, parallelizes across cores, and can
optimize circuits before running them.

## Quick paths

Through the fluent builder:

```rust
use simq::QuantumCircuit;

let mut qc = QuantumCircuit::new(2);
qc.h(0).cnot(0, 1);

let exact = qc.simulate().unwrap();                 // final state, no sampling
let shots = qc.simulate_with_shots(1024).unwrap();  // adds measurement counts
let probs = qc.probabilities().unwrap();            // exact probabilities
```

Or explicitly, with a configured simulator:

```rust
use simq::prelude::*;

let sim = Simulator::new(SimulatorConfig::default());
let result = sim.run(&circuit)?;
# Ok::<(), SimulatorError>(())
```

## `SimulatorConfig`

Configuration is a builder with sensible presets:

```rust
use simq::SimulatorConfig;

let config = SimulatorConfig::new()
    .with_shots(4096)
    .with_optimization_level(2)
    .with_seed(42)              // reproducible sampling
    .with_statistics(true)
    .with_memory_limit(8 * 1024 * 1024 * 1024); // 8 GiB
```

### Presets

| Preset | Intent |
|--------|--------|
| `SimulatorConfig::new()` / `default()` | Balanced defaults |
| `SimulatorConfig::fast()` | Maximum speed; favours aggressive optimization |
| `SimulatorConfig::accurate()` | Favours numerical accuracy |
| `SimulatorConfig::debug()` | Deterministic, statistics-heavy runs for debugging |

### Notable options

- **`sparse_threshold`** — controls when the simulator switches between
  sparse and dense state representations. Sparse states keep memory
  proportional to the number of nonzero amplitudes, which is what makes
  35–40 qubit simulations feasible on 32 GB RAM.
- **`parallel_threshold`** — minimum state size before work is fanned out
  across threads (via rayon).
- **`optimize_circuit` / `optimization_level`** — run
  [compiler passes](compiler.md) automatically before execution (levels
  0–3).
- **`seed`** — fixes the RNG for reproducible measurement sampling.
- **`collect_statistics`** — gathers execution statistics (gate timings,
  representation switches) in the result.
- **`memory_limit`** — hard cap; the simulator fails cleanly instead of
  swapping.

Call `config.validate()` to check a configuration before use.

## Results

`SimulationResult` carries:

- the final state (`num_qubits()`, state vector access),
- `measurements: Option<MeasurementCounts>` when shots were requested —
  with `get("0101")`, `sorted()`, and iteration helpers,
- optional execution statistics.

## Execution engine internals

For contributors: the `execution_engine` module contains the adaptive
executor (`adaptive.rs`), SIMD kernels (`kernels/`), parallel scheduling
(`parallel.rs`), checkpointing and recovery, result caching, and telemetry.
The [architecture overview](../architecture/index.md) describes how these
fit together, and `simq-sim/tests/execution_engine_internals.rs` exercises
them directly.

## GPU

`simq-sim` includes experimental GPU support (`gpu.rs`, WGSL shaders under
`shaders/`) behind the `use_gpu` config flag. Treat it as experimental —
the CPU path is the reference implementation.
