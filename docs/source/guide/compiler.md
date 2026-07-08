---
myst:
  html_meta:
    description: "Ferriq's circuit compiler — gate fusion, commutation, dead-code elimination, template matching, and optimization pipelines for quantum circuits."
---

# Circuit compiler

`ferriq-compiler` transforms circuits into more efficient, semantically
equivalent forms. The simulator can invoke it automatically
(`SimulatorConfig::with_optimization_level`), or you can run passes
yourself for full control.

## Optimization passes

| Pass | What it does |
|------|--------------|
| **Gate fusion** | Combines adjacent single-qubit gates on the same qubit into one composite gate (one matrix multiply instead of several) |
| **Gate commutation** | Reorders commuting gates to expose further fusion and cancellation opportunities |
| **Dead code elimination** | Removes gates that provably cannot affect measurement outcomes |
| **Template substitution** | Rewrites known gate patterns into cheaper equivalents |
| **Advanced template matching** | Larger pattern database with cost-driven matching |

## Running a single pass

Gate fusion, for example:

```rust
use ferriq_compiler::fusion::fuse_single_qubit_gates;
use ferriq_core::{Circuit, Gate, QubitId};
use ferriq_gates::standard::{Hadamard, PauliX, TGate};
use std::sync::Arc;

let mut circuit = Circuit::new(1);
let q0 = QubitId::new(0);
circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();
circuit.add_gate(Arc::new(TGate) as Arc<dyn Gate>, &[q0]).unwrap();
circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[q0]).unwrap();

let optimized = fuse_single_qubit_gates(&circuit, None).unwrap();
assert!(optimized.len() < circuit.len()); // 3 gates -> 1 fused gate
```

The math: sequential unitaries compose by matrix multiplication,
`U₃(U₂(U₁|ψ⟩)) = (U₃·U₂·U₁)|ψ⟩`, so the fused gate's matrix is the product
of the individual matrices (rightmost applied first).

## Pipelines

Passes compose into pipelines that run until a fixed point or a
configurable iteration budget. See these runnable demos in
`ferriq-compiler/examples/`:

| Example | Shows |
|---------|-------|
| `pipeline_demo.rs` | Basic pass pipeline |
| `advanced_pipeline_demo.rs` | Multi-pass pipelines with cost models |
| `gate_fusion.rs` | Fusion in isolation |
| `commutation_demo.rs` | Commutation analysis |
| `template_matching_demo.rs` | Template-based rewrites |
| `gate_decomposition.rs` | Decomposing multi-qubit gates |
| `circuit_analysis.rs` | Depth/width/gate-count analysis |
| `lazy_evaluation.rs`, `caching_demo.rs` | Lazy pass evaluation and result caching |
| `execution_plan_demo.rs` | Building execution plans for the simulator |

Run any of them with:

```bash
cargo run -p ferriq-compiler --example pipeline_demo
```

## Writing your own pass

A pass implements the `OptimizationPass` trait (take a circuit, return a
transformed circuit plus a "changed" indicator). Look at
`DeadCodeElimination` or `GateCommutation` in `ferriq-compiler/src/` as
templates, and add property tests that check semantic equivalence — the
existing test suites show the pattern (e.g. comparing state vectors before
and after optimization on random circuits).

More detail: [`ferriq-compiler/README.md`](https://github.com/glanzz/ferriq/blob/main/ferriq-compiler/README.md).
