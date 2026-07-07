---
myst:
  html_meta:
    description: "SimQ's gate library and compile-time gate matrix caching — multi-level caches give rotation gates ~0-5 ns matrix access with exact accuracy."
---

# Gates & compile-time caching

`simq-gates` implements the standard gate library with SIMD-optimized
kernels and a multi-level **compile-time gate matrix caching** system for
rotation gates.

## The gate library

Every standard gate is a zero-sized (or angle-carrying) struct implementing
the `Gate` trait from `simq-core`:

```rust
use simq::prelude::*; // pulls in simq_gates::standard::*

let mut circuit = Circuit::new(2);
circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)])?;
circuit.add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])?;
# Ok::<(), simq::QuantumError>(())
```

Custom gates are fully supported — implement the `Gate` trait and pass an
`Arc` of your type anywhere a standard gate goes. The complete walkthrough
is in
[`simq-gates/CUSTOM_GATES_GUIDE.md`](https://github.com/glanzz/simq/blob/main/simq-gates/CUSTOM_GATES_GUIDE.md)
and `simq-gates/examples/custom_gates_tutorial.rs`.

## Compile-time matrix caching

Rotation gates (RX, RY, RZ) dominate variational workloads, and computing a
2×2 unitary from `sin`/`cos` on every call is wasted work for the angles
that appear over and over. SimQ pre-computes matrices for common angles *at
compile time* and embeds them in the binary:

| Cache level | Coverage | Access time | Memory |
|-------------|----------|-------------|--------|
| 1 | Common angles (π/4, π/2, π) | ~0 ns | 0 bytes |
| 2 | Clifford+T (π/8, π/16, π/32) | ~1 ns | ~1 KB |
| 3 | π fractions (π/3, π/5, π/6, ...) | ~1 ns | ~2 KB |
| 4 | VQE range (0 to π/4, 256 steps) | ~2–5 ns | 48 KB |
| 5 | QAOA range (0 to π, 100 steps) | ~2–5 ns | 19 KB |
| 6 | Runtime compute (any angle) | ~20–50 ns | 0 bytes |

Total static memory: **~70 KB**, embedded in the binary.

```rust
use simq_gates::RotationX;
use std::f64::consts::PI;

// Automatically uses the optimal caching strategy
let rx1 = RotationX::new(PI / 4.0); // ~0 ns   (common-angle cache)
let rx2 = RotationX::new(0.1);      // ~2-5 ns (VQE range cache)
let rx3 = RotationX::new(10.0);     // ~20-50 ns (runtime fallback)
```

```{important}
**Accuracy guarantee** — every cache level is *exact-match only*: a cached
matrix is returned only when the requested angle equals the cached angle to
within 1e-12. Any other angle falls through to full-precision runtime
computation. Gate matrices are never approximated or snapped to a grid.
```

The full design document is
[`simq-gates/COMPILE_TIME_CACHING.md`](https://github.com/glanzz/simq/blob/main/simq-gates/COMPILE_TIME_CACHING.md),
and `simq-gates/examples/lookup_table_demo.rs` demonstrates the lookup
tables directly.

## Procedural macros

`simq-macros` generates the compile-time lookup tables and other
boilerplate. It is an implementation detail — you should never need to
depend on it directly.
