---
myst:
  html_meta:
    description: "Testing SimQ — unit, integration, doc, and property-based tests, end-to-end suites, Python pytest, and Criterion benchmarks."
---

# Testing

SimQ leans heavily on tests — from doc tests to property-based
semantic-equivalence checks. This page shows what exists and what new code
is expected to add.

## Running the suites

```bash
# Everything (except the Python extension)
cargo test --workspace --exclude simq-py

# A single crate
cargo test -p simq-state

# A single integration test file
cargo test -p simq-sim --test vqe_qaoa_e2e

# Python bindings
cd simq-py && maturin develop && pytest
```

## Test layers

### Unit tests

Live next to the code in `#[cfg(test)] mod tests` blocks. Fast checks of a
single function or type.

### Doc tests

Every rustdoc example compiles and runs under `cargo test`. Prefer doc
tests for public API examples — they document and verify at once.

### Integration / end-to-end tests

Each crate has a `tests/` directory. Highlights:

| Suite | Verifies |
|-------|----------|
| `simq/tests/comprehensive_e2e.rs` | The fluent API end to end |
| `simq-sim/tests/kernel_correctness.rs` | Gate kernels against reference math |
| `simq-sim/tests/vqe_qaoa_e2e.rs`, `qaoa_internals.rs` | Variational workflows converge |
| `simq-sim/tests/execution_engine_internals.rs` | Adaptive executor behaviour |
| `simq-state/tests/*_e2e.rs` | Sparse/dense/COW states, measurement, observables, SIMD |
| `simq-compiler` suites | Pass semantic equivalence |

### Property-based tests

`proptest` is a workspace dependency. Use it for anything with an algebraic
invariant, e.g. "optimized circuits produce the same state vector as the
original for random circuits" or "measurement counts sum to shots".

### Numerical assertions

Use `approx` (`assert_relative_eq!`, `assert_abs_diff_eq!`) rather than
exact float comparison. Simulator seeds (`SimulatorConfig::with_seed`) make
sampling deterministic in tests.

## What new code needs

- **Bug fix** → a regression test that fails without the fix.
- **New gate** → matrix correctness, unitarity, and application tests
  against a known state.
- **Optimization pass** → semantic-equivalence property tests plus
  before/after gate-count assertions.
- **Simulator/engine change** → run `kernel_correctness` and the e2e
  suites; add cases for new configuration paths.
- **Python API change** → pytest coverage in `simq-py/tests/`.

## Benchmarks

Criterion benchmarks live in each crate's `benches/` directory:

```bash
cargo bench                # all benchmarks
cargo bench -p simq-state  # one crate
```

Python micro-benchmarks live in `simq-py/tests/benchmarks/`. For
performance PRs, include before/after Criterion output.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs formatting, clippy (with
`-D warnings`), build, and tests on every push and PR to `main`/`develop`.
The docs site is built by `.github/workflows/docs.yml`
([details](documentation.md)).
