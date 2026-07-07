---
myst:
  html_meta:
    description: "Contribute to SimQ — development environment setup, workspace layout, coding standards, commit conventions, and the pull-request workflow."
---

# Contributing to SimQ

Contributions are welcome — bug reports, docs, examples, optimization
passes, new gates, backends, you name it. This guide gets you from clone to
merged PR.

## Development setup

### Prerequisites

- **Rust 1.75+** — install via [rustup](https://rustup.rs/), then
  `rustup component add rustfmt clippy`
- **Python 3.9+** and [maturin](https://www.maturin.rs/) — only if you work
  on the Python bindings
- No other system dependencies

### Clone and build

```bash
git clone https://github.com/glanzz/simq.git
cd simq

# Build everything except the Python extension
cargo build --workspace --exclude simq-py

# Run the test suite
cargo test --workspace --exclude simq-py
```

```{note}
`simq-py` is a PyO3 extension module and cannot be linked by a plain
`cargo build` — build it with maturin instead (below).
```

### Python bindings setup

```bash
cd simq-py
pip install -e ".[dev]"
maturin develop
pytest
```

## Making changes

1. **Fork and branch** — create a feature branch off `main` with a
   descriptive name (`gate_fusion_fix`, `docs-vqe-guide`, ...).
2. **Find the right crate** — the
   [architecture overview](../architecture/index.md) maps responsibilities.
   Types shared by several crates belong in `simq-core`.
3. **Write tests first (or at least alongside)** — see
   [Testing](testing.md). Every crate has unit tests plus `tests/`
   integration suites; optimization passes need semantic-equivalence tests.
4. **Document as you go** — public items need rustdoc comments, ideally
   with runnable examples (they're compiled by `cargo test`). User-facing
   features should also update this documentation site
   ([how](documentation.md)).

## Before you push

CI runs these exact checks — run them locally first:

```bash
cargo fmt -- --check                                             # formatting
cargo clippy --all-targets --all-features -- -D warnings         # lints
cargo build --verbose --all-features --workspace --exclude simq-py
cargo test  --workspace --exclude simq-py
```

Formatting is governed by the checked-in `rustfmt.toml`, lints by
`clippy.toml` — don't hand-tune around them.

For Python changes:

```bash
cd simq-py && maturin develop && pytest
```

## Pull requests

- Keep PRs focused — one logical change per PR.
- Write a clear description: what, why, and how it was tested.
- Benchmarks (`cargo bench`, Criterion) are appreciated for
  performance-sensitive changes — include before/after numbers.
- CI must be green (fmt, clippy, build, tests) before review.

## Performance-sensitive code

SimQ's value proposition is speed, so:

- Avoid allocation in hot loops (`smallvec` is available workspace-wide).
- Preserve SIMD paths — check `simq-state`/`simq-gates` kernels before
  touching state application code.
- Never regress the compile-time caching guarantees
  ([details](../guide/gates-caching.md)) — cached matrices must remain
  exact-match only.
- Run the relevant Criterion benchmarks and include results in the PR.

## License

By contributing you agree that your work is dual-licensed under
[MIT](https://github.com/glanzz/simq/blob/main/LICENSE-MIT) and
[Apache-2.0](https://github.com/glanzz/simq/blob/main/LICENSE-APACHE),
like the rest of the project.
