---
myst:
  html_meta:
    description: "Install the SimQ quantum computing SDK: add the Rust crate to Cargo.toml or build the Python bindings with maturin."
---

# Installation

SimQ ships as a Rust workspace with an optional Python extension. Pick the
track that matches how you want to use it.

## Rust

Add the umbrella crate to your `Cargo.toml`:

```toml
[dependencies]
simq = "0.1"
```

The `simq` crate re-exports every subcrate (`simq-core`, `simq-gates`,
`simq-sim`, `simq-state`, `simq-compiler`, `simq-backend`), so a single
dependency is all you need.

### Requirements

- Rust **1.75** or newer (`rustup update` to get the latest stable)
- No system dependencies: everything builds with `cargo`

### Building from source

```bash
git clone https://github.com/glanzz/simq.git
cd simq
cargo build --workspace --exclude simq-py
cargo test  --workspace --exclude simq-py
```

```{note}
`simq-py` is a PyO3 *extension module*; it can only be linked when loaded
by a Python interpreter, so it is excluded from plain `cargo build`/`cargo
test` runs and built separately with maturin (see below).
```

## Python

The Python bindings live in the [`simq-py`](https://github.com/glanzz/simq/tree/main/simq-py)
crate and are built with [maturin](https://www.maturin.rs/):

```bash
git clone https://github.com/glanzz/simq.git
cd simq/simq-py

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Build and install the extension in development mode
pip install maturin
maturin develop --release
```

Verify the install:

```python
import simq
print(simq.__version__)
```

### Development install

For working on the bindings themselves (tests, benchmarks, docs tooling):

```bash
cd simq-py
pip install -e ".[dev]"
maturin develop
pytest
```

## Next steps

- [Quickstart (Rust)](quickstart-rust.md): your first circuit in Rust
- [Quickstart (Python)](quickstart-python.md): your first circuit in Python
