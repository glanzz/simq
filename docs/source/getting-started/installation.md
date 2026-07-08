---
myst:
  html_meta:
    description: "Install the Ferriq quantum computing SDK — add the Rust crate to Cargo.toml or build the Python bindings with maturin."
---

# Installation

Ferriq ships as a Rust workspace with an optional Python extension. Pick the
track that matches how you want to use it.

## Rust

Add the umbrella crate to your `Cargo.toml` as a git dependency:

```toml
[dependencies]
ferriq = { git = "https://github.com/glanzz/ferriq" }
```

```{note}
Ferriq isn't published to crates.io quite yet, so the git dependency is
the supported install path until `cargo add ferriq` lands. (Ferriq was
briefly known as SimQ; it was renamed because the `simq` name on
crates.io and PyPI belongs to unrelated projects.)
```

The `ferriq` crate re-exports every subcrate (`ferriq-core`, `ferriq-gates`,
`ferriq-sim`, `ferriq-state`, `ferriq-compiler`, `ferriq-backend`), so a single
dependency is all you need.

### Requirements

- Rust **1.75** or newer (`rustup update` to get the latest stable)
- No system dependencies — everything builds with `cargo`

### Building from source

```bash
git clone https://github.com/glanzz/ferriq.git
cd ferriq
cargo build --workspace --exclude ferriq-py
cargo test  --workspace --exclude ferriq-py
```

```{note}
`ferriq-py` is a PyO3 *extension module* — it can only be linked when loaded
by a Python interpreter, so it is excluded from plain `cargo build`/`cargo
test` runs and built separately with maturin (see below).
```

## Python

The Python bindings live in the [`ferriq-py`](https://github.com/glanzz/ferriq/tree/main/ferriq-py)
crate and are built with [maturin](https://www.maturin.rs/):

```bash
git clone https://github.com/glanzz/ferriq.git
cd ferriq/ferriq-py

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Build and install the extension in development mode
pip install maturin
maturin develop --release
```

Verify the install:

```python
import ferriq
print(ferriq.__version__)
```

### Development install

For working on the bindings themselves (tests, benchmarks, docs tooling):

```bash
cd ferriq-py
pip install -e ".[dev]"
maturin develop
pytest
```

## Next steps

- [Quickstart (Rust)](quickstart-rust.md) — your first circuit in Rust
- [Quickstart (Python)](quickstart-python.md) — your first circuit in Python
