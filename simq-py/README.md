# simq

Python bindings for [SimQ](https://github.com/glanzz/simq), a high performance
quantum computing simulator written in Rust.

## Installation

```bash
pip install maturin
maturin develop --release
```

## Quick start

```python
import simq

builder = simq.CircuitBuilder(2)
builder.h(0)
builder.cx(0, 1)
circuit = builder.build()

simulator = simq.Simulator()
result = simulator.run(circuit)
print(result.probabilities)
```

## Features

- Circuit construction with standard, parameterized, and custom gates
- Local simulation (dense and sparse state vectors) and IBM Quantum backend support
- Noise channels and hardware noise models
- Circuit compilation and optimization passes
- Circuit visualization (ASCII, LaTeX, Bloch sphere, histograms)

## Development

```bash
pip install -e ".[dev]"
maturin develop
pytest
```

## License

MIT OR Apache-2.0
