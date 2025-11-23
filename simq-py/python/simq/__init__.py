"""
SimQ: High-Performance Quantum Computing SDK

SimQ is a Rust-based quantum computing framework with Python bindings,
designed for high-performance quantum circuit simulation and execution.

Quick Start
-----------

Create and simulate a simple quantum circuit:

    >>> import simq
    >>>
    >>> # Create a 2-qubit Bell state
    >>> builder = simq.CircuitBuilder(2)
    >>> builder.h(0)
    >>> builder.cx(0, 1)
    >>> circuit = builder.build()
    >>>
    >>> # Simulate the circuit
    >>> simulator = simq.Simulator()
    >>> result = simulator.run(circuit)
    >>> print(result.probabilities)

Features
--------

- **High Performance**: Rust-powered quantum simulation with minimal Python overhead
- **Easy to Use**: Pythonic API for circuit building and execution
- **Flexible**: Support for parameterized circuits and custom gates
- **Realistic**: Comprehensive noise models for accurate simulation
- **Extensible**: Plugin architecture for custom backends and gates

Modules
-------

circuit
    Circuit building and manipulation
gates
    Standard quantum gate library
simulation
    Quantum circuit simulation
noise
    Noise models for realistic simulation
backend
    Hardware backend abstraction
visualization
    Circuit and state visualization

"""

from typing import TYPE_CHECKING

# Import the Rust extension module
try:
    from simq._simq import (
        __version__,
        __author__,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import the SimQ Rust extension. "
        "Make sure the package is properly installed. "
        f"Error: {e}"
    ) from e

# Version information
__all__ = [
    "__version__",
    "__author__",
]

# Type checking imports (will be populated in Phase 1+)
if TYPE_CHECKING:
    pass
    # from simq._simq import (
    #     Circuit,
    #     CircuitBuilder,
    #     Simulator,
    #     SimulatorConfig,
    #     SimulationResult,
    #     # ... more types as we implement them
    # )
