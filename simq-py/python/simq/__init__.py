"""
SimQ: High-Performance Quantum Computing SDK
=============================================

SimQ is a quantum computing SDK with Python bindings built on a high-performance
Rust core. It provides a comprehensive set of tools for quantum circuit construction,
simulation, noise modeling, and hardware backend integration.

Quick Start
-----------

Create and simulate a simple Bell state::

    import simq
    
    # Build circuit
    builder = simq.CircuitBuilder(2)
    builder.h(0)
    builder.cx(0, 1)
    circuit = builder.build()
    
    # Simulate
    config = simq.SimulatorConfig(shots=1000)
    simulator = simq.Simulator(config)
    result = simulator.run(circuit)
    print(result.state_vector)

Modules
-------

- :mod:`simq.gates`: Quantum gate library (Pauli, Hadamard, CNOT, etc.)
- :mod:`simq.noise`: Noise models for realistic quantum simulations
- :mod:`simq.simulation`: High-performance quantum circuit simulator
- :mod:`simq.visualization`: Plotting and visualization utilities

Core Classes
------------

Circuit
    Immutable quantum circuit representation
CircuitBuilder
    Builder for constructing quantum circuits
Simulator
    High-performance quantum circuit simulator
HardwareNoiseModel
    Hardware noise model for realistic simulations

For more information, see https://github.com/your-org/simq
"""

from ._simq import (
    __version__,
    __author__,
    __doc__,
    # Core types
    Circuit,
    CircuitBuilder,
    QubitId,
    Parameter,
    # Exceptions
    QuantumException,
    InvalidQubitError,
    InvalidGateError,
    InvalidParameterError,
    # Compiler
    Compiler,
    OptimizationLevel,
    CircuitAnalysis,
    # Backend
    BackendResult,
    JobStatus,
    BackendType,
    LocalSimulatorConfig,
    LocalSimulatorBackend,
    # IBM Quantum (optional)
    IBMConfig,
    IBMQuantumBackend,
)

from .gates import (
    # Single-qubit gates
    HGate, XGate, YGate, ZGate,
    SGate, SdgGate, TGate, TdgGate,
    SXGate, SXdgGate,
    
    # Parameterized single-qubit gates
    RXGate, RYGate, RZGate, PhaseGate, U3Gate,
    
    # Two-qubit gates
    CXGate, CZGate, # CYGate, CHGate,
    SwapGate, iSwapGate, ECRGate,
    
    # Parameterized two-qubit gates
    RXXGate, RYYGate, RZZGate, CPhaseGate,
    
    # Three-qubit gates
    ToffoliGate, FredkinGate,
    
    # Custom gates
    CustomGate,
    
    # Factory functions
    h, x, y, z, s, sdg, t, tdg, sx, sxdg,
    rx, ry, rz, phase, u3,
    cx, cz, # cy, ch, 
    swap, iswap, ecr,
    rxx, ryy, rzz, cphase,
    toffoli, fredkin, custom,
)

from .noise import (
    DepolarizingChannel,
    AmplitudeDamping,
    PhaseDamping,
    ReadoutError,
    HardwareNoiseModel,
)

from .simulation import (
    Simulator,
    SimulatorConfig,
    SimulationResult,
)

from .visualization import (
    plot_histogram,
    plot_bloch_vector,
)

__all__ = [
    "__version__",
    "__author__",
    "__doc__",
    "Circuit",
    "CircuitBuilder",
    "QubitId",
    "Parameter",
    "QuantumException",
    "InvalidQubitError",
    "InvalidGateError",
    "InvalidParameterError",
    
    # Gates
    "HGate", "XGate", "YGate", "ZGate",
    "SGate", "SdgGate", "TGate", "TdgGate",
    "SXGate", "SXdgGate",
    "RXGate", "RYGate", "RZGate", "PhaseGate", "U3Gate",
    "CXGate", "CZGate", # "CYGate", "CHGate",
    "SwapGate", "iSwapGate", "ECRGate",
    "RXXGate", "RYYGate", "RZZGate", "CPhaseGate",
    "ToffoliGate", "FredkinGate",
    "CustomGate",
    
    # Factory functions
    "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg",
    "rx", "ry", "rz", "phase", "u3",
    "cx", "cz", # "cy", "ch", 
    "swap", "iswap", "ecr",
    "rxx", "ryy", "rzz", "cphase",
    "toffoli", "fredkin", "custom",

    # Noise
    "DepolarizingChannel",
    "AmplitudeDamping",
    "PhaseDamping",
    "ReadoutError",
    "HardwareNoiseModel",
    
    # Simulation
    "Simulator",
    "SimulatorConfig",
    "SimulationResult",
    
    # Compiler
    "Compiler",
    "OptimizationLevel",
    "CircuitAnalysis",
    
    # Backend
    "BackendResult",
    "JobStatus",
    "BackendType",
    "LocalSimulatorConfig",
    "LocalSimulatorBackend",
    "IBMConfig",
    "IBMQuantumBackend",
    
    # Visualization
    "plot_histogram",
    "plot_bloch_vector",
]


