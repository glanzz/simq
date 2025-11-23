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

from .simulation import (
    Simulator,
    SimulatorConfig,
    SimulationResult,
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
    
    # Simulation
    "Simulator",
    "SimulatorConfig",
    "SimulationResult",
]
