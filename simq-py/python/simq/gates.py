"""
SimQ Gates Module
=================

This module provides a comprehensive library of quantum gates for the SimQ framework.
All gates are available both as class-based constructors and convenience factory functions.

Gate Categories
---------------

Single-Qubit Gates
    Basic gates: H, X, Y, Z
    Phase gates: S, T, and their adjoints
    SX gate and its adjoint
    
Parameterized Single-Qubit Gates
    Rotation gates: RX, RY, RZ
    Phase gate: Phase
    Universal gate: U3
    
Two-Qubit Gates
    CNOT (CX), CZ
    SWAP, iSWAP, ECR (echoed cross-resonance)
    
Parameterized Two-Qubit Gates
    RXX, RYY, RZZ (two-qubit rotations)
    Controlled phase: CPhase
    
Three-Qubit Gates
    Toffoli (CCNOT)
    Fredkin (CSWAP)
    
Custom Gates
    User-defined gates with custom unitary matrices

Examples
--------

Using factory functions::

    import simq
    
    builder = simq.CircuitBuilder(2)
    builder.h(0)              # Hadamard on qubit 0
    builder.rx(0, np.pi/4)    # RX rotation
    builder.cx(0, 1)          # CNOT
    
Using gate classes::

    from simq.gates import HGate, CNOTGate
    
    h = HGate()
    cnot = CXGate(control=0, target=1)
"""

from .._simq import (
    # Single-qubit gates
    HGate, XGate, YGate, ZGate,
    SGate, SdgGate, TGate, TdgGate,
    SXGate, SXdgGate,
    
    # Parameterized single-qubit gates
    RXGate, RYGate, RZGate, PhaseGate, U3Gate,
    
    # Two-qubit gates
    CXGate, CZGate,
    SwapGate, iSwapGate, ECRGate,
    
    # Parameterized two-qubit gates
    RXXGate, RYYGate, RZZGate, CPhaseGate,
    
    # Three-qubit gates
    ToffoliGate, FredkinGate,
    
    # Custom gates
    CustomGate,
)

# Factory functions for convenience
def h(qubit):
    """Apply Hadamard gate to qubit."""
    return HGate()

def x(qubit):
    """Apply Pauli-X gate to qubit."""
    return XGate()

def y(qubit):
    """Apply Pauli-Y gate to qubit."""
    return YGate()

def z(qubit):
    """Apply Pauli-Z gate to qubit."""
    return ZGate()

def s(qubit):
    """Apply S gate (phase π/2) to qubit."""
    return SGate()

def sdg(qubit):
    """Apply S-dagger gate to qubit."""
    return SdgGate()

def t(qubit):
    """Apply T gate (phase π/4) to qubit."""
    return TGate()

def tdg(qubit):
    """Apply T-dagger gate to qubit."""
    return TdgGate()

def sx(qubit):
    """Apply √X gate to qubit."""
    return SXGate()

def sxdg(qubit):
    """Apply √X-dagger gate to qubit."""
    return SXdgGate()

def rx(qubit, theta):
    """Apply RX rotation gate to qubit."""
    return RXGate(theta=theta)

def ry(qubit, theta):
    """Apply RY rotation gate to qubit."""
    return RYGate(theta=theta)

def rz(qubit, theta):
    """Apply RZ rotation gate to qubit."""
    return RZGate(theta=theta)

def phase(qubit, theta):
    """Apply phase gate to qubit."""
    return PhaseGate(theta=theta)

def u3(qubit, theta, phi, lambda_):
    """Apply universal single-qubit gate."""
    return U3Gate(theta=theta, phi=phi, lambda_=lambda_)

def cx(control, target):
    """Apply CNOT (controlled-X) gate."""
    return CXGate()

def cz(control, target):
    """Apply CZ (controlled-Z) gate."""
    return CZGate()

def swap(qubit1, qubit2):
    """Apply SWAP gate."""
    return SwapGate()

def iswap(qubit1, qubit2):
    """Apply iSWAP gate."""
    return iSwapGate()

def ecr(qubit1, qubit2):
    """Apply ECR (echoed cross-resonance) gate."""
    return ECRGate()

def rxx(qubit1, qubit2, theta):
    """Apply RXX rotation gate."""
    return RXXGate(theta=theta)

def ryy(qubit1, qubit2, theta):
    """Apply RYY rotation gate."""
    return RYYGate(theta=theta)

def rzz(qubit1, qubit2, theta):
    """Apply RZZ rotation gate."""
    return RZZGate(theta=theta)

def cphase(control, target, theta):
    """Apply controlled phase gate."""
    return CPhaseGate(theta=theta)

def toffoli(control1, control2, target):
    """Apply Toffoli (CCNOT) gate."""
    return ToffoliGate()

def fredkin(control, target1, target2):
    """Apply Fredkin (CSWAP) gate."""
    return FredkinGate()

def custom(qubits, matrix, name=None):
    """Create custom gate with given unitary matrix."""
    return CustomGate(matrix=matrix, name=name)

__all__ = [
    # Gate classes
    "HGate", "XGate", "YGate", "ZGate",
    "SGate", "SdgGate", "TGate", "TdgGate",
    "SXGate", "SXdgGate",
    "RXGate", "RYGate", "RZGate", "PhaseGate", "U3Gate",
    "CXGate", "CZGate",
    "SwapGate", "iSwapGate", "ECRGate",
    "RXXGate", "RYYGate", "RZZGate", "CPhaseGate",
    "ToffoliGate", "FredkinGate",
    "CustomGate",
    
    # Factory functions
    "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg",
    "rx", "ry", "rz", "phase", "u3",
    "cx", "cz",
    "swap", "iswap", "ecr",
    "rxx", "ryy", "rzz", "cphase",
    "toffoli", "fredkin", "custom",
]
