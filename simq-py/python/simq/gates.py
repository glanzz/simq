from typing import List, Optional, Union
import numpy as np
from ._simq import gates as _rust_gates

# Single-qubit gates
HGate = _rust_gates.HGate
XGate = _rust_gates.XGate
YGate = _rust_gates.YGate
ZGate = _rust_gates.ZGate
SGate = _rust_gates.SGate
SdgGate = _rust_gates.SdgGate
TGate = _rust_gates.TGate
TdgGate = _rust_gates.TdgGate
SXGate = _rust_gates.SXGate
SXdgGate = _rust_gates.SXdgGate

# Parameterized single-qubit gates
RXGate = _rust_gates.RXGate
RYGate = _rust_gates.RYGate
RZGate = _rust_gates.RZGate
PhaseGate = _rust_gates.PhaseGate
U3Gate = _rust_gates.U3Gate

# Two-qubit gates
CXGate = _rust_gates.CXGate
CZGate = _rust_gates.CZGate
# CYGate = _rust_gates.CYGate
# CHGate = _rust_gates.CHGate
SwapGate = _rust_gates.SwapGate
iSwapGate = _rust_gates.iSwapGate
ECRGate = _rust_gates.ECRGate

# Parameterized two-qubit gates
RXXGate = _rust_gates.RXXGate
RYYGate = _rust_gates.RYYGate
RZZGate = _rust_gates.RZZGate
CPhaseGate = _rust_gates.CPhaseGate

# Three-qubit gates
ToffoliGate = _rust_gates.ToffoliGate
FredkinGate = _rust_gates.FredkinGate

# Custom gates
CustomGate = _rust_gates.CustomGate

def h() -> HGate:
    """Create a Hadamard gate."""
    return HGate()

def x() -> XGate:
    """Create a Pauli-X gate."""
    return XGate()

def y() -> YGate:
    """Create a Pauli-Y gate."""
    return YGate()

def z() -> ZGate:
    """Create a Pauli-Z gate."""
    return ZGate()

def s() -> SGate:
    """Create an S gate."""
    return SGate()

def sdg() -> SdgGate:
    """Create an S-dagger gate."""
    return SdgGate()

def t() -> TGate:
    """Create a T gate."""
    return TGate()

def tdg() -> TdgGate:
    """Create a T-dagger gate."""
    return TdgGate()

def sx() -> SXGate:
    """Create a square-root X gate."""
    return SXGate()

def sxdg() -> SXdgGate:
    """Create an inverse square-root X gate."""
    return SXdgGate()

def rx(theta: float) -> RXGate:
    """Create a rotation-X gate."""
    return RXGate(theta)

def ry(theta: float) -> RYGate:
    """Create a rotation-Y gate."""
    return RYGate(theta)

def rz(theta: float) -> RZGate:
    """Create a rotation-Z gate."""
    return RZGate(theta)

def phase(theta: float) -> PhaseGate:
    """Create a phase gate."""
    return PhaseGate(theta)

def u3(theta: float, phi: float, lambda_: float) -> U3Gate:
    """Create a U3 gate."""
    return U3Gate(theta, phi, lambda_)

def cx() -> CXGate:
    """Create a controlled-X (CNOT) gate."""
    return CXGate()

def cz() -> CZGate:
    """Create a controlled-Z gate."""
    return CZGate()

# def cy() -> CYGate:
#     """Create a controlled-Y gate."""
#     return CYGate()

# def ch() -> CHGate:
#     """Create a controlled-Hadamard gate."""
#     return CHGate()

def swap() -> SwapGate:
    """Create a SWAP gate."""
    return SwapGate()

def iswap() -> iSwapGate:
    """Create an iSWAP gate."""
    return iSwapGate()

def ecr() -> ECRGate:
    """Create an ECR gate."""
    return ECRGate()

def rxx(theta: float) -> RXXGate:
    """Create a rotation-XX gate."""
    return RXXGate(theta)

def ryy(theta: float) -> RYYGate:
    """Create a rotation-YY gate."""
    return RYYGate(theta)

def rzz(theta: float) -> RZZGate:
    """Create a rotation-ZZ gate."""
    return RZZGate(theta)

def cphase(theta: float) -> CPhaseGate:
    """Create a controlled-phase gate."""
    return CPhaseGate(theta)

def toffoli() -> ToffoliGate:
    """Create a Toffoli (CCNOT) gate."""
    return ToffoliGate()

def fredkin() -> FredkinGate:
    """Create a Fredkin (CSWAP) gate."""
    return FredkinGate()

def custom(name: str, matrix: Union[List[List[complex]], np.ndarray]) -> CustomGate:
    """Create a custom gate from a unitary matrix.
    
    Args:
        name: Name of the gate
        matrix: Unitary matrix as a list of lists or numpy array
        
    Returns:
        CustomGate: The custom gate object
    """
    if isinstance(matrix, np.ndarray):
        matrix = matrix.tolist()
    return CustomGate(name, matrix)
