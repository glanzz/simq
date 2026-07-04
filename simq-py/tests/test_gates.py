import pytest
import numpy as np
import simq
from simq import (
    HGate, XGate, YGate, ZGate,
    SGate, SdgGate, TGate, TdgGate,
    SXGate, SXdgGate,
    RXGate, RYGate, RZGate, PhaseGate, U3Gate,
    CXGate, CZGate, # CYGate, CHGate,
    SwapGate, iSwapGate, ECRGate,
    RXXGate, RYYGate, RZZGate, CPhaseGate,
    ToffoliGate, FredkinGate,
    CustomGate,
)

def test_single_qubit_gates():
    gates = [
        HGate(), XGate(), YGate(), ZGate(),
        SGate(), SdgGate(), TGate(), TdgGate(),
        SXGate(), SXdgGate()
    ]
    for gate in gates:
        assert gate.num_qubits() == 1
        assert isinstance(gate.name(), str)
        assert len(gate.matrix()) == 2
        assert len(gate.matrix()[0]) == 2

def test_parameterized_single_qubit_gates():
    theta = np.pi / 2
    gates = [
        RXGate(theta), RYGate(theta), RZGate(theta), PhaseGate(theta)
    ]
    for gate in gates:
        assert gate.num_qubits() == 1
        assert isinstance(gate.name(), str)
        assert len(gate.matrix()) == 2

    u3 = U3Gate(theta, theta, theta)
    assert u3.num_qubits() == 1
    assert len(u3.matrix()) == 2

def test_two_qubit_gates():
    gates = [
        CXGate(), CZGate(), # CYGate(), CHGate(),
        SwapGate(), iSwapGate(), ECRGate()
    ]
    for gate in gates:
        assert gate.num_qubits() == 2
        assert isinstance(gate.name(), str)
        assert len(gate.matrix()) == 4
        assert len(gate.matrix()[0]) == 4

def test_parameterized_two_qubit_gates():
    theta = np.pi / 2
    gates = [
        RXXGate(theta), RYYGate(theta), RZZGate(theta), CPhaseGate(theta)
    ]
    for gate in gates:
        assert gate.num_qubits() == 2
        assert isinstance(gate.name(), str)
        assert len(gate.matrix()) == 4

def test_three_qubit_gates():
    gates = [
        ToffoliGate(), FredkinGate()
    ]
    for gate in gates:
        assert gate.num_qubits() == 3
        assert isinstance(gate.name(), str)
        assert len(gate.matrix()) == 8
        assert len(gate.matrix()[0]) == 8

def test_custom_gate():
    matrix = [[1, 0], [0, 1]]
    gate = CustomGate("Identity", matrix)
    assert gate.num_qubits() == 1
    assert gate.name() == "Identity"
    assert gate.matrix() == [[1+0j, 0+0j], [0+0j, 1+0j]]

    # Test invalid matrix
    with pytest.raises(ValueError):
        CustomGate("Invalid", [[1, 0]])  # Not square

    with pytest.raises(ValueError):
        CustomGate("Invalid", [[1, 0], [0, 1], [0, 0]]) # Not power of 2
