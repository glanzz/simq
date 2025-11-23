"""Tests for Circuit and CircuitBuilder"""

import pytest
import simq


class TestCircuitBuilder:
    """Tests for CircuitBuilder"""

    def test_create_empty_circuit(self):
        """Test creating an empty circuit"""
        builder = simq.CircuitBuilder(3)
        circuit = builder.build()
        assert circuit.num_qubits == 3
        assert circuit.gate_count == 0
        assert circuit.depth == 0

    def test_single_gate(self):
        """Test adding a single gate"""
        builder = simq.CircuitBuilder(1)
        builder.h(0)
        circuit = builder.build()
        assert circuit.gate_count == 1
        assert circuit.depth == 1

    def test_bell_state(self):
        """Test creating a Bell state circuit"""
        builder = simq.CircuitBuilder(2)
        builder.h(0)
        builder.cx(0, 1)
        circuit = builder.build()

        assert circuit.num_qubits == 2
        assert circuit.gate_count == 2
        assert circuit.depth == 2

    def test_ghz_state(self):
        """Test creating a GHZ state"""
        builder = simq.CircuitBuilder(3)
        builder.h(0)
        builder.cx(0, 1)
        builder.cx(0, 2)
        circuit = builder.build()

        assert circuit.num_qubits == 3
        assert circuit.gate_count == 3

    def test_all_single_qubit_gates(self):
        """Test all single-qubit gates"""
        builder = simq.CircuitBuilder(6)
        builder.h(0)
        builder.x(1)
        builder.y(2)
        builder.z(3)
        builder.s(4)
        builder.t(5)
        circuit = builder.build()

        assert circuit.gate_count == 6

    def test_two_qubit_gates(self):
        """Test two-qubit gates"""
        builder = simq.CircuitBuilder(3)
        builder.cx(0, 1)
        builder.cz(1, 2)
        builder.swap(0, 2)
        circuit = builder.build()

        assert circuit.gate_count == 3

    def test_rotation_gates(self):
        """Test rotation gates"""
        import math
        builder = simq.CircuitBuilder(3)
        builder.rx(0, math.pi / 4)
        builder.ry(1, math.pi / 2)
        builder.rz(2, math.pi)
        circuit = builder.build()

        assert circuit.gate_count == 3

    def test_invalid_qubit(self):
        """Test that invalid qubit indices raise errors"""
        builder = simq.CircuitBuilder(2)
        with pytest.raises(simq.InvalidQubitError):
            builder.h(5)

    def test_duplicate_qubit_in_two_qubit_gate(self):
        """Test that using same qubit for control and target raises error"""
        builder = simq.CircuitBuilder(2)
        with pytest.raises(simq.QuantumException):
            builder.cx(0, 0)

    def test_repr(self):
        """Test string representation"""
        builder = simq.CircuitBuilder(2)
        assert "CircuitBuilder" in repr(builder)
        assert "2" in repr(builder)


class TestCircuit:
    """Tests for Circuit (read-only)"""

    @pytest.fixture
    def simple_circuit(self):
        """Create a simple circuit for testing"""
        builder = simq.CircuitBuilder(2)
        builder.h(0)
        builder.cx(0, 1)
        return builder.build()

    def test_properties(self, simple_circuit):
        """Test circuit properties"""
        assert simple_circuit.num_qubits == 2
        assert simple_circuit.gate_count == 2
        assert simple_circuit.depth == 2

    def test_repr(self, simple_circuit):
        """Test circuit string representation"""
        repr_str = repr(simple_circuit)
        assert "Circuit" in repr_str
        assert "qubits=2" in repr_str
        assert "gates=2" in repr_str

    def test_str_ascii(self, simple_circuit):
        """Test ASCII circuit diagram"""
        ascii_str = str(simple_circuit)
        assert "q0" in ascii_str
        assert "q1" in ascii_str
        assert "H" in ascii_str  # Hadamard gate
        assert "●" in ascii_str or "C" in ascii_str  # Control
        assert "⊕" in ascii_str or "X" in ascii_str  # Target

    def test_to_ascii(self, simple_circuit):
        """Test to_ascii method"""
        ascii_diagram = simple_circuit.to_ascii()
        assert isinstance(ascii_diagram, str)
        assert len(ascii_diagram) > 0


class TestMultipleBuilds:
    """Test that builder can be used multiple times"""

    def test_build_multiple_times(self):
        """Test calling build() multiple times"""
        builder = simq.CircuitBuilder(2)
        builder.h(0)

        circuit1 = builder.build()
        circuit2 = builder.build()

        # Both should have same properties
        assert circuit1.num_qubits == circuit2.num_qubits
        assert circuit1.gate_count == circuit2.gate_count

    def test_modify_after_build(self):
        """Test modifying builder after build"""
        builder = simq.CircuitBuilder(2)
        builder.h(0)
        circuit1 = builder.build()

        # Add more gates
        builder.cx(0, 1)
        circuit2 = builder.build()

        # Second circuit should have more gates
        assert circuit2.gate_count > circuit1.gate_count
