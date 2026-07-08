"""Basic tests for Ferriq Python bindings."""

import sys


def test_import():
    """Test that the ferriq module can be imported."""
    import ferriq

    assert ferriq is not None


def test_version(version):
    """Test that version information is available."""
    assert version is not None
    assert isinstance(version, str)
    assert len(version) > 0


def test_version_format(version):
    """Test that version follows semantic versioning."""
    parts = version.split(".")
    assert len(parts) >= 3, f"Version should be semver format, got: {version}"
    # Check that first three parts are numbers
    for part in parts[:3]:
        assert part.isdigit(), f"Version part should be numeric, got: {part}"


def test_author():
    """Test that author information is available."""
    import ferriq

    assert hasattr(ferriq, "__author__")
    assert isinstance(ferriq.__author__, str)
    assert len(ferriq.__author__) > 0


def test_python_version():
    """Ensure we're running on a supported Python version."""
    version_info = sys.version_info
    # ferriq requires Python 3.8+
    assert version_info >= (3, 8), f"Python 3.8+ required, got {sys.version}"


# Phase 1+ tests will be added as features are implemented:
#
# class TestCircuit:
#     """Tests for Circuit and CircuitBuilder."""
#
#     def test_create_builder(self):
#         import ferriq
#         builder = ferriq.CircuitBuilder(3)
#         assert builder is not None
#
#     def test_build_circuit(self, basic_circuit):
#         assert basic_circuit.num_qubits == 3
#         assert basic_circuit.gate_count == 3
