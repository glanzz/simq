"""Pytest configuration and fixtures for SimQ tests."""

import pytest


@pytest.fixture
def version():
    """Get the SimQ version."""
    import simq

    return simq.__version__


# Additional fixtures will be added as we implement more features:
#
# @pytest.fixture
# def basic_circuit():
#     """Create a basic 3-qubit circuit with H and CNOT gates."""
#     import simq
#     builder = simq.CircuitBuilder(3)
#     builder.h(0)
#     builder.cx(0, 1)
#     builder.cx(1, 2)
#     return builder.build()
#
# @pytest.fixture
# def simulator():
#     """Create a default simulator instance."""
#     import simq
#     return simq.Simulator()
