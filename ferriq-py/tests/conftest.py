"""Pytest configuration and fixtures for Ferriq tests."""

import pytest


@pytest.fixture
def version():
    """Get the Ferriq version."""
    import ferriq

    return ferriq.__version__


# Additional fixtures will be added as we implement more features:
#
# @pytest.fixture
# def basic_circuit():
#     """Create a basic 3-qubit circuit with H and CNOT gates."""
#     import ferriq
#     builder = ferriq.CircuitBuilder(3)
#     builder.h(0)
#     builder.cx(0, 1)
#     builder.cx(1, 2)
#     return builder.build()
#
# @pytest.fixture
# def simulator():
#     """Create a default simulator instance."""
#     import ferriq
#     return ferriq.Simulator()
