import pytest
import simq

def build_circuit(n_qubits, depth):
    builder = simq.CircuitBuilder(n_qubits)
    for _ in range(depth):
        for i in range(n_qubits):
            builder.h(i)
        for i in range(0, n_qubits - 1, 2):
            builder.cx(i, i + 1)
    return builder.build()

@pytest.mark.benchmark(group="circuit_construction")
@pytest.mark.parametrize("n_qubits", [10, 20, 50])
def test_circuit_construction_small_depth(benchmark, n_qubits):
    benchmark(build_circuit, n_qubits, depth=10)

@pytest.mark.benchmark(group="circuit_construction")
@pytest.mark.parametrize("n_qubits", [10, 20])
def test_circuit_construction_large_depth(benchmark, n_qubits):
    benchmark(build_circuit, n_qubits, depth=100)
