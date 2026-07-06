"""
Getting Started with SimQ

Builds a Bell state circuit and runs it on the local simulator.
"""

import simq


def main():
    # Create a simple 2-qubit circuit
    builder = simq.CircuitBuilder(2)
    builder.h(0)
    builder.cx(0, 1)

    # Build and display
    circuit = builder.build()
    print(f"Circuit: {circuit.num_qubits} qubits, {circuit.gate_count} gates")

    # Simulate
    simulator = simq.Simulator()
    result = simulator.run(circuit)
    print(f"Probabilities: {result.probabilities}")


if __name__ == "__main__":
    main()
