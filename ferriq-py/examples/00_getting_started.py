"""
Getting Started with Ferriq

Builds a Bell state circuit and runs it on the local simulator.
"""

import ferriq


def main():
    # Create a simple 2-qubit circuit
    builder = ferriq.CircuitBuilder(2)
    builder.h(0)
    builder.cx(0, 1)

    # Build and display
    circuit = builder.build()
    print(f"Circuit: {circuit.num_qubits} qubits, {circuit.gate_count} gates")

    # Simulate
    simulator = ferriq.Simulator()
    result = simulator.run(circuit)
    print(f"Probabilities: {result.probabilities}")


if __name__ == "__main__":
    main()
