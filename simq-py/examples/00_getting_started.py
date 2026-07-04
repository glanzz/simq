"""
Getting Started with SimQ

This example will be implemented in Phase 1 once the core API is ready.

Expected usage:
---------------

import simq

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
"""

if __name__ == "__main__":
    print("This example will be available in Phase 1 of development.")
    print("Current phase: Phase 0 - Project Setup Complete")
