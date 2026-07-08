"""
Basic Circuit Example

This example demonstrates how to create and simulate a simple quantum circuit
using Ferriq. We'll create a Bell state (maximally entangled state) and measure it.
"""

import ferriq

def main():
    # Create a circuit with 2 qubits
    print("Creating a 2-qubit circuit...")
    builder = ferriq.CircuitBuilder(2)
    
    # Apply Hadamard gate to first qubit
    builder.h(0)
    
    # Apply CNOT gate (control=0, target=1)
    builder.cx(0, 1)
    
    # Build the circuit
    circuit = builder.build()
    
    print(f"Circuit created with {circuit.num_qubits} qubits")
    print(f"Circuit depth: {circuit.depth}")
    print(f"Gate count: {circuit.gate_count}")
    
    # Configure simulator
    config = ferriq.SimulatorConfig(shots=1000)
    simulator = ferriq.Simulator(config)
    
    # Run statevector simulation
    print("\nRunning statevector simulation...")
    result = simulator.run(circuit)
    print(f"State vector: {result.state_vector}")
    print(f"Probabilities: {result.probabilities}")
    
    # Run with shots (measurements)
    print("\nRunning simulation with 1024 shots...")
    counts = simulator.run_with_shots(circuit, shots=1024)
    print(f"Measurement counts: {counts}")
    
    # Expected result: roughly 50% |00⟩ and 50% |11⟩
    print("\nExpected: Bell state with ~50% |00⟩ and ~50% |11⟩")

if __name__ == "__main__":
    main()
