"""
Parameterized Circuit Example

This example demonstrates how to use parameterized gates in SimQ,
which is essential for variational quantum algorithms like VQE and QAOA.
"""

import simq
import numpy as np

def create_variational_circuit(n_qubits, params):
    """
    Create a variational circuit with parameterized rotation gates.
    
    Args:
        n_qubits: Number of qubits
        params: List of rotation angles
        
    Returns:
        Circuit object
    """
    builder = simq.CircuitBuilder(n_qubits)
    
    # Layer 1: Hadamard gates
    for i in range(n_qubits):
        builder.h(i)
    
    # Layer 2: Parameterized rotations
    param_idx = 0
    for i in range(n_qubits):
        if param_idx < len(params):
            builder.rx(i, theta=params[param_idx])
            param_idx += 1
        if param_idx < len(params):
            builder.ry(i, theta=params[param_idx])
            param_idx += 1
    
    # Layer 3: Entangling layer
    for i in range(n_qubits - 1):
        builder.cx(i, i + 1)
    
    # Layer 4: More parameterized rotations
    for i in range(n_qubits):
        if param_idx < len(params):
            builder.rz(i, theta=params[param_idx])
            param_idx += 1
    
    return builder.build()

def main():
    n_qubits = 3
    n_params = 9  # 3 qubits × 3 parameters per qubit
    
    # Initialize random parameters
    print(f"Creating variational circuit with {n_qubits} qubits...")
    params = np.random.uniform(0, 2 * np.pi, n_params)
    print(f"Parameters: {params}")
    
    # Create circuit
    circuit = create_variational_circuit(n_qubits, params)
    print(f"\nCircuit depth: {circuit.depth}")
    print(f"Gate count: {circuit.gate_count}")
    
    # Simulate
    config = simq.SimulatorConfig(shots=1000)
    simulator = simq.Simulator(config)
    
    result = simulator.run(circuit)
    print(f"\nState vector amplitudes (first 4): {result.state_vector[:4]}")
    print(f"Probabilities (first 4): {result.probabilities[:4]}")
    
    # Run with different parameters
    print("\n" + "="*50)
    print("Testing with different parameters...")
    new_params = np.random.uniform(0, 2 * np.pi, n_params)
    print(f"New parameters: {new_params}")
    
    new_circuit = create_variational_circuit(n_qubits, new_params)
    new_result = simulator.run(new_circuit)
    print(f"New state vector amplitudes (first 4): {new_result.state_vector[:4]}")
    
    # Demonstrate parameter scanning
    print("\n" + "="*50)
    print("Scanning first parameter...")
    theta_values = np.linspace(0, 2*np.pi, 5)
    for theta in theta_values:
        scan_params = params.copy()
        scan_params[0] = theta
        scan_circuit = create_variational_circuit(n_qubits, scan_params)
        scan_result = simulator.run(scan_circuit)
        prob_sum = np.sum(scan_result.probabilities[:4])
        print(f"  θ={theta:.2f}: P(first 4 states) = {prob_sum:.4f}")

if __name__ == "__main__":
    main()
