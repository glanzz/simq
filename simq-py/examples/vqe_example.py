"""
VQE (Variational Quantum Eigensolver) Example

This example demonstrates a simple VQE implementation using SimQ.
We'll find the ground state energy of a simple Hamiltonian using
a variational quantum circuit.

Note: This is a simplified educational example. Production VQE
implementations would use more sophisticated optimizers and ansatzes.
"""

import simq
import numpy as np

def create_ansatz(n_qubits, params):
    """
    Create a hardware-efficient ansatz circuit.
    
    Args:
        n_qubits: Number of qubits
        params: Variational parameters (list of angles)
        
    Returns:
        Circuit object
    """
    builder = simq.CircuitBuilder(n_qubits)
    
    param_idx = 0
    depth = 2  # Number of layers
    
    for layer in range(depth):
        # Rotation layer
        for i in range(n_qubits):
            if param_idx < len(params):
                builder.ry(i, theta=params[param_idx])
                param_idx += 1
        
        # Entanglement layer
        for i in range(n_qubits - 1):
            builder.cx(i, i + 1)
        
        # More rotations
        for i in range(n_qubits):
            if param_idx < len(params):
                builder.rz(i, theta=params[param_idx])
                param_idx += 1
    
    return builder.build()

def measure_z_expectation(simulator, circuit, qubit_idx):
    """
    Measure expectation value of Z operator on a specific qubit.
    
    For a qubit, <Z> = P(|0⟩) - P(|1⟩)
    
    Args:
        simulator: Simulator instance
        circuit: Circuit to measure
        qubit_idx: Which qubit to measure
        
    Returns:
        Expectation value of Z
    """
    result = simulator.run(circuit)
    probs = result.probabilities
    
    # Calculate <Z> from probabilities
    expectation = 0.0
    n_states = len(probs)
    
    for state_idx in range(n_states):
        # Check if qubit_idx is |0⟩ or |1⟩ in this computational basis state
        if (state_idx >> qubit_idx) & 1:  # Qubit is in |1⟩
            expectation -= probs[state_idx]
        else:  # Qubit is in |0⟩
            expectation += probs[state_idx]
    
    return expectation

def compute_energy(simulator, circuit, hamiltonian):
    """
    Compute energy for a simple Hamiltonian.
    
    Hamiltonian is represented as [(coeff, qubit, pauli), ...]
    where pauli is 'Z', 'X', or 'I'
    
    For simplicity, we only implement Z measurements here.
    """
    energy = 0.0
    
    for coeff, qubit, pauli in hamiltonian:
        if pauli == 'Z':
            exp_val = measure_z_expectation(simulator, circuit, qubit)
            energy += coeff * exp_val
        elif pauli == 'I':
            energy += coeff
    
    return energy

def vqe_optimization(n_qubits, hamiltonian, n_iterations=20):
    """
    Run VQE optimization.
    
    Args:
        n_qubits: Number of qubits
        hamiltonian: List of (coefficient, qubit, pauli) tuples
        n_iterations: Number of optimization iterations
        
    Returns:
        Optimal parameters and final energy
    """
    # Initialize parameters
    n_params = n_qubits * 4  # 2 params per qubit per layer, 2 layers
    params = np.random.uniform(0, 2*np.pi, n_params)
    
    # Configure simulator
    config = simq.SimulatorConfig(shots=1000)
    simulator = simq.Simulator(config)
    
    # Simple gradient descent
    learning_rate = 0.1
    epsilon = 0.01  # For numerical gradients
    
    print("Starting VQE optimization...")
    print(f"Initial parameters: {params}")
    
    for iteration in range(n_iterations):
        # Create circuit with current parameters
        circuit = create_ansatz(n_qubits, params)
        
        # Compute current energy
        energy = compute_energy(simulator, circuit, hamiltonian)
        
        if iteration % 5 == 0:
            print(f"Iteration {iteration}: Energy = {energy:.6f}")
        
        # Compute gradients (finite difference)
        gradients = np.zeros_like(params)
        for i in range(len(params)):
            # Forward difference
            params_plus = params.copy()
            params_plus[i] += epsilon
            circuit_plus = create_ansatz(n_qubits, params_plus)
            energy_plus = compute_energy(simulator, circuit_plus, hamiltonian)
            
            # Backward difference
            params_minus = params.copy()
            params_minus[i] -= epsilon
            circuit_minus = create_ansatz(n_qubits, params_minus)
            energy_minus = compute_energy(simulator, circuit_minus, hamiltonian)
            
            # Central difference
            gradients[i] = (energy_plus - energy_minus) / (2 * epsilon)
        
        # Update parameters
        params -= learning_rate * gradients
        
        # Decay learning rate
        learning_rate *= 0.95
    
    # Final energy
    final_circuit = create_ansatz(n_qubits, params)
    final_energy = compute_energy(simulator, final_circuit, hamiltonian)
    
    return params, final_energy

def main():
    n_qubits = 2
    
    # Define a simple Hamiltonian: H = -0.5*Z0 - 0.3*Z1 + 0.2*I
    # This has ground state energy around -0.8
    hamiltonian = [
        (-0.5, 0, 'Z'),  # -0.5 * Z on qubit 0
        (-0.3, 1, 'Z'),  # -0.3 * Z on qubit 1
        (0.2, 0, 'I'),   # Constant offset
    ]
    
    print("="*60)
    print("VQE Example: Finding Ground State Energy")
    print("="*60)
    print(f"Hamiltonian: H = -0.5*Z₀ - 0.3*Z₁ + 0.2*I")
    print(f"Number of qubits: {n_qubits}")
    print(f"Theoretical minimum: -0.5 - 0.3 + 0.2 = -0.6")
    print()
    
    # Run VQE
    optimal_params, final_energy = vqe_optimization(n_qubits, hamiltonian, n_iterations=30)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Final energy: {final_energy:.6f}")
    print(f"Optimal parameters: {optimal_params}")
    print(f"Theoretical ground state: -0.6")
    print(f"Error: {abs(final_energy - (-0.6)):.6f}")

if __name__ == "__main__":
    main()
