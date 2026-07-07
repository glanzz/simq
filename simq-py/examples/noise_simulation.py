"""
Noise Simulation Example

This example demonstrates how to simulate quantum circuits with realistic
noise models in SimQ. We'll compare ideal and noisy simulations.
"""

import simq
import numpy as np

def create_test_circuit(n_qubits):
    """Create a test circuit with multiple gate layers."""
    builder = simq.CircuitBuilder(n_qubits)
    
    # Layer 1: Hadamard
    for i in range(n_qubits):
        builder.h(i)
    
    # Layer 2: CNOT chain
    for i in range(n_qubits - 1):
        builder.cx(i, i + 1)
    
    # Layer 3: More rotations
    for i in range(n_qubits):
        builder.rz(i, theta=np.pi/4)
    
    return builder.build()

def main():
    n_qubits = 3
    n_shots = 2000
    
    # Create circuit
    print(f"Creating {n_qubits}-qubit test circuit...")
    circuit = create_test_circuit(n_qubits)
    print(f"Circuit depth: {circuit.depth}, gates: {circuit.gate_count}")
    
    # Ideal simulation
    print("\n" + "="*50)
    print("IDEAL SIMULATION (no noise)")
    print("="*50)
    
    ideal_config = simq.SimulatorConfig(shots=n_shots)
    ideal_sim = simq.Simulator(ideal_config)
    
    ideal_result = ideal_sim.run(circuit)
    ideal_counts = ideal_sim.run_with_shots(circuit, shots=n_shots)
    
    print(f"State vector (first 4): {ideal_result.state_vector[:4]}")
    print(f"Measurement counts: {ideal_counts}")
    
    # Noisy simulation - Depolarizing noise
    print("\n" + "="*50)
    print("NOISY SIMULATION (Depolarizing Channel)")
    print("="*50)
    
    noise_model = simq.HardwareNoiseModel()
    
    # Add depolarizing noise to CNOT gates (typically the noisiest)
    depol_channel = simq.DepolarizingChannel(0.05)  # 5% error rate
    noise_model.add_gate_error("cx", depol_channel)
    
    print("Noise model: 5% depolarizing error on CNOT gates")
    
    noisy_config = simq.SimulatorConfig(noise_model=noise_model, shots=n_shots)
    noisy_sim = simq.Simulator(noisy_config)
    
    noisy_counts = noisy_sim.run_with_shots(circuit, shots=n_shots)
    print(f"Measurement counts: {noisy_counts}")
    
    # Noisy simulation - Amplitude Damping
    print("\n" + "="*50)
    print("NOISY SIMULATION (Amplitude Damping)")
    print("="*50)
    
    noise_model2 = simq.HardwareNoiseModel()
    amp_damp = simq.AmplitudeDamping(0.1)  # 10% amplitude damping
    noise_model2.add_gate_error("cx", amp_damp)
    
    print("Noise model: 10% amplitude damping on CNOT gates")
    
    noisy_config2 = simq.SimulatorConfig(noise_model=noise_model2, shots=n_shots)
    noisy_sim2 = simq.Simulator(noisy_config2)
    
    noisy_counts2 = noisy_sim2.run_with_shots(circuit, shots=n_shots)
    print(f"Measurement counts: {noisy_counts2}")
    
    # Compare distributions
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    
    all_states = set(ideal_counts.keys()) | set(noisy_counts.keys()) | set(noisy_counts2.keys())
    print(f"{'State':<10} {'Ideal':<10} {'Depol':<10} {'AmpDamp':<10}")
    print("-" * 40)
    
    for state in sorted(all_states)[:8]:  # Show first 8 states
        ideal_prob = ideal_counts.get(state, 0) / n_shots
        noisy_prob = noisy_counts.get(state, 0) / n_shots
        noisy_prob2 = noisy_counts2.get(state, 0) / n_shots
        print(f"{state:<10} {ideal_prob:<10.4f} {noisy_prob:<10.4f} {noisy_prob2:<10.4f}")

if __name__ == "__main__":
    main()
