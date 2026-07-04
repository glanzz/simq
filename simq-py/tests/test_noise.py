import pytest
import simq
from simq.noise import (
    DepolarizingChannel,
    AmplitudeDamping,
    PhaseDamping,
    HardwareNoiseModel,
)

def test_noise_channels():
    # Depolarizing
    depol = DepolarizingChannel(0.1)
    assert depol.error_probability == 0.1
    
    # Amplitude Damping
    amp = AmplitudeDamping(0.05)
    assert amp.gamma == 0.05
    amp_t1 = AmplitudeDamping.from_t1(t1=100.0, gate_time=0.1)
    assert 0.0 < amp_t1.gamma < 1.0
    
    # Phase Damping
    phase = PhaseDamping(0.05)
    assert phase.lambda_value == 0.05
    phase_t2 = PhaseDamping.from_t2(t2=100.0, gate_time=0.1)
    assert 0.0 < phase_t2.lambda_value < 1.0

def test_hardware_noise_model():
    model = HardwareNoiseModel(5)
    assert model.num_qubits() == 5
    
    # Set properties
    model.set_qubit_t1(0, 50.0)
    model.set_qubit_t2(0, 30.0)
    model.set_single_qubit_fidelity(0, 0.99)
    
    # Check derived channels
    amp = model.amplitude_damping_single_gate(0)
    assert amp.gamma > 0.0
    
    depol = model.depolarizing_single_gate(0)
    assert abs(depol.error_probability - 0.01) < 1e-10

def test_noisy_simulation_depolarizing():
    # Create a circuit that should be |0> -> |0> (Identity)
    # But with strong depolarizing noise, it should be mixed.
    builder = simq.CircuitBuilder(1)
    # Apply X then X (Identity)
    builder.x(0)
    builder.x(0)
    circuit = builder.build()
    
    # Create noise model with strong depolarizing noise
    model = HardwareNoiseModel(1)
    model.set_single_qubit_fidelity(0, 0.5) # 50% error rate!
    
    config = simq.SimulatorConfig(shots=1000, noise_model=model)
    sim = simq.Simulator(config)
    
    counts = sim.run_with_shots(circuit, 1000)
    
    # Without noise, should be 100% "0"
    # With 50% depolarizing noise on each gate (2 gates),
    # we expect significant "1" counts.
    assert "0" in counts
    assert "1" in counts
    assert counts["1"] > 50 # Should be significant

def test_noisy_simulation_amplitude_damping():
    # Prepare |1> state
    builder = simq.CircuitBuilder(1)
    builder.x(0)
    
    # Apply Identity gates to allow decay
    for _ in range(10):
        # Use pairs of X as identity
        builder.x(0)
        builder.x(0)
    
    circuit = builder.build()
    
    # Create noise model with short T1
    model = HardwareNoiseModel(1)
    model.set_qubit_t1(0, 0.1) # Very short T1
    # Gate time defaults to 0.02us. 20 gates * 0.02 = 0.4us.
    # exp(-0.4/0.1) = exp(-4) ~ 0.018 survival probability.
    # So most should decay to |0>.
    
    config = simq.SimulatorConfig(shots=1000, noise_model=model)
    sim = simq.Simulator(config)
    
    counts = sim.run_with_shots(circuit, 1000)
    
    # Should be mostly "0" due to decay, even though logic says |1>
    # Logic: X -> |1>. Then 20 gates.
    # If no noise: |1>.
    # With noise: Decay to |0>.
    
    print(f"Counts: {counts}")
    assert counts.get("0", 0) > counts.get("1", 0)

def test_readout_error():
    # Prepare |0> state
    builder = simq.CircuitBuilder(1)
    circuit = builder.build()
    
    # Create noise model with 100% readout error (flip 0->1)
    model = HardwareNoiseModel(1)
    model.set_readout_error(0, 1.0, 0.0) # p01=1.0 (0->1), p10=0.0
    
    config = simq.SimulatorConfig(shots=100, noise_model=model)
    sim = simq.Simulator(config)
    
    counts = sim.run_with_shots(circuit, 100)
    
    # Should be all "1"
    assert counts.get("1", 0) == 100

if __name__ == "__main__":
    test_noise_channels()
    test_hardware_noise_model()
    test_noisy_simulation_depolarizing()
    test_noisy_simulation_amplitude_damping()
    test_readout_error()
    print("All tests passed!")
