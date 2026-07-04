import pytest
import simq
import numpy as np

def test_simulator_config():
    config = simq.SimulatorConfig(shots=2048)
    assert config.shots == 2048
    
    # Test default values
    default_config = simq.SimulatorConfig()
    assert default_config.shots == 1024

def test_simulator_creation():
    sim = simq.Simulator()
    assert sim is not None
    
    config = simq.SimulatorConfig(shots=100)
    sim_with_config = simq.Simulator(config)
    assert sim_with_config is not None

def test_run_bell_state():
    # Create Bell state (|00> + |11>) / sqrt(2)
    builder = simq.CircuitBuilder(2)
    builder.h(0)
    builder.cx(0, 1)
    circuit = builder.build()
    
    sim = simq.Simulator()
    result = sim.run(circuit)
    
    # Check state vector
    state_vector = result.state_vector
    assert len(state_vector) == 4
    
    # Check probabilities
    probs = result.probabilities
    assert len(probs) == 4
    # Bell state should have |00> and |11> with 50% probability each
    assert abs(probs[0] - 0.5) < 0.01 or abs(probs[3] - 0.5) < 0.01

def test_single_qubit_gate():
    builder = simq.CircuitBuilder(1)
    builder.x(0)  # |1>
    circuit = builder.build()
    
    sim = simq.Simulator()
    result = sim.run(circuit)
    
    # Check probabilities - should be 100% in |1> state
    probs = result.probabilities
    assert abs(probs[1] - 1.0) < 0.01
    assert abs(probs[0]) < 0.01

def test_hadamard_gate():
    builder = simq.CircuitBuilder(1)
    builder.h(0)
    circuit = builder.build()
    
    sim = simq.Simulator()
    result = sim.run(circuit)
    
    # Check probabilities - should be 50/50
    probs = result.probabilities
    assert abs(probs[0] - 0.5) < 0.01
    assert abs(probs[1] - 0.5) < 0.01
