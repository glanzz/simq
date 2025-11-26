import pytest
import simq

def create_ghz_circuit(n_qubits):
    builder = simq.CircuitBuilder(n_qubits)
    builder.h(0)
    for i in range(n_qubits - 1):
        builder.cx(i, i + 1)
    return builder.build()

@pytest.mark.benchmark(group="simulation")
@pytest.mark.parametrize("n_qubits", [5, 10, 15])
def test_simulation_ghz(benchmark, n_qubits):
    circuit = create_ghz_circuit(n_qubits)
    config = simq.SimulatorConfig(shots=1000)
    simulator = simq.Simulator(config)
    
    def run_sim():
        simulator.run(circuit)
        
    benchmark(run_sim)

@pytest.mark.benchmark(group="simulation")
@pytest.mark.parametrize("n_qubits", [5, 10])
def test_simulation_ghz_shots(benchmark, n_qubits):
    circuit = create_ghz_circuit(n_qubits)
    config = simq.SimulatorConfig(shots=1000)
    simulator = simq.Simulator(config)
    
    def run_sim_shots():
        simulator.run_with_shots(circuit, 1000)
        
    benchmark(run_sim_shots)
