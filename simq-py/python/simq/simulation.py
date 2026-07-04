from ._simq import simulation as _rust_sim

# Re-export Rust classes directly
SimulatorConfig = _rust_sim.SimulatorConfig
SimulationResult = _rust_sim.SimulationResult
Simulator = _rust_sim.Simulator
