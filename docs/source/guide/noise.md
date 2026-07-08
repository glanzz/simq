---
myst:
  html_meta:
    description: "Noise modelling in Ferriq — depolarizing, amplitude damping, phase damping, readout error channels and hardware noise models in Rust and Python."
---

# Noise modelling

Real hardware is noisy. Ferriq's noise system (in `ferriq-core::noise`, exposed
to Python as `ferriq.noise`) lets you attach realistic error channels to
simulations.

## Noise channels

| Channel | Models |
|---------|--------|
| `DepolarizingChannel` | Uniform Pauli errors with probability *p* |
| `AmplitudeDamping` | Energy relaxation (T₁ decay) |
| `PhaseDamping` | Pure dephasing (T₂) |
| `ReadoutError` | Bit-flip errors at measurement time |

Each channel has a Monte-Carlo counterpart (`DepolarizingMC`,
`AmplitudeDampingMC`, ...) used for trajectory-based sampling during shot
simulation.

## Hardware noise models

`HardwareNoiseModel` composes per-qubit and per-gate noise into a device
profile:

- `QubitProperties` — T₁/T₂ times and per-qubit error rates
- `GateTiming` — gate durations, so damping is applied for the right time
- `TwoQubitGateProperties` and `CrosstalkProperties` — coupling-dependent
  errors
- `GateNoise` — attaches channels to specific gate types

## Using noise from Python

```python
import ferriq

# Build a device-like noise model
noise_model = ferriq.HardwareNoiseModel()
noise_model.add_gate_error("cx", ferriq.DepolarizingChannel(0.01))

# Simulate with noise
config = ferriq.SimulatorConfig(noise_model=noise_model, shots=1000)
simulator = ferriq.Simulator(config)
result = simulator.run(circuit)
```

A complete, commented walkthrough lives in
[`ferriq-py/examples/noise_simulation.py`](https://github.com/glanzz/ferriq/blob/main/ferriq-py/examples/noise_simulation.py) —
it builds noise channels, attaches them to a hardware model, and compares
noisy versus ideal measurement distributions.

## Tips

- Start with a single `DepolarizingChannel` on two-qubit gates — they
  dominate error budgets on real devices.
- Use `SimulatorConfig(seed=...)` for reproducible noisy runs when
  debugging.
- Readout error is often the largest effect on shallow circuits; model it
  separately from gate noise.
