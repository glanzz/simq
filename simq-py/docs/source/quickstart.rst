Quickstart
==========

Installation
------------

Install SimQ using pip:

.. code-block:: bash

   pip install simq

Basic Usage
-----------

Creating a simple Bell State circuit:

.. code-block:: python

   import simq

   # Build a circuit
   builder = simq.CircuitBuilder(2)
   builder.h(0)
   builder.cx(0, 1)
   circuit = builder.build()

   # Simulate
   config = simq.SimulatorConfig(shots=1000)
   simulator = simq.Simulator(config)
   result = simulator.run(circuit)

   # Get results
   print(f"State vector: {result.state_vector}")
   counts = simulator.run_with_shots(circuit, shots=1024)
   print(f"Measurement counts: {counts}")

Parameterized Circuits
----------------------

.. code-block:: python

   import simq
   import numpy as np

   builder = simq.CircuitBuilder(2)
   builder.rx(0, theta=np.pi/4)
   builder.ry(1, theta=np.pi/2)
   builder.cx(0, 1)
   circuit = builder.build()

Noise Simulation
----------------

.. code-block:: python

   import simq

   # Create noise model
   noise_model = simq.HardwareNoiseModel()
   noise_model.add_gate_error("cx", simq.DepolarizingChannel(0.01))
   
   # Simulate with noise
   config = simq.SimulatorConfig(noise_model=noise_model, shots=1000)
   simulator = simq.Simulator(config)
   result = simulator.run(circuit)
