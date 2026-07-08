Quickstart
==========

Installation
------------

Install Ferriq using pip:

.. code-block:: bash

   pip install ferriq

Basic Usage
-----------

Creating a simple Bell State circuit:

.. code-block:: python

   import ferriq

   # Build a circuit
   builder = ferriq.CircuitBuilder(2)
   builder.h(0)
   builder.cx(0, 1)
   circuit = builder.build()

   # Simulate
   config = ferriq.SimulatorConfig(shots=1000)
   simulator = ferriq.Simulator(config)
   result = simulator.run(circuit)

   # Get results
   print(f"State vector: {result.state_vector}")
   counts = simulator.run_with_shots(circuit, shots=1024)
   print(f"Measurement counts: {counts}")

Parameterized Circuits
----------------------

.. code-block:: python

   import ferriq
   import numpy as np

   builder = ferriq.CircuitBuilder(2)
   builder.rx(0, theta=np.pi/4)
   builder.ry(1, theta=np.pi/2)
   builder.cx(0, 1)
   circuit = builder.build()

Noise Simulation
----------------

.. code-block:: python

   import ferriq

   # Create noise model
   noise_model = ferriq.HardwareNoiseModel()
   noise_model.add_gate_error("cx", ferriq.DepolarizingChannel(0.01))
   
   # Simulate with noise
   config = ferriq.SimulatorConfig(noise_model=noise_model, shots=1000)
   simulator = ferriq.Simulator(config)
   result = simulator.run(circuit)
