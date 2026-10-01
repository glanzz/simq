# Migration Guide: Qiskit $\rightarrow$ SimQ

This guide provides a mapping of concepts and code patterns to help users transition from Qiskit to the SimQ SDK.

## Core Concepts Mapping

| Qiskit | SimQ | Note |
| :--- | :--- | :--- |
| `QuantumCircuit(n)` | `CircuitBuilder::<n>::new()` | SimQ uses a fluent builder pattern with compile-time size checking. |
| `circuit.h(0)` | `builder.h(0)` | Equivalent single-qubit gates. |
| `circuit.cx(0, 1)` | `builder.cx(0, 1)` | Equivalent controlled gates. |
| `AerSimulator()` | `Simulator::new()` | SimQ provides a high-performance native simulator. |
| `execute(qc, backend)` | `simulator.run(circuit)` | SimQ uses a direct run method returning probabilities. |
| `get_counts()` | `result.probabilities` | SimQ returns normalized probabilities by default. |

## Code Translation Example

### Qiskit
```python
from qiskit import QuantumCircuit, Aer, execute

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
counts = job.result().get_counts()
```

### SimQ
```python
import simq

builder = simq.CircuitBuilder(2)
builder.h(0)
builder.cx(0, 1)
circuit = builder.build()

simulator = simq.Simulator()
result = simulator.run(circuit)
print(result.probabilities)
```

## Key Differences
- **Type Safety**: SimQ leverages Rust's type system via Python bindings to catch circuit errors earlier.
- **Performance**: SimQ's core is written in Rust, offering significant speedups for statevector simulations over standard Python-based backends.
- **Explicit Building**: In SimQ, you use a `CircuitBuilder` to define the layout, and then `build()` it into an immutable `Circuit` for execution.
