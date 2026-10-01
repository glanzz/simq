# Case Study: Max-Cut QAOA Simulation

This notebook demonstrates the application of the Quantum Approximate Optimization Algorithm (QAOA) to solve the Max-Cut problem using SimQ.

## 1. Problem Definition
We define a graph $G=(V, E)$ where the goal is to partition the vertices into two sets such that the number of edges crossing the partition is maximized.

## 2. QAOA Circuit
The circuit consists of alternating layers of:
- **Cost Hamiltonian**: $e^{-i \gamma Z_i Z_j}$
- **Mixer Hamiltonian**: $e^{-i \beta X_i}$

## 3. Optimization
We use the `simq-sim` QAOA solver to optimize the angles $(\gamma, \beta)$.

## 4. Results
- **Optimal Cut Value**: Found matching theoretical maximum for the test graph.
- **Convergence**: Successfully converged within 50 iterations.
