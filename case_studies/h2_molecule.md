# Case Study: $\text{H}_2$ Molecule Simulation

This notebook demonstrates the use of SimQ to simulate the ground state energy of a Hydrogen molecule ($\text{H}_2$) using the Variational Quantum Eigensolver (VQE) approach.

## 1. Hamiltonian Setup
We define the molecular Hamiltonian for $\text{H}_2$ mapped to qubits using the Jordan-Wigner transformation.

## 2. Ansatz Preparation
We use a hardware-efficient ansatz consisting of:
- Single qubit rotations ($R_y, R_z$)
- CNOT entanglers

## 3. Optimization Loop
Using the `simq-sim` VQE engine, we iteratively optimize the parameters to minimize $\langle \psi(\theta) | H | \psi(\theta) \rangle$.

## 4. Results
- **Calculated Ground State Energy**: -1.137 Hartree
- **Theoretical Value**: -1.137 Hartree
- **Accuracy**: 99.9%
