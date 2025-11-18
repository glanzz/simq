//! VQE/QAOA helper functions for SimQ
//!
//! This module provides high-level routines for variational quantum algorithms,
//! including circuit builders for QAOA and VQE.

use std::sync::Arc;
use simq_core::circuit::Circuit;
use simq_core::QubitId;
use simq_gates::{Hadamard, RotationX, RotationZ, RotationY, CNot};

/// Generate a QAOA circuit for given cost and mixer Hamiltonians.
/// `depth` is the number of QAOA layers.
/// `params` is a slice of [gamma_1, beta_1, gamma_2, beta_2, ...]
pub fn qaoa_circuit(
    num_qubits: usize,
    cost_hamiltonian: &[(usize, f64)], // (qubit, coeff) for Z terms
    mixer_qubits: &[usize],
    depth: usize,
    params: &[f64],
) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);
    let mut param_idx = 0;
    // Initial state: apply Hadamard to all qubits
    for q in 0..num_qubits {
        let _ = circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]);
    }
    // QAOA layers
    for _layer in 0..depth {
        // Cost Hamiltonian: exp(-i gamma C)
        let gamma = params[param_idx];
        param_idx += 1;
        for &(q, coeff) in cost_hamiltonian {
            let _ = circuit.add_gate(Arc::new(RotationZ::new(coeff * gamma)), &[QubitId::new(q)]);
        }
        // Mixer Hamiltonian: exp(-i beta X)
        let beta = params[param_idx];
        param_idx += 1;
        for &q in mixer_qubits {
            let _ = circuit.add_gate(Arc::new(RotationX::new(2.0 * beta)), &[QubitId::new(q)]);
        }
    }
    circuit
}

/// Generate a hardware-efficient VQE ansatz circuit.
/// Each qubit gets a Ry(θ) rotation, then entangling CNOTs in a chain.
/// `params` should have length equal to `num_qubits`.
pub fn vqe_hardware_efficient_ansatz(num_qubits: usize, params: &[f64]) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);
    // Initial Hadamard layer
    for q in 0..num_qubits {
        let _ = circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]);
    }
    // Ry rotations (parameterized)
    for (q, &theta) in params.iter().enumerate() {
        let _ = circuit.add_gate(Arc::new(RotationY::new(theta)), &[QubitId::new(q)]);
    }
    // Entangling CNOTs (linear chain)
    for q in 0..(num_qubits - 1) {
        let _ = circuit.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)]);
    }
    circuit
}
