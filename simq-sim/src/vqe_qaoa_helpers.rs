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

//! VQE/QAOA helper functions for SimQ
//!
//! This module provides high-level routines for variational quantum algorithms,
//! including batch gradient computation and expectation evaluation.

use crate::simulator::Simulator;
use crate::gradient::compute_gradient_parameter_shift;
use crate::finite_diff::compute_gradient_finite_difference;
use crate::batch_eval::evaluate_batch_expectation;
use crate::circuit::ParametricCircuit;
use crate::observable::Observable;

/// Compute gradients for a parametric circuit using the parameter shift rule.
pub fn vqe_gradient_parameter_shift(
    simulator: &Simulator,
    circuit: &ParametricCircuit,
    observable: &Observable,
    params: &[f64],
) -> Vec<f64> {
    compute_gradient_parameter_shift(simulator, circuit, observable, params)
}

/// Compute gradients for a parametric circuit using finite differences.
pub fn vqe_gradient_finite_difference(
    simulator: &Simulator,
    circuit: &ParametricCircuit,
    observable: &Observable,
    params: &[f64],
    epsilon: f64,
) -> Vec<f64> {
    compute_gradient_finite_difference(simulator, circuit, observable, params, epsilon)
}

/// Evaluate expectation values for a batch of parameter sets.
pub fn vqe_batch_expectation(
    simulator: &Simulator,
    circuit: &ParametricCircuit,
    observable: &Observable,
    batch_params: &[Vec<f64>],
) -> Vec<f64> {
    evaluate_batch_expectation(simulator, circuit, observable, batch_params)
}

// TODO: Add QAOA circuit generator and VQE ansatz templates here.
