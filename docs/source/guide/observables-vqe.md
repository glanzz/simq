---
myst:
  html_meta:
    description: "Variational algorithms with Ferriq — Pauli observables, exact expectation values, gradients, built-in optimizers, and VQE/QAOA helpers."
---

# Observables, VQE & QAOA

Ferriq is built for variational algorithms: exact expectation values are one
call away, gradients come built in, and `ferriq-sim` ships classical
optimizers plus VQE/QAOA circuit helpers.

## Pauli observables

Hamiltonians are sums of weighted Pauli strings:

```rust
use ferriq::{PauliObservable, PauliString};

// H = 1.0 * Z
let h = PauliObservable::from_pauli_string(
    PauliString::from_str("Z").unwrap(), 1.0);

// Multi-qubit strings work the same way: "XXZI", "ZZII", ...
let zz = PauliObservable::from_pauli_string(
    PauliString::from_str("ZZ").unwrap(), 0.5);
```

## Exact expectation values

An energy function for VQE is a few lines:

```rust
use ferriq::{PauliObservable, PauliString, QuantumCircuit};

let hamiltonian = PauliObservable::from_pauli_string(
    PauliString::from_str("Z").unwrap(), 1.0);

let energy = |theta: f64| {
    let mut qc = QuantumCircuit::new(1);
    qc.ry(theta, 0);
    qc.expectation_value(&hamiltonian).unwrap()
};
// energy(θ) = cos(θ); minimize with your favourite optimizer
```

## A complete VQE loop

The runnable example `ferriq/examples/vqe_fluent.rs` optimizes RY(θ)|0⟩
against H = Z with gradient descent and converges to the ground state
energy −1 at θ = π:

```bash
cargo run -p ferriq --example vqe_fluent
```

The core of the loop is nothing more than the energy function above plus a
central finite-difference gradient:

```rust
let grad = (energy(theta + eps) - energy(theta - eps)) / (2.0 * eps);
theta -= learning_rate * grad;
```

Exact gate matrices make finite differences reliable at any angle.

## Gradient methods

The `ferriq_sim::gradient` module provides several strategies:

| Method | Module | Notes |
|--------|--------|-------|
| Finite differences | `finite_difference` | Simple, works with any circuit |
| Parameter shift | `parameter_shift` | Exact gradients for rotation gates |
| Forward-mode autodiff | `autodiff` (`Dual`) | Dual-number based |
| Reverse-mode autodiff | `autodiff` (`ReverseTape`, `HybridAD`) | Scales to many parameters |
| Batch evaluation | `batch`, `batch_advanced` | Parallel energy/gradient batches, grid search, importance sampling |

`gradient_fallback.rs` (in `ferriq-sim/examples/`) shows how methods degrade
gracefully when a gate has no analytic rule.

## Classical optimizers

`ferriq_sim::gradient::classical_optimizers` includes ready-made
implementations of **L-BFGS** (`LBFGSOptimizer`) and **Nelder–Mead**
(`NelderMeadOptimizer`), each with a config struct. Convergence monitoring
lives in `gradient::convergence` (`ConvergenceMonitor`,
`StoppingCriterion`, `BestTracker`) — see
`ferriq-sim/examples/convergence_monitoring.rs` and
`optimizer_comparison.rs`.

## VQE / QAOA circuit helpers

`ferriq_sim` also provides ansatz constructors:

- `vqe_hardware_efficient_ansatz(num_qubits, params)` — the standard
  hardware-efficient layered ansatz
- `qaoa_circuit(...)` — builds a QAOA circuit from a problem Hamiltonian

## End-to-end examples

| Example | Run with |
|---------|----------|
| H₂ molecule ground state (VQE) | `cargo run -p ferriq-sim --example vqe_h2_molecule` |
| MaxCut (QAOA) | `cargo run -p ferriq-sim --example qaoa_maxcut` |
| Comprehensive QAOA workflow | `cargo run -p ferriq-sim --example qaoa_comprehensive` |
| Optimizer comparison | `cargo run -p ferriq-sim --example optimizer_comparison` |
