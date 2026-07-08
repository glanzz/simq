//! Fluent quantum circuit builder
//!
//! [`QuantumCircuit`] is an ergonomic facade over [`ferriq_core::Circuit`] with
//! named gate methods, so circuits read like the algorithm they implement:
//!
//! ```
//! use ferriq::QuantumCircuit;
//!
//! let mut qc = QuantumCircuit::new(2);
//! qc.h(0).cnot(0, 1).rz(0.8, 1);
//! let circuit = qc.build().unwrap();
//! assert_eq!(circuit.len(), 3);
//! ```
//!
//! # Error handling
//!
//! Gate methods return `&mut Self` so calls chain without `.unwrap()` noise.
//! The first error (e.g. a qubit index out of range) is recorded and all
//! subsequent gate calls become no-ops; the error surfaces from
//! [`QuantumCircuit::build`] or [`QuantumCircuit::simulate`]. This keeps the
//! fluent style while remaining panic-free and impossible to silently ignore.

use rand::{Rng, SeedableRng};
use ferriq_core::{Circuit, Gate, QuantumError, QubitId};
use ferriq_gates::standard::{
    CNot, CPhase, Fredkin, Hadamard, ISwap, Identity, PauliX, PauliY, PauliZ, Phase, RotationX,
    RotationY, RotationZ, SGate, SGateDagger, SXGate, SXGateDagger, Swap, TGate, TGateDagger,
    Toffoli, CY, CZ, ECR, RXX, RYY, RZZ, U1, U2, U3,
};
use ferriq_sim::{MeasurementCounts, SimulationResult, Simulator, SimulatorConfig, SimulatorError};
use ferriq_state::{measurement::ComputationalBasis, AdaptiveState, DenseState, PauliObservable};
use std::sync::Arc;

/// A quantum circuit under construction, with Qiskit-style gate methods
///
/// See the [module documentation](self) for the error-handling contract.
#[derive(Debug)]
pub struct QuantumCircuit {
    circuit: Circuit,
    error: Option<QuantumError>,
}

impl QuantumCircuit {
    /// Create a circuit on `num_qubits` qubits
    pub fn new(num_qubits: usize) -> Self {
        Self {
            circuit: Circuit::new(num_qubits),
            error: None,
        }
    }

    /// Create a circuit with pre-allocated capacity for `capacity` operations
    pub fn with_capacity(num_qubits: usize, capacity: usize) -> Self {
        Self {
            circuit: Circuit::with_capacity(num_qubits, capacity),
            error: None,
        }
    }

    /// Apply an arbitrary gate to the given qubit indices
    ///
    /// Escape hatch for gates without a dedicated method (custom gates,
    /// gates from [`ferriq_gates::custom`], ...).
    pub fn gate(&mut self, gate: Arc<dyn Gate>, qubits: &[usize]) -> &mut Self {
        if self.error.is_none() {
            let ids: Vec<QubitId> = qubits.iter().map(|&q| QubitId::new(q)).collect();
            if let Err(e) = self.circuit.add_gate(gate, &ids) {
                self.error = Some(e);
            }
        }
        self
    }

    // --- Single-qubit gates ---

    /// Hadamard gate
    pub fn h(&mut self, qubit: usize) -> &mut Self {
        self.gate(Arc::new(Hadamard), &[qubit])
    }

    /// Pauli-X (NOT) gate
    pub fn x(&mut self, qubit: usize) -> &mut Self {
        self.gate(Arc::new(PauliX), &[qubit])
    }

    /// Pauli-Y gate
    pub fn y(&mut self, qubit: usize) -> &mut Self {
        self.gate(Arc::new(PauliY), &[qubit])
    }

    /// Pauli-Z gate
    pub fn z(&mut self, qubit: usize) -> &mut Self {
        self.gate(Arc::new(PauliZ), &[qubit])
    }

    /// Identity gate
    pub fn id(&mut self, qubit: usize) -> &mut Self {
        self.gate(Arc::new(Identity), &[qubit])
    }

    /// S gate (√Z phase gate)
    pub fn s(&mut self, qubit: usize) -> &mut Self {
        self.gate(Arc::new(SGate), &[qubit])
    }

    /// S† gate
    pub fn sdg(&mut self, qubit: usize) -> &mut Self {
        self.gate(Arc::new(SGateDagger), &[qubit])
    }

    /// T gate (π/8 gate)
    pub fn t(&mut self, qubit: usize) -> &mut Self {
        self.gate(Arc::new(TGate), &[qubit])
    }

    /// T† gate
    pub fn tdg(&mut self, qubit: usize) -> &mut Self {
        self.gate(Arc::new(TGateDagger), &[qubit])
    }

    /// √X gate
    pub fn sx(&mut self, qubit: usize) -> &mut Self {
        self.gate(Arc::new(SXGate), &[qubit])
    }

    /// √X† gate
    pub fn sxdg(&mut self, qubit: usize) -> &mut Self {
        self.gate(Arc::new(SXGateDagger), &[qubit])
    }

    /// Rotation around the X-axis by `theta` radians
    pub fn rx(&mut self, theta: f64, qubit: usize) -> &mut Self {
        self.gate(Arc::new(RotationX::new(theta)), &[qubit])
    }

    /// Rotation around the Y-axis by `theta` radians
    pub fn ry(&mut self, theta: f64, qubit: usize) -> &mut Self {
        self.gate(Arc::new(RotationY::new(theta)), &[qubit])
    }

    /// Rotation around the Z-axis by `theta` radians
    pub fn rz(&mut self, theta: f64, qubit: usize) -> &mut Self {
        self.gate(Arc::new(RotationZ::new(theta)), &[qubit])
    }

    /// Phase gate: adds phase `theta` to |1⟩
    pub fn p(&mut self, theta: f64, qubit: usize) -> &mut Self {
        self.gate(Arc::new(Phase::new(theta)), &[qubit])
    }

    /// U1 gate (diagonal phase gate)
    pub fn u1(&mut self, lambda: f64, qubit: usize) -> &mut Self {
        self.gate(Arc::new(U1::new(lambda)), &[qubit])
    }

    /// U2 gate
    pub fn u2(&mut self, phi: f64, lambda: f64, qubit: usize) -> &mut Self {
        self.gate(Arc::new(U2::new(phi, lambda)), &[qubit])
    }

    /// U3 gate (generic single-qubit unitary)
    pub fn u3(&mut self, theta: f64, phi: f64, lambda: f64, qubit: usize) -> &mut Self {
        self.gate(Arc::new(U3::new(theta, phi, lambda)), &[qubit])
    }

    // --- Two-qubit gates ---

    /// Controlled-NOT gate
    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        self.gate(Arc::new(CNot), &[control, target])
    }

    /// Controlled-NOT gate (alias for [`cnot`](Self::cnot))
    pub fn cx(&mut self, control: usize, target: usize) -> &mut Self {
        self.cnot(control, target)
    }

    /// Controlled-Y gate
    pub fn cy(&mut self, control: usize, target: usize) -> &mut Self {
        self.gate(Arc::new(CY), &[control, target])
    }

    /// Controlled-Z gate
    pub fn cz(&mut self, control: usize, target: usize) -> &mut Self {
        self.gate(Arc::new(CZ), &[control, target])
    }

    /// Controlled-phase gate
    pub fn cp(&mut self, theta: f64, control: usize, target: usize) -> &mut Self {
        self.gate(Arc::new(CPhase::new(theta)), &[control, target])
    }

    /// SWAP gate
    pub fn swap(&mut self, a: usize, b: usize) -> &mut Self {
        self.gate(Arc::new(Swap), &[a, b])
    }

    /// iSWAP gate
    pub fn iswap(&mut self, a: usize, b: usize) -> &mut Self {
        self.gate(Arc::new(ISwap), &[a, b])
    }

    /// Echoed cross-resonance gate
    pub fn ecr(&mut self, a: usize, b: usize) -> &mut Self {
        self.gate(Arc::new(ECR), &[a, b])
    }

    /// XX-interaction rotation by `theta` radians
    pub fn rxx(&mut self, theta: f64, a: usize, b: usize) -> &mut Self {
        self.gate(Arc::new(RXX::new(theta)), &[a, b])
    }

    /// YY-interaction rotation by `theta` radians
    pub fn ryy(&mut self, theta: f64, a: usize, b: usize) -> &mut Self {
        self.gate(Arc::new(RYY::new(theta)), &[a, b])
    }

    /// ZZ-interaction rotation by `theta` radians
    pub fn rzz(&mut self, theta: f64, a: usize, b: usize) -> &mut Self {
        self.gate(Arc::new(RZZ::new(theta)), &[a, b])
    }

    // --- Three-qubit gates ---

    /// Toffoli (CCX) gate
    pub fn toffoli(&mut self, control1: usize, control2: usize, target: usize) -> &mut Self {
        self.gate(Arc::new(Toffoli), &[control1, control2, target])
    }

    /// Toffoli gate (alias for [`toffoli`](Self::toffoli))
    pub fn ccx(&mut self, control1: usize, control2: usize, target: usize) -> &mut Self {
        self.toffoli(control1, control2, target)
    }

    /// Fredkin (controlled-SWAP) gate
    pub fn cswap(&mut self, control: usize, a: usize, b: usize) -> &mut Self {
        self.gate(Arc::new(Fredkin), &[control, a, b])
    }

    // --- Inspection ---

    /// Number of qubits in the circuit
    pub fn num_qubits(&self) -> usize {
        self.circuit.num_qubits()
    }

    /// Number of gate operations added so far
    pub fn len(&self) -> usize {
        self.circuit.len()
    }

    /// Whether the circuit contains no operations
    pub fn is_empty(&self) -> bool {
        self.circuit.is_empty()
    }

    /// Circuit depth (longest path of dependent gates)
    pub fn depth(&self) -> usize {
        self.circuit.depth()
    }

    /// The first error encountered while building, if any
    pub fn error(&self) -> Option<&QuantumError> {
        self.error.as_ref()
    }

    /// Borrow the underlying [`Circuit`] (ignores any deferred error)
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    /// Render the circuit as an ASCII diagram
    pub fn to_ascii(&self) -> String {
        self.circuit.to_ascii()
    }

    // --- Finalization ---

    /// Finish building and return the underlying [`Circuit`]
    ///
    /// Returns the first error recorded during building, if any.
    pub fn build(self) -> Result<Circuit, QuantumError> {
        match self.error {
            Some(e) => Err(e),
            None => Ok(self.circuit),
        }
    }

    /// Simulate the circuit with default simulator settings (1024 shots)
    ///
    /// The returned [`SimulationResult`] contains both the final state vector
    /// and sampled measurement counts (`result.measurements`).
    pub fn simulate(&self) -> Result<SimulationResult, SimulatorError> {
        self.simulate_with_config(SimulatorConfig::default())
    }

    /// Simulate the circuit with the given number of measurement shots
    pub fn simulate_with_shots(&self, shots: usize) -> Result<SimulationResult, SimulatorError> {
        self.simulate_with_config(SimulatorConfig::default().with_shots(shots))
    }

    /// Simulate the circuit with a custom [`SimulatorConfig`]
    ///
    /// Measurement counts are sampled from the final state using
    /// `config.shots` shots (and `config.seed` for reproducibility).
    pub fn simulate_with_config(
        &self,
        config: SimulatorConfig,
    ) -> Result<SimulationResult, SimulatorError> {
        if let Some(e) = &self.error {
            return Err(SimulatorError::InvalidCircuit(e.to_string()));
        }
        let shots = config.shots;
        let seed = config.seed;
        let mut result = Simulator::new(config).run(&self.circuit)?;
        if shots > 0 {
            result.measurements = Some(sample_counts(&result.state, shots, seed)?);
        }
        Ok(result)
    }

    /// Simulate and compute the expectation value ⟨ψ|O|ψ⟩ of an observable
    ///
    /// This is the core VQE/QAOA primitive: run the circuit (statevector,
    /// no measurement sampling) and evaluate the observable on the final
    /// state exactly.
    ///
    /// ```
    /// use ferriq::{QuantumCircuit, PauliObservable, PauliString};
    ///
    /// let mut qc = QuantumCircuit::new(2);
    /// qc.h(0).cnot(0, 1);
    ///
    /// // ⟨Z₀Z₁⟩ = 1 for the Bell state (|00⟩ + |11⟩)/√2
    /// let zz = PauliObservable::from_pauli_string(
    ///     PauliString::from_str("ZZ").unwrap(), 1.0);
    /// let value = qc.expectation_value(&zz).unwrap();
    /// assert!((value - 1.0).abs() < 1e-10);
    /// ```
    pub fn expectation_value(&self, observable: &PauliObservable) -> Result<f64, SimulatorError> {
        if let Some(e) = &self.error {
            return Err(SimulatorError::InvalidCircuit(e.to_string()));
        }
        let result = Simulator::new(SimulatorConfig::default()).run(&self.circuit)?;
        let dense =
            DenseState::from_amplitudes(result.state.num_qubits(), &result.state.to_dense_vec())
                .map_err(|e| SimulatorError::StateError {
                    message: e.to_string(),
                })?;
        observable
            .expectation_value(&dense)
            .map_err(|e| SimulatorError::StateError {
                message: e.to_string(),
            })
    }

    /// Simulate and return the exact probability of each basis state
    ///
    /// Returns a vector of length 2^n where entry `i` is |⟨i|ψ⟩|². Useful
    /// for verifying results against theory without sampling noise.
    pub fn probabilities(&self) -> Result<Vec<f64>, SimulatorError> {
        if let Some(e) = &self.error {
            return Err(SimulatorError::InvalidCircuit(e.to_string()));
        }
        let result = Simulator::new(SimulatorConfig::default()).run(&self.circuit)?;
        Ok(result
            .state
            .to_dense_vec()
            .iter()
            .map(|amp| amp.norm_sqr())
            .collect())
    }
}

/// Sample computational-basis measurement counts from a final state
fn sample_counts(
    state: &AdaptiveState,
    shots: usize,
    seed: Option<u64>,
) -> Result<MeasurementCounts, SimulatorError> {
    let num_qubits = state.num_qubits();
    let amplitudes = state.to_dense_vec();
    let dense = DenseState::from_amplitudes(num_qubits, &amplitudes)
        .map_err(|e| SimulatorError::MeasurementFailed(e.to_string()))?;

    let mut rng: rand::rngs::StdRng = match seed {
        Some(s) => SeedableRng::seed_from_u64(s),
        None => SeedableRng::from_entropy(),
    };
    let mut rng_fn = || rng.gen::<f64>();

    let sampling = ComputationalBasis::new()
        .with_collapse(false)
        .sample(&dense, shots, &mut rng_fn)
        .map_err(|e| SimulatorError::MeasurementFailed(e.to_string()))?;

    Ok(MeasurementCounts::from_counts(sampling.to_bitstring_counts(num_qubits)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bell_state_fluent() {
        let mut qc = QuantumCircuit::new(2);
        qc.h(0).cnot(0, 1);

        assert_eq!(qc.len(), 2);
        assert!(qc.error().is_none());

        let result = qc.simulate_with_shots(4096).unwrap();
        let counts = result.measurements.expect("measurements present");

        // Only |00⟩ and |11⟩ should appear, in roughly equal proportion
        assert_eq!(counts.get("01"), 0);
        assert_eq!(counts.get("10"), 0);
        let p00 = counts.probability("00");
        let p11 = counts.probability("11");
        assert!((p00 - 0.5).abs() < 0.1, "p00 = {p00}");
        assert!((p11 - 0.5).abs() < 0.1, "p11 = {p11}");
    }

    #[test]
    fn all_gate_methods_build() {
        let mut qc = QuantumCircuit::new(3);
        qc.h(0)
            .x(1)
            .y(2)
            .z(0)
            .id(1)
            .s(0)
            .sdg(0)
            .t(1)
            .tdg(1)
            .sx(2)
            .sxdg(2)
            .rx(0.3, 0)
            .ry(0.4, 1)
            .rz(0.5, 2)
            .p(0.6, 0)
            .u1(0.7, 1)
            .u2(0.1, 0.2, 2)
            .u3(0.1, 0.2, 0.3, 0)
            .cnot(0, 1)
            .cx(1, 2)
            .cy(0, 2)
            .cz(1, 0)
            .cp(0.8, 0, 1)
            .swap(1, 2)
            .iswap(0, 1)
            .ecr(1, 2)
            .rxx(0.2, 0, 1)
            .ryy(0.3, 1, 2)
            .rzz(0.4, 0, 2)
            .toffoli(0, 1, 2)
            .ccx(2, 1, 0)
            .cswap(0, 1, 2);

        let circuit = qc.build().unwrap();
        assert_eq!(circuit.len(), 32);
    }

    #[test]
    fn invalid_qubit_defers_error() {
        let mut qc = QuantumCircuit::new(2);
        qc.h(0).cnot(0, 5).x(1); // qubit 5 out of range

        assert!(qc.error().is_some());
        // Gates after the error are not added
        assert_eq!(qc.len(), 1);
        assert!(qc.simulate().is_err());
        assert!(qc.build().is_err());
    }

    #[test]
    fn expectation_value_matches_theory() {
        use ferriq_state::PauliString;

        // RX(0.8)|0⟩: ⟨Z⟩ = cos(0.8). Uses an angle in the range issue #37
        // used to corrupt, so this doubles as an accuracy check through the
        // whole facade → simulator → gate-cache stack.
        let theta = 0.8;
        let mut qc = QuantumCircuit::new(1);
        qc.rx(theta, 0);

        let z = PauliObservable::from_pauli_string(PauliString::from_str("Z").unwrap(), 1.0);
        let value = qc.expectation_value(&z).unwrap();
        assert!(
            (value - theta.cos()).abs() < 1e-10,
            "⟨Z⟩ = {value}, expected cos(0.8) = {}",
            theta.cos()
        );
    }

    #[test]
    fn probabilities_match_theory() {
        let mut qc = QuantumCircuit::new(2);
        qc.h(0).cnot(0, 1);

        let probs = qc.probabilities().unwrap();
        assert_eq!(probs.len(), 4);
        assert!((probs[0] - 0.5).abs() < 1e-10); // |00⟩
        assert!(probs[1].abs() < 1e-10); // |01⟩
        assert!(probs[2].abs() < 1e-10); // |10⟩
        assert!((probs[3] - 0.5).abs() < 1e-10); // |11⟩
    }

    #[test]
    fn expectation_value_propagates_build_error() {
        use ferriq_state::PauliString;

        let mut qc = QuantumCircuit::new(1);
        qc.h(9);
        let z = PauliObservable::from_pauli_string(PauliString::from_str("Z").unwrap(), 1.0);
        assert!(qc.expectation_value(&z).is_err());
        assert!(qc.probabilities().is_err());
    }

    #[test]
    fn seeded_simulation_is_reproducible() {
        let mut qc = QuantumCircuit::new(2);
        qc.h(0).cnot(0, 1);

        let config = || SimulatorConfig::default().with_shots(256).with_seed(42);
        let a = qc.simulate_with_config(config()).unwrap();
        let b = qc.simulate_with_config(config()).unwrap();
        assert_eq!(a.measurements.unwrap(), b.measurements.unwrap());
    }

    #[test]
    fn custom_gate_escape_hatch() {
        let mut qc = QuantumCircuit::new(2);
        qc.gate(Arc::new(Hadamard), &[1]);
        assert_eq!(qc.len(), 1);
    }

    #[test]
    fn inspection_methods() {
        let mut qc = QuantumCircuit::with_capacity(2, 8);
        assert!(qc.is_empty());
        qc.h(0).cnot(0, 1);
        assert!(!qc.is_empty());
        assert_eq!(qc.num_qubits(), 2);
        assert_eq!(qc.depth(), 2);
        assert_eq!(qc.circuit().len(), 2);
        assert!(!qc.to_ascii().is_empty());
    }
}
