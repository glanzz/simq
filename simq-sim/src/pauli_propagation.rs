//! Pauli-propagation observable engine (Heisenberg-picture expectation values).
//!
//! `Simulator::run` + `PauliObservable::expectation_value` compute ⟨ψ|O|ψ⟩ by
//! materializing the full `2^n`-amplitude state `|ψ⟩ = U|0...0⟩` — but a
//! VQE/QAOA energy evaluation only ever *reads* that state through a handful
//! of Pauli-string observables, never the raw amplitudes. This module
//! computes the same expectation value without ever building a statevector,
//! by evolving the *observable* backward through the circuit instead of
//! evolving the *state* forward through it (the Heisenberg picture):
//!
//! ```text
//! ⟨ψ|O|ψ⟩ = ⟨0| U_1^† U_2^† ... U_L^† O U_L ... U_2 U_1 |0⟩
//! ```
//!
//! Starting from `O` and conjugating by each gate from last-applied to
//! first-applied (`O <- U^† O U`) produces `⟨0|O_final|0⟩`, which — because
//! the initial state is always `|0...0⟩` in this crate — reduces to summing
//! the coefficients of every surviving Pauli term that contains no `X`/`Y`
//! (an `X` or `Y` anywhere makes `⟨0...0|P|0...0⟩ = 0`).
//!
//! # Why this is tractable at all
//!
//! A Pauli string conjugated by a **Clifford** gate maps to exactly one
//! other Pauli string (with a real +-1 coefficient) — the term count never
//! grows. A **non-Clifford** gate (any rotation, T, ...) generally splits one
//! term into up to `dim` new terms (`dim = 2^k` for a `k`-qubit gate), so the
//! term count can grow with the number of non-Clifford gates. This is the
//! same trade-off the research write-up this module implements describes:
//! polynomial for near-Clifford circuits, exponential in the worst case.
//! [`PropagationConfig`] bounds the blow-up by dropping negligible-magnitude
//! terms after every gate (a term with coefficient `c` can change `⟨O⟩` by
//! at most `|c|`, since `|⟨ψ|P|ψ⟩| <= 1` for any Pauli string `P` and unit
//! `|ψ⟩` — so the sum of dropped `|c|` is a rigorous upper bound on the
//! introduced error, returned as `truncation_error_bound`).
//!
//! # Implementation: matrix conjugation, not hand-derived gate rules
//!
//! Rather than hand-deriving a conjugation rule per gate name (as
//! `simq_sim::stabilizer` does for its fixed Clifford set), this module
//! conjugates generically via each gate's [`simq_core::gate::Gate::matrix`]:
//! build the local Pauli operator as a `2^k x 2^k` matrix, compute
//! `U^† P U`, and decompose the result back into the Pauli basis via the
//! trace formula `c_Q = Tr(Q · M) / 2^k` (valid because Pauli tensor
//! products are Hermitian and pairwise trace-orthogonal). This works
//! uniformly for Clifford and non-Clifford, 1- and 2-qubit gates alike, and
//! needs no per-gate special-casing — the trade-off is that gates with more
//! than 2 qubits, or without a `matrix()`, are out of scope (see
//! [`PauliPropagationError::UnsupportedGate`]).

use num_complex::Complex64;
use simq_core::Circuit;
use simq_state::{Pauli, PauliObservable};
use std::collections::HashMap;

type C = Complex64;

/// A Pauli string in sparse form: `(qubit, Pauli)` pairs for every
/// non-identity qubit, sorted ascending by qubit index. Two keys compare
/// equal iff they represent the same Pauli string, so this doubles as the
/// `HashMap` key that merges equal terms produced by different branches.
type SparseKey = Vec<(u32, Pauli)>;

fn local_pauli(key: &SparseKey, qubit: u32) -> Pauli {
    key.iter()
        .find(|(q, _)| *q == qubit)
        .map(|(_, p)| *p)
        .unwrap_or(Pauli::I)
}

fn rekeyed(key: &SparseKey, gate_qubits: &[u32], new_local: &[Pauli]) -> SparseKey {
    let mut out: SparseKey = key
        .iter()
        .copied()
        .filter(|(q, _)| !gate_qubits.contains(q))
        .collect();
    for (&q, &p) in gate_qubits.iter().zip(new_local) {
        if p != Pauli::I {
            out.push((q, p));
        }
    }
    out.sort_unstable_by_key(|(q, _)| *q);
    out
}

const ALL_PAULIS: [Pauli; 4] = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];

/// Every length-`k` combination of {I,X,Y,Z}, most-significant slot first —
/// `4^k` of them, matching the number of independent Pauli basis elements
/// on `k` qubits.
fn all_combos(k: usize) -> Vec<Vec<Pauli>> {
    let mut combos = vec![Vec::new()];
    for _ in 0..k {
        let mut next = Vec::with_capacity(combos.len() * 4);
        for c in &combos {
            for &p in &ALL_PAULIS {
                let mut nc = c.clone();
                nc.push(p);
                next.push(nc);
            }
        }
        combos = next;
    }
    combos
}

/// Flattened row-major 2x2 matrix for a single-qubit Pauli. Shared with
/// [`crate::mps`], which needs the same convention to apply Pauli strings
/// to an MPS for expectation-value computation.
pub(crate) fn pauli_matrix_2x2(p: Pauli) -> [C; 4] {
    let z = C::new(0.0, 0.0);
    let o = C::new(1.0, 0.0);
    let i = C::new(0.0, 1.0);
    match p {
        Pauli::I => [o, z, z, o],
        Pauli::X => [z, o, o, z],
        Pauli::Y => [z, -i, i, z],
        Pauli::Z => [o, z, z, -o],
    }
}

/// `kron(paulis[0], paulis[1], ...)`, flattened row-major, `paulis[0]` the
/// most-significant tensor slot — matching this crate's convention that
/// `op.qubits()[0]` is the more-significant bit of a multi-qubit gate
/// matrix (see `simq-sim/src/execution_engine/kernels/two_qubit.rs`).
fn kron_paulis(paulis: &[Pauli]) -> Vec<C> {
    let mut result = vec![C::new(1.0, 0.0)];
    let mut dim = 1usize;
    for &p in paulis {
        let m = pauli_matrix_2x2(p);
        let new_dim = dim * 2;
        let mut next = vec![C::new(0.0, 0.0); new_dim * new_dim];
        for i in 0..dim {
            for j in 0..dim {
                let a = result[i * dim + j];
                if a == C::new(0.0, 0.0) {
                    continue;
                }
                for pi in 0..2 {
                    for pj in 0..2 {
                        next[(i * 2 + pi) * new_dim + (j * 2 + pj)] = a * m[pi * 2 + pj];
                    }
                }
            }
        }
        result = next;
        dim = new_dim;
    }
    result
}

fn matmul(a: &[C], b: &[C], dim: usize) -> Vec<C> {
    let mut out = vec![C::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for k in 0..dim {
            let aik = a[i * dim + k];
            if aik == C::new(0.0, 0.0) {
                continue;
            }
            for j in 0..dim {
                out[i * dim + j] += aik * b[k * dim + j];
            }
        }
    }
    out
}

fn conjugate_transpose(a: &[C], dim: usize) -> Vec<C> {
    let mut out = vec![C::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            out[j * dim + i] = a[i * dim + j].conj();
        }
    }
    out
}

/// `Tr(a * b)`, both `dim x dim`, flattened row-major.
fn trace_product(a: &[C], b: &[C], dim: usize) -> C {
    let mut s = C::new(0.0, 0.0);
    for i in 0..dim {
        for k in 0..dim {
            s += a[i * dim + k] * b[k * dim + i];
        }
    }
    s
}

/// Configuration for [`expectation_value_with_config`].
#[derive(Debug, Clone, Copy)]
pub struct PropagationConfig {
    /// Drop any term with `|coefficient| < truncation_threshold` after every
    /// gate. Its magnitude is added to `truncation_error_bound` on the
    /// result (see the module docs for why that sum is a rigorous bound).
    pub truncation_threshold: f64,
    /// Hard cap on the number of live terms. If exceeded after a gate, the
    /// smallest-magnitude terms are dropped (contributing to
    /// `truncation_error_bound`) until the cap is met — a safety valve
    /// against pathological blow-up independent of `truncation_threshold`.
    pub max_terms: usize,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            truncation_threshold: 1e-10,
            max_terms: 20_000,
        }
    }
}

/// Result of a Pauli-propagation expectation-value computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropagationResult {
    /// The (possibly truncation-approximated) expectation value.
    pub expectation: f64,
    /// Rigorous upper bound on the error `truncation_threshold`/`max_terms`
    /// may have introduced (0.0 if nothing was ever dropped, i.e. exact).
    pub truncation_error_bound: f64,
    /// Largest live term count seen during propagation.
    pub peak_terms: usize,
}

/// Error returned when a circuit contains something this engine can't
/// propagate through, or an observable that doesn't match the circuit.
#[derive(Debug, Clone, PartialEq)]
pub enum PauliPropagationError {
    /// `gate` acts on `num_qubits` qubits, or has no [`Gate::matrix`]
    /// representation — this engine only conjugates through 1- and 2-qubit
    /// gates with an explicit matrix.
    ///
    /// [`Gate::matrix`]: simq_core::gate::Gate::matrix
    UnsupportedGate { gate: String, num_qubits: usize },
    /// A term in the observable doesn't have one Pauli per circuit qubit.
    ObservableQubitMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for PauliPropagationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PauliPropagationError::UnsupportedGate { gate, num_qubits } => {
                write!(
                    f,
                    "gate '{gate}' ({num_qubits} qubits) is not supported by Pauli propagation \
                     (needs a 1- or 2-qubit gate with a matrix representation)"
                )
            },
            PauliPropagationError::ObservableQubitMismatch { expected, actual } => {
                write!(f, "observable has {actual} qubits, circuit has {expected}")
            },
        }
    }
}

impl std::error::Error for PauliPropagationError {}

/// Compute `⟨0...0|U^† O U|0...0⟩` for `circuit` (which prepares
/// `U|0...0⟩`) and observable `O`, without ever building a statevector. See
/// the module docs for the algorithm and [`PropagationConfig::default`] for
/// the truncation settings used.
pub fn expectation_value(
    circuit: &Circuit,
    observable: &PauliObservable,
) -> Result<PropagationResult, PauliPropagationError> {
    expectation_value_with_config(circuit, observable, &PropagationConfig::default())
}

/// Same as [`expectation_value`] with an explicit [`PropagationConfig`].
pub fn expectation_value_with_config(
    circuit: &Circuit,
    observable: &PauliObservable,
    config: &PropagationConfig,
) -> Result<PropagationResult, PauliPropagationError> {
    let n = circuit.num_qubits();
    let mut terms: HashMap<SparseKey, f64> = HashMap::new();
    for (pauli_string, coeff) in observable.terms() {
        if pauli_string.num_qubits() != n {
            return Err(PauliPropagationError::ObservableQubitMismatch {
                expected: n,
                actual: pauli_string.num_qubits(),
            });
        }
        let mut key = SparseKey::new();
        for q in 0..n {
            if let Some(p) = pauli_string.get(q) {
                if p != Pauli::I {
                    key.push((q as u32, p));
                }
            }
        }
        key.sort_unstable_by_key(|(q, _)| *q);
        *terms.entry(key).or_insert(0.0) += coeff * pauli_string.coeff() as f64;
    }

    let mut truncation_error_bound = 0.0;
    let mut peak_terms = terms.len();

    for op in circuit.operations_slice().iter().rev() {
        let gate = op.gate();
        let qubits: Vec<u32> = op.qubits().iter().map(|q| q.index() as u32).collect();
        let k = qubits.len();
        let dim = 1usize << k;
        let u = gate
            .matrix()
            .filter(|m| (k == 1 || k == 2) && m.len() == dim * dim)
            .ok_or_else(|| PauliPropagationError::UnsupportedGate {
                gate: gate.name().to_string(),
                num_qubits: k,
            })?;
        let u_dag = conjugate_transpose(&u, dim);
        let combos = all_combos(k);

        let mut new_terms: HashMap<SparseKey, f64> = HashMap::new();
        for (key, &coeff) in &terms {
            let local: Vec<Pauli> = qubits.iter().map(|&q| local_pauli(key, q)).collect();
            let p_mat = kron_paulis(&local);
            let m = matmul(&matmul(&u_dag, &p_mat, dim), &u, dim);

            for combo in &combos {
                let b_mat = kron_paulis(combo);
                let c = trace_product(&b_mat, &m, dim).re / dim as f64;
                if c.abs() < 1e-14 {
                    continue;
                }
                let new_key = rekeyed(key, &qubits, combo);
                *new_terms.entry(new_key).or_insert(0.0) += coeff * c;
            }
        }

        terms = HashMap::with_capacity(new_terms.len());
        for (key, coeff) in new_terms {
            if coeff.abs() < config.truncation_threshold {
                truncation_error_bound += coeff.abs();
            } else {
                terms.insert(key, coeff);
            }
        }

        if terms.len() > config.max_terms {
            // `select_nth_unstable_by` partitions around the cutoff in
            // O(n) average instead of a full O(n log n) sort — the only
            // thing this needs is "which terms are in the top
            // `max_terms`", not a total order.
            let mut ranked: Vec<(SparseKey, f64)> = terms.into_iter().collect();
            ranked.select_nth_unstable_by(config.max_terms, |a, b| {
                b.1.abs().partial_cmp(&a.1.abs()).unwrap()
            });
            for (_, coeff) in &ranked[config.max_terms..] {
                truncation_error_bound += coeff.abs();
            }
            ranked.truncate(config.max_terms);
            terms = ranked.into_iter().collect();
        }

        peak_terms = peak_terms.max(terms.len());
    }

    // <0...0| P |0...0> is 1 if P has no X/Y (I/Z entries are stored as "no
    // entry"/Z, both eigenvalue +1 on |0>), else 0.
    let expectation: f64 = terms
        .iter()
        .filter(|(key, _)| key.iter().all(|(_, p)| *p == Pauli::Z))
        .map(|(_, coeff)| coeff)
        .sum();

    Ok(PropagationResult {
        expectation,
        truncation_error_bound,
        peak_terms,
    })
}

/// Which backend [`recommend_backend`] suggests for an expectation-value
/// computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedBackend {
    /// Use Pauli propagation ([`expectation_value`]).
    PauliPropagation,
    /// Use the statevector simulator + [`PauliObservable::expectation_value`].
    Statevector,
}

/// Qubit count above which materializing a statevector starts to dominate
/// runtime/memory on the reference machine documented in `BENCHMARKS.md`
/// (well under the ~30-qubit hard wall, since this is a "prefer the cheaper
/// path" threshold, not a capability limit).
const STATEVECTOR_COMFORTABLE_QUBITS: usize = 20;

/// Heuristic cost model choosing between Pauli propagation and the
/// statevector simulator for an expectation-value computation.
///
/// A Clifford gate maps one Pauli term to exactly one other term (no
/// branching); only non-Clifford gates can multiply the live term count
/// (by at most `dim = 2^k` per gate, and never past `4^n`, the total size
/// of the Pauli basis). `2^(non_clifford_gate_count)` is therefore a valid
/// — if loose — upper bound on how many terms propagation will ever carry.
/// Propagation is recommended when the circuit is past the statevector
/// comfort zone *and* that bound stays small enough to be cheap; otherwise
/// the statevector path (exact, and not worse than exponential-in-qubits
/// either) is recommended.
pub fn recommend_backend(circuit: &Circuit) -> RecommendedBackend {
    let n = circuit.num_qubits();
    if n <= STATEVECTOR_COMFORTABLE_QUBITS {
        return RecommendedBackend::Statevector;
    }
    let non_clifford = circuit
        .operations()
        .filter(|op| !crate::stabilizer::is_clifford_gate_name(op.gate().name()))
        .count();
    // Cap the exponent so this never overflows regardless of circuit size.
    let log2_term_bound = non_clifford.min(STATEVECTOR_COMFORTABLE_QUBITS);
    if log2_term_bound < STATEVECTOR_COMFORTABLE_QUBITS {
        RecommendedBackend::PauliPropagation
    } else {
        RecommendedBackend::Statevector
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use simq_core::QubitId;
    use simq_gates::standard::{CNot, Hadamard, RotationX, RotationY, RotationZ, Toffoli};
    use simq_state::{DenseState, PauliString};
    use std::sync::Arc;

    fn qc(n: usize) -> Circuit {
        Circuit::new(n)
    }

    fn dense_expectation(circuit: &Circuit, observable: &PauliObservable) -> f64 {
        let unitary = simq_gates::matrix_ops::circuit_matrix(circuit).unwrap();
        let dim = 1usize << circuit.num_qubits();
        let amps: Vec<C> = (0..dim).map(|row| unitary[row * dim]).collect();
        let state = DenseState::from_amplitudes(circuit.num_qubits(), &amps).unwrap();
        observable.expectation_value(&state).unwrap()
    }

    #[test]
    fn matches_statevector_on_ghz_clifford_circuit() {
        // Clifford-only: propagation never branches, so this must be exact.
        let n = 5;
        let mut c = qc(n);
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
        for q in 0..n - 1 {
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
                .unwrap();
        }

        let mut obs = PauliObservable::new();
        obs.add_term(PauliString::all_z(n), 1.0);
        obs.add_term(
            PauliString::from_paulis(vec![Pauli::Z, Pauli::Z, Pauli::I, Pauli::I, Pauli::I]),
            0.5,
        );

        let expected = dense_expectation(&c, &obs);
        let got = expectation_value(&c, &obs).unwrap();
        assert_relative_eq!(got.expectation, expected, epsilon = 1e-9);
        assert_eq!(got.truncation_error_bound, 0.0);
    }

    #[test]
    fn matches_statevector_on_vqe_style_circuit_with_rotations() {
        // Non-Clifford RY/RZ rotations force real term branching; with
        // truncation effectively off, this still must match the exact
        // statevector expectation value.
        let n = 4;
        let mut c = qc(n);
        for q in 0..n {
            c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
        }
        for q in 0..n {
            c.add_gate(Arc::new(RotationY::new(0.3 + 0.2 * q as f64)), &[QubitId::new(q)])
                .unwrap();
        }
        for q in 0..n - 1 {
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
                .unwrap();
        }
        for q in 0..n {
            c.add_gate(Arc::new(RotationZ::new(0.1 + 0.15 * q as f64)), &[QubitId::new(q)])
                .unwrap();
        }

        let mut obs = PauliObservable::new();
        for q in 0..n - 1 {
            let mut paulis = vec![Pauli::I; n];
            paulis[q] = Pauli::Z;
            paulis[q + 1] = Pauli::Z;
            obs.add_term(PauliString::from_paulis(paulis), 1.0);
        }
        for q in 0..n {
            let mut paulis = vec![Pauli::I; n];
            paulis[q] = Pauli::X;
            obs.add_term(PauliString::from_paulis(paulis), 0.5);
        }

        let expected = dense_expectation(&c, &obs);
        let config = PropagationConfig {
            truncation_threshold: 0.0,
            max_terms: 1_000_000,
        };
        let got = expectation_value_with_config(&c, &obs, &config).unwrap();
        assert_relative_eq!(got.expectation, expected, epsilon = 1e-9);
        assert!(got.peak_terms > 1, "rotations should have branched at least once");
    }

    #[test]
    fn matches_statevector_with_rx_rotation_too() {
        // RX is the Y/Z-mixing rotation (as opposed to RY's X/Z mix and
        // RZ's X/Y mix), exercising a different pair of conjugation
        // branches than the previous test.
        let n = 2;
        let mut c = qc(n);
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
        c.add_gate(Arc::new(RotationX::new(0.7)), &[QubitId::new(1)])
            .unwrap();
        c.add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();

        let mut obs = PauliObservable::new();
        obs.add_term(PauliString::from_str("YY").unwrap(), 1.0);
        obs.add_term(PauliString::from_str("ZI").unwrap(), 0.3);

        let expected = dense_expectation(&c, &obs);
        let got = expectation_value(&c, &obs).unwrap();
        assert_relative_eq!(got.expectation, expected, epsilon = 1e-9);
    }

    #[test]
    fn truncation_bounds_the_error() {
        let n = 4;
        let mut c = qc(n);
        for q in 0..n {
            c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
            c.add_gate(Arc::new(RotationY::new(0.4 + 0.1 * q as f64)), &[QubitId::new(q)])
                .unwrap();
        }
        for q in 0..n - 1 {
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
                .unwrap();
        }

        let mut obs = PauliObservable::new();
        obs.add_term(PauliString::all_z(n), 1.0);

        let exact = expectation_value_with_config(
            &c,
            &obs,
            &PropagationConfig {
                truncation_threshold: 0.0,
                max_terms: 1_000_000,
            },
        )
        .unwrap();
        let truncated = expectation_value_with_config(
            &c,
            &obs,
            &PropagationConfig {
                truncation_threshold: 0.05,
                max_terms: 1_000_000,
            },
        )
        .unwrap();

        assert!(truncated.peak_terms <= exact.peak_terms);
        let actual_error = (truncated.expectation - exact.expectation).abs();
        assert!(
            actual_error <= truncated.truncation_error_bound + 1e-12,
            "actual error {actual_error} exceeded its own bound {}",
            truncated.truncation_error_bound
        );
    }

    #[test]
    fn unsupported_three_qubit_gate_is_rejected() {
        let mut c = qc(3);
        c.add_gate(Arc::new(Toffoli), &[QubitId::new(0), QubitId::new(1), QubitId::new(2)])
            .unwrap();
        let obs = PauliObservable::from_pauli_string(PauliString::all_z(3), 1.0);

        let err = expectation_value(&c, &obs).unwrap_err();
        assert_eq!(
            err,
            PauliPropagationError::UnsupportedGate {
                gate: "CCNOT".to_string(),
                num_qubits: 3
            }
        );
    }

    #[test]
    fn observable_qubit_mismatch_is_rejected() {
        let c = qc(3);
        let obs = PauliObservable::from_pauli_string(PauliString::all_z(2), 1.0);
        let err = expectation_value(&c, &obs).unwrap_err();
        assert_eq!(
            err,
            PauliPropagationError::ObservableQubitMismatch {
                expected: 3,
                actual: 2
            }
        );
    }

    #[test]
    fn recommend_backend_prefers_statevector_below_threshold() {
        let c = qc(4);
        assert_eq!(recommend_backend(&c), RecommendedBackend::Statevector);
    }

    #[test]
    fn recommend_backend_prefers_propagation_for_wide_near_clifford_circuit() {
        let mut c = qc(30);
        for q in 0..30 {
            c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
        }
        for q in 0..29 {
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
                .unwrap();
        }
        assert_eq!(recommend_backend(&c), RecommendedBackend::PauliPropagation);
    }

    #[test]
    fn recommend_backend_falls_back_to_statevector_when_too_many_non_clifford_gates() {
        let mut c = qc(30);
        for q in 0..30 {
            c.add_gate(Arc::new(RotationY::new(0.37)), &[QubitId::new(q)])
                .unwrap();
        }
        assert_eq!(recommend_backend(&c), RecommendedBackend::Statevector);
    }

    #[test]
    fn display_impls_are_human_readable() {
        let e1 = PauliPropagationError::UnsupportedGate {
            gate: "CCNOT".to_string(),
            num_qubits: 3,
        };
        assert!(e1.to_string().contains("CCNOT"));
        let e2 = PauliPropagationError::ObservableQubitMismatch {
            expected: 3,
            actual: 2,
        };
        assert!(e2.to_string().contains('3') && e2.to_string().contains('2'));
    }
}
