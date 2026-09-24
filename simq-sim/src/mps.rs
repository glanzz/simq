//! Matrix-product-state (MPS) backend for near-1D, low-entanglement circuits.
//!
//! A statevector needs `2^n` amplitudes regardless of what the circuit does.
//! An MPS instead represents the state as a chain of `n` tensors joined by
//! "bond" indices whose dimension only needs to grow with the circuit's
//! *entanglement*, not its qubit count — a circuit whose two-qubit gates
//! stay local (a 1D chain or ring, e.g. this crate's own VQE ansatz and QAOA
//! MaxCut benchmark circuits) can often be represented exactly, or to a
//! tight numerical tolerance, at a bond dimension far smaller than `2^n`.
//! This unlocks qubit counts a statevector can never reach for that
//! circuit family, at the cost of no longer being exact for circuits that
//! *do* entangle across the whole chain (a bond-dimension cap turns this
//! into a controlled approximation there — see [`MpsConfig`]).
//!
//! # Layout
//!
//! Tensor `i` has shape `(left_bond, 2, right_bond)`; boundary tensors have
//! bond dimension 1. A two-qubit gate on *adjacent* sites is applied by
//! contracting the pair into one `(left_bond, 2, 2, right_bond)` tensor,
//! applying the gate, reshaping to a matrix, and re-splitting via SVD
//! (`nalgebra`, which implements the complex SVD natively — no external
//! BLAS/LAPACK needed), truncating to [`MpsConfig::max_bond_dim`] or to
//! [`MpsConfig::truncation_cutoff`], whichever is smaller.
//!
//! A gate on *non-adjacent* qubits (e.g. the one wraparound edge in a QAOA
//! ring ansatz) is handled by a swap network: adjacent-site SWAP gates move
//! the two qubits together, the gate is applied, and — deliberately, to
//! keep this module's scope bounded — the chain is *not* swapped back
//! afterwards. `qubit_to_site`/`site_to_qubit` track the resulting
//! permutation so every later operation still resolves the right physical
//! site for a given qubit; this is correct in general, just not
//! swap-count-optimal for circuits with many long-range gates (which are
//! exactly the circuits this backend is the wrong tool for anyway — see
//! [`is_1d_candidate`]).

use nalgebra::DMatrix;
use num_complex::Complex64;
use simq_core::gate::Gate as _;
use simq_core::Circuit;
use simq_gates::standard::Swap;
use simq_state::{Pauli, PauliObservable, PauliString};

type C = Complex64;

fn czero() -> C {
    C::new(0.0, 0.0)
}

/// One MPS tensor, flattened row-major as `(left, physical=2, right)`.
#[derive(Debug, Clone)]
struct Tensor {
    dl: usize,
    dr: usize,
    data: Vec<C>,
}

impl Tensor {
    fn product_state_zero() -> Self {
        Self {
            dl: 1,
            dr: 1,
            data: vec![C::new(1.0, 0.0), czero()],
        }
    }

    #[inline]
    fn get(&self, l: usize, p: usize, r: usize) -> C {
        self.data[(l * 2 + p) * self.dr + r]
    }
}

/// Configuration for the MPS backend.
#[derive(Debug, Clone, Copy)]
pub struct MpsConfig {
    /// Hard cap on the bond dimension between any two sites.
    pub max_bond_dim: usize,
    /// After every two-qubit gate, drop singular values below
    /// `truncation_cutoff * (largest singular value)`. Set to `0.0` for
    /// exact (up to floating-point) compression, bounded only by
    /// `max_bond_dim`.
    pub truncation_cutoff: f64,
}

impl Default for MpsConfig {
    fn default() -> Self {
        Self {
            max_bond_dim: 64,
            truncation_cutoff: 1e-10,
        }
    }
}

/// Error returned when a circuit contains something this backend can't
/// apply to an MPS.
#[derive(Debug, Clone, PartialEq)]
pub enum MpsError {
    /// `gate` acts on `num_qubits` qubits, or has no matrix representation
    /// — this backend only applies 1- and 2-qubit gates.
    UnsupportedGate { gate: String, num_qubits: usize },
}

impl std::fmt::Display for MpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MpsError::UnsupportedGate { gate, num_qubits } => {
                write!(f, "gate '{gate}' ({num_qubits} qubits) is not supported by the MPS backend")
            },
        }
    }
}

impl std::error::Error for MpsError {}

/// A matrix-product-state quantum register.
#[derive(Debug, Clone)]
pub struct MpsState {
    n: usize,
    tensors: Vec<Tensor>,
    qubit_to_site: Vec<usize>,
    site_to_qubit: Vec<usize>,
    config: MpsConfig,
}

impl MpsState {
    /// A fresh `|0...0>` product state (every bond dimension 1).
    pub fn new(n: usize, config: MpsConfig) -> Self {
        assert!(n > 0, "MpsState must have at least one qubit");
        Self {
            n,
            tensors: (0..n).map(|_| Tensor::product_state_zero()).collect(),
            qubit_to_site: (0..n).collect(),
            site_to_qubit: (0..n).collect(),
            config,
        }
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// The largest bond dimension anywhere in the chain — the number that
    /// determines this state's memory footprint (`O(n * d^2)` vs a
    /// statevector's `O(2^n)`).
    pub fn max_bond_dimension(&self) -> usize {
        self.tensors.iter().map(|t| t.dr).max().unwrap_or(1)
    }

    /// Apply a single-qubit gate (`mat`: row-major flattened 2x2) to
    /// `qubit`. Never changes any bond dimension.
    pub fn apply_single_qubit(&mut self, qubit: usize, mat: &[C]) {
        let site = self.qubit_to_site[qubit];
        let t = &self.tensors[site];
        let (dl, dr) = (t.dl, t.dr);
        let mut out = vec![czero(); dl * 2 * dr];
        for l in 0..dl {
            for r in 0..dr {
                for pp in 0..2 {
                    let mut acc = czero();
                    for p in 0..2 {
                        acc += mat[pp * 2 + p] * t.get(l, p, r);
                    }
                    out[(l * 2 + pp) * dr + r] = acc;
                }
            }
        }
        self.tensors[site] = Tensor { dl, dr, data: out };
    }

    /// Apply a two-qubit gate (`mat`: row-major flattened 4x4, qubit
    /// `q1`'s bit more significant — matching this crate's
    /// `execution_engine::kernels::two_qubit` convention) to physical sites
    /// `site`/`site+1`, then SVD-compress back into two tensors.
    #[allow(clippy::needless_range_loop)]
    fn apply_adjacent(&mut self, site: usize, mat: &[C]) {
        let a = &self.tensors[site];
        let b = &self.tensors[site + 1];
        let (dl, mid, dr) = (a.dl, a.dr, b.dr);
        debug_assert_eq!(mid, b.dl);

        let rows = dl * 2;
        let cols = 2 * dr;
        let mut combined = DMatrix::<C>::from_element(rows, cols, czero());
        for l in 0..dl {
            for r in 0..dr {
                let mut theta = [[czero(); 2]; 2];
                for p1 in 0..2 {
                    for p2 in 0..2 {
                        let mut acc = czero();
                        for m in 0..mid {
                            acc += a.get(l, p1, m) * b.get(m, p2, r);
                        }
                        theta[p1][p2] = acc;
                    }
                }
                for p1p in 0..2 {
                    for p2p in 0..2 {
                        let mut acc = czero();
                        for p1 in 0..2 {
                            for p2 in 0..2 {
                                acc += mat[(p1p * 2 + p2p) * 4 + (p1 * 2 + p2)] * theta[p1][p2];
                            }
                        }
                        combined[(l * 2 + p1p, p2p * dr + r)] = acc;
                    }
                }
            }
        }

        let svd = combined.svd(true, true);
        let u = svd.u.expect("svd requested u");
        let vt = svd.v_t.expect("svd requested v_t");
        let s = svd.singular_values;

        let max_sv = s.iter().cloned().fold(0.0_f64, f64::max);
        let cutoff = self.config.truncation_cutoff * max_sv;
        let significant = s.iter().filter(|&&sv| sv > cutoff).count().max(1);
        let new_mid = significant.min(self.config.max_bond_dim).min(s.len());

        let mut a_data = vec![czero(); dl * 2 * new_mid];
        for l in 0..dl {
            for p1p in 0..2 {
                for k in 0..new_mid {
                    a_data[(l * 2 + p1p) * new_mid + k] = u[(l * 2 + p1p, k)];
                }
            }
        }
        let mut b_data = vec![czero(); new_mid * 2 * dr];
        for k in 0..new_mid {
            let sk = C::new(s[k], 0.0);
            for p2p in 0..2 {
                for r in 0..dr {
                    b_data[(k * 2 + p2p) * dr + r] = sk * vt[(k, p2p * dr + r)];
                }
            }
        }
        self.tensors[site] = Tensor {
            dl,
            dr: new_mid,
            data: a_data,
        };
        self.tensors[site + 1] = Tensor {
            dl: new_mid,
            dr,
            data: b_data,
        };
    }

    /// Swap the qubits currently living at adjacent sites `site`/`site+1`,
    /// keeping `qubit_to_site`/`site_to_qubit` consistent with the swap.
    fn swap_adjacent_sites(&mut self, site: usize) {
        let swap_mat = Swap.matrix().expect("SWAP has a matrix");
        self.apply_adjacent(site, &swap_mat);
        let qa = self.site_to_qubit[site];
        let qb = self.site_to_qubit[site + 1];
        self.site_to_qubit.swap(site, site + 1);
        self.qubit_to_site[qa] = site + 1;
        self.qubit_to_site[qb] = site;
    }

    /// Bring `q1` and `q2` to adjacent sites via a chain of adjacent swaps
    /// (see the module docs on why this doesn't swap back afterwards).
    fn move_adjacent(&mut self, q1: usize, q2: usize) {
        loop {
            let s1 = self.qubit_to_site[q1];
            let s2 = self.qubit_to_site[q2];
            let (lo, hi) = if s1 < s2 { (s1, s2) } else { (s2, s1) };
            if hi - lo <= 1 {
                return;
            }
            self.swap_adjacent_sites(hi - 1);
        }
    }

    /// Permutation that swaps the roles of the two qubit slots in a
    /// flattened 4x4 matrix — needed when `q1`'s site ends up to the right
    /// of `q2`'s after [`Self::move_adjacent`].
    fn permute_two_qubit_matrix(mat: &[C]) -> Vec<C> {
        let swap_bits = |i: usize| ((i & 1) << 1) | ((i >> 1) & 1);
        let mut out = vec![czero(); 16];
        for r in 0..4 {
            for c in 0..4 {
                out[swap_bits(r) * 4 + swap_bits(c)] = mat[r * 4 + c];
            }
        }
        out
    }

    /// Apply a two-qubit gate (`mat`: row-major flattened 4x4, `q1`'s bit
    /// more significant) to logical qubits `q1`, `q2` — need not be
    /// adjacent or ordered.
    pub fn apply_two_qubit(&mut self, q1: usize, q2: usize, mat: &[C]) {
        self.move_adjacent(q1, q2);
        let s1 = self.qubit_to_site[q1];
        let s2 = self.qubit_to_site[q2];
        if s1 < s2 {
            self.apply_adjacent(s1, mat);
        } else {
            let permuted = Self::permute_two_qubit_matrix(mat);
            self.apply_adjacent(s2, &permuted);
        }
    }

    /// `<self|other>`, contracting site by site. Both states must have the
    /// same qubit count and the same `site_to_qubit` mapping (true whenever
    /// `other` was derived from a clone of `self` via
    /// [`Self::apply_single_qubit`] only, as [`Self::expectation_pauli_string`]
    /// does).
    fn inner_product(&self, other: &MpsState) -> C {
        let mut env = vec![C::new(1.0, 0.0)];
        let mut cols = 1usize;
        for site in 0..self.n {
            let bt = &self.tensors[site];
            let kt = &other.tensors[site];
            let new_rows = bt.dr;
            let new_cols = kt.dr;
            let mut new_env = vec![czero(); new_rows * new_cols];
            for lb in 0..bt.dl {
                for lk in 0..kt.dl {
                    let ev = env[lb * cols + lk];
                    if ev == czero() {
                        continue;
                    }
                    for p in 0..2 {
                        for rb in 0..new_rows {
                            let bval = bt.get(lb, p, rb).conj();
                            if bval == czero() {
                                continue;
                            }
                            let coef = ev * bval;
                            for rk in 0..new_cols {
                                new_env[rb * new_cols + rk] += coef * kt.get(lk, p, rk);
                            }
                        }
                    }
                }
            }
            env = new_env;
            cols = new_cols;
        }
        env[0]
    }

    /// `<psi|P|psi>` for a single Pauli string, computed by applying `P` to
    /// a clone of this state and taking the overlap with the original —
    /// `O(n * D^2)` in the bond dimension `D`, never `O(2^n)`.
    pub fn expectation_pauli_string(&self, pauli: &PauliString) -> f64 {
        let mut ket = self.clone();
        for q in 0..self.n {
            if let Some(p) = pauli.get(q) {
                if p != Pauli::I {
                    ket.apply_single_qubit(q, &pauli_matrix_2x2(p));
                }
            }
        }
        (self.inner_product(&ket).re) * pauli.coeff() as f64
    }

    /// `<psi|O|psi>` for a full [`PauliObservable`].
    pub fn expectation_value(&self, observable: &PauliObservable) -> f64 {
        observable
            .terms()
            .iter()
            .map(|(pauli, coeff)| coeff * self.expectation_pauli_string(pauli))
            .sum()
    }

    /// Fully contract the chain into a `2^n`-amplitude statevector, in this
    /// crate's little-endian convention (qubit `q` is bit `q`).
    /// `O(2^n)` — for cross-validation on small circuits only, never on the
    /// benchmark path this backend exists for.
    #[allow(clippy::needless_range_loop)]
    pub fn to_statevector(&self) -> Vec<C> {
        let mut amp = vec![C::new(1.0, 0.0)];
        let mut bond = 1usize;
        let mut configs = 1usize;
        for t in &self.tensors {
            let new_bond = t.dr;
            let mut new_amp = vec![czero(); configs * 2 * new_bond];
            for cfg in 0..configs {
                for l in 0..bond {
                    let a = amp[cfg * bond + l];
                    if a == czero() {
                        continue;
                    }
                    for p in 0..2 {
                        for r in 0..new_bond {
                            new_amp[(cfg * 2 + p) * new_bond + r] += a * t.get(l, p, r);
                        }
                    }
                }
            }
            amp = new_amp;
            bond = new_bond;
            configs *= 2;
        }

        let n = self.n;
        let mut result = vec![czero(); 1 << n];
        for site_bits in 0..(1usize << n) {
            let mut qubit_bits = 0usize;
            for site in 0..n {
                let bit = (site_bits >> (n - 1 - site)) & 1;
                qubit_bits |= bit << self.site_to_qubit[site];
            }
            result[qubit_bits] = amp[site_bits];
        }
        result
    }
}

/// Flattened row-major 2x2 matrix for a single-qubit Pauli — shared with
/// [`crate::pauli_propagation`].
pub(crate) fn pauli_matrix_2x2(p: Pauli) -> [C; 4] {
    crate::pauli_propagation::pauli_matrix_2x2(p)
}

/// Run `circuit` (starting from `|0...0>`) on the MPS backend.
pub fn run_circuit(circuit: &Circuit, config: MpsConfig) -> Result<MpsState, MpsError> {
    let mut state = MpsState::new(circuit.num_qubits(), config);
    for op in circuit.operations() {
        let gate = op.gate();
        let qubits = op.qubits();
        let k = qubits.len();
        let dim = 1usize << k;
        let mat = gate
            .matrix()
            .filter(|m| (k == 1 || k == 2) && m.len() == dim * dim)
            .ok_or_else(|| MpsError::UnsupportedGate {
                gate: gate.name().to_string(),
                num_qubits: k,
            })?;
        match k {
            1 => state.apply_single_qubit(qubits[0].index(), &mat),
            2 => state.apply_two_qubit(qubits[0].index(), qubits[1].index(), &mat),
            _ => unreachable!("filtered to k in {{1,2}} above"),
        }
    }
    Ok(state)
}

/// Result of an MPS expectation-value computation, with the bond-dimension
/// telemetry needed to judge whether the approximation was tight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MpsExpectationResult {
    pub expectation: f64,
    pub max_bond_dimension: usize,
}

/// Run `circuit` and compute `<0...0|U^dagger O U|0...0>` on the MPS
/// backend in one call.
pub fn expectation_value(
    circuit: &Circuit,
    observable: &PauliObservable,
    config: MpsConfig,
) -> Result<MpsExpectationResult, MpsError> {
    let state = run_circuit(circuit, config)?;
    Ok(MpsExpectationResult {
        expectation: state.expectation_value(observable),
        max_bond_dimension: state.max_bond_dimension(),
    })
}

/// Whether every two-qubit gate in `circuit` acts on qubits that are
/// adjacent on a line (`|q0 - q1| == 1`) or wrap around a ring
/// (`|q0 - q1| == n - 1`) — the two connectivity shapes this crate's own
/// VQE (chain) and QAOA MaxCut (ring) benchmark ansätze use, and the shapes
/// MPS compresses well. A circuit that fails this check can still be run
/// through [`run_circuit`] (the swap network handles arbitrary
/// connectivity), just without the structural reason to expect a small
/// bond dimension.
pub fn is_1d_candidate(circuit: &Circuit) -> bool {
    let n = circuit.num_qubits();
    circuit.two_qubit_operations().all(|op| {
        let a = op.qubits()[0].index();
        let b = op.qubits()[1].index();
        let d = a.abs_diff(b);
        d == 1 || d == n.saturating_sub(1)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use simq_core::QubitId;
    use simq_gates::standard::{CNot, Hadamard, RotationX, RotationY, RotationZ, Toffoli};
    use simq_state::DenseState;
    use std::sync::Arc;

    fn qc(n: usize) -> Circuit {
        Circuit::new(n)
    }

    fn exact_config() -> MpsConfig {
        MpsConfig {
            max_bond_dim: 1 << 12,
            truncation_cutoff: 0.0,
        }
    }

    fn dense_amplitudes(circuit: &Circuit) -> Vec<Complex64> {
        let unitary = simq_gates::matrix_ops::circuit_matrix(circuit).unwrap();
        let dim = 1usize << circuit.num_qubits();
        (0..dim).map(|row| unitary[row * dim]).collect()
    }

    #[test]
    fn product_state_matches_statevector() {
        let mps = MpsState::new(3, exact_config());
        let sv = mps.to_statevector();
        let expected = dense_amplitudes(&qc(3));
        for (a, b) in sv.iter().zip(expected.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-12);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-12);
        }
    }

    #[test]
    fn ghz_chain_matches_statevector_exactly() {
        // H + CNOT chain: exactly bond-dimension-2 entanglement, so an
        // uncapped MPS must reproduce the statevector exactly.
        let n = 6;
        let mut c = qc(n);
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
        for q in 0..n - 1 {
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
                .unwrap();
        }

        let mps = run_circuit(&c, exact_config()).unwrap();
        assert!(mps.max_bond_dimension() <= 2);
        let got = mps.to_statevector();
        let expected = dense_amplitudes(&c);
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-9);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-9);
        }
    }

    #[test]
    fn vqe_chain_expectation_matches_statevector() {
        let n = 5;
        let mut c = qc(n);
        for q in 0..n {
            c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
        }
        for q in 0..n {
            c.add_gate(Arc::new(RotationY::new(0.31 + 0.11 * q as f64)), &[QubitId::new(q)])
                .unwrap();
        }
        for q in 0..n - 1 {
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
                .unwrap();
        }
        for q in 0..n {
            c.add_gate(Arc::new(RotationZ::new(0.05 + 0.07 * q as f64)), &[QubitId::new(q)])
                .unwrap();
        }
        assert!(is_1d_candidate(&c));

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

        let expected_amps = dense_amplitudes(&c);
        let expected_state = DenseState::from_amplitudes(n, &expected_amps).unwrap();
        let expected = obs.expectation_value(&expected_state).unwrap();

        let got = expectation_value(&c, &obs, exact_config()).unwrap();
        assert_relative_eq!(got.expectation, expected, epsilon = 1e-8);
    }

    #[test]
    fn qaoa_ring_expectation_matches_statevector_via_swap_network() {
        // The ring's wraparound edge (qubit n-1 <-> qubit 0) is not
        // adjacent, exercising `move_adjacent`'s swap network.
        let n = 5;
        let mut c = qc(n);
        for q in 0..n {
            c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
        }
        for q in 0..n {
            let r = (q + 1) % n;
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(r)])
                .unwrap();
            c.add_gate(Arc::new(RotationZ::new(1.6)), &[QubitId::new(r)])
                .unwrap();
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(r)])
                .unwrap();
        }
        for q in 0..n {
            c.add_gate(Arc::new(RotationX::new(1.4)), &[QubitId::new(q)])
                .unwrap();
        }
        assert!(is_1d_candidate(&c));

        let mut obs = PauliObservable::new();
        for q in 0..n {
            let r = (q + 1) % n;
            let mut paulis = vec![Pauli::I; n];
            paulis[q] = Pauli::Z;
            paulis[r] = Pauli::Z;
            obs.add_term(PauliString::from_paulis(paulis), 1.0);
        }

        let expected_amps = dense_amplitudes(&c);
        let expected_state = DenseState::from_amplitudes(n, &expected_amps).unwrap();
        let expected = obs.expectation_value(&expected_state).unwrap();

        let got = expectation_value(&c, &obs, exact_config()).unwrap();
        assert_relative_eq!(got.expectation, expected, epsilon = 1e-8);
    }

    #[test]
    fn bond_dimension_cap_still_gives_a_close_approximation() {
        let n = 6;
        let mut c = qc(n);
        for q in 0..n {
            c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
        }
        for q in 0..n {
            c.add_gate(Arc::new(RotationY::new(0.2 + 0.05 * q as f64)), &[QubitId::new(q)])
                .unwrap();
        }
        for q in 0..n - 1 {
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
                .unwrap();
        }

        let obs = PauliObservable::from_pauli_string(PauliString::all_z(n), 1.0);
        let expected_amps = dense_amplitudes(&c);
        let expected_state = DenseState::from_amplitudes(n, &expected_amps).unwrap();
        let expected = obs.expectation_value(&expected_state).unwrap();

        let capped = expectation_value(
            &c,
            &obs,
            MpsConfig {
                max_bond_dim: 2,
                truncation_cutoff: 0.0,
            },
        )
        .unwrap();
        assert!(capped.max_bond_dimension <= 2);
        // A tight chain like this stays close even at bond dimension 2;
        // this is a sanity bound, not a proof of exactness.
        assert!((capped.expectation - expected).abs() < 0.5);
    }

    #[test]
    fn unsupported_gate_is_rejected() {
        let mut c = qc(3);
        c.add_gate(Arc::new(Toffoli), &[QubitId::new(0), QubitId::new(1), QubitId::new(2)])
            .unwrap();
        let err = run_circuit(&c, MpsConfig::default()).unwrap_err();
        assert_eq!(
            err,
            MpsError::UnsupportedGate {
                gate: "CCNOT".to_string(),
                num_qubits: 3
            }
        );
        assert!(err.to_string().contains("CCNOT"));
    }

    #[test]
    fn is_1d_candidate_rejects_long_range_gates() {
        let n = 5;
        let mut chain = qc(n);
        for q in 0..n - 1 {
            chain
                .add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
                .unwrap();
        }
        assert!(is_1d_candidate(&chain));

        let mut long_range = qc(n);
        long_range
            .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(2)])
            .unwrap();
        assert!(!is_1d_candidate(&long_range));
    }

    #[test]
    fn max_bond_dimension_of_product_state_is_one() {
        let mps = MpsState::new(4, MpsConfig::default());
        assert_eq!(mps.max_bond_dimension(), 1);
    }
}
