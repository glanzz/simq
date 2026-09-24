//! Clifford/stabilizer tableau backend (Aaronson-Gottesman CHP algorithm).
//!
//! Statevector simulation of an `n`-qubit circuit costs `O(2^n)` memory no
//! matter what the circuit does. If every gate in the circuit is a Clifford
//! gate (H, S, S†, X, Y, Z, CNOT, CZ, SWAP), the state is always a
//! *stabilizer state* and can instead be tracked by an `O(n^2)`-bit tableau
//! of Pauli generators — polynomial in the qubit count instead of
//! exponential. This unlocks exact simulation of Clifford-only circuits
//! (e.g. GHZ preparation, most error-correction syndrome extraction) at
//! qubit counts a statevector could never reach.
//!
//! Reference: S. Aaronson & D. Gottesman, "Improved Simulation of
//! Stabilizer Circuits," Phys. Rev. A 70, 052328 (2004).
//!
//! # Tableau layout
//!
//! `2n` rows of `n`-bit `x`/`z` vectors plus a phase bit `r`. Rows
//! `0..n` are the *destabilizer* generators, rows `n..2n` are the
//! *stabilizer* generators — both are required (not just the stabilizers)
//! because [`StabilizerTableau::measure_qubit`]'s deterministic-outcome
//! case needs them to compute the sign without hitting an underdetermined
//! system. Row `i`'s Pauli on qubit `q` is `X^{x[i][q]} Z^{z[i][q]}` (up to
//! the sign carried by `r[i]`).

use rand::Rng;
use simq_core::Circuit;

/// A stabilizer tableau tracking an `n`-qubit stabilizer state exactly.
#[derive(Clone, Debug)]
pub struct StabilizerTableau {
    n: usize,
    /// `x[row][qubit]`, `row` in `0..2n` (destabilizers then stabilizers).
    x: Vec<Vec<bool>>,
    /// `z[row][qubit]`, same row layout as `x`.
    z: Vec<Vec<bool>>,
    /// Phase bit per row: the generator's sign is `(-1)^r[row]`.
    r: Vec<bool>,
}

impl StabilizerTableau {
    /// A new tableau for the `|0...0>` state: destabilizer `i` = `X_i`,
    /// stabilizer `i` = `Z_i`.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "StabilizerTableau must have at least one qubit");
        let mut x = vec![vec![false; n]; 2 * n];
        let mut z = vec![vec![false; n]; 2 * n];
        for i in 0..n {
            x[i][i] = true; // destabilizer i = X_i
            z[n + i][i] = true; // stabilizer i = Z_i
        }
        Self {
            n,
            x,
            z,
            r: vec![false; 2 * n],
        }
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Conjugate every generator by H on `qubit`: swaps the X/Z bits, and
    /// flips the sign of any generator whose local Pauli was Y (X and Z
    /// both set), since `H Y H = -Y`.
    pub fn apply_h(&mut self, qubit: usize) {
        for i in 0..2 * self.n {
            self.r[i] ^= self.x[i][qubit] && self.z[i][qubit];
            std::mem::swap(&mut self.x[i][qubit], &mut self.z[i][qubit]);
        }
    }

    /// Conjugate by S (phase gate): `S X S† = Y`, `S Z S† = Z`. Flips sign
    /// exactly when the local Pauli is Y (since `S Y S† = -X`... tracked via
    /// the same x&z sign rule as H) and folds X into Z.
    pub fn apply_s(&mut self, qubit: usize) {
        for i in 0..2 * self.n {
            self.r[i] ^= self.x[i][qubit] && self.z[i][qubit];
            self.z[i][qubit] ^= self.x[i][qubit];
        }
    }

    /// S† = S applied three times (S^4 = I), the simplest correct way to
    /// derive its tableau action from `apply_s` without re-deriving the
    /// sign rule by hand.
    pub fn apply_sdg(&mut self, qubit: usize) {
        self.apply_s(qubit);
        self.apply_s(qubit);
        self.apply_s(qubit);
    }

    /// Conjugate by Pauli X: flips the sign of any generator whose local
    /// Pauli anticommutes with X, i.e. has a Z component on `qubit`.
    pub fn apply_x(&mut self, qubit: usize) {
        for i in 0..2 * self.n {
            if self.z[i][qubit] {
                self.r[i] ^= true;
            }
        }
    }

    /// Conjugate by Pauli Z: flips the sign of any generator with an X
    /// component on `qubit` (the X/Z-anticommutation mirror of `apply_x`).
    pub fn apply_z(&mut self, qubit: usize) {
        for i in 0..2 * self.n {
            if self.x[i][qubit] {
                self.r[i] ^= true;
            }
        }
    }

    /// Conjugate by Pauli Y. `Y = iXZ` up to global phase, and global phase
    /// never affects a stabilizer group's signs, so conjugating by Z then
    /// by X reproduces the same generator-by-generator sign flips as
    /// conjugating by Y directly.
    pub fn apply_y(&mut self, qubit: usize) {
        self.apply_z(qubit);
        self.apply_x(qubit);
    }

    /// Conjugate by CNOT (`control` -> `target`): the standard
    /// Aaronson-Gottesman update rule.
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        for i in 0..2 * self.n {
            let (xc, zc) = (self.x[i][control], self.z[i][control]);
            let (xt, zt) = (self.x[i][target], self.z[i][target]);
            self.r[i] ^= xc && zt && (xt ^ zc ^ true);
            self.x[i][target] = xt ^ xc;
            self.z[i][control] = zc ^ zt;
        }
    }

    /// Conjugate by CZ, decomposed as `H(b) CNOT(a,b) H(b)` — CZ is
    /// symmetric in its two qubits, so which one plays `target` here is an
    /// implementation choice, not an observable one.
    pub fn apply_cz(&mut self, a: usize, b: usize) {
        self.apply_h(b);
        self.apply_cnot(a, b);
        self.apply_h(b);
    }

    /// Conjugate by SWAP, decomposed as the standard three-CNOT identity.
    pub fn apply_swap(&mut self, a: usize, b: usize) {
        self.apply_cnot(a, b);
        self.apply_cnot(b, a);
        self.apply_cnot(a, b);
    }

    /// `g(x1,z1,x2,z2)`: the phase exponent (as a signed count of factors of
    /// `i`) picked up when multiplying single-qubit Paulis `P1 = X^x1 Z^z1`
    /// and `P2 = X^x2 Z^z2` — Aaronson-Gottesman's Table/Lemma helper used
    /// by [`Self::rowsum`].
    fn g(x1: bool, z1: bool, x2: bool, z2: bool) -> i32 {
        match (x1, z1) {
            (false, false) => 0,
            (true, true) => (z2 as i32) - (x2 as i32),
            (true, false) => (z2 as i32) * (2 * (x2 as i32) - 1),
            (false, true) => (x2 as i32) * (1 - 2 * (z2 as i32)),
        }
    }

    /// Sets row `h` to the product of generators `h` and `i` (i.e.
    /// `h *= i`, in Pauli-group terms), including the correct sign. Used
    /// only by [`Self::measure_qubit`].
    fn rowsum(&mut self, h: usize, i: usize) {
        let mut exponent: i32 = 2 * (self.r[h] as i32) + 2 * (self.r[i] as i32);
        for q in 0..self.n {
            exponent += Self::g(self.x[i][q], self.z[i][q], self.x[h][q], self.z[h][q]);
        }
        exponent = exponent.rem_euclid(4);
        debug_assert!(
            exponent == 0 || exponent == 2,
            "rowsum phase must be 0 or 2 mod 4 for a valid stabilizer tableau, got {exponent}"
        );
        self.r[h] = exponent == 2;
        for q in 0..self.n {
            self.x[h][q] ^= self.x[i][q];
            self.z[h][q] ^= self.z[i][q];
        }
    }

    /// Whether measuring `qubit` in the computational basis has a fixed
    /// (non-random) outcome — true iff no *stabilizer* row anticommutes
    /// with Z_qubit (i.e. has an X component on it). Read-only: does not
    /// collapse the state. Returns the outcome when deterministic.
    pub fn deterministic_outcome(&self, qubit: usize) -> Option<bool> {
        if (self.n..2 * self.n).any(|p| self.x[p][qubit]) {
            return None;
        }
        let mut scratch_x = vec![false; self.n];
        let mut scratch_z = vec![false; self.n];
        let mut scratch_r = false;
        for i in 0..self.n {
            if self.x[i][qubit] {
                let src = self.n + i;
                let mut exponent: i32 = 2 * (scratch_r as i32) + 2 * (self.r[src] as i32);
                for q in 0..self.n {
                    exponent += Self::g(self.x[src][q], self.z[src][q], scratch_x[q], scratch_z[q]);
                }
                scratch_r = exponent.rem_euclid(4) == 2;
                for q in 0..self.n {
                    scratch_x[q] ^= self.x[src][q];
                    scratch_z[q] ^= self.z[src][q];
                }
            }
        }
        Some(scratch_r)
    }

    /// Measure `qubit` in the computational basis, collapsing the tableau
    /// and returning the outcome (`true` = |1>). If the outcome is random
    /// (see [`Self::deterministic_outcome`]) and `forced` is `Some(bit)`,
    /// that outcome is used instead of a fresh coin flip — this is how
    /// tests reconstruct a full basis-state probability distribution (walk
    /// both branches of every random qubit) without needing a second,
    /// independent simulation method.
    pub fn measure_qubit(
        &mut self,
        qubit: usize,
        forced: Option<bool>,
        rng: &mut impl Rng,
    ) -> bool {
        if let Some(p) = (self.n..2 * self.n).find(|&p| self.x[p][qubit]) {
            // Row `p - n` (the destabilizer paired with stabilizer `p`)
            // always anticommutes with row `p` -- that pairing is an
            // invariant preserved by every gate conjugation. `rowsum`
            // assumes its two rows can be combined into a real (+-1)
            // signed Pauli, which only holds for *commuting* rows (an
            // anticommuting pair's product is anti-Hermitian, i.e. has an
            // imaginary phase this tableau cannot represent) -- so it must
            // be skipped here. This is harmless: row `p - n`'s value is
            // about to be overwritten with the old row `p` a few lines
            // down regardless of what `rowsum` would have computed.
            for i in 0..2 * self.n {
                if i != p && i != p - self.n && self.x[i][qubit] {
                    self.rowsum(i, p);
                }
            }
            self.x[p - self.n] = self.x[p].clone();
            self.z[p - self.n] = self.z[p].clone();
            self.r[p - self.n] = self.r[p];
            for q in 0..self.n {
                self.x[p][q] = false;
                self.z[p][q] = false;
            }
            self.z[p][qubit] = true;
            let outcome = forced.unwrap_or_else(|| rng.gen_bool(0.5));
            self.r[p] = outcome;
            outcome
        } else {
            self.deterministic_outcome(qubit).unwrap_or(false)
        }
    }

    /// Measure every qubit (ascending order), collapsing the tableau, and
    /// return the resulting bitstring as `Vec<bool>` (index `q` -> qubit
    /// `q`, matching this crate's amplitude-index convention). Not a `u64`
    /// bitmask: this backend's entire reason to exist is qubit counts well
    /// past 64 (see BENCHMARKS.md's ~30-qubit statevector wall), so a
    /// 64-bit return type would defeat its own purpose.
    pub fn measure_all(&mut self, rng: &mut impl Rng) -> Vec<bool> {
        (0..self.n)
            .map(|q| self.measure_qubit(q, None, rng))
            .collect()
    }
}

/// Gate names this backend can execute exactly. Anything outside this set
/// (rotations, T, Toffoli, ...) is not a Clifford gate in general and must
/// fall back to statevector simulation.
pub(crate) fn is_clifford_gate_name(name: &str) -> bool {
    matches!(name, "H" | "X" | "Y" | "Z" | "S" | "S†" | "CNOT" | "CX" | "CZ" | "SWAP")
}

/// Whether every operation in `circuit` is a Clifford gate on 1 or 2
/// qubits — the condition under which [`StabilizerTableau`] simulates it
/// exactly. This is the auto-dispatch check: callers use it to decide
/// between this module and the statevector-based [`crate::Simulator`].
pub fn is_clifford_circuit(circuit: &Circuit) -> bool {
    circuit
        .operations()
        .all(|op| is_clifford_gate_name(op.gate().name()))
}

/// Errors from [`run_clifford_circuit`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StabilizerError {
    /// `circuit` contains a gate this backend cannot execute — see
    /// [`is_clifford_circuit`]. Carries the offending gate's name.
    NotClifford(String),
}

impl std::fmt::Display for StabilizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StabilizerError::NotClifford(name) => {
                write!(f, "gate '{name}' is not in the stabilizer backend's Clifford gate set")
            },
        }
    }
}

impl std::error::Error for StabilizerError {}

/// Apply every gate in `circuit` to `tableau`. Fails fast (leaving
/// `tableau` partially updated) on the first non-Clifford gate — callers
/// that need an all-or-nothing guarantee should check
/// [`is_clifford_circuit`] first, which this crate's own dispatch helpers
/// always do.
pub fn apply_circuit(
    tableau: &mut StabilizerTableau,
    circuit: &Circuit,
) -> Result<(), StabilizerError> {
    for op in circuit.operations() {
        let qubits = op.qubits();
        match (op.gate().name(), qubits.len()) {
            ("H", 1) => tableau.apply_h(qubits[0].index()),
            ("X", 1) => tableau.apply_x(qubits[0].index()),
            ("Y", 1) => tableau.apply_y(qubits[0].index()),
            ("Z", 1) => tableau.apply_z(qubits[0].index()),
            ("S", 1) => tableau.apply_s(qubits[0].index()),
            ("S†", 1) => tableau.apply_sdg(qubits[0].index()),
            ("CNOT" | "CX", 2) => tableau.apply_cnot(qubits[0].index(), qubits[1].index()),
            ("CZ", 2) => tableau.apply_cz(qubits[0].index(), qubits[1].index()),
            ("SWAP", 2) => tableau.apply_swap(qubits[0].index(), qubits[1].index()),
            (name, _) => return Err(StabilizerError::NotClifford(name.to_string())),
        }
    }
    Ok(())
}

/// Run `circuit` (which must be Clifford-only, see [`is_clifford_circuit`])
/// on a fresh `|0...0>` tableau and return it post-circuit, pre-measurement
/// — the shared starting point [`sample_bitstrings`] clones per shot.
pub fn run_clifford_circuit(circuit: &Circuit) -> Result<StabilizerTableau, StabilizerError> {
    let mut tableau = StabilizerTableau::new(circuit.num_qubits());
    apply_circuit(&mut tableau, circuit)?;
    Ok(tableau)
}

/// Sample `shots` independent full-qubit measurement outcomes from a
/// Clifford circuit's output state.
///
/// Runs the (measurement-free) circuit once, then clones the resulting
/// tableau per shot and measures each clone independently -- avoiding the
/// `shots`-fold cost of replaying every gate that a naive
/// "rebuild-and-measure per shot" loop would pay, while still giving each
/// shot an independent, correctly-collapsing measurement (unlike sampling
/// repeatedly from one frozen probability table, which is only valid
/// because computational-basis measurement of a *fixed* state is what both
/// approaches model here).
pub fn sample_bitstrings(
    circuit: &Circuit,
    shots: usize,
    seed: u64,
) -> Result<Vec<Vec<bool>>, StabilizerError> {
    use rand::SeedableRng;
    let base = run_clifford_circuit(circuit)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    Ok((0..shots)
        .map(|_| {
            let mut shot = base.clone();
            shot.measure_all(&mut rng)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use simq_core::QubitId;
    use simq_gates::standard::{
        CNot, Hadamard, PauliX, PauliY, PauliZ, SGate, SGateDagger, Swap, CZ,
    };
    use std::sync::Arc;

    fn qc(n: usize) -> Circuit {
        Circuit::new(n)
    }

    #[test]
    fn deterministic_zero_state_measures_all_zero() {
        let mut t = StabilizerTableau::new(3);
        let mut rng = StdRng::seed_from_u64(0);
        assert_eq!(t.measure_all(&mut rng), vec![false, false, false]);
    }

    #[test]
    fn x_gate_flips_deterministic_outcome() {
        let mut t = StabilizerTableau::new(1);
        t.apply_x(0);
        assert_eq!(t.deterministic_outcome(0), Some(true));
    }

    #[test]
    fn hadamard_makes_outcome_random() {
        let mut t = StabilizerTableau::new(1);
        t.apply_h(0);
        assert_eq!(t.deterministic_outcome(0), None);
    }

    #[test]
    fn bell_pair_outcomes_are_perfectly_correlated() {
        // H(0), CNOT(0,1): every shot must have q0 == q1.
        let mut base = StabilizerTableau::new(2);
        base.apply_h(0);
        base.apply_cnot(0, 1);

        let mut rng = StdRng::seed_from_u64(42);
        let mut saw_00 = false;
        let mut saw_11 = false;
        for _ in 0..200 {
            let mut shot = base.clone();
            let bits = shot.measure_all(&mut rng);
            assert_eq!(bits[0], bits[1], "Bell pair qubits must agree, got bits={bits:?}");
            saw_00 |= !bits[0] && !bits[1];
            saw_11 |= bits[0] && bits[1];
        }
        assert!(saw_00 && saw_11, "both Bell outcomes should occur over 200 shots");
    }

    #[test]
    fn ghz_sampling_matches_uniform_two_outcome_distribution() {
        // Mirrors simq::bench_workloads::ghz_circuit's structure: this is
        // the exact shape the benchmark suite's Clifford GHZ workload uses.
        let n = 10;
        let mut circuit = qc(n);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        for q in 0..n - 1 {
            circuit
                .add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
                .unwrap();
        }
        assert!(is_clifford_circuit(&circuit));

        let shots = 2000;
        let outcomes = sample_bitstrings(&circuit, shots, 0xC0FFEE).unwrap();
        let is_all_zero = |b: &Vec<bool>| b.iter().all(|&bit| !bit);
        let is_all_one = |b: &Vec<bool>| b.iter().all(|&bit| bit);
        let p0 = outcomes.iter().filter(|b| is_all_zero(b)).count() as f64 / shots as f64;
        let p1 = outcomes.iter().filter(|b| is_all_one(b)).count() as f64 / shots as f64;
        assert!(outcomes.iter().all(|b| is_all_zero(b) || is_all_one(b)));
        // Binomial standard error at p=0.5, n=2000 is ~0.011; 0.4/0.6 bounds
        // give a comfortable, non-flaky margin.
        assert!((0.4..0.6).contains(&p0), "p(all-zero)={p0} out of range");
        assert!((0.4..0.6).contains(&p1), "p(all-one)={p1} out of range");
    }

    #[test]
    fn non_clifford_gate_is_rejected() {
        use simq_gates::standard::TGate;
        let mut circuit = qc(1);
        circuit
            .add_gate(Arc::new(TGate), &[QubitId::new(0)])
            .unwrap();
        assert!(!is_clifford_circuit(&circuit));
        assert_eq!(
            run_clifford_circuit(&circuit).unwrap_err(),
            StabilizerError::NotClifford("T".to_string())
        );
    }

    /// Cross-validates the tableau against the exact statevector simulator
    /// on random small Clifford circuits: for every basis state, the two
    /// simulators must agree on which states have nonzero probability, and
    /// stabilizer-state probabilities (always a power of two) must match
    /// the statevector's amplitude-derived probability to high precision.
    #[test]
    fn matches_statevector_probabilities_on_random_clifford_circuits() {
        use simq_gates::standard::CNot as CNotGate;
        use simq_state::DenseState;

        let n = 4;
        for seed in 0u64..12 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut tableau = StabilizerTableau::new(n);
            let mut circuit = qc(n);

            for _ in 0..15 {
                let a = rng.gen_range(0..n);
                let b = (a + 1 + rng.gen_range(0..n - 1)) % n;
                match rng.gen_range(0..6) {
                    0 => {
                        tableau.apply_h(a);
                        circuit
                            .add_gate(Arc::new(Hadamard), &[QubitId::new(a)])
                            .unwrap();
                    },
                    1 => {
                        tableau.apply_s(a);
                        circuit
                            .add_gate(Arc::new(SGate), &[QubitId::new(a)])
                            .unwrap();
                    },
                    2 => {
                        tableau.apply_x(a);
                        circuit
                            .add_gate(Arc::new(PauliX), &[QubitId::new(a)])
                            .unwrap();
                    },
                    3 => {
                        tableau.apply_z(a);
                        circuit
                            .add_gate(Arc::new(PauliZ), &[QubitId::new(a)])
                            .unwrap();
                    },
                    4 => {
                        tableau.apply_cnot(a, b);
                        circuit
                            .add_gate(Arc::new(CNotGate), &[QubitId::new(a), QubitId::new(b)])
                            .unwrap();
                    },
                    _ => {
                        tableau.apply_cz(a, b);
                        circuit
                            .add_gate(Arc::new(CZ), &[QubitId::new(a), QubitId::new(b)])
                            .unwrap();
                    },
                }
            }
            // Exact statevector probabilities, computed independently of
            // simq-sim/simq-compiler via simq-gates' circuit-to-unitary
            // helper (unitary * |0...0> = the unitary's first column).
            let unitary = simq_gates::matrix_ops::circuit_matrix(&circuit).unwrap();
            let dim = 1usize << n;
            let amps: Vec<num_complex::Complex64> =
                (0..dim).map(|row| unitary[row * dim]).collect();
            let dense = DenseState::from_amplitudes(n, &amps).unwrap();

            for basis in 0u64..(1 << n) {
                let sv_prob = dense.amplitudes()[basis as usize].norm_sqr();

                // Determine the tableau's probability for this exact basis
                // state by walking (forcing) each qubit's outcome.
                let mut t = tableau.clone();
                let mut forced_rng = StdRng::seed_from_u64(0); // unused when forced
                let mut log2_branches = 0u32;
                for q in 0..n {
                    let bit = (basis >> q) & 1 == 1;
                    if t.deterministic_outcome(q).is_none() {
                        log2_branches += 1;
                    }
                    t.measure_qubit(q, Some(bit), &mut forced_rng);
                }
                let tableau_prob = 1.0 / (1u64 << log2_branches) as f64;
                // If the forced path was inconsistent with the state (this
                // basis state has zero amplitude), the deterministic
                // branches would have forced a *contradicting* outcome --
                // but `measure_qubit` with `forced` always honors the
                // request, so we detect zero-probability basis states by
                // comparing against a second, unforced pass on the
                // deterministic qubits only. Simpler: cross-check directly
                // against the statevector's zero/nonzero pattern instead.
                if sv_prob < 1e-12 {
                    // Can't assert tableau_prob is exactly 0 (forcing always
                    // "succeeds" combinatorially); instead assert this basis
                    // state is unreachable by confirming at least one
                    // originally-deterministic qubit disagreed with `basis`.
                    let mut t2 = tableau.clone();
                    let mut contradiction = false;
                    for q in 0..n {
                        let bit = (basis >> q) & 1 == 1;
                        if let Some(det) = t2.deterministic_outcome(q) {
                            if det != bit {
                                contradiction = true;
                            }
                        }
                        t2.measure_qubit(q, Some(bit), &mut forced_rng);
                    }
                    assert!(
                        contradiction,
                        "seed={seed} basis={basis}: statevector says zero probability but \
                         tableau found no contradicting deterministic qubit"
                    );
                } else {
                    assert!(
                        (tableau_prob - sv_prob).abs() < 1e-9,
                        "seed={seed} basis={basis}: tableau_prob={tableau_prob} sv_prob={sv_prob}"
                    );
                }
            }
        }
    }

    #[test]
    fn sdg_is_inverse_of_s() {
        let mut t = StabilizerTableau::new(1);
        t.apply_h(0); // make outcome random / put Y-like structure in play
        t.apply_s(0);
        t.apply_sdg(0);
        let mut ref_t = StabilizerTableau::new(1);
        ref_t.apply_h(0);
        assert_eq!(t.x, ref_t.x);
        assert_eq!(t.z, ref_t.z);
        assert_eq!(t.r, ref_t.r);
    }

    #[test]
    fn swap_exchanges_qubit_states() {
        let mut t = StabilizerTableau::new(2);
        t.apply_x(0); // qubit 0 -> |1>, qubit 1 stays |0>
        t.apply_swap(0, 1);
        assert_eq!(t.deterministic_outcome(0), Some(false));
        assert_eq!(t.deterministic_outcome(1), Some(true));
    }

    #[test]
    fn y_gate_matches_xz_composition_on_ancilla_free_check() {
        // Y|0> = i|1>; probability of measuring |1> must be 1 regardless
        // of the (unobservable) global phase.
        let mut t = StabilizerTableau::new(1);
        t.apply_y(0);
        assert_eq!(t.deterministic_outcome(0), Some(true));
    }

    #[test]
    fn cnot_cx_alias_and_swap_are_accepted_by_apply_circuit() {
        let mut circuit = qc(2);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        circuit
            .add_gate(Arc::new(Swap), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        circuit
            .add_gate(Arc::new(SGateDagger), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliY), &[QubitId::new(1)])
            .unwrap();
        assert!(is_clifford_circuit(&circuit));
        assert!(run_clifford_circuit(&circuit).is_ok());
    }
}
