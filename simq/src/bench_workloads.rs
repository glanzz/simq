//! Shared workload definitions for the end-to-end benchmark suite.
//!
//! These builders are used by BOTH `benches/end_to_end.rs` (timing) and
//! `examples/xcheck_bench.rs` (cross-validation against the Qiskit baseline
//! in `benchmarks/qiskit_baseline.py`). Keeping them in one place guarantees
//! that the circuits we time are exactly the circuits whose expectation
//! values are checked against Qiskit before any numbers are published.
//!
//! The Python side re-implements the same circuits; `benchmarks/compare.py`
//! refuses to print a comparison table unless all cross-check values agree
//! to 1e-12.

use simq_core::{Circuit, QubitId};
use simq_gates::standard::{
    CNot, CPhase, CZ, Hadamard, PauliX, RotationX, RotationY, RotationZ, SGate, SXGate, Swap,
    TGate,
};
use simq_sim::{Simulator, SimulatorConfig};
use simq_state::{
    measurement::ComputationalBasis, AdaptiveState, DenseState, Pauli, PauliObservable, PauliString,
};
use std::sync::Arc;

/// Number of randomized/perturbed instances benchmarked per workload for
/// the multi-instance variants below (`vqe_energy_instances`,
/// `qaoa_cost_instances`, `ghz_sample_instances`).
///
/// QED-C's application-oriented benchmarking guidance recommends >=10
/// instances per problem size specifically to avoid tuning/overfitting an
/// optimization to one convenient circuit (see BENCHMARKS.md's methodology
/// notes for the full citation). This suite uses 5, not 10+, as a
/// deliberate runtime trade-off: each instance re-runs the full
/// Rust/Qiskit/qsim cross-validation pipeline, and this repo already tests
/// 4 qubit sizes x 3 base workloads; running 10+ instances at all 4 sizes
/// would make `benchmarks/run.sh` several times slower for a marginal
/// increase in statistical confidence beyond what 5 already gives (the
/// question these instances answer is "is the fixed benchmark circuit an
/// outlier the optimizer happens to like," not "what is the precise
/// variance of performance," so 5 well-separated instances is enough to
/// catch the former).
pub const NUM_INSTANCES: usize = 5;

/// Deterministic RY angle for (layer, qubit) — same formula as the Python side.
pub fn vqe_theta(layer: usize, qubit: usize, num_qubits: usize) -> f64 {
    0.1 + 0.37 * (layer * num_qubits + qubit) as f64
}

/// Deterministic RZ angle for (layer, qubit) — same formula as the Python side.
pub fn vqe_phi(layer: usize, qubit: usize, num_qubits: usize) -> f64 {
    0.05 + 0.21 * (layer * num_qubits + qubit) as f64
}

/// QAOA parameters (gamma, beta) per layer — same values as the Python side.
pub const QAOA_GAMMA: [f64; 2] = [0.8, 0.4];
pub const QAOA_BETA: [f64; 2] = [0.7, 0.35];

/// Number of VQE ansatz layers.
pub const VQE_LAYERS: usize = 3;

/// Hardware-efficient VQE ansatz: H on all qubits, then `VQE_LAYERS` layers of
/// [RY(θ_lq) on each qubit, CNOT chain, RZ(φ_lq) on each qubit].
///
/// The per-qubit angles are deliberately asymmetric so any qubit-ordering
/// mismatch against the Qiskit implementation shows up in the cross-check.
pub fn vqe_circuit(num_qubits: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits);
    for q in 0..num_qubits {
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
    }
    for l in 0..VQE_LAYERS {
        for q in 0..num_qubits {
            c.add_gate(Arc::new(RotationY::new(vqe_theta(l, q, num_qubits))), &[QubitId::new(q)])
                .unwrap();
        }
        for q in 0..num_qubits - 1 {
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
                .unwrap();
        }
        for q in 0..num_qubits {
            c.add_gate(Arc::new(RotationZ::new(vqe_phi(l, q, num_qubits))), &[QubitId::new(q)])
                .unwrap();
        }
    }
    c
}

/// Per-instance VQE angle offset (deterministic, not a seeded PRNG — see
/// [`NUM_INSTANCES`] docs on why: it must produce bit-identical circuits in
/// the Rust and Python mirrors for the 1e-12 cross-check, and a fixed
/// additive offset guarantees that the way a seeded RNG's bit-level output
/// wouldn't).
pub fn vqe_theta_instance(layer: usize, qubit: usize, num_qubits: usize, instance: usize) -> f64 {
    vqe_theta(layer, qubit, num_qubits) + 0.91 * instance as f64
}

/// Per-instance VQE angle offset — see [`vqe_theta_instance`].
pub fn vqe_phi_instance(layer: usize, qubit: usize, num_qubits: usize, instance: usize) -> f64 {
    vqe_phi(layer, qubit, num_qubits) + 0.63 * instance as f64
}

/// Same ansatz as [`vqe_circuit`], with `instance` perturbing every angle by
/// a fixed offset — see [`NUM_INSTANCES`] for why this exists (QED-C-style
/// multi-instance testing, guarding against tuning fusion/optimization
/// specifically to the one fixed VQE circuit this suite otherwise always
/// times).
pub fn vqe_circuit_instance(num_qubits: usize, instance: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits);
    for q in 0..num_qubits {
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
    }
    for l in 0..VQE_LAYERS {
        for q in 0..num_qubits {
            let theta = vqe_theta_instance(l, q, num_qubits, instance);
            c.add_gate(Arc::new(RotationY::new(theta)), &[QubitId::new(q)])
                .unwrap();
        }
        for q in 0..num_qubits - 1 {
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
                .unwrap();
        }
        for q in 0..num_qubits {
            let phi = vqe_phi_instance(l, q, num_qubits, instance);
            c.add_gate(Arc::new(RotationZ::new(phi)), &[QubitId::new(q)])
                .unwrap();
        }
    }
    c
}

/// VQE observable: H = Σ_{q} Z_q Z_{q+1} + 0.5 Σ_q X_q
///
/// Mixes diagonal (ZZ) and off-diagonal (X) terms so both expectation paths
/// are exercised.
pub fn vqe_observable(num_qubits: usize) -> PauliObservable {
    let mut obs = PauliObservable::new();
    for q in 0..num_qubits - 1 {
        let mut paulis = vec![Pauli::I; num_qubits];
        paulis[q] = Pauli::Z;
        paulis[q + 1] = Pauli::Z;
        obs.add_term(PauliString::from_paulis(paulis), 1.0);
    }
    for q in 0..num_qubits {
        let mut paulis = vec![Pauli::I; num_qubits];
        paulis[q] = Pauli::X;
        obs.add_term(PauliString::from_paulis(paulis), 0.5);
    }
    obs
}

/// QAOA MaxCut circuit on the ring graph C_n at depth p=2.
///
/// Cost layers implement exp(-i γ Z_q Z_r) via CNOT·RZ(2γ)·CNOT, mixer layers
/// are RX(2β) on every qubit — the textbook construction, gate-for-gate the
/// same as the Python side.
pub fn qaoa_circuit(num_qubits: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits);
    for q in 0..num_qubits {
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
    }
    for l in 0..QAOA_GAMMA.len() {
        for q in 0..num_qubits {
            let r = (q + 1) % num_qubits;
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(r)])
                .unwrap();
            c.add_gate(Arc::new(RotationZ::new(2.0 * QAOA_GAMMA[l])), &[QubitId::new(r)])
                .unwrap();
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(r)])
                .unwrap();
        }
        for q in 0..num_qubits {
            c.add_gate(Arc::new(RotationX::new(2.0 * QAOA_BETA[l])), &[QubitId::new(q)])
                .unwrap();
        }
    }
    c
}

/// Per-instance QAOA (gamma, beta) pair — see [`vqe_theta_instance`] for why
/// this is a fixed deterministic offset, not a seeded PRNG.
pub fn qaoa_params_instance(layer: usize, instance: usize) -> (f64, f64) {
    (
        QAOA_GAMMA[layer] + 0.17 * instance as f64,
        QAOA_BETA[layer] + 0.11 * instance as f64,
    )
}

/// Same construction as [`qaoa_circuit`], with `instance` perturbing gamma/beta
/// per layer — see [`NUM_INSTANCES`].
pub fn qaoa_circuit_instance(num_qubits: usize, instance: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits);
    for q in 0..num_qubits {
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
    }
    for l in 0..QAOA_GAMMA.len() {
        let (gamma, beta) = qaoa_params_instance(l, instance);
        for q in 0..num_qubits {
            let r = (q + 1) % num_qubits;
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(r)])
                .unwrap();
            c.add_gate(Arc::new(RotationZ::new(2.0 * gamma)), &[QubitId::new(r)])
                .unwrap();
            c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(r)])
                .unwrap();
        }
        for q in 0..num_qubits {
            c.add_gate(Arc::new(RotationX::new(2.0 * beta)), &[QubitId::new(q)])
                .unwrap();
        }
    }
    c
}

/// QAOA MaxCut cost observable on C_n: C = Σ_edges 0.5 (1 - Z_q Z_r).
///
/// Returned as the ZZ part only; use [`qaoa_cost_from_zz`] to fold in the
/// constant.
pub fn qaoa_zz_observable(num_qubits: usize) -> PauliObservable {
    let mut obs = PauliObservable::new();
    for q in 0..num_qubits {
        let r = (q + 1) % num_qubits;
        let mut paulis = vec![Pauli::I; num_qubits];
        paulis[q] = Pauli::Z;
        paulis[r] = Pauli::Z;
        obs.add_term(PauliString::from_paulis(paulis), 1.0);
    }
    obs
}

/// Fold the ⟨ΣZZ⟩ expectation into the MaxCut cost value.
pub fn qaoa_cost_from_zz(num_qubits: usize, zz_expectation: f64) -> f64 {
    0.5 * (num_qubits as f64) - 0.5 * zz_expectation
}

/// GHZ preparation circuit: H(0) then a CNOT chain.
pub fn ghz_circuit(num_qubits: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits);
    c.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
    for q in 0..num_qubits - 1 {
        c.add_gate(Arc::new(CNot), &[QubitId::new(q), QubitId::new(q + 1)])
            .unwrap();
    }
    c
}

/// Run a circuit and return the final state as a `DenseState`.
pub fn run_to_dense(sim: &Simulator, circuit: &Circuit) -> DenseState {
    let result = sim.run(circuit).expect("simulation failed");
    match result.state {
        AdaptiveState::Dense(dense) => dense,
        sparse => {
            let amps = sparse.to_dense_vec();
            DenseState::from_amplitudes(sparse.num_qubits(), &amps).unwrap()
        },
    }
}

/// One full VQE energy evaluation: simulate the ansatz and take ⟨H⟩.
pub fn vqe_energy(sim: &Simulator, num_qubits: usize) -> f64 {
    let circuit = vqe_circuit(num_qubits);
    let state = run_to_dense(sim, &circuit);
    vqe_observable(num_qubits)
        .expectation_value(&state)
        .expect("expectation failed")
}

/// One VQE energy evaluation for a specific instance — see [`NUM_INSTANCES`].
pub fn vqe_energy_instance(sim: &Simulator, num_qubits: usize, instance: usize) -> f64 {
    let circuit = vqe_circuit_instance(num_qubits, instance);
    let state = run_to_dense(sim, &circuit);
    vqe_observable(num_qubits)
        .expectation_value(&state)
        .expect("expectation failed")
}

/// VQE energy across all [`NUM_INSTANCES`] instances, in instance order —
/// what the multi-instance benchmark times/checks (see [`NUM_INSTANCES`]).
pub fn vqe_energy_instances(sim: &Simulator, num_qubits: usize) -> Vec<f64> {
    (0..NUM_INSTANCES)
        .map(|i| vqe_energy_instance(sim, num_qubits, i))
        .collect()
}

/// One full QAOA cost evaluation: simulate the p=2 circuit and take the cut value.
pub fn qaoa_cost(sim: &Simulator, num_qubits: usize) -> f64 {
    let circuit = qaoa_circuit(num_qubits);
    let state = run_to_dense(sim, &circuit);
    let zz = qaoa_zz_observable(num_qubits)
        .expectation_value(&state)
        .expect("expectation failed");
    qaoa_cost_from_zz(num_qubits, zz)
}

/// One QAOA cost evaluation for a specific instance — see [`NUM_INSTANCES`].
pub fn qaoa_cost_instance(sim: &Simulator, num_qubits: usize, instance: usize) -> f64 {
    let circuit = qaoa_circuit_instance(num_qubits, instance);
    let state = run_to_dense(sim, &circuit);
    let zz = qaoa_zz_observable(num_qubits)
        .expectation_value(&state)
        .expect("expectation failed");
    qaoa_cost_from_zz(num_qubits, zz)
}

/// QAOA cost across all [`NUM_INSTANCES`] instances, in instance order.
pub fn qaoa_cost_instances(sim: &Simulator, num_qubits: usize) -> Vec<f64> {
    (0..NUM_INSTANCES)
        .map(|i| qaoa_cost_instance(sim, num_qubits, i))
        .collect()
}

/// GHZ shot sampling: simulate and draw `shots` samples.
pub fn ghz_sample(sim: &Simulator, num_qubits: usize, shots: usize, seed: u64) -> usize {
    let circuit = ghz_circuit(num_qubits);
    let state = run_to_dense(sim, &circuit);
    let mut rng = xorshift_star_rng(seed);
    let result = ComputationalBasis::new()
        .sample(&state, shots, &mut rng)
        .expect("sampling failed");
    result.sorted_outcomes().len()
}

/// xorshift* PRNG: deterministic and dependency-free. Only used for
/// *sampling* (shot draws, which are inherently stochastic and not
/// cross-validated bit-exactly against Qiskit/qsim) — never for circuit
/// *construction*, where a seeded PRNG's bit-level output isn't guaranteed
/// to match across the Rust/Python runtimes (see [`NUM_INSTANCES`] docs).
fn xorshift_star_rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    move || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let x = s.wrapping_mul(0x2545F4914F6CDD1D);
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// GHZ's circuit is entirely parameter-free (H + a fixed CNOT chain), so
/// there is no continuous-parameter axis to perturb the way VQE/QAOA's
/// instances do. Its multi-instance benchmark instead varies the
/// shot-sampling seed across [`NUM_INSTANCES`] draws — a legitimate
/// "instance" axis for a *sampling* workload (distinct RNG streams
/// exercise different parts of the sampler, e.g. different runs of binary
/// search over the CDF), while the underlying state-preparation circuit
/// (already cross-validated deterministically via `ghz_p0`) stays fixed.
pub fn ghz_sample_instances(sim: &Simulator, num_qubits: usize, shots: usize) -> Vec<usize> {
    (0..NUM_INSTANCES)
        .map(|i| ghz_sample(sim, num_qubits, shots, 0xB1A2_u64.wrapping_add(i as u64)))
        .collect()
}

// ============================================================================
// QFT: long-range (non-nearest-neighbor) entanglement, phase-sensitive check
// ============================================================================
//
// VQE/QAOA/GHZ above are all local: every two-qubit gate acts on adjacent
// qubits, every layer. That's exactly the pattern this crate's gate-fusion
// compiler pass (width-bounded, local blocks) is built to exploit, so a
// benchmark suite containing only local circuits risks making that specific
// optimization look better than it generalizes. QFT is the standard
// counter-example: qubit `i`'s controlled-phase gates reach every qubit
// `j > i`, so gate count is O(n^2) and the entangling structure is
// genuinely non-local — see BENCHMARKS.md's methodology notes for the
// literature basis (this shape is explicitly why simulator-comparison
// papers such as arXiv:2401.09076 include QFT alongside random circuits).

/// Deterministic nontrivial computational-basis input for the QFT workload:
/// prepares |k=1⟩ (an X gate on qubit 0 only, since qubit `q` represents
/// bit `q` of `k` — see [`qft_circuit`]). `k=0` (no state prep at all)
/// would make QFT's output identical to `H^{⊗n}|0...0⟩`, which is
/// convention-independent and useful as a sanity check (see this module's
/// tests) but does *not* exercise the controlled-phase ladder at all, so
/// it must not be the workload's actual input.
pub const QFT_INPUT_K: usize = 1;

/// Textbook QFT circuit (H + controlled-phase ladder + final qubit-order
/// swaps), applied to the computational basis state |k⟩ (qubit `q`
/// represents bit `q` of `k`, matching this crate's amplitude-index
/// convention: state index `k`'s bit `q` is qubit `q`'s value).
///
/// **QFT's output always has uniform amplitude *magnitude*, for every `k`,
/// whether or not the phase gates are implemented correctly** — that's the
/// defining property of the transform. This means a measurement-probability
/// observable (like GHZ's `p(|0...0⟩)` or a Z-expectation value) cannot
/// distinguish a correct implementation from a broken one here: *any*
/// unitary that preserves uniform output magnitude would pass such a check.
/// The workload's cross-validated value ([`qft_probe`]) is therefore a raw
/// amplitude component, which *is* phase-sensitive, not a probability.
pub fn qft_circuit(num_qubits: usize, k: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits);
    for q in 0..num_qubits {
        if (k >> q) & 1 == 1 {
            c.add_gate(Arc::new(PauliX), &[QubitId::new(q)]).unwrap();
        }
    }
    // Process targets from the highest qubit index down to 0. For target
    // `i`, every control `j < i` must still hold its *classical* input bit
    // (not yet superposed by its own H) for the controlled-phase ladder to
    // correctly build the phase 0.x_i x_{i-1}...x_0 on qubit i -- which is
    // only true if `j`'s own turn as a target hasn't happened yet, i.e. `j`
    // is processed *later* in this loop. Processing high-to-low guarantees
    // that: when we reach target `i`, every `j < i` is still pending.
    // (Getting this backwards -- low-to-high with j > i as control -- is a
    // subtle bug: it compiles, produces a normalized state, and *looks*
    // like a QFT, but silently applies no net phase for basis-state inputs,
    // since every control is trivially |0> at the time it's used. Verified
    // against the closed-form QFT|k=1> amplitude by hand before fixing.)
    for i in (0..num_qubits).rev() {
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(i)]).unwrap();
        for j in (0..i).rev() {
            let theta = std::f64::consts::PI / (1u64 << (i - j)) as f64;
            c.add_gate(Arc::new(CPhase::new(theta)), &[QubitId::new(j), QubitId::new(i)])
                .unwrap();
        }
    }
    for i in 0..num_qubits / 2 {
        c.add_gate(Arc::new(Swap), &[QubitId::new(i), QubitId::new(num_qubits - 1 - i)])
            .unwrap();
    }
    c
}

/// QFT workload probe: `Re(amplitude at basis state |1⟩)` after
/// `QFT|k=QFT_INPUT_K⟩`. Phase-sensitive (see [`qft_circuit`] docs) — this
/// is what actually exercises whether the controlled-phase ladder is
/// correct; a measurement probability would not.
pub fn qft_probe(sim: &Simulator, num_qubits: usize) -> f64 {
    let circuit = qft_circuit(num_qubits, QFT_INPUT_K);
    let state = run_to_dense(sim, &circuit);
    state.amplitudes()[1].re
}

// ============================================================================
// Random circuit sampling (RCS): structure-agnostic stress test
// ============================================================================
//
// The standard "no shortcuts" benchmark in the simulator-comparison
// literature (the shape of Google's/IBM's supremacy/utility circuits):
// alternating single-qubit gate layers and two-qubit entangling layers with
// *changing* qubit pairing, so there's no fixed, repeating structure for a
// structure-aware optimization (gate fusion, template matching,
// commutation) to exploit — unlike VQE/QAOA/GHZ, whose entangling pattern
// is the *same* linear chain every layer.
//
// This uses fixed, deterministic index-based formulas to choose gate types
// and angles rather than a seeded PRNG. That's a deliberate choice, not a
// simplification of "real" RCS: this suite's cross-validation requires the
// Rust and Python (Qiskit/Cirq) mirrors to build *bit-identical* circuits,
// and a seeded PRNG's bit-level output stream is not guaranteed to match
// across language runtimes even with "the same" seed and algorithm. A
// closed-form deterministic formula sidesteps that risk entirely while
// still producing a circuit with no fixed/repeating structure to exploit —
// which is the actual property this benchmark needs, not true randomness
// per se.

/// Number of alternating (single-qubit layer, entangling layer) rounds in
/// [`random_circuit`].
pub const RCS_LAYERS: usize = 8;

const RCS_SINGLE_QUBIT_GATE_COUNT: usize = 5;

/// Deterministic "which gate" index for (layer, qubit) — see module docs on
/// why this is a fixed formula, not a PRNG. The specific constants (7, 13)
/// are coprime to typical qubit counts and to each other, chosen only so
/// neighboring qubits/layers don't trivially get the same gate.
fn rcs_gate_index(layer: usize, qubit: usize) -> usize {
    (layer * 7 + qubit * 13 + 5) % RCS_SINGLE_QUBIT_GATE_COUNT
}

/// Deterministic rotation angle for the `RY` branch of
/// [`rcs_single_qubit_layer`] — same "index-linear, no modulo/transcendental
/// functions" style as [`vqe_theta`], for the same cross-language
/// floating-point exactness reasons (a global phase from an angle outside
/// `[0, 2*pi)` is harmless here since this workload's cross-validated value
/// is a measurement probability, not a raw amplitude — contrast
/// [`qft_probe`]).
fn rcs_angle(layer: usize, qubit: usize, num_qubits: usize) -> f64 {
    0.29 + 0.53 * (layer * num_qubits + qubit) as f64
}

/// One single-qubit layer: gate type chosen deterministically per
/// (layer, qubit) from a fixed 5-gate set {H, S, T, SX, RY(angle)} — a
/// deliberately more diverse gate vocabulary than VQE/QAOA/GHZ's
/// {H, CNOT, RX, RY, RZ}, since T/S/SX exercise the fixed-matrix-cache
/// dispatch path in `simq-sim`'s executor differently than the
/// always-parameterized rotation gates do.
fn rcs_single_qubit_layer(c: &mut Circuit, layer: usize, num_qubits: usize) {
    for q in 0..num_qubits {
        let qubit = QubitId::new(q);
        match rcs_gate_index(layer, q) {
            0 => {
                c.add_gate(Arc::new(Hadamard), &[qubit]).unwrap();
            },
            1 => {
                c.add_gate(Arc::new(SGate), &[qubit]).unwrap();
            },
            2 => {
                c.add_gate(Arc::new(TGate), &[qubit]).unwrap();
            },
            3 => {
                c.add_gate(Arc::new(SXGate), &[qubit]).unwrap();
            },
            _ => {
                c.add_gate(Arc::new(RotationY::new(rcs_angle(layer, q, num_qubits))), &[qubit])
                    .unwrap();
            },
        }
    }
}

/// One brickwork two-qubit entangling layer: pairs `(0,1),(2,3),...` on
/// even layers, `(1,2),(3,4),...` on odd layers — connectivity that
/// *alternates* every layer, unlike VQE/QAOA's fixed linear chain.
fn rcs_entangling_layer(c: &mut Circuit, layer: usize, num_qubits: usize) {
    let offset = layer % 2;
    let mut q = offset;
    while q + 1 < num_qubits {
        c.add_gate(Arc::new(CZ), &[QubitId::new(q), QubitId::new(q + 1)])
            .unwrap();
        q += 2;
    }
}

/// Random-circuit-sampling-style benchmark circuit: [`RCS_LAYERS`] rounds of
/// alternating single-qubit and brickwork CZ entangling layers. See the
/// module-level docs above for why this exists and how it differs from
/// VQE/QAOA/GHZ.
pub fn random_circuit(num_qubits: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits);
    for layer in 0..RCS_LAYERS {
        rcs_single_qubit_layer(&mut c, layer, num_qubits);
        rcs_entangling_layer(&mut c, layer, num_qubits);
    }
    c
}

/// RCS workload probe: probability of measuring the all-zeros bitstring.
/// Same style as `ghz_p0`, but informative here (unlike [`qft_probe`]'s
/// situation) because this circuit is not a stabilizer state and has no
/// uniform-magnitude degeneracy — its output distribution is genuinely
/// sensitive to every gate in the circuit.
pub fn random_circuit_p0(sim: &Simulator, num_qubits: usize) -> f64 {
    let circuit = random_circuit(num_qubits);
    let state = run_to_dense(sim, &circuit);
    state.amplitudes()[0].norm_sqr()
}

/// The default simulator used across the suite (out-of-the-box settings).
pub fn default_simulator() -> Simulator {
    Simulator::new(SimulatorConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vqe_theta_formula() {
        assert_eq!(vqe_theta(0, 0, 4), 0.1);
        assert_eq!(vqe_theta(1, 2, 4), 0.1 + 0.37 * 6.0);
    }

    #[test]
    fn vqe_phi_formula() {
        assert_eq!(vqe_phi(0, 0, 4), 0.05);
        assert_eq!(vqe_phi(1, 2, 4), 0.05 + 0.21 * 6.0);
    }

    #[test]
    fn vqe_circuit_gate_count() {
        let n = 4;
        let c = vqe_circuit(n);
        // H per qubit, then VQE_LAYERS * (RY per qubit + CNOT chain + RZ per qubit)
        let expected = n + VQE_LAYERS * (n + (n - 1) + n);
        assert_eq!(c.len(), expected);
    }

    #[test]
    fn vqe_circuit_minimum_two_qubits() {
        // num_qubits=1 would underflow the `num_qubits - 1` CNOT-chain bound;
        // the smallest sane workload size is 2.
        let c = vqe_circuit(2);
        assert_eq!(c.len(), 2 + VQE_LAYERS * (2 + 1 + 2));
    }

    #[test]
    fn vqe_observable_term_count_and_diagonality() {
        let n = 4;
        let obs = vqe_observable(n);
        // (n - 1) ZZ terms + n X terms
        assert_eq!(obs.num_terms(), (n - 1) + n);
        // Contains off-diagonal X terms, so it is not purely diagonal.
        assert!(!obs.is_diagonal());
    }

    #[test]
    fn qaoa_circuit_gate_count() {
        let n = 4;
        let c = qaoa_circuit(n);
        // H per qubit, then per layer: n edges * (CNOT, RZ, CNOT) + n RX mixers
        let per_layer = n * 3 + n;
        let expected = n + QAOA_GAMMA.len() * per_layer;
        assert_eq!(c.len(), expected);
    }

    #[test]
    fn qaoa_zz_observable_is_diagonal_ring() {
        let n = 5;
        let obs = qaoa_zz_observable(n);
        // One ZZ term per ring edge.
        assert_eq!(obs.num_terms(), n);
        assert!(obs.is_diagonal());
    }

    #[test]
    fn qaoa_cost_from_zz_matches_formula() {
        assert_eq!(qaoa_cost_from_zz(4, 1.0), 0.5 * 4.0 - 0.5);
        assert_eq!(qaoa_cost_from_zz(6, -1.0), 0.5 * 6.0 + 0.5);
        assert_eq!(qaoa_cost_from_zz(4, 0.0), 0.5 * 4.0);
    }

    #[test]
    fn ghz_circuit_gate_count() {
        let n = 5;
        let c = ghz_circuit(n);
        // 1 Hadamard + (n - 1) CNOTs
        assert_eq!(c.len(), 1 + (n - 1));
    }

    #[test]
    fn run_to_dense_produces_normalized_state() {
        let sim = default_simulator();
        let circuit = ghz_circuit(3);
        let dense = run_to_dense(&sim, &circuit);
        let norm: f64 = dense.get_all_probabilities().iter().sum();
        assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn run_to_dense_converts_sparse_state() {
        let sim = default_simulator();
        // GHZ state has exactly 2 non-zero amplitudes throughout, so its
        // density is 2 / 2^n. The executor's *effective* sparse->dense
        // threshold is capped at a measured cost-crossover density of
        // 1/1024 (see `COST_CROSSOVER_DENSITY` in
        // `execution_engine::adaptive`), not the naive 10% config default,
        // and the sparse->dense check runs on every gate (it's O(1)), so
        // the state converts as soon as density crosses that line. n=12
        // keeps density (2/4096 ≈ 0.00049) safely below 1/1024 (≈0.00098)
        // for the whole circuit, so `sim.run` returns `AdaptiveState::Sparse`.
        // Confirm that directly, then run it through `run_to_dense` to
        // exercise its sparse-to-dense conversion branch (the
        // `sparse => { ... }` arm).
        let circuit = ghz_circuit(12);
        let result = sim.run(&circuit).expect("simulation failed");
        assert!(
            matches!(result.state, AdaptiveState::Sparse { .. }),
            "expected GHZ(12) to stay sparse under the effective threshold"
        );

        let dense = run_to_dense(&sim, &circuit);
        assert_eq!(dense.num_qubits(), 12);
        let norm: f64 = dense.get_all_probabilities().iter().sum();
        assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn vqe_energy_is_finite_and_reproducible() {
        let sim = default_simulator();
        let e1 = vqe_energy(&sim, 4);
        let e2 = vqe_energy(&sim, 4);
        assert!(e1.is_finite());
        assert_eq!(e1, e2);
    }

    #[test]
    fn qaoa_cost_is_finite_and_reproducible() {
        let sim = default_simulator();
        let c1 = qaoa_cost(&sim, 4);
        let c2 = qaoa_cost(&sim, 4);
        assert!(c1.is_finite());
        assert_eq!(c1, c2);
    }

    #[test]
    fn ghz_sample_returns_two_outcomes_for_ghz_state() {
        let sim = default_simulator();
        // Enough shots that both |00..0> and |11..1> almost certainly appear.
        let distinct = ghz_sample(&sim, 3, 2000, 42);
        assert_eq!(distinct, 2);
    }

    #[test]
    fn ghz_sample_is_deterministic_given_seed() {
        let sim = default_simulator();
        let a = ghz_sample(&sim, 3, 500, 7);
        let b = ghz_sample(&sim, 3, 500, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn default_simulator_runs_a_trivial_circuit() {
        let sim = default_simulator();
        let circuit = ghz_circuit(2);
        let result = sim.run(&circuit).expect("simulation failed");
        assert_eq!(result.num_qubits(), 2);
    }

    // ------------------------------------------------------------------
    // Multi-instance variants
    // ------------------------------------------------------------------

    #[test]
    fn vqe_instance_angles_differ_from_base_and_each_other() {
        assert_eq!(vqe_theta_instance(0, 0, 4, 0), vqe_theta(0, 0, 4));
        assert_ne!(vqe_theta_instance(0, 0, 4, 1), vqe_theta_instance(0, 0, 4, 2));
        assert_eq!(vqe_phi_instance(1, 2, 4, 0), vqe_phi(1, 2, 4));
    }

    #[test]
    fn vqe_circuit_instance_same_shape_as_base() {
        let base = vqe_circuit(4);
        let instance = vqe_circuit_instance(4, 3);
        assert_eq!(base.len(), instance.len());
        assert_eq!(base.num_qubits(), instance.num_qubits());
    }

    #[test]
    fn vqe_energy_instances_are_finite_and_not_all_identical() {
        let sim = default_simulator();
        let energies = vqe_energy_instances(&sim, 4);
        assert_eq!(energies.len(), NUM_INSTANCES);
        assert!(energies.iter().all(|e| e.is_finite()));
        // Different instances perturb the angles, so the energies should
        // not all coincide (this would only fail by an astronomically
        // unlikely coincidence, or a bug that ignores `instance`).
        assert!(energies.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-9));
    }

    #[test]
    fn qaoa_params_instance_zero_matches_base() {
        assert_eq!(qaoa_params_instance(0, 0), (QAOA_GAMMA[0], QAOA_BETA[0]));
        assert_ne!(qaoa_params_instance(0, 1), qaoa_params_instance(0, 0));
    }

    #[test]
    fn qaoa_cost_instances_are_finite_and_not_all_identical() {
        let sim = default_simulator();
        let costs = qaoa_cost_instances(&sim, 4);
        assert_eq!(costs.len(), NUM_INSTANCES);
        assert!(costs.iter().all(|c| c.is_finite()));
        assert!(costs.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-9));
    }

    #[test]
    fn ghz_sample_instances_return_num_instances_values() {
        let sim = default_simulator();
        let samples = ghz_sample_instances(&sim, 4, 500);
        assert_eq!(samples.len(), NUM_INSTANCES);
        // Every instance draws from a GHZ state, so each should still see
        // both basis outcomes with 500 shots.
        assert!(samples.iter().all(|&d| d == 2));
    }

    // ------------------------------------------------------------------
    // QFT
    // ------------------------------------------------------------------

    #[test]
    fn qft_of_zero_matches_hadamard_layer_convention_independently() {
        // QFT|0...0> == H^{tensor n}|0...0> regardless of which bit-order/
        // sign convention the controlled-phase ladder uses -- a
        // convention-robust sanity check that doesn't depend on getting
        // the QFT phase convention exactly right (unlike qft_probe, which
        // does -- see that function's docs).
        let sim = default_simulator();
        let circuit = qft_circuit(4, 0);
        let state = run_to_dense(&sim, &circuit);
        let expected = 1.0 / (16.0_f64).sqrt();
        for amp in state.amplitudes() {
            assert!(
                (amp.norm() - expected).abs() < 1e-10,
                "expected uniform magnitude {expected}, got {amp:?}"
            );
            assert!(amp.im.abs() < 1e-10, "QFT|0> should be real-valued, got {amp:?}");
        }
    }

    #[test]
    fn qft_probe_matches_closed_form() {
        // QFT|k=1> = (1/sqrt(N)) * sum_y exp(2*pi*i*y/N) |y>, so
        // amp[1] = (1/sqrt(N)) * exp(2*pi*i/N) and Re(amp[1]) =
        // (1/sqrt(N)) * cos(2*pi/N). This is the exact check that caught a
        // real bug during development: an earlier version of the
        // controlled-phase ladder's loop direction compiled, produced a
        // normalized state, and passed a much looser "bounded and nonzero"
        // version of this test, while silently applying *no net phase at
        // all* for any basis-state input (every control qubit was still
        // classically |0> at the time it was used) -- the probe landed
        // exactly on the magnitude bound (0.25 at n=4) instead of the true
        // value (~0.231), which only a closed-form comparison catches.
        let sim = default_simulator();
        for n in [4usize, 6, 8, 10] {
            let probe = qft_probe(&sim, n);
            let dim = (1u64 << n) as f64;
            let expected = (1.0 / dim.sqrt()) * (2.0 * std::f64::consts::PI / dim).cos();
            assert!(
                (probe - expected).abs() < 1e-9,
                "n={n}: probe={probe} expected(closed form)={expected}"
            );
        }
    }

    #[test]
    fn qft_circuit_gate_count_is_quadratic() {
        // n state-prep X gates (just 1, for k=1) + n H + n(n-1)/2 CPhase +
        // floor(n/2) swaps.
        let n = 6;
        let c = qft_circuit(n, QFT_INPUT_K);
        let expected = 1 + n + (n * (n - 1)) / 2 + n / 2;
        assert_eq!(c.len(), expected);
    }

    // ------------------------------------------------------------------
    // Random circuit sampling
    // ------------------------------------------------------------------

    #[test]
    fn random_circuit_gate_count() {
        let n = 6;
        let c = random_circuit(n);
        // Each layer: n single-qubit gates + floor(n/2) or floor((n-1)/2)
        // entangling gates depending on the offset parity.
        let mut expected = 0usize;
        for layer in 0..RCS_LAYERS {
            expected += n;
            let offset = layer % 2;
            expected += (n - offset) / 2;
        }
        assert_eq!(c.len(), expected);
    }

    #[test]
    fn random_circuit_p0_is_finite_probability() {
        let sim = default_simulator();
        for n in [4usize, 6, 8] {
            let p0 = random_circuit_p0(&sim, n);
            assert!(p0.is_finite());
            assert!((0.0..=1.0).contains(&p0), "n={n}: p0={p0} out of [0,1]");
        }
    }

    #[test]
    fn random_circuit_entangling_layer_alternates_offset() {
        // Layer 0 pairs (0,1),(2,3),...; layer 1 pairs (1,2),(3,4),... --
        // spot-check by construction size rather than gate identity, since
        // Circuit doesn't expose per-gate qubit introspection here trivially.
        let mut even = Circuit::new(6);
        rcs_entangling_layer(&mut even, 0, 6);
        assert_eq!(even.len(), 3); // (0,1),(2,3),(4,5)

        let mut odd = Circuit::new(6);
        rcs_entangling_layer(&mut odd, 1, 6);
        assert_eq!(odd.len(), 2); // (1,2),(3,4) -- qubit 0 and 5 idle
    }
}
