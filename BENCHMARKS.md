# SimQ vs Qiskit vs qsim vs qulacs — cross-validated benchmarks

Every number in this document was produced on one machine, in one sitting, by
`./benchmarks/run.sh`, and the comparison tables are only printed after an
automatic cross-validation proves the suites simulate **identical circuits**.
The original 12 base observable values (VQE energies, QAOA cut values, GHZ
probabilities at 4/8/12/16 qubits) must agree to **1e-12** between SimQ and
Qiskit's exact `Statevector`, to **5e-6** between SimQ and qsim (Google's
simulator, whose Python wheel is float32-only — see the qsim fairness notes
below), and to **1e-12** between SimQ and qulacs, or the harness exits
without showing timings. A further 18 values (30 total) were added by the
QFT, random-circuit-sampling, and multi-instance VQE/QAOA workloads (see
"Closing a benchmark-methodology gap" below) and are checked against Qiskit
and qsim at the same tolerances — qulacs coverage of those specific
workloads is a tracked gap, not yet wired up (see that section). The
qsim/Cirq and qulacs legs are each optional (`pip install cirq qsimcirq` /
`pip install qulacs`); everything else runs without them.

## Results: SimQ vs Qiskit

Machine: Intel Xeon @ 2.10 GHz, 4 vCPUs, 15 GiB RAM (cloud container), Linux.
Toolchain: rustc 1.94.1 (release profile), Python 3 with Qiskit 2.5.0 /
qiskit-aer 0.17.2. Cross-validation: worst deviation across the 30 checked
values (12 base + 8 QFT/random-circuit + 5 VQE-multi-instance, see
"Closing a benchmark-methodology gap" below) was **1.51e-14** (tolerance
1e-12). Medians; `ratio` = Qiskit time ÷ SimQ time, so ratio > 1 means SimQ
is faster.

| Workload | SimQ (ms) | Statevector (ms) | ratio | Aer (ms) | ratio |
|----------|----------:|------------------:|------:|---------:|------:|
| vqe_energy/4q       |  0.035 |   1.960 |   56.2× |  1.377 |  39.5× |
| vqe_energy/8q       |  0.082 |   3.635 |   44.3× |  2.064 |  25.2× |
| vqe_energy/12q      |  0.555 |   8.331 |   15.0× |  4.121 |   7.4× |
| vqe_energy/16q      |  9.846 |  75.294 |    7.6× | 14.390 |   1.5× |
| qaoa_maxcut/4q      |  0.027 |   1.625 |   60.1× |  1.389 |  51.4× |
| qaoa_maxcut/8q      |  0.057 |   4.010 |   70.2× |  2.606 |  45.6× |
| qaoa_maxcut/12q     |  0.637 |   7.248 |   11.4× |  5.122 |   8.0× |
| qaoa_maxcut/16q     |  8.593 |  68.928 |    8.0× | 13.156 |   1.5× |
| ghz_sampling/4q     |  0.035 |   0.979 |   28.0× |  1.169 |  33.5× |
| ghz_sampling/8q     |  0.036 |   2.584 |   71.9× |  1.656 |  46.1× |
| ghz_sampling/12q    |  0.049 |  17.186 |  348.0× |  2.057 |  41.7× |
| ghz_sampling/16q    |  0.258 | 653.578 | 2533.8× |  4.129 |  16.0× |
| qft_probe/4q        |  0.017 |   0.768 |   45.1× |  1.237 |  72.6× |
| qft_probe/8q        |  0.065 |   2.398 |   36.9× |  2.427 |  37.4× |
| qft_probe/12q       |  0.957 |   7.027 |    7.3× |  6.230 |   6.5× |
| **qft_probe/16q**   | **25.986** | 69.171 |   2.7× | 15.563 | **1/1.7×** |
| random_circuit/4q   |  0.039 |   1.595 |   41.0× |  1.706 |  43.8× |
| random_circuit/8q   |  0.080 |   3.522 |   43.9× |  2.707 |  33.8× |
| random_circuit/12q  |  0.528 |   8.203 |   15.5× |  4.928 |   9.3× |
| random_circuit/16q  |  7.258 |  76.723 |   10.6× | 15.170 |   2.1× |

**SimQ is faster on 19 of 20 workloads against Aer** (1.5–72.6×) **and all 20
against exact Statevector** (2.7–2533.8×). The one Aer loss —
`qft_probe/16q` at 1/1.7× — is the direct, structural consequence of this
codebase's gate-fusion pass being local and width-bounded against QFT's
long-range controlled-phase gates; see "Closing a benchmark-methodology gap"
below for the full analysis. Before the issue #76 fixes (see below) SimQ
*lost* the original 16-qubit rows by 22–30×; the same suite produced both
sets of numbers, so the turnaround is measured, not asserted.

## Results: SimQ vs qsim (Google)

Same machine and circuits as above. Toolchain: Python 3 with `cirq` 1.7.0 /
`qsimcirq` 0.22.0. Cross-validation: worst deviation across the 30 checked
values was **2.29e-06** (tolerance 5e-6, loosened from Qiskit's 1e-12
because qsim's Python wheel carries no fp64 option — see Fairness notes).
Medians; `ratio` = qsim/Cirq time ÷ SimQ time.

| Workload | SimQ (ms) | Cirq (ms) | ratio | qsim (ms) | ratio |
|----------|----------:|----------:|------:|----------:|------:|
| vqe_energy/4q       |  0.035 |  3.194 |   91.6× | 0.810 |   23.2× |
| vqe_energy/8q       |  0.082 |  6.240 |   76.1× | 1.555 |   19.0× |
| vqe_energy/12q      |  0.555 | 10.213 |   18.4× | 2.394 |    4.3× |
| vqe_energy/16q      |  9.846 | 29.536 |    3.0× | 5.775 |  1/1.7× |
| qaoa_maxcut/4q      |  0.027 |  3.168 |  117.2× | 0.802 |   29.7× |
| qaoa_maxcut/8q      |  0.057 |  6.327 |  110.8× | 1.517 |   26.6× |
| qaoa_maxcut/12q     |  0.637 | 10.194 |   16.0× | 2.282 |    3.6× |
| qaoa_maxcut/16q     |  8.593 | 31.802 |    3.7× | 4.776 |  1/1.8× |
| ghz_sampling/4q     |  0.035 |  2.692 |   77.1× | 1.079 |   30.9× |
| ghz_sampling/8q     |  0.036 |  4.307 |  119.7× | 1.599 |   44.5× |
| ghz_sampling/12q    |  0.049 |  6.367 |  128.9× | 2.171 |   43.9× |
| ghz_sampling/16q    |  0.258 |  9.094 |   35.3× | 3.346 |   13.0× |
| qft_probe/4q        |  0.017 |  1.186 |   69.6× | 0.309 |   18.1× |
| qft_probe/8q        |  0.065 |  2.815 |   43.4× | 0.785 |   12.1× |
| qft_probe/12q       |  0.957 |  5.615 |    5.9× | 1.603 |    1.7× |
| **qft_probe/16q**   | **25.986** | 12.303 | **1/2.1×** | 5.203 | **1/5.0×** |
| random_circuit/4q   |  0.039 |  3.005 |   77.2× | 0.802 |   20.6× |
| random_circuit/8q   |  0.080 |  5.720 |   71.3× | 1.516 |   18.9× |
| random_circuit/12q  |  0.528 |  9.593 |   18.2× | 2.405 |    4.6× |
| **random_circuit/16q** | **7.258** | 31.274 |  4.3× | 4.702 | **1/1.5×** |

**SimQ is faster on 18 of 20 workloads against Cirq's pure-Python reference
simulator** (1.7–128.9×; the two losses are both `qft_probe`, discussed
below) **and on 16 of 20 against qsim's optimized AVX/SSE C++ core**. The
four qsim losses are all at 16 qubits: `vqe_energy` (1/1.7×), `qaoa_maxcut`
(1/1.8×), `qft_probe` (1/5.0×), and — the more interesting one —
`random_circuit` (1/1.5×). qsim is the strongest of the three Qiskit/Cirq/qsim
backends everywhere it's measured; see "Closing a benchmark-methodology gap"
for why `random_circuit`'s loss specifically revises this document's earlier
(machine-dependent) claim that it "wins comfortably at every size."

## Results: SimQ vs qulacs

Same machine and circuits as above. Toolchain: Python 3 with `qulacs` 0.6.13
(a single execution path — unlike Qiskit/Cirq, qulacs does not ship a
separate pure-Python reference simulator alongside its optimized core, so
there is one timing column here, not two). Cross-validation: worst deviation
across the 12 base values was **8.44e-15** (tolerance 1e-12 — qulacs, like
SimQ and Qiskit, is a float64 core, so it gets the tight tolerance, not
qsim's loosened one); `qulacs_baseline.py` does not yet cover
`qft_probe`/`random_circuit`/the multi-instance workloads (18 of the 30
values checked against Qiskit/qsim), so those are not cross-validated or
timed against qulacs below — a tracked gap, not an oversight (see Honest
limitations). Medians; `ratio` = qulacs time ÷ SimQ time.

| Workload | SimQ (ms) | qulacs (ms) | ratio |
|----------|----------:|------------:|------:|
| vqe_energy/4q    | 0.035 |  0.053 |    1.5× |
| vqe_energy/8q    | 0.082 |  0.089 |    1.1× |
| vqe_energy/12q   | 0.555 |  0.962 |    1.7× |
| vqe_energy/16q   | 9.846 | 14.070 |    1.4× |
| qaoa_maxcut/4q   | 0.027 |  0.047 |    1.7× |
| qaoa_maxcut/8q   | 0.057 |  0.078 |    1.4× |
| qaoa_maxcut/12q  | 0.637 |  0.857 |    1.3× |
| qaoa_maxcut/16q  | 8.593 |  8.639 |    1.0× |
| ghz_sampling/4q  | 0.035 |  0.064 |    1.8× |
| ghz_sampling/8q  | 0.036 |  0.055 |    1.5× |
| ghz_sampling/12q | 0.049 |  0.093 |    1.9× |
| ghz_sampling/16q | 0.258 |  0.302 |    1.2× |

**SimQ is faster on all 12 covered workloads against qulacs**, the strongest
of the three published competitors on the base suite — where Aer/Cirq/qsim
lose 19/18/16 of 20 workloads, qulacs stays within roughly 1.0–1.9× of SimQ
everywhere covered, with `qaoa_maxcut/16q` essentially a dead heat (1.0×).
That "all 12" is a recent result, not a given: qulacs *won*
`ghz_sampling/16q` (0.462 ms vs 0.790 ms, 1.7× ahead) the first time this
table was measured. Chasing that one loss down — at the suggestion that
memory allocation was the likely culprit — found and fixed a real O(2^n)
inefficiency in SimQ's own sampling path (see the next section), closing
today's `ghz_sampling/16q` gap to 1.2× in SimQ's favor. This is the most
instructive result in this document, not the embarrassing one: a competitor
benchmark exposed a genuine defect that pure unit testing hadn't caught, and
fixing it was a net win for every sampling-heavy workload at every qubit
count, not just this one row.

A genuine surprise found while wiring qulacs up, not something assumed from
its docs: **qulacs's `add_RX_gate`/`add_RY_gate`/`add_RZ_gate` use the
opposite sign convention from Qiskit/Cirq/simq** — `exp(+i·θ/2·P)` instead of
the near-universal `exp(-i·θ/2·P)`. Verified empirically (see
`benchmarks/qulacs_baseline.py`'s docstring): `qulacs.gate.RY(θ)` applied to
`|0⟩` produces `[cos(θ/2), −sin(θ/2)]`, the negated-angle result, not
`[cos(θ/2), +sin(θ/2)]`. Every angle passed to qulacs in this benchmark is
therefore negated to implement the same unitary as the other three suites —
exactly the kind of convention mismatch the cross-validation gate exists to
catch, and it would have silently produced wrong "reference" values (and a
failed cross-check) had it gone unnoticed.

## Why qulacs won `ghz_sampling/16q`, and the fix

**The circuit is nearly free; the measurement wasn't.** A GHZ state has
exactly 2 nonzero amplitudes — `|00…0⟩` and `|11…1⟩` — no matter how many
qubits it spans. Simulating H + a CNOT chain is trivial even at 16 qubits.
So a 0.790 ms result for `ghz_sampling/16q` (vs. several milliseconds for
`vqe_energy/16q`'s much deeper circuit, measured in the same investigation)
was itself a clue that almost all of that time had to be going somewhere
other than gate application.

Profiling `simq-state`'s `ComputationalBasis::sample` (the function
`ghz_sample` in `simq/src/bench_workloads.rs` calls to draw the 1024 shots)
confirmed it: computing the probability vector
(`DenseState::get_all_probabilities`, already SIMD-optimized) took **37 µs**,
but the full `sample()` call — the same computation plus building an alias
table and drawing 1024 shots from it — took **813 µs**. Over 95% of the
time was in `AliasTable::new`, which builds Vose's alias method table:
several more `Vec` allocations the size of the full **2^n = 65,536-entry**
Hilbert space (`prob`, `alias`, a `scaled` copy of the probabilities, a
`prob_copy` clone of that, plus growable `small`/`large` partition stacks),
built via inherently sequential, non-vectorizable stack popping/pushing —
**for a distribution with a support of exactly 2.** This is precisely the
"memory" angle worth chasing: the cost scaled with the size of the state
vector, not with the number of shots or the number of outcomes that could
ever actually be sampled.

**The fix** (`simq-state/src/measurement.rs`): compact the probability
vector to its actual support — every basis index with probability above the
crate's existing 1e-14 negligible-amplitude cutoff (the same convention
`dense_state.rs`/`sparse_state.rs` already use elsewhere) — in one linear
pass *before* building the alias table, then build the table over that
(potentially much smaller) compacted array and map sampled indices back to
the original basis states. The compaction pass costs no more than the
probability computation it rides alongside; the payoff is that
`AliasTable::new`'s allocations and construction cost now scale with **the
number of outcomes that can actually occur**, not with 2^n. For GHZ, that's
2 instead of 65,536 — for any other circuit whose output distribution isn't
already maximally spread across the full Hilbert space, some smaller but
still real win. A circuit that genuinely is maximally spread (all 2^n
amplitudes non-negligible, e.g. plain full superposition with no
structure) sees no asymptotic loss either: the compacted array is the same
size as before, plus one cheap linear filter pass.

**Result:** `sample(1024 shots)` on the 16-qubit GHZ state dropped from 813
µs to **97 µs** (8.4×) in isolation — a controlled, isolated A/B measurement
on this same machine, unaffected by cross-run noise — and the full
`ghz_sampling/16q` benchmark dropped from 0.790 ms to a range of
**0.258–0.302 ms** across the sessions this document draws from (still 2.6–3×
faster than the pre-fix number). `vqe_energy` and `qaoa_maxcut` never call
this sampling path, so they're unaffected. All 378 `simq-state` unit/e2e
tests pass unchanged; the fix only changes *which* entries `AliasTable` is
built from, not the sampling distribution itself, and the existing test that
samples a 2-of-4 support state (`test_batch_sampling`) covers exactly this
"already-sparse `DenseState`" path.

## Closing a benchmark-methodology gap: QFT, random circuits, multi-instance

The 12 base workload/backend combinations in the tables above are all built
from three circuit shapes — VQE, QAOA-MaxCut, GHZ — and all three share two
properties: every two-qubit gate acts on **adjacent qubits** in a **fixed
pattern that repeats every layer** (a linear chain for VQE, a ring for QAOA,
a chain for GHZ), and each is timed against **one fixed, hand-picked,
deterministic circuit instance** per qubit count. Neither property is unique
to this project's methodology — but a locality-exploiting compiler
optimization was added to SimQ's fusion pass in this repo (see "Issue #76
follow-up" below: gate fusion collapses adjacent local gates into
width-bounded blocks), which means a benchmark suite containing *only*
local, fixed-pattern circuits risks making that specific optimization look
better than it generalizes, and never tests whether the published numbers
are an artifact of the one circuit instance each workload happens to use.

This is a recognized methodology concern, not a project-specific worry:
QED-C's application-oriented benchmarking guidance (arXiv:2110.03137)
recommends multiple random circuit instances per problem size specifically
to guard against tuning results to one convenient circuit; MQT Bench
(arXiv:2204.13719) and the broader simulator-comparison literature (e.g.
arXiv:2401.09076, which benchmarks Aer/qsim-class simulators using exactly
Trotter circuits, **random circuit sampling**, and **QFT**) consistently
include non-local, structure-agnostic, and analytically-checkable circuit
families alongside application-specific ansätze — categories this repo's
suite didn't have. Three additions close that gap; their results are folded
into the "Results" tables above (rows are labeled `qft_probe`,
`random_circuit`, and the "Multi-instance results" section further down):

- **`qft_probe`** — textbook QFT. Its controlled-phase gates are **long-range**
  (qubit `i` connects to every qubit `j > i`, gate count O(n²)) — the direct
  structural opposite of VQE/QAOA/GHZ's fixed local chain, and exactly the
  shape a width-bounded *local* fusion pass can't help with. QFT's output
  also has uniform amplitude *magnitude* regardless of correctness (a
  defining property of the transform), so unlike every other workload here,
  its cross-checked value is a raw amplitude component (`Re(amplitude)`),
  not a measurement probability — a probability-based check would pass even
  for a completely broken phase ladder. That distinction caught a real bug
  during development: an earlier version of the controlled-phase loop
  compiled, produced a normalized state, and passed a looser
  bounds-only test, while silently applying *no net phase at all* (the
  ordering issue is documented at length as a code comment on
  `qft_circuit` in `simq/src/bench_workloads.rs`, since it is exactly the
  kind of mistake that looks correct without a value-level check).
- **`random_circuit`** — 8 alternating layers of single-qubit gates (type
  chosen deterministically per (layer, qubit) from {H, S, T, SX, RY}) and
  brickwork CZ layers whose qubit pairing **alternates every layer** —
  structurally the standard random-circuit-sampling shape (the family
  behind Google's/IBM's supremacy/utility circuits), with no fixed
  repeating pattern for a structure-aware optimizer to exploit. Uses
  deterministic index-based formulas for gate choice, not a seeded PRNG —
  a seeded PRNG's bit-level output isn't guaranteed to match across the
  Rust/Qiskit/Cirq runtimes, which this suite's cross-validation needs.
- **Multi-instance VQE/QAOA/GHZ** (`{vqe_energy,qaoa_maxcut}_multi/8q_i0..4`,
  `ghz_sampling_multi_instance/8q`) — the same ansätze, run at 5
  deterministically-offset parameter sets, at one representative qubit
  count (8; not swept across all four sizes, to bound how much slower this
  makes `benchmarks/run.sh`). GHZ has no continuous parameter to perturb
  this way (it's parameter-free), so its multi-instance variant instead
  varies the shot-sampling seed and is timing-only, not a new
  cross-checked value.

**The honest finding this exercise was for: `qft_probe/16q` is a loss for
SimQ against every backend** — 1/1.7× vs Aer, 1/2.1× vs Cirq, 1/5.0× vs
qsim (only exact-statevector references, which do no fusion/optimization of
their own, still lose to SimQ: 2.7× vs Statevector). This is not noise or a
machine artifact: it is the direct, structural consequence of this
project's own fusion work being width-bounded and *local* — QFT's
controlled-phase gates connect qubits up to 15 apart at n=16, far outside
any 3-qubit local block, so fusion simply doesn't engage for most of the
circuit, while Aer's, Cirq's, and (especially) qsim's own
fusion/optimization evidently handles the long-range case better. Note that
the width-bounded multi-qubit fusion pass doesn't even *activate* below 18
qubits (see "Issue #76 follow-up"), so this 16-qubit loss is not really
about that pass at all — it reflects how far a plain local
single-qubit-chain fusion (the pre-existing, unmodified code path at this
size) can carry a fundamentally non-local circuit, which is "not very far."

**`random_circuit` is a more nuanced finding than first measured, and this
document's earlier claim needs a correction.** On an 8 GiB Apple Silicon
machine (where these workloads were first measured, before being folded
into the reference-machine tables above), `random_circuit` won at every
size including 16q, against all four backends. Re-measured here on the 15
GiB Intel Xeon reference machine, that holds against Statevector (10.6×),
Aer (2.1×), and Cirq (4.3×) — **but `random_circuit/16q` loses to qsim,
1/1.5×**, the same pattern as `qft_probe`. `random_circuit`'s brickwork
connectivity alternates which pairs are adjacent each layer, but every
individual two-qubit gate is still local within that layer, so unlike QFT
this isn't a locality story — it looks instead like qsim's C++ core simply
being the strongest backend measured at 16 qubits across multiple workload
shapes on this specific machine, a finding the Apple Silicon run's
otherwise-clean sweep didn't surface. **The corrected, honest picture:**
`random_circuit` generalizes SimQ's advantage over Aer/Cirq/Statevector
cleanly; it does not generalize SimQ's advantage over qsim at 16 qubits,
and neither did this document's original claim that it did — that was a
measurement artifact of using a different machine, now fixed by remeasuring
on the reference box this document's other numbers all come from.

The multi-instance results (see "Multi-instance results" below) show a
reassuring finding: all 5 VQE and 5 QAOA instances cross-validated cleanly
against both Qiskit and qsim, so the headline VQE/QAOA numbers in the main
tables are not an artifact of the one parameter set each workload happens
to use.

### Multi-instance results

Same machine, same cross-validation run as the main tables above (30
checked values total; see their headers for the worst-deviation numbers).
Each row times the **whole 5-instance batch** as one unit, not one instance
— see `simq::bench_workloads::NUM_INSTANCES` docs.

| Workload (5 instances) | SimQ (ms) | Statevector | ratio | Aer | ratio | Cirq | ratio | qsim | ratio |
|----------|----------:|------------:|------:|----:|------:|-----:|------:|-----:|------:|
| vqe_energy_multi_instance/8q  | 0.415 | 18.059 | 43.5× | 10.096 | 24.3× | 31.461 | 75.9× | 8.400 | 20.3× |
| qaoa_cost_multi_instance/8q   | 0.290 | 16.660 | 57.4× | 13.011 | 44.8× | 32.189 |111.0× | 8.148 | 28.1× |
| ghz_sampling_multi_instance/8q| 0.179 |  9.803 | 54.9× |  7.907 | 44.3× | 22.400 |125.4× | 8.543 | 47.8× |

SimQ wins every multi-instance row against every backend, by margins in
line with the corresponding single-instance 8q rows above — consistent with
the reassuring finding above that the single-instance numbers generalize
across parameter sets.

## Workloads

Every workload is defined twice — in Rust (`simq/src/bench_workloads.rs`,
consumed by both the criterion bench and the cross-check binary) and in
Python (`benchmarks/qiskit_baseline.py` and `benchmarks/qsim_baseline.py`)
— with deterministic, qubit-asymmetric parameters so any qubit-ordering or
convention mismatch fails the cross-check instead of hiding in a symmetric
circuit. See "Closing a benchmark-methodology gap" below for why the bottom
three rows exist.

| Workload | Circuit | Measured quantity |
|----------|---------|-------------------|
| `vqe_energy/Nq` | H layer + 3 × [RY(θ_lq) ⊗ CNOT-chain ⊗ RZ(φ_lq)], θ/φ deterministic per (layer, qubit) | ⟨H⟩ with H = Σ Z_q Z_{q+1} + 0.5 Σ X_q |
| `qaoa_maxcut/Nq` | Ring-graph MaxCut QAOA, p = 2, cost blocks CNOT·RZ(2γ)·CNOT, mixers RX(2β) | cut value 0.5·N − 0.5·⟨Σ Z_q Z_{q+1}⟩ |
| `ghz_sampling/Nq` | H(0) + CNOT chain | 1024 measurement shots |
| `qft_probe/Nq` | Textbook QFT (H + controlled-phase ladder + swaps) applied to \|k=1⟩ | Re(amplitude at basis state \|1⟩) |
| `random_circuit/Nq` | 8 layers alternating {H,S,T,SX,RY} (deterministic per (layer,qubit)) with brickwork CZ, pairing alternating each layer | p(\|0...0⟩) |
| `{vqe_energy,qaoa_maxcut}_multi/8q_i{0..4}` | Same ansatz as above, 5 deterministic parameter offsets | same as above, per instance |

One benchmark iteration = one full cost-function evaluation, i.e. what a
variational optimizer pays per step.

## Fairness notes

These all favor Qiskit/qsim/qulacs; SimQ's wins are understated, not overstated:

- **SimQ rebuilds and re-optimizes its circuit inside every timed iteration**
  (its compiler runs at default `O2` in the loop). The Qiskit, Cirq, and
  qulacs circuits are all built once outside the timed region, and the Aer
  circuits are additionally **transpiled once outside the timed region**.
- Aer energy evaluations use `save_expectation_value`, so a single `run()`
  returns the energy with no statevector round-trip through Python; qulacs's
  `Observable.get_expectation_value` is the same shape (no round-trip).
- Aer and qulacs both run their own default multithreading; SimQ's defaults
  keep gate kernels single-threaded below 2^18 amplitudes because that is
  faster at these sizes (fork/join overhead exceeds the memory-bound kernel
  work) — these 4–16-qubit workloads are all below that threshold.
- Timings are medians: criterion's median estimate for SimQ, median of
  50/50/15/5 repetitions (4q/8q/12q/16q) after a warmup for Qiskit, qsim, and
  qulacs.
- **qsim's Python wheel (`qsimcirq`) ships a single-precision (float32) state
  vector core with no fp64 build option.** That is a genuine precision
  disadvantage baked into the distributed package, not something this harness
  imposes — it is why the qsim cross-check tolerance is 5e-6 instead of
  Qiskit's 1e-12, and the observable diffs against qsim (up to 3.28e-06) are
  consistent with float32 rounding compounding over each circuit's gate
  count, not a bug in either simulator. qulacs, unlike qsim, is float64, so
  it gets the tight 1e-12 tolerance.
- qsim's `qsimcirq.QSimSimulator` also runs its own default multithreading
  (AVX/SSE-vectorized C++ core), same as Aer and qulacs.
- qulacs has one execution path, not a slow-reference/fast-optimized split
  like the other two suites, so its table has a single timing column.

## Reproduction

```bash
pip install qiskit qiskit-aer cirq qsimcirq qulacs   # qsim/cirq/qulacs are optional
./benchmarks/run.sh
```

The pieces, if you want them separately:

```bash
cargo bench -p simq --bench end_to_end          # SimQ timings (criterion)
python3 benchmarks/qiskit_baseline.py           # Qiskit timings + reference values
python3 benchmarks/qsim_baseline.py             # qsim/Cirq timings + reference values (optional)
python3 benchmarks/qulacs_baseline.py           # qulacs timings + reference values (optional)
python3 benchmarks/compare.py                   # cross-check (1e-12 Qiskit/qulacs, 5e-6 qsim), then table
cargo run --release -p simq --example xcheck_bench  # SimQ observable values as JSON
```

The 18–30q scaling probes in "Where it actually fails" are separate,
manually-run scripts (not part of `run.sh`, since they can take minutes per
qubit count and the higher sizes need multiple GiB of free RAM) — run
individual sizes to bound memory, e.g. `... 20 22 24`:

```bash
cargo run --release -p simq-sim --example scaling_probe -- 20 22 24 26 28 30
python3 benchmarks/aer_scaling_probe.py 20 22 24 26 28 30
python3 benchmarks/qulacs_scaling_probe.py 20 22 24 26 28   # 30q not attempted, see Honest limitations
```

Update `benchmarks/scaling_results.json` with any new numbers, then regenerate
both charts:

```bash
python3 benchmarks/make_chart.py            # results-{light,dark}.svg (4-16q workloads)
python3 benchmarks/make_scaling_chart.py     # results-scaling-{light,dark}.svg (18-30q)
```

## How the ≥12-qubit gap was closed (issue #76)

The first run of this suite showed SimQ **losing 22–30×** to Aer at 16 qubits
(~2.5 ms/gate). Profiling traced it to five compounding defects on the gate
hot path, not to any single kernel:

1. **The state never left the sparse representation.** The executor's sparse
   kernels bypass the sparse state's incremental density bookkeeping, so the
   adaptive sparse→dense conversion never fired and 16-qubit circuits ran
   entire workloads on an `AHashMap` rebuilt per gate. Fixed by refreshing the
   density after each sparse gate (O(1)) — and by running adaptive conversion
   in the parallel execution path at all, which previously skipped it.
2. **Rayon micro-tasking.** Parallel kernels split gate application into
   per-pair/per-index tasks (down to 2 amplitudes per task, with
   `filter`-scans over all 2^n indices for CNOT/CZ/SWAP). All kernels now use
   cache-sized blocks (≥128 KiB per task, subspace enumeration instead of
   filtered scans), and parallelism only engages above 2^18 amplitudes, where
   it actually pays for itself.
3. **A qubits-vs-amplitudes unit bug** made every state with ≥8 *amplitudes*
   (the documented default was 8 *qubits*) take the parallel path.
4. **O(2^n) side-scans per gate**: telemetry recorded a full-state density
   scan after every gate, and the dense→sparse conversion check ran another
   one. Density telemetry now reads the O(1) sparse counter (dense states
   report 1.0), and the expensive check runs every 64 gates.
5. **Amplitude truncation** (sparse kernels dropped |amp| < 3e-8, the
   observable path skipped them too), which cost ~1e-9 of accuracy at 16
   qubits — found by this suite's own cross-check. The cutoffs are now 1e-14
   of amplitude / exact zero, and the worst deviation vs Qiskit is ~1.5e-14.

On top of that, single-qubit kernels detect diagonal gates (RZ/Phase/S/T) and
halve their memory traffic, and Pauli expectation values are evaluated with
bitmask + popcount in one allocation-free pass instead of walking the Pauli
vector per amplitude per term.

These are the standard optimizations of high-performance statevector
simulators — cache blocking, coarse-grained threading, gate specialization,
and gate fusion (SimQ's compiler already provided fusion) — as described for
[qsim/ProjectQ-style fusion and cache blocking](https://arxiv.org/pdf/2604.12256),
[QuEST](https://arxiv.org/pdf/1802.08032), and
[NUMA/SIMD-aware kernel design](https://arxiv.org/html/2506.09198v2); Aer
itself applies [cache-blocking remaps](https://arxiv.org/pdf/2604.12256) for
large circuits.

## Where it actually fails: 18–30 qubits

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benchmarks/results-scaling-dark.svg">
  <img alt="SimQ vs Aer vs qulacs scaling from 20 to 30 qubits, log scale. SimQ leads through 28 qubits; all three simulators hit a hard wall at 30 qubits (16 GiB state vs. 15 GiB RAM), shown as a dashed 'rejected / not attempted' marker rather than a bar." src="benchmarks/results-scaling-light.svg">
</picture>

We pushed the same 1-layer VQE circuit (H + RY + CNOT-chain + RZ, built by
`simq-sim/examples/scaling_probe.rs`, and its Python ports
`benchmarks/aer_scaling_probe.py` and `benchmarks/qulacs_scaling_probe.py`)
beyond the published table to find the edges. Same machine as every other
number in this document (15 GiB RAM, 4 vCPUs, this session); SimQ at default
settings including its optimizer in the timed run, Aer timed on a
once-transpiled circuit including final statevector retrieval, qulacs timed
on state creation + `update_quantum_state` (its equivalent of a full run).

**All three columns below — SimQ, Aer, and qulacs — were measured together
in this same sitting, on this same machine**, closing a gap this document
used to flag explicitly: earlier versions carried the Aer column over from a
different measurement session on a different instance of the same machine
class (2.80 GHz vs. this session's 2.10 GHz), because no Aer-side scaling
probe existed yet. `benchmarks/aer_scaling_probe.py` now exists specifically
to close that gap. Each value is the median of 3 fresh-process runs (a cold
first run at 28q was ~2.3× slower than steady state for SimQ, ~1.9× slower
for qulacs, and unaffected for Aer — see note below the table):

| Qubits | State | SimQ | Aer | qulacs | Verdict (vs Aer) |
|-------:|------:|-----:|----:|-------:|---------|
| 20 | 16 MiB | 17.9 ms | 37.2 ms | 30.3 ms | **SimQ 2.1× faster** |
| 22 | 64 MiB | 88.3 ms | 170.1 ms | 141.8 ms | **SimQ 1.9× faster** |
| 24 | 256 MiB | 580.9 ms | 783.5 ms | 855.2 ms | **SimQ 1.35× faster** |
| 26 | 1 GiB | 2.86 s | 3.16 s | 4.55 s | **SimQ 1.1× faster** |
| 28 | 4 GiB | 11.85 s | 13.46 s | 19.54 s | **SimQ 1.14× faster** |
| 30 | 16 GiB | rejected: "max supported is 29" | rejected: "maximum (29) in the coupling_map" | not attempted (see below) | physics: state > RAM |

So: **SimQ leads at every measured size from 20 through 28 qubits against
both Aer and qulacs**, and 30 remains a hard wall on this box (16 GiB state,
15 GiB RAM) for both simulators that attempt it. The margin against Aer
narrows steadily with qubit count (2.1× → 1.14×) rather than staying flat or
widening — a more modest, and now fully machine-consistent, picture than
this document's previous carried-over Aer numbers implied (which showed a
1.55× SimQ win at 28q using an Aer figure from a faster-for-Aer instance).
Against qulacs the margin is roughly flat at 1.4–2.1× throughout 20–28q — the
opposite trend from the 4–16q table, where qulacs was the closest
competitor by far and briefly beat SimQ outright (see "Why qulacs won
`ghz_sampling/16q`" above). qulacs was **not** run at 30 qubits: unlike
SimQ/Aer, it has no built-in memory-aware qubit cap and would have simply
attempted the 16 GiB allocation on a 15 GiB box, risking an OOM kill of the
whole container rather than a clean error — not worth the risk to confirm a
result the physics already answers.

What the probing found (and what got fixed along the way):

- **Sparse warm-up cliff (fixed).** Sparse→dense conversion used to trigger
  at 10% density — a relative threshold, so bigger registers hashed longer:
  at 24q the warm-up AHashMap grew to ~1.6M entries and consumed 55% of total
  runtime (1.3 s of 2.4 s). The conversion point is now the measured
  hash-vs-dense cost crossover (~1/1024 density), cutting the 26q warm-up
  from ~2.6 s to ~0.6 s.
- **28-qubit gap (closed by `scaling_30q`).** SimQ's dense kernels match
  Aer per pass (~0.6 ns/amplitude at 26q), but SimQ used to sweep the state
  once per gate (111 passes × 8 GiB of traffic at 28q) while Aer's runtime
  gate fusion collapses circuits into multi-qubit blocks and sweeps far
  fewer times. `simq-compiler`'s `fusion.rs` now does the same: a greedy,
  width-bounded frontier merge (`find_fusion_blocks`) fuses adjacent gates
  spanning up to `max_block_width` qubits (default 3, matching the "2–3-qubit"
  target this doc previously called out) into one composed unitary, cutting
  the number of full-state passes accordingly. Because a VQE/QAOA outer loop
  recompiles the *same-shaped* circuit on every step with only rotation
  angles changing, `fusion_cache.rs` additionally caches the fusion *block
  structure* (which operation indices group into which block — never the
  concrete matrix, which is always recomputed from that step's angles) keyed
  by a structural circuit fingerprint, so repeated evaluations skip
  re-deriving the same block partition. Net effect: 28q went from 58.6 s
  (measured before this work) to a steady-state 11.85 s (4.9×), enough to
  flip the 28q verdict from an Aer win to a SimQ win — 1.14× on this
  session's own fresh Aer measurement, larger still (3.4×) against the
  carried-over Aer figure this document used before `aer_scaling_probe.py`
  existed.
- **30-qubit failure mode (fixed, unchanged by this update).** A 30q dense
  state is 16 GiB; with 15 GiB of RAM both simulators must refuse — but SimQ
  used to *abort the process* inside an infallible `Vec` allocation (and,
  once the conversion was made fallible, would limp along sparse until the
  hash map itself blew up at ~234M entries). `Simulator::run` now derives the
  qubit cap from `MemAvailable` when no memory limit is configured and
  rejects upfront with `TooManyQubits`, exactly like Aer's clean "circuit too
  wide" error (independently reconfirmed this session: Aer raises
  `CircuitTooWideForTarget` at transpile time with the same "maximum (29)"
  message). Dense conversion also allocates once (null-checked) instead of
  twice. 30 qubits is a fundamental memory ceiling on a 15 GiB box (a
  double-precision statevector needs 16 GiB just for the amplitudes, before
  any working memory), not something deeper fusion can move — fusion cuts
  passes over a state that fits, it doesn't shrink the state itself.

## Issue #76 follow-up: deeper gate fusion

`simq-compiler`'s fusion pass previously only fused chains of *single-qubit*
gates on one wire — it broke the moment it hit a CNOT/CZ, which is why the
28q loss above happened at all. It now also does greedy, width-bounded
multi-qubit block fusion (up to 3 qubits by default), gated so it only ever
activates for circuits at or above an 18-qubit threshold (matching this
codebase's existing `parallel_threshold` — the point where per-pass memory
traffic starts to dominate over per-gate dispatch overhead). Below that
threshold, circuits are dispatched to the *original, unmodified*
single-qubit fusion function — not a re-implementation that behaves the
same, the literal pre-existing code path — so the 4–16q results in the
tables above are structurally guaranteed unaffected, not just expected to
be. Two additional pieces ride on top of the core fusion change:

- A **topology-keyed cache** for fusion block *structure* (which operations
  group into which blocks — never the composed matrix values), keyed by a
  structural circuit fingerprint that deliberately excludes gate parameter
  values. `Simulator` holds one across repeated `run()` calls, so a VQE/QAOA
  optimizer's outer loop — which recompiles the same-shaped circuit
  hundreds of times with only rotation angles differing — skips re-deriving
  which gates fuse together on every call; the actual fused matrices are
  still always recomputed fresh from each call's angles, so this can't
  return a stale result for a changed parameter (verified by a dedicated
  test that runs the same circuit shape through one `Simulator` at several
  different angles and checks every result against an unoptimized
  baseline).
- Fusion width is **gated on qubit count alone** (a static, one-comparison
  check, not a cost model or graph search) — this was a deliberate design
  choice made after a prior-art/patent search surfaced two granted IBM
  patents in this exact space (graph-based fusion scheduling with a
  shortest-path-selected schedule; and fusion of measurement gates
  specifically). This implementation's greedy, single-forward-pass,
  width-bounded approach — and its hard stop at any gate without a matrix
  representation, which includes measurement — stays outside both patents'
  specific claimed mechanisms and matches the simpler "greedy gate fusion"
  baseline Google's own qsim documents using. That diligence pass is not a
  substitute for legal review before any commercial release.

**Verification done, and what's still open:**

- Correctness: the full cross-validation harness (`benchmarks/compare.py`)
  was re-run after this change and still passes — worst deviation 3.19e-14
  vs Qiskit's exact `Statevector` (tolerance 1e-12) and 2.94e-06 vs qsim
  (tolerance 5e-6), unchanged from before this work. New unit and
  integration tests (`simq-compiler`'s `fusion.rs`/`fusion_cache.rs`,
  `simq-sim`'s `comprehensive_e2e.rs`) additionally check fused-vs-unfused
  execution agreement at the dense-kernel level, multiple qubit orderings
  (the bit-position convention between this crate's embedding math and the
  kernels' argument order is the single highest-risk spot in this kind of
  change), and that the fusion-structure cache never returns a stale result
  across differing gate parameters.
- Scaling, as measured *in this session*: the re-run happened on an 8 GiB
  Apple Silicon machine, not the 15 GiB Intel Xeon cloud container the rest
  of this document's numbers come from — a 28q dense state (4 GiB) doesn't
  fit safely in memory actually free on that machine, so this session could
  not measure the 28q/30q rows itself. `scaling_probe` did complete cleanly
  through 26 qubits (20/22/24/26 qubits: 11.4 ms / 71.3 ms / 437.1 ms /
  1834.3 ms on that machine), confirming no crashes or correctness issues at
  those sizes, but those absolute numbers aren't comparable to the
  reference-machine table and were never merged into it.
- **Reference-machine re-run: since done, in a later session** (see "Where
  it actually fails: 18–30 qubits" above) — this is no longer an open item.
  That run confirmed the 28q flip directly: 58.6 s → 11.85 s steady-state
  (4.9×), turning an Aer loss into a SimQ win, with SimQ now leading at
  every measured size from 20–28 qubits. A further session closed the last
  remaining gap here too: Aer's own scaling probe (`aer_scaling_probe.py`)
  didn't exist yet at the time of the first reference-machine re-run, so
  that run's Aer column was itself carried over from an even earlier,
  different-instance measurement (2.80 GHz vs. 2.10 GHz) — this is now
  fixed, and the 28q verdict on today's fully same-session numbers is a
  more modest 1.14× SimQ win, not 1.55×.

## Honest limitations

- Results above are from a 4-vCPU cloud container; ratios will differ on
  wider machines (more cores help Aer's and qulacs's threading too). Run
  `run.sh` on your own hardware — that is what it is for.
- **SimQ loses `qft_probe/16q`** against every backend measured — 1/1.7×
  vs Aer, 1/2.1× vs Cirq, 1/5.0× vs qsim (still 2.7× ahead of exact
  Statevector, which does no optimization of its own) — because this
  project's gate-fusion work is local and width-bounded, and QFT's
  controlled-phase gates are not. See "Closing a benchmark-methodology gap"
  above for the full result and why it's the direct structural consequence
  of that design choice, not noise.
- **SimQ also loses `random_circuit/16q` against qsim specifically**
  (1/1.5×), while still beating Statevector/Aer/Cirq there (10.6×/2.1×/4.3×).
  This corrects an earlier version of this document, which measured these
  workloads on a different machine (8 GiB Apple Silicon) and reported
  `random_circuit` winning at every size against all four backends —
  re-measured on this document's actual reference machine, that claim does
  not hold against qsim at 16q. See "Closing a benchmark-methodology gap"
  for the full analysis of why this isn't a locality story the way
  `qft_probe` is.
- GHZ sampling vs `Statevector` overstates SimQ's advantage at 16q because
  `Statevector.sample_counts` is known to be slow for wide registers; the Aer
  column is the meaningful one there.
- qulacs coverage does not yet extend to `qft_probe`/`random_circuit`/the
  multi-instance workloads — only Qiskit and qsim mirror those (see "Closing
  a benchmark-methodology gap"). Given qulacs is the strongest competitor
  found in the 4–16q base table, its performance on QFT's long-range gate
  structure specifically (where SimQ already loses to Aer, Cirq, and qsim
  at 16q) is an open, likely-unflattering-for-SimQ question this document
  does not yet answer.
- The 18–30q scaling table's SimQ, Aer, and qulacs numbers are each medians
  of 3 fresh-process runs. The very first 28q run measured after merging
  `scaling_30q` was ~25.8 s for SimQ — a ~2.3× cold-start outlier from
  first-touch page faults on a fresh 4 GiB allocation — while three
  subsequent runs were tightly clustered at 10.9–11.85 s across sessions;
  qulacs showed the same effect more mildly (~1.9× high on the first run,
  ~19.0–20.7 s steady). Aer showed no comparable cold-start effect. The
  table reports steady-state medians, not the cold numbers. Real workloads
  that allocate a 28q state once and reuse it should expect the
  steady-state figure; a one-shot cold allocation (e.g. a script run once)
  should expect something closer to the cold number.
- The qsim/Cirq scaling numbers only cover 4–16 qubits; no 18–30-qubit qsim
  probe exists yet (unlike Aer and qulacs, which now both have one — see
  "Where it actually fails"), so no claim is made about SimQ vs qsim beyond
  16 qubits. qsim's C++ core is expected to close the gap at larger qubit
  counts the same way Aer/qulacs do, this just has not been measured here.
- qulacs was not tested at 30 qubits (see "Where it actually fails") because
  it has no built-in memory-aware qubit cap the way SimQ and Aer do; running
  it there risked an OOM kill rather than a clean refusal, for a result
  (state doesn't fit in RAM) the other two simulators already establish.
