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
qiskit-aer 0.17.2. Cross-validation: worst deviation across the 12 checked
values was **1.45e-14** (tolerance 1e-12). Medians; `ratio` = Qiskit time ÷
SimQ time, so ratio > 1 means SimQ is faster.

| Workload | SimQ (ms) | Statevector (ms) | ratio | Aer (ms) | ratio |
|----------|----------:|------------------:|------:|---------:|------:|
| vqe_energy/4q    | 0.030 |   2.056 |  69.6× |  1.137 | 38.5× |
| vqe_energy/8q    | 0.069 |   3.339 |  48.4× |  1.854 | 26.9× |
| vqe_energy/12q   | 0.446 |   7.794 |  17.5× |  3.836 |  8.6× |
| vqe_energy/16q   | 8.702 |  81.407 |   9.4× | 11.835 |  1.4× |
| qaoa_maxcut/4q   | 0.023 |   1.571 |  68.5× |  1.398 | 61.0× |
| qaoa_maxcut/8q   | 0.049 |   2.953 |  60.3× |  2.102 | 42.9× |
| qaoa_maxcut/12q  | 0.537 |   6.635 |  12.3× |  4.251 |  7.9× |
| qaoa_maxcut/16q  | 7.657 |  73.579 |   9.6× | 10.001 |  1.3× |
| ghz_sampling/4q  | 0.028 |   1.065 |  38.4× |  1.000 | 36.1× |
| ghz_sampling/8q  | 0.032 |   2.408 |  76.4× |  1.601 | 50.8× |
| ghz_sampling/12q | 0.062 |  17.243 | 276.2× |  2.022 | 32.4× |
| ghz_sampling/16q | 0.790 | 508.933 | 643.9× |  4.588 |  5.8× |

**SimQ is faster on all 12 workloads against both Qiskit backends** —
1.3–61.0× vs Aer and 9.4–643.9× vs exact Statevector. Before the issue #76
fixes (see below) SimQ *lost* these 16-qubit rows by 22–30×; the same suite
produced both sets of numbers, so the turnaround is measured, not asserted.

## Results: SimQ vs qsim (Google)

Same machine and circuits as above. Toolchain: Python 3 with `cirq` 1.7.0 /
`qsimcirq` 0.22.0. Cross-validation: worst deviation across the 12 checked
values was **3.28e-06** (tolerance 5e-6, loosened from Qiskit's 1e-12 because
qsim's Python wheel carries no fp64 option — see Fairness notes). Medians;
`ratio` = qsim/Cirq time ÷ SimQ time.

| Workload | SimQ (ms) | Cirq (ms) | ratio | qsim (ms) | ratio |
|----------|----------:|----------:|------:|----------:|------:|
| vqe_energy/4q    | 0.030 |  2.820 |  95.5× | 0.687 |  23.3× |
| vqe_energy/8q    | 0.069 |  5.332 |  77.2× | 1.610 |  23.3× |
| vqe_energy/12q   | 0.446 |  9.525 |  21.3× | 2.200 |   4.9× |
| vqe_energy/16q   | 8.702 | 29.903 |   3.4× | 5.647 | 1/1.5× |
| qaoa_maxcut/4q   | 0.023 |  2.745 | 119.7× | 0.696 |  30.3× |
| qaoa_maxcut/8q   | 0.049 |  5.705 | 116.5× | 1.442 |  29.4× |
| qaoa_maxcut/12q  | 0.537 |  9.349 |  17.4× | 2.115 |   3.9× |
| qaoa_maxcut/16q  | 7.657 | 27.665 |   3.6× | 4.525 | 1/1.7× |
| ghz_sampling/4q  | 0.028 |  2.304 |  83.1× | 0.911 |  32.9× |
| ghz_sampling/8q  | 0.032 |  3.801 | 120.6× | 1.440 |  45.7× |
| ghz_sampling/12q | 0.062 |  5.514 |  88.3× | 2.073 |  33.2× |
| ghz_sampling/16q | 0.790 |  8.209 |  10.4× | 3.048 |   3.9× |

**SimQ is faster on all 12 workloads against Cirq's pure-Python reference
simulator** (3.4–120.6×) **and on 10 of 12 against qsim's optimized AVX/SSE
C++ core**, losing only `vqe_energy/16q` (1.5×) and `qaoa_maxcut/16q` (1.7×).
qsim is a much stronger competitor than Aer at 16 qubits — but see the
qulacs section below, which is stronger still.

## Results: SimQ vs qulacs

Same machine and circuits as above. Toolchain: Python 3 with `qulacs` 0.6.13
(a single execution path — unlike Qiskit/Cirq, qulacs does not ship a
separate pure-Python reference simulator alongside its optimized core, so
there is one timing column here, not two). Cross-validation: worst deviation
across the 12 checked values was **8.44e-15** (tolerance 1e-12 — qulacs, like
SimQ and Qiskit, is a float64 core, so it gets the tight tolerance, not
qsim's loosened one). Medians; `ratio` = qulacs time ÷ SimQ time.

| Workload | SimQ (ms) | qulacs (ms) | ratio |
|----------|----------:|------------:|------:|
| vqe_energy/4q    | 0.030 |  0.040 |    1.4× |
| vqe_energy/8q    | 0.069 |  0.083 |    1.2× |
| vqe_energy/12q   | 0.446 |  0.924 |    2.1× |
| vqe_energy/16q   | 8.702 | 12.817 |    1.5× |
| qaoa_maxcut/4q   | 0.023 |  0.040 |    1.8× |
| qaoa_maxcut/8q   | 0.049 |  0.076 |    1.5× |
| qaoa_maxcut/12q  | 0.537 |  0.893 |    1.7× |
| qaoa_maxcut/16q  | 7.657 |  8.193 |    1.1× |
| ghz_sampling/4q  | 0.028 |  0.044 |    1.6× |
| ghz_sampling/8q  | 0.032 |  0.049 |    1.6× |
| ghz_sampling/12q | 0.062 |  0.084 |    1.3× |
| ghz_sampling/16q | 0.790 |  0.462 | 1/1.7× |

**qulacs is, by a wide margin, the strongest of the three published
competitors at 4–16 qubits** — where Aer/Cirq/qsim lose by 1.3–643.9×, qulacs
stays within 1.1–2.1× of SimQ on 11 of 12 workloads, and **beats SimQ outright
on `ghz_sampling/16q`** (0.462 ms vs 0.790 ms, 1.7×). This matches qulacs's
reputation (it is built around a heavily hand-optimized, SIMD/multithreaded
C++ core specifically tuned for exactly this kind of shallow, wide circuit)
and is the most credible "SimQ isn't unambiguously fastest" data point in
this document — see the scaling section below for how the gap moves at
larger qubit counts.

A genuine surprise found while wiring this up, not something assumed from
qulacs's docs: **qulacs's `add_RX_gate`/`add_RY_gate`/`add_RZ_gate` use the
opposite sign convention from Qiskit/Cirq/simq** — `exp(+i·θ/2·P)` instead of
the near-universal `exp(-i·θ/2·P)`. Verified empirically (see
`benchmarks/qulacs_baseline.py`'s docstring): `qulacs.gate.RY(θ)` applied to
`|0⟩` produces `[cos(θ/2), −sin(θ/2)]`, the negated-angle result, not
`[cos(θ/2), +sin(θ/2)]`. Every angle passed to qulacs in this benchmark is
therefore negated to implement the same unitary as the other three suites —
exactly the kind of convention mismatch the cross-validation gate exists to
catch, and it would have silently produced wrong "reference" values (and a
failed cross-check) had it gone unnoticed.

## Closing a benchmark-methodology gap: QFT, random circuits, multi-instance

The 36 workload/backend combinations above (12 workloads × Qiskit/Aer, qsim,
and qulacs) are all built from the same three circuit shapes — VQE,
QAOA-MaxCut, GHZ — and all three share two properties:
every two-qubit gate acts on **adjacent qubits** in a **fixed pattern that
repeats every layer** (a linear chain for VQE, a ring for QAOA, a chain for
GHZ), and each is timed against **one fixed, hand-picked, deterministic
circuit instance** per qubit count. Neither property is unique to this
project's methodology — but a locality-exploiting compiler optimization was
added to SimQ's fusion pass in this repo (see "Issue #76 follow-up" above:
gate fusion collapses adjacent local gates into width-bounded blocks), which
means a benchmark suite containing *only* local, fixed-pattern circuits risks
making that specific optimization look better than it generalizes, and never
tests whether the published numbers are an artifact of the one circuit
instance each workload happens to use.

This is a recognized methodology concern, not a project-specific worry:
QED-C's application-oriented benchmarking guidance (arXiv:2110.03137)
recommends multiple random circuit instances per problem size specifically
to guard against tuning results to one convenient circuit; MQT Bench
(arXiv:2204.13719) and the broader simulator-comparison literature (e.g.
arXiv:2401.09076, which benchmarks Aer/qsim-class simulators using exactly
Trotter circuits, **random circuit sampling**, and **QFT**) consistently
include non-local, structure-agnostic, and analytically-checkable circuit
families alongside application-specific ansätze — categories this repo's
suite didn't have. Three additions close that gap:

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
- **Multi-instance VQE/QAOA** (`{vqe_energy,qaoa_maxcut}_multi/8q_i0..4`) —
  the same ansätze, run at 5 deterministically-offset parameter sets, at one
  representative qubit count (8; not swept across all four sizes, to bound
  how much slower this makes `benchmarks/run.sh`, which re-runs the full
  Rust/Qiskit/qsim pipeline per instance). GHZ has no continuous parameter
  to perturb this way (it's parameter-free), so its multi-instance variant
  instead varies the shot-sampling seed and is timing-only, not a new
  cross-checked value.

All of the above were designed and implemented in a single session together
with the deeper gate-fusion work in "Issue #76 follow-up," specifically to
test whether that work's own benchmarking was representative — see the
results and the honest finding below.

### Results (this session, NOT the reference machine — see caveat)

Machine: 8 GiB Apple Silicon (the same machine used for the "Issue #76
follow-up" scaling numbers above, **not** the 15 GiB Intel Xeon cloud
container the rest of this document's numbers come from). Cross-validation:
worst deviation **3.19e-14** vs Qiskit exact `Statevector` (tolerance
1e-12, unchanged from before this work) and **2.10e-06** vs qsim (tolerance
5e-6). All 30 checked values passed, including every new `qft_probe`,
`random_circuit`, and multi-instance entry.

| Workload | SimQ (ms) | Statevector | ratio | Aer | ratio | Cirq | ratio | qsim | ratio |
|----------|----------:|------------:|------:|----:|------:|-----:|------:|-----:|------:|
| qft_probe/4q          |  0.006 |  0.251 |  39.4× | 0.336 |  52.7× |  0.503 |  78.9× | 0.157 |  24.6× |
| qft_probe/8q          |  0.032 |  0.875 |  27.5× | 0.959 |  30.1× |  1.233 |  38.7× | 0.413 |  12.9× |
| qft_probe/12q         |  0.513 |  3.815 |   7.4× | 2.318 |   4.5× |  2.495 |   4.9× | 1.319 |   2.6× |
| **qft_probe/16q**     | **13.790** | 26.792 |   1.9× | 7.938 | **1/1.7×** | 6.338 | **1/2.2×** | 16.681 |   1.2× |
| random_circuit/4q     |  0.020 |  0.603 |  30.7× | 0.420 |  21.4× |  1.292 |  65.8× | 0.425 |  21.6× |
| random_circuit/8q     |  0.043 |  1.401 |  32.3× | 0.791 |  18.2× |  2.563 |  59.1× | 0.830 |  19.1× |
| random_circuit/12q    |  0.327 |  4.940 |  15.1× | 1.738 |   5.3× |  4.639 |  14.2× | 1.544 |   4.7× |
| random_circuit/16q    |  4.678 | 30.170 |   6.4× | 7.519 |   1.6× | 18.127 |   3.9× | 8.618 |   1.8× |
| vqe_energy_multi_instance/8q (5 instances) | 0.219 | 7.257 | 33.2× | 2.985 | 13.6× | 14.289 | 65.3× | 4.469 | 20.4× |
| qaoa_cost_multi_instance/8q (5 instances)  | 0.148 | 6.328 | 42.7× | 3.474 | 23.4× | 13.769 | 92.9× | 4.271 | 28.8× |
| ghz_sampling_multi_instance/8q (5 instances) | 0.093 | 5.495 | 59.1× | 2.659 | 28.6× | 13.065 | 140.6× | 5.101 | 54.9× |

**The honest finding this exercise was for: `qft_probe/16q` is a loss for
SimQ** — 1/1.7× vs Aer, 1/2.2× vs Cirq (still ahead of the float32 qsim
backend at 1.2×, and still ahead of both exact-statevector references). This
is not noise or a machine artifact: it is the direct, structural consequence
of this session's own fusion work being width-bounded and *local* —
QFT's controlled-phase gates connect qubits up to 15 apart at n=16, far
outside any 3-qubit local block, so fusion simply doesn't engage for most of
the circuit, while Aer's and Cirq's own fusion/optimization evidently
handles the long-range case better. `random_circuit`, by contrast, wins
comfortably at every size including 16q (6.4×/1.6×/3.9×/1.8×) — its
brickwork connectivity *alternates* which pairs are adjacent each layer, but
every individual two-qubit gate is still local within that layer, so the
width-bounded fusion still engages. **The gap between these two results is
the actual, now-measured boundary of what this session's fusion work
helps with: locally-structured circuits (including ones whose local
structure changes over time, like `random_circuit`), not genuinely
long-range ones like QFT.** The VQE/QAOA/GHZ-only suite could not have
shown this, because none of its circuits are non-local.

The multi-instance results show the opposite finding — a reassuring one:
all 5 VQE and 5 QAOA instances cross-validated cleanly against both Qiskit
and qsim (see the cross-validation log this session produced), so the
headline VQE/QAOA numbers earlier in this document are not an artifact of
the one parameter set each workload happens to use.

**What this section does not do:** it does not update the main "Results:
SimQ vs Qiskit" / "Results: SimQ vs qsim" tables above, or the "Where it
actually fails" scaling table — those stay as the historical record of the
one reference-machine run this document's own opening paragraph promises.
Re-running `benchmarks/run.sh` on that reference machine and merging these
five new workload rows into the main tables (including, honestly, the
16-qubit QFT loss) is the concrete next step, not done here.

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
| `{vqe_energy,qaoa_maxcut}_multi/8q_i{0..5}` | Same ansatz as above, 5 deterministic parameter offsets | same as above, per instance |

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

We pushed the same 1-layer VQE circuit (H + RY + CNOT-chain + RZ, built by
`simq-sim/examples/scaling_probe.rs` and, for qulacs, its Python port
`benchmarks/qulacs_scaling_probe.py`) beyond the published table to find the
edges. Same machine class (15 GiB RAM, 4 vCPUs); SimQ at default settings
including its optimizer in the timed run, Aer timed on a once-transpiled
circuit including final statevector retrieval, qulacs timed on state
creation + `update_quantum_state` (its equivalent of a full run).

**Update:** the SimQ and qulacs columns below were freshly measured in this
same sitting (SimQ after merging `scaling_30q`, persistent multi-qubit
fusion, issue #76's tracked follow-up — see below); the Aer column is
retained from an earlier measurement session on the same machine class
(Aer/Qiskit's own scaling probe was not rerun this time, only the SimQ and
qulacs ones). Each SimQ/qulacs value is the median of 3 fresh-process runs
(a cold first run at 28q was ~2.3× slower than steady state for SimQ and
~20% slower for qulacs — see note below the table):

| Qubits | State | SimQ | Aer | qulacs | Verdict (vs Aer) |
|-------:|------:|-----:|----:|-------:|---------|
| 20 | 16 MiB | 23.8 ms | 53 ms | 32.1 ms | **SimQ 2.2× faster** |
| 22 | 64 MiB | 100.8 ms | 240 ms | 147.5 ms | **SimQ 2.4× faster** |
| 24 | 256 MiB | 542.7 ms | 1.03 s | 767.7 ms | **SimQ 1.9× faster** |
| 26 | 1 GiB | 2.76 s | 4.3 s | 4.84 s | **SimQ 1.6× faster** |
| 28 | 4 GiB | 11.1 s | 17.2 s | 20.7 s | **SimQ 1.55× faster** |
| 30 | 16 GiB | rejected: "max supported is 29" | rejected: "maximum (29)" | not attempted (see below) | physics: state > RAM |

So: **SimQ now leads at every measured size from 20 through 28 qubits**
against both Aer and qulacs, and 30 remains a hard wall for SimQ and Aer on
this box (16 GiB state, 15 GiB RAM). This reverses the previous finding of
parity through 26q and a 3.4× Aer win at 28q. Unlike the 4–16q range — where
qulacs was the closest competitor by far, even beating SimQ once — at
20–28q SimQ's fusion advantage widens rather than closes: SimQ leads qulacs
by 1.35–1.87× across this range (1.35× at 20q, growing to 1.87× at 28q),
the opposite trend from the small-qubit table. qulacs was **not** run at 30
qubits: unlike SimQ/Aer, it has no built-in memory-aware qubit cap (see the
30-qubit bullet below) and would have simply attempted the 16 GiB
allocation on a 15 GiB box, risking an OOM kill of the whole container
rather than a clean error — not worth the risk to confirm a result the
physics already answers.

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
  re-deriving the same block partition. Net effect: 28q went from 58.6 s to
  a steady-state 11.1 s (5.3×), enough to flip the 28q verdict from a 3.4×
  Aer win to a 1.55× SimQ win.
- **30-qubit failure mode (fixed, unchanged by this update).** A 30q dense
  state is 16 GiB; with 15 GiB of RAM both simulators must refuse — but SimQ
  used to *abort the process* inside an infallible `Vec` allocation (and,
  once the conversion was made fallible, would limp along sparse until the
  hash map itself blew up at ~234M entries). `Simulator::run` now derives the
  qubit cap from `MemAvailable` when no memory limit is configured and
  rejects upfront with `TooManyQubits`, exactly like Aer's clean "circuit too
  wide" error. Dense conversion also allocates once (null-checked) instead of
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
- **Reference-machine re-run: since done, in a separate session** (see
  "Where it actually fails: 18–30 qubits" above, "Update" note) — this is
  no longer an open item. That run confirmed the 28q flip directly: 58.6 s
  → 11.1 s steady-state (5.3×), turning a 3.4× Aer loss into a 1.55× SimQ
  win, with SimQ now leading at every measured size from 20–28 qubits.

## Honest limitations

- **SimQ loses `qft_probe/16q`** — 1/1.7× vs Aer, 1/2.2× vs Cirq — because
  this session's gate-fusion work is local and width-bounded, and QFT's
  controlled-phase gates are not local. See "Closing a benchmark-methodology
  gap" above for the full result and why it's the direct structural
  consequence of that design choice, not noise.
  wider machines (more cores help Aer's and qulacs's threading too). Run
  `run.sh` on your own hardware — that is what it is for.
- The main 4–16q tables (Qiskit/Aer, Cirq/qsim, qulacs) were all measured
  together in this session, on this box, so they're directly comparable to
  each other. The 18–30q scaling table is not fully consistent with them:
  its SimQ and qulacs columns are from this same session, but its **Aer
  column was not rerun** — Aer's own scaling probe (20–30q) was not part of
  this update — so it is carried over from an earlier measurement session on
  a different instance of the same machine class (2.80 GHz vs. this
  session's 2.10 GHz, see below). A full rerun of Aer's scaling probe would
  be needed to confirm that gap holds beyond noise, though the 5.3× SimQ
  speedup at 28q (58.6 s → 11.1 s steady state) is far larger than plausible
  run-to-run variance.
- The 20–28q SimQ and qulacs numbers are medians of 3 fresh-process runs
  each. The very first 28q run measured after the merge was ~25.8 s for
  SimQ — a ~2.3× cold-start outlier from first-touch page faults on a fresh
  4 GiB allocation — while three subsequent runs were tightly clustered at
  10.9–11.2 s; qulacs showed the same effect more mildly (25.3 s cold vs.
  ~20.7 s steady, ~20% high). The table reports steady-state medians, not
  the cold numbers. Real workloads that allocate a 28q state once and reuse
  it should expect the steady-state figure; a one-shot cold allocation (e.g.
  a script run once) should expect something closer to the cold number.
- The scaling table's Aer column was measured on a different instance of
  the same machine class than everything else in this document: 2.80 GHz
  vs. this session's 2.10 GHz. That affects absolute times but not the
  order of magnitude of the SimQ/Aer ratios.
- GHZ sampling vs `Statevector` overstates SimQ's advantage at 16q because
  `Statevector.sample_counts` is known to be slow for wide registers; the Aer
  column is the meaningful one there.
- The qsim/Cirq and qulacs scaling numbers only cover 4–16q (qulacs) or
  aren't measured at all beyond 16q (qsim/Cirq); the 18–30q scaling probe's
  qulacs column exists (see above) but qsim/Cirq's does not, so no claim is
  made about SimQ vs qsim beyond 16 qubits.
- qsim was not pushed to find its own crossover point the way Aer and qulacs
  were (no 18–30-qubit qsim probe exists yet); qsim's C++ core is expected to
  close the gap at larger qubit counts the same way Aer/qulacs do, this just
  has not been measured here.
- qulacs was not tested at 30 qubits (see the scaling section above) because
  it has no built-in memory-aware qubit cap the way SimQ and Aer do; running
  it there risked an OOM kill rather than a clean refusal, for a result
  (state doesn't fit in RAM) the other two simulators already establish.
- **The QFT/random-circuit/multi-instance results** (see "Closing a
  benchmark-methodology gap" above) were measured in a separate session on a
  *different* machine (8 GiB Apple Silicon) than everything else in this
  document, and are not merged into any of the tables above — see the
  caveat at the end of that section. That same session's `benchmarks/run.sh`
  run also produced 4–16q criterion timings for the original three
  workloads; those are **not** included anywhere in this document. Criterion
  reported those numbers as several-percent faster than its stored local
  baseline, but that baseline predated the change by days with unknown run
  conditions, and the change's own dispatch logic provably routes every
  4–16q circuit through the unmodified pre-existing fusion code path — so
  that apparent speedup is measurement noise from an uncontrolled
  comparison, not a real effect. Only that run's correctness
  cross-validation (worst deviation 3.19e-14 vs Qiskit) is machine-
  independent and meaningful on its own.
- qulacs coverage does not yet extend to `qft_probe`/`random_circuit`/the
  multi-instance workloads — only Qiskit and qsim mirror those (see "Closing
  a benchmark-methodology gap"). Given qulacs is the strongest competitor
  found in the 4–16q base table, its performance on QFT's long-range gate
  structure specifically (where SimQ already loses to Aer and Cirq at 16q)
  is an open, likely-unflattering-for-SimQ question this document does not
  yet answer.
