# SimQ vs Qiskit vs qsim — cross-validated benchmarks

Every number in this document was produced on one machine, in one sitting, by
`./benchmarks/run.sh`, and the comparison table is only printed after an
automatic cross-validation proves the suites simulate **identical circuits**:
12 observable values (VQE energies, QAOA cut values, GHZ probabilities at
4/8/12/16 qubits) must agree to **1e-12** between SimQ and Qiskit's exact
`Statevector`, and to **5e-6** between SimQ and qsim (Google's simulator,
whose Python wheel is float32-only — see the qsim fairness notes below), or
the harness exits without showing timings. The qsim/Cirq leg is optional
(`pip install cirq qsimcirq`); everything else runs without it.

## Results: SimQ vs Qiskit

Machine: Intel Xeon @ 2.80 GHz, 4 vCPUs, 15 GiB RAM (cloud container), Linux.
Toolchain: rustc 1.94.1 (release profile), Python 3 with Qiskit 2.5.0 /
qiskit-aer 0.17.2. Cross-validation: worst deviation across the 12 checked
values was **1.52e-14** (tolerance 1e-12). Medians; `ratio` = Qiskit time ÷
SimQ time, so ratio > 1 means SimQ is faster.

| Workload | SimQ (ms) | Statevector (ms) | ratio | Aer (ms) | ratio |
|----------|----------:|-----------------:|------:|---------:|------:|
| vqe_energy/4q    | 0.019 |   1.731 |   92.3× |  1.540 |  82.1× |
| vqe_energy/8q    | 0.046 |   5.038 |  110.3× |  1.966 |  43.0× |
| vqe_energy/12q   | 0.334 |  10.582 |   31.7× |  4.671 |  14.0× |
| vqe_energy/16q   | 6.122 |  84.690 |   13.8× | 17.729 |   2.9× |
| qaoa_maxcut/4q   | 0.013 |   1.539 |  114.6× |  1.635 | 121.7× |
| qaoa_maxcut/8q   | 0.031 |   3.878 |  123.4× |  2.660 |  84.6× |
| qaoa_maxcut/12q  | 0.427 |   9.863 |   23.1× |  5.114 |  12.0× |
| qaoa_maxcut/16q  | 5.269 |  74.308 |   14.1× | 13.950 |   2.6× |
| ghz_sampling/4q  | 0.017 |   0.970 |   57.0× |  1.500 |  88.2× |
| ghz_sampling/8q  | 0.020 |   2.939 |  150.1× |  2.049 | 104.7× |
| ghz_sampling/12q | 0.051 |  21.144 |  414.8× |  3.362 |  65.9× |
| ghz_sampling/16q | 0.497 | 627.867 | 1263.7× |  4.784 |   9.6× |

**SimQ is faster on all 12 workloads against both Qiskit backends** —
2.6–121.7× vs Aer and 13.8–1263.7× vs exact Statevector. Before the issue #76
fixes (see below) SimQ *lost* these 16-qubit rows by 22–30×; the same suite
produced both sets of numbers, so the turnaround is measured, not asserted.

## Results: SimQ vs qsim (Google)

Same machine and circuits as above. Toolchain: Python 3 with `cirq` 1.7.0 /
`qsimcirq` 0.22.0. Cross-validation: worst deviation across the 12 checked
values was **2.94e-06** (tolerance 5e-6, loosened from Qiskit's 1e-12 because
qsim's Python wheel carries no fp64 option — see Fairness notes). Medians;
`ratio` = qsim/Cirq time ÷ SimQ time.

| Workload | SimQ (ms) | Cirq (ms) | ratio | qsim (ms) | ratio |
|----------|----------:|----------:|------:|----------:|------:|
| vqe_energy/4q    | 0.019 |  1.445 |  77.0× |  0.450 | 24.0× |
| vqe_energy/8q    | 0.046 |  2.811 |  61.5× |  0.947 | 20.7× |
| vqe_energy/12q   | 0.334 |  5.829 |  17.4× |  1.959 |  5.9× |
| vqe_energy/16q   | 6.122 | 16.592 |   2.7× | 13.031 |  2.1× |
| qaoa_maxcut/4q   | 0.013 |  1.395 | 103.9× |  0.462 | 34.4× |
| qaoa_maxcut/8q   | 0.031 |  2.827 |  89.9× |  0.953 | 30.3× |
| qaoa_maxcut/12q  | 0.427 |  5.337 |  12.5× |  1.688 |  3.9× |
| qaoa_maxcut/16q  | 5.269 | 19.346 |   3.7× |  9.176 |  1.7× |
| ghz_sampling/4q  | 0.017 |  1.560 |  91.7× |  0.703 | 41.3× |
| ghz_sampling/8q  | 0.020 |  2.765 | 141.2× |  1.103 | 56.3× |
| ghz_sampling/12q | 0.051 |  3.875 |  76.0× |  1.532 | 30.0× |
| ghz_sampling/16q | 0.497 |  7.560 |  15.2× |  3.898 |  7.8× |

**SimQ is faster on all 12 workloads against both qsim backends** — 1.7–56.3×
vs qsim's optimized AVX/SSE C++ core and 2.7–141.2× vs Cirq's pure-Python
reference simulator. qsim is a stronger competitor than Aer at every size
(its ratios are the smallest of the four backends across the board), and it
is the only backend that still beats Statevector/Cirq's own pure-Python
sibling at 16 qubits — but SimQ stays ahead of it everywhere tested.

## Workloads

All three workloads are defined twice — in Rust
(`simq/src/bench_workloads.rs`, consumed by both the criterion bench and the
cross-check binary) and in Python (`benchmarks/qiskit_baseline.py`) — with
deterministic, qubit-asymmetric parameters so any qubit-ordering or convention
mismatch fails the cross-check instead of hiding in a symmetric circuit.

| Workload | Circuit | Measured quantity |
|----------|---------|-------------------|
| `vqe_energy/Nq` | H layer + 3 × [RY(θ_lq) ⊗ CNOT-chain ⊗ RZ(φ_lq)], θ/φ deterministic per (layer, qubit) | ⟨H⟩ with H = Σ Z_q Z_{q+1} + 0.5 Σ X_q |
| `qaoa_maxcut/Nq` | Ring-graph MaxCut QAOA, p = 2, cost blocks CNOT·RZ(2γ)·CNOT, mixers RX(2β) | cut value 0.5·N − 0.5·⟨Σ Z_q Z_{q+1}⟩ |
| `ghz_sampling/Nq` | H(0) + CNOT chain | 1024 measurement shots |

One benchmark iteration = one full cost-function evaluation, i.e. what a
variational optimizer pays per step.

## Fairness notes

These all favor Qiskit/qsim; SimQ's wins are understated, not overstated:

- **SimQ rebuilds and re-optimizes its circuit inside every timed iteration**
  (its compiler runs at default `O2` in the loop). The Qiskit and Cirq
  circuits are built once outside the timed region, and the Aer circuits are
  additionally **transpiled once outside the timed region**.
- Aer energy evaluations use `save_expectation_value`, so a single `run()`
  returns the energy with no statevector round-trip through Python.
- Aer runs its default multithreading; SimQ's defaults keep gate kernels
  single-threaded below 2^18 amplitudes because that is faster at these sizes
  (fork/join overhead exceeds the memory-bound kernel work).
- Timings are medians: criterion's median estimate for SimQ, median of
  50/50/15/5 repetitions (4q/8q/12q/16q) after a warmup for Qiskit and qsim.
- **qsim's Python wheel (`qsimcirq`) ships a single-precision (float32) state
  vector core with no fp64 build option.** That is a genuine precision
  disadvantage baked into the distributed package, not something this harness
  imposes — it is why the qsim cross-check tolerance is 5e-6 instead of
  Qiskit's 1e-12, and the observable diffs against qsim (up to 2.94e-06) are
  consistent with float32 rounding compounding over each circuit's gate
  count, not a bug in either simulator.
- qsim's `qsimcirq.QSimSimulator` also runs its own default multithreading
  (AVX/SSE-vectorized C++ core), same as Aer.

## Reproduction

```bash
pip install qiskit qiskit-aer cirq qsimcirq   # qsim/cirq are optional
./benchmarks/run.sh
```

The pieces, if you want them separately:

```bash
cargo bench -p simq --bench end_to_end          # SimQ timings (criterion)
python3 benchmarks/qiskit_baseline.py           # Qiskit timings + reference values
python3 benchmarks/qsim_baseline.py             # qsim/Cirq timings + reference values (optional)
python3 benchmarks/compare.py                   # cross-check (1e-12 Qiskit, 5e-6 qsim), then table
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
`simq-sim/examples/scaling_probe.rs`) beyond the published table to find the
edges. Same machine (15 GiB RAM); SimQ at default settings including its
optimizer in the timed run, Aer timed on a once-transpiled circuit including
final statevector retrieval:

| Qubits | State | SimQ | Aer | Verdict |
|-------:|------:|-----:|----:|---------|
| 20 | 16 MiB | 44 ms | 53 ms | SimQ 1.2× faster |
| 22 | 64 MiB | 243 ms | 240 ms | tie |
| 24 | 256 MiB | 1.15 s | 1.03 s | Aer 1.1× faster |
| 26 | 1 GiB | 4.3 s | 4.3 s | tie |
| 28 | 4 GiB | 58.6 s | 17.2 s | **Aer 3.4× faster** |
| 30 | 16 GiB | rejected: "max supported is 29" | rejected: "maximum (29)" | physics: state > RAM |

So: **parity through 26 qubits, a real loss at 28, and a wall at 30** on this
box. What the probing found (and what got fixed along the way):

- **Sparse warm-up cliff (fixed).** Sparse→dense conversion used to trigger
  at 10% density — a relative threshold, so bigger registers hashed longer:
  at 24q the warm-up AHashMap grew to ~1.6M entries and consumed 55% of total
  runtime (1.3 s of 2.4 s). The conversion point is now the measured
  hash-vs-dense cost crossover (~1/1024 density), cutting the 26q warm-up
  from ~2.6 s to ~0.6 s.
- **28-qubit gap (open, the real ≥28q work item).** SimQ's dense kernels
  match Aer per pass (~0.6 ns/amplitude at 26q), but SimQ sweeps the state
  once per gate (111 passes × 8 GiB of traffic at 28q) while Aer's runtime
  gate fusion collapses circuits into multi-qubit blocks and sweeps far
  fewer times. Once the state outgrows the last-level cache, passes are the
  whole game. Deeper fusion in `simq-compiler` (to 2–3-qubit unitaries, like
  Aer/qsim fuse to ~5) is the tracked follow-up in issue #76.
- **30-qubit failure mode (fixed).** A 30q dense state is 16 GiB; with 15 GiB
  of RAM both simulators must refuse — but SimQ used to *abort the process*
  inside an infallible `Vec` allocation (and, once the conversion was made
  fallible, would limp along sparse until the hash map itself blew up at
  ~234M entries). `Simulator::run` now derives the qubit cap from
  `MemAvailable` when no memory limit is configured and rejects upfront with
  `TooManyQubits`, exactly like Aer's clean "circuit too wide" error. Dense
  conversion also allocates once (null-checked) instead of twice.

## Honest limitations

- Results above are from a 4-vCPU cloud container; ratios will differ on
  wider machines (more cores help Aer's OpenMP too). Run `run.sh` on your own
  hardware — that is what it is for.
- At ≥28 qubits Aer's deeper runtime gate fusion wins (see the scaling table
  above); issue #76 stays open to track it. The 28q Aer number *includes*
  copying the 4 GiB result statevector into Python, so Aer's compute-only
  advantage is larger than 3.4×.
- The 26–28q SimQ numbers are single cold runs (first-touch page faults
  included); warm-loop numbers would be a few percent better.
- GHZ sampling vs `Statevector` overstates SimQ's advantage at 16q because
  `Statevector.sample_counts` is known to be slow for wide registers; the Aer
  column is the meaningful one there.
- The qsim/Cirq numbers only cover 4–16 qubits (the same range as the Qiskit
  table); the scaling probe in the previous section was not repeated against
  qsim, so no claim is made about SimQ vs qsim beyond 16 qubits.
- qsim was not pushed to find its own crossover point the way Aer was (no
  18–30-qubit qsim probe exists yet); qsim's C++ core is expected to close
  the gap at larger qubit counts the same way Aer does, this just has not
  been measured here.
