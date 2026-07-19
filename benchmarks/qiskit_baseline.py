#!/usr/bin/env python3
"""Qiskit baseline for the SimQ end-to-end benchmark suite.

Implements EXACTLY the same workloads as ``simq::bench_workloads`` (used by
``cargo bench -p simq --bench end_to_end`` and ``examples/xcheck_bench.rs``):

* vqe_energy/{4,8,12,16}q  - hardware-efficient ansatz (H layer, then 3 layers
  of RY / CNOT-chain / RZ with deterministic per-qubit angles), observable
  H = sum_q Z_q Z_{q+1} + 0.5 * sum_q X_q
* qaoa_maxcut/{4,8,12,16}q - ring-graph MaxCut QAOA at p=2
  (CNOT-RZ(2*gamma)-CNOT cost blocks + RX(2*beta) mixers),
  cost C = 0.5 * n - 0.5 * <sum_edges Z Z>
* ghz_p0/{4,8,12,16}q      - GHZ preparation; sampling benchmark draws 1024
  shots, cross-check value is p(|0...0>)
* qft_probe/{4,8,12,16}q   - textbook QFT (H + controlled-phase ladder +
  swaps) applied to |k=1>; cross-check value is Re(amplitude at |1>).
  QFT's long-range (non-nearest-neighbor) two-qubit gates are the
  counterpoint to the three workloads above, which are all local -- see
  BENCHMARKS.md's methodology notes. Re(amplitude) is used, not a
  measurement probability, because QFT's output has uniform amplitude
  *magnitude* regardless of correctness -- a probability-based check
  couldn't distinguish a correct phase ladder from a broken one.
* random_circuit/{4,8,12,16}q - alternating single-qubit layers (gate type
  chosen deterministically per (layer, qubit), not via a seeded PRNG -- see
  ``rcs_gate_index``) and brickwork CZ entangling layers with *alternating*
  qubit pairing (unlike the three local workloads' fixed linear chain).
  Cross-check value is p(|0...0>).
* {vqe_energy,qaoa_maxcut}_multi/{MULTI_INSTANCE_SIZE}q_i{0..NUM_INSTANCES}
  - the same VQE/QAOA circuits above, run at NUM_INSTANCES different
  deterministic parameter offsets, at one representative qubit count. Not a
  statistical-robustness study (NUM_INSTANCES is small, see
  ``simq::bench_workloads::NUM_INSTANCES`` docs for why) -- it exists so
  this suite isn't only ever validated/timed against the one hand-picked
  circuit per workload, which QED-C's application-oriented benchmarking
  guidance flags as a real overfitting risk once a structure-aware compiler
  optimization (like this project's gate-fusion pass) is in the loop.

Timings are measured for two Qiskit execution paths:

* ``statevector``: qiskit.quantum_info.Statevector exact evolution
* ``aer``:         qiskit_aer.AerSimulator (statevector method)

Fairness notes (all favor Qiskit):
* Circuits are built ONCE outside the timed region; Aer circuits are also
  transpiled ONCE outside the timed region. The SimQ benchmark rebuilds and
  re-optimizes its circuit inside every timed iteration.
* Aer energy evaluations use ``save_expectation_value`` so a single execute
  returns the energy (no statevector round-trip through Python).

Writes ``benchmarks/qiskit_results.json``.
"""

import json
import math
import os
import statistics
import time

import qiskit
import qiskit_aer
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

SIZES = [4, 8, 12, 16]
VQE_LAYERS = 3
QAOA_GAMMA = [0.8, 0.4]
QAOA_BETA = [0.7, 0.35]
GHZ_SHOTS = 1024
NUM_INSTANCES = 5  # keep in sync with simq::bench_workloads::NUM_INSTANCES
MULTI_INSTANCE_SIZE = 8
RCS_LAYERS = 8  # keep in sync with simq::bench_workloads::RCS_LAYERS
QFT_INPUT_K = 1  # keep in sync with simq::bench_workloads::QFT_INPUT_K


def vqe_theta(layer, qubit, n):
    return 0.1 + 0.37 * (layer * n + qubit)


def vqe_phi(layer, qubit, n):
    return 0.05 + 0.21 * (layer * n + qubit)


def vqe_theta_instance(layer, qubit, n, instance):
    return vqe_theta(layer, qubit, n) + 0.91 * instance


def vqe_phi_instance(layer, qubit, n, instance):
    return vqe_phi(layer, qubit, n) + 0.63 * instance


def vqe_circuit(n, instance=None):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for l in range(VQE_LAYERS):
        for q in range(n):
            theta = (
                vqe_theta(l, q, n)
                if instance is None
                else vqe_theta_instance(l, q, n, instance)
            )
            qc.ry(theta, q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
        for q in range(n):
            phi = (
                vqe_phi(l, q, n)
                if instance is None
                else vqe_phi_instance(l, q, n, instance)
            )
            qc.rz(phi, q)
    return qc


def vqe_observable(n):
    terms = [("ZZ", [q, q + 1], 1.0) for q in range(n - 1)]
    terms += [("X", [q], 0.5) for q in range(n)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def qaoa_params_instance(layer, instance):
    return (
        QAOA_GAMMA[layer] + 0.17 * instance,
        QAOA_BETA[layer] + 0.11 * instance,
    )


def qaoa_circuit(n, instance=None):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for l in range(len(QAOA_GAMMA)):
        gamma, beta = (
            (QAOA_GAMMA[l], QAOA_BETA[l])
            if instance is None
            else qaoa_params_instance(l, instance)
        )
        for q in range(n):
            r = (q + 1) % n
            qc.cx(q, r)
            qc.rz(2.0 * gamma, r)
            qc.cx(q, r)
        for q in range(n):
            qc.rx(2.0 * beta, q)
    return qc


def qaoa_zz_observable(n):
    terms = [("ZZ", [q, (q + 1) % n], 1.0) for q in range(n)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def ghz_circuit(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    return qc


def qft_circuit(n, k):
    """Textbook QFT (H + controlled-phase ladder + swaps) applied to |k>.

    Hand-decomposed with explicit ``cp``/``swap`` gates (not Qiskit's
    ``QFT`` circuit-library class) so the gate sequence is guaranteed
    identical to ``simq::bench_workloads::qft_circuit`` -- using the
    library class would leave the exact decomposition/ordering to Qiskit,
    which this suite's 1e-12 cross-check can't tolerate any ambiguity in.
    """
    qc = QuantumCircuit(n)
    for q in range(n):
        if (k >> q) & 1:
            qc.x(q)
    # See simq::bench_workloads::qft_circuit's comment: targets must be
    # processed high-to-low, with each control `j < i` still classical
    # (unprocessed) at the time it's used, or the ladder silently applies no
    # net phase for basis-state inputs.
    for i in reversed(range(n)):
        qc.h(i)
        for j in reversed(range(i)):
            theta = math.pi / (1 << (i - j))
            qc.cp(theta, j, i)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc


def rcs_gate_index(layer, qubit):
    return (layer * 7 + qubit * 13 + 5) % 5


def rcs_angle(layer, qubit, n):
    return 0.29 + 0.53 * (layer * n + qubit)


def random_circuit(n):
    """Alternating single-qubit / brickwork-CZ layers -- see
    ``simq::bench_workloads``'s module docs on random circuit sampling for
    why the gate choices are deterministic formulas, not a seeded PRNG."""
    qc = QuantumCircuit(n)
    for layer in range(RCS_LAYERS):
        for q in range(n):
            idx = rcs_gate_index(layer, q)
            if idx == 0:
                qc.h(q)
            elif idx == 1:
                qc.s(q)
            elif idx == 2:
                qc.t(q)
            elif idx == 3:
                qc.sx(q)
            else:
                qc.ry(rcs_angle(layer, q, n), q)
        offset = layer % 2
        q = offset
        while q + 1 < n:
            qc.cz(q, q + 1)
            q += 2
    return qc


def median_ms(fn, reps):
    fn()  # warmup
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def reps_for(n):
    return {4: 50, 8: 50, 12: 15, 16: 5}[n]


def main():
    values = {}
    sv_times = {}
    aer_times = {}
    aer = AerSimulator(method="statevector")

    for n in SIZES:
        vqe_qc = vqe_circuit(n)
        vqe_op = vqe_observable(n)
        qaoa_qc = qaoa_circuit(n)
        qaoa_op = qaoa_zz_observable(n)
        ghz_qc = ghz_circuit(n)
        qft_qc = qft_circuit(n, QFT_INPUT_K)
        rcs_qc = random_circuit(n)
        reps = reps_for(n)

        # ---- Reference values (exact statevector) --------------------------
        sv = Statevector(vqe_qc)
        values[f"vqe_energy/{n}q"] = float(sv.expectation_value(vqe_op).real)
        sv = Statevector(qaoa_qc)
        zz = float(sv.expectation_value(qaoa_op).real)
        values[f"qaoa_maxcut/{n}q"] = 0.5 * n - 0.5 * zz
        sv = Statevector(ghz_qc)
        values[f"ghz_p0/{n}q"] = float(abs(sv.data[0]) ** 2)
        sv = Statevector(qft_qc)
        values[f"qft_probe/{n}q"] = float(sv.data[1].real)
        sv = Statevector(rcs_qc)
        values[f"random_circuit/{n}q"] = float(abs(sv.data[0]) ** 2)

        # ---- Statevector timings -------------------------------------------
        sv_times[f"vqe_energy/{n}q"] = median_ms(
            lambda: Statevector(vqe_qc).expectation_value(vqe_op).real, reps
        )
        sv_times[f"qaoa_maxcut/{n}q"] = median_ms(
            lambda: Statevector(qaoa_qc).expectation_value(qaoa_op).real, reps
        )
        sv_times[f"ghz_sampling/{n}q"] = median_ms(
            lambda: Statevector(ghz_qc).sample_counts(GHZ_SHOTS), reps
        )
        sv_times[f"qft_probe/{n}q"] = median_ms(
            lambda: Statevector(qft_qc).data[1].real, reps
        )
        sv_times[f"random_circuit/{n}q"] = median_ms(
            lambda: abs(Statevector(rcs_qc).data[0]) ** 2, reps
        )

        # ---- Aer timings (transpiled once, outside the timed region) --------
        vqe_aer = vqe_qc.copy()
        vqe_aer.save_expectation_value(vqe_op, list(range(n)))
        vqe_aer = transpile(vqe_aer, aer)
        aer_times[f"vqe_energy/{n}q"] = median_ms(
            lambda: aer.run(vqe_aer).result().data()["expectation_value"], reps
        )

        qaoa_aer = qaoa_qc.copy()
        qaoa_aer.save_expectation_value(qaoa_op, list(range(n)))
        qaoa_aer = transpile(qaoa_aer, aer)
        aer_times[f"qaoa_maxcut/{n}q"] = median_ms(
            lambda: aer.run(qaoa_aer).result().data()["expectation_value"], reps
        )

        ghz_aer = ghz_qc.copy()
        ghz_aer.measure_all()
        ghz_aer = transpile(ghz_aer, aer)
        aer_times[f"ghz_sampling/{n}q"] = median_ms(
            lambda: aer.run(ghz_aer, shots=GHZ_SHOTS).result().get_counts(), reps
        )

        # QFT/RCS have no save_expectation_value analogue (the probe value
        # is a raw amplitude/probability, not an operator expectation), so
        # Aer timing uses save_statevector -- still the full run+retrieve
        # SimQ itself pays (run_to_dense also materializes the whole dense
        # state before extracting one entry).
        qft_aer = qft_qc.copy()
        qft_aer.save_statevector()
        qft_aer = transpile(qft_aer, aer)
        aer_times[f"qft_probe/{n}q"] = median_ms(
            lambda: aer.run(qft_aer).result().data()["statevector"].data[1].real, reps
        )

        rcs_aer = rcs_qc.copy()
        rcs_aer.save_statevector()
        rcs_aer = transpile(rcs_aer, aer)
        aer_times[f"random_circuit/{n}q"] = median_ms(
            lambda: abs(aer.run(rcs_aer).result().data()["statevector"].data[0]) ** 2,
            reps,
        )

        print(f"  {n}q done")

    # ---- Multi-instance VQE/QAOA (see module docstring) ---------------------
    n = MULTI_INSTANCE_SIZE
    reps = reps_for(n)
    vqe_op = vqe_observable(n)
    qaoa_op = qaoa_zz_observable(n)

    vqe_instance_circuits = [vqe_circuit(n, instance=i) for i in range(NUM_INSTANCES)]
    qaoa_instance_circuits = [qaoa_circuit(n, instance=i) for i in range(NUM_INSTANCES)]

    for i, qc in enumerate(vqe_instance_circuits):
        values[f"vqe_energy_multi/{n}q_i{i}"] = float(
            Statevector(qc).expectation_value(vqe_op).real
        )
    for i, qc in enumerate(qaoa_instance_circuits):
        zz = float(Statevector(qc).expectation_value(qaoa_op).real)
        values[f"qaoa_maxcut_multi/{n}q_i{i}"] = 0.5 * n - 0.5 * zz

    def run_all_vqe_instances_sv():
        return [Statevector(qc).expectation_value(vqe_op).real for qc in vqe_instance_circuits]

    def run_all_qaoa_instances_sv():
        return [Statevector(qc).expectation_value(qaoa_op).real for qc in qaoa_instance_circuits]

    sv_times[f"vqe_energy_multi_instance/{n}q"] = median_ms(run_all_vqe_instances_sv, reps)
    sv_times[f"qaoa_cost_multi_instance/{n}q"] = median_ms(run_all_qaoa_instances_sv, reps)

    vqe_instance_aer = []
    for qc in vqe_instance_circuits:
        aqc = qc.copy()
        aqc.save_expectation_value(vqe_op, list(range(n)))
        vqe_instance_aer.append(transpile(aqc, aer))

    qaoa_instance_aer = []
    for qc in qaoa_instance_circuits:
        aqc = qc.copy()
        aqc.save_expectation_value(qaoa_op, list(range(n)))
        qaoa_instance_aer.append(transpile(aqc, aer))

    def run_all_vqe_instances_aer():
        return [aer.run(qc).result().data()["expectation_value"] for qc in vqe_instance_aer]

    def run_all_qaoa_instances_aer():
        return [aer.run(qc).result().data()["expectation_value"] for qc in qaoa_instance_aer]

    aer_times[f"vqe_energy_multi_instance/{n}q"] = median_ms(run_all_vqe_instances_aer, reps)
    aer_times[f"qaoa_cost_multi_instance/{n}q"] = median_ms(run_all_qaoa_instances_aer, reps)

    # ---- Multi-instance GHZ sampling (timing only, no cross-checked value --
    # GHZ's circuit has no continuous parameters to perturb; see
    # simq::bench_workloads::ghz_sample_instances docs) -----------------------
    ghz_qc = ghz_circuit(n)

    def run_all_ghz_seeds_sv():
        return [Statevector(ghz_qc).sample_counts(GHZ_SHOTS) for _ in range(NUM_INSTANCES)]

    sv_times[f"ghz_sampling_multi_instance/{n}q"] = median_ms(run_all_ghz_seeds_sv, reps)

    ghz_aer_multi = ghz_qc.copy()
    ghz_aer_multi.measure_all()
    ghz_aer_multi = transpile(ghz_aer_multi, aer)

    def run_all_ghz_seeds_aer():
        return [
            aer.run(ghz_aer_multi, shots=GHZ_SHOTS).result().get_counts()
            for _ in range(NUM_INSTANCES)
        ]

    aer_times[f"ghz_sampling_multi_instance/{n}q"] = median_ms(run_all_ghz_seeds_aer, reps)

    print(f"  multi-instance ({n}q, {NUM_INSTANCES} instances) done")

    out = {
        "versions": {
            "qiskit": qiskit.__version__,
            "qiskit_aer": qiskit_aer.__version__,
        },
        "ghz_shots": GHZ_SHOTS,
        "num_instances": NUM_INSTANCES,
        "multi_instance_size": MULTI_INSTANCE_SIZE,
        "values": values,
        "timings_ms": {"statevector": sv_times, "aer": aer_times},
    }

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qiskit_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
