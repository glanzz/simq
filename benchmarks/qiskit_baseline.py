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


def vqe_theta(layer, qubit, n):
    return 0.1 + 0.37 * (layer * n + qubit)


def vqe_phi(layer, qubit, n):
    return 0.05 + 0.21 * (layer * n + qubit)


def vqe_circuit(n):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for l in range(VQE_LAYERS):
        for q in range(n):
            qc.ry(vqe_theta(l, q, n), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
        for q in range(n):
            qc.rz(vqe_phi(l, q, n), q)
    return qc


def vqe_observable(n):
    terms = [("ZZ", [q, q + 1], 1.0) for q in range(n - 1)]
    terms += [("X", [q], 0.5) for q in range(n)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def qaoa_circuit(n):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for l in range(len(QAOA_GAMMA)):
        for q in range(n):
            r = (q + 1) % n
            qc.cx(q, r)
            qc.rz(2.0 * QAOA_GAMMA[l], r)
            qc.cx(q, r)
        for q in range(n):
            qc.rx(2.0 * QAOA_BETA[l], q)
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
        reps = reps_for(n)

        # ---- Reference values (exact statevector) --------------------------
        sv = Statevector(vqe_qc)
        values[f"vqe_energy/{n}q"] = float(sv.expectation_value(vqe_op).real)
        sv = Statevector(qaoa_qc)
        zz = float(sv.expectation_value(qaoa_op).real)
        values[f"qaoa_maxcut/{n}q"] = 0.5 * n - 0.5 * zz
        sv = Statevector(ghz_qc)
        values[f"ghz_p0/{n}q"] = float(abs(sv.data[0]) ** 2)

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

        print(f"  {n}q done")

    out = {
        "versions": {
            "qiskit": qiskit.__version__,
            "qiskit_aer": qiskit_aer.__version__,
        },
        "ghz_shots": GHZ_SHOTS,
        "values": values,
        "timings_ms": {"statevector": sv_times, "aer": aer_times},
    }

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qiskit_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
