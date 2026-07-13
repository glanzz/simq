#!/usr/bin/env python3
"""qsim (Google) baseline for the SimQ end-to-end benchmark suite.

Implements EXACTLY the same workloads as ``simq::bench_workloads`` and
``benchmarks/qiskit_baseline.py`` (see that file's docstring for the circuit
definitions):

* vqe_energy/{4,8,12,16}q
* qaoa_maxcut/{4,8,12,16}q
* ghz_p0/{4,8,12,16}q (sampling benchmark draws 1024 shots)

Timings are measured for two execution paths, mirroring the Qiskit side's
pure-Python-reference vs. optimized-C++-backend split:

* ``cirq``: cirq.Simulator - pure-Python state vector simulator (the
  Statevector-equivalent reference implementation)
* ``qsim``: qsimcirq.QSimSimulator - Google's AVX/SSE-vectorized C++
  state vector simulator (the Aer-equivalent optimized backend)

Fairness notes (all favor qsim/cirq, same spirit as the Qiskit baseline):
* Circuits are built ONCE outside the timed region. The SimQ benchmark
  rebuilds and re-optimizes its circuit inside every timed iteration.
* Expectation values use ``simulate_expectation_values`` - a single call
  that returns the energy with no state-vector round-trip through Python,
  the qsim/cirq analogue of Aer's ``save_expectation_value``.

Writes ``benchmarks/qsim_results.json``.
"""

import json
import os
import statistics
import time

import cirq
import qsimcirq

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
    qs = cirq.LineQubit.range(n)
    ops = [cirq.H(qs[q]) for q in range(n)]
    for l in range(VQE_LAYERS):
        ops += [cirq.ry(vqe_theta(l, q, n))(qs[q]) for q in range(n)]
        ops += [cirq.CNOT(qs[q], qs[q + 1]) for q in range(n - 1)]
        ops += [cirq.rz(vqe_phi(l, q, n))(qs[q]) for q in range(n)]
    return cirq.Circuit(ops), qs


def vqe_observable(n, qs):
    terms = [cirq.Z(qs[q]) * cirq.Z(qs[q + 1]) for q in range(n - 1)]
    terms += [0.5 * cirq.X(qs[q]) for q in range(n)]
    return sum(terms[1:], terms[0])


def qaoa_circuit(n):
    qs = cirq.LineQubit.range(n)
    ops = [cirq.H(qs[q]) for q in range(n)]
    for l in range(len(QAOA_GAMMA)):
        for q in range(n):
            r = (q + 1) % n
            ops.append(cirq.CNOT(qs[q], qs[r]))
            ops.append(cirq.rz(2.0 * QAOA_GAMMA[l])(qs[r]))
            ops.append(cirq.CNOT(qs[q], qs[r]))
        ops += [cirq.rx(2.0 * QAOA_BETA[l])(qs[q]) for q in range(n)]
    return cirq.Circuit(ops), qs


def qaoa_zz_observable(n, qs):
    terms = [cirq.Z(qs[q]) * cirq.Z(qs[(q + 1) % n]) for q in range(n)]
    return sum(terms[1:], terms[0])


def ghz_circuit(n):
    qs = cirq.LineQubit.range(n)
    ops = [cirq.H(qs[0])]
    ops += [cirq.CNOT(qs[q], qs[q + 1]) for q in range(n - 1)]
    return cirq.Circuit(ops), qs


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
    cirq_times = {}
    qsim_times = {}
    cirq_sim = cirq.Simulator()
    qsim_sim = qsimcirq.QSimSimulator()

    for n in SIZES:
        vqe_qc, vqe_qs = vqe_circuit(n)
        vqe_op = vqe_observable(n, vqe_qs)
        qaoa_qc, qaoa_qs = qaoa_circuit(n)
        qaoa_op = qaoa_zz_observable(n, qaoa_qs)
        ghz_qc, ghz_qs = ghz_circuit(n)
        ghz_measured = ghz_qc + cirq.measure(*ghz_qs, key="m")
        reps = reps_for(n)

        # ---- Reference values (exact statevector via qsim) ------------------
        e = qsim_sim.simulate_expectation_values(vqe_qc, [vqe_op])[0]
        values[f"vqe_energy/{n}q"] = float(e.real)
        zz = qsim_sim.simulate_expectation_values(qaoa_qc, [qaoa_op])[0]
        values[f"qaoa_maxcut/{n}q"] = float(0.5 * n - 0.5 * zz.real)
        sv = qsim_sim.simulate(ghz_qc)
        values[f"ghz_p0/{n}q"] = float(abs(sv.final_state_vector[0]) ** 2)

        # ---- cirq.Simulator timings (pure Python reference) -----------------
        cirq_times[f"vqe_energy/{n}q"] = median_ms(
            lambda: cirq_sim.simulate_expectation_values(vqe_qc, [vqe_op])[0].real, reps
        )
        cirq_times[f"qaoa_maxcut/{n}q"] = median_ms(
            lambda: cirq_sim.simulate_expectation_values(qaoa_qc, [qaoa_op])[0].real, reps
        )
        cirq_times[f"ghz_sampling/{n}q"] = median_ms(
            lambda: cirq_sim.run(ghz_measured, repetitions=GHZ_SHOTS), reps
        )

        # ---- qsim timings (optimized C++ backend) ----------------------------
        qsim_times[f"vqe_energy/{n}q"] = median_ms(
            lambda: qsim_sim.simulate_expectation_values(vqe_qc, [vqe_op])[0].real, reps
        )
        qsim_times[f"qaoa_maxcut/{n}q"] = median_ms(
            lambda: qsim_sim.simulate_expectation_values(qaoa_qc, [qaoa_op])[0].real, reps
        )
        qsim_times[f"ghz_sampling/{n}q"] = median_ms(
            lambda: qsim_sim.run(ghz_measured, repetitions=GHZ_SHOTS), reps
        )

        print(f"  {n}q done")

    out = {
        "versions": {
            "cirq": cirq.__version__,
            "qsimcirq": qsimcirq.__version__,
        },
        "ghz_shots": GHZ_SHOTS,
        "values": values,
        "timings_ms": {"cirq": cirq_times, "qsim": qsim_times},
    }

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qsim_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
