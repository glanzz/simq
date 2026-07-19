#!/usr/bin/env python3
"""qulacs baseline for the SimQ end-to-end benchmark suite.

Implements EXACTLY the same workloads as ``simq::bench_workloads`` and
``benchmarks/qiskit_baseline.py`` (see that file's docstring for the circuit
definitions):

* vqe_energy/{4,8,12,16}q
* qaoa_maxcut/{4,8,12,16}q
* ghz_p0/{4,8,12,16}q (sampling benchmark draws 1024 shots)

qulacs has one execution path (its own vectorized C++/SIMD core; there is no
separate "reference" simulator the way Qiskit has Statevector vs. Aer, or
Cirq has cirq.Simulator vs. qsim) so only one timing column is produced.

**Sign-convention gotcha (verified empirically, not from memory):** qulacs's
``add_RX_gate``/``add_RY_gate``/``add_RZ_gate`` implement
``exp(+i*theta/2*P)``, the opposite sign from Qiskit/Cirq's
``exp(-i*theta/2*P)`` convention that ``simq``, ``qiskit_baseline.py`` and
``qsim_baseline.py`` all use. Confirmed by comparing single-qubit output
states directly (e.g. qulacs's ``RY(theta)`` on |0> gives
``[cos(th/2), -sin(th/2)]``, not ``[cos(th/2), +sin(th/2)]``). Every angle
passed to qulacs below is therefore negated so the circuit implements the
same unitary as the other three suites; this is exactly the kind of
convention mismatch the cross-validation step exists to catch.

Writes ``benchmarks/qulacs_results.json``.
"""

import json
import os
import statistics
import time

import qulacs
from qulacs import Observable, QuantumCircuit, QuantumState

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
        qc.add_H_gate(q)
    for l in range(VQE_LAYERS):
        for q in range(n):
            qc.add_RY_gate(q, -vqe_theta(l, q, n))
        for q in range(n - 1):
            qc.add_CNOT_gate(q, q + 1)
        for q in range(n):
            qc.add_RZ_gate(q, -vqe_phi(l, q, n))
    return qc


def vqe_observable(n):
    obs = Observable(n)
    for q in range(n - 1):
        obs.add_operator(1.0, f"Z {q} Z {q + 1}")
    for q in range(n):
        obs.add_operator(0.5, f"X {q}")
    return obs


def qaoa_circuit(n):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.add_H_gate(q)
    for l in range(len(QAOA_GAMMA)):
        for q in range(n):
            r = (q + 1) % n
            qc.add_CNOT_gate(q, r)
            qc.add_RZ_gate(r, -2.0 * QAOA_GAMMA[l])
            qc.add_CNOT_gate(q, r)
        for q in range(n):
            qc.add_RX_gate(q, -2.0 * QAOA_BETA[l])
    return qc


def qaoa_zz_observable(n):
    obs = Observable(n)
    for q in range(n):
        obs.add_operator(1.0, f"Z {q} Z {(q + 1) % n}")
    return obs


def ghz_circuit(n):
    qc = QuantumCircuit(n)
    qc.add_H_gate(0)
    for q in range(n - 1):
        qc.add_CNOT_gate(q, q + 1)
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
    qulacs_times = {}

    for n in SIZES:
        vqe_qc = vqe_circuit(n)
        vqe_op = vqe_observable(n)
        qaoa_qc = qaoa_circuit(n)
        qaoa_op = qaoa_zz_observable(n)
        ghz_qc = ghz_circuit(n)
        reps = reps_for(n)

        # ---- Reference values (exact statevector) --------------------------
        st = QuantumState(n)
        vqe_qc.update_quantum_state(st)
        values[f"vqe_energy/{n}q"] = float(vqe_op.get_expectation_value(st).real)

        st = QuantumState(n)
        qaoa_qc.update_quantum_state(st)
        zz = float(qaoa_op.get_expectation_value(st).real)
        values[f"qaoa_maxcut/{n}q"] = 0.5 * n - 0.5 * zz

        st = QuantumState(n)
        ghz_qc.update_quantum_state(st)
        values[f"ghz_p0/{n}q"] = float(abs(st.get_vector()[0]) ** 2)

        # ---- qulacs timings --------------------------------------------------
        def run_vqe():
            st = QuantumState(n)
            vqe_qc.update_quantum_state(st)
            return vqe_op.get_expectation_value(st).real

        def run_qaoa():
            st = QuantumState(n)
            qaoa_qc.update_quantum_state(st)
            return qaoa_op.get_expectation_value(st).real

        def run_ghz():
            st = QuantumState(n)
            ghz_qc.update_quantum_state(st)
            return st.sampling(GHZ_SHOTS)

        qulacs_times[f"vqe_energy/{n}q"] = median_ms(run_vqe, reps)
        qulacs_times[f"qaoa_maxcut/{n}q"] = median_ms(run_qaoa, reps)
        qulacs_times[f"ghz_sampling/{n}q"] = median_ms(run_ghz, reps)

        print(f"  {n}q done")

    out = {
        "versions": {"qulacs": qulacs.__version__},
        "ghz_shots": GHZ_SHOTS,
        "values": values,
        "timings_ms": {"qulacs": qulacs_times},
    }

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qulacs_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
