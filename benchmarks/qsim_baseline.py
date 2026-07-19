#!/usr/bin/env python3
"""qsim (Google) baseline for the SimQ end-to-end benchmark suite.

Implements EXACTLY the same workloads as ``simq::bench_workloads`` and
``benchmarks/qiskit_baseline.py`` (see that file's docstring for the circuit
definitions, and in particular the module-level comments on why QFT's
cross-check is phase-sensitive (``Re(amplitude)``, not a probability) and
why the random-circuit workload uses deterministic index-based formulas
instead of a seeded PRNG):

* vqe_energy/{4,8,12,16}q
* qaoa_maxcut/{4,8,12,16}q
* ghz_p0/{4,8,12,16}q (sampling benchmark draws 1024 shots)
* qft_probe/{4,8,12,16}q
* random_circuit/{4,8,12,16}q
* {vqe_energy,qaoa_maxcut}_multi/{MULTI_INSTANCE_SIZE}q_i{0..NUM_INSTANCES}

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
import math
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
    qs = cirq.LineQubit.range(n)
    ops = [cirq.H(qs[q]) for q in range(n)]
    for l in range(VQE_LAYERS):
        if instance is None:
            ops += [cirq.ry(vqe_theta(l, q, n))(qs[q]) for q in range(n)]
        else:
            ops += [cirq.ry(vqe_theta_instance(l, q, n, instance))(qs[q]) for q in range(n)]
        ops += [cirq.CNOT(qs[q], qs[q + 1]) for q in range(n - 1)]
        if instance is None:
            ops += [cirq.rz(vqe_phi(l, q, n))(qs[q]) for q in range(n)]
        else:
            ops += [cirq.rz(vqe_phi_instance(l, q, n, instance))(qs[q]) for q in range(n)]
    return cirq.Circuit(ops), qs


def vqe_observable(n, qs):
    terms = [cirq.Z(qs[q]) * cirq.Z(qs[q + 1]) for q in range(n - 1)]
    terms += [0.5 * cirq.X(qs[q]) for q in range(n)]
    return sum(terms[1:], terms[0])


def qaoa_params_instance(layer, instance):
    return (
        QAOA_GAMMA[layer] + 0.17 * instance,
        QAOA_BETA[layer] + 0.11 * instance,
    )


def qaoa_circuit(n, instance=None):
    qs = cirq.LineQubit.range(n)
    ops = [cirq.H(qs[q]) for q in range(n)]
    for l in range(len(QAOA_GAMMA)):
        gamma, beta = (
            (QAOA_GAMMA[l], QAOA_BETA[l])
            if instance is None
            else qaoa_params_instance(l, instance)
        )
        for q in range(n):
            r = (q + 1) % n
            ops.append(cirq.CNOT(qs[q], qs[r]))
            ops.append(cirq.rz(2.0 * gamma)(qs[r]))
            ops.append(cirq.CNOT(qs[q], qs[r]))
        ops += [cirq.rx(2.0 * beta)(qs[q]) for q in range(n)]
    return cirq.Circuit(ops), qs


def qaoa_zz_observable(n, qs):
    terms = [cirq.Z(qs[q]) * cirq.Z(qs[(q + 1) % n]) for q in range(n)]
    return sum(terms[1:], terms[0])


def ghz_circuit(n):
    qs = cirq.LineQubit.range(n)
    ops = [cirq.H(qs[0])]
    ops += [cirq.CNOT(qs[q], qs[q + 1]) for q in range(n - 1)]
    return cirq.Circuit(ops), qs


def qft_circuit(n, k):
    """Textbook QFT -- see simq::bench_workloads::qft_circuit and
    qiskit_baseline.qft_circuit's comments on the target-processing order
    (high-to-low) this depends on for correctness."""
    qs = cirq.LineQubit.range(n)
    ops = []
    for q in range(n):
        if (k >> q) & 1:
            ops.append(cirq.X(qs[q]))
    for i in reversed(range(n)):
        ops.append(cirq.H(qs[i]))
        for j in reversed(range(i)):
            theta = math.pi / (1 << (i - j))
            # CZPowGate(exponent=t) applies phase exp(i*pi*t) to |11>, so
            # t = theta/pi gives exactly exp(i*theta) -- the same
            # diag(1,1,1,e^{i*theta}) convention as simq's CPhase/Qiskit's cp.
            ops.append(cirq.CZPowGate(exponent=theta / math.pi)(qs[j], qs[i]))
    for i in range(n // 2):
        ops.append(cirq.SWAP(qs[i], qs[n - 1 - i]))
    return cirq.Circuit(ops), qs


def rcs_gate_index(layer, qubit):
    return (layer * 7 + qubit * 13 + 5) % 5


def rcs_angle(layer, qubit, n):
    return 0.29 + 0.53 * (layer * n + qubit)


def random_circuit(n):
    """See simq::bench_workloads's module docs on random circuit sampling."""
    qs = cirq.LineQubit.range(n)
    ops = []
    for layer in range(RCS_LAYERS):
        for q in range(n):
            idx = rcs_gate_index(layer, q)
            if idx == 0:
                ops.append(cirq.H(qs[q]))
            elif idx == 1:
                ops.append(cirq.S(qs[q]))
            elif idx == 2:
                ops.append(cirq.T(qs[q]))
            elif idx == 3:
                # X**0.5 is the SX gate up to global phase; harmless here
                # since this workload's cross-check is a measurement
                # probability, not a raw amplitude (contrast qft_circuit).
                ops.append((cirq.X**0.5)(qs[q]))
            else:
                ops.append(cirq.ry(rcs_angle(layer, q, n))(qs[q]))
        offset = layer % 2
        q = offset
        while q + 1 < n:
            ops.append(cirq.CZ(qs[q], qs[q + 1]))
            q += 2
    return cirq.Circuit(ops), qs


def lsb_order(qs):
    """Cirq's default `qubit_order` is big-endian (the first-declared qubit
    is the state vector's MOST significant bit) -- the opposite of
    simq/Qiskit's little-endian convention (qubit 0 = LSB) this suite's
    circuits are all defined against. Passing `qubit_order=lsb_order(qs)`
    to `simulate`/`simulate_expectation_values` makes Cirq's output vector
    use the same bit-to-index mapping as the Rust and Qiskit mirrors.

    For `vqe_energy`/`qaoa_maxcut` (Pauli-sum expectation values, defined
    consistently against the same `qs` used to build the circuit) and
    `ghz_p0`/`random_circuit`'s p0 (both read state-vector index 0, which
    is "all qubits |0>" regardless of bit-order convention), this endianness
    difference happens not to change the result -- but it is NOT harmless
    in general, and did silently break `qft_probe`'s Re(amplitude-at-|1>)
    check during development (index 1 means "qubit 0 is |1>" only under a
    little-endian convention; under Cirq's default, index 1 means "the last
    declared qubit is |1>", a physically different amplitude). Applied
    everywhere below for defensive consistency, not just where currently
    required.
    """
    return list(reversed(qs))


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
        qft_qc, qft_qs = qft_circuit(n, QFT_INPUT_K)
        rcs_qc, rcs_qs = random_circuit(n)
        reps = reps_for(n)

        vqe_order, qaoa_order = lsb_order(vqe_qs), lsb_order(qaoa_qs)
        ghz_order, qft_order, rcs_order = lsb_order(ghz_qs), lsb_order(qft_qs), lsb_order(rcs_qs)

        # ---- Reference values (exact statevector via qsim) ------------------
        e = qsim_sim.simulate_expectation_values(vqe_qc, [vqe_op], qubit_order=vqe_order)[0]
        values[f"vqe_energy/{n}q"] = float(e.real)
        zz = qsim_sim.simulate_expectation_values(qaoa_qc, [qaoa_op], qubit_order=qaoa_order)[0]
        values[f"qaoa_maxcut/{n}q"] = float(0.5 * n - 0.5 * zz.real)
        sv = qsim_sim.simulate(ghz_qc, qubit_order=ghz_order)
        values[f"ghz_p0/{n}q"] = float(abs(sv.final_state_vector[0]) ** 2)
        sv = qsim_sim.simulate(qft_qc, qubit_order=qft_order)
        values[f"qft_probe/{n}q"] = float(sv.final_state_vector[1].real)
        sv = qsim_sim.simulate(rcs_qc, qubit_order=rcs_order)
        values[f"random_circuit/{n}q"] = float(abs(sv.final_state_vector[0]) ** 2)

        # ---- cirq.Simulator timings (pure Python reference) -----------------
        cirq_times[f"vqe_energy/{n}q"] = median_ms(
            lambda: cirq_sim.simulate_expectation_values(
                vqe_qc, [vqe_op], qubit_order=vqe_order
            )[0].real,
            reps,
        )
        cirq_times[f"qaoa_maxcut/{n}q"] = median_ms(
            lambda: cirq_sim.simulate_expectation_values(
                qaoa_qc, [qaoa_op], qubit_order=qaoa_order
            )[0].real,
            reps,
        )
        cirq_times[f"ghz_sampling/{n}q"] = median_ms(
            lambda: cirq_sim.run(ghz_measured, repetitions=GHZ_SHOTS), reps
        )
        cirq_times[f"qft_probe/{n}q"] = median_ms(
            lambda: cirq_sim.simulate(qft_qc, qubit_order=qft_order).final_state_vector[1].real,
            reps,
        )
        cirq_times[f"random_circuit/{n}q"] = median_ms(
            lambda: abs(
                cirq_sim.simulate(rcs_qc, qubit_order=rcs_order).final_state_vector[0]
            )
            ** 2,
            reps,
        )

        # ---- qsim timings (optimized C++ backend) ----------------------------
        qsim_times[f"vqe_energy/{n}q"] = median_ms(
            lambda: qsim_sim.simulate_expectation_values(
                vqe_qc, [vqe_op], qubit_order=vqe_order
            )[0].real,
            reps,
        )
        qsim_times[f"qaoa_maxcut/{n}q"] = median_ms(
            lambda: qsim_sim.simulate_expectation_values(
                qaoa_qc, [qaoa_op], qubit_order=qaoa_order
            )[0].real,
            reps,
        )
        qsim_times[f"ghz_sampling/{n}q"] = median_ms(
            lambda: qsim_sim.run(ghz_measured, repetitions=GHZ_SHOTS), reps
        )
        qsim_times[f"qft_probe/{n}q"] = median_ms(
            lambda: qsim_sim.simulate(qft_qc, qubit_order=qft_order).final_state_vector[1].real,
            reps,
        )
        qsim_times[f"random_circuit/{n}q"] = median_ms(
            lambda: abs(
                qsim_sim.simulate(rcs_qc, qubit_order=rcs_order).final_state_vector[0]
            )
            ** 2,
            reps,
        )

        print(f"  {n}q done")

    # ---- Multi-instance VQE/QAOA (see qiskit_baseline.py's module docstring) -
    n = MULTI_INSTANCE_SIZE
    reps = reps_for(n)

    vqe_instances = [vqe_circuit(n, instance=i) for i in range(NUM_INSTANCES)]
    vqe_ops = [vqe_observable(n, qs) for _, qs in vqe_instances]
    qaoa_instances = [qaoa_circuit(n, instance=i) for i in range(NUM_INSTANCES)]
    qaoa_ops = [qaoa_zz_observable(n, qs) for _, qs in qaoa_instances]

    for i, ((qc, qs), op) in enumerate(zip(vqe_instances, vqe_ops)):
        e = qsim_sim.simulate_expectation_values(qc, [op], qubit_order=lsb_order(qs))[0]
        values[f"vqe_energy_multi/{n}q_i{i}"] = float(e.real)
    for i, ((qc, qs), op) in enumerate(zip(qaoa_instances, qaoa_ops)):
        zz = qsim_sim.simulate_expectation_values(qc, [op], qubit_order=lsb_order(qs))[0]
        values[f"qaoa_maxcut_multi/{n}q_i{i}"] = float(0.5 * n - 0.5 * zz.real)

    def run_all_vqe_instances(sim):
        return [
            sim.simulate_expectation_values(qc, [op], qubit_order=lsb_order(qs))[0].real
            for (qc, qs), op in zip(vqe_instances, vqe_ops)
        ]

    def run_all_qaoa_instances(sim):
        return [
            sim.simulate_expectation_values(qc, [op], qubit_order=lsb_order(qs))[0].real
            for (qc, qs), op in zip(qaoa_instances, qaoa_ops)
        ]

    cirq_times[f"vqe_energy_multi_instance/{n}q"] = median_ms(
        lambda: run_all_vqe_instances(cirq_sim), reps
    )
    cirq_times[f"qaoa_cost_multi_instance/{n}q"] = median_ms(
        lambda: run_all_qaoa_instances(cirq_sim), reps
    )
    qsim_times[f"vqe_energy_multi_instance/{n}q"] = median_ms(
        lambda: run_all_vqe_instances(qsim_sim), reps
    )
    qsim_times[f"qaoa_cost_multi_instance/{n}q"] = median_ms(
        lambda: run_all_qaoa_instances(qsim_sim), reps
    )

    # ---- Multi-instance GHZ sampling (timing only, no cross-checked value) --
    ghz_qc, ghz_qs = ghz_circuit(n)
    ghz_measured = ghz_qc + cirq.measure(*ghz_qs, key="m")

    def run_all_ghz_seeds(sim):
        return [sim.run(ghz_measured, repetitions=GHZ_SHOTS) for _ in range(NUM_INSTANCES)]

    cirq_times[f"ghz_sampling_multi_instance/{n}q"] = median_ms(
        lambda: run_all_ghz_seeds(cirq_sim), reps
    )
    qsim_times[f"ghz_sampling_multi_instance/{n}q"] = median_ms(
        lambda: run_all_ghz_seeds(qsim_sim), reps
    )

    print(f"  multi-instance ({n}q, {NUM_INSTANCES} instances) done")

    out = {
        "versions": {
            "cirq": cirq.__version__,
            "qsimcirq": qsimcirq.__version__,
        },
        "ghz_shots": GHZ_SHOTS,
        "num_instances": NUM_INSTANCES,
        "multi_instance_size": MULTI_INSTANCE_SIZE,
        "values": values,
        "timings_ms": {"cirq": cirq_times, "qsim": qsim_times},
    }

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qsim_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
