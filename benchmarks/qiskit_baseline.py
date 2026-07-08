#!/usr/bin/env python3
"""Qiskit baseline for the SimQ end-to-end benchmarks.

Every workload here mirrors, gate for gate, a criterion benchmark in
`simq/benches/end_to_end.rs`; the shared definition lives in
`BENCHMARKS.md`. If you change a circuit here, change it there too.

One iteration always includes circuit *construction* as well as
simulation — that matches how a variational optimizer drives a
simulator, which is the workload being compared.

Two Qiskit execution paths are timed for each workload:

- ``statevector``: ``qiskit.quantum_info.Statevector`` (exact, in-process)
- ``aer``: ``qiskit-aer``'s C++ statevector simulator, using
  ``save_expectation_value`` for observables (no transpilation step, to
  be generous to Qiskit)

Usage:
    python qiskit_baseline.py [--out results.json] [--quick]
"""

import argparse
import json
import math
import platform
import statistics
import sys
import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

try:
    from qiskit_aer import AerSimulator

    HAVE_AER = True
except ImportError:  # pragma: no cover
    HAVE_AER = False

PI = math.pi


def param(e: int, j: int) -> float:
    """Deterministic parameter schedule shared with the Rust benchmark.

    Produces generic angles (irrational multiples of pi), so neither
    framework benefits from special-cased common angles.
    """
    x = ((e * 131 + j * 31) % 1000) / 1000.0
    return (2.0 * x - 1.0) * PI


def vqe_ansatz(n: int, layers: int, eval_idx: int) -> QuantumCircuit:
    """Hardware-efficient ansatz: layers x (RY + RZ on every qubit + CNOT chain)."""
    qc = QuantumCircuit(n)
    j = 0
    for _ in range(layers):
        for q in range(n):
            qc.ry(param(eval_idx, j), q)
            j += 1
        for q in range(n):
            qc.rz(param(eval_idx, j), q)
            j += 1
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc


def ising_hamiltonian(n: int) -> SparsePauliOp:
    """Transverse-field Ising on a line: H = sum Z_i Z_{i+1} + 0.5 sum X_i."""
    terms = [("ZZ", [i, i + 1], 1.0) for i in range(n - 1)]
    terms += [("X", [i], 0.5) for i in range(n)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def qaoa_circuit(n: int, p: int, eval_idx: int) -> QuantumCircuit:
    """QAOA MaxCut circuit on a ring graph, depth p."""
    qc = QuantumCircuit(n)
    qc.h(range(n))
    j = 0
    for _ in range(p):
        gamma = param(eval_idx, j)
        j += 1
        for q in range(n):
            qc.rzz(gamma, q, (q + 1) % n)
        beta = param(eval_idx, j)
        j += 1
        for q in range(n):
            qc.rx(beta, q)
    return qc


def maxcut_cost(n: int) -> SparsePauliOp:
    """MaxCut cost on the ring: C = sum_edges (I - Z_i Z_j) / 2."""
    terms = [("", [], n * 0.5)]
    terms += [("ZZ", [i, (i + 1) % n], -0.5) for i in range(n)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def ghz_circuit(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    return qc


# --- Execution paths -------------------------------------------------------


def sv_expectation(build, op):
    def run(eval_idx: int) -> float:
        qc = build(eval_idx)
        return Statevector.from_instruction(qc).expectation_value(op).real

    return run


def aer_expectation(build, op, backend):
    def run(eval_idx: int) -> float:
        qc = build(eval_idx)
        qc.save_expectation_value(op, qc.qubits)
        result = backend.run(qc).result()
        return result.data(0)["expectation_value"]

    return run


def sv_sampling(n: int, shots: int):
    def run(eval_idx: int):
        qc = ghz_circuit(n)
        return Statevector.from_instruction(qc).sample_counts(shots)

    return run


def aer_sampling(n: int, shots: int, backend):
    def run(eval_idx: int):
        qc = ghz_circuit(n)
        qc.measure_all()
        return backend.run(qc, shots=shots).result().get_counts()

    return run


# --- Timing harness --------------------------------------------------------


def time_workload(fn, min_batch_seconds=0.4, batches=5, warmup=3):
    """Median over `batches` of the mean per-iteration wall time."""
    eval_idx = 0
    for _ in range(warmup):
        eval_idx += 1
        fn(eval_idx)

    # Pick a batch size so one batch runs at least min_batch_seconds
    start = time.perf_counter()
    eval_idx += 1
    fn(eval_idx)
    once = time.perf_counter() - start
    batch = max(1, int(min_batch_seconds / max(once, 1e-9)))

    per_iter = []
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(batch):
            eval_idx += 1
            fn(eval_idx)
        per_iter.append((time.perf_counter() - start) / batch)
    return statistics.median(per_iter)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="write results JSON to this file")
    parser.add_argument(
        "--quick", action="store_true", help="fewer batches (faster, noisier)"
    )
    args = parser.parse_args()

    batches = 3 if args.quick else 5
    backend = AerSimulator(method="statevector") if HAVE_AER else None

    workloads = {}
    for n in [4, 8, 12, 16]:
        op = ising_hamiltonian(n)
        build = lambda e, n=n: vqe_ansatz(n, 3, e)
        workloads[f"vqe_energy/{n}q"] = {
            "statevector": sv_expectation(build, op),
            "aer": aer_expectation(build, op, backend) if backend else None,
        }
    for n in [4, 8, 12, 16]:
        op = maxcut_cost(n)
        build = lambda e, n=n: qaoa_circuit(n, 2, e)
        workloads[f"qaoa_maxcut/{n}q"] = {
            "statevector": sv_expectation(build, op),
            "aer": aer_expectation(build, op, backend) if backend else None,
        }
    for n in [10, 16, 20]:
        workloads[f"ghz_sampling/{n}q"] = {
            "statevector": sv_sampling(n, 1024),
            "aer": aer_sampling(n, 1024, backend) if backend else None,
        }

    import qiskit

    results = {
        "framework_versions": {
            "qiskit": qiskit.__version__,
            "qiskit_aer": __import__("qiskit_aer").__version__ if HAVE_AER else None,
            "python": platform.python_version(),
        },
        "platform": platform.platform(),
        "timings_seconds": {},
    }

    for name, paths in workloads.items():
        results["timings_seconds"][name] = {}
        for path_name, fn in paths.items():
            if fn is None:
                continue
            seconds = time_workload(fn, batches=batches)
            results["timings_seconds"][name][path_name] = seconds
            print(f"{name:24s} {path_name:12s} {seconds * 1e3:10.3f} ms/iter", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
