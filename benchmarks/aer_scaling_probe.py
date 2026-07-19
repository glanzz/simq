#!/usr/bin/env python3
"""High-qubit scaling probe for Qiskit Aer -- the Aer-side counterpart of
``simq-sim/examples/scaling_probe.rs`` and ``benchmarks/qulacs_scaling_probe.py``.

Times the identical 1-layer VQE-style circuit (H layer, then one layer of
RY / CNOT-chain / RZ) at increasing qubit counts, matching the other two
probes' circuit exactly (same angle formulas, same gate order). Circuit is
transpiled once outside the timed region (consistent with this suite's
existing Aer fairness convention -- see ``qiskit_baseline.py``); the timed
region includes running the circuit and retrieving the final statevector,
matching what "Where it actually fails" in BENCHMARKS.md has always
described for Aer.

Run individual sizes to bound memory, same as the other two probes:
``python3 benchmarks/aer_scaling_probe.py 20 22 24``.
"""

import sys
import time

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.exceptions import CircuitTooWideForTarget
from qiskit_aer import AerSimulator


def vqe_circuit(n, layers):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for l in range(layers):
        for q in range(n):
            theta = 0.1 + 0.37 * (l * n + q)
            qc.ry(theta, q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
        for q in range(n):
            phi = 0.05 + 0.21 * (l * n + q)
            qc.rz(phi, q)
    qc.save_statevector()
    return qc


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [16, 18, 20, 22]
    aer = AerSimulator(method="statevector")

    print(f"{'qubits':>6} {'gates':>7} {'total(ms)':>12} {'GiB state':>10}")
    for n in sizes:
        qc = vqe_circuit(n, 1)
        gates = qc.size() - 1  # exclude the save_statevector instruction
        try:
            transpiled = transpile(qc, aer)
        except CircuitTooWideForTarget as e:
            print(f"{n:>6}  FAILED: {e}")
            continue
        reps = 1 if n >= 24 else 3

        if n < 22:
            aer.run(transpiled).result()

        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            result = aer.run(transpiled).result()
            if not result.success:
                print(f"{n:>6}  FAILED: {result.status}")
                best = float("nan")
                break
            _ = result.get_statevector()
            elapsed = (time.perf_counter() - t0) * 1e3
            best = min(best, elapsed)

        gib = (1 << n) * 16 / (1 << 30)
        print(f"{n:>6} {gates:>7} {best:>12.1f} {gib:>10.3f}")


if __name__ == "__main__":
    main()
