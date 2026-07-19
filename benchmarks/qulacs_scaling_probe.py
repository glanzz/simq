#!/usr/bin/env python3
"""High-qubit scaling probe for qulacs -- the qulacs-side counterpart of
``simq-sim/examples/scaling_probe.rs``.

Times the identical 1-layer VQE-style circuit (H layer, then one layer of
RY / CNOT-chain / RZ) at increasing qubit counts, matching
``scaling_probe.rs``'s circuit exactly (same angle formulas, same gate
order). Run individual sizes to bound memory, same as the Rust probe:
``python3 benchmarks/qulacs_scaling_probe.py 20 22 24``.

Unlike SimQ/Aer (see BENCHMARKS.md's "30-qubit failure mode" note), qulacs
has no built-in memory-aware qubit cap -- it will simply try to allocate the
state and let the OS allocator/OOM-killer decide. 30 qubits (16 GiB) is
deliberately not attempted here for that reason.

Note the sign-convention fix from ``qulacs_baseline.py`` applies here too:
angles passed to ``add_RY_gate``/``add_RZ_gate`` are negated to match
Qiskit/simq's ``exp(-i*theta/2*P)`` convention (qulacs implements
``exp(+i*theta/2*P)``), verified empirically in ``qulacs_baseline.py``.
"""

import sys
import time

from qulacs import QuantumCircuit, QuantumState


def vqe_circuit(n, layers):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.add_H_gate(q)
    for l in range(layers):
        for q in range(n):
            theta = 0.1 + 0.37 * (l * n + q)
            qc.add_RY_gate(q, -theta)
        for q in range(n - 1):
            qc.add_CNOT_gate(q, q + 1)
        for q in range(n):
            phi = 0.05 + 0.21 * (l * n + q)
            qc.add_RZ_gate(q, -phi)
    return qc


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [16, 18, 20, 22]

    print(f"{'qubits':>6} {'gates':>7} {'total(ms)':>12} {'GiB state':>10}")
    for n in sizes:
        qc = vqe_circuit(n, 1)
        gates = qc.get_gate_count()
        reps = 1 if n >= 24 else 3

        if n < 22:
            st = QuantumState(n)
            qc.update_quantum_state(st)
            del st

        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            st = QuantumState(n)
            qc.update_quantum_state(st)
            elapsed = (time.perf_counter() - t0) * 1e3
            del st
            best = min(best, elapsed)

        gib = (1 << n) * 16 / (1 << 30)
        print(f"{n:>6} {gates:>7} {best:>12.1f} {gib:>10.3f}")


if __name__ == "__main__":
    main()
