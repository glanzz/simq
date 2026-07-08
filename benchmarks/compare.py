#!/usr/bin/env python3
"""Cross-validate SimQ against Qiskit, then print the benchmark comparison.

Refuses to print any timing comparison unless every observable value produced
by the SimQ benchmark workloads matches the Qiskit implementation to 1e-12.

Inputs:
* SimQ values:   `cargo run --release -p simq --example xcheck_bench` (run here)
* SimQ timings:  criterion output in target/criterion/*/new/estimates.json
                 (produced by `cargo bench -p simq --bench end_to_end`)
* Qiskit side:   benchmarks/qiskit_results.json
                 (produced by `python3 benchmarks/qiskit_baseline.py`)

Run everything in order with ./benchmarks/run.sh
"""

import json
import os
import subprocess
import sys

TOLERANCE = 1e-12
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = [4, 8, 12, 16]
GROUPS = ["vqe_energy", "qaoa_maxcut", "ghz_sampling"]


def simq_values():
    out = subprocess.run(
        ["cargo", "run", "--release", "-q", "-p", "simq", "--example", "xcheck_bench"],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=True,
    )
    return json.loads(out.stdout)


def criterion_median_ms(group, param):
    path = os.path.join(REPO, "target", "criterion", group, param, "new", "estimates.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        est = json.load(f)
    return est["median"]["point_estimate"] / 1e6  # ns -> ms


def main():
    qiskit_path = os.path.join(REPO, "benchmarks", "qiskit_results.json")
    if not os.path.exists(qiskit_path):
        sys.exit("benchmarks/qiskit_results.json missing - run benchmarks/qiskit_baseline.py first")
    with open(qiskit_path) as f:
        qiskit = json.load(f)

    print("== Cross-validation (SimQ vs Qiskit exact statevector) ==")
    simq = simq_values()
    worst = 0.0
    failed = False
    for key, sv in sorted(simq.items()):
        qv = qiskit["values"][key]
        diff = abs(sv - qv)
        worst = max(worst, diff)
        status = "ok" if diff <= TOLERANCE else "MISMATCH"
        print(f"  {key:22} simq={sv:+.12f}  qiskit={qv:+.12f}  diff={diff:.2e}  {status}")
        if diff > TOLERANCE:
            failed = True
    print(f"  worst deviation: {worst:.2e} (tolerance {TOLERANCE:.0e})")
    if failed:
        sys.exit("\nCROSS-VALIDATION FAILED - timings withheld. The suites are not "
                 "simulating the same circuits; fix that before comparing numbers.")
    print("  all values match - timing comparison is meaningful\n")

    print(f"== Timings (median, ms) - Qiskit {qiskit['versions']['qiskit']} / "
          f"Aer {qiskit['versions']['qiskit_aer']} ==")
    header = f"{'workload':<18} {'SimQ':>10} {'Statevector':>12} {'ratio':>8} {'Aer':>10} {'ratio':>8}"
    print(header)
    print("-" * len(header))
    missing = False
    for group in GROUPS:
        for n in SIZES:
            key = f"{group}/{n}q"
            simq_ms = criterion_median_ms(group, f"{n}q")
            sv_ms = qiskit["timings_ms"]["statevector"].get(key)
            aer_ms = qiskit["timings_ms"]["aer"].get(key)
            if simq_ms is None:
                print(f"{key:<18} {'(run cargo bench)':>10}")
                missing = True
                continue

            def ratio(other):
                if other is None:
                    return "-"
                r = other / simq_ms
                return f"{r:.1f}x" if r >= 1 else f"1/{1/r:.1f}x"

            sv_s = f"{sv_ms:.3f}" if sv_ms is not None else "-"
            aer_s = f"{aer_ms:.3f}" if aer_ms is not None else "-"
            print(f"{key:<18} {simq_ms:>10.3f} {sv_s:>12} {ratio(sv_ms):>8} "
                  f"{aer_s:>10} {ratio(aer_ms):>8}")
    if missing:
        print("\n(some SimQ rows missing - run: cargo bench -p simq --bench end_to_end)")
    print("\nratio > 1x means SimQ is faster; 1/Nx means SimQ is N times slower.")


if __name__ == "__main__":
    main()
