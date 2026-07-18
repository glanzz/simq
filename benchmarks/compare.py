#!/usr/bin/env python3
"""Cross-validate SimQ against Qiskit and qsim, then print the benchmark comparison.

Refuses to print any timing comparison unless every observable value produced
by the SimQ benchmark workloads matches the Qiskit implementation to 1e-12.
qsim values are checked too, at a looser 1e-6 tolerance -- qsim's Python
wheel (qsimcirq) ships a single-precision (float32) state vector core with no
fp64 build option, so its own numbers carry ~1e-7 relative error even against
its *own* re-runs. That precision gap is qsim's, not a bug in this harness;
see the fairness notes in BENCHMARKS.md.

Inputs:
* SimQ values:   `cargo run --release -p simq --example xcheck_bench` (run here)
* SimQ timings:  criterion output in target/criterion/*/new/estimates.json
                 (produced by `cargo bench -p simq --bench end_to_end`)
* Qiskit side:   benchmarks/qiskit_results.json
                 (produced by `python3 benchmarks/qiskit_baseline.py`)
* qsim side:     benchmarks/qsim_results.json (optional)
                 (produced by `python3 benchmarks/qsim_baseline.py`)

Run everything in order with ./benchmarks/run.sh
"""

import json
import os
import subprocess
import sys

QISKIT_TOLERANCE = 1e-12
QSIM_TOLERANCE = 5e-6  # qsim's Python wheel is float32-only, see module docstring
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


def cross_validate(name, simq, other_values, tolerance):
    print(f"== Cross-validation (SimQ vs {name}) ==")
    worst = 0.0
    failed = False
    for key, sv in sorted(simq.items()):
        ov = other_values[key]
        diff = abs(sv - ov)
        worst = max(worst, diff)
        status = "ok" if diff <= tolerance else "MISMATCH"
        print(f"  {key:22} simq={sv:+.12f}  {name.lower()}={ov:+.12f}  diff={diff:.2e}  {status}")
        if diff > tolerance:
            failed = True
    print(f"  worst deviation: {worst:.2e} (tolerance {tolerance:.0e})")
    if failed:
        sys.exit(f"\nCROSS-VALIDATION FAILED against {name} - timings withheld. The "
                  "suites are not simulating the same circuits; fix that before "
                  "comparing numbers.")
    print("  all values match - timing comparison is meaningful\n")


def main():
    qiskit_path = os.path.join(REPO, "benchmarks", "qiskit_results.json")
    if not os.path.exists(qiskit_path):
        sys.exit("benchmarks/qiskit_results.json missing - run benchmarks/qiskit_baseline.py first")
    with open(qiskit_path) as f:
        qiskit = json.load(f)

    qsim_path = os.path.join(REPO, "benchmarks", "qsim_results.json")
    qsim = None
    if os.path.exists(qsim_path):
        with open(qsim_path) as f:
            qsim = json.load(f)

    simq = simq_values()
    cross_validate("Qiskit exact statevector", simq, qiskit["values"], QISKIT_TOLERANCE)
    if qsim is not None:
        cross_validate("qsim exact statevector", simq, qsim["values"], QSIM_TOLERANCE)
    else:
        print("(benchmarks/qsim_results.json missing - skipping qsim cross-validation; "
              "run benchmarks/qsim_baseline.py to include it)\n")

    header_bits = [f"Qiskit {qiskit['versions']['qiskit']}/Aer {qiskit['versions']['qiskit_aer']}"]
    if qsim is not None:
        header_bits.append(f"cirq {qsim['versions']['cirq']}/qsim {qsim['versions']['qsimcirq']}")
    print(f"== Timings (median, ms) - {' vs '.join(header_bits)} ==")

    cols = [("SimQ", None), ("Statevector", "qiskit-sv"), ("Aer", "qiskit-aer")]
    if qsim is not None:
        cols += [("Cirq", "cirq-sv"), ("qsim", "qsim")]

    header = f"{'workload':<18}" + "".join(f" {name:>11}" + (f" {'ratio':>7}" if name != "SimQ" else "")
                                            for name, _ in cols)
    print(header)
    print("-" * len(header))
    missing = False
    for group in GROUPS:
        for n in SIZES:
            key = f"{group}/{n}q"
            simq_ms = criterion_median_ms(group, f"{n}q")
            if simq_ms is None:
                print(f"{key:<18} {'(run cargo bench)':>11}")
                missing = True
                continue

            values = {
                "qiskit-sv": qiskit["timings_ms"]["statevector"].get(key),
                "qiskit-aer": qiskit["timings_ms"]["aer"].get(key),
            }
            if qsim is not None:
                values["cirq-sv"] = qsim["timings_ms"]["cirq"].get(key)
                values["qsim"] = qsim["timings_ms"]["qsim"].get(key)

            def ratio(other):
                if other is None:
                    return "-"
                r = other / simq_ms
                return f"{r:.1f}x" if r >= 1 else f"1/{1/r:.1f}x"

            row = f"{key:<18} {simq_ms:>11.3f}"
            for name, tag in cols[1:]:
                ms = values.get(tag)
                ms_s = f"{ms:.3f}" if ms is not None else "-"
                row += f" {ms_s:>11} {ratio(ms):>7}"
            print(row)
    if missing:
        print("\n(some SimQ rows missing - run: cargo bench -p simq --bench end_to_end)")
    print("\nratio > 1x means SimQ is faster; 1/Nx means SimQ is N times slower.")


if __name__ == "__main__":
    main()
