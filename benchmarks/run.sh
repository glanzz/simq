#!/usr/bin/env bash
# One-command reproduction of the cross-validated SimQ vs Qiskit/qsim/qulacs benchmarks.
#
# Requires: Rust toolchain, Python 3 with
#   `pip install qiskit qiskit-aer cirq qsimcirq qulacs`.
# The qsim and qulacs legs are optional (see below); everything else still
# runs without them.
#
# 1. Times SimQ workloads with criterion
# 2. Times the identical workloads in Qiskit (Statevector + Aer)
# 3. Times the identical workloads in Cirq/qsim (Google), if qsimcirq is installed
# 4. Times the identical workloads in qulacs, if it is installed
# 5. Cross-validates 12 observable values to 1e-12 against Qiskit (and to
#    5e-6 against qsim, whose Python wheel is float32-only, and 1e-12
#    against qulacs) then prints the table
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/5] SimQ benchmarks (criterion)"
cargo bench -p simq --bench end_to_end

echo "[2/5] Qiskit baseline"
python3 benchmarks/qiskit_baseline.py

echo "[3/5] qsim baseline (optional)"
if python3 -c "import qsimcirq" >/dev/null 2>&1; then
    python3 benchmarks/qsim_baseline.py
else
    echo "  qsimcirq not installed - skipping (pip install cirq qsimcirq to include it)"
fi

echo "[4/5] qulacs baseline (optional)"
if python3 -c "import qulacs" >/dev/null 2>&1; then
    python3 benchmarks/qulacs_baseline.py
else
    echo "  qulacs not installed - skipping (pip install qulacs to include it)"
fi

echo "[5/5] Cross-validation + comparison"
python3 benchmarks/compare.py
