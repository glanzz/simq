#!/usr/bin/env bash
# One-command reproduction of the cross-validated SimQ vs Qiskit/qsim benchmarks.
#
# Requires: Rust toolchain, Python 3 with
#   `pip install qiskit qiskit-aer cirq qsimcirq`.
# The qsim leg is optional (see below); everything else still runs without it.
#
# 1. Times SimQ workloads with criterion
# 2. Times the identical workloads in Qiskit (Statevector + Aer)
# 3. Times the identical workloads in Cirq/qsim (Google), if qsimcirq is installed
# 4. Cross-validates 12 observable values to 1e-12 against Qiskit (and to
#    5e-6 against qsim, whose Python wheel is float32-only) then prints the table
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/4] SimQ benchmarks (criterion)"
cargo bench -p simq --bench end_to_end

echo "[2/4] Qiskit baseline"
python3 benchmarks/qiskit_baseline.py

echo "[3/4] qsim baseline (optional)"
if python3 -c "import qsimcirq" >/dev/null 2>&1; then
    python3 benchmarks/qsim_baseline.py
else
    echo "  qsimcirq not installed - skipping (pip install cirq qsimcirq to include it)"
fi

echo "[4/4] Cross-validation + comparison"
python3 benchmarks/compare.py
