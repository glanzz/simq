#!/usr/bin/env bash
# One-command reproduction of the cross-validated SimQ vs Qiskit benchmarks.
#
# Requires: Rust toolchain, Python 3 with `pip install qiskit qiskit-aer`.
#
# 1. Times SimQ workloads with criterion
# 2. Times the identical workloads in Qiskit (Statevector + Aer)
# 3. Cross-validates 12 observable values to 1e-12, then prints the table
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] SimQ benchmarks (criterion)"
cargo bench -p simq --bench end_to_end

echo "[2/3] Qiskit baseline"
python3 benchmarks/qiskit_baseline.py

echo "[3/3] Cross-validation + comparison"
python3 benchmarks/compare.py
