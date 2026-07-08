#!/usr/bin/env bash
# One-command reproduction of BENCHMARKS.md.
#
# Creates a local venv with the pinned Qiskit version, cross-validates that
# both suites simulate identical circuits, runs `cargo bench` and the Qiskit
# baseline, and writes the merged table + charts to benchmarks/results/<date>.
#
# Usage: ./benchmarks/run.sh [out-dir]

set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="${1:-benchmarks/results/$(date +%F)}"
VENV="benchmarks/.venv"

if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
fi
"$VENV/bin/pip" install --quiet -r benchmarks/requirements.txt

exec "$VENV/bin/python" benchmarks/compare.py --python "$VENV/bin/python" --out-dir "$OUT_DIR"
