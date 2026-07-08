#!/usr/bin/env python3
"""Merge Ferriq criterion results with the Qiskit baseline and report.

This script is the one-stop reproduction path for `BENCHMARKS.md`:

1. (optional, default on) runs the workload cross-validation so both
   suites provably simulate identical circuits,
2. runs ``cargo bench -p ferriq --bench end_to_end``,
3. runs ``qiskit_baseline.py`` with the Python interpreter you point it at,
4. parses criterion's ``estimates.json`` files and the baseline JSON,
5. prints a Markdown results table and writes ``results.json`` plus the
   speedup charts (``chart-light.svg`` / ``chart-dark.svg``).

Typical use (from the repository root, inside a venv with Qiskit):

    python benchmarks/compare.py --out-dir benchmarks/results/$(date +%F)

Use --skip-rust / --skip-qiskit to reuse existing results while iterating.
"""

import argparse
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

WORKLOADS = [
    *(f"vqe_energy/{n}q" for n in (4, 8, 12, 16)),
    *(f"qaoa_maxcut/{n}q" for n in (4, 8, 12, 16)),
    *(f"ghz_sampling/{n}q" for n in (10, 16, 20)),
]


def run(cmd, **kwargs):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def read_criterion_results():
    """Read mean per-iteration seconds from criterion's estimates.json."""
    timings = {}
    for name in WORKLOADS:
        est = REPO_ROOT / "target" / "criterion" / name / "new" / "estimates.json"
        if not est.exists():
            sys.exit(f"missing {est} — did `cargo bench -p ferriq` run?")
        with open(est) as f:
            timings[name] = json.load(f)["mean"]["point_estimate"] / 1e9
    return timings


def cross_validate(python):
    """Assert Ferriq and Qiskit compute identical expectation values."""
    rust = subprocess.run(
        ["cargo", "run", "--release", "-p", "ferriq", "--example", "xcheck_bench"],
        check=True, capture_output=True, text=True, cwd=REPO_ROOT,
    ).stdout
    qiskit = subprocess.run(
        [python, "-c", """
import sys
sys.path.insert(0, %r)
from qiskit_baseline import vqe_ansatz, ising_hamiltonian, qaoa_circuit, maxcut_cost
from qiskit.quantum_info import Statevector
for e in [1, 2, 7]:
    for n in [4, 8]:
        v = Statevector.from_instruction(vqe_ansatz(n, 3, e)).expectation_value(ising_hamiltonian(n)).real
        print(f'vqe {n}q eval{e}: {v:.12f}')
        q = Statevector.from_instruction(qaoa_circuit(n, 2, e)).expectation_value(maxcut_cost(n)).real
        print(f'qaoa {n}q eval{e}: {q:.12f}')
""" % str(REPO_ROOT / "benchmarks")],
        check=True, capture_output=True, text=True,
    ).stdout
    if rust.strip() != qiskit.strip():
        print(rust)
        print(qiskit)
        sys.exit("cross-validation FAILED: Ferriq and Qiskit disagree on workload values")
    print(f"cross-validation OK: {len(rust.strip().splitlines())} expectation values identical")


# --- Chart -----------------------------------------------------------------

INK = {
    "light": {
        "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "baseline": "#c3c2b7",
        "s1": "#2a78d6", "s2": "#1baf7a",
    },
    "dark": {
        "surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "baseline": "#383835",
        "s1": "#3987e5", "s2": "#199e70",
    },
}

ROW_LABELS = {
    "vqe_energy": "VQE energy",
    "qaoa_maxcut": "QAOA MaxCut",
    "ghz_sampling": "GHZ sampling",
}


def render_chart(rows, mode, versions):
    """Dot plot of Ferriq speedup vs Qiskit on a log axis. rows:
    (workload, speedup_vs_statevector, speedup_vs_aer)."""
    c = INK[mode]
    width, row_h = 880, 30
    left, right, top = 190, 30, 96
    plot_w = width - left - right
    height = top + row_h * len(rows) + 64
    lo, hi = math.log10(0.02), math.log10(2000)

    def x(v):
        return left + (math.log10(v) - lo) / (hi - lo) * plot_w

    font = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Ferriq speedup over Qiskit per workload, log scale">',
        f'<rect width="{width}" height="{height}" fill="{c["surface"]}" rx="8"/>',
        f'<text x="{left}" y="30" {font} font-size="15" font-weight="600" fill="{c["primary"]}">Ferriq speedup over Qiskit — one variational/sampling iteration</text>',
        f'<text x="{left}" y="50" {font} font-size="12" fill="{c["secondary"]}">Speedup = Qiskit time ÷ Ferriq time (log scale) · GHZ rows include 1024 shots · {versions}</text>',
    ]
    # legend
    lx = left
    for color, label in ((c["s1"], "vs Qiskit Statevector"), (c["s2"], "vs Qiskit Aer")):
        s.append(f'<circle cx="{lx + 5}" cy="68" r="5" fill="{color}"/>')
        s.append(f'<text x="{lx + 15}" y="72" {font} font-size="12" fill="{c["secondary"]}">{label}</text>')
        lx += 160

    plot_bottom = top + row_h * len(rows)
    # gridlines + ticks
    for tick in (0.1, 1, 10, 100, 1000):
        tx = x(tick)
        major = tick == 1
        color = c["baseline"] if major else c["grid"]
        w = 1.5 if major else 1
        s.append(f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{plot_bottom}" stroke="{color}" stroke-width="{w}"/>')
        label = "1× parity" if major else (f"{tick:g}×")
        weight = ' font-weight="600"' if major else ""
        s.append(f'<text x="{tx:.1f}" y="{plot_bottom + 18}" {font} font-size="11"{weight} fill="{c["muted"]}" text-anchor="middle">{label}</text>')
    s.append(f'<text x="{x(0.1):.1f}" y="{plot_bottom + 36}" {font} font-size="11" fill="{c["muted"]}" text-anchor="middle">← Qiskit faster</text>')
    s.append(f'<text x="{x(10):.1f}" y="{plot_bottom + 36}" {font} font-size="11" fill="{c["muted"]}" text-anchor="middle">Ferriq faster →</text>')

    prev_family = None
    for i, (name, sv, aer) in enumerate(rows):
        cy = top + row_h * i + row_h / 2
        family, size = name.split("/")
        label = ROW_LABELS[family] if family != prev_family else ""
        prev_family = family
        if label:
            s.append(f'<text x="12" y="{cy + 4}" {font} font-size="12" font-weight="600" fill="{c["primary"]}">{label}</text>')
        s.append(f'<text x="{left - 12}" y="{cy + 4}" {font} font-size="12" fill="{c["secondary"]}" text-anchor="end">{size.replace("q", " qubits")}</text>')
        # connector between the two dots of a row
        x1, x2 = sorted((x(sv), x(aer)))
        s.append(f'<line x1="{x1:.1f}" y1="{cy}" x2="{x2:.1f}" y2="{cy}" stroke="{c["grid"]}" stroke-width="1"/>')
        for v, color in ((sv, c["s1"]), (aer, c["s2"])):
            s.append(f'<circle cx="{x(v):.1f}" cy="{cy}" r="5" fill="{color}" stroke="{c["surface"]}" stroke-width="2"/>')
        # direct label at the row's extremes
        vmax = max(sv, aer)
        fmt = lambda v: (f"{v:.2f}×" if v < 1 else f"{v:.1f}×" if v < 30 else f"{v:.0f}×")
        s.append(f'<text x="{x(vmax) + 10:.1f}" y="{cy + 4}" {font} font-size="11" fill="{c["secondary"]}">{fmt(vmax)}</text>')
        vmin = min(sv, aer)
        if x(vmax) - x(vmin) > 46 and x(vmin) > left + 48:
            s.append(f'<text x="{x(vmin) - 10:.1f}" y="{cy + 4}" {font} font-size="11" fill="{c["secondary"]}" text-anchor="end">{fmt(vmin)}</text>')
    s.append("</svg>")
    return "\n".join(s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter with qiskit installed (default: current)")
    parser.add_argument("--out-dir", default="benchmarks/results/latest")
    parser.add_argument("--skip-rust", action="store_true", help="reuse existing criterion results")
    parser.add_argument("--skip-qiskit", action="store_true", help="reuse <out-dir>/qiskit.json")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_validation:
        cross_validate(args.python)
    if not args.skip_rust:
        run(["cargo", "bench", "-p", "ferriq", "--bench", "end_to_end"], cwd=REPO_ROOT)
    qiskit_json = out_dir / "qiskit.json"
    if not args.skip_qiskit:
        run([args.python, str(REPO_ROOT / "benchmarks" / "qiskit_baseline.py"),
             "--out", str(qiskit_json)])

    ferriq = read_criterion_results()
    with open(qiskit_json) as f:
        baseline = json.load(f)
    qk = baseline["timings_seconds"]

    versions = baseline["framework_versions"]
    results = {
        "hardware": {"platform": platform.platform(), "cpu_count": __import__("os").cpu_count()},
        "framework_versions": versions,
        "timings_ms": {},
    }
    rows = []
    lines = [
        "| Workload | Ferriq | Qiskit Statevector | Qiskit Aer | vs Statevector | vs Aer |",
        "|---|---|---|---|---|---|",
    ]
    fmt_ms = lambda s: f"{s * 1e3:,.3f} ms" if s < 1 else f"{s:,.2f} s"
    for name in WORKLOADS:
        t_ferriq = ferriq[name]
        t_sv = qk[name]["statevector"]
        t_aer = qk[name].get("aer")
        sv_speedup = t_sv / t_ferriq
        aer_speedup = t_aer / t_ferriq if t_aer else float("nan")
        rows.append((name, sv_speedup, aer_speedup))
        results["timings_ms"][name] = {
            "ferriq": t_ferriq * 1e3,
            "qiskit_statevector": t_sv * 1e3,
            "qiskit_aer": t_aer * 1e3 if t_aer else None,
        }
        fmt_x = lambda v: f"**{v:,.1f}× faster**" if v >= 1.05 else f"{1 / v:,.1f}× slower"
        lines.append(
            f"| `{name}` | {fmt_ms(t_ferriq)} | {fmt_ms(t_sv)} | {fmt_ms(t_aer)} | "
            f"{fmt_x(sv_speedup)} | {fmt_x(aer_speedup)} |"
        )

    table = "\n".join(lines)
    print("\n" + table + "\n")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    version_note = f"Qiskit {versions['qiskit']} / Aer {versions['qiskit_aer']}"
    for mode in ("light", "dark"):
        with open(out_dir / f"chart-{mode}.svg", "w") as f:
            f.write(render_chart(rows, mode, version_note))
    with open(out_dir / "table.md", "w") as f:
        f.write(table + "\n")
    print(f"wrote {out_dir}/results.json, chart-light.svg, chart-dark.svg, table.md")


if __name__ == "__main__":
    main()
