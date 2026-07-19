#!/usr/bin/env python3
"""Render benchmarks/results-scaling-{light,dark}.svg from the 18-30 qubit
scaling probe data (benchmarks/scaling_results.json).

This is a separate chart from make_chart.py's 4-16 qubit workload panels:
different circuit (1-layer VQE-style, not the three cross-validated
workloads), different backends (SimQ/Aer/qulacs only, no qsim/Cirq scaling
probe exists yet), and a genuinely unplottable 30-qubit row for all three
simulators (SimQ/Aer reject it outright, qulacs was never attempted) --
handled here as an explicit "wall" marker instead of a bar, rather than
omitting 30q or (worse) silently plotting it as zero/missing.

Run after regenerating benchmarks/scaling_results.json (see BENCHMARKS.md's
"Where it actually fails: 18-30 qubits" for how those numbers are measured):

    python3 benchmarks/make_scaling_chart.py
"""

import json
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIES = ["SimQ", "Aer", "qulacs"]
# Same color slots as make_chart.py's SERIES list (index 0/2/5: blue/yellow/teal)
# so the same simulator reads as the same color across both charts.
THEMES = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e",
        "grid": "#e4e3df", "wall": "#c7c6c1",
        "series": ["#2a78d6", "#eda100", "#0d9488"],
    },
    "dark": {
        "surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7",
        "grid": "#33322f", "wall": "#4a4945",
        "series": ["#3987e5", "#c98500", "#2dd4bf"],
    },
}

LOG_MIN, LOG_MAX = 10.0, 30000.0  # ms

# Short chart labels for the 30q wall -- full reasons live in
# scaling_results.json's "failure_30q" and in BENCHMARKS.md's prose.
SHORT_REASON = {
    "simq": 'rejected: "max supported is 29"',
    "aer": 'rejected: "maximum (29)"',
    "qulacs": "not attempted (OOM risk)",
}


def load_data():
    with open(os.path.join(REPO, "benchmarks", "scaling_results.json")) as f:
        return json.load(f)


def fmt(ms):
    if ms is None:
        return ""
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    if ms >= 100:
        return f"{ms:.0f} ms"
    return f"{ms:.1f} ms"


def render(mode, data):
    t = THEMES[mode]
    qubits = data["qubits"]
    cols = [data["simq_ms"], data["aer_ms"], data["qulacs_ms"]]
    n_series = len(SERIES)

    bar_h, bar_gap, group_gap = 16, 4, 20
    group_h = n_series * bar_h + (n_series - 1) * bar_gap
    left_pad, right_pad, top = 90, 330, 78
    plot_w = 560
    plot_h = len(qubits) * group_h + (len(qubits) - 1) * group_gap
    width = left_pad + plot_w + right_pad
    height = top + plot_h + 50

    def x_of(ms):
        f = (math.log10(ms) - math.log10(LOG_MIN)) / (math.log10(LOG_MAX) - math.log10(LOG_MIN))
        return left_pad + f * plot_w

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
              f'viewBox="0 0 {width} {height}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">')
    s.append(f'<rect width="{width}" height="{height}" fill="{t["surface"]}" rx="8"/>')
    s.append(f'<text x="{left_pad}" y="24" fill="{t["text"]}" font-size="15" font-weight="600">'
              f'SimQ vs Aer vs qulacs — scaling to 30 qubits (log scale, lower is better)</text>')
    s.append(f'<text x="{left_pad}" y="42" fill="{t["muted"]}" font-size="11">'
              f'1-layer VQE circuit (H+RY+CNOT-chain+RZ) · Xeon 2.10 GHz, 4 vCPU, 15 GiB · '
              f'medians of 3 fresh runs</text>')

    # Legend
    lx = left_pad
    for i, name in enumerate(SERIES):
        s.append(f'<rect x="{lx}" y="{top - 20}" width="10" height="10" rx="2" fill="{t["series"][i]}"/>')
        s.append(f'<text x="{lx + 14}" y="{top - 11}" fill="{t["muted"]}" font-size="11">{name}</text>')
        lx += 14 + 8 * len(name) + 22
    s.append(f'<rect x="{lx}" y="{top - 20}" width="10" height="10" rx="2" '
              f'fill="none" stroke="{t["wall"]}" stroke-width="1.5" stroke-dasharray="2,2"/>')
    s.append(f'<text x="{lx + 14}" y="{top - 11}" fill="{t["muted"]}" font-size="11">rejected / not attempted</text>')

    # Gridlines (decades)
    for d in [10, 100, 1000, 10000]:
        gx = x_of(d)
        s.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + plot_h}" '
                  f'stroke="{t["grid"]}" stroke-width="1"/>')
        label = f"{d / 1000:g}s" if d >= 1000 else f"{d:g}ms"
        s.append(f'<text x="{gx:.1f}" y="{top + plot_h + 16}" fill="{t["muted"]}" '
                  f'font-size="9" text-anchor="middle">{label}</text>')

    wall_x = x_of(LOG_MAX) - 4

    for gi, n in enumerate(qubits):
        gy = top + gi * (group_h + group_gap)
        gib = data["state_gib"][gi]
        gib_s = f"{gib * 1024:.0f} MiB" if gib < 1 else f"{gib:.0f} GiB"
        s.append(f'<text x="{left_pad - 8}" y="{gy + group_h / 2 + 3}" fill="{t["text"]}" '
                  f'font-size="11" font-weight="600" text-anchor="end">{n}q</text>')
        s.append(f'<text x="{left_pad - 8}" y="{gy + group_h / 2 + 15}" fill="{t["muted"]}" '
                  f'font-size="8.5" text-anchor="end">{gib_s}</text>')

        for si, col in enumerate(cols):
            ms = col[gi]
            by = gy + si * (bar_h + bar_gap)
            if ms is None:
                # 30q wall: dashed outline reaching the right edge of the
                # plotted range, not a fabricated bar height.
                s.append(f'<rect x="{left_pad}" y="{by}" width="{wall_x - left_pad:.1f}" height="{bar_h}" '
                          f'rx="2" fill="none" stroke="{t["wall"]}" stroke-width="1.5" stroke-dasharray="4,3"/>')
                reason = SHORT_REASON[SERIES[si].lower()]
                s.append(f'<text x="{wall_x + 6:.1f}" y="{by + bar_h - 3.5}" '
                          f'fill="{t["muted"]}" font-size="8.5" font-style="italic">{reason}</text>')
            else:
                bw = max(x_of(ms) - left_pad, 1.5)
                s.append(f'<rect x="{left_pad}" y="{by}" width="{bw:.1f}" height="{bar_h}" '
                          f'rx="2" fill="{t["series"][si]}"/>')
                s.append(f'<text x="{left_pad + bw + 5:.1f}" y="{by + bar_h - 3.5:.1f}" '
                          f'fill="{t["muted"]}" font-size="9.5">{fmt(ms)}</text>')

    s.append('</svg>')
    out = os.path.join(REPO, "benchmarks", f"results-scaling-{mode}.svg")
    with open(out, "w") as f:
        f.write("\n".join(s))
    print("wrote", out)


def main():
    data = load_data()
    for mode in ("light", "dark"):
        render(mode, data)


if __name__ == "__main__":
    main()
