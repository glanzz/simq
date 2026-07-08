#!/usr/bin/env python3
"""Render benchmarks/results-{light,dark}.svg from the measured data.

Reads SimQ medians from target/criterion/*/new/estimates.json and Qiskit
medians from benchmarks/qiskit_results.json, so the committed chart always
reflects the committed measurements. Run after ./benchmarks/run.sh:

    python3 benchmarks/make_chart.py
"""

import json
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = [4, 8, 12, 16]
PANELS = [("vqe_energy", "VQE energy"), ("qaoa_maxcut", "QAOA MaxCut"),
          ("ghz_sampling", "GHZ 1024 shots")]
SERIES = ["SimQ", "Statevector", "Aer"]

# Categorical slots 1-3 (blue, aqua, yellow), stepped per surface mode;
# palette validated with the six-checks script for both modes.
THEMES = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e",
        "grid": "#e4e3df", "series": ["#2a78d6", "#1baf7a", "#eda100"],
    },
    "dark": {
        "surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7",
        "grid": "#33322f", "series": ["#3987e5", "#199e70", "#c98500"],
    },
}

LOG_MIN, LOG_MAX = 0.01, 1000.0  # ms


def load_data():
    with open(os.path.join(REPO, "benchmarks", "qiskit_results.json")) as f:
        q = json.load(f)
    data = {}
    for group, _ in PANELS:
        for n in SIZES:
            est = os.path.join(REPO, "target", "criterion", group, f"{n}q",
                               "new", "estimates.json")
            with open(est) as f:
                simq_ms = json.load(f)["median"]["point_estimate"] / 1e6
            key = f"{group}/{n}q"
            data[key] = [simq_ms,
                         q["timings_ms"]["statevector"][key],
                         q["timings_ms"]["aer"][key]]
    return data


def fmt(ms):
    if ms >= 100:
        return f"{ms:.0f}"
    if ms >= 1:
        return f"{ms:.1f}"
    return f"{ms:.2f}"


def render(mode, data):
    t = THEMES[mode]
    panel_w, panel_gap, left_pad = 268, 26, 46
    bar_h, bar_gap, group_gap = 9, 2, 16
    group_h = 3 * bar_h + 2 * bar_gap
    top = 100
    plot_h = 4 * group_h + 3 * group_gap
    width = left_pad + 3 * panel_w + 2 * panel_gap + 12
    height = top + plot_h + 46

    def x_of(panel_x0, ms):
        f = (math.log10(ms) - math.log10(LOG_MIN)) / (math.log10(LOG_MAX) - math.log10(LOG_MIN))
        return panel_x0 + f * (panel_w - 40)

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">')
    s.append(f'<rect width="{width}" height="{height}" fill="{t["surface"]}" rx="8"/>')
    s.append(f'<text x="{left_pad}" y="26" fill="{t["text"]}" font-size="15" font-weight="600">'
             'SimQ vs Qiskit — median time per full evaluation (log scale, lower is better)</text>')
    s.append(f'<text x="{left_pad}" y="44" fill="{t["muted"]}" font-size="11">'
             'Identical cross-validated circuits (12 values agree to 1e-12) · Xeon 2.80 GHz, 4 vCPU · '
             'Qiskit 2.5.0 / Aer 0.17.2 · ./benchmarks/run.sh</text>')

    # Legend
    lx = left_pad
    for i, name in enumerate(SERIES):
        s.append(f'<rect x="{lx}" y="{top - 20}" width="10" height="10" rx="2" fill="{t["series"][i]}"/>')
        s.append(f'<text x="{lx + 14}" y="{top - 11}" fill="{t["muted"]}" font-size="11">{name}</text>')
        lx += 14 + 8 * len(name) + 22

    for p, (group, title) in enumerate(PANELS):
        px0 = left_pad + p * (panel_w + panel_gap)
        s.append(f'<text x="{px0}" y="{top - 40}" fill="{t["text"]}" font-size="12" '
                 f'font-weight="600">{title}</text>')
        # Gridlines each decade
        for d in [0.01, 0.1, 1, 10, 100, 1000]:
            gx = x_of(px0, d)
            s.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + plot_h}" '
                     f'stroke="{t["grid"]}" stroke-width="1"/>')
            label = f"{d:g}"
            s.append(f'<text x="{gx:.1f}" y="{top + plot_h + 16}" fill="{t["muted"]}" '
                     f'font-size="9" text-anchor="middle">{label}</text>')
        s.append(f'<text x="{x_of(px0, 3):.1f}" y="{top + plot_h + 30}" fill="{t["muted"]}" '
                 f'font-size="9" text-anchor="middle">ms</text>')

        for gi, n in enumerate(SIZES):
            gy = top + gi * (group_h + group_gap)
            if p == 0:
                s.append(f'<text x="{px0 - 8}" y="{gy + group_h / 2 + 3}" fill="{t["muted"]}" '
                         f'font-size="10" text-anchor="end">{n}q</text>')
            vals = data[f"{group}/{n}q"]
            for si, ms in enumerate(vals):
                by = gy + si * (bar_h + bar_gap)
                bw = max(x_of(px0, ms) - px0, 1.5)
                s.append(f'<rect x="{px0}" y="{by}" width="{bw:.1f}" height="{bar_h}" '
                         f'rx="2" fill="{t["series"][si]}"/>')
                s.append(f'<text x="{px0 + bw + 4:.1f}" y="{by + bar_h - 1.5}" '
                         f'fill="{t["muted"]}" font-size="8.5">{fmt(ms)}</text>')

    s.append('</svg>')
    out = os.path.join(REPO, "benchmarks", f"results-{mode}.svg")
    with open(out, "w") as f:
        f.write("\n".join(s))
    print("wrote", out)


def main():
    data = load_data()
    for mode in ("light", "dark"):
        render(mode, data)


if __name__ == "__main__":
    main()
