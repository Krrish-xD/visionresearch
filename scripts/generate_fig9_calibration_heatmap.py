"""
Generate Figure 9: Comprehensive Calibration Comparison Table + Heatmap.
Shows ECE-10, Brier, and NLL across all 4 methods x 9 conditions as a colour-coded
publication-grade figure suitable for a conference paper appendix.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

OUT_DIR = "paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────
# Data from validate_calibration_all_methods.py run (all 9 conditions)
# ────────────────────────────────────────────────────────────────────────

conditions = [
    "LLaVA-1.5-7B\nMMVP",
    "LLaVA-1.5-7B\nCLEVR",
    "LLaVA-1.5-7B\nGQA",
    "InternVL3-8B\nMMVP",
    "InternVL3-8B\nCLEVR",
    "InternVL3-8B\nGQA",
    "Qwen2.5-VL-7B\nMMVP",
    "Qwen2.5-VL-7B\nCLEVR",
    "Qwen2.5-VL-7B\nGQA",
]

# ECE-10 for [Raw, Temp, Iso, Conf] per condition
ece_data = np.array([
    [0.5893, 0.3039, 0.0178, 0.1524],
    [0.1167, 0.0763, 0.0505, 0.4014],
    [0.1605, 0.1174, 0.0035, 0.3536],
    [0.4852, 0.1242, 0.0737, 0.2210],
    [0.6082, 0.1786, 0.0632, 0.0695],
    [0.6131, 0.1868, 0.0448, 0.1315],
    [0.1306, 0.0791, 0.0676, 0.3421],
    [0.5508, 0.2147, 0.0250, 0.0870],
    [0.4721, 0.2660, 0.0296, 0.0719],
])

brier_data = np.array([
    [0.5285, 0.3116, 0.1715, 0.2014],
    [0.1691, 0.1563, 0.0995, 0.3372],
    [0.2023, 0.1771, 0.1248, 0.3206],
    [0.4847, 0.2815, 0.2695, 0.2979],
    [0.5818, 0.2414, 0.1947, 0.1954],
    [0.5870, 0.2654, 0.1981, 0.2342],
    [0.2148, 0.2033, 0.2076, 0.3197],
    [0.5092, 0.2297, 0.1990, 0.1735],
    [0.3946, 0.2480, 0.1705, 0.1617],
])

nll_data = np.array([
    [1.4667, 0.8996, 0.5265, 0.6293],
    [0.5387, 0.4875, 0.3322, 0.8752],
    [0.6431, 0.5349, 0.3883, 0.8407],
    [1.7330, 0.7837, 0.7677, 0.8112],
    [1.9345, 0.6845, 0.5985, 0.6180],
    [2.0112, 0.7790, 0.5922, 1.1431],
    [0.6477, 0.6307, 0.6321, 0.9189],
    [1.5572, 0.6487, 0.5814, 0.5109],
    [1.0946, 0.7159, 0.4997, 0.4848],
])

methods = ["Raw Conf.\n(Baseline)", "Temp. Scaling\n(TS)", "Isotonic Reg.\n(PAVA)", "APS Conformal\n(CP)"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), facecolor="#0d1117")
fig.suptitle(
    "Figure 9: Calibration Metrics Across All 4 Methods and 9 Model-Dataset Conditions\n"
    "(lower = better; best per row highlighted in gold)",
    fontsize=13, fontweight="bold", color="#e6edf3", y=1.01
)

metric_names = ["ECE-10 (lower = better)", "Brier Score (lower = better)", "NLL (lower = better)"]
metric_data  = [ece_data, brier_data, nll_data]

cmap = matplotlib.colormaps.get_cmap("RdYlGn_r")

for ax_i, (ax, data, metric_name) in enumerate(zip(axes, metric_data, metric_names)):
    ax.set_facecolor("#0d1117")

    normed = (data - data.min()) / (data.max() - data.min() + 1e-9)

    im = ax.imshow(normed, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9, color="#c9d1d9")
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions, fontsize=8.5, color="#c9d1d9")

    ax.tick_params(axis="both", colors="#8b949e", length=0)
    ax.set_title(metric_name, fontsize=10, fontweight="bold", color="#e6edf3", pad=10)

    # Cell text + best-column highlight
    best_col = np.argmin(data, axis=1)
    for ri in range(len(conditions)):
        for ci in range(len(methods)):
            val = data[ri, ci]
            text_color = "white" if normed[ri, ci] > 0.65 else "black"
            weight = "bold" if ci == best_col[ri] else "normal"
            ax.text(ci, ri, f"{val:.4f}", ha="center", va="center",
                    fontsize=7.8, color=text_color, fontweight=weight)
            if ci == best_col[ri]:
                ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                             fill=False, edgecolor="gold", lw=2.0, zorder=3))

    for spine in ax.spines.values():
        spine.set_color("#30363d")

# Colourbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.015, pad=0.02)
cbar.set_label("Relative metric (0=best, 1=worst)", color="#8b949e", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="#8b949e")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e", fontsize=8)

fig.tight_layout()

out_pdf = os.path.join(OUT_DIR, "fig9_calibration_heatmap.pdf")
out_png = os.path.join(OUT_DIR, "fig9_calibration_heatmap.png")
fig.savefig(out_pdf, dpi=150, bbox_inches="tight", facecolor="#0d1117")
fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="#0d1117")
print(f"[OK] Figure 9 saved to {out_pdf} and {out_png}")
