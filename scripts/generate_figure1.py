"""
Generate Figure 1: End-to-End Neuro-Symbolic Pipeline Diagram for Conference Publication.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_figure_1(output_path: str = "results/figures/fig1_pipeline_architecture.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # Colors
    c_blue = "#2b5c8f"
    c_cyan = "#e0f3f8"
    c_orange = "#d95f02"
    c_light_orange = "#fee8c8"
    c_purple = "#7570b3"
    c_light_purple = "#f2f0f7"
    c_green = "#238b45"
    c_light_green = "#e5f5e0"
    c_dark = "#222222"

    def draw_box(x, y, w, h, title, subtitle, bg_color, border_color):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15,rounding_size=0.2",
            facecolor=bg_color, edgecolor=border_color, linewidth=2.0
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center", fontsize=11, fontweight="bold", color=border_color)
        ax.text(x + w/2, y + h/2 - 0.2, subtitle, ha="center", va="center", fontsize=8.5, color=c_dark)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor="#555555", edgecolor="#555555", width=1.5, headwidth=7, shrink=0.05)
        )
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.18, label, ha="center", va="center", fontsize=8, fontweight="bold", color="#333333")

    # Stage 1: Input Query & Image
    draw_box(
        0.5, 2.0, 2.2, 3.2,
        "1. Multimodal Input",
        "Visual Query (Q)\n+\nAuthentic Image (I)\n[MMVP / CLEVR / GQA]\nPairs: (I_A, I_B)",
        c_cyan, c_blue
    )

    # Arrow 1 -> 2
    draw_arrow(2.7, 3.6, 3.4, 3.6, "Inference")

    # Stage 2: VLM Inference & Logprob Extraction
    draw_box(
        3.4, 2.0, 2.5, 3.2,
        "2. VLM Perception",
        "LLaVA / InternVL3 / Qwen2.5\n\n- Greedy Answer: a_top1\n- Candidate Set: {h_1, ..., h_K}\n- Token Logits: z\n- Raw Confidence: p_raw",
        c_light_orange, c_orange
    )

    # Arrow 2 -> 3 (Calibration)
    draw_arrow(5.9, 4.3, 6.7, 4.8, "p_raw, z")
    # Arrow 2 -> 4 (Autoformalization)
    draw_arrow(5.9, 2.9, 6.7, 2.4, "h_1 ... h_K")

    # Stage 3A: Calibration Layer (Top)
    draw_box(
        6.7, 3.8, 2.8, 2.3,
        "3. Post-Hoc Calibration",
        "• Temperature Scaling (T)\n• Isotonic Monotonic (p_iso)\n• Conformal Sets (APS: C(x))\nECE drops: 61% -> 0.3%",
        c_light_purple, c_purple
    )

    # Stage 3B: Deterministic Autoformalizer (Bottom)
    draw_box(
        6.7, 0.8, 2.8, 2.3,
        "4. Autoformalization",
        "Deterministic Parser\n\nh_i => Predicate Claim:\nchoice(item) == opt_i\ncount(x) == k\nrelation(x, r, y)",
        "#fff7bc", "#d95f0e"
    )

    # Convergence into Z3 MaxSMT
    draw_arrow(9.5, 4.8, 10.3, 4.0, "Weights w_i")
    draw_arrow(9.5, 2.0, 10.3, 3.2, "Claims h_i => C_i")

    # Stage 4: Z3 SMT / MaxSMT Verification
    draw_box(
        10.3, 1.5, 3.2, 4.2,
        "5. Z3 MaxSMT Verifier",
        "HARD Constraints (Sound Grounding):\n   /\\ F_gt  (Benchmark Reality)\n   sum(h_i) <= 1  (Mutex)\n\nSOFT Objectives (Calibrated):\n   max sum(w_i * h_i)\n\nDual Diagnostics:\n[+] Hard Error Rejection (SFAR bounded)\n[+] Active Error Recovery (+65% MMVP)\n[+] Conformal Abstention (p < tau)",
        c_light_green, c_green
    )

    # Title header
    ax.text(7.0, 6.2, "Calibrated Confidence for Symbolic Error Detection & Recovery in Visual Grounding",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#111111")
    ax.text(7.0, 5.85, "A Sound Neuro-Symbolic Framework Combining Multi-Family VLMs, Post-Hoc Calibration, and Z3 MaxSMT",
            ha="center", va="center", fontsize=10.5, style="italic", color="#555555")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 1 (Pipeline Architecture) -> {output_path}")

if __name__ == "__main__":
    generate_figure_1()
