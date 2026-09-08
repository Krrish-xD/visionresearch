"""Publication-grade figure generation for empirical calibration and verification results."""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

def plot_reliability_diagrams(
    cal_metrics_dict: Dict[str, Dict[str, Any]],
    output_path: str = "results/figures/fig2_reliability_diagrams.png"
):
    """
    Figure 2. Reliability diagrams comparing Raw vs Temperature vs Isotonic calibration.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    methods = list(cal_metrics_dict.keys())
    n = len(methods)

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]

    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]

    for idx, (method, data) in enumerate(cal_metrics_dict.items()):
        ax = axes[idx]
        bin_data = data.get("bin_data_10", {})
        bin_accs = bin_data.get("bin_accs", [])
        bin_confs = bin_data.get("bin_confs", [])
        bin_counts = bin_data.get("bin_counts", [])
        ece = data.get("ece_10", 0.0)

        # Plot diagonal ideal calibration line
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

        # Plot reliability curve
        valid_indices = [i for i, c in enumerate(bin_counts) if c > 0]
        if valid_indices:
            x_vals = [bin_confs[i] for i in valid_indices]
            y_vals = [bin_accs[i] for i in valid_indices]
            ax.plot(x_vals, y_vals, marker="o", linewidth=2, color=colors[idx % len(colors)], label=f"ECE = {ece:.3f}")
            ax.bar(x_vals, y_vals, width=0.08, alpha=0.2, color=colors[idx % len(colors)])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        if idx == 0:
            ax.set_ylabel("Empirical Accuracy")
        ax.set_title(method.replace("_", " ").title())
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Figure 2 -> {output_path}")

def plot_f1_bootstrap_ci(
    conditions: List[str],
    f1_scores: List[float],
    ci_lowers: List[float],
    ci_uppers: List[float],
    output_path: str = "results/figures/fig3_f1_bootstrap_ci.png"
):
    """
    Figure 3. Contradiction detection F1 scores with 95% bootstrap confidence intervals.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 4.5))

    x = np.arange(len(conditions))
    y_err = [
        np.array(f1_scores) - np.array(ci_lowers),
        np.array(ci_uppers) - np.array(f1_scores)
    ]

    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    plt.bar(x, f1_scores, yerr=y_err, capsize=5, color=colors[:len(conditions)], alpha=0.85, edgecolor="black")

    clean_labels = [c.replace("_", " ").title() for c in conditions]
    plt.xticks(x, clean_labels, rotation=20, ha="right")
    plt.ylabel("Contradiction Detection F1 Score")
    plt.ylim(0, 1.05)
    plt.title("Contradiction Detection F1 across Solver Conditions (95% Bootstrap CI)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Figure 3 -> {output_path}")

def plot_sfar_by_category(
    category_df,
    output_path: str = "results/figures/fig4_sfar_by_category.png"
):
    """
    Figure 4. Solver False-Accept Rate (SFAR) breakdown by visual grounding category.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 5))

    melted = category_df.melt(id_vars=["Category"], value_vars=[c for c in category_df.columns if "sfar" in c.lower()], var_name="Condition", value_name="SFAR")
    melted["Condition"] = melted["Condition"].str.replace("sfar_", "").str.replace("_", " ").str.title()

    sns.barplot(data=melted, x="Category", y="SFAR", hue="Condition", palette="viridis")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Solver False-Accept Rate (SFAR)")
    plt.title("Solver False-Accept Rate by Grounding Category (Lower is Better)")
    plt.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Figure 4 -> {output_path}")

def plot_ece_vs_sfar(
    ece_values: List[float],
    sfar_values: List[float],
    labels: List[str],
    output_path: str = "results/figures/fig5_ece_vs_sfar.png"
):
    """
    Figure 5. Correlation between Expected Calibration Error (ECE) and SFAR.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 5))

    plt.scatter(ece_values, sfar_values, s=120, color="#1f77b4", edgecolor="black", zorder=3)
    for i, label in enumerate(labels):
        plt.annotate(
            label.replace("_", " ").title(),
            (ece_values[i], sfar_values[i]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9
        )

    # Trendline
    if len(ece_values) > 1:
        z = np.polyfit(ece_values, sfar_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(ece_values) * 0.9, max(ece_values) * 1.1, 100)
        plt.plot(x_line, p(x_line), linestyle="--", color="gray", alpha=0.7)

    plt.xlabel("Expected Calibration Error (ECE-10)")
    plt.ylabel("Solver False-Accept Rate (SFAR)")
    plt.title("Correlation: Calibration Quality vs. Solver Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Figure 5 -> {output_path}")

def plot_coverage_vs_sfar(
    tradeoff_dict: Dict[str, Any],
    output_path: str = "results/figures/fig6_coverage_vs_sfar.png"
):
    """
    Figure 6. Conformal Selective Abstention: Coverage vs. SFAR Trade-off Curves.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(7, 5))

    colors = {"Raw": "#d95f02", "Temperature": "#1b9e77", "Isotonic": "#7570b3"}
    styles = {"Raw": "--", "Temperature": "-.", "Isotonic": "-"}

    for name, df in tradeoff_dict.items():
        if df is not None and not df.empty:
            c = colors.get(name, "#333333")
            s = styles.get(name, "-")
            # Sort by coverage descending
            df_sorted = df.sort_values("coverage", ascending=False)
            plt.plot(
                df_sorted["coverage"],
                df_sorted["sfar"],
                label=f"{name}",
                color=c,
                linestyle=s,
                linewidth=2.5
            )

    plt.xlabel("Coverage (Proportion of Retained Instances)")
    plt.ylabel("Solver False Acceptance Rate (SFAR)")
    plt.title("Selective Abstention: Risk Reduction via Calibration")
    plt.xlim(0.0, 1.05)
    plt.ylim(-0.02, 1.02)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Figure 6 -> {output_path}")

def plot_multi_hypothesis_recovery(
    recovery_dict: Dict[str, Dict[str, float]],
    output_path: str = "results/figures/fig7_multi_hypothesis_recovery.png"
):
    """
    Figure 7. Multi-Hypothesis N-Best MaxSMT Recovery across Models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models = list(recovery_dict.keys())
    base_accs = [recovery_dict[m].get("base_acc", 0.0) * 100.0 for m in models]
    recovered_accs = [recovery_dict[m].get("recovered_acc", 0.0) * 100.0 for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, base_accs, width, label="Top-1 Base Accuracy", color="#6baed6", edgecolor="black")
    plt.bar(x + width/2, recovered_accs, width, label="MaxSMT Recovered Accuracy (Hard-GT)", color="#31a354", edgecolor="black")

    plt.ylabel("Accuracy (%)")
    plt.title("Multi-Hypothesis N-Best MaxSMT Error Recovery")
    plt.xticks(x, [m.upper() for m in models])
    plt.ylim(0, 105)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Figure 7 -> {output_path}")

