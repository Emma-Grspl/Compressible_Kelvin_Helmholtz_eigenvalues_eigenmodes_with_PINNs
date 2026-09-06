#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path("assets/pinn_subsonic/compare_M050_detach_vs_true_nodetach")
df = pd.read_csv(out / "comparison_metrics.csv")

def plot_metric(metric, ylabel, name, logy=True):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run, g in df.groupby("run"):
        g = g.sort_values("alpha")
        ax.plot(g["alpha"], g[metric], marker="o", label=run)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"M=0.5 — {ylabel}")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / name, dpi=220, bbox_inches="tight")
    plt.close(fig)

plot_metric("p_rel", r"$p$ relative error", "01_p_rel_vs_alpha.png")
plot_metric("q_rel", r"$q=p_y$ relative error", "02_q_rel_vs_alpha.png")
plot_metric("u_rel", r"$u$ relative error", "03_u_rel_vs_alpha.png")
plot_metric("v_rel", r"$v$ relative error", "04_v_rel_vs_alpha.png")
plot_metric("gamma_rel", r"$\gamma$ relative error", "05_gamma_rel_vs_alpha_true_nodetach.png")
plot_metric("ci_rel_err", r"$c_i$ relative error", "06_ci_rel_err_true_nodetach.png")

# Aggregate score p/q/v, excluding u because u is systematically problematic.
score_rows = []
for _, r in df.iterrows():
    score_rows.append({
        "run": r["run"],
        "alpha": r["alpha"],
        "score_pqv": (r["p_rel"] + r["q_rel"] + r["v_rel"]) / 3,
        "score_pq": (r["p_rel"] + r["q_rel"]) / 2,
        "score_all_pquv": (r["p_rel"] + r["q_rel"] + r["u_rel"] + r["v_rel"]) / 4,
    })
score = pd.DataFrame(score_rows)
score.to_csv(out / "comparison_scores.csv", index=False)

for metric, ylabel, name in [
    ("score_pq", "mean relative error p/q", "07_score_pq_vs_alpha.png"),
    ("score_pqv", "mean relative error p/q/v", "08_score_pqv_vs_alpha.png"),
    ("score_all_pquv", "mean relative error p/q/u/v", "09_score_pquv_vs_alpha.png"),
]:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run, g in score.groupby("run"):
        g = g.sort_values("alpha")
        ax.plot(g["alpha"], g[metric], marker="o", label=run)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"M=0.5 — {ylabel}")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / name, dpi=220, bbox_inches="tight")
    plt.close(fig)

# Summary PDF
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages(out / "M050_detach_vs_true_nodetach_comparison.pdf") as pdf:
    for fname in [
        "01_p_rel_vs_alpha.png",
        "02_q_rel_vs_alpha.png",
        "03_u_rel_vs_alpha.png",
        "04_v_rel_vs_alpha.png",
        "05_gamma_rel_vs_alpha_true_nodetach.png",
        "06_ci_rel_err_true_nodetach.png",
        "07_score_pq_vs_alpha.png",
        "08_score_pqv_vs_alpha.png",
        "09_score_pquv_vs_alpha.png",
    ]:
        img = plt.imread(out / fname)
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.imshow(img)
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print("[OK] wrote plots in", out)
print("[OK] wrote", out / "M050_detach_vs_true_nodetach_comparison.pdf")
print(score.to_string(index=False))
