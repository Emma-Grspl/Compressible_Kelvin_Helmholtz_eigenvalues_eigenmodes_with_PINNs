#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path("assets/pinn_subsonic/compare_M050_all_methods")
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "pressure_stage1q": "assets/pinn_subsonic/stage1quater_pressure_path1500_xlim15/diagnostics_summary.csv",
    "pq_detach_old": "assets/pinn_subsonic/permach_pq_bootp_M050_a010070_cilearned_nodetach/diagnostics_summary.csv",
    "pq_true_nodetach": "assets/pinn_subsonic/permach_pq_bootp_M050_a010070_TRUE_cilearned_nodetach/diagnostics_summary.csv",
    "mini2d_pq_discrete": "assets/pinn_subsonic/mini2d_pq_firstorder_M050_a030_a070_bootp_discrete/diagnostics_summary.csv",
    "hybrid_switch_a040": "assets/pinn_subsonic/hybrid_pressure_pq_M050_a030_a070_switch_a040/diagnostics_summary.csv",
    "hybrid_oracle": "assets/pinn_subsonic/hybrid_pressure_pq_M050_a030_a070_switch_a040/diagnostics_summary_oracle.csv",
}

def first_existing_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

rows = []
for name, f in SOURCES.items():
    p = Path(f)
    if not p.exists():
        print("[missing]", name, f)
        continue

    df = pd.read_csv(p)

    alpha_col = first_existing_col(df, ["alpha", "Alpha"])
    mach_col = first_existing_col(df, ["Mach", "mach", "M"])

    if alpha_col is None:
        print("[skip no alpha]", name, list(df.columns))
        continue

    if mach_col is not None:
        df = df[np.isclose(df[mach_col].astype(float), 0.5, atol=1e-9)].copy()

    for _, r in df.iterrows():
        alpha = float(r[alpha_col])
        if not any(np.isclose(alpha, a, atol=5e-4) for a in [0.10, 0.30, 0.50, 0.55, 0.70, 0.80]):
            continue

        row = {"method": name, "alpha": alpha}
        for col in [
            "p_rel", "q_rel", "p_y_rel", "rho_rel", "u_rel", "v_rel",
            "gamma_rel", "ci_rel_err", "ci_abs_err", "ci_pred", "ci_ref",
        ]:
            row[col] = float(r[col]) if col in df.columns and pd.notna(r[col]) else np.nan

        if np.isnan(row["q_rel"]) and not np.isnan(row["p_y_rel"]):
            row["q_rel"] = row["p_y_rel"]

        rows.append(row)

comp = pd.DataFrame(rows).sort_values(["alpha", "method"]).reset_index(drop=True)
comp.to_csv(OUT / "comparison_all_methods.csv", index=False)

print(comp.to_string(index=False))
print("[OK] wrote", OUT / "comparison_all_methods.csv")

# Scores.
score = comp.copy()
score["score_pq"] = score[["p_rel", "q_rel"]].mean(axis=1, skipna=True)
score["score_pqv"] = score[["p_rel", "q_rel", "v_rel"]].mean(axis=1, skipna=True)
score["score_pquv"] = score[["p_rel", "q_rel", "u_rel", "v_rel"]].mean(axis=1, skipna=True)
score.to_csv(OUT / "comparison_scores_all_methods.csv", index=False)

def plot_metric(metric, ylabel, fname, logy=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, g in comp.groupby("method"):
        g = g.sort_values("alpha")
        if metric not in g.columns or g[metric].notna().sum() == 0:
            continue
        ax.plot(g["alpha"], g[metric], marker="o", label=method)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"M=0.5 — {ylabel}")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)

for metric, ylabel, fname in [
    ("p_rel", "p relative error", "01_p_rel_all_methods.png"),
    ("q_rel", "q relative error", "02_q_rel_all_methods.png"),
    ("u_rel", "u relative error", "03_u_rel_all_methods.png"),
    ("v_rel", "v relative error", "04_v_rel_all_methods.png"),
    ("gamma_rel", "gamma relative error", "05_gamma_rel_all_methods.png"),
    ("ci_rel_err", "ci relative error", "06_ci_rel_err_all_methods.png"),
]:
    plot_metric(metric, ylabel, fname)

for metric, ylabel, fname in [
    ("score_pq", "mean relative error p/q", "07_score_pq_all_methods.png"),
    ("score_pqv", "mean relative error p/q/v", "08_score_pqv_all_methods.png"),
    ("score_pquv", "mean relative error p/q/u/v", "09_score_pquv_all_methods.png"),
]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, g in score.groupby("method"):
        g = g.sort_values("alpha")
        if g[metric].notna().sum() == 0:
            continue
        ax.plot(g["alpha"], g[metric], marker="o", label=method)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"M=0.5 — {ylabel}")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)

with PdfPages(OUT / "M050_all_methods_comparison.pdf") as pdf:
    for fname in [
        "01_p_rel_all_methods.png",
        "02_q_rel_all_methods.png",
        "03_u_rel_all_methods.png",
        "04_v_rel_all_methods.png",
        "05_gamma_rel_all_methods.png",
        "06_ci_rel_err_all_methods.png",
        "07_score_pq_all_methods.png",
        "08_score_pqv_all_methods.png",
        "09_score_pquv_all_methods.png",
    ]:
        p = OUT / fname
        if not p.exists():
            continue
        img = plt.imread(p)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.imshow(img)
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print("[OK] wrote plots in", OUT)
print("[OK] wrote", OUT / "M050_all_methods_comparison.pdf")
print(score[["method", "alpha", "score_pq", "score_pqv", "score_pquv"]].to_string(index=False))
