#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BUNDLE = Path("assets/pinn_subsonic/subsonic_2d_review_bundle_velcurv1e4_INJECTED")
OUTPDF = BUNDLE / "subsonic_2d_review_velcurv1e4_INJECTED.pdf"
BUNDLE.mkdir(parents=True, exist_ok=True)

metrics = pd.DataFrame([
    {"alpha":0.3, "Mach":0.5, "ci_ref":0.446683, "ci_pred":0.446683, "p_rel":0.135129, "q_rel":0.310857, "rho_rel":0.135129, "u_rel":0.312260, "v_rel":0.305199, "gamma_rel":0.427655},
    {"alpha":0.5, "Mach":0.5, "ci_ref":0.267334, "ci_pred":0.267334, "p_rel":0.019103, "q_rel":0.054627, "rho_rel":0.019103, "u_rel":0.411024, "v_rel":0.073402, "gamma_rel":0.077269},
    {"alpha":0.7, "Mach":0.5, "ci_ref":0.114183, "ci_pred":0.114183, "p_rel":0.046361, "q_rel":0.078446, "rho_rel":0.046361, "u_rel":0.341885, "v_rel":0.085010, "gamma_rel":0.114208},
])

metrics.to_csv(BUNDLE / "subsonic_2d_review_metrics.csv", index=False)

def add_text_page(pdf, title, text, fontsize=10):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.02, 0.96, title, fontsize=16, weight="bold", va="top")
    ax.text(0.02, 0.88, text, fontsize=fontsize, family="monospace", va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def add_table_page(pdf, title, df):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, fontsize=16, weight="bold", pad=20)

    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind in "fc":
            d[c] = d[c].map(lambda x: f"{x:.6g}")

    table = ax.table(
        cellText=d.values,
        colLabels=d.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)

    for _, cell in table.get_celld().items():
        cell.set_linewidth(0.3)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def add_bar_page(pdf, title, df):
    plot_df = df[["alpha", "p_rel", "q_rel", "u_rel", "v_rel", "gamma_rel"]].copy()
    x = range(len(plot_df))
    width = 0.15

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fields = ["p_rel", "q_rel", "u_rel", "v_rel", "gamma_rel"]

    for i, field in enumerate(fields):
        ax.bar([xx + (i-2)*width for xx in x], plot_df[field], width=width, label=field)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"alpha={a:g}" for a in plot_df["alpha"]])
    ax.set_ylabel("relative error")
    ax.set_title(title, fontsize=16, weight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def add_image_page(pdf, image_path):
    try:
        img = plt.imread(image_path)
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(image_path.name, fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

with PdfPages(OUTPDF) as pdf:
    add_text_page(
        pdf,
        "Subsonic 2D PINN review - M=0.5",
        "Run:\n"
        "assets/pinn_subsonic/mini2d_pq_firstorder_M050_a030_a070_bootp_ucore005_velcurv1e4_INJECTED_no_modal_anchor\n\n"
        "Training setup:\n"
        "- p/q first-order PINN\n"
        "- pressure-only warm start\n"
        "- sparse scalar ci anchors only at alpha=0.3,0.5,0.7\n"
        "- no modal field anchors\n"
        "- detach_ci_in_mode_branch=False\n\n"
        "Important conclusion:\n"
        "The injected velocity-curvature/core regularization run completes, but final diagnostics are numerically identical to previous smooth005/smooth02 runs.\n"
    )

    add_table_page(pdf, "Final diagnostics", metrics)
    add_bar_page(pdf, "Relative errors by alpha", metrics)

    plot_files = sorted((BUNDLE / "plots").glob("*.png"))
    for p in plot_files:
        add_image_page(pdf, p)

print("[OK] wrote", OUTPDF)
print("[OK] wrote", BUNDLE / "subsonic_2d_review_metrics.csv")
