#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def cplx(df, re, im):
    return df[re].to_numpy(float) + 1j * df[im].to_numpy(float)


def savefig(fig, out, name, pdf):
    fig.tight_layout()
    fig.savefig(out / name, dpi=220, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields_csv = run / "validated_csv_diagnostics/fields_vs_validated_csv.csv"
    zones_csv = run / "validated_csv_diagnostics/zone_errors_vs_validated_csv.csv"
    diag_csv = run / "validated_csv_diagnostics/diagnostics_vs_validated_csv.csv"

    df = pd.read_csv(fields_csv)
    z = pd.read_csv(zones_csv) if zones_csv.exists() else None
    d = pd.read_csv(diag_csv) if diag_csv.exists() else None

    y = df["y"].to_numpy(float)

    p_ref = cplx(df, "p_ref_real", "p_ref_imag")
    p_pred = cplx(df, "p_pred_aligned_real", "p_pred_aligned_imag")

    q_ref = cplx(df, "q_ref_real", "q_ref_imag")
    q_pred = cplx(df, "q_pred_aligned_real", "q_pred_aligned_imag")

    py_pred = None
    if {"p_y_pred_num_aligned_real", "p_y_pred_num_aligned_imag"}.issubset(df.columns):
        py_pred = cplx(df, "p_y_pred_num_aligned_real", "p_y_pred_num_aligned_imag")

    with PdfPages(out / "supersonic_validated_review_plots.pdf") as pdf:

        # 0. Summary text
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        lines = [
            "Supersonic PINN vs validated modal CSV",
            f"Run: {run}",
            "",
        ]
        if d is not None and len(d):
            r = d.iloc[0]
            for k in ["alpha", "Mach", "p_rel", "q_rel", "p_y_num_rel", "q_vs_dpred_rel", "align_scale_abs", "pred_max_abs_p_raw"]:
                if k in r:
                    lines.append(f"{k}: {r[k]:.6g}")
        if z is not None:
            lines += ["", "Zone errors:"]
            for _, r in z.iterrows():
                lines.append(
                    f"{r['zone']}: p_rel={r['p_rel']:.4g}, q_rel={r['q_rel']:.4g}, "
                    f"max|p_pred|={r['max_abs_p_pred']:.4g}"
                )
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
        savefig(fig, out, "00_summary.png", pdf)

        # 1. p center real/imag
        m = np.abs(y) <= 80
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y[m], p_ref.real[m], label="Re p ref")
        ax.plot(y[m], p_pred.real[m], "--", label="Re p PINN")
        ax.plot(y[m], p_ref.imag[m], label="Im p ref")
        ax.plot(y[m], p_pred.imag[m], "--", label="Im p PINN")
        ax.set_title("p(y), centre |y| <= 80")
        ax.set_xlabel("y")
        ax.set_ylabel("p aligned")
        ax.grid(True, alpha=0.3)
        ax.legend()
        savefig(fig, out, "01_p_center_real_imag.png", pdf)

        # 2. q center real/imag
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y[m], q_ref.real[m], label="Re q ref")
        ax.plot(y[m], q_pred.real[m], "--", label="Re q PINN")
        ax.plot(y[m], q_ref.imag[m], label="Im q ref")
        ax.plot(y[m], q_pred.imag[m], "--", label="Im q PINN")
        ax.set_title("q(y), centre |y| <= 80")
        ax.set_xlabel("y")
        ax.set_ylabel("q aligned")
        ax.grid(True, alpha=0.3)
        ax.legend()
        savefig(fig, out, "02_q_center_real_imag.png", pdf)

        # 3. |p| full
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(y, np.abs(p_ref) + 1e-300, label="|p_ref|")
        ax.semilogy(y, np.abs(p_pred) + 1e-300, "--", label="|p_PINN|")
        ax.axvline(-80, linestyle=":", linewidth=1)
        ax.axvline(50, linestyle=":", linewidth=1)
        ax.axvline(80, linestyle=":", linewidth=1)
        ax.set_title("|p(y)| domaine complet")
        ax.set_xlabel("y")
        ax.set_ylabel("|p|")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        savefig(fig, out, "03_abs_p_full_semilogy.png", pdf)

        # 4. |q| full
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(y, np.abs(q_ref) + 1e-300, label="|q_ref|")
        ax.semilogy(y, np.abs(q_pred) + 1e-300, "--", label="|q_PINN|")
        ax.axvline(-80, linestyle=":", linewidth=1)
        ax.axvline(50, linestyle=":", linewidth=1)
        ax.axvline(80, linestyle=":", linewidth=1)
        ax.set_title("|q(y)| domaine complet")
        ax.set_xlabel("y")
        ax.set_ylabel("|q|")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        savefig(fig, out, "04_abs_q_full_semilogy.png", pdf)

        # 5. right tail p
        m = y >= 50
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(y[m], np.abs(p_ref[m]) + 1e-300, label="|p_ref|")
        ax.semilogy(y[m], np.abs(p_pred[m]) + 1e-300, "--", label="|p_PINN|")
        ax.set_title("Queue droite, |p(y)|, y >= 50")
        ax.set_xlabel("y")
        ax.set_ylabel("|p|")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        savefig(fig, out, "05_right_tail_abs_p_semilogy.png", pdf)

        # 6. left tail p
        m = y <= -80
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(y[m], np.abs(p_ref[m]) + 1e-300, label="|p_ref|")
        ax.semilogy(y[m], np.abs(p_pred[m]) + 1e-300, "--", label="|p_PINN|")
        ax.set_title("Queue gauche, |p(y)|, y <= -80")
        ax.set_xlabel("y")
        ax.set_ylabel("|p|")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        savefig(fig, out, "06_left_tail_abs_p_semilogy.png", pdf)

        # 7. absolute error p
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(y, np.abs(p_pred - p_ref) + 1e-300, label="|p_PINN - p_ref|")
        ax.semilogy(y, np.abs(q_pred - q_ref) + 1e-300, label="|q_PINN - q_ref|")
        ax.axvline(-80, linestyle=":", linewidth=1)
        ax.axvline(50, linestyle=":", linewidth=1)
        ax.axvline(80, linestyle=":", linewidth=1)
        ax.set_title("Erreur absolue domaine complet")
        ax.set_xlabel("y")
        ax.set_ylabel("erreur absolue")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        savefig(fig, out, "07_abs_error_full_semilogy.png", pdf)

        # 8. p complex plane
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(p_ref.real, p_ref.imag, label="p ref")
        ax.plot(p_pred.real, p_pred.imag, "--", label="p PINN")
        ax.set_title("Trajectoire complexe p(y)")
        ax.set_xlabel("Re p")
        ax.set_ylabel("Im p")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis("equal")
        savefig(fig, out, "08_p_complex_plane.png", pdf)

        # 9. zone errors
        if z is not None:
            keep = [
                "center_|y|<=50",
                "diagnostic_|y|<=80",
                "left_y<-80",
                "right_y>80",
                "outside_|y|>80",
                "full_domain",
            ]
            zz = z[z["zone"].isin(keep)].copy()
            x = np.arange(len(zz))
            width = 0.38

            fig, ax = plt.subplots(figsize=(11, 5))
            ax.bar(x - width / 2, zz["p_rel"], width, label="p_rel")
            ax.bar(x + width / 2, zz["q_rel"], width, label="q_rel")
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(zz["zone"], rotation=30, ha="right")
            ax.set_title("Erreurs relatives par zone")
            ax.set_ylabel("relative L2")
            ax.grid(True, which="both", axis="y", alpha=0.3)
            ax.legend()
            savefig(fig, out, "09_zone_errors.png", pdf)

        # 10. q vs numerical p_y if available
        if py_pred is not None:
            m = np.abs(y) <= 80
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y[m], q_pred.real[m], label="Re q PINN")
            ax.plot(y[m], py_pred.real[m], "--", label="Re p_y num")
            ax.plot(y[m], q_pred.imag[m], label="Im q PINN")
            ax.plot(y[m], py_pred.imag[m], "--", label="Im p_y num")
            ax.set_title("Compatibilité q ≈ p_y, centre |y| <= 80")
            ax.set_xlabel("y")
            ax.set_ylabel("aligned")
            ax.grid(True, alpha=0.3)
            ax.legend()
            savefig(fig, out, "10_q_vs_py_center.png", pdf)

    print("[OK] wrote plots to", out)
    print("[OK] PDF:", out / "supersonic_validated_review_plots.pdf")


if __name__ == "__main__":
    main()
