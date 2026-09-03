#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_inputs(blumen_csv: Path, classical_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    blumen = pd.read_csv(blumen_csv)
    classical = pd.read_csv(classical_csv)

    for df in (blumen, classical):
        for col in ["Mach", "alpha"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    if "target_ci" in blumen.columns:
        blumen["target_ci"] = pd.to_numeric(blumen["target_ci"], errors="coerce")

    for col in [
        "target_ci",
        "alpha_blumen",
        "alpha_classical",
        "delta_alpha",
        "delta_ci_at_blumen_alpha",
        "classical_cr",
        "classical_ci",
        "residual_norm",
    ]:
        if col in classical.columns:
            classical[col] = pd.to_numeric(classical[col], errors="coerce")

    return blumen, classical


def prepare_blumen_for_overlay(blumen: pd.DataFrame) -> pd.DataFrame:
    # si le CSV original n'a pas encore la colonne target_ci,
    # on suppose qu'il vient déjà du fichier enrichi final.
    # sinon on ne fait rien.
    needed = {"Mach", "alpha", "target_ci"}
    missing = needed - set(blumen.columns)
    if missing:
        raise ValueError(
            f"Le fichier Blumen doit contenir {needed}; manquant: {missing}"
        )

    out = blumen.dropna(subset=["Mach", "alpha", "target_ci"]).copy()
    out = out.sort_values(["target_ci", "Mach", "alpha"])
    return out


def make_color_values(levels: List[float]) -> dict[float, tuple]:
    cmap = plt.get_cmap("viridis")
    if len(levels) == 1:
        return {levels[0]: cmap(0.5)}
    vals = np.linspace(0.05, 0.95, len(levels))
    return {lev: cmap(v) for lev, v in zip(levels, vals)}


def plot_overlay(blumen: pd.DataFrame, classical: pd.DataFrame, output_pdf: Path, output_png: Path):
    levels = sorted(
        x for x in classical["target_ci"].dropna().unique().tolist()
        if np.isfinite(x)
    )
    colors = make_color_values(levels)

    fig, ax = plt.subplots(figsize=(8.0, 10.0))

    # points digitalisés Blumen
    for level in levels:
        sub_b = blumen.loc[np.isclose(blumen["target_ci"], level, rtol=0, atol=1e-12)].copy()
        if len(sub_b) == 0:
            continue
        sub_b = sub_b.sort_values(["Mach", "alpha"])
        ax.scatter(
            sub_b["alpha"],
            sub_b["Mach"],
            s=8,
            alpha=0.5,
            color=colors[level],
            label=f"Blumen ci={level:.4g}" if level == levels[0] else None,
        )

    # isolignes classiques reconstruites
    for level in levels:
        sub = classical.loc[
            np.isclose(classical["target_ci"], level, rtol=0, atol=1e-12)
            & classical["status"].astype(str).str.startswith("converged")
        ].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values("Mach")
        ax.plot(
            sub["alpha_classical"],
            sub["Mach"],
            "-",
            lw=1.6,
            color=colors[level],
        )

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$M$")
    ax.set_title("Isolignes classiques reconstruites aux niveaux de Blumen\n(plan alpha–Mach)")
    ax.grid(True, alpha=0.25)

    # légende compacte
    txt = (
        f"n curves: {classical['curve_key'].nunique()} | "
        f"n converged points: "
        f"{classical['status'].astype(str).str.startswith('converged').sum()}"
    )
    ax.text(
        0.01,
        0.01,
        txt,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def plot_delta_alpha_heatmap(classical: pd.DataFrame, output_pdf: Path, output_png: Path):
    sub = classical.loc[
        classical["status"].astype(str).str.startswith("converged")
        & np.isfinite(classical["alpha_blumen"])
        & np.isfinite(classical["Mach"])
        & np.isfinite(classical["delta_alpha"])
    ].copy()

    fig, ax = plt.subplots(figsize=(8.0, 10.0))
    sc = ax.scatter(
        sub["alpha_blumen"],
        sub["Mach"],
        c=sub["delta_alpha"],
        s=18,
        cmap="coolwarm",
        vmin=-np.nanmax(np.abs(sub["delta_alpha"])),
        vmax= np.nanmax(np.abs(sub["delta_alpha"])),
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\Delta\alpha = \alpha_{\mathrm{classique}} - \alpha_{\mathrm{Blumen}}$")

    ax.set_xlabel(r"$\alpha_{\mathrm{Blumen}}$")
    ax.set_ylabel(r"$M$")
    ax.set_title("Erreur géométrique entre Blumen et les vraies isolignes classiques")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blumen-csv", type=Path, required=True)
    parser.add_argument("--classical-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    blumen_raw, classical = load_inputs(args.blumen_csv, args.classical_csv)
    blumen = prepare_blumen_for_overlay(blumen_raw)

    merged_csv = outdir / "blumen_classical_pointwise_comparison.csv"
    classical.to_csv(merged_csv, index=False)

    overlay_pdf = outdir / "blumen_true_classical_isolines_overlay_alpha_mach.pdf"
    overlay_png = outdir / "blumen_true_classical_isolines_overlay_alpha_mach.png"

    heat_pdf = outdir / "blumen_true_classical_delta_alpha_heatmap.pdf"
    heat_png = outdir / "blumen_true_classical_delta_alpha_heatmap.png"

    plot_overlay(blumen, classical, overlay_pdf, overlay_png)
    plot_delta_alpha_heatmap(classical, heat_pdf, heat_png)

    metadata = {
        "status": "PASS",
        "blumen_csv": str(args.blumen_csv),
        "classical_csv": str(args.classical_csv),
        "overlay_pdf": str(overlay_pdf),
        "heatmap_pdf": str(heat_pdf),
        "merged_csv": str(merged_csv),
        "n_blumen_rows": int(len(blumen)),
        "n_classical_rows": int(len(classical)),
        "n_converged": int(classical["status"].astype(str).str.startswith("converged").sum()),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("=== BLUMEN TRUE-ISOLINE ASSETS ===")
    print("Wrote:", overlay_pdf)
    print("Wrote:", heat_pdf)
    print("Wrote:", merged_csv)
    print("Wrote:", outdir / "metadata.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
