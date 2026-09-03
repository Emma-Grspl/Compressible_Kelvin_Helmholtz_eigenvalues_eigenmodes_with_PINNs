#!/usr/bin/env python3
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def pick_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    raise SystemExit(f"[FAIL] missing one of columns {names}; available={list(df.columns)}")

def parse_ci_label(label):
    s = str(label).strip()

    # Ex: "0.01", "0.03", ...
    if re.fullmatch(r"[0-9]*\.?[0-9]+", s):
        v = float(s)
        return v if v > 0 else None

    # Ex: "ci=0.03"
    m = re.fullmatch(r"ci\s*=\s*([0-9]*\.?[0-9]+)", s)
    if m:
        v = float(m.group(1))
        return v if v > 0 else None

    return None

def load_blumen_ci_isolines(path):
    raw = pd.read_csv(path, low_memory=False)
    cols = list(raw.columns)

    curves = []
    for j in range(0, len(cols) - 1, 2):
        label_col = cols[j]
        x_col = cols[j]
        y_col = cols[j + 1]

        ci_value = parse_ci_label(label_col)
        if ci_value is None:
            continue

        x = pd.to_numeric(raw[x_col].iloc[1:], errors="coerce")
        y = pd.to_numeric(raw[y_col].iloc[1:], errors="coerce")

        df = pd.DataFrame({"Mach": x, "alpha": y}).dropna()
        df = df[(df["Mach"] > 0.5) & (df["Mach"] < 3.0) & (df["alpha"] > -0.05) & (df["alpha"] < 0.8)]
        df = df.drop_duplicates(["Mach", "alpha"]).reset_index(drop=True)

        if len(df) >= 2:
            curves.append({
                "ci": ci_value,
                "label": f"ci={ci_value:g}",
                "df": df,
            })

    if not curves:
        raise SystemExit(f"[FAIL] no positive ci isolines found in {path}")

    curves = sorted(curves, key=lambda d: d["ci"])
    return curves

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--ci-datasets", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    points_path = Path(args.points)
    ci_path = Path(args.ci_datasets)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not points_path.exists():
        raise SystemExit(f"[FAIL] missing points CSV: {points_path}")
    if not ci_path.exists():
        raise SystemExit(f"[FAIL] missing Blumen ci dataset: {ci_path}")

    pts = pd.read_csv(points_path, low_memory=False)
    M_col = pick_col(pts, ["Mach", "M"])
    a_col = pick_col(pts, ["alpha", "a"])
    ci_col = pick_col(pts, ["ci", "reference_ci"])

    pts[M_col] = pd.to_numeric(pts[M_col], errors="coerce")
    pts[a_col] = pd.to_numeric(pts[a_col], errors="coerce")
    pts[ci_col] = pd.to_numeric(pts[ci_col], errors="coerce")
    pts = pts.dropna(subset=[M_col, a_col, ci_col]).copy()
    pts = pts.sort_values([M_col, a_col]).reset_index(drop=True)

    curves = load_blumen_ci_isolines(ci_path)

    fig, ax = plt.subplots(figsize=(10.5, 7.2))

    # Plot isolines
    for curve in curves:
        dfc = curve["df"]
        ax.plot(dfc["Mach"], dfc["alpha"], linewidth=2.0, label=curve["label"])

        # place label near the rightmost part of each curve
        ir = dfc["Mach"].idxmax()
        row = dfc.loc[ir]
        ax.text(row["Mach"] + 0.01, row["alpha"], curve["label"], fontsize=9)

    # Plot ALL points (no rejection)
    sc = ax.scatter(
        pts[M_col],
        pts[a_col],
        c=pts[ci_col],
        s=110,
        edgecolors="black",
        linewidths=0.7,
        alpha=0.95,
        zorder=5,
        label=f"all sparse points ({len(pts)})",
    )

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("point ci")

    # generous axis limits to see M=1.8 and M=1.9 clearly
    xmin = min(1.05, float(pts[M_col].min()) - 0.03)
    xmax = max(1.95, float(pts[M_col].max()) + 0.03)
    ymin = min(0.00, float(pts[a_col].min()) - 0.01)
    ymax = max(0.42, float(pts[a_col].max()) + 0.02)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Mach M")
    ax.set_ylabel("alpha")
    ax.set_title("Supersonic sparse PINN points over digitized Blumen ci isolines")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()

    pdf_path = outdir / "blumen_ci_isolines_overlay_ALL_sparse_points.pdf"
    png_path = outdir / "blumen_ci_isolines_overlay_ALL_sparse_points.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    readme = outdir / "README_overlay_all_points.md"
    readme.write_text(
        "# Overlay of all sparse supersonic points on Blumen ci isolines\n\n"
        f"Points CSV: `{points_path}`\n\n"
        f"Blumen ci digitization: `{ci_path}`\n\n"
        "This figure keeps **all points** and does **no rejection/filtering**.\n"
    )

    print("[OK] total points plotted:", len(pts))
    print("[OK] Mach range in points:", float(pts[M_col].min()), "->", float(pts[M_col].max()))
    print("[OK] alpha range in points:", float(pts[a_col].min()), "->", float(pts[a_col].max()))
    print("[OK] wrote:", pdf_path)
    print("[OK] wrote:", png_path)
    print("[OK] wrote:", readme)

if __name__ == "__main__":
    main()
