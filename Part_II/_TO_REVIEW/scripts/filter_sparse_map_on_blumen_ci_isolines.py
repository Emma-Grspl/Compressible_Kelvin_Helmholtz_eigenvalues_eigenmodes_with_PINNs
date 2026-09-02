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

    # Keep only positive numeric ci labels: 0.01, 0.03, 0.05, 0.07, 0.1, etc.
    if re.fullmatch(r"[0-9]*\.?[0-9]+", s):
        v = float(s)
        return v if v > 0 else None

    # Optional support for labels like ci=0.05
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

        # Row 0 usually contains X/Y text.
        x = pd.to_numeric(raw[x_col].iloc[1:], errors="coerce")
        y = pd.to_numeric(raw[y_col].iloc[1:], errors="coerce")

        df = pd.DataFrame({"Mach": x, "alpha": y}).dropna()
        df = df[(df["Mach"] > 0.5) & (df["Mach"] < 2.5) & (df["alpha"] > -0.05) & (df["alpha"] < 0.7)]
        df = df.drop_duplicates(["Mach", "alpha"]).reset_index(drop=True)

        if len(df) >= 2:
            curves.append({
                "ci_isoline": ci_value,
                "label": f"ci={ci_value:g}",
                "points": df[["Mach", "alpha"]].to_numpy(float),
                "dataframe": df,
            })

    if not curves:
        raise SystemExit(f"[FAIL] no positive ci isolines found in {path}")

    curves = sorted(curves, key=lambda d: d["ci_isoline"])
    return curves

def point_segment_distance_scaled(P, A, B, scale_M, scale_alpha):
    p = np.array([P[0] / scale_M, P[1] / scale_alpha], dtype=float)
    a = np.array([A[0] / scale_M, A[1] / scale_alpha], dtype=float)
    b = np.array([B[0] / scale_M, B[1] / scale_alpha], dtype=float)

    ab = b - a
    den = float(np.dot(ab, ab))
    if den <= 0:
        t = 0.0
    else:
        t = float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0))

    q = a + t * ab
    d = float(np.linalg.norm(p - q))
    q_orig = np.array([q[0] * scale_M, q[1] * scale_alpha], dtype=float)
    return d, q_orig[0], q_orig[1]

def point_curve_distance_scaled(M, alpha, curve_points, scale_M, scale_alpha):
    best_d = np.inf
    best_M = np.nan
    best_alpha = np.nan

    for k in range(len(curve_points) - 1):
        d, qM, qa = point_segment_distance_scaled(
            (M, alpha),
            curve_points[k],
            curve_points[k + 1],
            scale_M,
            scale_alpha,
        )
        if d < best_d:
            best_d = d
            best_M = qM
            best_alpha = qa

    return best_d, best_M, best_alpha

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--ci-datasets", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tol-ci", type=float, default=0.012)
    ap.add_argument("--tol-scaled", type=float, default=1.0)
    ap.add_argument("--scale-M", type=float, default=0.05)
    ap.add_argument("--scale-alpha", type=float, default=0.025)
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

    curves = load_blumen_ci_isolines(ci_path)

    rows = []
    for idx, row in pts.iterrows():
        M = float(row[M_col])
        alpha = float(row[a_col])
        ci = float(row[ci_col])

        best_any = None
        best_eligible = None

        for curve in curves:
            iso_ci = float(curve["ci_isoline"])
            d_scaled, near_M, near_alpha = point_curve_distance_scaled(
                M,
                alpha,
                curve["points"],
                args.scale_M,
                args.scale_alpha,
            )
            ci_abs_err = abs(ci - iso_ci)

            # Combined score only for ranking; hard keep uses both tolerances.
            score = np.sqrt((ci_abs_err / max(args.tol_ci, 1e-12))**2 + (d_scaled / max(args.tol_scaled, 1e-12))**2)

            candidate = {
                "matched_ci_isoline": iso_ci,
                "ci_abs_err_to_isoline": ci_abs_err,
                "scaled_distance_to_isoline": d_scaled,
                "nearest_isoline_M": near_M,
                "nearest_isoline_alpha": near_alpha,
                "match_score": score,
            }

            if best_any is None or candidate["match_score"] < best_any["match_score"]:
                best_any = candidate

            if ci_abs_err <= args.tol_ci and d_scaled <= args.tol_scaled:
                if best_eligible is None or candidate["match_score"] < best_eligible["match_score"]:
                    best_eligible = candidate

        out = row.to_dict()
        out["original_row_index"] = idx
        out["tol_ci"] = args.tol_ci
        out["tol_scaled"] = args.tol_scaled
        out["scale_M"] = args.scale_M
        out["scale_alpha"] = args.scale_alpha

        if best_eligible is not None:
            out.update(best_eligible)
            out["keep_on_blumen_ci_isoline"] = True
            out["filter_reason"] = "kept: ci close to isoline and point close to isoline in (M,alpha)"
        else:
            out.update(best_any)
            out["keep_on_blumen_ci_isoline"] = False
            out["filter_reason"] = "rejected: no ci isoline satisfies both tolerances"

        rows.append(out)

    all_df = pd.DataFrame(rows)
    all_df = all_df.sort_values([M_col, a_col]).reset_index(drop=True)

    kept = all_df[all_df["keep_on_blumen_ci_isoline"]].copy()
    rejected = all_df[~all_df["keep_on_blumen_ci_isoline"]].copy()

    all_out = outdir / "supersonic_sparse_map_ALL_with_blumen_ci_isoline_distance.csv"
    kept_out = outdir / "supersonic_sparse_map_ON_BLUMEN_CI_ISOLINES.csv"
    rejected_out = outdir / "supersonic_sparse_map_OFF_BLUMEN_CI_ISOLINES.csv"

    all_df.to_csv(all_out, index=False)
    kept.to_csv(kept_out, index=False)
    rejected.to_csv(rejected_out, index=False)

    # Plot.
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    for curve in curves:
        dfc = curve["dataframe"]
        ax.plot(dfc["Mach"], dfc["alpha"], linewidth=1.6, label=curve["label"])

        # Label near last valid point.
        last = dfc.iloc[-1]
        ax.text(last["Mach"], last["alpha"], curve["label"], fontsize=8)

    if len(rejected):
        ax.scatter(
            rejected[M_col],
            rejected[a_col],
            s=55,
            marker="x",
            label=f"rejected points ({len(rejected)})",
        )

    if len(kept):
        sc = ax.scatter(
            kept[M_col],
            kept[a_col],
            c=kept[ci_col],
            s=70,
            edgecolors="black",
            linewidths=0.5,
            label=f"kept sparse points ({len(kept)})",
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("point ci")

    ax.set_xlabel("Mach M")
    ax.set_ylabel("alpha")
    ax.set_title("Supersonic sparse PINN points filtered on digitized Blumen ci isolines")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(1.05, 1.95)
    ax.set_ylim(0.0, 0.42)

    plot_pdf = outdir / "blumen_ci_isolines_overlay_sparse_points.pdf"
    plot_png = outdir / "blumen_ci_isolines_overlay_sparse_points.png"
    fig.tight_layout()
    fig.savefig(plot_pdf)
    fig.savefig(plot_png, dpi=220)
    plt.close(fig)

    readme = outdir / "README_blumen_ci_isoline_filter.md"
    readme.write_text(
        "# Blumen ci-isoline filtering of supersonic sparse PINN map\n\n"
        f"Input points: `{points_path}`\n\n"
        f"Input Blumen ci digitization: `{ci_path}`\n\n"
        f"Tolerances:\n"
        f"- tol_ci = {args.tol_ci}\n"
        f"- tol_scaled = {args.tol_scaled}\n"
        f"- scale_M = {args.scale_M}\n"
        f"- scale_alpha = {args.scale_alpha}\n\n"
        "A point is kept if at least one digitized positive-ci Blumen isoline satisfies:\n\n"
        "`abs(ci_point - ci_isoline) <= tol_ci`\n\n"
        "and\n\n"
        "`scaled_distance((M,alpha), isoline) <= tol_scaled`.\n\n"
        "Main output:\n"
        "- `supersonic_sparse_map_ON_BLUMEN_CI_ISOLINES.csv`\n"
        "- `supersonic_sparse_map_ALL_with_blumen_ci_isoline_distance.csv`\n"
        "- `blumen_ci_isolines_overlay_sparse_points.pdf`\n"
    )

    print("[OK] parsed ci isolines:", [c["ci_isoline"] for c in curves])
    print("[OK] total points:", len(all_df))
    print("[OK] kept:", len(kept))
    print("[OK] rejected:", len(rejected))
    print("")
    print("[COUNTS kept by Mach]")
    if len(kept):
        print(kept.groupby(M_col).size().to_string())
    else:
        print("none")
    print("")
    print("[COUNTS kept by matched ci isoline]")
    if len(kept):
        print(kept.groupby("matched_ci_isoline").size().to_string())
    else:
        print("none")
    print("")
    print("[WRITTEN]")
    print(all_out)
    print(kept_out)
    print(rejected_out)
    print(plot_pdf)
    print(plot_png)
    print(readme)

if __name__ == "__main__":
    main()
