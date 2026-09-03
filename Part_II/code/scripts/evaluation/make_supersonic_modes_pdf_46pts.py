#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def pick_col(df, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    if required:
        raise SystemExit(f"[FAIL] missing one of {candidates}\nAvailable columns:\n{list(df.columns)}")
    return None

def pick_complex_pair(df, base):
    candidates_re = [
        f"{base}_real", f"{base}_re", f"re_{base}", f"{base}r",
        f"{base}.real", f"{base}_R",
    ]
    candidates_im = [
        f"{base}_imag", f"{base}_im", f"im_{base}", f"{base}i",
        f"{base}.imag", f"{base}_I",
    ]

    re_col = pick_col(df, candidates_re, required=False)
    im_col = pick_col(df, candidates_im, required=False)

    if re_col is not None and im_col is not None:
        return re_col, im_col

    # Fallback names often used in exported modal-field CSVs.
    alt = {
        "p": [
            ("p_re", "p_im"), ("p_real", "p_imag"), ("pressure_re", "pressure_im"),
            ("p_ref_re", "p_ref_im"), ("p_modal_re", "p_modal_im"),
        ],
        "rho": [
            ("rho_re", "rho_im"), ("rho_real", "rho_imag"), ("density_re", "density_im"),
            ("rho_ref_re", "rho_ref_im"), ("rho_modal_re", "rho_modal_im"),
        ],
        "u": [
            ("u_re", "u_im"), ("u_real", "u_imag"), ("velocity_u_re", "velocity_u_im"),
            ("u_ref_re", "u_ref_im"), ("u_modal_re", "u_modal_im"),
        ],
        "v": [
            ("v_re", "v_im"), ("v_real", "v_imag"), ("velocity_v_re", "velocity_v_im"),
            ("v_ref_re", "v_ref_im"), ("v_modal_re", "v_modal_im"),
        ],
    }

    for a, b in alt.get(base, []):
        re_col = pick_col(df, [a], required=False)
        im_col = pick_col(df, [b], required=False)
        if re_col is not None and im_col is not None:
            return re_col, im_col

    raise SystemExit(
        f"[FAIL] cannot find real/imag columns for field `{base}`.\n"
        f"Available columns:\n{list(df.columns)}"
    )

def closest_rows_by_M_alpha(fields, M_col, a_col, M, alpha, tol_M=5e-7, tol_a=5e-7):
    sub = fields[
        (np.abs(fields[M_col] - M) <= tol_M)
        & (np.abs(fields[a_col] - alpha) <= tol_a)
    ].copy()

    if len(sub) == 0:
        # More permissive fallback for decimal representation mismatch.
        fields["_dist_Ma_tmp"] = (fields[M_col] - M).abs() + (fields[a_col] - alpha).abs()
        nearest = fields["_dist_Ma_tmp"].min()
        if nearest <= 5e-4:
            sub = fields[fields["_dist_Ma_tmp"] <= nearest + 1e-12].copy()
        fields.drop(columns=["_dist_Ma_tmp"], inplace=True, errors="ignore")

    return sub

def normalize_phase_by_p(y, arrays):
    p = arrays["p"]
    if len(p) == 0:
        return arrays, 1.0 + 0.0j

    k = int(np.nanargmax(np.abs(p)))
    z = p[k]
    if not np.isfinite(z.real) or not np.isfinite(z.imag) or abs(z) < 1e-14:
        return arrays, 1.0 + 0.0j

    phase = np.exp(-1j * np.angle(z))
    out = {name: val * phase for name, val in arrays.items()}
    return out, phase

def plot_one_page(pdf, point, sub, cols, normalize_phase=True, y_window=None):
    y_col = cols["y"]

    sub = sub.sort_values(y_col)
    y = sub[y_col].to_numpy(float)

    arrays = {}
    for field in ["p", "rho", "u", "v"]:
        re_col, im_col = cols[field]
        arrays[field] = (
            pd.to_numeric(sub[re_col], errors="coerce").to_numpy(float)
            + 1j * pd.to_numeric(sub[im_col], errors="coerce").to_numpy(float)
        )

    if normalize_phase:
        arrays, phase = normalize_phase_by_p(y, arrays)
    else:
        phase = 1.0 + 0.0j

    if y_window is not None:
        mask = np.abs(y) <= float(y_window)
    else:
        mask = np.ones_like(y, dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), sharex=True)
    axes = axes.ravel()

    title = (
        f"Supersonic KH mode - "
        f"M={point['Mach']:.6g}, alpha={point['alpha']:.6g}, "
        f"cr={point['cr']:.6g}, ci={point['ci']:.6g}"
    )
    fig.suptitle(title, fontsize=14, weight="bold")

    for ax, field in zip(axes, ["p", "rho", "u", "v"]):
        z = arrays[field]
        ax.plot(y[mask], z.real[mask], linewidth=1.4, label=f"Re {field}")
        ax.plot(y[mask], z.imag[mask], linewidth=1.4, linestyle="--", label=f"Im {field}")
        ax.axhline(0.0, linewidth=0.7)
        ax.set_title(field)
        ax.set_ylabel(field)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    for ax in axes[2:]:
        ax.set_xlabel("y")

    info = (
        f"point_id={point.get('point_id', '')} | "
        f"omega_i=alpha*ci={point['alpha']*point['ci']:.6g} | "
        f"phase normalization by p max: {normalize_phase} | "
        f"phase={phase.real:.4g}+{phase.imag:.4g}i | "
        f"n_y={len(y)}"
    )
    fig.text(0.02, 0.02, info, fontsize=8, family="monospace")

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--fields", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pdf-name", default="supersonic_modes_p_rho_u_v_46pts.pdf")
    ap.add_argument("--no-phase-normalize", action="store_true")
    ap.add_argument("--y-window", type=float, default=None)
    args = ap.parse_args()

    points_path = Path(args.points)
    fields_path = Path(args.fields)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not points_path.exists():
        raise SystemExit(f"[FAIL] missing points CSV: {points_path}")
    if not fields_path.exists():
        raise SystemExit(f"[FAIL] missing modal fields CSV: {fields_path}")

    points = pd.read_csv(points_path, low_memory=False)
    fields = pd.read_csv(fields_path, low_memory=False)

    # Point columns.
    pM = pick_col(points, ["Mach", "M"])
    pa = pick_col(points, ["alpha", "a"])
    pcr = pick_col(points, ["cr", "reference_cr"])
    pci = pick_col(points, ["ci", "reference_ci"])

    points = points.copy()
    points["Mach"] = pd.to_numeric(points[pM], errors="coerce")
    points["alpha"] = pd.to_numeric(points[pa], errors="coerce")
    points["cr"] = pd.to_numeric(points[pcr], errors="coerce")
    points["ci"] = pd.to_numeric(points[pci], errors="coerce")
    points = points.dropna(subset=["Mach", "alpha", "cr", "ci"]).copy()
    points = points.sort_values(["Mach", "alpha"]).reset_index(drop=True)

    # Field columns.
    fM = pick_col(fields, ["Mach", "M"])
    fa = pick_col(fields, ["alpha", "a"])
    fy = pick_col(fields, ["y", "Y"])

    fields = fields.copy()
    fields[fM] = pd.to_numeric(fields[fM], errors="coerce")
    fields[fa] = pd.to_numeric(fields[fa], errors="coerce")
    fields[fy] = pd.to_numeric(fields[fy], errors="coerce")
    fields = fields.dropna(subset=[fM, fa, fy]).copy()

    cols = {
        "Mach": fM,
        "alpha": fa,
        "y": fy,
        "p": pick_complex_pair(fields, "p"),
        "rho": pick_complex_pair(fields, "rho"),
        "u": pick_complex_pair(fields, "u"),
        "v": pick_complex_pair(fields, "v"),
    }

    print("[INFO] field columns:")
    print(cols)

    pdf_path = outdir / args.pdf_name
    summary_rows = []
    missing_rows = []

    with PdfPages(pdf_path) as pdf:
        for _, point in points.iterrows():
            M = float(point["Mach"])
            alpha = float(point["alpha"])

            sub = closest_rows_by_M_alpha(fields, fM, fa, M, alpha)

            if len(sub) == 0:
                missing_rows.append({
                    "Mach": M,
                    "alpha": alpha,
                    "cr": float(point["cr"]),
                    "ci": float(point["ci"]),
                    "reason": "no modal field rows found for this Mach/alpha",
                })
                continue

            plot_one_page(
                pdf,
                point,
                sub,
                cols,
                normalize_phase=not args.no_phase_normalize,
                y_window=args.y_window,
            )

            summary_rows.append({
                "Mach": M,
                "alpha": alpha,
                "cr": float(point["cr"]),
                "ci": float(point["ci"]),
                "omega_i": float(point["alpha"] * point["ci"]),
                "n_y": len(sub),
                "y_min": float(sub[fy].min()),
                "y_max": float(sub[fy].max()),
            })

    summary = pd.DataFrame(summary_rows)
    missing = pd.DataFrame(missing_rows)

    summary_path = outdir / "supersonic_modes_p_rho_u_v_46pts_summary.csv"
    missing_path = outdir / "supersonic_modes_p_rho_u_v_46pts_missing.csv"

    summary.to_csv(summary_path, index=False)
    missing.to_csv(missing_path, index=False)

    readme = outdir / "README_modes_pdf_46pts.md"
    readme.write_text(
        "# Supersonic modal fields PDF for sparse 46-point map\n\n"
        f"Input points: `{points_path}`\n\n"
        f"Input modal fields: `{fields_path}`\n\n"
        f"PDF: `{pdf_path}`\n\n"
        "Each page shows p, rho, u, v as complex modal profiles versus y.\n\n"
        "By default, all modes are phase-normalized so that p at its maximum-amplitude location is real positive.\n"
    )

    print("[OK] wrote PDF:", pdf_path)
    print("[OK] wrote summary:", summary_path)
    print("[OK] wrote missing:", missing_path)
    print("[OK] wrote README:", readme)
    print("")
    print("[SUMMARY]")
    print("requested points:", len(points))
    print("plotted points:", len(summary))
    print("missing points:", len(missing))
    if len(summary):
        print("\nplotted by Mach:")
        print(summary.groupby("Mach").size().to_string())
    if len(missing):
        print("\n[MISSING]")
        print(missing.to_string(index=False))

if __name__ == "__main__":
    main()
