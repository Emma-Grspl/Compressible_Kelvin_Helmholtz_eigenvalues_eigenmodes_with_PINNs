#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def norm(c: str) -> str:
    return c.strip().lower().replace(" ", "_").replace("-", "_")


def exact_col(df: pd.DataFrame, names):
    cm = {norm(c): c for c in df.columns}
    for n in names:
        if norm(n) in cm:
            return cm[norm(n)]
    return None


def infer_mach_from_path(path: Path):
    s = str(path)

    # M16 -> 1.6, M140 -> 1.40, Mach1.6 -> 1.6
    patterns = [
        r"Mach[_\- ]?([0-9]+(?:\.[0-9]+)?)",
        r"\bM[_\- ]?([0-9]+(?:\.[0-9]+)?)",
        r"\bM([0-9])([0-9])([0-9])\b",
        r"\bM([0-9])([0-9])\b",
    ]

    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if not m:
            continue

        if len(m.groups()) == 1:
            val = float(m.group(1))
            if val >= 10:
                if val < 100:
                    return val / 10.0
                return val / 100.0
            return val

        digits = "".join(m.groups())
        if len(digits) == 2:
            return float(digits) / 10.0
        if len(digits) == 3:
            return float(digits) / 100.0

    return np.nan


def load_long_blumen(path: Path):
    df = pd.read_csv(path)
    if df.empty:
        return None

    # Exact matches only. No fuzzy "M" substring matching.
    M_col = exact_col(df, ["Mach", "mach", "M"])
    a_col = exact_col(df, ["alpha", "Alpha", "a", "k"])
    ci_col = exact_col(df, ["ci", "c_i", "cimag", "c_imag", "growth_ci"])

    if a_col is None or ci_col is None:
        return None

    if M_col is not None:
        M = pd.to_numeric(df[M_col], errors="coerce")
    else:
        inferred = infer_mach_from_path(path)
        if not np.isfinite(inferred):
            return None
        M = inferred

    out = pd.DataFrame({
        "Mach": M,
        "alpha": pd.to_numeric(df[a_col], errors="coerce"),
        "ci": pd.to_numeric(df[ci_col], errors="coerce"),
        "source": str(path),
    }).dropna()

    if out.empty:
        return None

    # Hard safety: these are supersonic comparisons; Mach must overlap M >= 1.
    out = out[(out["Mach"] >= 0.95) & (out["Mach"] <= 2.1)]

    if out.empty:
        return None

    return out


def load_wide_blumen(path: Path):
    df = pd.read_csv(path)
    if df.empty:
        return None

    rows = []

    cols = list(df.columns)

    # Try pairs like alpha_M16 / ci_M16, M16_alpha / M16_ci, alpha_M1.6 / ci_M1.6.
    for c in cols:
        cn = norm(c)
        if not ("alpha" in cn or cn in {"a", "k"}):
            continue

        tag = cn
        candidates = []
        for d in cols:
            dn = norm(d)
            if d == c:
                continue
            if ("ci" in dn or "c_i" in dn or "cimag" in dn) and any(tok in dn for tok in re.findall(r"m[0-9_\.]+", tag)):
                candidates.append(d)

        if not candidates:
            continue

        for ci_c in candidates:
            combo = c + "_" + ci_c
            M = infer_mach_from_path(Path(combo))
            if not np.isfinite(M):
                continue

            tmp = pd.DataFrame({
                "Mach": M,
                "alpha": pd.to_numeric(df[c], errors="coerce"),
                "ci": pd.to_numeric(df[ci_c], errors="coerce"),
                "source": str(path),
            }).dropna()

            tmp = tmp[(tmp["Mach"] >= 0.95) & (tmp["Mach"] <= 2.1)]
            if not tmp.empty:
                rows.append(tmp)

    if not rows:
        return None

    return pd.concat(rows, ignore_index=True)


def load_blumen(paths):
    frames = []
    rejected = []

    for p in paths:
        p = Path(p)
        if not p.exists():
            continue

        out = load_long_blumen(p)
        if out is None:
            out = load_wide_blumen(p)

        if out is None or out.empty:
            rejected.append(str(p))
            continue

        frames.append(out)

    if not frames:
        print("[WARN] no usable supersonic Blumen CSV found.")
        if rejected:
            print("[INFO] rejected Blumen-like CSVs:")
            for r in rejected:
                print(" ", r)
        return pd.DataFrame(columns=["Mach", "alpha", "ci", "source"])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["Mach", "alpha", "ci"])
    out = out.sort_values(["Mach", "alpha"]).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuilt-dir", default="assets/classic_supersonic/validated_modal_points/rebuilt_aggregates_latest")
    ap.add_argument("--out-pdf", default=None)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    d = Path(args.rebuilt_dir)

    spectral = pd.read_csv(d / "supersonic_reference_core_local_spectral_REBUILT.csv")
    modal = pd.read_csv(d / "supersonic_reference_core_local_modal_REBUILT.csv")

    blumen_paths = []
    for pat in [
        "KH_RT_Blumen/**/*.csv",
        "assets/**/*blumen*.csv",
        "assets/**/*ci*digit*.csv",
        "assets/**/*ci*digital*.csv",
    ]:
        blumen_paths.extend(Path(".").glob(pat))

    blumen_paths = sorted(set(blumen_paths))
    print("[INFO] Blumen-like CSV candidates:")
    for p in blumen_paths:
        print(" ", p)

    blumen = load_blumen(blumen_paths)

    out_csv = Path(args.out_csv) if args.out_csv else d / "blumen_ci_points_FIXED.csv"
    blumen.to_csv(out_csv, index=False)
    print("[OK] wrote", out_csv)
    print("[INFO] usable Blumen points:", len(blumen))
    if not blumen.empty:
        print(blumen.groupby("Mach").size().reset_index(name="n").to_string(index=False))

    out_pdf = Path(args.out_pdf) if args.out_pdf else d / "supersonic_ci_shooting_vs_blumen_FIXED.pdf"

    machs = sorted(set(np.round(spectral["Mach"], 10)) | set(np.round(modal["Mach"], 10)))
    if not blumen.empty:
        machs = sorted(set(machs) | set(np.round(blumen["Mach"], 10)))

    with PdfPages(out_pdf) as pp:
        for M in machs:
            fig, ax = plt.subplots(figsize=(7.8, 5.0))

            s = spectral[np.isclose(spectral["Mach"], M)].sort_values("alpha")
            m = modal[np.isclose(modal["Mach"], M)].sort_values("alpha")
            b = blumen[np.isclose(blumen["Mach"], M)].sort_values("alpha")

            if not s.empty:
                ax.plot(s["alpha"], s["reference_ci"], "o-", linewidth=1.4, markersize=4, label="shooting spectral validé")

            if not m.empty:
                ax.plot(m["alpha"], m["reference_ci"], "s-", linewidth=1.2, markersize=3.5, label="shooting modal validé")

            if not b.empty:
                ax.plot(b["alpha"], b["ci"], "x--", linewidth=1.2, markersize=5, label="Blumen")

            ax.set_title(f"ci(alpha), M={M:g}")
            ax.set_xlabel("alpha")
            ax.set_ylabel("ci")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            pp.savefig(fig)
            plt.close(fig)

    print("[OK] wrote", out_pdf)


if __name__ == "__main__":
    main()
