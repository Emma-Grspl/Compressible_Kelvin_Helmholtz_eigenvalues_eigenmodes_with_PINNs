#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bool_like(x) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"1", "true", "yes", "y", "ok"}


def apply_optional_bool_filter(mask, df, col, enabled=True):
    """
    Apply a boolean-like filter only if the column is populated
    on the current subset. If the column is all NaN, skip it.
    This is important for v2 campaigns where validation flags may be
    stored in other metadata columns.
    """
    if not enabled:
        return mask

    if col not in df.columns:
        return mask

    vals = df.loc[mask, col]

    if vals.notna().sum() == 0:
        print(f"[warn] skipping filter {col}: all values are NaN on current subset")
        return mask

    parsed = df[col].map(bool_like)
    new_mask = mask & parsed

    print(
        f"[info] filter {col}: "
        f"{int(mask.sum())} -> {int(new_mask.sum())} rows"
    )

    return new_mask


def find_col(columns, candidates, required=True, label="column"):
    cols = list(columns)
    lower = {c.lower(): c for c in cols}

    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]

    compact = {
        c.lower().replace("_", "").replace("-", "").replace(" ", ""): c
        for c in cols
    }

    for cand in candidates:
        key = cand.lower().replace("_", "").replace("-", "").replace(" ", "")
        if key in compact:
            return compact[key]

    if required:
        raise RuntimeError(
            f"Could not find required {label}. Tried {candidates}. "
            f"Available columns:\n{cols}"
        )

    return None


def mach_tag(mach: float) -> str:
    return f"M{int(round(100 * mach)):03d}"


def finite_complex_gradient(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Compute q = dp/dy on a 1D grid.

    Robust to duplicated y values:
    - sort by y;
    - merge duplicate y values by averaging p;
    - compute dp/dy on the unique y grid;
    - map the derivative back to the original rows.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=complex)

    if len(y) < 2:
        raise RuntimeError("Need at least two y-points to compute q=dp/dy.")

    if not np.isfinite(y).all() or not np.isfinite(p.real).all() or not np.isfinite(p.imag).all():
        raise RuntimeError("Non-finite y or p values in finite_complex_gradient.")

    order = np.argsort(y)
    y_sorted = y[order]
    p_sorted = p[order]

    unique_y, inverse, counts = np.unique(
        y_sorted,
        return_inverse=True,
        return_counts=True,
    )

    if len(unique_y) < 2:
        raise RuntimeError("Need at least two unique y-points to compute q=dp/dy.")

    if np.any(counts > 1):
        p_unique = np.zeros(len(unique_y), dtype=complex)
        np.add.at(p_unique, inverse, p_sorted)
        p_unique = p_unique / counts
    else:
        p_unique = p_sorted

    dy = np.diff(unique_y)
    if np.any(dy <= 0):
        raise RuntimeError("Unique y grid is not strictly increasing.")

    edge_order = 2 if len(unique_y) >= 3 else 1
    q_unique = np.gradient(p_unique, unique_y, edge_order=edge_order)

    q_sorted = q_unique[inverse]

    q = np.empty_like(q_sorted)
    q[order] = q_sorted

    return q


def complex_alignment_factor(p_ref: np.ndarray, p_pred: np.ndarray) -> complex:
    """
    Returns a minimizing ||a p_pred - p_ref||_2.
    Useful as diagnostic. Not used to alter the dataset by default.
    """
    denom = np.vdot(p_pred, p_pred)
    if abs(denom) < 1e-30:
        return 1.0 + 0.0j
    return np.vdot(p_pred, p_ref) / denom


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare fixed-M supersonic PINN dataset with spectral anchors "
            "cr/ci and modal pressure anchors p,q where q=dp/dy."
        )
    )

    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--mach", type=float, default=1.80)
    parser.add_argument(
        "--field-version",
        choices=["raw_confirmed", "tail_polished"],
        default="raw_confirmed",
    )
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--mach-tol", type=float, default=1e-10)
    parser.add_argument("--alpha-min", type=float, default=None)
    parser.add_argument("--alpha-max", type=float, default=None)
    parser.add_argument(
        "--normalize-gauge",
        action="store_true",
        help=(
            "Normalize each mode so that p(y closest to 0) is real positive and "
            "approximately equal to 1. Default: keep reference gauge unchanged."
        ),
    )
    parser.add_argument(
        "--no-trusted-filter",
        action="store_true",
        help="Do not filter with trusted_spectral/trusted_modal flags.",
    )
    parser.add_argument("--out", type=Path, default=None)

    args = parser.parse_args()

    repo = args.repo.resolve()
    freeze = repo / "assets/classic_supersonic/supersonic_sparse_PINN_reference_v2_FINAL_FREEZE"
    data_dir = freeze / "data"

    spectral_csv = data_dir / "supersonic_sparse_PINN_reference_v2_FINAL_spectral.csv"

    if args.field_version == "raw_confirmed":
        fields_csv = data_dir / "supersonic_sparse_PINN_reference_v2_FINAL_modal_fields_raw_confirmed.csv"
    else:
        fields_csv = data_dir / "supersonic_sparse_PINN_reference_v2_FINAL_modal_fields_tail_polished_v1.csv"

    if not spectral_csv.exists():
        raise FileNotFoundError(f"Missing spectral CSV: {spectral_csv}")

    if not fields_csv.exists():
        raise FileNotFoundError(f"Missing modal fields CSV: {fields_csv}")

    out_dir = repo / "assets/pinn_supersonic/datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out is None:
        out_npz = out_dir / (
            f"fixedM_{mach_tag(args.mach)}_alpha_branch_"
            f"pq_modal_anchors_{args.field_version}.npz"
        )
    else:
        out_npz = args.out

    print("[info] repo:", repo)
    print("[info] spectral CSV:", spectral_csv)
    print("[info] modal fields CSV:", fields_csv)
    print("[info] fixed Mach:", args.mach)
    print("[info] field version:", args.field_version)
    print("[info] normalize gauge:", args.normalize_gauge)
    print("[info] output:", out_npz)

    spec = pd.read_csv(spectral_csv)

    mach_col = find_col(spec.columns, ["Mach", "M"], label="Mach")
    alpha_col = find_col(spec.columns, ["alpha"], label="alpha")
    cr_col = find_col(spec.columns, ["cr", "c_r"], label="cr")
    ci_col = find_col(spec.columns, ["ci", "c_i"], label="ci")
    omega_i_col = find_col(spec.columns, ["omega_i", "omegai"], required=False, label="omega_i")
    point_id_col = find_col(spec.columns, ["point_id", "id"], required=False, label="point_id")

    mask = np.isclose(spec[mach_col].astype(float), args.mach, atol=args.mach_tol)

    if args.alpha_min is not None:
        mask &= spec[alpha_col].astype(float) >= args.alpha_min

    if args.alpha_max is not None:
        mask &= spec[alpha_col].astype(float) <= args.alpha_max

    mask = apply_optional_bool_filter(
        mask,
        spec,
        "has_exported_modal_fields",
        enabled=True,
    )

    if not args.no_trusted_filter:
        mask = apply_optional_bool_filter(
            mask,
            spec,
            "trusted_spectral",
            enabled=True,
        )
        mask = apply_optional_bool_filter(
            mask,
            spec,
            "trusted_modal",
            enabled=True,
        )

    specM = spec.loc[mask].copy()
    specM = specM.sort_values(alpha_col).reset_index(drop=True)

    if specM.empty:
        raise RuntimeError(f"No spectral/modal anchors found for Mach={args.mach}")

    alpha_anchors = specM[alpha_col].astype(float).to_numpy()
    cr_ref = specM[cr_col].astype(float).to_numpy()
    ci_ref = specM[ci_col].astype(float).to_numpy()

    if omega_i_col is not None:
        omega_i_ref = specM[omega_i_col].astype(float).to_numpy()
    else:
        omega_i_ref = alpha_anchors * ci_ref

    if point_id_col is not None:
        point_ids = specM[point_id_col].astype(str).to_numpy()
    else:
        point_ids = np.array(
            [f"{mach_tag(args.mach)}_alpha_{a:.12g}" for a in alpha_anchors],
            dtype=str,
        )

    print("\n[info] selected anchors")
    print(f"  n anchors      : {len(alpha_anchors)}")
    print(f"  alpha range    : [{alpha_anchors.min():.12g}, {alpha_anchors.max():.12g}]")
    print(f"  cr range       : [{cr_ref.min():.12g}, {cr_ref.max():.12g}]")
    print(f"  ci range       : [{ci_ref.min():.12g}, {ci_ref.max():.12g}]")
    print(f"  min |ci|       : {np.min(np.abs(ci_ref)):.12g}")

    if len(alpha_anchors) > 1:
        gaps = np.diff(np.sort(alpha_anchors))
        print(f"  alpha step med : {np.median(gaps):.12g}")
        print(f"  alpha step max : {np.max(gaps):.12g}")

    subset_csv = out_npz.with_suffix(".spectral_subset.csv")
    specM.to_csv(subset_csv, index=False)
    print("[info] wrote spectral subset:", subset_csv)

    header = pd.read_csv(fields_csv, nrows=0)
    fcols = list(header.columns)

    f_point_id_col = find_col(fcols, ["point_id", "id"], required=False, label="field point_id")
    f_mach_col = find_col(fcols, ["Mach", "M"], required=False, label="field Mach")
    f_alpha_col = find_col(fcols, ["alpha"], required=False, label="field alpha")
    f_y_col = find_col(fcols, ["y"], label="field y")

    p_real_col = find_col(
        fcols,
        ["p_real", "p_re", "preal", "p_r", "real_p", "Re_p"],
        label="p_real",
    )
    p_imag_col = find_col(
        fcols,
        ["p_imag", "p_im", "pimag", "p_i", "imag_p", "Im_p"],
        label="p_imag",
    )

    q_real_col = find_col(
        fcols,
        ["q_real", "q_re", "qreal", "q_r", "p_y_real", "dpdy_real", "pprime_real"],
        required=False,
        label="q_real",
    )
    q_imag_col = find_col(
        fcols,
        ["q_imag", "q_im", "qimag", "q_i", "p_y_imag", "dpdy_imag", "pprime_imag"],
        required=False,
        label="q_imag",
    )

    q_available = q_real_col is not None and q_imag_col is not None
    print("[info] q columns available in CSV:", q_available)

    usecols = []
    for c in [
        f_point_id_col,
        f_mach_col,
        f_alpha_col,
        f_y_col,
        p_real_col,
        p_imag_col,
        q_real_col,
        q_imag_col,
    ]:
        if c is not None and c not in usecols:
            usecols.append(c)

    if f_point_id_col is None and (f_mach_col is None or f_alpha_col is None):
        raise RuntimeError(
            "Cannot filter modal fields: need either point_id, or both Mach and alpha columns."
        )

    point_id_set = set(point_ids)
    alpha_set = set(np.round(alpha_anchors.astype(float), 12))

    chunks = []
    n_seen = 0
    n_kept = 0

    print("\n[info] reading modal fields by chunks...")

    for k, chunk in enumerate(
        pd.read_csv(fields_csv, usecols=usecols, chunksize=args.chunksize),
        start=1,
    ):
        n_seen += len(chunk)

        if f_point_id_col is not None and point_id_col is not None:
            keep = chunk[f_point_id_col].astype(str).isin(point_id_set)
        else:
            keep = np.ones(len(chunk), dtype=bool)

            if f_mach_col is not None:
                keep &= np.isclose(
                    chunk[f_mach_col].astype(float),
                    args.mach,
                    atol=args.mach_tol,
                )

            if f_alpha_col is not None:
                keep &= chunk[f_alpha_col].astype(float).round(12).isin(alpha_set)

        sub = chunk.loc[keep].copy()

        if not sub.empty:
            chunks.append(sub)
            n_kept += len(sub)

        if k % 10 == 0:
            print(f"  chunks={k:04d} rows_seen={n_seen:,} rows_kept={n_kept:,}")

    if not chunks:
        raise RuntimeError(f"No modal field rows found for Mach={args.mach}")

    df = pd.concat(chunks, ignore_index=True)

    rename = {
        f_y_col: "y",
        p_real_col: "p_real",
        p_imag_col: "p_imag",
    }

    if f_point_id_col is not None:
        rename[f_point_id_col] = "point_id"
    if f_mach_col is not None:
        rename[f_mach_col] = "Mach"
    if f_alpha_col is not None:
        rename[f_alpha_col] = "alpha"
    if q_available:
        rename[q_real_col] = "q_real_csv"
        rename[q_imag_col] = "q_imag_csv"

    df = df.rename(columns=rename)

    if "Mach" not in df.columns:
        df["Mach"] = args.mach

    if "alpha" not in df.columns:
        if "point_id" not in df.columns or point_id_col is None:
            raise RuntimeError("Cannot reconstruct alpha column in modal fields.")
        alpha_map = dict(zip(specM[point_id_col].astype(str), specM[alpha_col].astype(float)))
        df["alpha"] = df["point_id"].astype(str).map(alpha_map)

    alpha_to_pid = {
        round(float(a), 12): str(pid)
        for a, pid in zip(alpha_anchors, point_ids)
    }

    if "point_id" not in df.columns:
        df["point_id"] = df["alpha"].astype(float).round(12).map(alpha_to_pid)
    else:
        pid_str = df["point_id"].astype(str).str.strip()
        bad_pid = df["point_id"].isna() | pid_str.isin(["", "nan", "NaN", "None", "none"])
        if bad_pid.any():
            print(f"[warn] filling {int(bad_pid.sum())} empty point_id values from alpha")
            df.loc[bad_pid, "point_id"] = (
                df.loc[bad_pid, "alpha"].astype(float).round(12).map(alpha_to_pid)
            )

    for c in ["Mach", "alpha", "y", "p_real", "p_imag"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if q_available:
        df["q_real_csv"] = pd.to_numeric(df["q_real_csv"], errors="coerce")
        df["q_imag_csv"] = pd.to_numeric(df["q_imag_csv"], errors="coerce")

    df = df.dropna(subset=["Mach", "alpha", "y", "p_real", "p_imag"]).copy()
    df = df.sort_values(["alpha", "y"]).reset_index(drop=True)

    # Attach spectral anchors to every modal row.
    rounded_alpha_to_cr = {
        round(float(a), 12): float(c)
        for a, c in zip(alpha_anchors, cr_ref)
    }
    rounded_alpha_to_ci = {
        round(float(a), 12): float(c)
        for a, c in zip(alpha_anchors, ci_ref)
    }
    rounded_alpha_to_omega_i = {
        round(float(a), 12): float(w)
        for a, w in zip(alpha_anchors, omega_i_ref)
    }

    a_round = df["alpha"].astype(float).round(12)
    df["cr_ref"] = a_round.map(rounded_alpha_to_cr)
    df["ci_ref"] = a_round.map(rounded_alpha_to_ci)
    df["omega_i_ref"] = a_round.map(rounded_alpha_to_omega_i)

    # Compute q=dp/dy from p unless q exists in the CSV.
    q_real = np.full(len(df), np.nan, dtype=float)
    q_imag = np.full(len(df), np.nan, dtype=float)

    # Group by alpha rather than point_id: in some exports point_id exists
    # but is empty/NaN, which would leave q uncomputed.
    df["_alpha_key_for_q"] = df["alpha"].astype(float).round(12)

    for key, idx_values in df.groupby("_alpha_key_for_q").groups.items():
        idx = np.asarray(list(idx_values), dtype=int)

        y = df.loc[idx, "y"].to_numpy(float)
        p = (
            df.loc[idx, "p_real"].to_numpy(float)
            + 1j * df.loc[idx, "p_imag"].to_numpy(float)
        )

        if args.normalize_gauge:
            j0 = int(np.argmin(np.abs(y)))
            p0 = p[j0]
            if abs(p0) > 1e-30:
                factor = 1.0 / p0
                p = factor * p
                df.loc[idx, "p_real"] = p.real
                df.loc[idx, "p_imag"] = p.imag

        if q_available:
            q = (
                df.loc[idx, "q_real_csv"].to_numpy(float)
                + 1j * df.loc[idx, "q_imag_csv"].to_numpy(float)
            )
            if args.normalize_gauge:
                # Same factor as p normalization.
                j0 = int(np.argmin(np.abs(y)))
                p0_original = (
                    df.loc[idx, "p_real"].to_numpy(float)[j0]
                    + 1j * df.loc[idx, "p_imag"].to_numpy(float)[j0]
                )
                # After normalization p0 is already changed. Safer: recompute q from p.
                q = finite_complex_gradient(y, p)
        else:
            q = finite_complex_gradient(y, p)

        q_real[idx] = q.real
        q_imag[idx] = q.imag

    if not np.isfinite(q_real).all() or not np.isfinite(q_imag).all():
        n_bad_q = int((~np.isfinite(q_real) | ~np.isfinite(q_imag)).sum())
        raise RuntimeError(
            f"q computation failed: {n_bad_q} rows were not assigned finite q values."
        )

    df["q_real"] = q_real
    df["q_imag"] = q_imag

    if "_alpha_key_for_q" in df.columns:
        df = df.drop(columns=["_alpha_key_for_q"])

    # Diagnostics: gamma=q/p and Q=q/alpha.
    p_all = df["p_real"].to_numpy(float) + 1j * df["p_imag"].to_numpy(float)
    q_all = df["q_real"].to_numpy(float) + 1j * df["q_imag"].to_numpy(float)
    alpha_all = df["alpha"].to_numpy(float)

    eps_p = 1e-12
    gamma = np.full_like(p_all, np.nan + 1j * np.nan, dtype=complex)
    good_p = np.abs(p_all) > eps_p
    gamma[good_p] = q_all[good_p] / p_all[good_p]

    eps_alpha = 1e-12
    Q = q_all / np.maximum(np.abs(alpha_all), eps_alpha)

    df["gamma_real"] = gamma.real
    df["gamma_imag"] = gamma.imag
    df["Q_real"] = Q.real
    df["Q_imag"] = Q.imag

    required = [
        "Mach",
        "alpha",
        "y",
        "cr_ref",
        "ci_ref",
        "omega_i_ref",
        "p_real",
        "p_imag",
        "q_real",
        "q_imag",
    ]

    finite_mask = np.isfinite(df[required].to_numpy()).all(axis=1)
    n_bad = int((~finite_mask).sum())

    if n_bad:
        print(f"[warn] dropping {n_bad} non-finite rows")
        df = df.loc[finite_mask].copy()

    df = df.sort_values(["alpha", "y"]).reset_index(drop=True)

    by_alpha = df.groupby("alpha").size()

    print("\n[info] modal anchor subset")
    print(f"  rows kept       : {len(df):,}")
    print(f"  alphas in modes : {len(by_alpha)}")
    print(f"  rows/alpha min  : {int(by_alpha.min())}")
    print(f"  rows/alpha max  : {int(by_alpha.max())}")

    missing_alphas = sorted(
        set(np.round(alpha_anchors, 12))
        - set(np.round(by_alpha.index.to_numpy(float), 12))
    )

    if missing_alphas:
        print("[warn] missing modal fields for alpha anchors:")
        for a in missing_alphas:
            print(f"  alpha={a:.12g}")

    # Store a compact table too, useful for debugging but much smaller than original CSV.
    modal_subset_csv = out_npz.with_suffix(".modal_subset_head.csv")
    df.head(2000).to_csv(modal_subset_csv, index=False)
    print("[info] wrote modal subset head:", modal_subset_csv)

    metadata = {
        "dataset_type": "fixed-M supersonic alpha-branch p/q modal anchors",
        "Mach_fixed": args.mach,
        "field_version": args.field_version,
        "normalize_gauge": bool(args.normalize_gauge),
        "spectral_csv": str(spectral_csv),
        "fields_csv": str(fields_csv),
        "n_spectral_anchors": int(len(alpha_anchors)),
        "n_modal_rows": int(len(df)),
        "alpha_min": float(alpha_anchors.min()),
        "alpha_max": float(alpha_anchors.max()),
        "cr_min": float(cr_ref.min()),
        "cr_max": float(cr_ref.max()),
        "ci_min": float(ci_ref.min()),
        "ci_max": float(ci_ref.max()),
        "min_abs_ci": float(np.min(np.abs(ci_ref))),
        "q_definition": "q = dp/dy, computed from modal pressure p(y) unless q columns exist.",
        "gamma_definition": "gamma = q/p, stored as diagnostic only.",
        "Q_definition": "Q = q/alpha, stored for p/Qscaled diagnostics or low-alpha variants.",
        "method_role": (
            "Dataset for local supersonic PINN chart. The PINN should learn "
            "spectral seeds cr/ci and modal seeds p/q, then pass them to "
            "a dense GEP or shooting refinement."
        ),
    }

    metadata_json_path = out_npz.with_suffix(".metadata.json")
    metadata_json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    np.savez_compressed(
        out_npz,
        Mach_fixed=np.array(args.mach, dtype=float),
        alpha_anchors=alpha_anchors.astype(float),
        cr_ref=cr_ref.astype(float),
        ci_ref=ci_ref.astype(float),
        omega_i_ref=omega_i_ref.astype(float),
        spectral_point_id=point_ids.astype(str),
        row_point_id=df["point_id"].astype(str).to_numpy(),
        row_alpha=df["alpha"].astype(float).to_numpy(),
        row_cr_ref=df["cr_ref"].astype(float).to_numpy(),
        row_ci_ref=df["ci_ref"].astype(float).to_numpy(),
        row_omega_i_ref=df["omega_i_ref"].astype(float).to_numpy(),
        y=df["y"].astype(float).to_numpy(),
        p_real=df["p_real"].astype(float).to_numpy(),
        p_imag=df["p_imag"].astype(float).to_numpy(),
        q_real=df["q_real"].astype(float).to_numpy(),
        q_imag=df["q_imag"].astype(float).to_numpy(),
        gamma_real=df["gamma_real"].astype(float).to_numpy(),
        gamma_imag=df["gamma_imag"].astype(float).to_numpy(),
        Q_real=df["Q_real"].astype(float).to_numpy(),
        Q_imag=df["Q_imag"].astype(float).to_numpy(),
        metadata_json=json.dumps(metadata, indent=2),
    )

    print("\n[done] wrote dataset:", out_npz)
    print("[done] wrote metadata:", metadata_json_path)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
