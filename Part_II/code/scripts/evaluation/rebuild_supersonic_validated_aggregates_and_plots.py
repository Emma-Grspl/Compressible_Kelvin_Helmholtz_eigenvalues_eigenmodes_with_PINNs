#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path("assets/classic_supersonic")
SHOOTING = ROOT / "shooting"
VALIDATED = ROOT / "validated_modal_points"
BLUMEN_CANDIDATES = [
    ROOT / "blumen_reference" / "supersonic_ci_digitized_points.csv",
    Path("assets/classic_supersonic/csv/blumen_validation/supersonic/table_ci_datasets.csv"),
]

CANON_SPECTRAL = SHOOTING / "supersonic_reference_core_local_spectral.csv"
CANON_MODAL = SHOOTING / "supersonic_reference_core_local_modal.csv"
CANON_FIELDS = SHOOTING / "supersonic_reference_core_local_modal_fields.csv"
CANON_STRICT = VALIDATED / "supersonic_validated_modal_points.csv"


def norm_name(c: str) -> str:
    return c.strip().lower().replace(" ", "_").replace("-", "_")


def colmap(df: pd.DataFrame) -> dict[str, str]:
    return {norm_name(c): c for c in df.columns}


def pick(df: pd.DataFrame, names, required=False, avoid=()):
    cm = colmap(df)
    for n in names:
        key = norm_name(n)
        if key in cm:
            return cm[key]

    avoid = tuple(a.lower() for a in avoid)
    for c in df.columns:
        lc = norm_name(c)
        if any(a in lc for a in avoid):
            continue
        for n in names:
            if norm_name(n) in lc:
                return c

    if required:
        raise KeyError(f"Missing column among {names}. Columns={list(df.columns)}")
    return None


def truthy(x) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes", "y", "validated", "success", "accepted"}


def infer_mach_from_path(path: Path) -> float | None:
    s = str(path)
    pats = [
        r"M([0-9])([0-9])\b",        # M16 -> 1.6, M14 -> 1.4
        r"M([0-9])([0-9])([0-9])",   # M160 -> 1.60
        r"m([0-9])([0-9])\b",
        r"m([0-9])([0-9])([0-9])",
    ]
    for pat in pats:
        m = re.search(pat, s)
        if m:
            digits = "".join(m.groups())
            if len(digits) == 2:
                return float(digits) / 10.0
            if len(digits) == 3:
                return float(digits) / 100.0
    return None


def source_label(path: Path) -> str:
    name = path.name
    name = re.sub(r"\.csv$", "", name)
    return name


def read_csv_safe(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] cannot read {path}: {e}")
        return None


def normalize_summary(path: Path) -> pd.DataFrame | None:
    df = read_csv_safe(path)
    if df is None or df.empty:
        return None

    a_col = pick(df, ["alpha", "a"], required=False)
    M_col = pick(df, ["Mach", "mach", "M"], required=False)

    if a_col is None:
        return None

    cr_col = pick(
        df,
        ["reference_cr", "best_cr", "cr", "c_r", "root_cr"],
        required=False,
        avoid=("blumen", "err", "delta", "diff"),
    )
    ci_col = pick(
        df,
        ["reference_ci", "best_ci", "ci", "c_i", "root_ci"],
        required=False,
        avoid=("blumen", "err", "delta", "diff", "rel", "abs", "dci"),
    )
    omega_col = pick(df, ["reference_omega_i", "omega_i", "best_omega_i"], required=False)

    # Summary/candidate files without c are not useful for spectral aggregation.
    if cr_col is None or ci_col is None:
        return None

    out = pd.DataFrame()
    out["alpha"] = pd.to_numeric(df[a_col], errors="coerce")

    if M_col is not None:
        out["Mach"] = pd.to_numeric(df[M_col], errors="coerce")
    else:
        inferred = infer_mach_from_path(path)
        out["Mach"] = inferred if inferred is not None else np.nan

    out["reference_cr"] = pd.to_numeric(df[cr_col], errors="coerce")
    out["reference_ci"] = pd.to_numeric(df[ci_col], errors="coerce")

    if omega_col is not None:
        out["reference_omega_i"] = pd.to_numeric(df[omega_col], errors="coerce")
    else:
        out["reference_omega_i"] = out["alpha"] * out["reference_ci"]

    line_col = pick(df, ["line_id", "point_id", "id"], required=False)
    if line_col is not None:
        out["line_id"] = df[line_col].astype(str)
    else:
        out["line_id"] = [
            f"M{M:.2f}_a{a:.5f}" if np.isfinite(M) and np.isfinite(a) else ""
            for M, a in zip(out["Mach"], out["alpha"])
        ]

    status_col = pick(df, ["best_status", "status", "acceptance_mode"], required=False)
    out["best_status"] = df[status_col].astype(str) if status_col is not None else ""

    for dst, candidates in [
        ("best_stage1_mismatch", ["best_stage1_mismatch", "stage1_mismatch", "mismatch_stage1"]),
        ("best_stage2_mismatch", ["best_stage2_mismatch", "stage2_mismatch", "mismatch_stage2"]),
    ]:
        c = pick(df, candidates, required=False)
        out[dst] = pd.to_numeric(df[c], errors="coerce") if c else np.nan

    for dst, candidates in [
        ("best_spectral_success", ["best_spectral_success", "spectral_success", "success"]),
        ("best_mode_success", ["best_mode_success", "mode_success"]),
        ("trusted_spectral", ["trusted_spectral"]),
        ("trusted_modal", ["trusted_modal"]),
    ]:
        c = pick(df, candidates, required=False)
        out[dst] = df[c].map(truthy) if c else False

    # Conservative inference.
    status_l = out["best_status"].astype(str).str.lower()
    valid_status = status_l.str.contains("validated|success|accepted|spectral|modal", regex=True, na=False)
    failed_status = status_l.str.contains("fail|reject|bad|invalid", regex=True, na=False)

    finite_core = (
        np.isfinite(out["Mach"])
        & np.isfinite(out["alpha"])
        & np.isfinite(out["reference_cr"])
        & np.isfinite(out["reference_ci"])
    )

    out["valid_spectral_candidate"] = finite_core & ~failed_status & (
        out["trusted_spectral"]
        | out["best_spectral_success"]
        | valid_status
        | (out["best_stage2_mismatch"] < 1e-8)
    )

    out["valid_modal_candidate"] = finite_core & ~failed_status & (
        out["trusted_modal"]
        | out["best_mode_success"]
        | status_l.str.contains("validated|modal", regex=True, na=False)
    )

    out["source_csv"] = str(path)
    out["source_label"] = source_label(path)
    out["source_group"] = path.parent.name

    return out


def gather_summaries() -> pd.DataFrame:
    candidates = []
    for pat in ["**/*summary*.csv", "**/*candidates*.csv"]:
        candidates.extend(SHOOTING.glob(pat))

    # Exclude canonical outputs from previous aggregation to avoid self-feeding.
    exclude_names = {
        "supersonic_reference_core_local_spectral.csv",
        "supersonic_reference_core_local_modal.csv",
        "supersonic_reference_core_local_modal_fields.csv",
    }

    frames = []
    for p in sorted(set(candidates)):
        if p.name in exclude_names:
            continue
        df = normalize_summary(p)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No usable summary/candidate CSVs found.")

    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_df = all_df.dropna(subset=["Mach", "alpha", "reference_cr", "reference_ci"])
    return all_df


def deduplicate_points(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    x = df.copy()
    if kind == "spectral":
        x = x[x["valid_spectral_candidate"]].copy()
    elif kind == "modal":
        x = x[x["valid_modal_candidate"]].copy()
    else:
        raise ValueError(kind)

    if x.empty:
        return x

    x["Mach_round"] = x["Mach"].round(10)
    x["alpha_round"] = x["alpha"].round(10)

    # Prefer trusted, then lower stage2 mismatch, then source with fields/core-band.
    x["_trusted_score"] = 0
    if kind == "spectral":
        x["_trusted_score"] += x["trusted_spectral"].astype(int) * 10
        x["_trusted_score"] += x["best_spectral_success"].astype(int) * 5
    else:
        x["_trusted_score"] += x["trusted_modal"].astype(int) * 10
        x["_trusted_score"] += x["best_mode_success"].astype(int) * 5

    x["_source_score"] = (
        x["source_csv"].astype(str).str.contains("core_band|pointwise|modal_front|continuation_core", regex=True).astype(int)
    )

    x["_stage2_sort"] = x["best_stage2_mismatch"].fillna(np.inf)

    x = x.sort_values(
        ["Mach_round", "alpha_round", "_trusted_score", "_source_score", "_stage2_sort"],
        ascending=[True, True, False, False, True],
    )

    x = x.drop_duplicates(["Mach_round", "alpha_round"], keep="first")
    x = x.drop(columns=[c for c in ["Mach_round", "alpha_round", "_trusted_score", "_source_score", "_stage2_sort"] if c in x.columns])
    x = x.sort_values(["Mach", "alpha"]).reset_index(drop=True)

    return x


def normalize_fields(path: Path) -> pd.DataFrame | None:
    df = read_csv_safe(path)
    if df is None or df.empty:
        return None

    a_col = pick(df, ["alpha", "a"], required=False)
    y_col = pick(df, ["y"], required=False)
    pre_col = pick(df, ["p_real", "p_re", "real_p"], required=False)

    if a_col is None or y_col is None or pre_col is None:
        return None

    M_col = pick(df, ["Mach", "mach", "M"], required=False)
    pim_col = pick(df, ["p_imag", "p_im", "imag_p"], required=False)
    line_col = pick(df, ["line_id", "point_id", "id"], required=False)

    out = pd.DataFrame()
    out["alpha"] = pd.to_numeric(df[a_col], errors="coerce")

    if M_col is not None:
        out["Mach"] = pd.to_numeric(df[M_col], errors="coerce")
    else:
        inferred = infer_mach_from_path(path)
        out["Mach"] = inferred if inferred is not None else np.nan

    out["y"] = pd.to_numeric(df[y_col], errors="coerce")
    out["p_real"] = pd.to_numeric(df[pre_col], errors="coerce")
    out["p_imag"] = pd.to_numeric(df[pim_col], errors="coerce") if pim_col else 0.0

    if line_col:
        out["line_id"] = df[line_col].astype(str)
    else:
        out["line_id"] = ""

    for col in ["rho_real", "rho_imag", "u_real", "u_imag", "v_real", "v_imag"]:
        c = pick(df, [col], required=False)
        if c:
            out[col] = pd.to_numeric(df[c], errors="coerce")

    out["source_fields_csv"] = str(path)
    out["source_label"] = source_label(path)
    out["source_group"] = path.parent.name

    out = out.dropna(subset=["Mach", "alpha", "y", "p_real", "p_imag"])
    return out


def gather_fields() -> pd.DataFrame:
    paths = sorted(SHOOTING.glob("**/*fields*.csv"))
    paths = [p for p in paths if p.name != "supersonic_reference_core_local_modal_fields.csv"]

    frames = []
    for p in paths:
        df = normalize_fields(p)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No usable fields CSVs found.")

    return pd.concat(frames, ignore_index=True, sort=False)


def select_modal_fields(modal: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fields = fields.copy()
    fields["Mach_round"] = fields["Mach"].round(10)
    fields["alpha_round"] = fields["alpha"].round(10)

    for _, r in modal.iterrows():
        M = round(float(r["Mach"]), 10)
        a = round(float(r["alpha"]), 10)
        line_id = str(r.get("line_id", ""))

        sub = fields[(fields["Mach_round"] == M) & (fields["alpha_round"] == a)].copy()
        if sub.empty:
            print(f"[WARN] no fields for modal point M={M}, alpha={a}, line_id={line_id}")
            continue

        # Prefer same line_id when possible.
        if line_id and "line_id" in sub.columns and (sub["line_id"].astype(str) == line_id).any():
            sub = sub[sub["line_id"].astype(str) == line_id].copy()
        else:
            # Otherwise choose the field source with the most samples for this point.
            counts = sub.groupby("source_fields_csv").size().sort_values(ascending=False)
            sub = sub[sub["source_fields_csv"] == counts.index[0]].copy()

        for c in [
            "reference_cr", "reference_ci", "reference_omega_i", "trusted_modal",
            "best_status", "source_csv", "source_group"
        ]:
            sub[c] = r.get(c, np.nan)

        rows.append(sub)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True, sort=False)
    out = out.drop(columns=[c for c in ["Mach_round", "alpha_round"] if c in out.columns])
    return out.sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)


def load_blumen_ci() -> pd.DataFrame:
    frames = []
    for p in BLUMEN_CANDIDATES:
        if not p.exists():
            continue
        df = read_csv_safe(p)
        if df is None or df.empty:
            continue

        a_col = pick(df, ["alpha", "a"], required=False)
        M_col = pick(df, ["Mach", "mach", "M"], required=False)
        ci_col = pick(df, ["ci", "blumen_ci", "c_i"], required=False, avoid=("err", "rel", "abs"))

        if a_col is None or M_col is None or ci_col is None:
            continue

        out = pd.DataFrame({
            "Mach": pd.to_numeric(df[M_col], errors="coerce"),
            "alpha": pd.to_numeric(df[a_col], errors="coerce"),
            "ci": pd.to_numeric(df[ci_col], errors="coerce"),
            "source": str(p),
        }).dropna()

        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["Mach", "alpha", "ci", "source"])

    return pd.concat(frames, ignore_index=True, sort=False).drop_duplicates()


def plot_ci_curves(spectral, modal, blumen, outdir: Path):
    figdir = outdir / "ci_curves"
    figdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / "supersonic_ci_vs_alpha_by_mach.pdf"

    machs = sorted(set(spectral["Mach"].dropna().round(10)) | set(modal["Mach"].dropna().round(10)) | set(blumen["Mach"].dropna().round(10)))

    with PdfPages(pdf) as pp:
        for M in machs:
            fig, ax = plt.subplots(figsize=(7.5, 4.8))

            for df, label, marker in [
                (spectral, "spectral validé", "o"),
                (modal, "modal validé", "s"),
            ]:
                sub = df[np.isclose(df["Mach"], M)].sort_values("alpha")
                if not sub.empty:
                    ax.plot(sub["alpha"], sub["reference_ci"], marker=marker, linewidth=1.5, label=label)

            b = blumen[np.isclose(blumen["Mach"], M)].sort_values("alpha")
            if not b.empty:
                ax.plot(b["alpha"], b["ci"], "x--", linewidth=1.2, label="Blumen/digitized")

            ax.set_title(f"Supersonic ci(alpha), M={M:g}")
            ax.set_xlabel("alpha")
            ax.set_ylabel("ci")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            pp.savefig(fig)

            fig.savefig(figdir / f"ci_vs_alpha_M{int(round(M*100)):03d}.png", dpi=220)
            plt.close(fig)

    print("[OK] wrote", pdf)


def plot_modes(modal_fields: pd.DataFrame, outdir: Path):
    modes_dir = outdir / "modes"
    modes_dir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / "supersonic_validated_modal_modes.pdf"

    if modal_fields.empty:
        print("[WARN] no modal fields to plot")
        return

    groups = list(modal_fields.groupby(["Mach", "alpha"], sort=True))

    with PdfPages(pdf) as pp:
        for i in range(0, len(groups), 6):
            chunk = groups[i:i+6]
            fig, axes = plt.subplots(3, 2, figsize=(10, 12))
            axes = axes.ravel()

            for ax in axes:
                ax.axis("off")

            for ax, ((M, a), sub) in zip(axes, chunk):
                ax.axis("on")
                sub = sub.sort_values("y")
                y = sub["y"].to_numpy(float)
                p = sub["p_real"].to_numpy(float) + 1j * sub["p_imag"].to_numpy(float)
                norm = np.nanmax(np.abs(p))
                if not np.isfinite(norm) or norm <= 0:
                    continue

                ci = sub["reference_ci"].iloc[0] if "reference_ci" in sub.columns else np.nan
                ax.plot(y, p.real / norm, linewidth=0.9, label="Re(p)/max|p|")
                ax.plot(y, np.abs(p) / norm, "--", linewidth=0.9, label="|p|/max|p|")
                ax.axhline(0.0, color="0.75", linewidth=0.6)
                ax.grid(True, alpha=0.25)
                ax.set_title(f"M={M:.2f}, alpha={a:.5f}\nci={ci:.5g}, N={len(sub)}", fontsize=9)
                ax.set_xlabel("y")
                ax.set_ylabel("normalized mode")

                tag = f"M{int(round(M*100)):03d}_a{int(round(a*100000)):05d}"
                fig1, ax1 = plt.subplots(figsize=(8.5, 4.5))
                ax1.plot(y, p.real / norm, linewidth=1.1, label="Re(p)/max|p|")
                ax1.plot(y, np.abs(p) / norm, "--", linewidth=1.1, label="|p|/max|p|")
                ax1.axhline(0.0, color="0.75", linewidth=0.6)
                ax1.grid(True, alpha=0.3)
                ax1.set_title(f"M={M:.2f}, alpha={a:.5f}, ci={ci:.5g}, N={len(sub)}")
                ax1.set_xlabel("y")
                ax1.set_ylabel("normalized mode")
                ax1.legend()
                fig1.tight_layout()
                fig1.savefig(modes_dir / f"mode_{tag}.png", dpi=220)
                plt.close(fig1)

            axes[0].legend(fontsize=8)
            fig.tight_layout()
            pp.savefig(fig)
            plt.close(fig)

    print("[OK] wrote", pdf)
    print("[OK] wrote individual modes in", modes_dir)


def backup_and_write(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = dst.with_suffix(dst.suffix + f".bak_{stamp}")
        shutil.copy2(dst, bak)
        print("[backup]", dst, "->", bak)
    shutil.copy2(src, dst)
    print("[promote]", src, "->", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="assets/classic_supersonic/validated_modal_points/rebuilt_aggregates")
    ap.add_argument("--promote", action="store_true", help="Overwrite canonical aggregate CSVs after backup.")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/5] gather summaries")
    summaries = gather_summaries()
    summaries.to_csv(outdir / "all_candidate_summaries_normalized.csv", index=False)
    print("candidate rows:", len(summaries))

    print("[2/5] build spectral/modal point tables")
    spectral = deduplicate_points(summaries, "spectral")
    modal = deduplicate_points(summaries, "modal")

    spectral_path = outdir / "supersonic_reference_core_local_spectral_REBUILT.csv"
    modal_path = outdir / "supersonic_reference_core_local_modal_REBUILT.csv"

    spectral.to_csv(spectral_path, index=False)
    modal.to_csv(modal_path, index=False)

    print("spectral points:", len(spectral), "->", spectral_path)
    print("modal points:", len(modal), "->", modal_path)

    print("[3/5] gather modal fields")
    all_fields = gather_fields()
    all_fields.to_csv(outdir / "all_modal_fields_candidates_normalized.csv", index=False)
    modal_fields = select_modal_fields(modal, all_fields)
    fields_path = outdir / "supersonic_reference_core_local_modal_fields_REBUILT.csv"
    modal_fields.to_csv(fields_path, index=False)
    print("modal field rows:", len(modal_fields), "->", fields_path)

    missing_ci = spectral[~np.isfinite(spectral["reference_ci"])]
    if not missing_ci.empty:
        missing_ci.to_csv(outdir / "missing_ci_spectral_rows.csv", index=False)
        print("[WARN] spectral rows with missing ci:", len(missing_ci))

    missing_fields = []
    for _, r in modal.iterrows():
        has = (
            np.isclose(modal_fields["Mach"], r["Mach"]).any()
            and np.isclose(modal_fields["alpha"], r["alpha"]).any()
        )
        if not has:
            missing_fields.append(r)
    if missing_fields:
        pd.DataFrame(missing_fields).to_csv(outdir / "modal_points_missing_fields.csv", index=False)
        print("[WARN] modal points missing fields:", len(missing_fields))

    print("[4/5] plot ci curves vs Blumen")
    blumen = load_blumen_ci()
    blumen.to_csv(outdir / "blumen_ci_points_used.csv", index=False)
    plot_ci_curves(spectral, modal, blumen, outdir)

    print("[5/5] plot modes")
    plot_modes(modal_fields, outdir)

    if args.promote:
        backup_and_write(spectral_path, CANON_SPECTRAL)
        backup_and_write(modal_path, CANON_MODAL)
        backup_and_write(fields_path, CANON_FIELDS)
        # Strict PDF table: use modal points as current validated modal list.
        backup_and_write(modal_path, CANON_STRICT)

    print("\n[SUMMARY]")
    print("Output dir:", outdir)
    print("Spectral CSV:", spectral_path)
    print("Modal CSV:", modal_path)
    print("Fields CSV:", fields_path)
    print("ci PDF:", outdir / "supersonic_ci_vs_alpha_by_mach.pdf")
    print("modes PDF:", outdir / "supersonic_validated_modal_modes.pdf")


if __name__ == "__main__":
    main()
