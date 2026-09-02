#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


D = Path("assets/classic_supersonic/validated_modal_points/rebuilt_aggregates_latest")
OUT = D / "high_mach_triage"
OUT.mkdir(parents=True, exist_ok=True)

CAND = D / "all_candidate_summaries_normalized.csv"
FIELDS = D / "all_modal_fields_candidates_normalized.csv"


GRID_CI = np.array([0.015, 0.025, 0.040, 0.055, 0.070, 0.085], dtype=float)


def close_grid_ci(x):
    try:
        x = float(x)
    except Exception:
        return False
    return bool(np.min(np.abs(GRID_CI - x)) < 5e-7)


def finite(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def pick_summary_row(cand, M, a, line_id, source_label):
    sub = cand[
        np.isclose(cand["Mach"].astype(float), float(M), atol=1e-10)
        & np.isclose(cand["alpha"].astype(float), float(a), atol=1e-10)
    ].copy()

    if sub.empty:
        return None

    if line_id and "line_id" in sub.columns:
        s = sub[sub["line_id"].astype(str) == str(line_id)]
        if not s.empty:
            sub = s.copy()

    elif source_label and "source_label" in sub.columns:
        s = sub[sub["source_label"].astype(str) == str(source_label)]
        if not s.empty:
            sub = s.copy()

    # Garde prioritairement les lignes avec cr/ci finis et faible mismatch.
    sub["_finite_score"] = (
        np.isfinite(pd.to_numeric(sub["reference_cr"], errors="coerce")).astype(int)
        + np.isfinite(pd.to_numeric(sub["reference_ci"], errors="coerce")).astype(int)
    )

    if "best_stage2_mismatch" in sub.columns:
        sub["_stage2"] = pd.to_numeric(sub["best_stage2_mismatch"], errors="coerce").fillna(np.inf)
    else:
        sub["_stage2"] = np.inf

    if "best_stage1_mismatch" in sub.columns:
        sub["_stage1"] = pd.to_numeric(sub["best_stage1_mismatch"], errors="coerce").fillna(np.inf)
    else:
        sub["_stage1"] = np.inf

    sub = sub.sort_values(["_finite_score", "_stage2", "_stage1"], ascending=[False, True, True])
    return sub.iloc[0]


def mode_metrics(sub):
    y = pd.to_numeric(sub["y"], errors="coerce").to_numpy(float)
    p = (
        pd.to_numeric(sub["p_real"], errors="coerce").to_numpy(float)
        + 1j * pd.to_numeric(sub["p_imag"], errors="coerce").to_numpy(float)
    )

    good = np.isfinite(y) & np.isfinite(p.real) & np.isfinite(p.imag)
    y = y[good]
    p = p[good]

    if len(y) < 10:
        return None

    order = np.argsort(y)
    y = y[order]
    p = p[order]

    pabs = np.abs(p)
    norm = np.nanmax(pabs)
    if not np.isfinite(norm) or norm <= 0:
        return None

    yy = np.nanmax(np.abs(y))
    if not np.isfinite(yy) or yy <= 0:
        yy = 1.0

    imax = int(np.nanargmax(pabs))
    peak_y_abs = abs(float(y[imax]))
    peak_y_rel = peak_y_abs / yy

    tail = pabs[np.abs(y) > 0.85 * yy]
    tail_ratio = float(np.nanmax(tail) / norm) if len(tail) else np.nan

    edge_ratio = float(max(pabs[0], pabs[-1]) / norm)

    # Variation normalisée : utile pour repérer des modes numériquement hachés.
    dp = np.abs(np.diff(p / norm))
    jump95 = float(np.nanpercentile(dp, 95)) if len(dp) else np.nan
    jumpmax = float(np.nanmax(dp)) if len(dp) else np.nan

    return {
        "n_y": int(len(y)),
        "y_min": float(np.nanmin(y)),
        "y_max": float(np.nanmax(y)),
        "peak_y_abs": peak_y_abs,
        "peak_y_rel": float(peak_y_rel),
        "tail_ratio": tail_ratio,
        "edge_ratio": edge_ratio,
        "jump95": jump95,
        "jumpmax": jumpmax,
    }


def main():
    cand = pd.read_csv(CAND, low_memory=False)
    fields = pd.read_csv(FIELDS, low_memory=False)

    cand["Mach"] = pd.to_numeric(cand["Mach"], errors="coerce")
    cand["alpha"] = pd.to_numeric(cand["alpha"], errors="coerce")
    fields["Mach"] = pd.to_numeric(fields["Mach"], errors="coerce")
    fields["alpha"] = pd.to_numeric(fields["alpha"], errors="coerce")

    fields_hi = fields[fields["Mach"] >= 1.55].copy()

    group_cols = ["Mach", "alpha"]
    for c in ["line_id", "source_label", "source_fields_csv"]:
        if c in fields_hi.columns:
            group_cols.append(c)

    rows = []

    for key, sub in fields_hi.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)

        meta = dict(zip(group_cols, key))
        M = float(meta["Mach"])
        a = float(meta["alpha"])
        line_id = str(meta.get("line_id", ""))
        source_label = str(meta.get("source_label", ""))

        mm = mode_metrics(sub)
        if mm is None:
            continue

        sr = pick_summary_row(cand, M, a, line_id=line_id, source_label=source_label)

        row = dict(meta)
        row.update(mm)

        if sr is not None:
            for c in [
                "reference_cr", "reference_ci", "reference_omega_i",
                "best_status", "best_stage1_mismatch", "best_stage2_mismatch",
                "trusted_spectral", "trusted_modal", "source_csv", "source_label", "line_id"
            ]:
                if c in sr.index:
                    row[f"summary_{c}"] = sr[c]

        ci = row.get("summary_reference_ci", np.nan)
        cr = row.get("summary_reference_cr", np.nan)

        row["finite_cr_ci"] = finite(cr) and finite(ci)
        row["grid_ci_flag"] = close_grid_ci(ci)

        # Score bas = meilleur. Ce n'est pas une validation, seulement un tri de lecture.
        stage2 = row.get("summary_best_stage2_mismatch", np.nan)
        try:
            stage2 = float(stage2)
        except Exception:
            stage2 = np.nan

        row["triage_score"] = (
            10.0 * float(row["tail_ratio"])
            + 10.0 * float(row["edge_ratio"])
            + 2.0 * float(row["peak_y_rel"])
            + 0.2 * float(row["jump95"])
            + (0.0 if not np.isfinite(stage2) else min(10.0, abs(np.log10(stage2 + 1e-300))) * 0.0)
            + (2.0 if row["grid_ci_flag"] else 0.0)
        )

        rows.append(row)

    triage = pd.DataFrame(rows)

    if triage.empty:
        raise RuntimeError("No high-Mach field candidates found.")

    triage = triage.sort_values(["Mach", "alpha", "triage_score"]).reset_index(drop=True)
    triage.to_csv(OUT / "high_mach_triage_all_field_groups.csv", index=False)

    eligible = triage[
        triage["finite_cr_ci"].astype(bool)
        & (triage["n_y"] >= 1000)
        & (triage["tail_ratio"] < 0.08)
        & (triage["edge_ratio"] < 0.05)
        & (triage["peak_y_rel"] < 0.25)
    ].copy()

    selected = (
        eligible.sort_values(["Mach", "alpha", "triage_score"])
        .groupby(["Mach", "alpha"], as_index=False)
        .head(1)
        .sort_values(["Mach", "alpha"])
        .reset_index(drop=True)
    )

    selected.to_csv(OUT / "high_mach_selected_for_visual_review.csv", index=False)

    print("[OK] wrote", OUT / "high_mach_triage_all_field_groups.csv")
    print("[OK] wrote", OUT / "high_mach_selected_for_visual_review.csv")

    print("\n=== all field groups by Mach/alpha ===")
    print(triage.groupby(["Mach", "alpha"]).size().reset_index(name="n_groups").to_string(index=False))

    print("\n=== selected for visual review ===")
    cols = [
        "Mach", "alpha", "summary_reference_cr", "summary_reference_ci",
        "summary_reference_omega_i", "n_y", "tail_ratio", "edge_ratio",
        "peak_y_rel", "jump95", "grid_ci_flag", "triage_score",
        "summary_best_status", "summary_source_label"
    ]
    cols = [c for c in cols if c in selected.columns]
    print(selected[cols].to_string(index=False))

    # Plot cr/ci selected.
    with PdfPages(OUT / "high_mach_selected_cr_ci_omega.pdf") as pp:
        for M in sorted(selected["Mach"].unique()):
            ss = selected[np.isclose(selected["Mach"], M)].sort_values("alpha")

            for col, ylabel in [
                ("summary_reference_ci", "ci"),
                ("summary_reference_cr", "cr"),
                ("summary_reference_omega_i", "omega_i"),
            ]:
                if col not in ss.columns:
                    continue

                fig, ax = plt.subplots(figsize=(7.5, 4.7))
                ax.plot(ss["alpha"], pd.to_numeric(ss[col], errors="coerce"), "o-", linewidth=1.4)
                ax.set_title(f"High Mach selected {ylabel}(alpha), M={M:g}")
                ax.set_xlabel("alpha")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                pp.savefig(fig)
                plt.close(fig)

    # Plot modes selected.
    with PdfPages(OUT / "high_mach_selected_modes_for_visual_review.pdf") as pp:
        selected_keys = []
        for _, r in selected.iterrows():
            selected_keys.append({
                "Mach": float(r["Mach"]),
                "alpha": float(r["alpha"]),
                "line_id": str(r.get("line_id", "")),
                "source_fields_csv": str(r.get("source_fields_csv", "")),
                "ci": float(r.get("summary_reference_ci", np.nan)),
            })

        for i in range(0, len(selected_keys), 6):
            chunk = selected_keys[i:i+6]
            fig, axes = plt.subplots(3, 2, figsize=(10, 12))
            axes = axes.ravel()

            for ax in axes:
                ax.axis("off")

            for ax, item in zip(axes, chunk):
                M = item["Mach"]
                a = item["alpha"]

                sub = fields_hi[
                    np.isclose(fields_hi["Mach"], M, atol=1e-10)
                    & np.isclose(fields_hi["alpha"], a, atol=1e-10)
                ].copy()

                if item["line_id"] and "line_id" in sub.columns:
                    s2 = sub[sub["line_id"].astype(str) == item["line_id"]]
                    if not s2.empty:
                        sub = s2.copy()

                if item["source_fields_csv"] and "source_fields_csv" in sub.columns:
                    s2 = sub[sub["source_fields_csv"].astype(str) == item["source_fields_csv"]]
                    if not s2.empty:
                        sub = s2.copy()

                sub = sub.sort_values("y")

                y = pd.to_numeric(sub["y"], errors="coerce").to_numpy(float)
                p = (
                    pd.to_numeric(sub["p_real"], errors="coerce").to_numpy(float)
                    + 1j * pd.to_numeric(sub["p_imag"], errors="coerce").to_numpy(float)
                )
                good = np.isfinite(y) & np.isfinite(p.real) & np.isfinite(p.imag)
                y = y[good]
                p = p[good]

                if len(y) < 10:
                    continue

                norm = np.nanmax(np.abs(p))
                if not np.isfinite(norm) or norm <= 0:
                    continue

                ax.axis("on")
                ax.plot(y, p.real / norm, linewidth=1.0, label="Re(p)/max|p|")
                ax.plot(y, np.abs(p) / norm, "--", linewidth=1.0, label="|p|/max|p|")
                ax.axhline(0.0, linewidth=0.6)
                ax.grid(True, alpha=0.25)
                ax.set_title(f"M={M:.2f}, alpha={a:.5f}\nci={item['ci']:.5g}, N={len(y)}", fontsize=9)
                ax.set_xlabel("y")
                ax.set_ylabel("normalized mode")

            axes[0].legend(fontsize=8)
            fig.tight_layout()
            pp.savefig(fig)
            plt.close(fig)

    print("[OK] wrote", OUT / "high_mach_selected_cr_ci_omega.pdf")
    print("[OK] wrote", OUT / "high_mach_selected_modes_for_visual_review.pdf")


if __name__ == "__main__":
    main()
