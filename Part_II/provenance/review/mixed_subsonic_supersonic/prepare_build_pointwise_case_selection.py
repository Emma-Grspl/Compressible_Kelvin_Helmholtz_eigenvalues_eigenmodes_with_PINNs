#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SUPERSONIC_SPECTRAL = (
    REPO_ROOT
    / "assets/classic_supersonic/csv/pinn_direct/spectral/table_supersonic_primary_modal_spectral_133pts.csv"
)
SUPERSONIC_MODAL = (
    REPO_ROOT
    / "experiments/modal_reconstruction/support/modal/supersonic_reference_v2_modal_raw.parquet"
)
SUBSONIC_SPECTRAL = (
    REPO_ROOT / "assets/classic_subsonic/data/subsonic_hybrid_growth_map.csv"
)
POINTWISE_DIR = REPO_ROOT / "classic_supersonic/configs/convergence/pointwise"
SUBSONIC_OUTPUT_DIR = (
    REPO_ROOT / "assets/classic_subsonic/article/convergence"
)

SUBSONIC_MACHS = tuple(np.round(np.arange(0.1, 1.0, 0.1), 1))
SUPERSONIC_MACHS = tuple(np.round(np.arange(1.1, 2.0, 0.1), 1))
SELECTION_COLUMNS = [
    "case_id",
    "regime",
    "Mach",
    "alpha",
    "reference_cr",
    "reference_ci",
    "reference_omega_i",
    "validation_status",
    "spectral_source",
    "modal_source",
    "branch_identifier",
    "selection_rank",
    "selection_rule",
]


def point_key(mach: float, alpha: float) -> tuple[float, float]:
    return round(float(mach), 8), round(float(alpha), 8)


def _case_id(regime: str, mach: float) -> str:
    return f"{regime}_M{int(round(100.0 * mach)):03d}"


def build_subsonic_candidates() -> pd.DataFrame:
    source = pd.read_csv(SUBSONIC_SPECTRAL)
    required = {"Mach", "alpha", "ci", "omega_i", "source", "primary_success"}
    missing = required - set(source.columns)
    if missing:
        raise KeyError(f"Subsonic canonical table is missing {sorted(missing)}")
    for column in ("Mach", "alpha", "ci", "omega_i"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source = source.dropna(subset=["Mach", "alpha", "ci"])
    mach_grid = np.sort(source["Mach"].unique())
    alpha_grid = np.sort(source["alpha"].unique())

    records: list[dict[str, object]] = []
    for target_mach in SUBSONIC_MACHS:
        lower_values = mach_grid[mach_grid <= target_mach]
        upper_values = mach_grid[mach_grid >= target_mach]
        if not len(lower_values) or not len(upper_values):
            raise RuntimeError(f"Cannot bracket subsonic Mach={target_mach}")
        lower = float(lower_values[-1])
        upper = float(upper_values[0])
        for alpha in alpha_grid:
            rows = source[
                source["Mach"].isin([lower, upper])
                & np.isclose(source["alpha"], alpha)
            ].sort_values("Mach")
            if rows.empty or len(rows) != (1 if lower == upper else 2):
                continue
            ci = float(np.interp(target_mach, rows["Mach"], rows["ci"]))
            if not np.isfinite(ci) or ci <= 0.0:
                continue
            records.append(
                {
                    "case_id": _case_id("subsonic", target_mach),
                    "regime": "subsonic",
                    "Mach": target_mach,
                    "alpha": round(float(alpha), 8),
                    "reference_cr": 0.0,
                    "reference_ci": ci,
                    "reference_omega_i": round(float(alpha), 8) * ci,
                    "validation_status": "canonical_M_interpolation_spectral_modal_reconstructible",
                    "spectral_source": str(SUBSONIC_SPECTRAL.relative_to(REPO_ROOT)),
                    "modal_source": "classical_solver.subsonic.mstab17_subsonic_solver:Mstab17SubsonicSolver",
                    "branch_identifier": (
                        f"subsonic_cr0_M{target_mach:.2f}_from_M{lower:.2f}_M{upper:.2f}"
                    ),
                    "selection_rule": "max(alpha*ci) on canonical alpha grid after linear Mach interpolation",
                    "source_mach_lower": lower,
                    "source_mach_upper": upper,
                }
            )
    candidates = pd.DataFrame(records)
    candidates["selection_rank"] = (
        candidates.groupby("Mach")["reference_omega_i"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return candidates[candidates["selection_rank"] <= 3].sort_values(
        ["Mach", "selection_rank"]
    ).reset_index(drop=True)


def build_supersonic_candidates() -> pd.DataFrame:
    spectral = pd.read_csv(SUPERSONIC_SPECTRAL)
    required = {"Mach", "alpha", "cr", "ci", "omega_i", "validation_status"}
    missing = required - set(spectral.columns)
    if missing:
        raise KeyError(f"Supersonic canonical table is missing {sorted(missing)}")
    for column in ("Mach", "alpha", "cr", "ci", "omega_i"):
        spectral[column] = pd.to_numeric(spectral[column], errors="coerce")

    modal = pd.read_parquet(SUPERSONIC_MODAL, columns=[
        "Mach", "alpha", "y", "p_real", "p_imag", "rho_real", "rho_imag",
        "u_real", "u_imag", "v_real", "v_imag",
    ])
    modal_keys = {
        point_key(mach, alpha)
        for mach, alpha in modal[["Mach", "alpha"]].drop_duplicates().itertuples(index=False)
    }
    accepted_statuses = {
        "modal_spectral_validated_with_exported_fields",
        "validated_core_stable_tail_sensitive",
    }
    admissible = spectral[
        spectral["validation_status"].isin(accepted_statuses)
        & spectral[["Mach", "alpha", "cr", "ci", "omega_i"]].notna().all(axis=1)
    ].copy()
    admissible = admissible[
        [point_key(m, a) in modal_keys for m, a in admissible[["Mach", "alpha"]].itertuples(index=False)]
    ].copy()

    records: list[dict[str, object]] = []
    for target_mach in SUPERSONIC_MACHS:
        rows = admissible[np.isclose(admissible["Mach"], target_mach)].copy()
        if rows.empty:
            raise RuntimeError(f"No modal-spectral canonical candidate for Mach={target_mach}")
        rows = rows.sort_values(["omega_i", "alpha"], ascending=[False, True]).head(3)
        for rank, (_, row) in enumerate(rows.iterrows(), start=1):
            point_id = row.get("point_id")
            if not isinstance(point_id, str) or not point_id.strip():
                point_id = f"M{target_mach:.2f}_a{float(row['alpha']):.6f}"
            records.append(
                {
                    "case_id": _case_id("supersonic", target_mach),
                    "regime": "supersonic",
                    "Mach": target_mach,
                    "alpha": float(row["alpha"]),
                    "reference_cr": float(row["cr"]),
                    "reference_ci": float(row["ci"]),
                    "reference_omega_i": float(row["omega_i"]),
                    "validation_status": str(row["validation_status"]),
                    "spectral_source": str(SUPERSONIC_SPECTRAL.relative_to(REPO_ROOT)),
                    "modal_source": str(SUPERSONIC_MODAL.relative_to(REPO_ROOT)),
                    "branch_identifier": point_id,
                    "selection_rank": rank,
                    "selection_rule": "max(alpha*ci) among canonical modal-spectral points",
                    "canonical_stage1_mismatch": (
                        row.get("best_stage1_mismatch")
                        if pd.notna(row.get("best_stage1_mismatch"))
                        else row.get("max_stage1")
                    ),
                    "best_stage2_mismatch": row.get("best_stage2_mismatch"),
                    "canonical_provenance_source": row.get("source"),
                }
            )
    return pd.DataFrame(records).sort_values(["Mach", "selection_rank"]).reset_index(drop=True)


def apply_provenance_selection(
    candidates: pd.DataFrame,
    provenance_path: Path | None,
) -> pd.DataFrame:
    if provenance_path is None or not provenance_path.exists():
        return candidates[candidates["selection_rank"].eq(1)].copy()
    audit = pd.read_csv(provenance_path)
    required = {"Mach", "selection_rank", "reproduced"}
    missing = required - set(audit.columns)
    if missing:
        raise KeyError(f"Provenance audit is missing {sorted(missing)}")
    audit["reproduced"] = audit["reproduced"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    chosen = (
        audit[audit["reproduced"]]
        .sort_values(["Mach", "selection_rank"])
        .drop_duplicates("Mach", keep="first")
    )
    if set(np.round(chosen["Mach"], 1)) != set(SUPERSONIC_MACHS):
        missing_mach = sorted(set(SUPERSONIC_MACHS) - set(np.round(chosen["Mach"], 1)))
        raise RuntimeError(f"No reproduced candidate for Mach values {missing_mach}")
    keys = {
        (round(float(row.Mach), 8), int(row.selection_rank))
        for row in chosen.itertuples(index=False)
    }
    selected = candidates[
        [
            (round(float(mach), 8), int(rank)) in keys
            for mach, rank in candidates[["Mach", "selection_rank"]].itertuples(index=False)
        ]
    ].copy()
    provenance_by_mach = {
        round(float(row.Mach), 8): str(getattr(row, "provenance_mode", "unknown"))
        for row in chosen.itertuples(index=False)
    }
    selected["provenance_mode"] = selected["Mach"].map(
        lambda value: provenance_by_mach[round(float(value), 8)]
    )
    selected["validation_status"] = selected["validation_status"].astype(str) + selected[
        "provenance_mode"
    ].map(
        {
            "rerun_current_solver": "+current_shooting_reproduced",
            "reused_canonical_shooting": "+canonical_shooting_provenance_reused",
        }
    ).fillna("+shooting_provenance_unknown")
    return selected


def write_source_audit(sub_candidates: pd.DataFrame, sup_candidates: pd.DataFrame) -> Path:
    sub_source = pd.read_csv(SUBSONIC_SPECTRAL)
    sup_source = pd.read_csv(SUPERSONIC_SPECTRAL)
    modal_counts = (
        pd.read_parquet(SUPERSONIC_MODAL, columns=["Mach", "alpha"])
        .drop_duplicates()
        .groupby("Mach")
        .size()
    )
    lines = [
        "# Pointwise canonical source audit",
        "",
        "## Subsonic",
        f"- Path: `{SUBSONIC_SPECTRAL.relative_to(REPO_ROOT)}`",
        f"- Columns: `{', '.join(sub_source.columns)}`",
        f"- Rows: {len(sub_source)}; points per source Mach: {sub_source.groupby('Mach').size().to_dict()}",
        "- Spectral status: `primary_success` plus optional mstab17 secondary diagnostics.",
        "- Modal fields: not stored in this CSV; p, rho, u and v are reconstructible with Mstab17SubsonicSolver.",
        "- Provenance: output of `classical_solver/subsonic/hybrid_subsonic_scan.py`.",
        "- Limitation: requested Mach values are not all table nodes. Linear interpolation is applied only in Mach; alpha is never interpolated.",
        "",
        "## Supersonic",
        f"- Spectral path: `{SUPERSONIC_SPECTRAL.relative_to(REPO_ROOT)}`",
        f"- Spectral columns: `{', '.join(sup_source.columns)}`",
        f"- Rows per Mach: {sup_source.groupby('Mach').size().to_dict()}",
        f"- Modal path: `{SUPERSONIC_MODAL.relative_to(REPO_ROOT)}`",
        f"- Unique modal points per Mach: {modal_counts.to_dict()}",
        "- Spectral/modal status: the two accepted canonical statuses are modal-spectral validated and core-modal validated/tail-sensitive.",
        "- Modal fields: p, rho, u and v real/imaginary arrays are present in the canonical Parquet.",
        "- Provenance: current 133-point primary spectral table and current v2 raw modal Parquet; UNFROZEN archives are not used.",
        "",
        "## Selection coverage",
        f"- Subsonic rank-1 coverage: {sub_candidates[sub_candidates.selection_rank.eq(1)].Mach.tolist()}",
        f"- Supersonic rank-1 coverage: {sup_candidates[sup_candidates.selection_rank.eq(1)].Mach.tolist()}",
        "- Some Mach values have fewer than three admissible canonical candidates; no placeholder candidate is invented.",
    ]
    path = POINTWISE_DIR / "canonical_source_audit.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_cases_config(sub_selected: pd.DataFrame, sup_selected: pd.DataFrame) -> Path:
    cases: list[dict[str, object]] = []
    for row in sub_selected.sort_values("Mach").to_dict("records"):
        cases.append(
            {
                **{key: row[key] for key in SELECTION_COLUMNS if key != "selection_rule"},
                "seed_cr": float(row["reference_cr"]),
                "seed_ci": float(row["reference_ci"]),
                "branch_provenance_status": "canonical_M_interpolation_seed",
                "allowed_solvers": ["subsonic_mstab17"],
                "branch_check": {
                    "max_absolute_distance": 0.04,
                    "max_relative_distance": 0.35,
                },
                "nominal": {
                    "subsonic_mstab17": {
                        "Ly": 80.0,
                        "matching_y": 1.0,
                        "rtol": 1.0e-10,
                        "atol": 1.0e-12,
                        "max_step": None,
                        "n_scan": 61,
                        "ci_min": 1.0e-3,
                        "ci_max": 1.0,
                    }
                },
            }
        )
    for row in sup_selected.sort_values("Mach").to_dict("records"):
        high_mach = float(row["Mach"]) >= 1.8
        cases.append(
            {
                **{key: row[key] for key in SELECTION_COLUMNS if key != "selection_rule"},
                "seed_cr": float(row["reference_cr"]),
                "seed_ci": float(row["reference_ci"]),
                "branch_provenance_status": (
                    "resolved_current_shooting"
                    if "current_shooting_reproduced" in str(row["validation_status"])
                    else "resolved_reused_canonical_shooting"
                    if "canonical_shooting_provenance_reused" in str(row["validation_status"])
                    else "pending_current_shooting_audit"
                ),
                "allowed_solvers": ["supersonic_shooting"],
                "branch_check": {
                    "max_absolute_distance": 0.04,
                    "max_relative_distance": 0.35,
                },
                "nominal": {
                    "supersonic_shooting": {
                        "Ly": 2000.0 if high_mach else 500.0,
                        "matching_y": 1.0,
                        "rtol": 1.0e-10,
                        "atol": 1.0e-12,
                        "max_step": 0.25,
                        "mapping_kind": "pin",
                        "mapping_scale": 3.0 if high_mach else 5.0,
                        "y_limit_factor": 10.0 if high_mach else 6.0,
                        "cr_half_width": 0.025,
                        "ci_half_width": 0.015,
                        "grid_size": 5,
                        "max_iter": 20,
                        "search_tolerance": 1.0e-11 if high_mach else 1.0e-9,
                    }
                },
            }
        )
    document = {
        "version": 1,
        "selection_rule": "pointwise maximum temporal growth on canonical admissible points",
        "excluded_fixed_ci_points": [
            {"Mach": 1.8, "alpha": 0.238},
            {"Mach": 1.8, "alpha": 0.248},
            {"Mach": 1.8, "alpha": 0.258},
            {"Mach": 1.9, "alpha": 0.232},
            {"Mach": 1.9, "alpha": 0.242},
            {"Mach": 1.9, "alpha": 0.252},
        ],
        "cases": cases,
    }
    path = POINTWISE_DIR / "pointwise_cases.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build traceable pointwise KH case selections.")
    parser.add_argument("--provenance-audit", type=Path, default=None)
    args = parser.parse_args()
    POINTWISE_DIR.mkdir(parents=True, exist_ok=True)
    SUBSONIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sub_candidates = build_subsonic_candidates()
    sup_candidates = build_supersonic_candidates()
    sub_selected = sub_candidates[sub_candidates["selection_rank"].eq(1)].copy()
    sup_selected = apply_provenance_selection(sup_candidates, args.provenance_audit)

    if len(sub_selected) != 9 or len(sup_selected) != 9:
        raise RuntimeError("Pointwise selections must contain exactly nine points per regime.")
    if sub_selected["Mach"].nunique() != 9 or sup_selected["Mach"].nunique() != 9:
        raise RuntimeError("Pointwise selections must contain exactly one point per Mach.")

    sub_candidates.to_csv(
        SUBSONIC_OUTPUT_DIR / "subsonic_pointwise_candidate_ranking.csv", index=False
    )
    sup_candidates.to_csv(POINTWISE_DIR / "supersonic_pointwise_candidate_ranking.csv", index=False)
    sub_selected[SELECTION_COLUMNS].to_csv(
        SUBSONIC_OUTPUT_DIR / "subsonic_selected_points.csv", index=False
    )
    sup_selected[SELECTION_COLUMNS].to_csv(
        POINTWISE_DIR / "supersonic_selected_points.csv", index=False
    )
    audit = write_source_audit(sub_candidates, sup_candidates)
    cases_path = write_cases_config(sub_selected, sup_selected)
    print(f"Wrote {audit}")
    print(f"Wrote {cases_path}")
    print(sub_selected[SELECTION_COLUMNS].to_string(index=False))
    print(sup_selected[SELECTION_COLUMNS].to_string(index=False))


if __name__ == "__main__":
    main()
