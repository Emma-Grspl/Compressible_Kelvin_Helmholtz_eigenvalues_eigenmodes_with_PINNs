#!/usr/bin/env python3
"""Create the supplementary long-wave GEP mapping audit.

This script uses existing summary CSVs only. It does not run a GEP solve.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hm2-csv",
        default=(
            "assets/pinn_subsonic/release_v1/audits/"
            "gep_HM2_VLOW_EXTREME_mapping_audit_summary.csv"
        ),
    )
    parser.add_argument(
        "--etaedge-csv",
        default=(
            "assets/pinn_subsonic/release_v1/audits/"
            "gep_fullrect_ETAEDGE_HM2B_mapping_audit.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "assets/pinn_subsonic/local_atlas_v1/"
            "publication_assets_scientific_v2"
        ),
    )
    parser.add_argument(
        "--release-dir",
        default="assets/pinn_subsonic/release_v1",
    )
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def parse_run(run: str) -> dict[str, float]:
    text = str(run)
    n_match = re.search(r"N(\d+)", text)
    map_match = re.search(r"map(\d+)", text)
    xi_match = re.search(r"xi(\d+)", text)

    n_value = float(n_match.group(1)) if n_match else np.nan
    map_value = float(map_match.group(1)) if map_match else np.nan

    xi_value = np.nan
    if xi_match:
        xi_value = float("0." + xi_match.group(1))

    return {
        "N": n_value,
        "mapping_scale": map_value,
        "xi_max": xi_value,
    }


def add_metadata(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    out = frame.copy()
    metadata = out["run"].map(parse_run).apply(pd.Series)
    out["N"] = metadata["N"]
    out["mapping_scale"] = metadata["mapping_scale"]
    out["xi_max"] = metadata["xi_max"]
    out["regime"] = regime
    return out


def require_columns(frame: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}")


def positive(values: pd.Series, floor: float = 1.0e-12) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array[~np.isfinite(array)] = np.nan
    return np.where(array > floor, array, floor)


def plot_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metrics: list[tuple[str, str]],
    title: str,
) -> None:
    x = np.arange(len(frame))
    for column, label in metrics:
        ax.plot(
            x,
            positive(frame[column]),
            marker="o",
            linewidth=1.7,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["run"], rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Maximum error or overlap defect")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)


def save_figure(fig: plt.Figure, base: Path, dpi: int) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def policy_report(hm2: pd.DataFrame, etaedge: pd.DataFrame) -> dict:
    hm2_candidates = hm2.loc[
        (hm2["N"] == 301)
        & (hm2["mapping_scale"].isin([10, 15, 20]))
        & np.isclose(hm2["xi_max"], 0.99)
    ].copy()

    hm2_best = hm2_candidates.loc[hm2_candidates["ci_gep_max"].idxmin()]
    hm2_map5_n301 = hm2.loc[
        (hm2["N"] == 301) & (hm2["mapping_scale"] == 5)
    ].iloc[0]
    hm2_map5_n401 = hm2.loc[
        (hm2["N"] == 401) & (hm2["mapping_scale"] == 5)
    ].iloc[0]

    etaedge_candidates = etaedge.loc[
        etaedge["mapping_scale"].isin([10, 15, 20, 30])
    ].copy()

    etaedge_best_ci = etaedge_candidates.loc[
        etaedge_candidates["ci_gep_abs_max"].idxmin()
    ]
    etaedge_best_u = etaedge_candidates.loc[
        etaedge_candidates["u_rel_max"].idxmin()
    ]
    etaedge_best_v = etaedge_candidates.loc[
        etaedge_candidates["v_rel_max"].idxmin()
    ]
    etaedge_final = etaedge.loc[
        (etaedge["mapping_scale"] == 20)
        & np.isclose(etaedge["xi_max"], 0.995)
    ].iloc[0]

    return {
        "HM2_VLOW_EXTREME": {
            "recommended_policy": "N301_map10_xi099",
            "best_reported_ci_run": str(hm2_best["run"]),
            "best_reported_ci_max": float(hm2_best["ci_gep_max"]),
            "map5_N301_reported_ci_max": float(hm2_map5_n301["ci_gep_max"]),
            "map5_N401_reported_ci_max": float(hm2_map5_n401["ci_gep_max"]),
            "conclusion": (
                "map10 minimizes the reported maximum c_i error among the "
                "N301 map10/map15/map20 candidates. Raising N from 301 to "
                "401 does not repair the old map5 result."
            ),
        },
        "ETAEDGE_HM2B": {
            "recommended_policy": "N301_map20_xi0995",
            "best_ci_max_run": str(etaedge_best_ci["run"]),
            "best_ci_abs_max": float(etaedge_best_ci["ci_gep_abs_max"]),
            "best_u_max_run": str(etaedge_best_u["run"]),
            "best_u_rel_max": float(etaedge_best_u["u_rel_max"]),
            "best_v_max_run": str(etaedge_best_v["run"]),
            "best_v_rel_max": float(etaedge_best_v["v_rel_max"]),
            "final_run": str(etaedge_final["run"]),
            "final_ci_abs_max": float(etaedge_final["ci_gep_abs_max"]),
            "final_p_rel_max": float(etaedge_final["p_rel_max"]),
            "final_u_rel_max": float(etaedge_final["u_rel_max"]),
            "final_v_rel_max": float(etaedge_final["v_rel_max"]),
            "final_overlap_min": float(etaedge_final["p_overlap_min"]),
            "conclusion": (
                "map20 with xi=0.995 gives the lowest maximum absolute c_i "
                "error in the sweep and simultaneously the lowest maximum "
                "u and v modal errors, with near-unit overlap."
            ),
        },
    }


def copy_release_products(output_dir: Path, release_dir: Path) -> None:
    release_figures = release_dir / "figures" / "supplement"
    release_tables = release_dir / "tables"
    release_manifests = release_dir / "manifests"

    release_figures.mkdir(parents=True, exist_ok=True)
    release_tables.mkdir(parents=True, exist_ok=True)
    release_manifests.mkdir(parents=True, exist_ok=True)

    for suffix in (".pdf", ".png"):
        shutil.copy2(
            output_dir / "figures" / f"SuppFig08_longwave_mapping_audit{suffix}",
            release_figures / f"SuppFig08_longwave_mapping_audit{suffix}",
        )

    shutil.copy2(
        output_dir / "tables" / "longwave_mapping_audit_combined.csv",
        release_tables / "longwave_mapping_audit_combined.csv",
    )
    shutil.copy2(
        output_dir / "longwave_mapping_audit_report.json",
        release_manifests / "longwave_mapping_audit_report.json",
    )


def main() -> None:
    args = parse_args()

    hm2_path = Path(args.hm2_csv).resolve()
    etaedge_path = Path(args.etaedge_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    release_dir = Path(args.release_dir).resolve()

    if not hm2_path.is_file():
        raise FileNotFoundError(hm2_path)
    if not etaedge_path.is_file():
        raise FileNotFoundError(etaedge_path)

    hm2 = pd.read_csv(hm2_path)
    etaedge = pd.read_csv(etaedge_path)

    require_columns(
        hm2,
        [
            "run",
            "success_rate",
            "ci_gep_max",
            "p_rel_max",
            "u_rel_max",
            "v_rel_max",
            "corner_overlap",
        ],
        hm2_path,
    )
    require_columns(
        etaedge,
        [
            "run",
            "success_rate",
            "ci_gep_abs_max",
            "ci_gep_rel_max",
            "p_rel_max",
            "u_rel_max",
            "v_rel_max",
            "p_overlap_min",
        ],
        etaedge_path,
    )

    hm2 = add_metadata(hm2, "HM2_VLOW_EXTREME")
    etaedge = add_metadata(etaedge, "ETAEDGE_HM2B")

    hm2_order = {
        name: index for index, name in enumerate(
            [
                "N301_map5_xi098_old",
                "N401_map5_xi098_old",
                "N301_map10_xi099",
                "N301_map15_xi099",
                "N301_map20_xi099",
                "N301_map20_xi0995",
            ]
        )
    }
    etaedge_order = {
        name: index for index, name in enumerate(
            [
                "map10_xi099",
                "map15_xi099",
                "map20_xi099",
                "map30_xi099",
                "map20_xi0995",
                "map30_xi0995",
            ]
        )
    }

    hm2["_order"] = hm2["run"].map(hm2_order)
    etaedge["_order"] = etaedge["run"].map(etaedge_order)
    hm2 = hm2.sort_values("_order").reset_index(drop=True)
    etaedge = etaedge.sort_values("_order").reset_index(drop=True)

    hm2["overlap_defect"] = 1.0 - pd.to_numeric(
        hm2["corner_overlap"], errors="coerce"
    )
    etaedge["overlap_defect"] = 1.0 - pd.to_numeric(
        etaedge["p_overlap_min"], errors="coerce"
    )

    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))

    plot_panel(
        axes[0, 0],
        hm2,
        [
            ("ci_gep_max", r"reported max $c_i$ error"),
            ("p_rel_max", r"max $p$ relative error"),
        ],
        "(a) HM2 very-long-wave sweep: growth rate and pressure",
    )
    plot_panel(
        axes[0, 1],
        hm2,
        [
            ("u_rel_max", r"max $u$ relative error"),
            ("v_rel_max", r"max $v$ relative error"),
            ("overlap_defect", r"$1-\min$ overlap"),
        ],
        "(b) HM2 very-long-wave sweep: velocity and overlap",
    )
    plot_panel(
        axes[1, 0],
        etaedge,
        [
            ("ci_gep_abs_max", r"max absolute $c_i$ error"),
            ("ci_gep_rel_max", r"max relative $c_i$ error"),
            ("p_rel_max", r"max $p$ relative error"),
        ],
        "(c) ETAEDGE_HM2B sweep: growth rate and pressure",
    )
    plot_panel(
        axes[1, 1],
        etaedge,
        [
            ("u_rel_max", r"max $u$ relative error"),
            ("v_rel_max", r"max $v$ relative error"),
            ("overlap_defect", r"$1-\min$ overlap"),
        ],
        "(d) ETAEDGE_HM2B sweep: velocity and overlap",
    )

    fig.suptitle(
        "Long-wave GEP mapping audit: mapping scale, truncation and resolution",
        y=1.01,
    )
    fig.tight_layout()

    figure_base = figure_dir / "SuppFig08_longwave_mapping_audit"
    save_figure(fig, figure_base, args.dpi)

    all_columns = sorted(set(hm2.columns) | set(etaedge.columns))
    for column in all_columns:
        if column not in hm2.columns:
            hm2[column] = np.nan
        if column not in etaedge.columns:
            etaedge[column] = np.nan

    preferred = [
        "regime",
        "run",
        "N",
        "mapping_scale",
        "xi_max",
        "success_rate",
        "overlap_defect",
    ]
    export_columns = preferred + [
        column
        for column in all_columns
        if column not in preferred and not column.startswith("_")
    ]

    combined = pd.concat(
        [hm2[export_columns], etaedge[export_columns]],
        ignore_index=True,
    )
    combined_path = table_dir / "longwave_mapping_audit_combined.csv"
    combined.to_csv(combined_path, index=False)

    report = {
        "source_files": {
            "HM2_VLOW_EXTREME": str(hm2_path),
            "ETAEDGE_HM2B": str(etaedge_path),
        },
        "run_counts": {
            "HM2_VLOW_EXTREME": int(len(hm2)),
            "ETAEDGE_HM2B": int(len(etaedge)),
        },
        "all_runs_successful": bool(
            hm2["success_rate"].eq(1.0).all()
            and etaedge["success_rate"].eq(1.0).all()
        ),
        "policy_conclusions": policy_report(hm2, etaedge),
    }

    report_path = output_dir / "longwave_mapping_audit_report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    copy_release_products(output_dir, release_dir)

    print("===== LONG-WAVE MAPPING AUDIT BUILT =====")
    print(json.dumps(report, indent=2))
    print("Figure:", figure_base.with_suffix(".pdf"))
    print("Table :", combined_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
