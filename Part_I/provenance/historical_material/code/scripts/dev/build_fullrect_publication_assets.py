#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


RECTANGLE_SOURCES = (
    ("core", "atlas_manifest.csv"),
    ("extension", "atlas_extension_plan_M005_098_eta005_098.csv"),
    ("fullrect", "atlas_fullrect_plan_M002_098_eta002_098.csv"),
)

RECTANGLE_ALIASES = {
    "chart_id": ("chart_id", "atlas_id", "name"),
    "M_min": ("M_min", "mach_min", "Mach_min"),
    "M_max": ("M_max", "mach_max", "Mach_max"),
    "eta_min": ("eta_min", "Eta_min"),
    "eta_max": ("eta_max", "Eta_max"),
    "path": ("path", "output_dir", "run_dir"),
    "status": ("status", "chart_status"),
    "priority": ("priority",),
}

REGIME_LABELS = {
    0: "standard_N301",
    1: "longwave_map10_N301",
    2: "extreme_longwave_map20_N301",
    3: "near_neutral_N401",
}

FIGURE_DPI = 320


def first_existing_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": FIGURE_DPI,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def normalize_rectangles(atlas_root: Path) -> pd.DataFrame:
    rows: list[dict] = []

    for source_name, filename in RECTANGLE_SOURCES:
        path = atlas_root / filename
        if not path.exists():
            print(f"[WARN] rectangle source missing: {path}")
            continue

        frame = pd.read_csv(path)
        resolved = {
            key: first_existing_column(frame, aliases)
            for key, aliases in RECTANGLE_ALIASES.items()
        }

        required = ("chart_id", "M_min", "M_max", "eta_min", "eta_max")
        missing = [key for key in required if resolved[key] is None]
        if missing:
            print(
                f"[WARN] cannot normalize {path}; missing {missing}; "
                f"columns={list(frame.columns)}"
            )
            continue

        for _, row in frame.iterrows():
            normalized = {
                "source": source_name,
                "chart_id": str(row[resolved["chart_id"]]),
                "M_min": float(row[resolved["M_min"]]),
                "M_max": float(row[resolved["M_max"]]),
                "eta_min": float(row[resolved["eta_min"]]),
                "eta_max": float(row[resolved["eta_max"]]),
                "path": "",
                "status": "",
                "priority": np.nan,
            }
            if resolved["path"] is not None and pd.notna(row[resolved["path"]]):
                normalized["path"] = str(row[resolved["path"]])
            if resolved["status"] is not None and pd.notna(row[resolved["status"]]):
                normalized["status"] = str(row[resolved["status"]])
            if resolved["priority"] is not None and pd.notna(row[resolved["priority"]]):
                normalized["priority"] = float(row[resolved["priority"]])
            normalized["area"] = (
                (normalized["M_max"] - normalized["M_min"])
                * (normalized["eta_max"] - normalized["eta_min"])
            )
            rows.append(normalized)

    rectangles = pd.DataFrame(rows)
    if rectangles.empty:
        raise RuntimeError("No rectangle definitions could be loaded.")

    has_ultralow = (
        (rectangles["M_min"] <= 0.02)
        & (rectangles["M_max"] >= 0.06)
        & (rectangles["eta_min"] <= 0.02)
        & (rectangles["eta_max"] >= 0.06)
    ).any()

    if not has_ultralow:
        rectangles = pd.concat(
            [
                rectangles,
                pd.DataFrame(
                    [
                        {
                            "source": "ultralow_explicit",
                            "chart_id": "ULTRALOW_M002_006_eta002_006",
                            "M_min": 0.02,
                            "M_max": 0.06,
                            "eta_min": 0.02,
                            "eta_max": 0.06,
                            "path": "",
                            "status": "validated",
                            "priority": 200.0,
                            "area": 0.04 * 0.04,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    rectangles = (
        rectangles.drop_duplicates(
            ["source", "chart_id", "M_min", "M_max", "eta_min", "eta_max"]
        )
        .sort_values(["source", "M_min", "eta_min", "chart_id"])
        .reset_index(drop=True)
    )
    return rectangles


def classify_regime(mach: np.ndarray, eta: np.ndarray) -> np.ndarray:
    mach = np.asarray(mach, dtype=float)
    eta = np.asarray(eta, dtype=float)
    alpha = eta * np.sqrt(np.maximum(0.0, 1.0 - mach**2))

    regime = np.zeros(np.broadcast(mach, eta).shape, dtype=int)
    regime[(eta <= 0.06) | (alpha < 0.06)] = 1
    regime[(mach >= 0.92) & (eta <= 0.06)] = 2
    regime[eta >= 0.96] = 3
    return regime


def regime_parameters(regime_code: int) -> tuple[int, float, float, bool]:
    if regime_code == 3:
        return 401, 5.0, 0.98, True
    if regime_code == 2:
        return 301, 20.0, 0.995, False
    if regime_code == 1:
        return 301, 10.0, 0.99, False
    return 301, 5.0, 0.98, False


def build_or_load_coverage(atlas_root: Path, rectangles: pd.DataFrame) -> pd.DataFrame:
    candidates = (
        atlas_root / "atlas_fullrect_final_coverage_grid.csv",
        atlas_root / "release_fullrect_v1" / "atlas_fullrect_final_coverage_grid.csv",
    )
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            if {"Mach", "eta", "coverage_count"}.issubset(frame.columns):
                return frame

    mach_values = np.round(np.arange(0.02, 0.9800001, 0.0025), 7)
    eta_values = np.round(np.arange(0.02, 0.9800001, 0.0025), 7)
    grid_mach, grid_eta = np.meshgrid(mach_values, eta_values)
    coverage_count = np.zeros_like(grid_mach, dtype=np.int16)

    # Vectorized fallback: increment the full grid once per rectangle.
    for _, rectangle in rectangles.iterrows():
        inside = (
            (grid_mach >= float(rectangle["M_min"]) - 1e-12)
            & (grid_mach <= float(rectangle["M_max"]) + 1e-12)
            & (grid_eta >= float(rectangle["eta_min"]) - 1e-12)
            & (grid_eta <= float(rectangle["eta_max"]) + 1e-12)
        )
        coverage_count[inside] += 1

    return pd.DataFrame(
        {
            "Mach": grid_mach.ravel(),
            "eta": grid_eta.ravel(),
            "alpha": (
                grid_eta * np.sqrt(np.maximum(0.0, 1.0 - grid_mach**2))
            ).ravel(),
            "coverage_count": coverage_count.ravel(),
            "covering_charts": "",
        }
    )


def load_required_frame(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
    return frame


def normalized_release(release: pd.DataFrame) -> pd.DataFrame:
    release = release.copy()
    if "ci_final" not in release.columns:
        release["ci_final"] = release["gep_ci"]
    if "ci_final_source" not in release.columns:
        release["ci_final_source"] = "independent_GEP"

    for final_name, raw_name in (
        ("p_rel_final", "p_rel"),
        ("rho_rel_final", "rho_rel"),
        ("u_rel_final", "u_rel"),
        ("v_rel_final", "v_rel"),
        ("p_overlap_final", "p_overlap"),
    ):
        if final_name not in release.columns:
            release[final_name] = release[raw_name]

    release["ci_final_abs_err_vs_raw_classic"] = (
        release["ci_final"] - release["ci_classic"]
    ).abs()
    release["ci_final_rel_err_vs_raw_classic"] = (
        release["ci_final_abs_err_vs_raw_classic"]
        / release["ci_classic"].abs().clip(lower=1e-12)
    )
    release["modal_error_max"] = release[
        ["p_rel_final", "rho_rel_final", "u_rel_final", "v_rel_final"]
    ].max(axis=1)
    release["continued"] = release.get(
        "continuation_status", pd.Series("", index=release.index)
    ).fillna("").astype(str).str.startswith("validated_")
    release["reference_corrected"] = (
        release.get("reference_status", pd.Series("", index=release.index))
        .fillna("")
        .astype(str)
        .eq("continuation_corrected_reference")
    )
    return release


def plot_chart_map(rectangles: pd.DataFrame, figure_dir: Path) -> None:
    source_styles = {
        "core": ("C0", "-"),
        "extension": ("C1", "--"),
        "fullrect": ("C2", "-."),
        "ultralow_explicit": ("C3", ":"),
    }
    fig, ax = plt.subplots(figsize=(10.5, 8.2))

    for _, row in rectangles.iterrows():
        color, linestyle = source_styles.get(str(row["source"]), ("0.4", "-"))
        rectangle = Rectangle(
            (row["M_min"], row["eta_min"]),
            row["M_max"] - row["M_min"],
            row["eta_max"] - row["eta_min"],
            fill=False,
            linewidth=1.0 if row["source"] != "fullrect" else 1.7,
            linestyle=linestyle,
            edgecolor=color,
            alpha=0.85,
        )
        ax.add_patch(rectangle)

        if row["source"] in {"fullrect", "ultralow_explicit"}:
            ax.text(
                0.5 * (row["M_min"] + row["M_max"]),
                0.5 * (row["eta_min"] + row["eta_max"]),
                str(row["chart_id"]),
                ha="center",
                va="center",
                fontsize=5.8,
                rotation=0,
                clip_on=True,
            )

    ax.set_xlim(0.015, 0.985)
    ax.set_ylim(0.015, 0.985)
    ax.set_xlabel(r"Mach number $M$")
    ax.set_ylabel(r"Scaled wavenumber $\eta$")
    ax.set_title("Local PINN atlas footprints in the operational rectangle")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        handles=[
            Patch(facecolor="none", edgecolor=color, linestyle=linestyle, label=source)
            for source, (color, linestyle) in source_styles.items()
            if source in set(rectangles["source"])
        ],
        loc="upper left",
        frameon=True,
    )
    save_figure(fig, figure_dir / "fig01_atlas_chart_footprints")


def plot_coverage(coverage: pd.DataFrame, figure_dir: Path) -> None:
    pivot = coverage.pivot_table(
        index="eta", columns="Mach", values="coverage_count", aggfunc="max"
    ).sort_index().sort_index(axis=1)
    mach = pivot.columns.to_numpy(dtype=float)
    eta = pivot.index.to_numpy(dtype=float)
    values = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    mesh = ax.pcolormesh(mach, eta, values, shading="auto", cmap="viridis")
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("Number of covering charts")
    ax.set_xlabel(r"Mach number $M$")
    ax.set_ylabel(r"Scaled wavenumber $\eta$")
    ax.set_title("Atlas coverage multiplicity")
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.02, 0.98)
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, figure_dir / "fig02_coverage_multiplicity")


def plot_regime_map(figure_dir: Path, data_dir: Path) -> pd.DataFrame:
    mach = np.linspace(0.02, 0.98, 385)
    eta = np.linspace(0.02, 0.98, 385)
    grid_mach, grid_eta = np.meshgrid(mach, eta)
    regime = classify_regime(grid_mach, grid_eta)

    cmap = plt.get_cmap("viridis", 4)
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    ax.pcolormesh(mach, eta, regime, cmap=cmap, norm=norm, shading="auto")
    ax.set_xlabel(r"Mach number $M$")
    ax.set_ylabel(r"Scaled wavenumber $\eta$")
    ax.set_title("Operational GEP regime policy")
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.02, 0.98)
    ax.set_aspect("equal", adjustable="box")

    handles = [
        Patch(facecolor=cmap(code), edgecolor="none", label=REGIME_LABELS[code])
        for code in range(4)
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True)
    save_figure(fig, figure_dir / "fig03_gep_regime_policy")

    flat = pd.DataFrame(
        {
            "Mach": grid_mach.ravel(),
            "eta": grid_eta.ravel(),
            "alpha": (
                grid_eta * np.sqrt(np.maximum(0.0, 1.0 - grid_mach**2))
            ).ravel(),
            "regime_code": regime.ravel(),
        }
    )
    flat["gep_regime"] = flat["regime_code"].map(REGIME_LABELS)
    parameters = flat["regime_code"].map(regime_parameters)
    flat[["N", "mapping_scale", "xi_max", "continuation_required"]] = pd.DataFrame(
        parameters.tolist(), index=flat.index
    )
    flat.to_csv(data_dir / "gep_regime_policy_grid.csv", index=False)
    return flat


def positive_lognorm(values: pd.Series) -> LogNorm | None:
    finite = values[np.isfinite(values.to_numpy(dtype=float)) & (values > 0)]
    if finite.empty:
        return None
    lower = max(float(finite.min()), 1e-12)
    upper = max(float(finite.max()), lower * 1.0001)
    return LogNorm(vmin=lower, vmax=upper)


def scatter_metric(
    release: pd.DataFrame,
    metric: str,
    title: str,
    colorbar_label: str,
    stem: Path,
    *,
    log_scale: bool = True,
    cmap: str = "viridis",
) -> None:
    values = release[metric].astype(float)
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    norm = positive_lognorm(values) if log_scale else None
    scatter = ax.scatter(
        release["Mach"],
        release["eta"],
        c=values,
        cmap=cmap,
        norm=norm,
        s=34,
        linewidths=0.25,
        edgecolors="black",
        alpha=0.92,
    )

    corrected = release[release["reference_corrected"]]
    if not corrected.empty:
        ax.scatter(
            corrected["Mach"],
            corrected["eta"],
            facecolors="none",
            edgecolors="red",
            linewidths=1.1,
            s=68,
            label="continuation-corrected reference",
        )
        ax.legend(loc="best", frameon=True)

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(colorbar_label)
    ax.set_xlabel(r"Mach number $M$")
    ax.set_ylabel(r"Scaled wavenumber $\eta$")
    ax.set_title(title)
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.02, 0.98)
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, stem)


def plot_near_neutral(release: pd.DataFrame, figure_dir: Path) -> None:
    neutral = release[
        (release["eta"] >= 0.94) | release["continued"]
    ].copy()
    neutral = neutral.sort_values(["eta", "Mach"])

    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    ax.scatter(
        neutral["eta"],
        neutral["ci_classic"],
        marker="o",
        facecolors="none",
        edgecolors="black",
        s=42,
        label="raw classical reference",
    )
    ax.scatter(
        neutral["eta"],
        neutral["gep_ci"],
        marker="x",
        s=38,
        label="independent GEP",
    )
    ax.scatter(
        neutral["eta"],
        neutral["ci_final"],
        marker="s",
        s=28,
        label="final branch-selected value",
    )
    ax.set_xlabel(r"Scaled wavenumber $\eta$")
    ax.set_ylabel(r"Growth-rate phase speed $c_i$")
    ax.set_title("Near-neutral branch selection")
    ax.set_xlim(max(0.935, float(neutral["eta"].min()) - 0.002), 0.982)
    ax.legend(loc="best", frameon=True)
    save_figure(fig, figure_dir / "fig07_near_neutral_ci_zoom")


def plot_reference_corrections(
    release: pd.DataFrame, figure_dir: Path, table_dir: Path
) -> pd.DataFrame:
    corrections = release[release["reference_corrected"]].copy()
    corrections["ci_correction"] = corrections["ci_final"] - corrections["ci_classic"]
    corrections["ci_correction_abs"] = corrections["ci_correction"].abs()
    corrections = corrections.sort_values("ci_correction_abs", ascending=True).reset_index(drop=True)
    corrections.to_csv(table_dir / "table04_reference_corrections.csv", index=False)

    if corrections.empty:
        return corrections

    labels = corrections["point_id"].astype(str).tolist()
    positions = np.arange(len(corrections))

    fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.27 * len(corrections) + 1.8)))
    ax.hlines(
        positions,
        corrections["ci_classic"],
        corrections["ci_final"],
        linewidth=1.2,
    )
    ax.scatter(corrections["ci_classic"], positions, marker="o", s=32, label="raw classical")
    ax.scatter(corrections["ci_final"], positions, marker="s", s=28, label="continued/final")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$c_i$")
    ax.set_title("Continuation-corrected reference points")
    ax.legend(loc="best", frameon=True)
    save_figure(fig, figure_dir / "fig08_reference_corrections")
    return corrections


def plot_offgrid_0246(atlas_root: Path, release: pd.DataFrame, figure_dir: Path) -> None:
    offgrid_root = atlas_root / "offgrid_validation_384"
    path_file = (
        offgrid_root
        / "continuation_audit_OFFGRID_0246"
        / "primary"
        / "OFFGRID_0246"
        / "continuation_paths.csv"
    )
    if not path_file.exists():
        print(f"[WARN] OFFGRID_0246 path audit missing: {path_file}")
        return

    paths = pd.read_csv(path_file)
    target = release[release["point_id"].astype(str) == "OFFGRID_0246"]
    if target.empty:
        print("[WARN] OFFGRID_0246 absent from release table")
        return
    target_row = target.iloc[0]

    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    for direction, group in paths.groupby("direction"):
        ordered = group.sort_values("step_index")
        ax.plot(ordered["eta"], ordered["gep_ci"], marker="o", markersize=2.8, label=direction)
    ax.scatter(
        [target_row["eta"]],
        [target_row["ci_classic"]],
        marker="D",
        s=58,
        label="classical target",
        zorder=5,
    )
    ax.scatter(
        [target_row["eta"]],
        [target_row["ci_final"]],
        marker="s",
        s=54,
        label="selected final branch",
        zorder=5,
    )
    ax.set_xlabel(r"Scaled wavenumber $\eta$")
    ax.set_ylabel(r"GEP phase speed $c_i$")
    ax.set_title("OFFGRID_0246: primary and secondary continuation branches")
    ax.legend(loc="best", frameon=True)
    save_figure(fig, figure_dir / "fig09_OFFGRID_0246_branch_ci")

    overlap = paths.dropna(subset=["adjacent_overlap"]).copy()
    if not overlap.empty:
        fig, ax = plt.subplots(figsize=(9.0, 5.8))
        for direction, group in overlap.groupby("direction"):
            ordered = group.sort_values("step_index")
            ax.plot(
                ordered["eta"],
                ordered["adjacent_overlap"],
                marker="o",
                markersize=2.8,
                label=direction,
            )
        ax.axhline(0.995, linestyle="--", linewidth=1.0, label="operational threshold")
        ax.set_xlabel(r"Scaled wavenumber $\eta$")
        ax.set_ylabel("Adjacent modal overlap")
        ax.set_title("OFFGRID_0246: continuation overlap audit")
        ax.set_ylim(min(0.98, float(overlap["adjacent_overlap"].min()) - 0.002), 1.0005)
        ax.legend(loc="best", frameon=True)
        save_figure(fig, figure_dir / "fig10_OFFGRID_0246_branch_overlap")


def grouped_metrics(release: pd.DataFrame, group_column: str) -> pd.DataFrame:
    metrics = (
        "ci_final_abs_err_vs_raw_classic",
        "ci_final_rel_err_vs_raw_classic",
        "p_rel_final",
        "rho_rel_final",
        "u_rel_final",
        "v_rel_final",
        "modal_error_max",
        "p_overlap_final",
    )
    rows: list[dict] = []
    for name, group in release.groupby(group_column, dropna=False):
        row: dict[str, object] = {
            group_column: name,
            "n_points": len(group),
            "n_continued": int(group["continued"].sum()),
            "n_reference_corrected": int(group["reference_corrected"].sum()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_max"] = float(group[metric].max())
            row[f"{metric}_min"] = float(group[metric].min())
        rows.append(row)
    return pd.DataFrame(rows)


def copy_if_present(source: Path, destination: Path) -> None:
    if source.exists():
        shutil.copy2(source, destination)
    else:
        print(f"[WARN] optional source missing: {source}")


def build_data_assets(
    atlas_root: Path,
    offgrid_root: Path,
    rectangles: pd.DataFrame,
    coverage: pd.DataFrame,
    release: pd.DataFrame,
    data_dir: Path,
    table_dir: Path,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    rectangles.to_csv(data_dir / "atlas_catalog.csv", index=False)
    rectangles.to_csv(data_dir / "atlas_manifest_final.csv", index=False)
    coverage.to_csv(data_dir / "coverage_grid.csv", index=False)
    release.to_csv(data_dir / "validation_pointwise.csv", index=False)

    policy_source = atlas_root / "gep_regime_policy_fullrect.csv"
    if policy_source.exists():
        shutil.copy2(policy_source, data_dir / "gep_regime_policy.csv")
    else:
        pd.DataFrame(
            [
                {
                    "regime": "standard_N301",
                    "N": 301,
                    "mapping_scale": 5.0,
                    "xi_max": 0.98,
                    "continuation_required": False,
                },
                {
                    "regime": "longwave_map10_N301",
                    "N": 301,
                    "mapping_scale": 10.0,
                    "xi_max": 0.99,
                    "continuation_required": False,
                },
                {
                    "regime": "extreme_longwave_map20_N301",
                    "N": 301,
                    "mapping_scale": 20.0,
                    "xi_max": 0.995,
                    "continuation_required": False,
                },
                {
                    "regime": "near_neutral_N401",
                    "N": 401,
                    "mapping_scale": 5.0,
                    "xi_max": 0.98,
                    "continuation_required": True,
                },
            ]
        ).to_csv(data_dir / "gep_regime_policy.csv", index=False)

    copy_if_present(
        atlas_root / "atlas_fullrect_gep_final_all_points.csv",
        data_dir / "atlas_fullrect_gep_final_all_points.csv",
    )
    copy_if_present(
        atlas_root / "atlas_fullrect_gep_final_metrics_by_chart.csv",
        data_dir / "atlas_fullrect_gep_final_metrics_by_chart.csv",
    )
    copy_if_present(
        offgrid_root / "offgrid_continuation_N401_summary_55.csv",
        data_dir / "offgrid_continuation_N401_summary_55.csv",
    )
    copy_if_present(
        offgrid_root / "offgrid_reference_branch_corrections.csv",
        data_dir / "offgrid_reference_branch_corrections.csv",
    )
    copy_if_present(
        offgrid_root / "offgrid_0246_branch_resolution.csv",
        data_dir / "offgrid_0246_branch_resolution.csv",
    )
    copy_if_present(
        offgrid_root / "offgrid_validation_release_checks.csv",
        data_dir / "release_checks.csv",
    )

    grouped_metrics(release, "gep_regime").to_csv(
        table_dir / "table01_metrics_by_gep_regime.csv", index=False
    )
    grouped_metrics(release, "chart_id").to_csv(
        table_dir / "table02_metrics_by_chart.csv", index=False
    )

    worst = release.sort_values(
        ["modal_error_max", "ci_final_abs_err_vs_raw_classic"],
        ascending=False,
    ).head(30)
    worst.to_csv(table_dir / "table03_worst_offgrid_points.csv", index=False)


def write_release_summary(
    rectangles: pd.DataFrame,
    coverage: pd.DataFrame,
    release: pd.DataFrame,
    output_path: Path,
) -> None:
    summary = {
        "domain": {"Mach": [0.02, 0.98], "eta": [0.02, 0.98]},
        "n_rectangles": int(len(rectangles)),
        "n_coverage_points": int(len(coverage)),
        "coverage_min": int(coverage["coverage_count"].min()),
        "coverage_max": int(coverage["coverage_count"].max()),
        "n_offgrid_points": int(len(release)),
        "technical_success_rate": float(parse_bool(release["success"]).mean()),
        "n_continuation_points": int(release["continued"].sum()),
        "n_reference_corrections": int(release["reference_corrected"].sum()),
        "modal_metrics": {
            "p_rel_max": float(release["p_rel_final"].max()),
            "rho_rel_max": float(release["rho_rel_final"].max()),
            "u_rel_max": float(release["u_rel_final"].max()),
            "v_rel_max": float(release["v_rel_final"].max()),
            "p_overlap_min": float(release["p_overlap_final"].min()),
        },
        "continuation_status_counts": {
            str(key): int(value)
            for key, value in release.get(
                "continuation_status", pd.Series("not_continued", index=release.index)
            )
            .fillna("not_continued")
            .value_counts()
            .items()
        },
        "release_validated": bool(
            len(release) == 384
            and release["point_id"].nunique() == 384
            and parse_bool(release["success"]).all()
            and int(coverage["coverage_count"].min()) >= 1
            and float(release["p_rel_final"].max()) <= 0.01
            and float(release["rho_rel_final"].max()) <= 0.01
            and float(release["u_rel_final"].max()) <= 0.03
            and float(release["v_rel_final"].max()) <= 0.02
            and float(release["p_overlap_final"].min()) >= 0.999
        ),
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def write_asset_manifest(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "asset_manifest.csv":
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(output_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "asset_manifest.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build publication-ready assets for the validated fullrect subsonic KH atlas."
    )
    parser.add_argument(
        "--atlas-root",
        type=Path,
        default=Path("assets/pinn_subsonic/local_atlas_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "assets/pinn_subsonic/local_atlas_v1/publication_assets_fullrect_v1"
        ),
    )
    args = parser.parse_args()

    configure_matplotlib()

    atlas_root = args.atlas_root
    offgrid_root = atlas_root / "offgrid_validation_384"
    output_dir = args.output_dir
    figure_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    table_dir = output_dir / "tables"

    for directory in (output_dir, figure_dir, data_dir, table_dir):
        directory.mkdir(parents=True, exist_ok=True)

    release_path = offgrid_root / "offgrid_validation_results_384_release.csv"
    release = load_required_frame(
        release_path,
        {
            "point_id",
            "Mach",
            "eta",
            "alpha",
            "chart_id",
            "gep_regime",
            "ci_classic",
            "gep_ci",
            "p_rel",
            "rho_rel",
            "u_rel",
            "v_rel",
            "p_overlap",
            "success",
        },
    )
    release = normalized_release(release)

    rectangles = normalize_rectangles(atlas_root)
    coverage = build_or_load_coverage(atlas_root, rectangles)

    build_data_assets(
        atlas_root,
        offgrid_root,
        rectangles,
        coverage,
        release,
        data_dir,
        table_dir,
    )

    plot_chart_map(rectangles, figure_dir)
    plot_coverage(coverage, figure_dir)
    plot_regime_map(figure_dir, data_dir)
    scatter_metric(
        release,
        "ci_final_abs_err_vs_raw_classic",
        "Off-grid absolute discrepancy in $c_i$ versus raw classical reference",
        r"$|c_i^{final}-c_i^{classic}|$",
        figure_dir / "fig04_offgrid_ci_abs_discrepancy",
        log_scale=True,
        cmap="magma",
    )
    scatter_metric(
        release,
        "modal_error_max",
        "Maximum off-grid modal relative error",
        r"$\max(\epsilon_p,\epsilon_\rho,\epsilon_u,\epsilon_v)$",
        figure_dir / "fig05_offgrid_modal_error_max",
        log_scale=True,
        cmap="viridis",
    )
    scatter_metric(
        release,
        "p_overlap_final",
        "Final pressure-mode overlap",
        "Pressure-mode overlap",
        figure_dir / "fig06_offgrid_pressure_overlap",
        log_scale=False,
        cmap="viridis",
    )
    plot_near_neutral(release, figure_dir)
    plot_reference_corrections(release, figure_dir, table_dir)
    plot_offgrid_0246(atlas_root, release, figure_dir)

    write_release_summary(
        rectangles,
        coverage,
        release,
        data_dir / "release_summary.json",
    )

    manifest = write_asset_manifest(output_dir)

    print("===== PUBLICATION ASSETS BUILT =====")
    print("Output directory:", output_dir)
    print("Figures:", len(list(figure_dir.glob("*.pdf"))), "PDF and", len(list(figure_dir.glob("*.png"))), "PNG")
    print("Data files:", len(list(data_dir.glob("*"))))
    print("Tables:", len(list(table_dir.glob("*.csv"))))
    print("Manifest rows:", len(manifest))
    print("Release rows:", len(release))
    print("Coverage min/max:", coverage["coverage_count"].min(), coverage["coverage_count"].max())
    print("Modal error max:", release["modal_error_max"].max())
    print("Pressure overlap min:", release["p_overlap_final"].min())


if __name__ == "__main__":
    main()
