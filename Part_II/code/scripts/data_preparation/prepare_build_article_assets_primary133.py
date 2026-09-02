#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


STRICT_STATUS = "modal_spectral_validated_with_exported_fields"
TAIL_STATUS = "validated_core_stable_tail_sensitive"

EXPECTED_STATUS_COUNTS = {
    STRICT_STATUS: 44,
    TAIL_STATUS: 89,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build article assets for the classical supersonic reference."
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/classic_supersonic/article"),
    )
    parser.add_argument("--mode-mach", type=float, default=1.4)
    parser.add_argument("--mode-alpha-target", type=float, default=0.18)
    parser.add_argument("--active-threshold", type=float, default=1.0e-4)
    parser.add_argument(
        "--digitization-error-mode",
        choices=("explicit", "local-spacing"),
        default="local-spacing",
        help=(
            "Use explicit error columns when present, or a documented "
            "half-local-spacing proxy for the digitized Blumen coordinates."
        ),
    )
    parser.add_argument(
        "--errorbar-stride",
        type=int,
        default=1,
        help="Draw one Blumen error bar every N digitized points.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point_key(mach: object, alpha: object) -> tuple[float, float]:
    return round(float(mach), 8), round(float(alpha), 8)


def truthy(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def locate_column(
    columns: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        match = lower.get(candidate.lower())
        if match is not None:
            return match
    return None


def load_spectral(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    required = {"Mach", "alpha", "cr", "ci", "validation_status"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing spectral columns: {sorted(missing)}")

    for column in ("Mach", "alpha", "cr", "ci", "omega_i"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "omega_i" not in frame.columns:
        frame["omega_i"] = frame["alpha"] * frame["ci"]

    frame = (
        frame.dropna(subset=["Mach", "alpha", "cr", "ci"])
        .drop_duplicates(["Mach", "alpha"])
        .sort_values(["Mach", "alpha"])
        .reset_index(drop=True)
    )

    observed = frame["validation_status"].value_counts().to_dict()
    if observed != EXPECTED_STATUS_COUNTS:
        raise RuntimeError(
            "Unexpected primary-reference status counts.\n"
            f"Observed: {observed}\n"
            f"Expected: {EXPECTED_STATUS_COUNTS}"
        )

    frame["key"] = [
        point_key(mach, alpha)
        for mach, alpha in zip(frame["Mach"], frame["alpha"])
    ]
    return frame


def transform_blumen_mach(
    values: np.ndarray,
    repo: Path,
) -> tuple[np.ndarray, str]:
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        raise RuntimeError(
            "No finite Blumen Mach coordinate."
        )

    minimum = float(np.nanmin(finite))
    maximum = float(np.nanmax(finite))

    # The canonical Blumen CSV stores physical Mach
    # coordinates and may legitimately begin slightly
    # below M=1.  Do not infer the coordinate system
    # from the minimum value alone.
    if 0.8 <= minimum and maximum <= 2.2:
        return (
            values,
            "already physical Mach",
        )

    # A genuinely normalized/digitized horizontal
    # coordinate must remain entirely near [0, 1].
    if maximum <= 1.1:
        src = repo / "classic_supersonic/src"
        sys.path.insert(
            0,
            str(src),
        )

        from classic_supersonic_reference.io.blumen_reference import (
            supersonic_digitized_x_to_mach,
        )

        transformed = (
            supersonic_digitized_x_to_mach(
                values
            )
        )

        return (
            np.asarray(
                transformed,
                dtype=float,
            ),
            "canonical repository mapping",
        )

    raise RuntimeError(
        "Ambiguous Blumen Mach coordinates: "
        f"observed range [{minimum}, {maximum}]. "
        "Refusing to apply an inferred transformation."
    )


def local_spacing_errors(group: pd.DataFrame) -> tuple[float, float]:
    mach = np.sort(group["Mach_physical"].dropna().unique())
    alpha = np.sort(group["alpha"].dropna().unique())

    dm = np.diff(mach)
    da = np.diff(alpha)

    dm = dm[np.isfinite(dm) & (dm > 0.0)]
    da = da[np.isfinite(da) & (da > 0.0)]

    xerr = 0.5 * float(np.median(dm)) if len(dm) else 0.0
    yerr = 0.5 * float(np.median(da)) if len(da) else 0.0

    return xerr, yerr


def load_blumen(
    points_path: Path,
    levels_path: Path,
    repo: Path,
    error_mode: str,
) -> tuple[pd.DataFrame, dict]:
    for path in (points_path, levels_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    points = pd.read_csv(points_path)
    levels = pd.read_csv(levels_path)

    required_points = {"curve_id", "Mach", "alpha"}
    required_levels = {"curve_id", "curve_label", "family", "ci_level"}

    missing_points = required_points - set(points.columns)
    missing_levels = required_levels - set(levels.columns)

    if missing_points:
        raise KeyError(f"Missing Blumen point columns: {sorted(missing_points)}")
    if missing_levels:
        raise KeyError(f"Missing Blumen level columns: {sorted(missing_levels)}")

    points["curve_id"] = pd.to_numeric(
        points["curve_id"], errors="coerce"
    ).astype("Int64")
    levels["curve_id"] = pd.to_numeric(
        levels["curve_id"], errors="coerce"
    ).astype("Int64")
    levels["ci_level"] = pd.to_numeric(levels["ci_level"], errors="coerce")

    include_column = locate_column(
        levels.columns.tolist(),
        ("include_in_overlay", "include_in_quantitative_comparison"),
    )
    if include_column is not None:
        levels = levels.loc[truthy(levels[include_column])].copy()

    merged = points.merge(
        levels[["curve_id", "curve_label", "family", "ci_level"]],
        on="curve_id",
        how="inner",
        validate="many_to_one",
    )

    merged = merged.loc[
        ~merged["family"].astype(str).str.strip().eq("cr_special")
    ].copy()

    mach_input = pd.to_numeric(merged["Mach"], errors="coerce").to_numpy(
        dtype=float
    )
    alpha = pd.to_numeric(merged["alpha"], errors="coerce").to_numpy(dtype=float)

    mach_physical, calibration = transform_blumen_mach(mach_input, repo)

    merged["Mach_physical"] = mach_physical
    merged["alpha"] = alpha
    merged = merged.dropna(
        subset=["Mach_physical", "alpha", "ci_level"]
    ).reset_index(drop=True)

    xerr_column = locate_column(
        merged.columns.tolist(),
        ("Mach_error", "mach_error", "xerr", "x_error"),
    )
    yerr_column = locate_column(
        merged.columns.tolist(),
        ("alpha_error", "yerr", "y_error"),
    )

    explicit_available = xerr_column is not None and yerr_column is not None

    if error_mode == "explicit" and not explicit_available:
        raise RuntimeError(
            "Explicit Blumen uncertainty columns were requested, but no "
            "Mach/x and alpha/y error columns were found."
        )

    if explicit_available:
        merged["Mach_error"] = pd.to_numeric(
            merged[xerr_column], errors="coerce"
        ).fillna(0.0)
        merged["alpha_error"] = pd.to_numeric(
            merged[yerr_column], errors="coerce"
        ).fillna(0.0)
        error_method = (
            f"explicit columns: {xerr_column!r}, {yerr_column!r}"
        )
    else:
        merged["Mach_error"] = 0.0
        merged["alpha_error"] = 0.0

        for curve_id, group in merged.groupby("curve_id", sort=False):
            xerr, yerr = local_spacing_errors(group)
            mask = merged["curve_id"].eq(curve_id)
            merged.loc[mask, "Mach_error"] = xerr
            merged.loc[mask, "alpha_error"] = yerr

        error_method = (
            "conservative digitization/sampling proxy: one half of the "
            "median positive local coordinate spacing on each digitized curve"
        )

    metadata = {
        "mach_calibration": calibration,
        "uncertainty_method": error_method,
        "point_count": int(len(merged)),
        "curve_count": int(merged["curve_id"].nunique()),
        "median_mach_error": float(merged["Mach_error"].median()),
        "median_alpha_error": float(merged["alpha_error"].median()),
    }
    return merged, metadata


def save_figure(
    figure: plt.Figure,
    pdf_path: Path,
    png_path: Path,
) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_overlay(
    spectral: pd.DataFrame,
    blumen: pd.DataFrame,
    pdf_path: Path,
    png_path: Path,
    errorbar_stride: int,
) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)

    ci_min = min(
        float(spectral["ci"].min()),
        float(blumen["ci_level"].min()),
    )
    ci_max = max(
        float(spectral["ci"].max()),
        float(blumen["ci_level"].max()),
    )
    normalization = Normalize(vmin=ci_min, vmax=ci_max)
    cmap = plt.get_cmap("viridis")

    stride = max(1, int(errorbar_stride))
    for _, group in blumen.groupby("curve_id", sort=True):
        selected = group.iloc[::stride]

        axis.errorbar(
            selected["Mach_physical"],
            selected["alpha"],
            xerr=selected["Mach_error"],
            yerr=selected["alpha_error"],
            fmt="none",
            ecolor="0.62",
            elinewidth=0.55,
            capsize=1.3,
            alpha=0.50,
            zorder=1,
        )

        axis.scatter(
            group["Mach_physical"],
            group["alpha"],
            c=group["ci_level"],
            norm=normalization,
            cmap=cmap,
            marker=".",
            s=18,
            linewidths=0.0,
            alpha=0.88,
            zorder=2,
        )

    strict = spectral.loc[
        spectral["validation_status"].eq(STRICT_STATUS)
    ]
    tail = spectral.loc[
        spectral["validation_status"].eq(TAIL_STATUS)
    ]

    classical = axis.scatter(
        strict["Mach"],
        strict["alpha"],
        c=strict["ci"],
        norm=normalization,
        cmap=cmap,
        marker="o",
        s=48,
        edgecolors="black",
        linewidths=0.55,
        label="joint modal-spectral validation (44)",
        zorder=5,
    )

    axis.scatter(
        tail["Mach"],
        tail["alpha"],
        c=tail["ci"],
        norm=normalization,
        cmap=cmap,
        marker="s",
        s=40,
        edgecolors="black",
        linewidths=0.45,
        label="spectral and core-modal validation; tail-sensitive (89)",
        zorder=4,
    )

    colorbar = figure.colorbar(classical, ax=axis, pad=0.02)
    colorbar.set_label(r"$c_i$")

    axis.set_xlabel("Mach number")
    axis.set_ylabel(r"Wavenumber $\alpha$")
    axis.set_title(
        "Digitized Blumen isolines and classical shooting reference"
    )
    axis.grid(True, alpha=0.20)
    axis.legend(frameon=False, fontsize=7.5, loc="best")

    save_figure(figure, pdf_path, png_path)


def build_coverage_map(
    spectral: pd.DataFrame,
    pdf_path: Path,
    png_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)

    normalization = Normalize(
        vmin=float(spectral["ci"].min()),
        vmax=float(spectral["ci"].max()),
    )
    cmap = plt.get_cmap("viridis")

    strict = spectral.loc[
        spectral["validation_status"].eq(STRICT_STATUS)
    ]
    tail = spectral.loc[
        spectral["validation_status"].eq(TAIL_STATUS)
    ]

    plotted = axis.scatter(
        strict["Mach"],
        strict["alpha"],
        c=strict["ci"],
        norm=normalization,
        cmap=cmap,
        marker="o",
        s=52,
        edgecolors="black",
        linewidths=0.55,
        label=f"joint modal-spectral ({len(strict)})",
    )

    axis.scatter(
        tail["Mach"],
        tail["alpha"],
        c=tail["ci"],
        norm=normalization,
        cmap=cmap,
        marker="s",
        s=42,
        edgecolors="black",
        linewidths=0.45,
        label=f"core-modal, tail-sensitive ({len(tail)})",
    )

    colorbar = figure.colorbar(plotted, ax=axis, pad=0.02)
    colorbar.set_label(r"$c_i$")

    axis.set_xlabel("Mach number")
    axis.set_ylabel(r"Wavenumber $\alpha$")
    axis.set_title("Coverage of the 133-point validated classical reference")
    axis.grid(True, alpha=0.20)
    axis.legend(frameon=False, loc="best")

    save_figure(figure, pdf_path, png_path)


def select_mode_point(
    spectral: pd.DataFrame,
    mach_target: float,
    alpha_target: float,
) -> pd.Series:
    exact_mach = spectral.loc[
        np.isclose(
            spectral["Mach"].to_numpy(dtype=float),
            mach_target,
            atol=5.0e-8,
            rtol=0.0,
        )
    ].copy()

    if exact_mach.empty:
        available = sorted(spectral["Mach"].unique())
        raise RuntimeError(
            f"No point exists at Mach={mach_target}. Available Mach values: {available}"
        )

    strict = exact_mach.loc[
        exact_mach["validation_status"].eq(STRICT_STATUS)
    ]
    candidates = strict if not strict.empty else exact_mach

    index = (
        candidates["alpha"].astype(float).sub(alpha_target).abs().idxmin()
    )
    return spectral.loc[index]


def common_active_limit(
    point_modal: pd.DataFrame,
    threshold: float,
) -> float:
    y = point_modal["y"].to_numpy(dtype=float)
    full_limit = float(np.nanmax(np.abs(y)))
    envelope = np.zeros(len(point_modal), dtype=float)

    for field in ("p", "rho", "u", "v"):
        values = (
            point_modal[f"{field}_real"].to_numpy(dtype=float)
            + 1j * point_modal[f"{field}_imag"].to_numpy(dtype=float)
        )
        peak = float(np.nanmax(np.abs(values)))
        if np.isfinite(peak) and peak > 0.0:
            envelope = np.maximum(envelope, np.abs(values) / peak)

    active = np.isfinite(envelope) & (envelope >= threshold)
    if not active.any():
        return full_limit

    detected = float(np.nanmax(np.abs(y[active])))
    return min(
        full_limit,
        max(min(10.0, full_limit), 1.08 * detected),
    )


def load_mode(
    modal_path: Path,
    point: pd.Series,
) -> pd.DataFrame:
    if not modal_path.is_file():
        raise FileNotFoundError(modal_path)

    columns = [
        "Mach",
        "alpha",
        "y",
        "p_real",
        "p_imag",
        "rho_real",
        "rho_imag",
        "u_real",
        "u_imag",
        "v_real",
        "v_imag",
    ]
    frame = pd.read_parquet(modal_path, columns=columns)

    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    expected_key = point_key(point["Mach"], point["alpha"])
    frame["key"] = [
        point_key(mach, alpha)
        for mach, alpha in zip(frame["Mach"], frame["alpha"])
    ]
    matching_key = frame["key"].map(
        lambda value: value == expected_key
    )

    selected = (
        frame.loc[matching_key]
        .dropna(subset=["y"])
        .drop_duplicates(subset=["y"], keep="first")
        .sort_values("y")
        .reset_index(drop=True)
    )

    if selected.empty:
        raise RuntimeError(f"No modal rows found for point {expected_key}")

    return selected


def build_mode_figure(
    point: pd.Series,
    modal: pd.DataFrame,
    threshold: float,
    pdf_path: Path,
    png_path: Path,
) -> None:
    y = modal["y"].to_numpy(dtype=float)
    limit = common_active_limit(modal, threshold)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(9.2, 7.2),
        constrained_layout=True,
    )

    fields = [
        ("p", r"Pressure $p$"),
        ("rho", r"Density $\rho$"),
        ("u", r"Streamwise velocity $u$"),
        ("v", r"Transverse velocity $v$"),
    ]

    for axis, (field, title) in zip(axes.flat, fields):
        values = (
            modal[f"{field}_real"].to_numpy(dtype=float)
            + 1j * modal[f"{field}_imag"].to_numpy(dtype=float)
        )
        peak = float(np.nanmax(np.abs(values)))
        if not np.isfinite(peak) or peak <= 0.0:
            raise RuntimeError(f"Invalid {field} normalization.")

        values = values / peak
        visible = np.abs(y) <= limit

        axis.plot(y[visible], values.real[visible], label="real", linewidth=1.0)
        axis.plot(y[visible], values.imag[visible], label="imaginary", linewidth=1.0)
        axis.plot(
            y[visible],
            np.abs(values[visible]),
            linestyle="--",
            label="modulus",
            linewidth=1.0,
        )
        axis.axhline(0.0, linewidth=0.55)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-1.06, 1.06)
        axis.set_xlabel(r"$y$")
        axis.set_ylabel("normalized mode")
        axis.set_title(title)
        axis.grid(True, alpha=0.20)

    axes.flat[0].legend(frameon=False, ncols=3, fontsize=7.5)

    figure.suptitle(
        (
            f"Representative classical mode: "
            f"M={point['Mach']:.2f}, "
            f"alpha={point['alpha']:.6f}, "
            f"cr={point['cr']:.6f}, "
            f"ci={point['ci']:.6f}\n"
            f"status={point['validation_status']}; "
            f"common active window at amplitude >= {threshold:g}"
        ),
        fontsize=10.5,
    )

    save_figure(figure, pdf_path, png_path)


def build_summary_table(spectral: pd.DataFrame) -> pd.DataFrame:
    grouped = spectral.groupby("Mach", sort=True)

    summary = grouped.agg(
        n_points=("alpha", "size"),
        alpha_min=("alpha", "min"),
        alpha_max=("alpha", "max"),
        ci_min=("ci", "min"),
        ci_max=("ci", "max"),
    ).reset_index()

    strict_counts = (
        spectral.assign(
            is_strict=spectral["validation_status"].eq(STRICT_STATUS)
        )
        .groupby("Mach")["is_strict"]
        .sum()
    )

    tail_counts = (
        spectral.assign(
            is_tail=spectral["validation_status"].eq(TAIL_STATUS)
        )
        .groupby("Mach")["is_tail"]
        .sum()
    )

    summary["n_joint_modal_spectral"] = (
        summary["Mach"].map(strict_counts).fillna(0).astype(int)
    )
    summary["n_core_modal_tail_sensitive"] = (
        summary["Mach"].map(tail_counts).fillna(0).astype(int)
    )

    return summary


def write_readme(
    path: Path,
    point: pd.Series,
    blumen_metadata: dict,
) -> None:
    text = f"""# Classical supersonic article assets

This directory contains the compact set of figures and tables intended for the
article.

## Figures

- `figures/fig_ci_blumen_vs_classical_overlay.*`
  Digitized Blumen $c_i$ isolines and the 133-point classical shooting
  reference. Blumen points are displayed as scatter points only and are never
  connected by plotting lines.

- `figures/fig_modes_example_M140.*`
  Representative normalized $p$, $\\rho$, $u$ and $v$ mode at
  $M={point['Mach']:.2f}$ and $\\alpha={point['alpha']:.6f}$.

- `figures/fig_validated_points_map.*`
  Coverage of the validated reference in the $(M,\\alpha)$ plane.

## Blumen uncertainty

The current error-bar method is:

`{blumen_metadata['uncertainty_method']}`

These bars are a digitization/sampling uncertainty proxy, not a statistical
confidence interval. Replace the proxy with calibrated pixel-to-coordinate
uncertainties when explicit digitization calibration uncertainties are
available.

Median displayed coordinate uncertainties:

- Mach: `{blumen_metadata['median_mach_error']:.8g}`
- alpha: `{blumen_metadata['median_alpha_error']:.8g}`

Mach-coordinate calibration:

`{blumen_metadata['mach_calibration']}`

## Reference scope

The primary reference contains 133 points:

- 44 jointly validated modal-spectral points with exported fields;
- 89 spectral and core-modal validated points with documented tail
  sensitivity.

Raw-confirmed modal fields are used for the representative mode.
"""
    path.write_text(text)


def main() -> None:
    arguments = parse_args()
    repo = arguments.repo.expanduser().resolve()

    if not (repo / ".git").exists():
        raise SystemExit(f"Not a Git repository root: {repo}")

    output = arguments.output_dir
    if not output.is_absolute():
        output = repo / output
    output = output.resolve()

    figures = output / "figures"
    tables = output / "tables"
    provenance = output / "provenance"
    supplementary = output / "supplementary"

    for directory in (figures, tables, provenance, supplementary):
        directory.mkdir(parents=True, exist_ok=True)

    spectral_path = (
        repo
        / "assets/classic_supersonic/csv/pinn_direct/spectral/"
        "table_supersonic_primary_modal_spectral_133pts.csv"
    )
    modal_path = (
        repo
        / "experiments/modal_reconstruction/support/modal/"
        "supersonic_reference_v2_modal_raw.parquet"
    )
    blumen_points_path = (
        repo
        / "article/tables/table_blumen_ci_digitized_points.csv"
    )
    blumen_levels_path = (
        repo
        / "article/tables/table_blumen_ci_curve_levels.csv"
    )

    spectral = load_spectral(spectral_path)
    blumen, blumen_metadata = load_blumen(
        blumen_points_path,
        blumen_levels_path,
        repo,
        arguments.digitization_error_mode,
    )

    overlay_pdf = figures / "fig_ci_blumen_vs_classical_overlay.pdf"
    overlay_png = figures / "fig_ci_blumen_vs_classical_overlay.png"
    build_overlay(
        spectral,
        blumen,
        overlay_pdf,
        overlay_png,
        arguments.errorbar_stride,
    )

    coverage_pdf = figures / "fig_validated_points_map.pdf"
    coverage_png = figures / "fig_validated_points_map.png"
    build_coverage_map(
        spectral,
        coverage_pdf,
        coverage_png,
    )

    mode_point = select_mode_point(
        spectral,
        arguments.mode_mach,
        arguments.mode_alpha_target,
    )
    modal = load_mode(modal_path, mode_point)

    mode_pdf = figures / "fig_modes_example_M140.pdf"
    mode_png = figures / "fig_modes_example_M140.png"
    build_mode_figure(
        mode_point,
        modal,
        arguments.active_threshold,
        mode_pdf,
        mode_png,
    )

    summary = build_summary_table(spectral)
    summary_path = tables / "table_reference_summary.csv"
    summary.to_csv(summary_path, index=False)

    selected_mode_path = tables / "table_selected_mode.csv"
    pd.DataFrame(
        [
            {
                "Mach": float(mode_point["Mach"]),
                "alpha": float(mode_point["alpha"]),
                "cr": float(mode_point["cr"]),
                "ci": float(mode_point["ci"]),
                "omega_i": float(mode_point["omega_i"]),
                "validation_status": str(mode_point["validation_status"]),
                "active_threshold": float(arguments.active_threshold),
                "modal_source": str(modal_path.relative_to(repo)),
            }
        ]
    ).to_csv(selected_mode_path, index=False)

    readme_path = output / "README.md"
    write_readme(readme_path, mode_point, blumen_metadata)

    generated = [
        overlay_pdf,
        overlay_png,
        coverage_pdf,
        coverage_png,
        mode_pdf,
        mode_png,
        summary_path,
        selected_mode_path,
        readme_path,
    ]

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "spectral_source": str(spectral_path.relative_to(repo)),
        "modal_source": str(modal_path.relative_to(repo)),
        "blumen_points_source": str(blumen_points_path.relative_to(repo)),
        "blumen_levels_source": str(blumen_levels_path.relative_to(repo)),
        "point_count": int(len(spectral)),
        "status_counts": spectral["validation_status"].value_counts().to_dict(),
        "selected_mode": {
            "Mach": float(mode_point["Mach"]),
            "alpha": float(mode_point["alpha"]),
            "cr": float(mode_point["cr"]),
            "ci": float(mode_point["ci"]),
            "omega_i": float(mode_point["omega_i"]),
            "validation_status": str(mode_point["validation_status"]),
        },
        "blumen": blumen_metadata,
        "outputs": [
            {
                "path": str(path.relative_to(repo)),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256(path),
            }
            for path in generated
        ],
    }

    manifest_path = provenance / "article_asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print("Article directory:", output)
    print("Primary points:", len(spectral))
    print(
        "Selected mode:",
        f"M={mode_point['Mach']:.2f}",
        f"alpha={mode_point['alpha']:.6f}",
        mode_point["validation_status"],
    )
    print("Blumen uncertainty:", blumen_metadata["uncertainty_method"])
    print()
    for path in generated:
        print("Wrote:", path)
    print("Wrote:", manifest_path)
    print()
    print("ARTICLE ASSETS BUILD: PASS")


if __name__ == "__main__":
    main()
