#!/usr/bin/env python3
"""Compute quantitative Blumen-versus-shooting metrics.

The script uses the same one-sided geometric definition as the existing error
heatmap: each digitized Blumen point is compared with the corresponding
isoline reconstructed from the shooting ``omega_i(M, alpha)`` grid.

For the neutral level ``omega_i = 0``, the shooting field is zero throughout
the stable region and therefore has no unique zero contour. Its geometric
reference is the analytical neutral boundary ``M**2 + alpha**2 = 1``.

Example:
    python scripts/reference_classique/subsonique/compute_blumen_shooting_metrics.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.subsonic.hybrid_subsonic_scan import (
    point_to_polyline_distance,
)


DEFAULT_GROWTH_MAP = (
    ROOT_DIR / "assets/classic_subsonic/data/subsonic_hybrid_growth_map.csv"
)
DEFAULT_BLUMEN_DIR = ROOT_DIR / "KH_RT_Blumen/subsonic"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "assets/classic_subsonic/data"
DEFAULT_OUTPUT_STEM = "subsonic_blumen_shooting_metrics"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--growth-map", type=Path, default=DEFAULT_GROWTH_MAP)
    parser.add_argument("--blumen-dir", type=Path, default=DEFAULT_BLUMEN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    return parser


def parse_level(path: Path) -> float:
    return float(path.stem.strip().replace("_", ".").replace(",", "."))


def load_blumen_curves(directory: Path) -> list[dict[str, object]]:
    curves: list[dict[str, object]] = []
    for filename in sorted(glob.glob(str(directory / "*.csv"))):
        path = Path(filename)
        data = (
            pd.read_csv(
                path,
                header=None,
                names=["Mach", "alpha"],
                sep=";",
                decimal=",",
                engine="python",
            )
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        curves.append({"level": parse_level(path), "path": path, "data": data})
    if not curves:
        raise FileNotFoundError(f"No digitized Blumen curves found in {directory}")
    return curves


def load_growth_grid(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_csv(path)
    required = {"Mach", "alpha", "omega_i"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing growth-map columns: {sorted(missing)}")
    if data[["Mach", "alpha"]].duplicated().any():
        raise ValueError("The growth map contains duplicate (Mach, alpha) points.")

    pivot = (
        data.pivot(index="alpha", columns="Mach", values="omega_i")
        .sort_index()
        .sort_index(axis=1)
    )
    if pivot.isna().any().any():
        raise ValueError("The growth map is not a complete rectangular grid.")
    return (
        data,
        pivot.columns.to_numpy(dtype=float),
        pivot.index.to_numpy(dtype=float),
        pivot.to_numpy(dtype=float),
    )


def build_positive_contours(
    machs: np.ndarray,
    alphas: np.ndarray,
    omega_i: np.ndarray,
    levels: list[float],
) -> dict[float, list[np.ndarray]]:
    mach_grid, alpha_grid = np.meshgrid(machs, alphas)
    figure, axis = plt.subplots()
    contours = axis.contour(
        mach_grid,
        alpha_grid,
        omega_i,
        levels=sorted(levels),
    )
    result = {
        float(level): [
            np.asarray(segment, dtype=float)
            for segment in segments
            if len(segment) >= 2
        ]
        for level, segments in zip(contours.levels, contours.allsegs)
    }
    plt.close(figure)
    return result


def neutral_boundary_distance(point: np.ndarray) -> float:
    """Euclidean distance to the first-quadrant unit circle."""
    return abs(float(np.linalg.norm(point)) - 1.0)


def finite_mean(values: pd.Series) -> float:
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    return float(finite.mean()) if len(finite) else np.nan


def finite_max_row(frame: pd.DataFrame, column: str) -> pd.Series | None:
    finite = frame[np.isfinite(frame[column].to_numpy(dtype=float))]
    if finite.empty:
        return None
    return finite.loc[finite[column].idxmax()]


def compute_metrics(
    growth_map: Path,
    blumen_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    growth_data, machs, alphas, omega_i = load_growth_grid(growth_map)
    curves = load_blumen_curves(blumen_dir)
    positive_levels = sorted(
        float(curve["level"]) for curve in curves if float(curve["level"]) > 0.0
    )
    contours = build_positive_contours(machs, alphas, omega_i, positive_levels)
    interpolator = RegularGridInterpolator(
        (alphas, machs),
        omega_i,
        bounds_error=False,
        fill_value=np.nan,
    )

    point_rows: list[dict[str, object]] = []
    for curve in curves:
        level = float(curve["level"])
        source = Path(curve["path"])
        data = curve["data"]
        assert isinstance(data, pd.DataFrame)
        points = data[["Mach", "alpha"]].to_numpy(dtype=float)
        predictions = interpolator(data[["alpha", "Mach"]].to_numpy(dtype=float))

        for point, prediction in zip(points, predictions):
            if np.isclose(level, 0.0):
                distance = neutral_boundary_distance(point)
                distance_reference = "analytic neutral boundary M^2 + alpha^2 = 1"
            else:
                polylines = contours.get(level, [])
                distance = (
                    min(
                        point_to_polyline_distance(point, polyline)
                        for polyline in polylines
                    )
                    if polylines
                    else np.nan
                )
                distance_reference = "shooting contour"
            point_rows.append(
                {
                    "blumen_level_omega_i": level,
                    "Mach": float(point[0]),
                    "alpha": float(point[1]),
                    "shooting_omega_i": (
                        float(prediction) if np.isfinite(prediction) else np.nan
                    ),
                    "abs_error_omega_i": (
                        abs(float(prediction) - level)
                        if np.isfinite(prediction)
                        else np.nan
                    ),
                    "geometric_distance_M_alpha": distance,
                    "distance_reference": distance_reference,
                    "blumen_source_csv": str(source.relative_to(ROOT_DIR)),
                }
            )

    points = pd.DataFrame(point_rows).sort_values(
        ["blumen_level_omega_i", "Mach", "alpha"]
    )
    level_rows: list[dict[str, object]] = []
    for level, group in points.groupby("blumen_level_omega_i", sort=True):
        max_error = finite_max_row(group, "abs_error_omega_i")
        max_distance = finite_max_row(group, "geometric_distance_M_alpha")
        level_rows.append(
            {
                "blumen_level_omega_i": float(level),
                "n_digitized_points": int(len(group)),
                "n_finite_omega_errors": int(group["abs_error_omega_i"].notna().sum()),
                "mean_abs_error_omega_i": finite_mean(group["abs_error_omega_i"]),
                "max_abs_error_omega_i": (
                    float(max_error["abs_error_omega_i"])
                    if max_error is not None
                    else np.nan
                ),
                "max_error_Mach": (
                    float(max_error["Mach"]) if max_error is not None else np.nan
                ),
                "max_error_alpha": (
                    float(max_error["alpha"]) if max_error is not None else np.nan
                ),
                "mean_geometric_distance_M_alpha": finite_mean(
                    group["geometric_distance_M_alpha"]
                ),
                "max_geometric_distance_M_alpha": (
                    float(max_distance["geometric_distance_M_alpha"])
                    if max_distance is not None
                    else np.nan
                ),
                "max_distance_Mach": (
                    float(max_distance["Mach"])
                    if max_distance is not None
                    else np.nan
                ),
                "max_distance_alpha": (
                    float(max_distance["alpha"])
                    if max_distance is not None
                    else np.nan
                ),
            }
        )
    by_level = pd.DataFrame(level_rows)

    max_error = finite_max_row(points, "abs_error_omega_i")
    max_distance = finite_max_row(points, "geometric_distance_M_alpha")
    positive_points = points[points["blumen_level_omega_i"] > 0.0]
    max_positive_error = finite_max_row(positive_points, "abs_error_omega_i")
    max_positive_distance = finite_max_row(
        positive_points, "geometric_distance_M_alpha"
    )
    summary: dict[str, object] = {
        "growth_map": str(growth_map.relative_to(ROOT_DIR)),
        "blumen_directory": str(blumen_dir.relative_to(ROOT_DIR)),
        "grid_num_mach": int(len(machs)),
        "grid_num_alpha": int(len(alphas)),
        "n_digitized_points": int(len(points)),
        "n_finite_omega_errors": int(points["abs_error_omega_i"].notna().sum()),
        "mean_abs_error_omega_i": finite_mean(points["abs_error_omega_i"]),
        "isoline_balanced_mean_abs_error_omega_i": float(
            by_level["mean_abs_error_omega_i"].mean()
        ),
        "max_abs_error_omega_i": float(max_error["abs_error_omega_i"]),
        "max_abs_error_level_omega_i": float(max_error["blumen_level_omega_i"]),
        "max_abs_error_Mach": float(max_error["Mach"]),
        "max_abs_error_alpha": float(max_error["alpha"]),
        "max_abs_error_shooting_omega_i": float(max_error["shooting_omega_i"]),
        "mean_abs_error_positive_levels_only": finite_mean(
            positive_points["abs_error_omega_i"]
        ),
        "max_abs_error_positive_levels_only": float(
            max_positive_error["abs_error_omega_i"]
        ),
        "max_abs_error_positive_level_omega_i": float(
            max_positive_error["blumen_level_omega_i"]
        ),
        "max_abs_error_positive_Mach": float(max_positive_error["Mach"]),
        "max_abs_error_positive_alpha": float(max_positive_error["alpha"]),
        "mean_geometric_distance_M_alpha": finite_mean(
            points["geometric_distance_M_alpha"]
        ),
        "isoline_balanced_mean_geometric_distance_M_alpha": float(
            by_level["mean_geometric_distance_M_alpha"].mean()
        ),
        "max_geometric_distance_M_alpha": float(
            max_distance["geometric_distance_M_alpha"]
        ),
        "max_geometric_distance_level_omega_i": float(
            max_distance["blumen_level_omega_i"]
        ),
        "max_geometric_distance_Mach": float(max_distance["Mach"]),
        "max_geometric_distance_alpha": float(max_distance["alpha"]),
        "mean_geometric_distance_positive_levels_only": finite_mean(
            positive_points["geometric_distance_M_alpha"]
        ),
        "max_geometric_distance_positive_levels_only": float(
            max_positive_distance["geometric_distance_M_alpha"]
        ),
        "max_geometric_distance_positive_level_omega_i": float(
            max_positive_distance["blumen_level_omega_i"]
        ),
        "max_geometric_distance_positive_Mach": float(
            max_positive_distance["Mach"]
        ),
        "max_geometric_distance_positive_alpha": float(
            max_positive_distance["alpha"]
        ),
        "geometric_metric": (
            "one-sided Euclidean distance from digitized Blumen points to the "
            "corresponding shooting contour in the (Mach, alpha) plane; the "
            "neutral level uses M^2 + alpha^2 = 1"
        ),
        "aggregation": "point-weighted over digitized Blumen points",
        "source_counts": {
            str(key): int(value)
            for key, value in growth_data["source"].value_counts().items()
        }
    }
    return points, by_level, summary


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points, by_level, summary = compute_metrics(
        args.growth_map.resolve(),
        args.blumen_dir.resolve(),
    )

    points_path = args.output_dir / f"{args.output_stem}_by_point.csv"
    levels_path = args.output_dir / f"{args.output_stem}_by_level.csv"
    summary_csv_path = args.output_dir / f"{args.output_stem}_summary.csv"
    summary_json_path = args.output_dir / f"{args.output_stem}_summary.json"

    points.to_csv(points_path, index=False)
    by_level.to_csv(levels_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {points_path}")
    print(f"Wrote {levels_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {summary_json_path}")


if __name__ == "__main__":
    main()
