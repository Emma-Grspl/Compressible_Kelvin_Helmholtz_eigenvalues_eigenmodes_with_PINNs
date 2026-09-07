#!/usr/bin/env python3
"""Create separated supersonic Blumen/classical validation figures.

Figure 1: digitized Blumen points + classical isolines.
Figure 2: pointwise error map + point-to-solver-isoline distance.

The script never searches for another alpha and never displaces a Blumen point.
For every digitized level it reconstructs a continuous geometric ordering of the
(M, alpha) samples. An explicit order column is used when available; otherwise
several geometry-aware candidates are compared, including a two-branch fold
reconstruction and a graph-spectral ordering.
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D


LEVELS = np.array([0.01, 0.03, 0.05, 0.07, 0.10], dtype=float)

COLUMN_CANDIDATES = {
    "mach": ("Mach", "mach", "M", "m"),
    "alpha": ("alpha", "Alpha", "wavenumber", "wave_number"),
    "blumen_ci": (
        "blumen_ci",
        "ci_blumen",
        "Blumen_ci",
        "c_i_blumen",
        "ci_B",
        "ci_b",
    ),
    "classical_ci": (
        "classical_ci",
        "ci_classical",
        "classical_c_i",
        "ci_classic",
        "ci",
    ),
    "classical_cr": (
        "classical_cr",
        "cr_classical",
        "classical_c_r",
        "cr_classic",
        "cr",
    ),
    "status": ("status", "solver_status", "root_status", "convergence_status"),
}

ORDER_COLUMN_CANDIDATES = (
    "digitized_order",
    "digitisation_order",
    "original_order",
    "source_order",
    "contour_order",
    "curve_order",
    "path_order",
    "point_order",
    "sequence",
    "seq",
    "path_index",
    "point_index",
    "original_index",
)

REJECT_STATUS_RE = re.compile(
    r"fail|non[-_ ]?converg|no[-_ ]?root|missing|reject|invalid|nan|negative",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Columns:
    mach: str
    alpha: str
    blumen_ci: str
    classical_ci: str
    classical_cr: str | None
    status: str | None
    order: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot separated supersonic Blumen/classical validation figures."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Pointwise CSV. Defaults to the article source-data path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to assets/article/classical_supersonic/figures.",
    )
    parser.add_argument(
        "--order-column",
        type=str,
        default=None,
        help="Force an explicit digitization-order column.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution (default: 300 dpi).",
    )
    return parser.parse_args()


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [Path.cwd(), *here.parents]
    for candidate in candidates:
        if (candidate / "assets").is_dir():
            return candidate
    if len(here.parents) >= 4:
        return here.parents[3]
    return Path.cwd()


def resolve_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    required: bool,
    semantic_name: str,
) -> str | None:
    exact = {str(col): str(col) for col in df.columns}
    folded = {str(col).casefold(): str(col) for col in df.columns}

    for name in candidates:
        if name in exact:
            return exact[name]
        if name.casefold() in folded:
            return folded[name.casefold()]

    if required:
        raise KeyError(
            f"Missing required column for {semantic_name!r}. "
            f"Tried {list(candidates)}. Available columns: {list(df.columns)}"
        )
    return None


def resolve_columns(df: pd.DataFrame, forced_order: str | None) -> Columns:
    if forced_order is not None:
        if forced_order not in df.columns:
            raise KeyError(
                f"--order-column={forced_order!r} is absent. "
                f"Available columns: {list(df.columns)}"
            )
        order_col = forced_order
    else:
        order_col = resolve_column(
            df,
            ORDER_COLUMN_CANDIDATES,
            required=False,
            semantic_name="digitization order",
        )

    return Columns(
        mach=resolve_column(
            df, COLUMN_CANDIDATES["mach"], required=True, semantic_name="Mach"
        ),
        alpha=resolve_column(
            df, COLUMN_CANDIDATES["alpha"], required=True, semantic_name="alpha"
        ),
        blumen_ci=resolve_column(
            df,
            COLUMN_CANDIDATES["blumen_ci"],
            required=True,
            semantic_name="Blumen ci",
        ),
        classical_ci=resolve_column(
            df,
            COLUMN_CANDIDATES["classical_ci"],
            required=True,
            semantic_name="classical ci",
        ),
        classical_cr=resolve_column(
            df,
            COLUMN_CANDIDATES["classical_cr"],
            required=False,
            semantic_name="classical cr",
        ),
        status=resolve_column(
            df,
            COLUMN_CANDIDATES["status"],
            required=False,
            semantic_name="solver status",
        ),
        order=order_col,
    )


def prepare_dataframe(df: pd.DataFrame, cols: Columns) -> pd.DataFrame:
    out = df.copy()
    out["__row_order__"] = np.arange(len(out), dtype=int)

    numeric_cols: Iterable[str | None] = (
        cols.mach,
        cols.alpha,
        cols.blumen_ci,
        cols.classical_ci,
        cols.classical_cr,
        cols.order,
    )
    for col in numeric_cols:
        if col is not None:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    essential = [cols.mach, cols.alpha, cols.blumen_ci]
    before = len(out)
    out = out.dropna(subset=essential).copy()
    if len(out) < before:
        warnings.warn(f"Dropped {before - len(out)} rows with invalid M, alpha or Blumen ci.")

    finite_positive_ci = np.isfinite(out[cols.classical_ci]) & (
        out[cols.classical_ci] > 0.0
    )
    if cols.status is None:
        status_ok = np.ones(len(out), dtype=bool)
    else:
        status_text = out[cols.status].fillna("").astype(str)
        status_ok = ~status_text.str.contains(REJECT_STATUS_RE, na=False)

    out["accepted"] = finite_positive_ci & status_ok
    out["abs_error"] = np.where(
        out["accepted"],
        np.abs(out[cols.classical_ci] - out[cols.blumen_ci]),
        np.nan,
    )
    out["solver_distance"] = np.nan
    return out


def normalize_xy(xy: np.ndarray) -> np.ndarray:
    lo = np.nanmin(xy, axis=0)
    hi = np.nanmax(xy, axis=0)
    span = hi - lo
    span[~np.isfinite(span) | (span <= 0.0)] = 1.0
    return (xy - lo) / span


def remove_consecutive_duplicates(order: np.ndarray) -> np.ndarray:
    if len(order) <= 1:
        return order.astype(int, copy=False)
    keep = np.r_[True, order[1:] != order[:-1]]
    return order[keep].astype(int, copy=False)


def count_segment_intersections(path: np.ndarray) -> int:
    if len(path) < 4:
        return 0

    def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ab = b - a
        ac = c - a
        return float(ab[0] * ac[1] - ab[1] * ac[0])

    count = 0
    eps = 1.0e-12
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        for j in range(i + 2, len(path) - 1):
            if j == i + 1:
                continue
            c, d = path[j], path[j + 1]
            o1 = orientation(a, b, c)
            o2 = orientation(a, b, d)
            o3 = orientation(c, d, a)
            o4 = orientation(c, d, b)
            if (o1 * o2 < -eps) and (o3 * o4 < -eps):
                count += 1
    return count


def path_score(order: np.ndarray, xy_norm: np.ndarray) -> float:
    order = remove_consecutive_duplicates(np.asarray(order, dtype=int))
    if len(order) != len(xy_norm) or len(np.unique(order)) != len(xy_norm):
        return np.inf
    if len(order) < 2:
        return 0.0

    path = xy_norm[order]
    delta = np.diff(path, axis=0)
    steps = np.linalg.norm(delta, axis=1)
    positive_steps = steps[steps > 1.0e-14]
    if len(positive_steps) == 0:
        return np.inf

    median_step = float(np.median(positive_steps))
    max_step = float(np.max(positive_steps))
    long_edge_penalty = float(
        np.sum(np.clip(positive_steps / max(median_step, 1.0e-12) - 3.0, 0.0, None) ** 2)
    )

    smoothness = 0.0
    if len(delta) >= 2:
        norms = np.linalg.norm(delta, axis=1)
        valid = norms > 1.0e-14
        unit = np.zeros_like(delta)
        unit[valid] = delta[valid] / norms[valid, None]
        cos_angles = np.sum(unit[:-1] * unit[1:], axis=1)
        smoothness = float(np.sum((1.0 - np.clip(cos_angles, -1.0, 1.0)) ** 2))

    dx = delta[:, 0]
    tol = max(1.0e-4, 0.02 * float(np.median(np.abs(dx[np.abs(dx) > 0])))) if np.any(np.abs(dx) > 0) else 1.0e-4
    signs = np.sign(dx[np.abs(dx) > tol])
    mach_turns = int(np.sum(signs[1:] != signs[:-1])) if len(signs) >= 2 else 0
    extra_turn_penalty = float(max(0, mach_turns - 1) ** 2)

    intersections = count_segment_intersections(path)

    return (
        float(np.sum(positive_steps))
        + 2.5 * max_step
        + 2.0 * smoothness
        + 12.0 * long_edge_penalty
        + 30.0 * extra_turn_penalty
        + 100.0 * intersections
    )


def folded_candidates(xy: np.ndarray) -> list[np.ndarray]:
    n = len(xy)
    if n <= 2:
        return [np.arange(n, dtype=int)]

    x = xy[:, 0]
    y = xy[:, 1]
    candidates: list[np.ndarray] = []

    for fold_kind in ("max", "min"):
        if fold_kind == "max":
            near = x >= np.quantile(x, 0.85)
            x_direction_to_fold = +1
        else:
            near = x <= np.quantile(x, 0.15)
            x_direction_to_fold = -1

        if not np.any(near):
            continue

        alpha_fold = float(np.median(y[near]))
        lower = np.flatnonzero(y <= alpha_fold)
        upper = np.flatnonzero(y > alpha_fold)

        split_variants: list[tuple[np.ndarray, np.ndarray]] = [(lower, upper)]
        y_sort = np.argsort(y, kind="mergesort")
        y_gap = np.diff(y[y_sort])
        if len(y_gap):
            gap_idx = int(np.argmax(y_gap))
            split_variants.append((y_sort[: gap_idx + 1], y_sort[gap_idx + 1 :]))

        for branch_a, branch_b in split_variants:
            if len(branch_a) == 0 or len(branch_b) == 0:
                continue

            if x_direction_to_fold > 0:
                a = branch_a[np.lexsort((y[branch_a], x[branch_a]))]
                b = branch_b[np.lexsort((-y[branch_b], -x[branch_b]))]
            else:
                a = branch_a[np.lexsort((-y[branch_a], -x[branch_a]))]
                b = branch_b[np.lexsort((y[branch_b], x[branch_b]))]

            path = np.concatenate([a, b])
            candidates.extend([path, path[::-1]])

        fold_index = int(np.argmax(x) if fold_kind == "max" else np.argmin(x))
        below = np.flatnonzero(y < y[fold_index])
        above = np.flatnonzero(y > y[fold_index])
        equal = np.flatnonzero(np.isclose(y, y[fold_index], rtol=0.0, atol=1.0e-14))
        if len(below) and len(above):
            if fold_kind == "max":
                below = below[np.argsort(x[below], kind="mergesort")]
                above = above[np.argsort(-x[above], kind="mergesort")]
            else:
                below = below[np.argsort(-x[below], kind="mergesort")]
                above = above[np.argsort(x[above], kind="mergesort")]
            equal = equal[np.argsort(y[equal], kind="mergesort")]
            path = np.concatenate([below, equal, above])
            candidates.extend([path, path[::-1]])

    return candidates


def graph_spectral_candidates(xy_norm: np.ndarray) -> list[np.ndarray]:
    n = len(xy_norm)
    if n <= 2:
        return [np.arange(n, dtype=int)]

    distance = np.linalg.norm(xy_norm[:, None, :] - xy_norm[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    candidates: list[np.ndarray] = []

    for k in range(2, min(7, n)):
        neighbours = np.argsort(distance, axis=1)[:, :k]
        adjacency = np.zeros((n, n), dtype=bool)
        for i in range(n):
            adjacency[i, neighbours[i]] = True
        adjacency |= adjacency.T

        seen = {0}
        stack = [0]
        while stack:
            i = stack.pop()
            for j in np.flatnonzero(adjacency[i]):
                if int(j) not in seen:
                    seen.add(int(j))
                    stack.append(int(j))
        if len(seen) != n:
            continue

        finite_edges = distance[adjacency]
        sigma = float(np.median(finite_edges[np.isfinite(finite_edges)]))
        sigma = max(sigma, 1.0e-12)
        weights = np.zeros((n, n), dtype=float)
        edge_values = np.exp(-((distance[adjacency] / sigma) ** 2))
        weights[adjacency] = edge_values
        weights = np.maximum(weights, weights.T)

        degree = np.sum(weights, axis=1)
        laplacian = np.diag(degree) - weights
        eigvals, eigvecs = np.linalg.eigh(laplacian)
        fiedler = eigvecs[:, np.argsort(eigvals)[1]]
        order = np.argsort(fiedler, kind="mergesort")
        candidates.extend([order, order[::-1]])
        break

    return candidates


def pca_candidates(xy_norm: np.ndarray) -> list[np.ndarray]:
    centered = xy_norm - np.mean(xy_norm, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    candidates: list[np.ndarray] = []
    for axis in vh:
        projection = centered @ axis
        order = np.argsort(projection, kind="mergesort")
        candidates.extend([order, order[::-1]])
    return candidates


def geometric_order(sub: pd.DataFrame, cols: Columns) -> np.ndarray:
    n = len(sub)
    if n <= 2:
        return np.arange(n, dtype=int)

    if cols.order is not None:
        values = pd.to_numeric(sub[cols.order], errors="coerce")
        if values.notna().sum() >= max(2, int(np.ceil(0.8 * n))) and values.nunique(dropna=True) >= 2:
            fill = values.max(skipna=True) + 1.0 + np.arange(n)
            sortable = values.to_numpy(dtype=float)
            sortable[~np.isfinite(sortable)] = fill[~np.isfinite(sortable)]
            return np.argsort(sortable, kind="mergesort")

    xy = sub[[cols.mach, cols.alpha]].to_numpy(dtype=float)
    xy_norm = normalize_xy(xy)

    candidates: list[np.ndarray] = []
    row_order = np.argsort(sub["__row_order__"].to_numpy(), kind="mergesort")
    candidates.extend([row_order, row_order[::-1]])
    candidates.extend(folded_candidates(xy))
    candidates.extend(graph_spectral_candidates(xy_norm))
    candidates.extend(pca_candidates(xy_norm))

    by_m = np.lexsort((xy[:, 1], xy[:, 0]))
    by_alpha = np.lexsort((xy[:, 0], xy[:, 1]))
    candidates.extend([by_m, by_m[::-1], by_alpha, by_alpha[::-1]])

    best_order: np.ndarray | None = None
    best_score = np.inf
    for candidate in candidates:
        candidate = np.asarray(candidate, dtype=int)
        if len(candidate) != n or len(np.unique(candidate)) != n:
            continue
        score = path_score(candidate, xy_norm)
        if score < best_score:
            best_score = score
            best_order = candidate

    if best_order is None:
        raise RuntimeError("Could not reconstruct a geometric order for one contour.")
    return best_order


def split_polyline_at_large_gaps(xy: np.ndarray) -> list[np.ndarray]:
    if len(xy) <= 2:
        return [xy]
    xy_norm = normalize_xy(xy)
    steps = np.linalg.norm(np.diff(xy_norm, axis=0), axis=1)
    positive = steps[steps > 1.0e-14]
    if len(positive) == 0:
        return [xy]

    median = float(np.median(positive))
    q90 = float(np.quantile(positive, 0.90))
    threshold = max(3.5 * median, 1.8 * q90)
    cuts = np.flatnonzero(steps > threshold) + 1
    return [segment for segment in np.split(xy, cuts) if len(segment) >= 2]


def level_subset(df: pd.DataFrame, cols: Columns, level: float) -> pd.DataFrame:
    mask = np.isclose(
        df[cols.blumen_ci].to_numpy(dtype=float),
        level,
        rtol=0.0,
        atol=5.0e-6,
    )
    return df.loc[mask].copy()


def padded_limits(values: np.ndarray, fraction: float = 0.025) -> tuple[float, float]:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    span = hi - lo
    pad = fraction * span if span > 0.0 else 0.05
    return lo - pad, hi + pad


def point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1

    seg_len2 = vx * vx + vy * vy
    if seg_len2 == 0.0:
        return float(np.hypot(px - x1, py - y1))

    t = (wx * vx + wy * vy) / seg_len2
    t = float(np.clip(t, 0.0, 1.0))
    proj_x = x1 + t * vx
    proj_y = y1 + t * vy
    return float(np.hypot(px - proj_x, py - proj_y))


def point_to_polyline_distance(px: float, py: float, poly_x: np.ndarray, poly_y: np.ndarray) -> float:
    if len(poly_x) == 0:
        return float("nan")
    if len(poly_x) == 1:
        return float(np.hypot(px - poly_x[0], py - poly_y[0]))

    dmin = np.inf
    for i in range(len(poly_x) - 1):
        d = point_to_segment_distance(px, py, poly_x[i], poly_y[i], poly_x[i + 1], poly_y[i + 1])
        if d < dmin:
            dmin = d
    return float(dmin)


def build_ordered_curves(df: pd.DataFrame, cols: Columns, present_levels: Sequence[float]) -> dict[float, pd.DataFrame]:
    ordered_curves: dict[float, pd.DataFrame] = {}
    for level in present_levels:
        accepted = level_subset(df, cols, level)
        accepted = accepted.loc[accepted["accepted"]].copy()
        if len(accepted) < 2:
            continue
        order = geometric_order(accepted, cols)
        ordered = accepted.iloc[order].copy().reset_index(drop=True)
        ordered_curves[float(level)] = ordered
    return ordered_curves


def attach_solver_distances(df: pd.DataFrame, cols: Columns, ordered_curves: dict[float, pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()
    distances = np.full(len(out), np.nan, dtype=float)

    for i, row in out.iterrows():
        level = float(row[cols.blumen_ci])
        curve = ordered_curves.get(level)
        if curve is None or len(curve) == 0:
            continue
        distances[i] = point_to_polyline_distance(
            float(row[cols.mach]),
            float(row[cols.alpha]),
            curve[cols.mach].to_numpy(dtype=float),
            curve[cols.alpha].to_numpy(dtype=float),
        )

    out["solver_distance"] = distances
    return out


def make_isoline_figure(
    df: pd.DataFrame,
    cols: Columns,
    output_dir: Path,
    dpi: int,
    present_levels: Sequence[float],
    ordered_curves: dict[float, pd.DataFrame],
    level_colors: dict[float, tuple],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    for level in present_levels:
        sub_all = level_subset(df, cols, level)
        color = level_colors[level]

        ordered = ordered_curves.get(float(level))
        if ordered is not None and len(ordered) >= 2:
            ordered_xy = ordered[[cols.mach, cols.alpha]].to_numpy(dtype=float)
            for segment in split_polyline_at_large_gaps(ordered_xy):
                ax.plot(
                    segment[:, 0],
                    segment[:, 1],
                    color=color,
                    linewidth=2.0,
                    solid_capstyle="round",
                    zorder=2,
                )

        ax.scatter(
            sub_all[cols.mach],
            sub_all[cols.alpha],
            s=32,
            marker="o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )

    missing = df.loc[~df["accepted"]]
    if not missing.empty:
        ax.scatter(
            missing[cols.mach],
            missing[cols.alpha],
            s=62,
            marker="x",
            color="black",
            linewidth=1.8,
            zorder=5,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=level_colors[level],
            linewidth=2.0,
            marker="o",
            markersize=6,
            markerfacecolor=level_colors[level],
            markeredgecolor="white",
            label=rf"Blumen $c_i={level:.2f}$",
        )
        for level in present_levels
    ]
    if not missing.empty:
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="x",
                markersize=7,
                markeredgewidth=1.6,
                color="black",
                label="No accepted classical root",
            )
        )

    ax.set_title("Supersonic classical reconstruction by shooting")
    ax.set_xlabel("Mach number (M)")
    ax.set_ylabel(r"Wavenumber ($\alpha$)")
    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlim(*padded_limits(df[cols.mach].to_numpy(dtype=float)))
    ax.set_ylim(*padded_limits(df[cols.alpha].to_numpy(dtype=float)))
    ax.legend(handles=handles, loc="best", frameon=True, fontsize=9)

    pdf_path = output_dir / "Fig_supersonic_blumen_isolines.pdf"
    png_path = output_dir / "Fig_supersonic_blumen_isolines.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def make_error_distance_figure(
    df: pd.DataFrame,
    cols: Columns,
    output_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    ax_err, ax_dist = axes

    valid_error = df.loc[df["accepted"] & np.isfinite(df["abs_error"])].copy()
    if valid_error.empty:
        raise ValueError("No accepted point has a finite pointwise error.")

    positive_error = valid_error.loc[valid_error["abs_error"] > 0.0, "abs_error"]
    if positive_error.empty:
        vmin_err = 1.0e-12
        vmax_err = 1.0e-11
    else:
        vmin_err = float(positive_error.min())
        vmax_err = float(valid_error["abs_error"].max())
        if not np.isfinite(vmax_err) or vmax_err <= vmin_err:
            vmax_err = 10.0 * vmin_err

    error_for_color = np.clip(valid_error["abs_error"].to_numpy(dtype=float), vmin_err, None)
    sc_err = ax_err.scatter(
        valid_error[cols.mach],
        valid_error[cols.alpha],
        c=error_for_color,
        cmap="plasma",
        norm=LogNorm(vmin=vmin_err, vmax=vmax_err),
        s=34,
        edgecolors="none",
        zorder=3,
    )

    n_top_err = max(1, int(np.ceil(0.10 * len(valid_error))))
    top_err = valid_error.nlargest(n_top_err, "abs_error")
    ax_err.scatter(
        top_err[cols.mach],
        top_err[cols.alpha],
        s=72,
        facecolors="none",
        edgecolors="cyan",
        linewidths=1.5,
        label="Top 10% errors",
        zorder=6,
    )

    missing_error = df.loc[~df["accepted"]]
    if not missing_error.empty:
        ax_err.scatter(
            missing_error[cols.mach],
            missing_error[cols.alpha],
            marker="x",
            s=70,
            c="k",
            linewidths=1.8,
            zorder=5,
            label="No accepted classical root",
        )

    cbar1 = fig.colorbar(sc_err, ax=ax_err)
    cbar1.set_label(r"$|c_i^{\mathrm{classical}}-c_i^B|$")

    ax_err.set_title(r"Error in $c_i$ at Blumen points")
    ax_err.set_xlabel("Mach number (M)")
    ax_err.set_ylabel(r"Wavenumber ($\alpha$)")
    ax_err.grid(True, alpha=0.25, linewidth=0.6)
    ax_err.tick_params(direction="in", top=True, right=True)
    ax_err.legend(loc="best", fontsize=9, frameon=True)

    valid_dist = df.loc[np.isfinite(df["solver_distance"])].copy()
    if valid_dist.empty:
        raise ValueError("No finite point-to-solver-isoline distance could be computed.")

    positive_dist = valid_dist.loc[valid_dist["solver_distance"] > 0.0, "solver_distance"]
    if positive_dist.empty:
        vmin_dist = 1.0e-12
        vmax_dist = 1.0e-11
    else:
        vmin_dist = float(positive_dist.min())
        vmax_dist = float(valid_dist["solver_distance"].max())
        if not np.isfinite(vmax_dist) or vmax_dist <= vmin_dist:
            vmax_dist = 10.0 * vmin_dist

    dist_for_color = np.clip(valid_dist["solver_distance"].to_numpy(dtype=float), vmin_dist, None)
    sc_dist = ax_dist.scatter(
        valid_dist[cols.mach],
        valid_dist[cols.alpha],
        c=dist_for_color,
        cmap="plasma",
        norm=LogNorm(vmin=vmin_dist, vmax=vmax_dist),
        s=34,
        edgecolors="none",
        zorder=3,
    )

    n_top_dist = max(1, int(np.ceil(0.10 * len(valid_dist))))
    top_dist = valid_dist.nlargest(n_top_dist, "solver_distance")
    ax_dist.scatter(
        top_dist[cols.mach],
        top_dist[cols.alpha],
        s=72,
        facecolors="none",
        edgecolors="cyan",
        linewidths=1.5,
        label="Top 10% errors",
        zorder=6,
    )

    missing_dist = df.loc[~np.isfinite(df["solver_distance"])]
    if not missing_dist.empty:
        ax_dist.scatter(
            missing_dist[cols.mach],
            missing_dist[cols.alpha],
            marker="x",
            s=70,
            c="k",
            linewidths=1.8,
            zorder=5,
            label="No solver distance",
        )

    cbar2 = fig.colorbar(sc_dist, ax=ax_dist)
    cbar2.set_label("Geometric distance in the (M, alpha) plane")

    ax_dist.set_title("Point-to-solver-isoline distance")
    ax_dist.set_xlabel("Mach number (M)")
    ax_dist.set_ylabel(r"Wavenumber ($\alpha$)")
    ax_dist.grid(True, alpha=0.25, linewidth=0.6)
    ax_dist.tick_params(direction="in", top=True, right=True)
    ax_dist.legend(loc="best", fontsize=9, frameon=True)

    xlim = padded_limits(df[cols.mach].to_numpy(dtype=float))
    ylim = padded_limits(df[cols.alpha].to_numpy(dtype=float))
    ax_err.set_xlim(*xlim)
    ax_err.set_ylim(*ylim)

    fig.suptitle("Supersonic shooting error map", fontsize=18)

    pdf_path = output_dir / "Fig_supersonic_blumen_error_distance.pdf"
    png_path = output_dir / "Fig_supersonic_blumen_error_distance.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()
    input_csv = args.input or (
        repo_root
        / "assets/article/classical_supersonic/source_data/"
        / "supersonic_blumen_positive_pointwise_values.csv"
    )
    output_dir = args.output_dir or (
        repo_root / "assets/article/classical_supersonic/figures"
    )

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df_raw = pd.read_csv(input_csv)
    cols = resolve_columns(df_raw, args.order_column)
    df = prepare_dataframe(df_raw, cols)

    present_levels = [
        float(level)
        for level in LEVELS
        if np.any(
            np.isclose(
                df[cols.blumen_ci].to_numpy(dtype=float),
                level,
                rtol=0.0,
                atol=5.0e-6,
            )
        )
    ]
    if not present_levels:
        raise ValueError(f"None of the expected Blumen levels {LEVELS.tolist()} is present.")

    output_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    cmap_levels = mpl.colormaps["tab10"]
    level_colors = {level: cmap_levels(i) for i, level in enumerate(present_levels)}

    ordered_curves = build_ordered_curves(df, cols, present_levels)
    df = attach_solver_distances(df, cols, ordered_curves)

    make_isoline_figure(df, cols, output_dir, args.dpi, present_levels, ordered_curves, level_colors)
    make_error_distance_figure(df, cols, output_dir, args.dpi)

    print(
        f"Rows: {len(df)} | accepted: {int(df['accepted'].sum())} | "
        f"missing/rejected: {int((~df['accepted']).sum())}"
    )


if __name__ == "__main__":
    main()
