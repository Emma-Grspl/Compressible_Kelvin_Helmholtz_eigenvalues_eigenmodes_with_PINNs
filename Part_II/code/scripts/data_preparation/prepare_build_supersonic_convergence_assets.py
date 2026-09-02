#!/usr/bin/env python3
"""Build article-ready supersonic convergence and robustness assets.

Inputs
------
Article-ready aggregated tables:
  assets/article/classical_supersonic/tables/
      supersonic_high_mach_convergence_by_point.csv
  assets/article/classical_supersonic/source_data/
      supersonic_high_mach_convergence_by_setting.csv
      supersonic_high_mach_core_tail_phase_convergence.csv

Raw robustness runs:
  classic_supersonic/reproducibility/results/**/runs/convergence_runs.csv

Every path containing QUARANTINE, OLD, backup, archive, obsolete or superseded
is excluded.
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    }
)

EXCLUDED_RE = re.compile(
    r"quarantine|(?:^|[_-])old(?:[_-]|$)|backup|archive|obsolete|superseded",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def find_repo_root(start: Path) -> Path:
    start = start.expanduser().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "assets").is_dir():
            return candidate
    raise FileNotFoundError("Impossible de trouver la racine du dépôt contenant assets/.")


def is_allowed(path: Path) -> bool:
    return not any(EXCLUDED_RE.search(part) for part in path.parts)


def normalized(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).casefold())


def print_columns(df: pd.DataFrame, label: str) -> None:
    print(f"{label} columns:")
    for column in df.columns:
        print(f"  - {column}")


def find_column(
    df: pd.DataFrame,
    aliases: Sequence[str] = (),
    *,
    contains_all: Sequence[str] = (),
    contains_any: Sequence[str] = (),
    excludes: Sequence[str] = (),
    required: bool = True,
    label: str = "",
) -> str | None:
    norm_to_original = {normalized(column): str(column) for column in df.columns}

    for alias in aliases:
        key = normalized(alias)
        if key in norm_to_original:
            return norm_to_original[key]

    all_tokens = [normalized(token) for token in contains_all]
    any_tokens = [normalized(token) for token in contains_any]
    excluded_tokens = [normalized(token) for token in excludes]

    scored: list[tuple[int, int, str]] = []
    for column in df.columns:
        original = str(column)
        key = normalized(original)

        if all_tokens and not all(token in key for token in all_tokens):
            continue
        if any_tokens and not any(token in key for token in any_tokens):
            continue
        if excluded_tokens and any(token in key for token in excluded_tokens):
            continue

        score = (
            4 * sum(token in key for token in all_tokens)
            + 2 * sum(token in key for token in any_tokens)
            - len(key)
        )
        scored.append((score, -len(original), original))

    if scored:
        scored.sort(reverse=True)
        return scored[0][2]

    if required:
        raise KeyError(
            f"Colonne introuvable pour {label or aliases}. "
            f"Colonnes disponibles : {list(df.columns)}"
        )
    return None


def to_numeric(df: pd.DataFrame, columns: Iterable[str | None]) -> None:
    for column in columns:
        if column is not None:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def save_both(fig: plt.Figure, pdf_path: Path, dpi: int) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = pdf_path.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def choose_label_column(df: pd.DataFrame) -> str:
    explicit = find_column(
        df,
        aliases=(
            "setting",
            "setting_name",
            "configuration",
            "configuration_name",
            "config",
            "config_name",
            "numerical_setting",
            "label",
            "case",
            "case_name",
        ),
        required=False,
    )
    if explicit is not None:
        return explicit

    object_columns = [
        str(column)
        for column in df.columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if object_columns:
        return object_columns[0]

    df["_configuration"] = np.arange(1, len(df) + 1).astype(str)
    return "_configuration"


def resolve_delta_column(df: pd.DataFrame, component: str) -> str:
    return find_column(
        df,
        aliases=(
            f"max_delta_{component}",
            f"max_abs_delta_{component}",
            f"maximum_delta_{component}",
            f"max_error_{component}",
            f"max_abs_error_{component}",
            f"delta_{component}_max",
            f"error_{component}_max",
            f"max_d{component}",
        ),
        contains_all=(component,),
        contains_any=("delta", "error", "err", "diff", "deviation"),
        excludes=("median", "mean", "avg"),
        label=f"maximum delta {component}",
    )


def resolve_residual_column(
    df: pd.DataFrame,
    statistic: str,
    *,
    required: bool,
) -> str | None:
    aliases = {
        "median": (
            "residual_median",
            "median_residual",
            "spectral_residual_median",
            "median_spectral_residual",
            "resid_median",
        ),
        "max": (
            "residual_max",
            "max_residual",
            "maximum_residual",
            "spectral_residual_max",
            "max_spectral_residual",
            "resid_max",
        ),
    }[statistic]

    column = find_column(df, aliases=aliases, required=False)
    if column is not None:
        return column

    return find_column(
        df,
        contains_all=("resid", statistic),
        required=required,
        label=f"{statistic} residual",
    )


def prepare_setting_table(path: Path) -> tuple[pd.DataFrame, dict[str, str | None]]:
    df = pd.read_csv(path)
    print_columns(df, "Convergence-by-setting")

    label = choose_label_column(df)
    delta_cr = resolve_delta_column(df, "cr")
    delta_ci = resolve_delta_column(df, "ci")
    residual_median = resolve_residual_column(df, "median", required=False)
    residual_max = resolve_residual_column(df, "max", required=False)

    to_numeric(df, (delta_cr, delta_ci, residual_median, residual_max))
    df = df.loc[df[delta_cr].notna() & df[delta_ci].notna()].copy()

    if residual_median is not None:
        sort_columns = [delta_ci, delta_cr, residual_median]
    else:
        sort_columns = [delta_ci, delta_cr]

    # largest deviations first, strict/reference configuration last
    df = df.sort_values(
        sort_columns,
        ascending=[False] * len(sort_columns),
        kind="mergesort",
    ).reset_index(drop=True)
    df["_rank"] = np.arange(1, len(df) + 1)

    return df, {
        "label": label,
        "delta_cr": delta_cr,
        "delta_ci": delta_ci,
        "residual_median": residual_median,
        "residual_max": residual_max,
    }


def plot_eigenvalue_convergence(
    df: pd.DataFrame,
    columns: dict[str, str | None],
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.8, 5.2),
        constrained_layout=True,
    )
    x = df["_rank"].to_numpy()
    floor = 1e-16

    axes[0].semilogy(
        x,
        np.maximum(df[columns["delta_cr"]].to_numpy(float), floor),
        marker="o",
        linewidth=2,
    )
    axes[0].set_title(r"Convergence of $c_r$")
    axes[0].set_ylabel(r"$\max |\Delta c_r|$")

    axes[1].semilogy(
        x,
        np.maximum(df[columns["delta_ci"]].to_numpy(float), floor),
        marker="o",
        linewidth=2,
    )
    axes[1].set_title(r"Convergence of $c_i$")
    axes[1].set_ylabel(r"$\max |\Delta c_i|$")

    residual_median = columns["residual_median"]
    residual_max = columns["residual_max"]

    if residual_median is not None:
        axes[2].semilogy(
            x,
            np.maximum(df[residual_median].to_numpy(float), floor),
            marker="o",
            linewidth=2,
            label="median",
        )
    if residual_max is not None:
        axes[2].semilogy(
            x,
            np.maximum(df[residual_max].to_numpy(float), floor),
            marker="s",
            linestyle="--",
            linewidth=2,
            label="maximum",
        )

    if residual_median is None and residual_max is None:
        axes[2].text(
            0.5,
            0.5,
            "No residual column in the\nby-setting article table",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
    else:
        axes[2].legend(loc="best", frameon=True)

    axes[2].set_title("Spectral residual convergence")
    axes[2].set_ylabel("Spectral residual")

    for axis in axes:
        axis.set_xlabel("Numerical configuration rank (coarse → strict)")
        axis.set_xticks(x)
        axis.grid(alpha=0.25)

    strict_label = str(df.iloc[-1][columns["label"]])
    fig.suptitle(
        f"Supersonic convergence toward the strictest configuration: {strict_label}",
        fontsize=14,
    )
    save_both(fig, output_path, dpi)


def resolve_point_table(path: Path) -> tuple[pd.DataFrame, dict[str, str | None]]:
    df = pd.read_csv(path)
    print_columns(df, "Convergence-by-point")

    mach = find_column(df, aliases=("Mach", "M", "mach_number"), label="Mach")
    alpha = find_column(df, aliases=("alpha", "wavenumber"), label="alpha")
    delta_cr = resolve_delta_column(df, "cr")
    delta_ci = resolve_delta_column(df, "ci")
    residual_median = resolve_residual_column(df, "median", required=False)
    residual_max = resolve_residual_column(df, "max", required=False)

    n_configurations = find_column(
        df,
        aliases=(
            "n_configurations",
            "n_configs",
            "number_of_configurations",
            "n_settings",
            "n_cases",
        ),
        contains_any=("nconfig", "nsetting", "numberconfig"),
        required=False,
    )
    status = find_column(
        df,
        aliases=("status", "validation_status", "convergence_status"),
        required=False,
    )

    to_numeric(
        df,
        (
            mach,
            alpha,
            delta_cr,
            delta_ci,
            residual_median,
            residual_max,
            n_configurations,
        ),
    )

    if n_configurations is None:
        df["_n_configurations"] = np.nan
        n_configurations = "_n_configurations"
    if status is None:
        df["_status"] = np.where(
            np.isfinite(df[delta_cr]) & np.isfinite(df[delta_ci]),
            "validated",
            "check",
        )
        status = "_status"

    return df, {
        "mach": mach,
        "alpha": alpha,
        "delta_cr": delta_cr,
        "delta_ci": delta_ci,
        "residual_median": residual_median,
        "residual_max": residual_max,
        "n_configurations": n_configurations,
        "status": status,
    }


def representative_point_rows(
    df: pd.DataFrame,
    columns: dict[str, str | None],
    maximum_rows: int = 8,
) -> pd.DataFrame:
    valid = df.loc[
        df[columns["mach"]].notna()
        & df[columns["alpha"]].notna()
        & df[columns["delta_cr"]].notna()
        & df[columns["delta_ci"]].notna()
    ].copy()

    if len(valid) <= maximum_rows:
        return valid.sort_values(
            [columns["mach"], columns["alpha"]],
            kind="mergesort",
        )

    selected_indices: list[int] = []

    for coordinate in (columns["mach"], columns["alpha"]):
        selected_indices.append(int(valid[coordinate].idxmin()))
        selected_indices.append(int(valid[coordinate].idxmax()))

    selected_indices.extend(
        valid.nlargest(maximum_rows, columns["delta_ci"]).index.tolist()
    )
    selected_indices.extend(
        valid.nlargest(maximum_rows, columns["delta_cr"]).index.tolist()
    )

    unique_indices = list(dict.fromkeys(selected_indices))[:maximum_rows]
    return (
        valid.loc[unique_indices]
        .sort_values([columns["mach"], columns["alpha"]], kind="mergesort")
        .reset_index(drop=True)
    )


def write_summary_tables(
    point_df: pd.DataFrame,
    columns: dict[str, str | None],
    csv_path: Path,
    tex_path: Path,
) -> None:
    selected = representative_point_rows(point_df, columns)

    output = pd.DataFrame(
        {
            "Mach": selected[columns["mach"]],
            "alpha": selected[columns["alpha"]],
            "max_delta_cr": selected[columns["delta_cr"]],
            "max_delta_ci": selected[columns["delta_ci"]],
            "residual_median": (
                selected[columns["residual_median"]]
                if columns["residual_median"] is not None
                else np.nan
            ),
            "residual_max": (
                selected[columns["residual_max"]]
                if columns["residual_max"] is not None
                else np.nan
            ),
            "n_configurations": selected[columns["n_configurations"]],
            "status": selected[columns["status"]].astype(str),
        }
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(csv_path, index=False)

    formatters = {
        "Mach": lambda value: f"{value:.3f}",
        "alpha": lambda value: f"{value:.4f}",
        "max_delta_cr": lambda value: f"{value:.3e}",
        "max_delta_ci": lambda value: f"{value:.3e}",
        "residual_median": lambda value: (
            f"{value:.3e}" if pd.notna(value) else "--"
        ),
        "residual_max": lambda value: (
            f"{value:.3e}" if pd.notna(value) else "--"
        ),
        "n_configurations": lambda value: (
            str(int(value)) if pd.notna(value) else "--"
        ),
        "status": str,
    }

    latex = output.to_latex(
        index=False,
        escape=False,
        formatters=formatters,
        column_format="rrcccccl",
        caption=(
            "Summary of supersonic numerical convergence and robustness "
            "for representative spectral points."
        ),
        label="tab:supersonic_convergence_robustness_summary",
    )
    tex_path.write_text(latex, encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")


def choose_core_tail_metric(
    df: pd.DataFrame,
    family: str,
    statistic: str,
) -> str | None:
    aliases = (
        f"{family}_{statistic}",
        f"{statistic}_{family}",
        f"{family}_error_{statistic}",
        f"{family}_rel_{statistic}",
        f"{family}_relative_{statistic}",
    )
    column = find_column(df, aliases=aliases, required=False)
    if column is not None:
        return column

    return find_column(
        df,
        contains_all=(family, statistic),
        required=False,
    )


def prepare_core_tail_phase(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(path)
    print_columns(df, "Core-tail-phase")

    label = choose_label_column(df)

    metrics: dict[str, dict[str, str | None]] = {}
    numeric_columns: list[str] = []

    for family in ("core", "tail", "phase"):
        median = choose_core_tail_metric(df, family, "median")
        maximum = choose_core_tail_metric(df, family, "max")

        if median is None and maximum is None:
            single = find_column(
                df,
                contains_all=(family,),
                contains_any=("error", "rel", "l2", "metric", family),
                required=False,
            )
            median = single

        metrics[family] = {"median": median, "max": maximum}
        numeric_columns.extend(
            column
            for column in (median, maximum)
            if column is not None
        )

    if any(
        metrics[family]["median"] is None
        and metrics[family]["max"] is None
        for family in metrics
    ):
        raise KeyError(
            "Le tableau core-tail-phase ne contient pas une métrique "
            "identifiable pour chacune des trois familles."
        )

    to_numeric(df, numeric_columns)

    ranking_columns = [
        metrics[family]["median"] or metrics[family]["max"]
        for family in ("core", "tail", "phase")
    ]
    df = df.sort_values(
        ranking_columns,
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    df["_rank"] = np.arange(1, len(df) + 1)

    return df, {"label": label, "metrics": metrics}


def plot_core_tail_phase(
    df: pd.DataFrame,
    configuration: dict[str, object],
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, 5.1),
        constrained_layout=True,
    )
    floor = 1e-16
    x = df["_rank"].to_numpy()
    metrics = configuration["metrics"]

    for axis, family in zip(axes, ("core", "tail", "phase")):
        median = metrics[family]["median"]
        maximum = metrics[family]["max"]

        if median is not None:
            axis.semilogy(
                x,
                np.maximum(df[median].to_numpy(float), floor),
                marker="o",
                linewidth=2,
                label="median" if maximum is not None else family,
            )
        if maximum is not None:
            axis.semilogy(
                x,
                np.maximum(df[maximum].to_numpy(float), floor),
                marker="s",
                linestyle="--",
                linewidth=2,
                label="maximum",
            )

        axis.set_title(f"{family.capitalize()} convergence")
        axis.set_xlabel("Numerical configuration rank (coarse → strict)")
        axis.set_ylabel("Diagnostic metric")
        axis.set_xticks(x)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", frameon=True)

    fig.suptitle(
        "Core-versus-tail-versus-phase convergence",
        fontsize=14,
    )
    save_both(fig, output_path, dpi)


def infer_parameter_from_filename(
    filename: str,
    tokens: Sequence[str],
) -> float | None:
    lowered = filename.casefold()
    for token in tokens:
        match = re.search(
            rf"{re.escape(token.casefold())}[_-]?([0-9]+(?:p[0-9]+|\.[0-9]+)?)",
            lowered,
        )
        if match:
            return float(match.group(1).replace("p", "."))
    return None


def resolve_raw_eigenvalue_column(df: pd.DataFrame, component: str) -> str:
    aliases = {
        "cr": (
            "cr",
            "c_r",
            "classical_cr",
            "candidate_cr",
            "root_cr",
            "eigenvalue_cr",
        ),
        "ci": (
            "ci",
            "c_i",
            "classical_ci",
            "candidate_ci",
            "root_ci",
            "eigenvalue_ci",
        ),
    }[component]

    column = find_column(df, aliases=aliases, required=False)
    if column is not None:
        return column

    return find_column(
        df,
        contains_all=(component,),
        excludes=("delta", "error", "err", "reference", "target"),
        label=component,
    )


def load_robustness_tables(paths: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path in paths:
        if not is_allowed(path):
            continue

        df = pd.read_csv(path)
        print_columns(df, f"Robustness: {path.name}")

        try:
            cr = resolve_raw_eigenvalue_column(df, "cr")
            ci = resolve_raw_eigenvalue_column(df, "ci")
        except KeyError as exc:
            warnings.warn(f"Skipping {path}: {exc}")
            continue

        mach = find_column(
            df,
            aliases=("Mach", "M", "mach_number"),
            required=False,
        )
        alpha = find_column(
            df,
            aliases=("alpha", "wavenumber"),
            required=False,
        )
        residual = find_column(
            df,
            aliases=(
                "residual",
                "spectral_residual",
                "root_residual",
                "matching_residual",
                "max_residual",
                "resid",
            ),
            contains_any=("resid", "mismatch"),
            required=False,
        )
        domain = find_column(
            df,
            aliases=(
                "y_limit",
                "ymax",
                "y_max",
                "domain_half_width",
                "box_half_width",
                "domain_size",
                "box_size",
                "L",
            ),
            required=False,
        )
        matching = find_column(
            df,
            aliases=(
                "match_y",
                "matching_y",
                "y_match",
                "matching_position",
                "match_position",
            ),
            required=False,
        )

        to_numeric(df, (cr, ci, mach, alpha, residual, domain, matching))

        work = pd.DataFrame(
            {
                "cr": df[cr],
                "ci": df[ci],
                "Mach": df[mach] if mach is not None else np.nan,
                "alpha": df[alpha] if alpha is not None else np.nan,
                "residual": df[residual] if residual is not None else np.nan,
                "domain_parameter": (
                    df[domain] if domain is not None else np.nan
                ),
                "matching_parameter": (
                    df[matching] if matching is not None else np.nan
                ),
                "source_file": path.name,
                "source_path": str(path),
            }
        )

        if work["domain_parameter"].isna().all():
            inferred = infer_parameter_from_filename(
                path.name,
                ("bigbox", "box", "ymax", "domain", "limit"),
            )
            if inferred is not None:
                work["domain_parameter"] = inferred

        if work["matching_parameter"].isna().all():
            inferred = infer_parameter_from_filename(
                path.name,
                ("match", "matching", "ymatch"),
            )
            if inferred is not None:
                work["matching_parameter"] = inferred

        work = work.loc[
            work["cr"].notna() & work["ci"].notna()
        ].copy()

        if work["Mach"].notna().any():
            distance = np.abs(work["Mach"] - 1.4)
            work = work.loc[
                np.isclose(
                    distance,
                    distance.min(),
                    atol=1e-12,
                    rtol=0.0,
                )
            ].copy()

        if work["alpha"].notna().any():
            distance = np.abs(work["alpha"] - 0.18)
            work = work.loc[
                np.isclose(
                    distance,
                    distance.min(),
                    atol=1e-12,
                    rtol=0.0,
                )
            ].copy()

        if not work.empty:
            frames.append(work)

    if not frames:
        raise ValueError(
            "Aucun convergence_runs.csv ne contient des colonnes "
            "d'eigenvaleur cr et ci exploitables."
        )

    combined = pd.concat(frames, ignore_index=True)

    if combined["residual"].notna().any():
        reference = combined.sort_values(
            "residual",
            kind="mergesort",
        ).iloc[0]
        reference_cr = float(reference["cr"])
        reference_ci = float(reference["ci"])
    else:
        reference_cr = float(combined["cr"].median())
        reference_ci = float(combined["ci"].median())

    combined["delta_cr"] = np.abs(combined["cr"] - reference_cr)
    combined["delta_ci"] = np.abs(combined["ci"] - reference_ci)

    return combined


def aggregate_robustness(
    df: pd.DataFrame,
    parameter: str,
) -> pd.DataFrame:
    valid = df.loc[df[parameter].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    return (
        valid.groupby(parameter, as_index=False)
        .agg(
            max_delta_cr=("delta_cr", "max"),
            max_delta_ci=("delta_ci", "max"),
            n_configurations=("source_file", "nunique"),
        )
        .sort_values(parameter, kind="mergesort")
    )


def plot_robustness(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    domain = aggregate_robustness(df, "domain_parameter")
    matching = aggregate_robustness(df, "matching_parameter")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.6, 5.1),
        constrained_layout=True,
    )
    floor = 1e-16

    def draw(
        axis: plt.Axes,
        summary: pd.DataFrame,
        x_column: str,
        title: str,
        x_label: str,
    ) -> None:
        if summary.empty:
            axis.text(
                0.5,
                0.5,
                "No explicit parameter found\nin valid non-quarantine runs",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        else:
            axis.semilogy(
                summary[x_column],
                np.maximum(summary["max_delta_cr"], floor),
                marker="o",
                linewidth=2,
                label=r"$\max|\Delta c_r|$",
            )
            axis.semilogy(
                summary[x_column],
                np.maximum(summary["max_delta_ci"], floor),
                marker="s",
                linestyle="--",
                linewidth=2,
                label=r"$\max|\Delta c_i|$",
            )
            axis.legend(loc="best", frameon=True)

        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Eigenvalue deviation")
        axis.grid(alpha=0.25)

    draw(
        axes[0],
        domain,
        "domain_parameter",
        "Domain-size robustness",
        r"Domain half-width / limit",
    )
    draw(
        axes[1],
        matching,
        "matching_parameter",
        "Matching-position robustness",
        r"Matching position $y_m$",
    )

    fig.suptitle(
        r"Supersonic robustness near $M=1.4$, $\alpha=0.18$",
        fontsize=14,
    )
    save_both(fig, output_path, dpi)


def main() -> None:
    args = parse_args()
    repo = find_repo_root(args.repo_root)

    point_path = (
        repo
        / "assets"
        / "article"
        / "classical_supersonic"
        / "tables"
        / "supersonic_high_mach_convergence_by_point.csv"
    )
    setting_path = (
        repo
        / "assets"
        / "article"
        / "classical_supersonic"
        / "source_data"
        / "supersonic_high_mach_convergence_by_setting.csv"
    )
    core_tail_path = (
        repo
        / "assets"
        / "article"
        / "classical_supersonic"
        / "source_data"
        / "supersonic_high_mach_core_tail_phase_convergence.csv"
    )

    for path in (point_path, setting_path, core_tail_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    reproducibility_root = (
        repo / "classic_supersonic" / "reproducibility" / "results"
    )
    robustness_paths = sorted(
        path
        for path in reproducibility_root.rglob("convergence_runs.csv")
        if path.is_file() and is_allowed(path)
    )
    if not robustness_paths:
        raise FileNotFoundError(
            f"No valid convergence_runs.csv under {reproducibility_root}"
        )

    figure_directory = (
        repo / "assets" / "article" / "classical_supersonic" / "figures"
    )
    table_directory = (
        repo / "assets" / "article" / "classical_supersonic" / "source_data"
    )

    setting_df, setting_columns = prepare_setting_table(setting_path)
    plot_eigenvalue_convergence(
        setting_df,
        setting_columns,
        figure_directory / "Fig_supersonic_eigenvalue_convergence.pdf",
        args.dpi,
    )

    point_df, point_columns = resolve_point_table(point_path)
    write_summary_tables(
        point_df,
        point_columns,
        table_directory / "Tab_supersonic_convergence_robustness_summary.csv",
        table_directory / "Tab_supersonic_convergence_robustness_summary.tex",
    )

    core_df, core_configuration = prepare_core_tail_phase(core_tail_path)
    plot_core_tail_phase(
        core_df,
        core_configuration,
        figure_directory / "Fig_supersonic_core_tail_phase_convergence.pdf",
        args.dpi,
    )

    robustness_df = load_robustness_tables(robustness_paths)
    plot_robustness(
        robustness_df,
        figure_directory / "Fig_supersonic_box_and_matching_robustness.pdf",
        args.dpi,
    )

    print("All convergence and robustness assets were generated.")


if __name__ == "__main__":
    main()
