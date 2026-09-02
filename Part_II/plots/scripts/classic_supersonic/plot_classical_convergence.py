#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_ERRORS = (
    PACKAGE_ROOT
    / "reproducibility/results/classical_convergence_modalfix_v1_full/errors/convergence_errors.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "assets/classic_supersonic/article/convergence_modalfix_v2"
COMMON_REQUIRED_COLUMNS = {
    "case_id",
    "regime",
    "solver",
    "sweep_type",
    "Ly",
    "accuracy_label",
    "accuracy_order",
    "runtime_seconds",
    "converged",
    "spectral_success",
    "modal_reconstruction_success",
    "branch_check_passed",
    "overall_validated",
    "mode_file",
    "branch_distance",
    "abs_error_cr",
    "abs_error_ci",
    "abs_error_omega_i",
    "mode_error_max_core",
    "mode_error_max_full",
}
FIGURE_REQUIRED_COLUMNS = {
    "shooting-spectral": COMMON_REQUIRED_COLUMNS,
    "modal": COMMON_REQUIRED_COLUMNS,
    "gep": COMMON_REQUIRED_COLUMNS | {"n_points"},
    "all": COMMON_REQUIRED_COLUMNS | {"n_points"},
    "box": COMMON_REQUIRED_COLUMNS
    | {"reference_run_id", "complex_error_c", "core_threshold"},
    "integration": COMMON_REQUIRED_COLUMNS
    | {
        "reference_run_id",
        "complex_error_c",
        "core_threshold",
        "rtol",
        "atol",
        "max_step",
    },
    "both": COMMON_REQUIRED_COLUMNS
    | {
        "reference_run_id",
        "complex_error_c",
        "core_threshold",
        "rtol",
        "atol",
        "max_step",
    },
}
REQUIRED_COLUMNS = FIGURE_REQUIRED_COLUMNS["all"]


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_plot_frame(path: Path, *, figure: str = "all") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Convergence error CSV not found: {path}")
    frame = pd.read_csv(path)
    if figure not in FIGURE_REQUIRED_COLUMNS:
        raise ValueError(f"Unsupported figure selection: {figure!r}")
    missing = sorted(FIGURE_REQUIRED_COLUMNS[figure] - set(frame.columns))
    if missing:
        raise ValueError(f"Convergence error CSV is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Convergence error CSV is empty.")
    return frame


def _subset(frame: pd.DataFrame, *, regime: str | None, sweep_type: str, solver: str | None = None) -> pd.DataFrame:
    selected = frame[frame["sweep_type"].eq(sweep_type)].copy()
    if regime is not None:
        selected = selected[selected["regime"].eq(regime)]
    if solver is not None:
        selected = selected[selected["solver"].eq(solver)]
    return selected


def _positive(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return np.where(array > 0.0, array, np.nan)


def _case_label(case_id: str) -> str:
    return case_id.replace("subsonic_", "sub ").replace("supersonic_", "sup ").replace("_", " ")


def _plot_box_errors(ax, frame: pd.DataFrame, metrics: list[tuple[str, str]]) -> None:
    for metric, short_name in metrics:
        metric_frame = frame.dropna(subset=[metric]).copy()
        grouping = ["case_id"]
        if metric.endswith("_core") and "core_threshold" in metric_frame:
            grouping.append("core_threshold")
        else:
            metric_frame = metric_frame.drop_duplicates("run_id")
        for key, group in metric_frame.groupby(grouping, sort=True):
            case_id = key[0] if isinstance(key, tuple) else key
            threshold_label = (
                f" threshold={key[1]:g}"
                if isinstance(key, tuple) and len(key) > 1
                else ""
            )
            group = group.sort_values("Ly")
            ax.plot(
                group["Ly"],
                _positive(group[metric]),
                marker="o",
                label=f"{_case_label(case_id)} {short_name}{threshold_label}",
            )
    ax.set_xlabel(r"Demi-boite $L_y$")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    if ax.lines:
        ax.legend(fontsize=7)


def _plot_accuracy_errors(ax, frame: pd.DataFrame, metrics: list[tuple[str, str]]) -> None:
    tick_positions: set[int] = set()
    tick_labels: dict[int, str] = {}
    for metric, short_name in metrics:
        metric_frame = frame.dropna(subset=[metric]).copy()
        grouping = ["case_id"]
        if metric.endswith("_core") and "core_threshold" in metric_frame:
            grouping.append("core_threshold")
        else:
            metric_frame = metric_frame.drop_duplicates("run_id")
        for key, group in metric_frame.groupby(grouping, sort=True):
            case_id = key[0] if isinstance(key, tuple) else key
            threshold_label = (
                f" threshold={key[1]:g}"
                if isinstance(key, tuple) and len(key) > 1
                else ""
            )
            group = group.sort_values("accuracy_order")
            x = group["accuracy_order"].astype(int).to_numpy()
            for position, label in zip(x, group["accuracy_label"].astype(str)):
                tick_positions.add(int(position))
                tick_labels[int(position)] = label.replace("strict_", "").replace("_tolerance", "")
            ax.plot(
                x,
                _positive(group[metric]),
                marker="o",
                label=f"{_case_label(case_id)} {short_name}{threshold_label}",
            )
    ticks = sorted(tick_positions)
    ax.set_xticks(ticks, [tick_labels[value] for value in ticks], rotation=25, ha="right")
    ax.set_xlabel("Niveau de precision RK45")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    if ax.lines:
        ax.legend(fontsize=7)


def _empty_panel(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.2)


def _mark_modal_failures(ax, frame: pd.DataFrame, axis: str) -> None:
    if frame.empty:
        return
    failed = frame[~_as_bool(frame["modal_reconstruction_success"])].drop_duplicates("run_id")
    if failed.empty:
        return
    x = pd.to_numeric(failed[axis], errors="coerce")
    valid = x.notna()
    ax.scatter(
        x[valid],
        np.full(int(valid.sum()), 0.035),
        transform=ax.get_xaxis_transform(),
        marker="x",
        color="crimson",
        label="reconstruction modale echouee (marqueur de statut)",
        zorder=5,
    )


def _annotate_unresolved_m140(ax, frame: pd.DataFrame) -> None:
    if "branch_provenance_status" not in frame:
        return
    if frame["branch_provenance_status"].astype(str).eq("unresolved_current_solver_mismatch").any():
        ax.text(
            0.02,
            0.03,
            "M=1.4 : provenance de branche non resolue",
            transform=ax.transAxes,
            fontsize=8,
            color="darkred",
        )


def _title_panels(axes, titles: list[str]) -> None:
    for label, ax, title in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat, titles):
        ax.set_title(f"{label} {title}", loc="left", fontsize=10)


def build_shooting_spectral_figure(frame: pd.DataFrame):
    spectral = frame[_as_bool(frame["spectral_success"])].drop_duplicates("run_id")
    sub_box = _subset(spectral, regime="subsonic", sweep_type="shooting_box")
    sub_accuracy = _subset(spectral, regime="subsonic", sweep_type="shooting_accuracy")
    sup_box = _subset(spectral, regime="supersonic", sweep_type="shooting_box")
    sup_accuracy = _subset(spectral, regime="supersonic", sweep_type="shooting_accuracy")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    _plot_box_errors(axes[0, 0], sub_box, [("abs_error_ci", r"$c_i$"), ("abs_error_omega_i", r"$\omega_i$")])
    _plot_accuracy_errors(
        axes[0, 1], sub_accuracy, [("abs_error_ci", r"$c_i$"), ("abs_error_omega_i", r"$\omega_i$")]
    )
    for ax, subset in zip(axes.flat, (sub_box, sub_accuracy, sup_box, sup_accuracy)):
        if subset.empty:
            _empty_panel(ax, "Donnees spectrales indisponibles")
    _annotate_unresolved_m140(axes[1, 0], sup_box)
    _annotate_unresolved_m140(axes[1, 1], sup_accuracy)
    _plot_box_errors(axes[1, 0], sup_box, [("abs_error_cr", r"$c_r$"), ("abs_error_ci", r"$c_i$")])
    _plot_accuracy_errors(
        axes[1, 1], sup_accuracy, [("abs_error_cr", r"$c_r$"), ("abs_error_ci", r"$c_i$")]
    )
    for ax in axes.flat:
        ax.set_ylabel("Erreur absolue")
    _title_panels(
        axes,
        [
            "Subsonique, taille de boite",
            "Subsonique, precision d'integration",
            "Supersonique, taille de boite",
            "Supersonique, precision d'integration",
        ],
    )
    return fig


def build_gep_figure(frame: pd.DataFrame):
    gep = _subset(
        frame[_as_bool(frame["spectral_success"])].drop_duplicates("run_id"),
        regime="supersonic",
        sweep_type="supersonic_gep_resolution",
        solver="supersonic_gep",
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)

    for case_id, group in gep.groupby("case_id", sort=True):
        group = group.sort_values("n_points")
        x = group["n_points"]
        axes[0, 0].plot(x, _positive(group["abs_error_cr"]), marker="o", label=f"{_case_label(case_id)} $c_r$")
        axes[0, 0].plot(x, _positive(group["abs_error_ci"]), marker="s", label=f"{_case_label(case_id)} $c_i$")
        axes[0, 1].plot(x, _positive(group["branch_distance"]), marker="o", label=_case_label(case_id))
        axes[1, 1].plot(x, group["runtime_seconds"], marker="o", label=_case_label(case_id))

        modal = group.dropna(subset=["mode_error_max_core", "mode_error_max_full"])
        if not modal.empty:
            axes[1, 0].plot(
                modal["n_points"], _positive(modal["mode_error_max_core"]), marker="o", label=f"{_case_label(case_id)} core"
            )
            axes[1, 0].plot(
                modal["n_points"], _positive(modal["mode_error_max_full"]), marker="s", label=f"{_case_label(case_id)} full"
            )

    axes[0, 0].set_ylabel("Erreur absolue")
    axes[0, 1].set_ylabel(r"$|c_{GEP}-c_{seed}|$")
    axes[1, 0].set_ylabel("Erreur modale relative L2")
    axes[1, 1].set_ylabel("Temps [s]")
    for ax in axes.flat:
        ax.set_xlabel("Points de collocation GEP $n_{points}$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_yscale("log")
    if gep.empty:
        for ax in axes.flat:
            _empty_panel(ax, "Donnees GEP indisponibles")
    elif not axes[1, 0].lines:
        axes[1, 0].text(0.5, 0.5, "Modes GEP indisponibles", ha="center", va="center", transform=axes[1, 0].transAxes)
    _title_panels(
        axes,
        [
            "Valeurs propres",
            "Distance GEP-shooting",
            "Modes core et full",
            "Cout de calcul",
        ],
    )
    return fig


def build_modal_figure(frame: pd.DataFrame):
    has_mode = frame["mode_file"].fillna("").astype(str).str.len().gt(0)
    modal_ok = _as_bool(frame["modal_reconstruction_success"])
    modal_frame = frame[has_mode & modal_ok].copy()
    sub_box = _subset(modal_frame, regime="subsonic", sweep_type="shooting_box")
    sub_accuracy = _subset(
        modal_frame, regime="subsonic", sweep_type="shooting_accuracy"
    )
    sup_box = _subset(modal_frame, regime="supersonic", sweep_type="shooting_box")
    sup_accuracy = _subset(
        modal_frame,
        regime="supersonic",
        sweep_type="shooting_accuracy",
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    _plot_box_errors(axes[0, 0], sub_box, [("mode_error_max_full", "full")])
    _plot_accuracy_errors(axes[0, 1], sub_accuracy, [("mode_error_max_full", "full")])
    _plot_box_errors(
        axes[1, 0], sup_box, [("mode_error_max_core", "core"), ("mode_error_max_full", "full")]
    )
    _plot_accuracy_errors(
        axes[1, 1], sup_accuracy, [("mode_error_max_core", "core"), ("mode_error_max_full", "full")]
    )
    all_subsets = (
        _subset(frame, regime="subsonic", sweep_type="shooting_box"),
        _subset(frame, regime="subsonic", sweep_type="shooting_accuracy"),
        _subset(frame, regime="supersonic", sweep_type="shooting_box"),
        _subset(frame, regime="supersonic", sweep_type="shooting_accuracy"),
    )
    for ax, subset, axis in zip(axes.flat, all_subsets, ("Ly", "accuracy_order", "Ly", "accuracy_order")):
        _mark_modal_failures(ax, subset, axis)
        if subset.empty:
            _empty_panel(ax, "Donnees modales indisponibles")
        if ax.lines or ax.collections:
            ax.legend(fontsize=7)
    _annotate_unresolved_m140(axes[1, 0], sup_box)
    _annotate_unresolved_m140(axes[1, 1], sup_accuracy)
    for ax in axes.flat:
        ax.set_ylabel("Erreur modale relative L2")
    _title_panels(
        axes,
        [
            "Subsonique, taille de boite",
            "Subsonique, precision d'integration",
            "Supersonique, taille de boite",
            "Supersonique, precision d'integration",
        ],
    )
    return fig


ARTICLE_CASES = {
    "subsonic_interior": ("Interior", "#16697A"),
    "subsonic_neutral": ("Near-neutral", "#D17A22"),
    "supersonic_M180": (r"$M=1.8$", "#16697A"),
    "supersonic_M190": (r"$M=1.9$", "#D17A22"),
}


def validate_reference_integrity(frame: pd.DataFrame) -> None:
    """Ensure every referenced run belongs to the same case and sweep."""

    rows = frame.drop_duplicates("run_id").set_index("run_id")
    for reference_id in frame["reference_run_id"].dropna().astype(str).unique():
        if reference_id not in rows.index:
            raise ValueError(f"Reference run is absent from the input CSV: {reference_id}")
    for _, row in frame.dropna(subset=["reference_run_id"]).iterrows():
        reference = rows.loc[str(row["reference_run_id"])]
        if (
            str(reference["case_id"]) != str(row["case_id"])
            or str(reference["sweep_type"]) != str(row["sweep_type"])
        ):
            raise ValueError(
                "Reference mismatch: run and reference must share case_id and sweep_type "
                f"({row['run_id']} -> {row['reference_run_id']})."
            )


def select_core_threshold(frame: pd.DataFrame, core_threshold: float) -> pd.DataFrame:
    if not 0.0 < core_threshold < 1.0:
        raise ValueError("core_threshold must be strictly between zero and one.")
    values = pd.to_numeric(frame["core_threshold"], errors="coerce")
    selected = frame[np.isclose(values, core_threshold, rtol=0.0, atol=1.0e-12)].copy()
    if selected.empty:
        available = sorted(values.dropna().unique())
        raise ValueError(
            f"core_threshold={core_threshold:g} is unavailable; available values: {available}"
        )
    return selected


def exclude_reference_runs(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove self-comparisons before logarithmic plotting."""

    reference = frame["reference_run_id"].fillna("").astype(str)
    return frame[frame["run_id"].astype(str).ne(reference)].copy()


def validate_one_parameter_sweep(
    frame: pd.DataFrame,
    *,
    parameter: str,
    fixed_columns: Iterable[str],
    minimum_levels: int = 3,
) -> None:
    if frame.empty:
        raise ValueError("The proposed integration sweep is empty.")
    levels = pd.to_numeric(frame[parameter], errors="coerce").dropna().unique()
    if len(levels) < minimum_levels:
        raise ValueError(
            f"At least {minimum_levels} numeric {parameter} levels are required; found {len(levels)}."
        )
    varying = [column for column in fixed_columns if frame[column].dropna().nunique() > 1]
    if varying:
        raise ValueError(
            "Pseudo-sweep rejected because parameters other than "
            f"{parameter} vary: {varying}"
        )


def _reference_note(frame: pd.DataFrame, *, parameter: str) -> str:
    rows = frame.drop_duplicates("run_id").set_index("run_id")
    reference_values: list[float] = []
    for case_id, group in frame.groupby("case_id", sort=True):
        references = group["reference_run_id"].dropna().astype(str).unique()
        if len(references) != 1:
            raise ValueError(f"Expected one reference for {case_id}, found {len(references)}.")
        reference_id = references[0]
        reference = rows.loc[reference_id]
        reference_values.append(float(reference[parameter]))
    unique_values = sorted(set(reference_values))
    if len(unique_values) == 1:
        return f"Reference: {parameter}={unique_values[0]:g} (excluded)."
    values = ", ".join(f"{value:g}" for value in unique_values)
    return f"References: {parameter}={values} (excluded)."


def _plot_article_series(
    ax,
    frame: pd.DataFrame,
    *,
    x_column: str,
    metric: str,
    label: str,
    color: str,
    linestyle: str = "-",
    marker: str = "o",
    zero_tolerance: float,
) -> str:
    group = frame.sort_values(x_column)
    x = pd.to_numeric(group[x_column], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if not len(y):
        return f"{label}: unavailable"

    resolved = y > zero_tolerance
    censored = ~resolved
    if resolved.any():
        ax.plot(
            x[resolved],
            y[resolved],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.8,
            markersize=5,
            label=label,
        )
    if censored.any():
        ax.plot(
            x[censored],
            np.full(int(censored.sum()), zero_tolerance),
            color=color,
            linestyle="None",
            marker="v",
            markerfacecolor="none" if linestyle == "--" else color,
            markersize=6,
            label=label if not resolved.any() else rf"{label} ($<\tau$)",
        )
    if not resolved.any():
        return f"{label}: maximum variation < {zero_tolerance:.0e}"
    if censored.any():
        return f"{label}: {int(censored.sum())} value(s) < {zero_tolerance:.0e}"
    return ""


def _finish_article_panel(
    ax,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    notes: list[str],
) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, which="both", color="#B8C0C2", alpha=0.35, linewidth=0.7)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8, frameon=False)
    text = "\n".join(note for note in notes if note)
    if text:
        ax.text(
            0.02,
            0.03,
            text,
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            color="#3E4B4D",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
        )


def _article_case_frame(
    frame: pd.DataFrame,
    *,
    case_id: str,
    sweep_type: str,
) -> pd.DataFrame:
    selected = frame[
        frame["case_id"].eq(case_id)
        & frame["sweep_type"].eq(sweep_type)
        & _as_bool(frame["overall_validated"])
    ].copy()
    if selected.empty:
        raise ValueError(f"No overall-validated data for {case_id} / {sweep_type}.")
    return selected


def build_box_convergence_figure(
    frame: pd.DataFrame,
    *,
    core_threshold: float = 1.0e-2,
    zero_tolerance: float = 1.0e-14,
):
    validate_reference_integrity(frame)
    selected = select_core_threshold(frame, core_threshold)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)

    sub_full = pd.concat(
        [
            _article_case_frame(selected, case_id=case, sweep_type="shooting_box")
            for case in ("subsonic_interior", "subsonic_neutral")
        ],
        ignore_index=True,
    )
    sub_data = exclude_reference_runs(sub_full)
    notes = []
    for case in ("subsonic_interior", "subsonic_neutral"):
        label, color = ARTICLE_CASES[case]
        notes.append(
            _plot_article_series(
                axes[0, 0],
                sub_data[sub_data["case_id"].eq(case)],
                x_column="Ly",
                metric="abs_error_ci",
                label=label,
                color=color,
                zero_tolerance=zero_tolerance,
            )
        )
    notes.append(_reference_note(sub_full, parameter="Ly"))
    _finish_article_panel(
        axes[0, 0],
        title="(a) Subsonic spectral convergence",
        xlabel=r"Domain half-width $L_y$",
        ylabel=r"$|c_i-c_{i,\mathrm{ref}}|$",
        notes=notes,
    )

    sup_full = pd.concat(
        [
            _article_case_frame(selected, case_id=case, sweep_type="shooting_box")
            for case in ("supersonic_M180", "supersonic_M190")
        ],
        ignore_index=True,
    )
    sup_data = exclude_reference_runs(sup_full)
    notes = []
    for case in ("supersonic_M180", "supersonic_M190"):
        label, color = ARTICLE_CASES[case]
        notes.append(
            _plot_article_series(
                axes[0, 1],
                sup_data[sup_data["case_id"].eq(case)],
                x_column="Ly",
                metric="complex_error_c",
                label=label,
                color=color,
                zero_tolerance=zero_tolerance,
            )
        )
    notes.extend(
        [
            _reference_note(sup_full, parameter="Ly"),
            "M=1.4 excluded: unresolved branch provenance.",
            r"Open triangles denote upper bounds $<\tau$, not measured errors.",
        ]
    )
    _finish_article_panel(
        axes[0, 1],
        title="(b) Supersonic spectral convergence",
        xlabel=r"Domain half-width $L_y$",
        ylabel=r"$|c-c_{\mathrm{ref}}|$",
        notes=notes,
    )

    modal_sub = sub_data[
        _as_bool(sub_data["modal_reconstruction_success"])
        & sub_data["mode_file"].fillna("").astype(str).str.len().gt(0)
    ]
    notes = []
    for case in ("subsonic_interior", "subsonic_neutral"):
        label, color = ARTICLE_CASES[case]
        notes.append(
            _plot_article_series(
                axes[1, 0],
                modal_sub[modal_sub["case_id"].eq(case)],
                x_column="Ly",
                metric="mode_error_max_full",
                label=f"{label}, full",
                color=color,
                linestyle="--",
                zero_tolerance=zero_tolerance,
            )
        )
    notes.extend(
        [
            _reference_note(sub_full, parameter="Ly"),
            f"Complex alignment uses core threshold {core_threshold:g}.",
        ]
    )
    _finish_article_panel(
        axes[1, 0],
        title="(c) Subsonic modal convergence",
        xlabel=r"Domain half-width $L_y$",
        ylabel="Maximum relative modal error (full domain)",
        notes=notes,
    )

    modal_sup = sup_data[
        _as_bool(sup_data["modal_reconstruction_success"])
        & sup_data["mode_file"].fillna("").astype(str).str.len().gt(0)
    ]
    notes = []
    for case in ("supersonic_M180", "supersonic_M190"):
        label, color = ARTICLE_CASES[case]
        notes.append(
            _plot_article_series(
                axes[1, 1],
                modal_sup[modal_sup["case_id"].eq(case)],
                x_column="Ly",
                metric="mode_error_max_core",
                label=f"{label}, core",
                color=color,
                linestyle="-",
                zero_tolerance=zero_tolerance,
            )
        )
        notes.append(
            _plot_article_series(
                axes[1, 1],
                modal_sup[modal_sup["case_id"].eq(case)],
                x_column="Ly",
                metric="mode_error_max_full",
                label=f"{label}, full",
                color=color,
                linestyle="--",
                marker="s",
                zero_tolerance=zero_tolerance,
            )
        )
    notes.extend(
        [
            _reference_note(sup_full, parameter="Ly"),
            f"Core threshold: {core_threshold:g}. M=1.4 excluded.",
        ]
    )
    _finish_article_panel(
        axes[1, 1],
        title="(d) Supersonic modal convergence",
        xlabel=r"Domain half-width $L_y$",
        ylabel="Maximum relative modal error",
        notes=notes,
    )
    return fig


def prepare_integration_sweep(
    frame: pd.DataFrame,
    *,
    core_threshold: float = 1.0e-2,
) -> pd.DataFrame:
    validate_reference_integrity(frame)
    selected = select_core_threshold(frame, core_threshold)
    selected = selected[
        selected["case_id"].isin(["supersonic_M180", "supersonic_M190"])
        & selected["sweep_type"].eq("shooting_accuracy")
        & _as_bool(selected["overall_validated"])
        & np.isclose(pd.to_numeric(selected["rtol"], errors="coerce"), 1.0e-10)
        & np.isclose(pd.to_numeric(selected["atol"], errors="coerce"), 1.0e-12)
        & pd.to_numeric(selected["max_step"], errors="coerce").notna()
    ].copy()
    for case_id, group in selected.groupby("case_id", sort=True):
        validate_one_parameter_sweep(
            group,
            parameter="max_step",
            fixed_columns=(
                "Mach",
                "alpha",
                "Ly",
                "rtol",
                "atol",
                "mapping_kind",
                "mapping_scale",
                "matching_y",
            ),
            minimum_levels=4,
        )
        non_reference = exclude_reference_runs(group)
        if non_reference["max_step"].nunique() < 3:
            raise ValueError(
                f"{case_id} has fewer than three comparable non-reference max_step levels."
            )
    if set(selected["case_id"]) != {"supersonic_M180", "supersonic_M190"}:
        raise ValueError("The integration figure requires valid M1.8 and M1.9 sweeps.")
    return selected


def build_integration_convergence_figure(
    frame: pd.DataFrame,
    *,
    core_threshold: float = 1.0e-2,
    zero_tolerance: float = 1.0e-14,
):
    selected = prepare_integration_sweep(frame, core_threshold=core_threshold)
    data = exclude_reference_runs(selected)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)

    statuses = []
    for case in ("supersonic_M180", "supersonic_M190"):
        label, color = ARTICLE_CASES[case]
        statuses.append(
            _plot_article_series(
                axes[0],
                data[data["case_id"].eq(case)],
                x_column="max_step",
                metric="complex_error_c",
                label=label,
                color=color,
                zero_tolerance=zero_tolerance,
            )
        )
    all_censored = statuses and all("maximum variation" in status for status in statuses)
    notes = [
        r"All variations $<10^{-14}$; triangles are upper bounds."
        if all_censored
        else "; ".join(status for status in statuses if status),
        _reference_note(selected, parameter="max_step"),
        r"Fixed $rtol=10^{-10}$, $atol=10^{-12}$.",
    ]
    _finish_article_panel(
        axes[0],
        title="(a) Spectral integration convergence",
        xlabel="Maximum RK45 step",
        ylabel=r"$|c-c_{\mathrm{ref}}|$",
        notes=notes,
    )

    statuses = []
    for case in ("supersonic_M180", "supersonic_M190"):
        label, color = ARTICLE_CASES[case]
        statuses.append(
            _plot_article_series(
                axes[1],
                data[data["case_id"].eq(case)],
                x_column="max_step",
                metric="mode_error_max_core",
                label=f"{label}, core",
                color=color,
                zero_tolerance=zero_tolerance,
            )
        )
        statuses.append(
            _plot_article_series(
                axes[1],
                data[data["case_id"].eq(case)],
                x_column="max_step",
                metric="mode_error_max_full",
                label=f"{label}, full",
                color=color,
                linestyle="--",
                marker="s",
                zero_tolerance=zero_tolerance,
            )
        )
    all_censored = statuses and all("maximum variation" in status for status in statuses)
    notes = [
        r"All variations $<10^{-14}$; triangles are upper bounds."
        if all_censored
        else "; ".join(status for status in statuses if status),
        _reference_note(selected, parameter="max_step"),
        f"Core threshold: {core_threshold:g}.",
    ]
    _finish_article_panel(
        axes[1],
        title="(b) Modal integration convergence",
        xlabel="Maximum RK45 step",
        ylabel="Maximum relative modal error",
        notes=notes,
    )
    for ax in axes:
        ax.set_xticks([2.0, 1.0, 0.5], ["2", "1", "0.5"])
        ax.set_xlim(2.15, 0.42)
    return fig


def _save(fig, output_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        path = output_dir / f"{stem}.{file_format}"
        kwargs = {"dpi": dpi} if file_format == "png" else {}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create article figures from validated classical convergence errors.")
    parser.add_argument("--errors-csv", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formats", nargs="+", choices=["pdf", "png"], default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--core-threshold", type=float, default=1.0e-2)
    parser.add_argument(
        "--zero-tolerance",
        type=float,
        default=1.0e-14,
        help="Reporting threshold for censored zero/roundoff values; never used as data.",
    )
    parser.add_argument(
        "--figure",
        choices=["box", "integration", "both", "shooting-spectral", "modal", "gep", "all"],
        default="both",
        help="Build corrected article figures or an explicitly requested legacy family.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate all panel inputs without creating figures.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.zero_tolerance <= 0.0:
        raise ValueError("--zero-tolerance must be positive.")
    frame = load_plot_frame(args.errors_csv, figure=args.figure)
    builders = {
        "box": (
            "fig_classical_box_convergence",
            lambda values: build_box_convergence_figure(
                values,
                core_threshold=args.core_threshold,
                zero_tolerance=args.zero_tolerance,
            ),
        ),
        "integration": (
            "fig_classical_integration_convergence",
            lambda values: build_integration_convergence_figure(
                values,
                core_threshold=args.core_threshold,
                zero_tolerance=args.zero_tolerance,
            ),
        ),
        "shooting-spectral": (
            "fig_classical_shooting_spectral_convergence",
            build_shooting_spectral_figure,
        ),
        "modal": ("fig_classical_modal_convergence", build_modal_figure),
        "gep": ("fig_supersonic_gep_resolution_convergence", build_gep_figure),
    }
    if args.figure == "both":
        selected = ["box", "integration"]
    elif args.figure == "all":
        selected = ["shooting-spectral", "modal", "gep"]
    else:
        selected = [args.figure]
    figures = [(builders[name][0], builders[name][1](frame)) for name in selected]
    if args.dry_run:
        for _, fig in figures:
            plt.close(fig)
        print(f"Validated {len(figures)} selected figures from {len(frame)} rows; no files written.")
        return 0
    for stem, fig in figures:
        for path in _save(fig, args.output_dir, stem, args.formats, args.dpi):
            print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
