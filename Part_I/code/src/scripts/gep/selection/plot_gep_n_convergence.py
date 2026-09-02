#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).casefold())


def resolve_column(
    frame: pd.DataFrame,
    *,
    exact: tuple[str, ...] = (),
    all_tokens: tuple[str, ...] = (),
    any_tokens: tuple[str, ...] = (),
    exclude_tokens: tuple[str, ...] = (),
    required: bool = True,
    label: str,
) -> str | None:
    columns = [str(column) for column in frame.columns]
    by_normalised = {normalise(column): column for column in columns}

    for candidate in exact:
        key = normalise(candidate)
        if key in by_normalised:
            return by_normalised[key]

    all_tokens_n = tuple(normalise(token) for token in all_tokens)
    any_tokens_n = tuple(normalise(token) for token in any_tokens)
    exclude_tokens_n = tuple(normalise(token) for token in exclude_tokens)

    matches: list[str] = []
    for column in columns:
        key = normalise(column)
        if all_tokens_n and not all(token in key for token in all_tokens_n):
            continue
        if any_tokens_n and not any(token in key for token in any_tokens_n):
            continue
        if any(token in key for token in exclude_tokens_n):
            continue
        matches.append(column)

    if matches:
        matches.sort(key=lambda item: (len(normalise(item)), item))
        return matches[0]

    if required:
        raise KeyError(
            f"Unable to identify the column for {label}.\n"
            f"Available columns: {columns}"
        )
    return None


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def resolve_or_compute_difference(
    frame: pd.DataFrame,
    *,
    quantity: str,
) -> tuple[pd.Series, str]:
    if quantity == "ci":
        difference = resolve_column(
            frame,
            exact=(
                "ci_abs_diff",
                "ci_abs_error",
                "delta_ci",
                "abs_delta_ci",
                "ci_diff_to_ref",
                "ci_diff_to_finest",
                "ci_abs_diff_to_ref",
                "ci_abs_diff_to_finest",
                "ci_error_to_ref",
                "ci_error_to_finest",
            ),
            all_tokens=("ci",),
            any_tokens=("diff", "error", "delta"),
            exclude_tokens=("omega",),
            required=False,
            label="|ci(N)-ci(Nmax)|",
        )
        current_exact = ("ci", "ci_gep", "ci_N", "ci_value")
        reference_exact = (
            "ci_ref",
            "ci_reference",
            "ci_finest",
            "ci_Nmax",
            "ci_gep_ref",
        )
    else:
        difference = resolve_column(
            frame,
            exact=(
                "omega_abs_diff",
                "omega_i_abs_diff",
                "delta_omega_i",
                "abs_delta_omega_i",
                "omega_i_diff_to_ref",
                "omega_i_diff_to_finest",
                "omega_i_abs_diff_to_ref",
                "omega_i_abs_diff_to_finest",
                "omega_i_error_to_ref",
                "omega_i_error_to_finest",
            ),
            all_tokens=("omega",),
            any_tokens=("diff", "error", "delta"),
            exclude_tokens=(),
            required=False,
            label="|omega_i(N)-omega_i(Nmax)|",
        )
        current_exact = ("omega_i", "omega", "omega_i_gep", "omega_i_N")
        reference_exact = (
            "omega_i_ref",
            "omega_ref",
            "omega_i_reference",
            "omega_i_finest",
            "omega_i_Nmax",
        )

    if difference is not None:
        return numeric(frame, difference).abs(), difference

    current = resolve_column(
        frame,
        exact=current_exact,
        required=True,
        label=f"current {quantity}",
    )
    reference = resolve_column(
        frame,
        exact=reference_exact,
        all_tokens=(("ci",) if quantity == "ci" else ("omega",)),
        any_tokens=("ref", "reference", "finest", "nmax"),
        required=True,
        label=f"reference {quantity}",
    )
    return (numeric(frame, current) - numeric(frame, reference)).abs(), f"{current} - {reference}"


def main() -> None:
    repo = Path.cwd().resolve()
    input_candidates = [
        repo / "assets/pinn_subsonic/csv/article/results_pinn/release_final/tables/Table_GEP_N_convergence.csv",
        repo / "assets/pinn_subsonic/joint_ci_mode_global_validation_v1/gep_n_convergence/GEP_N_convergence.csv",
    ]
    input_path = next((path for path in input_candidates if path.is_file()), None)
    if input_path is None:
        raise FileNotFoundError(
            "GEP_N_convergence.csv was not found. Tested:\n  "
            + "\n  ".join(str(path) for path in input_candidates)
        )

    output_directory = repo / "assets/pinn_subsonic/article/results_pinn/release_final/figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_pdf = output_directory / "SuppFig07_GEP_N_convergence.pdf"
    output_png = output_directory / "SuppFig07_GEP_N_convergence.png"

    frame = pd.read_csv(input_path)

    n_column = resolve_column(
        frame,
        exact=("N", "gep_N", "grid_N", "grid_size_N", "grid_size"),
        required=True,
        label="GEP grid size N",
    )
    eta_column = resolve_column(
        frame,
        exact=("eta", "η"),
        required=False,
        label="eta",
    )
    nmax_column = resolve_column(
        frame,
        exact=("Nmax", "N_max", "N_ref", "N_finest", "reference_N", "finest_N"),
        all_tokens=("n",),
        any_tokens=("max", "ref", "finest"),
        required=False,
        label="point-specific finest resolution Nmax",
    )

    overlap_column = resolve_column(
        frame,
        exact=(
            "p_overlap",
            "pressure_overlap",
            "p_overlap_to_ref",
            "p_overlap_to_finest",
            "pressure_overlap_to_ref",
            "pressure_overlap_to_finest",
        ),
        all_tokens=("overlap",),
        any_tokens=("p", "pressure"),
        required=True,
        label="pressure overlap",
    )

    modal_column = resolve_column(
        frame,
        exact=(
            "max_rel_l2",
            "max_rel_l2_to_ref",
            "max_rel_l2_to_finest",
            "max_field_rel_l2",
            "max_field_relative_error",
            "max_modal_error",
            "modal_error_max",
            "modal_rel_max_to_finest",
        ),
        any_tokens=("maxrell2", "maxfield", "maxmodal", "modalerrormax"),
        required=True,
        label="maximum relative error among p, rho, u and v",
    )

    n_values = numeric(frame, n_column)
    frame = frame.loc[n_values.notna()].copy()
    frame["_N"] = n_values.loc[frame.index].astype(int)

    ci_difference, ci_source = resolve_or_compute_difference(frame, quantity="ci")
    omega_difference, omega_source = resolve_or_compute_difference(frame, quantity="omega")
    frame["_delta_ci"] = ci_difference
    frame["_delta_omega"] = omega_difference
    frame["_overlap_defect"] = (1.0 - numeric(frame, overlap_column)).clip(lower=0.0)
    frame["_modal_error"] = numeric(frame, modal_column).abs()

    if nmax_column is not None:
        frame["_Nmax"] = numeric(frame, nmax_column)
    elif eta_column is not None:
        eta = numeric(frame, eta_column)
        # The historical files sometimes display 200/300/... although the
        # actual odd collocation sizes are 201/301/.... Infer the convention.
        uses_round_hundreds = bool((frame["_N"] % 100 == 0).all())
        low_reference = 500 if uses_round_hundreds else 501
        high_reference = 600 if uses_round_hundreds else 601
        frame["_Nmax"] = np.where(eta < 0.92, low_reference, high_reference)
    else:
        frame["_Nmax"] = np.nan

    if frame["_Nmax"].notna().any():
        frame = frame.loc[frame["_N"] < frame["_Nmax"]].copy()
    else:
        exact_self_comparison = (
            frame["_delta_ci"].fillna(np.inf).le(1e-15)
            & frame["_delta_omega"].fillna(np.inf).le(1e-15)
            & frame["_overlap_defect"].fillna(np.inf).le(1e-15)
            & frame["_modal_error"].fillna(np.inf).le(1e-15)
        )
        frame = frame.loc[~exact_self_comparison].copy()

    required_metrics = ["_delta_ci", "_delta_omega", "_overlap_defect", "_modal_error"]
    frame = frame.dropna(subset=["_N", *required_metrics])
    if frame.empty:
        raise ValueError("No non-reference convergence rows remain after filtering.")

    summary = (
        frame.groupby("_N", as_index=False)
        .agg(
            delta_ci_median=("_delta_ci", "median"),
            delta_ci_max=("_delta_ci", "max"),
            delta_omega_median=("_delta_omega", "median"),
            delta_omega_max=("_delta_omega", "max"),
            overlap_defect_median=("_overlap_defect", "median"),
            overlap_defect_max=("_overlap_defect", "max"),
            modal_error_median=("_modal_error", "median"),
            modal_error_max=("_modal_error", "max"),
            n_points=("_N", "size"),
        )
        .sort_values("_N")
    )

    floor = 1e-16
    x = summary["_N"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2), constrained_layout=True)

    panels = (
        (
            axes[0, 0],
            "delta_ci_median",
            "delta_ci_max",
            r"$|c_i^{(N)}-c_i^{(N_{\max})}|$",
            r"(a) Eigenvalue convergence",
        ),
        (
            axes[0, 1],
            "delta_omega_median",
            "delta_omega_max",
            r"$|\omega_i^{(N)}-\omega_i^{(N_{\max})}|$",
            r"(b) Growth-rate convergence",
        ),
        (
            axes[1, 0],
            "overlap_defect_median",
            "overlap_defect_max",
            r"$1-\mathcal{O}_p(N,N_{\max})$",
            r"(c) Pressure-overlap defect",
        ),
        (
            axes[1, 1],
            "modal_error_median",
            "modal_error_max",
            r"$\max_{f\in\{p,\rho,u,v\}}\varepsilon_f(N,N_{\max})$",
            r"(d) Complete-field modal convergence",
        ),
    )

    for axis, median_column, maximum_column, ylabel, title in panels:
        axis.semilogy(
            x,
            np.maximum(summary[median_column].to_numpy(float), floor),
            marker="o",
            linewidth=1.8,
            label="median",
        )
        axis.semilogy(
            x,
            np.maximum(summary[maximum_column].to_numpy(float), floor),
            marker="s",
            linewidth=1.8,
            label="maximum",
        )
        axis.set_xlabel("GEP grid size $N$")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(loc="best")

    fig.suptitle(
        "PINN-seeded dense GEP resolution convergence on 20 representative points",
        fontsize=15,
    )

    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)

    print(f"Input: {input_path}")
    print(f"N column: {n_column}")
    print(f"Nmax column: {nmax_column or 'derived from eta'}")
    print(f"ci difference: {ci_source}")
    print(f"omega difference: {omega_source}")
    print(f"pressure overlap: {overlap_column}")
    print(f"modal error: {modal_column}")
    print("\nAggregated non-reference rows:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {output_pdf}")
    print(f"Saved: {output_png}")


if __name__ == "__main__":
    main()
