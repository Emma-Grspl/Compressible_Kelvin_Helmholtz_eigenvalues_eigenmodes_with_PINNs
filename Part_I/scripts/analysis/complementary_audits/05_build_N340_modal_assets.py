#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import scripts.dev.joint_pinn_global_validation as V

from classical_solver.gep.dense_gep_notebook_style import (
    NotebookStyleDenseGEPSolver,
)

from scripts.compare_kh_subsonic_fixed_mach_modal_candidates import (
    load_classic_full_mode,
)

from scripts.dev.run_joint_chart_full_gep import (
    choose_regime,
)


ROOT = Path(__file__).resolve().parents[3]

RUN = (
    ROOT
    / "assets/pinn_subsonic/"
    "anchor_budget_runs/N340"
)

OUT = (
    ROOT
    / "assets/pinn_subsonic/"
    "article/N340"
)

OLD_POINTS = (
    ROOT
    / "assets/pinn_subsonic/"
    "paper_results_v1/data/"
    "paired_modal_validation_20.csv"
)

BASE_PLAN = (
    ROOT
    / "assets/pinn_subsonic/"
    "joint_ci_mode_atlas_v2/"
    "training_plan.tsv"
)

FIELDS = (
    "p",
    "rho",
    "u",
    "v",
)


def save(fig, name):
    fig.savefig(
        OUT / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT / f"{name}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def build_plan():
    plan = pd.read_csv(
        BASE_PLAN,
        sep="\t",
    ).copy()

    checkpoints = []

    for chart in plan["chart_id"].astype(str):
        p = (
            RUN
            / "joint"
            / chart
            / "model_state.pt"
        )

        if not p.is_file():
            p = (
                RUN
                / "joint"
                / chart
                / "model_best.pt"
            )

        if not p.is_file():
            raise FileNotFoundError(
                f"No N340 checkpoint for {chart}"
            )

        checkpoints.append(str(p))

    plan["checkpoint"] = checkpoints

    plan["chart_area"] = (
        (
            plan["mach_max"]
            - plan["mach_min"]
        )
        * (
            plan["eta_max"]
            - plan["eta_min"]
        )
    )

    return plan


def interp_complex(
    x_source,
    values,
    x_target,
):
    x_source = np.asarray(
        x_source,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=np.complex128,
    )

    order = np.argsort(x_source)

    x_source = x_source[order]
    values = values[order]

    return (
        np.interp(
            x_target,
            x_source,
            np.real(values),
        )
        + 1j
        * np.interp(
            x_target,
            x_source,
            np.imag(values),
        )
    )


def inner(a, b, y):
    return np.trapz(
        np.conj(a) * b,
        y,
    )


def pressure_align(
    predicted,
    reference,
    y,
):
    den = inner(
        predicted["p"],
        predicted["p"],
        y,
    )

    if abs(den) < 1.0e-30:
        factor = 1.0 + 0.0j
    else:
        factor = (
            inner(
                predicted["p"],
                reference["p"],
                y,
            )
            / den
        )

    return {
        name:
            factor
            * np.asarray(
                predicted[name],
                dtype=np.complex128,
            )
        for name in FIELDS
    }


def relative_l2(
    predicted,
    reference,
    y,
):
    numerator = np.trapz(
        np.abs(
            predicted - reference
        ) ** 2,
        y,
    )

    denominator = np.trapz(
        np.abs(reference) ** 2,
        y,
    )

    return float(
        np.sqrt(
            numerator
            / max(
                denominator,
                1.0e-30,
            )
        )
    )


def overlap(
    predicted,
    reference,
    y,
):
    num = abs(
        inner(
            predicted,
            reference,
            y,
        )
    )

    den = math.sqrt(
        max(
            float(
                np.real(
                    inner(
                        predicted,
                        predicted,
                        y,
                    )
                )
            ),
            0.0,
        )
        * max(
            float(
                np.real(
                    inner(
                        reference,
                        reference,
                        y,
                    )
                )
            ),
            0.0,
        )
    )

    return float(
        num / max(den, 1.0e-30)
    )


def direct_pinn(
    checkpoint,
    mach,
    alpha,
    y,
    device,
):
    (
        field,
        ci_net,
        module,
        _args,
        family,
    ) = V.evaluate_pinn(
        checkpoint_path=Path(checkpoint),
        device=device,
    )

    with torch.no_grad():
        p, q, ci = (
            V.call_pinn_profiles(
                field=field,
                ci_net=ci_net,
                module=module,
                family=family,
                y=y,
                alpha=alpha,
                mach=mach,
                device=device,
            )
        )

    fields = V.fields_from_pq(
        y,
        p,
        q,
        alpha,
        mach,
        ci,
    )

    result = {
        name: np.asarray(
            fields[name],
            dtype=np.complex128,
        )
        for name in FIELDS
    }

    ci = float(ci)

    del field
    del ci_net
    del module

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result, ci


def gep_mode(
    chart_id,
    mach,
    eta,
    alpha,
    ci_seed,
):
    regime_name, regime, _ = (
        choose_regime(
            chart_id,
            mach,
            eta,
        )
    )

    solver = (
        NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=mach,
            n_points=int(
                regime["N"]
            ),
            mapping_kind=str(
                regime[
                    "mapping_kind"
                ]
            ),
            mapping_scale=float(
                regime[
                    "mapping_scale"
                ]
            ),
            xi_max=float(
                regime["xi_max"]
            ),
        )
    )

    mode, source, n_modes = (
        solver.get_nearest_mode_to_target(
            target_guess=(
                0.0,
                float(ci_seed),
            ),
            prefer_positive_cr=False,
            ci_weight=2.0,
        )
    )

    if mode is None:
        raise RuntimeError(
            f"GEP failed: "
            f"{chart_id}, "
            f"M={mach}, eta={eta}, "
            f"source={source}, "
            f"n_modes={n_modes}"
        )

    vector = np.asarray(
        mode["vector"],
        dtype=np.complex128,
    )

    n = int(regime["N"])

    p = vector[
        2*n : 3*n
    ]

    fields = {
        "u": vector[0:n],
        "v": vector[n:2*n],
        "p": p,
        "rho": mach**2 * p,
    }

    return (
        np.asarray(
            solver.y,
            dtype=float,
        ),
        fields,
        float(mode["ci"]),
        float(mode.get("cr", 0.0)),
        regime_name,
    )


def classical_mode(
    mach,
    alpha,
    y_target,
):
    fields, ci = (
        load_classic_full_mode(
            alpha,
            mach,
        )
    )

    y_source = np.asarray(
        fields["y"],
        dtype=float,
    )

    interpolated = {
        name: interp_complex(
            y_source,
            fields[name],
            y_target,
        )
        for name in FIELDS
    }

    return interpolated, float(ci)


def solve_point(
    plan,
    mach,
    eta,
    alpha,
    device,
    y,
):
    routed = V.route_chart(
        plan,
        mach,
        eta,
    )

    chart_id = str(
        routed["chart_id"]
    )

    checkpoint = str(
        routed["checkpoint"]
    )

    direct_raw, ci_pinn = (
        direct_pinn(
            checkpoint,
            mach,
            alpha,
            y,
            device,
        )
    )

    (
        y_gep,
        gep_raw_native,
        ci_gep,
        cr_gep,
        regime,
    ) = gep_mode(
        chart_id,
        mach,
        eta,
        alpha,
        ci_pinn,
    )

    gep_raw = {
        name: interp_complex(
            y_gep,
            gep_raw_native[name],
            y,
        )
        for name in FIELDS
    }

    classic, ci_classic = (
        classical_mode(
            mach,
            alpha,
            y,
        )
    )

    direct = pressure_align(
        direct_raw,
        classic,
        y,
    )

    gep = pressure_align(
        gep_raw,
        classic,
        y,
    )

    return {
        "chart_id": chart_id,
        "checkpoint": checkpoint,
        "gep_regime": regime,
        "ci_classic": ci_classic,
        "ci_pinn": ci_pinn,
        "ci_gep": ci_gep,
        "cr_gep": cr_gep,
        "classic": classic,
        "direct": direct,
        "gep": gep,
    }


def evaluate_20(
    plan,
    device,
    y,
):
    points = pd.read_csv(
        OLD_POINTS
    ).copy()

    if len(points) != 20:
        raise RuntimeError(
            f"Expected 20 points, "
            f"found {len(points)}."
        )

    rows = []

    for index, point in points.iterrows():
        mach = float(
            point["Mach"]
        )

        eta = float(
            point["eta"]
        )

        alpha = float(
            point["alpha"]
        )

        print(
            f"[{index+1:02d}/20] "
            f"M={mach:.6f} "
            f"eta={eta:.6f}",
            flush=True,
        )

        result = solve_point(
            plan,
            mach,
            eta,
            alpha,
            device,
            y,
        )

        row = {
            "point_id":
                point.get(
                    "point_id",
                    f"P{index:02d}",
                ),
            "selection_stratum":
                point.get(
                    "selection_stratum",
                    "",
                ),
            "sample_group":
                point.get(
                    "sample_group",
                    "",
                ),
            "Mach": mach,
            "eta": eta,
            "alpha": alpha,
            "chart_id":
                result["chart_id"],
            "gep_regime":
                result["gep_regime"],
            "ci_classic":
                result["ci_classic"],
            "ci_pinn":
                result["ci_pinn"],
            "ci_gep":
                result["ci_gep"],
            "cr_gep":
                result["cr_gep"],
        }

        direct_errors = []
        gep_errors = []

        for name in FIELDS:
            d = relative_l2(
                result["direct"][name],
                result["classic"][name],
                y,
            )

            g = relative_l2(
                result["gep"][name],
                result["classic"][name],
                y,
            )

            row[
                f"{name}_rel_direct"
            ] = d

            row[
                f"{name}_rel_final"
            ] = g

            direct_errors.append(d)
            gep_errors.append(g)

        row["p_overlap_direct"] = (
            overlap(
                result["direct"]["p"],
                result["classic"]["p"],
                y,
            )
        )

        row["p_overlap_final"] = (
            overlap(
                result["gep"]["p"],
                result["classic"]["p"],
                y,
            )
        )

        row[
            "modal_error_mean_direct"
        ] = float(
            np.mean(
                direct_errors
            )
        )

        row[
            "modal_error_mean_final"
        ] = float(
            np.mean(
                gep_errors
            )
        )

        row[
            "modal_error_max_direct"
        ] = float(
            np.max(
                direct_errors
            )
        )

        row[
            "modal_error_max_final"
        ] = float(
            np.max(
                gep_errors
            )
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        OUT
        / "Table_paired_modal_validation_20_N340.csv",
        index=False,
    )

    return df


def plot_ecdf(df):
    fig, ax = plt.subplots(
        figsize=(7.2, 5.1)
    )

    for column, label in [
        (
            "modal_error_mean_direct",
            "Direct PINN",
        ),
        (
            "modal_error_mean_final",
            "Selected GEP",
        ),
    ]:
        values = np.sort(
            df[column]
            .to_numpy(float)
        )

        values = values[
            np.isfinite(values)
            & (values > 0)
        ]

        cumulative = (
            np.arange(
                1,
                len(values) + 1,
            )
            / len(values)
        )

        ax.step(
            values,
            cumulative,
            where="post",
            lw=2.0,
            label=(
                f"{label} "
                f"(n={len(values)})"
            ),
        )

    ax.set_xscale("log")

    ax.set_xlabel(
        "Mean relative modal error"
    )

    ax.set_ylabel(
        "Empirical cumulative fraction"
    )

    ax.set_title(
        "Paired modal reconstruction accuracy — N340"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.23,
    )

    ax.legend()

    fig.tight_layout()

    save(
        fig,
        "Fig_paired_modal_error_ecdf_20_N340",
    )


def plot_representative(
    plan,
    device,
    y,
):
    mach = 0.5
    alpha = 0.5

    eta = (
        alpha
        / math.sqrt(
            1.0 - mach**2
        )
    )

    result = solve_point(
        plan,
        mach,
        eta,
        alpha,
        device,
        y,
    )

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(10.5, 13.5),
        sharex=True,
    )

    field_titles = [
        ("p", r"$\hat p$"),
        ("rho", r"$\hat\rho$"),
        ("u", r"$\hat u$"),
        ("v", r"$\hat v$"),
    ]

    for row, (
        name,
        latex_name,
    ) in enumerate(field_titles):

        for col, (
            part_name,
            transform,
        ) in enumerate([
            ("Re", np.real),
            ("Im", np.imag),
        ]):

            ax = axes[row, col]

            ax.plot(
                y,
                transform(
                    result[
                        "classic"
                    ][name]
                ),
                lw=2.0,
                label="Classical",
            )

            ax.plot(
                y,
                transform(
                    result[
                        "direct"
                    ][name]
                ),
                "--",
                lw=1.7,
                label="Direct PINN",
            )

            ax.plot(
                y,
                transform(
                    result[
                        "gep"
                    ][name]
                ),
                ":",
                lw=2.2,
                label="Selected GEP",
            )

            ax.set_title(
                rf"{part_name}"
                rf"({latex_name})"
            )

            ax.grid(
                alpha=0.22
            )

            if row == 3:
                ax.set_xlabel(r"$y$")

    axes[0, 0].legend(
        loc="best"
    )

    fig.suptitle(
        (
            r"Eigenmode reconstruction at "
            r"$M=0.5,\ \alpha=0.5$"
            "\n"
            rf"$c_i^{{classical}}="
            f"{result['ci_classic']:.6f}"
            "  |  "
            rf"c_i^{{PINN}}="
            f"{result['ci_pinn']:.6f}"
            "  |  "
            rf"c_i^{{GEP}}="
            f"{result['ci_gep']:.6f}"
        ),
        y=0.995,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    save(
        fig,
        "Fig_representative_mode_M05_a05_N340",
    )

    # Also save numerical profiles.
    data = {
        "y": y,
    }

    for method in (
        "classic",
        "direct",
        "gep",
    ):
        for name in FIELDS:
            values = result[
                method
            ][name]

            data[
                f"{method}_{name}_real"
            ] = np.real(values)

            data[
                f"{method}_{name}_imag"
            ] = np.imag(values)

    pd.DataFrame(data).to_csv(
        OUT
        / "Data_representative_mode_M05_a05_N340.csv",
        index=False,
    )


def print_summary(df):
    print()
    print("=" * 78)
    print("N340 PAIRED MODAL SUMMARY")
    print("=" * 78)

    for metric in [
        "modal_error_mean_direct",
        "modal_error_mean_final",
        "modal_error_max_direct",
        "modal_error_max_final",
    ]:
        x = df[metric]

        print(
            f"{metric:30s}"
            f" median={x.median():.6e}"
            f" p95={x.quantile(.95):.6e}"
            f" max={x.max():.6e}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    plan = build_plan()

    y = np.linspace(
        -12.0,
        12.0,
        1601,
    )

    df = evaluate_20(
        plan,
        device,
        y,
    )

    plot_ecdf(df)

    plot_representative(
        plan,
        device,
        y,
    )

    print_summary(df)

    print()
    print("Wrote assets to:", OUT)


if __name__ == "__main__":
    main()
