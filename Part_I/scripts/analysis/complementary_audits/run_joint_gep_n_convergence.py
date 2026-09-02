#!/usr/bin/env python3
"""
Resolution-convergence audit for the joint subsonic PINN-guided dense GEP.

Subcommands
-----------
build-plan
    Select 20 representative points from the already computed central-branch
    audit and expand them over the requested N grids.

run
    Execute one (point, N) task: full dense diagonalization, select the central
    KH branch |c_r| <= cr_max by maximum c_i, compare it with the classical
    mode, and save one compact profile.

merge
    Merge all task results, compare every resolution with the finest resolution
    of the same physical point, and generate convergence tables/figures.

This script never retrains a PINN and never modifies existing GEP assets.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classical_solver.gep.dense_gep_notebook_style import (
    NotebookStyleDenseGEPSolver,
)
from scripts.compare_kh_subsonic_fixed_mach_modal_candidates import (
    load_classic_full_mode,
)
from scripts.dev.test_mid_joint_pinn_full_gep import (
    call_pinn_profiles,
    compare_mode_to_classic,
    evaluate_pinn,
    make_match_mask,
    mode_overlap_with_pinn,
    overlap_complex,
    phase_alignment,
    rel_l2,
)

DEFAULT_TRAINING_PLAN = (
    "assets/pinn_subsonic/joint_ci_mode_atlas_v2/training_plan.tsv"
)
DEFAULT_CENTRAL_AUDIT = (
    "assets/pinn_subsonic/joint_ci_mode_full_gep_atlas_v2/"
    "central_branch_audit/atlas_pointwise_central_branch.csv"
)
DEFAULT_OUTPUT = (
    "assets/pinn_subsonic/joint_ci_mode_global_validation_v1/"
    "gep_n_convergence"
)

TARGETS = [
    ("ultralow", 0.04, 0.05),
    ("low_Mach_edge", 0.10, 0.05),
    ("longwave_mid_Mach", 0.50, 0.08),
    ("longwave_high_Mach", 0.85, 0.08),
    ("extreme_longwave_high_Mach", 0.95, 0.05),
    ("interior_low_Mach", 0.20, 0.30),
    ("interior_mid_1", 0.40, 0.50),
    ("interior_mid_2", 0.60, 0.70),
    ("high_Mach_interior", 0.80, 0.40),
    ("very_high_Mach_interior", 0.95, 0.40),
    ("near_neutral_low_Mach", 0.20, 0.95),
    ("near_neutral_mid_Mach", 0.50, 0.95),
    ("near_neutral_high_Mach", 0.80, 0.95),
    ("near_neutral_HM1", 0.915, 0.95),
    ("near_neutral_HM2_eta930", 0.98, 0.93),
    ("near_neutral_HM2_eta950", 0.98, 0.95),
    ("branch_correction_case", 0.98, 0.9725),
    ("HM2_LOW_MID_seam_1", 0.915, 0.375),
    ("HM2_LOW_MID_seam_2", 0.954, 0.360),
    ("vlow_transition", 0.70, 0.11125),
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None:
        return None
    return value


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    aliases = {
        "M": "Mach",
        "mach": "Mach",
        "Eta": "eta",
        "chart": "chart_id",
    }
    for old, new in aliases.items():
        if old in result and new not in result:
            result = result.rename(columns={old: new})
    required = {"Mach", "eta", "alpha", "chart_id"}
    missing = sorted(required.difference(result.columns))
    if missing:
        raise KeyError(f"Central audit is missing columns {missing}")
    for column in ("Mach", "eta", "alpha"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["Mach", "eta", "alpha", "chart_id"])
    return result


def choose_mapping(chart_id: str, mach: float, eta: float) -> tuple[str, float, float]:
    if chart_id == "ETAEDGE_HM2B" or (mach >= 0.88 and eta <= 0.06):
        return "extreme_longwave_map20", 20.0, 0.995
    if eta <= 0.12:
        return "longwave_map10", 10.0, 0.99
    if eta >= 0.92:
        return "near_neutral_map5", 5.0, 0.98
    return "standard_map5", 5.0, 0.98


def nearest_unique_points(source: pd.DataFrame) -> pd.DataFrame:
    chosen_rows: list[dict[str, Any]] = []
    used_indices: set[int] = set()

    for stratum, target_mach, target_eta in TARGETS:
        candidates = source.loc[~source.index.isin(used_indices)].copy()
        if candidates.empty:
            raise RuntimeError("Not enough unique central-audit rows for 20 targets.")

        distance = (
            ((candidates["Mach"] - target_mach) / 0.08) ** 2
            + ((candidates["eta"] - target_eta) / 0.04) ** 2
        )
        selected_index = int(distance.idxmin())
        selected = source.loc[selected_index].to_dict()
        used_indices.add(selected_index)
        selected.update(
            {
                "selection_stratum": stratum,
                "target_Mach": float(target_mach),
                "target_eta": float(target_eta),
                "selection_distance": float(distance.loc[selected_index]),
            }
        )
        chosen_rows.append(selected)

    result = pd.DataFrame(chosen_rows)
    result.insert(
        1,
        "convergence_point_id",
        [f"CONV_{index:02d}" for index in range(len(result))],
    )
    return result


def command_build_plan(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    central = normalize_columns(pd.read_csv(args.central_audit))
    central = (
        central.sort_values(["Mach", "eta", "chart_id"])
        .drop_duplicates(["Mach", "eta", "chart_id"])
        .reset_index(drop=True)
    )
    selected = nearest_unique_points(central)

    training = pd.read_csv(args.training_plan, sep="\t")
    required_training = {"chart_id", "output_dir"}
    missing = sorted(required_training.difference(training.columns))
    if missing:
        raise KeyError(f"Training plan is missing {missing}")

    checkpoint_map = {
        str(row["chart_id"]): str(Path(str(row["output_dir"])) / "model_state.pt")
        for _, row in training.iterrows()
    }

    selected["checkpoint"] = selected["chart_id"].astype(str).map(checkpoint_map)
    if selected["checkpoint"].isna().any():
        missing_charts = sorted(
            selected.loc[selected["checkpoint"].isna(), "chart_id"]
            .astype(str)
            .unique()
        )
        raise KeyError(f"No checkpoint found for charts {missing_charts}")

    selected_rows = []
    task_rows = []

    for _, row in selected.iterrows():
        mach = float(row["Mach"])
        eta = float(row["eta"])
        chart_id = str(row["chart_id"])
        regime, mapping_scale, xi_max = choose_mapping(chart_id, mach, eta)
        n_values = [301, 401, 501, 601] if eta >= 0.92 else [201, 301, 401, 501]

        selected_payload = row.to_dict()
        selected_payload.update(
            {
                "convergence_regime": regime,
                "mapping_kind": "pin",
                "mapping_scale": mapping_scale,
                "xi_max": xi_max,
                "N_values": " ".join(str(value) for value in n_values),
            }
        )
        selected_rows.append(selected_payload)

        for n_points in n_values:
            task_id = len(task_rows)
            task_rows.append(
                {
                    "task_id": task_id,
                    "convergence_point_id": row["convergence_point_id"],
                    "selection_stratum": row["selection_stratum"],
                    "chart_id": chart_id,
                    "checkpoint": row["checkpoint"],
                    "Mach": mach,
                    "eta": eta,
                    "alpha": float(row["alpha"]),
                    "N": int(n_points),
                    "mapping_kind": "pin",
                    "mapping_scale": mapping_scale,
                    "xi_max": xi_max,
                    "convergence_regime": regime,
                }
            )

    selected_frame = pd.DataFrame(selected_rows)
    task_frame = pd.DataFrame(task_rows)

    selected_path = output_dir / "validation_mode_points_20.csv"
    task_path = output_dir / "GEP_N_convergence_plan.csv"
    selected_frame.to_csv(selected_path, index=False)
    task_frame.to_csv(task_path, index=False)

    report = {
        "n_points": int(len(selected_frame)),
        "n_tasks": int(len(task_frame)),
        "selected_points_csv": str(selected_path),
        "task_plan_csv": str(task_path),
        "N_counts": {
            str(key): int(value)
            for key, value in task_frame["N"].value_counts().sort_index().items()
        },
    }
    write_json(output_dir / "plan_summary.json", report)
    print(selected_frame[
        [
            "convergence_point_id", "selection_stratum", "Mach", "eta",
            "alpha", "chart_id", "convergence_regime", "N_values",
        ]
    ].to_string(index=False))
    print(json.dumps(report, indent=2, sort_keys=True))



def direct_fields_from_pq(
    y: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    alpha: float,
    mach: float,
    ci: float,
) -> dict[str, np.ndarray]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=np.complex128)
    q = np.asarray(q, dtype=np.complex128)
    ubar = np.tanh(y)
    ubar_y = 1.0 - ubar**2
    c = 1j * float(ci)
    denominator = ubar - c
    rho = float(mach) ** 2 * p
    v = -q / (1j * float(alpha) * denominator)
    u = -(ubar_y * v + 1j * float(alpha) * p) / (
        1j * float(alpha) * denominator
    )
    return {"p": p, "rho": rho, "u": u, "v": v}


def select_central_mode(
    solver: NotebookStyleDenseGEPSolver,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    p_pinn: np.ndarray,
    q_pinn: np.ndarray,
    match_mask: np.ndarray,
    *,
    cr_max: float,
    ci_max: float,
    p_weight: float,
) -> tuple[int, dict[str, float], str]:
    values = np.asarray(eigenvalues, dtype=np.complex128)
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    central = np.where(
        finite
        & (values.imag > 0.0)
        & (values.imag <= float(ci_max))
        & (np.abs(values.real) <= float(cr_max))
    )[0]

    source = "central_max_ci"
    if len(central) == 0:
        central = np.where(
            finite
            & (values.imag > 0.0)
            & (values.imag <= float(ci_max))
            & (np.abs(values.real) <= 1.05)
        )[0]
        source = "fallback_best_pinn_overlap"

    if len(central) == 0:
        raise RuntimeError("No unstable finite GEP candidate was found.")

    candidate_rows = []
    for raw_index in central:
        p_overlap, q_overlap, combined = mode_overlap_with_pinn(
            solver=solver,
            vector=eigenvectors[:, raw_index],
            p_pinn=p_pinn,
            q_pinn=q_pinn,
            match_mask=match_mask,
            p_weight=p_weight,
        )
        candidate_rows.append(
            {
                "raw_index": int(raw_index),
                "cr": float(values[raw_index].real),
                "ci": float(values[raw_index].imag),
                "omega_i": float(solver.alpha * values[raw_index].imag),
                "p_overlap_pinn": float(p_overlap),
                "q_overlap_pinn": float(q_overlap),
                "combined_overlap_pinn": float(combined),
            }
        )

    if source == "central_max_ci":
        selected = max(candidate_rows, key=lambda item: item["ci"])
    else:
        selected = max(
            candidate_rows,
            key=lambda item: (item["combined_overlap_pinn"], item["ci"]),
        )
    return int(selected["raw_index"]), selected, source


def save_profile(
    path: Path,
    frame: pd.DataFrame,
    direct_fields: dict[str, np.ndarray],
) -> None:
    payload: dict[str, np.ndarray] = {"y": frame["y"].to_numpy(dtype=float)}
    for field in ("p", "rho", "u", "v"):
        payload[f"{field}_gep"] = (
            frame[f"{field}_gep_real"].to_numpy(dtype=float)
            + 1j * frame[f"{field}_gep_imag"].to_numpy(dtype=float)
        )
        payload[f"{field}_classic"] = (
            frame[f"{field}_classic_real"].to_numpy(dtype=float)
            + 1j * frame[f"{field}_classic_imag"].to_numpy(dtype=float)
        )
        payload[f"{field}_direct"] = np.asarray(
            direct_fields[field],
            dtype=np.complex128,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def command_run(args: argparse.Namespace) -> None:
    plan = pd.read_csv(args.plan)
    task = plan.loc[plan["task_id"].astype(int).eq(int(args.task_index))]
    if len(task) != 1:
        raise RuntimeError(
            f"Expected exactly one row for task {args.task_index}, found {len(task)}"
        )
    row = task.iloc[0]

    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "shards"
    profile_dir = output_dir / "profiles"
    shard_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    result = row.to_dict()
    result.update(
        {
            "technical_success": False,
            "reference_success": False,
            "selection_source": "",
            "error": "",
        }
    )

    try:
        device = torch.device(args.device)
        checkpoint = Path(str(row["checkpoint"]))
        field, ci_net, module, _, family = evaluate_pinn(
            checkpoint_path=checkpoint,
            device=device,
        )

        solver = NotebookStyleDenseGEPSolver(
            alpha=float(row["alpha"]),
            Mach=float(row["Mach"]),
            n_points=int(row["N"]),
            mapping_kind=str(row["mapping_kind"]),
            mapping_scale=float(row["mapping_scale"]),
            xi_max=float(row["xi_max"]),
        )

        p_pinn, q_pinn, ci_pinn = call_pinn_profiles(
            field=field,
            ci_net=ci_net,
            module=module,
            family=family,
            y=solver.y,
            alpha=float(row["alpha"]),
            mach=float(row["Mach"]),
            device=device,
        )
        match_mask = make_match_mask(
            solver.y,
            p_pinn,
            y_match_max=args.y_match_max,
            amplitude_floor_fraction=args.amplitude_floor_fraction,
        )

        eigenvalues, eigenvectors = solver.solve_all()
        raw_index, selected, source = select_central_mode(
            solver,
            eigenvalues,
            eigenvectors,
            p_pinn,
            q_pinn,
            match_mask,
            cr_max=args.cr_max,
            ci_max=args.ci_max,
            p_weight=args.p_overlap_weight,
        )

        classic_fields, ci_classic = load_classic_full_mode(
            float(row["alpha"]),
            float(row["Mach"]),
        )
        metrics, profile = compare_mode_to_classic(
            solver=solver,
            vector=eigenvectors[:, raw_index],
            classic_fields=classic_fields,
            y_match_max=args.y_match_max,
        )

        y_reference = profile["y"].to_numpy(dtype=float)
        p_direct = interp_complex(
            solver.y,
            np.asarray(p_pinn, dtype=np.complex128),
            y_reference,
        )
        q_direct = interp_complex(
            solver.y,
            np.asarray(q_pinn, dtype=np.complex128),
            y_reference,
        )
        direct_fields = direct_fields_from_pq(
            y_reference,
            p_direct,
            q_direct,
            float(row["alpha"]),
            float(row["Mach"]),
            float(ci_pinn),
        )
        classic_p = (
            profile["p_classic_real"].to_numpy(dtype=float)
            + 1j * profile["p_classic_imag"].to_numpy(dtype=float)
        )
        direct_mask = np.abs(y_reference) <= float(args.y_match_max)
        if int(np.count_nonzero(direct_mask)) < 20:
            direct_mask = np.ones_like(y_reference, dtype=bool)
        direct_scale = phase_alignment(
            direct_fields["p"],
            classic_p,
            y_reference,
            direct_mask,
        )
        for field_name in direct_fields:
            direct_fields[field_name] = direct_scale * direct_fields[field_name]

        profile_path = (
            profile_dir
            / f"{row['convergence_point_id']}__N{int(row['N'])}.npz"
        )
        save_profile(profile_path, profile, direct_fields)

        result.update(
            {
                "technical_success": True,
                "reference_success": True,
                "field_family": family,
                "ci_pinn": float(ci_pinn),
                "ci_classic": float(ci_classic),
                "selected_raw_index": int(raw_index),
                "selected_cr": float(selected["cr"]),
                "selected_ci": float(selected["ci"]),
                "selected_omega_i": float(selected["omega_i"]),
                "selected_p_overlap_pinn": float(selected["p_overlap_pinn"]),
                "selected_q_overlap_pinn": float(selected["q_overlap_pinn"]),
                "selected_combined_overlap_pinn": float(
                    selected["combined_overlap_pinn"]
                ),
                "selection_source": source,
                "ci_abs_err_classic": abs(
                    float(selected["ci"]) - float(ci_classic)
                ),
                "profile_path": str(profile_path),
                **{key: float(value) for key, value in metrics.items()},
            }
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()

    shard_path = shard_dir / f"shard_{int(args.task_index):05d}.csv"
    pd.DataFrame([result]).to_csv(shard_path, index=False)
    print(pd.DataFrame([result]).to_string(index=False))

    if not bool(result["technical_success"]):
        raise SystemExit(2)


def interp_complex(y_source: np.ndarray, values: np.ndarray, y_target: np.ndarray) -> np.ndarray:
    return (
        np.interp(y_target, y_source, values.real)
        + 1j * np.interp(y_target, y_source, values.imag)
    )


def load_profile(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def finite_stats(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }



def plot_one_mode_page(
    row: pd.Series,
    *,
    profile: dict[str, np.ndarray],
) -> plt.Figure:
    y = profile["y"].astype(float)
    mask = np.abs(y) <= 12.0
    if int(np.count_nonzero(mask)) < 20:
        mask = np.ones_like(y, dtype=bool)
    y_plot = y[mask]

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(11.0, 12.0),
        sharex=True,
    )
    for field_index, field in enumerate(("p", "rho", "u", "v")):
        classic = profile[f"{field}_classic"][mask]
        direct = profile[f"{field}_direct"][mask]
        gep = profile[f"{field}_gep"][mask]

        for component_index, (operator, component) in enumerate(
            ((np.real, "Re"), (np.imag, "Im"))
        ):
            axis = axes[field_index, component_index]
            axis.plot(
                y_plot,
                operator(classic),
                linewidth=1.8,
                label="Classical",
            )
            axis.plot(
                y_plot,
                operator(direct),
                linewidth=1.3,
                linestyle="--",
                label="Direct PINN",
            )
            axis.plot(
                y_plot,
                operator(gep),
                linewidth=1.3,
                linestyle="-.",
                label="PINN + GEP",
            )
            axis.set_title(fr"{component} $({field})$")
            axis.grid(alpha=0.22)
            if component_index == 0:
                axis.set_ylabel("Amplitude")

    axes[-1, 0].set_xlabel(r"$y$")
    axes[-1, 1].set_xlabel(r"$y$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        fr"{row['selection_stratum']} — "
        fr"$M={float(row['Mach']):.5f}$, "
        fr"$\eta={float(row['eta']):.5f}$, "
        fr"$\alpha={float(row['alpha']):.6f}$, "
        fr"$N={int(row['N'])}$",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def build_mode_pdf(
    finest_rows: pd.DataFrame,
    output_dir: Path,
) -> None:
    ordered = finest_rows.sort_values(
        ["selection_stratum", "Mach", "eta"]
    ).reset_index(drop=True)
    pdf_path = (
        output_dir
        / "supp_modes_classical_vs_direct_PINN_vs_PINN_GEP_20_points.pdf"
    )

    with PdfPages(pdf_path) as pdf:
        for _, row in ordered.iterrows():
            profile = load_profile(row["profile_path"])
            fig = plot_one_mode_page(row, profile=profile)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    representative_rows = ordered.loc[
        ordered["selection_stratum"].astype(str).eq("branch_correction_case")
    ]
    representative = (
        representative_rows.iloc[0]
        if not representative_rows.empty
        else ordered.iloc[0]
    )
    representative_profile = load_profile(representative["profile_path"])
    figure = plot_one_mode_page(
        representative,
        profile=representative_profile,
    )
    figure.savefig(
        output_dir / "Fig_representative_mode.pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "Fig_representative_mode.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def command_merge(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    shard_paths = sorted((output_dir / "shards").glob("shard_*.csv"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found under {output_dir / 'shards'}")

    raw = pd.concat(
        [pd.read_csv(path) for path in shard_paths],
        ignore_index=True,
        sort=False,
    )
    raw.to_csv(output_dir / "GEP_N_convergence_raw.csv", index=False)

    success = as_bool(raw["technical_success"]) & as_bool(raw["reference_success"])
    failed = raw.loc[~success].copy()
    failed.to_csv(output_dir / "GEP_N_convergence_failures.csv", index=False)
    valid = raw.loc[success].copy()

    rows = []
    for point_id, group in valid.groupby("convergence_point_id", sort=True):
        group = group.sort_values("N")
        finest = group.iloc[-1]
        finest_profile = load_profile(finest["profile_path"])
        y_reference = finest_profile["y"].astype(float)

        for _, current in group.iterrows():
            profile = load_profile(current["profile_path"])
            y_current = profile["y"].astype(float)
            mask = np.abs(y_reference) <= float(args.y_match_max)
            if int(mask.sum()) < 20:
                mask = np.ones_like(y_reference, dtype=bool)

            aligned_fields: dict[str, np.ndarray] = {}
            for field in ("p", "rho", "u", "v"):
                aligned_fields[field] = interp_complex(
                    y_current,
                    profile[f"{field}_gep"],
                    y_reference,
                )

            scale = phase_alignment(
                aligned_fields["p"],
                finest_profile["p_gep"],
                y_reference,
                mask,
            )
            for field in aligned_fields:
                aligned_fields[field] = scale * aligned_fields[field]

            row = current.to_dict()
            row.update(
                {
                    "N_finest": int(finest["N"]),
                    "ci_finest": float(finest["selected_ci"]),
                    "omega_finest": float(finest["selected_omega_i"]),
                    "ci_delta_to_finest": abs(
                        float(current["selected_ci"])
                        - float(finest["selected_ci"])
                    ),
                    "omega_delta_to_finest": abs(
                        float(current["selected_omega_i"])
                        - float(finest["selected_omega_i"])
                    ),
                }
            )
            for field in ("p", "rho", "u", "v"):
                row[f"{field}_rel_to_finest"] = rel_l2(
                    aligned_fields[field],
                    finest_profile[f"{field}_gep"],
                    y_reference,
                    mask,
                )
                row[f"{field}_overlap_to_finest"] = overlap_complex(
                    aligned_fields[field],
                    finest_profile[f"{field}_gep"],
                    y_reference,
                    mask,
                )
            row["modal_rel_max_to_finest"] = max(
                row[f"{field}_rel_to_finest"]
                for field in ("p", "rho", "u", "v")
            )
            rows.append(row)

    convergence = pd.DataFrame(rows)
    convergence.to_csv(output_dir / "GEP_N_convergence.csv", index=False)

    finest_rows = (
        convergence.sort_values(["convergence_point_id", "N"])
        .groupby("convergence_point_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    finest_rows.to_csv(
        output_dir / "validation_mode_points_20_finest.csv",
        index=False,
    )
    build_mode_pdf(finest_rows, output_dir)

    summary_rows = []
    for (regime, n_points), group in convergence.groupby(
        ["convergence_regime", "N"], sort=True
    ):
        summary_rows.append(
            {
                "convergence_regime": regime,
                "N": int(n_points),
                "n_points": int(len(group)),
                "ci_delta_median": finite_stats(group["ci_delta_to_finest"]).get(
                    "median"
                ),
                "ci_delta_max": finite_stats(group["ci_delta_to_finest"]).get("max"),
                "omega_delta_median": finite_stats(
                    group["omega_delta_to_finest"]
                ).get("median"),
                "omega_delta_max": finite_stats(
                    group["omega_delta_to_finest"]
                ).get("max"),
                "p_overlap_min": float(
                    pd.to_numeric(
                        group["p_overlap_to_finest"], errors="coerce"
                    ).min()
                ),
                "modal_rel_max": float(
                    pd.to_numeric(
                        group["modal_rel_max_to_finest"], errors="coerce"
                    ).max()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "GEP_N_convergence_summary.csv", index=False)

    metrics = {
        "n_raw_rows": int(len(raw)),
        "n_success_rows": int(len(valid)),
        "n_failed_rows": int(len(failed)),
        "n_points": int(convergence["convergence_point_id"].nunique()),
        "metrics": {
            column: finite_stats(convergence[column])
            for column in (
                "ci_delta_to_finest",
                "omega_delta_to_finest",
                "p_rel_to_finest",
                "rho_rel_to_finest",
                "u_rel_to_finest",
                "v_rel_to_finest",
                "p_overlap_to_finest",
                "modal_rel_max_to_finest",
            )
        },
    }
    write_json(output_dir / "GEP_N_convergence_metrics.json", json_safe(metrics))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    plot_specs = [
        ("ci_delta_to_finest", r"$|c_i^{(N)}-c_i^{(N_{\max})}|$", True),
        (
            "omega_delta_to_finest",
            r"$|\omega_i^{(N)}-\omega_i^{(N_{\max})}|$",
            True,
        ),
        ("p_overlap_to_finest", r"$\mathcal{O}_p(N,N_{\max})$", False),
        (
            "modal_rel_max_to_finest",
            r"$\max_{p,\rho,u,v}\varepsilon_f(N,N_{\max})$",
            True,
        ),
    ]

    for axis, (column, ylabel, use_log) in zip(axes.ravel(), plot_specs):
        grouped = convergence.groupby("N")[column]
        n_values = sorted(convergence["N"].astype(int).unique())
        median = [
            float(pd.to_numeric(grouped.get_group(n), errors="coerce").median())
            for n in n_values
        ]
        maximum = [
            float(pd.to_numeric(grouped.get_group(n), errors="coerce").max())
            for n in n_values
        ]
        axis.plot(n_values, median, marker="o", label="median")
        axis.plot(n_values, maximum, marker="s", label="maximum")
        if use_log:
            axis.set_yscale("log")
        axis.set_xlabel("GEP grid size N")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False)

    fig.suptitle("GEP resolution convergence on 20 representative points")
    fig.tight_layout()
    fig.savefig(
        output_dir / "SuppFig07_GEP_N_convergence.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "SuppFig07_GEP_N_convergence.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(summary.to_string(index=False))
    print(json.dumps(json_safe(metrics), indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-plan")
    build.add_argument("--central-audit", default=DEFAULT_CENTRAL_AUDIT)
    build.add_argument("--training-plan", default=DEFAULT_TRAINING_PLAN)
    build.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    build.set_defaults(function=command_build_plan)

    run = sub.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--task-index", type=int, required=True)
    run.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    run.add_argument("--device", default="cpu")
    run.add_argument("--y-match-max", type=float, default=12.0)
    run.add_argument("--amplitude-floor-fraction", type=float, default=0.02)
    run.add_argument("--cr-max", type=float, default=0.05)
    run.add_argument("--ci-max", type=float, default=2.0)
    run.add_argument("--p-overlap-weight", type=float, default=0.75)
    run.set_defaults(function=command_run)

    merge = sub.add_parser("merge")
    merge.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    merge.add_argument("--y-match-max", type=float, default=12.0)
    merge.set_defaults(function=command_merge)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
