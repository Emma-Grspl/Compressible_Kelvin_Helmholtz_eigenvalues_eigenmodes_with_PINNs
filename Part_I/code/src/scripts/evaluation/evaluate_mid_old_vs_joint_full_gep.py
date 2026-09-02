#!/usr/bin/env python3
"""
Compare the historical MID PINN/IDW pipeline with the jointly trained MID PINN.

The comparison is deliberately fair:

- the same 18 off-anchor (Mach, eta) points are used;
- the same dense GEP matrices are diagonalized only once per point;
- the complete GEP spectrum is shared by both comparisons;
- the old PINN uses its historical fixed CiGridIDW provider;
- the new PINN uses its trained CiAtlasNet provider;
- for each PINN, c_i is the primary branch-identification signal and
  the p/q profiles are secondary;
- the classical solution is used only after branch selection, for validation.

The script also compares the already generated direct-PINN diagnostics CSVs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from src.scripts.gep.selection.solve_dense_gep_notebook_style import (
    NotebookStyleDenseGEPSolver,
)
from src.scripts.evaluation.evaluate_kh_subsonic_fixed_mach_modal_candidates import (
    load_classic_full_mode,
)
from src.scripts.gep.selection.audit_mid_joint_pinn_full_gep import (
    alpha_from_eta,
    call_pinn_profiles,
    compare_mode_to_classic,
    evaluate_pinn,
    make_match_mask,
    point_tag,
    save_spectrum_plot,
    select_modes,
)
from src.scripts.training.atlas.direct_pinn.train_subsonic_joint_spectral_modal_chart import (
    MODULES,
    call_supported,
    infer_field_family,
)


def parse_float_list(value: str) -> list[float]:
    return [
        float(item)
        for item in str(value).replace(",", " ").split()
        if item.strip()
    ]


def instantiate_field(
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, dict[str, Any], str]:
    args = dict(checkpoint.get("args", {}))
    family = args.get("field_family_resolved")
    if not family:
        family = infer_field_family(
            "auto",
            checkpoint_path,
            checkpoint,
        )

    if family not in MODULES:
        raise RuntimeError(
            f"Unsupported field family {family!r}; "
            f"known families: {sorted(MODULES)}"
        )

    module = importlib.import_module(MODULES[family])

    required = ("mach_min", "mach_max", "eta_min", "eta_max")
    missing = [name for name in required if args.get(name) is None]
    if missing:
        raise RuntimeError(
            f"{checkpoint_path} is missing chart bounds: {missing}"
        )

    mach_min = float(args["mach_min"])
    mach_max = float(args["mach_max"])
    eta_min = float(args["eta_min"])
    eta_max = float(args["eta_max"])

    alpha_corners = [
        eta * math.sqrt(max(1.0 - mach**2, 1.0e-14))
        for eta in (eta_min, eta_max)
        for mach in (mach_min, mach_max)
    ]

    field = call_supported(
        module.FieldPQNet,
        ymax=float(args.get("ymax", 100.0)),
        alpha_min=min(alpha_corners),
        alpha_max=max(alpha_corners),
        mach_min=mach_min,
        mach_max=mach_max,
        eta_min=eta_min,
        eta_max=eta_max,
        width=int(args.get("width", 256)),
        depth=int(args.get("depth", 7)),
        n_freq=int(args.get("n_freq", 12)),
    ).to(device=device, dtype=torch.float64)

    load_result = field.load_state_dict(
        checkpoint["field_state_dict"],
        strict=True,
    )
    field.eval()

    print("Loaded field:", checkpoint_path)
    print("  family:", family)
    print("  result:", load_result)

    return field, module, args, family


def load_old_pinn(
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    Any,
    dict[str, Any],
    str,
    pd.DataFrame,
]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    field, module, args, family = instantiate_field(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    anchor_df = pd.DataFrame(checkpoint.get("anchor_df", {}))
    if anchor_df.empty:
        raise RuntimeError(
            f"Old checkpoint {checkpoint_path} has no anchor_df."
        )

    ci_provider = call_supported(
        module.CiGridIDW,
        anchor_df=anchor_df,
        eta_scale=float(args.get("ci_idw_eta_scale", 0.25)),
        mach_scale=float(args.get("ci_idw_mach_scale", 0.25)),
        power=float(args.get("ci_idw_power", 4.0)),
        eps=float(args.get("ci_idw_eps", 1.0e-12)),
    ).to(device=device, dtype=torch.float64)
    ci_provider.eval()

    return (
        field,
        ci_provider,
        module,
        args,
        family,
        anchor_df,
    )


def load_new_pinn(
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    Any,
    dict[str, Any],
    str,
    pd.DataFrame,
]:
    field, ci_net, module, args, family = evaluate_pinn(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    anchor_df = pd.DataFrame(checkpoint.get("anchor_df", {}))
    if anchor_df.empty:
        raise RuntimeError(
            f"New checkpoint {checkpoint_path} has no anchor_df."
        )

    counts = anchor_df.groupby("Mach").size()
    if not bool((counts == 4).all()):
        raise RuntimeError(
            "The new checkpoint does not have exactly four c_i anchors "
            f"per Mach: {counts.to_dict()}"
        )

    return field, ci_net, module, args, family, anchor_df


def compare_direct_diagnostics(
    *,
    old_csv: Path,
    new_csv: Path,
    output_dir: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "old_csv": str(old_csv),
        "new_csv": str(new_csv),
    }

    if not old_csv.is_file() or not new_csv.is_file():
        print(
            "[WARN] Direct-PINN CSV comparison skipped. "
            f"old_exists={old_csv.is_file()} "
            f"new_exists={new_csv.is_file()}"
        )
        return result

    old = pd.read_csv(old_csv)
    new = pd.read_csv(new_csv)

    keys = ["Mach", "eta"]
    metrics = [
        "ci_abs_err",
        "ci_rel_err",
        "p_rel",
        "q_rel",
        "rho_rel",
        "u_rel",
        "v_rel",
        "gamma_rel",
    ]

    merged = old.merge(
        new,
        on=keys,
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )

    pointwise_columns: dict[str, Any] = {
        "Mach": merged["Mach"],
        "eta": merged["eta"],
    }
    summary_rows: list[dict[str, Any]] = []

    for metric in metrics:
        old_column = f"{metric}_old"
        new_column = f"{metric}_new"
        if old_column not in merged or new_column not in merged:
            continue

        old_values = pd.to_numeric(
            merged[old_column],
            errors="coerce",
        )
        new_values = pd.to_numeric(
            merged[new_column],
            errors="coerce",
        )

        pointwise_columns[old_column] = old_values
        pointwise_columns[new_column] = new_values
        pointwise_columns[f"{metric}_new_minus_old"] = (
            new_values - old_values
        )
        pointwise_columns[f"{metric}_new_over_old"] = (
            new_values / old_values.replace(0.0, np.nan)
        )
        pointwise_columns[f"{metric}_new_better"] = (
            new_values < old_values
        )

        valid = np.isfinite(old_values) & np.isfinite(new_values)
        old_valid = old_values[valid]
        new_valid = new_values[valid]

        if old_valid.empty:
            continue

        old_mean = float(old_valid.mean())
        new_mean = float(new_valid.mean())
        reduction = (
            100.0 * (old_mean - new_mean) / old_mean
            if old_mean != 0.0
            else float("nan")
        )

        summary_rows.append(
            {
                "metric": metric,
                "n_points": int(valid.sum()),
                "old_mean": old_mean,
                "new_mean": new_mean,
                "mean_reduction_percent": reduction,
                "old_max": float(old_valid.max()),
                "new_max": float(new_valid.max()),
                "n_new_better": int(
                    (new_valid < old_valid).sum()
                ),
            }
        )

    pointwise = pd.DataFrame(pointwise_columns)
    summary = pd.DataFrame(summary_rows)

    pointwise.to_csv(
        output_dir / "direct_pinn_comparison_pointwise.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "direct_pinn_comparison_metrics.csv",
        index=False,
    )

    print("=" * 100)
    print("DIRECT PINN COMPARISON")
    print("=" * 100)
    print(summary.to_string(index=False))

    result.update(
        {
            "available": True,
            "n_points": int(len(pointwise)),
            "metrics": summary.to_dict(orient="records"),
        }
    )
    return result


def summarize_gep_comparison(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    metric_pairs = [
        (
            "ci_seed_rel_err_classic",
            "old_ci_seed_rel_err_classic",
            "new_ci_seed_rel_err_classic",
        ),
        (
            "matched_ci_rel_err_classic",
            "old_matched_ci_rel_err_classic",
            "new_matched_ci_rel_err_classic",
        ),
        (
            "matched_p_overlap_pinn",
            "old_matched_p_overlap_pinn",
            "new_matched_p_overlap_pinn",
        ),
        (
            "matched_q_overlap_pinn",
            "old_matched_q_overlap_pinn",
            "new_matched_q_overlap_pinn",
        ),
        (
            "matched_p_rel_classic",
            "old_matched_p_rel_classic",
            "new_matched_p_rel_classic",
        ),
        (
            "matched_u_rel_classic",
            "old_matched_u_rel_classic",
            "new_matched_u_rel_classic",
        ),
        (
            "matched_v_rel_classic",
            "old_matched_v_rel_classic",
            "new_matched_v_rel_classic",
        ),
    ]

    rows: list[dict[str, Any]] = []

    for metric, old_column, new_column in metric_pairs:
        if old_column not in comparison or new_column not in comparison:
            continue

        old_values = pd.to_numeric(
            comparison[old_column],
            errors="coerce",
        )
        new_values = pd.to_numeric(
            comparison[new_column],
            errors="coerce",
        )
        valid = np.isfinite(old_values) & np.isfinite(new_values)
        old_valid = old_values[valid]
        new_valid = new_values[valid]

        if old_valid.empty:
            continue

        error_like = (
            "overlap" not in metric
        )

        if error_like:
            n_new_better = int((new_valid < old_valid).sum())
            old_mean = float(old_valid.mean())
            new_mean = float(new_valid.mean())
            reduction = (
                100.0 * (old_mean - new_mean) / old_mean
                if old_mean != 0.0
                else float("nan")
            )
        else:
            n_new_better = int((new_valid > old_valid).sum())
            old_mean = float(old_valid.mean())
            new_mean = float(new_valid.mean())
            reduction = float("nan")

        rows.append(
            {
                "metric": metric,
                "n_points": int(valid.sum()),
                "old_mean": old_mean,
                "new_mean": new_mean,
                "mean_reduction_percent": reduction,
                "old_max": float(old_valid.max()),
                "new_max": float(new_valid.max()),
                "n_new_better": n_new_better,
                "better_direction": (
                    "lower" if error_like else "higher"
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--old-checkpoint",
        default="models_saved/archive/pinn_subsonic/models/MID/model_state.pt",
    )
    parser.add_argument(
        "--new-checkpoint",
        default=(
            "models_saved/production/atlas/N340/MID/model_state.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "assets/pinn_subsonic/"
            "mid_old_vs_joint_full_gep_comparison"
        ),
    )

    parser.add_argument(
        "--old-direct-csv",
        default=(
            "assets/pinn_subsonic/"
            "joint_ci_mode_baseline/MID/diagnostics_summary.csv"
        ),
    )
    parser.add_argument(
        "--new-direct-csv",
        default=(
            "archive/csv/assets/pinn_subsonic/joint_ci_mode_atlas_v2/"
            "MID/diagnostics_summary.csv"
        ),
    )

    parser.add_argument(
        "--mach-values",
        default="0.15 0.25 0.35 0.45 0.55 0.65",
    )
    parser.add_argument(
        "--eta-values",
        default="0.375 0.45 0.525",
    )

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--N", type=int, default=301)
    parser.add_argument("--mapping-kind", default="pin")
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--xi-max", type=float, default=0.98)

    parser.add_argument("--y-match-max", type=float, default=12.0)
    parser.add_argument(
        "--amplitude-floor-fraction",
        type=float,
        default=0.02,
    )
    parser.add_argument("--ci-window-rel", type=float, default=0.02)
    parser.add_argument("--ci-window-factor", type=float, default=3.0)
    parser.add_argument("--shortlist-max", type=int, default=8)
    parser.add_argument("--p-overlap-weight", type=float, default=0.75)
    parser.add_argument("--cr-physical-max", type=float, default=1.05)
    parser.add_argument("--ci-physical-max", type=float, default=2.0)

    args = parser.parse_args()

    old_checkpoint = Path(args.old_checkpoint)
    new_checkpoint = Path(args.new_checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not old_checkpoint.is_file():
        raise FileNotFoundError(old_checkpoint)
    if not new_checkpoint.is_file():
        raise FileNotFoundError(new_checkpoint)

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    (
        old_field,
        old_ci_provider,
        old_module,
        old_args,
        old_family,
        old_anchors,
    ) = load_old_pinn(
        checkpoint_path=old_checkpoint,
        device=device,
    )

    (
        new_field,
        new_ci_provider,
        new_module,
        new_args,
        new_family,
        new_anchors,
    ) = load_new_pinn(
        checkpoint_path=new_checkpoint,
        device=device,
    )

    print("=" * 100)
    print("ANCHOR AUDIT")
    print("=" * 100)
    print("Old anchor counts per Mach:")
    print(old_anchors.groupby("Mach").size().to_string())
    print("Old total scalar c_i anchors:", len(old_anchors))
    print()
    print("New anchor counts per Mach:")
    print(new_anchors.groupby("Mach").size().to_string())
    print("New total scalar c_i anchors:", len(new_anchors))

    direct_report = compare_direct_diagnostics(
        old_csv=Path(args.old_direct_csv),
        new_csv=Path(args.new_direct_csv),
        output_dir=output_dir,
    )

    mach_values = parse_float_list(args.mach_values)
    eta_values = parse_float_list(args.eta_values)

    comparison_rows: list[dict[str, Any]] = []

    for mach in mach_values:
        for eta in eta_values:
            alpha = alpha_from_eta(eta, mach)
            tag = point_tag(mach, eta, alpha)
            point_dir = output_dir / tag
            point_dir.mkdir(parents=True, exist_ok=True)

            print("=" * 100)
            print(
                f"{tag}: M={mach:.6f}, eta={eta:.6f}, "
                f"alpha={alpha:.9f}"
            )

            solver = NotebookStyleDenseGEPSolver(
                alpha=alpha,
                Mach=mach,
                n_points=args.N,
                mapping_kind=args.mapping_kind,
                mapping_scale=args.mapping_scale,
                xi_max=args.xi_max,
            )

            # Fair comparison: one and only one full dense diagonalization.
            eigenvalues, eigenvectors = solver.solve_all()

            classic_fields, ci_classic = load_classic_full_mode(
                alpha,
                mach,
            )

            model_results: dict[str, dict[str, Any]] = {}

            model_specs = {
                "old": (
                    old_field,
                    old_ci_provider,
                    old_module,
                    old_family,
                ),
                "new": (
                    new_field,
                    new_ci_provider,
                    new_module,
                    new_family,
                ),
            }

            for label, (
                field,
                ci_provider,
                module,
                family,
            ) in model_specs.items():
                p_pinn, q_pinn, ci_pinn = call_pinn_profiles(
                    field=field,
                    ci_net=ci_provider,
                    module=module,
                    family=family,
                    y=solver.y,
                    alpha=alpha,
                    mach=mach,
                    device=device,
                )

                match_mask = make_match_mask(
                    solver.y,
                    p_pinn,
                    y_match_max=args.y_match_max,
                    amplitude_floor_fraction=(
                        args.amplitude_floor_fraction
                    ),
                )

                (
                    spectrum,
                    nearest_ci,
                    pinn_matched,
                    most_unstable,
                ) = select_modes(
                    solver=solver,
                    eigenvalues=eigenvalues,
                    eigenvectors=eigenvectors,
                    ci_pinn=ci_pinn,
                    p_pinn=p_pinn,
                    q_pinn=q_pinn,
                    match_mask=match_mask,
                    p_weight=args.p_overlap_weight,
                    ci_window_rel=args.ci_window_rel,
                    ci_window_factor=args.ci_window_factor,
                    shortlist_max=args.shortlist_max,
                    cr_physical_max=args.cr_physical_max,
                    ci_physical_max=args.ci_physical_max,
                )

                spectrum.to_csv(
                    point_dir / f"full_spectrum_{label}.csv",
                    index=False,
                )

                save_spectrum_plot(
                    spectrum=spectrum,
                    ci_pinn=ci_pinn,
                    output_path=(
                        point_dir / f"full_spectrum_{label}.png"
                    ),
                    title=(
                        f"{label.upper()} PINN: "
                        f"M={mach:.3f}, eta={eta:.3f}"
                    ),
                )

                matched_metrics, matched_profile = (
                    compare_mode_to_classic(
                        solver=solver,
                        vector=eigenvectors[
                            :,
                            int(pinn_matched["raw_index"]),
                        ],
                        classic_fields=classic_fields,
                        y_match_max=args.y_match_max,
                    )
                )
                matched_profile.to_csv(
                    point_dir
                    / f"{label}_pinn_matched_vs_classic.csv",
                    index=False,
                )

                model_results[label] = {
                    "ci_pinn": float(ci_pinn),
                    "ci_seed_abs_err_classic": abs(
                        float(ci_pinn) - float(ci_classic)
                    ),
                    "ci_seed_rel_err_classic": (
                        abs(float(ci_pinn) - float(ci_classic))
                        / max(abs(float(ci_classic)), 1.0e-12)
                    ),
                    "nearest_ci_raw_index": int(
                        nearest_ci["raw_index"]
                    ),
                    "nearest_ci_ci": float(nearest_ci["ci"]),
                    "nearest_ci_rel_distance": float(
                        nearest_ci["ci_rel_distance_to_pinn"]
                    ),
                    "matched_raw_index": int(
                        pinn_matched["raw_index"]
                    ),
                    "matched_cr": float(pinn_matched["cr"]),
                    "matched_ci": float(pinn_matched["ci"]),
                    "matched_ci_rel_distance_to_pinn": float(
                        pinn_matched[
                            "ci_rel_distance_to_pinn"
                        ]
                    ),
                    "matched_p_overlap_pinn": float(
                        pinn_matched["p_overlap_pinn"]
                    ),
                    "matched_q_overlap_pinn": float(
                        pinn_matched["q_overlap_pinn"]
                    ),
                    "matched_combined_overlap_pinn": float(
                        pinn_matched[
                            "combined_overlap_pinn"
                        ]
                    ),
                    "matched_is_most_unstable": bool(
                        int(pinn_matched["raw_index"])
                        == int(most_unstable["raw_index"])
                    ),
                    "matched_ci_abs_err_classic": abs(
                        float(pinn_matched["ci"])
                        - float(ci_classic)
                    ),
                    "matched_ci_rel_err_classic": (
                        abs(
                            float(pinn_matched["ci"])
                            - float(ci_classic)
                        )
                        / max(abs(float(ci_classic)), 1.0e-12)
                    ),
                    "most_unstable_raw_index": int(
                        most_unstable["raw_index"]
                    ),
                    "most_unstable_cr": float(
                        most_unstable["cr"]
                    ),
                    "most_unstable_ci": float(
                        most_unstable["ci"]
                    ),
                    "matched_p_rel_classic": float(
                        matched_metrics["p_rel_classic"]
                    ),
                    "matched_rho_rel_classic": float(
                        matched_metrics["rho_rel_classic"]
                    ),
                    "matched_u_rel_classic": float(
                        matched_metrics["u_rel_classic"]
                    ),
                    "matched_v_rel_classic": float(
                        matched_metrics["v_rel_classic"]
                    ),
                    "matched_p_overlap_classic": float(
                        matched_metrics["p_overlap_classic"]
                    ),
                }

                print(
                    f"{label.upper()}: "
                    f"ci_PINN={ci_pinn:.9e}; "
                    f"matched index={pinn_matched['raw_index']}; "
                    f"most unstable index="
                    f"{most_unstable['raw_index']}; "
                    f"same={model_results[label]['matched_is_most_unstable']}; "
                    f"O_p={pinn_matched['p_overlap_pinn']:.6f}; "
                    f"O_q={pinn_matched['q_overlap_pinn']:.6f}"
                )

            old_result = model_results["old"]
            new_result = model_results["new"]

            row: dict[str, Any] = {
                "Mach": mach,
                "eta": eta,
                "alpha": alpha,
                "ci_classic": float(ci_classic),
                "n_raw_eigenvalues": int(len(eigenvalues)),
                "old_new_same_matched_raw_index": bool(
                    old_result["matched_raw_index"]
                    == new_result["matched_raw_index"]
                ),
                "old_new_both_match_most_unstable": bool(
                    old_result["matched_is_most_unstable"]
                    and new_result["matched_is_most_unstable"]
                ),
                "new_only_recovers_most_unstable": bool(
                    (not old_result["matched_is_most_unstable"])
                    and new_result["matched_is_most_unstable"]
                ),
                "old_only_recovers_most_unstable": bool(
                    old_result["matched_is_most_unstable"]
                    and (not new_result["matched_is_most_unstable"])
                ),
            }

            for label, result in model_results.items():
                for key, value in result.items():
                    row[f"{label}_{key}"] = value

            row["new_ci_seed_error_smaller"] = bool(
                new_result["ci_seed_rel_err_classic"]
                < old_result["ci_seed_rel_err_classic"]
            )
            row["new_p_overlap_higher"] = bool(
                new_result["matched_p_overlap_pinn"]
                > old_result["matched_p_overlap_pinn"]
            )
            row["new_q_overlap_higher"] = bool(
                new_result["matched_q_overlap_pinn"]
                > old_result["matched_q_overlap_pinn"]
            )

            comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["Mach", "eta"]
    ).reset_index(drop=True)
    comparison.to_csv(
        output_dir / "gep_old_vs_new_pointwise.csv",
        index=False,
    )

    gep_metrics = summarize_gep_comparison(comparison)
    gep_metrics.to_csv(
        output_dir / "gep_old_vs_new_metrics.csv",
        index=False,
    )

    report = {
        "old_checkpoint": str(old_checkpoint),
        "new_checkpoint": str(new_checkpoint),
        "n_points": int(len(comparison)),
        "new_supervision": {
            "type": "scalar_c_i_only",
            "anchors_per_mach": 4,
            "n_mach_values": int(new_anchors["Mach"].nunique()),
            "total_scalar_anchors": int(len(new_anchors)),
            "classical_modal_supervision": False,
        },
        "old_anchor_counts_per_mach": {
            str(key): int(value)
            for key, value
            in old_anchors.groupby("Mach").size().items()
        },
        "new_anchor_counts_per_mach": {
            str(key): int(value)
            for key, value
            in new_anchors.groupby("Mach").size().items()
        },
        "n_old_matches_most_unstable": int(
            comparison["old_matched_is_most_unstable"].sum()
        ),
        "n_new_matches_most_unstable": int(
            comparison["new_matched_is_most_unstable"].sum()
        ),
        "n_same_selected_mode": int(
            comparison["old_new_same_matched_raw_index"].sum()
        ),
        "n_new_only_recovers_most_unstable": int(
            comparison["new_only_recovers_most_unstable"].sum()
        ),
        "n_new_ci_seed_error_smaller": int(
            comparison["new_ci_seed_error_smaller"].sum()
        ),
        "direct_pinn_comparison": direct_report,
        "gep_metric_comparison": gep_metrics.to_dict(
            orient="records"
        ),
    }

    (output_dir / "comparison_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=" * 100)
    print("FULL GEP OLD VS NEW — POINTWISE")
    print("=" * 100)
    print(
        comparison[
            [
                "Mach",
                "eta",
                "old_ci_seed_rel_err_classic",
                "new_ci_seed_rel_err_classic",
                "old_matched_p_overlap_pinn",
                "new_matched_p_overlap_pinn",
                "old_matched_q_overlap_pinn",
                "new_matched_q_overlap_pinn",
                "old_matched_is_most_unstable",
                "new_matched_is_most_unstable",
                "old_new_same_matched_raw_index",
                "old_matched_p_rel_classic",
                "new_matched_p_rel_classic",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 100)
    print("FULL GEP OLD VS NEW — METRICS")
    print("=" * 100)
    print(gep_metrics.to_string(index=False))

    print()
    print(json.dumps(report, indent=2, sort_keys=True))
    print()
    print(
        "Wrote:",
        output_dir / "gep_old_vs_new_pointwise.csv",
    )
    print(
        "Wrote:",
        output_dir / "gep_old_vs_new_metrics.csv",
    )
    print(
        "Wrote:",
        output_dir / "comparison_report.json",
    )


if __name__ == "__main__":
    main()
