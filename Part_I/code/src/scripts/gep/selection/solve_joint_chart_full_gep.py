#!/usr/bin/env python3
"""
Run the complete dense GEP on all diagnostic points of one jointly trained
subsonic KH PINN chart.

The script computes every generalized eigenpair. The PINN c_i prediction is
the primary branch-identification signal; PINN p/q overlaps are secondary.
Classical fields are loaded only after selection for validation.

Outputs are aggregated per chart to limit inode usage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from src.scripts.gep.selection.solve_dense_gep_notebook_style import (
    NotebookStyleDenseGEPSolver,
)
from src.scripts.evaluation.evaluate_kh_subsonic_fixed_mach_modal_candidates import (
    load_classic_full_mode,
)
from src.scripts.gep.selection.audit_mid_joint_pinn_full_gep import (
    call_pinn_profiles,
    compare_mode_to_classic,
    evaluate_pinn,
    make_match_mask,
    select_modes,
    split_gep_vector,
)

REGIMES = {
    "standard_N301": {
        "N": 301,
        "mapping_kind": "pin",
        "mapping_scale": 5.0,
        "xi_max": 0.98,
        "continuation_required": False,
    },
    "longwave_map10_N301": {
        "N": 301,
        "mapping_kind": "pin",
        "mapping_scale": 10.0,
        "xi_max": 0.99,
        "continuation_required": False,
    },
    "extreme_longwave_map20_N301": {
        "N": 301,
        "mapping_kind": "pin",
        "mapping_scale": 20.0,
        "xi_max": 0.995,
        "continuation_required": False,
    },
    "near_neutral_N401": {
        "N": 401,
        "mapping_kind": "pin",
        "mapping_scale": 5.0,
        "xi_max": 0.98,
        "continuation_required": True,
    },
}


def choose_regime(chart_id: str, mach: float, eta: float):
    if chart_id == "ETAEDGE_HM2B" or (mach >= 0.88 and eta <= 0.06):
        name = "extreme_longwave_map20_N301"
        reason = "extreme long-wave high-Mach corner"
    elif eta <= 0.12:
        name = "longwave_map10_N301"
        reason = "long-wave eta <= 0.12"
    elif eta >= 0.92:
        name = "near_neutral_N401"
        reason = "near-neutral eta >= 0.92"
    else:
        name = "standard_N301"
        reason = "standard interior regime"
    return name, dict(REGIMES[name]), reason


def scalar(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def make_point_id(index: int, mach: float, eta: float) -> str:
    return (
        f"P{index:03d}_M{int(round(1000 * mach)):04d}_"
        f"eta{int(round(1000 * eta)):04d}"
    )


def split_mode(vector: np.ndarray, solver, mach: float):
    return split_gep_vector(vector, solver.n_points, mach)


def add_array(store, prefix: str, name: str, values):
    store[f"{prefix}__{name}"] = np.asarray(values)


def add_frame(store, prefix: str, frame: pd.DataFrame):
    for column in frame.columns:
        add_array(store, prefix, str(column), frame[column].to_numpy())


def fallback_spectrum(
    eigenvalues,
    chart_id,
    point_id,
    mach,
    eta,
    alpha,
    regime,
):
    values = np.asarray(eigenvalues)
    finite = np.isfinite(np.real(values)) & np.isfinite(np.imag(values))
    return pd.DataFrame(
        {
            "chart_id": chart_id,
            "point_id": point_id,
            "Mach": mach,
            "eta": eta,
            "alpha": alpha,
            "gep_regime": regime,
            "raw_index": np.arange(len(values)),
            "cr": np.where(finite, np.real(values), np.nan),
            "ci": np.where(finite, np.imag(values), np.nan),
            "omega_i": np.where(finite, alpha * np.imag(values), np.nan),
            "finite": finite,
        }
    )


def finite_metric(frame: pd.DataFrame, column: str):
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]
    return values if not values.empty else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chart-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--diagnostics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--y-match-max", type=float, default=12.0)
    parser.add_argument("--amplitude-floor-fraction", type=float, default=0.02)
    parser.add_argument("--ci-window-rel", type=float, default=0.02)
    parser.add_argument("--ci-window-factor", type=float, default=3.0)
    parser.add_argument("--shortlist-max", type=int, default=8)
    parser.add_argument("--p-overlap-weight", type=float, default=0.75)
    parser.add_argument("--cr-physical-max", type=float, default=1.05)
    parser.add_argument("--ci-physical-max", type=float, default=2.0)
    args = parser.parse_args()

    chart_id = str(args.chart_id)
    checkpoint_path = Path(args.checkpoint)
    diagnostics_path = Path(args.diagnostics_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not diagnostics_path.is_file():
        raise FileNotFoundError(diagnostics_path)

    device_name = str(args.device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    field, ci_net, module, _, family = evaluate_pinn(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    diagnostics = pd.read_csv(diagnostics_path)
    required = {"Mach", "eta", "alpha"}
    missing = sorted(required.difference(diagnostics.columns))
    if missing:
        raise KeyError(f"{diagnostics_path} is missing {missing}")

    points = (
        diagnostics.sort_values(["Mach", "eta", "alpha"])
        .drop_duplicates(["Mach", "eta", "alpha"])
        .reset_index(drop=True)
    )

    plan_rows = []
    for index, row in points.iterrows():
        mach = float(row["Mach"])
        eta = float(row["eta"])
        alpha = float(row["alpha"])
        regime_name, regime, reason = choose_regime(chart_id, mach, eta)
        plan_rows.append(
            {
                "chart_id": chart_id,
                "point_id": make_point_id(index, mach, eta),
                "Mach": mach,
                "eta": eta,
                "alpha": alpha,
                "gep_regime": regime_name,
                "regime_reason": reason,
                **regime,
            }
        )

    point_plan = pd.DataFrame(plan_rows)
    point_plan.to_csv(output_dir / "point_plan.csv", index=False)

    print("=" * 100)
    print("JOINT PINN -> COMPLETE DENSE GEP")
    print("=" * 100)
    print("chart_id:", chart_id)
    print("checkpoint:", checkpoint_path)
    print("diagnostics:", diagnostics_path)
    print("output:", output_dir)
    print("family:", family)
    print(point_plan.to_string(index=False))

    summary_rows = []
    spectrum_frames = []
    profile_store = {}

    for _, plan in point_plan.iterrows():
        pid = str(plan["point_id"])
        mach = float(plan["Mach"])
        eta = float(plan["eta"])
        alpha = float(plan["alpha"])
        regime_name = str(plan["gep_regime"])
        n_points = int(plan["N"])
        mapping_kind = str(plan["mapping_kind"])
        mapping_scale = float(plan["mapping_scale"])
        xi_max = float(plan["xi_max"])

        source = points.loc[
            np.isclose(points["Mach"], mach)
            & np.isclose(points["eta"], eta)
            & np.isclose(points["alpha"], alpha)
        ].iloc[0]

        result = {
            "chart_id": chart_id,
            "point_id": pid,
            "Mach": mach,
            "eta": eta,
            "alpha": alpha,
            "field_family": family,
            "gep_regime": regime_name,
            "N": n_points,
            "mapping_kind": mapping_kind,
            "mapping_scale": mapping_scale,
            "xi_max": xi_max,
            "continuation_required": bool(plan["continuation_required"]),
            "ci_ref_diagnostic": scalar(source.get("ci_ref")),
            "ci_pred_diagnostic": scalar(source.get("ci_pred")),
            "technical_success": False,
            "reference_success": False,
            "selection_error": "",
            "reference_error": "",
        }

        print("-" * 100)
        print(
            f"{pid}: M={mach:.7f}, eta={eta:.7f}, alpha={alpha:.9f}, "
            f"regime={regime_name}"
        )

        try:
            solver = NotebookStyleDenseGEPSolver(
                alpha=alpha,
                Mach=mach,
                n_points=n_points,
                mapping_kind=mapping_kind,
                mapping_scale=mapping_scale,
                xi_max=xi_max,
            )
            p_pinn, q_pinn, ci_pinn = call_pinn_profiles(
                field=field,
                ci_net=ci_net,
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
                amplitude_floor_fraction=args.amplitude_floor_fraction,
            )

            eigenvalues, eigenvectors = solver.solve_all()

            try:
                spectrum, nearest_ci, matched, most_unstable = select_modes(
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
            except Exception:
                spectrum_frames.append(
                    fallback_spectrum(
                        eigenvalues,
                        chart_id,
                        pid,
                        mach,
                        eta,
                        alpha,
                        regime_name,
                    )
                )
                raise

            for position, (name, value) in enumerate(
                [
                    ("chart_id", chart_id),
                    ("point_id", pid),
                    ("Mach", mach),
                    ("eta", eta),
                    ("alpha", alpha),
                    ("gep_regime", regime_name),
                ]
            ):
                spectrum.insert(position, name, value)
            spectrum_frames.append(spectrum)

            matched_index = int(matched["raw_index"])
            unstable_index = int(most_unstable["raw_index"])
            same_mode = matched_index == unstable_index

            matched_fields = split_mode(
                eigenvectors[:, matched_index],
                solver,
                mach,
            )
            unstable_fields = split_mode(
                eigenvectors[:, unstable_index],
                solver,
                mach,
            )

            add_array(profile_store, pid, "y_gep", solver.y)
            add_array(profile_store, pid, "p_pinn", p_pinn)
            add_array(profile_store, pid, "q_pinn", q_pinn)
            add_array(profile_store, pid, "match_mask", match_mask.astype(np.int8))
            for name, values in matched_fields.items():
                add_array(profile_store, pid, f"{name}_gep_matched", values)
            for name, values in unstable_fields.items():
                add_array(profile_store, pid, f"{name}_gep_most_unstable", values)

            result.update(
                {
                    "technical_success": True,
                    "ci_pinn": float(ci_pinn),
                    "n_raw_eigenvalues": int(len(eigenvalues)),
                    "n_finite_eigenvalues": int(spectrum["finite"].sum()),
                    "n_solver_unstable_modes": int(
                        spectrum["solver_finite_mode"].sum()
                    ),
                    "n_physically_admissible_modes": int(
                        spectrum["physically_admissible"].sum()
                    ),
                    "nearest_ci_raw_index": int(nearest_ci["raw_index"]),
                    "nearest_ci_cr": float(nearest_ci["cr"]),
                    "nearest_ci_ci": float(nearest_ci["ci"]),
                    "nearest_ci_rel_distance": float(
                        nearest_ci["ci_rel_distance_to_pinn"]
                    ),
                    "pinn_matched_raw_index": matched_index,
                    "pinn_matched_cr": float(matched["cr"]),
                    "pinn_matched_ci": float(matched["ci"]),
                    "pinn_matched_omega_i": float(matched["omega_i"]),
                    "pinn_matched_ci_rel_distance": float(
                        matched["ci_rel_distance_to_pinn"]
                    ),
                    "pinn_matched_p_overlap": float(
                        matched["p_overlap_pinn"]
                    ),
                    "pinn_matched_q_overlap": float(
                        matched["q_overlap_pinn"]
                    ),
                    "pinn_matched_combined_overlap": float(
                        matched["combined_overlap_pinn"]
                    ),
                    "most_unstable_raw_index": unstable_index,
                    "most_unstable_cr": float(most_unstable["cr"]),
                    "most_unstable_ci": float(most_unstable["ci"]),
                    "most_unstable_omega_i": float(
                        most_unstable["omega_i"]
                    ),
                    "most_unstable_p_overlap": float(
                        most_unstable["p_overlap_pinn"]
                    ),
                    "most_unstable_q_overlap": float(
                        most_unstable["q_overlap_pinn"]
                    ),
                    "pinn_matched_is_most_unstable": same_mode,
                }
            )

            try:
                classic_fields, ci_classic = load_classic_full_mode(alpha, mach)
                matched_metrics, matched_profile = compare_mode_to_classic(
                    solver=solver,
                    vector=eigenvectors[:, matched_index],
                    classic_fields=classic_fields,
                    y_match_max=args.y_match_max,
                )
                unstable_metrics, unstable_profile = compare_mode_to_classic(
                    solver=solver,
                    vector=eigenvectors[:, unstable_index],
                    classic_fields=classic_fields,
                    y_match_max=args.y_match_max,
                )
                add_frame(
                    profile_store,
                    f"{pid}__matched_vs_classic",
                    matched_profile,
                )
                add_frame(
                    profile_store,
                    f"{pid}__unstable_vs_classic",
                    unstable_profile,
                )

                result.update(
                    {
                        "reference_success": True,
                        "ci_classic": float(ci_classic),
                        "ci_pinn_abs_err_classic": abs(
                            ci_pinn - float(ci_classic)
                        ),
                        "ci_pinn_rel_err_classic": abs(
                            ci_pinn - float(ci_classic)
                        )
                        / max(abs(float(ci_classic)), 1e-12),
                        "pinn_matched_ci_abs_err_classic": abs(
                            float(matched["ci"]) - float(ci_classic)
                        ),
                        "pinn_matched_ci_rel_err_classic": abs(
                            float(matched["ci"]) - float(ci_classic)
                        )
                        / max(abs(float(ci_classic)), 1e-12),
                        "most_unstable_ci_abs_err_classic": abs(
                            float(most_unstable["ci"]) - float(ci_classic)
                        ),
                        "most_unstable_ci_rel_err_classic": abs(
                            float(most_unstable["ci"]) - float(ci_classic)
                        )
                        / max(abs(float(ci_classic)), 1e-12),
                    }
                )
                for key, value in matched_metrics.items():
                    result[f"pinn_matched_{key}"] = float(value)
                for key, value in unstable_metrics.items():
                    result[f"most_unstable_{key}"] = float(value)

            except Exception as error:
                result["reference_error"] = (
                    f"{type(error).__name__}: {error}"
                )
                print("REFERENCE WARNING:", result["reference_error"])

            print(
                f"ci_PINN={ci_pinn:.8e}; matched_ci={float(matched['ci']):.8e}; "
                f"O_p={float(matched['p_overlap_pinn']):.6f}; "
                f"O_q={float(matched['q_overlap_pinn']):.6f}; "
                f"matched==max={same_mode}"
            )

        except Exception as error:
            result["selection_error"] = f"{type(error).__name__}: {error}"
            print("POINT FAILURE:", result["selection_error"])
            traceback.print_exc()

        summary_rows.append(result)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)

    if spectrum_frames:
        pd.concat(spectrum_frames, ignore_index=True, sort=False).to_csv(
            output_dir / "full_spectra.csv.gz",
            index=False,
            compression="gzip",
        )
    if profile_store:
        np.savez_compressed(
            output_dir / "selected_mode_profiles.npz",
            **profile_store,
        )

    technical = summary["technical_success"].fillna(False).astype(bool)
    reference = summary["reference_success"].fillna(False).astype(bool)
    matched_flag = (
        summary.get(
            "pinn_matched_is_most_unstable",
            pd.Series(False, index=summary.index),
        )
        .fillna(False)
        .astype(bool)
    )

    report = {
        "chart_id": chart_id,
        "checkpoint": str(checkpoint_path),
        "diagnostics_csv": str(diagnostics_path),
        "field_family": family,
        "n_points": int(len(summary)),
        "n_technical_success": int(technical.sum()),
        "n_reference_success": int(reference.sum()),
        "n_pinn_matches_most_unstable": int(
            (technical & matched_flag).sum()
        ),
        "all_technical_success": bool(technical.all()),
        "all_pinn_matches_most_unstable": bool(
            technical.all() and matched_flag.all()
        ),
        "regime_counts": {
            str(key): int(value)
            for key, value in summary["gep_regime"].value_counts().items()
        },
    }

    for column in [
        "ci_pinn_rel_err_classic",
        "pinn_matched_ci_rel_err_classic",
        "pinn_matched_p_overlap",
        "pinn_matched_q_overlap",
        "pinn_matched_p_rel_classic",
        "pinn_matched_rho_rel_classic",
        "pinn_matched_u_rel_classic",
        "pinn_matched_v_rel_classic",
        "pinn_matched_p_overlap_classic",
    ]:
        values = finite_metric(summary, column)
        if values is not None:
            report[f"{column}_mean"] = float(values.mean())
            report[f"{column}_max"] = float(values.max())
            report[f"{column}_min"] = float(values.min())

    (output_dir / "summary_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("CHART SUMMARY")
    print(json.dumps(report, indent=2, sort_keys=True))

    if not technical.all():
        raise SystemExit(2)


if __name__ == "__main__":
    main()
