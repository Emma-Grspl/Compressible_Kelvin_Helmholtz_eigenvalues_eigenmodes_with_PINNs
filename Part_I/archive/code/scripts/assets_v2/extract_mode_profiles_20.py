#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from scripts.compare_kh_subsonic_fixed_mach_modal_candidates import load_classic_full_mode
from scripts.dev.benchmark_subsonic_local_atlas_core_ci_seeded_gep_v2 import (
    align_complex,
    interp_complex,
    overlap_complex,
    rel_l2,
    split_gep_vector,
)
from scripts.dev.train_subsonic_seedGEP_pq2d_continuous_M_alpha_etaaware import (
    FieldPQNet as EtaAwareFieldPQNet,
)
from scripts.train_kh_subsonic_2d_pressure_pq_firstorder_mini import (
    FieldPQNet as LegacyFieldPQNet,
    fields_from_pq,
)

FIELDS = ("p", "rho", "u", "v")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract direct-PINN and final-GEP modal profiles for one release point."
    )
    parser.add_argument("--policy-csv", default="mode_extraction_policy_20.csv")
    parser.add_argument("--model-root", default="pinn_subsonic/models")
    parser.add_argument(
        "--output-root",
        default=(
            "assets/pinn_subsonic/local_atlas_v1/"
            "publication_assets_scientific_v2/data/mode_profiles"
        ),
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--index", type=int)
    selector.add_argument("--point-id")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def linear_dimensions(state: dict[str, torch.Tensor]) -> tuple[int, int]:
    layers: list[tuple[int, tuple[int, ...]]] = []
    for key, tensor in state.items():
        match = re.fullmatch(r"net\.(\d+)\.weight", key)
        if match:
            layers.append((int(match.group(1)), tuple(tensor.shape)))
    if not layers:
        raise RuntimeError("No linear layers found in checkpoint")
    layers.sort()
    return int(layers[0][1][1]), int(layers[-1][1][0])


def build_field_model(checkpoint: dict[str, Any]):
    args = dict(checkpoint["args"])
    state = checkpoint["field_state_dict"]
    input_dimension, output_dimension = linear_dimensions(state)
    if output_dimension != 4:
        raise RuntimeError(f"Unexpected output dimension: {output_dimension}")

    n_freq = int(args["n_freq"])
    legacy_dimension = 3 + 2 * n_freq
    eta_aware_dimension = 7 + 2 * n_freq

    mach_min = float(args["mach_min"])
    mach_max = float(args["mach_max"])
    eta_min = float(args["eta_min"])
    eta_max = float(args["eta_max"])
    alpha_min = eta_min * math.sqrt(max(1.0 - mach_max**2, 1.0e-14))
    alpha_max = eta_max * math.sqrt(max(1.0 - mach_min**2, 1.0e-14))

    common = dict(
        ymax=float(args["ymax"]),
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        mach_min=mach_min,
        mach_max=mach_max,
        width=int(args["width"]),
        depth=int(args["depth"]),
        n_freq=n_freq,
    )

    if input_dimension == legacy_dimension:
        architecture = "legacy"
        model = LegacyFieldPQNet(**common).double()
    elif input_dimension == eta_aware_dimension:
        architecture = "eta-aware"
        model = EtaAwareFieldPQNet(
            **common,
            eta_min=eta_min,
            eta_max=eta_max,
        ).double()
    else:
        raise RuntimeError(
            f"Unsupported architecture: input={input_dimension}, "
            f"legacy={legacy_dimension}, eta-aware={eta_aware_dimension}"
        )

    model.load_state_dict(state, strict=True)
    model.eval()
    family = "pQscaled" if "qscaled" in str(args.get("output_dir", "")).lower() else "pq"
    metadata = {
        "architecture": architecture,
        "field_family": family,
        "input_dimension": input_dimension,
        "output_dimension": output_dimension,
        "n_freq": n_freq,
        "width": int(args["width"]),
        "depth": int(args["depth"]),
        "amp_mask_frac": float(args.get("amp_mask_frac", 0.05)),
    }
    return model, family, metadata


def evaluate_direct(model, family, y, alpha, mach, ci_seed):
    yt = torch.tensor(y[:, None], dtype=torch.float64)
    at = torch.full_like(yt, alpha)
    mt = torch.full_like(yt, mach)
    with torch.no_grad():
        p_t, second_t = model(yt, at, mt)
    p = p_t.cpu().numpy().reshape(-1).astype(np.complex128)
    second = second_t.cpu().numpy().reshape(-1).astype(np.complex128)
    q = alpha * second if family == "pQscaled" else second
    rho, u, v, _ = fields_from_pq(y, p, q, alpha, mach, ci_seed)
    return {
        "p": np.asarray(p, np.complex128),
        "rho": np.asarray(rho, np.complex128),
        "u": np.asarray(u, np.complex128),
        "v": np.asarray(v, np.complex128),
    }


def solve_gep(alpha, mach, n_points, mapping_scale, xi_max, target_ci):
    solver = NotebookStyleDenseGEPSolver(
        alpha=alpha,
        Mach=mach,
        n_points=n_points,
        mapping_kind="pin",
        mapping_scale=mapping_scale,
        xi_max=xi_max,
    )
    mode, source, n_modes = solver.get_nearest_mode_to_target(
        target_guess=(0.0, float(target_ci)),
        prefer_positive_cr=False,
        ci_weight=2.0,
    )
    if mode is None:
        raise RuntimeError(f"No GEP mode near ci={target_ci:.16g}")
    return solver, mode, source, int(n_modes)


def interpolate_gep(solver, mode, mach, y_target):
    fields = split_gep_vector(mode["vector"], solver.n_points, mach)
    return {
        field: interp_complex(
            np.asarray(solver.y, float),
            np.asarray(fields[field], np.complex128),
            y_target,
        )
        for field in FIELDS
    }


def scale_fields(fields, scale):
    return {key: np.asarray(value, np.complex128) * scale for key, value in fields.items()}


def metrics(pred, ref, y, mask):
    result = {
        f"{field}_rel": rel_l2(pred[field], ref[field], y, mask)
        for field in FIELDS
    }
    result["p_overlap"] = overlap_complex(pred["p"], ref["p"], y, mask)
    return result


def save_profile(path, y, ref, pred, extras, shooting=None):
    payload: dict[str, Any] = {"y": np.asarray(y, float)}
    for field in FIELDS:
        payload[f"{field}_ref"] = np.asarray(ref[field], np.complex128)
        payload[f"{field}_pred"] = np.asarray(pred[field], np.complex128)
        if shooting is not None:
            payload[f"{field}_shooting"] = np.asarray(shooting[field], np.complex128)
    payload.update(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))

    policy = pd.read_csv(args.policy_csv)
    if args.index is not None:
        if args.index < 0 or args.index >= len(policy):
            raise IndexError(f"index={args.index}; valid range 0..{len(policy)-1}")
        row = policy.iloc[args.index]
    else:
        selected = policy.loc[policy["point_id"].astype(str) == str(args.point_id)]
        if len(selected) != 1:
            raise RuntimeError(f"Expected one row for {args.point_id}, found {len(selected)}")
        row = selected.iloc[0]

    point_id = str(row["point_id"])
    chart_id = str(row["chart_id"])
    mach = float(row["Mach"])
    eta = float(row["eta"])
    alpha = float(row["alpha"])
    ci_ref = float(row["ci_ref"])
    ci_seed = float(row["ci_seed"])
    ci_final = float(row["ci_final"])
    n_points = int(row["N"])
    mapping_scale = float(row["mapping_scale"])
    xi_max = float(row["xi_max"])

    output_root = Path(args.output_root)
    direct_path = output_root / "direct" / f"{point_id}.npz"
    gep_path = output_root / "gep" / f"{point_id}.npz"
    report_path = output_root / "reports" / f"{point_id}.json"
    if not args.overwrite and direct_path.exists() and gep_path.exists() and report_path.exists():
        print(f"{point_id}: already complete")
        return

    checkpoint_path = Path(args.model_root) / chart_id / "model_state.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model, family, model_metadata = build_field_model(checkpoint)

    classic_fields, ci_shooting = load_classic_full_mode(alpha, mach)
    y_ref = np.asarray(classic_fields["y"], float)
    shooting = {
        field: np.asarray(classic_fields[field], np.complex128)
        for field in FIELDS
    }

    direct = evaluate_direct(model, family, y_ref, alpha, mach, ci_seed)
    amp_frac = float(model_metadata["amp_mask_frac"])
    direct_mask = (
        np.isfinite(y_ref)
        & np.isfinite(shooting["p"])
        & np.isfinite(direct["p"])
        & (np.abs(shooting["p"]) >= amp_frac * np.nanmax(np.abs(shooting["p"])))
    )
    if direct_mask.sum() < 2:
        raise RuntimeError("Empty direct alignment mask")
    direct_scale = align_complex(direct["p"], shooting["p"], direct_mask)
    direct = scale_fields(direct, direct_scale)
    direct_metrics = metrics(direct, shooting, y_ref, direct_mask)

    solver_ref, mode_ref, source_ref, n_modes_ref = solve_gep(
        alpha, mach, n_points, mapping_scale, xi_max, ci_ref
    )
    classic_gep = interpolate_gep(solver_ref, mode_ref, mach, y_ref)

    solver_pred, mode_pred, source_pred, n_modes_pred = solve_gep(
        alpha, mach, n_points, mapping_scale, xi_max, ci_final
    )
    final_gep = interpolate_gep(solver_pred, mode_pred, mach, y_ref)

    y_min = max(float(np.min(y_ref)), float(np.min(solver_ref.y)), float(np.min(solver_pred.y)), -12.0)
    y_max = min(float(np.max(y_ref)), float(np.max(solver_ref.y)), float(np.max(solver_pred.y)), 12.0)
    gep_mask = (
        np.isfinite(y_ref)
        & (y_ref >= y_min)
        & (y_ref <= y_max)
        & np.isfinite(classic_gep["p"])
        & np.isfinite(final_gep["p"])
    )
    if gep_mask.sum() < 2:
        raise RuntimeError("Empty GEP alignment mask")

    classic_scale = align_complex(classic_gep["p"], shooting["p"], gep_mask)
    classic_gep = scale_fields(classic_gep, classic_scale)
    final_scale = align_complex(final_gep["p"], classic_gep["p"], gep_mask)
    final_gep = scale_fields(final_gep, final_scale)

    gep_metrics = metrics(final_gep, classic_gep, y_ref, gep_mask)
    gep_vs_shooting = metrics(final_gep, shooting, y_ref, gep_mask)

    common = {
        "point_id": np.array(point_id),
        "chart_id": np.array(chart_id),
        "Mach": np.array(mach),
        "eta": np.array(eta),
        "alpha": np.array(alpha),
        "ci_ref": np.array(ci_ref),
        "ci_seed": np.array(ci_seed),
        "ci_final": np.array(ci_final),
        "ci_shooting_recomputed": np.array(float(ci_shooting)),
        "architecture": np.array(model_metadata["architecture"]),
        "field_family": np.array(family),
    }

    save_profile(
        direct_path,
        y_ref,
        shooting,
        direct,
        {
            **common,
            "pipeline": np.array("direct"),
            "alignment_scale": np.array(direct_scale),
        },
    )
    save_profile(
        gep_path,
        y_ref,
        classic_gep,
        final_gep,
        {
            **common,
            "pipeline": np.array("gep"),
            "N": np.array(n_points),
            "mapping_kind": np.array("pin"),
            "mapping_scale": np.array(mapping_scale),
            "xi_max": np.array(xi_max),
            "ci_classic_gep": np.array(float(mode_ref["ci"])),
            "ci_final_gep": np.array(float(mode_pred["ci"])),
            "cr_classic_gep": np.array(float(mode_ref["cr"])),
            "cr_final_gep": np.array(float(mode_pred["cr"])),
            "classic_selection_source": np.array(source_ref),
            "final_selection_source": np.array(source_pred),
        },
        shooting=shooting,
    )

    report = {
        "point_id": point_id,
        "chart_id": chart_id,
        "Mach": mach,
        "eta": eta,
        "alpha": alpha,
        "ci_ref_csv": ci_ref,
        "ci_shooting_recomputed": float(ci_shooting),
        "ci_seed": ci_seed,
        "ci_final_csv": ci_final,
        "model": model_metadata,
        "numerical_policy": {
            "N": n_points,
            "mapping_kind": "pin",
            "mapping_scale": mapping_scale,
            "xi_max": xi_max,
        },
        "direct": direct_metrics,
        "classic_gep": {
            "cr": float(mode_ref["cr"]),
            "ci": float(mode_ref["ci"]),
            "selection_source": source_ref,
            "n_finite_modes": n_modes_ref,
        },
        "final_gep": {
            "cr": float(mode_pred["cr"]),
            "ci": float(mode_pred["ci"]),
            "selection_source": source_pred,
            "n_finite_modes": n_modes_pred,
            "vs_classic_gep": gep_metrics,
            "vs_shooting": gep_vs_shooting,
        },
        "outputs": {"direct": str(direct_path), "gep": str(gep_path)},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
