#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from src.scripts.gep.selection.solve_dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from src.scripts.evaluation.evaluate_kh_subsonic_fixed_mach_modal_candidates import load_classic_full_mode
from src.scripts.gep.selection.audit_mid_joint_pinn_full_gep import (
    call_pinn_profiles,
    compare_mode_to_classic,
    evaluate_pinn,
    make_match_mask,
    mode_overlap_with_pinn,
    phase_alignment,
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
FIELDS = ("p", "rho", "u", "v")


def normalize_audit(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {"M": "Mach", "mach": "Mach", "Eta": "eta"}
    frame = frame.rename(
        columns={old: new for old, new in aliases.items() if old in frame and new not in frame}
    ).copy()
    required = {"Mach", "eta", "alpha", "chart_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Audit missing {missing}")
    for col in ("Mach", "eta", "alpha"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["Mach", "eta", "alpha", "chart_id"])


def choose_points(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    used = set()
    for name, target_m, target_eta in TARGETS:
        candidates = audit.loc[~audit.index.isin(used)].copy()
        distance = (
            ((candidates["Mach"] - target_m) / 0.08) ** 2
            + ((candidates["eta"] - target_eta) / 0.04) ** 2
        )
        index = distance.idxmin()
        used.add(index)
        row = audit.loc[index].to_dict()
        row.update({
            "selection_stratum": name,
            "target_Mach": target_m,
            "target_eta": target_eta,
            "selection_distance": float(distance.loc[index]),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    out.insert(0, "task_id", np.arange(len(out), dtype=int))
    out.insert(1, "mode_point_id", [f"MODE_{i:02d}" for i in range(len(out))])
    return out


def policy(mach: float, eta: float) -> tuple[int, float, float, str]:
    if mach >= 0.88 and eta <= 0.06:
        return 401, 20.0, 0.995, "extreme_longwave"
    if eta <= 0.12:
        return 401, 10.0, 0.99, "longwave"
    if eta >= 0.92:
        return 501, 5.0, 0.98, "near_neutral"
    return 401, 5.0, 0.98, "standard"


def command_prepare(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit = normalize_audit(pd.read_csv(args.central_audit))
    training = pd.read_csv(args.training_plan, sep="\t")
    required = {"chart_id", "output_dir"}
    missing = sorted(required.difference(training.columns))
    if missing:
        raise KeyError(f"Training plan missing {missing}")
    checkpoint = {
        str(row.chart_id): str(Path(str(row.output_dir)) / "model_state.pt")
        for _, row in training.iterrows()
    }

    selected = choose_points(audit)
    selected["checkpoint"] = selected["chart_id"].astype(str).map(checkpoint)
    if selected["checkpoint"].isna().any():
        raise KeyError("Missing checkpoint for one or more selected charts")

    policies = selected.apply(
        lambda row: policy(float(row.Mach), float(row.eta)), axis=1
    )
    selected["N"] = [item[0] for item in policies]
    selected["mapping_scale"] = [item[1] for item in policies]
    selected["xi_max"] = [item[2] for item in policies]
    selected["regime"] = [item[3] for item in policies]
    selected.to_csv(output / "mode_points_20_plan.csv", index=False)
    print(selected[
        ["task_id", "mode_point_id", "selection_stratum", "Mach", "eta",
         "alpha", "chart_id", "N", "mapping_scale", "xi_max"]
    ].to_string(index=False))


def select_central(solver, values, vectors, p_pinn, q_pinn, mask):
    values = np.asarray(values, np.complex128)
    candidate = np.where(
        np.isfinite(values.real) & np.isfinite(values.imag)
        & (values.imag > 0) & (values.imag <= 2.0)
        & (np.abs(values.real) <= 0.05)
    )[0]
    if len(candidate) == 0:
        raise RuntimeError("No unstable central mode")
    index = int(candidate[np.argmax(values[candidate].imag)])
    p_overlap, q_overlap, combined = mode_overlap_with_pinn(
        solver=solver, vector=vectors[:, index],
        p_pinn=p_pinn, q_pinn=q_pinn, match_mask=mask, p_weight=0.75,
    )
    return index, float(values[index].real), float(values[index].imag), float(p_overlap), float(q_overlap), float(combined)


def interp_complex(y_src, values, y_dst):
    return np.interp(y_dst, y_src, values.real) + 1j * np.interp(y_dst, y_src, values.imag)


def direct_fields(y, p, q, alpha, mach, ci):
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


def save_profile(path: Path, frame: pd.DataFrame, direct: dict[str, np.ndarray]):
    payload = {"y": frame["y"].to_numpy(float)}
    for field in FIELDS:
        payload[f"{field}_classic"] = (
            frame[f"{field}_classic_real"].to_numpy(float)
            + 1j * frame[f"{field}_classic_imag"].to_numpy(float)
        )
        payload[f"{field}_gep"] = (
            frame[f"{field}_gep_real"].to_numpy(float)
            + 1j * frame[f"{field}_gep_imag"].to_numpy(float)
        )
        payload[f"{field}_direct"] = np.asarray(direct[field], np.complex128)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def command_run(args):
    plan = pd.read_csv(args.plan)
    selected = plan.loc[plan["task_id"].astype(int).eq(int(args.index))]
    if len(selected) != 1:
        raise RuntimeError(f"Task {args.index}: found {len(selected)}")
    row = selected.iloc[0]
    output = Path(args.output_dir)
    (output / "shards").mkdir(parents=True, exist_ok=True)
    (output / "profiles").mkdir(parents=True, exist_ok=True)
    result = row.to_dict()
    result.update({"success": False, "error": ""})

    try:
        device = torch.device("cpu")
        field, ci_net, module, _, family = evaluate_pinn(
            checkpoint_path=Path(str(row.checkpoint)), device=device
        )
        solver = NotebookStyleDenseGEPSolver(
            alpha=float(row.alpha), Mach=float(row.Mach), n_points=int(row.N),
            mapping_kind="pin", mapping_scale=float(row.mapping_scale),
            xi_max=float(row.xi_max),
        )
        p_pinn, q_pinn, ci_pinn = call_pinn_profiles(
            field=field, ci_net=ci_net, module=module, family=family,
            y=solver.y, alpha=float(row.alpha), mach=float(row.Mach), device=device,
        )
        mask = make_match_mask(
            solver.y, p_pinn, y_match_max=12.0, amplitude_floor_fraction=0.02
        )
        values, vectors = solver.solve_all()
        index, cr, ci, p_ov, q_ov, combined = select_central(
            solver, values, vectors, p_pinn, q_pinn, mask
        )
        classic, ci_classic = load_classic_full_mode(float(row.alpha), float(row.Mach))
        metrics, profile = compare_mode_to_classic(
            solver=solver, vector=vectors[:, index],
            classic_fields=classic, y_match_max=12.0,
        )

        y = profile["y"].to_numpy(float)
        p_direct = interp_complex(solver.y, np.asarray(p_pinn, np.complex128), y)
        q_direct = interp_complex(solver.y, np.asarray(q_pinn, np.complex128), y)
        direct = direct_fields(
            y, p_direct, q_direct, float(row.alpha), float(row.Mach), float(ci_pinn)
        )
        classic_p = (
            profile["p_classic_real"].to_numpy(float)
            + 1j * profile["p_classic_imag"].to_numpy(float)
        )
        align_mask = np.abs(y) <= 12.0
        if align_mask.sum() < 20:
            align_mask = np.ones_like(y, dtype=bool)
        scale = phase_alignment(direct["p"], classic_p, y, align_mask)
        direct = {key: value * scale for key, value in direct.items()}

        profile_path = output / "profiles" / f"{row.mode_point_id}.npz"
        save_profile(profile_path, profile, direct)
        result.update({
            "success": True,
            "field_family": family,
            "ci_pinn": float(ci_pinn),
            "ci_classic": float(ci_classic),
            "ci_gep": ci,
            "cr_gep": cr,
            "p_overlap_pinn": p_ov,
            "q_overlap_pinn": q_ov,
            "combined_overlap_pinn": combined,
            "profile_path": str(profile_path),
            **{key: float(value) for key, value in metrics.items()},
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    pd.DataFrame([result]).to_csv(
        output / "shards" / f"shard_{int(args.index):05d}.csv", index=False
    )
    print(pd.DataFrame([result]).to_string(index=False))
    if not bool(result["success"]):
        raise SystemExit(2)


def command_merge(args):
    output = Path(args.output_dir)
    paths = sorted((output / "shards").glob("shard_*.csv"))
    if not paths:
        raise FileNotFoundError(output / "shards")
    frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True, sort=False)
    success = frame["success"].astype(str).str.lower().isin({"true", "1", "yes"})
    failed = frame.loc[~success]
    if not failed.empty:
        print(failed[["task_id", "mode_point_id", "error"]].to_string(index=False))
        raise RuntimeError(f"{len(failed)} mode-profile tasks failed")
    frame = frame.sort_values("task_id").reset_index(drop=True)
    frame.to_csv(output / "validation_mode_points_20_finest.csv", index=False)
    print(output / "validation_mode_points_20_finest.csv")


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--central-audit", required=True)
    prepare.add_argument("--training-plan", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.set_defaults(func=command_prepare)

    run = sub.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--output-dir", required=True)
    run.set_defaults(func=command_run)

    merge = sub.add_parser("merge")
    merge.add_argument("--output-dir", required=True)
    merge.set_defaults(func=command_merge)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
