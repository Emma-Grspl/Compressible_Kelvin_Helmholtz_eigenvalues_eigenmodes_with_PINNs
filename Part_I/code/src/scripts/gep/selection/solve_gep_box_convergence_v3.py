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
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[5]
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
    overlap_complex,
    phase_alignment,
    rel_l2,
)

FIELDS = ("p", "rho", "u", "v")
BOXES = [
    ("compact", 4.0, 0.975, 0),
    ("reference", 5.0, 0.980, 1),
    ("extended", 6.0, 0.985, 2),
]
N_VALUES = [201, 301, 401, 501]


def load_profile(path):
    with np.load(path) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def interp_complex(y_src, values, y_dst):
    return np.interp(y_dst, y_src, values.real) + 1j * np.interp(y_dst, y_src, values.imag)


def save_profile(path: Path, frame: pd.DataFrame) -> None:
    payload = {"y": frame["y"].to_numpy(float)}
    for field in FIELDS:
        payload[f"{field}_gep"] = (
            frame[f"{field}_gep_real"].to_numpy(float)
            + 1j * frame[f"{field}_gep_imag"].to_numpy(float)
        )
        payload[f"{field}_classic"] = (
            frame[f"{field}_classic_real"].to_numpy(float)
            + 1j * frame[f"{field}_classic_imag"].to_numpy(float)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def choose_point(audit: pd.DataFrame, training: pd.DataFrame, target_m: float, target_eta: float):
    aliases = {"M": "Mach", "mach": "Mach", "Eta": "eta"}
    audit = audit.rename(
        columns={old: new for old, new in aliases.items() if old in audit and new not in audit}
    ).copy()
    for col in ("Mach", "eta", "alpha"):
        audit[col] = pd.to_numeric(audit[col], errors="coerce")
    audit = audit.dropna(subset=["Mach", "eta", "alpha", "chart_id"])
    interior = audit.loc[
        audit["Mach"].between(0.25, 0.75)
        & audit["eta"].between(0.25, 0.75)
    ].copy()
    if interior.empty:
        interior = audit
    distance = (
        ((interior["Mach"] - target_m) / 0.15) ** 2
        + ((interior["eta"] - target_eta) / 0.15) ** 2
    )
    row = interior.loc[distance.idxmin()].copy()
    match = training.loc[training["chart_id"].astype(str).eq(str(row.chart_id))]
    if match.empty:
        raise KeyError(f"No chart {row.chart_id} in training plan")
    row["checkpoint"] = str(Path(str(match.iloc[0].output_dir)) / "model_state.pt")
    return row


def command_prepare(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(args.central_audit)
    training = pd.read_csv(args.training_plan, sep="\t")
    point = choose_point(audit, training, args.target_mach, args.target_eta)

    rows = []
    task = 0
    for name, scale, xi, rank in BOXES:
        ymax = scale * xi / (1.0 - xi * xi)
        for n in N_VALUES:
            rows.append({
                "task_id": task,
                "point_id": "BOXCONV",
                "chart_id": str(point.chart_id),
                "checkpoint": str(point.checkpoint),
                "Mach": float(point.Mach),
                "eta": float(point.eta),
                "alpha": float(point.alpha),
                "box_name": name,
                "box_rank": rank,
                "mapping_scale": scale,
                "xi_max": xi,
                "effective_ymax": ymax,
                "N": n,
            })
            task += 1
    plan = pd.DataFrame(rows)
    plan.to_csv(output / "GEP_box_N_plan.csv", index=False)
    pd.DataFrame([point]).to_csv(output / "selected_point.csv", index=False)
    print(json.dumps({
        "n_tasks": len(plan),
        "point": {
            "Mach": float(point.Mach),
            "eta": float(point.eta),
            "alpha": float(point.alpha),
            "chart_id": str(point.chart_id),
        }
    }, indent=2))


def select_mode(solver, eigenvalues, eigenvectors, p_pinn, q_pinn, mask):
    values = np.asarray(eigenvalues, np.complex128)
    candidate = np.where(
        np.isfinite(values.real) & np.isfinite(values.imag)
        & (values.imag > 0) & (values.imag <= 2.0)
        & (np.abs(values.real) <= 0.05)
    )[0]
    if len(candidate) == 0:
        raise RuntimeError("No unstable central candidate")
    index = int(candidate[np.argmax(values[candidate].imag)])
    p_ov, q_ov, _ = mode_overlap_with_pinn(
        solver=solver, vector=eigenvectors[:, index],
        p_pinn=p_pinn, q_pinn=q_pinn, match_mask=mask, p_weight=0.75,
    )
    return index, float(values[index].real), float(values[index].imag), float(p_ov), float(q_ov)


def command_run(args):
    plan = pd.read_csv(args.plan)
    selected = plan.loc[plan["task_id"].astype(int).eq(int(args.index))]
    if len(selected) != 1:
        raise RuntimeError(f"Task {args.index}: {len(selected)} rows")
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
        eigenvalues, eigenvectors = solver.solve_all()
        index, cr, ci, p_ov, q_ov = select_mode(
            solver, eigenvalues, eigenvectors, p_pinn, q_pinn, mask
        )
        classic, ci_classic = load_classic_full_mode(float(row.alpha), float(row.Mach))
        metrics, profile = compare_mode_to_classic(
            solver=solver, vector=eigenvectors[:, index],
            classic_fields=classic, y_match_max=12.0,
        )
        profile_path = output / "profiles" / f"{row.box_name}_N{int(row.N)}.npz"
        save_profile(profile_path, profile)
        result.update({
            "success": True,
            "field_family": family,
            "ci_pinn": float(ci_pinn),
            "ci_classic": float(ci_classic),
            "cr": cr,
            "ci": ci,
            "p_overlap_pinn": p_ov,
            "q_overlap_pinn": q_ov,
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


def save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def command_merge(args):
    output = Path(args.output_dir)
    asset = Path(args.asset_dir)
    fig_dir = asset / "figures"
    table_dir = asset / "tables"
    data_dir = asset / "data"
    for p in (fig_dir, table_dir, data_dir):
        p.mkdir(parents=True, exist_ok=True)

    shards = sorted((output / "shards").glob("shard_*.csv"))
    if not shards:
        raise FileNotFoundError(output / "shards")
    raw = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True, sort=False)
    valid = raw.loc[
        raw["success"].astype(str).str.lower().isin({"true", "1", "yes"})
    ].copy()
    if len(valid) != len(raw):
        print(raw.loc[~raw.index.isin(valid.index), ["task_id", "error"]].to_string(index=False))
    if valid.empty:
        raise RuntimeError("No successful convergence tasks")

    reference = valid.sort_values(["box_rank", "N"]).iloc[-1]
    ref_profile = load_profile(reference.profile_path)
    y_ref = ref_profile["y"].astype(float)
    mask = np.abs(y_ref) <= 12.0
    if mask.sum() < 20:
        mask = np.ones_like(y_ref, dtype=bool)

    rows = []
    for _, row in valid.iterrows():
        profile = load_profile(row.profile_path)
        y = profile["y"].astype(float)
        fields = {}
        for field in FIELDS:
            fields[field] = interp_complex(y, profile[f"{field}_gep"], y_ref)
        scale = phase_alignment(fields["p"], ref_profile["p_gep"], y_ref, mask)
        for field in FIELDS:
            fields[field] *= scale
        out = row.to_dict()
        out["ci_abs_err_to_reference"] = abs(float(row.ci) - float(reference.ci))
        rels = []
        overlaps = []
        for field in FIELDS:
            rel = rel_l2(fields[field], ref_profile[f"{field}_gep"], y_ref, mask)
            overlap = overlap_complex(fields[field], ref_profile[f"{field}_gep"], y_ref, mask)
            out[f"{field}_rel_to_reference"] = rel
            out[f"{field}_overlap_to_reference"] = overlap
            rels.append(rel)
            overlaps.append(overlap)
        out["modal_rel_mean_to_reference"] = float(np.mean(rels))
        out["modal_rel_max_to_reference"] = float(np.max(rels))
        out["modal_overlap_min_to_reference"] = float(np.min(overlaps))
        rows.append(out)
    conv = pd.DataFrame(rows).sort_values(["box_rank", "N"])
    conv.to_csv(table_dir / "GEP_box_N_convergence.csv", index=False)
    conv.to_csv(output / "GEP_box_N_convergence.csv", index=False)
    pd.DataFrame([reference]).to_csv(data_dir / "GEP_box_N_reference_configuration.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4))
    for box_name, group in conv.groupby("box_name", sort=False):
        group = group.sort_values("N")
        label = (
            f"{box_name}: L={group.mapping_scale.iloc[0]:g}, "
            f"xi_max={group.xi_max.iloc[0]:.3f}, "
            f"|y|max≈{group.effective_ymax.iloc[0]:.1f}"
        )
        axes[0, 0].plot(group["N"], group["ci"], marker="o", label=label)
        axes[0, 1].plot(group["N"], np.clip(group["ci_abs_err_to_reference"], 1e-16, None),
                        marker="o", label=label)
        axes[1, 0].plot(group["N"], group["modal_overlap_min_to_reference"],
                        marker="o", label=label)
        axes[1, 1].plot(group["N"], np.clip(group["modal_rel_max_to_reference"], 1e-16, None),
                        marker="o", label=label)

    axes[0, 0].axhline(float(reference.ci), color="black", ls=":", lw=1.0,
                       label="finest reference")
    axes[0, 0].set(title=r"Selected $c_i$", xlabel="GEP grid size N", ylabel=r"$c_i$")
    axes[0, 1].set(title=r"$|c_i-c_i^{ref}|$", xlabel="GEP grid size N",
                   ylabel="absolute error", yscale="log")
    axes[1, 0].set(title=r"Minimum modal overlap over $p,\rho,u,v$",
                   xlabel="GEP grid size N", ylabel="minimum overlap")
    axes[1, 0].set_ylim(0.95, 1.0005)
    axes[1, 1].set(title=r"Maximum modal error over $p,\rho,u,v$",
                   xlabel="GEP grid size N", ylabel="relative error", yscale="log")
    for ax in axes.ravel():
        ax.grid(alpha=0.22, which="both")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        fr"Single-point GEP convergence: $M={float(reference.Mach):.5f}$, "
        fr"$\eta={float(reference.eta):.5f}$, $\alpha={float(reference.alpha):.6f}$"
    )
    fig.tight_layout()
    save(fig, fig_dir / "Fig_GEP_single_point_box_and_N_convergence")

    selected_configs = [
        conv.loc[(conv["box_name"] == "compact") & (conv["N"] == 201)].iloc[0],
        conv.loc[(conv["box_name"] == "reference") & (conv["N"] == 301)].iloc[0],
        reference,
    ]
    fig, axes = plt.subplots(4, 2, figsize=(11.2, 12.5), sharex=True)
    classic = ref_profile
    styles = [
        ("compact N=201", "tab:blue", "--"),
        ("reference N=301", "tab:green", "-."),
        ("extended N=501", "tab:orange", ":"),
    ]
    for i, field in enumerate(FIELDS):
        for j, (component, operator) in enumerate((("Re", np.real), ("Im", np.imag))):
            ax = axes[i, j]
            ax.plot(y_ref[mask], operator(classic[f"{field}_classic"][mask]),
                    color="black", lw=1.8, label="Classical")
            for row, (label, color, ls) in zip(selected_configs, styles):
                profile = load_profile(row.profile_path)
                values = interp_complex(
                    profile["y"].astype(float), profile[f"{field}_gep"], y_ref
                )
                scale = phase_alignment(values, ref_profile["p_gep"] if field == "p" else
                                        interp_complex(profile["y"].astype(float),
                                                       profile["p_gep"], y_ref),
                                        y_ref, mask) if False else 1.0
                # Use the p-based phase already implicit in comparison to classical:
                p_values = interp_complex(profile["y"].astype(float), profile["p_gep"], y_ref)
                p_scale = phase_alignment(p_values, ref_profile["p_gep"], y_ref, mask)
                values *= p_scale
                ax.plot(y_ref[mask], operator(values[mask]), color=color, ls=ls,
                        lw=1.35, label=label)
            ax.set_title(fr"{component}$({field})$")
            ax.grid(alpha=0.2)
            if j == 0:
                ax.set_ylabel("Amplitude")
    axes[-1, 0].set_xlabel(r"$y$")
    axes[-1, 1].set_xlabel(r"$y$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.972),
               ncol=4, frameon=False)
    fig.suptitle("Mode stability under simultaneous box-size and grid refinement",
                 y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    save(fig, fig_dir / "Fig_GEP_single_point_mode_overlay_box_and_N")

    summary = {
        "n_tasks": int(len(raw)),
        "n_success": int(len(valid)),
        "reference": {
            "box_name": str(reference.box_name),
            "N": int(reference.N),
            "mapping_scale": float(reference.mapping_scale),
            "xi_max": float(reference.xi_max),
            "ci": float(reference.ci),
        },
    }
    (output / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--central-audit", required=True)
    prepare.add_argument("--training-plan", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--target-mach", type=float, default=0.50)
    prepare.add_argument("--target-eta", type=float, default=0.50)
    prepare.set_defaults(func=command_prepare)

    run = sub.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--output-dir", required=True)
    run.set_defaults(func=command_run)

    merge = sub.add_parser("merge")
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--asset-dir", required=True)
    merge.set_defaults(func=command_merge)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
