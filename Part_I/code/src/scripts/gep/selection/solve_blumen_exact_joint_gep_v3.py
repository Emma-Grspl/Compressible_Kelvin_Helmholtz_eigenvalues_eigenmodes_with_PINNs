#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
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
    evaluate_pinn,
    make_match_mask,
    mode_overlap_with_pinn,
)


def parse_decimal_comma(text: str) -> float:
    return float(text.strip().replace(" ", "").replace(",", "."))


def read_blumen(directory: Path) -> pd.DataFrame:
    rows = []
    files = []
    for path in directory.glob("*.csv"):
        try:
            level = float(path.stem)
        except ValueError:
            continue
        files.append((level, path))
    if not files:
        raise FileNotFoundError(f"No numeric CSV files in {directory}")
    for level, path in sorted(files):
        for point_index, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not line.strip():
                continue
            parts = line.split(";")
            if len(parts) != 2:
                raise ValueError(f"{path}:{point_index}: expected alpha;Mach")
            alpha = parse_decimal_comma(parts[0])
            mach = parse_decimal_comma(parts[1])
            if -2e-3 <= alpha < 0:
                alpha = 0.0
            eta = (
                alpha / math.sqrt(max(1.0 - mach * mach, 1e-30))
                if 0 < mach < 1
                else math.nan
            )
            rows.append({
                "blumen_point_id": f"BLUMEN_{len(rows):04d}",
                "curve_id": path.stem,
                "ci_blumen": float(level),
                "alpha": float(alpha),
                "Mach": float(mach),
                "eta": float(eta),
                "source_file": path.name,
                "point_index": point_index,
            })
    return pd.DataFrame(rows)


def normalize_plan(path: Path) -> pd.DataFrame:
    plan = pd.read_csv(path, sep="\t").copy()
    required = {"chart_id", "output_dir", "mach_min", "mach_max", "eta_min", "eta_max"}
    missing = sorted(required.difference(plan.columns))
    if missing:
        raise KeyError(f"Training plan missing {missing}")
    plan["checkpoint"] = plan["output_dir"].map(
        lambda x: str(Path(str(x)) / "model_state.pt")
    )
    plan["chart_area"] = (
        (pd.to_numeric(plan["mach_max"]) - pd.to_numeric(plan["mach_min"]))
        * (pd.to_numeric(plan["eta_max"]) - pd.to_numeric(plan["eta_min"]))
    )
    return plan


def route_chart(plan: pd.DataFrame, mach: float, eta: float) -> pd.Series:
    tol = 5e-10
    covering = plan.loc[
        (pd.to_numeric(plan["mach_min"]) - tol <= mach)
        & (mach <= pd.to_numeric(plan["mach_max"]) + tol)
        & (pd.to_numeric(plan["eta_min"]) - tol <= eta)
        & (eta <= pd.to_numeric(plan["eta_max"]) + tol)
    ].copy()
    if covering.empty:
        raise RuntimeError(f"No chart covers M={mach}, eta={eta}")
    m_half = 0.5 * (
        pd.to_numeric(covering["mach_max"]) - pd.to_numeric(covering["mach_min"])
    ).clip(lower=1e-12)
    e_half = 0.5 * (
        pd.to_numeric(covering["eta_max"]) - pd.to_numeric(covering["eta_min"])
    ).clip(lower=1e-12)
    m_center = 0.5 * (
        pd.to_numeric(covering["mach_max"]) + pd.to_numeric(covering["mach_min"])
    )
    e_center = 0.5 * (
        pd.to_numeric(covering["eta_max"]) + pd.to_numeric(covering["eta_min"])
    )
    covering["route_distance"] = (
        ((mach - m_center) / m_half) ** 2 + ((eta - e_center) / e_half) ** 2
    )
    return covering.sort_values(
        ["chart_area", "route_distance", "chart_id"], kind="mergesort"
    ).iloc[0]


def numerical_policy(mach: float, eta: float) -> tuple[int, float, float, str]:
    if mach >= 0.88 and eta <= 0.06:
        return 301, 20.0, 0.995, "extreme_longwave"
    if eta <= 0.12:
        return 301, 10.0, 0.99, "longwave"
    if eta >= 0.92:
        return 401, 5.0, 0.98, "near_neutral"
    return 301, 5.0, 0.98, "standard"


def select_central(
    solver,
    eigenvalues,
    eigenvectors,
    p_pinn,
    q_pinn,
    match_mask,
) -> tuple[int, float, float, float, float]:
    values = np.asarray(eigenvalues, dtype=np.complex128)
    candidate = np.where(
        np.isfinite(values.real)
        & np.isfinite(values.imag)
        & (values.imag > 0)
        & (values.imag <= 2.0)
        & (np.abs(values.real) <= 0.05)
    )[0]
    if len(candidate) == 0:
        raise RuntimeError("No unstable central candidate")
    index = int(candidate[np.argmax(values[candidate].imag)])
    p_overlap, q_overlap, _ = mode_overlap_with_pinn(
        solver=solver,
        vector=eigenvectors[:, index],
        p_pinn=p_pinn,
        q_pinn=q_pinn,
        match_mask=match_mask,
        p_weight=0.75,
    )
    return (
        index,
        float(values[index].real),
        float(values[index].imag),
        float(p_overlap),
        float(q_overlap),
    )


def command_prepare(args) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_points = read_blumen(Path(args.blumen_dir))
    plan = normalize_plan(Path(args.training_plan))

    rows = []
    for _, point in all_points.iterrows():
        mach = float(point.Mach)
        alpha = float(point.alpha)
        eta = float(point.eta)
        status = "eligible"
        chart_id = ""
        checkpoint = ""
        if not (0 < mach < 1 and alpha > 0 and 0.02 <= eta <= 0.98):
            status = "outside_validated_atlas"
        else:
            try:
                chart = route_chart(plan, mach, eta)
                chart_id = str(chart.chart_id)
                checkpoint = str(chart.checkpoint)
            except Exception:
                status = "outside_validated_atlas"
        n, scale, xi, regime = numerical_policy(mach, eta) if status == "eligible" else (0, 0, 0, "")
        row = point.to_dict()
        row.update({
            "eligibility": status,
            "chart_id": chart_id,
            "checkpoint": checkpoint,
            "N": n,
            "mapping_scale": scale,
            "xi_max": xi,
            "regime": regime,
        })
        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "blumen_all_points.csv", index=False)
    eligible = manifest.loc[manifest["eligibility"].eq("eligible")].reset_index(drop=True)
    eligible.insert(0, "task_id", np.arange(len(eligible), dtype=int))
    eligible.to_csv(out / "blumen_exact_plan.csv", index=False)
    print(json.dumps({
        "n_all": int(len(manifest)),
        "n_eligible": int(len(eligible)),
        "plan": str(out / "blumen_exact_plan.csv"),
    }, indent=2))


def command_run(args) -> None:
    plan = pd.read_csv(args.plan)
    row = plan.loc[plan["task_id"].astype(int).eq(int(args.index))]
    if len(row) != 1:
        raise RuntimeError(f"Task {args.index}: found {len(row)} rows")
    row = row.iloc[0]
    output = Path(args.output_dir)
    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    result = row.to_dict()
    result.update({"success": False, "error": ""})
    try:
        device = torch.device("cpu")
        field, ci_net, module, _, family = evaluate_pinn(
            checkpoint_path=Path(str(row.checkpoint)),
            device=device,
        )
        solver = NotebookStyleDenseGEPSolver(
            alpha=float(row.alpha),
            Mach=float(row.Mach),
            n_points=int(row.N),
            mapping_kind="pin",
            mapping_scale=float(row.mapping_scale),
            xi_max=float(row.xi_max),
        )
        p_pinn, q_pinn, ci_pinn = call_pinn_profiles(
            field=field,
            ci_net=ci_net,
            module=module,
            family=family,
            y=solver.y,
            alpha=float(row.alpha),
            mach=float(row.Mach),
            device=device,
        )
        mask = make_match_mask(
            solver.y, p_pinn, y_match_max=12.0, amplitude_floor_fraction=0.02
        )
        eigenvalues, eigenvectors = solver.solve_all()
        raw_index, cr, ci_gep, p_overlap, q_overlap = select_central(
            solver, eigenvalues, eigenvectors, p_pinn, q_pinn, mask
        )
        _, ci_classic = load_classic_full_mode(float(row.alpha), float(row.Mach))
        result.update({
            "success": True,
            "field_family": family,
            "ci_pinn": float(ci_pinn),
            "ci_gep": float(ci_gep),
            "ci_classic": float(ci_classic),
            "cr_gep": float(cr),
            "raw_index": int(raw_index),
            "p_overlap_pinn": p_overlap,
            "q_overlap_pinn": q_overlap,
            "gep_classic_abs_err": abs(float(ci_gep) - float(ci_classic)),
            "classic_blumen_abs_err": abs(float(ci_classic) - float(row.ci_blumen)),
            "gep_blumen_abs_err": abs(float(ci_gep) - float(row.ci_blumen)),
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    path = shard_dir / f"shard_{int(args.index):05d}.csv"
    pd.DataFrame([result]).to_csv(path, index=False)
    print(pd.DataFrame([result]).to_string(index=False))
    if not bool(result["success"]):
        raise SystemExit(2)


def save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def command_merge(args) -> None:
    output = Path(args.output_dir)
    asset = Path(args.asset_dir)
    data_dir = asset / "data"
    fig_dir = asset / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_points = pd.read_csv(output / "blumen_all_points.csv")
    shards = sorted((output / "shards").glob("shard_*.csv"))
    if not shards:
        raise FileNotFoundError(output / "shards")
    results = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True, sort=False)
    merged = all_points.merge(
        results.drop(columns=[c for c in all_points.columns if c in results.columns and c != "blumen_point_id"]),
        on="blumen_point_id",
        how="left",
    )
    merged.to_csv(data_dir / "Blumen_exact_point_comparison.csv", index=False)

    valid = merged.loc[
        merged["success"].astype(str).str.lower().isin({"true", "1", "yes"})
    ].copy()
    for col in ("ci_blumen", "ci_classic", "ci_gep"):
        valid[col] = pd.to_numeric(valid[col], errors="coerce")
    valid = valid.dropna(subset=["Mach", "alpha", "ci_blumen", "ci_classic", "ci_gep"])
    if valid.empty:
        raise RuntimeError("No successful exact Blumen evaluations")

    all_ci = np.concatenate([
        valid["ci_blumen"].to_numpy(),
        valid["ci_classic"].to_numpy(),
        valid["ci_gep"].to_numpy(),
    ])
    norm = Normalize(vmin=float(np.min(all_ci)), vmax=float(np.max(all_ci)))

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.3), sharex=True, sharey=True)
    panels = [
        ("ci_blumen", "Blumen digitized values"),
        ("ci_classic", "Classical shooting at identical points"),
        ("ci_gep", "PINN + GEP at identical points"),
    ]
    scatter = None
    for ax, (column, title) in zip(axes, panels):
        scatter = ax.scatter(
            valid["alpha"], valid["Mach"], c=valid[column],
            cmap="viridis", norm=norm, s=22, marker="o",
            linewidths=0.35, edgecolors="black",
        )
        ax.set_title(title)
        ax.set_xlabel(r"Wavenumber $\alpha$")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel(r"Mach number $M$")
    fig.colorbar(scatter, ax=axes, label=r"$c_i$", shrink=0.90)
    fig.suptitle(
        r"Pointwise comparison at the exact digitized Blumen $(\alpha,M)$ pairs",
        y=1.02,
    )
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.13, top=0.85, wspace=0.08)
    save(fig, fig_dir / "Fig_Blumen_exact_points_classical_PINN_GEP")

    errors = valid[["Mach", "alpha", "gep_classic_abs_err", "gep_blumen_abs_err"]].copy()
    values = pd.concat([
        pd.to_numeric(errors["gep_classic_abs_err"], errors="coerce"),
        pd.to_numeric(errors["gep_blumen_abs_err"], errors="coerce"),
    ]).dropna()
    positive = values[values > 0]
    vmin = max(float(positive.min()), 1e-14)
    vmax = max(float(positive.max()), vmin * 1.01)
    lognorm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), sharex=True, sharey=True)
    for ax, column, title in [
        (axes[0], "gep_classic_abs_err", "PINN + GEP versus classical shooting"),
        (axes[1], "gep_blumen_abs_err", "PINN + GEP versus digitized Blumen"),
    ]:
        sc = ax.scatter(
            valid["alpha"], valid["Mach"],
            c=np.clip(pd.to_numeric(valid[column], errors="coerce"), vmin, None),
            norm=lognorm, cmap="magma", s=22, linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel(r"Wavenumber $\alpha$")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel(r"Mach number $M$")
    fig.colorbar(sc, ax=axes, label=r"Absolute $c_i$ error", shrink=0.90)
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.14, top=0.87, wspace=0.10)
    save(fig, fig_dir / "Fig_Blumen_exact_points_absolute_errors")

    summary = {
        "n_all_digitized": int(len(all_points)),
        "n_eligible": int((all_points["eligibility"] == "eligible").sum()),
        "n_success": int(len(valid)),
        "n_failed": int(len(results) - results["success"].astype(str).str.lower().isin({"true", "1", "yes"}).sum()),
    }
    (output / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--blumen-dir", required=True)
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
    merge.add_argument("--asset-dir", required=True)
    merge.set_defaults(func=command_merge)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
