#!/usr/bin/env python3
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
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from scripts.compare_kh_subsonic_fixed_mach_modal_candidates import load_classic_full_mode
from scripts.dev.test_mid_joint_pinn_full_gep import (
    call_pinn_profiles,
    compare_mode_to_classic,
    evaluate_pinn,
    make_match_mask,
    mode_overlap_with_pinn,
)

FIELDS = ("p", "rho", "u", "v")


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 320) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def interp_complex(y0: np.ndarray, z0: np.ndarray, y1: np.ndarray) -> np.ndarray:
    return np.interp(y1, y0, z0.real) + 1j * np.interp(y1, y0, z0.imag)


def phase_scale(candidate: np.ndarray, reference: np.ndarray, y: np.ndarray, mask: np.ndarray) -> complex:
    yc = y[mask]
    cand = candidate[mask]
    ref = reference[mask]
    num = np.trapezoid(np.conj(cand) * ref, yc)
    den = np.trapezoid(np.conj(cand) * cand, yc)
    return 1.0 + 0.0j if abs(den) < 1.0e-30 else num / den


def physical_eta(alpha: float, mach: float) -> float:
    return alpha / math.sqrt(max(1.0 - mach * mach, 1.0e-30))


def load_training_plan(path: Path) -> pd.DataFrame:
    plan = pd.read_csv(path, sep="\t").copy()
    required = {"chart_id", "output_dir", "mach_min", "mach_max", "eta_min", "eta_max"}
    missing = sorted(required.difference(plan.columns))
    if missing:
        raise KeyError(f"Training plan missing columns: {missing}")
    plan["checkpoint"] = plan["output_dir"].map(lambda x: str(Path(str(x)) / "model_state.pt"))
    plan["chart_area"] = (
        (pd.to_numeric(plan["mach_max"]) - pd.to_numeric(plan["mach_min"]))
        * (pd.to_numeric(plan["eta_max"]) - pd.to_numeric(plan["eta_min"]))
    )
    return plan


def route_chart(plan: pd.DataFrame, alpha: float, mach: float) -> pd.Series:
    eta = physical_eta(alpha, mach)
    tol = 5.0e-10
    covering = plan.loc[
        (pd.to_numeric(plan["mach_min"]) - tol <= mach)
        & (mach <= pd.to_numeric(plan["mach_max"]) + tol)
        & (pd.to_numeric(plan["eta_min"]) - tol <= eta)
        & (eta <= pd.to_numeric(plan["eta_max"]) + tol)
    ].copy()
    if covering.empty:
        raise RuntimeError(f"No chart covers alpha={alpha:.8g}, M={mach:.8g}, eta={eta:.8g}")

    mh = 0.5 * (pd.to_numeric(covering["mach_max"]) - pd.to_numeric(covering["mach_min"])).clip(lower=1e-12)
    eh = 0.5 * (pd.to_numeric(covering["eta_max"]) - pd.to_numeric(covering["eta_min"])).clip(lower=1e-12)
    mc = 0.5 * (pd.to_numeric(covering["mach_max"]) + pd.to_numeric(covering["mach_min"]))
    ec = 0.5 * (pd.to_numeric(covering["eta_max"]) + pd.to_numeric(covering["eta_min"]))
    covering["route_distance"] = ((mach - mc) / mh) ** 2 + ((eta - ec) / eh) ** 2
    row = covering.sort_values(["chart_area", "route_distance", "chart_id"]).iloc[0].copy()

    checkpoint = Path(str(row["checkpoint"]))
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    elif not checkpoint.is_file() and "assets" in checkpoint.parts:
        # training_plan.tsv may contain an absolute Jean-Zay path. Rebase the
        # suffix beginning at assets/ onto the local repository root.
        assets_index = checkpoint.parts.index("assets")
        checkpoint = ROOT.joinpath(*checkpoint.parts[assets_index:])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing local checkpoint for chart {row['chart_id']}:\n  {checkpoint}\n"
            "Synchronize assets/pinn_subsonic/joint_ci_mode_atlas_v2 first."
        )
    row["checkpoint"] = str(checkpoint)
    return row


def numerical_policy(alpha: float, mach: float, exact_mode: bool) -> dict[str, Any]:
    eta = physical_eta(alpha, mach)
    if mach >= 0.88 and eta <= 0.06:
        return {"N": 401 if exact_mode else 301, "mapping_scale": 20.0, "xi_max": 0.995, "regime": "extreme_longwave"}
    if eta <= 0.12:
        return {"N": 401 if exact_mode else 301, "mapping_scale": 10.0, "xi_max": 0.990, "regime": "longwave"}
    if eta >= 0.92:
        return {"N": 501 if exact_mode else 401, "mapping_scale": 5.0, "xi_max": 0.980, "regime": "near_neutral"}
    return {"N": 401 if exact_mode else 301, "mapping_scale": 5.0, "xi_max": 0.980, "regime": "standard"}


def select_central_mode(solver, eigenvalues, eigenvectors, p_pinn, q_pinn, mask):
    values = np.asarray(eigenvalues, dtype=np.complex128)
    candidates = np.where(
        np.isfinite(values.real) & np.isfinite(values.imag)
        & (values.imag > 0.0) & (values.imag <= 2.0)
        & (np.abs(values.real) <= 0.05)
    )[0]
    if len(candidates) == 0:
        raise RuntimeError("No unstable central GEP eigenvalue found")
    index = int(candidates[np.argmax(values[candidates].imag)])
    p_ov, q_ov, combined = mode_overlap_with_pinn(
        solver=solver,
        vector=eigenvectors[:, index],
        p_pinn=p_pinn,
        q_pinn=q_pinn,
        match_mask=mask,
        p_weight=0.75,
    )
    return index, {
        "cr": float(values[index].real),
        "ci": float(values[index].imag),
        "omega_i": float(solver.alpha * values[index].imag),
        "p_overlap_pinn": float(p_ov),
        "q_overlap_pinn": float(q_ov),
        "combined_overlap_pinn": float(combined),
    }


def direct_fields(y, p, q, alpha, mach, ci):
    ubar = np.tanh(y)
    ubar_y = 1.0 - ubar**2
    c = 1j * float(ci)
    denominator = ubar - c
    rho = float(mach) ** 2 * p
    v = -q / (1j * float(alpha) * denominator)
    u = -(ubar_y * v + 1j * float(alpha) * p) / (1j * float(alpha) * denominator)
    return {"p": p, "rho": rho, "u": u, "v": v}


def profile_family(profile: pd.DataFrame, suffix: str) -> dict[str, np.ndarray]:
    return {
        field: profile[f"{field}_{suffix}_real"].to_numpy(float)
        + 1j * profile[f"{field}_{suffix}_imag"].to_numpy(float)
        for field in FIELDS
    }


def align_family(family, reference_p, y, mask):
    scale = phase_scale(family["p"], reference_p, y, mask)
    return {name: scale * np.asarray(values, np.complex128) for name, values in family.items()}, scale


def get_model(cache: dict[str, tuple], checkpoint: Path, device: torch.device):
    key = str(checkpoint.resolve())
    if key not in cache:
        cache[key] = evaluate_pinn(checkpoint_path=checkpoint, device=device)
    return cache[key]


def evaluate_point(alpha, mach, plan, cache, device, exact_mode, need_profile):
    chart = route_chart(plan, alpha, mach)
    checkpoint = Path(str(chart["checkpoint"]))
    field, ci_net, module, _, family = get_model(cache, checkpoint, device)
    policy = numerical_policy(alpha, mach, exact_mode)

    solver = NotebookStyleDenseGEPSolver(
        alpha=float(alpha),
        Mach=float(mach),
        n_points=int(policy["N"]),
        mapping_kind="pin",
        mapping_scale=float(policy["mapping_scale"]),
        xi_max=float(policy["xi_max"]),
    )
    p_pinn, q_pinn, ci_pinn = call_pinn_profiles(
        field=field, ci_net=ci_net, module=module, family=family,
        y=solver.y, alpha=float(alpha), mach=float(mach), device=device,
    )
    mask = make_match_mask(solver.y, p_pinn, y_match_max=12.0, amplitude_floor_fraction=0.02)
    eigenvalues, eigenvectors = solver.solve_all()
    raw_index, selected = select_central_mode(solver, eigenvalues, eigenvectors, p_pinn, q_pinn, mask)
    classic_fields, ci_classic = load_classic_full_mode(float(alpha), float(mach))

    result = {
        "alpha": float(alpha), "Mach": float(mach), "eta": float(physical_eta(alpha, mach)),
        "chart_id": str(chart["chart_id"]), "checkpoint": str(checkpoint),
        "field_family": str(family), "N": int(policy["N"]),
        "mapping_scale": float(policy["mapping_scale"]), "xi_max": float(policy["xi_max"]),
        "regime": str(policy["regime"]), "ci_classic": float(ci_classic),
        "ci_pinn": float(ci_pinn), "ci_gep": float(selected["ci"]),
        "cr_gep": float(selected["cr"]), "omega_gep": float(selected["omega_i"]),
        "p_overlap_pinn": float(selected["p_overlap_pinn"]),
        "q_overlap_pinn": float(selected["q_overlap_pinn"]), "raw_index": int(raw_index),
    }
    if not need_profile:
        return result

    metrics, profile = compare_mode_to_classic(
        solver=solver, vector=eigenvectors[:, raw_index],
        classic_fields=classic_fields, y_match_max=12.0,
    )
    y = profile["y"].to_numpy(float)
    classic = profile_family(profile, "classic")
    gep = profile_family(profile, "gep")
    p_direct = interp_complex(solver.y, np.asarray(p_pinn, np.complex128), y)
    q_direct = interp_complex(solver.y, np.asarray(q_pinn, np.complex128), y)
    direct = direct_fields(y, p_direct, q_direct, alpha, mach, ci_pinn)

    align_mask = np.abs(y) <= 12.0
    if int(align_mask.sum()) < 20:
        align_mask = np.ones_like(y, dtype=bool)
    direct, direct_scale = align_family(direct, classic["p"], y, align_mask)
    gep, gep_scale = align_family(gep, classic["p"], y, align_mask)

    norm = float(np.max(np.abs(classic["rho"])))
    if not math.isfinite(norm) or norm < 1e-30:
        norm = 1.0
    for fam in (classic, direct, gep):
        for field_name in FIELDS:
            fam[field_name] = fam[field_name] / norm

    result.update({
        "y": y, "classic": classic, "direct": direct, "gep": gep,
        "direct_scale": direct_scale, "gep_scale": gep_scale,
        "normalization_rho_classic": norm,
        "metrics": {key: float(value) for key, value in metrics.items()},
    })
    return result


def write_mode_csv(result, path: Path):
    frame = pd.DataFrame({"y": result["y"]})
    for method in ("classic", "direct", "gep"):
        for field in FIELDS:
            frame[f"{field}_{method}_real"] = result[method][field].real
            frame[f"{field}_{method}_imag"] = result[method][field].imag
    for key in ("alpha", "Mach", "eta", "ci_classic", "ci_pinn", "ci_gep", "cr_gep", "N", "mapping_scale", "xi_max"):
        frame[key] = result[key]
    frame["chart_id"] = result["chart_id"]
    frame["field_family"] = result["field_family"]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def plot_mode(result, stem: Path):
    colors = {"classic": "black", "direct": "tab:blue", "gep": "tab:orange"}
    labels = {"classic": "Classical shooting", "direct": "Direct PINN", "gep": "PINN + GEP"}
    panels = (("p", r"Pressure $\hat p$"), ("rho", r"Density $\hat\rho$"),
              ("v", r"Transverse velocity $\hat v$"), ("u", r"Streamwise velocity $\hat u$"))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.4), sharex=True)
    for axis, (field, title) in zip(axes.ravel(), panels):
        for method in ("classic", "direct", "gep"):
            z = result[method][field]
            axis.plot(result["y"], z.real, color=colors[method], lw=1.9)
            axis.plot(result["y"], z.imag, color=colors[method], lw=1.7, ls="--")
        axis.axhline(0.0, color="0.72", lw=0.7)
        axis.set_title(title, fontsize=14)
        axis.set_xlabel(r"$y$")
        axis.set_ylabel("Amplitude")
        axis.grid(alpha=0.22)

    methods = [Line2D([0], [0], color=colors[m], lw=2.3, label=labels[m]) for m in ("classic", "direct", "gep")]
    components = [Line2D([0], [0], color="0.25", lw=2, label="Real part"),
                  Line2D([0], [0], color="0.25", lw=2, ls="--", label="Imaginary part")]
    first = fig.legend(handles=methods, loc="upper center", bbox_to_anchor=(0.36, 0.945), ncol=3, frameon=False, title="Method")
    fig.add_artist(first)
    fig.legend(handles=components, loc="upper center", bbox_to_anchor=(0.82, 0.945), ncol=2, frameon=False, title="Component")
    fig.suptitle(
        rf"Subsonic mode comparison at $\alpha={result['alpha']:.3f}$ and $M={result['Mach']:.3f}$" "\n"
        rf"$c_i^{{class}}={result['ci_classic']:.6f}$, $c_i^{{PINN}}={result['ci_pinn']:.6f}$, $c_i^{{GEP}}={result['ci_gep']:.6f}$",
        fontsize=15, y=0.995,
    )
    fig.tight_layout(rect=(0.025, 0.025, 0.975, 0.875))
    save_figure(fig, stem)


def load_heatmap(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Mach", "alpha", "ci_final_abs_err"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Canonical CSV missing columns: {missing}")
    out = frame[["Mach", "alpha", "ci_final_abs_err"]].apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def alpha_grid(mach, alpha_min, alpha_max, n_alpha, required_alpha):
    physical_max = 0.98 * math.sqrt(max(1.0 - mach * mach, 0.0))
    upper = physical_max if alpha_max is None else min(alpha_max, physical_max)
    if alpha_min <= 0 or upper <= alpha_min:
        raise ValueError(f"Invalid alpha interval [{alpha_min}, {upper}]")
    values = np.linspace(alpha_min, upper, int(n_alpha))
    if alpha_min <= required_alpha <= upper:
        values = np.concatenate([values, [required_alpha]])
    return np.unique(np.round(values, 12))


def valid_rows(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "success" not in frame:
        return pd.DataFrame()
    success = frame["success"].astype(str).str.lower().isin({"true", "1", "yes"})
    return frame.loc[success].copy()


def compute_cut(alphas, mach, plan, cache, device, path: Path):
    completed = valid_rows(path)
    done = set(np.round(pd.to_numeric(completed.get("alpha", pd.Series(dtype=float)), errors="coerce").dropna(), 12))
    rows = completed.to_dict("records") if not completed.empty else []

    for i, alpha in enumerate(alphas, start=1):
        if round(float(alpha), 12) in done:
            print(f"[cut {i:02d}/{len(alphas):02d}] alpha={alpha:.8f}: cached")
            continue
        print(f"[cut {i:02d}/{len(alphas):02d}] alpha={alpha:.8f}: evaluating", flush=True)
        row = {"alpha": float(alpha), "Mach": float(mach), "success": False, "error": ""}
        try:
            row.update(evaluate_point(float(alpha), float(mach), plan, cache, device, False, False))
            row["success"] = True
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
            print(row["error"], file=sys.stderr)
        rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)

    frame = pd.DataFrame(rows)
    success = frame["success"].astype(str).str.lower().isin({"true", "1", "yes"})
    return frame.loc[success].sort_values("alpha").reset_index(drop=True)


def plot_cut_heatmap(cut, heatmap, mach, stem: Path):
    if len(cut) < 3:
        raise RuntimeError(f"Only {len(cut)} successful spectral-cut points")
    errors = heatmap["ci_final_abs_err"].to_numpy(float)
    positive = errors[np.isfinite(errors) & (errors > 0)]
    vmin = max(float(positive.min()), 1e-14)
    vmax = max(float(positive.max()), 10 * vmin)

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.7), gridspec_kw={"width_ratios": [1.05, 1.0]})
    axes[0].plot(cut["alpha"], cut["ci_classic"], color="black", lw=1.9, marker="o", ms=3.8, label="Classical shooting")
    axes[0].plot(cut["alpha"], cut["ci_pinn"], color="tab:blue", lw=1.5, ls="--", marker="s", ms=3.5, mfc="white", label="Direct PINN")
    axes[0].plot(cut["alpha"], cut["ci_gep"], color="tab:orange", lw=1.6, ls="-.", marker="^", ms=3.8, mfc="white", label="PINN + GEP")
    axes[0].set(xlabel=r"Wavenumber $\alpha$", ylabel=r"$c_i$", title=rf"Exact spectral cut at $M={mach:.3f}$")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)

    scatter = axes[1].scatter(heatmap["Mach"], heatmap["alpha"], c=np.clip(errors, vmin, vmax), cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax), marker="s", s=34, linewidths=0)
    axes[1].axvline(mach, color="white", lw=2.4, ls="--", label=rf"$M={mach:.3f}$ cut")
    axes[1].axvline(mach, color="black", lw=0.8, ls="--")
    mm = np.linspace(0, 1, 600)
    axes[1].plot(mm, np.sqrt(np.clip(1-mm**2, 0, None)), color="black", ls=":", lw=1)
    axes[1].set(xlabel=r"Mach number $M$", ylabel=r"Wavenumber $\alpha$", title=r"Pointwise $|c_i^{GEP}-c_i^{class}|$", xlim=(0, 1), ylim=(0, 1))
    axes[1].grid(alpha=0.12)
    axes[1].legend(frameon=False, loc="lower left")
    cbar = fig.colorbar(scatter, ax=axes[1], pad=0.025)
    cbar.set_label(r"Absolute $c_i$ error")
    fig.suptitle("Subsonic spectral comparison and fixed-Mach cut", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, stem)


def serializable_metadata(result):
    payload = {k: v for k, v in result.items() if k not in {"y", "classic", "direct", "gep", "direct_scale", "gep_scale"}}
    payload["direct_scale"] = {"real": float(result["direct_scale"].real), "imag": float(result["direct_scale"].imag)}
    payload["gep_scale"] = {"real": float(result["gep_scale"].real), "imag": float(result["gep_scale"].imag)}
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-plan", type=Path, default=Path("assets/pinn_subsonic/joint_ci_mode_atlas_v2/training_plan.tsv"))
    parser.add_argument("--canonical-csv", type=Path, default=Path("assets/pinn_subsonic/joint_ci_mode_final_assets_v3/data/validation_pointwise_canonical.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets/pinn_subsonic/joint_ci_mode_final_assets_v3"))
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--mode-alpha", type=float, default=0.5)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=None)
    parser.add_argument("--n-alpha", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-cut", action="store_true")
    args = parser.parse_args()

    training_path = args.training_plan if args.training_plan.is_absolute() else ROOT / args.training_plan
    canonical_path = args.canonical_csv if args.canonical_csv.is_absolute() else ROOT / args.canonical_csv
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if not training_path.is_file():
        raise FileNotFoundError(training_path)
    if not canonical_path.is_file():
        raise FileNotFoundError(canonical_path)

    plan = load_training_plan(training_path)
    device = torch.device(args.device)
    cache: dict[str, tuple] = {}
    data_dir = output_dir / "data" / "article_M050_alpha0500"
    fig_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exact modal evaluation: alpha={args.mode_alpha}, M={args.mach}", flush=True)
    mode = evaluate_point(args.mode_alpha, args.mach, plan, cache, device, True, True)
    mode_csv = data_dir / f"mode_comparison_M{args.mach:.3f}_alpha{args.mode_alpha:.3f}.csv"
    write_mode_csv(mode, mode_csv)
    (data_dir / f"mode_comparison_M{args.mach:.3f}_alpha{args.mode_alpha:.3f}.json").write_text(json.dumps(serializable_metadata(mode), indent=2, sort_keys=True, default=str) + "\n")
    mode_stem = fig_dir / f"Fig_subsonic_mode_comparison_M{int(round(100*args.mach)):03d}_alpha{int(round(1000*args.mode_alpha)):04d}_classical_PINN_GEP"
    plot_mode(mode, mode_stem)
    print(mode_csv)
    print(mode_stem.with_suffix(".pdf"))

    if args.skip_cut:
        return

    alphas = alpha_grid(args.mach, args.alpha_min, args.alpha_max, args.n_alpha, args.mode_alpha)
    cut_csv = data_dir / f"spectral_cut_M{args.mach:.3f}.csv"
    cut = compute_cut(alphas, args.mach, plan, cache, device, cut_csv)
    heatmap = load_heatmap(canonical_path)
    spectral_stem = fig_dir / f"Fig_subsonic_spectral_cut_and_error_heatmap_M{int(round(100*args.mach)):03d}"
    plot_cut_heatmap(cut, heatmap, args.mach, spectral_stem)
    print(cut_csv)
    print(spectral_stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
