#!/usr/bin/env python3
"""
Chart-wise, active-region normalization of the P1-C physical residuals.

This is a companion to:
    scripts/article/audit_subsonic_atlas_independent_physics_residuals.py

It does NOT retrain any model and does NOT use classical eigenfunctions.
It reloads the same final joint-atlas checkpoints, draws the same independent
Sobol interior audit points (same default seed and chart ordering), converts all
modal families to the common physical (p, q_p) representation, and computes
one normalized residual value per chart.

For each chart k, on the active modal region Omega_active^k defined by
    sqrt(|p|^2 + |q_p|^2) >= active_fraction * max sqrt(|p|^2 + |q_p|^2),
we compute

    E_ode = RMS(R_ode) /
            [RMS(q_p') + RMS(P q_p) + RMS(alpha^2 R p)]

with
    R_ode = q_p' + P q_p - alpha^2 R p,
    P = -2 U'/(U-c),
    R = 1 - M^2 (U-c)^2,

and

    E_compat = RMS(p' - q_p) / [RMS(p') + RMS(q_p)].

The denominator is therefore a chart-wise scale built from norms of the
individual equation terms, rather than a pointwise ratio that can become ill
conditioned when all terms simultaneously approach zero.
"""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Reuse the exact P1-C checkpoint loader, Sobol sampler, and complex derivative.
import scripts.article.audit_subsonic_atlas_independent_physics_residuals as p1c

complex_grad = p1c.complex_grad
independent_interior_points = p1c.independent_interior_points


def torch_load_checkpoint(path: Path, device: torch.device):
    """Load full training checkpoints on both old and new PyTorch releases."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_models(checkpoint_path: Path, device: torch.device):
    """Local-safe equivalent of the original P1-C model loader."""
    ckpt = torch_load_checkpoint(checkpoint_path, device)
    args = dict(ckpt.get("args", {}))

    family = args.get("field_family_resolved")
    if not family:
        requested = str(args.get("field_family", "auto"))
        family = p1c.infer_field_family(
            requested, Path(checkpoint_path), ckpt
        )

    if family not in p1c.MODULES:
        raise RuntimeError(f"Unsupported field family: {family}")

    import importlib
    mod = importlib.import_module(p1c.MODULES[family])

    mach_min = float(args["mach_min"])
    mach_max = float(args["mach_max"])
    eta_min = float(args["eta_min"])
    eta_max = float(args["eta_max"])
    ymax = float(args["ymax"])
    width = int(args["width"])
    depth = int(args["depth"])
    n_freq = int(args["n_freq"])

    alpha_corners = [
        eta * math.sqrt(max(1.0 - mach**2, 1.0e-14))
        for eta in (eta_min, eta_max)
        for mach in (mach_min, mach_max)
    ]

    field = p1c.call_supported(
        mod.FieldPQNet,
        ymax=ymax,
        alpha_min=min(alpha_corners),
        alpha_max=max(alpha_corners),
        mach_min=mach_min,
        mach_max=mach_max,
        eta_min=eta_min,
        eta_max=eta_max,
        width=width,
        depth=depth,
        n_freq=n_freq,
    ).to(device=device, dtype=torch.float64)
    field.load_state_dict(ckpt["field_state_dict"], strict=True)

    anchor_df = pd.DataFrame(ckpt.get("anchor_df", {}))
    ci_init = 0.1 if anchor_df.empty else float(anchor_df["ci"].mean())

    ci_net = p1c.CiAtlasNet(
        mach_min=mach_min,
        mach_max=mach_max,
        eta_min=eta_min,
        eta_max=eta_max,
        ci_init=ci_init,
        width=int(args.get("ci_width", 96)),
        depth=int(args.get("ci_depth", 3)),
    ).to(device=device, dtype=torch.float64)
    ci_net.load_state_dict(ckpt["ci_state_dict"], strict=True)

    field.eval()
    ci_net.eval()
    return ckpt, args, family, mod, field, ci_net


def rms_complex(z: torch.Tensor) -> torch.Tensor:
    """RMS magnitude of a real or complex tensor."""
    return torch.sqrt(torch.mean(torch.abs(z) ** 2))


def q95(values: pd.Series) -> float:
    return float(values.quantile(0.95))


def chart_statistics(
    checkpoint_path: Path,
    *,
    n_interior: int,
    seed: int,
    active_fraction: float,
    device: torch.device,
) -> dict[str, float | int | str]:
    chart_id = checkpoint_path.parent.name

    ckpt, args, family, _mod, field, ci_net = build_models(
        checkpoint_path, device
    )

    mach_min = float(args["mach_min"])
    mach_max = float(args["mach_max"])
    eta_min = float(args["eta_min"])
    eta_max = float(args["eta_max"])
    ymax = float(args["ymax"])
    central_ymax = float(args["central_ymax"])

    # Same independent Sobol construction as the original P1-C audit.
    y, mach, eta, alpha, _region = independent_interior_points(
        n=n_interior,
        ymax=ymax,
        central_ymax=central_ymax,
        mach_min=mach_min,
        mach_max=mach_max,
        eta_min=eta_min,
        eta_max=eta_max,
        seed=seed,
        device=device,
    )

    p, q_or_Q = field(y, alpha, mach)
    ci = ci_net(alpha, mach)
    c = torch.complex(torch.zeros_like(ci), ci)

    U = torch.tanh(y)
    Uy = 1.0 - U**2
    Uc = torch.complex(U, torch.zeros_like(U))
    Uyc = torch.complex(Uy, torch.zeros_like(Uy))
    D = Uc - c

    dp = complex_grad(p, y)
    dq_raw = complex_grad(q_or_Q, y)

    alpha_c = torch.complex(alpha, torch.zeros_like(alpha))

    # Convert every family to the same PHYSICAL q_p = dp/dy representation.
    if family == "pQscaled":
        q_phys = alpha_c * q_or_Q
        dq_phys = alpha_c * dq_raw
    else:
        q_phys = q_or_Q
        dq_phys = dq_raw

    # Common physical first-order equation:
    # q_p' + P q_p - alpha^2 R p = 0
    # with P = -2 U'/(U-c), R = 1 - M^2(U-c)^2.
    P = -2.0 * Uyc / D
    R = 1.0 - mach**2 * D**2

    term_qprime = dq_phys
    term_Pq = P * q_phys
    term_Rp = -(alpha_c**2) * R * p
    r_ode = term_qprime + term_Pq + term_Rp

    term_pprime = dp
    term_q = -q_phys
    r_compat = term_pprime + term_q

    # Same active-region definition used in the final P1-C interpretation.
    activity = torch.sqrt(torch.abs(p) ** 2 + torch.abs(q_phys) ** 2)
    threshold = active_fraction * torch.max(activity)
    active = (activity >= threshold).reshape(-1)

    if int(active.sum()) == 0:
        raise RuntimeError(f"No active audit point for chart {chart_id}.")

    def a(z: torch.Tensor) -> torch.Tensor:
        return z.reshape(-1)[active]

    ode_rms = rms_complex(a(r_ode))
    qprime_rms = rms_complex(a(term_qprime))
    Pq_rms = rms_complex(a(term_Pq))
    Rp_rms = rms_complex(a(term_Rp))
    ode_scale = qprime_rms + Pq_rms + Rp_rms
    ode_norm = ode_rms / ode_scale

    compat_rms = rms_complex(a(r_compat))
    pprime_rms = rms_complex(a(term_pprime))
    q_rms = rms_complex(a(q_phys))
    compat_scale = pprime_rms + q_rms
    compat_norm = compat_rms / compat_scale

    # A few internal checks. These are intentionally strict enough to catch
    # formula/loader mistakes but not intended as publication criteria.
    if not torch.isfinite(ode_norm):
        raise RuntimeError(f"Non-finite ODE normalization for {chart_id}.")
    if not torch.isfinite(compat_norm):
        raise RuntimeError(f"Non-finite compatibility normalization for {chart_id}.")

    row = {
        "chart_id": chart_id,
        "field_family": family,
        "n_interior": int(n_interior),
        "n_active": int(active.sum().item()),
        "active_fraction": float(active.double().mean().item()),
        "active_threshold_fraction": float(active_fraction),
        "ode_rms_abs": float(ode_rms.item()),
        "ode_term_qprime_rms": float(qprime_rms.item()),
        "ode_term_Pq_rms": float(Pq_rms.item()),
        "ode_term_alpha2Rp_rms": float(Rp_rms.item()),
        "ode_scale_rms_sum": float(ode_scale.item()),
        "ode_norm": float(ode_norm.item()),
        "compat_rms_abs": float(compat_rms.item()),
        "compat_term_pprime_rms": float(pprime_rms.item()),
        "compat_term_q_rms": float(q_rms.item()),
        "compat_scale_rms_sum": float(compat_scale.item()),
        "compat_norm": float(compat_norm.item()),
        "checkpoint": str(checkpoint_path),
        "best_epoch": ckpt.get("best_epoch"),
    }

    # Explicit cleanup helps on laptops when moving through 49 float64 models.
    del field, ci_net, y, mach, eta, alpha, p, q_or_Q, ci
    gc.collect()

    return row


def summarize_metric(df: pd.DataFrame, metric: str, group: str) -> dict[str, float | int | str]:
    x = df[metric].astype(float)
    return {
        "group": group,
        "metric": metric,
        "n_charts": int(len(x)),
        "median": float(x.median()),
        "p95": q95(x),
        "max": float(x.max()),
    }


def build_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = [
        summarize_metric(df, "ode_norm", "all_charts"),
        summarize_metric(df, "compat_norm", "all_charts"),
    ]
    overall = pd.DataFrame(overall_rows)

    family_rows: list[dict[str, float | int | str]] = []
    for family, g in df.groupby("field_family", sort=True):
        family_rows.append(summarize_metric(g, "ode_norm", str(family)))
        family_rows.append(summarize_metric(g, "compat_norm", str(family)))
    by_family = pd.DataFrame(family_rows)
    return overall, by_family


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atlas-root",
        type=Path,
        default=Path("assets/pinn_subsonic/joint_ci_mode_atlas_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "assets/pinn_subsonic/article_work/"
            "p1c_chartwise_active_normalized_residuals"
        ),
    )
    parser.add_argument("--n-interior", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--active-fraction", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chart", default=None)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional torch CPU thread count.",
    )
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(max(1, int(args.threads)))

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_checkpoints = sorted(args.atlas_root.glob("*/model_best.pt"))
    indexed_checkpoints = list(enumerate(all_checkpoints))
    if args.chart:
        indexed_checkpoints = [
            (idx, p) for idx, p in indexed_checkpoints
            if p.parent.name == args.chart
        ]

    if not indexed_checkpoints:
        raise RuntimeError(
            "No model_best.pt found under "
            f"{args.atlas_root}. Use the same final atlas root as P1-C."
        )

    print("=" * 100)
    print("P1-C CHART-WISE ACTIVE-REGION NORMALIZED RESIDUALS")
    print("=" * 100)
    print("atlas root     :", args.atlas_root)
    print("charts         :", len(indexed_checkpoints))
    print("device         :", device)
    print("interior/chart :", args.n_interior)
    print("base seed      :", args.seed)
    print("active fraction:", args.active_fraction)
    print()

    rows = []
    for run_idx, (original_idx, checkpoint_path) in enumerate(indexed_checkpoints):
        chart_id = checkpoint_path.parent.name
        # Preserve the exact per-chart seed used by the original full P1-C run,
        # even when --chart is used for a one-chart smoke test.
        chart_seed = args.seed + 1000 * original_idx
        print(
            f"[{run_idx + 1:02d}/{len(indexed_checkpoints):02d}] "
            f"{chart_id} | original_index={original_idx} | seed={chart_seed}"
        )

        row = chart_statistics(
            checkpoint_path,
            n_interior=args.n_interior,
            seed=chart_seed,
            active_fraction=args.active_fraction,
            device=device,
        )
        rows.append(row)
        print(
            "    "
            f"active={100.0 * row['active_fraction']:.1f}% | "
            f"E_ode={row['ode_norm']:.6e} | "
            f"E_compat={row['compat_norm']:.6e}"
        )

    df = pd.DataFrame(rows)
    overall, by_family = build_summaries(df)

    chart_path = args.output_dir / "p1c_chartwise_active_normalized_residuals.csv"
    overall_path = args.output_dir / "p1c_chartwise_active_normalized_summary.csv"
    family_path = args.output_dir / "p1c_chartwise_active_normalized_summary_by_family.csv"

    df.to_csv(chart_path, index=False)
    overall.to_csv(overall_path, index=False)
    by_family.to_csv(family_path, index=False)

    print()
    print("=" * 100)
    print("OVERALL — 49 CHARTS")
    print("=" * 100)
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    print()
    print("=" * 100)
    print("BY MODAL FAMILY")
    print("=" * 100)
    print(by_family.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    print()
    print("Worst ODE-normalized charts:")
    print(
        df.sort_values("ode_norm", ascending=False)[
            [
                "chart_id",
                "field_family",
                "active_fraction",
                "ode_rms_abs",
                "ode_scale_rms_sum",
                "ode_norm",
                "compat_norm",
            ]
        ].head(10).to_string(index=False, float_format=lambda x: f"{x:.6e}")
    )

    print()
    print("Saved:")
    print(" ", chart_path)
    print(" ", overall_path)
    print(" ", family_path)


if __name__ == "__main__":
    main()
