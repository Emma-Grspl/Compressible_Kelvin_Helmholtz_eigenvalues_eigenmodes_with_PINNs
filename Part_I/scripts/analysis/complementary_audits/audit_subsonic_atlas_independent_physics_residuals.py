#!/usr/bin/env python3
"""
P1-C — Independent physics-residual audit of the final subsonic joint atlas.

Purpose
-------
Evaluate the 49 already-trained joint spectral--modal checkpoints on completely
independent Sobol points, without retraining and without using classical modal
fields.

For each chart this script audits:
    - local compatibility residual
    - local first-order ODE residual
    - outgoing boundary residuals
    - gauge condition at y=0
    - center q/Q condition
    - parity constraints

The implementation mirrors the actual training equations:
    pq_legacy / pq_etaaware:
        r_compat = p_y - q
        r_ode    = q_y - A q - B p

    pQscaled:
        Q = q / alpha
        r_compat = p_y / alpha - Q
        r_ode    = Q_y - A Q - alpha (1 - M^2 D^2) p

No classical eigenfunction is used anywhere in this audit.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.dev.train_subsonic_joint_spectral_modal_chart import (
    CiAtlasNet,
    infer_field_family,
)


MODULES = {
    "pq_legacy":
        "scripts.dev.train_subsonic_seedGEP_pq2d_continuous_M_alpha",
    "pq_etaaware":
        "scripts.dev.train_subsonic_seedGEP_pq2d_continuous_M_alpha_etaaware",
    "pQscaled":
        "scripts.dev.train_subsonic_seedGEP_pQscaled2d_continuous_M_alpha",
}


def call_supported(function, **kwargs):
    parameters = inspect.signature(function).parameters
    return function(**{k: v for k, v in kwargs.items() if k in parameters})


def first_linear_input_dimension(state_dict):
    candidates = []
    for key, tensor in state_dict.items():
        if key.endswith(".weight") and tensor.ndim == 2:
            pieces = key.split(".")
            idx = next((int(p) for p in pieces if p.isdigit()), 10**9)
            candidates.append((idx, int(tensor.shape[1])))
    return min(candidates)[1] if candidates else None


def alpha_from_eta(eta, mach):
    return eta * torch.sqrt(torch.clamp(1.0 - mach**2, min=1.0e-14))


def sobol_box(n, lo, hi, dim, seed, device, dtype):
    engine = torch.quasirandom.SobolEngine(
        dimension=dim,
        scramble=True,
        seed=int(seed),
    )
    u = engine.draw(n).to(device=device, dtype=dtype)
    lo_t = torch.tensor(lo, device=device, dtype=dtype).reshape(1, dim)
    hi_t = torch.tensor(hi, device=device, dtype=dtype).reshape(1, dim)
    return lo_t + (hi_t - lo_t) * u


def complex_grad(z, x):
    ones = torch.ones_like(x)
    zr = torch.autograd.grad(
        z.real, x, grad_outputs=ones,
        create_graph=False, retain_graph=True
    )[0]
    zi = torch.autograd.grad(
        z.imag, x, grad_outputs=ones,
        create_graph=False, retain_graph=True
    )[0]
    return torch.complex(zr, zi)


def sqrt_outgoing(z):
    # Match the branch choice used by the standard atlas modules.
    root = torch.sqrt(z)
    flip = root.real < 0
    root = torch.where(flip, -root, root)
    return root


def build_models(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = dict(ckpt.get("args", {}))

    family = args.get("field_family_resolved")
    if not family:
        requested = str(args.get("field_family", "auto"))
        family = infer_field_family(
            requested,
            Path(checkpoint_path),
            ckpt,
        )

    if family not in MODULES:
        raise RuntimeError(f"Unsupported field family: {family}")

    mod = importlib.import_module(MODULES[family])

    mach_min = float(args["mach_min"])
    mach_max = float(args["mach_max"])
    eta_min = float(args["eta_min"])
    eta_max = float(args["eta_max"])
    ymax = float(args["ymax"])
    central_ymax = float(args["central_ymax"])
    sym_ymax = float(args["sym_ymax"])
    width = int(args["width"])
    depth = int(args["depth"])
    n_freq = int(args["n_freq"])

    alpha_corners = [
        eta * math.sqrt(max(1.0 - mach**2, 1.0e-14))
        for eta in (eta_min, eta_max)
        for mach in (mach_min, mach_max)
    ]
    alpha_min = min(alpha_corners)
    alpha_max = max(alpha_corners)

    field = call_supported(
        mod.FieldPQNet,
        ymax=ymax,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        mach_min=mach_min,
        mach_max=mach_max,
        eta_min=eta_min,
        eta_max=eta_max,
        width=width,
        depth=depth,
        n_freq=n_freq,
    ).to(device=device, dtype=torch.float64)

    field.load_state_dict(ckpt["field_state_dict"], strict=True)

    anchor_df_raw = ckpt.get("anchor_df", {})
    anchor_df = pd.DataFrame(anchor_df_raw)
    if len(anchor_df) == 0:
        ci_init = 0.1
    else:
        ci_init = float(anchor_df["ci"].mean())

    ci_net = CiAtlasNet(
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


def independent_interior_points(
    n,
    ymax,
    central_ymax,
    mach_min,
    mach_max,
    eta_min,
    eta_max,
    seed,
    device,
):
    """
    Preserve the training support structure but not its random points:
    half on the full y-domain, half on the central y-domain.
    """
    dtype = torch.float64
    n_full = n // 2
    n_center = n - n_full

    full = sobol_box(
        n_full,
        [-ymax, mach_min, eta_min],
        [ymax, mach_max, eta_max],
        3, seed, device, dtype,
    )
    center = sobol_box(
        n_center,
        [-central_ymax, mach_min, eta_min],
        [central_ymax, mach_max, eta_max],
        3, seed + 1, device, dtype,
    )

    pts = torch.cat([full, center], dim=0)
    y = pts[:, 0:1].clone().detach().requires_grad_(True)
    mach = pts[:, 1:2]
    eta = pts[:, 2:3]
    alpha = alpha_from_eta(eta, mach)

    region = np.array(
        ["full"] * n_full + ["central"] * n_center,
        dtype=object,
    )
    return y, mach, eta, alpha, region


def independent_parameter_points(
    n,
    mach_min,
    mach_max,
    eta_min,
    eta_max,
    seed,
    device,
):
    pts = sobol_box(
        n,
        [mach_min, eta_min],
        [mach_max, eta_max],
        2, seed, device, torch.float64,
    )
    mach = pts[:, 0:1]
    eta = pts[:, 1:2]
    alpha = alpha_from_eta(eta, mach)
    return mach, eta, alpha


def local_residuals(family, field, ci_net, y, alpha, mach):
    p, q_or_Q = field(y, alpha, mach)
    ci = ci_net(alpha, mach)
    c = torch.complex(torch.zeros_like(ci), ci)

    U = torch.tanh(y)
    Uy = 1.0 - U**2
    Uc = torch.complex(U, torch.zeros_like(U))
    Uyc = torch.complex(Uy, torch.zeros_like(Uy))
    D = Uc - c

    dp = complex_grad(p, y)
    dq = complex_grad(q_or_Q, y)

    if family == "pQscaled":
        alpha_c = torch.complex(alpha, torch.zeros_like(alpha))
        r_compat = dp / alpha_c - q_or_Q
        r_ode = (
            dq
            - (2.0 * Uyc / D) * q_or_Q
            - alpha_c * (1.0 - mach**2 * D**2) * p
        )
        physical_q = alpha_c * q_or_Q
    else:
        A = 2.0 * Uyc / D
        B = alpha**2 * (1.0 - mach**2 * D**2)
        r_compat = dp - q_or_Q
        r_ode = dq - A * q_or_Q - B * p
        physical_q = q_or_Q

    return p, q_or_Q, physical_q, ci, r_compat, r_ode


def constraint_residuals(
    family,
    mod,
    field,
    ci_net,
    mach,
    eta,
    alpha,
    ymax,
):
    ci = ci_net(alpha, mach)
    c = torch.complex(torch.zeros_like(ci), ci)
    alpha_c = torch.complex(alpha, torch.zeros_like(alpha))

    y0 = torch.zeros_like(alpha)
    p0, q0 = field(y0, alpha, mach)

    gauge_re = p0.real - 1.0
    gauge_im = p0.imag

    # The training q_center term uses real(q) or real(Q) for pQscaled.
    q_center = q0.real

    yL = torch.full_like(alpha, -ymax)
    yR = torch.full_like(alpha, ymax)
    pL, qL = field(yL, alpha, mach)
    pR, qR = field(yR, alpha, mach)

    D_left = torch.complex(
        -torch.ones_like(alpha), torch.zeros_like(alpha)
    ) - c
    D_right = torch.complex(
        torch.ones_like(alpha), torch.zeros_like(alpha)
    ) - c

    if family == "pQscaled":
        lam_left = alpha_c * torch.sqrt(
            1.0 - mach**2 * D_left**2
        )
        lam_right = alpha_c * torch.sqrt(
            1.0 - mach**2 * D_right**2
        )
        bc_left = qL - (lam_left / alpha_c) * pL
        bc_right = qR + (lam_right / alpha_c) * pR
    else:
        sqrt_fn = getattr(mod, "torch_sqrt_pos", sqrt_outgoing)
        lam_left = alpha * sqrt_fn(
            1.0 - mach**2 * D_left**2
        )
        lam_right = alpha * sqrt_fn(
            1.0 - mach**2 * D_right**2
        )
        bc_left = qL - lam_left * pL
        bc_right = qR + lam_right * pR

    return {
        "ci": ci,
        "gauge_re": gauge_re,
        "gauge_im": gauge_im,
        "gauge_abs": torch.sqrt(gauge_re**2 + gauge_im**2),
        "q_center": q_center,
        "q_center_abs": torch.abs(q_center),
        "bc_left": bc_left,
        "bc_right": bc_right,
        "bc_left_abs": torch.abs(bc_left),
        "bc_right_abs": torch.abs(bc_right),
    }


def parity_residuals(
    family,
    field,
    n,
    sym_ymax,
    mach_min,
    mach_max,
    eta_min,
    eta_max,
    seed,
    device,
):
    pts = sobol_box(
        n,
        [0.0, mach_min, eta_min],
        [sym_ymax, mach_max, eta_max],
        3, seed, device, torch.float64,
    )
    y = pts[:, 0:1]
    mach = pts[:, 1:2]
    eta = pts[:, 2:3]
    alpha = alpha_from_eta(eta, mach)

    p_plus, q_plus = field(y, alpha, mach)
    p_minus, q_minus = field(-y, alpha, mach)

    return {
        "y": y,
        "mach": mach,
        "eta": eta,
        "alpha": alpha,
        "p_re_even": p_minus.real - p_plus.real,
        "p_im_odd": p_minus.imag + p_plus.imag,
        "q_re_odd": q_minus.real + q_plus.real,
        "q_im_even": q_minus.imag - q_plus.imag,
    }


def tensor_np(x):
    return x.detach().cpu().numpy().reshape(-1)


def quantiles(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def summarize_chart(chart_id, family, point_df, constraint_df, parity_df):
    rows = []

    metrics = {
        "compat_abs": point_df["compat_abs"].values,
        "ode_abs": point_df["ode_abs"].values,
        "bc_left_abs": constraint_df["bc_left_abs"].values,
        "bc_right_abs": constraint_df["bc_right_abs"].values,
        "gauge_abs": constraint_df["gauge_abs"].values,
        "q_center_abs": constraint_df["q_center_abs"].values,
        "parity_p_re_even_abs": np.abs(parity_df["p_re_even"].values),
        "parity_p_im_odd_abs": np.abs(parity_df["p_im_odd"].values),
        "parity_q_re_odd_abs": np.abs(parity_df["q_re_odd"].values),
        "parity_q_im_even_abs": np.abs(parity_df["q_im_even"].values),
    }

    for metric, values in metrics.items():
        row = {
            "chart_id": chart_id,
            "field_family": family,
            "metric": metric,
        }
        row.update(quantiles(values))
        rows.append(row)

    return rows


def audit_chart(
    checkpoint_path,
    n_interior,
    n_constraints,
    n_parity,
    seed,
    device,
):
    chart_id = checkpoint_path.parent.name

    ckpt, args, family, mod, field, ci_net = build_models(
        checkpoint_path, device
    )

    mach_min = float(args["mach_min"])
    mach_max = float(args["mach_max"])
    eta_min = float(args["eta_min"])
    eta_max = float(args["eta_max"])
    ymax = float(args["ymax"])
    central_ymax = float(args["central_ymax"])
    sym_ymax = float(args["sym_ymax"])

    y, mach, eta, alpha, region = independent_interior_points(
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

    p, qraw, qphys, ci, r_compat, r_ode = local_residuals(
        family, field, ci_net, y, alpha, mach
    )

    point_df = pd.DataFrame({
        "chart_id": chart_id,
        "field_family": family,
        "sample_region": region,
        "Mach": tensor_np(mach),
        "eta": tensor_np(eta),
        "alpha": tensor_np(alpha),
        "y": tensor_np(y),
        "ci": tensor_np(ci),
        "compat_real": tensor_np(r_compat.real),
        "compat_imag": tensor_np(r_compat.imag),
        "compat_abs": tensor_np(torch.abs(r_compat)),
        "ode_real": tensor_np(r_ode.real),
        "ode_imag": tensor_np(r_ode.imag),
        "ode_abs": tensor_np(torch.abs(r_ode)),
        "p_abs": tensor_np(torch.abs(p)),
        "qraw_abs": tensor_np(torch.abs(qraw)),
        "qphysical_abs": tensor_np(torch.abs(qphys)),
    })

    cmach, ceta, calpha = independent_parameter_points(
        n_constraints,
        mach_min,
        mach_max,
        eta_min,
        eta_max,
        seed + 101,
        device,
    )

    cons = constraint_residuals(
        family,
        mod,
        field,
        ci_net,
        cmach,
        ceta,
        calpha,
        ymax,
    )

    constraint_df = pd.DataFrame({
        "chart_id": chart_id,
        "field_family": family,
        "Mach": tensor_np(cmach),
        "eta": tensor_np(ceta),
        "alpha": tensor_np(calpha),
        "ci": tensor_np(cons["ci"]),
        "bc_left_real": tensor_np(cons["bc_left"].real),
        "bc_left_imag": tensor_np(cons["bc_left"].imag),
        "bc_left_abs": tensor_np(cons["bc_left_abs"]),
        "bc_right_real": tensor_np(cons["bc_right"].real),
        "bc_right_imag": tensor_np(cons["bc_right"].imag),
        "bc_right_abs": tensor_np(cons["bc_right_abs"]),
        "gauge_re": tensor_np(cons["gauge_re"]),
        "gauge_im": tensor_np(cons["gauge_im"]),
        "gauge_abs": tensor_np(cons["gauge_abs"]),
        "q_center": tensor_np(cons["q_center"]),
        "q_center_abs": tensor_np(cons["q_center_abs"]),
    })

    par = parity_residuals(
        family,
        field,
        n_parity,
        sym_ymax,
        mach_min,
        mach_max,
        eta_min,
        eta_max,
        seed + 202,
        device,
    )

    parity_df = pd.DataFrame({
        "chart_id": chart_id,
        "field_family": family,
        "Mach": tensor_np(par["mach"]),
        "eta": tensor_np(par["eta"]),
        "alpha": tensor_np(par["alpha"]),
        "y": tensor_np(par["y"]),
        "p_re_even": tensor_np(par["p_re_even"]),
        "p_im_odd": tensor_np(par["p_im_odd"]),
        "q_re_odd": tensor_np(par["q_re_odd"]),
        "q_im_even": tensor_np(par["q_im_even"]),
    })

    summary_rows = summarize_chart(
        chart_id, family, point_df, constraint_df, parity_df
    )

    chart_meta = {
        "chart_id": chart_id,
        "field_family": family,
        "mach_min": mach_min,
        "mach_max": mach_max,
        "eta_min": eta_min,
        "eta_max": eta_max,
        "ymax": ymax,
        "central_ymax": central_ymax,
        "sym_ymax": sym_ymax,
        "n_interior": n_interior,
        "n_constraints": n_constraints,
        "n_parity": n_parity,
        "best_epoch": ckpt.get("best_epoch"),
        "best_loss": ckpt.get("best_loss"),
    }

    return point_df, constraint_df, parity_df, summary_rows, chart_meta


def build_global_summary(points_df, constraints_df, parity_df):
    """Pooled statistics over every independent audit point."""
    metrics = {
        "compat_abs": points_df["compat_abs"].to_numpy(dtype=float),
        "ode_abs": points_df["ode_abs"].to_numpy(dtype=float),
        "bc_left_abs": constraints_df["bc_left_abs"].to_numpy(dtype=float),
        "bc_right_abs": constraints_df["bc_right_abs"].to_numpy(dtype=float),
        "gauge_abs": constraints_df["gauge_abs"].to_numpy(dtype=float),
        "q_center_abs": constraints_df["q_center_abs"].to_numpy(dtype=float),
        "parity_p_re_even_abs": np.abs(parity_df["p_re_even"].to_numpy(dtype=float)),
        "parity_p_im_odd_abs": np.abs(parity_df["p_im_odd"].to_numpy(dtype=float)),
        "parity_q_re_odd_abs": np.abs(parity_df["q_re_odd"].to_numpy(dtype=float)),
        "parity_q_im_even_abs": np.abs(parity_df["q_im_even"].to_numpy(dtype=float)),
    }

    rows = []
    for metric, values in metrics.items():
        row = {"metric": metric, "aggregation": "pooled_points"}
        row.update(quantiles(values))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atlas-root",
        type=Path,
        default=Path("assets/pinn_subsonic/joint_ci_mode_atlas_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/pinn_subsonic/article_work/p1c_independent_physics_audit"),
    )
    parser.add_argument("--n-interior", type=int, default=4096)
    parser.add_argument("--n-constraints", type=int, default=1024)
    parser.add_argument("--n-parity", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chart", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(args.atlas_root.glob("*/model_best.pt"))
    if args.chart:
        checkpoints = [
            p for p in checkpoints if p.parent.name == args.chart
        ]

    if not checkpoints:
        raise RuntimeError("No matching model_best.pt found.")

    print("=" * 100)
    print("P1-C INDEPENDENT PHYSICS RESIDUAL AUDIT")
    print("=" * 100)
    print("atlas root :", args.atlas_root)
    print("charts     :", len(checkpoints))
    print("device     :", device)
    print("interior   :", args.n_interior)
    print("constraints:", args.n_constraints)
    print("parity     :", args.n_parity)
    print("seed       :", args.seed)
    print()

    all_points = []
    all_constraints = []
    all_parity = []
    all_summary = []
    all_meta = []

    for idx, checkpoint_path in enumerate(checkpoints):
        chart_id = checkpoint_path.parent.name
        chart_seed = args.seed + 1000 * idx

        print(
            f"[{idx+1:02d}/{len(checkpoints):02d}] "
            f"{chart_id} | seed={chart_seed}"
        )

        point_df, constraint_df, parity_df, summary_rows, meta = audit_chart(
            checkpoint_path=checkpoint_path,
            n_interior=args.n_interior,
            n_constraints=args.n_constraints,
            n_parity=args.n_parity,
            seed=chart_seed,
            device=device,
        )

        all_points.append(point_df)
        all_constraints.append(constraint_df)
        all_parity.append(parity_df)
        all_summary.extend(summary_rows)
        all_meta.append(meta)

    points_df = pd.concat(all_points, ignore_index=True)
    constraints_df = pd.concat(all_constraints, ignore_index=True)
    parity_df = pd.concat(all_parity, ignore_index=True)
    summary_df = pd.DataFrame(all_summary)
    meta_df = pd.DataFrame(all_meta)
    global_df = build_global_summary(
        points_df, constraints_df, parity_df
    )

    points_df.to_csv(
        args.output_dir / "p1c_pointwise_interior_residuals.csv",
        index=False,
    )
    constraints_df.to_csv(
        args.output_dir / "p1c_pointwise_constraints.csv",
        index=False,
    )
    parity_df.to_csv(
        args.output_dir / "p1c_pointwise_parity.csv",
        index=False,
    )
    summary_df.to_csv(
        args.output_dir / "p1c_summary_by_chart.csv",
        index=False,
    )
    global_df.to_csv(
        args.output_dir / "p1c_summary_global.csv",
        index=False,
    )
    meta_df.to_csv(
        args.output_dir / "p1c_chart_metadata.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(args.output_dir)
    print()
    print("Family counts:")
    print(meta_df["field_family"].value_counts().to_string())
    print()
    print("Worst charts by ODE P95:")
    ode = summary_df[summary_df["metric"] == "ode_abs"].copy()
    print(
        ode.sort_values("p95", ascending=False)[
            ["chart_id", "field_family", "median", "p95", "p99", "max"]
        ].head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()
