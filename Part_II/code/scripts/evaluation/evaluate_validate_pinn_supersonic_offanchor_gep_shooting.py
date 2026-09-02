#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from classical_solver.gep.dense_gep_notebook_style import (
    NotebookStyleDenseGEPSolver,
)
from classical_solver.supersonic.mstab17_supersonic_solver import (
    Mstab17SupersonicSolver,
)
from scripts.audits.audit_supersonic_shooting_visual_validation_6969b4f1bf import (
    reconstruct_shooting_fields,
)


EPS = 1.0e-30


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int,
        depth: int,
        activation: str,
    ):
        super().__init__()

        activations = {
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }

        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")

        act = activations[activation]

        layers = [nn.Linear(in_dim, width), act()]

        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]

        layers += [nn.Linear(width, out_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def alpha_feature(alpha, alpha_min, alpha_max):
    return (
        2.0 * (alpha - alpha_min)
        / (alpha_max - alpha_min)
        - 1.0
    )


def y_feature(y, transform, scale, ymax):
    if transform == "tanh":
        return torch.tanh(y / scale)

    if transform == "asinh":
        return (
            torch.asinh(y / scale)
            / math.asinh(ymax / scale)
        )

    if transform == "compact":
        return y / (scale + torch.abs(y))

    raise ValueError(f"Unknown y transform: {transform}")


def interpolate_complex(
    y_target: np.ndarray,
    y_source: np.ndarray,
    field_source: np.ndarray,
) -> np.ndarray:
    order = np.argsort(y_source)

    y_sorted = np.asarray(y_source)[order]
    field_sorted = np.asarray(field_source)[order]

    real = np.interp(
        y_target,
        y_sorted,
        np.real(field_sorted),
    )
    imag = np.interp(
        y_target,
        y_sorted,
        np.imag(field_sorted),
    )

    return real + 1j * imag


def spectral_distance(
    cr_a: float,
    ci_a: float,
    cr_b: float,
    ci_b: float,
    ci_weight: float = 2.0,
) -> float:
    return float(
        np.sqrt(
            (cr_a - cr_b) ** 2
            + (
                ci_weight
                * (ci_a - ci_b)
            )
            ** 2
        )
    )


def complex_fit_metrics(
    target_p: np.ndarray,
    candidate_p: np.ndarray,
    *,
    target_q: np.ndarray | None = None,
    candidate_q: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> dict:
    if mask is None:
        mask = np.ones(
            target_p.shape,
            dtype=bool,
        )

    p_target = target_p[mask]
    p_candidate = candidate_p[mask]

    denominator = np.vdot(
        p_candidate,
        p_candidate,
    )

    if abs(denominator) <= EPS:
        return {
            "scale_real": np.nan,
            "scale_imag": np.nan,
            "p_overlap": 0.0,
            "p_rel_after_fit": np.inf,
            "q_rel_after_fit": np.nan,
        }

    # Complex least-squares phase and amplitude alignment.
    scale = (
        np.vdot(
            p_candidate,
            p_target,
        )
        / denominator
    )

    p_aligned = scale * p_candidate

    p_overlap = float(
        abs(
            np.vdot(
                p_target,
                p_candidate,
            )
        )
        / max(
            np.linalg.norm(p_target)
            * np.linalg.norm(p_candidate),
            EPS,
        )
    )

    p_rel = float(
        np.linalg.norm(
            p_aligned - p_target
        )
        / max(
            np.linalg.norm(p_target),
            EPS,
        )
    )

    q_rel = np.nan

    if (
        target_q is not None
        and candidate_q is not None
    ):
        q_target = target_q[mask]
        q_aligned = (
            scale
            * candidate_q[mask]
        )

        q_rel = float(
            np.linalg.norm(
                q_aligned - q_target
            )
            / max(
                np.linalg.norm(q_target),
                EPS,
            )
        )

    return {
        "scale_real": float(
            np.real(scale)
        ),
        "scale_imag": float(
            np.imag(scale)
        ),
        "p_overlap": p_overlap,
        "p_rel_after_fit": p_rel,
        "q_rel_after_fit": q_rel,
    }


class PINNChart:
    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
    ):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        self.config = dict(
            checkpoint["config"]
        )
        self.device = device

        width = int(
            self.config["width"]
        )
        depth = int(
            self.config["depth"]
        )
        activation = str(
            self.config["activation"]
        )

        self.modal_net = MLP(
            2,
            4,
            width,
            depth,
            activation,
        ).to(device)

        self.spectral_net = MLP(
            1,
            2,
            max(64, width // 2),
            max(3, depth - 1),
            activation,
        ).to(device)

        self.modal_net.load_state_dict(
            checkpoint["modal_net"]
        )
        self.spectral_net.load_state_dict(
            checkpoint["spectral_net"]
        )

        self.modal_net.eval()
        self.spectral_net.eval()

        normalization = self.config[
            "normalization"
        ]

        def tensor(value):
            return torch.tensor(
                value,
                dtype=torch.float32,
                device=device,
            )

        self.x_mean = tensor(
            normalization["x_mean"]
        )
        self.x_std = tensor(
            normalization["x_std"]
        )
        self.modal_mean = tensor(
            normalization["modal_mean"]
        )
        self.modal_std = tensor(
            normalization["modal_std"]
        )
        self.spec_mean = tensor(
            normalization["spec_mean"]
        )
        self.spec_std = tensor(
            normalization["spec_std"]
        )

        self.alpha_min = float(
            self.config["alpha_min"]
        )
        self.alpha_max = float(
            self.config["alpha_max"]
        )

        self.y_scale = float(
            self.config.get(
                "y_scale",
                10.0,
            )
        )
        self.y_transform = str(
            self.config.get(
                "y_transform",
                "asinh",
            )
        )

        dataset_path = Path(
            self.config["dataset"]
        )
        dataset = np.load(
            dataset_path,
            allow_pickle=True,
        )

        self.Mach = float(
            self.config.get(
                "Mach_fixed",
                np.asarray(
                    dataset["Mach_fixed"]
                ).item(),
            )
        )

        self.ymax_training = max(
            float(
                np.max(
                    np.abs(
                        dataset["y"]
                    )
                )
            ),
            1.0,
        )

        self.anchor_alphas = np.unique(
            np.asarray(
                dataset["alpha_anchors"],
                dtype=float,
            )
        )

    def predict(
        self,
        alpha_value: float,
        y_np: np.ndarray,
    ) -> dict:
        y = torch.tensor(
            y_np,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        alpha = torch.full_like(
            y,
            float(alpha_value),
        )

        features = torch.stack(
            [
                y_feature(
                    y,
                    self.y_transform,
                    self.y_scale,
                    self.ymax_training,
                ),
                alpha_feature(
                    alpha,
                    self.alpha_min,
                    self.alpha_max,
                ),
            ],
            dim=1,
        )

        x_normalized = (
            features - self.x_mean
        ) / self.x_std

        modal_normalized = (
            self.modal_net(
                x_normalized
            )
        )

        modal = (
            modal_normalized
            * self.modal_std
            + self.modal_mean
        )

        p_real = modal[:, 0]
        p_imag = modal[:, 1]
        q_real = modal[:, 2]
        q_imag = modal[:, 3]

        def derivative(value):
            return torch.autograd.grad(
                value.sum(),
                y,
                create_graph=False,
                retain_graph=True,
            )[0]

        dp_real = derivative(p_real)
        dp_imag = derivative(p_imag)
        dq_real = derivative(q_real)
        dq_imag = derivative(q_imag)

        alpha_scalar = torch.tensor(
            [[
                alpha_feature(
                    torch.tensor(
                        float(alpha_value),
                        device=self.device,
                    ),
                    self.alpha_min,
                    self.alpha_max,
                )
            ]],
            dtype=torch.float32,
            device=self.device,
        )

        alpha_scalar_normalized = (
            alpha_scalar
            - self.x_mean[1:2]
        ) / self.x_std[1:2]

        with torch.no_grad():
            spectral = (
                self.spectral_net(
                    alpha_scalar_normalized
                )
                * self.spec_std
                + self.spec_mean
            )

        cr = float(
            spectral[0, 0].cpu()
        )
        ci = float(
            spectral[0, 1].cpu()
        )

        p = (
            p_real.detach().cpu().numpy()
            + 1j
            * p_imag.detach().cpu().numpy()
        )
        q = (
            q_real.detach().cpu().numpy()
            + 1j
            * q_imag.detach().cpu().numpy()
        )
        p_y = (
            dp_real.detach().cpu().numpy()
            + 1j
            * dp_imag.detach().cpu().numpy()
        )
        q_y = (
            dq_real.detach().cpu().numpy()
            + 1j
            * dq_imag.detach().cpu().numpy()
        )

        c = complex(cr, ci)
        U = np.tanh(y_np)
        U_y = 1.0 - U**2
        D = U - c

        r1 = p_y - q

        shear = (
            2.0
            * U_y
            / D
            * q
        )

        pressure_term = (
            alpha_value**2
            * (
                1.0
                - self.Mach**2
                * D**2
            )
            * p
        )

        r2 = (
            q_y
            - shear
            - pressure_term
        )

        r1_rel = float(
            np.linalg.norm(r1)
            / max(
                np.linalg.norm(q),
                EPS,
            )
        )

        r2_scale = (
            np.linalg.norm(q_y)
            + np.linalg.norm(shear)
            + np.linalg.norm(
                pressure_term
            )
        )

        r2_rel = float(
            np.linalg.norm(r2)
            / max(r2_scale, EPS)
        )

        tail_mask = (
            np.abs(y_np)
            >= 0.9
            * np.max(
                np.abs(y_np)
            )
        )

        tail_ratio = float(
            np.max(
                np.abs(p[tail_mask])
            )
            / max(
                np.max(np.abs(p)),
                EPS,
            )
        )

        return {
            "alpha": float(alpha_value),
            "Mach": self.Mach,
            "cr": cr,
            "ci": ci,
            "omega_i": float(
                alpha_value * ci
            ),
            "p": p,
            "q": q,
            "r1_rel": r1_rel,
            "r2_rel": r2_rel,
            "tail_ratio": tail_ratio,
            "min_abs_U_minus_c": float(
                np.min(np.abs(D))
            ),
        }


def choose_offanchor_alphas(
    anchors: np.ndarray,
    n_cases: int,
) -> np.ndarray:
    midpoints = 0.5 * (
        anchors[:-1]
        + anchors[1:]
    )

    n_cases = min(
        max(1, n_cases),
        len(midpoints),
    )

    indices = np.unique(
        np.rint(
            np.linspace(
                0,
                len(midpoints) - 1,
                n_cases,
            )
        ).astype(int)
    )

    return midpoints[indices]


def run_gep(
    *,
    alpha: float,
    Mach: float,
    target: tuple[float, float],
    n_points: int,
    mapping_kind: str,
    mapping_scale: float,
    xi_max: float,
    ci_weight: float,
):
    solver = NotebookStyleDenseGEPSolver(
        alpha=alpha,
        Mach=Mach,
        n_points=n_points,
        mapping_kind=mapping_kind,
        mapping_scale=mapping_scale,
        xi_max=xi_max,
    )

    mode, selection_source, n_modes = (
        solver.get_nearest_mode_to_target(
            target_guess=target,
            prefer_positive_cr=True,
            ci_weight=ci_weight,
        )
    )

    return solver, mode, selection_source, n_modes


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--n-cases",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--n-y",
        type=int,
        default=2001,
    )

    parser.add_argument(
        "--gep-n",
        type=int,
        default=301,
    )
    parser.add_argument(
        "--gep-mapping-kind",
        choices=["pin", "cubic"],
        default="pin",
    )
    parser.add_argument(
        "--gep-mapping-scale",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--gep-xi-max",
        type=float,
        default=0.98,
    )
    parser.add_argument(
        "--ci-weight",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--match-y",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--use-mapping",
        action="store_true",
    )
    parser.add_argument(
        "--shooting-mapping-scale",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--min-y-limit",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--max-y-limit",
        type=float,
        default=2000.0,
    )
    parser.add_argument(
        "--y-limit-factor",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--cr-half-width",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--ci-half-width",
        type=float,
        default=0.003,
    )
    parser.add_argument(
        "--ci-relative-half-width",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--shooting-max-iter",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--shooting-grid-size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--gep-seed-distance-max",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--gep-overlap-min",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--shooting-seed-distance-max",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--shooting-stage1-mismatch-max",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--shooting-spectral-distance-max",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--shooting-overlap-min",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--shooting-p-rel-max",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--shooting-q-rel-max",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields_dir = (
        args.output_dir / "fields"
    )
    fields_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if (
            args.device == "cuda"
            and torch.cuda.is_available()
        )
        else "cpu"
    )

    chart = PINNChart(
        args.checkpoint,
        device,
    )

    offanchor_alphas = (
        choose_offanchor_alphas(
            chart.anchor_alphas,
            args.n_cases,
        )
    )

    y_seed = np.linspace(
        -args.ymax,
        args.ymax,
        args.n_y,
    )

    core_mask = (
        np.abs(y_seed)
        <= args.ymax
    )

    rows = []

    config = {
        **vars(args),
        "checkpoint": str(
            args.checkpoint
        ),
        "output_dir": str(
            args.output_dir
        ),
        "device_resolved": str(device),
        "Mach_fixed": chart.Mach,
        "anchor_alphas": (
            chart.anchor_alphas.tolist()
        ),
        "offanchor_alphas": (
            offanchor_alphas.tolist()
        ),
        "interpretation": (
            "No pre-existing classical reference is used "
            "at off-anchor alpha. Validation is based on "
            "PINN physics residuals, GEP agreement, and "
            "independent local shooting convergence."
        ),
    }

    (
        args.output_dir
        / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    for case_index, alpha in enumerate(
        offanchor_alphas
    ):
        print(
            f"\n[case {case_index + 1}/"
            f"{len(offanchor_alphas)}] "
            f"alpha={alpha:.8f}"
        )

        row = {
            "case_index": case_index,
            "Mach": chart.Mach,
            "alpha": float(alpha),
            "is_offanchor": True,
            "nearest_anchor_distance": float(
                np.min(
                    np.abs(
                        chart.anchor_alphas
                        - alpha
                    )
                )
            ),
        }

        try:
            pinn = chart.predict(
                float(alpha),
                y_seed,
            )

            row.update(
                {
                    "pinn_cr": pinn["cr"],
                    "pinn_ci": pinn["ci"],
                    "pinn_omega_i": (
                        pinn["omega_i"]
                    ),
                    "pinn_r1_rel": (
                        pinn["r1_rel"]
                    ),
                    "pinn_r2_rel": (
                        pinn["r2_rel"]
                    ),
                    "pinn_tail_ratio": (
                        pinn["tail_ratio"]
                    ),
                    "pinn_min_abs_U_minus_c": (
                        pinn[
                            "min_abs_U_minus_c"
                        ]
                    ),
                }
            )

            print(
                "  PINN "
                f"c={pinn['cr']:.8f}"
                f"+i{pinn['ci']:.8f} "
                f"R1rel={pinn['r1_rel']:.3e} "
                f"R2rel={pinn['r2_rel']:.3e}"
            )

            gep_solver = None
            gep_mode = None
            gep_p_on_seed = None

            try:
                (
                    gep_solver,
                    gep_mode,
                    gep_source,
                    n_modes,
                ) = run_gep(
                    alpha=float(alpha),
                    Mach=chart.Mach,
                    target=(
                        pinn["cr"],
                        pinn["ci"],
                    ),
                    n_points=args.gep_n,
                    mapping_kind=(
                        args.gep_mapping_kind
                    ),
                    mapping_scale=(
                        args.gep_mapping_scale
                    ),
                    xi_max=args.gep_xi_max,
                    ci_weight=args.ci_weight,
                )

                row["gep_selection_source"] = (
                    gep_source
                )
                row["gep_n_finite_modes"] = (
                    n_modes
                )

                if gep_mode is not None:
                    gep_cr = float(
                        gep_mode["cr"]
                    )
                    gep_ci = float(
                        gep_mode["ci"]
                    )

                    gep_distance = (
                        spectral_distance(
                            gep_cr,
                            gep_ci,
                            pinn["cr"],
                            pinn["ci"],
                            args.ci_weight,
                        )
                    )

                    gep_p_native = np.asarray(
                        gep_mode["vector"][
                            2
                            * gep_solver.n_points
                            :
                            3
                            * gep_solver.n_points
                        ],
                        dtype=complex,
                    )

                    gep_p_on_seed = (
                        interpolate_complex(
                            y_seed,
                            gep_solver.y,
                            gep_p_native,
                        )
                    )

                    gep_fit = (
                        complex_fit_metrics(
                            pinn["p"],
                            gep_p_on_seed,
                            mask=core_mask,
                        )
                    )

                    row.update(
                        {
                            "gep_found": True,
                            "gep_cr": gep_cr,
                            "gep_ci": gep_ci,
                            "gep_omega_i": float(
                                alpha * gep_ci
                            ),
                            "gep_pinn_spectral_distance": (
                                gep_distance
                            ),
                            "gep_pinn_p_overlap": (
                                gep_fit[
                                    "p_overlap"
                                ]
                            ),
                            "gep_pinn_p_rel_after_fit": (
                                gep_fit[
                                    "p_rel_after_fit"
                                ]
                            ),
                        }
                    )

                    print(
                        "  GEP  "
                        f"c={gep_cr:.8f}"
                        f"+i{gep_ci:.8f} "
                        f"distance={gep_distance:.3e} "
                        f"overlap={gep_fit['p_overlap']:.4f}"
                    )
                else:
                    row["gep_found"] = False
                    print("  GEP: no finite mode")

            except Exception as gep_error:
                row["gep_found"] = False
                row["gep_error"] = repr(
                    gep_error
                )
                print(
                    "  GEP ERROR:",
                    repr(gep_error),
                )
                traceback.print_exc()

            gep_agrees = bool(
                row.get("gep_found", False)
                and row.get(
                    "gep_pinn_spectral_distance",
                    np.inf,
                )
                <= args.gep_seed_distance_max
                and row.get(
                    "gep_pinn_p_overlap",
                    0.0,
                )
                >= args.gep_overlap_min
            )

            row["gep_agrees_with_pinn"] = (
                gep_agrees
            )

            # Use the GEP center only when it already agrees with
            # both the spectral and modal PINN seed.
            if gep_agrees:
                center_source = "gep"
                center_cr = float(
                    row["gep_cr"]
                )
                center_ci = float(
                    row["gep_ci"]
                )
            else:
                center_source = "pinn"
                center_cr = float(
                    pinn["cr"]
                )
                center_ci = float(
                    pinn["ci"]
                )

            row["shooting_center_source"] = (
                center_source
            )
            row["shooting_center_cr"] = (
                center_cr
            )
            row["shooting_center_ci"] = (
                center_ci
            )

            ci_half_width = max(
                args.ci_half_width,
                args.ci_relative_half_width
                * abs(center_ci),
            )

            cr_min = max(
                0.0,
                center_cr
                - args.cr_half_width,
            )
            cr_max = (
                center_cr
                + args.cr_half_width
            )
            ci_min = max(
                1.0e-6,
                center_ci
                - ci_half_width,
            )
            ci_max = (
                center_ci
                + ci_half_width
            )

            row.update(
                {
                    "shooting_cr_min": cr_min,
                    "shooting_cr_max": cr_max,
                    "shooting_ci_min": ci_min,
                    "shooting_ci_max": ci_max,
                }
            )

            shooting_solver = (
                Mstab17SupersonicSolver(
                    alpha=float(alpha),
                    Mach=chart.Mach,
                    match_y=args.match_y,
                    use_mapping=(
                        args.use_mapping
                    ),
                    mapping_scale=(
                        args.shooting_mapping_scale
                    ),
                    min_y_limit=(
                        args.min_y_limit
                    ),
                    max_y_limit=(
                        args.max_y_limit
                    ),
                    y_limit_factor=(
                        args.y_limit_factor
                    ),
                )
            )

            shooting = shooting_solver.solve(
                cr_min=cr_min,
                cr_max=cr_max,
                ci_min=ci_min,
                ci_max=ci_max,
                max_iter=(
                    args.shooting_max_iter
                ),
                grid_size=(
                    args.shooting_grid_size
                ),
            )

            row.update(
                {
                    "shooting_cr": float(
                        shooting.cr
                    ),
                    "shooting_ci": float(
                        shooting.ci
                    ),
                    "shooting_omega_i": float(
                        shooting.omega_i
                    ),
                    "shooting_stage1_mismatch": float(
                        shooting.stage1_mismatch
                    ),
                    "shooting_stage2_mismatch": float(
                        shooting.stage2_mismatch
                    ),
                    "shooting_spectral_success": bool(
                        shooting.spectral_success
                    ),
                    "shooting_mode_success": bool(
                        shooting.mode_success
                    ),
                    "shooting_success": bool(
                        shooting.success
                    ),
                    "shooting_y_limit": float(
                        shooting.y_limit
                    ),
                    "shooting_ln_p_start_right": float(
                        shooting.ln_p_start_right
                    ),
                }
            )

            shooting_pinn_distance = (
                spectral_distance(
                    shooting.cr,
                    shooting.ci,
                    pinn["cr"],
                    pinn["ci"],
                    args.ci_weight,
                )
            )

            shooting_center_distance = (
                spectral_distance(
                    shooting.cr,
                    shooting.ci,
                    center_cr,
                    center_ci,
                    args.ci_weight,
                )
            )

            row[
                "shooting_pinn_spectral_distance"
            ] = shooting_pinn_distance
            row[
                "shooting_center_spectral_distance"
            ] = shooting_center_distance

            if row.get("gep_found", False):
                row[
                    "shooting_gep_spectral_distance"
                ] = spectral_distance(
                    shooting.cr,
                    shooting.ci,
                    row["gep_cr"],
                    row["gep_ci"],
                    args.ci_weight,
                )

            print(
                "  SHOOT "
                f"c={shooting.cr:.8f}"
                f"+i{shooting.ci:.8f} "
                f"stage1="
                f"{shooting.stage1_mismatch:.3e} "
                f"stage2="
                f"{shooting.stage2_mismatch:.3e} "
                f"success={shooting.success}"
            )

            shooting_fields = (
                reconstruct_shooting_fields(
                    alpha=float(alpha),
                    mach=chart.Mach,
                    cr=float(shooting.cr),
                    ci=float(shooting.ci),
                    ln_p_start_right=float(
                        shooting.ln_p_start_right
                    ),
                    match_y=args.match_y,
                    use_mapping=args.use_mapping,
                    mapping_scale=(
                        args.shooting_mapping_scale
                    ),
                    min_y_limit=(
                        args.min_y_limit
                    ),
                    max_y_limit=(
                        args.max_y_limit
                    ),
                    y_limit_factor=(
                        args.y_limit_factor
                    ),
                )
            )

            shooting_y = np.asarray(
                shooting_fields["y"],
                dtype=float,
            )

            shooting_p = np.asarray(
                shooting_fields["p"],
                dtype=np.complex128,
            )

            # Use the exact first-order pressure variable reconstructed
            # from the Riccati solution: q = p_y = gamma * p.
            if "q" in shooting_fields:
                shooting_q = np.asarray(
                    shooting_fields["q"],
                    dtype=np.complex128,
                )
                shooting_q_source = "returned_q"
            elif "gamma" in shooting_fields:
                shooting_gamma = np.asarray(
                    shooting_fields["gamma"],
                    dtype=np.complex128,
                )
                shooting_q = (
                    shooting_gamma
                    * shooting_p
                )
                shooting_q_source = "gamma_times_p"
            else:
                raise KeyError(
                    "The corrected shooting reconstruction must "
                    "return either q or gamma."
                )

            if not (
                shooting_y.shape
                == shooting_p.shape
                == shooting_q.shape
            ):
                raise RuntimeError(
                    "Inconsistent shooting field shapes: "
                    f"y={shooting_y.shape}, "
                    f"p={shooting_p.shape}, "
                    f"q={shooting_q.shape}."
                )

            finite = (
                np.isfinite(shooting_y)
                & np.isfinite(shooting_p.real)
                & np.isfinite(shooting_p.imag)
                & np.isfinite(shooting_q.real)
                & np.isfinite(shooting_q.imag)
            )

            shooting_y = shooting_y[finite]
            shooting_p = shooting_p[finite]
            shooting_q = shooting_q[finite]

            order = np.argsort(shooting_y)

            shooting_y = shooting_y[order]
            shooting_p = shooting_p[order]
            shooting_q = shooting_q[order]

            unique_y, inverse, counts = np.unique(
                shooting_y,
                return_inverse=True,
                return_counts=True,
            )

            if unique_y.size != shooting_y.size:
                pressure_sum = np.zeros(
                    unique_y.shape,
                    dtype=np.complex128,
                )
                q_sum = np.zeros(
                    unique_y.shape,
                    dtype=np.complex128,
                )

                np.add.at(
                    pressure_sum,
                    inverse,
                    shooting_p,
                )
                np.add.at(
                    q_sum,
                    inverse,
                    shooting_q,
                )

                shooting_p = (
                    pressure_sum / counts
                )
                shooting_q = (
                    q_sum / counts
                )
                shooting_y = unique_y

            if shooting_y.size < 3:
                raise RuntimeError(
                    "Too few shooting points after filtering."
                )

            if np.any(
                np.diff(shooting_y) <= 0.0
            ):
                raise RuntimeError(
                    "The shooting grid is not strictly increasing."
                )

            row["shooting_q_source"] = (
                shooting_q_source
            )

            shooting_p_on_seed = (
                interpolate_complex(
                    y_seed,
                    shooting_y,
                    shooting_p,
                )
            )

            shooting_q_on_seed = (
                interpolate_complex(
                    y_seed,
                    shooting_y,
                    shooting_q,
                )
            )

            shooting_fit = (
                complex_fit_metrics(
                    pinn["p"],
                    shooting_p_on_seed,
                    target_q=pinn["q"],
                    candidate_q=(
                        shooting_q_on_seed
                    ),
                    mask=core_mask,
                )
            )

            row.update(
                {
                    "shooting_pinn_p_overlap": (
                        shooting_fit[
                            "p_overlap"
                        ]
                    ),
                    "shooting_pinn_p_rel_after_fit": (
                        shooting_fit[
                            "p_rel_after_fit"
                        ]
                    ),
                    "shooting_pinn_q_rel_after_fit": (
                        shooting_fit[
                            "q_rel_after_fit"
                        ]
                    ),
                    "shooting_fit_scale_real": (
                        shooting_fit[
                            "scale_real"
                        ]
                    ),
                    "shooting_fit_scale_imag": (
                        shooting_fit[
                            "scale_imag"
                        ]
                    ),
                }
            )

            print(
                "  MODAL "
                f"p_overlap="
                f"{shooting_fit['p_overlap']:.4f} "
                f"p_rel="
                f"{shooting_fit['p_rel_after_fit']:.3e} "
                f"q_rel="
                f"{shooting_fit['q_rel_after_fit']:.3e}"
            )

            # Stage-1 verifies that the PINN spectral seed lies on
            # a local shooting root.
            shooting_stage1_seed_consistent = bool(
                shooting.stage1_mismatch
                <= args.shooting_stage1_mismatch_max
                and shooting_pinn_distance
                <= args.shooting_spectral_distance_max
                and shooting.ci > 0.0
            )

            # Complete modal validation additionally requires the
            # exact amplitude match and agreement of p and q.
            shooting_modal_validated = bool(
                shooting.mode_success
                and shooting_fit[
                    "p_overlap"
                ]
                >= args.shooting_overlap_min
                and shooting_fit[
                    "p_rel_after_fit"
                ]
                <= args.shooting_p_rel_max
                and shooting_fit[
                    "q_rel_after_fit"
                ]
                <= args.shooting_q_rel_max
            )

            pinn_seed_validated = bool(
                shooting.success
                and shooting_stage1_seed_consistent
                and shooting_modal_validated
            )

            full_pipeline_consistent = bool(
                pinn_seed_validated
                and gep_agrees
                and row.get(
                    "shooting_gep_spectral_distance",
                    np.inf,
                )
                <= args.shooting_seed_distance_max
            )

            row[
                "shooting_stage1_seed_consistent"
            ] = shooting_stage1_seed_consistent

            row[
                "shooting_modal_validated"
            ] = shooting_modal_validated

            row[
                "pinn_seed_validated_by_shooting"
            ] = pinn_seed_validated

            row[
                "full_pipeline_consistent"
            ] = full_pipeline_consistent

            if full_pipeline_consistent:
                status = (
                    "pinn_gep_shooting_consistent"
                )
            elif pinn_seed_validated:
                status = (
                    "pinn_shooting_fully_consistent_"
                    "gep_disagrees_or_missing"
                )
            elif (
                shooting_stage1_seed_consistent
                and not shooting.mode_success
            ):
                status = (
                    "shooting_stage1_seed_consistent_"
                    "but_amplitude_match_failed"
                )
            elif shooting_stage1_seed_consistent:
                status = (
                    "shooting_stage1_seed_consistent_"
                    "but_modal_fields_disagree"
                )
            elif shooting.spectral_success:
                status = (
                    "shooting_internal_spectral_success_"
                    "but_pinn_seed_disagrees"
                )
            else:
                status = (
                    "shooting_spectral_failed"
                )

            row["status"] = status

            np.savez_compressed(
                fields_dir
                / (
                    f"case_{case_index:03d}_"
                    f"alpha_{alpha:.8f}.npz"
                ),
                Mach=np.array(chart.Mach),
                alpha=np.array(alpha),
                y_seed=y_seed,
                pinn_cr=np.array(
                    pinn["cr"]
                ),
                pinn_ci=np.array(
                    pinn["ci"]
                ),
                pinn_p_real=np.real(
                    pinn["p"]
                ),
                pinn_p_imag=np.imag(
                    pinn["p"]
                ),
                pinn_q_real=np.real(
                    pinn["q"]
                ),
                pinn_q_imag=np.imag(
                    pinn["q"]
                ),
                shooting_cr=np.array(
                    shooting.cr
                ),
                shooting_ci=np.array(
                    shooting.ci
                ),
                shooting_p_real=np.real(
                    shooting_p_on_seed
                ),
                shooting_p_imag=np.imag(
                    shooting_p_on_seed
                ),
                shooting_q_real=np.real(
                    shooting_q_on_seed
                ),
                shooting_q_imag=np.imag(
                    shooting_q_on_seed
                ),
                gep_cr=np.array(
                    row.get(
                        "gep_cr",
                        np.nan,
                    )
                ),
                gep_ci=np.array(
                    row.get(
                        "gep_ci",
                        np.nan,
                    )
                ),
                gep_p_real=(
                    np.real(
                        gep_p_on_seed
                    )
                    if gep_p_on_seed
                    is not None
                    else np.full_like(
                        y_seed,
                        np.nan,
                    )
                ),
                gep_p_imag=(
                    np.imag(
                        gep_p_on_seed
                    )
                    if gep_p_on_seed
                    is not None
                    else np.full_like(
                        y_seed,
                        np.nan,
                    )
                ),
            )

        except Exception as error:
            row["status"] = "exception"
            row["error"] = repr(error)

            row[
                "pinn_seed_validated_by_shooting"
            ] = False
            row[
                "full_pipeline_consistent"
            ] = False

            print(
                "  CASE ERROR:",
                repr(error),
            )
            traceback.print_exc()

        rows.append(row)

        summary = pd.DataFrame(rows)
        summary.to_csv(
            args.output_dir
            / "offanchor_validation_summary.csv",
            index=False,
        )

    summary = pd.DataFrame(rows)

    counts = {
        str(key): int(value)
        for key, value
        in summary["status"]
        .value_counts()
        .items()
    }

    report = {
        "n_cases": int(len(summary)),
        "n_pinn_seed_validated_by_shooting": int(
            summary[
                "pinn_seed_validated_by_shooting"
            ]
            .fillna(False)
            .sum()
        ),
        "n_full_pipeline_consistent": int(
            summary[
                "full_pipeline_consistent"
            ]
            .fillna(False)
            .sum()
        ),
        "status_counts": counts,
        "scientific_note": (
            "These are off-anchor consistency checks. "
            "No pre-existing classical reference values "
            "were used at the tested alpha values."
        ),
    }

    (
        args.output_dir
        / "validation_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n[summary]")
    print(
        summary[
            [
                "alpha",
                "pinn_cr",
                "pinn_ci",
                "pinn_r1_rel",
                "pinn_r2_rel",
                "gep_pinn_spectral_distance",
                "shooting_pinn_spectral_distance",
                "shooting_stage1_mismatch",
                "shooting_stage2_mismatch",
                "shooting_pinn_p_overlap",
                "shooting_pinn_p_rel_after_fit",
                "shooting_pinn_q_rel_after_fit",
                "shooting_stage1_seed_consistent",
                "shooting_modal_validated",
                "pinn_seed_validated_by_shooting",
                "full_pipeline_consistent",
                "status",
            ]
        ].to_string(index=False)
    )

    print("\n[report]")
    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
