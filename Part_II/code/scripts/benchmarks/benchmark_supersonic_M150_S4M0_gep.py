#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from classical_solver.gep.dense_gep_notebook_style import (
    NotebookStyleDenseGEPSolver,
)
from src.models.kh_supersonic_kappa_q_logamp import (
    KHSupersonicLocalPINN,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

ANCHOR_ALPHAS = np.asarray(
    [
        0.070,
        0.140,
        0.205,
        0.250,
    ],
    dtype=float,
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def build_pinn(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> KHSupersonicLocalPINN:
    config = checkpoint["config"]
    model_config = config["model"]

    model = KHSupersonicLocalPINN(
        mach=float(config["Mach"]),
        alpha_min=float(config["alpha_min"]),
        alpha_max=float(config["alpha_max"]),
        xi_max=float(model_config["xi_max"]),
        mapping_scale=float(
            model_config["mapping_scale"]
        ),
        spectral_width=int(
            model_config["spectral_width"]
        ),
        spectral_depth=int(
            model_config["spectral_depth"]
        ),
        modal_width=int(
            model_config["modal_width"]
        ),
        modal_depth=int(
            model_config["modal_depth"]
        ),
        n_frequencies=int(
            model_config["n_frequencies"]
        ),
        mode_experts=int(
            model_config["mode_experts"]
        ),
        alpha_split=float(
            model_config["alpha_split"]
        ),
        alpha_gate_width=float(
            model_config["alpha_gate_width"]
        ),
        cr_min=float(model_config["cr_min"]),
        cr_max=float(model_config["cr_max"]),
        ci_floor=float(
            model_config["ci_floor"]
        ),
        ci_max=float(model_config["ci_max"]),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    return model


def predict_pinn_spectrum(
    model: KHSupersonicLocalPINN,
    alpha: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dtype = next(model.parameters()).dtype

    alpha_tensor = torch.tensor(
        alpha,
        dtype=dtype,
        device=device,
    ).reshape(-1, 1)

    with torch.inference_mode():
        cr, ci = model.get_spectrum(
            alpha_tensor
        )

    return (
        cr.detach().cpu().numpy().reshape(-1),
        ci.detach().cpu().numpy().reshape(-1),
    )


def interpolate_complex(
    y_source: np.ndarray,
    values_source: np.ndarray,
    y_target: np.ndarray,
) -> np.ndarray:
    order = np.argsort(y_source)

    y_source = y_source[order]
    values_source = values_source[order]

    real = np.interp(
        y_target,
        y_source,
        values_source.real,
        left=np.nan,
        right=np.nan,
    )

    imag = np.interp(
        y_target,
        y_source,
        values_source.imag,
        left=np.nan,
        right=np.nan,
    )

    return real + 1j * imag


def relative_l2(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> float:
    numerator = np.linalg.norm(
        prediction[mask]
        - reference[mask]
    )

    denominator = max(
        float(
            np.linalg.norm(
                reference[mask]
            )
        ),
        1.0e-14,
    )

    return float(
        numerator / denominator
    )


def overlap_error(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> float:
    reference_masked = reference[mask]
    prediction_masked = prediction[mask]

    denominator = (
        np.linalg.norm(reference_masked)
        * np.linalg.norm(prediction_masked)
    )

    if denominator <= 1.0e-14:
        return float("nan")

    overlap = abs(
        np.vdot(
            reference_masked,
            prediction_masked,
        )
    ) / denominator

    return float(
        1.0
        - np.clip(
            overlap,
            0.0,
            1.0,
        )
    )


def alignment_factor(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> complex:
    denominator = np.vdot(
        prediction[mask],
        prediction[mask],
    )

    if abs(denominator) <= 1.0e-20:
        raise RuntimeError(
            "Near-zero GEP pressure mode"
        )

    return complex(
        np.vdot(
            prediction[mask],
            reference[mask],
        )
        / denominator
    )


def generalized_residual(
    solver: NotebookStyleDenseGEPSolver,
    mode: dict,
) -> float:
    matrix_a, matrix_b = (
        solver.construct_matrices()
    )

    n = solver.n_points

    keep = np.ones(
        3 * n,
        dtype=bool,
    )

    # v(-y_max)=v(+y_max)=0
    keep[
        [
            n,
            2 * n - 1,
        ]
    ] = False

    vector = np.asarray(
        mode["vector"],
        dtype=complex,
    )[keep]

    matrix_a = matrix_a[keep][:, keep]
    matrix_b = matrix_b[keep][:, keep]

    left = matrix_a @ vector
    right = mode["c"] * (
        matrix_b @ vector
    )

    denominator = max(
        float(
            np.linalg.norm(left)
            + np.linalg.norm(right)
        ),
        1.0e-14,
    )

    return float(
        np.linalg.norm(left - right)
        / denominator
    )


def evaluate_mode(
    *,
    solver: NotebookStyleDenseGEPSolver,
    mode: dict,
    classical_mode: pd.DataFrame,
    mach: float,
    y_max: float,
    amplitude_floor: float,
) -> dict[str, float]:
    classical_mode = classical_mode.sort_values(
        "coordinate_index"
    )

    y_reference = classical_mode[
        "y"
    ].to_numpy(float)

    reference_fields = {
        "p": (
            classical_mode[
                "p_real"
            ].to_numpy(float)
            + 1j
            * classical_mode[
                "p_imag"
            ].to_numpy(float)
        ),
        "rho": (
            classical_mode[
                "rho_real"
            ].to_numpy(float)
            + 1j
            * classical_mode[
                "rho_imag"
            ].to_numpy(float)
        ),
        "u": (
            classical_mode[
                "u_real"
            ].to_numpy(float)
            + 1j
            * classical_mode[
                "u_imag"
            ].to_numpy(float)
        ),
        "v": (
            classical_mode[
                "v_real"
            ].to_numpy(float)
            + 1j
            * classical_mode[
                "v_imag"
            ].to_numpy(float)
        ),
    }

    n = solver.n_points

    vector = np.asarray(
        mode["vector"],
        dtype=complex,
    )

    gep_fields = {
        "u": vector[0:n],
        "v": vector[n:2 * n],
        "p": vector[2 * n:3 * n],
    }

    gep_fields["rho"] = (
        float(mach) ** 2
        * gep_fields["p"]
    )

    y_gep = np.asarray(
        solver.y,
        dtype=float,
    )

    reference_on_gep = {
        field: interpolate_complex(
            y_reference,
            values,
            y_gep,
        )
        for field, values
        in reference_fields.items()
    }

    mask = (
        np.isfinite(y_gep)
        & np.isfinite(
            reference_on_gep["p"].real
        )
        & np.isfinite(
            reference_on_gep["p"].imag
        )
        & (np.abs(y_gep) <= y_max)
        & (
            np.abs(
                reference_on_gep["p"]
            )
            >= amplitude_floor
        )
    )

    if int(mask.sum()) < 32:
        raise RuntimeError(
            "Insufficient comparison points: "
            f"{int(mask.sum())}"
        )

    factor = alignment_factor(
        reference_on_gep["p"],
        gep_fields["p"],
        mask,
    )

    aligned = {
        field: factor * values
        for field, values
        in gep_fields.items()
    }

    metrics = {
        f"{field}_rel_l2": relative_l2(
            reference_on_gep[field],
            aligned[field],
            mask,
        )
        for field in [
            "p",
            "rho",
            "u",
            "v",
        ]
    }

    metrics[
        "pressure_overlap_error"
    ] = overlap_error(
        reference_on_gep["p"],
        aligned["p"],
        mask,
    )

    metrics[
        "generalized_residual"
    ] = generalized_residual(
        solver,
        mode,
    )

    metrics[
        "n_comparison_points"
    ] = int(mask.sum())

    return metrics


def make_solver(
    *,
    alpha: float,
    mach: float,
    n_points: int,
    mapping_scale: float,
    xi_max: float,
) -> NotebookStyleDenseGEPSolver:
    return NotebookStyleDenseGEPSolver(
        alpha=float(alpha),
        Mach=float(mach),
        n_points=int(n_points),
        mapping_kind="pin",
        mapping_scale=float(
            mapping_scale
        ),
        xi_max=float(xi_max),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed-source",
        choices=[
            "classical",
            "pinn",
        ],
        required=True,
    )

    parser.add_argument(
        "--alpha-set",
        choices=[
            "anchors",
            "all",
        ],
        required=True,
    )

    parser.add_argument(
        "--branch-continuation",
        action="store_true",
    )

    parser.add_argument(
        "--start-alpha",
        type=float,
        default=0.140,
    )

    parser.add_argument(
        "--mach",
        type=float,
        default=1.50,
    )

    parser.add_argument(
        "--n-points",
        type=int,
        default=301,
    )

    parser.add_argument(
        "--mapping-scale",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--xi-max",
        type=float,
        default=0.985,
    )

    parser.add_argument(
        "--ci-weight",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--y-max",
        type=float,
        default=80.0,
    )

    parser.add_argument(
        "--amplitude-floor",
        type=float,
        default=1.0e-6,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "models_saved/supersonic_general/long_S4/"
            "best_joint_checkpoint_601b535fde.pt"
        ),
    )

    parser.add_argument(
        "--spectral-reference",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--modal-reference",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    args = parser.parse_args()

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    spectral = pd.read_csv(
        resolve_path(
            args.spectral_reference
        )
    )

    spectral = spectral[
        np.isclose(
            spectral["Mach"],
            float(args.mach),
            rtol=0.0,
            atol=1.0e-12,
        )
    ].sort_values(
        "alpha"
    ).reset_index(drop=True)

    modal = pd.read_csv(
        resolve_path(
            args.modal_reference
        ),
        compression="gzip",
    )

    modal = modal[
        np.isclose(
            modal["Mach"],
            float(args.mach),
            rtol=0.0,
            atol=1.0e-12,
        )
    ].copy()

    if args.alpha_set == "anchors":
        selected_rows = []

        for alpha in ANCHOR_ALPHAS:
            matches = spectral[
                np.isclose(
                    spectral["alpha"],
                    alpha,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ]

            if len(matches) != 1:
                raise RuntimeError(
                    f"Missing anchor alpha={alpha}"
                )

            selected_rows.append(
                matches.iloc[0]
            )

        spectral = pd.DataFrame(
            selected_rows
        ).sort_values("alpha").reset_index(
            drop=True
        )

    alpha_values = spectral[
        "alpha"
    ].to_numpy(float)

    if args.seed_source == "classical":
        seed_cr = spectral[
            "cr"
        ].to_numpy(float)

        seed_ci = spectral[
            "ci"
        ].to_numpy(float)

    else:
        device = torch.device(
            args.device
        )

        checkpoint = torch.load(
            resolve_path(
                args.checkpoint
            ),
            map_location=device,
        )

        model = build_pinn(
            checkpoint,
            device,
        )

        seed_cr, seed_ci = (
            predict_pinn_spectrum(
                model,
                alpha_values,
                device,
            )
        )

    targets = {
        float(alpha): (
            float(cr),
            float(ci),
        )
        for alpha, cr, ci in zip(
            alpha_values,
            seed_cr,
            seed_ci,
        )
    }

    selected_modes: dict[
        float,
        tuple[
            NotebookStyleDenseGEPSolver,
            dict | None,
            str,
            int,
        ],
    ] = {}

    def solve_initial(
        alpha: float,
    ) -> tuple[
        NotebookStyleDenseGEPSolver,
        dict | None,
        str,
        int,
    ]:
        solver = make_solver(
            alpha=alpha,
            mach=float(args.mach),
            n_points=int(
                args.n_points
            ),
            mapping_scale=float(
                args.mapping_scale
            ),
            xi_max=float(
                args.xi_max
            ),
        )

        mode, source, n_modes = (
            solver.get_nearest_mode_to_target(
                target_guess=targets[alpha],
                prefer_positive_cr=True,
                ci_weight=float(
                    args.ci_weight
                ),
            )
        )

        return (
            solver,
            mode,
            source,
            n_modes,
        )

    if not args.branch_continuation:
        for alpha in alpha_values:
            alpha = float(alpha)

            selected_modes[alpha] = (
                solve_initial(alpha)
            )

    else:
        start_index = int(
            np.argmin(
                np.abs(
                    alpha_values
                    - float(args.start_alpha)
                )
            )
        )

        start_alpha = float(
            alpha_values[start_index]
        )

        selected_modes[start_alpha] = (
            solve_initial(start_alpha)
        )

        (
            _,
            start_mode,
            _,
            _,
        ) = selected_modes[start_alpha]

        if start_mode is None:
            raise RuntimeError(
                "No initial GEP mode found"
            )

        previous_mode = start_mode

        for alpha in alpha_values[
            start_index + 1:
        ]:
            alpha = float(alpha)

            solver = make_solver(
                alpha=alpha,
                mach=float(args.mach),
                n_points=int(
                    args.n_points
                ),
                mapping_scale=float(
                    args.mapping_scale
                ),
                xi_max=float(
                    args.xi_max
                ),
            )

            mode, source, n_modes = (
                solver.get_branch_mode(
                    target_guess=targets[alpha],
                    previous_guess=(
                        float(
                            previous_mode["cr"]
                        ),
                        float(
                            previous_mode["ci"]
                        ),
                    ),
                    previous_signature=(
                        previous_mode[
                            "signature"
                        ]
                    ),
                    prefer_positive_cr=True,
                    ci_weight=float(
                        args.ci_weight
                    ),
                )
            )

            selected_modes[alpha] = (
                solver,
                mode,
                source,
                n_modes,
            )

            if mode is not None:
                previous_mode = mode

        previous_mode = start_mode

        for alpha in alpha_values[
            :start_index
        ][::-1]:
            alpha = float(alpha)

            solver = make_solver(
                alpha=alpha,
                mach=float(args.mach),
                n_points=int(
                    args.n_points
                ),
                mapping_scale=float(
                    args.mapping_scale
                ),
                xi_max=float(
                    args.xi_max
                ),
            )

            mode, source, n_modes = (
                solver.get_branch_mode(
                    target_guess=targets[alpha],
                    previous_guess=(
                        float(
                            previous_mode["cr"]
                        ),
                        float(
                            previous_mode["ci"]
                        ),
                    ),
                    previous_signature=(
                        previous_mode[
                            "signature"
                        ]
                    ),
                    prefer_positive_cr=True,
                    ci_weight=float(
                        args.ci_weight
                    ),
                )
            )

            selected_modes[alpha] = (
                solver,
                mode,
                source,
                n_modes,
            )

            if mode is not None:
                previous_mode = mode

    rows: list[
        dict[str, Any]
    ] = []

    for index, reference_row in (
        spectral.iterrows()
    ):
        alpha = float(
            reference_row["alpha"]
        )

        (
            solver,
            mode,
            selection_source,
            n_modes,
        ) = selected_modes[alpha]

        row: dict[
            str,
            Any,
        ] = {
            "Mach": float(args.mach),
            "alpha": alpha,
            "seed_source": (
                args.seed_source
            ),
            "seed_cr": targets[alpha][0],
            "seed_ci": targets[alpha][1],
            "cr_reference": float(
                reference_row["cr"]
            ),
            "ci_reference": float(
                reference_row["ci"]
            ),
            "selection_source": (
                selection_source
            ),
            "n_finite_modes": int(
                n_modes
            ),
            "success": mode is not None,
        }

        if mode is None:
            rows.append(row)
            continue

        classical_mode = modal[
            np.isclose(
                modal["alpha"],
                alpha,
                rtol=0.0,
                atol=1.0e-12,
            )
        ]

        if classical_mode.empty:
            raise RuntimeError(
                f"No modal reference for alpha={alpha}"
            )

        modal_metrics = evaluate_mode(
            solver=solver,
            mode=mode,
            classical_mode=classical_mode,
            mach=float(args.mach),
            y_max=float(args.y_max),
            amplitude_floor=float(
                args.amplitude_floor
            ),
        )

        row.update(
            {
                "cr_gep": float(
                    mode["cr"]
                ),
                "ci_gep": float(
                    mode["ci"]
                ),
                "omega_i_gep": float(
                    mode["omega_i"]
                ),
                "seed_cr_abs_error": abs(
                    targets[alpha][0]
                    - float(
                        reference_row["cr"]
                    )
                ),
                "seed_ci_abs_error": abs(
                    targets[alpha][1]
                    - float(
                        reference_row["ci"]
                    )
                ),
                "gep_cr_abs_error": abs(
                    float(mode["cr"])
                    - float(
                        reference_row["cr"]
                    )
                ),
                "gep_ci_abs_error": abs(
                    float(mode["ci"])
                    - float(
                        reference_row["ci"]
                    )
                ),
                "seed_complex_error": float(
                    np.hypot(
                        targets[alpha][0]
                        - float(
                            reference_row["cr"]
                        ),
                        targets[alpha][1]
                        - float(
                            reference_row["ci"]
                        ),
                    )
                ),
                "gep_complex_error": float(
                    np.hypot(
                        float(mode["cr"])
                        - float(
                            reference_row["cr"]
                        ),
                        float(mode["ci"])
                        - float(
                            reference_row["ci"]
                        ),
                    )
                ),
                **modal_metrics,
            }
        )

        rows.append(row)

        print(
            f"{index + 1:3d}/{len(spectral):3d} "
            f"alpha={alpha:.6f} "
            f"seed=({targets[alpha][0]:.6f},"
            f"{targets[alpha][1]:.6f}) "
            f"GEP=({mode['cr']:.6f},"
            f"{mode['ci']:.6f}) "
            f"p_rel={modal_metrics['p_rel_l2']:.3e} "
            f"res={modal_metrics['generalized_residual']:.3e}",
            flush=True,
        )

    results = pd.DataFrame(
        rows
    ).sort_values("alpha")

    csv_path = (
        output_dir
        / "gep_results.csv"
    )

    results.to_csv(
        csv_path,
        index=False,
    )

    successful = results[
        results["success"].eq(True)
    ]

    if successful.empty:
        raise RuntimeError(
            "No successful GEP selections"
        )

    summary = {
        "status": "COMPLETED",
        "Mach": float(args.mach),
        "seed_source": args.seed_source,
        "alpha_set": args.alpha_set,
        "branch_continuation": bool(
            args.branch_continuation
        ),
        "start_alpha": float(
            args.start_alpha
        ),
        "n_points": int(
            args.n_points
        ),
        "mapping_scale": float(
            args.mapping_scale
        ),
        "xi_max": float(
            args.xi_max
        ),
        "n_requested": int(
            len(results)
        ),
        "n_successful": int(
            len(successful)
        ),
        "seed_cr_mae": float(
            successful[
                "seed_cr_abs_error"
            ].mean()
        ),
        "seed_ci_mae": float(
            successful[
                "seed_ci_abs_error"
            ].mean()
        ),
        "gep_cr_mae": float(
            successful[
                "gep_cr_abs_error"
            ].mean()
        ),
        "gep_ci_mae": float(
            successful[
                "gep_ci_abs_error"
            ].mean()
        ),
        "seed_complex_error_mean": float(
            successful[
                "seed_complex_error"
            ].mean()
        ),
        "gep_complex_error_mean": float(
            successful[
                "gep_complex_error"
            ].mean()
        ),
        "gep_complex_improved_count": int(
            (
                successful[
                    "gep_complex_error"
                ]
                < successful[
                    "seed_complex_error"
                ]
            ).sum()
        ),
        "p_rel_l2_mean": float(
            successful[
                "p_rel_l2"
            ].mean()
        ),
        "p_rel_l2_max": float(
            successful[
                "p_rel_l2"
            ].max()
        ),
        "rho_rel_l2_mean": float(
            successful[
                "rho_rel_l2"
            ].mean()
        ),
        "u_rel_l2_mean": float(
            successful[
                "u_rel_l2"
            ].mean()
        ),
        "v_rel_l2_mean": float(
            successful[
                "v_rel_l2"
            ].mean()
        ),
        "pressure_overlap_error_mean": (
            float(
                successful[
                    "pressure_overlap_error"
                ].mean()
            )
        ),
        "generalized_residual_max": float(
            successful[
                "generalized_residual"
            ].max()
        ),
        "results_csv": str(
            csv_path
        ),
    }

    summary_path = (
        output_dir
        / "gep_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(json.dumps(summary, indent=2))
    print()
    print("SUPERSONIC S4M0 + GEP: COMPLETED")


if __name__ == "__main__":
    main()
