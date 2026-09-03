#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.models.kh_supersonic_kappa_q_logamp import (
    KHSupersonicLocalPINN,
)
from src.physics.kh_supersonic_riccati_residual import (
    y_to_xi,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def build_model(
    config: dict[str, Any],
    device: torch.device,
) -> KHSupersonicLocalPINN:
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
    )

    return model.to(device)


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[KHSupersonicLocalPINN, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    config = checkpoint["config"]

    model = build_model(
        config,
        device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, config


def read_reference_modes(
    path: Path,
    *,
    mach: float,
    chunksize: int,
) -> pd.DataFrame:
    columns = [
        "final_reference_id",
        "Mach",
        "alpha",
        "cr",
        "ci",
        "omega_i",
        "coordinate_index",
        "y",
        "kappa",
        "q",
        "p_real",
        "p_imag",
    ]

    selected_chunks: list[pd.DataFrame] = []

    for chunk_index, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=columns,
            chunksize=chunksize,
        ),
        start=1,
    ):
        mask = np.isclose(
            chunk["Mach"].to_numpy(float),
            mach,
            rtol=0.0,
            atol=1.0e-12,
        )

        if mask.any():
            selected = chunk.loc[mask].copy()

            selected_chunks.append(selected)

            print(
                f"[read] chunk={chunk_index} "
                f"selected={len(selected)}",
                flush=True,
            )

    if not selected_chunks:
        raise RuntimeError(
            f"No modal rows found for Mach={mach}"
        )

    reference = pd.concat(
        selected_chunks,
        ignore_index=True,
    )

    reference = reference.sort_values(
        [
            "alpha",
            "final_reference_id",
            "coordinate_index",
        ]
    ).reset_index(drop=True)

    return reference


def predict_mode(
    model: KHSupersonicLocalPINN,
    *,
    y: np.ndarray,
    alpha_value: float,
    device: torch.device,
    batch_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    dtype = next(model.parameters()).dtype

    predictions: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(
            0,
            len(y),
            batch_size,
        ):
            stop = min(
                start + batch_size,
                len(y),
            )

            y_batch = torch.tensor(
                y[start:stop],
                device=device,
                dtype=dtype,
            ).view(-1, 1)

            xi_batch = y_to_xi(
                y_batch,
                model.get_mapping_scale(),
            )

            alpha_batch = torch.full_like(
                xi_batch,
                float(alpha_value),
            )

            prediction = model(
                xi_batch,
                alpha_batch,
            )

            predictions.append(
                prediction.detach()
                .cpu()
                .numpy()
            )

        alpha_tensor = torch.tensor(
            [[float(alpha_value)]],
            device=device,
            dtype=dtype,
        )

        cr_prediction, ci_prediction = (
            model.get_spectrum(
                alpha_tensor
            )
        )

    prediction_array = np.concatenate(
        predictions,
        axis=0,
    )

    return (
        prediction_array[:, 0],
        prediction_array[:, 1],
        prediction_array[:, 2],
        float(cr_prediction.item()),
        float(ci_prediction.item()),
    )


def integrate_phase(
    y: np.ndarray,
    phase_gradient: np.ndarray,
    center_index: int,
) -> np.ndarray:
    phase = np.zeros_like(
        phase_gradient,
        dtype=float,
    )

    if center_index < len(y) - 1:
        right_steps = np.diff(
            y[center_index:]
        )

        right_integrands = (
            0.5
            * (
                phase_gradient[
                    center_index:-1
                ]
                + phase_gradient[
                    center_index + 1:
                ]
            )
            * right_steps
        )

        phase[center_index + 1:] = (
            np.cumsum(right_integrands)
        )

    if center_index > 0:
        left_steps = np.diff(
            y[: center_index + 1]
        )

        left_integrands = (
            0.5
            * (
                phase_gradient[:center_index]
                + phase_gradient[
                    1: center_index + 1
                ]
            )
            * left_steps
        )

        phase[:center_index] = (
            -np.cumsum(
                left_integrands[::-1]
            )[::-1]
        )

    return phase


def relative_l2(
    prediction: np.ndarray,
    reference: np.ndarray,
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
        1.0e-12,
    )

    return float(
        numerator / denominator
    )


def normalized_rmse(
    prediction: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> float:
    error_rms = float(
        np.sqrt(
            np.mean(
                (
                    prediction[mask]
                    - reference[mask]
                )
                ** 2
            )
        )
    )

    reference_rms = max(
        float(
            np.sqrt(
                np.mean(
                    reference[mask] ** 2
                )
            )
        ),
        1.0e-10,
    )

    return error_rms / reference_rms


def overlap_error(
    prediction: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> float:
    prediction_masked = prediction[mask]
    reference_masked = reference[mask]

    denominator = (
        np.linalg.norm(prediction_masked)
        * np.linalg.norm(reference_masked)
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
        1.0 - np.clip(overlap, 0.0, 1.0)
    )


def is_close_to_any(
    value: float,
    values: np.ndarray,
) -> bool:
    if values.size == 0:
        return False

    return bool(
        np.any(
            np.isclose(
                value,
                values,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
    )


def evaluate_one_model(
    name: str,
    model: KHSupersonicLocalPINN,
    config: dict[str, Any],
    reference: pd.DataFrame,
    *,
    spectral_anchor_alphas: np.ndarray,
    modal_anchor_alphas: np.ndarray,
    device: torch.device,
    batch_size: int,
    y_max: float,
    amplitude_floor: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    grouped = reference.groupby(
        [
            "final_reference_id",
            "alpha",
        ],
        sort=True,
    )

    number_of_modes = grouped.ngroups

    for mode_number, (
        (reference_id, alpha_value),
        mode,
    ) in enumerate(
        grouped,
        start=1,
    ):
        mode = mode.sort_values(
            "coordinate_index"
        )

        y = mode["y"].to_numpy(float)

        kappa_reference = (
            mode["kappa"].to_numpy(float)
        )

        q_reference = (
            mode["q"].to_numpy(float)
        )

        p_reference = (
            mode["p_real"].to_numpy(float)
            + 1j
            * mode["p_imag"].to_numpy(float)
        )

        center_index = int(
            np.argmin(np.abs(y))
        )

        center_pressure = p_reference[
            center_index
        ]

        if abs(center_pressure) <= 1.0e-14:
            raise RuntimeError(
                "Near-zero center pressure for "
                f"reference_id={reference_id}"
            )

        p_reference_gauge = (
            p_reference
            / center_pressure
        )

        amplitude_reference = np.abs(
            p_reference_gauge
        )

        log_amp_reference = np.log(
            np.maximum(
                amplitude_reference,
                1.0e-30,
            )
        )

        phase_reference = np.unwrap(
            np.angle(
                p_reference_gauge
            )
        )

        phase_reference = (
            phase_reference
            - phase_reference[
                center_index
            ]
        )

        (
            kappa_prediction,
            q_prediction,
            log_amp_prediction,
            cr_prediction,
            ci_prediction,
        ) = predict_mode(
            model,
            y=y,
            alpha_value=float(alpha_value),
            device=device,
            batch_size=batch_size,
        )

        phase_prediction = integrate_phase(
            y,
            q_prediction,
            center_index,
        )

        clipped_log_amp_prediction = (
            np.clip(
                log_amp_prediction,
                -50.0,
                20.0,
            )
        )

        amplitude_prediction = np.exp(
            clipped_log_amp_prediction
        )

        p_prediction = (
            amplitude_prediction
            * np.exp(
                1j * phase_prediction
            )
        )

        finite_mask = (
            np.isfinite(y)
            & np.isfinite(kappa_reference)
            & np.isfinite(q_reference)
            & np.isfinite(
                log_amp_reference
            )
            & np.isfinite(kappa_prediction)
            & np.isfinite(q_prediction)
            & np.isfinite(
                log_amp_prediction
            )
            & (np.abs(y) <= y_max)
        )

        amplitude_mask = (
            finite_mask
            & (
                amplitude_reference
                >= amplitude_floor
            )
        )

        if finite_mask.sum() < 32:
            raise RuntimeError(
                "Insufficient finite modal points "
                f"at alpha={alpha_value}"
            )

        if amplitude_mask.sum() < 32:
            raise RuntimeError(
                "Insufficient amplitude-supported "
                f"points at alpha={alpha_value}"
            )

        cr_reference = float(
            mode["cr"].iloc[0]
        )

        ci_reference = float(
            mode["ci"].iloc[0]
        )

        is_spectral_anchor = (
            is_close_to_any(
                float(alpha_value),
                spectral_anchor_alphas,
            )
        )

        is_modal_anchor = (
            is_close_to_any(
                float(alpha_value),
                modal_anchor_alphas,
            )
        )

        if is_modal_anchor:
            role = "S4M2_modal_anchor"
        elif is_spectral_anchor:
            role = "spectral_only_anchor"
        else:
            role = "off_anchor"

        row = {
            "model": name,
            "final_reference_id": (
                reference_id
            ),
            "alpha": float(alpha_value),
            "role": role,
            "is_spectral_anchor": (
                is_spectral_anchor
            ),
            "is_S4M2_modal_anchor": (
                is_modal_anchor
            ),
            "n_domain": int(
                finite_mask.sum()
            ),
            "n_amplitude_mask": int(
                amplitude_mask.sum()
            ),
            "cr_reference": cr_reference,
            "cr_prediction": cr_prediction,
            "cr_abs_error": abs(
                cr_prediction - cr_reference
            ),
            "ci_reference": ci_reference,
            "ci_prediction": ci_prediction,
            "ci_abs_error": abs(
                ci_prediction - ci_reference
            ),
            "kappa_nrmse": normalized_rmse(
                kappa_prediction,
                kappa_reference,
                finite_mask,
            ),
            "q_nrmse": normalized_rmse(
                q_prediction,
                q_reference,
                finite_mask,
            ),
            "kappa_max_abs": float(
                np.max(
                    np.abs(
                        kappa_prediction[
                            finite_mask
                        ]
                        - kappa_reference[
                            finite_mask
                        ]
                    )
                )
            ),
            "q_max_abs": float(
                np.max(
                    np.abs(
                        q_prediction[
                            finite_mask
                        ]
                        - q_reference[
                            finite_mask
                        ]
                    )
                )
            ),
            "log_amp_rmse": float(
                np.sqrt(
                    np.mean(
                        (
                            log_amp_prediction[
                                amplitude_mask
                            ]
                            - log_amp_reference[
                                amplitude_mask
                            ]
                        )
                        ** 2
                    )
                )
            ),
            "amplitude_rel_l2": (
                relative_l2(
                    amplitude_prediction,
                    amplitude_reference,
                    amplitude_mask,
                )
            ),
            "phase_rmse": float(
                np.sqrt(
                    np.mean(
                        (
                            phase_prediction[
                                amplitude_mask
                            ]
                            - phase_reference[
                                amplitude_mask
                            ]
                        )
                        ** 2
                    )
                )
            ),
            "pressure_rel_l2": relative_l2(
                p_prediction,
                p_reference_gauge,
                amplitude_mask,
            ),
            "pressure_overlap_error": (
                overlap_error(
                    p_prediction,
                    p_reference_gauge,
                    amplitude_mask,
                )
            ),
        }

        rows.append(row)

        if (
            mode_number == 1
            or mode_number % 10 == 0
            or mode_number == number_of_modes
        ):
            print(
                f"[{name}] "
                f"mode={mode_number}/"
                f"{number_of_modes} "
                f"alpha={float(alpha_value):.6f} "
                f"p_rel={row['pressure_rel_l2']:.3e} "
                f"overlap={row['pressure_overlap_error']:.3e}",
                flush=True,
            )

    return pd.DataFrame(rows)


def build_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "cr_abs_error",
        "ci_abs_error",
        "kappa_nrmse",
        "q_nrmse",
        "log_amp_rmse",
        "amplitude_rel_l2",
        "phase_rmse",
        "pressure_rel_l2",
        "pressure_overlap_error",
    ]

    rows: list[dict[str, Any]] = []

    for model_name in sorted(
        metrics["model"].unique()
    ):
        model_metrics = metrics[
            metrics["model"].eq(model_name)
        ]

        subsets = {
            "all": model_metrics,
            "off_anchor": model_metrics[
                model_metrics[
                    "role"
                ].eq("off_anchor")
            ],
            "S4M2_modal_anchor": model_metrics[
                model_metrics[
                    "role"
                ].eq("S4M2_modal_anchor")
            ],
            "spectral_only_anchor": (
                model_metrics[
                    model_metrics[
                        "role"
                    ].eq(
                        "spectral_only_anchor"
                    )
                ]
            ),
        }

        for subset_name, subset in (
            subsets.items()
        ):
            if subset.empty:
                continue

            row: dict[str, Any] = {
                "model": model_name,
                "subset": subset_name,
                "n_modes": int(len(subset)),
            }

            for column in metric_columns:
                values = pd.to_numeric(
                    subset[column],
                    errors="coerce",
                )

                row[f"{column}_mean"] = (
                    float(values.mean())
                )

                row[f"{column}_median"] = (
                    float(values.median())
                )

                row[f"{column}_max"] = (
                    float(values.max())
                )

            rows.append(row)

    return pd.DataFrame(rows)


def build_paired_comparison(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    s4 = metrics[
        metrics["model"].eq("S4")
    ].copy()

    s4m2 = metrics[
        metrics["model"].eq("S4M2")
    ].copy()

    merge_columns = [
        "final_reference_id",
        "alpha",
        "role",
    ]

    metric_columns = [
        "cr_abs_error",
        "ci_abs_error",
        "kappa_nrmse",
        "q_nrmse",
        "log_amp_rmse",
        "amplitude_rel_l2",
        "phase_rmse",
        "pressure_rel_l2",
        "pressure_overlap_error",
    ]

    comparison = s4[
        merge_columns + metric_columns
    ].merge(
        s4m2[
            merge_columns + metric_columns
        ],
        on=merge_columns,
        suffixes=("_S4", "_S4M2"),
        validate="one_to_one",
    )

    for column in metric_columns:
        comparison[
            f"delta_{column}_S4M2_minus_S4"
        ] = (
            comparison[f"{column}_S4M2"]
            - comparison[f"{column}_S4"]
        )

    return comparison.sort_values(
        "alpha"
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--s4-checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--s4m2-checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--modal-reference",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--spectral-anchor-file",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--modal-anchor-bank",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mach",
        type=float,
        default=1.50,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
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

    args = parser.parse_args()

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(args.device)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    spectral_anchors = pd.read_csv(
        resolve_path(
            args.spectral_anchor_file
        )
    )

    spectral_anchor_alphas = (
        spectral_anchors[
            "alpha"
        ].to_numpy(float)
    )

    with np.load(
        resolve_path(
            args.modal_anchor_bank
        ),
        allow_pickle=False,
    ) as modal_bank:
        modal_anchor_alphas = (
            modal_bank["alpha"].astype(float)
        )

    reference = read_reference_modes(
        resolve_path(
            args.modal_reference
        ),
        mach=float(args.mach),
        chunksize=int(args.chunksize),
    )

    print()
    print("REFERENCE")
    print(
        "  rows :",
        len(reference),
    )
    print(
        "  modes:",
        reference[
            "final_reference_id"
        ].nunique(),
    )
    print(
        "  alpha:",
        reference["alpha"].min(),
        reference["alpha"].max(),
    )
    print()

    models = {}

    models["S4"] = load_checkpoint_model(
        resolve_path(args.s4_checkpoint),
        device,
    )

    models["S4M2"] = (
        load_checkpoint_model(
            resolve_path(
                args.s4m2_checkpoint
            ),
            device,
        )
    )

    metric_frames = []

    for name, (
        model,
        config,
    ) in models.items():
        metrics = evaluate_one_model(
            name,
            model,
            config,
            reference,
            spectral_anchor_alphas=(
                spectral_anchor_alphas
            ),
            modal_anchor_alphas=(
                modal_anchor_alphas
            ),
            device=device,
            batch_size=int(args.batch_size),
            y_max=float(args.y_max),
            amplitude_floor=float(
                args.amplitude_floor
            ),
        )

        metric_frames.append(metrics)

    all_metrics = pd.concat(
        metric_frames,
        ignore_index=True,
    )

    all_metrics.to_csv(
        output_dir
        / "modal_dense_per_mode_metrics.csv",
        index=False,
    )

    summary = build_summary(
        all_metrics
    )

    summary.to_csv(
        output_dir
        / "modal_dense_summary.csv",
        index=False,
    )

    paired = build_paired_comparison(
        all_metrics
    )

    paired.to_csv(
        output_dir
        / "modal_dense_paired_comparison.csv",
        index=False,
    )

    metadata = {
        "Mach": float(args.mach),
        "n_reference_rows": int(
            len(reference)
        ),
        "n_reference_modes": int(
            reference[
                "final_reference_id"
            ].nunique()
        ),
        "spectral_anchor_alphas": (
            spectral_anchor_alphas.tolist()
        ),
        "S4M2_modal_anchor_alphas": (
            modal_anchor_alphas.tolist()
        ),
        "y_max": float(args.y_max),
        "amplitude_floor": float(
            args.amplitude_floor
        ),
    }

    (
        output_dir / "audit_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    display_columns = [
        "model",
        "subset",
        "n_modes",
        "kappa_nrmse_mean",
        "q_nrmse_mean",
        "log_amp_rmse_mean",
        "pressure_rel_l2_mean",
        "pressure_overlap_error_mean",
    ]

    print()
    print("DENSE MODAL SUMMARY")
    print(
        summary[
            display_columns
        ].to_string(index=False)
    )

    print()
    print(
        "LOCAL SUPERSONIC DENSE "
        "MODAL AUDIT: COMPLETED"
    )


if __name__ == "__main__":
    main()
