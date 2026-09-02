#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
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

MODAL_COLUMNS = [
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
    "rho_real",
    "rho_imag",
    "u_real",
    "u_imag",
    "v_real",
    "v_imag",
]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def sanitize_filename(value: object) -> str:
    text = str(value)

    text = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        text,
    )

    return text.strip("_") or "mode"


def complex_relative_l2(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> float:
    reference_masked = reference[mask]
    prediction_masked = prediction[mask]

    denominator = max(
        float(np.linalg.norm(reference_masked)),
        1.0e-14,
    )

    return float(
        np.linalg.norm(
            prediction_masked
            - reference_masked
        )
        / denominator
    )


def pressure_overlap_error(
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


def complex_alignment_factor(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> complex:
    reference_masked = reference[mask]
    prediction_masked = prediction[mask]

    denominator = np.vdot(
        prediction_masked,
        prediction_masked,
    )

    if abs(denominator) <= 1.0e-20:
        raise RuntimeError(
            "Cannot align a near-zero predicted mode"
        )

    return complex(
        np.vdot(
            prediction_masked,
            reference_masked,
        )
        / denominator
    )


def integrate_phase_from_center(
    y: np.ndarray,
    phase_gradient: np.ndarray,
) -> np.ndarray:
    if len(y) != len(phase_gradient):
        raise ValueError(
            "y and phase_gradient have "
            "different lengths"
        )

    order = np.argsort(y)

    y_sorted = np.asarray(
        y[order],
        dtype=float,
    )

    q_sorted = np.asarray(
        phase_gradient[order],
        dtype=float,
    )

    phase_sorted = np.zeros_like(
        y_sorted,
        dtype=float,
    )

    center_index = int(
        np.argmin(np.abs(y_sorted))
    )

    if center_index < len(y_sorted) - 1:
        increments_right = (
            0.5
            * (
                q_sorted[center_index:-1]
                + q_sorted[
                    center_index + 1:
                ]
            )
            * np.diff(
                y_sorted[center_index:]
            )
        )

        phase_sorted[
            center_index + 1:
        ] = np.cumsum(increments_right)

    if center_index > 0:
        increments_left = (
            0.5
            * (
                q_sorted[:center_index]
                + q_sorted[
                    1:center_index + 1
                ]
            )
            * np.diff(
                y_sorted[:center_index + 1]
            )
        )

        phase_sorted[:center_index] = (
            -np.cumsum(
                increments_left[::-1]
            )[::-1]
        )

    phase = np.empty_like(
        phase_sorted
    )

    phase[order] = phase_sorted

    return phase


def reconstruct_primitive_fields(
    *,
    y: np.ndarray,
    alpha: float,
    mach: float,
    cr: float,
    ci: float,
    pressure: np.ndarray,
    kappa: np.ndarray,
    phase_gradient: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Reconstruction corresponding to the pressure equation

        p'' - 2 U_y/(U-c) p'
            - alpha^2 [1-M^2(U-c)^2] p = 0,

    with normal-mode convention exp(i alpha (x-c t)).

    The implementation is validated against rho/u/v stored
    in the classical reference before any asset is generated.
    """
    c = complex(cr, ci)

    velocity = np.tanh(y)

    velocity_derivative = (
        1.0
        - velocity**2
    )

    doppler = velocity - c

    pressure_derivative = (
        (
            kappa
            + 1j * phase_gradient
        )
        * pressure
    )

    denominator = (
        1j
        * float(alpha)
        * doppler
    )

    if np.any(
        np.abs(denominator) <= 1.0e-14
    ):
        raise RuntimeError(
            "Near-zero reconstruction denominator"
        )

    v_mode = (
        -pressure_derivative
        / denominator
    )

    u_mode = (
        -(
            velocity_derivative
            * v_mode
            + 1j
            * float(alpha)
            * pressure
        )
        / denominator
    )

    rho_mode = (
        float(mach) ** 2
        * pressure
    )

    return {
        "p": pressure,
        "rho": rho_mode,
        "u": u_mode,
        "v": v_mode,
    }


def build_model(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[
    KHSupersonicLocalPINN,
    dict[str, Any],
]:
    config = checkpoint["config"]
    model_config = config["model"]

    model = KHSupersonicLocalPINN(
        mach=float(config["Mach"]),
        alpha_min=float(
            config["alpha_min"]
        ),
        alpha_max=float(
            config["alpha_max"]
        ),
        xi_max=float(
            model_config["xi_max"]
        ),
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
        cr_min=float(
            model_config["cr_min"]
        ),
        cr_max=float(
            model_config["cr_max"]
        ),
        ci_floor=float(
            model_config["ci_floor"]
        ),
        ci_max=float(
            model_config["ci_max"]
        ),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    return model, config


def load_selected_model(
    *,
    pilot_dir: Path,
    experiment: str,
    device: torch.device,
) -> tuple[
    KHSupersonicLocalPINN,
    dict[str, Any],
    Path,
]:
    run_dir = (
        pilot_dir
        / "runs"
        / experiment
    )

    selection_path = (
        run_dir
        / "checkpoint_selection.json"
    )

    selection = json.loads(
        selection_path.read_text(
            encoding="utf-8"
        )
    )

    checkpoint_path = resolve_path(
        selection["selected_checkpoint"]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model, config = build_model(
        checkpoint,
        device,
    )

    return (
        model,
        config,
        checkpoint_path,
    )


def predict_spectrum(
    model: KHSupersonicLocalPINN,
    alpha: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dtype = next(
        model.parameters()
    ).dtype

    alpha_tensor = torch.tensor(
        alpha,
        device=device,
        dtype=dtype,
    ).reshape(-1, 1)

    with torch.inference_mode():
        cr, ci = model.get_spectrum(
            alpha_tensor
        )

    return (
        cr.detach().cpu().numpy()[:, 0],
        ci.detach().cpu().numpy()[:, 0],
    )


def predict_modal_fields(
    model: KHSupersonicLocalPINN,
    *,
    y: np.ndarray,
    alpha: float,
    device: torch.device,
    batch_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    dtype = next(
        model.parameters()
    ).dtype

    outputs: list[np.ndarray] = []

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

            y_tensor = torch.tensor(
                y[start:stop],
                device=device,
                dtype=dtype,
            ).reshape(-1, 1)

            xi_tensor = y_to_xi(
                y_tensor,
                model.get_mapping_scale(),
            )

            alpha_tensor = torch.full_like(
                xi_tensor,
                float(alpha),
            )

            prediction = model(
                xi_tensor,
                alpha_tensor,
            )

            outputs.append(
                prediction.detach()
                .cpu()
                .numpy()
            )

    output = np.concatenate(
        outputs,
        axis=0,
    )

    if output.shape != (
        len(y),
        3,
    ):
        raise RuntimeError(
            "Unexpected modal prediction shape: "
            f"{output.shape}"
        )

    return (
        output[:, 0],
        output[:, 1],
        output[:, 2],
    )


def mode_comparison_mask(
    *,
    y: np.ndarray,
    pressure_reference: np.ndarray,
    pressure_prediction: np.ndarray,
    y_max: float,
    amplitude_floor: float,
) -> np.ndarray:
    return (
        np.isfinite(y)
        & np.isfinite(
            pressure_reference.real
        )
        & np.isfinite(
            pressure_reference.imag
        )
        & np.isfinite(
            pressure_prediction.real
        )
        & np.isfinite(
            pressure_prediction.imag
        )
        & (np.abs(y) <= y_max)
        & (
            np.abs(pressure_reference)
            >= amplitude_floor
        )
    )


def audit_reconstruction_formulas(
    modal_reference: pd.DataFrame,
    *,
    y_max: float,
    amplitude_floor: float,
    tolerance: float,
) -> pd.DataFrame:
    audit_rows: list[
        dict[str, float | str]
    ] = []

    for mach in sorted(
        modal_reference["Mach"].unique()
    ):
        mach_modes = modal_reference[
            np.isclose(
                modal_reference["Mach"],
                mach,
                rtol=0.0,
                atol=1.0e-12,
            )
        ]

        identifiers = (
            mach_modes[
                "final_reference_id"
            ]
            .drop_duplicates()
            .tolist()
        )

        selected_identifiers = (
            identifiers
            if len(identifiers) <= 3
            else [
                identifiers[0],
                identifiers[
                    len(identifiers) // 2
                ],
                identifiers[-1],
            ]
        )

        for reference_id in (
            selected_identifiers
        ):
            mode = mach_modes[
                mach_modes[
                    "final_reference_id"
                ].eq(reference_id)
            ].sort_values(
                "coordinate_index"
            )

            y = mode["y"].to_numpy(float)

            pressure = (
                mode["p_real"].to_numpy(float)
                + 1j
                * mode[
                    "p_imag"
                ].to_numpy(float)
            )

            kappa = mode[
                "kappa"
            ].to_numpy(float)

            q = mode["q"].to_numpy(float)

            reconstructed = (
                reconstruct_primitive_fields(
                    y=y,
                    alpha=float(
                        mode["alpha"].iloc[0]
                    ),
                    mach=float(mach),
                    cr=float(
                        mode["cr"].iloc[0]
                    ),
                    ci=float(
                        mode["ci"].iloc[0]
                    ),
                    pressure=pressure,
                    kappa=kappa,
                    phase_gradient=q,
                )
            )

            mask = (
                np.isfinite(y)
                & (np.abs(y) <= y_max)
                & (
                    np.abs(pressure)
                    >= amplitude_floor
                )
            )

            classical = {
                "rho": (
                    mode[
                        "rho_real"
                    ].to_numpy(float)
                    + 1j
                    * mode[
                        "rho_imag"
                    ].to_numpy(float)
                ),
                "u": (
                    mode[
                        "u_real"
                    ].to_numpy(float)
                    + 1j
                    * mode[
                        "u_imag"
                    ].to_numpy(float)
                ),
                "v": (
                    mode[
                        "v_real"
                    ].to_numpy(float)
                    + 1j
                    * mode[
                        "v_imag"
                    ].to_numpy(float)
                ),
            }

            row: dict[
                str,
                float | str,
            ] = {
                "reference_id": str(
                    reference_id
                ),
                "Mach": float(mach),
                "alpha": float(
                    mode["alpha"].iloc[0]
                ),
            }

            for field in [
                "rho",
                "u",
                "v",
            ]:
                row[
                    f"{field}_rel_l2"
                ] = complex_relative_l2(
                    classical[field],
                    reconstructed[field],
                    mask,
                )

            audit_rows.append(row)

    audit = pd.DataFrame(
        audit_rows
    )

    metric_columns = [
        "rho_rel_l2",
        "u_rel_l2",
        "v_rel_l2",
    ]

    maximum_error = float(
        audit[
            metric_columns
        ].to_numpy(float).max()
    )

    print()
    print(
        "CLASSICAL PRIMITIVE "
        "RECONSTRUCTION AUDIT"
    )

    print(
        audit.to_string(index=False)
    )

    print()
    print(
        "maximum reconstruction error:",
        f"{maximum_error:.12e}",
    )

    if maximum_error > tolerance:
        raise RuntimeError(
            "Primitive reconstruction formulas "
            "do not reproduce the classical "
            "reference accurately enough: "
            f"max_error={maximum_error:.6e}, "
            f"tolerance={tolerance:.6e}"
        )

    print(
        "PRIMITIVE RECONSTRUCTION: OK"
    )

    return audit


def save_figure(
    figure: plt.Figure,
    output_stem: Path,
    *,
    dpi: int,
) -> None:
    output_stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
    )

    figure.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )


def make_spectral_overlay(
    *,
    alpha: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    symbol: str,
    title: str,
    output_stem: Path,
    dpi: int,
) -> None:
    absolute_error = np.abs(
        prediction
        - reference
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.5, 7.0),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3, 1],
        },
    )

    axes[0].plot(
        alpha,
        reference,
        linestyle="-",
        marker="o",
        markersize=3.5,
        linewidth=1.4,
        label="Classique",
    )

    axes[0].plot(
        alpha,
        prediction,
        linestyle="--",
        linewidth=1.8,
        label="PINN S4M4",
    )

    axes[0].set_ylabel(
        f"${symbol}$"
    )

    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        alpha,
        absolute_error,
        linewidth=1.4,
    )

    axes[1].set_xlabel(
        r"$\alpha$"
    )

    axes[1].set_ylabel(
        "Erreur abs."
    )

    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()

    save_figure(
        figure,
        output_stem,
        dpi=dpi,
    )

    plt.close(figure)


def make_error_heatmap(
    *,
    dataframe: pd.DataFrame,
    value_column: str,
    title: str,
    colorbar_label: str,
    output_stem: Path,
    dpi: int,
) -> None:
    pivot = dataframe.pivot(
        index="Mach",
        columns="alpha",
        values=value_column,
    ).sort_index()

    values = np.ma.masked_invalid(
        pivot.to_numpy(float)
    )

    color_map = plt.cm.viridis.copy()

    color_map.set_bad(
        color="lightgray"
    )

    figure, axis = plt.subplots(
        figsize=(12.0, 4.5)
    )

    image = axis.imshow(
        values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
    )

    alpha_values = (
        pivot.columns.to_numpy(float)
    )

    mach_values = (
        pivot.index.to_numpy(float)
    )

    number_of_ticks = min(
        12,
        len(alpha_values),
    )

    tick_indices = np.unique(
        np.linspace(
            0,
            len(alpha_values) - 1,
            number_of_ticks,
        ).round().astype(int)
    )

    axis.set_xticks(
        tick_indices
    )

    axis.set_xticklabels(
        [
            f"{alpha_values[index]:.3f}"
            for index in tick_indices
        ],
        rotation=45,
        ha="right",
    )

    axis.set_yticks(
        np.arange(len(mach_values))
    )

    axis.set_yticklabels(
        [
            f"{mach:.2f}"
            for mach in mach_values
        ]
    )

    axis.set_xlabel(
        r"$\alpha$"
    )

    axis.set_ylabel(
        "Mach"
    )

    axis.set_title(title)

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        colorbar_label
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_stem,
        dpi=dpi,
    )

    plt.close(figure)


def make_mode_figure(
    *,
    label: str,
    reference_id: object,
    mach: float,
    alpha: float,
    cr_reference: float,
    ci_reference: float,
    cr_prediction: float,
    ci_prediction: float,
    y: np.ndarray,
    classical: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    errors: dict[str, float],
    overlap_error: float,
    plot_y_max: float,
) -> plt.Figure:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 9.2),
        sharex=True,
    )

    plot_mask = (
        np.isfinite(y)
        & (
            np.abs(y)
            <= plot_y_max
        )
    )

    fields = [
        ("p", r"$p$"),
        ("rho", r"$\rho$"),
        ("u", r"$u$"),
        ("v", r"$v$"),
    ]

    for axis, (
        field,
        display_name,
    ) in zip(
        axes.ravel(),
        fields,
    ):
        reference = classical[field]
        predicted = prediction[field]

        axis.plot(
            y[plot_mask],
            reference.real[plot_mask],
            color="black",
            linestyle="-",
            linewidth=1.35,
        )

        axis.plot(
            y[plot_mask],
            reference.imag[plot_mask],
            color="black",
            linestyle="--",
            linewidth=1.35,
        )

        axis.plot(
            y[plot_mask],
            predicted.real[plot_mask],
            color="tab:red",
            linestyle="-",
            linewidth=1.15,
        )

        axis.plot(
            y[plot_mask],
            predicted.imag[plot_mask],
            color="tab:red",
            linestyle="--",
            linewidth=1.15,
        )

        axis.set_title(
            f"{display_name}   "
            f"rel. L2 = "
            f"{errors[field]:.3e}"
        )

        axis.set_ylabel(
            "Amplitude"
        )

        axis.grid(
            True,
            alpha=0.25,
        )

    axes[1, 0].set_xlabel(
        r"$y$"
    )

    axes[1, 1].set_xlabel(
        r"$y$"
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            label="Classique - réel",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            label="Classique - imaginaire",
        ),
        Line2D(
            [0],
            [0],
            color="tab:red",
            linestyle="-",
            label="PINN - réel",
        ),
        Line2D(
            [0],
            [0],
            color="tab:red",
            linestyle="--",
            label="PINN - imaginaire",
        ),
    ]

    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(
            0.5,
            0.935,
        ),
    )

    figure.suptitle(
        (
            f"{label} - mode {reference_id} - "
            f"M={mach:.2f}, "
            f"alpha={alpha:.6f}\n"
            f"Classique: "
            f"c={cr_reference:.6f}"
            f"+i{ci_reference:.6f}  |  "
            f"PINN: "
            f"c={cr_prediction:.6f}"
            f"+i{ci_prediction:.6f}  |  "
            f"overlap error(p)="
            f"{overlap_error:.3e}"
        ),
        fontsize=11,
    )

    figure.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
            0.88,
        ]
    )

    return figure


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
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
        default="cuda",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--audit-y-max",
        type=float,
        default=80.0,
    )

    parser.add_argument(
        "--plot-y-max",
        type=float,
        default=40.0,
    )

    parser.add_argument(
        "--amplitude-floor",
        type=float,
        default=1.0e-6,
    )

    parser.add_argument(
        "--reconstruction-tolerance",
        type=float,
        default=5.0e-4,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--formula-check-only",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    manifest_path = resolve_path(
        args.manifest
    )

    spectral_path = resolve_path(
        args.spectral_reference
    )

    modal_path = resolve_path(
        args.modal_reference
    )

    output_dir = resolve_path(
        args.output_dir
    )

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    modal_reference = pd.read_csv(
        modal_path,
        compression="gzip",
        usecols=MODAL_COLUMNS,
    )

    reconstruction_audit = (
        audit_reconstruction_formulas(
            modal_reference,
            y_max=float(
                args.audit_y_max
            ),
            amplitude_floor=float(
                args.amplitude_floor
            ),
            tolerance=float(
                args.reconstruction_tolerance
            ),
        )
    )

    if args.formula_check_only:
        print()
        print(
            "FORMULA CHECK ONLY: COMPLETED"
        )

        return

    if output_dir.exists() and args.overwrite:
        for relative_path in [
            "spectra",
            "modes",
            "pdf",
            "summary_fixed_mach_assets.csv",
            "asset_report.json",
            "primitive_reconstruction_audit.csv",
        ]:
            target = (
                output_dir
                / relative_path
            )

            if target.is_dir():
                shutil.rmtree(target)

            elif target.exists():
                target.unlink()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    spectra_dir = (
        output_dir / "spectra"
    )

    modes_dir = (
        output_dir / "modes"
    )

    pdf_dir = (
        output_dir / "pdf"
    )

    for directory in [
        spectra_dir,
        modes_dir,
        pdf_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    reconstruction_audit.to_csv(
        output_dir
        / "primitive_reconstruction_audit.csv",
        index=False,
    )

    manifest = pd.read_csv(
        manifest_path
    )

    required_manifest_columns = {
        "label",
        "mach",
        "pilot_dir",
        "experiment",
    }

    missing_columns = (
        required_manifest_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Manifest missing columns: "
            f"{sorted(missing_columns)}"
        )

    spectral_reference = pd.read_csv(
        spectral_path,
        usecols=[
            "Mach",
            "alpha",
            "cr",
            "ci",
            "omega_i",
        ],
    )

    heatmap_rows: list[
        pd.DataFrame
    ] = []

    global_summary_rows: list[
        dict[str, Any]
    ] = []

    complete_pdf_path = (
        pdf_dir
        / "modes_fixed_mach_S4M4_complete.pdf"
    )

    total_modes_exported = 0

    with PdfPages(
        complete_pdf_path
    ) as complete_pdf:
        for manifest_row in (
            manifest.itertuples(index=False)
        ):
            label = str(
                manifest_row.label
            )

            mach = float(
                manifest_row.mach
            )

            pilot_dir = resolve_path(
                manifest_row.pilot_dir
            )

            experiment = str(
                manifest_row.experiment
            )

            print()
            print("=" * 80)
            print(
                f"{label}: M={mach:.2f}"
            )
            print("=" * 80)

            (
                model,
                config,
                checkpoint_path,
            ) = load_selected_model(
                pilot_dir=pilot_dir,
                experiment=experiment,
                device=device,
            )

            model_mach = float(
                config["Mach"]
            )

            if not np.isclose(
                model_mach,
                mach,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    f"{label}: manifest/checkpoint "
                    "Mach mismatch"
                )

            spectrum = spectral_reference[
                np.isclose(
                    spectral_reference["Mach"],
                    mach,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ].sort_values(
                "alpha"
            ).reset_index(drop=True)

            if spectrum.empty:
                raise RuntimeError(
                    f"No spectral reference "
                    f"for Mach={mach}"
                )

            alpha_values = spectrum[
                "alpha"
            ].to_numpy(float)

            (
                cr_prediction,
                ci_prediction,
            ) = predict_spectrum(
                model,
                alpha_values,
                device,
            )

            spectrum = spectrum.copy()

            spectrum[
                "cr_pinn"
            ] = cr_prediction

            spectrum[
                "ci_pinn"
            ] = ci_prediction

            spectrum[
                "cr_abs_error"
            ] = np.abs(
                spectrum["cr_pinn"]
                - spectrum["cr"]
            )

            spectrum[
                "ci_abs_error"
            ] = np.abs(
                spectrum["ci_pinn"]
                - spectrum["ci"]
            )

            label_spectra_dir = (
                spectra_dir / label
            )

            label_spectra_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            spectrum.to_csv(
                label_spectra_dir
                / "spectral_predictions.csv",
                index=False,
            )

            make_spectral_overlay(
                alpha=alpha_values,
                reference=spectrum[
                    "ci"
                ].to_numpy(float),
                prediction=ci_prediction,
                symbol="c_i",
                title=(
                    f"{label} - M={mach:.2f} - "
                    r"$c_i(\alpha)$"
                ),
                output_stem=(
                    label_spectra_dir
                    / f"ci_overlay_{label}"
                ),
                dpi=int(args.dpi),
            )

            make_spectral_overlay(
                alpha=alpha_values,
                reference=spectrum[
                    "cr"
                ].to_numpy(float),
                prediction=cr_prediction,
                symbol="c_r",
                title=(
                    f"{label} - M={mach:.2f} - "
                    r"$c_r(\alpha)$"
                ),
                output_stem=(
                    label_spectra_dir
                    / f"cr_overlay_{label}"
                ),
                dpi=int(args.dpi),
            )

            heatmap_rows.append(
                spectrum[
                    [
                        "Mach",
                        "alpha",
                        "cr_abs_error",
                        "ci_abs_error",
                    ]
                ].copy()
            )

            modal_mach = modal_reference[
                np.isclose(
                    modal_reference["Mach"],
                    mach,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ].copy()

            modal_mach = modal_mach.sort_values(
                [
                    "alpha",
                    "final_reference_id",
                    "coordinate_index",
                ]
            )

            grouped_modes = modal_mach.groupby(
                [
                    "final_reference_id",
                    "alpha",
                ],
                sort=True,
            )

            number_of_modes = (
                grouped_modes.ngroups
            )

            label_modes_dir = (
                modes_dir / label
            )

            label_modes_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            per_mode_rows: list[
                dict[str, Any]
            ] = []

            mach_pdf_path = (
                pdf_dir
                / f"modes_{label}.pdf"
            )

            with PdfPages(
                mach_pdf_path
            ) as mach_pdf:
                for mode_number, (
                    (
                        reference_id,
                        alpha,
                    ),
                    mode,
                ) in enumerate(
                    grouped_modes,
                    start=1,
                ):
                    mode = mode.sort_values(
                        "coordinate_index"
                    )

                    alpha = float(alpha)

                    y = mode[
                        "y"
                    ].to_numpy(float)

                    classical = {
                        "p": (
                            mode[
                                "p_real"
                            ].to_numpy(float)
                            + 1j
                            * mode[
                                "p_imag"
                            ].to_numpy(float)
                        ),
                        "rho": (
                            mode[
                                "rho_real"
                            ].to_numpy(float)
                            + 1j
                            * mode[
                                "rho_imag"
                            ].to_numpy(float)
                        ),
                        "u": (
                            mode[
                                "u_real"
                            ].to_numpy(float)
                            + 1j
                            * mode[
                                "u_imag"
                            ].to_numpy(float)
                        ),
                        "v": (
                            mode[
                                "v_real"
                            ].to_numpy(float)
                            + 1j
                            * mode[
                                "v_imag"
                            ].to_numpy(float)
                        ),
                    }

                    (
                        kappa_prediction,
                        q_prediction,
                        log_amp_prediction,
                    ) = predict_modal_fields(
                        model,
                        y=y,
                        alpha=alpha,
                        device=device,
                        batch_size=int(
                            args.batch_size
                        ),
                    )

                    phase_prediction = (
                        integrate_phase_from_center(
                            y,
                            q_prediction,
                        )
                    )

                    pressure_prediction = (
                        np.exp(
                            np.clip(
                                log_amp_prediction,
                                -50.0,
                                20.0,
                            )
                        )
                        * np.exp(
                            1j
                            * phase_prediction
                        )
                    )

                    (
                        cr_mode_prediction,
                        ci_mode_prediction,
                    ) = predict_spectrum(
                        model,
                        np.asarray(
                            [alpha],
                            dtype=float,
                        ),
                        device,
                    )

                    cr_mode_prediction = float(
                        cr_mode_prediction[0]
                    )

                    ci_mode_prediction = float(
                        ci_mode_prediction[0]
                    )

                    prediction_raw = (
                        reconstruct_primitive_fields(
                            y=y,
                            alpha=alpha,
                            mach=mach,
                            cr=cr_mode_prediction,
                            ci=ci_mode_prediction,
                            pressure=(
                                pressure_prediction
                            ),
                            kappa=(
                                kappa_prediction
                            ),
                            phase_gradient=(
                                q_prediction
                            ),
                        )
                    )

                    comparison_mask = (
                        mode_comparison_mask(
                            y=y,
                            pressure_reference=(
                                classical["p"]
                            ),
                            pressure_prediction=(
                                prediction_raw["p"]
                            ),
                            y_max=float(
                                args.audit_y_max
                            ),
                            amplitude_floor=float(
                                args.amplitude_floor
                            ),
                        )
                    )

                    if (
                        comparison_mask.sum()
                        < 32
                    ):
                        raise RuntimeError(
                            "Insufficient comparison "
                            f"points for M={mach}, "
                            f"alpha={alpha}"
                        )

                    alignment = (
                        complex_alignment_factor(
                            classical["p"],
                            prediction_raw["p"],
                            comparison_mask,
                        )
                    )

                    prediction = {
                        field: (
                            alignment
                            * prediction_raw[field]
                        )
                        for field in [
                            "p",
                            "rho",
                            "u",
                            "v",
                        ]
                    }

                    errors = {
                        field: (
                            complex_relative_l2(
                                classical[field],
                                prediction[field],
                                comparison_mask,
                            )
                        )
                        for field in [
                            "p",
                            "rho",
                            "u",
                            "v",
                        ]
                    }

                    overlap_error = (
                        pressure_overlap_error(
                            classical["p"],
                            prediction["p"],
                            comparison_mask,
                        )
                    )

                    figure = make_mode_figure(
                        label=label,
                        reference_id=reference_id,
                        mach=mach,
                        alpha=alpha,
                        cr_reference=float(
                            mode["cr"].iloc[0]
                        ),
                        ci_reference=float(
                            mode["ci"].iloc[0]
                        ),
                        cr_prediction=(
                            cr_mode_prediction
                        ),
                        ci_prediction=(
                            ci_mode_prediction
                        ),
                        y=y,
                        classical=classical,
                        prediction=prediction,
                        errors=errors,
                        overlap_error=(
                            overlap_error
                        ),
                        plot_y_max=float(
                            args.plot_y_max
                        ),
                    )

                    alpha_name = (
                        f"{alpha:.6f}"
                        .replace(".", "p")
                    )

                    image_name = (
                        f"mode_{mode_number:03d}_"
                        f"{sanitize_filename(reference_id)}_"
                        f"alpha_{alpha_name}.png"
                    )

                    figure.savefig(
                        label_modes_dir
                        / image_name,
                        dpi=int(args.dpi),
                        bbox_inches="tight",
                    )

                    mach_pdf.savefig(
                        figure,
                        bbox_inches="tight",
                    )

                    complete_pdf.savefig(
                        figure,
                        bbox_inches="tight",
                    )

                    plt.close(figure)

                    per_mode_rows.append(
                        {
                            "label": label,
                            "Mach": mach,
                            "reference_id": (
                                reference_id
                            ),
                            "alpha": alpha,
                            "cr_reference": float(
                                mode["cr"].iloc[0]
                            ),
                            "ci_reference": float(
                                mode["ci"].iloc[0]
                            ),
                            "cr_pinn": (
                                cr_mode_prediction
                            ),
                            "ci_pinn": (
                                ci_mode_prediction
                            ),
                            "p_rel_l2": (
                                errors["p"]
                            ),
                            "rho_rel_l2": (
                                errors["rho"]
                            ),
                            "u_rel_l2": (
                                errors["u"]
                            ),
                            "v_rel_l2": (
                                errors["v"]
                            ),
                            "pressure_overlap_error": (
                                overlap_error
                            ),
                            "n_comparison_points": (
                                int(
                                    comparison_mask.sum()
                                )
                            ),
                        }
                    )

                    if (
                        mode_number == 1
                        or mode_number % 10 == 0
                        or mode_number
                        == number_of_modes
                    ):
                        print(
                            f"[{label}] "
                            f"{mode_number}/"
                            f"{number_of_modes} "
                            f"alpha={alpha:.6f} "
                            f"p_rel="
                            f"{errors['p']:.3e} "
                            f"overlap_error="
                            f"{overlap_error:.3e}",
                            flush=True,
                        )

            per_mode = pd.DataFrame(
                per_mode_rows
            )

            per_mode.to_csv(
                label_modes_dir
                / f"mode_metrics_{label}.csv",
                index=False,
            )

            total_modes_exported += int(
                len(per_mode)
            )

            global_summary_rows.append(
                {
                    "label": label,
                    "Mach": mach,
                    "checkpoint": str(
                        checkpoint_path
                    ),
                    "n_spectral_points": int(
                        len(spectrum)
                    ),
                    "n_modes": int(
                        len(per_mode)
                    ),
                    "cr_mae": float(
                        spectrum[
                            "cr_abs_error"
                        ].mean()
                    ),
                    "cr_max_abs": float(
                        spectrum[
                            "cr_abs_error"
                        ].max()
                    ),
                    "ci_mae": float(
                        spectrum[
                            "ci_abs_error"
                        ].mean()
                    ),
                    "ci_max_abs": float(
                        spectrum[
                            "ci_abs_error"
                        ].max()
                    ),
                    "p_rel_l2_mean": float(
                        per_mode[
                            "p_rel_l2"
                        ].mean()
                    ),
                    "rho_rel_l2_mean": float(
                        per_mode[
                            "rho_rel_l2"
                        ].mean()
                    ),
                    "u_rel_l2_mean": float(
                        per_mode[
                            "u_rel_l2"
                        ].mean()
                    ),
                    "v_rel_l2_mean": float(
                        per_mode[
                            "v_rel_l2"
                        ].mean()
                    ),
                    "pressure_overlap_error_mean": (
                        float(
                            per_mode[
                                "pressure_overlap_error"
                            ].mean()
                        )
                    ),
                    "mach_pdf": str(
                        mach_pdf_path
                    ),
                }
            )

    heatmap = pd.concat(
        heatmap_rows,
        ignore_index=True,
    )

    heatmap.to_csv(
        spectra_dir
        / "spectral_error_heatmap_data.csv",
        index=False,
    )

    make_error_heatmap(
        dataframe=heatmap,
        value_column="ci_abs_error",
        title=(
            r"Erreur absolue "
            r"$|c_i^{PINN}-c_i^{classique}|$"
        ),
        colorbar_label=(
            r"$|\Delta c_i|$"
        ),
        output_stem=(
            spectra_dir
            / "ci_abs_error_heatmap"
        ),
        dpi=int(args.dpi),
    )

    make_error_heatmap(
        dataframe=heatmap,
        value_column="cr_abs_error",
        title=(
            r"Erreur absolue "
            r"$|c_r^{PINN}-c_r^{classique}|$"
        ),
        colorbar_label=(
            r"$|\Delta c_r|$"
        ),
        output_stem=(
            spectra_dir
            / "cr_abs_error_heatmap"
        ),
        dpi=int(args.dpi),
    )

    summary = pd.DataFrame(
        global_summary_rows
    ).sort_values("Mach")

    summary.to_csv(
        output_dir
        / "summary_fixed_mach_assets.csv",
        index=False,
    )

    report = {
        "status": "COMPLETED",
        "manifest": str(manifest_path),
        "spectral_reference": str(
            spectral_path
        ),
        "modal_reference": str(
            modal_path
        ),
        "output_dir": str(output_dir),
        "number_of_pilots": int(
            len(summary)
        ),
        "labels": summary[
            "label"
        ].tolist(),
        "total_modes_exported": int(
            total_modes_exported
        ),
        "complete_pdf": str(
            complete_pdf_path
        ),
        "plot_y_max": float(
            args.plot_y_max
        ),
        "audit_y_max": float(
            args.audit_y_max
        ),
        "amplitude_floor": float(
            args.amplitude_floor
        ),
    }

    (
        output_dir
        / "asset_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("ASSET EXPORT SUMMARY")
    print(summary.to_string(index=False))
    print()
    print(json.dumps(report, indent=2))
    print()
    print(
        "FIXED-MACH S4M4 "
        "ASSET EXPORT: COMPLETED"
    )


if __name__ == "__main__":
    main()
