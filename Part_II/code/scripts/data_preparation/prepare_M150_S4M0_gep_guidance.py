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


def resolve_path(path: str | Path) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def integrate_phase_from_center(
    y: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    order = np.argsort(y)

    y_sorted = np.asarray(
        y[order],
        dtype=float,
    )

    q_sorted = np.asarray(
        q[order],
        dtype=float,
    )

    phase_sorted = np.zeros_like(
        y_sorted,
        dtype=float,
    )

    center = int(
        np.argmin(np.abs(y_sorted))
    )

    if center < len(y_sorted) - 1:
        increments = (
            0.5
            * (
                q_sorted[center:-1]
                + q_sorted[center + 1:]
            )
            * np.diff(y_sorted[center:])
        )

        phase_sorted[
            center + 1:
        ] = np.cumsum(increments)

    if center > 0:
        increments = (
            0.5
            * (
                q_sorted[:center]
                + q_sorted[1:center + 1]
            )
            * np.diff(y_sorted[:center + 1])
        )

        phase_sorted[:center] = (
            -np.cumsum(
                increments[::-1]
            )[::-1]
        )

    phase = np.empty_like(
        phase_sorted
    )

    phase[order] = phase_sorted

    return phase


def build_model(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[KHSupersonicLocalPINN, dict]:
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

    return model, config


def predict_spectrum(
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


def predict_mode(
    model: KHSupersonicLocalPINN,
    *,
    y: np.ndarray,
    alpha: float,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dtype = next(model.parameters()).dtype

    chunks: list[np.ndarray] = []

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
                dtype=dtype,
                device=device,
            ).reshape(-1, 1)

            xi_tensor = y_to_xi(
                y_tensor,
                model.get_mapping_scale(),
            )

            alpha_tensor = torch.full_like(
                xi_tensor,
                float(alpha),
            )

            output = model(
                xi_tensor,
                alpha_tensor,
            )

            chunks.append(
                output.detach()
                .cpu()
                .numpy()
            )

    output = np.concatenate(
        chunks,
        axis=0,
    )

    if output.shape != (len(y), 3):
        raise RuntimeError(
            f"Unexpected modal output shape: {output.shape}"
        )

    return (
        output[:, 0],
        output[:, 1],
        output[:, 2],
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
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
        "--spectral-anchor-file",
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
        default="cpu",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
    )

    args = parser.parse_args()

    checkpoint_path = resolve_path(
        args.checkpoint
    )

    spectral_path = resolve_path(
        args.spectral_reference
    )

    modal_path = resolve_path(
        args.modal_reference
    )

    anchor_path = resolve_path(
        args.spectral_anchor_file
    )

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(args.device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model, config = build_model(
        checkpoint,
        device,
    )

    if not np.isclose(
        float(config["Mach"]),
        float(args.mach),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Checkpoint Mach does not match "
            f"requested Mach={args.mach}"
        )

    spectral = pd.read_csv(
        spectral_path
    )

    spectral = spectral[
        np.isclose(
            spectral["Mach"],
            float(args.mach),
            rtol=0.0,
            atol=1.0e-12,
        )
    ].sort_values("alpha").reset_index(
        drop=True
    )

    modal = pd.read_csv(
        modal_path,
        compression="gzip",
    )

    modal = modal[
        np.isclose(
            modal["Mach"],
            float(args.mach),
            rtol=0.0,
            atol=1.0e-12,
        )
    ].sort_values(
        [
            "alpha",
            "coordinate_index",
        ]
    )

    anchors = pd.read_csv(
        anchor_path
    ).sort_values("alpha")

    if len(anchors) != 4:
        raise RuntimeError(
            f"Expected four anchors, found {len(anchors)}"
        )

    alpha_values = spectral[
        "alpha"
    ].to_numpy(float)

    cr_pinn, ci_pinn = predict_spectrum(
        model,
        alpha_values,
        device,
    )

    spectral_output = spectral.copy()

    spectral_output[
        "cr_pinn_S4M0"
    ] = cr_pinn

    spectral_output[
        "ci_pinn_S4M0"
    ] = ci_pinn

    spectral_output[
        "cr_abs_error_S4M0"
    ] = np.abs(
        spectral_output["cr"]
        - spectral_output["cr_pinn_S4M0"]
    )

    spectral_output[
        "ci_abs_error_S4M0"
    ] = np.abs(
        spectral_output["ci"]
        - spectral_output["ci_pinn_S4M0"]
    )

    anchor_alphas = anchors[
        "alpha"
    ].to_numpy(float)

    spectral_output[
        "is_spectral_anchor"
    ] = [
        bool(
            np.any(
                np.isclose(
                    float(alpha),
                    anchor_alphas,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
        )
        for alpha in spectral_output["alpha"]
    ]

    spectral_csv = (
        output_dir
        / "pinn_S4M0_spectral_guidance.csv"
    )

    spectral_output.to_csv(
        spectral_csv,
        index=False,
    )

    reference_ids: list[str] = []
    mode_alpha: list[float] = []
    mode_cr_pinn: list[float] = []
    mode_ci_pinn: list[float] = []
    mode_cr_reference: list[float] = []
    mode_ci_reference: list[float] = []
    mode_ptr = [0]

    y_flat: list[np.ndarray] = []
    kappa_flat: list[np.ndarray] = []
    q_flat: list[np.ndarray] = []
    log_amp_flat: list[np.ndarray] = []
    phase_flat: list[np.ndarray] = []
    p_real_flat: list[np.ndarray] = []
    p_imag_flat: list[np.ndarray] = []

    grouped = modal.groupby(
        [
            "final_reference_id",
            "alpha",
        ],
        sort=True,
    )

    number_of_modes = grouped.ngroups

    for mode_number, (
        (
            reference_id,
            alpha,
        ),
        group,
    ) in enumerate(
        grouped,
        start=1,
    ):
        group = group.sort_values(
            "coordinate_index"
        )

        alpha = float(alpha)

        y = group["y"].to_numpy(float)

        (
            kappa,
            q,
            log_amp,
        ) = predict_mode(
            model,
            y=y,
            alpha=alpha,
            device=device,
            batch_size=int(args.batch_size),
        )

        phase = integrate_phase_from_center(
            y,
            q,
        )

        pressure = (
            np.exp(
                np.clip(
                    log_amp,
                    -50.0,
                    20.0,
                )
            )
            * np.exp(1j * phase)
        )

        cr_mode, ci_mode = predict_spectrum(
            model,
            np.asarray([alpha]),
            device,
        )

        reference_ids.append(
            str(reference_id)
        )

        mode_alpha.append(alpha)

        mode_cr_pinn.append(
            float(cr_mode[0])
        )

        mode_ci_pinn.append(
            float(ci_mode[0])
        )

        mode_cr_reference.append(
            float(group["cr"].iloc[0])
        )

        mode_ci_reference.append(
            float(group["ci"].iloc[0])
        )

        y_flat.append(y)
        kappa_flat.append(kappa)
        q_flat.append(q)
        log_amp_flat.append(log_amp)
        phase_flat.append(phase)
        p_real_flat.append(pressure.real)
        p_imag_flat.append(pressure.imag)

        mode_ptr.append(
            mode_ptr[-1] + len(y)
        )

        if (
            mode_number == 1
            or mode_number % 10 == 0
            or mode_number == number_of_modes
        ):
            print(
                f"{mode_number}/{number_of_modes} "
                f"alpha={alpha:.6f} "
                f"cPINN={cr_mode[0]:.6f}"
                f"+i{ci_mode[0]:.6f}",
                flush=True,
            )

    modal_npz = (
        output_dir
        / "pinn_S4M0_modal_guidance.npz"
    )

    np.savez_compressed(
        modal_npz,
        Mach=np.asarray(
            [float(args.mach)],
            dtype=float,
        ),
        reference_id=np.asarray(
            reference_ids,
            dtype=str,
        ),
        alpha=np.asarray(
            mode_alpha,
            dtype=float,
        ),
        cr_pinn=np.asarray(
            mode_cr_pinn,
            dtype=float,
        ),
        ci_pinn=np.asarray(
            mode_ci_pinn,
            dtype=float,
        ),
        cr_reference=np.asarray(
            mode_cr_reference,
            dtype=float,
        ),
        ci_reference=np.asarray(
            mode_ci_reference,
            dtype=float,
        ),
        spectral_anchor_alpha=np.asarray(
            anchor_alphas,
            dtype=float,
        ),
        mode_ptr=np.asarray(
            mode_ptr,
            dtype=np.int64,
        ),
        y=np.concatenate(y_flat),
        kappa_pinn=np.concatenate(
            kappa_flat
        ),
        q_pinn=np.concatenate(q_flat),
        log_amp_pinn=np.concatenate(
            log_amp_flat
        ),
        phase_pinn=np.concatenate(
            phase_flat
        ),
        p_real_pinn=np.concatenate(
            p_real_flat
        ),
        p_imag_pinn=np.concatenate(
            p_imag_flat
        ),
    )

    report = {
        "status": "COMPLETED",
        "experiment": "M150_S4M0_GEP_GUIDANCE",
        "Mach": float(args.mach),
        "checkpoint": str(checkpoint_path),
        "spectral_anchor_count": 4,
        "modal_anchor_count": 0,
        "spectral_anchor_alphas": (
            anchor_alphas.tolist()
        ),
        "number_of_spectral_points": int(
            len(spectral_output)
        ),
        "number_of_modal_guidance_modes": int(
            number_of_modes
        ),
        "number_of_modal_points": int(
            mode_ptr[-1]
        ),
        "spectral_guidance_csv": str(
            spectral_csv
        ),
        "modal_guidance_npz": str(
            modal_npz
        ),
    }

    report_path = (
        output_dir
        / "pinn_S4M0_guidance_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(json.dumps(report, indent=2))
    print()
    print("S4M0 GEP GUIDANCE: COMPLETED")


if __name__ == "__main__":
    main()
