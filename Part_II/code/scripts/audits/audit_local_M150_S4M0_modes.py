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


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_model(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[KHSupersonicLocalPINN, dict[str, Any]]:
    config = checkpoint["config"]
    model_config = config["model"]

    model = KHSupersonicLocalPINN(
        mach=float(config["Mach"]),
        alpha_min=float(config["alpha_min"]),
        alpha_max=float(config["alpha_max"]),
        xi_max=float(model_config["xi_max"]),
        mapping_scale=float(model_config["mapping_scale"]),
        spectral_width=int(model_config["spectral_width"]),
        spectral_depth=int(model_config["spectral_depth"]),
        modal_width=int(model_config["modal_width"]),
        modal_depth=int(model_config["modal_depth"]),
        n_frequencies=int(model_config["n_frequencies"]),
        mode_experts=int(model_config["mode_experts"]),
        alpha_split=float(model_config["alpha_split"]),
        alpha_gate_width=float(model_config["alpha_gate_width"]),
        cr_min=float(model_config["cr_min"]),
        cr_max=float(model_config["cr_max"]),
        ci_floor=float(model_config["ci_floor"]),
        ci_max=float(model_config["ci_max"]),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()
    return model, config


def integrate_phase(
    y: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    order = np.argsort(y)
    y_sorted = y[order]
    q_sorted = q[order]

    phase_sorted = np.zeros_like(y_sorted)
    center = int(np.argmin(np.abs(y_sorted)))

    if center < len(y_sorted) - 1:
        increments = (
            0.5
            * (
                q_sorted[center:-1]
                + q_sorted[center + 1:]
            )
            * np.diff(y_sorted[center:])
        )
        phase_sorted[center + 1:] = np.cumsum(increments)

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
            -np.cumsum(increments[::-1])[::-1]
        )

    phase = np.empty_like(phase_sorted)
    phase[order] = phase_sorted
    return phase


def relative_l2(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(
            prediction[mask] - reference[mask]
        )
        / max(
            np.linalg.norm(reference[mask]),
            1.0e-14,
        )
    )


def overlap_error(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> float:
    denominator = (
        np.linalg.norm(reference[mask])
        * np.linalg.norm(prediction[mask])
    )

    if denominator <= 1.0e-14:
        return float("nan")

    overlap = abs(
        np.vdot(
            reference[mask],
            prediction[mask],
        )
    ) / denominator

    return float(
        1.0 - np.clip(overlap, 0.0, 1.0)
    )


def phase_rmse(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> float:
    phase_difference = np.angle(
        prediction[mask]
        * np.conjugate(reference[mask])
    )

    weights = np.abs(reference[mask]) ** 2
    weights /= max(float(weights.sum()), 1.0e-14)

    return float(
        np.sqrt(
            np.sum(
                weights * phase_difference**2
            )
        )
    )


def align_complex(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, complex]:
    denominator = np.vdot(
        prediction[mask],
        prediction[mask],
    )

    if abs(denominator) <= 1.0e-20:
        raise RuntimeError("Near-zero predicted pressure")

    factor = (
        np.vdot(
            prediction[mask],
            reference[mask],
        )
        / denominator
    )

    return factor * prediction, complex(factor)


def predict_mode(
    model: KHSupersonicLocalPINN,
    *,
    y: np.ndarray,
    alpha: float,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dtype = next(model.parameters()).dtype
    chunks = []

    with torch.inference_mode():
        for start in range(0, len(y), batch_size):
            stop = min(start + batch_size, len(y))

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
                output.detach().cpu().numpy()
            )

    output = np.concatenate(chunks, axis=0)

    if output.shape != (len(y), 3):
        raise RuntimeError(
            f"Unexpected output shape: {output.shape}"
        )

    return (
        output[:, 0],
        output[:, 1],
        output[:, 2],
    )


def predict_spectrum(
    model: KHSupersonicLocalPINN,
    *,
    alpha: float,
    device: torch.device,
) -> tuple[float, float]:
    dtype = next(model.parameters()).dtype

    alpha_tensor = torch.tensor(
        [[alpha]],
        dtype=dtype,
        device=device,
    )

    with torch.inference_mode():
        cr, ci = model.get_spectrum(alpha_tensor)

    return (
        float(cr.detach().cpu().item()),
        float(ci.detach().cpu().item()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--modal-reference",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--anchor-file",
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
        default=16384,
    )

    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    checkpoint = torch.load(
        resolve_path(args.checkpoint),
        map_location=device,
    )

    model, config = build_model(
        checkpoint,
        device,
    )

    reference = pd.read_csv(
        resolve_path(args.modal_reference),
        compression="gzip",
    )

    reference = reference[
        np.isclose(
            reference["Mach"],
            float(args.mach),
            rtol=0.0,
            atol=1.0e-12,
        )
    ].copy()

    anchors = pd.read_csv(
        resolve_path(args.anchor_file)
    )

    anchor_alphas = anchors["alpha"].to_numpy(float)

    rows = []

    grouped = reference.groupby(
        ["final_reference_id", "alpha"],
        sort=True,
    )

    for mode_index, (
        (reference_id, alpha),
        mode,
    ) in enumerate(grouped, start=1):
        mode = mode.sort_values("coordinate_index")

        alpha = float(alpha)
        y = mode["y"].to_numpy(float)

        p_reference = (
            mode["p_real"].to_numpy(float)
            + 1j * mode["p_imag"].to_numpy(float)
        )

        kappa_reference = mode["kappa"].to_numpy(float)
        q_reference = mode["q"].to_numpy(float)

        (
            kappa_prediction,
            q_prediction,
            log_amp_prediction,
        ) = predict_mode(
            model,
            y=y,
            alpha=alpha,
            device=device,
            batch_size=int(args.batch_size),
        )

        phase_prediction = integrate_phase(
            y,
            q_prediction,
        )

        p_prediction = (
            np.exp(
                np.clip(
                    log_amp_prediction,
                    -50.0,
                    20.0,
                )
            )
            * np.exp(1j * phase_prediction)
        )

        amplitude_reference = np.abs(p_reference)
        amplitude_max = max(
            float(amplitude_reference.max()),
            1.0e-14,
        )

        amplitude_mask = (
            amplitude_reference
            >= 1.0e-6 * amplitude_max
        )

        core20 = amplitude_mask & (np.abs(y) <= 20.0)
        core40 = amplitude_mask & (np.abs(y) <= 40.0)

        if int(core20.sum()) < 32:
            raise RuntimeError(
                f"Too few core points at alpha={alpha}"
            )

        p_aligned, factor = align_complex(
            p_reference,
            p_prediction,
            core20,
        )

        cr_prediction, ci_prediction = predict_spectrum(
            model,
            alpha=alpha,
            device=device,
        )

        cr_reference = float(mode["cr"].iloc[0])
        ci_reference = float(mode["ci"].iloc[0])

        row = {
            "final_reference_id": str(reference_id),
            "Mach": float(args.mach),
            "alpha": alpha,
            "is_anchor": bool(
                np.any(
                    np.isclose(
                        alpha,
                        anchor_alphas,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                )
            ),
            "cr_reference": cr_reference,
            "ci_reference": ci_reference,
            "cr_pinn": cr_prediction,
            "ci_pinn": ci_prediction,
            "cr_abs_error": abs(
                cr_prediction - cr_reference
            ),
            "ci_abs_error": abs(
                ci_prediction - ci_reference
            ),
            "complex_spectral_error": float(
                np.hypot(
                    cr_prediction - cr_reference,
                    ci_prediction - ci_reference,
                )
            ),
            "p_rel_l2_core20": relative_l2(
                p_reference,
                p_aligned,
                core20,
            ),
            "p_rel_l2_core40": relative_l2(
                p_reference,
                p_aligned,
                core40,
            ),
            "overlap_error_core20": overlap_error(
                p_reference,
                p_aligned,
                core20,
            ),
            "overlap_error_core40": overlap_error(
                p_reference,
                p_aligned,
                core40,
            ),
            "envelope_rel_l2_core40": relative_l2(
                np.abs(p_reference),
                np.abs(p_aligned),
                core40,
            ),
            "phase_rmse_core40": phase_rmse(
                p_reference,
                p_aligned,
                core40,
            ),
            "kappa_rel_l2_core40": relative_l2(
                kappa_reference,
                kappa_prediction,
                core40,
            ),
            "q_rel_l2_core40": relative_l2(
                q_reference,
                q_prediction,
                core40,
            ),
            "alignment_real": factor.real,
            "alignment_imag": factor.imag,
            "n_core20": int(core20.sum()),
            "n_core40": int(core40.sum()),
        }

        rows.append(row)

        print(
            f"{mode_index:3d}/{grouped.ngroups:3d} "
            f"alpha={alpha:.6f} "
            f"spec={row['complex_spectral_error']:.3e} "
            f"p40={row['p_rel_l2_core40']:.3e} "
            f"ov40={row['overlap_error_core40']:.3e}",
            flush=True,
        )

    results = pd.DataFrame(rows).sort_values("alpha")

    results_path = output_dir / "s4m0_modal_audit.csv"
    results.to_csv(results_path, index=False)

    off_anchor = results[~results["is_anchor"]].copy()

    summary = {
        "status": "COMPLETED",
        "Mach": float(args.mach),
        "checkpoint": str(resolve_path(args.checkpoint)),
        "n_modes": int(len(results)),
        "n_anchors": int(results["is_anchor"].sum()),
        "n_off_anchor": int(len(off_anchor)),
        "spectral_complex_error_mean": float(
            results["complex_spectral_error"].mean()
        ),
        "spectral_complex_error_max": float(
            results["complex_spectral_error"].max()
        ),
        "p_rel_l2_core20_mean": float(
            off_anchor["p_rel_l2_core20"].mean()
        ),
        "p_rel_l2_core40_mean": float(
            off_anchor["p_rel_l2_core40"].mean()
        ),
        "p_rel_l2_core40_max": float(
            off_anchor["p_rel_l2_core40"].max()
        ),
        "overlap_error_core20_mean": float(
            off_anchor["overlap_error_core20"].mean()
        ),
        "overlap_error_core40_mean": float(
            off_anchor["overlap_error_core40"].mean()
        ),
        "overlap_error_core40_max": float(
            off_anchor["overlap_error_core40"].max()
        ),
        "envelope_rel_l2_core40_mean": float(
            off_anchor["envelope_rel_l2_core40"].mean()
        ),
        "phase_rmse_core40_mean": float(
            off_anchor["phase_rmse_core40"].mean()
        ),
        "kappa_rel_l2_core40_mean": float(
            off_anchor["kappa_rel_l2_core40"].mean()
        ),
        "q_rel_l2_core40_mean": float(
            off_anchor["q_rel_l2_core40"].mean()
        ),
        "results_csv": str(results_path),
    }

    summary_path = output_dir / "s4m0_modal_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(json.dumps(summary, indent=2))
    print()
    print("S4M0 MODAL AUDIT: COMPLETED")


if __name__ == "__main__":
    main()
