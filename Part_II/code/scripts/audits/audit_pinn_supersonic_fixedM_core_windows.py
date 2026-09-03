#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1.0e-30


def load_array(data: np.lib.npyio.NpzFile, names: list[str]) -> np.ndarray:
    for name in names:
        if name in data.files:
            return np.asarray(data[name], dtype=np.float64)
    raise KeyError(
        f"None of the expected keys exists: {names}\n"
        f"Available keys: {sorted(data.files)}"
    )


def load_complex(
    data: np.lib.npyio.NpzFile,
    real_names: list[str],
    imag_names: list[str],
) -> np.ndarray:
    real = load_array(data, real_names)
    imag = load_array(data, imag_names)

    if real.shape != imag.shape:
        raise ValueError(
            f"Real/imaginary shape mismatch: {real.shape} versus {imag.shape}"
        )

    return real + 1j * imag


def relative_l2(reference: np.ndarray, prediction: np.ndarray) -> float:
    numerator = np.sum(np.abs(prediction - reference) ** 2)
    denominator = np.sum(np.abs(reference) ** 2)
    return float(np.sqrt(numerator / max(denominator, EPS)))


def finite_difference_by_alpha(
    alpha: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    """
    Compute dp/dy independently for every alpha branch.

    This is a finite-difference diagnostic. It is not the final autodiff
    q-definition residual that will be used during PINN training.
    """
    derivative = np.full(p.shape, np.nan + 1j * np.nan, dtype=np.complex128)

    for alpha_value in np.unique(alpha):
        indices = np.flatnonzero(
            np.isclose(alpha, alpha_value, atol=1.0e-12, rtol=0.0)
        )

        if indices.size < 3:
            raise ValueError(
                f"Only {indices.size} rows for alpha={alpha_value}; "
                "at least three are required."
            )

        order = np.argsort(y[indices])
        sorted_indices = indices[order]
        y_sorted = y[sorted_indices]
        p_sorted = p[sorted_indices]

        dy = np.diff(y_sorted)
        if np.any(dy <= 0.0):
            raise ValueError(
                f"Non-increasing y grid for alpha={alpha_value}. "
                f"Minimum dy={np.min(dy)}"
            )

        dp_dy_sorted = np.gradient(
            p_sorted,
            y_sorted,
            edge_order=2,
        )

        derivative[sorted_indices] = dp_dy_sorted

    if not np.all(np.isfinite(derivative.real)):
        raise RuntimeError("Non-finite real values in finite-difference dp/dy.")

    if not np.all(np.isfinite(derivative.imag)):
        raise RuntimeError("Non-finite imaginary values in finite-difference dp/dy.")

    return derivative


def compute_metrics(
    mask: np.ndarray,
    p_ref: np.ndarray,
    p_pred: np.ndarray,
    q_ref: np.ndarray,
    q_pred: np.ndarray,
    dp_ref_dy: np.ndarray,
    dp_pred_dy: np.ndarray,
    total_p_error_energy: float,
    total_q_error_energy: float,
) -> dict[str, float | int]:
    if not np.any(mask):
        raise ValueError("Empty audit mask.")

    p_ref_m = p_ref[mask]
    p_pred_m = p_pred[mask]
    q_ref_m = q_ref[mask]
    q_pred_m = q_pred[mask]
    dp_ref_m = dp_ref_dy[mask]
    dp_pred_m = dp_pred_dy[mask]

    p_error = p_pred_m - p_ref_m
    q_error = q_pred_m - q_ref_m

    p_error_energy = float(np.sum(np.abs(p_error) ** 2))
    q_error_energy = float(np.sum(np.abs(q_error) ** 2))

    return {
        "n": int(np.count_nonzero(mask)),
        "p_rel_l2": relative_l2(p_ref_m, p_pred_m),
        "q_rel_l2": relative_l2(q_ref_m, q_pred_m),
        "p_max_abs_err": float(np.max(np.abs(p_error))),
        "q_max_abs_err": float(np.max(np.abs(q_error))),
        "p_rmse": float(np.sqrt(np.mean(np.abs(p_error) ** 2))),
        "q_rmse": float(np.sqrt(np.mean(np.abs(q_error) ** 2))),
        "p_error_energy_fraction": (
            p_error_energy / max(total_p_error_energy, EPS)
        ),
        "q_error_energy_fraction": (
            q_error_energy / max(total_q_error_energy, EPS)
        ),
        # Finite-difference consistency of the reference dataset.
        "qdef_reference_rel_l2": relative_l2(q_ref_m, dp_ref_m),
        # Consistency of the two independent PINN outputs p and q.
        "qdef_prediction_rel_l2": relative_l2(q_pred_m, dp_pred_m),
        # Difference between dp_pred/dy and the target q_ref.
        "dp_pred_vs_q_ref_rel_l2": relative_l2(q_ref_m, dp_pred_m),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--windows",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 2000.0],
    )
    parser.add_argument("--core-window", type=float, default=20.0)
    parser.add_argument("--edge-count", type=int, default=4)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    prediction_file = args.run_dir / "modal_predictions_all_rows.npz"
    if not prediction_file.exists():
        raise FileNotFoundError(prediction_file)

    data = np.load(prediction_file, allow_pickle=True)

    alpha = load_array(data, ["row_alpha", "alpha"])
    y = load_array(data, ["y", "row_y"])

    p_ref = load_complex(
        data,
        ["p_ref_real", "p_real", "reference_p_real"],
        ["p_ref_imag", "p_imag", "reference_p_imag"],
    )
    q_ref = load_complex(
        data,
        ["q_ref_real", "q_real", "reference_q_real"],
        ["q_ref_imag", "q_imag", "reference_q_imag"],
    )
    p_pred = load_complex(
        data,
        ["p_pred_real", "prediction_p_real"],
        ["p_pred_imag", "prediction_p_imag"],
    )
    q_pred = load_complex(
        data,
        ["q_pred_real", "prediction_q_real"],
        ["q_pred_imag", "prediction_q_imag"],
    )

    expected_shape = alpha.shape
    arrays = {
        "y": y,
        "p_ref": p_ref,
        "q_ref": q_ref,
        "p_pred": p_pred,
        "q_pred": q_pred,
    }

    for name, array in arrays.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {name}: {array.shape}, "
                f"expected {expected_shape}"
            )

    finite = (
        np.isfinite(alpha)
        & np.isfinite(y)
        & np.isfinite(p_ref.real)
        & np.isfinite(p_ref.imag)
        & np.isfinite(q_ref.real)
        & np.isfinite(q_ref.imag)
        & np.isfinite(p_pred.real)
        & np.isfinite(p_pred.imag)
        & np.isfinite(q_pred.real)
        & np.isfinite(q_pred.imag)
    )

    if not np.all(finite):
        raise RuntimeError(
            f"Found {np.count_nonzero(~finite)} non-finite prediction rows."
        )

    unique_alpha = np.unique(alpha)

    print("[audit-core] prediction file:", prediction_file)
    print("[audit-core] rows:", len(alpha))
    print("[audit-core] unique alpha:", len(unique_alpha))
    print("[audit-core] alpha range:", float(alpha.min()), float(alpha.max()))
    print("[audit-core] y range:", float(y.min()), float(y.max()))

    print("\n[audit-core] computing finite-difference derivatives")
    dp_ref_dy = finite_difference_by_alpha(alpha, y, p_ref)
    dp_pred_dy = finite_difference_by_alpha(alpha, y, p_pred)

    total_p_error_energy = float(np.sum(np.abs(p_pred - p_ref) ** 2))
    total_q_error_energy = float(np.sum(np.abs(q_pred - q_ref) ** 2))

    window_rows: list[dict] = []

    all_metrics = compute_metrics(
        np.ones(alpha.shape, dtype=bool),
        p_ref,
        p_pred,
        q_ref,
        q_pred,
        dp_ref_dy,
        dp_pred_dy,
        total_p_error_energy,
        total_q_error_energy,
    )
    window_rows.append(
        {
            "scope": "all",
            "y_window": np.inf,
            **all_metrics,
        }
    )

    for window in sorted(set(args.windows)):
        mask = np.abs(y) <= window
        metrics = compute_metrics(
            mask,
            p_ref,
            p_pred,
            q_ref,
            q_pred,
            dp_ref_dy,
            dp_pred_dy,
            total_p_error_energy,
            total_q_error_energy,
        )

        window_rows.append(
            {
                "scope": f"|y|<={window:g}",
                "y_window": window,
                **metrics,
            }
        )

    window_df = pd.DataFrame(window_rows)
    window_csv = args.run_dir / "modal_error_by_y_window.csv"
    window_df.to_csv(window_csv, index=False)

    print("\n[audit-core] global metrics by y window")
    display_columns = [
        "scope",
        "n",
        "p_rel_l2",
        "q_rel_l2",
        "qdef_reference_rel_l2",
        "qdef_prediction_rel_l2",
        "p_error_energy_fraction",
        "q_error_energy_fraction",
    ]
    print(
        window_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6e}",
        )
    )

    per_alpha_rows: list[dict] = []

    windows_with_core = sorted(set(args.windows + [args.core_window]))

    for alpha_value in unique_alpha:
        alpha_mask = np.isclose(
            alpha,
            alpha_value,
            atol=1.0e-12,
            rtol=0.0,
        )

        for window in windows_with_core:
            mask = alpha_mask & (np.abs(y) <= window)

            if not np.any(mask):
                continue

            metrics = compute_metrics(
                mask,
                p_ref,
                p_pred,
                q_ref,
                q_pred,
                dp_ref_dy,
                dp_pred_dy,
                total_p_error_energy,
                total_q_error_energy,
            )

            per_alpha_rows.append(
                {
                    "alpha": float(alpha_value),
                    "y_window": float(window),
                    **metrics,
                }
            )

    per_alpha_df = pd.DataFrame(per_alpha_rows)
    per_alpha_csv = args.run_dir / "modal_error_by_alpha_y_window.csv"
    per_alpha_df.to_csv(per_alpha_csv, index=False)

    core_df = per_alpha_df[
        np.isclose(
            per_alpha_df["y_window"],
            args.core_window,
            atol=1.0e-12,
            rtol=0.0,
        )
    ].copy()

    print(f"\n[audit-core] worst p errors for |y|<={args.core_window:g}")
    print(
        core_df.sort_values("p_rel_l2", ascending=False)
        .head(args.top)[
            [
                "alpha",
                "n",
                "p_rel_l2",
                "q_rel_l2",
                "qdef_prediction_rel_l2",
                "p_max_abs_err",
                "q_max_abs_err",
            ]
        ]
        .to_string(index=False, float_format=lambda value: f"{value:.6e}")
    )

    print(f"\n[audit-core] worst q errors for |y|<={args.core_window:g}")
    print(
        core_df.sort_values("q_rel_l2", ascending=False)
        .head(args.top)[
            [
                "alpha",
                "n",
                "p_rel_l2",
                "q_rel_l2",
                "qdef_prediction_rel_l2",
                "p_max_abs_err",
                "q_max_abs_err",
            ]
        ]
        .to_string(index=False, float_format=lambda value: f"{value:.6e}")
    )

    print(
        f"\n[audit-core] worst q=dp/dy consistency for "
        f"|y|<={args.core_window:g}"
    )
    print(
        core_df.sort_values("qdef_prediction_rel_l2", ascending=False)
        .head(args.top)[
            [
                "alpha",
                "n",
                "qdef_reference_rel_l2",
                "qdef_prediction_rel_l2",
                "dp_pred_vs_q_ref_rel_l2",
            ]
        ]
        .to_string(index=False, float_format=lambda value: f"{value:.6e}")
    )

    edge_count = max(1, min(args.edge_count, len(unique_alpha) // 2))
    edge_alphas = np.concatenate(
        [
            unique_alpha[:edge_count],
            unique_alpha[-edge_count:],
        ]
    )

    edge_df = core_df[
        core_df["alpha"].isin(edge_alphas)
    ].sort_values("alpha")

    edge_csv = args.run_dir / "modal_error_edge_alpha_core.csv"
    edge_df.to_csv(edge_csv, index=False)

    print(
        f"\n[audit-core] edge-alpha summary for "
        f"|y|<={args.core_window:g}"
    )
    print(
        edge_df[
            [
                "alpha",
                "p_rel_l2",
                "q_rel_l2",
                "qdef_reference_rel_l2",
                "qdef_prediction_rel_l2",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6e}")
    )

    print("\n[audit-core] wrote:")
    print(" ", window_csv)
    print(" ", per_alpha_csv)
    print(" ", edge_csv)


if __name__ == "__main__":
    main()
