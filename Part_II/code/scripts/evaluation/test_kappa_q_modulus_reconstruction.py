#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, solve_ivp


DEFAULT_MACH = 1.4
DEFAULT_ALPHA = 0.18125
DEFAULT_CR = 0.3153193678544234
DEFAULT_CI = 0.0427209194542578


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the non-oscillatory supersonic reconstruction based on "
            "(kappa, q, |p|) at a fixed known eigenvalue."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--Mach", type=float, default=DEFAULT_MACH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--cr", type=float, default=DEFAULT_CR)
    parser.add_argument("--ci", type=float, default=DEFAULT_CI)
    parser.add_argument("--Ly", type=float, default=500.0)
    parser.add_argument("--matching-y", type=float, default=1.0)
    parser.add_argument("--output-dy", type=float, default=0.05)
    parser.add_argument("--max-step", type=float, default=0.25)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--method", choices=("RK45", "DOP853"), default="DOP853")
    parser.add_argument("--core-y", type=float, default=20.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to classic_supersonic/reproducibility/results/kappa_q_modulus_test_M140_a018125.",
    )
    parser.add_argument(
        "--reference-parquet",
        type=Path,
        default=None,
        help=(
            "Optional old modal reference. By default the script tries "
            "experiments/modal_reconstruction/support/modal/supersonic_reference_v2_modal_raw.parquet."
        ),
    )
    return parser.parse_args()


def base_flow(y: np.ndarray | float) -> np.ndarray | float:
    return np.tanh(y)


def base_flow_prime(y: np.ndarray | float) -> np.ndarray | float:
    u = np.tanh(y)
    return 1.0 - u * u


def coefficients(
    y: np.ndarray | float,
    *,
    Mach: float,
    c: complex,
) -> tuple[np.ndarray | complex, np.ndarray | complex]:
    u = base_flow(y)
    up = base_flow_prime(y)
    P = -2.0 * up / (u - c)
    R = 1.0 - Mach**2 * (u - c) ** 2
    return P, R


def asymptotic_gamma(
    *,
    side: str,
    Mach: float,
    alpha: float,
    c: complex,
) -> complex:
    if side == "left":
        u_inf = -1.0
        required_sign = 1.0
    elif side == "right":
        u_inf = 1.0
        required_sign = -1.0
    else:
        raise ValueError(f"Unknown side: {side}")

    R_inf = 1.0 - Mach**2 * (u_inf - c) ** 2
    root = alpha * np.sqrt(complex(R_inf))

    if required_sign > 0.0:
        if root.real < 0.0 or (abs(root.real) < 1.0e-14 and root.imag < 0.0):
            root = -root
    else:
        if root.real > 0.0 or (abs(root.real) < 1.0e-14 and root.imag > 0.0):
            root = -root

    return complex(root)


def rhs_kappa_q_modulus(
    y: float,
    state: np.ndarray,
    *,
    Mach: float,
    alpha: float,
    c: complex,
) -> np.ndarray:
    kappa, q, modulus = state
    P, R = coefficients(y, Mach=Mach, c=c)
    Pr = float(np.real(P))
    Pi = float(np.imag(P))
    Rr = float(np.real(R))
    Ri = float(np.imag(R))

    dkappa = (
        -kappa**2
        + q**2
        - Pr * kappa
        + Pi * q
        + alpha**2 * Rr
    )
    dq = (
        -2.0 * kappa * q
        - Pi * kappa
        - Pr * q
        + alpha**2 * Ri
    )
    dmodulus = kappa * modulus
    return np.asarray([dkappa, dq, dmodulus], dtype=float)


def make_eval_grid(start: float, stop: float, dy: float) -> np.ndarray:
    n = int(round(abs(stop - start) / dy)) + 1
    return np.linspace(start, stop, max(n, 2), dtype=float)


def integrate_branch(
    *,
    side: str,
    Mach: float,
    alpha: float,
    c: complex,
    Ly: float,
    matching_y: float,
    output_dy: float,
    max_step: float,
    rtol: float,
    atol: float,
    method: str,
) -> dict[str, Any]:
    if side == "left":
        start, stop = -Ly, matching_y
    elif side == "right":
        start, stop = Ly, matching_y
    else:
        raise ValueError(side)

    gamma0 = asymptotic_gamma(
        side=side,
        Mach=Mach,
        alpha=alpha,
        c=c,
    )
    initial = np.asarray([gamma0.real, gamma0.imag, 1.0], dtype=float)
    t_eval = make_eval_grid(start, stop, output_dy)

    solution = solve_ivp(
        lambda y, z: rhs_kappa_q_modulus(
            y,
            z,
            Mach=Mach,
            alpha=alpha,
            c=c,
        ),
        (start, stop),
        initial,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    if not solution.success:
        raise RuntimeError(f"{side} integration failed: {solution.message}")
    if not np.all(np.isfinite(solution.t)):
        raise RuntimeError(f"{side} integration produced non-finite coordinates.")
    if not np.all(np.isfinite(solution.y)):
        raise RuntimeError(f"{side} integration produced non-finite states.")

    return {
        "side": side,
        "gamma0": gamma0,
        "y": np.asarray(solution.t, dtype=float),
        "kappa": np.asarray(solution.y[0], dtype=float),
        "q": np.asarray(solution.y[1], dtype=float),
        "modulus": np.asarray(solution.y[2], dtype=float),
        "nfev": int(solution.nfev),
        "njev": int(solution.njev),
        "nlu": int(solution.nlu),
    }


def reconstruct_mode(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    y_left = left["y"]
    k_left = left["kappa"]
    q_left = left["q"]
    a_left = left["modulus"]

    # The right branch was integrated from +Ly down to matching_y.
    y_right = right["y"][::-1]
    k_right = right["kappa"][::-1]
    q_right = right["q"][::-1]
    a_right = right["modulus"][::-1]

    delta_kappa = float(k_left[-1] - k_right[0])
    delta_q = float(q_left[-1] - q_right[0])

    if not np.isfinite(a_right[0]) or abs(a_right[0]) < 1.0e-300:
        raise RuntimeError("Right modulus cannot be rescaled at the matching point.")

    amplitude_scale_right = float(a_left[-1] / a_right[0])
    a_right = amplitude_scale_right * a_right

    # Phase is not part of the shooting state. It is reconstructed afterwards
    # by quadrature of q = phi', with phi(matching_y)=0.
    phi_left = cumulative_trapezoid(q_left, y_left, initial=0.0)
    phi_left = phi_left - phi_left[-1]
    phi_right = cumulative_trapezoid(q_right, y_right, initial=0.0)

    # Avoid duplicating the matching point.
    y = np.concatenate([y_left, y_right[1:]])
    kappa = np.concatenate([k_left, k_right[1:]])
    q = np.concatenate([q_left, q_right[1:]])
    modulus_raw = np.concatenate([a_left, a_right[1:]])
    phi = np.concatenate([phi_left, phi_right[1:]])
    branch = np.concatenate(
        [
            np.full(y_left.size, "left", dtype=object),
            np.full(y_right.size - 1, "right", dtype=object),
        ]
    )

    if np.any(modulus_raw <= 0.0):
        minimum = float(np.min(modulus_raw))
        raise RuntimeError(
            f"The directly integrated modulus became non-positive (min={minimum:.3e})."
        )

    normalization = float(np.max(modulus_raw))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise RuntimeError("Invalid global modulus normalization.")

    modulus = modulus_raw / normalization
    pressure = modulus * np.exp(1j * phi)
    gamma = kappa + 1j * q
    pressure_prime_from_gamma = gamma * pressure

    frame = pd.DataFrame(
        {
            "y": y,
            "branch": branch,
            "kappa": kappa,
            "q": q,
            "modulus_raw": modulus_raw,
            "modulus": modulus,
            "phi_postprocessed": phi,
            "p_real": pressure.real,
            "p_imag": pressure.imag,
            "pprime_real_from_gamma": pressure_prime_from_gamma.real,
            "pprime_imag_from_gamma": pressure_prime_from_gamma.imag,
        }
    )

    diagnostics = {
        "delta_kappa_at_match": delta_kappa,
        "delta_q_at_match": delta_q,
        "complex_gamma_mismatch_at_match": float(
            np.hypot(delta_kappa, delta_q)
        ),
        "amplitude_scale_right": amplitude_scale_right,
        "global_modulus_normalization": normalization,
    }
    return frame, diagnostics


def numerical_residuals(
    frame: pd.DataFrame,
    *,
    Mach: float,
    alpha: float,
    c: complex,
    matching_y: float,
    core_y: float,
) -> dict[str, float]:
    y = frame["y"].to_numpy(float)
    pressure = (
        frame["p_real"].to_numpy(float)
        + 1j * frame["p_imag"].to_numpy(float)
    )
    gamma = (
        frame["kappa"].to_numpy(float)
        + 1j * frame["q"].to_numpy(float)
    )

    p_y_fd = np.gradient(pressure, y, edge_order=2)
    p_yy_fd = np.gradient(p_y_fd, y, edge_order=2)
    p_y_gamma = gamma * pressure

    P, R = coefficients(y, Mach=Mach, c=c)
    residual = p_yy_fd + P * p_y_fd - alpha**2 * R * pressure
    residual_scale = (
        np.abs(p_yy_fd)
        + np.abs(P * p_y_fd)
        + np.abs(alpha**2 * R * pressure)
        + 1.0e-14
    )
    relative_residual = np.abs(residual) / residual_scale

    derivative_scale = np.abs(p_y_gamma) + np.abs(p_y_fd) + 1.0e-14
    relative_derivative_error = np.abs(p_y_fd - p_y_gamma) / derivative_scale

    # Remove boundaries and a narrow interval around the left/right splice.
    interior = np.ones(y.size, dtype=bool)
    interior[:4] = False
    interior[-4:] = False
    splice_guard = max(1.0, 10.0 * float(np.median(np.diff(y))))
    interior &= np.abs(y - matching_y) > splice_guard

    core = interior & (np.abs(y) <= core_y)
    significant = interior & (
        frame["modulus"].to_numpy(float) >= 1.0e-8
    )

    def safe_max(values: np.ndarray, mask: np.ndarray) -> float:
        selected = values[mask]
        return float(np.nanmax(selected)) if selected.size else float("nan")

    def safe_rms(values: np.ndarray, mask: np.ndarray) -> float:
        selected = values[mask]
        return (
            float(np.sqrt(np.nanmean(selected**2)))
            if selected.size
            else float("nan")
        )

    return {
        "fd_relative_derivative_error_max_core": safe_max(
            relative_derivative_error, core
        ),
        "fd_relative_derivative_error_rms_core": safe_rms(
            relative_derivative_error, core
        ),
        "fd_relative_ode_residual_max_core": safe_max(
            relative_residual, core
        ),
        "fd_relative_ode_residual_rms_core": safe_rms(
            relative_residual, core
        ),
        "fd_relative_ode_residual_max_significant": safe_max(
            relative_residual, significant
        ),
        "fd_relative_ode_residual_rms_significant": safe_rms(
            relative_residual, significant
        ),
    }


def first_existing_column(
    columns: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    exact = {str(column): str(column) for column in columns}
    lowered = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def compare_with_reference(
    frame: pd.DataFrame,
    *,
    reference_path: Path,
    Mach: float,
    alpha: float,
    core_y: float,
    output_dir: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reference_attempted": True,
        "reference_path": str(reference_path),
        "reference_loaded": False,
    }
    if not reference_path.is_file():
        result["reference_message"] = "Reference parquet not found."
        return result

    try:
        reference = pd.read_parquet(reference_path)
    except Exception as exc:
        result["reference_message"] = (
            f"Reference parquet could not be read: {type(exc).__name__}: {exc}"
        )
        return result
    result["reference_columns"] = [str(c) for c in reference.columns]

    mach_col = first_existing_column(
        list(reference.columns), ("Mach", "M", "mach")
    )
    alpha_col = first_existing_column(
        list(reference.columns), ("alpha", "Alpha", "wavenumber")
    )
    y_col = first_existing_column(
        list(reference.columns), ("y", "Y", "coordinate_y")
    )
    pre_col = first_existing_column(
        list(reference.columns),
        ("p_real", "p_re", "pressure_real", "Re_p", "p_r"),
    )
    pim_col = first_existing_column(
        list(reference.columns),
        ("p_imag", "p_im", "pressure_imag", "Im_p", "p_i"),
    )
    pcomplex_col = first_existing_column(
        list(reference.columns), ("p", "pressure", "p_complex")
    )

    required_basic = (mach_col, alpha_col, y_col)
    if any(column is None for column in required_basic):
        result["reference_message"] = (
            "Reference schema not recognized for Mach/alpha/y. "
            "See reference_columns in summary.json."
        )
        return result

    selected = reference[
        np.isclose(reference[mach_col].astype(float), Mach, rtol=0.0, atol=5.0e-8)
        & np.isclose(
            reference[alpha_col].astype(float),
            alpha,
            rtol=0.0,
            atol=5.0e-8,
        )
    ].copy()

    if selected.empty:
        result["reference_message"] = (
            f"No reference rows found for M={Mach}, alpha={alpha}."
        )
        return result

    selected = selected.sort_values(y_col)
    y_ref = selected[y_col].to_numpy(float)

    if pre_col is not None and pim_col is not None:
        p_ref = (
            selected[pre_col].to_numpy(float)
            + 1j * selected[pim_col].to_numpy(float)
        )
    elif pcomplex_col is not None:
        p_ref = selected[pcomplex_col].to_numpy(complex)
    else:
        result["reference_message"] = (
            "Reference pressure columns not recognized. "
            "See reference_columns in summary.json."
        )
        return result

    new_y = frame["y"].to_numpy(float)
    new_p = (
        frame["p_real"].to_numpy(float)
        + 1j * frame["p_imag"].to_numpy(float)
    )
    new_interp = (
        np.interp(y_ref, new_y, new_p.real)
        + 1j * np.interp(y_ref, new_y, new_p.imag)
    )

    overlap = (
        np.isfinite(y_ref)
        & np.isfinite(p_ref.real)
        & np.isfinite(p_ref.imag)
        & (y_ref >= new_y.min())
        & (y_ref <= new_y.max())
    )
    core = overlap & (np.abs(y_ref) <= core_y)
    if np.count_nonzero(core) < 4:
        core = overlap

    denominator = np.vdot(new_interp[core], new_interp[core])
    if abs(denominator) < 1.0e-30:
        result["reference_message"] = "New mode is zero on comparison core."
        return result

    complex_scale = np.vdot(new_interp[core], p_ref[core]) / denominator
    aligned = complex_scale * new_interp

    def relative_l2(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
        den = np.linalg.norm(b[mask])
        return (
            float(np.linalg.norm(a[mask] - b[mask]) / den)
            if den > 0.0
            else float("nan")
        )

    result.update(
        {
            "reference_loaded": True,
            "reference_rows": int(selected.shape[0]),
            "alignment_scale_real": float(complex_scale.real),
            "alignment_scale_imag": float(complex_scale.imag),
            "pressure_relative_l2_core": relative_l2(
                aligned, p_ref, core
            ),
            "pressure_relative_l2_overlap": relative_l2(
                aligned, p_ref, overlap
            ),
            "envelope_relative_l2_core": relative_l2(
                np.abs(aligned), np.abs(p_ref), core
            ),
            "envelope_relative_l2_overlap": relative_l2(
                np.abs(aligned), np.abs(p_ref), overlap
            ),
        }
    )

    comparison = pd.DataFrame(
        {
            "y": y_ref,
            "p_ref_real": p_ref.real,
            "p_ref_imag": p_ref.imag,
            "p_new_aligned_real": aligned.real,
            "p_new_aligned_imag": aligned.imag,
            "modulus_ref": np.abs(p_ref),
            "modulus_new_aligned": np.abs(aligned),
            "in_overlap": overlap,
            "in_core": core,
        }
    )
    comparison.to_csv(output_dir / "comparison_with_old_reference.csv", index=False)

    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    axes[0].plot(y_ref[overlap], p_ref.real[overlap], label="old Re(p)")
    axes[0].plot(
        y_ref[overlap],
        aligned.real[overlap],
        linestyle="--",
        label="new aligned Re(p)",
    )
    axes[0].plot(y_ref[overlap], p_ref.imag[overlap], label="old Im(p)")
    axes[0].plot(
        y_ref[overlap],
        aligned.imag[overlap],
        linestyle="--",
        label="new aligned Im(p)",
    )
    axes[0].set_ylabel("Complex pressure")
    axes[0].legend(ncols=2)
    axes[0].grid(True, alpha=0.25)

    axes[1].semilogy(
        y_ref[overlap],
        np.maximum(np.abs(p_ref[overlap]), 1.0e-16),
        label="old |p|",
    )
    axes[1].semilogy(
        y_ref[overlap],
        np.maximum(np.abs(aligned[overlap]), 1.0e-16),
        linestyle="--",
        label="new aligned |p|",
    )
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("Pressure modulus")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.25)

    figure.suptitle(
        f"Old/new modal comparison — M={Mach:g}, alpha={alpha:g}"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "comparison_with_old_reference.png", dpi=220)
    figure.savefig(output_dir / "comparison_with_old_reference.pdf")
    plt.close(figure)

    return result


def plot_reconstruction(
    frame: pd.DataFrame,
    *,
    Mach: float,
    alpha: float,
    c: complex,
    matching_y: float,
    output_dir: Path,
) -> None:
    y = frame["y"].to_numpy(float)

    figure, axes = plt.subplots(4, 1, figsize=(11.0, 11.5), sharex=True)

    axes[0].plot(y, frame["kappa"], label=r"$\kappa$")
    axes[0].plot(y, frame["q"], label=r"$q$")
    axes[0].axvline(matching_y, linestyle=":")
    axes[0].set_ylabel("Riccati variables")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].semilogy(
        y,
        np.maximum(frame["modulus"].to_numpy(float), 1.0e-16),
    )
    axes[1].axvline(matching_y, linestyle=":")
    axes[1].set_ylabel(r"Normalized $|\hat p|$")
    axes[1].grid(True, which="both", alpha=0.25)

    axes[2].plot(y, frame["phi_postprocessed"])
    axes[2].axvline(matching_y, linestyle=":")
    axes[2].set_ylabel(r"Postprocessed $\phi$")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(y, frame["p_real"], label=r"$\Re(\hat p)$")
    axes[3].plot(y, frame["p_imag"], label=r"$\Im(\hat p)$")
    axes[3].axvline(matching_y, linestyle=":")
    axes[3].set_xlabel("y")
    axes[3].set_ylabel("Reconstructed pressure")
    axes[3].legend()
    axes[3].grid(True, alpha=0.25)

    figure.suptitle(
        "Non-oscillatory shooting reconstruction\n"
        f"M={Mach:g}, alpha={alpha:g}, "
        f"c={c.real:.10g}+{c.imag:.10g}i"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "kappa_q_modulus_reconstruction.png", dpi=220)
    figure.savefig(output_dir / "kappa_q_modulus_reconstruction.pdf")
    plt.close(figure)


def main() -> int:
    args = parse_args()

    if args.Ly <= 0.0:
        raise ValueError("--Ly must be positive.")
    if not (-args.Ly < args.matching_y < args.Ly):
        raise ValueError("--matching-y must lie strictly inside (-Ly, Ly).")
    if args.output_dy <= 0.0:
        raise ValueError("--output-dy must be positive.")
    if args.max_step <= 0.0:
        raise ValueError("--max-step must be positive.")

    repo = args.repo.expanduser().resolve()
    c = complex(args.cr, args.ci)

    if args.output_dir is None:
        output_dir = (
            repo
            / "classic_supersonic/reproducibility/results"
            / "kappa_q_modulus_test_M140_a018125"
        )
    else:
        output_dir = args.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = repo / output_dir
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reference_parquet is None:
        reference_path = (
            repo
            / "classic_supersonic/data/modal"
            / "supersonic_reference_v2_modal_raw.parquet"
        )
    else:
        reference_path = args.reference_parquet.expanduser()
        if not reference_path.is_absolute():
            reference_path = repo / reference_path
        reference_path = reference_path.resolve()

    print("=== Fixed-eigenvalue kappa-q-modulus test ===", flush=True)
    print(f"Repository : {repo}", flush=True)
    print(f"Output     : {output_dir}", flush=True)
    print(f"M          : {args.Mach}", flush=True)
    print(f"alpha      : {args.alpha}", flush=True)
    print(f"c fixed    : {c.real:.15g} + {c.imag:.15g} i", flush=True)
    print(f"Ly         : {args.Ly}", flush=True)
    print(f"matching_y : {args.matching_y}", flush=True)
    print(f"method     : {args.method}", flush=True)
    print(f"max_step   : {args.max_step}", flush=True)

    left = integrate_branch(
        side="left",
        Mach=args.Mach,
        alpha=args.alpha,
        c=c,
        Ly=args.Ly,
        matching_y=args.matching_y,
        output_dy=args.output_dy,
        max_step=args.max_step,
        rtol=args.rtol,
        atol=args.atol,
        method=args.method,
    )
    right = integrate_branch(
        side="right",
        Mach=args.Mach,
        alpha=args.alpha,
        c=c,
        Ly=args.Ly,
        matching_y=args.matching_y,
        output_dy=args.output_dy,
        max_step=args.max_step,
        rtol=args.rtol,
        atol=args.atol,
        method=args.method,
    )

    print(
        "left gamma(-infty)  = "
        f"{left['gamma0'].real:.12g} + {left['gamma0'].imag:.12g} i",
        flush=True,
    )
    print(
        "right gamma(+infty) = "
        f"{right['gamma0'].real:.12g} + {right['gamma0'].imag:.12g} i",
        flush=True,
    )
    print(
        f"function evaluations: left={left['nfev']}, right={right['nfev']}",
        flush=True,
    )

    frame, matching_diagnostics = reconstruct_mode(left, right)
    residual_diagnostics = numerical_residuals(
        frame,
        Mach=args.Mach,
        alpha=args.alpha,
        c=c,
        matching_y=args.matching_y,
        core_y=args.core_y,
    )

    frame.to_csv(output_dir / "kappa_q_modulus_reconstruction.csv", index=False)
    plot_reconstruction(
        frame,
        Mach=args.Mach,
        alpha=args.alpha,
        c=c,
        matching_y=args.matching_y,
        output_dir=output_dir,
    )

    reference_diagnostics = compare_with_reference(
        frame,
        reference_path=reference_path,
        Mach=args.Mach,
        alpha=args.alpha,
        core_y=args.core_y,
        output_dir=output_dir,
    )

    summary: dict[str, Any] = {
        "parameters": {
            "Mach": args.Mach,
            "alpha": args.alpha,
            "cr": args.cr,
            "ci": args.ci,
            "Ly": args.Ly,
            "matching_y": args.matching_y,
            "output_dy": args.output_dy,
            "max_step": args.max_step,
            "rtol": args.rtol,
            "atol": args.atol,
            "method": args.method,
            "core_y": args.core_y,
        },
        "asymptotic_conditions": {
            "gamma_left_real": left["gamma0"].real,
            "gamma_left_imag": left["gamma0"].imag,
            "gamma_right_real": right["gamma0"].real,
            "gamma_right_imag": right["gamma0"].imag,
        },
        "integrator": {
            "left_nfev": left["nfev"],
            "right_nfev": right["nfev"],
        },
        "matching": matching_diagnostics,
        "finite_difference_checks": residual_diagnostics,
        "old_reference_comparison": reference_diagnostics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== Matching diagnostics ===", flush=True)
    for key, value in matching_diagnostics.items():
        print(f"{key}: {value:.12e}", flush=True)

    print("\n=== Independent finite-difference checks ===", flush=True)
    for key, value in residual_diagnostics.items():
        print(f"{key}: {value:.12e}", flush=True)

    print("\n=== Old reference comparison ===", flush=True)
    for key, value in reference_diagnostics.items():
        if key == "reference_columns":
            print(f"{key}: {value}", flush=True)
        else:
            print(f"{key}: {value}", flush=True)

    print("\nWritten files:", flush=True)
    for path in sorted(output_dir.iterdir()):
        print(f"  {path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
