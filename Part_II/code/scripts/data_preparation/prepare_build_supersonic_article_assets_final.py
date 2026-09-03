#!/usr/bin/env python3
"""Regenerate scientifically auditable classical-supersonic KH article assets.

The script uses only existing local results. It never launches a spectral solve.
Paths containing QUARANTINE_INVALID (or other quarantine/old markers) are rejected.

Outputs
-------
Figures (PDF + PNG, 300 dpi):
  Fig_supersonic_representative_mode_M140_a018
  Fig_supersonic_modal_reconstruction_residuals_M140_a018
  Fig_supersonic_eigenvalue_convergence
  Fig_supersonic_box_and_matching_robustness
  Fig_supersonic_core_tail_phase_convergence  [only when raw run modes exist]

Tables/reports:
  Tab_supersonic_modal_reconstruction_residuals_M140_a018.csv
  Tab_supersonic_asymptotic_decay_M140_a018.csv
  Tab_supersonic_eigenvalue_convergence.csv
  Tab_supersonic_box_and_matching_robustness.csv
  Tab_supersonic_core_tail_phase_convergence.csv [only when computable]
  Supersonic_asset_generation_report.txt
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Global policy and plotting configuration
# -----------------------------------------------------------------------------

EXCLUDED_PATH_RE = re.compile(
    r"QUARANTINE_INVALID|QUARANTINE|(?:^|[/_\\-])OLD(?:[/_\\-]|$)|"
    r"backup|archive|obsolete|superseded",
    re.IGNORECASE,
)

UPPER_BOUND = 1.0e-14
DISPLAY_FLOOR = 1.0e-16
CORE_WINDOW = 250.0
TARGET_MACH = 1.4
TARGET_ALPHA = 0.18
FIELDS = ("p", "rho", "u", "v")

mpl.rcParams.update(
    {
        "font.size": 10.5,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8.2,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


@dataclass
class Audit:
    inputs: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    upper_bounds: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def add_input(self, path: Path) -> None:
        value = str(path.resolve())
        if value not in self.inputs:
            self.inputs.append(value)

    def add_output(self, path: Path) -> None:
        value = str(path.resolve())
        if value not in self.outputs:
            self.outputs.append(value)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def find_repo_root(start: Path) -> Path:
    start = start.expanduser().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "assets").is_dir() and (candidate / "classic_supersonic").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate repository root containing assets/ and classic_supersonic/."
    )


def reject_excluded_path(path: Path) -> None:
    if EXCLUDED_PATH_RE.search(str(path)):
        raise RuntimeError(f"Excluded/quarantined path must not be read: {path}")


def require_file(path: Path, audit: Audit) -> Path:
    path = path.expanduser()
    reject_excluded_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    audit.add_input(path)
    return path


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "validated", "ok"}
    )


def save_figure(fig: plt.Figure, pdf_path: Path, dpi: int, audit: Audit) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = pdf_path.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    audit.add_output(pdf_path)
    audit.add_output(png_path)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def finite_minmax(values: Iterable[float]) -> tuple[float, float] | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.min(array)), float(np.max(array))


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def configuration_label(row: pd.Series) -> str:
    if str(row.get("sweep_type", "")) == "shooting_box":
        return f"Ly={safe_float(row.get('Ly')):g}"
    label = str(row.get("accuracy_label", "")).strip()
    return label if label and label.lower() != "nan" else str(row.get("run_id", ""))


# -----------------------------------------------------------------------------
# Canonical modal case and reconstruction checks
# -----------------------------------------------------------------------------


def load_canonical_mode(repo: Path, audit: Audit) -> tuple[pd.DataFrame, dict[str, Any]]:
    campaign = (
        repo
        / "assets/classic_supersonic/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS"
    )
    index_path = require_file(
        campaign / "classical_supersonic_final_modes_index.csv", audit
    )
    long_path = require_file(
        campaign / "classical_supersonic_final_modes_long.csv.gz", audit
    )

    index = pd.read_csv(index_path)
    required_index = {
        "final_reference_id",
        "Mach",
        "alpha",
        "cr",
        "ci",
        "omega_i",
        "n_coordinates",
        "y_min",
        "y_max",
    }
    missing = sorted(required_index.difference(index.columns))
    if missing:
        raise KeyError(f"Missing modal-index columns: {missing}")

    selection = index.loc[
        np.isclose(index["Mach"], TARGET_MACH, atol=1e-12, rtol=0.0)
        & np.isclose(index["alpha"], TARGET_ALPHA, atol=1e-12, rtol=0.0)
    ]
    if len(selection) != 1:
        raise ValueError(
            f"Expected exactly one canonical modal row, found {len(selection)}."
        )
    meta_row = selection.iloc[0]
    reference_id = str(meta_row["final_reference_id"])

    pieces: list[pd.DataFrame] = []
    usecols = [
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
    for chunk in pd.read_csv(long_path, usecols=usecols, chunksize=250_000):
        mask = chunk["final_reference_id"].astype(str).eq(reference_id)
        if mask.any():
            pieces.append(chunk.loc[mask].copy())

    if not pieces:
        raise ValueError(f"Canonical reference {reference_id} is absent from modal long CSV.")

    mode = (
        pd.concat(pieces, ignore_index=True)
        .sort_values("coordinate_index", kind="mergesort")
        .reset_index(drop=True)
    )
    expected_n = int(meta_row["n_coordinates"])
    if len(mode) != expected_n:
        raise ValueError(f"Canonical mode has {len(mode)} rows, expected {expected_n}.")

    y = mode["y"].to_numpy(float)
    if not np.all(np.isfinite(y)) or not np.all(np.diff(y) > 0.0):
        raise ValueError("Canonical modal coordinate must be finite and strictly increasing.")
    if mode["y"].duplicated().any():
        raise ValueError("Duplicate y values detected in canonical raw mode.")

    def complex_field(name: str) -> np.ndarray:
        return mode[f"{name}_real"].to_numpy(float) + 1j * mode[
            f"{name}_imag"
        ].to_numpy(float)

    p_raw = complex_field("p")
    peak_index = int(np.nanargmax(np.abs(p_raw)))
    peak_value = p_raw[peak_index]
    if not np.isfinite(abs(peak_value)) or abs(peak_value) <= 0.0:
        raise ValueError("Degenerate pressure mode.")

    # One global complex factor: max|p|=1 and p at its maximum is positive real.
    factor = np.exp(-1j * np.angle(peak_value)) / abs(peak_value)
    for name in FIELDS:
        values = factor * complex_field(name)
        mode[f"{name}_real"] = values.real
        mode[f"{name}_imag"] = values.imag

    metadata = {
        "reference_id": reference_id,
        "Mach": float(meta_row["Mach"]),
        "alpha": float(meta_row["alpha"]),
        "cr": float(meta_row["cr"]),
        "ci": float(meta_row["ci"]),
        "omega_i": float(meta_row["omega_i"]),
        "normalization_factor": factor,
        "y_min": float(y.min()),
        "y_max": float(y.max()),
    }

    audit.references.append(
        "Canonical mode: "
        f"{reference_id}, M={metadata['Mach']}, alpha={metadata['alpha']}, "
        f"cr={metadata['cr']:.16g}, ci={metadata['ci']:.16g}"
    )
    return mode, metadata


def complex_from_frame(frame: pd.DataFrame, field: str) -> np.ndarray:
    return frame[f"{field}_real"].to_numpy(float) + 1j * frame[
        f"{field}_imag"
    ].to_numpy(float)


def plot_representative_mode(
    mode: pd.DataFrame,
    metadata: dict[str, Any],
    output_path: Path,
    dpi: int,
    audit: Audit,
) -> None:
    y = mode["y"].to_numpy(float)
    p = complex_from_frame(mode, "p")
    rho = complex_from_frame(mode, "rho")
    u = complex_from_frame(mode, "u")
    v = complex_from_frame(mode, "v")

    rho_error = rho - metadata["Mach"] ** 2 * p
    rho_relative_l2 = float(np.linalg.norm(rho_error) / np.linalg.norm(rho))
    rho_relative_linf = float(np.max(np.abs(rho_error)) / np.max(np.abs(rho)))

    u_peak_index = int(np.nanargmax(np.abs(u)))
    u_peak_y = float(y[u_peak_index])
    u_peak = complex(u[u_peak_index])
    local_spacing = (
        float(y[u_peak_index] - y[u_peak_index - 1]) if u_peak_index > 0 else float("nan"),
        float(y[u_peak_index + 1] - y[u_peak_index])
        if u_peak_index + 1 < len(y)
        else float("nan"),
    )

    print(
        "Canonical raw-mode check: "
        f"max|u|={abs(u_peak):.16g} at y={u_peak_y:.16g}; "
        f"u={u_peak.real:.16g}{u_peak.imag:+.16g}i; "
        f"neighbor spacings={local_spacing}; duplicate_y={bool(mode['y'].duplicated().any())}"
    )
    print(
        "rho=M^2 p check: "
        f"relative L2={rho_relative_l2:.6e}, relative Linf={rho_relative_linf:.6e}"
    )

    audit.diagnostics.extend(
        [
            f"Canonical max|u|={abs(u_peak):.16g} at y={u_peak_y:.16g}; "
            f"u={u_peak.real:.16g}{u_peak.imag:+.16g}i.",
            "Canonical mode contains no duplicate y coordinates; "
            f"local spacings around max|u| are {local_spacing}.",
            f"rho=M^2 p relative L2={rho_relative_l2:.6e}; "
            f"relative Linf={rho_relative_linf:.6e}.",
        ]
    )

    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 8.1), sharex=True, constrained_layout=True
    )
    specifications = (
        (p, r"$p$"),
        (rho, r"$\rho$"),
        (u, r"$u$"),
        (v, r"$v$"),
    )
    for label, axis, (values, symbol) in zip(
        ("(a)", "(b)", "(c)", "(d)"), axes.flat, specifications
    ):
        central = np.abs(y) <= CORE_WINDOW
        math_symbol = symbol.strip("$")
        axis.plot(
            y[central],
            values.real[central],
            linewidth=1.8,
            label=rf"$\Re({math_symbol})$",
        )
        axis.plot(
            y[central],
            values.imag[central],
            linewidth=1.8,
            linestyle="--",
            label=rf"$\Im({math_symbol})$",
        )
        axis.axvline(0.0, linewidth=0.8, linestyle=":", alpha=0.65)
        axis.set_xlim(-CORE_WINDOW, CORE_WINDOW)
        axis.set_title(f"{label} {symbol}", loc="left")
        axis.set_ylabel("Amplitude")
        axis.grid(alpha=0.22)
        axis.legend(loc="best")

    axes[1, 0].set_xlabel(r"Transverse coordinate $y$")
    axes[1, 1].set_xlabel(r"Transverse coordinate $y$")
    fig.suptitle(
        r"Representative supersonic mode, $\max_y|p|=1$: "
        rf"$M={metadata['Mach']:.1f}$, $\alpha={metadata['alpha']:.2f}$, "
        rf"$c_r={metadata['cr']:.6f}$, $c_i={metadata['ci']:.6f}$",
        fontsize=13.5,
    )
    save_figure(fig, output_path, dpi, audit)


def relative_metrics(
    residual: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, int]:
    valid = mask & np.isfinite(residual) & np.isfinite(target.real) & np.isfinite(target.imag)
    count = int(valid.sum())
    if count == 0:
        return float("nan"), float("nan"), 0
    epsilon = np.finfo(float).eps
    l2 = float(
        np.linalg.norm(residual[valid])
        / max(np.linalg.norm(target[valid]), epsilon)
    )
    linf = float(
        np.max(np.abs(residual[valid]))
        / max(np.max(np.abs(target[valid])), epsilon)
    )
    return l2, linf, count


def theoretical_gamma(alpha: float, mach: float, c: complex, u_inf: float, side: str) -> complex:
    gamma_squared = alpha**2 * (1.0 - mach**2 * (u_inf - c) ** 2)
    root = complex(np.sqrt(gamma_squared))
    roots = (root, -root)
    if side == "left":
        valid = [candidate for candidate in roots if candidate.real > 0.0]
    else:
        valid = [candidate for candidate in roots if candidate.real < 0.0]
    if len(valid) != 1:
        raise ValueError(f"Could not select unique asymptotic root on {side} side: {roots}")
    return valid[0]


def longest_contiguous_segment(indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return indices
    segments = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    return max(segments, key=len)


def fit_asymptotic_side(
    y: np.ndarray,
    pressure_amplitude: np.ndarray,
    base_velocity: np.ndarray,
    side: str,
    theory: float,
) -> dict[str, Any]:
    side_mask = y < 0.0 if side == "left" else y > 0.0
    mask = (
        side_mask
        & (np.abs(np.abs(base_velocity) - 1.0) < 1.0e-8)
        & (pressure_amplitude >= 1.0e-10)
        & (pressure_amplitude <= 1.0e-3)
        & np.isfinite(pressure_amplitude)
        & (pressure_amplitude > 0.0)
    )
    indices = longest_contiguous_segment(np.flatnonzero(mask))
    if indices.size < 8:
        raise ValueError(
            f"Fewer than eight contiguous asymptotic fitting points on {side} side."
        )
    coefficients = np.polyfit(y[indices], np.log(pressure_amplitude[indices]), 1)
    slope = float(coefficients[0])
    intercept = float(coefficients[1])
    absolute_difference = abs(slope - theory)
    relative_difference = absolute_difference / max(abs(theory), np.finfo(float).eps)
    return {
        "side": side,
        "indices": indices,
        "y_fit_min": float(y[indices[0]]),
        "y_fit_max": float(y[indices[-1]]),
        "n_fit_points": int(indices.size),
        "kappa_theory": float(theory),
        "kappa_fit": slope,
        "intercept": intercept,
        "absolute_difference": float(absolute_difference),
        "relative_difference": float(relative_difference),
    }


def build_modal_reconstruction_assets(
    mode: pd.DataFrame,
    metadata: dict[str, Any],
    figure_path: Path,
    table_directory: Path,
    dpi: int,
    audit: Audit,
) -> None:
    y = mode["y"].to_numpy(float)
    kappa = mode["kappa"].to_numpy(float)
    q = mode["q"].to_numpy(float)
    gamma = kappa + 1j * q
    p = complex_from_frame(mode, "p")
    rho = complex_from_frame(mode, "rho")
    u = complex_from_frame(mode, "u")
    v = complex_from_frame(mode, "v")

    mach = float(metadata["Mach"])
    alpha = float(metadata["alpha"])
    c = complex(metadata["cr"], metadata["ci"])
    base_velocity = np.tanh(y)
    base_velocity_prime = 1.0 - base_velocity**2
    pressure_derivative = np.gradient(p, y, edge_order=2)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        residuals = {
            "p_y=(kappa+i q)p": np.abs(pressure_derivative - gamma * p),
            "rho=M^2 p": np.abs(rho - mach**2 * p),
            "v=i p_y/[alpha(U-c)]": np.abs(
                v - 1j * pressure_derivative / (alpha * (base_velocity - c))
            ),
            "u=-p/(U-c)+i Uprime v/[alpha(U-c)]": np.abs(
                u
                + p / (base_velocity - c)
                - 1j
                * base_velocity_prime
                * v
                / (alpha * (base_velocity - c))
            ),
        }

    targets = {
        "p_y=(kappa+i q)p": gamma * p,
        "rho=M^2 p": rho,
        "v=i p_y/[alpha(U-c)]": v,
        "u=-p/(U-c)+i Uprime v/[alpha(U-c)]": u,
    }
    normalizations = {
        "p_y=(kappa+i q)p": "||gamma p|| on |y|<=250",
        "rho=M^2 p": "||rho|| on |y|<=250",
        "v=i p_y/[alpha(U-c)]": "||v|| on |y|<=250",
        "u=-p/(U-c)+i Uprime v/[alpha(U-c)]": "||u|| on |y|<=250",
    }

    core = np.abs(y) <= CORE_WINDOW
    metric_rows: list[dict[str, Any]] = []
    display_curves: dict[str, np.ndarray] = {}
    for relation, residual in residuals.items():
        target = targets[relation]
        l2, linf, n_valid = relative_metrics(residual, target, core)
        metric_rows.append(
            {
                "relation": relation,
                "relative_L2": l2,
                "relative_Linf": linf,
                "normalization": normalizations[relation],
                "n_valid_points": n_valid,
            }
        )
        denominator = max(
            float(np.nanmax(np.abs(target[core]))), np.finfo(float).eps
        )
        display_curves[relation] = residual / denominator

    metric_table = pd.DataFrame(metric_rows)
    table_directory.mkdir(parents=True, exist_ok=True)
    residual_table_path = (
        table_directory / "Tab_supersonic_modal_reconstruction_residuals_M140_a018.csv"
    )
    metric_table.to_csv(residual_table_path, index=False)
    audit.add_output(residual_table_path)

    normalized_pressure = np.abs(p) / np.max(np.abs(p))
    left_gamma = theoretical_gamma(alpha, mach, c, -1.0, "left")
    right_gamma = theoretical_gamma(alpha, mach, c, 1.0, "right")
    left_fit = fit_asymptotic_side(
        y, normalized_pressure, base_velocity, "left", left_gamma.real
    )
    right_fit = fit_asymptotic_side(
        y, normalized_pressure, base_velocity, "right", right_gamma.real
    )

    fit_rows = []
    for result in (left_fit, right_fit):
        fit_rows.append(
            {
                key: result[key]
                for key in (
                    "side",
                    "y_fit_min",
                    "y_fit_max",
                    "n_fit_points",
                    "kappa_theory",
                    "kappa_fit",
                    "absolute_difference",
                    "relative_difference",
                )
            }
        )
        print(
            f"{result['side']} asymptotic fit: "
            f"y=[{result['y_fit_min']:.9g},{result['y_fit_max']:.9g}], "
            f"n={result['n_fit_points']}, theory={result['kappa_theory']:.12g}, "
            f"fit={result['kappa_fit']:.12g}, "
            f"relative difference={result['relative_difference']:.6e}"
        )
        audit.diagnostics.append(
            f"{result['side']} asymptotic fit y=[{result['y_fit_min']:.9g}, "
            f"{result['y_fit_max']:.9g}], n={result['n_fit_points']}, "
            f"kappa_theory={result['kappa_theory']:.12g}, "
            f"kappa_fit={result['kappa_fit']:.12g}, "
            f"relative_difference={result['relative_difference']:.6e}."
        )

    asymptotic_table_path = (
        table_directory / "Tab_supersonic_asymptotic_decay_M140_a018.csv"
    )
    pd.DataFrame(fit_rows).to_csv(asymptotic_table_path, index=False)
    audit.add_output(asymptotic_table_path)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.35), constrained_layout=True)
    central = np.abs(y) <= CORE_WINDOW
    labels = {
        "p_y=(kappa+i q)p": r"$p_y=(\kappa+iq)p$",
        "rho=M^2 p": r"$\rho=M^2p$",
        "v=i p_y/[alpha(U-c)]": r"$v=i p_y/[\alpha(U-c)]$",
        "u=-p/(U-c)+i Uprime v/[alpha(U-c)]": (
            r"$u=-p/(U-c)+iU'v/[\alpha(U-c)]$"
        ),
    }
    for relation, values in display_curves.items():
        valid = central & np.isfinite(values)
        axes[0].semilogy(
            y[valid],
            np.maximum(values[valid], DISPLAY_FLOOR),
            linewidth=1.55,
            label=labels[relation],
        )
    axes[0].set_xlim(-CORE_WINDOW, CORE_WINDOW)
    axes[0].set_xlabel(r"Transverse coordinate $y$")
    axes[0].set_ylabel("Relative reconstruction residual")
    axes[0].set_title("(a) Central reconstruction checks", loc="left")
    axes[0].grid(alpha=0.23, which="both")
    axes[0].legend(loc="best")
    axes[0].text(
        0.02,
        0.03,
        r"Display floor: $10^{-16}$ (not a resolved value)",
        transform=axes[0].transAxes,
        fontsize=8.2,
    )

    axes[1].semilogy(y, normalized_pressure, linewidth=1.65, label=r"$|p|/\max_y|p|$")
    for result, linestyle in ((left_fit, "--"), (right_fit, ":")):
        indices = result["indices"]
        fit_curve = np.exp(result["intercept"] + result["kappa_fit"] * y[indices])
        axes[1].semilogy(
            y[indices],
            fit_curve,
            linestyle=linestyle,
            linewidth=2.0,
            label=(
                f"{result['side']} fit: "
                rf"$\kappa_{{fit}}={result['kappa_fit']:.6f}$, "
                rf"$\kappa_{{th}}={result['kappa_theory']:.6f}$"
            ),
        )
    axes[1].set_xlabel(r"Transverse coordinate $y$")
    axes[1].set_ylabel(r"Normalized pressure amplitude")
    axes[1].set_title("(b) Full-domain asymptotic decay", loc="left")
    axes[1].grid(alpha=0.23, which="both")
    axes[1].legend(loc="best")

    fig.suptitle(
        rf"Modal reconstruction audit: $M={mach:.1f}$, $\alpha={alpha:.2f}$, "
        rf"$c_r={metadata['cr']:.6f}$, $c_i={metadata['ci']:.6f}$",
        fontsize=13.5,
    )
    save_figure(fig, figure_path, dpi, audit)

    for row in metric_rows:
        audit.diagnostics.append(
            f"Reconstruction {row['relation']}: relative L2={row['relative_L2']:.6e}, "
            f"relative Linf={row['relative_Linf']:.6e}, "
            f"n={row['n_valid_points']}."
        )


# -----------------------------------------------------------------------------
# Spectral convergence and robustness
# -----------------------------------------------------------------------------


def load_spectral_error_table(repo: Path, audit: Audit) -> pd.DataFrame:
    path = require_file(
        repo
        / "assets/classic_supersonic/csv/computational_cost/errors/"
        "table_convergence_errors_spectral_b1818272d8.csv",
        audit,
    )
    frame = pd.read_csv(path)
    required = {
        "run_id",
        "regime",
        "Mach",
        "alpha",
        "sweep_type",
        "Ly",
        "rtol",
        "atol",
        "max_step",
        "matching_y",
        "cr",
        "ci",
        "reference_run_id",
        "abs_error_cr",
        "abs_error_ci",
        "branch_provenance_status",
        "branch_check_passed",
        "overall_validated",
        "validation_status",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing spectral-error columns: {missing}")

    # The file repeats each spectral run for three modal core thresholds.
    frame = frame.drop_duplicates("run_id", keep="first").copy()
    frame = frame.loc[
        frame["regime"].astype(str).eq("supersonic")
        & frame["Mach"].isin([1.8, 1.9])
        & frame["sweep_type"].isin(["shooting_box", "shooting_accuracy"])
        & frame["branch_provenance_status"].astype(str).eq("resolved")
        & as_bool(frame["branch_check_passed"])
        & as_bool(frame["overall_validated"])
    ].copy()
    if frame.empty:
        raise ValueError("No validated, branch-resolved M=1.8/M=1.9 spectral rows found.")

    for column in (
        "Mach",
        "alpha",
        "Ly",
        "rtol",
        "atol",
        "max_step",
        "matching_y",
        "cr",
        "ci",
        "abs_error_cr",
        "abs_error_ci",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Recover reference eigenvalues from the referenced run, rather than from the
    # unrelated configured-reference columns in the original run table.
    by_run = frame.set_index("run_id", drop=False)
    reference_cr: list[float] = []
    reference_ci: list[float] = []
    for _, row in frame.iterrows():
        reference_id = str(row["reference_run_id"])
        if reference_id not in by_run.index:
            # The reference itself can be in the full source table but removed by
            # validation filtering; fall back to all deduplicated rows.
            full = pd.read_csv(path).drop_duplicates("run_id", keep="first").set_index(
                "run_id", drop=False
            )
            if reference_id not in full.index:
                raise KeyError(f"Reference run not found: {reference_id}")
            reference_row = full.loc[reference_id]
        else:
            reference_row = by_run.loc[reference_id]
        reference_cr.append(float(reference_row["cr"]))
        reference_ci.append(float(reference_row["ci"]))
    frame["cr_reference"] = reference_cr
    frame["ci_reference"] = reference_ci

    # Recompute from printed CSV values as a cross-check. Keep the original error
    # columns, which were generated before serialization, when they are larger.
    recomputed_cr = np.abs(frame["cr"] - frame["cr_reference"])
    recomputed_ci = np.abs(frame["ci"] - frame["ci_reference"])
    frame["delta_cr"] = np.maximum(frame["abs_error_cr"].fillna(0.0), recomputed_cr)
    frame["delta_ci"] = np.maximum(frame["abs_error_ci"].fillna(0.0), recomputed_ci)
    frame["is_upper_bound_cr"] = frame["delta_cr"] <= UPPER_BOUND
    frame["is_upper_bound_ci"] = frame["delta_ci"] <= UPPER_BOUND
    frame["branch_status"] = (
        frame["branch_provenance_status"].astype(str)
        + "; "
        + frame["validation_status"].astype(str)
    )
    frame["configuration"] = frame.apply(configuration_label, axis=1)

    for (mach, sweep), group in frame.groupby(["Mach", "sweep_type"], sort=True):
        references = sorted(group["reference_run_id"].astype(str).unique())
        audit.references.append(
            f"Spectral {sweep}, M={mach:g}: reference run(s)={references}."
        )
    audit.excluded.append(
        "All M=1.4 convergence rows excluded because branch provenance is not "
        "accepted for these figures."
    )
    return frame


def plot_series_with_upper_bounds(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    upper_bound: np.ndarray,
    *,
    label: str,
    marker: str,
    linestyle: str,
) -> None:
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    upper_bound = np.asarray(upper_bound, dtype=bool)
    resolved = (~upper_bound) & np.isfinite(y)
    color = axis._get_lines.get_next_color()
    if resolved.any():
        axis.plot(
            x[resolved],
            y[resolved],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            color=color,
            label=label,
        )
    if upper_bound.any():
        axis.scatter(
            x[upper_bound],
            np.full(int(upper_bound.sum()), UPPER_BOUND),
            marker="v",
            facecolors="none",
            edgecolors=color,
            linewidths=1.2,
            label=f"{label} (<$10^{{-14}}$)" if not resolved.any() else None,
            zorder=4,
        )


def write_eigenvalue_table(frame: pd.DataFrame, path: Path, audit: Audit) -> None:
    rows: list[dict[str, Any]] = []
    for _, row in frame.sort_values(
        ["sweep_type", "Mach", "Ly", "accuracy_order"], kind="mergesort"
    ).iterrows():
        common = {
            "Mach": row["Mach"],
            "alpha": row["alpha"],
            "study_type": (
                "domain_size" if row["sweep_type"] == "shooting_box" else "integration_accuracy"
            ),
            "configuration": row["configuration"],
            "Ly": row["Ly"],
            "rtol": row["rtol"],
            "atol": row["atol"],
            "max_step": row["max_step"],
            "cr": row["cr"],
            "ci": row["ci"],
            "cr_reference": row["cr_reference"],
            "ci_reference": row["ci_reference"],
            "delta_cr": row["delta_cr"],
            "delta_ci": row["delta_ci"],
            "is_upper_bound": bool(
                row["is_upper_bound_cr"] and row["is_upper_bound_ci"]
            ),
            "is_upper_bound_cr": bool(row["is_upper_bound_cr"]),
            "is_upper_bound_ci": bool(row["is_upper_bound_ci"]),
            "reference_run_id": row["reference_run_id"],
            "branch_status": row["branch_status"],
        }
        rows.append(common)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    audit.add_output(path)


def build_eigenvalue_convergence_asset(
    frame: pd.DataFrame,
    figure_path: Path,
    table_path: Path,
    dpi: int,
    audit: Audit,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.5), constrained_layout=True)
    marker_by_metric = {"cr": "o", "ci": "s"}
    linestyle_by_mach = {1.8: "-", 1.9: "--"}

    box = frame.loc[frame["sweep_type"].eq("shooting_box")].copy()
    for mach in (1.8, 1.9):
        group = box.loc[np.isclose(box["Mach"], mach)].sort_values("Ly")
        for metric in ("cr", "ci"):
            plot_series_with_upper_bounds(
                axes[0],
                group["Ly"].to_numpy(float),
                group[f"delta_{metric}"].to_numpy(float),
                group[f"is_upper_bound_{metric}"].to_numpy(bool),
                label=rf"$M={mach:.1f}$, $\Delta c_{metric[-1]}$",
                marker=marker_by_metric[metric],
                linestyle=linestyle_by_mach[mach],
            )
    axes[0].set_xlabel(r"Domain half-width $L_y$")
    axes[0].set_ylabel("Absolute eigenvalue difference")
    axes[0].set_title("(a) Domain-size convergence", loc="left")
    axes[0].set_yscale("log")
    axes[0].set_ylim(UPPER_BOUND / 2.5, None)
    axes[0].grid(alpha=0.23, which="both")
    axes[0].legend(loc="best", ncol=2)
    axes[0].text(
        0.02,
        0.04,
        r"Open $\triangledown$: upper bound $<10^{-14}$",
        transform=axes[0].transAxes,
        fontsize=8.2,
    )

    accuracy = frame.loc[frame["sweep_type"].eq("shooting_accuracy")].copy()
    # Same ordered labels for both Mach numbers; use the actual configuration names.
    ordering = (
        accuracy[["configuration", "accuracy_order"]]
        .drop_duplicates()
        .sort_values("accuracy_order", kind="mergesort")
    )
    labels = ordering["configuration"].astype(str).tolist()
    x_map = {label: index for index, label in enumerate(labels)}
    for mach in (1.8, 1.9):
        group = accuracy.loc[np.isclose(accuracy["Mach"], mach)].copy()
        group["_x"] = group["configuration"].map(x_map)
        group = group.sort_values("_x")
        for metric in ("cr", "ci"):
            plot_series_with_upper_bounds(
                axes[1],
                group["_x"].to_numpy(int),
                group[f"delta_{metric}"].to_numpy(float),
                group[f"is_upper_bound_{metric}"].to_numpy(bool),
                label=rf"$M={mach:.1f}$, $\Delta c_{metric[-1]}$",
                marker=marker_by_metric[metric],
                linestyle=linestyle_by_mach[mach],
            )
    axes[1].set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axes[1].set_xlabel("Exact integration configuration")
    axes[1].set_ylabel("Absolute eigenvalue difference")
    axes[1].set_title("(b) Integration-parameter convergence", loc="left")
    axes[1].set_yscale("log")
    axes[1].set_ylim(UPPER_BOUND / 2.5, None)
    axes[1].grid(alpha=0.23, which="both")
    axes[1].legend(loc="best", ncol=2)
    axes[1].text(
        0.02,
        0.04,
        r"Open $\triangledown$: upper bound $<10^{-14}$",
        transform=axes[1].transAxes,
        fontsize=8.2,
    )

    fig.suptitle(
        r"Supersonic eigenvalue convergence on resolved branches "
        r"($M=1.8$ and $M=1.9$, $\alpha=0.1$)",
        fontsize=13.3,
    )
    save_figure(fig, figure_path, dpi, audit)
    write_eigenvalue_table(frame, table_path, audit)

    for metric in ("delta_cr", "delta_ci"):
        limits = finite_minmax(frame.loc[frame[metric] > UPPER_BOUND, metric])
        audit.diagnostics.append(
            f"Eigenvalue {metric} resolved range (>1e-14): {limits}; "
            f"upper-bound rows={int((frame[metric] <= UPPER_BOUND).sum())}."
        )
    audit.upper_bounds.append(
        f"Eigenvalue differences <= {UPPER_BOUND:.0e} are reported and plotted as "
        f"upper bounds <{UPPER_BOUND:.0e}, not as measured values."
    )


def build_box_matching_asset(
    frame: pd.DataFrame,
    figure_path: Path,
    table_path: Path,
    dpi: int,
    audit: Audit,
) -> None:
    box = frame.loc[frame["sweep_type"].eq("shooting_box")].copy()
    matching_values = np.sort(box["matching_y"].dropna().unique())
    matching_varied = len(matching_values) > 1

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.3), constrained_layout=True)
    for mach, linestyle in ((1.8, "-"), (1.9, "--")):
        group = box.loc[np.isclose(box["Mach"], mach)].sort_values("Ly")
        for metric, marker in (("cr", "o"), ("ci", "s")):
            plot_series_with_upper_bounds(
                axes[0],
                group["Ly"].to_numpy(float),
                group[f"delta_{metric}"].to_numpy(float),
                group[f"is_upper_bound_{metric}"].to_numpy(bool),
                label=rf"$M={mach:.1f}$, $\Delta c_{metric[-1]}$",
                marker=marker,
                linestyle=linestyle,
            )
    axes[0].set_xlabel(r"Domain half-width $L_y$")
    axes[0].set_ylabel("Absolute eigenvalue difference")
    axes[0].set_title("(a) Box-size robustness", loc="left")
    axes[0].set_yscale("log")
    axes[0].set_ylim(UPPER_BOUND / 2.5, None)
    axes[0].grid(alpha=0.23, which="both")
    axes[0].legend(loc="best", ncol=2)
    axes[0].text(
        0.02,
        0.04,
        r"Open $\triangledown$: upper bound $<10^{-14}$",
        transform=axes[0].transAxes,
        fontsize=8.2,
    )

    if matching_varied:
        # This branch is deliberately implemented, although the audited data do
        # not currently exercise it.
        for mach, linestyle in ((1.8, "-"), (1.9, "--")):
            group = box.loc[np.isclose(box["Mach"], mach)].sort_values("matching_y")
            for metric, marker in (("cr", "o"), ("ci", "s")):
                plot_series_with_upper_bounds(
                    axes[1],
                    group["matching_y"].to_numpy(float),
                    group[f"delta_{metric}"].to_numpy(float),
                    group[f"is_upper_bound_{metric}"].to_numpy(bool),
                    label=rf"$M={mach:.1f}$, $\Delta c_{metric[-1]}$",
                    marker=marker,
                    linestyle=linestyle,
                )
        axes[1].set_yscale("log")
        axes[1].set_ylim(UPPER_BOUND / 2.5, None)
        axes[1].set_xlabel(r"Matching position $y_{match}$")
        axes[1].set_ylabel("Absolute eigenvalue difference")
        axes[1].set_title("(b) Matching-position robustness", loc="left")
        axes[1].grid(alpha=0.23, which="both")
        axes[1].legend(loc="best", ncol=2)
        overall_title = "Supersonic spectral robustness to box size and matching position"
    else:
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.57,
            "Matching-position study unavailable",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            transform=axes[1].transAxes,
        )
        axes[1].text(
            0.5,
            0.40,
            "Every validated M=1.8/M=1.9 run in the audited dataset uses\n"
            + rf"the same matching position: $y_{{match}}={matching_values[0]:g}$."
            + "\nNo matching-robustness curve is inferred.",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title("(b) Data availability", loc="left")
        overall_title = "Supersonic spectral box-size robustness audit"
        audit.warnings.append(
            "Asset 4 matching-position panel not computable: all validated "
            f"M=1.8/M=1.9 rows have matching_y={matching_values.tolist()}."
        )

    fig.suptitle(overall_title, fontsize=13.3)
    save_figure(fig, figure_path, dpi, audit)

    table_rows = []
    for _, row in box.sort_values(["Mach", "Ly"]).iterrows():
        table_rows.append(
            {
                "Mach": row["Mach"],
                "alpha": row["alpha"],
                "configuration": row["configuration"],
                "Ly": row["Ly"],
                "y_left": -row["Ly"] if np.isfinite(row["Ly"]) else np.nan,
                "y_right": row["Ly"],
                "y_match": row["matching_y"],
                "rtol": row["rtol"],
                "atol": row["atol"],
                "max_step": row["max_step"],
                "cr": row["cr"],
                "ci": row["ci"],
                "delta_cr": row["delta_cr"],
                "delta_ci": row["delta_ci"],
                "is_upper_bound_cr": bool(row["is_upper_bound_cr"]),
                "is_upper_bound_ci": bool(row["is_upper_bound_ci"]),
                "reference_configuration": row["reference_run_id"],
                "branch_status": row["branch_status"],
            }
        )
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table_rows).to_csv(table_path, index=False)
    audit.add_output(table_path)


# -----------------------------------------------------------------------------
# Modal core/tail/phase convergence from raw per-run NPZ modes
# -----------------------------------------------------------------------------


def resolve_repo_path(repo: Path, path_value: Any) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = repo / path
    return path.resolve()


def load_npz_mode(path: Path) -> dict[str, np.ndarray]:
    reject_excluded_path(path)
    with np.load(path) as payload:
        required = {"y"} | {
            f"{field}_{part}" for field in FIELDS for part in ("real", "imag")
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise KeyError(f"Mode {path} is missing arrays: {missing}")
        result = {name: np.asarray(payload[name]) for name in required}
    y = np.asarray(result["y"], dtype=float)
    if len(y) < 3 or not np.all(np.diff(y) > 0.0):
        raise ValueError(f"Mode grid is not strictly increasing: {path}")
    return result


def mode_complex(mode: dict[str, np.ndarray], field: str) -> np.ndarray:
    return np.asarray(mode[f"{field}_real"], float) + 1j * np.asarray(
        mode[f"{field}_imag"], float
    )


def interpolate_complex(
    source_y: np.ndarray, source_values: np.ndarray, target_y: np.ndarray
) -> np.ndarray:
    return np.interp(target_y, source_y, source_values.real) + 1j * np.interp(
        target_y, source_y, source_values.imag
    )


def trapezoidal_weights(y: np.ndarray) -> np.ndarray:
    weights = np.empty_like(y, dtype=float)
    weights[0] = 0.5 * (y[1] - y[0])
    weights[-1] = 0.5 * (y[-1] - y[-2])
    weights[1:-1] = 0.5 * (y[2:] - y[:-2])
    return weights


def weighted_inner(a: np.ndarray, b: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> complex:
    return complex(np.sum(weights[mask] * np.conjugate(a[mask]) * b[mask]))


def weighted_norm(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    value = weighted_inner(values, values, weights, mask).real
    return float(np.sqrt(max(value, 0.0)))


def modal_pair_metrics(test: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, Any]:
    test_y = np.asarray(test["y"], float)
    reference_y = np.asarray(reference["y"], float)
    lower = max(float(test_y.min()), float(reference_y.min()))
    upper = min(float(test_y.max()), float(reference_y.max()))
    common = (reference_y >= lower) & (reference_y <= upper)
    common_y = reference_y[common]
    if common_y.size < 16:
        raise ValueError("Fewer than 16 common reference-grid points.")

    test_fields: dict[str, np.ndarray] = {}
    reference_fields: dict[str, np.ndarray] = {}
    for name in FIELDS:
        test_fields[name] = interpolate_complex(
            test_y, mode_complex(test, name), common_y
        )
        reference_fields[name] = mode_complex(reference, name)[common]

    p_test = test_fields["p"]
    p_reference = reference_fields["p"]
    pressure_scale = float(np.max(np.abs(p_reference)))
    if pressure_scale <= 0.0:
        raise ValueError("Degenerate reference pressure mode.")

    core = np.abs(p_reference) >= 1.0e-4 * pressure_scale
    tail = (~core) & ~(
        (np.abs(p_reference) < 1.0e-12 * pressure_scale)
        & (np.abs(p_test) < 1.0e-12 * pressure_scale)
    )
    full = core | tail
    if int(core.sum()) < 8 or int(full.sum()) < 16:
        raise ValueError("Insufficient core/full points for modal comparison.")

    weights = trapezoidal_weights(common_y)
    denominator = weighted_inner(p_test, p_test, weights, core)
    if abs(denominator) <= np.finfo(float).eps:
        raise ValueError("Degenerate test pressure in modal core.")
    a_star = weighted_inner(p_test, p_reference, weights, core) / denominator
    aligned = {name: a_star * values for name, values in test_fields.items()}

    def rel_error(name: str, mask: np.ndarray, denominator_mask: np.ndarray | None = None) -> float:
        denominator_mask = mask if denominator_mask is None else denominator_mask
        denominator_norm = weighted_norm(
            reference_fields[name], weights, denominator_mask
        )
        if denominator_norm <= np.finfo(float).eps:
            return float("nan")
        return weighted_norm(
            reference_fields[name] - aligned[name], weights, mask
        ) / denominator_norm

    p_test_norm = weighted_norm(p_test, weights, core)
    p_ref_norm = weighted_norm(p_reference, weights, core)
    correlation = abs(weighted_inner(p_test, p_reference, weights, core)) / max(
        p_test_norm * p_ref_norm, np.finfo(float).eps
    )
    phase_mismatch = float(np.clip(1.0 - correlation, 0.0, 1.0))

    return {
        "A_star_real": float(a_star.real),
        "A_star_imag": float(a_star.imag),
        "core_relative_L2_p": rel_error("p", core),
        "tail_contribution_p": rel_error("p", tail, full) if tail.any() else 0.0,
        "full_relative_L2_p": rel_error("p", full),
        "phase_mismatch_p": phase_mismatch,
        "full_relative_L2_rho": rel_error("rho", full),
        "full_relative_L2_u": rel_error("u", full),
        "full_relative_L2_v": rel_error("v", full),
        "n_common_points": int(common_y.size),
        "n_core_points": int(core.sum()),
        "n_tail_points": int(tail.sum()),
    }


def build_modal_convergence_asset(
    repo: Path,
    spectral_frame: pd.DataFrame,
    figure_path: Path,
    table_path: Path,
    report_path: Path,
    dpi: int,
    audit: Audit,
) -> bool:
    rows: list[dict[str, Any]] = []
    excluded_modes: list[str] = []

    by_run = spectral_frame.set_index("run_id", drop=False)
    cache: dict[Path, dict[str, np.ndarray]] = {}

    for _, row in spectral_frame.iterrows():
        reference_id = str(row["reference_run_id"])
        if str(row["run_id"]) == reference_id:
            audit.excluded.append(
                f"Modal curve reference run excluded from errors: {reference_id}."
            )
            continue
        if reference_id not in by_run.index:
            excluded_modes.append(f"{row['run_id']}: reference row absent after filtering")
            continue
        reference_row = by_run.loc[reference_id]
        test_path = resolve_repo_path(repo, row["mode_file"])
        reference_path = resolve_repo_path(repo, reference_row["mode_file"])
        if not test_path.is_file() or not reference_path.is_file():
            excluded_modes.append(
                f"{row['run_id']}: missing test/reference NPZ "
                f"({test_path.is_file()=}, {reference_path.is_file()=})"
            )
            continue
        try:
            if test_path not in cache:
                cache[test_path] = load_npz_mode(test_path)
                audit.add_input(test_path)
            if reference_path not in cache:
                cache[reference_path] = load_npz_mode(reference_path)
                audit.add_input(reference_path)
            metrics = modal_pair_metrics(cache[test_path], cache[reference_path])
        except Exception as exc:  # preserve an auditable exclusion rather than hiding it
            excluded_modes.append(f"{row['run_id']}: {type(exc).__name__}: {exc}")
            continue

        rows.append(
            {
                "Mach": row["Mach"],
                "alpha": row["alpha"],
                "study_type": (
                    "domain_size"
                    if row["sweep_type"] == "shooting_box"
                    else "integration_accuracy"
                ),
                "configuration": row["configuration"],
                "reference_configuration": reference_id,
                **metrics,
                "n_common_points": metrics["n_common_points"],
                "n_core_points": metrics["n_core_points"],
                "n_tail_points": metrics["n_tail_points"],
                "branch_status": row["branch_status"],
                "Ly": row["Ly"],
                "rtol": row["rtol"],
                "atol": row["atol"],
                "max_step": row["max_step"],
                "accuracy_order": row.get("accuracy_order", np.nan),
                "run_id": row["run_id"],
            }
        )

    for item in excluded_modes:
        audit.excluded.append(f"Modal convergence: {item}")

    if not rows:
        if figure_path.exists():
            figure_path.unlink()
        if figure_path.with_suffix(".png").exists():
            figure_path.with_suffix(".png").unlink()
        if table_path.exists():
            table_path.unlink()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_text = (
            "ASSET 5 NOT GENERATED\n"
            "=====================\n\n"
            "The audited convergence tables identify valid M=1.8 and M=1.9 runs, "
            "but the raw complex NPZ modes required to recompute interpolation, "
            "one common complex alignment factor, core/tail errors and phase mismatch "
            "were not accessible. Existing precomputed 0.44--0.58 modal errors were "
            "not reused as evidence of convergence.\n\n"
            "Excluded/missing mode details:\n- "
            + "\n- ".join(excluded_modes)
            + "\n"
        )
        report_path.write_text(report_text, encoding="utf-8")
        audit.add_output(report_path)
        audit.warnings.append(
            "Asset 5 not generated because raw complex per-run NPZ modes were unavailable."
        )
        print(f"Asset 5 not generated; report saved: {report_path}")
        return False

    table = pd.DataFrame(rows)
    requested_columns = [
        "Mach",
        "alpha",
        "study_type",
        "configuration",
        "reference_configuration",
        "A_star_real",
        "A_star_imag",
        "core_relative_L2_p",
        "tail_contribution_p",
        "full_relative_L2_p",
        "phase_mismatch_p",
        "full_relative_L2_rho",
        "full_relative_L2_u",
        "full_relative_L2_v",
        "n_common_points",
        "n_core_points",
        "n_tail_points",
        "branch_status",
    ]
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table[requested_columns].to_csv(table_path, index=False)
    audit.add_output(table_path)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.5), constrained_layout=True)
    metrics = (
        ("core_relative_L2_p", "core relative L2", "o"),
        ("tail_contribution_p", "tail contribution", "s"),
        ("phase_mismatch_p", "phase mismatch", "^"),
    )

    box = table.loc[table["study_type"].eq("domain_size")].copy()
    for mach, linestyle in ((1.8, "-"), (1.9, "--")):
        group = box.loc[np.isclose(box["Mach"], mach)].sort_values("Ly")
        for metric, label, marker in metrics:
            values = group[metric].to_numpy(float)
            positive = values > 0.0
            if positive.any():
                axes[0].plot(
                    group.loc[positive, "Ly"],
                    values[positive],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.4,
                    label=f"M={mach:.1f}, {label}",
                )
            zero = ~positive & np.isfinite(values)
            if zero.any():
                axes[0].scatter(
                    group.loc[zero, "Ly"],
                    np.full(int(zero.sum()), UPPER_BOUND),
                    marker="v",
                    facecolors="none",
                        )
    axes[0].set_xlabel(r"Domain half-width $L_y$")
    axes[0].set_ylabel("Modal diagnostic")
    axes[0].set_title("(a) Domain-size convergence", loc="left")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.23, which="both")
    axes[0].legend(loc="best", ncol=2)

    accuracy = table.loc[table["study_type"].eq("integration_accuracy")].copy()
    order = (
        accuracy[["configuration", "accuracy_order"]]
        .drop_duplicates()
        .sort_values("accuracy_order")
    )
    labels = order["configuration"].astype(str).tolist()
    x_map = {name: idx for idx, name in enumerate(labels)}
    for mach, linestyle in ((1.8, "-"), (1.9, "--")):
        group = accuracy.loc[np.isclose(accuracy["Mach"], mach)].copy()
        group["_x"] = group["configuration"].map(x_map)
        group = group.sort_values("_x")
        for metric, label, marker in metrics:
            values = group[metric].to_numpy(float)
            positive = values > 0.0
            if positive.any():
                axes[1].plot(
                    group.loc[positive, "_x"],
                    values[positive],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.4,
                    label=f"M={mach:.1f}, {label}",
                )
            zero = ~positive & np.isfinite(values)
            if zero.any():
                axes[1].scatter(
                    group.loc[zero, "_x"],
                    np.full(int(zero.sum()), UPPER_BOUND),
                    marker="v",
                    facecolors="none",
                        )
    axes[1].set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axes[1].set_xlabel("Exact integration configuration")
    axes[1].set_ylabel("Modal diagnostic")
    axes[1].set_title("(b) Integration-parameter convergence", loc="left")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.23, which="both")
    axes[1].legend(loc="best", ncol=2)

    fig.suptitle(
        "Supersonic modal convergence after common complex alignment",
        fontsize=13.3,
    )
    save_figure(fig, figure_path, dpi, audit)
    if report_path.exists():
        report_path.unlink()

    for metric, _, _ in metrics:
        audit.diagnostics.append(
            f"Modal {metric} range: {finite_minmax(table[metric])}."
        )
    return True


# -----------------------------------------------------------------------------
# Final report
# -----------------------------------------------------------------------------


def write_report(path: Path, audit: Audit, modal_asset_generated: bool) -> None:
    sections = [
        ("INPUT FILES USED", audit.inputs),
        ("EXCLUDED CONFIGURATIONS / DATA", audit.excluded),
        ("REFERENCE CONFIGURATIONS", audit.references),
        ("UPPER-BOUND POLICY", audit.upper_bounds),
        ("NUMERICAL DIAGNOSTICS", audit.diagnostics),
        ("WARNINGS / UNAVAILABLE PANELS", audit.warnings),
        ("OUTPUT FILES", audit.outputs),
    ]
    lines = [
        "CLASSICAL SUPERSONIC ARTICLE-ASSET AUDIT REPORT",
        "================================================",
        "",
        "No spectral solve or new numerical campaign was launched.",
        "Scientific figure text is in English.",
        "Mach values used for convergence assets: 1.8 and 1.9 only.",
        "Canonical M=1.4 is used only for the representative modal assets.",
        f"Asset 5 generated from raw modes: {modal_asset_generated}.",
        "No path matching QUARANTINE_INVALID/QUARANTINE was read.",
        "",
    ]
    for title, values in sections:
        lines.extend([title, "-" * len(title)])
        if values:
            lines.extend(f"- {value}" for value in values)
        else:
            lines.append("- None")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    audit.add_output(path)
    print(f"Saved audit report: {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    repo = find_repo_root(args.repo_root)
    audit = Audit()

    figure_directory = repo / "assets/article/classical_supersonic/figures"
    table_directory = repo / "assets/article/classical_supersonic/tables"
    figure_directory.mkdir(parents=True, exist_ok=True)
    table_directory.mkdir(parents=True, exist_ok=True)

    mode, metadata = load_canonical_mode(repo, audit)
    plot_representative_mode(
        mode,
        metadata,
        figure_directory / "Fig_supersonic_representative_mode_M140_a018.pdf",
        args.dpi,
        audit,
    )
    build_modal_reconstruction_assets(
        mode,
        metadata,
        figure_directory
        / "Fig_supersonic_modal_reconstruction_residuals_M140_a018.pdf",
        table_directory,
        args.dpi,
        audit,
    )

    spectral = load_spectral_error_table(repo, audit)
    build_eigenvalue_convergence_asset(
        spectral,
        figure_directory / "Fig_supersonic_eigenvalue_convergence.pdf",
        table_directory / "Tab_supersonic_eigenvalue_convergence.csv",
        args.dpi,
        audit,
    )
    build_box_matching_asset(
        spectral,
        figure_directory / "Fig_supersonic_box_and_matching_robustness.pdf",
        table_directory / "Tab_supersonic_box_and_matching_robustness.csv",
        args.dpi,
        audit,
    )

    modal_asset_generated = build_modal_convergence_asset(
        repo,
        spectral,
        figure_directory / "Fig_supersonic_core_tail_phase_convergence.pdf",
        table_directory / "Tab_supersonic_core_tail_phase_convergence.csv",
        table_directory / "Report_supersonic_core_tail_phase_unavailable.txt",
        args.dpi,
        audit,
    )

    report_path = table_directory / "Supersonic_asset_generation_report.txt"
    write_report(report_path, audit, modal_asset_generated)

    print("\nFINAL SUMMARY")
    print("=============")
    print(f"Repository: {repo}")
    print("Canonical mode: M=1.4, alpha=0.18")
    print("Convergence modes: M=1.8 and M=1.9 only")
    print(f"Asset 5 generated from raw NPZ modes: {modal_asset_generated}")
    print(f"Upper-bound display threshold: < {UPPER_BOUND:.0e}")
    print("No QUARANTINE_INVALID path was read.")
    print(f"Detailed report: {report_path}")


if __name__ == "__main__":
    main()
