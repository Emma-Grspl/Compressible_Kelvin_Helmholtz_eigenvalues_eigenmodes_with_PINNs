#!/usr/bin/env python3
"""Generate modal-reconstruction assets for the classical supersonic KH reference."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--mach", type=float, default=1.4)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def find_repo_root(start: Path) -> Path:
    start = start.expanduser().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "assets").is_dir():
            return candidate
    raise FileNotFoundError("Impossible de trouver la racine du dépôt contenant assets/.")


def pick_column(
    df: pd.DataFrame,
    names: Iterable[str],
    *,
    required: bool = True,
    label: str = "",
) -> str | None:
    exact = {str(column).casefold(): str(column) for column in df.columns}
    normalized = {
        re.sub(r"[^a-z0-9]+", "", str(column).casefold()): str(column)
        for column in df.columns
    }

    for name in names:
        if name.casefold() in exact:
            return exact[name.casefold()]
        key = re.sub(r"[^a-z0-9]+", "", name.casefold())
        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            f"Colonne introuvable pour {label or tuple(names)}. "
            f"Colonnes disponibles : {list(df.columns)}"
        )
    return None


def pick_complex_columns(df: pd.DataFrame, field: str) -> tuple[str, str]:
    real_names = (
        f"{field}_real",
        f"{field}_re",
        f"real_{field}",
        f"re_{field}",
        f"{field}r",
    )
    imag_names = (
        f"{field}_imag",
        f"{field}_im",
        f"imag_{field}",
        f"im_{field}",
        f"{field}i",
    )
    return (
        pick_column(df, real_names, label=f"Re({field})"),
        pick_column(df, imag_names, label=f"Im({field})"),
    )


def save_both(fig: plt.Figure, pdf_path: Path, dpi: int) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = pdf_path.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def load_selected_mode(
    modes_path: Path,
    index_path: Path,
    target_mach: float,
    target_alpha: float,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, str]]:
    modes = pd.read_csv(modes_path, compression="infer")
    index = pd.read_csv(index_path)

    mach_i = pick_column(index, ("Mach", "mach", "M", "mach_number"), label="Mach index")
    alpha_i = pick_column(index, ("alpha", "Alpha", "wavenumber"), label="alpha index")
    cr_i = pick_column(index, ("cr", "c_r", "classical_cr", "real_c"), label="cr index")
    ci_i = pick_column(index, ("ci", "c_i", "classical_ci", "imag_c"), label="ci index")

    for column in (mach_i, alpha_i, cr_i, ci_i):
        index[column] = pd.to_numeric(index[column], errors="coerce")

    candidates = index.dropna(subset=[mach_i, alpha_i, cr_i, ci_i]).copy()
    candidates["_distance"] = (
        (candidates[mach_i] - target_mach) ** 2
        + (candidates[alpha_i] - target_alpha) ** 2
    )
    if candidates.empty:
        raise ValueError("L'index modal ne contient aucun cas exploitable.")

    selected = candidates.sort_values("_distance", kind="mergesort").iloc[0]
    metadata = {
        "Mach": float(selected[mach_i]),
        "alpha": float(selected[alpha_i]),
        "cr": float(selected[cr_i]),
        "ci": float(selected[ci_i]),
    }

    # Strategy 1: modes_long itself contains Mach and alpha.
    mach_m = pick_column(modes, ("Mach", "mach", "M", "mach_number"), required=False)
    alpha_m = pick_column(modes, ("alpha", "Alpha", "wavenumber"), required=False)

    selected_modes: pd.DataFrame | None = None

    if mach_m is not None and alpha_m is not None:
        modes[mach_m] = pd.to_numeric(modes[mach_m], errors="coerce")
        modes[alpha_m] = pd.to_numeric(modes[alpha_m], errors="coerce")
        mask = (
            np.isclose(modes[mach_m], metadata["Mach"], atol=5e-6, rtol=0.0)
            & np.isclose(modes[alpha_m], metadata["alpha"], atol=5e-6, rtol=0.0)
        )
        selected_modes = modes.loc[mask].copy()

    # Strategy 2: common mode/case identifier.
    if selected_modes is None or selected_modes.empty:
        for identifier in ("mode_id", "point_id", "case_id", "spectral_id", "id"):
            mode_key = pick_column(modes, (identifier,), required=False)
            index_key = pick_column(index, (identifier,), required=False)
            if mode_key is not None and index_key is not None:
                selected_modes = modes.loc[modes[mode_key] == selected[index_key]].copy()
                if not selected_modes.empty:
                    break

    # Strategy 3: index stores a row range in the long table.
    if selected_modes is None or selected_modes.empty:
        start_col = pick_column(
            index,
            ("start_row", "row_start", "start", "offset", "mode_start", "first_row"),
            required=False,
        )
        end_col = pick_column(
            index,
            ("end_row", "row_end", "end", "stop", "mode_end", "last_row"),
            required=False,
        )
        count_col = pick_column(
            index,
            ("n_rows", "row_count", "n_points", "count", "length"),
            required=False,
        )

        if start_col is not None:
            start = int(selected[start_col])
            if end_col is not None:
                end = int(selected[end_col])
                selected_modes = modes.iloc[start:end].copy()
            elif count_col is not None:
                count = int(selected[count_col])
                selected_modes = modes.iloc[start : start + count].copy()

    if selected_modes is None or selected_modes.empty:
        raise RuntimeError(
            "Impossible de relier l'index au fichier modal long. "
            "Le script gère Mach/alpha, un identifiant commun, ou une plage de lignes."
        )

    y_col = pick_column(selected_modes, ("y", "Y", "eta"), label="y")
    kappa_col = pick_column(selected_modes, ("kappa", "kap", "κ"), label="kappa")
    q_col = pick_column(selected_modes, ("q",), label="q")

    complex_columns: dict[str, tuple[str, str]] = {
        field: pick_complex_columns(selected_modes, field)
        for field in ("p", "rho", "u", "v")
    }

    numeric_columns = [y_col, kappa_col, q_col]
    for real_col, imag_col in complex_columns.values():
        numeric_columns.extend((real_col, imag_col))
    for column in numeric_columns:
        selected_modes[column] = pd.to_numeric(selected_modes[column], errors="coerce")

    selected_modes = (
        selected_modes.dropna(subset=[y_col])
        .sort_values(y_col, kind="mergesort")
        .reset_index(drop=True)
    )

    columns = {
        "y": y_col,
        "kappa": kappa_col,
        "q": q_col,
    }
    for field, (real_col, imag_col) in complex_columns.items():
        columns[f"{field}_real"] = real_col
        columns[f"{field}_imag"] = imag_col

    print(
        "Selected modal case: "
        f"M={metadata['Mach']:.8g}, alpha={metadata['alpha']:.8g}, "
        f"cr={metadata['cr']:.8g}, ci={metadata['ci']:.8g}, "
        f"rows={len(selected_modes)}"
    )
    return selected_modes, metadata, columns


def build_modal_arrays(
    frame: pd.DataFrame,
    metadata: dict[str, float],
    columns: dict[str, str],
) -> dict[str, np.ndarray]:
    y = frame[columns["y"]].to_numpy(float)

    def complex_field(field: str) -> np.ndarray:
        return (
            frame[columns[f"{field}_real"]].to_numpy(float)
            + 1j * frame[columns[f"{field}_imag"]].to_numpy(float)
        )

    p = complex_field("p")
    rho = complex_field("rho")
    u = complex_field("u")
    v = complex_field("v")
    kappa = frame[columns["kappa"]].to_numpy(float)
    q = frame[columns["q"]].to_numpy(float)

    pressure_norm = float(np.nanmax(np.abs(p)))
    if not np.isfinite(pressure_norm) or pressure_norm <= 0.0:
        raise ValueError("La norme max |p| est invalide.")

    p /= pressure_norm
    rho /= pressure_norm
    u /= pressure_norm
    v /= pressure_norm

    mach = metadata["Mach"]
    alpha = metadata["alpha"]
    c = metadata["cr"] + 1j * metadata["ci"]
    velocity = np.tanh(y)
    velocity_prime = 1.0 / np.cosh(y) ** 2

    # Numerical derivative used solely as an independent reconstruction check.
    p_y_numeric = np.gradient(p, y, edge_order=2)
    gamma = kappa + 1j * q

    with np.errstate(divide="ignore", invalid="ignore"):
        p_y_from_gamma = gamma * p
        rho_from_p = mach**2 * p
        v_from_reconstruction = 1j * p_y_numeric / (alpha * (velocity - c))
        u_from_reconstruction = (
            -p / (velocity - c)
            + 1j * velocity_prime * v / (alpha * (velocity - c))
        )

    return {
        "y": y,
        "p": p,
        "rho": rho,
        "u": u,
        "v": v,
        "kappa": kappa,
        "q": q,
        "residual_p": np.abs(p_y_numeric - p_y_from_gamma),
        "residual_rho": np.abs(rho - rho_from_p),
        "residual_v": np.abs(v - v_from_reconstruction),
        "residual_u": np.abs(u - u_from_reconstruction),
    }


def plot_representative_mode(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, float],
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.2, 8.2),
        sharex=True,
        constrained_layout=True,
    )

    fields = (
        ("p", r"$p$"),
        ("rho", r"$\rho$"),
        ("u", r"$u$"),
        ("v", r"$v$"),
    )

    for axis, (field, symbol) in zip(axes.ravel(), fields):
        values = arrays[field]
        axis.plot(arrays["y"], values.real, linewidth=2.0, label=rf"$\Re({symbol})$")
        axis.plot(
            arrays["y"],
            values.imag,
            linewidth=2.0,
            linestyle="--",
            label=rf"$\Im({symbol})$",
        )
        axis.set_ylabel(symbol)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", frameon=True)

    axes[1, 0].set_xlabel(r"Transverse coordinate $y$")
    axes[1, 1].set_xlabel(r"Transverse coordinate $y$")

    fig.suptitle(
        rf"Representative supersonic mode, $\max_y|p|=1$: "
        rf"$M={metadata['Mach']:.3f}$, "
        rf"$\alpha={metadata['alpha']:.3f}$, "
        rf"$c_r={metadata['cr']:.6f}$, "
        rf"$c_i={metadata['ci']:.6f}$",
        fontsize=14,
    )

    save_both(fig, output_path, dpi)


def plot_modal_checks(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, float],
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.0),
        constrained_layout=True,
    )

    floor = np.finfo(float).tiny
    y = arrays["y"]

    axes[0].semilogy(
        y,
        np.maximum(arrays["residual_p"], floor),
        linewidth=1.9,
        label=r"$|p_y-(\kappa+iq)p|$",
    )
    axes[0].semilogy(
        y,
        np.maximum(arrays["residual_rho"], floor),
        linewidth=1.9,
        label=r"$|\rho-M^2p|$",
    )
    axes[0].semilogy(
        y,
        np.maximum(arrays["residual_v"], floor),
        linewidth=1.9,
        label=r"$|v-i p_y/[\alpha(U-c)]|$",
    )
    axes[0].semilogy(
        y,
        np.maximum(arrays["residual_u"], floor),
        linewidth=1.9,
        label=r"$|u+p/(U-c)-iU'v/[\alpha(U-c)]|$",
    )
    axes[0].set_xlabel(r"Transverse coordinate $y$")
    axes[0].set_ylabel("Absolute reconstruction residual")
    axes[0].set_title("Modal reconstruction residuals")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", frameon=True)

    pressure_abs = np.maximum(np.abs(arrays["p"]), floor)
    axes[1].semilogy(y, pressure_abs, linewidth=2.2, label=r"$|p|$")

    # Asymptotic decay predicted by d log|p| / dy = kappa.
    number_tail = max(5, len(y) // 10)
    left_slope = float(np.nanmedian(arrays["kappa"][:number_tail]))
    right_slope = float(np.nanmedian(arrays["kappa"][-number_tail:]))

    left_anchor = number_tail - 1
    right_anchor = len(y) - number_tail

    left_expected = pressure_abs[left_anchor] * np.exp(
        left_slope * (y[: left_anchor + 1] - y[left_anchor])
    )
    right_expected = pressure_abs[right_anchor] * np.exp(
        right_slope * (y[right_anchor:] - y[right_anchor])
    )

    axes[1].semilogy(
        y[: left_anchor + 1],
        left_expected,
        linestyle="--",
        linewidth=1.6,
        label=rf"left asymptotic slope $\kappa_\infty\approx{left_slope:.3f}$",
    )
    axes[1].semilogy(
        y[right_anchor:],
        right_expected,
        linestyle=":",
        linewidth=1.8,
        label=rf"right asymptotic slope $\kappa_\infty\approx{right_slope:.3f}$",
    )
    axes[1].set_xlabel(r"Transverse coordinate $y$")
    axes[1].set_ylabel(r"Normalized pressure amplitude $|p|$")
    axes[1].set_title("Asymptotic pressure-mode decay")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", frameon=True)

    fig.suptitle(
        rf"Modal validation: $M={metadata['Mach']:.3f}$, "
        rf"$\alpha={metadata['alpha']:.3f}$, "
        rf"$c_r={metadata['cr']:.6f}$, "
        rf"$c_i={metadata['ci']:.6f}$",
        fontsize=14,
    )
    save_both(fig, output_path, dpi)


def main() -> None:
    args = parse_args()
    repo = find_repo_root(args.repo_root)

    campaign = (
        repo
        / "assets"
        / "classic_supersonic"
        / "dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS"
    )
    modes_path = campaign / "classical_supersonic_final_modes_long.csv.gz"
    index_path = campaign / "classical_supersonic_final_modes_index.csv"

    for path in (modes_path, index_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_directory = (
        repo / "assets" / "article" / "classical_supersonic" / "figures"
    )

    mode_frame, metadata, columns = load_selected_mode(
        modes_path,
        index_path,
        args.mach,
        args.alpha,
    )
    arrays = build_modal_arrays(mode_frame, metadata, columns)

    plot_representative_mode(
        arrays,
        metadata,
        output_directory / "Fig_supersonic_representative_mode_M140_a018.pdf",
        args.dpi,
    )
    plot_modal_checks(
        arrays,
        metadata,
        output_directory
        / "Fig_supersonic_modal_reconstruction_residuals_M140_a018.pdf",
        args.dpi,
    )


if __name__ == "__main__":
    main()
