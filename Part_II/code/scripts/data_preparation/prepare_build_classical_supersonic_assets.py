#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

MAP_FILENAMES = {
    "cr": "cr_map.pdf",
    "ci": "ci_map.pdf",
    "omega_i": "omega_i_map.pdf",
}


def resolve_path(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def alpha_grid(config: dict[str, Any]) -> np.ndarray:
    start = float(config["alpha_min"])
    stop = float(config["alpha_max"])
    step = float(config["alpha_step"])
    count = int(round((stop - start) / step)) + 1
    return start + step * np.arange(count, dtype=float)


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5])
    mid = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate([[first], mid, [last]])


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def key_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Mach_key": pd.to_numeric(frame["Mach"], errors="raise").round(10),
            "alpha_key": pd.to_numeric(frame["alpha"], errors="raise").round(12),
        }
    )


def build_reference_csv(
    *,
    retained: pd.DataFrame,
    modes: pd.DataFrame,
    output: Path,
    freeze_id: str,
) -> pd.DataFrame:
    retained = retained.copy()
    modes = modes.copy()
    retained[["_Mach", "_alpha"]] = key_columns(retained).to_numpy()
    modes[["_Mach", "_alpha"]] = key_columns(modes).to_numpy()
    mode_metadata_columns = [
        name
        for name in (
            "Mach",
            "alpha",
            "status",
            "gamma_mismatch_at_match",
            "numerical_extent",
            "tail_tolerance",
            "far_left",
            "far_right",
            "mu_left",
            "mu_right",
            "n_coordinates",
        )
        if name in modes.columns
    ]
    mode_metadata = modes[mode_metadata_columns + ["_Mach", "_alpha"]].copy()
    rename = {
        name: f"mode_{name}"
        for name in mode_metadata_columns
        if name not in ("Mach", "alpha")
    }
    mode_metadata = mode_metadata.drop(columns=[c for c in ("Mach", "alpha") if c in mode_metadata])
    mode_metadata = mode_metadata.rename(columns=rename)
    reference = retained.merge(mode_metadata, on=["_Mach", "_alpha"], how="left", validate="one_to_one")
    reference["mode_available"] = reference.get("mode_status", pd.Series(index=reference.index, dtype=object)).eq("converged")
    reference["freeze_id"] = freeze_id
    reference["reference_status"] = "retained_classical_supersonic"
    reference = reference.drop(columns=["_Mach", "_alpha"])
    reference = reference.sort_values(["Mach", "alpha"]).reset_index(drop=True)
    if not reference["mode_available"].all():
        missing = reference.loc[~reference["mode_available"], ["Mach", "alpha"]].to_dict("records")
        raise ValueError(f"Missing modal data for retained points: {missing[:20]}")
    atomic_csv(output, reference)
    return reference


def pivot_grid(
    frame: pd.DataFrame,
    *,
    mach_values: np.ndarray,
    alpha_values: np.ndarray,
    value: str,
) -> np.ma.MaskedArray:
    table = frame.pivot(index="Mach", columns="alpha", values=value)
    table.index = pd.to_numeric(table.index, errors="raise").round(10)
    table.columns = pd.to_numeric(table.columns, errors="raise").round(12)
    table = table.reindex(
        index=np.round(mach_values, 10),
        columns=np.round(alpha_values, 12),
    )
    array = table.to_numpy(dtype=float)
    return np.ma.masked_invalid(array)


def save_map(
    *,
    matrix: np.ma.MaskedArray,
    mach_values: np.ndarray,
    alpha_values: np.ndarray,
    label: str,
    title: str,
    output: Path,
    cmap: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    mesh = ax.pcolormesh(
        centers_to_edges(alpha_values),
        centers_to_edges(mach_values),
        matrix,
        shading="flat",
        cmap=cmap,
        rasterized=True,
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label(label)
    ax.set_xlabel(r"Wavenumber $\alpha$")
    ax.set_ylabel(r"Mach number $M$")
    ax.set_title(title)
    ax.set_xlim(alpha_values.min(), alpha_values.max())
    ax.set_ylim(mach_values.min(), mach_values.max())
    ax.grid(False)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def neutral_table(
    *,
    targets: pd.DataFrame,
    retained: pd.DataFrame,
    mach_values: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for Mach in mach_values:
        target_subset = targets[np.isclose(pd.to_numeric(targets["Mach"]), Mach, atol=5e-11)].copy()
        retained_subset = retained[np.isclose(pd.to_numeric(retained["Mach"]), Mach, atol=5e-11)].copy()
        neutral = pd.to_numeric(target_subset.get("neutral_alpha_estimate"), errors="coerce").dropna()
        estimate = float(neutral.iloc[-1]) if len(neutral) else math.nan
        fit_r2 = pd.to_numeric(target_subset.get("neutral_fit_r2"), errors="coerce").dropna()
        rows.append(
            {
                "Mach": float(Mach),
                "neutral_alpha_estimate": estimate,
                "neutral_fit_r2": float(fit_r2.iloc[-1]) if len(fit_r2) else math.nan,
                "alpha_max_retained": float(pd.to_numeric(retained_subset["alpha"]).max())
                if not retained_subset.empty
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def save_neutral_curve(table: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)
    finite = table[np.isfinite(table["neutral_alpha_estimate"])].copy()
    if not finite.empty:
        ax.plot(
            finite["Mach"],
            finite["neutral_alpha_estimate"],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=r"Estimated neutral boundary $\alpha_n(M)$",
        )
    ax.scatter(
        table["Mach"],
        table["alpha_max_retained"],
        marker="x",
        s=24,
        label="Largest retained target",
    )
    ax.set_xlabel(r"Mach number $M$")
    ax.set_ylabel(r"Wavenumber $\alpha$")
    ax.set_title("Classical supersonic neutral boundary")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_mask(
    *,
    retained: pd.DataFrame,
    mach_values: np.ndarray,
    alpha_values: np.ndarray,
    output: Path,
) -> None:
    retained_keys = set(
        zip(
            pd.to_numeric(retained["Mach"]).round(10),
            pd.to_numeric(retained["alpha"]).round(12),
        )
    )
    mask = np.zeros((mach_values.size, alpha_values.size), dtype=float)
    for i, Mach in enumerate(mach_values):
        for j, alpha in enumerate(alpha_values):
            mask[i, j] = float((round(float(Mach), 10), round(float(alpha), 12)) in retained_keys)
    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    mesh = ax.pcolormesh(
        centers_to_edges(alpha_values),
        centers_to_edges(mach_values),
        mask,
        shading="flat",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    colorbar = fig.colorbar(mesh, ax=ax, ticks=[0.0, 1.0], pad=0.02)
    colorbar.ax.set_yticklabels(["Not retained", "Retained"])
    ax.set_xlabel(r"Wavenumber $\alpha$")
    ax.set_ylabel(r"Mach number $M$")
    ax.set_title(f"Retained classical points: {int(mask.sum())}/{mask.size}")
    ax.set_xlim(alpha_values.min(), alpha_values.max())
    ax.set_ylim(mach_values.min(), mach_values.max())
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def downsample_indices(size: int, maximum: int) -> np.ndarray:
    if size <= maximum:
        return np.arange(size)
    return np.unique(np.linspace(0, size - 1, maximum, dtype=int))


def mode_records(per_mach_root: Path) -> Iterable[tuple[float, Path]]:
    for mach_dir in sorted(per_mach_root.glob("M*")):
        npz = mach_dir / "modes" / "modes_compact_with_analytic_tails.npz"
        if npz.is_file():
            yield float(mach_dir.name[1:].replace("p", ".")), npz


def add_cover_page(
    pdf: PdfPages,
    *,
    reference: pd.DataFrame,
    freeze_id: str,
    modes_per_page: int,
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.78, "Classical supersonic modes", ha="center", fontsize=22, weight="bold")
    fig.text(0.5, 0.69, f"Freeze: {freeze_id}", ha="center", fontsize=12)
    fig.text(
        0.5,
        0.58,
        f"{len(reference)} retained eigenpairs and reconstructed pressure modes",
        ha="center",
        fontsize=14,
    )
    fig.text(
        0.12,
        0.40,
        "Each mode contains:\n"
        "- numerical Riccati reconstruction in the compact core;\n"
        "- normalized pressure modulus, real part and imaginary part;\n"
        "- analytic far-field envelope extensions;\n"
        f"- {modes_per_page} modes per page after Mach section pages.",
        fontsize=12,
        va="top",
    )
    fig.text(0.5, 0.08, "Generated from the frozen dense classical supersonic reference.", ha="center", fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_mach_page(pdf: PdfPages, *, Mach: float, count: int) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.58, rf"Mach $M={Mach:.2f}$", ha="center", fontsize=26, weight="bold")
    fig.text(0.5, 0.45, f"{count} retained modes", ha="center", fontsize=14)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_one_mode(
    *,
    ax_core: plt.Axes,
    ax_tail: plt.Axes,
    payload: Any,
    index: int,
    core_limit: float,
    max_plot_points: int,
) -> None:
    y = np.asarray(payload["y"][index], dtype=float)
    region = np.asarray(payload["region"][index], dtype=int)
    modulus = np.asarray(payload["modulus"][index], dtype=float)
    p_real = np.asarray(payload["p_real"][index], dtype=float)
    p_imag = np.asarray(payload["p_imag"][index], dtype=float)
    alpha = float(payload["alpha"][index])
    cr = float(payload["cr"][index])
    ci = float(payload["ci"][index])
    omega_i = float(payload["omega_i"][index])

    core = (region == 0) & (np.abs(y) <= core_limit)
    core_indices = np.flatnonzero(core)
    core_indices = core_indices[downsample_indices(core_indices.size, max_plot_points)]
    ax_core.plot(y[core_indices], p_real[core_indices], linewidth=0.9, label=r"$\Re p$")
    ax_core.plot(y[core_indices], p_imag[core_indices], linewidth=0.9, label=r"$\Im p$")
    ax_core.plot(y[core_indices], modulus[core_indices], linewidth=1.1, label=r"$|p|$")
    ax_core.axhline(0.0, linewidth=0.5)
    ax_core.set_xlim(-core_limit, core_limit)
    ax_core.set_ylim(-1.08, 1.08)
    ax_core.set_xlabel(r"$y$")
    ax_core.set_ylabel("Normalized pressure")
    ax_core.grid(True, alpha=0.2)
    ax_core.legend(loc="upper right", fontsize=6, ncol=3)
    ax_core.set_title(
        rf"$\alpha={alpha:.5f}$, $c={cr:.7f}+{ci:.3e}i$, $\omega_i={omega_i:.3e}$",
        fontsize=8,
    )

    all_indices = downsample_indices(y.size, max_plot_points)
    positive = np.maximum(modulus[all_indices], 1e-300)
    ax_tail.plot(y[all_indices], positive, linewidth=1.0)
    ax_tail.set_xscale("symlog", linthresh=core_limit)
    ax_tail.set_yscale("log")
    ax_tail.set_ylim(max(1e-8, float(np.nanmin(positive)) * 0.5), 1.2)
    ax_tail.set_xlabel(r"$y$ (symlog)")
    ax_tail.set_ylabel(r"$|p|$")
    ax_tail.grid(True, which="both", alpha=0.2)
    ax_tail.set_title("Full envelope with analytic tails", fontsize=8)


def build_all_modes_pdf(
    *,
    per_mach_root: Path,
    reference: pd.DataFrame,
    output: Path,
    freeze_id: str,
    modes_per_page: int,
    core_limit: float,
    max_plot_points: int,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    page_count = 0
    with PdfPages(temporary) as pdf:
        add_cover_page(pdf, reference=reference, freeze_id=freeze_id, modes_per_page=modes_per_page)
        page_count += 1
        total_modes = 0
        for Mach, path in mode_records(per_mach_root):
            with np.load(path, allow_pickle=False) as payload:
                required = {
                    "Mach", "alpha", "cr", "ci", "omega_i", "y", "region", "modulus", "p_real", "p_imag"
                }
                missing = required.difference(payload.files)
                if missing:
                    raise KeyError(f"{path}: missing NPZ arrays {sorted(missing)}")
                n_modes = int(payload["alpha"].shape[0])
                add_mach_page(pdf, Mach=Mach, count=n_modes)
                page_count += 1
                order = np.argsort(payload["alpha"])
                for start in range(0, n_modes, modes_per_page):
                    batch = order[start : start + modes_per_page]
                    fig, axes = plt.subplots(
                        len(batch),
                        2,
                        figsize=(11.69, 8.27),
                        squeeze=False,
                        constrained_layout=True,
                    )
                    fig.suptitle(rf"Classical pressure modes - $M={Mach:.2f}$", fontsize=13)
                    for row_index, mode_index in enumerate(batch):
                        plot_one_mode(
                            ax_core=axes[row_index, 0],
                            ax_tail=axes[row_index, 1],
                            payload=payload,
                            index=int(mode_index),
                            core_limit=core_limit,
                            max_plot_points=max_plot_points,
                        )
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    page_count += 1
                total_modes += n_modes
        if total_modes != len(reference):
            raise ValueError(
                f"NPZ mode count {total_modes} does not match reference rows {len(reference)}."
            )
    os.replace(temporary, output)
    return page_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen dense classical supersonic maps and one all-modes PDF."
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=Path("assets/classic_supersonic/dense_kappa_q_campaign_v1_FINAL_FREEZE"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--modes-per-page", type=int, default=2)
    parser.add_argument("--core-limit", type=float, default=20.0)
    parser.add_argument("--max-plot-points", type=int, default=1600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.modes_per_page < 1 or args.modes_per_page > 4:
        raise ValueError("--modes-per-page must be between 1 and 4.")
    repo = args.repo.expanduser().resolve()
    freeze_dir = resolve_path(repo, args.freeze_dir)
    metadata = load_json(freeze_dir / "freeze_metadata.json")
    config = load_json(freeze_dir / "provenance" / "dense_supersonic_campaign_config.json")
    aggregate = freeze_dir / "frozen_results" / "aggregated"
    per_mach = freeze_dir / "frozen_results" / "per_mach"
    maps_dir = freeze_dir / "classical_supersonic_maps"
    all_modes_pdf = freeze_dir / "classical_supersonic_all_modes.pdf"
    outputs = [maps_dir / name for name in MAP_FILENAMES.values()] + [
        maps_dir / "neutral_curve.pdf",
        maps_dir / "retained_point_mask.pdf",
        maps_dir / "classical_supersonic_dense_reference.csv",
        all_modes_pdf,
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Assets already exist. Use --overwrite only when intentionally regenerating: "
            + ", ".join(str(path) for path in existing)
        )
    maps_dir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(aggregate / "dense_spectral_targets.csv")
    retained = pd.read_csv(aggregate / "dense_spectral_retained.csv")
    modes = pd.read_csv(aggregate / "all_mode_summaries.csv")
    freeze_id = str(metadata.get("freeze_id", freeze_dir.name))
    reference = build_reference_csv(
        retained=retained,
        modes=modes,
        output=maps_dir / "classical_supersonic_dense_reference.csv",
        freeze_id=freeze_id,
    )

    mach_values = np.asarray([float(value) for value in config["mach_values"]], dtype=float)
    alpha_values = alpha_grid(config)
    for field, filename in MAP_FILENAMES.items():
        matrix = pivot_grid(
            retained,
            mach_values=mach_values,
            alpha_values=alpha_values,
            value=field,
        )
        labels = {
            "cr": (r"Phase speed $c_r$", r"Classical supersonic $c_r(M,\alpha)$", "viridis"),
            "ci": (r"Growth component $c_i$", r"Classical supersonic $c_i(M,\alpha)$", "magma"),
            "omega_i": (r"Temporal growth rate $\omega_i=\alpha c_i$", r"Classical supersonic $\omega_i(M,\alpha)$", "magma"),
        }
        label, title, cmap = labels[field]
        save_map(
            matrix=matrix,
            mach_values=mach_values,
            alpha_values=alpha_values,
            label=label,
            title=title,
            output=maps_dir / filename,
            cmap=cmap,
        )

    neutral = neutral_table(targets=targets, retained=retained, mach_values=mach_values)
    save_neutral_curve(neutral, maps_dir / "neutral_curve.pdf")
    save_mask(
        retained=retained,
        mach_values=mach_values,
        alpha_values=alpha_values,
        output=maps_dir / "retained_point_mask.pdf",
    )
    pages = build_all_modes_pdf(
        per_mach_root=per_mach,
        reference=reference,
        output=all_modes_pdf,
        freeze_id=freeze_id,
        modes_per_page=args.modes_per_page,
        core_limit=args.core_limit,
        max_plot_points=args.max_plot_points,
    )

    print(f"Reference rows: {len(reference)}")
    print(f"All-modes PDF pages: {pages}")
    print(f"Maps written to: {maps_dir}")
    print(f"Modes PDF: {all_modes_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
