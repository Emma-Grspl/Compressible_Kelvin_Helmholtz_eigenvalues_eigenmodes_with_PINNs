#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

BASE_FREEZE_DEFAULT = Path(
    "assets/classic_supersonic/dense_kappa_q_campaign_v1_FINAL_FREEZE"
)
EXTENSION_DEFAULT = Path(
    "classic_supersonic/reproducibility/results/dense_kappa_q_lowM_canonical_full_branches_v1"
)
OUTPUT_DEFAULT = Path(
    "assets/classic_supersonic/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(repo: Path, value: Path) -> Path:
    value = value.expanduser()
    return value if value.is_absolute() else repo / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point_key(Mach: float, alpha: float) -> tuple[float, float]:
    return round(float(Mach), 10), round(float(alpha), 12)


def locate(columns: Iterable[str], variants: tuple[str, ...]) -> str | None:
    mapping = {str(c).strip().lower(): str(c) for c in columns}
    for variant in variants:
        found = mapping.get(variant.lower())
        if found is not None:
            return found
    return None


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def parse_ci_from_label(value: object) -> float:
    import re
    numbers = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        str(value),
    )
    return float(numbers[-1]) if numbers else math.nan


def find_blumen_points(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = resolve(repo, explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = [
        repo / "article/tables/table_blumen_ci_digitized_points.csv",
        repo / (
            "assets/classic_supersonic/"
            "supersonic_sparse_PINN_reference_v2_FINAL_FREEZE/"
            "ci/blumen_ci_digitized_points.csv"
        ),
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(repo.glob("**/blumen_ci_digitized_points.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError("Could not locate blumen_ci_digitized_points.csv")


def find_blumen_levels(repo: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        path = resolve(repo, explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = [
        repo / "article/tables/table_blumen_ci_curve_levels.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(repo.glob("**/blumen_ci_curve_levels.csv"))
    return matches[0] if matches else None


def load_blumen(
    points_path: Path,
    levels_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    points = pd.read_csv(points_path)
    mach_col = locate(points.columns, ("Mach", "M", "mach_physical"))
    alpha_col = locate(points.columns, ("alpha", "wavenumber"))
    if mach_col is None or alpha_col is None:
        raise KeyError(
            f"{points_path}: expected Mach and alpha columns; "
            f"observed {points.columns.tolist()}"
        )

    points = points.copy()
    points["Mach_input"] = pd.to_numeric(points[mach_col], errors="coerce")
    points["alpha"] = pd.to_numeric(points[alpha_col], errors="coerce")

    level_source = "unavailable"
    if "ci_level" in points.columns:
        points["ci_level"] = pd.to_numeric(points["ci_level"], errors="coerce")
        level_source = "points:ci_level"
    elif levels_path is not None and "curve_id" in points.columns:
        levels = pd.read_csv(levels_path)
        if {"curve_id", "ci_level"}.issubset(levels.columns):
            # Keep neutral ci=0 curves. Only discard explicitly excluded special cr curves.
            if "family" in levels.columns:
                levels = levels.loc[
                    ~levels["family"].astype(str).str.strip().eq("cr_special")
                ].copy()
            levels["curve_id"] = pd.to_numeric(
                levels["curve_id"], errors="coerce"
            ).astype("Int64")
            levels["ci_level"] = pd.to_numeric(
                levels["ci_level"], errors="coerce"
            )
            points["curve_id"] = pd.to_numeric(
                points["curve_id"], errors="coerce"
            ).astype("Int64")
            keep = ["curve_id", "ci_level"]
            for extra in ("curve_label", "family"):
                if extra in levels.columns:
                    keep.append(extra)
            points = points.merge(
                levels[keep],
                on="curve_id",
                how="inner",
                validate="many_to_one",
            )
            level_source = str(levels_path)
    else:
        label_col = locate(points.columns, ("curve_label", "label", "series"))
        if label_col is not None:
            points["ci_level"] = points[label_col].map(parse_ci_from_label)
            level_source = f"parsed from {label_col}"

    finite_mach = points["Mach_input"].dropna().to_numpy(float)
    if finite_mach.size == 0:
        raise RuntimeError("No finite Blumen Mach coordinates.")
    if float(np.nanmin(finite_mach)) < 1.0 and float(np.nanmax(finite_mach)) <= 1.1:
        points["Mach"] = points["Mach_input"] + 0.9
        calibration = "digitized Mach coordinate + 0.9"
    else:
        points["Mach"] = points["Mach_input"]
        calibration = "already physical Mach"

    points = points.dropna(subset=["Mach", "alpha"]).copy()
    points = points.loc[
        points["Mach"].between(0.95, 2.1)
        & points["alpha"].between(0.0, 1.0)
    ].reset_index(drop=True)
    if points.empty:
        raise RuntimeError("No Blumen points remain after validation.")

    metadata = {
        "points_path": str(points_path),
        "levels_path": str(levels_path) if levels_path else None,
        "point_count": int(len(points)),
        "neutral_point_count": int(
            np.isclose(
                pd.to_numeric(points.get("ci_level"), errors="coerce"),
                0.0,
                atol=1e-14,
            ).sum()
        ),
        "mach_calibration": calibration,
        "ci_level_source": level_source,
    }
    return points, metadata


def load_base_reference(base_freeze: Path) -> pd.DataFrame:
    path = (
        base_freeze
        / "classical_supersonic_maps/classical_supersonic_dense_reference.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["reference_subset"] = "base_dense_campaign"
    frame["reference_source_path"] = str(path)
    return frame


def find_extension_reference(extension_dir: Path) -> Path:
    candidates = [
        extension_dir / "lowM_lowalpha_dense_reference.csv",
        extension_dir / "lowM_lowalpha_spectral_retained.csv",
        extension_dir / "dense_spectral_retained.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(extension_dir.glob("*retained*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No retained extension CSV found under {extension_dir}"
    )


def load_extension_reference(extension_dir: Path) -> pd.DataFrame:
    path = find_extension_reference(extension_dir)
    frame = pd.read_csv(path)
    frame["reference_subset"] = "lowM_canonical_full_branches"
    frame["reference_source_path"] = str(path)
    return frame


def standardize_reference(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"Mach", "alpha", "cr", "ci"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Reference table missing {sorted(missing)}")
    result = frame.copy()
    for column in ("Mach", "alpha", "cr", "ci"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["Mach", "alpha", "cr", "ci"]).copy()
    if "omega_i" not in result.columns:
        result["omega_i"] = result["alpha"] * result["ci"]
    else:
        result["omega_i"] = pd.to_numeric(
            result["omega_i"], errors="coerce"
        )
        result["omega_i"] = result["omega_i"].fillna(
            result["alpha"] * result["ci"]
        )
    return result


def merge_references(base: pd.DataFrame, extension: pd.DataFrame) -> pd.DataFrame:
    base = standardize_reference(base)
    extension = standardize_reference(extension)

    # Prefer the sealed base result on exact overlap; extension fills new domain.
    base["_priority"] = 0
    extension["_priority"] = 1
    merged = pd.concat([base, extension], ignore_index=True, sort=False)
    merged["_M_key"] = merged["Mach"].round(10)
    merged["_a_key"] = merged["alpha"].round(12)
    merged = (
        merged.sort_values(["_priority", "Mach", "alpha"])
        .drop_duplicates(["_M_key", "_a_key"], keep="first")
        .sort_values(["Mach", "alpha"])
        .reset_index(drop=True)
    )
    merged["mode_available"] = True
    merged["final_reference_id"] = [
        f"M{m:.6f}_a{a:.8f}" for m, a in merged[["Mach", "alpha"]].itertuples(index=False)
    ]
    return merged.drop(columns=["_priority", "_M_key", "_a_key"])


def save_ci_map(
    reference: pd.DataFrame,
    blumen: pd.DataFrame,
    output: Path,
) -> None:
    finite_b = pd.to_numeric(blumen.get("ci_level"), errors="coerce")
    neutral = finite_b.notna() & np.isclose(finite_b, 0.0, atol=1e-14)
    positive = finite_b.notna() & (finite_b > 0.0)

    vmax = max(
        float(reference["ci"].max()),
        float(finite_b[positive].max()) if positive.any() else 0.0,
    )
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9.1, 6.3), constrained_layout=True)
    dense_plot = ax.scatter(
        reference["Mach"],
        reference["alpha"],
        c=reference["ci"],
        norm=norm,
        cmap="viridis",
        marker="o",
        s=18,
        linewidths=0,
        alpha=0.84,
        label=f"Classical retained points ({len(reference)})",
        zorder=2,
    )

    if positive.any():
        ax.scatter(
            blumen.loc[positive, "Mach"],
            blumen.loc[positive, "alpha"],
            c=finite_b.loc[positive],
            norm=norm,
            cmap="viridis",
            marker="x",
            s=34,
            linewidths=1.0,
            label=rf"Blumen digitized $c_i>0$ ({int(positive.sum())})",
            zorder=5,
        )

    if neutral.any():
        ax.scatter(
            blumen.loc[neutral, "Mach"],
            blumen.loc[neutral, "alpha"],
            facecolors="none",
            edgecolors="black",
            marker="s",
            s=42,
            linewidths=1.1,
            label=rf"Blumen neutral $c_i=0$ ({int(neutral.sum())})",
            zorder=6,
        )

    if "neutral_alpha_estimate" in reference.columns:
        neutral_columns = ["Mach", "neutral_alpha_estimate"]
        for column in ("neutral_alpha_lower", "neutral_alpha_upper"):
            if column in reference.columns:
                neutral_columns.append(column)

        neutral_classic = reference[neutral_columns].copy()
        for column in neutral_columns[1:]:
            neutral_classic[column] = pd.to_numeric(
                neutral_classic[column], errors="coerce"
            )
        neutral_classic = (
            neutral_classic.dropna(subset=["Mach", "neutral_alpha_estimate"])
            .groupby("Mach", as_index=False)
            .median(numeric_only=True)
            .sort_values("Mach")
        )
        if not neutral_classic.empty:
            ax.plot(
                neutral_classic["Mach"],
                neutral_classic["neutral_alpha_estimate"],
                marker=".",
                linewidth=1.1,
                label=r"Classical neutral boundary $c_i=0$",
                zorder=4,
            )
            if {
                "neutral_alpha_lower",
                "neutral_alpha_upper",
            }.issubset(neutral_classic.columns):
                bracketed = neutral_classic.dropna(
                    subset=["neutral_alpha_lower", "neutral_alpha_upper"]
                )
                if not bracketed.empty:
                    midpoint = bracketed["neutral_alpha_estimate"].to_numpy(float)
                    lower = bracketed["neutral_alpha_lower"].to_numpy(float)
                    upper = bracketed["neutral_alpha_upper"].to_numpy(float)
                    yerr = np.vstack((midpoint - lower, upper - midpoint))
                    ax.errorbar(
                        bracketed["Mach"],
                        midpoint,
                        yerr=yerr,
                        fmt="D",
                        markersize=4,
                        capsize=3,
                        linewidth=0.9,
                        label="Low-M neutral brackets",
                        zorder=7,
                    )

    fig.colorbar(dense_plot, ax=ax, pad=0.02, label=r"$c_i$")
    ax.set_xlabel(r"Mach number $M$")
    ax.set_ylabel(r"Wavenumber $\alpha$")
    ax.set_title(
        r"Final classical supersonic reference and Blumen digitization"
    )
    ax.set_xlim(
        min(0.98, float(reference["Mach"].min()) - 0.02),
        max(2.02, float(reference["Mach"].max()) + 0.03),
    )
    ax.set_ylim(
        min(0.0, float(reference["alpha"].min()) - 0.005),
        max(0.37, float(reference["alpha"].max()) + 0.01),
    )
    ax.grid(True, alpha=0.20)
    ax.legend(frameon=False, fontsize=8, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def load_npz_paths(root: Path) -> list[Path]:
    return sorted(root.glob("M*/modes/modes_compact_with_analytic_tails.npz"))


def mach_from_dir(path: Path) -> float:
    name = path.parents[1].name
    if not name.startswith("M"):
        raise ValueError(f"Unexpected Mach directory: {name}")
    return float(name[1:].replace("p", "."))


def reconstruct_fields(
    Mach: float,
    alpha: float,
    cr: float,
    ci: float,
    y: np.ndarray,
    kappa: np.ndarray,
    q: np.ndarray,
    p_real: np.ndarray,
    p_imag: np.ndarray,
) -> dict[str, np.ndarray]:
    p = np.asarray(p_real, float) + 1j * np.asarray(p_imag, float)
    gamma = np.asarray(kappa, float) + 1j * np.asarray(q, float)
    py = gamma * p
    U = np.tanh(y)
    Up = 1.0 - U * U
    c = complex(cr, ci)
    difference = U - c
    rho = (Mach ** 2) * p
    v = 1j * py / (alpha * difference)
    u = -p / difference + 1j * Up * v / (alpha * difference)
    result = {
        "p": p,
        "rho": rho,
        "u": u,
        "v": v,
        "kappa": np.asarray(kappa, float),
        "q": np.asarray(q, float),
    }
    for name in ("p", "rho", "u", "v"):
        values = result[name]
        if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
            raise FloatingPointError(
                f"Non-finite reconstructed field {name} at M={Mach}, alpha={alpha}"
            )
    return result


def downsample_indices(size: int, maximum: int) -> np.ndarray:
    if size <= maximum:
        return np.arange(size)
    return np.unique(np.linspace(0, size - 1, maximum, dtype=int))


def active_limit(
    y: np.ndarray,
    fields: dict[str, np.ndarray],
    threshold: float,
) -> float:
    envelope = np.zeros(y.size, dtype=float)
    for name in ("p", "rho", "u", "v"):
        amplitude = np.abs(fields[name])
        peak = float(np.nanmax(amplitude))
        if np.isfinite(peak) and peak > 0.0:
            envelope = np.maximum(envelope, amplitude / peak)
    selected = np.isfinite(envelope) & (envelope >= threshold)
    if not selected.any():
        return min(float(np.nanmax(np.abs(y))), 40.0)
    return min(
        float(np.nanmax(np.abs(y))),
        max(10.0, 1.08 * float(np.nanmax(np.abs(y[selected])))),
    )


def plot_field(
    ax: plt.Axes,
    y: np.ndarray,
    values: np.ndarray,
    limit: float,
    title: str,
    max_points: int,
) -> None:
    peak = float(np.nanmax(np.abs(values)))
    normalized = values / peak
    indices = np.flatnonzero(np.abs(y) <= limit)
    indices = indices[downsample_indices(indices.size, max_points)]
    ax.plot(y[indices], normalized.real[indices], linewidth=0.85, label="real")
    ax.plot(
        y[indices],
        normalized.imag[indices],
        linewidth=0.85,
        linestyle="--",
        label="imaginary",
    )
    ax.axhline(0.0, linewidth=0.5)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-1.06, 1.06)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel("independently normalized mode")
    ax.set_title(title)
    ax.grid(True, alpha=0.20)


LONG_COLUMNS = [
    "final_reference_id",
    "Mach",
    "alpha",
    "cr",
    "ci",
    "omega_i",
    "reference_subset",
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


def write_long_rows(
    writer: csv.writer,
    ref_row: pd.Series,
    y: np.ndarray,
    fields: dict[str, np.ndarray],
) -> int:
    n = int(y.size)
    for i in range(n):
        writer.writerow([
            ref_row["final_reference_id"],
            float(ref_row["Mach"]),
            float(ref_row["alpha"]),
            float(ref_row["cr"]),
            float(ref_row["ci"]),
            float(ref_row["omega_i"]),
            str(ref_row["reference_subset"]),
            i,
            float(y[i]),
            float(fields["kappa"][i]),
            float(fields["q"][i]),
            float(fields["p"][i].real),
            float(fields["p"][i].imag),
            float(fields["rho"][i].real),
            float(fields["rho"][i].imag),
            float(fields["u"][i].real),
            float(fields["u"][i].imag),
            float(fields["v"][i].real),
            float(fields["v"][i].imag),
        ])
    return n


def build_modes_and_long_csv(
    reference: pd.DataFrame,
    roots: list[Path],
    pdf_output: Path,
    long_output: Path,
    summary_output: Path,
    active_threshold: float,
    max_plot_points: int,
) -> tuple[int, int, pd.DataFrame]:
    expected = {
        point_key(m, a): row
        for (_, row), (m, a) in zip(
            reference.iterrows(),
            reference[["Mach", "alpha"]].itertuples(index=False),
        )
    }
    observed: set[tuple[float, float]] = set()
    summaries: list[dict[str, Any]] = []
    page_count = 0
    coordinate_rows = 0

    pdf_tmp = pdf_output.with_name(f".{pdf_output.name}.tmp")
    csv_tmp = long_output.with_name(f".{long_output.name}.tmp")

    with gzip.open(csv_tmp, "wt", newline="", encoding="utf-8") as csv_handle, \
         PdfPages(pdf_tmp, metadata={
             "Title": "Extended classical supersonic modes: p, rho, u and v",
             "Author": "Emma Grospellier",
         }) as pdf:
        writer = csv.writer(csv_handle)
        writer.writerow(LONG_COLUMNS)

        cover = plt.figure(figsize=(11.69, 8.27))
        cover.text(
            0.5, 0.84,
            "Final classical supersonic modal atlas",
            ha="center", fontsize=20, weight="bold",
        )
        cover.text(
            0.5, 0.76,
            f"{len(reference)} unique retained points - one point per page",
            ha="center", fontsize=12,
        )
        cover.text(
            0.10, 0.57,
            "Fields reconstructed from the pressure mode:\n"
            r"$\hat\rho=M^2\hat p$" "\n"
            r"$\hat v=i\hat p_y/[\alpha(U-c)]$" "\n"
            r"$\hat u=-\hat p/(U-c)+iU'\hat v/[\alpha(U-c)]$" "\n\n"
            "Each field is normalized independently in the PDF.\n"
            "The compressed long-format CSV stores the unnormalized real and "
            "imaginary values at every y coordinate.",
            va="top", fontsize=11,
        )
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)
        page_count += 1

        global_index = 0
        # Base first, then extension; exact duplicate keys are skipped.
        for path in roots:
            with np.load(path, allow_pickle=False) as payload:
                required = {
                    "Mach", "alpha", "cr", "ci", "omega_i",
                    "y", "kappa", "q", "p_real", "p_imag",
                }
                missing = required.difference(payload.files)
                if missing:
                    raise KeyError(f"{path}: missing arrays {sorted(missing)}")
                for raw_index in np.argsort(payload["alpha"]):
                    idx = int(raw_index)
                    Mach = float(payload["Mach"][idx])
                    alpha = float(payload["alpha"][idx])
                    key = point_key(Mach, alpha)
                    if key not in expected or key in observed:
                        continue
                    ref_row = expected[key]
                    observed.add(key)

                    cr = float(ref_row["cr"])
                    ci = float(ref_row["ci"])
                    omega_i = float(ref_row["omega_i"])
                    y = np.asarray(payload["y"][idx], dtype=float)
                    fields = reconstruct_fields(
                        Mach=Mach,
                        alpha=alpha,
                        cr=cr,
                        ci=ci,
                        y=y,
                        kappa=np.asarray(payload["kappa"][idx], dtype=float),
                        q=np.asarray(payload["q"][idx], dtype=float),
                        p_real=np.asarray(payload["p_real"][idx], dtype=float),
                        p_imag=np.asarray(payload["p_imag"][idx], dtype=float),
                    )

                    coordinate_rows += write_long_rows(
                        writer, ref_row, y, fields
                    )
                    limit = active_limit(y, fields, active_threshold)
                    global_index += 1

                    fig, axes = plt.subplots(
                        2, 2, figsize=(11.69, 8.27), constrained_layout=True
                    )
                    labels = (
                        ("p", r"Pressure $\hat p$"),
                        ("rho", r"Density $\hat\rho$"),
                        ("u", r"Streamwise velocity $\hat u$"),
                        ("v", r"Transverse velocity $\hat v$"),
                    )
                    for ax, (name, title) in zip(axes.flat, labels):
                        plot_field(
                            ax, y, fields[name], limit, title, max_plot_points
                        )
                    axes.flat[0].legend(
                        frameon=False, ncols=2, fontsize=8, loc="best"
                    )
                    fig.suptitle(
                        rf"$M={Mach:.3f}$, $\alpha={alpha:.6f}$, "
                        rf"$c={cr:.8f}+{ci:.8e}i$, "
                        rf"$\omega_i={omega_i:.8e}$"
                        f"\npoint {global_index}/{len(reference)}; "
                        f"displayed |y| <= {limit:.4g}",
                        fontsize=10.5,
                    )
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    page_count += 1

                    summaries.append({
                        "final_reference_id": ref_row["final_reference_id"],
                        "Mach": Mach,
                        "alpha": alpha,
                        "cr": cr,
                        "ci": ci,
                        "omega_i": omega_i,
                        "reference_subset": ref_row["reference_subset"],
                        "mode_npz_path": str(path),
                        "mode_index_in_npz": idx,
                        "n_coordinates": int(y.size),
                        "y_min": float(y.min()),
                        "y_max": float(y.max()),
                        "display_limit": limit,
                        "min_abs_U_minus_c": float(
                            np.min(np.abs(np.tanh(y) - complex(cr, ci)))
                        ),
                    })

    missing_keys = sorted(set(expected).difference(observed))
    if missing_keys:
        raise RuntimeError(
            f"Missing {len(missing_keys)} modal keys; first={missing_keys[:20]}"
        )
    if page_count != len(reference) + 1:
        raise RuntimeError(
            f"Expected {len(reference)+1} PDF pages, generated {page_count}"
        )

    os.replace(pdf_tmp, pdf_output)
    os.replace(csv_tmp, long_output)
    summary = pd.DataFrame(summaries).sort_values(
        ["Mach", "alpha"]
    ).reset_index(drop=True)
    summary.to_csv(summary_output, index=False)
    return page_count, coordinate_rows, summary


def write_checksums(output_dir: Path) -> None:
    paths = sorted(
        p for p in output_dir.rglob("*")
        if p.is_file() and p.name not in {"SHA256SUMS.txt", "manifest.csv"}
    )
    (output_dir / "SHA256SUMS.txt").write_text(
        "\n".join(
            f"{sha256(p)}  {p.relative_to(output_dir).as_posix()}"
            for p in paths
        ) + "\n",
        encoding="utf-8",
    )
    manifest = pd.DataFrame([
        {
            "relative_path": p.relative_to(output_dir).as_posix(),
            "size_bytes": p.stat().st_size,
            "sha256": sha256(p),
        }
        for p in sorted(q for q in output_dir.rglob("*") if q.is_file())
    ])
    manifest.to_csv(output_dir / "manifest.csv", index=False)


def make_bundle(output_dir: Path) -> tuple[Path, Path]:
    bundle = output_dir.with_suffix(".tar.gz")
    temporary = bundle.with_name(f".{bundle.name}.tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(output_dir, arcname=output_dir.name)
    os.replace(temporary, bundle)
    checksum = bundle.with_suffix(bundle.suffix + ".sha256")
    checksum.write_text(
        f"{sha256(bundle)}  {bundle.name}\n", encoding="utf-8"
    )
    return bundle, checksum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the sealed dense campaign with the validated low-M/"
            "low-alpha extension and regenerate the ci map, modal atlas, "
            "point-reference CSV and long-format modal CSV."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base-freeze", type=Path, default=BASE_FREEZE_DEFAULT)
    parser.add_argument("--extension-dir", type=Path, default=EXTENSION_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--blumen-points", type=Path, default=None)
    parser.add_argument("--blumen-levels", type=Path, default=None)
    parser.add_argument("--active-threshold", type=float, default=1e-4)
    parser.add_argument("--max-plot-points", type=int, default=1800)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    base_freeze = resolve(repo, args.base_freeze)
    extension_dir = resolve(repo, args.extension_dir)
    output_dir = resolve(repo, args.output_dir)

    if not base_freeze.is_dir():
        raise FileNotFoundError(base_freeze)
    if not extension_dir.is_dir():
        raise FileNotFoundError(extension_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output exists: {output_dir}; pass --overwrite to regenerate"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    base = load_base_reference(base_freeze)
    extension = load_extension_reference(extension_dir)
    reference = merge_references(base, extension)

    points_path = find_blumen_points(repo, args.blumen_points)
    levels_path = find_blumen_levels(repo, args.blumen_levels)
    blumen, blumen_metadata = load_blumen(points_path, levels_path)

    point_csv = output_dir / "classical_supersonic_final_reference.csv"
    ci_pdf = output_dir / "classical_supersonic_final_ci_map.pdf"
    modes_pdf = (
        output_dir
        / "classical_supersonic_final_all_modes_p_rho_u_v.pdf"
    )
    modes_long = (
        output_dir
        / "classical_supersonic_final_modes_long.csv.gz"
    )
    modes_index = output_dir / "classical_supersonic_final_modes_index.csv"
    blumen_csv = output_dir / "blumen_overlay_points_with_neutral.csv"

    reference.to_csv(point_csv, index=False)
    blumen_export = blumen.copy()
    blumen_export.to_csv(blumen_csv, index=False)
    save_ci_map(reference, blumen, ci_pdf)

    base_root = base_freeze / "frozen_results/per_mach"
    extension_roots = load_npz_paths(extension_dir)
    base_roots = load_npz_paths(base_root)
    if not base_roots:
        raise FileNotFoundError(f"No base mode NPZ files under {base_root}")
    if not extension_roots:
        raise FileNotFoundError(
            f"No extension mode NPZ files under {extension_dir}"
        )

    pages, coordinate_rows, summary = build_modes_and_long_csv(
        reference=reference,
        roots=base_roots + extension_roots,
        pdf_output=modes_pdf,
        long_output=modes_long,
        summary_output=modes_index,
        active_threshold=args.active_threshold,
        max_plot_points=args.max_plot_points,
    )

    metadata = {
        "generated_at": utc_now(),
        "base_reference_points": int(len(standardize_reference(base))),
        "extension_reference_points": int(len(standardize_reference(extension))),
        "unique_extended_reference_points": int(len(reference)),
        "mach_values": sorted(float(v) for v in reference["Mach"].unique()),
        "alpha_min": float(reference["alpha"].min()),
        "alpha_max": float(reference["alpha"].max()),
        "modal_pdf_pages": int(pages),
        "modal_coordinate_rows": int(coordinate_rows),
        "blumen": blumen_metadata,
        "outputs": {
            "ci_map": ci_pdf.name,
            "point_reference_csv": point_csv.name,
            "modal_pdf": modes_pdf.name,
            "modal_long_csv_gz": modes_long.name,
            "modal_index_csv": modes_index.name,
            "blumen_points_csv": blumen_csv.name,
        },
        "status": "PASS",
    }
    (output_dir / "asset_build_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__).resolve(), output_dir / Path(__file__).name)
    (output_dir / "README.md").write_text(
        f"""# Final full-branch classical supersonic assets

This package merges the sealed dense classical campaign with the validated
complete M=1.00 and M=1.05 unstable branches.

- Unique spectral points: {len(reference)}
- Modal PDF pages: {pages}
- Long-format modal rows: {coordinate_rows}

Files:

- `classical_supersonic_final_ci_map.pdf`: Mach-alpha map colored by ci,
  including positive and neutral digitized Blumen points.
- `classical_supersonic_final_reference.csv`: one row per unique point,
  with Mach, alpha, cr, ci, omega_i and provenance.
- `classical_supersonic_final_all_modes_p_rho_u_v.pdf`: one page per
  point, with real and imaginary parts of p, rho, u and v.
- `classical_supersonic_final_modes_long.csv.gz`: one row per
  (point, y-coordinate), containing unnormalized p, rho, u and v real and
  imaginary parts, plus kappa and q.
- `classical_supersonic_final_modes_index.csv`: compact lookup table
  linking each point to its NPZ source and mode index.
""",
        encoding="utf-8",
    )

    write_checksums(output_dir)
    bundle, checksum = make_bundle(output_dir)

    print("=== FINAL FULL-BRANCH SUPERSONIC ASSETS ===")
    print(f"Unique reference points : {len(reference)}")
    print(f"Mach values             : {reference['Mach'].nunique()}")
    print(f"Modal PDF pages         : {pages}")
    print(f"Modal coordinate rows   : {coordinate_rows}")
    print(f"Output directory        : {output_dir}")
    print(f"Transfer bundle         : {bundle}")
    print(f"Bundle checksum         : {checksum}")
    print("ASSET STATUS                 : PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
