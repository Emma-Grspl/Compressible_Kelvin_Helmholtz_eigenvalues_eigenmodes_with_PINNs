#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

FREEZE_DEFAULT = Path(
    "assets/classic_supersonic/dense_kappa_q_campaign_v1_FINAL_FREEZE"
)
OUTPUT_DEFAULT = Path(
    "assets/classic_supersonic/dense_kappa_q_campaign_v1_REQUESTED_ASSETS"
)
EXPECTED_POINTS = 770
EXPECTED_MACHS = 17


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(repo: Path, value: Path) -> Path:
    value = value.expanduser()
    return value if value.is_absolute() else repo / value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point_key(Mach: float, alpha: float) -> tuple[float, float]:
    return round(float(Mach), 10), round(float(alpha), 12)


def find_blumen_points(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = resolve(repo, explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = [
        repo / "article/tables/table_blumen_ci_digitized_points.csv",
        repo / "assets/classic_supersonic/supersonic_sparse_PINN_reference_v2_FINAL_FREEZE/ci/blumen_ci_digitized_points.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(repo.glob("**/blumen_ci_digitized_points.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Could not locate blumen_ci_digitized_points.csv in the repository."
    )


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


def locate(columns: list[str], variants: tuple[str, ...]) -> str | None:
    mapping = {str(column).strip().lower(): column for column in columns}
    for variant in variants:
        found = mapping.get(variant.lower())
        if found is not None:
            return found
    return None


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def parse_ci_from_label(value: object) -> float:
    import re
    text = str(value)
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not numbers:
        return math.nan
    try:
        return float(numbers[-1])
    except ValueError:
        return math.nan


def load_blumen(points_path: Path, levels_path: Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    points = pd.read_csv(points_path)
    mach_col = locate(points.columns.tolist(), ("Mach", "M", "mach_physical"))
    alpha_col = locate(points.columns.tolist(), ("alpha", "wavenumber"))
    if mach_col is None or alpha_col is None:
        raise KeyError(
            f"{points_path}: expected Mach and alpha columns; observed {points.columns.tolist()}"
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
        required = {"curve_id", "ci_level"}
        if required.issubset(levels.columns):
            if "include_in_overlay" in levels.columns:
                levels = levels.loc[truthy(levels["include_in_overlay"])].copy()
            elif "include_in_quantitative_comparison" in levels.columns:
                levels = levels.loc[truthy(levels["include_in_quantitative_comparison"])].copy()
            if "family" in levels.columns:
                levels = levels.loc[
                    ~levels["family"].astype(str).str.strip().eq("cr_special")
                ].copy()
            levels["curve_id"] = pd.to_numeric(levels["curve_id"], errors="coerce").astype("Int64")
            levels["ci_level"] = pd.to_numeric(levels["ci_level"], errors="coerce")
            points["curve_id"] = pd.to_numeric(points["curve_id"], errors="coerce").astype("Int64")
            keep = ["curve_id", "ci_level"]
            for extra in ("curve_label", "family"):
                if extra in levels.columns:
                    keep.append(extra)
            points = points.merge(levels[keep], on="curve_id", how="inner", validate="many_to_one")
            level_source = str(levels_path)
    else:
        label_col = locate(points.columns.tolist(), ("curve_label", "label", "series"))
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
        points["Mach"].between(1.0, 2.1) & points["alpha"].between(0.0, 1.0)
    ].reset_index(drop=True)
    if points.empty:
        raise RuntimeError("No Blumen points remain after coordinate validation.")

    metadata = {
        "points_path": str(points_path),
        "levels_path": str(levels_path) if levels_path is not None else None,
        "point_count": int(len(points)),
        "curve_count": int(points["curve_id"].nunique()) if "curve_id" in points.columns else None,
        "mach_calibration": calibration,
        "ci_level_source": level_source,
        "ci_levels_available": bool("ci_level" in points.columns and points["ci_level"].notna().any()),
    }
    return points, metadata


def load_reference(freeze_dir: Path, expected_points: int, expected_machs: int) -> pd.DataFrame:
    path = freeze_dir / "classical_supersonic_maps/classical_supersonic_dense_reference.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"Mach", "alpha", "cr", "ci", "omega_i", "mode_available"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{path}: missing columns {sorted(missing)}")
    for column in ("Mach", "alpha", "cr", "ci", "omega_i"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Mach", "alpha", "cr", "ci", "omega_i"])
    frame = frame.sort_values(["Mach", "alpha"]).drop_duplicates(["Mach", "alpha"]).reset_index(drop=True)
    available = frame["mode_available"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    if not available.all():
        raise RuntimeError(f"Reference has {int((~available).sum())} rows without modes.")
    if len(frame) != expected_points:
        raise RuntimeError(f"Expected {expected_points} retained points; observed {len(frame)}.")
    if frame["Mach"].nunique() != expected_machs:
        raise RuntimeError(f"Expected {expected_machs} Mach values; observed {frame['Mach'].nunique()}.")
    return frame


def save_overlay(
    *, reference: pd.DataFrame, blumen: pd.DataFrame, metadata: dict[str, Any], output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    has_levels = bool(metadata["ci_levels_available"])
    if has_levels:
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), constrained_layout=True)
    else:
        fig, single = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
        axes = np.asarray([single])

    coverage = axes[0]
    coverage.scatter(
        reference["alpha"], reference["Mach"], marker="o", s=12, linewidths=0,
        alpha=0.55, label=f"Dense classical reference ({len(reference)})", zorder=2,
    )
    coverage.scatter(
        blumen["alpha"], blumen["Mach"], marker="x", s=18, linewidths=0.75,
        label=f"Blumen digitized points ({len(blumen)})", zorder=4,
    )
    coverage.set_title("Point coverage")
    coverage.set_xlabel(r"Wavenumber $\alpha$")
    coverage.set_ylabel(r"Mach number $M$")
    coverage.grid(True, alpha=0.20)
    coverage.legend(frameon=False, fontsize=8, loc="best")

    if has_levels:
        colored = axes[1]
        finite_levels = pd.to_numeric(blumen["ci_level"], errors="coerce")
        vmin = min(float(reference["ci"].min()), float(finite_levels.min()))
        vmax = max(float(reference["ci"].max()), float(finite_levels.max()))
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("viridis")
        dense_plot = colored.scatter(
            reference["alpha"], reference["Mach"], c=reference["ci"],
            norm=norm, cmap=cmap, marker="o", s=14, linewidths=0, alpha=0.70,
            label="Dense reference", zorder=2,
        )
        colored.scatter(
            blumen.loc[finite_levels.notna(), "alpha"],
            blumen.loc[finite_levels.notna(), "Mach"],
            c=finite_levels.loc[finite_levels.notna()], norm=norm, cmap=cmap,
            marker="x", s=22, linewidths=0.85, label="Blumen digitized $c_i$", zorder=4,
        )
        fig.colorbar(dense_plot, ax=colored, pad=0.02, label=r"$c_i$")
        colored.set_title(r"Shared $c_i$ scale")
        colored.set_xlabel(r"Wavenumber $\alpha$")
        colored.set_ylabel(r"Mach number $M$")
        colored.grid(True, alpha=0.20)
        colored.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle("Blumen digitized points vs dense classical supersonic reference", fontsize=13)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def downsample_indices(size: int, maximum: int) -> np.ndarray:
    if size <= maximum:
        return np.arange(size)
    return np.unique(np.linspace(0, size - 1, maximum, dtype=int))


def reconstruct_fields(
    *, Mach: float, alpha: float, cr: float, ci: float,
    y: np.ndarray, kappa: np.ndarray, q: np.ndarray,
    p_real: np.ndarray, p_imag: np.ndarray,
) -> dict[str, np.ndarray]:
    p = np.asarray(p_real, float) + 1j * np.asarray(p_imag, float)
    gamma = np.asarray(kappa, float) + 1j * np.asarray(q, float)
    py = gamma * p
    U = np.tanh(y)
    Up = 1.0 - U * U
    c = complex(cr, ci)
    difference = U - c
    min_abs_difference = float(np.nanmin(np.abs(difference)))
    if not np.isfinite(min_abs_difference) or min_abs_difference <= 0.0:
        raise FloatingPointError("Invalid U-c denominator in modal reconstruction.")
    rho = (Mach ** 2) * p
    v = 1j * py / (alpha * difference)
    u = -p / difference + 1j * Up * v / (alpha * difference)
    fields = {"p": p, "rho": rho, "u": u, "v": v}
    for name, values in fields.items():
        if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
            raise FloatingPointError(f"Non-finite reconstructed field: {name}")
    fields["min_abs_U_minus_c"] = np.asarray(min_abs_difference)
    return fields


def active_limit(y: np.ndarray, fields: dict[str, np.ndarray], threshold: float) -> float:
    full = float(np.nanmax(np.abs(y)))
    envelope = np.zeros(y.size, dtype=float)
    for name in ("p", "rho", "u", "v"):
        amp = np.abs(fields[name])
        peak = float(np.nanmax(amp))
        if np.isfinite(peak) and peak > 0.0:
            envelope = np.maximum(envelope, amp / peak)
    selected = np.isfinite(envelope) & (envelope >= threshold)
    if not selected.any():
        return min(full, 40.0)
    detected = 1.08 * float(np.nanmax(np.abs(y[selected])))
    return min(full, max(10.0, detected))


def plot_field(ax: plt.Axes, y: np.ndarray, values: np.ndarray, limit: float, title: str, max_points: int) -> None:
    peak = float(np.nanmax(np.abs(values)))
    if not np.isfinite(peak) or peak <= 0.0:
        raise RuntimeError(f"Invalid normalization for {title}.")
    normalized = values / peak
    mask = np.abs(y) <= limit
    indices = np.flatnonzero(mask)
    indices = indices[downsample_indices(indices.size, max_points)]
    ax.plot(y[indices], normalized.real[indices], linewidth=0.85, label="real")
    ax.plot(y[indices], normalized.imag[indices], linewidth=0.85, linestyle="--", label="imaginary")
    ax.axhline(0.0, linewidth=0.5)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-1.06, 1.06)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel("independently normalized mode")
    ax.set_title(title)
    ax.grid(True, alpha=0.20)


def load_npz_by_mach(per_mach_root: Path) -> dict[float, Path]:
    result: dict[float, Path] = {}
    for mach_dir in sorted(per_mach_root.glob("M*")):
        path = mach_dir / "modes/modes_compact_with_analytic_tails.npz"
        if not path.is_file():
            continue
        Mach = float(mach_dir.name[1:].replace("p", "."))
        result[round(Mach, 10)] = path
    return result


def build_modes_pdf(
    *, reference: pd.DataFrame, per_mach_root: Path, output: Path,
    summary_output: Path, active_threshold: float, max_plot_points: int,
) -> tuple[int, pd.DataFrame]:
    paths = load_npz_by_mach(per_mach_root)
    expected_keys = {point_key(m, a) for m, a in reference[["Mach", "alpha"]].itertuples(index=False)}
    observed_keys: set[tuple[float, float]] = set()
    summaries: list[dict[str, Any]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    page_count = 0

    with PdfPages(temporary, metadata={
        "Title": "Dense classical supersonic modes: p, rho, u and v",
        "Author": "Emma Grospellier",
        "Subject": "770 retained classical Kelvin-Helmholtz eigenmodes",
    }) as pdf:
        cover = plt.figure(figsize=(11.69, 8.27))
        cover.text(0.5, 0.84, "Dense classical supersonic modal atlas", ha="center", fontsize=20, weight="bold")
        cover.text(0.5, 0.76, f"{len(reference)} retained points - one point per page", ha="center", fontsize=12)
        cover.text(
            0.10, 0.57,
            "Fields reconstructed from the frozen pressure mode:\n"
            r"$\hat\rho=M^2\hat p$" "\n"
            r"$\hat v=i\hat p_y/[\alpha(U-c)]$" "\n"
            r"$\hat u=-\hat p/(U-c)+iU'\hat v/[\alpha(U-c)]$" "\n\n"
            "Each field is normalized independently. Solid line: real part. Dashed line: imaginary part.\n"
            "The displayed window is selected from the common active envelope of p, rho, u and v.",
            va="top", fontsize=11,
        )
        cover.text(0.5, 0.08, "Generated from the sealed dense_kappa_q_campaign_v1 classical reference.", ha="center", fontsize=8)
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)
        page_count += 1

        global_index = 0
        for Mach in sorted(paths):
            path = paths[Mach]
            with np.load(path, allow_pickle=False) as payload:
                required = {
                    "Mach", "alpha", "cr", "ci", "omega_i", "y", "kappa", "q", "p_real", "p_imag"
                }
                missing = required.difference(payload.files)
                if missing:
                    raise KeyError(f"{path}: missing arrays {sorted(missing)}")
                order = np.argsort(payload["alpha"])
                for raw_index in order:
                    idx = int(raw_index)
                    mode_Mach = float(payload["Mach"][idx])
                    alpha = float(payload["alpha"][idx])
                    cr = float(payload["cr"][idx])
                    ci = float(payload["ci"][idx])
                    omega_i = float(payload["omega_i"][idx])
                    key = point_key(mode_Mach, alpha)
                    if key not in expected_keys:
                        raise RuntimeError(f"Mode {key} is absent from the retained reference.")
                    if key in observed_keys:
                        raise RuntimeError(f"Duplicate modal key: {key}")
                    observed_keys.add(key)

                    y = np.asarray(payload["y"][idx], dtype=float)
                    fields = reconstruct_fields(
                        Mach=mode_Mach, alpha=alpha, cr=cr, ci=ci, y=y,
                        kappa=np.asarray(payload["kappa"][idx], dtype=float),
                        q=np.asarray(payload["q"][idx], dtype=float),
                        p_real=np.asarray(payload["p_real"][idx], dtype=float),
                        p_imag=np.asarray(payload["p_imag"][idx], dtype=float),
                    )
                    limit = active_limit(y, fields, active_threshold)
                    global_index += 1

                    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
                    labels = (
                        ("p", r"Pressure $\hat p$"),
                        ("rho", r"Density $\hat\rho$"),
                        ("u", r"Streamwise velocity $\hat u$"),
                        ("v", r"Transverse velocity $\hat v$"),
                    )
                    for ax, (name, title) in zip(axes.flat, labels):
                        plot_field(ax, y, fields[name], limit, title, max_plot_points)
                    axes.flat[0].legend(frameon=False, ncols=2, fontsize=8, loc="best")
                    fig.suptitle(
                        rf"$M={mode_Mach:.2f}$, $\alpha={alpha:.6f}$, "
                        rf"$c={cr:.8f}+{ci:.8e}i$, $\omega_i={omega_i:.8e}$"
                        f"\npoint {global_index}/{len(reference)}; displayed |y| <= {limit:.4g}",
                        fontsize=10.5,
                    )
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    page_count += 1

                    summaries.append({
                        "Mach": mode_Mach,
                        "alpha": alpha,
                        "cr": cr,
                        "ci": ci,
                        "omega_i": omega_i,
                        "n_coordinates": int(y.size),
                        "y_min": float(np.min(y)),
                        "y_max": float(np.max(y)),
                        "display_limit": limit,
                        "active_threshold": active_threshold,
                        "min_abs_U_minus_c": float(fields["min_abs_U_minus_c"]),
                        "finite_p": bool(np.isfinite(fields["p"]).all()),
                        "finite_rho": bool(np.isfinite(fields["rho"]).all()),
                        "finite_u": bool(np.isfinite(fields["u"]).all()),
                        "finite_v": bool(np.isfinite(fields["v"]).all()),
                    })

    if observed_keys != expected_keys:
        missing = sorted(expected_keys.difference(observed_keys))
        extra = sorted(observed_keys.difference(expected_keys))
        raise RuntimeError(f"Modal key mismatch: missing={missing[:20]}, extra={extra[:20]}")
    if page_count != len(reference) + 1:
        raise RuntimeError(f"Expected {len(reference) + 1} PDF pages; generated {page_count}.")
    os.replace(temporary, output)
    summary = pd.DataFrame(summaries).sort_values(["Mach", "alpha"]).reset_index(drop=True)
    atomic_csv(summary_output, summary)
    return page_count, summary


def write_checksums(output_dir: Path) -> None:
    checksum_path = output_dir / "SHA256SUMS.txt"
    paths = sorted(
        path for path in output_dir.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS.txt", "manifest.csv"}
    )
    lines = [f"{sha256(path)}  {path.relative_to(output_dir).as_posix()}" for path in paths]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = pd.DataFrame([
        {
            "relative_path": path.relative_to(output_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(path for path in output_dir.rglob("*") if path.is_file())
    ])
    atomic_csv(output_dir / "manifest.csv", manifest)


def make_bundle(output_dir: Path) -> tuple[Path, Path]:
    bundle = output_dir.with_suffix(".tar.gz")
    temporary = bundle.with_name(f".{bundle.name}.tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(output_dir, arcname=output_dir.name)
    os.replace(temporary, bundle)
    checksum = bundle.with_suffix(bundle.suffix + ".sha256")
    checksum.write_text(f"{sha256(bundle)}  {bundle.name}\n", encoding="utf-8")
    return bundle, checksum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the exact requested dense supersonic Blumen overlay and p/rho/u/v modal atlas."
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--freeze-dir", type=Path, default=FREEZE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--blumen-points", type=Path, default=None)
    parser.add_argument("--blumen-levels", type=Path, default=None)
    parser.add_argument("--active-threshold", type=float, default=1.0e-4)
    parser.add_argument("--expected-points", type=int, default=EXPECTED_POINTS)
    parser.add_argument("--expected-machs", type=int, default=EXPECTED_MACHS)
    parser.add_argument("--max-plot-points", type=int, default=1800)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0.0 < args.active_threshold < 1.0):
        raise ValueError("--active-threshold must be between 0 and 1.")
    if args.max_plot_points < 200:
        raise ValueError("--max-plot-points must be at least 200.")

    repo = args.repo.expanduser().resolve()
    freeze_dir = resolve(repo, args.freeze_dir)
    output_dir = resolve(repo, args.output_dir)
    if not freeze_dir.is_dir():
        raise FileNotFoundError(freeze_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}. Use --overwrite to regenerate.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    reference = load_reference(freeze_dir, args.expected_points, args.expected_machs)
    points_path = find_blumen_points(repo, args.blumen_points)
    levels_path = find_blumen_levels(repo, args.blumen_levels)
    blumen, blumen_metadata = load_blumen(points_path, levels_path)

    overlay_pdf = output_dir / "blumen_vs_dense_alpha_mach_map.pdf"
    overlay_csv = output_dir / "blumen_vs_dense_overlay_points.csv"
    modes_pdf = output_dir / "classical_supersonic_all_modes_p_rho_u_v.pdf"
    modal_summary_csv = output_dir / "modal_field_reconstruction_summary.csv"

    save_overlay(reference=reference, blumen=blumen, metadata=blumen_metadata, output=overlay_pdf)
    dense_export = reference[["Mach", "alpha", "ci"]].copy()
    dense_export["source"] = "dense_classical_reference"
    dense_export = dense_export.rename(columns={"ci": "ci_value"})
    blumen_export = blumen[["Mach", "alpha"]].copy()
    blumen_export["ci_value"] = pd.to_numeric(blumen.get("ci_level"), errors="coerce")
    blumen_export["source"] = "Blumen_digitized"
    atomic_csv(overlay_csv, pd.concat([dense_export, blumen_export], ignore_index=True))

    pages, modal_summary = build_modes_pdf(
        reference=reference,
        per_mach_root=freeze_dir / "frozen_results/per_mach",
        output=modes_pdf,
        summary_output=modal_summary_csv,
        active_threshold=args.active_threshold,
        max_plot_points=args.max_plot_points,
    )

    provenance = output_dir / "provenance"
    provenance.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve(), provenance / Path(__file__).name)
    shutil.copy2(points_path, provenance / points_path.name)
    if levels_path is not None:
        shutil.copy2(levels_path, provenance / levels_path.name)
    convergence_summary = repo / "assets/classic_supersonic/dense_kappa_q_campaign_v1_CONVERGENCE_AUDIT/convergence_tables/convergence_summary.json"
    if convergence_summary.is_file():
        shutil.copy2(convergence_summary, provenance / "convergence_summary.json")

    metadata = {
        "generated_at": utc_now(),
        "reference_freeze": str(freeze_dir.relative_to(repo)),
        "reference_points": int(len(reference)),
        "reference_mach_values": int(reference["Mach"].nunique()),
        "overlay_pdf": overlay_pdf.name,
        "overlay_points_csv": overlay_csv.name,
        "blumen": blumen_metadata,
        "modal_pdf": modes_pdf.name,
        "modal_pdf_pages": int(pages),
        "modal_points": int(len(modal_summary)),
        "modal_fields": ["p", "rho", "u", "v"],
        "modal_components": ["real", "imaginary"],
        "field_reconstruction": {
            "rho": "rho = M^2 p",
            "v": "v = i p_y / (alpha (U-c))",
            "u": "u = -p/(U-c) + i U_prime v/(alpha (U-c))",
            "p_y": "p_y = (kappa + i q) p",
            "U": "tanh(y)",
        },
        "active_threshold": args.active_threshold,
        "status": "PASS",
    }
    atomic_json(output_dir / "asset_build_metadata.json", metadata)

    readme = f"""# Dense classical supersonic requested assets\n\nGenerated: {metadata['generated_at']}\n\nThis package contains the two explicitly requested deliverables:\n\n1. `blumen_vs_dense_alpha_mach_map.pdf`: all {len(reference)} retained dense classical points overlaid with {len(blumen)} digitized Blumen points. Blumen points are never connected by plotting lines.\n2. `classical_supersonic_all_modes_p_rho_u_v.pdf`: one cover page plus one page for each of the {len(reference)} retained points, displaying the real and imaginary parts of p, rho, u and v.\n\nTraceability tables and the exact build script are included.\n"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    write_checksums(output_dir)
    bundle, checksum = make_bundle(output_dir)

    print("REQUESTED ASSET BUILD: PASS")
    print(f"Reference points: {len(reference)}")
    print(f"Blumen points: {len(blumen)}")
    print(f"Modal PDF pages: {pages}")
    print(f"Overlay: {overlay_pdf}")
    print(f"Modes: {modes_pdf}")
    print(f"Asset directory: {output_dir}")
    print(f"Transfer bundle: {bundle}")
    print(f"Checksum: {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
