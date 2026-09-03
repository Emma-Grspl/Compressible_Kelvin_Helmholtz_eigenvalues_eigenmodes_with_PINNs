#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VALID_STATUSES = {"accepted_root", "neutral_limit_root"}


def numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def load_spectral(manifest: pd.DataFrame, work_root: Path) -> pd.DataFrame:
    rows = []
    for _, target in manifest.sort_values("task_index").iterrows():
        index = int(target["task_index"])
        path = work_root / f"point_{index:03d}" / "spectral.csv"
        if path.is_file() and path.stat().st_size > 0:
            frame = pd.read_csv(path)
            if len(frame) == 1:
                rows.append(frame.iloc[0].to_dict())
                continue
        row = target.to_dict()
        row.update(
            {
                "status": "missing_result",
                "classical_cr": math.nan,
                "classical_ci": math.nan,
                "classical_omega_i": math.nan,
                "delta_ci": math.nan,
                "residual_norm": math.nan,
                "mode_status": "missing_result",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def curve_segments(frame: pd.DataFrame):
    ordered = frame.sort_values("source_row_id").copy()
    if len(ordered) <= 1:
        yield ordered
        return
    mach_jump = ordered["Mach"].diff().abs().fillna(0.0)
    alpha_jump = ordered["alpha"].diff().abs().fillna(0.0)
    split = (mach_jump > 0.15) | (alpha_jump > 0.10)
    for _, segment in ordered.groupby(split.cumsum(), sort=False):
        yield segment


def draw_blumen_lines(ax, spectral: pd.DataFrame) -> None:
    for _, curve in spectral.groupby("curve_key", sort=False):
        for segment in curve_segments(curve):
            ax.plot(
                segment["Mach"],
                segment["alpha"],
                color="0.35",
                linestyle="--",
                linewidth=0.9,
                alpha=0.55,
            )


def plot_scatter(
    spectral: pd.DataFrame,
    *,
    value: str,
    label: str,
    title: str,
    output_pdf: Path,
    output_png: Path,
    diverging: bool = False,
) -> None:
    good = spectral.loc[
        spectral["status"].astype(str).isin(VALID_STATUSES)
    ].copy()
    good = good.dropna(subset=["Mach", "alpha", value])

    fig, ax = plt.subplots(figsize=(8.8, 6.8), constrained_layout=True)
    draw_blumen_lines(ax, spectral)

    kwargs = {}
    if diverging and not good.empty:
        limit = float(np.nanmax(np.abs(good[value])))
        if not np.isfinite(limit) or limit == 0.0:
            limit = 1e-12
        kwargs.update(cmap="coolwarm", vmin=-limit, vmax=limit)

    scatter = ax.scatter(
        good["Mach"],
        good["alpha"],
        c=good[value],
        s=34,
        linewidths=0.25,
        edgecolors="black",
        **kwargs,
    )
    fig.colorbar(scatter, ax=ax, label=label)
    ax.set_xlabel("Mach")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(title)
    ax.grid(True, alpha=0.20)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    work_root = args.work_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(work_root / "blumen_pointwise_manifest.csv")
    spectral = load_spectral(manifest, work_root)
    spectral = numeric(
        spectral,
        (
            "task_index", "blumen_row_id", "source_row_id", "Mach", "alpha",
            "blumen_ci", "classical_cr", "classical_ci", "classical_omega_i",
            "delta_ci", "residual_norm",
        ),
    ).sort_values("blumen_row_id").reset_index(drop=True)

    spectral_csv = output_root / "blumen_pointwise_classical_spectral.csv"
    spectral.to_csv(spectral_csv, index=False)

    failures = spectral.loc[
        ~spectral["status"].astype(str).isin(VALID_STATUSES)
    ].copy()
    failures.to_csv(
        output_root / "blumen_pointwise_classical_failures.csv",
        index=False,
    )

    mode_paths = []
    for index in spectral.loc[
        spectral["mode_status"].astype(str).eq("reconstructed"), "task_index"
    ].dropna().astype(int):
        path = work_root / f"point_{index:03d}" / "mode.csv.gz"
        if path.is_file():
            mode_paths.append(path)

    mode_frames = [pd.read_csv(path) for path in mode_paths]
    modes = (
        pd.concat(mode_frames, ignore_index=True, sort=False)
        if mode_frames else pd.DataFrame()
    )
    modes_csv = output_root / "blumen_pointwise_classical_modes.csv.gz"
    modes.to_csv(modes_csv, index=False, compression="gzip")

    plot_scatter(
        spectral,
        value="classical_ci",
        label=r"Classical $c_i$",
        title="Classical $c_i$ at the digitized Blumen coordinates",
        output_pdf=output_root / "blumen_pointwise_classical_ci_map.pdf",
        output_png=output_root / "blumen_pointwise_classical_ci_map.png",
    )
    plot_scatter(
        spectral,
        value="classical_cr",
        label=r"Classical $c_r$",
        title="Classical $c_r$ at the digitized Blumen coordinates",
        output_pdf=output_root / "blumen_pointwise_classical_cr_map.pdf",
        output_png=output_root / "blumen_pointwise_classical_cr_map.png",
    )
    plot_scatter(
        spectral,
        value="delta_ci",
        label=r"$\Delta c_i=c_{i,\mathrm{classique}}-c_{i,\mathrm{Blumen}}$",
        title="Pointwise error at the digitized Blumen coordinates",
        output_pdf=output_root / "blumen_pointwise_delta_ci_heatmap.pdf",
        output_png=output_root / "blumen_pointwise_delta_ci_heatmap.png",
        diverging=True,
    )

    valid = spectral["status"].astype(str).isin(VALID_STATUSES)
    mode_ok = spectral["mode_status"].astype(str).eq("reconstructed")
    n_total = int(len(spectral))
    n_valid = int(valid.sum())
    n_modes = int(mode_ok.sum())
    status = "PASS" if n_valid == n_total and n_modes == n_total else "INCOMPLETE"

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_total": n_total,
        "n_spectral_valid": n_valid,
        "n_modes_reconstructed": n_modes,
        "n_failures": int(n_total - n_valid),
        "status": status,
        "spectral_csv": str(spectral_csv),
        "modes_csv": str(modes_csv),
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    archive = output_root.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output_root, arcname=output_root.name)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(
        f"{sha256(archive)}  {archive.name}\n",
        encoding="utf-8",
    )

    print("=== BLUMEN POINTWISE CLASSICAL ASSETS ===")
    print(f"Spectral roots : {n_valid}/{n_total}")
    print(f"Modes          : {n_modes}/{n_total}")
    print(f"Spectral CSV   : {spectral_csv}")
    print(f"Modal CSV      : {modes_csv}")
    print(f"ci map         : {output_root / 'blumen_pointwise_classical_ci_map.pdf'}")
    print(f"cr map         : {output_root / 'blumen_pointwise_classical_cr_map.pdf'}")
    print(f"Error heatmap  : {output_root / 'blumen_pointwise_delta_ci_heatmap.pdf'}")
    print(f"Archive        : {archive}")
    print(f"ASSET STATUS   : {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
