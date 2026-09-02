#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader

FIELDS = [
    ("p", "Pressure p"),
    ("rho", "Density rho"),
    ("u", "Streamwise velocity u"),
    ("v", "Transverse velocity v"),
]


def key(mach: object, alpha: object) -> tuple[float, float]:
    return round(float(mach), 8), round(float(alpha), 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points-csv", type=Path, required=True)
    parser.add_argument("--modal-parquet", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--n-dense", type=int, default=16000)
    parser.add_argument("--active-threshold", type=float, default=1e-4)
    return parser.parse_args()


def normalize(real: np.ndarray, imag: np.ndarray):
    amp = np.hypot(real, imag)
    peak = float(np.nanmax(amp))
    if not np.isfinite(peak) or peak <= 0.0:
        raise RuntimeError("Invalid modal normalization")
    return real / peak, imag / peak, amp / peak


def active_limit(group: pd.DataFrame, threshold: float) -> float:
    y = group["y"].to_numpy(dtype=float)
    full = float(np.nanmax(np.abs(y)))
    envelope = np.zeros(len(group), dtype=float)

    for field, _ in FIELDS:
        real = group[f"{field}_real"].to_numpy(dtype=float)
        imag = group[f"{field}_imag"].to_numpy(dtype=float)
        amp = np.hypot(real, imag)
        peak = float(np.nanmax(amp))
        if np.isfinite(peak) and peak > 0.0:
            envelope = np.maximum(envelope, amp / peak)

    selected = np.isfinite(envelope) & (envelope >= threshold)
    if not selected.any():
        return full

    detected = float(np.nanmax(np.abs(y[selected])))
    return min(full, max(min(10.0, full), 1.08 * detected))


def plot_field(
    axis,
    y_dense: np.ndarray,
    real_dense: np.ndarray,
    imag_dense: np.ndarray,
    x_limit: float,
    title: str,
    legend: bool,
) -> None:
    real_n, imag_n, amp_n = normalize(real_dense, imag_dense)
    visible = np.abs(y_dense) <= x_limit

    axis.plot(y_dense[visible], real_n[visible], linewidth=0.75, label="real")
    axis.plot(y_dense[visible], imag_n[visible], linewidth=0.75, label="imaginary")
    axis.plot(
        y_dense[visible],
        amp_n[visible],
        linestyle="--",
        linewidth=0.8,
        label="modulus",
    )
    axis.axhline(0.0, linewidth=0.5)
    axis.set_xlim(-x_limit, x_limit)
    axis.set_ylim(-1.06, 1.06)
    axis.set_title(title)
    axis.set_xlabel("y")
    axis.set_ylabel("normalized mode")
    axis.grid(True, alpha=0.2)
    if legend:
        axis.legend(frameon=False, ncols=3, fontsize=7)


def main() -> None:
    args = parse_args()

    points_path = args.points_csv.expanduser()
    modal_path = args.modal_parquet.expanduser()
    output_path = args.output_pdf.expanduser()

    if not points_path.is_file():
        raise SystemExit(f"Points CSV not found: {points_path}")
    if not modal_path.is_file():
        raise SystemExit(f"Modal Parquet not found: {modal_path}")
    if args.n_dense < 2000:
        raise SystemExit("--n-dense must be at least 2000")
    if not 0.0 < args.active_threshold < 1.0:
        raise SystemExit("--active-threshold must be between 0 and 1")

    points = pd.read_csv(points_path)
    for column in ["Mach", "alpha"]:
        if column not in points.columns:
            raise KeyError(f"Missing {column!r} in {points_path}")
        points[column] = pd.to_numeric(points[column], errors="coerce")

    points = (
        points.dropna(subset=["Mach", "alpha"])
        .loc[lambda frame: frame["Mach"] > 1.5]
        .drop_duplicates(["Mach", "alpha"])
        .sort_values(["Mach", "alpha"])
        .reset_index(drop=True)
    )
    points["key"] = [key(m, a) for m, a in zip(points["Mach"], points["alpha"])]

    columns = [
        "Mach", "alpha", "y",
        "p_real", "p_imag",
        "rho_real", "rho_imag",
        "u_real", "u_imag",
        "v_real", "v_imag",
    ]
    modal = pd.read_parquet(modal_path, columns=columns)
    for column in columns:
        modal[column] = pd.to_numeric(modal[column], errors="coerce")
    modal = modal.dropna(subset=["Mach", "alpha", "y"])
    modal["key"] = [key(m, a) for m, a in zip(modal["Mach"], modal["alpha"])]

    expected = set(points["key"])
    modal = modal.loc[modal["key"].isin(expected)].copy()
    observed = set(modal["key"])
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"Missing modal points: {missing}")

    groups = {
        point_key: group.sort_values("y")
        for point_key, group in modal.groupby("key", sort=False)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_label = modal_path.stem

    with PdfPages(
        output_path,
        metadata={"Title": "Dense modal review for M > 1.5", "Subject": source_label},
    ) as pdf:
        cover = plt.figure(figsize=(11.69, 8.27))
        cover.text(0.5, 0.84, "Dense modal review for M > 1.5", ha="center", fontsize=18)
        cover.text(0.5, 0.76, f"Source: {modal_path}", ha="center", fontsize=10)
        cover.text(0.5, 0.70, f"Points: {len(points)}", ha="center", fontsize=11)
        cover.text(0.5, 0.64, f"Dense plotting grid: {args.n_dense}", ha="center", fontsize=11)
        cover.text(
            0.5,
            0.52,
            (
                "Top row: complete domain. Bottom row: active modal window.\n"
                "Linear interpolation is for rendering only; canonical data are unchanged."
            ),
            ha="center",
            fontsize=10,
        )
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)

        for index, point in points.iterrows():
            group = (
                groups[point["key"]]
                .drop_duplicates(subset=["y"], keep="first")
                .sort_values("y")
            )
            y = group["y"].to_numpy(dtype=float)
            y_dense = np.linspace(float(y.min()), float(y.max()), args.n_dense)
            full_limit = float(np.nanmax(np.abs(y)))
            zoom_limit = active_limit(group, args.active_threshold)

            figure, axes = plt.subplots(2, 4, figsize=(16.0, 8.5), constrained_layout=True)
            for column, (field, label) in enumerate(FIELDS):
                real = group[f"{field}_real"].to_numpy(dtype=float)
                imag = group[f"{field}_imag"].to_numpy(dtype=float)
                real_dense = np.interp(y_dense, y, real)
                imag_dense = np.interp(y_dense, y, imag)

                plot_field(
                    axes[0, column], y_dense, real_dense, imag_dense,
                    full_limit, f"{label}: complete domain", column == 0,
                )
                plot_field(
                    axes[1, column], y_dense, real_dense, imag_dense,
                    zoom_limit, f"{label}: active window", False,
                )

            figure.suptitle(
                f"M={point['Mach']:.2f}, alpha={point['alpha']:.6f}  "
                f"point {index + 1}/{len(points)}\nsource={source_label}",
                fontsize=11,
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    reader = PdfReader(str(output_path))
    expected_pages = len(points) + 1
    if len(reader.pages) != expected_pages:
        raise RuntimeError(
            f"Expected {expected_pages} pages; observed {len(reader.pages)}"
        )

    print("Source:", modal_path)
    print("Points:", len(points))
    print("PDF pages:", len(reader.pages))
    print("Wrote:", output_path)
    print()
    print("DENSE HIGH-MACH RENDER: PASS")


if __name__ == "__main__":
    main()
