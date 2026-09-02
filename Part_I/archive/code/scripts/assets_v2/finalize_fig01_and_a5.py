#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
SCI = ROOT / "assets" / "pinn_subsonic" / "local_atlas_v1" / "publication_assets_scientific_v2"
RELEASE = ROOT / "assets" / "pinn_subsonic" / "release_v1"
MODEL_ROOT = ROOT / "pinn_subsonic" / "models"

FIG_DIR = SCI / "figures"
DATA_DIR = SCI / "data"
MANIFEST_DIR = SCI / "manifests"

ANCHOR_CSV = DATA_DIR / "spectral_supervision_anchors_explicit.csv"
ANCHOR_REPORT = MANIFEST_DIR / "spectral_supervision_anchors_report.json"
BUILD_REPORT = SCI / "build_report.json"


def first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    for name in names:
        match = lookup.get(name.lower())
        if match is not None:
            return match
    return None


def to_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict):
        return pd.DataFrame(value)
    if isinstance(value, (list, tuple)):
        return pd.DataFrame(value)
    if hasattr(value, "to_dict"):
        return pd.DataFrame(value.to_dict())
    raise TypeError(f"Unsupported anchor_df type: {type(value)!r}")


def extract_anchors() -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []

    checkpoints = sorted(MODEL_ROOT.glob("*/model_state.pt"))
    if not checkpoints:
        raise RuntimeError(f"No curated checkpoints found under {MODEL_ROOT}")

    for checkpoint_path in checkpoints:
        chart_id = checkpoint_path.parent.name
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        raw = checkpoint.get("anchor_df")

        if raw is None:
            inventory.append({
                "chart_id": chart_id,
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                "status": "missing_anchor_df",
                "n_rows": 0,
            })
            continue

        frame = to_frame(raw).reset_index(drop=False)
        frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")].copy()

        mach_col = first_column(frame, ("Mach", "M", "mach"))
        eta_col = first_column(frame, ("eta", "Eta"))
        alpha_col = first_column(frame, ("alpha", "Alpha", "a"))
        ci_col = first_column(
            frame,
            ("ci", "c_i", "ci_ref", "ci_target", "target_ci", "growth_rate"),
        )

        if mach_col is None:
            raise RuntimeError(f"{chart_id}: no Mach column in anchor_df")
        if eta_col is None and alpha_col is None:
            raise RuntimeError(f"{chart_id}: neither eta nor alpha is present")

        normalized = frame.copy()
        normalized["Mach"] = pd.to_numeric(normalized[mach_col], errors="coerce")

        if alpha_col is not None:
            normalized["alpha"] = pd.to_numeric(normalized[alpha_col], errors="coerce")

        if eta_col is not None:
            normalized["eta"] = pd.to_numeric(normalized[eta_col], errors="coerce")
        else:
            denominator = np.sqrt(np.clip(1.0 - normalized["Mach"] ** 2, 1.0e-15, None))
            normalized["eta"] = normalized["alpha"] / denominator

        if "alpha" not in normalized.columns:
            normalized["alpha"] = normalized["eta"] * np.sqrt(
                np.clip(1.0 - normalized["Mach"] ** 2, 1.0e-15, None)
            )

        if ci_col is not None:
            normalized["ci_anchor"] = pd.to_numeric(normalized[ci_col], errors="coerce")
        else:
            normalized["ci_anchor"] = np.nan

        normalized.insert(0, "chart_id", chart_id)
        normalized.insert(1, "source_checkpoint", str(checkpoint_path.relative_to(ROOT)))

        required = (
            np.isfinite(normalized["Mach"])
            & np.isfinite(normalized["eta"])
            & np.isfinite(normalized["alpha"])
        )
        normalized = normalized.loc[required].copy()

        canonical = [
            "chart_id",
            "source_checkpoint",
            "Mach",
            "eta",
            "alpha",
            "ci_anchor",
        ]
        remaining = [column for column in normalized.columns if column not in canonical]
        normalized = normalized[canonical + remaining]

        frames.append(normalized)
        inventory.append({
            "chart_id": chart_id,
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "status": "extracted",
            "n_rows": int(len(normalized)),
            "source_columns": [str(column) for column in frame.columns],
        })

    if not frames:
        raise RuntimeError("No explicit anchors were extracted")

    anchors = pd.concat(frames, ignore_index=True, sort=False)
    anchors = anchors.drop_duplicates(
        subset=["chart_id", "Mach", "eta", "alpha", "ci_anchor"],
        keep="first",
    )
    anchors = anchors.sort_values(
        ["chart_id", "Mach", "eta", "alpha"],
        kind="mergesort",
    ).reset_index(drop=True)

    report = {
        "source": "checkpoint.anchor_df",
        "n_checkpoints_scanned": len(checkpoints),
        "n_charts_with_anchors": int(anchors["chart_id"].nunique()),
        "n_anchor_rows": int(len(anchors)),
        "mach_range": [float(anchors["Mach"].min()), float(anchors["Mach"].max())],
        "eta_range": [float(anchors["eta"].min()), float(anchors["eta"].max())],
        "alpha_range": [float(anchors["alpha"].min()), float(anchors["alpha"].max())],
        "inventory": inventory,
    }
    return anchors, report


def save_all_formats(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def plot_sparse_supervision(anchors: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    chart_ids = sorted(anchors["chart_id"].astype(str).unique())
    cmap = plt.get_cmap("tab20", max(len(chart_ids), 1))

    for index, chart_id in enumerate(chart_ids):
        subset = anchors.loc[anchors["chart_id"].astype(str) == chart_id]
        ax.scatter(
            subset["Mach"],
            subset["eta"],
            s=24,
            alpha=0.82,
            label=chart_id,
            color=cmap(index),
            edgecolors="none",
        )

    ax.set(
        xlabel=r"Mach number $M$",
        ylabel=r"$\eta=\alpha/\sqrt{1-M^2}$",
        title="Explicit sparse spectral-supervision anchors",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    ax.grid(True, alpha=0.25)

    if len(chart_ids) <= 16:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=7,
            frameon=False,
        )

    fig.tight_layout()
    save_all_formats(fig, FIG_DIR / "Fig03_sparse_spectral_supervision")
    plt.close(fig)


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
) -> tuple[float, float]:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.25,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=9,
        linespacing=1.25,
    )
    return x + width / 2, y + height / 2


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.25,
            color="black",
            shrinkA=3,
            shrinkB=3,
        )
    )


def plot_method() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        "Subsonic Kelvin–Helmholtz PINN atlas and seeded-GEP workflow",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
    )

    width = 0.145
    height = 0.19
    y_top = 0.63
    xs = [0.025, 0.19, 0.355, 0.52, 0.685, 0.85]
    boxes = [
        ("Physical coordinates", r"$(M,\eta)$" + "\n" + r"$\alpha=\eta\sqrt{1-M^2}$"),
        ("Atlas routing", "coverage map\nchart assignment"),
        ("Local PINN", r"$c_i$ seed and fields" + "\n" + r"$p,\ q$ (or $\alpha Q$)"),
        ("Direct modes", r"reconstruct $p,\rho,u,v$" + "\n" + "phase/amplitude alignment"),
        ("Seeded dense GEP", "standard / long-wave /\nnear-neutral policy"),
        ("Final validation", "branch controls, continuation,\nmodal errors and overlap"),
    ]

    centers = [
        add_box(ax, x, y_top, width, height, title, body)
        for x, (title, body) in zip(xs, boxes)
    ]

    for left, right in zip(centers[:-1], centers[1:]):
        add_arrow(
            ax,
            (left[0] + width / 2 - 0.004, left[1]),
            (right[0] - width / 2 + 0.004, right[1]),
        )

    lower_y = 0.22
    lower_width = 0.245
    lower_height = 0.17

    add_box(
        ax,
        0.20,
        lower_y,
        lower_width,
        lower_height,
        "Sparse spectral supervision",
        "explicit checkpoint anchors\nacross local charts",
    )
    add_box(
        ax,
        0.555,
        lower_y,
        lower_width,
        lower_height,
        "Publication assets",
        r"$c_i$ maps, mode PDFs, heatmaps," + "\n" + "audits and release manifests",
    )

    add_arrow(ax, (0.2725, lower_y + lower_height), (0.2725, y_top))
    add_arrow(ax, (0.9275, y_top), (0.6775, lower_y + lower_height))

    ax.text(
        0.5,
        0.08,
        "The PINN provides a continuous local surrogate; the dense GEP "
        "performs branch-controlled modal refinement.",
        ha="center",
        va="center",
        fontsize=10,
    )

    fig.tight_layout()
    save_all_formats(fig, FIG_DIR / "Fig01_method")
    plt.close(fig)


def update_build_report(anchor_report: dict[str, Any]) -> None:
    if BUILD_REPORT.exists():
        report = json.loads(BUILD_REPORT.read_text(encoding="utf-8"))
    else:
        report = {}

    relative_anchor = str(ANCHOR_CSV.relative_to(ROOT))
    report["anchor_files"] = [relative_anchor]
    report["anchor_source"] = "checkpoint.anchor_df"
    report["anchor_count"] = int(anchor_report["n_anchor_rows"])
    report["anchor_chart_count"] = int(anchor_report["n_charts_with_anchors"])

    generated = report.setdefault("generated", [])
    for path in (
        FIG_DIR / "Fig01_method.pdf",
        FIG_DIR / "Fig01_method.png",
        FIG_DIR / "Fig03_sparse_spectral_supervision.pdf",
        FIG_DIR / "Fig03_sparse_spectral_supervision.png",
        ANCHOR_CSV,
        ANCHOR_REPORT,
    ):
        relative = str(path.relative_to(ROOT))
        if relative not in generated:
            generated.append(relative)

    BUILD_REPORT.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def copy_release_assets() -> None:
    destinations = {
        FIG_DIR / "Fig01_method.pdf":
            RELEASE / "figures" / "main" / "Fig01_method.pdf",
        FIG_DIR / "Fig01_method.png":
            RELEASE / "figures" / "main" / "Fig01_method.png",
        FIG_DIR / "Fig03_sparse_spectral_supervision.pdf":
            RELEASE / "figures" / "main" / "Fig03_sparse_spectral_supervision.pdf",
        FIG_DIR / "Fig03_sparse_spectral_supervision.png":
            RELEASE / "figures" / "main" / "Fig03_sparse_spectral_supervision.png",
        ANCHOR_CSV:
            RELEASE / "data" / "spectral_supervision_anchors_explicit.csv",
        ANCHOR_REPORT:
            RELEASE / "manifests" / "spectral_supervision_anchors_report.json",
        BUILD_REPORT:
            RELEASE / "manifests" / "publication_assets_build_report.json",
    }

    for source, destination in destinations.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    anchors, anchor_report = extract_anchors()
    anchors.to_csv(ANCHOR_CSV, index=False)
    ANCHOR_REPORT.write_text(
        json.dumps(anchor_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    plot_sparse_supervision(anchors)
    plot_method()
    update_build_report(anchor_report)
    copy_release_assets()

    print(json.dumps({
        "anchors_csv": str(ANCHOR_CSV),
        "n_anchor_rows": int(len(anchors)),
        "n_anchor_charts": int(anchors["chart_id"].nunique()),
        "fig01_pdf": str(FIG_DIR / "Fig01_method.pdf"),
        "fig03_pdf": str(FIG_DIR / "Fig03_sparse_spectral_supervision.pdf"),
        "build_report": str(BUILD_REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
