#!/usr/bin/env python3

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[8]

FIGURE_SCRIPTS = (
    ROOT
    / "code"
    / "plots"
    / "scripts"
    / "pinn_subsonic"
    / "curated_entrypoint"
    / "source_tree"
    / "pinn_subsonic"
    / "scripts"
    / "figures"
)
sys.path.insert(0, str(FIGURE_SCRIPTS))

import build_ci_and_atlas_assets as assets


SCI = (
    ROOT
    / "assets"
    / "pinn_subsonic"
    / "local_atlas_v1"
    / "publication_assets_scientific_v2"
)

POINTS_CSV = (
    ROOT
    / "assets"
    / "pinn_subsonic"
    / "release_v1"
    / "data"
    / "validation_mode_points_20.csv"
)

REPORT_DIR = (
    SCI
    / "data"
    / "mode_profiles"
    / "reports"
)

OUTPUT_CSV = (
    SCI
    / "data"
    / "direct_PINN_modal_errors_20.csv"
)

OUTPUT_REPORT = (
    SCI
    / "manifests"
    / "direct_PINN_modal_heatmaps_report.json"
)

OUTPUT_STEM = (
    SCI
    / "figures"
    / "SuppFig_direct_PINN_modal_error_heatmaps"
)


def main() -> None:
    assets.configure_plotting()

    points = pd.read_csv(
        POINTS_CSV,
        dtype={"point_id": str},
    )

    rows = []

    for path in sorted(
        REPORT_DIR.glob("OFFGRID_*.json")
    ):
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            report = json.load(handle)

        direct = report["direct"]

        rows.append(
            {
                "point_id": str(
                    report["point_id"]
                ),
                "p_rel_direct": float(
                    direct["p_rel"]
                ),
                "rho_rel_direct": float(
                    direct["rho_rel"]
                ),
                "u_rel_direct": float(
                    direct["u_rel"]
                ),
                "v_rel_direct": float(
                    direct["v_rel"]
                ),
                "p_overlap_direct": float(
                    direct["p_overlap"]
                ),
            }
        )

    metrics = pd.DataFrame(rows)

    if len(metrics) != 20:
        raise RuntimeError(
            "Expected 20 extraction reports, "
            f"found {len(metrics)}"
        )

    if metrics["point_id"].duplicated().any():
        duplicated = metrics.loc[
            metrics["point_id"].duplicated(
                keep=False
            ),
            "point_id",
        ].tolist()

        raise RuntimeError(
            f"Duplicated reports: {duplicated}"
        )

    frame = points.merge(
        metrics,
        on="point_id",
        how="left",
        validate="one_to_one",
    )

    direct_columns = [
        "p_rel_direct",
        "rho_rel_direct",
        "u_rel_direct",
        "v_rel_direct",
    ]

    missing = frame.loc[
        frame[direct_columns].isna().any(axis=1),
        "point_id",
    ].tolist()

    if missing:
        raise RuntimeError(
            f"Missing direct metrics: {missing}"
        )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_STEM.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    labels = [
        ("p_rel_direct", r"$p$"),
        ("rho_rel_direct", r"$\rho$"),
        ("u_rel_direct", r"$u$"),
        ("v_rel_direct", r"$v$"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 9),
    )

    for ax, (column, label) in zip(
        axes.ravel(),
        labels,
    ):
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        positive = values[
            np.isfinite(values)
            & (values > 0)
        ]

        if positive.empty:
            raise RuntimeError(
                f"No positive values for {column}"
            )

        vmin = max(
            float(positive.min()),
            1.0e-8,
        )

        vmax = max(
            float(positive.max()),
            1.01 * vmin,
        )

        norm = LogNorm(
            vmin=vmin,
            vmax=vmax,
        )

        assets.scatter_map(
            ax,
            frame,
            column,
            norm=norm,
            cmap="plasma",
            title=(
                "Direct PINN relative error: "
                f"{label}"
            ),
            cbar_label="relative error",
        )

    fig.suptitle(
        "Direct PINN modal validation — "
        "20 stratified off-grid points",
        y=1.01,
    )

    fig.tight_layout()

    assets.save_figure(
        fig,
        OUTPUT_STEM,
    )

    plt.close(fig)

    summary = {
        "n_points": int(len(frame)),
        "source": str(REPORT_DIR),
        "output_csv": str(OUTPUT_CSV),
        "output_figure_stem": str(
            OUTPUT_STEM
        ),
        "maxima": {
            column: float(frame[column].max())
            for column in direct_columns
        },
        "min_pressure_overlap": float(
            frame["p_overlap_direct"].min()
        ),
    }

    with OUTPUT_REPORT.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
