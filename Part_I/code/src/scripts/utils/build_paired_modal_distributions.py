#!/usr/bin/env python3

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def finite_pair(df, direct_col, final_col):
    direct = pd.to_numeric(
        df[direct_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    final = pd.to_numeric(
        df[final_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    mask = np.isfinite(direct) & np.isfinite(final)

    return direct[mask], final[mask]


def summary(values):
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"
    data_dir = args.output_dir / "data"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    required = [
        "p_rel_direct",
        "rho_rel_direct",
        "u_rel_direct",
        "v_rel_direct",
        "p_overlap_direct",
        "p_rel_final",
        "rho_rel_final",
        "u_rel_final",
        "v_rel_final",
        "p_overlap_final",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise KeyError(f"Colonnes absentes : {missing}")

    direct_fields = [
        "p_rel_direct",
        "rho_rel_direct",
        "u_rel_direct",
        "v_rel_direct",
    ]
    final_fields = [
        "p_rel_final",
        "rho_rel_final",
        "u_rel_final",
        "v_rel_final",
    ]

    direct_matrix = (
        df[direct_fields]
        .apply(pd.to_numeric, errors="coerce")
    )
    final_matrix = (
        df[final_fields]
        .apply(pd.to_numeric, errors="coerce")
    )

    df["modal_error_mean_direct"] = (
        direct_matrix.mean(axis=1, skipna=False)
    )
    df["modal_error_mean_final"] = (
        final_matrix.mean(axis=1, skipna=False)
    )

    df["modal_error_max_direct"] = (
        direct_matrix.max(axis=1, skipna=False)
    )
    df["modal_error_max_final"] = (
        final_matrix.max(axis=1, skipna=False)
    )

    df["p_overlap_defect_direct"] = (
        1.0
        - pd.to_numeric(
            df["p_overlap_direct"],
            errors="coerce",
        )
    )
    df["p_overlap_defect_final"] = (
        1.0
        - pd.to_numeric(
            df["p_overlap_final"],
            errors="coerce",
        )
    )

    metrics = {
        "pressure_relative_error": (
            "p_rel_direct",
            "p_rel_final",
        ),
        "density_relative_error": (
            "rho_rel_direct",
            "rho_rel_final",
        ),
        "streamwise_velocity_relative_error": (
            "u_rel_direct",
            "u_rel_final",
        ),
        "transverse_velocity_relative_error": (
            "v_rel_direct",
            "v_rel_final",
        ),
        "mean_modal_relative_error": (
            "modal_error_mean_direct",
            "modal_error_mean_final",
        ),
        "maximum_modal_relative_error": (
            "modal_error_max_direct",
            "modal_error_max_final",
        ),
        "pressure_overlap_defect": (
            "p_overlap_defect_direct",
            "p_overlap_defect_final",
        ),
    }

    rows = []

    for metric, (direct_col, final_col) in metrics.items():
        direct, final = finite_pair(
            df,
            direct_col,
            final_col,
        )

        direct_stats = summary(direct)
        final_stats = summary(final)

        denominator = np.maximum(
            np.abs(direct),
            1.0e-15,
        )
        paired_ratio = final / denominator

        rows.append({
            "metric": metric,
            "n_pairs": len(direct),
            "direct_mean": direct_stats["mean"],
            "direct_median": direct_stats["median"],
            "direct_p90": direct_stats["p90"],
            "direct_p95": direct_stats["p95"],
            "direct_p99": direct_stats["p99"],
            "direct_max": direct_stats["max"],
            "final_mean": final_stats["mean"],
            "final_median": final_stats["median"],
            "final_p90": final_stats["p90"],
            "final_p95": final_stats["p95"],
            "final_p99": final_stats["p99"],
            "final_max": final_stats["max"],
            "n_final_better": int(
                np.sum(final < direct)
            ),
            "fraction_final_better": float(
                np.mean(final < direct)
            ),
            "median_ratio_final_over_direct": float(
                np.median(paired_ratio)
            ),
            "median_reduction_percent": float(
                100.0
                * np.median(1.0 - paired_ratio)
            ),
        })

    table = pd.DataFrame(rows)

    table_path = (
        tables_dir
        / "Table_paired_modal_distributions_20.csv"
    )
    table.to_csv(table_path, index=False)

    point_columns = [
        column
        for column in [
            "selection_stratum",
            "point_id",
            "sample_group",
            "Mach",
            "eta",
            "alpha",
            "chart_id",
            "modal_error_mean_direct",
            "modal_error_mean_final",
            "modal_error_max_direct",
            "modal_error_max_final",
            "p_overlap_direct",
            "p_overlap_final",
        ]
        if column in df.columns
    ]

    points_path = (
        data_dir
        / "paired_modal_validation_20.csv"
    )
    df[point_columns].to_csv(
        points_path,
        index=False,
    )

    direct, final = finite_pair(
        df,
        "modal_error_mean_direct",
        "modal_error_mean_final",
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.5))

    for values, label in [
        (direct, "Direct PINN"),
        (final, "PINN-seeded GEP"),
    ]:
        values = np.sort(values[values > 0.0])
        cumulative = (
            np.arange(1, len(values) + 1)
            / len(values)
        )

        ax.step(
            values,
            cumulative,
            where="post",
            label=f"{label} (n={len(values)})",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Mean relative modal error")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    png_path = (
        figures_dir
        / "Fig_paired_modal_error_ecdf_20.png"
    )
    pdf_path = (
        figures_dir
        / "Fig_paired_modal_error_ecdf_20.pdf"
    )

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(table.to_string(index=False))

    print("\nFichiers produits :")
    print(table_path)
    print(points_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
