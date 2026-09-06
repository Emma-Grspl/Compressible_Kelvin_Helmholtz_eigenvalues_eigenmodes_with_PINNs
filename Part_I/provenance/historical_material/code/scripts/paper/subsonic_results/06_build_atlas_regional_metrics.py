#!/usr/bin/env python3

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


def find_col(df, names, required=True):
    lookup = {str(c).lower(): str(c) for c in df.columns}

    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lookup:
            return lookup[name.lower()]

    if required:
        raise KeyError(
            f"Colonnes recherchées : {names}\n"
            f"Colonnes disponibles : {list(df.columns)}"
        )

    return None


def values(df, name):
    return pd.to_numeric(df[name], errors="coerce")


def get_error(
    df,
    error_names,
    prediction_names,
    reference_names,
):
    error_col = find_col(df, error_names, required=False)

    if error_col is not None:
        return values(df, error_col).abs(), {
            "error_column": error_col,
            "prediction_column": None,
            "reference_column": None,
        }

    prediction_col = find_col(df, prediction_names)
    reference_col = find_col(df, reference_names)

    error = (
        values(df, prediction_col)
        - values(df, reference_col)
    ).abs()

    return error, {
        "error_column": None,
        "prediction_column": prediction_col,
        "reference_column": reference_col,
    }


def stats(series):
    array = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float)

    array = array[np.isfinite(array)]

    if len(array) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }

    return {
        "n": len(array),
        "mae": np.mean(array),
        "median": np.median(array),
        "p95": np.quantile(array, 0.95),
        "p99": np.quantile(array, 0.99),
        "max": np.max(array),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--eta-long-wave-max",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--eta-near-neutral-min",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--mach-high-min",
        type=float,
        default=0.90,
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    tables_dir = args.output_dir / "tables"
    data_dir = args.output_dir / "data"

    tables_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    mach_col = find_col(df, ["Mach", "mach", "M"])
    eta_col = find_col(df, ["eta", "Eta"], required=False)
    alpha_col = find_col(
        df,
        ["alpha", "Alpha"],
        required=False,
    )

    mach = values(df, mach_col)

    if eta_col is not None:
        eta = values(df, eta_col)
    else:
        if alpha_col is None:
            raise KeyError("Aucune colonne eta ou alpha.")

        alpha = values(df, alpha_col)
        denominator = np.sqrt(
            np.maximum(1.0 - mach.to_numpy() ** 2, 0.0)
        )

        eta = pd.Series(
            np.divide(
                alpha.to_numpy(),
                denominator,
                out=np.full(len(df), np.nan),
                where=denominator > 0.0,
            ),
            index=df.index,
        )

    if alpha_col is not None:
        alpha = values(df, alpha_col)
    else:
        alpha = pd.Series(
            eta.to_numpy()
            * np.sqrt(
                np.maximum(
                    1.0 - mach.to_numpy() ** 2,
                    0.0,
                )
            ),
            index=df.index,
        )

    seed_error, seed_source = get_error(
        df,
        [
            "ci_seed_abs_err",
            "seed_abs_err",
            "ci_pinn_abs_err",
            "ci_direct_abs_err",
        ],
        [
            "ci_seed",
            "ci_pinn",
            "ci_direct",
            "ci_pred",
        ],
        [
            "ci_reference",
            "ci_classic",
            "ci_ref",
            "ci_true",
        ],
    )

    final_error, final_source = get_error(
        df,
        [
            "ci_final_abs_err",
            "gep_ci_abs_err",
            "ci_gep_abs_err",
            "final_abs_err",
        ],
        [
            "ci_final",
            "gep_ci",
            "ci_gep",
            "ci_refined",
        ],
        [
            "ci_reference",
            "ci_classic",
            "ci_ref",
            "ci_true",
        ],
    )

    valid = (
        np.isfinite(mach)
        & np.isfinite(eta)
        & np.isfinite(seed_error)
        & np.isfinite(final_error)
    )

    long_wave = valid & (
        eta <= args.eta_long_wave_max
    )
    near_neutral = valid & (
        eta >= args.eta_near_neutral_min
    )
    high_mach = valid & (
        mach >= args.mach_high_min
    )

    edge_union = long_wave | near_neutral | high_mach
    interior = valid & ~edge_union

    regions = {
        "all": valid,
        "interior": interior,
        "long_wavelength": long_wave,
        "near_neutral": near_neutral,
        "high_mach": high_mach,
        "edge_corner_union": edge_union,
    }

    definitions = {
        "all": "all valid points",
        "interior": (
            f"eta>{args.eta_long_wave_max:g}, "
            f"eta<{args.eta_near_neutral_min:g}, "
            f"Mach<{args.mach_high_min:g}"
        ),
        "long_wavelength": (
            f"eta<={args.eta_long_wave_max:g}"
        ),
        "near_neutral": (
            f"eta>={args.eta_near_neutral_min:g}"
        ),
        "high_mach": (
            f"Mach>={args.mach_high_min:g}"
        ),
        "edge_corner_union": (
            f"eta<={args.eta_long_wave_max:g} or "
            f"eta>={args.eta_near_neutral_min:g} or "
            f"Mach>={args.mach_high_min:g}"
        ),
    }

    rows = []

    for region, mask in regions.items():
        seed = stats(seed_error[mask])
        final = stats(final_error[mask])

        if np.isfinite(seed["mae"]) and seed["mae"] > 0:
            reduction = (
                100.0
                * (seed["mae"] - final["mae"])
                / seed["mae"]
            )
        else:
            reduction = np.nan

        rows.append({
            "region": region,
            "definition": definitions[region],
            "n": int(mask.sum()),
            "ci_seed_mae": seed["mae"],
            "ci_seed_median": seed["median"],
            "ci_seed_p95": seed["p95"],
            "ci_seed_p99": seed["p99"],
            "ci_seed_max": seed["max"],
            "ci_final_mae": final["mae"],
            "ci_final_median": final["median"],
            "ci_final_p95": final["p95"],
            "ci_final_p99": final["p99"],
            "ci_final_max": final["max"],
            "gep_mae_reduction_percent": reduction,
        })

    table = pd.DataFrame(rows)

    table_path = (
        tables_dir
        / "Table_atlas_regional_spectral_metrics.csv"
    )
    table.to_csv(table_path, index=False)

    membership = pd.DataFrame({
        "source_row": np.arange(len(df)),
        "Mach": mach,
        "eta": eta,
        "alpha": alpha,
        "ci_seed_abs_err": seed_error,
        "ci_final_abs_err": final_error,
        "valid": valid,
        "region_interior": interior,
        "region_long_wavelength": long_wave,
        "region_near_neutral": near_neutral,
        "region_high_mach": high_mach,
        "region_edge_corner_union": edge_union,
    })

    membership_path = (
        data_dir / "atlas_regional_membership.csv"
    )
    membership.to_csv(membership_path, index=False)

    metadata = {
        "input": str(args.input),
        "input_rows": len(df),
        "valid_rows": int(valid.sum()),
        "mach_column": mach_col,
        "eta_column": eta_col,
        "alpha_column": alpha_col,
        "seed_error_source": seed_source,
        "final_error_source": final_source,
        "thresholds": {
            "eta_long_wave_max": args.eta_long_wave_max,
            "eta_near_neutral_min": (
                args.eta_near_neutral_min
            ),
            "mach_high_min": args.mach_high_min,
        },
    }

    metadata_path = (
        data_dir
        / "atlas_regional_metrics_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2)
    )

    print("Colonnes détectées :")
    print(json.dumps(metadata, indent=2))

    print("\nRésultats :")
    print(table.to_string(index=False))

    print("\nFichiers produits :")
    print(table_path)
    print(membership_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
