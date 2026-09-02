#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def complex_interp(
    x_new: np.ndarray,
    x_old: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return (
        np.interp(
            x_new,
            x_old,
            values.real,
        )
        + 1j
        * np.interp(
            x_new,
            x_old,
            values.imag,
        )
    )


def finite_complex(
    values: np.ndarray,
) -> bool:
    return bool(
        np.all(np.isfinite(values.real))
        and np.all(np.isfinite(values.imag))
    )


def l2_energy(
    y: np.ndarray,
    values: np.ndarray,
) -> float:
    return float(
        np.trapz(
            np.abs(values) ** 2,
            y,
        )
    )


def load_case(
    path: Path,
) -> dict[str, np.ndarray]:
    with np.load(
        path,
        allow_pickle=True,
    ) as data:
        return {
            key: data[key]
            for key in data.files
        }


def normalize_with_analytic_tails(
    data: dict[str, np.ndarray],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, float],
]:
    result = dict(data)

    y = np.asarray(
        result["y"],
        dtype=float,
    )

    p = np.asarray(
        result["p"],
        dtype=np.complex128,
    )

    q = np.asarray(
        result["q"],
        dtype=np.complex128,
    )

    gamma = np.asarray(
        result["gamma"],
        dtype=np.complex128,
    )

    if not (
        y.shape
        == p.shape
        == q.shape
        == gamma.shape
    ):
        raise RuntimeError(
            "Inconsistent field shapes."
        )

    if not (
        finite_complex(p)
        and finite_complex(q)
        and finite_complex(gamma)
    ):
        raise RuntimeError(
            "Non-finite modal field."
        )

    gamma_left = complex(
        np.asarray(
            result[
                "gamma_asymptotic_left"
            ]
        ).item()
    )

    gamma_right = complex(
        np.asarray(
            result[
                "gamma_asymptotic_right"
            ]
        ).item()
    )

    if gamma_left.real <= 0.0:
        raise RuntimeError(
            "Left asymptotic exponent "
            f"is not decaying: {gamma_left}"
        )

    if gamma_right.real >= 0.0:
        raise RuntimeError(
            "Right asymptotic exponent "
            f"is not decaying: {gamma_right}"
        )

    interior_energy = l2_energy(
        y,
        p,
    )

    left_tail_energy = float(
        abs(p[0]) ** 2
        / (
            2.0
            * gamma_left.real
        )
    )

    right_tail_energy = float(
        abs(p[-1]) ** 2
        / (
            -2.0
            * gamma_right.real
        )
    )

    total_energy = (
        interior_energy
        + left_tail_energy
        + right_tail_energy
    )

    if (
        not np.isfinite(total_energy)
        or total_energy <= 0.0
    ):
        raise RuntimeError(
            f"Invalid total modal energy: "
            f"{total_energy}"
        )

    normalization = float(
        np.sqrt(total_energy)
    )

    result["p"] = (
        p / normalization
    )

    result["q"] = (
        q / normalization
    )

    if "q_returned" in result:
        result["q_returned"] = (
            np.asarray(
                result["q_returned"],
                dtype=np.complex128,
            )
            / normalization
        )

    result[
        "analytic_tail_normalization"
    ] = np.array(normalization)

    result[
        "interior_energy_before_tail_norm"
    ] = np.array(interior_energy)

    result[
        "left_tail_energy_before_tail_norm"
    ] = np.array(left_tail_energy)

    result[
        "right_tail_energy_before_tail_norm"
    ] = np.array(right_tail_energy)

    tail_energy = (
        left_tail_energy
        + right_tail_energy
    )

    tail_fraction = float(
        tail_energy
        / total_energy
    )

    result[
        "analytic_tail_energy_fraction"
    ] = np.array(tail_fraction)

    diagnostics = {
        "interior_energy_before_tail_norm": (
            interior_energy
        ),
        "left_tail_energy_before_tail_norm": (
            left_tail_energy
        ),
        "right_tail_energy_before_tail_norm": (
            right_tail_energy
        ),
        "analytic_tail_energy_fraction": (
            tail_fraction
        ),
        "analytic_tail_normalization": (
            normalization
        ),
        "gamma_asymptotic_left_real": (
            gamma_left.real
        ),
        "gamma_asymptotic_left_imag": (
            gamma_left.imag
        ),
        "gamma_asymptotic_right_real": (
            gamma_right.real
        ),
        "gamma_asymptotic_right_imag": (
            gamma_right.imag
        ),
    }

    return result, diagnostics


def phase_to_reference(
    reference: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    window: float = 20.0,
) -> tuple[
    float,
    complex,
]:
    reference_y = np.asarray(
        reference["y"],
        dtype=float,
    )

    current_y = np.asarray(
        current["y"],
        dtype=float,
    )

    lower = max(
        float(reference_y[0]),
        float(current_y[0]),
        -window,
    )

    upper = min(
        float(reference_y[-1]),
        float(current_y[-1]),
        window,
    )

    mask = (
        (current_y >= lower)
        & (current_y <= upper)
    )

    if int(mask.sum()) < 32:
        raise RuntimeError(
            "Insufficient common overlap grid."
        )

    y_common = current_y[mask]

    reference_p = complex_interp(
        y_common,
        reference_y,
        np.asarray(
            reference["p"],
            dtype=np.complex128,
        ),
    )

    current_p = np.asarray(
        current["p"],
        dtype=np.complex128,
    )[mask]

    overlap = np.trapz(
        np.conjugate(reference_p)
        * current_p,
        y_common,
    )

    reference_norm = np.sqrt(
        max(
            l2_energy(
                y_common,
                reference_p,
            ),
            0.0,
        )
    )

    current_norm = np.sqrt(
        max(
            l2_energy(
                y_common,
                current_p,
            ),
            0.0,
        )
    )

    normalized_overlap = float(
        abs(overlap)
        / max(
            reference_norm
            * current_norm,
            1.0e-30,
        )
    )

    if abs(overlap) == 0.0:
        phase = 1.0 + 0.0j
    else:
        phase = np.exp(
            -1j
            * np.angle(overlap)
        )

    return (
        normalized_overlap,
        complex(phase),
    )


def apply_phase(
    data: dict[str, np.ndarray],
    phase: complex,
) -> None:
    for key in [
        "p",
        "q",
        "q_returned",
    ]:
        if key in data:
            data[key] = (
                np.asarray(
                    data[key],
                    dtype=np.complex128,
                )
                * phase
            )


def scalar(
    data: dict[str, np.ndarray],
    key: str,
) -> float:
    return float(
        np.asarray(
            data[key]
        ).item()
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-points",
        type=int,
        default=242,
    )
    parser.add_argument(
        "--expected-mach",
        type=int,
        default=22,
    )
    parser.add_argument(
        "--expected-s",
        type=int,
        default=11,
    )
    parser.add_argument(
        "--anchor-mach",
        type=float,
        default=1.8,
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalog_paths = sorted(
        args.input_dir.glob(
            "M_*/catalog_M_*.csv"
        )
    )

    if not catalog_paths:
        raise RuntimeError(
            "No per-Mach catalogs found."
        )

    catalog = pd.concat(
        [
            pd.read_csv(path)
            for path in catalog_paths
        ],
        ignore_index=True,
    )

    catalog = (
        catalog
        .sort_values(
            ["Mach", "s"]
        )
        .reset_index(drop=True)
    )

    catalog.to_csv(
        args.output_dir
        / "catalog_all_raw.csv",
        index=False,
    )

    modal_mask = (
        catalog["quality_level"]
        == "modal_certified"
    )

    initial_report = {
        "n_catalog_files": int(
            len(catalog_paths)
        ),
        "n_catalog_points": int(
            len(catalog)
        ),
        "n_mach": int(
            catalog["Mach"].nunique()
        ),
        "n_s": int(
            catalog["s"].nunique()
        ),
        "n_modal_certified": int(
            modal_mask.sum()
        ),
        "n_not_modal_certified": int(
            (~modal_mask).sum()
        ),
    }

    complete_input = bool(
        len(catalog)
        == args.expected_points
        and catalog["Mach"].nunique()
        == args.expected_mach
        and catalog["s"].nunique()
        == args.expected_s
        and modal_mask.all()
    )

    if (
        args.require_complete
        and not complete_input
    ):
        (
            args.output_dir
            / "assembly_report.json"
        ).write_text(
            json.dumps(
                {
                    **initial_report,
                    "complete_input": False,
                    "complete": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        raise RuntimeError(
            "The atlas input is incomplete "
            "or contains non-certified cases."
        )

    certified = (
        catalog[modal_mask]
        .copy()
        .reset_index(drop=True)
    )

    cases: dict[
        int,
        dict[str, np.ndarray],
    ] = {}

    for index, row in (
        certified.iterrows()
    ):
        field_path = Path(
            str(row["field_file"])
        )

        if not field_path.is_file():
            raise FileNotFoundError(
                field_path
            )

        case, diagnostics = (
            normalize_with_analytic_tails(
                load_case(field_path)
            )
        )

        cases[index] = case

        for key, value in (
            diagnostics.items()
        ):
            certified.loc[
                index,
                key,
            ] = value

    certified[
        "phase_cross_mach_real"
    ] = np.nan

    certified[
        "phase_cross_mach_imag"
    ] = np.nan

    certified[
        "overlap_cross_mach"
    ] = np.nan

    for s_value, group in (
        certified.groupby(
            "s",
            sort=True,
        )
    ):
        indices = (
            group
            .sort_values("Mach")
            .index
            .tolist()
        )

        anchor_index = min(
            indices,
            key=lambda index: abs(
                float(
                    certified.loc[
                        index,
                        "Mach",
                    ]
                )
                - args.anchor_mach
            ),
        )

        certified.loc[
            anchor_index,
            "phase_cross_mach_real",
        ] = 1.0

        certified.loc[
            anchor_index,
            "phase_cross_mach_imag",
        ] = 0.0

        certified.loc[
            anchor_index,
            "overlap_cross_mach",
        ] = 1.0

        anchor_position = (
            indices.index(
                anchor_index
            )
        )

        for position in range(
            anchor_position - 1,
            -1,
            -1,
        ):
            reference_index = (
                indices[
                    position + 1
                ]
            )

            current_index = (
                indices[position]
            )

            overlap, phase = (
                phase_to_reference(
                    cases[
                        reference_index
                    ],
                    cases[
                        current_index
                    ],
                )
            )

            apply_phase(
                cases[current_index],
                phase,
            )

            certified.loc[
                current_index,
                "phase_cross_mach_real",
            ] = phase.real

            certified.loc[
                current_index,
                "phase_cross_mach_imag",
            ] = phase.imag

            certified.loc[
                current_index,
                "overlap_cross_mach",
            ] = overlap

        for position in range(
            anchor_position + 1,
            len(indices),
        ):
            reference_index = (
                indices[
                    position - 1
                ]
            )

            current_index = (
                indices[position]
            )

            overlap, phase = (
                phase_to_reference(
                    cases[
                        reference_index
                    ],
                    cases[
                        current_index
                    ],
                )
            )

            apply_phase(
                cases[current_index],
                phase,
            )

            certified.loc[
                current_index,
                "phase_cross_mach_real",
            ] = phase.real

            certified.loc[
                current_index,
                "phase_cross_mach_imag",
            ] = phase.imag

            certified.loc[
                current_index,
                "overlap_cross_mach",
            ] = overlap

    certified = (
        certified
        .sort_values(
            ["Mach", "s"]
        )
        .reset_index(drop=True)
    )

    ordered_cases = [
        cases[index]
        for index in certified.index
    ]

    # cases correspond encore à l'ancien index avant sort.
    # Recharger l'ordre à partir de Mach et s.
    case_lookup: dict[
        tuple[float, float],
        dict[str, np.ndarray],
    ] = {}

    for old_index, row in (
        catalog[modal_mask]
        .copy()
        .reset_index(drop=True)
        .iterrows()
    ):
        key = (
            round(
                float(row["Mach"]),
                10,
            ),
            round(
                float(row["s"]),
                10,
            ),
        )

        case_lookup[key] = (
            cases[old_index]
        )

    ordered_cases = [
        case_lookup[
            (
                round(
                    float(row["Mach"]),
                    10,
                ),
                round(
                    float(row["s"]),
                    10,
                ),
            )
        ]
        for _, row
        in certified.iterrows()
    ]

    xi = np.asarray(
        ordered_cases[0]["xi"],
        dtype=float,
    )

    for case in ordered_cases[1:]:
        np.testing.assert_allclose(
            np.asarray(
                case["xi"],
                dtype=float,
            ),
            xi,
            rtol=0.0,
            atol=0.0,
        )

    np.savez_compressed(
        args.output_dir
        / "atlas_certified_v2.npz",
        Mach=certified[
            "Mach"
        ].to_numpy(float),
        s=certified[
            "s"
        ].to_numpy(float),
        alpha=certified[
            "alpha"
        ].to_numpy(float),
        alpha_neutral=certified[
            "alpha_neutral"
        ].to_numpy(float),
        cr=certified[
            "cr"
        ].to_numpy(float),
        ci=certified[
            "ci"
        ].to_numpy(float),
        xi=xi,
        y=np.stack(
            [
                np.asarray(
                    case["y"],
                    dtype=float,
                )
                for case in ordered_cases
            ]
        ),
        p=np.stack(
            [
                np.asarray(
                    case["p"],
                    dtype=np.complex128,
                )
                for case in ordered_cases
            ]
        ),
        q=np.stack(
            [
                np.asarray(
                    case["q"],
                    dtype=np.complex128,
                )
                for case in ordered_cases
            ]
        ),
        gamma=np.stack(
            [
                np.asarray(
                    case["gamma"],
                    dtype=np.complex128,
                )
                for case in ordered_cases
            ]
        ),
        analytic_tail_energy_fraction=(
            certified[
                "analytic_tail_energy_fraction"
            ].to_numpy(float)
        ),
        overlap_cross_mach=(
            certified[
                "overlap_cross_mach"
            ].to_numpy(float)
        ),
        source_field_file=(
            certified[
                "field_file"
            ].astype(str).to_numpy()
        ),
    )

    certified.to_csv(
        args.output_dir
        / "catalog_modal_certified.csv",
        index=False,
    )

    noncertified = catalog[
        ~modal_mask
    ].copy()

    noncertified.to_csv(
        args.output_dir
        / "catalog_not_modal_certified.csv",
        index=False,
    )

    cross_overlap = (
        certified[
            "overlap_cross_mach"
        ].dropna()
    )

    report: dict[str, Any] = {
        **initial_report,
        "complete_input": (
            complete_input
        ),
        "n_assembled": int(
            len(certified)
        ),
        "tail_energy_fraction_min": float(
            certified[
                "analytic_tail_energy_fraction"
            ].min()
        ),
        "tail_energy_fraction_mean": float(
            certified[
                "analytic_tail_energy_fraction"
            ].mean()
        ),
        "tail_energy_fraction_max": float(
            certified[
                "analytic_tail_energy_fraction"
            ].max()
        ),
        "cross_mach_overlap_min": float(
            cross_overlap.min()
        ),
        "cross_mach_overlap_mean": float(
            cross_overlap.mean()
        ),
        "complete": bool(
            complete_input
            and len(certified)
            == args.expected_points
        ),
    }

    (
        args.output_dir
        / "assembly_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
