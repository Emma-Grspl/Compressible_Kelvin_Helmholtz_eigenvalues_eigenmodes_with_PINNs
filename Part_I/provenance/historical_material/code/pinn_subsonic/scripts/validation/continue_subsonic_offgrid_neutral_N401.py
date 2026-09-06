from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from classical_solver.gep.dense_gep_notebook_style import (
    NotebookStyleDenseGEPSolver,
)

from scripts.compare_kh_subsonic_fixed_mach_modal_candidates import (
    load_classic_full_mode,
)

from scripts.dev.benchmark_subsonic_local_atlas_core_ci_seeded_gep_v2 import (
    align_complex,
    interp_complex,
    overlap_complex,
    rel_l2,
    split_gep_vector,
)

from scripts.dev.validate_subsonic_atlas_offgrid import (
    SeedProvider,
)


def alpha_from_eta(
    eta: float,
    mach: float,
) -> float:
    return float(
        eta * math.sqrt(max(0.0, 1.0 - mach**2))
    )


def build_path(
    start: float,
    target: float,
    max_step: float,
) -> np.ndarray:
    distance = abs(target - start)

    n_intervals = max(
        1,
        int(math.ceil(distance / max_step)),
    )

    return np.linspace(
        start,
        target,
        n_intervals + 1,
        dtype=float,
    )


def solve_single_mode(
    *,
    eta: float,
    mach: float,
    target_guess: tuple[float, float],
    n_points: int,
    mapping_scale: float,
    xi_max: float,
) -> dict:
    alpha = alpha_from_eta(
        eta=eta,
        mach=mach,
    )

    solver = NotebookStyleDenseGEPSolver(
        alpha=alpha,
        Mach=mach,
        n_points=n_points,
        mapping_kind="pin",
        mapping_scale=mapping_scale,
        xi_max=xi_max,
    )

    mode, selection_source, n_modes = (
        solver.get_nearest_mode_to_target(
            target_guess=target_guess,
            prefer_positive_cr=False,
            ci_weight=2.0,
        )
    )

    if mode is None:
        raise RuntimeError(
            "No finite GEP mode found for "
            f"M={mach}, eta={eta}, alpha={alpha}, "
            f"target_guess={target_guess}"
        )

    fields = split_gep_vector(
        mode["vector"],
        solver.n_points,
        mach,
    )

    return {
        "eta": float(eta),
        "alpha": float(alpha),
        "Mach": float(mach),
        "cr": float(mode["cr"]),
        "ci": float(mode["ci"]),
        "y": np.asarray(
            solver.y,
            dtype=float,
        ),
        "fields": fields,
        "selection_source": str(
            selection_source
        ),
        "n_finite_modes": int(n_modes),
    }


def state_overlap(
    candidate: dict,
    reference: dict,
) -> float:
    y_reference = np.asarray(
        reference["y"],
        dtype=float,
    )

    p_reference = np.asarray(
        reference["fields"]["p"],
        dtype=np.complex128,
    )

    p_candidate = interp_complex(
        np.asarray(
            candidate["y"],
            dtype=float,
        ),
        np.asarray(
            candidate["fields"]["p"],
            dtype=np.complex128,
        ),
        y_reference,
    )

    mask = (
        (y_reference >= -12.0)
        & (y_reference <= 12.0)
    )

    scale = align_complex(
        p_candidate,
        p_reference,
        mask,
    )

    p_candidate = scale * p_candidate

    return float(
        overlap_complex(
            p_candidate,
            p_reference,
            y_reference,
            mask,
        )
    )


def run_direction(
    *,
    direction: str,
    eta_values: np.ndarray,
    mach: float,
    provider: SeedProvider,
    n_points: int,
    mapping_scale: float,
    xi_max: float,
    fallback_overlap: float,
) -> tuple[dict, pd.DataFrame]:
    records: list[dict] = []

    previous: dict | None = None

    for step_index, eta in enumerate(
        eta_values
    ):
        eta = float(eta)

        alpha = alpha_from_eta(
            eta=eta,
            mach=mach,
        )

        ci_seed = provider.predict(
            alpha=alpha,
            Mach=mach,
        )

        if previous is None:
            selected = solve_single_mode(
                eta=eta,
                mach=mach,
                target_guess=(
                    0.0,
                    float(ci_seed),
                ),
                n_points=n_points,
                mapping_scale=mapping_scale,
                xi_max=xi_max,
            )

            selection_basis = "pinn_seed_start"
            adjacent_overlap = np.nan

        else:
            continuation_candidate = (
                solve_single_mode(
                    eta=eta,
                    mach=mach,
                    target_guess=(
                        float(previous["cr"]),
                        float(previous["ci"]),
                    ),
                    n_points=n_points,
                    mapping_scale=mapping_scale,
                    xi_max=xi_max,
                )
            )

            continuation_overlap = (
                state_overlap(
                    continuation_candidate,
                    previous,
                )
            )

            selected = continuation_candidate
            adjacent_overlap = (
                continuation_overlap
            )
            selection_basis = (
                "previous_eigenvalue"
            )

            # Si la sélection spectrale se décorrèle du mode
            # précédent, tester également le seed PINN local.
            if (
                continuation_overlap
                < fallback_overlap
            ):
                seed_candidate = (
                    solve_single_mode(
                        eta=eta,
                        mach=mach,
                        target_guess=(
                            0.0,
                            float(ci_seed),
                        ),
                        n_points=n_points,
                        mapping_scale=mapping_scale,
                        xi_max=xi_max,
                    )
                )

                seed_overlap = state_overlap(
                    seed_candidate,
                    previous,
                )

                if (
                    seed_overlap
                    > continuation_overlap
                ):
                    selected = seed_candidate
                    adjacent_overlap = (
                        seed_overlap
                    )
                    selection_basis = (
                        "pinn_seed_fallback"
                    )

        records.append(
            {
                "direction": direction,
                "step_index": step_index,
                "Mach": mach,
                "eta": eta,
                "alpha": alpha,
                "ci_seed": ci_seed,
                "gep_cr": selected["cr"],
                "gep_ci": selected["ci"],
                "adjacent_overlap": (
                    adjacent_overlap
                ),
                "selection_basis": (
                    selection_basis
                ),
                "selection_source": (
                    selected[
                        "selection_source"
                    ]
                ),
                "n_finite_modes": (
                    selected[
                        "n_finite_modes"
                    ]
                ),
            }
        )

        previous = selected

    if previous is None:
        raise RuntimeError(
            f"Empty continuation path: {direction}"
        )

    return previous, pd.DataFrame(records)


def compare_to_classic(
    state: dict,
    *,
    mach: float,
    alpha: float,
) -> dict:
    classic_fields, ci_classic = (
        load_classic_full_mode(
            alpha,
            mach,
        )
    )

    y_reference = np.asarray(
        classic_fields["y"],
        dtype=float,
    )

    references = {
        name: np.asarray(
            classic_fields[name],
            dtype=np.complex128,
        )
        for name in [
            "p",
            "rho",
            "u",
            "v",
        ]
    }

    predictions = {
        name: interp_complex(
            np.asarray(
                state["y"],
                dtype=float,
            ),
            np.asarray(
                state["fields"][name],
                dtype=np.complex128,
            ),
            y_reference,
        )
        for name in [
            "p",
            "rho",
            "u",
            "v",
        ]
    }

    mask = (
        (y_reference >= -12.0)
        & (y_reference <= 12.0)
    )

    scale = align_complex(
        predictions["p"],
        references["p"],
        mask,
    )

    for name in predictions:
        predictions[name] = (
            scale * predictions[name]
        )

    return {
        "ci_classic_raw": float(
            ci_classic
        ),
        "ci_cont_abs_err_vs_classic": abs(
            float(state["ci"])
            - float(ci_classic)
        ),
        "ci_cont_rel_err_vs_classic": (
            abs(
                float(state["ci"])
                - float(ci_classic)
            )
            / max(
                abs(float(ci_classic)),
                1.0e-12,
            )
        ),
        "p_rel_cont_vs_classic": rel_l2(
            predictions["p"],
            references["p"],
            y_reference,
            mask,
        ),
        "rho_rel_cont_vs_classic": rel_l2(
            predictions["rho"],
            references["rho"],
            y_reference,
            mask,
        ),
        "u_rel_cont_vs_classic": rel_l2(
            predictions["u"],
            references["u"],
            y_reference,
            mask,
        ),
        "v_rel_cont_vs_classic": rel_l2(
            predictions["v"],
            references["v"],
            y_reference,
            mask,
        ),
        "p_overlap_cont_vs_classic": (
            overlap_complex(
                predictions["p"],
                references["p"],
                y_reference,
                mask,
            )
        ),
    }


def resolve_chart_path(
    row: pd.Series,
) -> Path:
    chart_path_value = row.get(
        "chart_path",
        np.nan,
    )

    if pd.notna(chart_path_value):
        chart_path = Path(
            str(chart_path_value)
        )

        if (
            chart_path
            / "model_best.pt"
        ).is_file():
            return chart_path

    checkpoint_value = row.get(
        "checkpoint_path",
        np.nan,
    )

    if pd.notna(checkpoint_value):
        checkpoint_path = Path(
            str(checkpoint_value)
        )

        if checkpoint_path.is_file():
            return checkpoint_path.parent

    raise FileNotFoundError(
        "Unable to resolve model_best.pt for "
        f"point_id={row.get('point_id')}, "
        f"chart={row.get('chart_id')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--targets-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--target-index",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--eta-low",
        type=float,
        default=0.92,
    )

    parser.add_argument(
        "--eta-high",
        type=float,
        default=0.98,
    )

    parser.add_argument(
        "--max-step",
        type=float,
        default=0.0025,
    )

    parser.add_argument(
        "--N",
        type=int,
        default=401,
    )

    parser.add_argument(
        "--mapping-scale",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--xi-max",
        type=float,
        default=0.98,
    )

    parser.add_argument(
        "--fallback-overlap",
        type=float,
        default=0.995,
    )

    parser.add_argument(
        "--accept-overlap",
        type=float,
        default=0.999,
    )

    parser.add_argument(
        "--accept-ci-abs",
        type=float,
        default=5.0e-4,
    )

    args = parser.parse_args()

    targets = pd.read_csv(
        args.targets_csv
    ).reset_index(drop=True)

    if not (
        0 <= args.target_index < len(targets)
    ):
        raise IndexError(
            f"target-index={args.target_index}; "
            f"valid range is 0..{len(targets)-1}"
        )

    row = targets.iloc[
        args.target_index
    ]

    point_id = str(row["point_id"])
    mach = float(row["Mach"])
    eta_target = float(row["eta"])
    alpha_target = float(row["alpha"])
    chart_id = str(row["chart_id"])

    output_dir = (
        args.output_dir / point_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir / "summary.csv"
    )

    path_output = (
        output_dir
        / "continuation_paths.csv"
    )

    try:
        chart_path = resolve_chart_path(
            row
        )

        provider = SeedProvider(
            chart_path
        )

        eta_forward = build_path(
            args.eta_low,
            eta_target,
            args.max_step,
        )

        eta_backward = build_path(
            args.eta_high,
            eta_target,
            args.max_step,
        )

        print(
            f"point_id={point_id}"
        )
        print(
            f"chart_id={chart_id}"
        )
        print(
            f"Mach={mach:.12g}"
        )
        print(
            f"eta_target={eta_target:.12g}"
        )
        print(
            f"forward_steps={len(eta_forward)}"
        )
        print(
            f"backward_steps={len(eta_backward)}"
        )
        print(
            f"chart_path={chart_path}"
        )

        forward_state, forward_path = (
            run_direction(
                direction="low_to_target",
                eta_values=eta_forward,
                mach=mach,
                provider=provider,
                n_points=args.N,
                mapping_scale=(
                    args.mapping_scale
                ),
                xi_max=args.xi_max,
                fallback_overlap=(
                    args.fallback_overlap
                ),
            )
        )

        backward_state, backward_path = (
            run_direction(
                direction="high_to_target",
                eta_values=eta_backward,
                mach=mach,
                provider=provider,
                n_points=args.N,
                mapping_scale=(
                    args.mapping_scale
                ),
                xi_max=args.xi_max,
                fallback_overlap=(
                    args.fallback_overlap
                ),
            )
        )

        direction_overlap = state_overlap(
            backward_state,
            forward_state,
        )

        direction_ci_abs = abs(
            float(forward_state["ci"])
            - float(backward_state["ci"])
        )

        forward_min_overlap = (
            forward_path[
                "adjacent_overlap"
            ]
            .dropna()
            .min()
        )

        backward_min_overlap = (
            backward_path[
                "adjacent_overlap"
            ]
            .dropna()
            .min()
        )

        if pd.isna(forward_min_overlap):
            forward_min_overlap = 1.0

        if pd.isna(backward_min_overlap):
            backward_min_overlap = 1.0

        minimum_adjacent_overlap = min(
            float(forward_min_overlap),
            float(backward_min_overlap),
        )

        classic = compare_to_classic(
            forward_state,
            mach=mach,
            alpha=alpha_target,
        )

        ci_seed_target = provider.predict(
            alpha=alpha_target,
            Mach=mach,
        )

        continuation_valid = (
            direction_overlap
            >= args.accept_overlap
            and direction_ci_abs
            <= args.accept_ci_abs
            and minimum_adjacent_overlap
            >= args.fallback_overlap
        )

        if continuation_valid:
            continuation_status = (
                "validated_bidirectional"
            )
        else:
            continuation_status = (
                "needs_manual_audit"
            )

        modal_classic_overlap = float(
            classic[
                "p_overlap_cont_vs_classic"
            ]
        )

        classic_ci_abs = float(
            classic[
                "ci_cont_abs_err_vs_classic"
            ]
        )

        if classic_ci_abs <= 5.0e-4:
            reference_status = (
                "classic_consistent"
            )
        elif modal_classic_overlap >= 0.999:
            reference_status = (
                "continuation_corrected_reference"
            )
        else:
            reference_status = (
                "branch_switch_suspected"
            )

        summary = {
            "target_index": args.target_index,
            "point_id": point_id,
            "sample_group": row.get(
                "sample_group",
                "",
            ),
            "Mach": mach,
            "eta": eta_target,
            "alpha": alpha_target,
            "chart_id": chart_id,
            "chart_path": str(
                chart_path
            ),
            "N": args.N,
            "mapping_scale": (
                args.mapping_scale
            ),
            "xi_max": args.xi_max,
            "eta_low": args.eta_low,
            "eta_high": args.eta_high,
            "max_step": args.max_step,
            "forward_n_steps": len(
                eta_forward
            ),
            "backward_n_steps": len(
                eta_backward
            ),
            "ci_seed_target": (
                ci_seed_target
            ),
            "ci_independent_original": (
                row.get(
                    "gep_ci",
                    np.nan,
                )
            ),
            "ci_forward": float(
                forward_state["ci"]
            ),
            "cr_forward": float(
                forward_state["cr"]
            ),
            "ci_backward": float(
                backward_state["ci"]
            ),
            "cr_backward": float(
                backward_state["cr"]
            ),
            "forward_backward_ci_abs": (
                direction_ci_abs
            ),
            "forward_backward_overlap": (
                direction_overlap
            ),
            "forward_min_adjacent_overlap": (
                forward_min_overlap
            ),
            "backward_min_adjacent_overlap": (
                backward_min_overlap
            ),
            "minimum_adjacent_overlap": (
                minimum_adjacent_overlap
            ),
            "continuation_status": (
                continuation_status
            ),
            "reference_status": (
                reference_status
            ),
            "success": True,
            "error": "",
            "traceback": "",
            **classic,
        }

        paths = pd.concat(
            [
                forward_path,
                backward_path,
            ],
            ignore_index=True,
        )

        paths.insert(
            0,
            "point_id",
            point_id,
        )

        paths.to_csv(
            path_output,
            index=False,
        )

    except Exception as error:
        summary = {
            "target_index": args.target_index,
            "point_id": point_id,
            "sample_group": row.get(
                "sample_group",
                "",
            ),
            "Mach": mach,
            "eta": eta_target,
            "alpha": alpha_target,
            "chart_id": chart_id,
            "continuation_status": (
                "failed"
            ),
            "reference_status": (
                "unknown"
            ),
            "success": False,
            "error": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
            "traceback": (
                traceback.format_exc()
            ),
        }

        print(
            summary["traceback"]
        )

    pd.DataFrame(
        [summary]
    ).to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        pd.DataFrame(
            [summary]
        ).to_string(index=False)
    )
    print()
    print("Summary:", summary_path)

    if not summary["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
