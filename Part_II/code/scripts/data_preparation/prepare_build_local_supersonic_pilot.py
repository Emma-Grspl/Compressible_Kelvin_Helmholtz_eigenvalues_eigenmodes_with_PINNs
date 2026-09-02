#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_relative(path: Path) -> str:
    path = path.resolve()

    try:
        return str(
            path.relative_to(REPO_ROOT.resolve())
        )
    except ValueError:
        return str(path)


def select_mach(
    dataframe: pd.DataFrame,
    mach: float,
) -> pd.DataFrame:
    mask = np.isclose(
        dataframe["Mach"].to_numpy(float),
        float(mach),
        rtol=0.0,
        atol=1.0e-12,
    )

    result = dataframe.loc[mask].copy()

    if "alpha" in result.columns:
        result = result.sort_values("alpha")

    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mach",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--tag",
        required=True,
        help="Example: M110 or M190",
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "assets/pinn_supersonic/"
            "datasets/sparse_v1"
        ),
    )

    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=Path(
            "assets/pinn_supersonic/pilots"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    source = (
        args.dataset_dir
        if args.dataset_dir.is_absolute()
        else REPO_ROOT / args.dataset_dir
    )

    pilot_root = (
        args.pilot_root
        if args.pilot_root.is_absolute()
        else REPO_ROOT / args.pilot_root
    )

    pilot = (
        pilot_root
        / f"local_{args.tag}_sparse_v1"
    )

    if pilot.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{pilot} already exists"
            )

        shutil.rmtree(pilot)

    data_dir = pilot / "data"
    config_dir = pilot / "configs"

    for directory in [
        data_dir,
        config_dir,
        pilot / "runs",
        pilot / "logs",
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    spectral = pd.read_csv(
        source / "spectral_full_audit.csv"
    )

    anchors = pd.read_csv(
        source
        / "spectral_anchors_4_per_mach_all.csv"
    )

    modal_2 = pd.read_csv(
        source
        / "modal_anchor_manifest_2_per_mach_all.csv"
    )

    modal_4 = pd.read_csv(
        source
        / "modal_anchor_manifest_4_per_mach_all.csv"
    )

    spectral_mach = select_mach(
        spectral,
        args.mach,
    )

    anchors_mach = (
        select_mach(anchors, args.mach)
        .sort_values("anchor_rank")
        .reset_index(drop=True)
    )

    modal_2_mach = (
        select_mach(modal_2, args.mach)
        .sort_values("anchor_rank")
        .reset_index(drop=True)
    )

    modal_4_mach = (
        select_mach(modal_4, args.mach)
        .sort_values("anchor_rank")
        .reset_index(drop=True)
    )

    if spectral_mach.empty:
        raise RuntimeError(
            f"No spectral data at Mach={args.mach}"
        )

    if len(anchors_mach) != 4:
        raise RuntimeError(
            "Expected four spectral anchors, "
            f"found {len(anchors_mach)}"
        )

    if len(modal_2_mach) != 2:
        raise RuntimeError(
            "Expected two S4M2 modal anchors, "
            f"found {len(modal_2_mach)}"
        )

    if len(modal_4_mach) != 4:
        raise RuntimeError(
            "Expected four S4M4 modal anchors, "
            f"found {len(modal_4_mach)}"
        )

    if anchors_mach[
        "anchor_rank"
    ].tolist() != [1, 2, 3, 4]:
        raise RuntimeError(
            "Unexpected spectral anchor ranks"
        )

    if modal_2_mach[
        "anchor_rank"
    ].tolist() != [1, 4]:
        raise RuntimeError(
            "Unexpected S4M2 anchor ranks"
        )

    if modal_4_mach[
        "anchor_rank"
    ].tolist() != [1, 2, 3, 4]:
        raise RuntimeError(
            "Unexpected S4M4 anchor ranks"
        )

    tag = args.tag

    spectral_path = (
        data_dir / f"spectral_full_{tag}.csv"
    )

    spectral_anchor_path = (
        data_dir
        / f"spectral_anchors_S4_{tag}.csv"
    )

    modal_2_manifest_path = (
        data_dir
        / f"modal_manifest_S4M2_{tag}.csv"
    )

    modal_4_manifest_path = (
        data_dir
        / f"modal_manifest_S4M4_{tag}.csv"
    )

    modal_2_bank_path = (
        data_dir
        / f"modal_bank_S4M2_{tag}.npz"
    )

    modal_4_bank_path = (
        data_dir
        / f"modal_bank_S4M4_{tag}.npz"
    )

    spectral_mach.to_csv(
        spectral_path,
        index=False,
    )

    anchors_mach.to_csv(
        spectral_anchor_path,
        index=False,
    )

    modal_2_mach.to_csv(
        modal_2_manifest_path,
        index=False,
    )

    modal_4_mach.to_csv(
        modal_4_manifest_path,
        index=False,
    )

    alpha_min = float(
        spectral_mach["alpha"].min()
    )

    alpha_max = float(
        spectral_mach["alpha"].max()
    )

    cr_data_min = float(
        min(
            spectral_mach["cr"].min(),
            anchors_mach["cr"].min(),
        )
    )

    cr_data_max = float(
        max(
            spectral_mach["cr"].max(),
            anchors_mach["cr"].max(),
        )
    )

    cr_span = max(
        cr_data_max - cr_data_min,
        0.05,
    )

    cr_margin = max(
        0.10,
        0.50 * cr_span,
    )

    cr_min = cr_data_min - cr_margin
    cr_max = cr_data_max + cr_margin

    ci_peak = float(
        max(
            spectral_mach["ci"].max(),
            anchors_mach["ci"].max(),
        )
    )

    ci_max = max(
        0.15,
        1.50 * ci_peak + 0.01,
    )

    alpha_split = 0.5 * (
        float(anchors_mach.iloc[1]["alpha"])
        + float(
            anchors_mach.iloc[2]["alpha"]
        )
    )

    alpha_gate_width = max(
        0.01,
        0.08 * (alpha_max - alpha_min),
    )

    model_config = {
        "xi_max": 0.985,
        "mapping_scale": 3.0,
        "spectral_width": 96,
        "spectral_depth": 3,
        "modal_width": 256,
        "modal_depth": 7,
        "n_frequencies": 12,
        "mode_experts": 2,
        "alpha_split": alpha_split,
        "alpha_gate_width": (
            alpha_gate_width
        ),
        "cr_min": cr_min,
        "cr_max": cr_max,
        "ci_floor": 1.0e-6,
        "ci_max": ci_max,
    }

    loss_weights = {
        "riccati_kappa": 1.0,
        "riccati_q": 1.0,
        "log_amp_compatibility": 5.0,
        "boundary_kappa": 10.0,
        "boundary_q": 10.0,
        "spectral": 50.0,
        "modal_kappa": 1.0,
        "modal_q": 1.0,
        "modal_log_amp": 1.0,
    }

    common = {
        "experiment_family": (
            f"local_supersonic_{tag}_sparse_v1"
        ),
        "Mach": float(args.mach),
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "spectral_reference": repo_relative(
            spectral_path
        ),
        "spectral_anchor_file": repo_relative(
            spectral_anchor_path
        ),
        "seed": 12345,
        "model": model_config,
        "loss_weights": loss_weights,
        "modal_representation": [
            "kappa",
            "q",
            "logabs_p_center_gauge",
        ],
        "predict_phase": False,
    }

    experiments = {
        "P0": {
            "use_spectral_supervision": False,
            "use_modal_supervision": False,
            "modal_anchor_file": None,
        },
        "S4": {
            "use_spectral_supervision": True,
            "use_modal_supervision": False,
            "modal_anchor_file": None,
        },
        "S4M2": {
            "use_spectral_supervision": True,
            "use_modal_supervision": True,
            "modal_anchor_file": (
                repo_relative(
                    modal_2_bank_path
                )
            ),
        },
        "S4M4": {
            "use_spectral_supervision": True,
            "use_modal_supervision": True,
            "modal_anchor_file": (
                repo_relative(
                    modal_4_bank_path
                )
            ),
        },
    }

    for experiment, overrides in (
        experiments.items()
    ):
        config = {
            **common,
            **overrides,
            "experiment": experiment,
            "output_dir": repo_relative(
                pilot / "runs" / experiment
            ),
        }

        (
            config_dir
            / f"{experiment}.json"
        ).write_text(
            json.dumps(
                config,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    report = {
        "Mach": float(args.mach),
        "tag": tag,
        "n_spectral_reference": int(
            len(spectral_mach)
        ),
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "S4_alphas": (
            anchors_mach[
                "alpha"
            ].astype(float).tolist()
        ),
        "S4_cr": (
            anchors_mach[
                "cr"
            ].astype(float).tolist()
        ),
        "S4_ci": (
            anchors_mach[
                "ci"
            ].astype(float).tolist()
        ),
        "S4M2_alphas": (
            modal_2_mach[
                "alpha"
            ].astype(float).tolist()
        ),
        "S4M4_alphas": (
            modal_4_mach[
                "alpha"
            ].astype(float).tolist()
        ),
        "model": model_config,
    }

    (
        data_dir / "pilot_data_report.json"
    ).write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print()
    print("pilot:", repo_relative(pilot))


if __name__ == "__main__":
    main()
