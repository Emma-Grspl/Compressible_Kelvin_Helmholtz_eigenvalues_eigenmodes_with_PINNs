from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO = Path.cwd()

PREDICTIONS = (
    REPO / 'assets/pinn_supersonic/csv/pinn_direct/validation/table_N76_validation_predictions_64_cf57c58769.csv'
)

RUN_ROOT = (
    REPO
    / "assets/pinn_supersonic/"
      "atlas2d_v1_continuousM/N76/runs"
)

MODE_BANK = (
    REPO / 'experiments/modal_reconstruction/support/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS/classical_supersonic_final_modes_long.csv.gz'
)

TRAINER = (
    REPO / 'code/scripts/training/train_global_supersonic_kappa_q_logamp_continuousM.py'
)

AUDIT = (
    REPO / 'code/scripts/audits/audit_local_supersonic_modal_dense.py'
)

OUTPUT_ROOT = (
    REPO
    / "assets/pinn_supersonic/"
      "atlas2d_v1_continuousM/N76/"
      "modal_validation"
)


def load_module(
    name: str,
    path: Path,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module

    spec.loader.exec_module(
        module
    )

    return module


def as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(
            value.detach().cpu().item()
        )

    return float(value)


def unpack_modal(
    prediction,
) -> torch.Tensor:

    if torch.is_tensor(prediction):
        result = prediction

    elif isinstance(
        prediction,
        (tuple, list),
    ):
        parts = []

        for part in prediction:
            if not torch.is_tensor(part):
                continue

            if part.ndim == 1:
                part = part[:, None]

            parts.append(part)

        if not parts:
            raise RuntimeError(
                "Model returned no tensor "
                "modal outputs."
            )

        result = torch.cat(
            parts,
            dim=1,
        )

    else:
        raise RuntimeError(
            "Unsupported model output type: "
            f"{type(prediction)}"
        )

    if result.ndim != 2:
        raise RuntimeError(
            f"Unexpected modal output shape "
            f"{tuple(result.shape)}"
        )

    if result.shape[1] < 3:
        raise RuntimeError(
            "Expected at least 3 modal "
            "outputs: kappa, q, log_amp."
        )

    return result[:, :3]


def model_forward(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
) -> torch.Tensor:

    params = inspect.signature(
        model.forward
    ).parameters

    # The global model may either receive Mach
    # explicitly or use set_mach_context().
    if "Mach" in params:
        prediction = model(
            xi,
            alpha,
            Mach=mach,
        )

    elif "mach" in params:
        prediction = model(
            xi,
            alpha,
            mach=mach,
        )

    else:
        prediction = model(
            xi,
            alpha,
        )

    return unpack_modal(
        prediction
    )


def mapping_scale(
    model,
    config: dict[str, Any],
    mach_tensor: torch.Tensor,
) -> float:

    if hasattr(
        model,
        "get_mapping_scale",
    ):
        try:
            value = (
                model.get_mapping_scale()
            )

            return as_float(value)

        except TypeError:
            value = (
                model.get_mapping_scale(
                    mach_tensor[:1]
                )
            )

            return as_float(value)

    if "mapping_scale" in config:
        return float(
            config["mapping_scale"]
        )

    raise RuntimeError(
        "Cannot determine mapping scale."
    )


def overlap(
    prediction: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> float:

    p = prediction[mask]
    r = reference[mask]

    denominator = (
        np.linalg.norm(p)
        * np.linalg.norm(r)
    )

    if denominator <= 1e-14:
        return float("nan")

    return float(
        abs(
            np.vdot(
                r,
                p,
            )
        )
        / denominator
    )


def aligned_relative_l2(
    prediction: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> float:

    p = prediction[mask]
    r = reference[mask]

    denominator = np.vdot(
        p,
        p,
    )

    if abs(denominator) <= 1e-20:
        return float("nan")

    # Least-squares complex amplitude+phase
    # alignment of PINN pressure to reference.
    scale = (
        np.vdot(
            p,
            r,
        )
        / denominator
    )

    aligned = (
        scale
        * p
    )

    ref_norm = np.linalg.norm(r)

    if ref_norm <= 1e-20:
        return float("nan")

    return float(
        np.linalg.norm(
            aligned - r
        )
        / ref_norm
    )


def normalized_rmse(
    prediction: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> float:

    p = prediction[mask]
    r = reference[mask]

    denom = max(
        float(
            np.sqrt(
                np.mean(r * r)
            )
        ),
        1e-12,
    )

    return float(
        np.sqrt(
            np.mean(
                (p - r) ** 2
            )
        )
        / denom
    )


def make_key(
    mach: float,
    alpha: float,
) -> str:
    return (
        f"{float(mach):.6f}|"
        f"{float(alpha):.8f}"
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    for path in [
        PREDICTIONS,
        MODE_BANK,
        TRAINER,
        AUDIT,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        args.device
    )

    trainer = load_module(
        "atlas_continuousM_trainer_modal_eval",
        TRAINER,
    )

    audit = load_module(
        "atlas_modal_audit_helpers",
        AUDIT,
    )

    predictions = pd.read_csv(
        PREDICTIONS
    )

    if len(predictions) != 64:
        raise RuntimeError(
            f"Expected 64 validation points, "
            f"got {len(predictions)}"
        )

    required_prediction_columns = {
        "Mach",
        "alpha",
        "atlas_chart",
        "cr",
        "ci",
        "cr_pred",
        "ci_pred",
    }

    missing = (
        required_prediction_columns
        - set(predictions.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing prediction columns: "
            f"{sorted(missing)}"
        )

    predictions = (
        predictions
        .sort_values(
            [
                "atlas_chart",
                "Mach",
                "alpha",
            ]
        )
        .reset_index(drop=True)
    )

    if args.limit is not None:
        predictions = (
            predictions
            .iloc[
                : int(args.limit)
            ]
            .copy()
        )

    print(
        "Loading classical mode bank...",
        flush=True,
    )

    bank = pd.read_csv(
        MODE_BANK,
        usecols=[
            "final_reference_id",
            "Mach",
            "alpha",
            "cr",
            "ci",
            "coordinate_index",
            "y",
            "kappa",
            "q",
            "p_real",
            "p_imag",
        ],
    )

    bank["_key"] = [
        make_key(m, a)
        for m, a in zip(
            bank["Mach"],
            bank["alpha"],
        )
    ]

    grouped_bank = {
        key: group.copy()
        for key, group
        in bank.groupby(
            "_key",
            sort=False,
        )
    }

    rows: list[
        dict[str, Any]
    ] = []

    current_chart = None
    model = None
    config = None

    for index, source in (
        predictions.iterrows()
    ):

        chart = str(
            source["atlas_chart"]
        )

        mach = float(
            source["Mach"]
        )

        alpha = float(
            source["alpha"]
        )

        if chart != current_chart:

            checkpoint_path = (
                RUN_ROOT
                / chart
                / "best_joint_checkpoint.pt"
            )

            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    checkpoint_path
                )

            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
            )

            if (
                not isinstance(
                    checkpoint,
                    dict,
                )
                or "model_state_dict"
                not in checkpoint
            ):
                raise RuntimeError(
                    f"Invalid checkpoint: "
                    f"{checkpoint_path}"
                )

            config = checkpoint.get(
                "config"
            )

            if not isinstance(
                config,
                dict,
            ):
                raise RuntimeError(
                    f"{chart}: checkpoint "
                    "does not contain config."
                )

            model = trainer.build_model(
                config
            )

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

            model.to(device)
            model.eval()

            current_chart = chart

            print(
                f"Loaded {chart}",
                flush=True,
            )

        assert model is not None
        assert config is not None

        if hasattr(
            model,
            "set_mach_context",
        ):
            model.set_mach_context(
                mach
            )

        key = make_key(
            mach,
            alpha,
        )

        if key not in grouped_bank:
            raise RuntimeError(
                "Reference mode absent for "
                f"M={mach}, alpha={alpha}"
            )

        ref = (
            grouped_bank[key]
            .sort_values(
                "coordinate_index"
            )
            .reset_index(drop=True)
        )

        y = ref[
            "y"
        ].to_numpy(
            dtype=float
        )

        order = np.argsort(y)

        y = y[order]

        kappa_ref = (
            ref["kappa"]
            .to_numpy(dtype=float)
            [order]
        )

        q_ref = (
            ref["q"]
            .to_numpy(dtype=float)
            [order]
        )

        p_ref = (
            ref["p_real"]
            .to_numpy(dtype=float)
            [order]
            + 1j
            * ref["p_imag"]
            .to_numpy(dtype=float)
            [order]
        )

        center_index = int(
            np.argmin(
                np.abs(y)
            )
        )

        center_pressure = (
            p_ref[
                center_index
            ]
        )

        if (
            not np.isfinite(
                center_pressure.real
            )
            or not np.isfinite(
                center_pressure.imag
            )
            or abs(
                center_pressure
            ) <= 1e-14
        ):
            raise RuntimeError(
                "Invalid reference pressure "
                f"at center for {key}"
            )

        p_ref_gauge = (
            p_ref
            / center_pressure
        )

        log_amp_ref = np.log(
            np.maximum(
                np.abs(
                    p_ref_gauge
                ),
                1e-30,
            )
        )

        dtype = next(
            model.parameters()
        ).dtype

        y_tensor = torch.tensor(
            y[:, None],
            dtype=dtype,
            device=device,
        )

        alpha_tensor = torch.full_like(
            y_tensor,
            alpha,
        )

        mach_tensor = torch.full_like(
            y_tensor,
            mach,
        )

        scale = mapping_scale(
            model,
            config,
            mach_tensor,
        )

        xi_tensor = audit.y_to_xi(
            y_tensor,
            scale,
        )

        with torch.no_grad():

            modal = model_forward(
                model,
                xi_tensor,
                alpha_tensor,
                mach_tensor,
            )

        modal_np = (
            modal.detach()
            .cpu()
            .numpy()
        )

        kappa_pred = (
            modal_np[:, 0]
        )

        q_pred = (
            modal_np[:, 1]
        )

        log_amp_pred = (
            modal_np[:, 2]
        )

        phase_pred = (
            audit.integrate_phase(
                y,
                q_pred,
                center_index,
            )
        )

        p_pred = (
            np.exp(
                np.clip(
                    log_amp_pred,
                    -50.0,
                    20.0,
                )
            )
            * np.exp(
                1j
                * phase_pred
            )
        )

        finite = (
            np.isfinite(y)
            & np.isfinite(kappa_ref)
            & np.isfinite(q_ref)
            & np.isfinite(log_amp_ref)
            & np.isfinite(kappa_pred)
            & np.isfinite(q_pred)
            & np.isfinite(log_amp_pred)
            & np.isfinite(
                p_pred.real
            )
            & np.isfinite(
                p_pred.imag
            )
        )

        core20 = (
            finite
            & (
                np.abs(y)
                <= 20.0
            )
        )

        core40 = (
            finite
            & (
                np.abs(y)
                <= 40.0
            )
        )

        support40 = (
            core40
            & (
                np.abs(
                    p_ref_gauge
                )
                >= 1e-4
            )
        )

        for name, mask in [
            ("core20", core20),
            ("core40", core40),
            ("support40", support40),
        ]:
            if int(mask.sum()) < 16:
                raise RuntimeError(
                    f"{key}: insufficient "
                    f"points in {name}: "
                    f"{int(mask.sum())}"
                )

        ov20 = overlap(
            p_pred,
            p_ref_gauge,
            core20,
        )

        ov40 = overlap(
            p_pred,
            p_ref_gauge,
            core40,
        )

        ov_support = overlap(
            p_pred,
            p_ref_gauge,
            support40,
        )

        row = {
            "atlas_chart":
                chart,

            "Mach":
                mach,

            "alpha":
                alpha,

            "reference_id":
                str(
                    ref[
                        "final_reference_id"
                    ].iloc[0]
                ),

            "cr_reference":
                float(source["cr"]),

            "ci_reference":
                float(source["ci"]),

            "cr_pinn":
                float(
                    source["cr_pred"]
                ),

            "ci_pinn":
                float(
                    source["ci_pred"]
                ),

            "n_y":
                int(len(y)),

            "center_y":
                float(
                    y[
                        center_index
                    ]
                ),

            "mapping_scale":
                float(scale),

            "kappa_nrmse_core40":
                normalized_rmse(
                    kappa_pred,
                    kappa_ref,
                    core40,
                ),

            "q_nrmse_core40":
                normalized_rmse(
                    q_pred,
                    q_ref,
                    core40,
                ),

            "log_amp_rmse_support40":
                float(
                    np.sqrt(
                        np.mean(
                            (
                                log_amp_pred[
                                    support40
                                ]
                                - log_amp_ref[
                                    support40
                                ]
                            )
                            ** 2
                        )
                    )
                ),

            "pressure_overlap_core20":
                ov20,

            "pressure_overlap_defect_core20":
                1.0 - ov20,

            "pressure_overlap_core40":
                ov40,

            "pressure_overlap_defect_core40":
                1.0 - ov40,

            "pressure_overlap_support40":
                ov_support,

            "pressure_overlap_defect_support40":
                1.0 - ov_support,

            "pressure_rel_l2_aligned_core20":
                aligned_relative_l2(
                    p_pred,
                    p_ref_gauge,
                    core20,
                ),

            "pressure_rel_l2_aligned_core40":
                aligned_relative_l2(
                    p_pred,
                    p_ref_gauge,
                    core40,
                ),

            "pressure_rel_l2_aligned_support40":
                aligned_relative_l2(
                    p_pred,
                    p_ref_gauge,
                    support40,
                ),
        }

        rows.append(row)

        print(
            f"[{index+1:02d}/"
            f"{len(predictions):02d}] "
            f"{chart} "
            f"M={mach:.3f} "
            f"a={alpha:.3f} "
            f"overlap40="
            f"{ov40:.6f} "
            f"defect="
            f"{1.0-ov40:.3e} "
            f"L2="
            f"{row['pressure_rel_l2_aligned_core40']:.3e}",
            flush=True,
        )

    result = pd.DataFrame(
        rows
    )

    result_path = (
        OUTPUT_ROOT
        / "N76_modal_validation_64.csv"
    )

    result.to_csv(
        result_path,
        index=False,
    )

    metric_names = [
        "pressure_overlap_defect_core20",
        "pressure_overlap_defect_core40",
        "pressure_overlap_defect_support40",
        "pressure_rel_l2_aligned_core20",
        "pressure_rel_l2_aligned_core40",
        "pressure_rel_l2_aligned_support40",
        "kappa_nrmse_core40",
        "q_nrmse_core40",
        "log_amp_rmse_support40",
    ]

    summary: dict[
        str,
        Any,
    ] = {
        "n":
            int(len(result)),
    }

    for metric in metric_names:

        values = pd.to_numeric(
            result[metric],
            errors="coerce",
        )

        summary[
            f"{metric}_mean"
        ] = float(
            values.mean()
        )

        summary[
            f"{metric}_median"
        ] = float(
            values.median()
        )

        summary[
            f"{metric}_p95"
        ] = float(
            values.quantile(
                0.95
            )
        )

        summary[
            f"{metric}_max"
        ] = float(
            values.max()
        )

    summary_path = (
        OUTPUT_ROOT
        / "N76_modal_validation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    by_chart_rows = []

    for chart, group in (
        result.groupby(
            "atlas_chart"
        )
    ):
        by_chart_rows.append(
            {
                "atlas_chart":
                    chart,

                "n":
                    int(
                        len(group)
                    ),

                "overlap40_mean":
                    float(
                        group[
                            "pressure_overlap_core40"
                        ].mean()
                    ),

                "overlap40_min":
                    float(
                        group[
                            "pressure_overlap_core40"
                        ].min()
                    ),

                "defect40_mean":
                    float(
                        group[
                            "pressure_overlap_defect_core40"
                        ].mean()
                    ),

                "defect40_max":
                    float(
                        group[
                            "pressure_overlap_defect_core40"
                        ].max()
                    ),

                "aligned_l2_40_mean":
                    float(
                        group[
                            "pressure_rel_l2_aligned_core40"
                        ].mean()
                    ),

                "aligned_l2_40_max":
                    float(
                        group[
                            "pressure_rel_l2_aligned_core40"
                        ].max()
                    ),
            }
        )

    by_chart = pd.DataFrame(
        by_chart_rows
    )

    by_chart.to_csv(
        OUTPUT_ROOT
        / "N76_modal_validation_by_chart.csv",
        index=False,
    )

    worst = (
        result
        .sort_values(
            "pressure_overlap_defect_core40",
            ascending=False,
        )
    )

    worst.to_csv(
        OUTPUT_ROOT
        / "N76_modal_validation_worst.csv",
        index=False,
    )

    print()
    print(
        "=" * 100
    )
    print(
        "GLOBAL MODAL SUMMARY"
    )
    print(
        "=" * 100
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    target = result[
        np.isclose(
            result["Mach"],
            1.10,
            atol=1e-12,
            rtol=0.0,
        )
        & np.isclose(
            result["alpha"],
            0.09,
            atol=1e-12,
            rtol=0.0,
        )
    ]

    print()
    print(
        "=" * 100
    )
    print(
        "AMBIGUOUS POINT "
        "M=1.10 alpha=0.09"
    )
    print(
        "=" * 100
    )

    if len(target) == 1:
        print(
            target.to_string(
                index=False,
            )
        )
    else:
        print(
            "Point not present "
            "in current subset."
        )

    print()
    print(
        "WRITTEN:",
        result_path,
    )
    print(
        "WRITTEN:",
        summary_path,
    )


if __name__ == "__main__":
    main()
