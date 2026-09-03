from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[3]

TRAINER_PATH = (
    REPO
    / "code/scripts/training/"
      "train_global_supersonic_kappa_q_logamp_continuousM.py"
)

PHYSICS_PATH = (
    REPO
    / "code/src/physics/"
      "kh_supersonic_riccati_residual.py"
)

DEFAULT_V64 = (
    REPO
    / "article/supplementary/data/reviewer_multiseed/"
      "table_N76_production_shooting_validation_64.csv"
)

DEFAULT_T401 = (
    REPO
    / "article/tables/"
      "table_test_reference_401_SEALED.csv"
)

DEFAULT_OUTPUT = (
    REPO
    / "experiments/physics_residual_audit/"
      "N76_independent"
)

MODEL_ROOT = (
    REPO
    / "models_saved/atlas/N76"
)


# ---------------------------------------------------------------------
# Production N76 checkpoint set.
#
# These are the checkpoints used by the retained N76 production atlas.
# Phase 1 uses "joint" only.
#
# The modal-stage map is retained here for the later Stage-2 vs Stage-3
# audit, so that exactly the same code can be reused without changing the
# evaluation protocol.
# ---------------------------------------------------------------------

CHECKPOINTS: dict[str, dict[str, str]] = {
    "joint": {
        "C00": "best_joint_checkpoint_33647c0c65.pt",
        "C01": "best_joint_checkpoint_bd8dd50299.pt",
        "C02": "best_joint_checkpoint_be0e4833f5.pt",
        "C10": "best_joint_checkpoint_06bde09f2f.pt",
        "C11": "best_joint_checkpoint_b7ea53532f.pt",
        "C12": "best_joint_checkpoint_d4f26e5b75.pt",
        "C20": "best_joint_checkpoint_b236e307a9.pt",
        "C21": "best_joint_checkpoint_d89a36cc31.pt",
        "C22": "best_joint_checkpoint_4f0defe56d.pt",
        "C30": "best_joint_checkpoint_2d630591a9.pt",
        "C31": "best_joint_checkpoint_eafc27be12.pt",
        "C32": "best_joint_checkpoint_cd639cf2a1.pt",
    },
    "modal": {
        "C00": "best_modal_spectral_fixed_spectrum_checkpoint_5a837f7a71.pt",
        "C01": "best_modal_spectral_fixed_spectrum_checkpoint_ddd8f57373.pt",
        "C02": "best_modal_spectral_fixed_spectrum_checkpoint_6b3bb11f83.pt",
        "C10": "best_modal_spectral_fixed_spectrum_checkpoint_971ac1b272.pt",
        "C11": "best_modal_spectral_fixed_spectrum_checkpoint_e4262f203b.pt",
        "C12": "best_modal_spectral_fixed_spectrum_checkpoint_57a6789b3f.pt",
        "C20": "best_modal_spectral_fixed_spectrum_checkpoint_7c7b3b47b6.pt",
        "C21": "best_modal_spectral_fixed_spectrum_checkpoint_98fcb4be82.pt",
        "C22": "best_modal_spectral_fixed_spectrum_checkpoint_0eb1b4e8ff.pt",
        "C30": "best_modal_spectral_fixed_spectrum_checkpoint_283e527dd2.pt",
        "C31": "best_modal_spectral_fixed_spectrum_checkpoint_7426609015.pt",
        "C32": "best_modal_spectral_fixed_spectrum_checkpoint_5413fac496.pt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independent post-training physics-residual audit for the "
            "retained N76 supersonic KH atlas."
        )
    )

    parser.add_argument(
        "--v64",
        type=Path,
        default=DEFAULT_V64,
    )

    parser.add_argument(
        "--t401",
        type=Path,
        default=DEFAULT_T401,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["joint", "modal"],
        default=["joint"],
        help=(
            "Phase 1 should use '--stages joint'. "
            "The modal option is reserved for the later "
            "Stage-2 vs Stage-3 audit."
        ),
    )

    parser.add_argument(
        "--n-xi",
        type=int,
        default=256,
        help="Independent interior xi samples per (Mach, alpha).",
    )

    parser.add_argument(
        "--t401-limit",
        type=int,
        default=100,
        help=(
            "Number of T401 coordinates, sampled reproducibly "
            "and approximately stratified by chart. "
            "Use 401 for the full sealed test set."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260827,
        help="Independent audit seed.",
    )

    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import module from {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def tensor_scalar(value: Any) -> float:
    if torch.is_tensor(value):
        arr = (
            value
            .detach()
            .cpu()
            .reshape(-1)
        )

        if arr.numel() == 0:
            raise RuntimeError(
                "Cannot convert empty tensor to scalar."
            )

        return float(arr[0].item())

    return float(value)


def model_dtype(model) -> torch.dtype:
    for parameter in model.parameters():
        return parameter.dtype

    for buffer in model.buffers():
        if torch.is_floating_point(buffer):
            return buffer.dtype

    return torch.float32


def set_mach_context(
    model,
    mach: float,
    n: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """
    The continuous-M atlas can store Mach through set_mach_context().
    Training uses this context before modal/spectral evaluation.

    The canonical expected shape is (N, 1). Fallbacks are provided only
    for compatibility with older checkpoint/model implementations.
    """

    if not hasattr(model, "set_mach_context"):
        return

    candidates = [
        torch.full(
            (n, 1),
            float(mach),
            dtype=dtype,
            device=device,
        ),
        torch.full(
            (n,),
            float(mach),
            dtype=dtype,
            device=device,
        ),
        torch.tensor(
            float(mach),
            dtype=dtype,
            device=device,
        ),
    ]

    errors = []

    for candidate in candidates:
        try:
            model.set_mach_context(candidate)
            return
        except (TypeError, RuntimeError, ValueError) as exc:
            errors.append(
                f"{tuple(candidate.shape)}: {exc}"
            )

    raise RuntimeError(
        "Could not set Mach context. Attempts:\n"
        + "\n".join(errors)
    )


def get_xi_boundary(
    model,
    config: dict[str, Any],
) -> float:
    """
    Recover the exact xi support stored by the trained model/config.
    No hard-coded fallback is allowed in this audit.
    """

    if hasattr(model, "xi_max"):
        value = tensor_scalar(
            getattr(model, "xi_max")
        )

        if 0.0 < value < 1.0:
            return value

    if "xi_max" in config:
        value = float(config["xi_max"])

        if 0.0 < value < 1.0:
            return value

    state = model.state_dict()

    if "xi_max" in state:
        value = tensor_scalar(state["xi_max"])

        if 0.0 < value < 1.0:
            return value

    raise RuntimeError(
        "Cannot recover xi_max from model/config."
    )


def get_mapping_scale(model) -> float:
    if not hasattr(model, "get_mapping_scale"):
        raise RuntimeError(
            "Model has no get_mapping_scale()."
        )

    value = model.get_mapping_scale()
    return tensor_scalar(value)


def normalize_coordinate_table(
    path: Path,
    dataset: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {"Mach", "alpha"}

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{path}: missing columns {sorted(missing)}"
        )

    chart_col = None

    for candidate in [
        "atlas_chart",
        "primary_chart",
        "chart",
    ]:
        if candidate in df.columns:
            chart_col = candidate
            break

    if chart_col is None:
        raise RuntimeError(
            f"{path}: no chart column found."
        )

    id_col = None

    for candidate in [
        "test_id",
        "validation_id",
        "benchmark_id",
        "id",
    ]:
        if candidate in df.columns:
            id_col = candidate
            break

    result = pd.DataFrame(
        {
            "dataset": dataset,
            "source_id": (
                df[id_col].astype(str)
                if id_col is not None
                else np.arange(len(df)).astype(str)
            ),
            "chart": df[chart_col].astype(str),
            "Mach": pd.to_numeric(
                df["Mach"],
                errors="raise",
            ),
            "alpha": pd.to_numeric(
                df["alpha"],
                errors="raise",
            ),
        }
    )

    # Keep reference c_i if available, only for descriptive downstream
    # stratification. It is NEVER used in the residual computation.
    for candidate in [
        "ci",
        "ci_reference",
        "reference_ci",
        "ci_ref",
    ]:
        if candidate in df.columns:
            result["reference_ci"] = pd.to_numeric(
                df[candidate],
                errors="coerce",
            )
            break

    return result


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """
    Approximately proportional sampling by primary chart, with at least
    one point per chart when the requested budget permits it.
    """

    if n <= 0:
        return df.iloc[0:0].copy()

    if n >= len(df):
        return df.copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)

    groups = {
        chart: group.copy()
        for chart, group in df.groupby(
            "chart",
            sort=True,
        )
    }

    charts = sorted(groups)

    if n < len(charts):
        raise ValueError(
            f"t401-limit={n} is smaller than "
            f"the number of charts={len(charts)}."
        )

    total = len(df)

    raw = {
        chart: n * len(groups[chart]) / total
        for chart in charts
    }

    allocation = {
        chart: max(
            1,
            int(math.floor(raw[chart])),
        )
        for chart in charts
    }

    # Never allocate more than available.
    for chart in charts:
        allocation[chart] = min(
            allocation[chart],
            len(groups[chart]),
        )

    current = sum(allocation.values())

    # Add remaining points according to largest fractional remainder.
    if current < n:
        ranking = sorted(
            charts,
            key=lambda c: (
                raw[c] - math.floor(raw[c]),
                len(groups[c]),
                c,
            ),
            reverse=True,
        )

        while current < n:
            changed = False

            for chart in ranking:
                if allocation[chart] < len(groups[chart]):
                    allocation[chart] += 1
                    current += 1
                    changed = True

                    if current == n:
                        break

            if not changed:
                break

    # Remove excess while preserving >= 1/chart.
    if current > n:
        ranking = sorted(
            charts,
            key=lambda c: (
                raw[c] - math.floor(raw[c]),
                len(groups[c]),
                c,
            ),
        )

        while current > n:
            changed = False

            for chart in ranking:
                if allocation[chart] > 1:
                    allocation[chart] -= 1
                    current -= 1
                    changed = True

                    if current == n:
                        break

            if not changed:
                break

    selected = []

    for chart in charts:
        group = groups[chart]

        indices = rng.choice(
            len(group),
            size=allocation[chart],
            replace=False,
        )

        selected.append(
            group.iloc[
                np.sort(indices)
            ]
        )

    return (
        pd.concat(
            selected,
            ignore_index=True,
        )
        .sort_values(
            ["chart", "Mach", "alpha"]
        )
        .reset_index(drop=True)
    )


def coordinate_seed(
    base_seed: int,
    *,
    stage: str,
    dataset: str,
    source_id: str,
    chart: str,
) -> int:
    # IMPORTANT: the audit coordinates must be identical across
    # training stages so that Stage 2 -> Stage 3 comparisons are paired.
    # The stage is deliberately excluded from the sampling seed.
    payload = (
        f"{base_seed}|{dataset}|"
        f"{source_id}|{chart}"
    ).encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    offset = int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )

    return (
        int(base_seed)
        + offset
    ) % (2**32)


def summarize_array(
    values: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return {
            "median": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def load_model(
    *,
    trainer,
    stage: str,
    chart: str,
    device: torch.device,
):
    filename = CHECKPOINTS[stage][chart]

    path = (
        MODEL_ROOT
        / chart
        / filename
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if (
        not isinstance(checkpoint, dict)
        or "config" not in checkpoint
        or "model_state_dict" not in checkpoint
    ):
        raise RuntimeError(
            f"Invalid checkpoint: {path}"
        )

    config = checkpoint["config"]

    if not isinstance(config, dict):
        raise RuntimeError(
            f"{path}: config is not a dict."
        )

    model = trainer.build_model(config)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    # We need derivatives with respect to xi, not parameter gradients.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return (
        model,
        config,
        path,
        checkpoint,
    )


def boundary_diagnostics(
    *,
    physics,
    model,
    mach: float,
    alpha_value: float,
    xi_boundary: float,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, float]:

    alpha = torch.tensor(
        [[float(alpha_value)]],
        dtype=dtype,
        device=device,
    )

    set_mach_context(
        model,
        mach,
        1,
        dtype=dtype,
        device=device,
    )

    with torch.no_grad():
        official = (
            physics.riccati_boundary_losses(
                model,
                alpha,
                mach=float(mach),
                xi_boundary=float(
                    xi_boundary
                ),
            )
        )

        # Reconstruct the individual errors using the same official
        # asymptotic gamma helper, solely to report max absolute BC error.
        xi_left = torch.full_like(
            alpha,
            -float(xi_boundary),
        )

        xi_right = torch.full_like(
            alpha,
            float(xi_boundary),
        )

        pred_left = model(
            xi_left,
            alpha,
        )

        pred_right = model(
            xi_right,
            alpha,
        )

        cr, ci = model.get_spectrum(
            alpha
        )

        gamma_left, gamma_right = (
            physics.asymptotic_riccati_gammas(
                alpha,
                float(mach),
                cr,
                ci,
            )
        )

        kappa_errors = torch.cat(
            [
                (
                    pred_left[:, 0:1]
                    - gamma_left.real
                ).reshape(-1),
                (
                    pred_right[:, 0:1]
                    - gamma_right.real
                ).reshape(-1),
            ]
        )

        q_errors = torch.cat(
            [
                (
                    pred_left[:, 1:2]
                    - gamma_left.imag
                ).reshape(-1),
                (
                    pred_right[:, 1:2]
                    - gamma_right.imag
                ).reshape(-1),
            ]
        )

    # The official BC loss is mean(left^2) + mean(right^2).
    # For one coordinate this is the sum of two squared errors.
    # Dividing by two gives the combined left/right MSE.
    bc_kappa_rmse = math.sqrt(
        max(
            tensor_scalar(
                official[
                    "loss_bc_kappa"
                ]
            )
            / 2.0,
            0.0,
        )
    )

    bc_q_rmse = math.sqrt(
        max(
            tensor_scalar(
                official[
                    "loss_bc_phase_gradient"
                ]
            )
            / 2.0,
            0.0,
        )
    )

    return {
        "bc_kappa_rmse": (
            bc_kappa_rmse
        ),
        "bc_q_rmse": (
            bc_q_rmse
        ),
        "bc_kappa_max_abs": float(
            kappa_errors
            .abs()
            .max()
            .cpu()
            .item()
        ),
        "bc_q_max_abs": float(
            q_errors
            .abs()
            .max()
            .cpu()
            .item()
        ),
    }


def evaluate_coordinate(
    *,
    physics,
    model,
    config: dict[str, Any],
    stage: str,
    dataset: str,
    source_id: str,
    chart: str,
    mach: float,
    alpha_value: float,
    n_xi: int,
    audit_seed: int,
    device: torch.device,
):
    dtype = model_dtype(model)

    # Set a one-point context first so that mapping properties depending
    # on Mach are evaluated at the target coordinate.
    set_mach_context(
        model,
        mach,
        1,
        dtype=dtype,
        device=device,
    )

    xi_boundary = get_xi_boundary(
        model,
        config,
    )

    mapping_scale = get_mapping_scale(
        model
    )

    rng = np.random.default_rng(
        coordinate_seed(
            audit_seed,
            stage=stage,
            dataset=dataset,
            source_id=source_id,
            chart=chart,
        )
    )

    # Draw fresh continuous points strictly inside the training support.
    # Probability of reproducing a training collocation coordinate is zero
    # for continuous sampling; the tiny margin avoids evaluating exactly
    # on the BC.
    margin = max(
        1.0e-6,
        1.0e-5 * xi_boundary,
    )

    xi_np = rng.uniform(
        -xi_boundary + margin,
        xi_boundary - margin,
        size=(n_xi, 1),
    )

    xi = torch.tensor(
        xi_np,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )

    alpha = torch.full(
        (n_xi, 1),
        float(alpha_value),
        dtype=dtype,
        device=device,
    )

    set_mach_context(
        model,
        mach,
        n_xi,
        dtype=dtype,
        device=device,
    )

    residual = (
        physics.riccati_regularized_residuals(
            model,
            xi,
            alpha,
            mach=float(mach),
        )
    )

    r_kappa = (
        residual["residual_kappa"]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    r_q = (
        residual[
            "residual_phase_gradient"
        ]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    r_logamp = (
        residual["residual_log_amp"]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    y = (
        residual["y"]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    cr = (
        residual["cr"]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    ci = (
        residual["ci"]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    # These are exactly sqrt of the normalized MSE losses used by the
    # production residual module.
    norm_rms_kappa = math.sqrt(
        max(
            tensor_scalar(
                residual["loss_kappa"]
            ),
            0.0,
        )
    )

    norm_rms_q = math.sqrt(
        max(
            tensor_scalar(
                residual[
                    "loss_phase_gradient"
                ]
            ),
            0.0,
        )
    )

    norm_rms_logamp = math.sqrt(
        max(
            tensor_scalar(
                residual["loss_log_amp"]
            ),
            0.0,
        )
    )

    bc = boundary_diagnostics(
        physics=physics,
        model=model,
        mach=mach,
        alpha_value=alpha_value,
        xi_boundary=xi_boundary,
        dtype=dtype,
        device=device,
    )

    abs_kappa = np.abs(r_kappa)
    abs_q = np.abs(r_q)
    abs_logamp = np.abs(r_logamp)

    kappa_stats = summarize_array(
        abs_kappa
    )

    q_stats = summarize_array(
        abs_q
    )

    logamp_stats = summarize_array(
        abs_logamp
    )

    coordinate_row = {
        "stage": stage,
        "dataset": dataset,
        "source_id": source_id,
        "chart": chart,
        "Mach": float(mach),
        "alpha": float(alpha_value),
        "n_xi": int(n_xi),
        "xi_boundary": float(
            xi_boundary
        ),
        "mapping_scale": float(
            mapping_scale
        ),
        "cr_pred": float(
            np.median(cr)
        ),
        "ci_pred": float(
            np.median(ci)
        ),

        "r_kappa_abs_median":
            kappa_stats["median"],
        "r_kappa_abs_p90":
            kappa_stats["p90"],
        "r_kappa_abs_p95":
            kappa_stats["p95"],
        "r_kappa_abs_max":
            kappa_stats["max"],

        "r_q_abs_median":
            q_stats["median"],
        "r_q_abs_p90":
            q_stats["p90"],
        "r_q_abs_p95":
            q_stats["p95"],
        "r_q_abs_max":
            q_stats["max"],

        "r_logamp_abs_median":
            logamp_stats["median"],
        "r_logamp_abs_p90":
            logamp_stats["p90"],
        "r_logamp_abs_p95":
            logamp_stats["p95"],
        "r_logamp_abs_max":
            logamp_stats["max"],

        "r_kappa_normalized_rms":
            norm_rms_kappa,
        "r_q_normalized_rms":
            norm_rms_q,
        "r_logamp_normalized_rms":
            norm_rms_logamp,

        **bc,
    }

    pointwise = pd.DataFrame(
        {
            "stage": stage,
            "dataset": dataset,
            "source_id": source_id,
            "chart": chart,
            "Mach": float(mach),
            "alpha": float(
                alpha_value
            ),
            "sample_id": np.arange(
                n_xi
            ),
            "xi": xi_np.reshape(-1),
            "y": y,
            "r_kappa_abs":
                abs_kappa,
            "r_q_abs":
                abs_q,
            "r_logamp_abs":
                abs_logamp,
        }
    )

    return (
        coordinate_row,
        pointwise,
    )


def make_long_summary(
    df: pd.DataFrame,
    *,
    group_columns: list[str],
    metrics: list[str],
    source: str,
) -> pd.DataFrame:
    rows = []

    grouped = (
        df.groupby(
            group_columns,
            dropna=False,
            sort=True,
        )
        if group_columns
        else [((), df)]
    )

    for key, group in grouped:
        if group_columns:
            if not isinstance(
                key,
                tuple,
            ):
                key = (key,)

            base = dict(
                zip(
                    group_columns,
                    key,
                )
            )
        else:
            base = {}

        for metric in metrics:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).to_numpy(float)

            stats = summarize_array(
                values
            )

            rows.append(
                {
                    **base,
                    "source": source,
                    "metric": metric,
                    "n": int(
                        np.isfinite(
                            values
                        ).sum()
                    ),
                    **stats,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    if args.n_xi < 8:
        raise ValueError(
            "--n-xi must be >= 8."
        )

    if args.device != "cpu":
        print(
            "WARNING: this audit was designed "
            "to run locally on CPU.",
            file=sys.stderr,
        )

    device = torch.device(
        args.device
    )

    for path in [
        TRAINER_PATH,
        PHYSICS_PATH,
        args.v64,
        args.t401,
    ]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer = load_module(
        "n76_physics_audit_trainer",
        TRAINER_PATH,
    )

    physics = load_module(
        "n76_physics_audit_equations",
        PHYSICS_PATH,
    )

    v64 = normalize_coordinate_table(
        args.v64,
        "V64",
    )

    if len(v64) != 64:
        raise RuntimeError(
            f"Expected 64 V64 coordinates, "
            f"got {len(v64)}."
        )

    t401_full = normalize_coordinate_table(
        args.t401,
        "T401",
    )

    if len(t401_full) != 401:
        raise RuntimeError(
            f"Expected 401 T401 coordinates, "
            f"got {len(t401_full)}."
        )

    t401 = stratified_sample(
        t401_full,
        int(args.t401_limit),
        int(args.seed),
    )

    coordinates = pd.concat(
        [
            v64,
            t401,
        ],
        ignore_index=True,
    )

    valid_charts = set(
        CHECKPOINTS["joint"]
    )

    unknown = (
        set(coordinates["chart"])
        - valid_charts
    )

    if unknown:
        raise RuntimeError(
            "Unknown charts in evaluation "
            f"datasets: {sorted(unknown)}"
        )

    coordinate_rows = []
    pointwise_frames = []

    checkpoint_metadata = {}

    for stage in args.stages:
        print()
        print("=" * 78)
        print(
            f"STAGE: {stage}"
        )
        print("=" * 78)

        for chart in sorted(
            coordinates["chart"].unique()
        ):
            chart_coordinates = (
                coordinates[
                    coordinates[
                        "chart"
                    ] == chart
                ]
                .copy()
                .reset_index(drop=True)
            )

            model, config, checkpoint_path, checkpoint = (
                load_model(
                    trainer=trainer,
                    stage=stage,
                    chart=chart,
                    device=device,
                )
            )

            checkpoint_metadata[
                f"{stage}:{chart}"
            ] = {
                "path": str(
                    checkpoint_path
                    .relative_to(REPO)
                ),
                "sha256": sha256(
                    checkpoint_path
                ),
                "checkpoint_stage":
                    checkpoint.get(
                        "stage"
                    ),
                "checkpoint_step":
                    checkpoint.get(
                        "step"
                    ),
                "checkpoint_loss":
                    checkpoint.get(
                        "loss"
                    ),
                "config_seed":
                    config.get(
                        "seed"
                    ),
                "mach_min":
                    config.get(
                        "mach_min"
                    ),
                "mach_max":
                    config.get(
                        "mach_max"
                    ),
                "alpha_min":
                    config.get(
                        "alpha_min"
                    ),
                "alpha_max":
                    config.get(
                        "alpha_max"
                    ),
                "loss_weights":
                    config.get(
                        "loss_weights"
                    ),
            }

            print(
                f"{chart}: "
                f"{checkpoint_path.name} "
                f"| n_coordinates="
                f"{len(chart_coordinates)}",
                flush=True,
            )

            for _, row in (
                chart_coordinates.iterrows()
            ):
                coordinate_row, pointwise = (
                    evaluate_coordinate(
                        physics=physics,
                        model=model,
                        config=config,
                        stage=stage,
                        dataset=str(
                            row["dataset"]
                        ),
                        source_id=str(
                            row[
                                "source_id"
                            ]
                        ),
                        chart=chart,
                        mach=float(
                            row["Mach"]
                        ),
                        alpha_value=float(
                            row["alpha"]
                        ),
                        n_xi=int(
                            args.n_xi
                        ),
                        audit_seed=int(
                            args.seed
                        ),
                        device=device,
                    )
                )

                if (
                    "reference_ci"
                    in row.index
                    and pd.notna(
                        row[
                            "reference_ci"
                        ]
                    )
                ):
                    coordinate_row[
                        "reference_ci"
                    ] = float(
                        row[
                            "reference_ci"
                        ]
                    )

                coordinate_rows.append(
                    coordinate_row
                )

                pointwise_frames.append(
                    pointwise
                )

            del model
            del checkpoint

    coordinate_df = pd.DataFrame(
        coordinate_rows
    )

    pointwise_df = pd.concat(
        pointwise_frames,
        ignore_index=True,
    )

    point_metrics = [
        "r_kappa_abs",
        "r_q_abs",
        "r_logamp_abs",
    ]

    coordinate_metrics = [
        "r_kappa_normalized_rms",
        "r_q_normalized_rms",
        "r_logamp_normalized_rms",
        "bc_kappa_rmse",
        "bc_q_rmse",
        "bc_kappa_max_abs",
        "bc_q_max_abs",
    ]

    global_point = make_long_summary(
        pointwise_df,
        group_columns=[
            "stage",
            "dataset",
        ],
        metrics=point_metrics,
        source="independent_xi_pointwise",
    )

    global_coordinate = (
        make_long_summary(
            coordinate_df,
            group_columns=[
                "stage",
                "dataset",
            ],
            metrics=coordinate_metrics,
            source="coordinate_level",
        )
    )

    # Additional combined V64+T401 summary.
    combined_point = (
        make_long_summary(
            pointwise_df,
            group_columns=[
                "stage",
            ],
            metrics=point_metrics,
            source=(
                "independent_xi_"
                "pointwise_combined"
            ),
        )
    )

    combined_coordinate = (
        make_long_summary(
            coordinate_df,
            group_columns=[
                "stage",
            ],
            metrics=coordinate_metrics,
            source=(
                "coordinate_level_"
                "combined"
            ),
        )
    )

    global_summary = pd.concat(
        [
            global_point,
            global_coordinate,
            combined_point,
            combined_coordinate,
        ],
        ignore_index=True,
        sort=False,
    )

    chart_point = make_long_summary(
        pointwise_df,
        group_columns=[
            "stage",
            "dataset",
            "chart",
        ],
        metrics=point_metrics,
        source="independent_xi_pointwise",
    )

    chart_coordinate = (
        make_long_summary(
            coordinate_df,
            group_columns=[
                "stage",
                "dataset",
                "chart",
            ],
            metrics=coordinate_metrics,
            source="coordinate_level",
        )
    )

    by_chart = pd.concat(
        [
            chart_point,
            chart_coordinate,
        ],
        ignore_index=True,
        sort=False,
    )

    # Worst coordinates are reported separately for each physically
    # meaningful coordinate-level metric.
    worst_frames = []

    for metric in coordinate_metrics:
        tmp = (
            coordinate_df[
                [
                    "stage",
                    "dataset",
                    "source_id",
                    "chart",
                    "Mach",
                    "alpha",
                    metric,
                ]
            ]
            .dropna(
                subset=[metric]
            )
            .sort_values(
                metric,
                ascending=False,
            )
            .head(20)
            .copy()
        )

        tmp.insert(
            0,
            "ranking_metric",
            metric,
        )

        tmp = tmp.rename(
            columns={
                metric:
                    "ranking_value"
            }
        )

        worst_frames.append(tmp)

    worst = pd.concat(
        worst_frames,
        ignore_index=True,
    )

    # Save the actual T401 subset so that the audit can be reproduced
    # exactly without relying on the sampling implementation.
    t401.to_csv(
        output_dir
        / "T401_stratified_subset.csv",
        index=False,
    )

    coordinate_df.to_csv(
        output_dir
        / "physics_residual_coordinate_summary.csv",
        index=False,
    )

    pointwise_df.to_csv(
        output_dir
        / "physics_residual_pointwise.csv",
        index=False,
    )

    global_summary.to_csv(
        output_dir
        / "physics_residual_global_summary.csv",
        index=False,
    )

    by_chart.to_csv(
        output_dir
        / "physics_residual_by_chart.csv",
        index=False,
    )

    worst.to_csv(
        output_dir
        / "physics_residual_worst_coordinates.csv",
        index=False,
    )

    metadata = {
        "audit": (
            "N76 independent post-training "
            "physics-residual audit"
        ),
        "audit_seed": int(
            args.seed
        ),
        "device": str(device),
        "stages": list(
            args.stages
        ),
        "n_xi_per_coordinate": int(
            args.n_xi
        ),
        "v64_coordinates": int(
            len(v64)
        ),
        "t401_total_coordinates": int(
            len(t401_full)
        ),
        "t401_sampled_coordinates": int(
            len(t401)
        ),
        "sampling": {
            "parameter_coordinates": (
                "V64 + chart-stratified "
                "subset of sealed T401"
            ),
            "xi": (
                "fresh independent continuous "
                "uniform samples strictly inside "
                "[-xi_max, xi_max]"
            ),
        },
        "reference_data_usage": (
            "Reference eigenvalues are not used "
            "in residual or BC computation."
        ),
        "v64_path": str(
            args.v64.relative_to(REPO)
            if args.v64.is_relative_to(REPO)
            else args.v64
        ),
        "v64_sha256": sha256(
            args.v64
        ),
        "t401_path": str(
            args.t401.relative_to(REPO)
            if args.t401.is_relative_to(REPO)
            else args.t401
        ),
        "t401_sha256": sha256(
            args.t401
        ),
        "trainer_path": str(
            TRAINER_PATH.relative_to(
                REPO
            )
        ),
        "trainer_sha256": sha256(
            TRAINER_PATH
        ),
        "physics_path": str(
            PHYSICS_PATH.relative_to(
                REPO
            )
        ),
        "physics_sha256": sha256(
            PHYSICS_PATH
        ),
        "checkpoints":
            checkpoint_metadata,
    }

    (
        output_dir
        / "physics_residual_audit_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)
    print(
        f"coordinates: "
        f"{len(coordinate_df)}"
    )
    print(
        f"independent xi evaluations: "
        f"{len(pointwise_df)}"
    )
    print(
        f"output: {output_dir}"
    )

    print()
    print(
        global_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
