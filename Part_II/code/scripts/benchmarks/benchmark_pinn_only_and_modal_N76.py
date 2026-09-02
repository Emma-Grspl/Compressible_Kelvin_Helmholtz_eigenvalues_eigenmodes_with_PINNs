from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.integrate import cumulative_trapezoid


REPO = Path.cwd()

TRAINER_PATH = (
    REPO / 'code/scripts/training/train_global_supersonic_kappa_q_logamp.py'
)

ATLAS_ROOT = (
    REPO
    / "assets/pinn_supersonic/"
      "atlas2d_v1_continuousM/N76"
)

COST500_PREDICTIONS = (
    ATLAS_ROOT
    / "cost500/N76_cost500_predictions_500.csv"
)

COST500_COORDS = (
    REPO / 'assets/pinn_supersonic/csv/computational_cost/cost500/table_cost500_coordinates.csv'
)

T401_SHOOTING = (
    ATLAS_ROOT
    / "shooting_T401/"
      "N76_T401_shooting_401.csv"
)

FINAL_MODE_ROOT = (
    REPO
    / "assets/classic_supersonic/"
      "dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS"
)

FINAL_MODE_INDEX = (
    FINAL_MODE_ROOT
    / "classical_supersonic_final_modes_index.csv"
)

OUT = (
    ATLAS_ROOT
    / "final_benchmarks/pinn_only"
)

CHARTS = [
    "C00", "C01", "C02",
    "C10", "C11", "C12",
    "C20", "C21", "C22",
    "C30", "C31", "C32",
]


def load_python_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def coord_key(mach: float, alpha: float):
    return (
        round(float(mach), 12),
        round(float(alpha), 12),
    )


def load_models(device: torch.device):
    trainer = load_python_module(
        TRAINER_PATH,
        "final_pinn_trainer",
    )

    models = {}
    configs = {}

    # Use the already frozen T401 direct-PINN predictions to identify
    # unambiguously the checkpoint used for each atlas chart.
    if not T401_SHOOTING.is_file():
        raise FileNotFoundError(
            T401_SHOOTING
        )

    frozen = pd.read_csv(
        T401_SHOOTING
    )

    required = {
        "Mach",
        "alpha",
        "atlas_chart",
        "cr_pinn",
        "ci_pinn",
    }

    missing = (
        required
        - set(frozen.columns)
    )

    if missing:
        raise RuntimeError(
            "T401 frozen predictions are missing "
            f"{sorted(missing)}"
        )

    search_root = (
        REPO
        / "assets/pinn_supersonic/"
          "atlas2d_v1_continuousM"
    )

    all_candidates = sorted(
        search_root.rglob(
            "best_joint_checkpoint.pt"
        )
    )

    print(
        "checkpoint candidates found:",
        len(all_candidates),
        flush=True,
    )

    if not all_candidates:
        raise FileNotFoundError(
            "No best_joint_checkpoint.pt found under "
            f"{search_root}"
        )

    t0 = time.perf_counter()

    selected_paths = {}

    for chart in CHARTS:

        reference_rows = frozen[
            frozen["atlas_chart"]
            .astype(str)
            .eq(chart)
        ]

        if reference_rows.empty:
            raise RuntimeError(
                f"No frozen T401 row for {chart}"
            )

        reference = (
            reference_rows
            .iloc[0]
        )

        ref_mach = float(
            reference["Mach"]
        )

        ref_alpha = float(
            reference["alpha"]
        )

        ref_cr = float(
            reference["cr_pinn"]
        )

        ref_ci = float(
            reference["ci_pinn"]
        )

        candidates = [
            cp
            for cp in all_candidates
            if (
                chart in cp.parts
                and "N76" in cp.parts
            )
        ]

        # Fallback in case N76 is embedded in a longer directory name.
        if not candidates:
            candidates = [
                cp
                for cp in all_candidates
                if (
                    chart in cp.parts
                    and "N76" in str(cp)
                )
            ]

        if not candidates:
            raise FileNotFoundError(
                f"No N76 checkpoint candidate for {chart}"
            )

        scored = []

        for cp in candidates:

            checkpoint = torch.load(
                cp,
                map_location=device,
            )

            if (
                not isinstance(
                    checkpoint,
                    dict,
                )
                or "model_state_dict"
                not in checkpoint
                or "config"
                not in checkpoint
            ):
                continue

            config = checkpoint[
                "config"
            ]

            model = (
                trainer.build_model(
                    config
                )
                .to(device)
            )

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

            model.eval()

            dtype = next(
                model.parameters()
            ).dtype

            alpha_t = torch.tensor(
                [[ref_alpha]],
                dtype=dtype,
                device=device,
            )

            mach_t = torch.tensor(
                [[ref_mach]],
                dtype=dtype,
                device=device,
            )

            with torch.no_grad():
                cr_t, ci_t = (
                    model.get_spectrum(
                        alpha_t,
                        mach_t,
                    )
                )

            pred_cr = float(
                cr_t.detach()
                .cpu()
                .reshape(-1)[0]
            )

            pred_ci = float(
                ci_t.detach()
                .cpu()
                .reshape(-1)[0]
            )

            error = float(
                np.hypot(
                    pred_cr - ref_cr,
                    pred_ci - ref_ci,
                )
            )

            scored.append(
                (
                    error,
                    cp,
                    pred_cr,
                    pred_ci,
                )
            )

            del model
            del checkpoint

        if not scored:
            raise RuntimeError(
                f"No readable checkpoint candidate for {chart}"
            )

        scored.sort(
            key=lambda item:
                (
                    item[0],
                    str(item[1]),
                )
        )

        best_error, best_path, _, _ = (
            scored[0]
        )

        print()
        print(
            chart,
            "checkpoint candidates =",
            len(scored),
            flush=True,
        )

        for score, cp, pred_cr, pred_ci in scored:
            print(
                "  ",
                f"delta={score:.3e}",
                cp,
                f"pred=({pred_cr:.9e},{pred_ci:.9e})",
                flush=True,
            )

        if best_error > 1.0e-7:
            raise RuntimeError(
                f"{chart}: no checkpoint reproduces "
                "the frozen T401 prediction. "
                f"Best discrepancy={best_error:.6e}; "
                f"path={best_path}"
            )

        selected_paths[
            chart
        ] = best_path

        checkpoint = torch.load(
            best_path,
            map_location=device,
        )

        config = checkpoint[
            "config"
        ]

        model = (
            trainer.build_model(
                config
            )
            .to(device)
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        model.eval()

        models[chart] = model
        configs[chart] = config

        print(
            chart,
            "SELECTED:",
            best_path,
            f"frozen discrepancy={best_error:.3e}",
            flush=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    load_seconds = (
        time.perf_counter()
        - t0
    )

    manifest = {
        chart: str(path)
        for chart, path
        in selected_paths.items()
    }

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        OUT
        / "selected_checkpoint_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    return (
        trainer,
        models,
        configs,
        load_seconds,
    )


def get_chart_column(frame: pd.DataFrame) -> str:
    for c in [
        "atlas_chart",
        "primary_chart",
        "chart",
    ]:
        if c in frame.columns:
            return c

    raise RuntimeError(
        "No chart column found."
    )


def load_cost500_frame() -> pd.DataFrame:

    if not COST500_PREDICTIONS.is_file():
        raise FileNotFoundError(
            COST500_PREDICTIONS
        )

    df = pd.read_csv(
        COST500_PREDICTIONS
    )

    chart_col = get_chart_column(df)

    df = df.rename(
        columns={
            chart_col: "atlas_chart"
        }
    )

    if "cost_id" not in df.columns:

        coords = pd.read_csv(
            COST500_COORDS
        )

        if "cost_id" not in coords.columns:
            coords = coords.copy()
            coords["cost_id"] = np.arange(
                len(coords)
            )

        ids = {
            coord_key(m, a): int(cid)
            for m, a, cid in zip(
                coords["Mach"],
                coords["alpha"],
                coords["cost_id"],
            )
        }

        df["cost_id"] = [
            ids[coord_key(m, a)]
            for m, a in zip(
                df["Mach"],
                df["alpha"],
            )
        ]

    df = (
        df.sort_values("cost_id")
        .reset_index(drop=True)
    )

    assert len(df) == 500
    assert df["cost_id"].tolist() == list(
        range(500)
    )

    return df


def tensor_column(
    values,
    *,
    model,
    device,
):
    dtype = next(
        model.parameters()
    ).dtype

    return torch.as_tensor(
        np.asarray(values, dtype=float),
        dtype=dtype,
        device=device,
    ).reshape(-1, 1)


def make_spectral_groups(
    frame,
    *,
    models,
    device,
):
    groups = {}

    for chart, sub in frame.groupby(
        "atlas_chart",
        sort=True,
    ):
        chart = str(chart)

        model = models[chart]

        alpha = tensor_column(
            sub["alpha"],
            model=model,
            device=device,
        )

        mach = tensor_column(
            sub["Mach"],
            model=model,
            device=device,
        )

        groups[chart] = (
            alpha,
            mach,
        )

    return groups


@torch.inference_mode()
def run_spectral_groups(
    groups,
    models,
):
    outputs = []

    for chart, (
        alpha,
        mach,
    ) in groups.items():

        cr, ci = models[
            chart
        ].get_spectrum(
            alpha,
            mach,
        )

        outputs.append(
            (cr, ci)
        )

    return outputs


def benchmark_callable(
    fn,
    *,
    device,
    warmup,
    repeats,
):

    for _ in range(warmup):
        fn()

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    for _ in range(repeats):
        fn()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - t0
    )

    return {
        "repeats": int(repeats),
        "total_seconds": float(elapsed),
        "seconds_per_call":
            float(elapsed / repeats),
    }


def resolve_modal_callable(model):

    for name in [
        "predict_modal",
        "predict_mode",
        "get_modal",
        "get_mode",
    ]:
        if hasattr(model, name):
            return getattr(
                model,
                name,
            )

    return model.forward


def call_modal(
    model,
    *,
    xi,
    alpha,
    mach,
):

    if hasattr(
        model,
        "set_mach_context",
    ):
        try:
            model.set_mach_context(
                float(
                    mach.reshape(-1)[0]
                    .detach()
                    .cpu()
                )
            )
        except Exception:
            pass

    fn = resolve_modal_callable(
        model
    )

    sig = inspect.signature(fn)

    kwargs = {}

    for name, param in sig.parameters.items():

        lname = name.lower()

        if lname == "self":
            continue

        if (
            "xi" in lname
            or lname in {
                "x",
                "coordinate",
                "coordinates",
            }
        ):
            kwargs[name] = xi
            continue

        if (
            "alpha" in lname
            or lname in {
                "a",
                "wavenumber",
            }
        ):
            kwargs[name] = alpha
            continue

        if (
            "mach" in lname
            or lname in {
                "m",
                "mach_number",
            }
        ):
            kwargs[name] = mach
            continue

        if (
            param.default
            is not inspect.Parameter.empty
        ):
            continue

        raise RuntimeError(
            "Cannot resolve modal argument "
            f"{name!r} for signature {sig}"
        )

    return fn(**kwargs)


def parse_modal_output(output):

    if isinstance(output, dict):

        lower = {
            str(k).lower(): v
            for k, v in output.items()
        }

        def pick(names):
            for name in names:
                if name in lower:
                    return lower[name]
            return None

        kappa = pick([
            "kappa",
            "k",
        ])

        q = pick([
            "q",
        ])

        log_amp = pick([
            "log_amp",
            "logamp",
            "log_abs_p",
            "log_absp",
            "log_modulus",
            "log_p_abs",
        ])

        if (
            kappa is None
            or q is None
            or log_amp is None
        ):
            raise RuntimeError(
                "Could not identify modal outputs "
                f"from dict keys {list(output)}"
            )

        return kappa, q, log_amp

    if isinstance(
        output,
        (tuple, list),
    ):
        if len(output) < 3:
            raise RuntimeError(
                "Modal tuple has fewer than "
                "three outputs."
            )

        return (
            output[0],
            output[1],
            output[2],
        )

    if torch.is_tensor(output):

        if (
            output.ndim >= 2
            and output.shape[-1] >= 3
        ):
            return (
                output[..., 0:1],
                output[..., 1:2],
                output[..., 2:3],
            )

    raise RuntimeError(
        "Unsupported modal return type: "
        f"{type(output)}"
    )


def make_modal_groups(
    frame,
    *,
    models,
    configs,
    device,
    n_xi=801,
):
    groups = {}

    for chart, sub in frame.groupby(
        "atlas_chart",
        sort=True,
    ):
        chart = str(chart)
        model = models[chart]

        cfg = configs[chart]
        xi_max = float(
            cfg["model"]["xi_max"]
        )

        xi_grid = np.linspace(
            -xi_max,
            xi_max,
            n_xi,
        )

        n = len(sub)

        xi = np.tile(
            xi_grid,
            n,
        )

        alpha = np.repeat(
            sub["alpha"].to_numpy(float),
            n_xi,
        )

        mach = np.repeat(
            sub["Mach"].to_numpy(float),
            n_xi,
        )

        groups[chart] = (
            tensor_column(
                xi,
                model=model,
                device=device,
            ),
            tensor_column(
                alpha,
                model=model,
                device=device,
            ),
            tensor_column(
                mach,
                model=model,
                device=device,
            ),
        )

    return groups


@torch.inference_mode()
def run_full_groups(
    spectral_groups,
    modal_groups,
    models,
):
    outputs = []

    for chart in spectral_groups:

        alpha_s, mach_s = (
            spectral_groups[chart]
        )

        cr, ci = models[
            chart
        ].get_spectrum(
            alpha_s,
            mach_s,
        )

        xi, alpha_m, mach_m = (
            modal_groups[chart]
        )

        modal = call_modal(
            models[chart],
            xi=xi,
            alpha=alpha_m,
            mach=mach_m,
        )

        parsed = parse_modal_output(
            modal
        )

        outputs.append(
            (
                cr,
                ci,
                *parsed,
            )
        )

    return outputs


def find_reference_fields_csv():

    required = {
        "Mach",
        "alpha",
        "y",
        "p_real",
        "p_imag",
    }

    candidates = []

    if FINAL_MODE_ROOT.is_dir():

        for p in FINAL_MODE_ROOT.rglob(
            "*.csv"
        ):
            try:
                cols = set(
                    pd.read_csv(
                        p,
                        nrows=0,
                    ).columns
                )
            except Exception:
                continue

            if required.issubset(cols):
                candidates.append(p)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda p: p.stat().st_size,
    )


def select_npz_mode_array(
    arr,
    index,
):

    arr = np.asarray(arr)

    if (
        arr.dtype == object
        and arr.ndim == 1
        and len(arr) > index
    ):
        return np.asarray(
            arr[index]
        )

    if arr.ndim == 1:
        return arr

    if arr.ndim == 2:

        if (
            arr.shape[0] > index
            and arr.shape[1] > 10
        ):
            return np.asarray(
                arr[index]
            )

        if (
            arr.shape[1] > index
            and arr.shape[0] > 10
        ):
            return np.asarray(
                arr[:, index]
            )

    raise RuntimeError(
        f"Cannot select mode {index} "
        f"from shape {arr.shape}"
    )


def first_key(data, names):
    for name in names:
        if name in data.files:
            return name
    return None


def resolve_npz_path(value):

    p = Path(str(value))

    candidates = [
        p,
        REPO / p,
        FINAL_MODE_ROOT / p,
    ]

    for c in candidates:
        if c.is_file():
            return c.resolve()

    raise FileNotFoundError(
        f"Cannot resolve NPZ {value}"
    )


def load_mode_from_npz(
    index_row,
):

    path = resolve_npz_path(
        index_row["mode_npz_path"]
    )

    mode_index = int(
        index_row[
            "mode_index_in_npz"
        ]
    )

    data = np.load(
        path,
        allow_pickle=True,
    )

    y_key = first_key(
        data,
        [
            "y",
            "y_grid",
            "coordinates",
            "coordinate",
        ],
    )

    if y_key is None:
        raise RuntimeError(
            f"No y array in {path}; "
            f"keys={data.files}"
        )

    y = select_npz_mode_array(
        data[y_key],
        mode_index,
    ).astype(float)

    p_key = first_key(
        data,
        [
            "p",
            "pressure",
            "p_complex",
        ],
    )

    if p_key is not None:

        p = select_npz_mode_array(
            data[p_key],
            mode_index,
        ).astype(
            np.complex128
        )

    else:

        pr_key = first_key(
            data,
            [
                "p_real",
                "pressure_real",
            ],
        )

        pi_key = first_key(
            data,
            [
                "p_imag",
                "pressure_imag",
            ],
        )

        if (
            pr_key is None
            or pi_key is None
        ):
            raise RuntimeError(
                f"No pressure arrays in {path}; "
                f"keys={data.files}"
            )

        p = (
            select_npz_mode_array(
                data[pr_key],
                mode_index,
            ).astype(float)
            + 1j
            * select_npz_mode_array(
                data[pi_key],
                mode_index,
            ).astype(float)
        )

    return y, p


def build_reference_accessor():

    long_csv = (
        find_reference_fields_csv()
    )

    if long_csv is not None:

        print(
            "Using long modal reference:",
            long_csv,
        )

        fields = pd.read_csv(
            long_csv
        )

        fields["_key"] = [
            coord_key(m, a)
            for m, a in zip(
                fields["Mach"],
                fields["alpha"],
            )
        ]

        groups = {
            k: g.copy()
            for k, g in fields.groupby(
                "_key",
                sort=False,
            )
        }

        def accessor(mach, alpha):

            k = coord_key(
                mach,
                alpha,
            )

            if k not in groups:
                raise KeyError(k)

            g = groups[k]

            return (
                g["y"].to_numpy(float),
                g["p_real"].to_numpy(float)
                + 1j
                * g[
                    "p_imag"
                ].to_numpy(float),
            )

        return accessor, str(long_csv)

    if not FINAL_MODE_INDEX.is_file():
        raise FileNotFoundError(
            FINAL_MODE_INDEX
        )

    index = pd.read_csv(
        FINAL_MODE_INDEX
    )

    index["_key"] = [
        coord_key(m, a)
        for m, a in zip(
            index["Mach"],
            index["alpha"],
        )
    ]

    rows = {
        k: g.iloc[0]
        for k, g in index.groupby(
            "_key",
            sort=False,
        )
    }

    print(
        "Using modal NPZ index:",
        FINAL_MODE_INDEX,
    )

    def accessor(mach, alpha):

        k = coord_key(
            mach,
            alpha,
        )

        if k not in rows:
            raise KeyError(k)

        return load_mode_from_npz(
            rows[k]
        )

    return (
        accessor,
        str(FINAL_MODE_INDEX),
    )


def y_to_xi(
    y,
    mapping_scale,
):
    y = np.asarray(
        y,
        dtype=float,
    )

    out = np.zeros_like(y)

    mask = ~np.isclose(
        y,
        0.0,
    )

    out[mask] = (
        2.0 * y[mask]
        / (
            mapping_scale
            + np.sqrt(
                mapping_scale**2
                + 4.0
                * y[mask] ** 2
            )
        )
    )

    return out


@torch.inference_mode()
def modal_metrics_for_point(
    *,
    model,
    config,
    mach,
    alpha,
    y_ref,
    p_ref,
    device,
):

    y_ref = np.asarray(
        y_ref,
        dtype=float,
    )

    p_ref = np.asarray(
        p_ref,
        dtype=np.complex128,
    )

    finite = (
        np.isfinite(y_ref)
        & np.isfinite(p_ref.real)
        & np.isfinite(p_ref.imag)
    )

    y_ref = y_ref[finite]
    p_ref = p_ref[finite]

    order = np.argsort(
        y_ref
    )

    y_ref = y_ref[order]
    p_ref = p_ref[order]

    core = (
        np.abs(y_ref)
        <= 20.0
    )

    y = y_ref[core]
    p = p_ref[core]

    if len(y) < 50:
        raise RuntimeError(
            "Too few reference points "
            "inside |y|<=20."
        )

    if len(y) > 1201:
        idx = np.linspace(
            0,
            len(y) - 1,
            1201,
        ).round().astype(int)

        idx = np.unique(idx)

        y = y[idx]
        p = p[idx]

    mapping_scale = float(
        config["model"][
            "mapping_scale"
        ]
    )

    xi_max = float(
        config["model"]["xi_max"]
    )

    xi_np = y_to_xi(
        y,
        mapping_scale,
    )

    valid = (
        np.abs(xi_np)
        <= xi_max
    )

    y = y[valid]
    p = p[valid]
    xi_np = xi_np[valid]

    alpha_np = np.full(
        len(y),
        float(alpha),
    )

    mach_np = np.full(
        len(y),
        float(mach),
    )

    xi = tensor_column(
        xi_np,
        model=model,
        device=device,
    )

    alpha_t = tensor_column(
        alpha_np,
        model=model,
        device=device,
    )

    mach_t = tensor_column(
        mach_np,
        model=model,
        device=device,
    )

    output = call_modal(
        model,
        xi=xi,
        alpha=alpha_t,
        mach=mach_t,
    )

    kappa_t, q_t, logamp_t = (
        parse_modal_output(output)
    )

    kappa = (
        kappa_t.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    q = (
        q_t.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    logamp = (
        logamp_t.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    phase = cumulative_trapezoid(
        q,
        y,
        initial=0.0,
    )

    phase0 = np.interp(
        0.0,
        y,
        phase,
    )

    phase = phase - phase0

    amp = np.exp(
        np.clip(
            logamp,
            -80.0,
            80.0,
        )
    )

    p_pred = (
        amp
        * np.exp(1j * phase)
    )

    denominator = np.vdot(
        p_pred,
        p_pred,
    )

    if abs(denominator) == 0:
        raise RuntimeError(
            "Zero predicted modal norm."
        )

    beta = (
        np.vdot(
            p_pred,
            p,
        )
        / denominator
    )

    p_aligned = (
        beta * p_pred
    )

    p_norm = np.linalg.norm(
        p
    )

    pressure_rel_l2 = float(
        np.linalg.norm(
            p_aligned - p
        )
        / p_norm
    )

    amplitude_rel_l2 = float(
        np.linalg.norm(
            np.abs(p_aligned)
            - np.abs(p)
        )
        / np.linalg.norm(
            np.abs(p)
        )
    )

    overlap = float(
        abs(
            np.vdot(
                p_aligned,
                p,
            )
        )
        / (
            np.linalg.norm(
                p_aligned
            )
            * p_norm
        )
    )

    return {
        "n_modal_points":
            int(len(y)),
        "pressure_rel_l2":
            pressure_rel_l2,
        "amplitude_rel_l2":
            amplitude_rel_l2,
        "pressure_overlap":
            overlap,
        "alignment_real":
            float(beta.real),
        "alignment_imag":
            float(beta.imag),
        "kappa_abs_mean":
            float(
                np.mean(
                    np.abs(kappa)
                )
            ),
        "q_abs_mean":
            float(
                np.mean(
                    np.abs(q)
                )
            ),
    }


def summarize(values):
    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return {}

    return {
        "n": int(len(x)),
        "mean": float(
            np.mean(x)
        ),
        "median": float(
            np.median(x)
        ),
        "p90": float(
            np.quantile(
                x,
                0.90,
            )
        ),
        "p95": float(
            np.quantile(
                x,
                0.95,
            )
        ),
        "max": float(
            np.max(x)
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--modal-grid",
        type=int,
        default=801,
    )

    args = parser.parse_args()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA unavailable."
        )

    (
        trainer,
        models,
        configs,
        load_seconds,
    ) = load_models(device)

    cost500 = load_cost500_frame()

    cost1 = (
        cost500.iloc[[0]]
        .copy()
        .reset_index(drop=True)
    )

    print(
        "COST1 coordinate:",
        float(cost1["Mach"].iloc[0]),
        float(cost1["alpha"].iloc[0]),
        str(
            cost1[
                "atlas_chart"
            ].iloc[0]
        ),
    )

    spectral500 = make_spectral_groups(
        cost500,
        models=models,
        device=device,
    )

    spectral1 = make_spectral_groups(
        cost1,
        models=models,
        device=device,
    )

    spectral_timing_1 = (
        benchmark_callable(
            lambda:
                run_spectral_groups(
                    spectral1,
                    models,
                ),
            device=device,
            warmup=100,
            repeats=5000,
        )
    )

    spectral_timing_500 = (
        benchmark_callable(
            lambda:
                run_spectral_groups(
                    spectral500,
                    models,
                ),
            device=device,
            warmup=50,
            repeats=1000,
        )
    )

    modal1 = make_modal_groups(
        cost1,
        models=models,
        configs=configs,
        device=device,
        n_xi=args.modal_grid,
    )

    modal500 = make_modal_groups(
        cost500,
        models=models,
        configs=configs,
        device=device,
        n_xi=args.modal_grid,
    )

    full_timing_1 = (
        benchmark_callable(
            lambda:
                run_full_groups(
                    spectral1,
                    modal1,
                    models,
                ),
            device=device,
            warmup=20,
            repeats=500,
        )
    )

    full_timing_500 = (
        benchmark_callable(
            lambda:
                run_full_groups(
                    spectral500,
                    modal500,
                    models,
                ),
            device=device,
            warmup=5,
            repeats=30,
        )
    )

    t401 = pd.read_csv(
        T401_SHOOTING
    )

    assert len(t401) == 401

    t401["ci_abs_error"] = np.abs(
        t401["ci_pinn"]
        - t401["ci_reference"]
    )

    t401["spectral_error"] = np.hypot(
        t401["cr_pinn"]
        - t401["cr_reference"],
        t401["ci_pinn"]
        - t401["ci_reference"],
    )

    t401["shoot_error"] = np.hypot(
        t401["shoot_cr"]
        - t401["cr_reference"],
        t401["shoot_ci"]
        - t401["ci_reference"],
    )

    technical = (
        as_bool(
            t401[
                "shoot_spectral_success"
            ]
        )
        & as_bool(
            t401[
                "shoot_mode_success"
            ]
        )
    )

    localized = (
        technical
        & (
            t401[
                "shoot_error"
            ]
            <= 1e-4
        )
    )

    reference_accessor, reference_source = (
        build_reference_accessor()
    )

    modal_rows = []

    for idx, row in t401.iterrows():

        chart = str(
            row["atlas_chart"]
        )

        try:

            y_ref, p_ref = (
                reference_accessor(
                    float(row["Mach"]),
                    float(row["alpha"]),
                )
            )

            metrics = (
                modal_metrics_for_point(
                    model=models[chart],
                    config=configs[chart],
                    mach=float(
                        row["Mach"]
                    ),
                    alpha=float(
                        row["alpha"]
                    ),
                    y_ref=y_ref,
                    p_ref=p_ref,
                    device=device,
                )
            )

            modal_rows.append(
                {
                    "t401_id":
                        row.get(
                            "t401_id",
                            idx,
                        ),
                    "atlas_chart":
                        chart,
                    "Mach":
                        float(
                            row["Mach"]
                        ),
                    "alpha":
                        float(
                            row["alpha"]
                        ),
                    "status":
                        "OK",
                    **metrics,
                }
            )

        except Exception as exc:

            modal_rows.append(
                {
                    "t401_id":
                        row.get(
                            "t401_id",
                            idx,
                        ),
                    "atlas_chart":
                        chart,
                    "Mach":
                        float(
                            row["Mach"]
                        ),
                    "alpha":
                        float(
                            row["alpha"]
                        ),
                    "status":
                        "FAILED",
                    "error":
                        repr(exc),
                }
            )

            print(
                "MODAL FAIL:",
                idx,
                chart,
                row["Mach"],
                row["alpha"],
                repr(exc),
                flush=True,
            )

    modal_df = pd.DataFrame(
        modal_rows
    )

    modal_df.to_csv(
        OUT
        / "N76_T401_direct_PINN_modal_metrics.csv",
        index=False,
    )

    good = modal_df[
        modal_df["status"].eq(
            "OK"
        )
    ]

    modal_summary = {
        "reference_source":
            reference_source,
        "n_total": 401,
        "n_modal_success":
            int(len(good)),
        "pressure_rel_l2":
            summarize(
                good[
                    "pressure_rel_l2"
                ]
                if len(good)
                else []
            ),
        "amplitude_rel_l2":
            summarize(
                good[
                    "amplitude_rel_l2"
                ]
                if len(good)
                else []
            ),
        "pressure_overlap":
            summarize(
                good[
                    "pressure_overlap"
                ]
                if len(good)
                else []
            ),
    }

    summary = {
        "device": str(device),
        "model_load_seconds":
            float(load_seconds),
        "timing_excludes_checkpoint_load":
            True,
        "cost1_coordinate": {
            "Mach":
                float(
                    cost1[
                        "Mach"
                    ].iloc[0]
                ),
            "alpha":
                float(
                    cost1[
                        "alpha"
                    ].iloc[0]
                ),
            "atlas_chart":
                str(
                    cost1[
                        "atlas_chart"
                    ].iloc[0]
                ),
        },
        "spectral_only": {
            "cost1":
                spectral_timing_1,
            "cost500_batched":
                spectral_timing_500,
            "cost500_seconds_per_query_equivalent":
                float(
                    spectral_timing_500[
                        "seconds_per_call"
                    ]
                    / 500.0
                ),
        },
        "spectral_plus_modal_801": {
            "cost1":
                full_timing_1,
            "cost500_batched":
                full_timing_500,
            "cost500_seconds_per_query_equivalent":
                float(
                    full_timing_500[
                        "seconds_per_call"
                    ]
                    / 500.0
                ),
        },
        "T401_direct_PINN": {
            "n": 401,
            "ci_abs_error":
                summarize(
                    t401[
                        "ci_abs_error"
                    ]
                ),
            "spectral_error":
                summarize(
                    t401[
                        "spectral_error"
                    ]
                ),
            "branch_localization_success":
                int(
                    localized.sum()
                ),
            "branch_localization_fraction":
                float(
                    localized.mean()
                ),
        },
        "T401_modal":
            modal_summary,
    }

    (
        OUT
        / "N76_PINN_only_final_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 100)
    print("FINAL PINN-ONLY BENCHMARK")
    print("=" * 100)

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print(
        "output:",
        OUT,
    )


if __name__ == "__main__":
    main()
