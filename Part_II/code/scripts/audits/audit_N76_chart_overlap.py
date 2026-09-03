#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO = Path.cwd().resolve()

# Make the repository containing the ``src`` package importable.
_model_matches = list(
    REPO.glob(
        "**/src/models/"
        "kh_supersonic_global_kappa_q_logamp.py"
    )
)

if not _model_matches:
    raise FileNotFoundError(
        "Cannot locate src/models/"
        "kh_supersonic_global_kappa_q_logamp.py"
    )

_SRC_DIR = _model_matches[0].parent.parent
_SRC_PARENT = _SRC_DIR.parent

if str(_SRC_PARENT) not in sys.path:
    sys.path.insert(0, str(_SRC_PARENT))

TRAINER = (
    REPO / 'code/scripts/training/train_global_supersonic_kappa_q_logamp_continuousM.py'
)

CHECKPOINT_MANIFEST = (
    REPO / 'assets/pinn_supersonic/json/atlas_12charts/pinn_only/selected_checkpoint_manifest.json'
)

OUT = (
    REPO
    / "assets/p3-supersonic-results"
)

CHARTS = {
    "C00": (1.00, 1.25, 0.05, 0.13),
    "C01": (1.00, 1.25, 0.10, 0.22),
    "C02": (1.00, 1.25, 0.19, 0.36),

    "C10": (1.15, 1.45, 0.05, 0.13),
    "C11": (1.15, 1.45, 0.10, 0.22),
    "C12": (1.15, 1.45, 0.19, 0.36),

    "C20": (1.35, 1.65, 0.05, 0.13),
    "C21": (1.35, 1.65, 0.10, 0.22),
    "C22": (1.35, 1.65, 0.19, 0.36),

    "C30": (1.55, 1.90, 0.05, 0.13),
    "C31": (1.55, 1.90, 0.10, 0.22),
    "C32": (1.55, 1.90, 0.19, 0.36),
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def load_models(device: torch.device):
    trainer = load_module(
        TRAINER,
        "overlap_continuousM_trainer",
    )

    manifest = json.loads(
        CHECKPOINT_MANIFEST.read_text()
    )

    models = {}

    for chart in sorted(CHARTS):
        if chart not in manifest:
            raise RuntimeError(
                f"{chart} missing from checkpoint manifest"
            )

        cp = Path(
            manifest[chart]
        )

        assert cp.is_file(), cp

        checkpoint = torch.load(
            cp,
            map_location=device,
        )

        config = checkpoint["config"]

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

        print(
            chart,
            "->",
            cp,
            flush=True,
        )

    return models


def evaluate(
    model,
    mach: np.ndarray,
    alpha: np.ndarray,
    device: torch.device,
):
    dtype = next(
        model.parameters()
    ).dtype

    mt = torch.as_tensor(
        mach.reshape(-1, 1),
        dtype=dtype,
        device=device,
    )

    at = torch.as_tensor(
        alpha.reshape(-1, 1),
        dtype=dtype,
        device=device,
    )

    with torch.inference_mode():
        cr, ci = model.get_spectrum(
            at,
            mt,
        )

    return (
        cr.detach()
        .cpu()
        .numpy()
        .reshape(-1),
        ci.detach()
        .cpu()
        .numpy()
        .reshape(-1),
    )


def summary(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]

    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, .90)),
        "p95": float(np.quantile(x, .95)),
        "max": float(np.max(x)),
    }


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device =",
        device,
        flush=True,
    )

    models = load_models(
        device
    )

    rows = []

    pair_count = 0

    for c1, c2 in combinations(
        sorted(CHARTS),
        2,
    ):
        m1lo, m1hi, a1lo, a1hi = CHARTS[c1]
        m2lo, m2hi, a2lo, a2hi = CHARTS[c2]

        mlo = max(
            m1lo,
            m2lo,
        )

        mhi = min(
            m1hi,
            m2hi,
        )

        alo = max(
            a1lo,
            a2lo,
        )

        ahi = min(
            a1hi,
            a2hi,
        )

        # Positive-area overlap only.
        if not (
            mhi > mlo
            and ahi > alo
        ):
            continue

        pair_count += 1

        mach_grid = np.linspace(
            mlo,
            mhi,
            21,
        )

        alpha_grid = np.linspace(
            alo,
            ahi,
            21,
        )

        M, A = np.meshgrid(
            mach_grid,
            alpha_grid,
            indexing="ij",
        )

        mf = M.reshape(-1)
        af = A.reshape(-1)

        cr1, ci1 = evaluate(
            models[c1],
            mf,
            af,
            device,
        )

        cr2, ci2 = evaluate(
            models[c2],
            mf,
            af,
            device,
        )

        dcr = np.abs(
            cr1 - cr2
        )

        dci = np.abs(
            ci1 - ci2
        )

        dc = np.hypot(
            dcr,
            dci,
        )

        for k in range(
            len(mf)
        ):
            rows.append({
                "chart_1": c1,
                "chart_2": c2,
                "Mach": mf[k],
                "alpha": af[k],
                "cr_1": cr1[k],
                "ci_1": ci1[k],
                "cr_2": cr2[k],
                "ci_2": ci2[k],
                "delta_cr": dcr[k],
                "delta_ci": dci[k],
                "delta_c": dc[k],
            })

        print(
            c1,
            c2,
            "n=",
            len(mf),
            "median=",
            f"{np.median(dc):.6e}",
            "p95=",
            f"{np.quantile(dc, .95):.6e}",
            "max=",
            f"{np.max(dc):.6e}",
            flush=True,
        )

    result = pd.DataFrame(
        rows
    )

    assert not result.empty

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    points_path = (
        OUT
        / "Tab_supersonic_N76_chart_overlap_points.csv"
    )

    result.to_csv(
        points_path,
        index=False,
    )

    pair_rows = []

    for (c1, c2), g in result.groupby(
        [
            "chart_1",
            "chart_2",
        ],
        sort=True,
    ):
        for metric in [
            "delta_cr",
            "delta_ci",
            "delta_c",
        ]:
            pair_rows.append({
                "chart_1": c1,
                "chart_2": c2,
                "metric": metric,
                **summary(
                    g[metric]
                ),
            })

    pair_df = pd.DataFrame(
        pair_rows
    )

    pair_path = (
        OUT
        / "Tab_supersonic_N76_chart_overlap_by_pair.csv"
    )

    pair_df.to_csv(
        pair_path,
        index=False,
    )

    global_rows = []

    for metric in [
        "delta_cr",
        "delta_ci",
        "delta_c",
    ]:
        global_rows.append({
            "metric": metric,
            **summary(
                result[metric]
            ),
        })

    global_df = pd.DataFrame(
        global_rows
    )

    global_path = (
        OUT
        / "Tab_supersonic_N76_chart_overlap_global.csv"
    )

    global_df.to_csv(
        global_path,
        index=False,
    )

    print()
    print("=" * 100)
    print("INTER-CHART OVERLAP AUDIT COMPLETE")
    print("=" * 100)

    print(
        "overlapping chart pairs =",
        pair_count,
    )

    print(
        "total pairwise points =",
        len(result),
    )

    print()

    print(
        global_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8e}",
        )
    )

    print()
    print("saved:")
    print(points_path)
    print(pair_path)
    print(global_path)


if __name__ == "__main__":
    main()
