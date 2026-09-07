#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO = Path.cwd().resolve()

# Current PINN/physics package lives under code/src.
# Adding its parent makes imports such as `src.models...`
# resolve exactly as in the production environment.
for _path in [
    REPO,
    REPO / "code",
    REPO / "classic_supersonic" / "src",
]:
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

CONFIG_ROOT = (
    REPO
    / "configs/pinn_supersonic/"
      "atlas2d_v1_continuousM_fullc_v1/N76"
)

RUN_ROOT = (
    REPO
    / "assets/pinn_supersonic/"
      "atlas2d_v1_continuousM_fullc_v1/N76/runs"
)

TRAINER_PATH = (
    REPO
    / "scripts/pinn_supersonic/"
      "train_global_supersonic_kappa_q_logamp_continuousM.py"
)

CHARTS = [
    "C00", "C01", "C02",
    "C10", "C11", "C12",
    "C20", "C21", "C22",
    "C30", "C31", "C32",
]

# Routing tolerance used only for geometric covering.
ROUTE_TOL = 5.0e-10


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

    spec.loader.exec_module(
        module
    )

    return module


def build_plan() -> pd.DataFrame:
    rows = []

    for chart in CHARTS:
        path = CONFIG_ROOT / f"{chart}.json"

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        config = json.loads(
            path.read_text()
        )

        row = {
            "chart_id":
                chart,

            "mach_min":
                float(config["mach_min"]),

            "mach_max":
                float(config["mach_max"]),

            "alpha_min":
                float(config["alpha_min"]),

            "alpha_max":
                float(config["alpha_max"]),

            "config_path":
                str(path),
        }

        row["mach_center"] = 0.5 * (
            row["mach_min"]
            + row["mach_max"]
        )

        row["alpha_center"] = 0.5 * (
            row["alpha_min"]
            + row["alpha_max"]
        )

        row["mach_half"] = 0.5 * (
            row["mach_max"]
            - row["mach_min"]
        )

        row["alpha_half"] = 0.5 * (
            row["alpha_max"]
            - row["alpha_min"]
        )

        row["chart_area"] = (
            (
                row["mach_max"]
                - row["mach_min"]
            )
            *
            (
                row["alpha_max"]
                - row["alpha_min"]
            )
        )

        rows.append(row)

    plan = pd.DataFrame(rows)

    return (
        plan
        .sort_values("chart_id")
        .reset_index(drop=True)
    )


def route_chart(
    plan: pd.DataFrame,
    mach: float,
    alpha: float,
) -> pd.Series | None:
    covering = plan[
        (plan["mach_min"] - ROUTE_TOL <= mach)
        & (mach <= plan["mach_max"] + ROUTE_TOL)
        & (plan["alpha_min"] - ROUTE_TOL <= alpha)
        & (alpha <= plan["alpha_max"] + ROUTE_TOL)
    ].copy()

    if covering.empty:
        return None

    covering["route_center_distance"] = (
        (
            (
                mach
                - covering["mach_center"]
            )
            / covering["mach_half"]
        )
        ** 2
        +
        (
            (
                alpha
                - covering["alpha_center"]
            )
            / covering["alpha_half"]
        )
        ** 2
    )

    # Deterministic rule inherited from the atlas construction:
    # smallest chart area -> normalized center distance -> chart id.
    covering = covering.sort_values(
        [
            "chart_area",
            "route_center_distance",
            "chart_id",
        ],
        kind="stable",
    )

    return covering.iloc[0]


def route_id(
    plan: pd.DataFrame,
    mach: float,
    alpha: float,
) -> str:
    row = route_chart(
        plan,
        mach,
        alpha,
    )

    if row is None:
        return "NONE"

    return str(
        row["chart_id"]
    )


def detect_raw_edges(
    plan: pd.DataFrame,
    grid_n: int,
) -> pd.DataFrame:
    """
    Vectorized evaluation of the deterministic atlas router.

    Priority is identical to route_chart():
      1. smallest chart area
      2. smallest normalized distance to chart center
      3. lexicographically smallest chart_id

    plan is already sorted by chart_id, so exact area+distance ties
    naturally retain the earlier chart.
    """
    import time

    t0 = time.perf_counter()

    m_values = np.linspace(
        float(plan["mach_min"].min()),
        float(plan["mach_max"].max()),
        grid_n,
    )

    a_values = np.linspace(
        float(plan["alpha_min"].min()),
        float(plan["alpha_max"].max()),
        grid_n,
    )

    M, A = np.meshgrid(
        m_values,
        a_values,
        indexing="ij",
    )

    winners = np.full(
        M.shape,
        -1,
        dtype=np.int32,
    )

    best_area = np.full(
        M.shape,
        np.inf,
        dtype=float,
    )

    best_dist = np.full(
        M.shape,
        np.inf,
        dtype=float,
    )

    print(
        f"Routing grid: "
        f"{grid_n} x {grid_n}",
        flush=True,
    )

    for idx, chart in plan.iterrows():

        mach_min = float(chart["mach_min"])
        mach_max = float(chart["mach_max"])
        alpha_min = float(chart["alpha_min"])
        alpha_max = float(chart["alpha_max"])

        mach_center = float(
            chart["mach_center"]
        )
        alpha_center = float(
            chart["alpha_center"]
        )

        mach_half = max(
            float(chart["mach_half"]),
            1.0e-12,
        )

        alpha_half = max(
            float(chart["alpha_half"]),
            1.0e-12,
        )

        area = float(
            chart["chart_area"]
        )

        covering = (
            (M >= mach_min - ROUTE_TOL)
            & (M <= mach_max + ROUTE_TOL)
            & (A >= alpha_min - ROUTE_TOL)
            & (A <= alpha_max + ROUTE_TOL)
        )

        if not np.any(covering):
            continue

        dist = (
            ((M - mach_center) / mach_half) ** 2
            + ((A - alpha_center) / alpha_half) ** 2
        )

        # IMPORTANT:
        # reproduce the scalar lexicographic router EXACTLY.
        #
        # Do not use an artificial tolerance here: two nominally
        # identical chart areas can differ at floating-point level,
        # and route_chart() sorts their stored values directly.
        better_area = (
            area < best_area
        )

        same_area = (
            area == best_area
        )

        better_dist = (
            dist < best_dist
        )

        better = (
            covering
            & (
                better_area
                | (
                    same_area
                    & better_dist
                )
            )
        )

        winners[better] = int(idx)
        best_area[better] = area
        best_dist[better] = dist[better]

    print(
        "Routing grid complete in "
        f"{time.perf_counter() - t0:.3f} s",
        flush=True,
    )

    chart_names = (
        plan["chart_id"]
        .astype(str)
        .to_numpy()
    )

    # --------------------------------------------------------------
    # Deterministic equivalence audit:
    # vectorized router versus scalar route_chart().
    # --------------------------------------------------------------

    rng = np.random.default_rng(20260828)

    n_check = min(
        512,
        M.size,
    )

    flat_indices = rng.choice(
        M.size,
        size=n_check,
        replace=False,
    )

    mismatch_rows = []

    for flat_index in flat_indices:
        i, j = np.unravel_index(
            int(flat_index),
            M.shape,
        )

        scalar_name = route_id(
            plan,
            float(M[i, j]),
            float(A[i, j]),
        )

        vector_idx = int(
            winners[i, j]
        )

        vector_name = (
            "NONE"
            if vector_idx < 0
            else str(
                plan.iloc[
                    vector_idx
                ]["chart_id"]
            )
        )

        if scalar_name != vector_name:
            mismatch_rows.append(
                (
                    float(M[i, j]),
                    float(A[i, j]),
                    scalar_name,
                    vector_name,
                )
            )

    if mismatch_rows:
        raise RuntimeError(
            "Vectorized/scalar routing mismatch. "
            f"First mismatches: {mismatch_rows[:10]}"
        )

    print(
        f"Vectorized/scalar routing check: "
        f"{n_check}/{n_check} OK",
        flush=True,
    )

    rows = []

    # --------------------------------------------------------------
    # Interfaces whose normal direction is Mach
    # --------------------------------------------------------------

    changed_m = (
        (winners[:-1, :] != winners[1:, :])
        & (winners[:-1, :] >= 0)
        & (winners[1:, :] >= 0)
    )

    for i, j in np.argwhere(
        changed_m
    ):
        idx_minus = int(
            winners[i, j]
        )
        idx_plus = int(
            winners[i + 1, j]
        )

        rows.append({
            "normal_axis":
                "Mach",

            "chart_minus_grid":
                chart_names[idx_minus],

            "chart_plus_grid":
                chart_names[idx_plus],

            "Mach_minus_grid":
                float(m_values[i]),

            "Mach_plus_grid":
                float(m_values[i + 1]),

            "alpha_minus_grid":
                float(a_values[j]),

            "alpha_plus_grid":
                float(a_values[j]),

            "Mach_mid":
                float(
                    0.5
                    * (
                        m_values[i]
                        + m_values[i + 1]
                    )
                ),

            "alpha_mid":
                float(a_values[j]),
        })

    # --------------------------------------------------------------
    # Interfaces whose normal direction is alpha
    # --------------------------------------------------------------

    changed_a = (
        (winners[:, :-1] != winners[:, 1:])
        & (winners[:, :-1] >= 0)
        & (winners[:, 1:] >= 0)
    )

    for i, j in np.argwhere(
        changed_a
    ):
        idx_minus = int(
            winners[i, j]
        )
        idx_plus = int(
            winners[i, j + 1]
        )

        rows.append({
            "normal_axis":
                "alpha",

            "chart_minus_grid":
                chart_names[idx_minus],

            "chart_plus_grid":
                chart_names[idx_plus],

            "Mach_minus_grid":
                float(m_values[i]),

            "Mach_plus_grid":
                float(m_values[i]),

            "alpha_minus_grid":
                float(a_values[j]),

            "alpha_plus_grid":
                float(a_values[j + 1]),

            "Mach_mid":
                float(m_values[i]),

            "alpha_mid":
                float(
                    0.5
                    * (
                        a_values[j]
                        + a_values[j + 1]
                    )
                ),
        })

    edges = pd.DataFrame(
        rows
    )

    if edges.empty:
        raise RuntimeError(
            "No routing edge detected."
        )

    edges["pair_id"] = [
        "--".join(
            sorted(
                [
                    str(a),
                    str(b),
                ]
            )
        )
        for a, b in zip(
            edges[
                "chart_minus_grid"
            ],
            edges[
                "chart_plus_grid"
            ],
        )
    ]

    print(
        "Changed routing edges:",
        len(edges),
        flush=True,
    )

    return edges

def sample_edges(
    edges: pd.DataFrame,
    *,
    max_per_interface: int,
    grid_n: int,
    plan: pd.DataFrame,
) -> pd.DataFrame:
    dm = (
        float(plan["mach_max"].max())
        - float(plan["mach_min"].min())
    ) / (grid_n - 1)

    da = (
        float(plan["alpha_max"].max())
        - float(plan["alpha_min"].min())
    ) / (grid_n - 1)

    kept = []

    for (
        pair_id,
        normal_axis,
    ), group in edges.groupby(
        [
            "pair_id",
            "normal_axis",
        ],
        sort=True,
    ):
        g = (
            group
            .copy()
            .reset_index(drop=True)
        )

        tangent = (
            "alpha_mid"
            if normal_axis == "Mach"
            else "Mach_mid"
        )

        step = (
            da
            if normal_axis == "Mach"
            else dm
        )

        g = (
            g
            .sort_values(tangent)
            .reset_index(drop=True)
        )

        values = g[
            tangent
        ].to_numpy(float)

        component = np.zeros(
            len(g),
            dtype=int,
        )

        if len(g) > 1:
            component[1:] = np.cumsum(
                np.diff(values)
                > 3.1 * step
            )

        g["component"] = component

        for _, h in g.groupby(
            "component",
            sort=True,
        ):
            h = h.reset_index(
                drop=True
            )

            n = min(
                max_per_interface,
                len(h),
            )

            indices = sorted({
                int(round(x))
                for x in np.linspace(
                    0,
                    len(h) - 1,
                    n,
                )
            })

            kept.append(
                h.iloc[indices]
            )

    if not kept:
        raise RuntimeError(
            "No representative routing edge."
        )

    return pd.concat(
        kept,
        ignore_index=True,
    )


def refine_boundary(
    plan: pd.DataFrame,
    row: pd.Series,
    n_iter: int = 32,
):
    lo = np.array(
        [
            float(row["Mach_minus_grid"]),
            float(row["alpha_minus_grid"]),
        ],
        dtype=float,
    )

    hi = np.array(
        [
            float(row["Mach_plus_grid"]),
            float(row["alpha_plus_grid"]),
        ],
        dtype=float,
    )

    id_lo = route_id(
        plan,
        lo[0],
        lo[1],
    )

    id_hi = route_id(
        plan,
        hi[0],
        hi[1],
    )

    if (
        id_lo == "NONE"
        or id_hi == "NONE"
        or id_lo == id_hi
    ):
        return None

    original = {
        id_lo,
        id_hi,
    }

    # The refined scalar route must describe the same chart pair
    # as the vectorized candidate edge. Otherwise this is a
    # triple-junction / routing-disagreement artefact and must
    # not enter the audit.
    expected_pair = set(
        str(row["pair_id"]).split("--")
    )

    if original != expected_pair:
        return None

    for _ in range(n_iter):
        mid = 0.5 * (
            lo + hi
        )

        idx = route_id(
            plan,
            mid[0],
            mid[1],
        )

        if idx == id_lo:
            lo = mid

        elif idx == id_hi:
            hi = mid

        else:
            # A third chart cuts this crossing.
            return None

    center = 0.5 * (
        lo + hi
    )

    return {
        "chart_minus":
            id_lo,

        "chart_plus":
            id_hi,

        "Mach_boundary":
            float(center[0]),

        "alpha_boundary":
            float(center[1]),

        "clean_pair_switch":
            {
                route_id(
                    plan,
                    lo[0],
                    lo[1],
                ),
                route_id(
                    plan,
                    hi[0],
                    hi[1],
                ),
            }
            == original,
    }


def call_spectral_audit(
    fn,
    *,
    frame: pd.DataFrame,
    model,
    device: torch.device,
):
    sig = inspect.signature(
        fn
    )

    kwargs: dict[str, Any] = {}

    for name, param in sig.parameters.items():
        lname = name.lower()

        if (
            lname in {
                "frame",
                "df",
                "data",
                "dataset",
            }
            or "frame" in lname
        ):
            kwargs[name] = frame
            continue

        if (
            lname == "model"
            or lname.endswith("_model")
        ):
            kwargs[name] = model
            continue

        if lname == "device":
            kwargs[name] = device
            continue

        if lname in {
            "include_test",
            "audit_test",
            "expose_test",
            "allow_test",
        }:
            kwargs[name] = False
            continue

        if (
            param.default
            is not inspect.Parameter.empty
        ):
            continue

        raise RuntimeError(
            "Cannot resolve spectral_audit "
            f"argument {name!r}; "
            f"signature={sig}"
        )

    return fn(
        **kwargs
    )


class ChartEvaluator:
    def __init__(
        self,
        trainer,
        device: torch.device,
    ):
        self.trainer = trainer
        self.device = device
        self.cache = {}

    def load_chart(
        self,
        chart: str,
    ):
        if chart in self.cache:
            return self.cache[
                chart
            ]

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
            map_location=self.device,
        )

        if not isinstance(
            checkpoint,
            dict,
        ):
            raise RuntimeError(
                f"{chart}: invalid checkpoint"
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
                "has no config"
            )

        model = (
            self.trainer
            .build_model(config)
            .to(self.device)
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        model.eval()

        self.cache[chart] = model

        return model

    def predict(
        self,
        chart: str,
        mach: float,
        alpha: float,
    ):
        model = self.load_chart(
            chart
        )

        # Dummy reference values are intentionally not used
        # in inference. They are supplied only because the
        # historical spectral_audit returns error columns too.
        frame = pd.DataFrame({
            "Mach":
                [float(mach)],

            "alpha":
                [float(alpha)],

            "cr":
                [0.0],

            "ci":
                [0.0],

            "mach_split":
                ["validation"],

            "point_role":
                ["validation"],

            "usable_as_training_anchor":
                [False],
        })

        with torch.no_grad():
            output = call_spectral_audit(
                self.trainer.spectral_audit,
                frame=frame,
                model=model,
                device=self.device,
            )

        if not (
            isinstance(output, tuple)
            and len(output) >= 1
        ):
            raise RuntimeError(
                f"{chart}: unexpected "
                "spectral_audit output"
            )

        pred = output[0]

        if not isinstance(
            pred,
            pd.DataFrame,
        ):
            raise RuntimeError(
                f"{chart}: prediction "
                "is not DataFrame"
            )

        if len(pred) != 1:
            raise RuntimeError(
                f"{chart}: expected one "
                f"prediction, got {len(pred)}"
            )

        for column in [
            "cr_pred",
            "ci_pred",
        ]:
            if column not in pred:
                raise RuntimeError(
                    f"{chart}: missing {column}"
                )

        return (
            float(
                pred[
                    "cr_pred"
                ].iloc[0]
            ),
            float(
                pred[
                    "ci_pred"
                ].iloc[0]
            ),
        )


def delta_metrics(
    c1,
    c2,
):
    cr1, ci1 = c1
    cr2, ci2 = c2

    dcr = abs(
        cr1 - cr2
    )

    dci = abs(
        ci1 - ci2
    )

    dc = math.hypot(
        dcr,
        dci,
    )

    return {
        "cr_abs_diff":
            dcr,

        "ci_abs_diff":
            dci,

        "c_abs_diff":
            dc,
    }


def side_points(
    mach0: float,
    alpha0: float,
    axis: str,
    eps: float,
):
    if axis == "Mach":
        return (
            (
                mach0 - eps,
                alpha0,
            ),
            (
                mach0 + eps,
                alpha0,
            ),
        )

    if axis == "alpha":
        return (
            (
                mach0,
                alpha0 - eps,
            ),
            (
                mach0,
                alpha0 + eps,
            ),
        )

    raise ValueError(
        axis
    )


def q(values, p):
    arr = np.asarray(
        values,
        dtype=float,
    )

    arr = arr[
        np.isfinite(arr)
    ]

    if len(arr) == 0:
        return np.nan

    return float(
        np.quantile(
            arr,
            p,
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--grid-n",
        type=int,
        default=301,
    )

    parser.add_argument(
        "--max-per-interface",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[
            1.0e-6,
            1.0e-5,
            1.0e-4,
        ],
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "assets/pinn_supersonic/"
            "atlas2d_v1_continuousM_fullc_v1/"
            "N76/routing_boundary_audit"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    if not TRAINER_PATH.is_file():
        raise FileNotFoundError(
            TRAINER_PATH
        )

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    plan = build_plan()

    print("=" * 100)
    print(
        "PHASE 9 — FULLC ROUTING-BOUNDARY AUDIT"
    )
    print("=" * 100)

    print()
    print(plan[
        [
            "chart_id",
            "mach_min",
            "mach_max",
            "alpha_min",
            "alpha_max",
            "chart_area",
        ]
    ].to_string(index=False))

    raw = detect_raw_edges(
        plan,
        args.grid_n,
    )

    sampled = sample_edges(
        raw,
        max_per_interface=
            args.max_per_interface,
        grid_n=args.grid_n,
        plan=plan,
    )

    refined_rows = []

    for _, row in sampled.iterrows():
        refined = refine_boundary(
            plan,
            row,
        )

        if refined is None:
            continue

        merged = row.to_dict()
        merged.update(
            refined
        )

        refined_rows.append(
            merged
        )

    boundaries = pd.DataFrame(
        refined_rows
    )

    if boundaries.empty:
        raise RuntimeError(
            "No clean routing boundary."
        )

    boundaries = (
        boundaries
        .drop_duplicates(
            subset=[
                "normal_axis",
                "chart_minus",
                "chart_plus",
                "Mach_boundary",
                "alpha_boundary",
            ]
        )
        .reset_index(drop=True)
    )

    boundaries.insert(
        0,
        "boundary_id",
        [
            f"ROUTE_{i:04d}"
            for i in range(
                len(boundaries)
            )
        ],
    )

    print()
    print(
        "raw changed grid edges :",
        len(raw),
    )
    print(
        "representative samples :",
        len(sampled),
    )
    print(
        "refined boundaries     :",
        len(boundaries),
    )
    print(
        "unique chart pairs     :",
        boundaries[
            "pair_id"
        ].nunique(),
    )

    print()
    print(
        boundaries[
            [
                "boundary_id",
                "pair_id",
                "normal_axis",
                "chart_minus",
                "chart_plus",
                "Mach_boundary",
                "alpha_boundary",
            ]
        ].to_string(
            index=False
        )
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    boundaries.to_csv(
        output_dir
        / "routing_boundaries.csv",
        index=False,
    )

    raw.to_csv(
        output_dir
        / "all_detected_routing_edges.csv",
        index=False,
    )

    if args.dry_run:
        print()
        print(
            "DRY RUN — no neural "
            "evaluation performed."
        )
        return

    trainer = load_module(
        TRAINER_PATH,
        "fullc_routing_trainer",
    )

    if not hasattr(
        trainer,
        "build_model",
    ):
        raise RuntimeError(
            "Trainer has no build_model"
        )

    if not hasattr(
        trainer,
        "spectral_audit",
    ):
        raise RuntimeError(
            "Trainer has no spectral_audit"
        )

    evaluator = ChartEvaluator(
        trainer,
        device,
    )

    results = []

    for _, boundary in (
        boundaries.iterrows()
    ):
        mach0 = float(
            boundary[
                "Mach_boundary"
            ]
        )

        alpha0 = float(
            boundary[
                "alpha_boundary"
            ]
        )

        chart_a = str(
            boundary[
                "chart_minus"
            ]
        )

        chart_b = str(
            boundary[
                "chart_plus"
            ]
        )

        print()
        print(
            boundary["boundary_id"],
            chart_a,
            "<->",
            chart_b,
            f"M={mach0:.10f}",
            f"alpha={alpha0:.10f}",
            flush=True,
        )

        # ----------------------------------------------------------
        # A. Two adjacent chart networks at EXACTLY same point.
        # ----------------------------------------------------------

        pred_a = evaluator.predict(
            chart_a,
            mach0,
            alpha0,
        )

        pred_b = evaluator.predict(
            chart_b,
            mach0,
            alpha0,
        )

        same = delta_metrics(
            pred_a,
            pred_b,
        )

        # ----------------------------------------------------------
        # B. Actual routed prediction x-eps -> x+eps.
        # ----------------------------------------------------------

        for eps in args.eps:
            (
                point_minus,
                point_plus,
            ) = side_points(
                mach0,
                alpha0,
                str(
                    boundary[
                        "normal_axis"
                    ]
                ),
                float(eps),
            )

            (
                mach_minus,
                alpha_minus,
            ) = point_minus

            (
                mach_plus,
                alpha_plus,
            ) = point_plus

            route_minus = route_chart(
                plan,
                mach_minus,
                alpha_minus,
            )

            route_plus = route_chart(
                plan,
                mach_plus,
                alpha_plus,
            )

            if (
                route_minus is None
                or route_plus is None
            ):
                continue

            chart_minus = str(
                route_minus[
                    "chart_id"
                ]
            )

            chart_plus = str(
                route_plus[
                    "chart_id"
                ]
            )

            pred_minus = evaluator.predict(
                chart_minus,
                mach_minus,
                alpha_minus,
            )

            pred_plus = evaluator.predict(
                chart_plus,
                mach_plus,
                alpha_plus,
            )

            routed = delta_metrics(
                pred_minus,
                pred_plus,
            )

            row = {
                "boundary_id":
                    boundary[
                        "boundary_id"
                    ],

                "pair_id":
                    boundary[
                        "pair_id"
                    ],

                "normal_axis":
                    boundary[
                        "normal_axis"
                    ],

                "Mach_boundary":
                    mach0,

                "alpha_boundary":
                    alpha0,

                "chart_a":
                    chart_a,

                "chart_b":
                    chart_b,

                "samepoint_cr_a":
                    pred_a[0],

                "samepoint_ci_a":
                    pred_a[1],

                "samepoint_cr_b":
                    pred_b[0],

                "samepoint_ci_b":
                    pred_b[1],

                "samepoint_cr_abs_diff":
                    same[
                        "cr_abs_diff"
                    ],

                "samepoint_ci_abs_diff":
                    same[
                        "ci_abs_diff"
                    ],

                "samepoint_c_abs_diff":
                    same[
                        "c_abs_diff"
                    ],

                "eps":
                    float(eps),

                "Mach_minus":
                    mach_minus,

                "alpha_minus":
                    alpha_minus,

                "Mach_plus":
                    mach_plus,

                "alpha_plus":
                    alpha_plus,

                "routed_chart_minus":
                    chart_minus,

                "routed_chart_plus":
                    chart_plus,

                "routed_cr_minus":
                    pred_minus[0],

                "routed_ci_minus":
                    pred_minus[1],

                "routed_cr_plus":
                    pred_plus[0],

                "routed_ci_plus":
                    pred_plus[1],

                "routed_cr_abs_diff":
                    routed[
                        "cr_abs_diff"
                    ],

                "routed_ci_abs_diff":
                    routed[
                        "ci_abs_diff"
                    ],

                "routed_c_abs_diff":
                    routed[
                        "c_abs_diff"
                    ],

                "route_changes":
                    (
                        chart_minus
                        != chart_plus
                    ),
            }

            results.append(
                row
            )

    result = pd.DataFrame(
        results
    )

    if result.empty:
        raise RuntimeError(
            "No routing audit results."
        )

    result.to_csv(
        output_dir
        / "routing_boundary_results.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    # same-point metrics are repeated for the three eps;
    # count each boundary once.
    same_unique = (
        result
        .sort_values("eps")
        .drop_duplicates(
            subset=["boundary_id"]
        )
    )

    summary = {
        "n_boundaries":
            int(
                same_unique[
                    "boundary_id"
                ].nunique()
            ),

        "n_unique_chart_pairs":
            int(
                same_unique[
                    "pair_id"
                ].nunique()
            ),

        "samepoint_c_median":
            float(
                same_unique[
                    "samepoint_c_abs_diff"
                ].median()
            ),

        "samepoint_c_p95":
            q(
                same_unique[
                    "samepoint_c_abs_diff"
                ],
                0.95,
            ),

        "samepoint_c_max":
            float(
                same_unique[
                    "samepoint_c_abs_diff"
                ].max()
            ),

        "samepoint_cr_median":
            float(
                same_unique[
                    "samepoint_cr_abs_diff"
                ].median()
            ),

        "samepoint_cr_p95":
            q(
                same_unique[
                    "samepoint_cr_abs_diff"
                ],
                0.95,
            ),

        "samepoint_cr_max":
            float(
                same_unique[
                    "samepoint_cr_abs_diff"
                ].max()
            ),

        "samepoint_ci_median":
            float(
                same_unique[
                    "samepoint_ci_abs_diff"
                ].median()
            ),

        "samepoint_ci_p95":
            q(
                same_unique[
                    "samepoint_ci_abs_diff"
                ],
                0.95,
            ),

        "samepoint_ci_max":
            float(
                same_unique[
                    "samepoint_ci_abs_diff"
                ].max()
            ),
    }

    routed_by_eps = []

    for eps, group in result.groupby(
        "eps",
        sort=True,
    ):
        routed_by_eps.append({
            "eps":
                float(eps),

            "n":
                int(len(group)),

            "route_changes":
                int(
                    group[
                        "route_changes"
                    ].sum()
                ),

            "routed_c_median":
                float(
                    group[
                        "routed_c_abs_diff"
                    ].median()
                ),

            "routed_c_p95":
                q(
                    group[
                        "routed_c_abs_diff"
                    ],
                    0.95,
                ),

            "routed_c_max":
                float(
                    group[
                        "routed_c_abs_diff"
                    ].max()
                ),

            "routed_cr_median":
                float(
                    group[
                        "routed_cr_abs_diff"
                    ].median()
                ),

            "routed_cr_p95":
                q(
                    group[
                        "routed_cr_abs_diff"
                    ],
                    0.95,
                ),

            "routed_cr_max":
                float(
                    group[
                        "routed_cr_abs_diff"
                    ].max()
                ),

            "routed_ci_median":
                float(
                    group[
                        "routed_ci_abs_diff"
                    ].median()
                ),

            "routed_ci_p95":
                q(
                    group[
                        "routed_ci_abs_diff"
                    ],
                    0.95,
                ),

            "routed_ci_max":
                float(
                    group[
                        "routed_ci_abs_diff"
                    ].max()
                ),
        })

    routed_summary = pd.DataFrame(
        routed_by_eps
    )

    routed_summary.to_csv(
        output_dir
        / "routing_summary_by_eps.csv",
        index=False,
    )

    (
        output_dir
        / "routing_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    worst = (
        same_unique
        .sort_values(
            "samepoint_c_abs_diff",
            ascending=False,
        )
        .head(20)
    )

    worst.to_csv(
        output_dir
        / "routing_worst_boundaries.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("PHASE 9 — SAME-POINT CHART DISAGREEMENT")
    print("=" * 100)

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("=" * 100)
    print("PHASE 9 — ACTUAL ROUTED JUMPS")
    print("=" * 100)

    print(
        routed_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8e}",
        )
    )

    print()
    print("=" * 100)
    print("WORST SAME-POINT BOUNDARIES")
    print("=" * 100)

    print(
        worst[
            [
                "boundary_id",
                "pair_id",
                "normal_axis",
                "Mach_boundary",
                "alpha_boundary",
                "samepoint_cr_abs_diff",
                "samepoint_ci_abs_diff",
                "samepoint_c_abs_diff",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8e}",
        )
    )


if __name__ == "__main__":
    main()
