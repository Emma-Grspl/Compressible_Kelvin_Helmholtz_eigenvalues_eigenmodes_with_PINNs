from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(
    "assets/pinn_supersonic/datasets"
)

SOURCE = (
    ROOT
    / "sparse_v1"
    / "split_mach_interpolation.csv"
)

OUT = (
    ROOT
    / "atlas2d_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Atlas geometry
# ---------------------------------------------------------------------

M_BANDS = [
    ("M0", 1.00, 1.25),
    ("M1", 1.15, 1.45),
    ("M2", 1.35, 1.65),
    ("M3", 1.55, 1.90),
]

A_BANDS = [
    ("A0", 0.05, 0.13),
    ("A1", 0.10, 0.22),
    ("A2", 0.19, 0.36),
]

CHARTS = []

for m_name, m0, m1 in M_BANDS:
    for a_name, a0, a1 in A_BANDS:
        CHARTS.append(
            {
                "chart": f"C{m_name[1]}{a_name[1]}",
                "mach_band": m_name,
                "alpha_band": a_name,
                "mach_min": m0,
                "mach_max": m1,
                "alpha_min": a0,
                "alpha_max": a1,
            }
        )


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def stable_hash(mach: float, alpha: float) -> int:
    text = f"{mach:.8f}|{alpha:.8f}"
    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:16],
        16,
    )


def chart_margin(
    mach: float,
    alpha: float,
    chart: dict,
) -> float:

    m0 = chart["mach_min"]
    m1 = chart["mach_max"]

    a0 = chart["alpha_min"]
    a1 = chart["alpha_max"]

    if not (
        m0 - 1e-12 <= mach <= m1 + 1e-12
        and
        a0 - 1e-12 <= alpha <= a1 + 1e-12
    ):
        return -np.inf

    dm = m1 - m0
    da = a1 - a0

    distances = [
        (mach - m0) / dm,
        (m1 - mach) / dm,
        (alpha - a0) / da,
        (a1 - alpha) / da,
    ]

    return float(
        min(distances)
    )


def chart_membership(
    mach: float,
    alpha: float,
) -> tuple[list[str], str]:

    scores = []

    for chart in CHARTS:
        margin = chart_margin(
            mach,
            alpha,
            chart,
        )

        if np.isfinite(margin):
            scores.append(
                (
                    chart["chart"],
                    margin,
                )
            )

    if not scores:
        raise RuntimeError(
            f"No chart for M={mach}, alpha={alpha}"
        )

    scores.sort(
        key=lambda x: (
            -x[1],
            x[0],
        )
    )

    primary = scores[0][0]

    members = sorted(
        x[0]
        for x in scores
    )

    return members, primary


def allocate_counts(
    counts: pd.Series,
    total: int,
) -> dict[str, int]:

    weights = (
        counts
        / counts.sum()
    )

    raw = (
        weights
        * total
    )

    alloc = (
        np.floor(raw)
        .astype(int)
    )

    remaining = (
        total
        - int(alloc.sum())
    )

    fractional = (
        raw
        - alloc
    ).sort_values(
        ascending=False
    )

    for group in fractional.index[:remaining]:
        alloc.loc[group] += 1

    return {
        str(k): int(v)
        for k, v in alloc.items()
    }


def stratified_hash_select(
    df: pd.DataFrame,
    total: int,
) -> list[int]:

    counts = (
        df["primary_chart"]
        .value_counts()
        .sort_index()
    )

    alloc = allocate_counts(
        counts,
        total,
    )

    chosen = []

    for chart, n in alloc.items():
        sub = df[
            df["primary_chart"] == chart
        ].copy()

        sub["_hash"] = [
            stable_hash(
                float(m),
                float(a),
            )
            for m, a in zip(
                sub["Mach"],
                sub["alpha"],
            )
        ]

        sub = sub.sort_values(
            "_hash"
        )

        chosen.extend(
            sub.index[:n].tolist()
        )

    if len(chosen) != total:
        raise RuntimeError(
            f"Selection size mismatch: "
            f"{len(chosen)} != {total}"
        )

    return chosen


def normalized_xy(
    df: pd.DataFrame,
) -> np.ndarray:

    m = df["Mach"].to_numpy(
        dtype=float
    )

    a = df["alpha"].to_numpy(
        dtype=float
    )

    m = (
        (m - m.min())
        /
        max(
            m.max() - m.min(),
            1e-12,
        )
    )

    a = (
        (a - a.min())
        /
        max(
            a.max() - a.min(),
            1e-12,
        )
    )

    return np.column_stack(
        [m, a]
    )


def fps_order(
    df: pd.DataFrame,
    initial_indices: list[int],
    n_total: int,
) -> list[int]:

    xy = normalized_xy(df)

    index_to_pos = {
        idx: pos
        for pos, idx
        in enumerate(df.index)
    }

    selected = list(
        dict.fromkeys(
            initial_indices
        )
    )

    if len(selected) > n_total:
        selected = selected[:n_total]

    selected_pos = [
        index_to_pos[idx]
        for idx in selected
    ]

    dist = np.full(
        len(df),
        np.inf,
        dtype=float,
    )

    for p in selected_pos:
        d = np.sum(
            (xy - xy[p]) ** 2,
            axis=1,
        )

        dist = np.minimum(
            dist,
            d,
        )

    selected_set = set(
        selected_pos
    )

    while len(selected) < n_total:

        candidate_dist = dist.copy()

        if selected_set:
            candidate_dist[
                list(selected_set)
            ] = -np.inf

        p = int(
            np.argmax(
                candidate_dist
            )
        )

        selected_pos.append(p)

        selected.append(
            df.index[p]
        )

        selected_set.add(p)

        d = np.sum(
            (xy - xy[p]) ** 2,
            axis=1,
        )

        dist = np.minimum(
            dist,
            d,
        )

    return selected


def two_per_chart_seeds(
    df: pd.DataFrame,
) -> list[int]:

    """
    Select two maximally separated training points
    inside every primary chart.

    Purpose:
    - guarantee coverage of the chart extent already at N=24;
    - avoid a center + edge initialization;
    - keep the anchor budget purely geometric, without using cr/ci.
    """

    chosen: list[int] = []

    for chart in sorted(
        df["primary_chart"].unique()
    ):

        sub = df[
            df["primary_chart"] == chart
        ].copy()

        if len(sub) < 2:
            raise RuntimeError(
                f"{chart} has fewer than 2 train points"
            )

        xy = normalized_xy(sub)

        # Pairwise squared Euclidean distances in normalized
        # (Mach, alpha) coordinates.
        diff = (
            xy[:, None, :]
            - xy[None, :, :]
        )

        d2 = np.sum(
            diff**2,
            axis=2,
        )

        i, j = np.unravel_index(
            np.argmax(d2),
            d2.shape,
        )

        idx_i = sub.index[int(i)]
        idx_j = sub.index[int(j)]

        if idx_i == idx_j:
            raise RuntimeError(
                f"Could not find two distinct seeds for {chart}"
            )

        chosen.extend(
            [
                idx_i,
                idx_j,
            ]
        )

    if len(chosen) != 24:
        raise RuntimeError(
            f"Expected 24 initial seeds, got {len(chosen)}"
        )

    if len(set(chosen)) != len(chosen):
        raise RuntimeError(
            "Duplicate primary-chart seed detected."
        )

    return chosen


# ---------------------------------------------------------------------
# Read source
# ---------------------------------------------------------------------

df = pd.read_csv(
    SOURCE
)

required = {
    "Mach",
    "alpha",
    "cr",
    "ci",
}

missing = (
    required
    - set(df.columns)
)

if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )

df = (
    df.sort_values(
        ["Mach", "alpha"]
    )
    .drop_duplicates(
        ["Mach", "alpha"]
    )
    .reset_index(drop=True)
)


# ---------------------------------------------------------------------
# Chart assignment
# ---------------------------------------------------------------------

all_charts = []
primary = []

for row in df.itertuples(
    index=False
):

    members, p = chart_membership(
        float(row.Mach),
        float(row.alpha),
    )

    all_charts.append(
        ";".join(members)
    )

    primary.append(p)

df["charts"] = all_charts
df["primary_chart"] = primary


# ---------------------------------------------------------------------
# Fixed 2D split
# ---------------------------------------------------------------------

df["point_role"] = "train_pool"

test_idx = stratified_hash_select(
    df,
    total=128,
)

df.loc[
    test_idx,
    "point_role"
] = "test"


remaining = df[
    df["point_role"]
    == "train_pool"
].copy()

val_idx = stratified_hash_select(
    remaining,
    total=64,
)

df.loc[
    val_idx,
    "point_role"
] = "validation"


# ---------------------------------------------------------------------
# Nested anchor ordering
# ---------------------------------------------------------------------

train = df[
    df["point_role"]
    == "train_pool"
].copy()

initial = two_per_chart_seeds(
    train
)

if len(initial) != 24:
    raise RuntimeError(
        f"Expected 24 initial chart anchors, "
        f"got {len(initial)}"
    )

anchor_order = fps_order(
    train,
    initial_indices=initial,
    n_total=76,
)

budgets = [
    24,
    36,
    48,
    60,
    76,
]

for n in budgets:
    df[
        f"is_anchor_N{n}"
    ] = False

    df.loc[
        anchor_order[:n],
        f"is_anchor_N{n}"
    ] = True


# ---------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------

assert (
    df["point_role"]
    == "test"
).sum() == 128

assert (
    df["point_role"]
    == "validation"
).sum() == 64

for n in budgets:

    mask = df[
        f"is_anchor_N{n}"
    ]

    assert mask.sum() == n

    assert not (
        mask
        &
        df["point_role"].isin(
            ["validation", "test"]
        )
    ).any()

for n0, n1 in zip(
    budgets[:-1],
    budgets[1:],
):

    a0 = set(
        df.index[
            df[f"is_anchor_N{n0}"]
        ]
    )

    a1 = set(
        df.index[
            df[f"is_anchor_N{n1}"]
        ]
    )

    assert a0.issubset(a1)


# ---------------------------------------------------------------------
# Write files
# ---------------------------------------------------------------------

with open(
    OUT / "charts.json",
    "w",
) as f:

    json.dump(
        CHARTS,
        f,
        indent=2,
    )


df.to_csv(
    OUT
    / "atlas2d_point_manifest.csv",
    index=False,
)


df[
    df["point_role"]
    == "validation"
].to_csv(
    OUT
    / "validation_reference_64.csv",
    index=False,
)


# Coordinates visible during development.
df.loc[
    df["point_role"] == "test",
    [
        "Mach",
        "alpha",
        "charts",
        "primary_chart",
    ],
].to_csv(
    OUT
    / "test_coordinates_128.csv",
    index=False,
)


# Full classical test references: sealed until final evaluation.
df[
    df["point_role"]
    == "test"
].to_csv(
    OUT
    / "test_reference_128_SEALED.csv",
    index=False,
)


for n in budgets:

    df[
        df[
            f"is_anchor_N{n}"
        ]
    ].to_csv(
        OUT
        / f"anchors_N{n}.csv",
        index=False,
    )


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print()
print(
    "================ ATLAS 2D V1 ================"
)

print(
    "total points:",
    len(df),
)

print(
    "validation:",
    (df["point_role"] == "validation").sum(),
)

print(
    "test:",
    (df["point_role"] == "test").sum(),
)

print(
    "train pool:",
    (df["point_role"] == "train_pool").sum(),
)

print()

print(
    "PRIMARY CHART COUNTS"
)

print(
    df["primary_chart"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()

for n in budgets:

    sub = df[
        df[f"is_anchor_N{n}"]
    ]

    print(
        f"N={n}:",
        len(sub),
        "anchors | Mach coverage =",
        sub["Mach"].nunique(),
        "| primary charts =",
        sub["primary_chart"].nunique(),
    )

print()

print(
    "Written to:",
    OUT,
)

print(
    "TEST REFERENCES ARE SEALED."
)
