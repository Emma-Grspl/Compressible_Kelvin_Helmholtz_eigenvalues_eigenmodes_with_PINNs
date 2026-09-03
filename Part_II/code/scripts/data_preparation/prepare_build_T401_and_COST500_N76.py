from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()

DATASET = (
    ROOT
    / "assets/pinn_supersonic/datasets/atlas2d_v1"
)

MANIFEST = (
    DATASET
    / "atlas2d_point_manifest.csv"
)

V64 = (
    DATASET
    / "validation_reference_64.csv"
)

T128 = (
    DATASET
    / "test_coordinates_128.csv"
)

A76 = (
    DATASET
    / "anchors_N76.csv"
)

REFERENCE = (
    ROOT / 'assets/classic_supersonic/csv/modal_reconstruction/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS/table_classical_supersonic_final_reference.csv'
)

T401_ROOT = (
    DATASET
    / "article_test_401"
)

COST500_ROOT = (
    DATASET
    / "cost500"
)


# ----------------------------------------------------------------------
# Frozen atlas geometry
# ----------------------------------------------------------------------

MACH_BANDS = {
    "M0": (1.00, 1.25),
    "M1": (1.15, 1.45),
    "M2": (1.35, 1.65),
    "M3": (1.55, 1.90),
}

ALPHA_BANDS = {
    "A0": (0.05, 0.13),
    "A1": (0.10, 0.22),
    "A2": (0.19, 0.36),
}

CHARTS = {
    f"C{i}{j}": {
        "mach_min": MACH_BANDS[f"M{i}"][0],
        "mach_max": MACH_BANDS[f"M{i}"][1],
        "alpha_min": ALPHA_BANDS[f"A{j}"][0],
        "alpha_max": ALPHA_BANDS[f"A{j}"][1],
    }
    for i in range(4)
    for j in range(3)
}


def key(
    mach: float,
    alpha: float,
) -> tuple[float, float]:
    return (
        round(float(mach), 12),
        round(float(alpha), 12),
    )


def key_set(
    frame: pd.DataFrame,
) -> set[tuple[float, float]]:
    return {
        key(m, a)
        for m, a in zip(
            frame["Mach"],
            frame["alpha"],
        )
    }


def contains(
    chart: str,
    mach: float,
    alpha: float,
) -> bool:

    c = CHARTS[chart]

    return (
        c["mach_min"] - 1e-12
        <= mach
        <= c["mach_max"] + 1e-12
        and
        c["alpha_min"] - 1e-12
        <= alpha
        <= c["alpha_max"] + 1e-12
    )


def normalized_margin(
    chart: str,
    mach: float,
    alpha: float,
) -> float:

    c = CHARTS[chart]

    dm = (
        c["mach_max"]
        - c["mach_min"]
    )

    da = (
        c["alpha_max"]
        - c["alpha_min"]
    )

    return min(
        (mach - c["mach_min"]) / dm,
        (c["mach_max"] - mach) / dm,
        (alpha - c["alpha_min"]) / da,
        (c["alpha_max"] - alpha) / da,
    )


def primary_chart(
    mach: float,
    alpha: float,
) -> str:

    candidates = [
        chart
        for chart in sorted(CHARTS)
        if contains(
            chart,
            mach,
            alpha,
        )
    ]

    if not candidates:
        raise RuntimeError(
            f"No chart contains "
            f"M={mach}, alpha={alpha}"
        )

    ranked = sorted(
        candidates,
        key=lambda chart: (
            -normalized_margin(
                chart,
                mach,
                alpha,
            ),
            chart,
        ),
    )

    return ranked[0]


def normalized_xy(
    frame: pd.DataFrame,
    chart: str,
) -> np.ndarray:

    c = CHARTS[chart]

    dm = (
        c["mach_max"]
        - c["mach_min"]
    )

    da = (
        c["alpha_max"]
        - c["alpha_min"]
    )

    return np.column_stack(
        [
            (
                frame["Mach"].to_numpy(
                    dtype=float
                )
                - c["mach_min"]
            )
            / dm,

            (
                frame["alpha"].to_numpy(
                    dtype=float
                )
                - c["alpha_min"]
            )
            / da,
        ]
    )


def farthest_point_select(
    candidates: pd.DataFrame,
    *,
    chart: str,
    n_select: int,
    existing: pd.DataFrame,
) -> pd.DataFrame:

    if n_select <= 0:
        return candidates.iloc[:0].copy()

    if len(candidates) < n_select:
        raise RuntimeError(
            f"{chart}: requested "
            f"{n_select} candidates, "
            f"only {len(candidates)} available"
        )

    work = (
        candidates
        .sort_values(
            ["Mach", "alpha"]
        )
        .reset_index(drop=True)
        .copy()
    )

    xy = normalized_xy(
        work,
        chart,
    )

    if len(existing) > 0:
        seed_xy = normalized_xy(
            existing,
            chart,
        )

        d2 = (
            (
                xy[:, None, :]
                - seed_xy[None, :, :]
            )
            ** 2
        ).sum(axis=2)

        min_d2 = d2.min(axis=1)

    else:
        # No existing test point in chart:
        # start from point closest to chart centre.
        centre = np.array(
            [0.5, 0.5]
        )

        centre_d2 = (
            (xy - centre) ** 2
        ).sum(axis=1)

        first = int(
            np.argmin(centre_d2)
        )

        min_d2 = (
            (
                xy
                - xy[first]
            )
            ** 2
        ).sum(axis=1)

        min_d2[first] = -np.inf

        selected = [first]

        while len(selected) < n_select:
            idx = int(
                np.argmax(min_d2)
            )

            selected.append(idx)

            new_d2 = (
                (
                    xy
                    - xy[idx]
                )
                ** 2
            ).sum(axis=1)

            min_d2 = np.minimum(
                min_d2,
                new_d2,
            )

            min_d2[selected] = -np.inf

        return (
            work.iloc[selected]
            .copy()
            .reset_index(drop=True)
        )

    selected: list[int] = []

    while len(selected) < n_select:

        max_value = float(
            np.max(min_d2)
        )

        candidates_idx = np.where(
            np.isclose(
                min_d2,
                max_value,
                atol=1e-15,
                rtol=0.0,
            )
        )[0]

        # Work is sorted => deterministic tie break.
        idx = int(
            candidates_idx[0]
        )

        selected.append(idx)

        new_d2 = (
            (
                xy
                - xy[idx]
            )
            ** 2
        ).sum(axis=1)

        min_d2 = np.minimum(
            min_d2,
            new_d2,
        )

        min_d2[selected] = -np.inf

    return (
        work.iloc[selected]
        .copy()
        .reset_index(drop=True)
    )


def largest_remainder_allocation(
    counts: dict[str, int],
    total: int,
) -> dict[str, int]:

    available_total = sum(
        counts.values()
    )

    if available_total < total:
        raise RuntimeError(
            f"Need {total} points but only "
            f"{available_total} are available"
        )

    exact = {
        chart:
            total
            * counts[chart]
            / available_total
        for chart in counts
    }

    allocation = {
        chart:
            int(math.floor(exact[chart]))
        for chart in counts
    }

    remainder = (
        total
        - sum(allocation.values())
    )

    order = sorted(
        counts,
        key=lambda chart: (
            -(
                exact[chart]
                - allocation[chart]
            ),
            chart,
        ),
    )

    for chart in order[:remainder]:
        allocation[chart] += 1

    for chart in allocation:
        if allocation[chart] > counts[chart]:
            raise RuntimeError(
                f"Allocation exceeds "
                f"availability in {chart}"
            )

    return allocation


def write_sha256(
    directory: Path,
) -> None:

    files = sorted(
        p
        for p in directory.rglob("*")
        if (
            p.is_file()
            and p.name != "SHA256SUMS"
        )
    )

    lines = []

    for path in files:

        digest = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        relative = path.relative_to(
            directory
        )

        lines.append(
            f"{digest}  {relative}"
        )

    (
        directory
        / "SHA256SUMS"
    ).write_text(
        "\n".join(lines)
        + "\n"
    )


for p in [
    MANIFEST,
    V64,
    T128,
    A76,
    REFERENCE,
]:
    if not p.is_file():
        raise FileNotFoundError(p)


# ======================================================================
# T401
# ======================================================================

print("=" * 100)
print("BUILDING ARTICLE TEST T401")
print("=" * 100)

manifest = pd.read_csv(
    MANIFEST
)

v64 = pd.read_csv(
    V64
)

t128 = pd.read_csv(
    T128
)

a76 = pd.read_csv(
    A76
)

for frame_name, frame in [
    ("manifest", manifest),
    ("V64", v64),
    ("T128", t128),
    ("A76", a76),
]:
    if not {
        "Mach",
        "alpha",
    }.issubset(
        frame.columns
    ):
        raise RuntimeError(
            f"{frame_name} lacks Mach/alpha"
        )


# Primary chart from the frozen geometric rule,
# independent of any reference values.
manifest = manifest.copy()

manifest[
    "_primary_geometry"
] = [
    primary_chart(
        float(m),
        float(a),
    )
    for m, a in zip(
        manifest["Mach"],
        manifest["alpha"],
    )
]


MKEY = {
    key(m, a): i
    for i, (m, a)
    in enumerate(
        zip(
            manifest["Mach"],
            manifest["alpha"],
        )
    )
}

if len(MKEY) != len(manifest):
    raise RuntimeError(
        "Manifest has duplicate coordinates"
    )


T128_KEYS = key_set(t128)
V64_KEYS = key_set(v64)
A76_KEYS = key_set(a76)

if T128_KEYS & V64_KEYS:
    raise RuntimeError(
        "Existing T128 overlaps V64"
    )

if T128_KEYS & A76_KEYS:
    raise RuntimeError(
        "Existing T128 overlaps A76"
    )


# Recover existing T128 points from manifest so
# primary chart uses exactly the frozen geometric rule.
t128_manifest_rows = []

for _, row in t128.iterrows():

    k = key(
        row["Mach"],
        row["alpha"],
    )

    if k not in MKEY:
        raise RuntimeError(
            f"T128 point absent from manifest: {k}"
        )

    t128_manifest_rows.append(
        manifest.iloc[
            MKEY[k]
        ]
    )

t128_full = pd.DataFrame(
    t128_manifest_rows
).reset_index(drop=True)


excluded = (
    T128_KEYS
    | V64_KEYS
    | A76_KEYS
)

eligible = manifest[
    [
        key(m, a) not in excluded
        for m, a in zip(
            manifest["Mach"],
            manifest["alpha"],
        )
    ]
].copy()

if len(eligible) < 273:
    raise RuntimeError(
        f"Only {len(eligible)} eligible "
        "points for 273 additions"
    )


eligible_counts = {
    chart:
        int(
            (
                eligible[
                    "_primary_geometry"
                ]
                == chart
            ).sum()
        )
    for chart in sorted(CHARTS)
}

allocation = (
    largest_remainder_allocation(
        eligible_counts,
        total=273,
    )
)

print(
    "Eligible per primary chart:",
    eligible_counts,
)

print(
    "Added-point allocation:",
    allocation,
)


added_parts = []

for chart in sorted(CHARTS):

    cand = eligible[
        eligible[
            "_primary_geometry"
        ].eq(chart)
    ].copy()

    existing = t128_full[
        t128_full[
            "_primary_geometry"
        ].eq(chart)
    ].copy()

    selected = farthest_point_select(
        cand,
        chart=chart,
        n_select=allocation[chart],
        existing=existing,
    )

    added_parts.append(
        selected
    )


added = pd.concat(
    added_parts,
    ignore_index=True,
)

if len(added) != 273:
    raise RuntimeError(
        f"Expected 273 additions, "
        f"got {len(added)}"
    )


t128_coords = pd.DataFrame(
    {
        "Mach":
            t128_full[
                "Mach"
            ].to_numpy(
                dtype=float
            ),

        "alpha":
            t128_full[
                "alpha"
            ].to_numpy(
                dtype=float
            ),

        "primary_chart":
            t128_full[
                "_primary_geometry"
            ].astype(str),

        "test_source":
            "existing_T128",
    }
)

added_coords = pd.DataFrame(
    {
        "Mach":
            added[
                "Mach"
            ].to_numpy(
                dtype=float
            ),

        "alpha":
            added[
                "alpha"
            ].to_numpy(
                dtype=float
            ),

        "primary_chart":
            added[
                "_primary_geometry"
            ].astype(str),

        "test_source":
            "geometry_extension",
    }
)


t401 = pd.concat(
    [
        t128_coords,
        added_coords,
    ],
    ignore_index=True,
)

t401 = (
    t401
    .sort_values(
        [
            "primary_chart",
            "Mach",
            "alpha",
            "test_source",
        ]
    )
    .reset_index(drop=True)
)

t401.insert(
    0,
    "test_id",
    np.arange(
        len(t401),
        dtype=int,
    ),
)


if len(t401) != 401:
    raise RuntimeError(
        f"T401 has {len(t401)} rows"
    )

if len(key_set(t401)) != 401:
    raise RuntimeError(
        "T401 contains duplicates"
    )

if key_set(t401) & V64_KEYS:
    raise RuntimeError(
        "T401 overlaps V64"
    )

if key_set(t401) & A76_KEYS:
    raise RuntimeError(
        "T401 overlaps A76"
    )


T401_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

coords_path = (
    T401_ROOT
    / "test_coordinates_401.csv"
)

t401.to_csv(
    coords_path,
    index=False,
)


# ----------------------------------------------------------------------
# Create SEALED reference file only AFTER point selection.
# Reference values play no role in point selection.
# ----------------------------------------------------------------------

reference = pd.read_csv(
    REFERENCE
)

if not {
    "Mach",
    "alpha",
}.issubset(
    reference.columns
):
    raise RuntimeError(
        "Classical reference lacks Mach/alpha"
    )

reference = reference.copy()

reference["_key"] = [
    key(m, a)
    for m, a in zip(
        reference["Mach"],
        reference["alpha"],
    )
]

if reference["_key"].duplicated().any():
    dup = reference[
        reference[
            "_key"
        ].duplicated(
            keep=False
        )
    ]

    raise RuntimeError(
        "Classical final reference "
        "contains duplicate coordinates:\n"
        + dup[
            [
                "Mach",
                "alpha",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

reference_map = {
    row["_key"]: {
        key_: value
        for key_, value in row.items()
        if key_ != "_key"
    }
    for _, row in reference.iterrows()
}

sealed_rows = []

for _, row in t401.iterrows():

    k = key(
        row["Mach"],
        row["alpha"],
    )

    if k not in reference_map:
        raise RuntimeError(
            "T401 coordinate absent from "
            f"classical reference: {k}"
        )

    ref = dict(
        reference_map[k]
    )

    ref.update(
        {
            "test_id":
                int(
                    row["test_id"]
                ),

            "primary_chart":
                str(
                    row[
                        "primary_chart"
                    ]
                ),

            "test_source":
                str(
                    row[
                        "test_source"
                    ]
                ),
        }
    )

    sealed_rows.append(ref)


sealed = pd.DataFrame(
    sealed_rows
)

sealed_path = (
    T401_ROOT
    / "test_reference_401_SEALED.csv"
)

sealed.to_csv(
    sealed_path,
    index=False,
)


# Five chunks for later hybrid evaluation.
chunks_root = (
    T401_ROOT
    / "chunks"
)

chunks_root.mkdir(
    parents=True,
    exist_ok=True,
)

for old in chunks_root.glob(
    "T401_chunk_*.csv"
):
    old.unlink()

for chunk_id, indices in enumerate(
    np.array_split(
        np.arange(401),
        5,
    )
):
    part = t401.iloc[
        indices
    ].copy()

    part.to_csv(
        chunks_root
        / f"T401_chunk_{chunk_id:02d}.csv",
        index=False,
    )


freeze = (
    T401_ROOT
    / "freeze"
)

if freeze.exists():
    shutil.rmtree(
        freeze
    )

freeze.mkdir(
    parents=True,
)

shutil.copy2(
    coords_path,
    freeze
    / coords_path.name,
)

shutil.copy2(
    sealed_path,
    freeze
    / sealed_path.name,
)

metadata = {
    "name":
        "supersonic_article_test_T401",

    "n":
        401,

    "construction":
        (
            "Existing sealed T128 plus "
            "273 deterministic geometry-only "
            "farthest-point selections from "
            "the remaining atlas manifest."
        ),

    "selection_uses_reference_values":
        False,

    "disjoint_from_validation_V64":
        True,

    "disjoint_from_anchor_N76":
        True,

    "lower_budget_disjointness":
        (
            "Guaranteed because "
            "N24⊂N36⊂N48⊂N60⊂N76."
        ),

    "selection_rule":
        (
            "Per-primary-chart deterministic "
            "farthest-point sampling in "
            "normalized local chart coordinates."
        ),
}

(
    freeze
    / "README.json"
).write_text(
    json.dumps(
        metadata,
        indent=2,
    )
    + "\n"
)

write_sha256(
    freeze
)


print()
print("T401 WRITTEN:")
print(coords_path)

print(
    "T401 primary-chart counts:"
)

print(
    t401[
        "primary_chart"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print(
    "T401 sources:"
)

print(
    t401[
        "test_source"
    ]
    .value_counts()
    .to_string()
)

print(
    "T401∩V64 =",
    len(
        key_set(t401)
        & V64_KEYS
    ),
)

print(
    "T401∩A76 =",
    len(
        key_set(t401)
        & A76_KEYS
    ),
)

print(
    "SEALED reference exists:",
    sealed_path.is_file(),
)

print(
    "NOTE: reference values were NOT "
    "printed or used for selection."
)


# ======================================================================
# COST500
# ======================================================================

print()
print("=" * 100)
print("BUILDING COST500")
print("=" * 100)

n_mach = 25
n_alpha = 20

mach_edges = np.linspace(
    1.00,
    1.90,
    n_mach + 1,
)

alpha_edges = np.linspace(
    0.05,
    0.36,
    n_alpha + 1,
)

mach_values = (
    0.5
    * (
        mach_edges[:-1]
        + mach_edges[1:]
    )
)

alpha_values = (
    0.5
    * (
        alpha_edges[:-1]
        + alpha_edges[1:]
    )
)

rows = []

benchmark_id = 0

for i_m, mach in enumerate(
    mach_values
):
    for i_a, alpha in enumerate(
        alpha_values
    ):

        chart = primary_chart(
            float(mach),
            float(alpha),
        )

        rows.append(
            {
                "benchmark_id":
                    benchmark_id,

                "mach_index":
                    i_m,

                "alpha_index":
                    i_a,

                "Mach":
                    float(mach),

                "alpha":
                    float(alpha),

                "primary_chart":
                    chart,
            }
        )

        benchmark_id += 1


cost500 = pd.DataFrame(
    rows
)

if len(cost500) != 500:
    raise RuntimeError(
        f"COST500 has {len(cost500)} points"
    )

if len(
    key_set(cost500)
) != 500:
    raise RuntimeError(
        "COST500 duplicates"
    )


COST500_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

cost_path = (
    COST500_ROOT
    / "cost500_coordinates.csv"
)

cost500.to_csv(
    cost_path,
    index=False,
)


# Five chunks x 100.
chunks_root = (
    COST500_ROOT
    / "chunks"
)

chunks_root.mkdir(
    parents=True,
    exist_ok=True,
)

for old in chunks_root.glob(
    "COST500_chunk_*.csv"
):
    old.unlink()

for chunk_id, indices in enumerate(
    np.array_split(
        np.arange(500),
        5,
    )
):
    part = cost500.iloc[
        indices
    ].copy()

    if len(part) != 100:
        raise RuntimeError(
            "Expected 100 points "
            f"in COST500 chunk {chunk_id}"
        )

    part.to_csv(
        chunks_root
        / f"COST500_chunk_{chunk_id:02d}.csv",
        index=False,
    )


freeze = (
    COST500_ROOT
    / "freeze"
)

if freeze.exists():
    shutil.rmtree(
        freeze
    )

freeze.mkdir(
    parents=True,
)

shutil.copy2(
    cost_path,
    freeze
    / cost_path.name,
)

metadata = {
    "name":
        "supersonic_cost_benchmark_COST500",

    "n":
        500,

    "grid":
        "25 Mach cell centres x 20 alpha cell centres",

    "Mach_domain":
        [1.00, 1.90],

    "alpha_domain":
        [0.05, 0.36],

    "purpose":
        (
            "Performance benchmark only: "
            "same coordinates for classical "
            "and PINN+shooting workflows."
        ),

    "primary_chart_rule":
        (
            "Containing chart with maximum "
            "normalized interior margin; "
            "lexicographic tie break."
        ),
}

(
    freeze
    / "README.json"
).write_text(
    json.dumps(
        metadata,
        indent=2,
    )
    + "\n"
)

write_sha256(
    freeze
)


print()
print("COST500 WRITTEN:")
print(cost_path)

print(
    "COST500 primary-chart counts:"
)

print(
    cost500[
        "primary_chart"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print()
print(
    "COST500 overlap with V64:",
    len(
        key_set(cost500)
        & V64_KEYS
    ),
)

print(
    "COST500 overlap with A76:",
    len(
        key_set(cost500)
        & A76_KEYS
    ),
)

print(
    "COST500 overlap with T401:",
    len(
        key_set(cost500)
        & key_set(t401)
    ),
)

print()
print("=" * 100)
print("DONE")
print("=" * 100)
