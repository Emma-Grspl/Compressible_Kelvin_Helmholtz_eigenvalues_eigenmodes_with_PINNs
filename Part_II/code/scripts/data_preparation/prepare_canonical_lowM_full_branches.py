#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BASE_LOW_M_ROOT = Path(
    "classic_supersonic/reproducibility/results/"
    "dense_kappa_q_lowM_full_branches_v1"
)
M105_REPAIR_ROOT = Path(
    "classic_supersonic/reproducibility/results/"
    "dense_kappa_q_M105_full_branch_repair_v1"
)
OUTPUT_ROOT = Path(
    "classic_supersonic/reproducibility/results/"
    "dense_kappa_q_lowM_canonical_full_branches_v1"
)


def resolve(repo: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo / value


def numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def load_retained(root: Path, Mach: float) -> pd.DataFrame:
    path = root / "dense_spectral_retained.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = numeric(
        pd.read_csv(path),
        ("Mach", "alpha", "cr", "ci", "omega_i", "residual_norm"),
    )
    frame = frame[
        np.isclose(frame["Mach"], Mach, rtol=0.0, atol=5e-10)
        & np.isfinite(frame["alpha"])
        & np.isfinite(frame["cr"])
        & np.isfinite(frame["ci"])
        & (frame["ci"] > 0.0)
    ].copy()
    if frame.empty:
        raise RuntimeError(f"No retained branch found for M={Mach} in {path}")
    frame = (
        frame.sort_values("alpha")
        .drop_duplicates("alpha", keep="last")
        .reset_index(drop=True)
    )
    if "omega_i" not in frame.columns:
        frame["omega_i"] = frame["alpha"] * frame["ci"]
    else:
        frame["omega_i"] = frame["omega_i"].fillna(
            frame["alpha"] * frame["ci"]
        )
    return frame


def neutral_bracket(root: Path, mach_dir_name: str) -> dict[str, float]:
    path = root / mach_dir_name / "spectral_points.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = numeric(
        pd.read_csv(path),
        ("alpha", "ci", "residual_norm"),
    )
    status = frame["status"].astype(str)
    converged = frame[
        status.isin(("converged", "anchor_converged"))
        & np.isfinite(frame["alpha"])
        & np.isfinite(frame["ci"])
        & (frame["ci"] > 0.0)
    ].sort_values("alpha")
    if converged.empty:
        raise RuntimeError(f"No converged points in {path}")
    lower = float(converged["alpha"].max())
    last_ci = float(converged.iloc[-1]["ci"])
    failed = frame[
        status.isin(
            ("rejected", "stable_beyond_neutral", "not_reached_after_failure")
        )
        & np.isfinite(frame["alpha"])
        & (frame["alpha"] > lower)
    ].sort_values("alpha")
    if failed.empty:
        raise RuntimeError(f"No failed point above last converged point in {path}")
    upper = float(failed.iloc[0]["alpha"])
    if not upper > lower:
        raise RuntimeError(f"Invalid neutral bracket in {path}: {lower}, {upper}")
    return {
        "neutral_alpha_lower": lower,
        "neutral_alpha_upper": upper,
        "neutral_alpha_estimate": 0.5 * (lower + upper),
        "neutral_alpha_uncertainty": 0.5 * (upper - lower),
        "neutral_last_positive_ci": last_ci,
    }


def validate_regular_branch(frame: pd.DataFrame, Mach: float) -> None:
    expected = np.round(
        np.arange(0.05, float(frame["alpha"].max()) + 1e-12, 0.005),
        12,
    )
    observed = set(np.round(frame["alpha"].to_numpy(float), 12))
    missing = [float(a) for a in expected if round(float(a), 12) not in observed]
    if missing:
        raise RuntimeError(f"M={Mach}: missing regular alpha points: {missing}")
    required_low = np.round(np.arange(0.05, 0.1000001, 0.005), 12)
    missing_low = [
        float(a) for a in required_low if round(float(a), 12) not in observed
    ]
    if missing_low:
        raise RuntimeError(f"M={Mach}: missing alpha in [0.05,0.10]: {missing_low}")
    maximum_residual = float(
        pd.to_numeric(frame["residual_norm"], errors="coerce").max()
    )
    if maximum_residual > 1e-8:
        raise RuntimeError(
            f"M={Mach}: max residual {maximum_residual:.3e} > 1e-8"
        )


def copy_mach_directory(source_root: Path, target_root: Path, name: str) -> None:
    source = source_root / name
    target = target_root / name
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, target, dirs_exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    m100_root = resolve(repo, BASE_LOW_M_ROOT)
    m105_root = resolve(repo, M105_REPAIR_ROOT)
    output_root = resolve(repo, OUTPUT_ROOT)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_root} exists; pass --overwrite to rebuild it"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    m100 = load_retained(m100_root, 1.00)
    m105 = load_retained(m105_root, 1.05)
    validate_regular_branch(m100, 1.00)
    validate_regular_branch(m105, 1.05)

    bracket100 = neutral_bracket(m100_root, "M1p000000")
    bracket105 = neutral_bracket(m105_root, "M1p050000")

    for key, value in bracket100.items():
        m100[key] = value
    for key, value in bracket105.items():
        m105[key] = value

    m100["neutral_estimate_method"] = "last-positive/first-failed bracket"
    m105["neutral_estimate_method"] = "last-positive/first-failed bracket"
    m100["reference_subset"] = "M1.00_full_branch"
    m105["reference_subset"] = "M1.05_full_branch_repaired"
    m100["reference_source_path"] = str(m100_root)
    m105["reference_source_path"] = str(m105_root)

    combined = (
        pd.concat([m100, m105], ignore_index=True, sort=False)
        .sort_values(["Mach", "alpha"])
        .reset_index(drop=True)
    )
    if len(combined) != 125:
        raise RuntimeError(
            f"Expected 125 canonical low-M points (63+62), found {len(combined)}"
        )

    combined.to_csv(output_root / "dense_spectral_retained.csv", index=False)
    combined.to_csv(
        output_root / "lowM_canonical_full_branch_reference.csv", index=False
    )
    neutral = pd.DataFrame(
        [
            {"Mach": 1.00, **bracket100},
            {"Mach": 1.05, **bracket105},
        ]
    )
    neutral["neutral_estimate_method"] = (
        "last-positive/first-failed bracket midpoint"
    )
    neutral.to_csv(output_root / "classical_neutral_boundary_lowM.csv", index=False)

    copy_mach_directory(m100_root, output_root, "M1p000000")
    copy_mach_directory(m105_root, output_root, "M1p050000")

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "n_points": int(len(combined)),
        "n_M100": int(len(m100)),
        "n_M105": int(len(m105)),
        "M100_alpha_min": float(m100["alpha"].min()),
        "M100_alpha_max": float(m100["alpha"].max()),
        "M105_alpha_min": float(m105["alpha"].min()),
        "M105_alpha_max": float(m105["alpha"].max()),
        "neutral_M100": bracket100,
        "neutral_M105": bracket105,
        "source_M100": str(m100_root),
        "source_M105": str(m105_root),
    }
    (output_root / "canonical_lowM_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== CANONICAL LOW-M FULL BRANCHES ===")
    print(f"M=1.00 points : {len(m100)}")
    print(f"M=1.05 points : {len(m105)}")
    print(f"Total         : {len(combined)}")
    print(
        "Neutral M=1.00: "
        f"[{bracket100['neutral_alpha_lower']:.12f}, "
        f"{bracket100['neutral_alpha_upper']:.12f}]"
    )
    print(
        "Neutral M=1.05: "
        f"[{bracket105['neutral_alpha_lower']:.12f}, "
        f"{bracket105['neutral_alpha_upper']:.12f}]"
    )
    print(f"Written to    : {output_root}")
    print("PREP STATUS   : PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
