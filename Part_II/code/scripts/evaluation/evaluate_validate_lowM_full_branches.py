#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def decimal_grid(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step))
    return np.round(start + step * np.arange(count + 1), 12)


def key_frame(frame: pd.DataFrame) -> set[tuple[float, float]]:
    return {
        (round(float(M), 10), round(float(a), 12))
        for M, a in frame[["Mach", "alpha"]].itertuples(index=False)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repo / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))

    root = Path(config["output_root"])
    if not root.is_absolute():
        root = repo / root

    retained_path = root / "dense_spectral_retained.csv"
    targets_path = root / "dense_spectral_targets.csv"
    modes_path = root / "all_mode_summaries.csv"
    for path in (retained_path, targets_path, modes_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    retained = pd.read_csv(retained_path)
    targets = pd.read_csv(targets_path)
    modes = pd.read_csv(modes_path)

    for frame in (retained, targets, modes):
        frame["Mach"] = pd.to_numeric(frame["Mach"], errors="coerce")
        frame["alpha"] = pd.to_numeric(frame["alpha"], errors="coerce")

    retained["cr"] = pd.to_numeric(retained["cr"], errors="coerce")
    retained["ci"] = pd.to_numeric(retained["ci"], errors="coerce")
    retained["residual_norm"] = pd.to_numeric(
        retained["residual_norm"], errors="coerce"
    )
    retained = retained.dropna(
        subset=["Mach", "alpha", "cr", "ci", "residual_norm"]
    ).copy()
    modes = modes.dropna(subset=["Mach", "alpha"]).copy()

    required_low = decimal_grid(0.05, 0.10, 0.005)
    failures: list[str] = []
    summaries: list[dict[str, object]] = []

    for Mach in (1.0, 1.05):
        sub = retained[np.isclose(retained["Mach"], Mach, atol=5e-10)].copy()
        sub = sub.sort_values("alpha").drop_duplicates("alpha", keep="last")
        observed = set(np.round(sub["alpha"].to_numpy(float), 12))
        missing_low = [
            float(a) for a in required_low if round(float(a), 12) not in observed
        ]
        if missing_low:
            failures.append(
                f"M={Mach}: missing retained alpha in [0.05,0.10]: {missing_low}"
            )

        if sub.empty:
            failures.append(f"M={Mach}: no retained roots")
            continue

        max_residual = float(sub["residual_norm"].max())
        if max_residual > 1e-8:
            failures.append(
                f"M={Mach}: max residual {max_residual:.3e} > 1e-8"
            )

        # Require a contiguous target grid from 0.05 to the final retained point.
        maximum = float(sub["alpha"].max())
        expected_contiguous = decimal_grid(0.05, maximum, 0.005)
        missing_contiguous = [
            float(a)
            for a in expected_contiguous
            if round(float(a), 12) not in observed
        ]
        if missing_contiguous:
            failures.append(
                f"M={Mach}: gaps before neutral boundary: {missing_contiguous}"
            )

        neutral_values = pd.to_numeric(
            sub.get("neutral_alpha_estimate", pd.Series(dtype=float)),
            errors="coerce",
        )
        neutral_values = neutral_values[np.isfinite(neutral_values)]
        neutral = float(neutral_values.iloc[-1]) if len(neutral_values) else math.nan
        if not np.isfinite(neutral):
            failures.append(f"M={Mach}: no neutral-alpha estimate")
        elif neutral <= maximum:
            failures.append(
                f"M={Mach}: neutral estimate {neutral} <= max retained {maximum}"
            )
        elif neutral - maximum > 0.012:
            failures.append(
                f"M={Mach}: last retained alpha is too far from neutral "
                f"({maximum} vs {neutral})"
            )

        summaries.append(
            {
                "Mach": Mach,
                "n_retained": int(len(sub)),
                "alpha_min": float(sub["alpha"].min()),
                "alpha_max": maximum,
                "neutral_alpha_estimate": neutral,
                "max_residual_norm": max_residual,
            }
        )

    retained_keys = key_frame(retained)
    mode_keys = key_frame(modes)
    missing_modes = sorted(retained_keys.difference(mode_keys))
    extra_modes = sorted(mode_keys.difference(retained_keys))
    if missing_modes:
        failures.append(
            f"Missing {len(missing_modes)} modes; first={missing_modes[:20]}"
        )
    if extra_modes:
        failures.append(
            f"Unexpected {len(extra_modes)} modes; first={extra_modes[:20]}"
        )

    reference = retained.sort_values(["Mach", "alpha"]).reset_index(drop=True)
    reference["reference_subset"] = "lowM_full_branch"
    atomic_csv(root / "lowM_full_branch_reference.csv", reference)
    pd.DataFrame(summaries).to_csv(root / "lowM_full_branch_summary.csv", index=False)

    report = {
        "status": "PASS" if not failures else "FAIL",
        "root": str(root),
        "n_retained": int(len(retained)),
        "n_modes": int(len(modes)),
        "missing_modes": int(len(missing_modes)),
        "failures": failures,
        "branches": summaries,
    }
    (root / "lowM_full_branch_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== LOW-M FULL-BRANCH VALIDATION ===")
    for item in summaries:
        print(
            f"M={item['Mach']:.2f}: retained={item['n_retained']}, "
            f"alpha=[{item['alpha_min']:.3f},{item['alpha_max']:.3f}], "
            f"neutral={item['neutral_alpha_estimate']}, "
            f"max residual={item['max_residual_norm']:.3e}"
        )
    print(f"Modes       : {len(modes)}")
    print(f"Missing modes: {len(missing_modes)}")
    print(f"STATUS      : {report['status']}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
