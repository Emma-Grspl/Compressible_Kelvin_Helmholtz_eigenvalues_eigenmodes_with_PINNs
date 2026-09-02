#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.evaluation.run_dense_supersonic_campaign as campaign
import scripts.evaluation.test_kappa_q_modulus_reconstruction as base


def numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def robust_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["root_tolerance"] = min(float(config.get("root_tolerance", 1e-8)), 1e-8)
    config["ci_floor"] = min(float(config.get("ci_floor", 1e-12)), 1e-12)
    config["ci_upper"] = max(float(config.get("ci_upper", 0.20)), 0.30)
    config["cr_half_width"] = max(float(config.get("cr_half_width", 0.08)), 0.15)
    config["ci_factor"] = max(float(config.get("ci_factor", 100.0)), 500.0)
    config["max_nfev"] = max(int(config.get("max_nfev", 100)), 180)
    config["direct_ci_switch"] = max(float(config.get("direct_ci_switch", 1e-3)), 2e-3)
    config["direct_ci_scale_floor"] = min(
        float(config.get("direct_ci_scale_floor", 1e-4)), 1e-6
    )
    return config


def seed_candidates(
    reference: pd.DataFrame,
    *,
    Mach: float,
    alpha: float,
    blumen_ci: float,
) -> list[tuple[float, float, str]]:
    ref = reference.copy()
    ref = ref.loc[
        np.isfinite(ref["Mach"])
        & np.isfinite(ref["alpha"])
        & np.isfinite(ref["cr"])
        & np.isfinite(ref["ci"])
        & (ref["ci"] > 0.0)
    ].copy()

    ref["parameter_distance"] = (
        ((ref["Mach"] - Mach) / 0.05) ** 2
        + ((ref["alpha"] - alpha) / 0.02) ** 2
    )
    ci_target = max(blumen_ci, 1e-5)
    ref["ci_distance"] = np.abs(
        np.log10(np.maximum(ref["ci"], 1e-12)) - math.log10(ci_target)
    )
    ref["combined_distance"] = ref["parameter_distance"] + 0.35 * ref["ci_distance"] ** 2

    selected: list[tuple[float, float, str]] = []
    seen: set[tuple[float, float]] = set()

    def add_rows(rows: pd.DataFrame, source: str) -> None:
        for _, row in rows.iterrows():
            key = (round(float(row["cr"]), 12), round(float(row["ci"]), 12))
            if key in seen:
                continue
            seen.add(key)
            selected.append((float(row["cr"]), float(row["ci"]), source))

    add_rows(ref.nsmallest(10, "parameter_distance"), "nearest_parameter")
    add_rows(ref.nsmallest(10, "combined_distance"), "nearest_parameter_and_ci")

    nearest_mach = float(ref.iloc[(ref["Mach"] - Mach).abs().argmin()]["Mach"])
    same_mach = ref.loc[np.isclose(ref["Mach"], nearest_mach, atol=5e-10)].sort_values("alpha")
    if not same_mach.empty:
        add_rows(same_mach.head(3), "nearest_mach_low_alpha")
        add_rows(same_mach.tail(3), "nearest_mach_high_alpha")

    return selected[:24]


def reconstruct_mode(
    *,
    result: dict[str, Any],
    Mach: float,
    alpha: float,
    output_dy: float,
) -> pd.DataFrame:
    c = complex(float(result["cr"]), float(result["ci"]))
    extent = float(result["spectral_extent"])
    matching_y = float(result["matching_y"])

    left = base.integrate_branch(
        side="left",
        Mach=Mach,
        alpha=alpha,
        c=c,
        Ly=extent,
        matching_y=matching_y,
        output_dy=output_dy,
        max_step=float(result["max_step"]),
        rtol=float(result["rtol"]),
        atol=float(result["atol"]),
        method="DOP853",
    )
    right = base.integrate_branch(
        side="right",
        Mach=Mach,
        alpha=alpha,
        c=c,
        Ly=extent,
        matching_y=matching_y,
        output_dy=output_dy,
        max_step=float(result["max_step"]),
        rtol=float(result["rtol"]),
        atol=float(result["atol"]),
        method="DOP853",
    )
    frame, _ = base.reconstruct_mode(left, right)

    y = frame["y"].to_numpy(float)
    p = frame["p_real"].to_numpy(float) + 1j * frame["p_imag"].to_numpy(float)
    py = (
        frame["pprime_real_from_gamma"].to_numpy(float)
        + 1j * frame["pprime_imag_from_gamma"].to_numpy(float)
    )
    U = np.tanh(y)
    Up = 1.0 - U * U
    denominator = U - c

    rho = Mach * Mach * p
    v = 1j * py / (alpha * denominator)
    u = -p / denominator + 1j * Up * v / (alpha * denominator)

    frame["rho_real"] = rho.real
    frame["rho_imag"] = rho.imag
    frame["u_real"] = u.real
    frame["u_imag"] = u.imag
    frame["v_real"] = v.real
    frame["v_imag"] = v.imag
    return frame


def best_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        attempt
        for attempt in attempts
        if np.isfinite(float(attempt.get("residual_norm", math.nan)))
        and np.isfinite(float(attempt.get("cr", math.nan)))
        and np.isfinite(float(attempt.get("ci", math.nan)))
    ]
    return min(valid, key=lambda item: float(item["residual_norm"])) if valid else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-dy", type=float, default=0.025)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    selected = manifest.loc[
        pd.to_numeric(manifest["task_index"], errors="coerce") == args.task_index
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one Blumen target for task {args.task_index}; found {len(selected)}"
        )
    target = selected.iloc[0]

    Mach = float(target["Mach"])
    alpha = float(target["alpha"])
    blumen_ci = float(target["blumen_ci"])

    reference = numeric(
        pd.read_csv(args.reference_csv),
        ("Mach", "alpha", "cr", "ci", "residual_norm"),
    ).dropna(subset=["Mach", "alpha", "cr", "ci"])
    config = robust_config(args.config)

    point_root = args.output_root.resolve() / f"point_{args.task_index:03d}"
    point_root.mkdir(parents=True, exist_ok=True)

    all_attempts: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None
    accepted_seed_source = ""

    for seed_cr, seed_ci, seed_source in seed_candidates(
        reference, Mach=Mach, alpha=alpha, blumen_ci=blumen_ci
    ):
        result, attempts = campaign.solve_with_fallbacks(
            Mach=Mach,
            alpha=alpha,
            seed_cr=seed_cr,
            seed_ci=max(seed_ci, 1e-10),
            config=config,
        )
        for attempt in attempts:
            attempt = dict(attempt)
            attempt["seed_source"] = seed_source
            all_attempts.append(attempt)
        if result is not None:
            accepted = dict(result)
            accepted_seed_source = seed_source
            break

    selected_result = accepted
    status = "accepted_root" if accepted is not None else "no_accepted_root"

    if selected_result is None and np.isclose(blumen_ci, 0.0, atol=1e-14):
        candidate = best_attempt(all_attempts)
        if candidate is not None:
            residual = float(candidate["residual_norm"])
            ci = float(candidate["ci"])
            active_lower = bool(candidate.get("active_lower_ci", False))
            if residual <= 1e-8 and (ci <= 1e-6 or active_lower):
                selected_result = candidate
                status = "neutral_limit_root"
                accepted_seed_source = str(candidate.get("seed_source", ""))

    spectral: dict[str, Any] = {
        "task_index": int(args.task_index),
        "blumen_row_id": int(target["blumen_row_id"]),
        "source_row_id": int(target["source_row_id"]),
        "curve_id": str(target.get("curve_id", "")),
        "curve_label": str(target.get("curve_label", "")),
        "curve_key": str(target.get("curve_key", "")),
        "family": str(target.get("family", "")),
        "Mach": Mach,
        "alpha": alpha,
        "blumen_ci": blumen_ci,
        "status": status,
        "accepted_seed_source": accepted_seed_source,
        "n_attempts": len(all_attempts),
    }

    mode_status = "not_reconstructed"
    if selected_result is not None:
        classical_cr = float(selected_result["cr"])
        classical_ci = float(selected_result["ci"])
        spectral.update(
            {
                "classical_cr": classical_cr,
                "classical_ci": classical_ci,
                "classical_omega_i": alpha * classical_ci,
                "delta_ci": classical_ci - blumen_ci,
                "residual_norm": float(selected_result["residual_norm"]),
                "settings_name": str(selected_result.get("settings_name", "")),
                "spectral_extent": float(selected_result["spectral_extent"]),
                "matching_y": float(selected_result["matching_y"]),
                "max_step": float(selected_result["max_step"]),
                "rtol": float(selected_result["rtol"]),
                "atol": float(selected_result["atol"]),
            }
        )

        try:
            mode = reconstruct_mode(
                result=selected_result,
                Mach=Mach,
                alpha=alpha,
                output_dy=args.output_dy,
            )
            mode.insert(0, "coordinate_index", np.arange(len(mode), dtype=int))
            mode.insert(0, "classical_ci", classical_ci)
            mode.insert(0, "classical_cr", classical_cr)
            mode.insert(0, "blumen_ci", blumen_ci)
            mode.insert(0, "alpha", alpha)
            mode.insert(0, "Mach", Mach)
            mode.insert(0, "curve_key", spectral["curve_key"])
            mode.insert(0, "blumen_row_id", spectral["blumen_row_id"])
            mode.to_csv(
                point_root / "mode.csv.gz",
                index=False,
                compression="gzip",
            )
            mode_status = "reconstructed"
        except Exception as exc:
            mode_status = f"failed: {type(exc).__name__}: {exc}"

    spectral["mode_status"] = mode_status
    if selected_result is None:
        spectral.update(
            {
                "classical_cr": math.nan,
                "classical_ci": math.nan,
                "classical_omega_i": math.nan,
                "delta_ci": math.nan,
                "residual_norm": math.nan,
            }
        )

    pd.DataFrame(all_attempts).to_csv(point_root / "attempts.csv", index=False)
    pd.DataFrame([spectral]).to_csv(point_root / "spectral.csv", index=False)
    (point_root / "spectral.json").write_text(
        json.dumps(campaign.json_safe(spectral), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== BLUMEN POINTWISE CLASSICAL SOLVE ===")
    print(f"Task index     : {args.task_index}")
    print(f"Mach / alpha   : {Mach} / {alpha}")
    print(f"Blumen ci      : {blumen_ci}")
    print(f"Status         : {status}")
    print(f"Classical cr   : {spectral['classical_cr']}")
    print(f"Classical ci   : {spectral['classical_ci']}")
    print(f"Residual       : {spectral['residual_norm']}")
    print(f"Mode           : {mode_status}")
    # Always return zero after writing the diagnostic row. Completeness is
    # checked by the dependent asset job.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
