#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, solve_ivp

import scripts.evaluation.run_dense_supersonic_campaign as campaign
import scripts.evaluation.test_kappa_q_modulus_reconstruction as base


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Second pass for a completed dense spectral campaign. Reconstructs "
            "only accepted target modes on a compact numerical interval, extends "
            "both far fields analytically, stores each point immediately in one "
            "SQLite checkpoint per Mach, resumes automatically, and exports one "
            "compressed NPZ per Mach."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--Mach", type=float, default=None)
    parser.add_argument("--mach-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_mach(args: argparse.Namespace, config: dict[str, Any]) -> float:
    if args.Mach is not None:
        return float(args.Mach)
    index = args.mach_index
    if index is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise ValueError("Provide --Mach, --mach-index, or SLURM_ARRAY_TASK_ID.")
        index = int(raw)
    values = [float(value) for value in config["mach_values"]]
    if index < 0 or index >= len(values):
        raise IndexError(index)
    return values[index]


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def logmode_rhs(
    y: float,
    state: np.ndarray,
    *,
    Mach: float,
    alpha: float,
    c: complex,
) -> np.ndarray:
    gamma_derivative = campaign.gamma_rhs(
        y,
        state[:2],
        Mach=Mach,
        alpha=alpha,
        c=c,
    )
    return np.asarray(
        [gamma_derivative[0], gamma_derivative[1], float(state[0])],
        dtype=float,
    )


def integrate_branch(
    *,
    side: str,
    Mach: float,
    alpha: float,
    c: complex,
    extent: float,
    matching_y: float,
    max_step: float,
    rtol: float,
    atol: float,
    method: str,
):
    start = -extent if side == "left" else extent
    gamma0 = base.asymptotic_gamma(side=side, Mach=Mach, alpha=alpha, c=c)
    solution = solve_ivp(
        lambda y, state: logmode_rhs(
            y,
            state,
            Mach=Mach,
            alpha=alpha,
            c=c,
        ),
        (start, matching_y),
        np.asarray([gamma0.real, gamma0.imag, 0.0], dtype=float),
        method=method,
        dense_output=True,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"{side} mode integration failed: {solution.message}")
    return solution


def reconstruct_mode(
    *,
    row: pd.Series,
    Mach: float,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    alpha = float(row["alpha"])
    c = complex(float(row["cr"]), float(row["ci"]))
    extent = float(config.get("modal_numerical_extent", 40.0))
    dy = float(config.get("modal_dy", 0.025))
    tail_tolerance = float(config.get("tail_tolerance", 1.0e-6))
    n_tail = int(config.get("analytic_tail_points", 300))
    method = str(config.get("method", "DOP853"))
    matching_y = float(row.get("matching_y", config.get("modal_matching_y", 1.0)))
    max_step = float(config.get("modal_max_step", 0.25))
    rtol = float(config.get("modal_rtol", 1.0e-10))
    atol = float(config.get("modal_atol", 1.0e-12))

    if not (-extent < matching_y < extent):
        raise ValueError(f"matching_y={matching_y} outside modal interval ±{extent}.")

    left = integrate_branch(
        side="left",
        Mach=Mach,
        alpha=alpha,
        c=c,
        extent=extent,
        matching_y=matching_y,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
        method=method,
    )
    right = integrate_branch(
        side="right",
        Mach=Mach,
        alpha=alpha,
        c=c,
        extent=extent,
        matching_y=matching_y,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
        method=method,
    )

    left_match = left.sol(matching_y)
    right_match = right.sol(matching_y)
    right_ell_shift = float(left_match[2] - right_match[2])

    n_intervals = max(4, int(math.ceil(2.0 * extent / dy)))
    y_numeric = np.linspace(-extent, extent, n_intervals + 1)
    if not np.any(np.isclose(y_numeric, matching_y, atol=1.0e-13)):
        y_numeric = np.sort(np.unique(np.append(y_numeric, matching_y)))

    left_mask = y_numeric <= matching_y
    right_mask = ~left_mask
    values = np.empty((3, y_numeric.size), dtype=float)
    values[:, left_mask] = left.sol(y_numeric[left_mask])
    values[:, right_mask] = right.sol(y_numeric[right_mask])
    values[2, right_mask] += right_ell_shift

    ell_max = float(
        max(
            np.max(left.y[2]),
            np.max(right.y[2] + right_ell_shift),
            np.max(values[2]),
        )
    )
    kappa_numeric = values[0]
    q_numeric = values[1]
    log_modulus_numeric = values[2] - ell_max
    modulus_numeric = np.exp(np.clip(log_modulus_numeric, -745.0, 0.0))
    cumulative = cumulative_trapezoid(q_numeric, y_numeric, initial=0.0)
    match_index = int(np.argmin(np.abs(y_numeric - matching_y)))
    phase_numeric = cumulative - cumulative[match_index]
    pressure_numeric = modulus_numeric * np.exp(1j * phase_numeric)

    gamma_left = base.asymptotic_gamma(
        side="left",
        Mach=Mach,
        alpha=alpha,
        c=c,
    )
    gamma_right = base.asymptotic_gamma(
        side="right",
        Mach=Mach,
        alpha=alpha,
        c=c,
    )
    mu_left = float(gamma_left.real)
    mu_right = float(-gamma_right.real)
    if mu_left <= 0.0 or mu_right <= 0.0:
        raise RuntimeError(
            f"Non-decaying far field: mu_left={mu_left}, mu_right={mu_right}."
        )

    target_log = math.log(tail_tolerance)
    left_interface_log = float(log_modulus_numeric[0])
    right_interface_log = float(log_modulus_numeric[-1])
    left_distance = max(0.0, (left_interface_log - target_log) / mu_left)
    right_distance = max(0.0, (right_interface_log - target_log) / mu_right)
    far_left = -extent - left_distance
    far_right = extent + right_distance

    # Fixed number of analytic coordinates for every mode, which permits stacking.
    y_left = np.linspace(far_left, -extent, n_tail + 1, endpoint=True)[:-1]
    y_right = np.linspace(extent, far_right, n_tail + 1, endpoint=True)[1:]

    left_log = left_interface_log + gamma_left.real * (y_left + extent)
    left_phase = float(phase_numeric[0]) + gamma_left.imag * (y_left + extent)
    right_log = right_interface_log + gamma_right.real * (y_right - extent)
    right_phase = float(phase_numeric[-1]) + gamma_right.imag * (y_right - extent)
    left_modulus = np.exp(np.clip(left_log, -745.0, 0.0))
    right_modulus = np.exp(np.clip(right_log, -745.0, 0.0))
    left_pressure = left_modulus * np.exp(1j * left_phase)
    right_pressure = right_modulus * np.exp(1j * right_phase)

    y = np.concatenate([y_left, y_numeric, y_right])
    kappa = np.concatenate(
        [
            np.full(y_left.size, gamma_left.real),
            kappa_numeric,
            np.full(y_right.size, gamma_right.real),
        ]
    )
    q = np.concatenate(
        [
            np.full(y_left.size, gamma_left.imag),
            q_numeric,
            np.full(y_right.size, gamma_right.imag),
        ]
    )
    log_modulus = np.concatenate([left_log, log_modulus_numeric, right_log])
    modulus = np.concatenate([left_modulus, modulus_numeric, right_modulus])
    phase = np.concatenate([left_phase, phase_numeric, right_phase])
    pressure = np.concatenate([left_pressure, pressure_numeric, right_pressure])
    region = np.concatenate(
        [
            np.full(y_left.size, -1, dtype=np.int8),
            np.zeros(y_numeric.size, dtype=np.int8),
            np.full(y_right.size, 1, dtype=np.int8),
        ]
    )

    arrays = {
        "y": y.astype(np.float64),
        "kappa": kappa.astype(np.float64),
        "q": q.astype(np.float64),
        "log_modulus": log_modulus.astype(np.float64),
        "modulus": modulus.astype(np.float64),
        "phase": phase.astype(np.float64),
        "p_real": pressure.real.astype(np.float64),
        "p_imag": pressure.imag.astype(np.float64),
        "region": region,
    }
    metadata = {
        "Mach": Mach,
        "alpha": alpha,
        "cr": c.real,
        "ci": c.imag,
        "omega_i": alpha * c.imag,
        "matching_y": matching_y,
        "numerical_extent": extent,
        "modal_dy": dy,
        "tail_tolerance": tail_tolerance,
        "analytic_tail_points": n_tail,
        "gamma_left_real": gamma_left.real,
        "gamma_left_imag": gamma_left.imag,
        "gamma_right_real": gamma_right.real,
        "gamma_right_imag": gamma_right.imag,
        "mu_left": mu_left,
        "mu_right": mu_right,
        "far_left": far_left,
        "far_right": far_right,
        "left_distance": left_distance,
        "right_distance": right_distance,
        "left_interface_amplitude": math.exp(max(-745.0, left_interface_log)),
        "right_interface_amplitude": math.exp(max(-745.0, right_interface_log)),
        "left_far_amplitude": float(left_modulus[0]) if left_modulus.size else math.exp(left_interface_log),
        "right_far_amplitude": float(right_modulus[-1]) if right_modulus.size else math.exp(right_interface_log),
        "delta_kappa_at_match": float(left_match[0] - right_match[0]),
        "delta_q_at_match": float(left_match[1] - right_match[1]),
        "gamma_mismatch_at_match": float(
            math.hypot(left_match[0] - right_match[0], left_match[1] - right_match[1])
        ),
        "right_log_amplitude_shift": right_ell_shift,
        "n_coordinates": int(y.size),
        "timestamp": utc_now(),
    }
    return arrays, metadata


def serialize_arrays(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def deserialize_arrays(blob: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def initialize_database(path: Path, overwrite: bool) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        path.unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS modes (
            alpha REAL PRIMARY KEY,
            status TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            arrays_blob BLOB,
            error TEXT,
            updated TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def database_status(connection: sqlite3.Connection, alpha: float) -> str | None:
    row = connection.execute(
        "SELECT status FROM modes WHERE ABS(alpha - ?) < 5e-13",
        (alpha,),
    ).fetchone()
    return str(row[0]) if row else None


def save_success(
    connection: sqlite3.Connection,
    *,
    alpha: float,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    connection.execute(
        """
        INSERT INTO modes(alpha, status, metadata_json, arrays_blob, error, updated)
        VALUES (?, 'converged', ?, ?, '', ?)
        ON CONFLICT(alpha) DO UPDATE SET
            status=excluded.status,
            metadata_json=excluded.metadata_json,
            arrays_blob=excluded.arrays_blob,
            error=excluded.error,
            updated=excluded.updated
        """,
        (
            alpha,
            json.dumps(metadata, sort_keys=True),
            sqlite3.Binary(serialize_arrays(arrays)),
            utc_now(),
        ),
    )
    connection.commit()


def save_failure(
    connection: sqlite3.Connection,
    *,
    alpha: float,
    metadata: dict[str, Any],
    error: str,
) -> None:
    connection.execute(
        """
        INSERT INTO modes(alpha, status, metadata_json, arrays_blob, error, updated)
        VALUES (?, 'failed', ?, NULL, ?, ?)
        ON CONFLICT(alpha) DO UPDATE SET
            status=excluded.status,
            metadata_json=excluded.metadata_json,
            arrays_blob=NULL,
            error=excluded.error,
            updated=excluded.updated
        """,
        (alpha, json.dumps(metadata, sort_keys=True), error, utc_now()),
    )
    connection.commit()


def export_database(connection: sqlite3.Connection, output_dir: Path) -> None:
    rows = connection.execute(
        "SELECT alpha, status, metadata_json, arrays_blob, error, updated FROM modes ORDER BY alpha"
    ).fetchall()
    summary_rows: list[dict[str, Any]] = []
    successful: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    for alpha, status, metadata_json, blob, error, updated in rows:
        metadata = json.loads(metadata_json)
        summary_rows.append(
            {
                **metadata,
                "status": status,
                "error": error,
                "updated": updated,
            }
        )
        if status == "converged" and blob is not None:
            successful.append((metadata, deserialize_arrays(blob)))

    atomic_write_csv(output_dir / "mode_summary.csv", pd.DataFrame(summary_rows))
    if not successful:
        return

    names = tuple(successful[0][1].keys())
    lengths = {name: {arrays[name].shape for _, arrays in successful} for name in names}
    incompatible = {name: shapes for name, shapes in lengths.items() if len(shapes) != 1}
    if incompatible:
        raise RuntimeError(f"Cannot stack variable mode shapes: {incompatible}")

    payload: dict[str, np.ndarray] = {
        "Mach": np.asarray([item[0]["Mach"] for item in successful], dtype=float),
        "alpha": np.asarray([item[0]["alpha"] for item in successful], dtype=float),
        "cr": np.asarray([item[0]["cr"] for item in successful], dtype=float),
        "ci": np.asarray([item[0]["ci"] for item in successful], dtype=float),
        "omega_i": np.asarray([item[0]["omega_i"] for item in successful], dtype=float),
    }
    for name in names:
        payload[name] = np.stack([arrays[name] for _, arrays in successful], axis=0)

    final_path = output_dir / "modes_compact_with_analytic_tails.npz"
    temporary = output_dir / f".{final_path.name}.tmp.npz"
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, final_path)


def selected_spectral_points(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    is_target = frame["is_target"].astype(str).str.lower().isin(("true", "1"))
    selected = frame[
        is_target
        & frame["status"].astype(str).isin(
            ("converged", "anchor_converged")
        )
        & frame["direction"].astype(str).isin(
            ("low", "high", "anchor")
        )
        & np.isfinite(pd.to_numeric(frame["cr"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["ci"], errors="coerce"))
        & (pd.to_numeric(frame["ci"], errors="coerce") > 0.0)
    ].copy()
    selected = selected.sort_values("alpha").drop_duplicates("alpha", keep="last")
    return selected


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = repo / config_path
    config = load_config(config_path)
    Mach = resolve_mach(args, config)

    output_root = Path(str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = repo / output_root
    mach_dir = output_root / f"M{Mach:.6f}".replace(".", "p")
    spectral_path = mach_dir / "spectral_points.csv"
    mode_dir = mach_dir / "modes"
    mode_dir.mkdir(parents=True, exist_ok=True)
    database_path = mode_dir / "modes_checkpoint.sqlite"

    if not spectral_path.exists():
        print(f"No spectral_points.csv for Mach={Mach}; mode pass skipped.")
        return 0
    selected = selected_spectral_points(spectral_path)
    connection = initialize_database(database_path, args.overwrite)

    print("=== Dense supersonic mode reconstruction ===")
    print(f"Mach              : {Mach}")
    print(f"accepted targets  : {len(selected)}")
    print(f"database          : {database_path}")

    for index, (_, row) in enumerate(selected.iterrows(), start=1):
        alpha = float(row["alpha"])
        status = database_status(connection, alpha)
        if status == "converged":
            continue
        if status == "failed" and not args.retry_failed:
            continue
        print(
            f"[{index}/{len(selected)}] alpha={alpha:.12g}, "
            f"c={float(row['cr']):.12g}+{float(row['ci']):.6e}i",
            flush=True,
        )
        try:
            arrays, metadata = reconstruct_mode(row=row, Mach=Mach, config=config)
            save_success(
                connection,
                alpha=alpha,
                arrays=arrays,
                metadata=metadata,
            )
            print(
                f"  saved: y=[{metadata['far_left']:.4g}, {metadata['far_right']:.4g}], "
                f"match residual={metadata['gamma_mismatch_at_match']:.3e}",
                flush=True,
            )
        except Exception as exc:
            metadata = {
                "Mach": Mach,
                "alpha": alpha,
                "cr": float(row["cr"]),
                "ci": float(row["ci"]),
                "omega_i": float(row["omega_i"]),
                "timestamp": utc_now(),
            }
            save_failure(
                connection,
                alpha=alpha,
                metadata=metadata,
                error=f"{type(exc).__name__}: {exc}",
            )
            print(f"  FAILED: {exc}", flush=True)

    export_database(connection, mode_dir)
    connection.close()
    print(f"Written to: {mode_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
