#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import src.scripts.evaluation.evaluate_joint_pinn_global_validation as V
from src.scripts.gep.selection.solve_dense_gep_notebook_style import (
    NotebookStyleDenseGEPSolver,
)
from src.scripts.classical.solve_robust_subsonic_shooting import (
    RobustSubsonicShootingSolver,
)
from src.scripts.evaluation.evaluate_kh_subsonic_fixed_mach_modal_candidates import (
    load_classic_full_mode,
)

ROOT = Path(__file__).resolve().parents[4]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def interp_complex(y_source, values, y_target):
    return (
        np.interp(y_target, y_source, np.real(values))
        + 1j * np.interp(
            y_target,
            y_source,
            np.imag(values),
        )
    )


def normalize_fields(fields):
    output = {
        key: np.asarray(fields[key], dtype=np.complex128)
        for key in ("p", "rho", "u", "v")
    }

    amplitude = float(
        np.max(np.abs(output["p"]))
    )

    if amplitude > 1.0e-30:
        center = len(output["p"]) // 2
        phase = np.exp(
            -1j * np.angle(output["p"][center])
        )
        scale = phase / amplitude

        output = {
            key: scale * value
            for key, value in output.items()
        }

    return output


def statistics(values):
    values = np.asarray(values, dtype=float)

    q1, q3 = np.quantile(
        values,
        [0.25, 0.75],
    )

    return {
        "median_ms": float(np.median(values)),
        "q1_ms": float(q1),
        "q3_ms": float(q3),
        "iqr_ms": float(q3 - q1),
        "p95_ms": float(
            np.quantile(values, 0.95)
        ),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(
            np.std(values, ddof=1)
        ) if len(values) > 1 else 0.0,
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def benchmark(
    name,
    scope,
    execution_device,
    function,
    warmups,
    repeats,
    synchronization_device=None,
):
    for _ in range(warmups):
        function()

        if synchronization_device is not None:
            synchronize(synchronization_device)

    gc.collect()
    timings = []

    for repetition in range(repeats):
        if synchronization_device is not None:
            synchronize(synchronization_device)

        start = time.perf_counter_ns()

        function()

        if synchronization_device is not None:
            synchronize(synchronization_device)

        elapsed_ms = (
            time.perf_counter_ns() - start
        ) * 1.0e-6

        timings.append(elapsed_ms)

    row = {
        "method": name,
        "output_scope": scope,
        "execution_device": execution_device,
        "warmups": warmups,
        "repeats": repeats,
        **statistics(timings),
    }

    raw = pd.DataFrame({
        "method": name,
        "repetition": np.arange(repeats),
        "latency_ms": timings,
    })

    return row, raw


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=301,
    )
    parser.add_argument(
        "--fast-repeats",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--slow-repeats",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load-repeats",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA demandée mais indisponible."
        )

    tables_dir = args.output_dir / "tables"
    data_dir = args.output_dir / "data"

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plan_path = ROOT / (
        "assets/pinn_subsonic/"
        "joint_ci_mode_global_validation_v1/"
        "plans/offgrid_384_plan.csv"
    )

    plan = pd.read_csv(plan_path)

    mach_values = pd.to_numeric(
        plan["Mach"],
        errors="coerce",
    )
    eta_values = pd.to_numeric(
        plan["eta"],
        errors="coerce",
    )

    distance = (
        (mach_values - 0.5) ** 2
        + (eta_values - 0.5) ** 2
    )

    point = plan.loc[distance.idxmin()]

    mach = float(point["Mach"])
    eta = float(point["eta"])
    alpha = float(point["alpha"])

    n_points = int(args.n_points)

    mapping_scale = float(
        point["mapping_scale"]
    )
    xi_max = float(point["xi_max"])

    checkpoint = Path(
        str(point["checkpoint"])
    )

    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint

    with contextlib.redirect_stdout(
        io.StringIO()
    ):
        (
            field,
            ci_net,
            module,
            checkpoint_args,
            family,
        ) = V.evaluate_pinn(
            checkpoint_path=checkpoint,
            device=device,
        )

    prototype = NotebookStyleDenseGEPSolver(
        alpha=alpha,
        Mach=mach,
        n_points=n_points,
        mapping_kind="pin",
        mapping_scale=mapping_scale,
        xi_max=xi_max,
    )

    y_grid = np.asarray(
        prototype.y,
        dtype=float,
    )

    del prototype

    def pinn_profile():
        p_pred, q_pred, ci_pred = (
            V.call_pinn_profiles(
                field=field,
                ci_net=ci_net,
                module=module,
                family=family,
                y=y_grid,
                alpha=alpha,
                mach=mach,
                device=device,
            )
        )

        fields = V.fields_from_pq(
            y_grid,
            p_pred,
            q_pred,
            alpha,
            mach,
            ci_pred,
        )

        return (
            float(ci_pred),
            normalize_fields(fields),
        )

    ci_seed = pinn_profile()[0]
    synchronize(device)

    def gep_from_seed(seed):
        solver = NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=mach,
            n_points=n_points,
            mapping_kind="pin",
            mapping_scale=mapping_scale,
            xi_max=xi_max,
        )

        mode, source, n_modes = (
            solver.get_nearest_mode_to_target(
                target_guess=(
                    0.0,
                    float(seed),
                ),
                prefer_positive_cr=False,
                ci_weight=2.0,
            )
        )

        if mode is None:
            raise RuntimeError(
                f"GEP failed: {source}, "
                f"n_modes={n_modes}"
            )

        vector = np.asarray(
            mode["vector"],
            dtype=np.complex128,
        )

        pressure = vector[
            2 * n_points : 3 * n_points
        ]

        fields = normalize_fields({
            "u": vector[0:n_points],
            "v": vector[
                n_points : 2 * n_points
            ],
            "p": pressure,
            "rho": mach**2 * pressure,
        })

        return float(mode["ci"]), fields

    def gep_seed_ready():
        return gep_from_seed(ci_seed)

    def pinn_gep_end_to_end():
        seed, _ = pinn_profile()
        return gep_from_seed(seed)

    def classical_ci():
        result = (
            RobustSubsonicShootingSolver(
                alpha=alpha,
                Mach=mach,
            ).solve(
                force_cross_check=True
            )
        )

        return float(result.ci)

    def classical_mode():
        fields, ci_value = (
            load_classic_full_mode(
                alpha,
                mach,
            )
        )

        y_source = np.asarray(
            fields["y"],
            dtype=float,
        )

        interpolated = {
            key: interp_complex(
                y_source,
                np.asarray(
                    fields[key],
                    dtype=np.complex128,
                ),
                y_grid,
            )
            for key in (
                "p",
                "rho",
                "u",
                "v",
            )
        }

        return (
            float(ci_value),
            normalize_fields(interpolated),
        )

    def checkpoint_load():
        with contextlib.redirect_stdout(
            io.StringIO()
        ):
            loaded = V.evaluate_pinn(
                checkpoint_path=checkpoint,
                device=device,
            )

        loaded_family = loaded[-1]
        del loaded

        return loaded_family

    checks = {
        "pinn_ci": pinn_profile()[0],
        "gep_ci": gep_seed_ready()[0],
        "classical_ci": classical_ci(),
        "classical_mode_ci": (
            classical_mode()[0]
        ),
    }

    specifications = [
        (
            "PINN full mode, loaded",
            "full mode on N=301",
            "GPU",
            pinn_profile,
            20,
            args.fast_repeats,
            device,
        ),
        (
            "GEP refinement, seed ready",
            "full mode on N=301",
            "CPU",
            gep_seed_ready,
            1,
            args.slow_repeats,
            None,
        ),
        (
            "PINN-seeded GEP, end to end",
            "full mode on N=301",
            "GPU+CPU",
            pinn_gep_end_to_end,
            1,
            args.slow_repeats,
            device,
        ),
        (
            "Classical full mode",
            "full mode on N=301",
            "CPU",
            classical_mode,
            1,
            args.slow_repeats,
            None,
        ),
        (
            "Classical shooting",
            "c_i only",
            "CPU",
            classical_ci,
            1,
            args.slow_repeats,
            None,
        ),
        (
            "PINN checkpoint load",
            "model load only",
            "GPU",
            checkpoint_load,
            0,
            args.load_repeats,
            device,
        ),
    ]

    summary_rows = []
    raw_frames = []

    for (
        name,
        scope,
        execution_device,
        function,
        warmups,
        repeats,
        synchronization_device,
    ) in specifications:
        print(
            f"[BENCH] {name}: "
            f"warmups={warmups}, "
            f"repeats={repeats}",
            flush=True,
        )

        row, raw = benchmark(
            name,
            scope,
            execution_device,
            function,
            warmups,
            repeats,
            synchronization_device,
        )

        summary_rows.append(row)
        raw_frames.append(raw)

        print(
            f"        median="
            f"{row['median_ms']:.6f} ms; "
            f"IQR={row['iqr_ms']:.6f} ms",
            flush=True,
        )

    summary = pd.DataFrame(
        summary_rows
    )

    classical_baseline = float(
        summary.loc[
            summary["method"]
            == "Classical full mode",
            "median_ms",
        ].iloc[0]
    )

    summary[
        "speedup_vs_classical_full_mode"
    ] = np.nan

    full_mode_mask = (
        summary["output_scope"]
        == "full mode on N=301"
    )

    summary.loc[
        full_mode_mask,
        "speedup_vs_classical_full_mode",
    ] = (
        classical_baseline
        / summary.loc[
            full_mode_mask,
            "median_ms",
        ]
    )

    summary_path = (
        tables_dir
        / "Table_subsonic_runtime.csv"
    )
    raw_path = (
        data_dir
        / "subsonic_runtime_raw.csv"
    )
    metadata_path = (
        data_dir
        / "subsonic_runtime_metadata.json"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    pd.concat(
        raw_frames,
        ignore_index=True,
    ).to_csv(
        raw_path,
        index=False,
    )

    metadata = {
        "point_id": str(
            point["point_id"]
        ),
        "Mach": mach,
        "eta": eta,
        "alpha": alpha,
        "chart_id": str(
            point["chart_id"]
        ),
        "checkpoint": str(checkpoint),
        "field_family": family,
        "N": n_points,
        "checks": checks,
        "training_included": False,
        "checkpoint_load_separate": True,
        "end_to_end_note": (
            "The end-to-end PINN-seeded GEP "
            "row evaluates the full PINN "
            "profile to obtain the seed. "
            "It is therefore conservative."
        ),
        "hardware": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(
                    device
                )
                if device.type == "cuda"
                else None
            ),
            "slurm_job_id": os.environ.get(
                "SLURM_JOB_ID"
            ),
            "slurm_cpus_per_task": (
                os.environ.get(
                    "SLURM_CPUS_PER_TASK"
                )
            ),
            "omp_num_threads": os.environ.get(
                "OMP_NUM_THREADS"
            ),
            "mkl_num_threads": os.environ.get(
                "MKL_NUM_THREADS"
            ),
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print()
    print(summary.to_string(index=False))

    print("\nFichiers produits :")
    print(summary_path)
    print(raw_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
