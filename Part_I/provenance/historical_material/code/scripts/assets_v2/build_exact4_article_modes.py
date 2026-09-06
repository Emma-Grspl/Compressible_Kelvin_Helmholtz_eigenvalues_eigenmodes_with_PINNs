#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from scripts.compare_kh_subsonic_fixed_mach_modal_candidates import load_classic_full_mode
from scripts.dev.benchmark_subsonic_local_atlas_core_ci_seeded_gep_v2 import (
    align_complex,
    interp_complex,
    rel_l2,
    split_gep_vector,
)
from scripts.dev.train_subsonic_seedGEP_pq2d_continuous_M_alpha_etaaware import (
    CiGridIDW,
    FieldPQNet as EtaAwareFieldPQNet,
)
from scripts.train_kh_subsonic_2d_pressure_pq_firstorder_mini import (
    FieldPQNet as LegacyFieldPQNet,
    fields_from_pq,
)

FIELDS = ("rho", "u", "v", "p")
TARGETS = (
    ("M020_a020", 0.20, 0.20),
    ("M040_a025", 0.40, 0.25),
    ("M060_a018", 0.60, 0.18),
    ("M080_a020", 0.80, 0.20),
)

MODEL_ROOT = ROOT / "pinn_subsonic" / "models"
OUTPUT_ROOT = (
    ROOT
    / "assets"
    / "pinn_subsonic"
    / "article_work"
    / "exact4_modes"
)
PROFILE_ROOT = OUTPUT_ROOT / "profiles"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
REPORT_ROOT = OUTPUT_ROOT / "reports"


def linear_dimensions(state: dict[str, torch.Tensor]) -> tuple[int, int]:
    layers: list[tuple[int, tuple[int, ...]]] = []
    for key, tensor in state.items():
        match = re.fullmatch(r"net\.(\d+)\.weight", key)
        if match:
            layers.append((int(match.group(1)), tuple(tensor.shape)))
    if not layers:
        raise RuntimeError("No net.<index>.weight tensors found")
    layers.sort()
    return int(layers[0][1][1]), int(layers[-1][1][0])


def checkpoint_dataframe(raw: Any) -> pd.DataFrame:
    if isinstance(raw, pd.DataFrame):
        return raw.copy()
    if isinstance(raw, dict):
        return pd.DataFrame(raw)
    if isinstance(raw, (list, tuple)):
        return pd.DataFrame(raw)
    raise TypeError(f"Unsupported anchor_df type: {type(raw)!r}")


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    args = dict(checkpoint["args"])
    return {
        "path": path,
        "chart_id": path.parent.name,
        "checkpoint": checkpoint,
        "args": args,
        "mach_min": float(args["mach_min"]),
        "mach_max": float(args["mach_max"]),
        "eta_min": float(args["eta_min"]),
        "eta_max": float(args["eta_max"]),
    }


def assignment_candidates() -> list[Path]:
    exact = [
        ROOT / "pinn_subsonic" / "data" / "scientific_outputs" / "release_v1"
        / "data" / "coverage_and_assignment_grid.csv",
        ROOT / "pinn_subsonic" / "manifests"
        / "atlas_fullrect_final_coverage_grid.csv",
        ROOT / "assets" / "pinn_subsonic" / "release_v1"
        / "data" / "coverage_and_assignment_grid.csv",
    ]
    found = [path for path in exact if path.exists()]
    if found:
        return found
    return sorted(ROOT.rglob("coverage_and_assignment_grid.csv"))


def assigned_chart(mach: float, eta: float) -> str | None:
    chart_columns = (
        "selected_chart",
        "assigned_chart",
        "chart_id",
        "final_chart",
    )
    for path in assignment_candidates():
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "Mach" not in frame.columns or "eta" not in frame.columns:
            continue
        chart_column = next((c for c in chart_columns if c in frame.columns), None)
        if chart_column is None:
            continue
        work = frame[["Mach", "eta", chart_column]].copy()
        work["Mach"] = pd.to_numeric(work["Mach"], errors="coerce")
        work["eta"] = pd.to_numeric(work["eta"], errors="coerce")
        work = work.dropna(subset=["Mach", "eta", chart_column])
        if work.empty:
            continue
        work["distance"] = np.hypot(work["Mach"] - mach, work["eta"] - eta)
        chart = str(work.sort_values("distance").iloc[0][chart_column])
        if (MODEL_ROOT / chart / "model_state.pt").exists():
            return chart
    return None


def choose_checkpoint(mach: float, eta: float) -> dict[str, Any]:
    checkpoints = [
        inspect_checkpoint(path)
        for path in sorted(MODEL_ROOT.glob("*/model_state.pt"))
    ]
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found under {MODEL_ROOT}")

    preferred = assigned_chart(mach, eta)
    if preferred is not None:
        for item in checkpoints:
            if item["chart_id"] == preferred:
                if (
                    item["mach_min"] <= mach <= item["mach_max"]
                    and item["eta_min"] <= eta <= item["eta_max"]
                ):
                    return item

    covering = [
        item
        for item in checkpoints
        if (
            item["mach_min"] <= mach <= item["mach_max"]
            and item["eta_min"] <= eta <= item["eta_max"]
        )
    ]
    if not covering:
        raise RuntimeError(
            f"No curated chart covers M={mach:.6g}, eta={eta:.6g}"
        )

    def score(item: dict[str, Any]) -> tuple[float, float]:
        m_width = max(item["mach_max"] - item["mach_min"], 1.0e-12)
        e_width = max(item["eta_max"] - item["eta_min"], 1.0e-12)
        m_center = 0.5 * (item["mach_min"] + item["mach_max"])
        e_center = 0.5 * (item["eta_min"] + item["eta_max"])
        center_distance = math.hypot(
            (mach - m_center) / m_width,
            (eta - e_center) / e_width,
        )
        area = m_width * e_width
        return center_distance, area

    return min(covering, key=score)


def build_model(item: dict[str, Any]) -> tuple[torch.nn.Module, str, CiGridIDW]:
    checkpoint = item["checkpoint"]
    args = item["args"]
    state = checkpoint["field_state_dict"]

    input_dimension, output_dimension = linear_dimensions(state)
    if output_dimension != 4:
        raise RuntimeError(
            f"{item['chart_id']}: unexpected output dimension {output_dimension}"
        )

    n_freq = int(args["n_freq"])
    legacy_dimension = 3 + 2 * n_freq
    eta_aware_dimension = 7 + 2 * n_freq

    mach_min = float(args["mach_min"])
    mach_max = float(args["mach_max"])
    eta_min = float(args["eta_min"])
    eta_max = float(args["eta_max"])
    alpha_min = eta_min * math.sqrt(max(1.0 - mach_max**2, 1.0e-14))
    alpha_max = eta_max * math.sqrt(max(1.0 - mach_min**2, 1.0e-14))

    common = dict(
        ymax=float(args["ymax"]),
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        mach_min=mach_min,
        mach_max=mach_max,
        width=int(args["width"]),
        depth=int(args["depth"]),
        n_freq=n_freq,
    )

    if input_dimension == legacy_dimension:
        model = LegacyFieldPQNet(**common).double()
    elif input_dimension == eta_aware_dimension:
        model = EtaAwareFieldPQNet(
            **common,
            eta_min=eta_min,
            eta_max=eta_max,
        ).double()
    else:
        raise RuntimeError(
            f"{item['chart_id']}: unsupported input dimension "
            f"{input_dimension}; expected {legacy_dimension} or "
            f"{eta_aware_dimension}"
        )

    model.load_state_dict(state, strict=True)
    model.eval()

    family = (
        "pQscaled"
        if "qscaled" in str(args.get("output_dir", "")).lower()
        else "pq"
    )

    anchor_df = checkpoint_dataframe(checkpoint["anchor_df"])
    ci_provider = CiGridIDW(
        anchor_df=anchor_df,
        eta_scale=float(args.get("ci_idw_eta_scale", 0.25)),
        mach_scale=float(args.get("ci_idw_mach_scale", 0.25)),
        power=float(args.get("ci_idw_power", 4.0)),
    ).double()
    ci_provider.eval()

    return model, family, ci_provider


def evaluate_direct(
    model: torch.nn.Module,
    family: str,
    y: np.ndarray,
    alpha: float,
    mach: float,
    ci_seed: float,
) -> dict[str, np.ndarray]:
    yt = torch.tensor(y[:, None], dtype=torch.float64)
    at = torch.full_like(yt, alpha)
    mt = torch.full_like(yt, mach)

    with torch.no_grad():
        p_tensor, second_tensor = model(yt, at, mt)

    p = p_tensor.cpu().numpy().reshape(-1).astype(np.complex128)
    second = second_tensor.cpu().numpy().reshape(-1).astype(np.complex128)
    q = alpha * second if family == "pQscaled" else second
    rho, u, v, _ = fields_from_pq(y, p, q, alpha, mach, ci_seed)

    return {
        "p": np.asarray(p, np.complex128),
        "rho": np.asarray(rho, np.complex128),
        "u": np.asarray(u, np.complex128),
        "v": np.asarray(v, np.complex128),
    }


def solve_gep(
    alpha: float,
    mach: float,
    target_ci: float,
    n_points: int = 301,
    mapping_scale: float = 5.0,
    xi_max: float = 0.98,
):
    solver = NotebookStyleDenseGEPSolver(
        alpha=alpha,
        Mach=mach,
        n_points=n_points,
        mapping_kind="pin",
        mapping_scale=mapping_scale,
        xi_max=xi_max,
    )
    mode, source, n_modes = solver.get_nearest_mode_to_target(
        target_guess=(0.0, float(target_ci)),
        prefer_positive_cr=False,
        ci_weight=2.0,
    )
    if mode is None:
        raise RuntimeError(
            f"No finite GEP mode near ci={target_ci:.16g}"
        )
    return solver, mode, source, int(n_modes)


def interpolate_gep(
    solver,
    mode: dict[str, Any],
    mach: float,
    y_target: np.ndarray,
) -> dict[str, np.ndarray]:
    fields = split_gep_vector(mode["vector"], solver.n_points, mach)
    return {
        field: interp_complex(
            np.asarray(solver.y, float),
            np.asarray(fields[field], np.complex128),
            y_target,
        )
        for field in ("p", "rho", "u", "v")
    }


def multiply_fields(
    fields: dict[str, np.ndarray],
    scale: complex,
) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value, np.complex128) * scale
        for key, value in fields.items()
    }


def modal_errors(
    pred: dict[str, np.ndarray],
    ref: dict[str, np.ndarray],
    y: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    return {
        f"{field}_rel": float(rel_l2(pred[field], ref[field], y, mask))
        for field in ("p", "rho", "u", "v")
    }


def save_profile(
    label: str,
    *,
    y: np.ndarray,
    shooting: dict[str, np.ndarray],
    direct: dict[str, np.ndarray],
    classic_gep: dict[str, np.ndarray],
    pinn_gep: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> Path:
    path = PROFILE_ROOT / f"{label}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"y": np.asarray(y, float)}
    for field in ("p", "rho", "u", "v"):
        payload[f"{field}_shooting"] = shooting[field]
        payload[f"{field}_direct"] = direct[field]
        payload[f"{field}_classic_gep"] = classic_gep[field]
        payload[f"{field}_pinn_gep"] = pinn_gep[field]
    for key, value in metadata.items():
        payload[key] = np.array(value)
    np.savez_compressed(path, **payload)
    return path


def plot_modes(
    label: str,
    y: np.ndarray,
    shooting: dict[str, np.ndarray],
    direct: dict[str, np.ndarray],
    pinn_gep: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    for ax, field in zip(axes.ravel(), FIELDS):
        ax.plot(y, np.real(shooting[field]), linewidth=2.0, label="shooting")
        ax.plot(
            y,
            np.real(direct[field]),
            linewidth=1.5,
            linestyle=":",
            label="direct PINN",
        )
        ax.plot(
            y,
            np.real(pinn_gep[field]),
            linewidth=1.7,
            linestyle="--",
            label="PINN-seeded GEP",
        )
        ax.set_title(field)
        ax.set_xlabel(r"$y$")
        ax.set_ylabel("real part")
        ax.set_xlim(-10.0, 10.0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    figure.suptitle(
        (
            f"M={metadata['Mach']:.2f}, "
            f"alpha={metadata['alpha']:.2f}, "
            f"eta={metadata['eta']:.4f}\n"
            f"chart={metadata['chart_id']}"
        ),
        fontsize=13,
    )
    figure.tight_layout()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_ROOT / f"Fig_modes_{label}"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def find_manifest(filename: str) -> Path:
    exact = ROOT / "pinn_subsonic" / "manifests" / filename
    if exact.exists():
        return exact
    matches = [
        path
        for path in ROOT.rglob(filename)
        if "article_work" not in path.parts
    ]
    if not matches:
        raise FileNotFoundError(filename)
    matches.sort(
        key=lambda path: (
            "pinn_subsonic" in path.parts,
            "manifests" in path.parts,
            -len(path.parts),
        ),
        reverse=True,
    )
    return matches[0]


def plot_heatmap(
    csv_path: Path,
    columns: tuple[str, str, str, str],
    output_stem: str,
    title_prefix: str,
) -> None:
    frame = pd.read_csv(csv_path)
    required = {"Mach", "alpha", *columns}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f"{csv_path}: missing columns {sorted(missing)}"
        )

    if "success" in frame.columns:
        success = (
            frame["success"]
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"})
        )
        if success.any():
            frame = frame.loc[success].copy()

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    labels = (r"$p$", r"$\rho$", r"$u$", r"$v$")

    for ax, column, field_label in zip(axes.ravel(), columns, labels):
        data = frame[["Mach", "alpha", column]].copy()
        for name in ("Mach", "alpha", column):
            data[name] = pd.to_numeric(data[name], errors="coerce")
        data = data.dropna()
        data = data.loc[
            np.isfinite(data["Mach"])
            & np.isfinite(data["alpha"])
            & np.isfinite(data[column])
            & (data[column] > 0)
        ]
        data = (
            data.groupby(["Mach", "alpha"], as_index=False)[column]
            .median()
        )
        if len(data) < 4:
            raise RuntimeError(
                f"{csv_path}: insufficient data for {column}"
            )

        mach = data["Mach"].to_numpy(float)
        alpha = data["alpha"].to_numpy(float)
        error = data[column].to_numpy(float)
        triangulation = mtri.Triangulation(mach, alpha)

        lower = max(float(np.nanpercentile(error, 1.0)), 1.0e-8)
        upper = max(float(np.nanpercentile(error, 99.0)), 1.01 * lower)
        levels = np.geomspace(lower, upper, 24)

        contour = ax.tricontourf(
            triangulation,
            error,
            levels=levels,
            norm=LogNorm(vmin=lower, vmax=upper),
            extend="both",
        )
        ax.scatter(mach, alpha, s=4, alpha=0.18)
        for _, target_mach, target_alpha in TARGETS:
            ax.plot(
                target_mach,
                target_alpha,
                marker="*",
                markersize=9,
                markeredgecolor="black",
                markerfacecolor="white",
            )
        figure.colorbar(
            contour,
            ax=ax,
            label="relative modal error",
        )
        ax.set(
            xlabel=r"Mach number $M$",
            ylabel=r"Wavenumber $\alpha$",
            title=f"{title_prefix}: {field_label}",
        )

    figure.tight_layout()
    stem = FIGURE_ROOT / output_stem
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def process_target(label: str, mach: float, alpha: float) -> dict[str, Any]:
    eta = alpha / math.sqrt(1.0 - mach**2)
    item = choose_checkpoint(mach, eta)
    model, family, ci_provider = build_model(item)

    at = torch.tensor([[alpha]], dtype=torch.float64)
    mt = torch.tensor([[mach]], dtype=torch.float64)
    with torch.no_grad():
        ci_seed = float(ci_provider(at, mt).item())

    classic_fields, ci_ref = load_classic_full_mode(alpha, mach)
    y = np.asarray(classic_fields["y"], float)
    shooting = {
        field: np.asarray(classic_fields[field], np.complex128)
        for field in ("p", "rho", "u", "v")
    }

    direct = evaluate_direct(
        model,
        family,
        y,
        alpha,
        mach,
        ci_seed,
    )
    direct_mask = (
        np.isfinite(y)
        & np.isfinite(shooting["p"])
        & np.isfinite(direct["p"])
        & (
            np.abs(shooting["p"])
            >= 0.05 * np.nanmax(np.abs(shooting["p"]))
        )
    )
    direct_scale = align_complex(
        direct["p"],
        shooting["p"],
        direct_mask,
    )
    direct = multiply_fields(direct, direct_scale)

    solver_ref, mode_ref, source_ref, n_modes_ref = solve_gep(
        alpha,
        mach,
        float(ci_ref),
    )
    classic_gep = interpolate_gep(
        solver_ref,
        mode_ref,
        mach,
        y,
    )

    solver_seed, mode_seed, source_seed, n_modes_seed = solve_gep(
        alpha,
        mach,
        ci_seed,
    )
    pinn_gep = interpolate_gep(
        solver_seed,
        mode_seed,
        mach,
        y,
    )

    y_min = max(
        float(np.min(y)),
        float(np.min(solver_ref.y)),
        float(np.min(solver_seed.y)),
        -12.0,
    )
    y_max = min(
        float(np.max(y)),
        float(np.max(solver_ref.y)),
        float(np.max(solver_seed.y)),
        12.0,
    )
    gep_mask = (
        np.isfinite(y)
        & (y >= y_min)
        & (y <= y_max)
        & np.isfinite(classic_gep["p"])
        & np.isfinite(pinn_gep["p"])
    )
    classic_scale = align_complex(
        classic_gep["p"],
        shooting["p"],
        gep_mask,
    )
    classic_gep = multiply_fields(classic_gep, classic_scale)
    pinn_scale = align_complex(
        pinn_gep["p"],
        classic_gep["p"],
        gep_mask,
    )
    pinn_gep = multiply_fields(pinn_gep, pinn_scale)

    metadata = {
        "label": label,
        "Mach": mach,
        "alpha": alpha,
        "eta": eta,
        "chart_id": item["chart_id"],
        "field_family": family,
        "ci_shooting": float(ci_ref),
        "ci_seed": ci_seed,
        "ci_classic_gep": float(mode_ref["ci"]),
        "ci_pinn_gep": float(mode_seed["ci"]),
        "classic_selection_source": source_ref,
        "pinn_selection_source": source_seed,
        "classic_n_modes": n_modes_ref,
        "pinn_n_modes": n_modes_seed,
        "N": 301,
        "mapping_scale": 5.0,
        "xi_max": 0.98,
        "direct_errors": modal_errors(
            direct,
            shooting,
            y,
            direct_mask,
        ),
        "gep_errors": modal_errors(
            pinn_gep,
            classic_gep,
            y,
            gep_mask,
        ),
    }

    save_profile(
        label,
        y=y,
        shooting=shooting,
        direct=direct,
        classic_gep=classic_gep,
        pinn_gep=pinn_gep,
        metadata={
            key: value
            for key, value in metadata.items()
            if not isinstance(value, dict)
        },
    )
    plot_modes(
        label,
        y,
        shooting,
        direct,
        pinn_gep,
        metadata,
    )

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / f"{label}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    torch.set_num_threads(
        max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    )
    PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries = [
        process_target(label, mach, alpha)
        for label, mach, alpha in TARGETS
    ]

    final_csv = find_manifest(
        "atlas_fullrect_gep_final_all_points.csv"
    )
    direct_csv = find_manifest(
        "atlas_fullrect_pinn_diagnostics_all_points.csv"
    )

    plot_heatmap(
        final_csv,
        ("p_rel", "rho_rel", "u_rel", "v_rel"),
        "Fig07_final_modal_error_heatmaps_2D",
        "PINN-seeded GEP",
    )
    plot_heatmap(
        direct_csv,
        ("p_rel", "rho_rel", "u_rel", "v_rel"),
        "SuppFig_direct_PINN_modal_error_heatmaps_2D",
        "Direct PINN",
    )

    pd.DataFrame(
        [
            {
                "label": item["label"],
                "Mach": item["Mach"],
                "alpha": item["alpha"],
                "eta": item["eta"],
                "chart_id": item["chart_id"],
                "ci_shooting": item["ci_shooting"],
                "ci_seed": item["ci_seed"],
                "ci_classic_gep": item["ci_classic_gep"],
                "ci_pinn_gep": item["ci_pinn_gep"],
                **{
                    f"direct_{key}": value
                    for key, value in item["direct_errors"].items()
                },
                **{
                    f"gep_{key}": value
                    for key, value in item["gep_errors"].items()
                },
            }
            for item in summaries
        ]
    ).to_csv(OUTPUT_ROOT / "exact4_summary.csv", index=False)

    print(json.dumps({
        "output_root": str(OUTPUT_ROOT),
        "mode_figures": [
            str(FIGURE_ROOT / f"Fig_modes_{label}.pdf")
            for label, _, _ in TARGETS
        ],
        "final_heatmap": str(
            FIGURE_ROOT / "Fig07_final_modal_error_heatmaps_2D.pdf"
        ),
        "direct_heatmap": str(
            FIGURE_ROOT
            / "SuppFig_direct_PINN_modal_error_heatmaps_2D.pdf"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
