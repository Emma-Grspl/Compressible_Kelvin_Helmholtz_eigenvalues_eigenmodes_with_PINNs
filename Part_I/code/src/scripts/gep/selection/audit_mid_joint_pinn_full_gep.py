#!/usr/bin/env python3
"""
Full dense-GEP validation of a jointly trained subsonic KH PINN chart.

For every requested (Mach, eta) point, this script:

1. loads the jointly trained PINN checkpoint, including:
      - trainable c_i(M, eta),
      - modal fields p(y, M, eta), q(y, M, eta);
2. calls NotebookStyleDenseGEPSolver.solve_all(), which diagonalizes the
   complete reduced generalized eigenvalue problem;
3. saves the complete raw spectrum;
4. identifies:
      - the most unstable admissible GEP mode,
      - the mode matched to the PINN;
5. uses c_i^PINN as the primary matching information and the PINN p/q
   profiles only as a secondary disambiguation criterion;
6. checks whether the PINN-matched mode is the most unstable physical mode;
7. compares the selected GEP mode with the classical reference.

No classical modal profile is used to select the GEP mode. Classical data are
used only after selection, for validation metrics.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from src.scripts.gep.selection.solve_dense_gep_notebook_style import (
    NotebookStyleDenseGEPSolver,
)
from src.scripts.evaluation.evaluate_kh_subsonic_fixed_mach_modal_candidates import (
    load_classic_full_mode,
)
from src.scripts.training.atlas.direct_pinn.train_subsonic_joint_spectral_modal_chart import (
    CiAtlasNet,
    MODULES,
    call_supported,
    infer_field_family,
)


def parse_float_list(value: str) -> list[float]:
    return [
        float(item)
        for item in str(value).replace(",", " ").split()
        if item.strip()
    ]


def alpha_from_eta(eta: float, mach: float) -> float:
    return float(eta * math.sqrt(max(1.0 - mach**2, 1.0e-14)))


def interp_complex(
    x_source: np.ndarray,
    values: np.ndarray,
    x_target: np.ndarray,
) -> np.ndarray:
    x_source = np.asarray(x_source, dtype=float)
    values = np.asarray(values, dtype=np.complex128)
    x_target = np.asarray(x_target, dtype=float)
    return (
        np.interp(x_target, x_source, np.real(values))
        + 1j * np.interp(x_target, x_source, np.imag(values))
    )


def trapz(values: np.ndarray, x: np.ndarray) -> complex:
    return np.trapz(values, x)


def weighted_norm(
    values: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> float:
    values = np.asarray(values, dtype=np.complex128)[mask]
    y = np.asarray(y, dtype=float)[mask]
    if len(y) < 2:
        return 0.0
    value = float(np.real(trapz(np.abs(values) ** 2, y)))
    return math.sqrt(max(value, 0.0))


def overlap_complex(
    first: np.ndarray,
    second: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> float:
    first = np.asarray(first, dtype=np.complex128)[mask]
    second = np.asarray(second, dtype=np.complex128)[mask]
    y = np.asarray(y, dtype=float)[mask]

    if len(y) < 2:
        return float("nan")

    first_norm = weighted_norm(first, y, np.ones(len(y), dtype=bool))
    second_norm = weighted_norm(second, y, np.ones(len(y), dtype=bool))

    if first_norm <= 0.0 or second_norm <= 0.0:
        return 0.0

    numerator = abs(trapz(np.conjugate(first) * second, y))
    return float(numerator / (first_norm * second_norm))


def phase_alignment(
    source: np.ndarray,
    target: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> complex:
    source = np.asarray(source, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128)
    y = np.asarray(y, dtype=float)

    source_m = source[mask]
    target_m = target[mask]
    y_m = y[mask]

    denominator = trapz(np.conjugate(source_m) * source_m, y_m)
    if abs(denominator) <= 1.0e-30:
        return 1.0 + 0.0j
    return complex(trapz(np.conjugate(source_m) * target_m, y_m) / denominator)


def rel_l2(
    prediction: np.ndarray,
    reference: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> float:
    prediction = np.asarray(prediction, dtype=np.complex128)[mask]
    reference = np.asarray(reference, dtype=np.complex128)[mask]
    y = np.asarray(y, dtype=float)[mask]

    numerator = float(
        np.real(trapz(np.abs(prediction - reference) ** 2, y))
    )
    denominator = float(
        np.real(trapz(np.abs(reference) ** 2, y))
    )
    if denominator <= 0.0:
        return float("nan")
    return math.sqrt(max(numerator, 0.0) / denominator)


def split_gep_vector(
    vector: np.ndarray,
    n_points: int,
    mach: float,
) -> dict[str, np.ndarray]:
    vector = np.asarray(vector, dtype=np.complex128)
    u = vector[0:n_points]
    v = vector[n_points : 2 * n_points]
    p = vector[2 * n_points : 3 * n_points]
    rho = p * mach**2
    return {"u": u, "v": v, "p": p, "rho": rho}


def make_match_mask(
    y: np.ndarray,
    p_pinn: np.ndarray,
    *,
    y_match_max: float,
    amplitude_floor_fraction: float,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p_pinn = np.asarray(p_pinn, dtype=np.complex128)

    mask = np.abs(y) <= float(y_match_max)
    amplitude = np.abs(p_pinn)
    finite_amplitude = amplitude[np.isfinite(amplitude)]
    if finite_amplitude.size:
        threshold = (
            float(amplitude_floor_fraction)
            * float(np.max(finite_amplitude))
        )
        mask &= amplitude >= threshold

    if int(np.count_nonzero(mask)) < 20:
        mask = np.abs(y) <= float(y_match_max)

    if int(np.count_nonzero(mask)) < 20:
        mask = np.isfinite(y)

    return mask


def evaluate_pinn(
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    Any,
    dict[str, Any],
    str,
]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = dict(checkpoint.get("args", {}))

    family = args.get("field_family_resolved")
    if not family:
        family = infer_field_family(
            "auto",
            checkpoint_path,
            checkpoint,
        )

    if family not in MODULES:
        raise RuntimeError(
            f"Unsupported field family {family!r}. "
            f"Known families: {sorted(MODULES)}"
        )

    module = importlib.import_module(MODULES[family])

    required_ranges = ("mach_min", "mach_max", "eta_min", "eta_max")
    missing = [name for name in required_ranges if args.get(name) is None]
    if missing:
        raise RuntimeError(
            f"Checkpoint missing chart bounds: {missing}"
        )

    mach_min = float(args["mach_min"])
    mach_max = float(args["mach_max"])
    eta_min = float(args["eta_min"])
    eta_max = float(args["eta_max"])

    alpha_corners = [
        eta * math.sqrt(max(1.0 - mach**2, 1.0e-14))
        for eta in (eta_min, eta_max)
        for mach in (mach_min, mach_max)
    ]

    field = call_supported(
        module.FieldPQNet,
        ymax=float(args.get("ymax", 100.0)),
        alpha_min=min(alpha_corners),
        alpha_max=max(alpha_corners),
        mach_min=mach_min,
        mach_max=mach_max,
        eta_min=eta_min,
        eta_max=eta_max,
        width=int(args.get("width", 256)),
        depth=int(args.get("depth", 7)),
        n_freq=int(args.get("n_freq", 12)),
    ).to(device=device, dtype=torch.float64)

    load_field = field.load_state_dict(
        checkpoint["field_state_dict"],
        strict=True,
    )

    anchor_df = pd.DataFrame(checkpoint.get("anchor_df", {}))
    if anchor_df.empty:
        raise RuntimeError(
            "Joint checkpoint has no anchor_df; cannot reconstruct CiAtlasNet."
        )

    ci_net = CiAtlasNet(
        mach_min=mach_min,
        mach_max=mach_max,
        eta_min=eta_min,
        eta_max=eta_max,
        ci_init=float(anchor_df["ci"].mean()),
        width=int(args.get("ci_width", 96)),
        depth=int(args.get("ci_depth", 3)),
    ).to(device=device, dtype=torch.float64)

    load_ci = ci_net.load_state_dict(
        checkpoint["ci_state_dict"],
        strict=True,
    )

    field.eval()
    ci_net.eval()

    print("PINN checkpoint:", checkpoint_path)
    print("Resolved family:", family)
    print("Field load:", load_field)
    print("c_i load:", load_ci)
    print(
        "Chart bounds:",
        f"M=[{mach_min}, {mach_max}]",
        f"eta=[{eta_min}, {eta_max}]",
    )

    return field, ci_net, module, args, family


def call_pinn_profiles(
    *,
    field: torch.nn.Module,
    ci_net: torch.nn.Module,
    module: Any,
    family: str,
    y: np.ndarray,
    alpha: float,
    mach: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    if family == "pQscaled":
        function = getattr(module, "eval_model_scaledQ")
    else:
        function = getattr(module, "eval_model")

    p, q, ci = function(
        field,
        ci_net,
        np.asarray(y, dtype=float),
        float(alpha),
        float(mach),
        device,
    )
    return (
        np.asarray(p, dtype=np.complex128),
        np.asarray(q, dtype=np.complex128),
        float(ci),
    )


def mode_overlap_with_pinn(
    *,
    solver: NotebookStyleDenseGEPSolver,
    vector: np.ndarray,
    p_pinn: np.ndarray,
    q_pinn: np.ndarray,
    match_mask: np.ndarray,
    p_weight: float,
) -> tuple[float, float, float]:
    fields = split_gep_vector(
        vector,
        solver.n_points,
        solver.Mach,
    )
    p_gep = fields["p"]
    q_gep = np.asarray(
        solver.d_y @ p_gep,
        dtype=np.complex128,
    )

    p_overlap = overlap_complex(
        p_gep,
        p_pinn,
        solver.y,
        match_mask,
    )
    q_overlap = overlap_complex(
        q_gep,
        q_pinn,
        solver.y,
        match_mask,
    )
    combined = (
        float(p_weight) * p_overlap
        + (1.0 - float(p_weight)) * q_overlap
    )
    return p_overlap, q_overlap, combined


def select_modes(
    *,
    solver: NotebookStyleDenseGEPSolver,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    ci_pinn: float,
    p_pinn: np.ndarray,
    q_pinn: np.ndarray,
    match_mask: np.ndarray,
    p_weight: float,
    ci_window_rel: float,
    ci_window_factor: float,
    shortlist_max: int,
    cr_physical_max: float,
    ci_physical_max: float,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for raw_index, value in enumerate(eigenvalues):
        finite = bool(
            np.isfinite(np.real(value))
            and np.isfinite(np.imag(value))
        )

        cr = float(np.real(value)) if finite else float("nan")
        ci = float(np.imag(value)) if finite else float("nan")
        abs_c = float(abs(value)) if finite else float("nan")
        unstable = bool(finite and ci > 0.0)
        solver_finite_mode = bool(unstable and abs_c < 10.0)
        physically_admissible = bool(
            solver_finite_mode
            and abs(cr) <= float(cr_physical_max)
            and ci <= float(ci_physical_max)
        )

        row: dict[str, Any] = {
            "raw_index": int(raw_index),
            "cr": cr,
            "ci": ci,
            "omega_i": float(solver.alpha * ci) if finite else float("nan"),
            "abs_c": abs_c,
            "finite": finite,
            "unstable": unstable,
            "solver_finite_mode": solver_finite_mode,
            "physically_admissible": physically_admissible,
            "ci_abs_distance_to_pinn": (
                abs(ci - ci_pinn)
                if physically_admissible
                else float("nan")
            ),
            "ci_rel_distance_to_pinn": (
                abs(ci - ci_pinn) / max(abs(ci_pinn), 5.0e-2)
                if physically_admissible
                else float("nan")
            ),
            "p_overlap_pinn": float("nan"),
            "q_overlap_pinn": float("nan"),
            "combined_overlap_pinn": float("nan"),
            "in_ci_shortlist": False,
            "is_nearest_ci": False,
            "is_pinn_matched": False,
            "is_most_unstable": False,
        }

        if physically_admissible:
            p_overlap, q_overlap, combined = mode_overlap_with_pinn(
                solver=solver,
                vector=eigenvectors[:, raw_index],
                p_pinn=p_pinn,
                q_pinn=q_pinn,
                match_mask=match_mask,
                p_weight=p_weight,
            )
            row["p_overlap_pinn"] = p_overlap
            row["q_overlap_pinn"] = q_overlap
            row["combined_overlap_pinn"] = combined
            candidates.append(row)

        rows.append(row)

    if not candidates:
        raise RuntimeError(
            "No physically admissible unstable GEP mode was found."
        )

    candidates = sorted(
        candidates,
        key=lambda item: (
            item["ci_abs_distance_to_pinn"],
            -item["combined_overlap_pinn"],
        ),
    )

    nearest_ci = candidates[0]

    best_distance = float(nearest_ci["ci_abs_distance_to_pinn"])
    ci_floor = float(ci_window_rel) * max(abs(ci_pinn), 5.0e-2)
    ci_window = max(
        float(ci_window_factor) * best_distance,
        ci_floor,
    )

    shortlist = [
        candidate
        for candidate in candidates
        if candidate["ci_abs_distance_to_pinn"] <= ci_window
    ]
    shortlist = shortlist[: max(1, int(shortlist_max))]

    # c_i defines the shortlist. The PINN modal profiles only disambiguate
    # modes inside that spectrally close set.
    pinn_matched = max(
        shortlist,
        key=lambda item: (
            item["combined_overlap_pinn"],
            item["p_overlap_pinn"],
            -item["ci_abs_distance_to_pinn"],
        ),
    )

    most_unstable = max(
        candidates,
        key=lambda item: item["omega_i"],
    )

    row_by_index = {
        int(row["raw_index"]): row
        for row in rows
    }

    for candidate in shortlist:
        row_by_index[int(candidate["raw_index"])]["in_ci_shortlist"] = True

    row_by_index[int(nearest_ci["raw_index"])]["is_nearest_ci"] = True
    row_by_index[int(pinn_matched["raw_index"])]["is_pinn_matched"] = True
    row_by_index[int(most_unstable["raw_index"])]["is_most_unstable"] = True

    spectrum = pd.DataFrame(rows)
    spectrum["growth_rank"] = np.nan

    admissible_order = (
        spectrum.loc[spectrum["physically_admissible"]]
        .sort_values("omega_i", ascending=False)
        .index
    )
    for rank, index in enumerate(admissible_order, start=1):
        spectrum.loc[index, "growth_rank"] = rank

    return spectrum, nearest_ci, pinn_matched, most_unstable


def compare_mode_to_classic(
    *,
    solver: NotebookStyleDenseGEPSolver,
    vector: np.ndarray,
    classic_fields: dict[str, np.ndarray],
    y_match_max: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    y_ref = np.asarray(classic_fields["y"], dtype=float)
    fields = split_gep_vector(
        vector,
        solver.n_points,
        solver.Mach,
    )

    interpolated = {
        name: interp_complex(
            solver.y,
            fields[name],
            y_ref,
        )
        for name in ("p", "rho", "u", "v")
    }

    reference = {
        name: np.asarray(
            classic_fields[name],
            dtype=np.complex128,
        )
        for name in ("p", "rho", "u", "v")
    }

    mask = np.abs(y_ref) <= float(y_match_max)
    if int(np.count_nonzero(mask)) < 20:
        mask = np.ones_like(y_ref, dtype=bool)

    scale = phase_alignment(
        interpolated["p"],
        reference["p"],
        y_ref,
        mask,
    )
    for name in interpolated:
        interpolated[name] = scale * interpolated[name]

    metrics = {
        "p_rel_classic": rel_l2(
            interpolated["p"],
            reference["p"],
            y_ref,
            mask,
        ),
        "rho_rel_classic": rel_l2(
            interpolated["rho"],
            reference["rho"],
            y_ref,
            mask,
        ),
        "u_rel_classic": rel_l2(
            interpolated["u"],
            reference["u"],
            y_ref,
            mask,
        ),
        "v_rel_classic": rel_l2(
            interpolated["v"],
            reference["v"],
            y_ref,
            mask,
        ),
        "p_overlap_classic": overlap_complex(
            interpolated["p"],
            reference["p"],
            y_ref,
            mask,
        ),
    }

    profile = pd.DataFrame({"y": y_ref})
    for name in ("p", "rho", "u", "v"):
        profile[f"{name}_gep_real"] = np.real(interpolated[name])
        profile[f"{name}_gep_imag"] = np.imag(interpolated[name])
        profile[f"{name}_classic_real"] = np.real(reference[name])
        profile[f"{name}_classic_imag"] = np.imag(reference[name])

    return metrics, profile


def save_spectrum_plot(
    *,
    spectrum: pd.DataFrame,
    ci_pinn: float,
    output_path: Path,
    title: str,
) -> None:
    finite = spectrum.loc[spectrum["finite"]].copy()

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(
        finite["cr"],
        finite["ci"],
        s=10,
        alpha=0.45,
        label="full finite spectrum",
    )

    admissible = finite.loc[finite["physically_admissible"]]
    if not admissible.empty:
        ax.scatter(
            admissible["cr"],
            admissible["ci"],
            s=24,
            label="admissible unstable modes",
        )

    nearest = finite.loc[finite["is_nearest_ci"]]
    if not nearest.empty:
        ax.scatter(
            nearest["cr"],
            nearest["ci"],
            marker="x",
            s=90,
            label="nearest c_i",
        )

    matched = finite.loc[finite["is_pinn_matched"]]
    if not matched.empty:
        ax.scatter(
            matched["cr"],
            matched["ci"],
            marker="*",
            s=150,
            label="PINN matched",
        )

    unstable = finite.loc[finite["is_most_unstable"]]
    if not unstable.empty:
        ax.scatter(
            unstable["cr"],
            unstable["ci"],
            marker="D",
            s=70,
            label="most unstable",
        )

    ax.scatter(
        [0.0],
        [ci_pinn],
        marker="+",
        s=120,
        label=r"PINN target $(0,c_i)$",
    )

    ax.set_xlabel(r"$c_r$")
    ax.set_ylabel(r"$c_i$")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def point_tag(mach: float, eta: float, alpha: float) -> str:
    return (
        f"M{int(round(1000 * mach)):04d}_"
        f"eta{int(round(1000 * eta)):04d}_"
        f"a{int(round(1000 * alpha)):04d}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "models_saved/production/atlas/N340/MID/model_state.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "assets/pinn_subsonic/"
            "joint_ci_mode_full_gep_test/MID"
        ),
    )
    parser.add_argument(
        "--mach-values",
        default="0.15 0.25 0.35 0.45 0.55 0.65",
    )
    parser.add_argument(
        "--eta-values",
        default="0.375 0.45 0.525",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--N", type=int, default=301)
    parser.add_argument("--mapping-kind", default="pin")
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--xi-max", type=float, default=0.98)

    parser.add_argument("--y-match-max", type=float, default=12.0)
    parser.add_argument(
        "--amplitude-floor-fraction",
        type=float,
        default=0.02,
    )

    # c_i is the primary signal.
    parser.add_argument("--ci-window-rel", type=float, default=0.02)
    parser.add_argument("--ci-window-factor", type=float, default=3.0)
    parser.add_argument("--shortlist-max", type=int, default=8)

    # Modal information is secondary.
    parser.add_argument("--p-overlap-weight", type=float, default=0.75)

    # Operational physical box. The raw full spectrum is always saved.
    parser.add_argument("--cr-physical-max", type=float, default=1.05)
    parser.add_argument("--ci-physical-max", type=float, default=2.0)

    parser.add_argument(
        "--save-admissible-vectors",
        action="store_true",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    field, ci_net, module, checkpoint_args, family = evaluate_pinn(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    anchor_df = pd.DataFrame(
        torch.load(checkpoint_path, map_location="cpu")["anchor_df"]
    )
    anchor_counts = anchor_df.groupby("Mach").size()
    if not bool((anchor_counts == 4).all()):
        raise RuntimeError(
            "Checkpoint does not have exactly four c_i anchors per Mach: "
            f"{anchor_counts.to_dict()}"
        )

    print("Sparse c_i supervision audit:")
    print(anchor_counts.to_string())
    print("Total scalar anchors:", len(anchor_df))

    mach_values = parse_float_list(args.mach_values)
    eta_values = parse_float_list(args.eta_values)

    rows: list[dict[str, Any]] = []

    for mach in mach_values:
        for eta in eta_values:
            alpha = alpha_from_eta(eta, mach)
            tag = point_tag(mach, eta, alpha)
            point_dir = output_dir / tag
            point_dir.mkdir(parents=True, exist_ok=True)

            print("=" * 100)
            print(
                f"{tag}: M={mach:.6f}, eta={eta:.6f}, "
                f"alpha={alpha:.9f}"
            )

            solver = NotebookStyleDenseGEPSolver(
                alpha=alpha,
                Mach=mach,
                n_points=args.N,
                mapping_kind=args.mapping_kind,
                mapping_scale=args.mapping_scale,
                xi_max=args.xi_max,
            )

            p_pinn, q_pinn, ci_pinn = call_pinn_profiles(
                field=field,
                ci_net=ci_net,
                module=module,
                family=family,
                y=solver.y,
                alpha=alpha,
                mach=mach,
                device=device,
            )

            match_mask = make_match_mask(
                solver.y,
                p_pinn,
                y_match_max=args.y_match_max,
                amplitude_floor_fraction=args.amplitude_floor_fraction,
            )

            # This is the complete dense diagonalization.
            eigenvalues, eigenvectors = solver.solve_all()

            (
                spectrum,
                nearest_ci,
                pinn_matched,
                most_unstable,
            ) = select_modes(
                solver=solver,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                ci_pinn=ci_pinn,
                p_pinn=p_pinn,
                q_pinn=q_pinn,
                match_mask=match_mask,
                p_weight=args.p_overlap_weight,
                ci_window_rel=args.ci_window_rel,
                ci_window_factor=args.ci_window_factor,
                shortlist_max=args.shortlist_max,
                cr_physical_max=args.cr_physical_max,
                ci_physical_max=args.ci_physical_max,
            )

            spectrum.to_csv(
                point_dir / "full_spectrum.csv",
                index=False,
            )

            save_spectrum_plot(
                spectrum=spectrum,
                ci_pinn=ci_pinn,
                output_path=point_dir / "full_spectrum.png",
                title=(
                    f"M={mach:.3f}, eta={eta:.3f}, "
                    f"alpha={alpha:.4f}"
                ),
            )

            classic_fields, ci_classic = load_classic_full_mode(
                alpha,
                mach,
            )

            matched_metrics, matched_profile = compare_mode_to_classic(
                solver=solver,
                vector=eigenvectors[:, int(pinn_matched["raw_index"])],
                classic_fields=classic_fields,
                y_match_max=args.y_match_max,
            )
            matched_profile.to_csv(
                point_dir / "pinn_matched_vs_classic.csv",
                index=False,
            )

            unstable_metrics, unstable_profile = compare_mode_to_classic(
                solver=solver,
                vector=eigenvectors[:, int(most_unstable["raw_index"])],
                classic_fields=classic_fields,
                y_match_max=args.y_match_max,
            )
            unstable_profile.to_csv(
                point_dir / "most_unstable_vs_classic.csv",
                index=False,
            )

            pd.DataFrame(
                {
                    "y": solver.y,
                    "p_pinn_real": np.real(p_pinn),
                    "p_pinn_imag": np.imag(p_pinn),
                    "q_pinn_real": np.real(q_pinn),
                    "q_pinn_imag": np.imag(q_pinn),
                    "match_mask": match_mask.astype(int),
                }
            ).to_csv(
                point_dir / "pinn_profiles_on_gep_grid.csv",
                index=False,
            )

            admissible = spectrum.loc[
                spectrum["physically_admissible"]
            ].copy()

            if args.save_admissible_vectors:
                indices = admissible["raw_index"].astype(int).to_numpy()
                np.savez_compressed(
                    point_dir / "admissible_unstable_eigenvectors.npz",
                    y=solver.y,
                    raw_indices=indices,
                    eigenvalues=eigenvalues[indices],
                    eigenvectors=eigenvectors[:, indices],
                )

            same_mode = bool(
                int(pinn_matched["raw_index"])
                == int(most_unstable["raw_index"])
            )

            row = {
                "Mach": mach,
                "eta": eta,
                "alpha": alpha,
                "N": int(args.N),
                "mapping_kind": args.mapping_kind,
                "mapping_scale": float(args.mapping_scale),
                "xi_max": float(args.xi_max),
                "ci_pinn": ci_pinn,
                "ci_classic": float(ci_classic),
                "ci_pinn_abs_err": abs(ci_pinn - float(ci_classic)),
                "ci_pinn_rel_err": (
                    abs(ci_pinn - float(ci_classic))
                    / max(abs(float(ci_classic)), 1.0e-12)
                ),
                "n_raw_eigenvalues": int(len(eigenvalues)),
                "n_finite_eigenvalues": int(spectrum["finite"].sum()),
                "n_solver_unstable_modes": int(
                    spectrum["solver_finite_mode"].sum()
                ),
                "n_physically_admissible_modes": int(
                    spectrum["physically_admissible"].sum()
                ),
                "nearest_ci_raw_index": int(nearest_ci["raw_index"]),
                "nearest_ci_cr": float(nearest_ci["cr"]),
                "nearest_ci_ci": float(nearest_ci["ci"]),
                "nearest_ci_rel_distance": float(
                    nearest_ci["ci_rel_distance_to_pinn"]
                ),
                "pinn_matched_raw_index": int(
                    pinn_matched["raw_index"]
                ),
                "pinn_matched_cr": float(pinn_matched["cr"]),
                "pinn_matched_ci": float(pinn_matched["ci"]),
                "pinn_matched_omega_i": float(
                    pinn_matched["omega_i"]
                ),
                "pinn_matched_ci_rel_distance": float(
                    pinn_matched["ci_rel_distance_to_pinn"]
                ),
                "pinn_matched_p_overlap": float(
                    pinn_matched["p_overlap_pinn"]
                ),
                "pinn_matched_q_overlap": float(
                    pinn_matched["q_overlap_pinn"]
                ),
                "pinn_matched_combined_overlap": float(
                    pinn_matched["combined_overlap_pinn"]
                ),
                "most_unstable_raw_index": int(
                    most_unstable["raw_index"]
                ),
                "most_unstable_cr": float(most_unstable["cr"]),
                "most_unstable_ci": float(most_unstable["ci"]),
                "most_unstable_omega_i": float(
                    most_unstable["omega_i"]
                ),
                "most_unstable_p_overlap": float(
                    most_unstable["p_overlap_pinn"]
                ),
                "most_unstable_q_overlap": float(
                    most_unstable["q_overlap_pinn"]
                ),
                "most_unstable_combined_overlap": float(
                    most_unstable["combined_overlap_pinn"]
                ),
                "pinn_matched_is_most_unstable": same_mode,
                "pinn_matched_ci_abs_err_classic": abs(
                    float(pinn_matched["ci"]) - float(ci_classic)
                ),
                "pinn_matched_ci_rel_err_classic": (
                    abs(float(pinn_matched["ci"]) - float(ci_classic))
                    / max(abs(float(ci_classic)), 1.0e-12)
                ),
                "most_unstable_ci_abs_err_classic": abs(
                    float(most_unstable["ci"]) - float(ci_classic)
                ),
                "most_unstable_ci_rel_err_classic": (
                    abs(float(most_unstable["ci"]) - float(ci_classic))
                    / max(abs(float(ci_classic)), 1.0e-12)
                ),
            }

            for key, value in matched_metrics.items():
                row[f"pinn_matched_{key}"] = value
            for key, value in unstable_metrics.items():
                row[f"most_unstable_{key}"] = value

            rows.append(row)

            print(
                "c_i:",
                f"PINN={ci_pinn:.9e}",
                f"classic={float(ci_classic):.9e}",
            )
            print(
                "nearest c_i mode:",
                f"c=({nearest_ci['cr']:.6e},"
                f"{nearest_ci['ci']:.6e})",
            )
            print(
                "PINN matched mode:",
                f"c=({pinn_matched['cr']:.6e},"
                f"{pinn_matched['ci']:.6e})",
                f"O_p={pinn_matched['p_overlap_pinn']:.8f}",
                f"O_q={pinn_matched['q_overlap_pinn']:.8f}",
            )
            print(
                "most unstable mode:",
                f"c=({most_unstable['cr']:.6e},"
                f"{most_unstable['ci']:.6e})",
            )
            print(
                "PINN matched == most unstable:",
                same_mode,
            )
            print(
                "matched vs classical:",
                f"ci_rel={row['pinn_matched_ci_rel_err_classic']:.4e}",
                f"p_rel={row['pinn_matched_p_rel_classic']:.4e}",
                f"u_rel={row['pinn_matched_u_rel_classic']:.4e}",
                f"v_rel={row['pinn_matched_v_rel_classic']:.4e}",
            )

    summary = pd.DataFrame(rows).sort_values(
        ["Mach", "eta"]
    ).reset_index(drop=True)
    summary.to_csv(output_dir / "summary.csv", index=False)

    metric_columns = [
        "ci_pinn_rel_err",
        "pinn_matched_ci_rel_err_classic",
        "pinn_matched_p_rel_classic",
        "pinn_matched_rho_rel_classic",
        "pinn_matched_u_rel_classic",
        "pinn_matched_v_rel_classic",
        "pinn_matched_p_overlap_classic",
        "pinn_matched_p_overlap",
        "pinn_matched_q_overlap",
    ]

    metrics: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "field_family": family,
        "n_points_tested": int(len(summary)),
        "all_pinn_matches_most_unstable": bool(
            summary["pinn_matched_is_most_unstable"].all()
        ),
        "n_pinn_matches_most_unstable": int(
            summary["pinn_matched_is_most_unstable"].sum()
        ),
        "n_ci_anchors_per_mach": 4,
        "n_anchor_mach_values": int(anchor_df["Mach"].nunique()),
        "n_total_scalar_ci_anchors": int(len(anchor_df)),
    }

    for column in metric_columns:
        values = pd.to_numeric(
            summary[column],
            errors="coerce",
        )
        values = values[np.isfinite(values)]
        if not values.empty:
            metrics[f"{column}_mean"] = float(values.mean())
            metrics[f"{column}_max"] = float(values.max())

    (output_dir / "summary_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=" * 100)
    print("GLOBAL SUMMARY")
    print(summary[
        [
            "Mach",
            "eta",
            "ci_pinn_rel_err",
            "pinn_matched_ci_rel_err_classic",
            "pinn_matched_p_overlap",
            "pinn_matched_q_overlap",
            "pinn_matched_is_most_unstable",
            "pinn_matched_p_rel_classic",
            "pinn_matched_u_rel_classic",
            "pinn_matched_v_rel_classic",
        ]
    ].to_string(index=False))
    print()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print()
    print("Wrote:", output_dir / "summary.csv")
    print("Wrote:", output_dir / "summary_metrics.json")


if __name__ == "__main__":
    main()
