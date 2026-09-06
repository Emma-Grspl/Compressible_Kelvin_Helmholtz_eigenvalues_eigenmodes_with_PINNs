#!/usr/bin/env python3
"""
Exact local modal comparison at alpha=0.5, M=0.5.

This script is intentionally self-contained for the middle eta-aware chart:
it does not import the generic joint-atlas loader nor the missing pq_legacy /
pQscaled training modules.

Outputs
-------
- PDF + PNG: classical / direct PINN / PINN+GEP overlay for p, rho, v, u
- CSV: aligned complex fields used in the figure
- JSON: eigenvalues, chart and numerical metadata
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[8]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from src.scripts.gep.selection.solve_dense_gep_notebook_style import (
    NotebookStyleDenseGEPSolver,
)
from src.scripts.evaluation.evaluate_kh_subsonic_fixed_mach_modal_candidates import (
    load_classic_full_mode,
)

FIELDS = ("p", "rho", "u", "v")


class CiAtlasNet(nn.Module):
    def __init__(
        self,
        *,
        mach_min: float,
        mach_max: float,
        eta_min: float,
        eta_max: float,
        ci_init: float,
        width: int = 96,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.mach_min = float(mach_min)
        self.mach_max = float(mach_max)
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)

        layers: list[nn.Module] = [nn.Linear(5, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

        initial = max(float(ci_init), 1.0e-5)
        inverse_softplus = math.log(math.expm1(initial))
        with torch.no_grad():
            final = self.net[-1]
            assert isinstance(final, nn.Linear)
            final.weight.mul_(0.05)
            final.bias.fill_(inverse_softplus)

    def forward(
        self,
        alpha: torch.Tensor,
        mach: torch.Tensor,
    ) -> torch.Tensor:
        cutoff = torch.sqrt(torch.clamp(1.0 - mach**2, min=1.0e-12))
        eta = alpha / cutoff
        e = 2.0 * (eta - self.eta_min) / (
            self.eta_max - self.eta_min
        ) - 1.0
        m = 2.0 * (mach - self.mach_min) / (
            self.mach_max - self.mach_min
        ) - 1.0
        features = torch.cat([e, m, e**2, e * m, m**2], dim=1)
        return F.softplus(self.net(features)) + 1.0e-10


class EtaAwareFieldPQNet(nn.Module):
    def __init__(
        self,
        *,
        ymax: float,
        alpha_min: float,
        alpha_max: float,
        mach_min: float,
        mach_max: float,
        eta_min: float,
        eta_max: float,
        width: int = 256,
        depth: int = 7,
        n_freq: int = 12,
    ) -> None:
        super().__init__()
        self.ymax = float(ymax)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.mach_min = float(mach_min)
        self.mach_max = float(mach_max)
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        self.n_freq = int(n_freq)

        in_dim = 7 + 2 * self.n_freq
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.SiLU()])
        layers.append(nn.Linear(width, 4))
        self.net = nn.Sequential(*layers)

    def normalize_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        den = max(self.alpha_max - self.alpha_min, 1.0e-12)
        return 2.0 * (alpha - self.alpha_min) / den - 1.0

    def normalize_mach(self, mach: torch.Tensor) -> torch.Tensor:
        den = max(self.mach_max - self.mach_min, 1.0e-12)
        return 2.0 * (mach - self.mach_min) / den - 1.0

    def normalize_eta(self, eta: torch.Tensor) -> torch.Tensor:
        den = max(self.eta_max - self.eta_min, 1.0e-12)
        return 2.0 * (eta - self.eta_min) / den - 1.0

    def features(
        self,
        y: torch.Tensor,
        alpha: torch.Tensor,
        mach: torch.Tensor,
    ) -> torch.Tensor:
        z = y / self.ymax
        alpha_cut = torch.sqrt(
            torch.clamp(1.0 - mach**2, min=1.0e-14)
        )
        eta = alpha / alpha_cut

        a = self.normalize_alpha(alpha)
        m = self.normalize_mach(mach)
        e = self.normalize_eta(eta)
        k = 2.0 * alpha_cut - 1.0
        ay = torch.tanh(alpha * y)
        ky = torch.tanh(alpha_cut * y)

        features = [z, a, m, e, k, ay, ky]
        for j in range(1, self.n_freq + 1):
            features.append(torch.sin(math.pi * j * z))
            features.append(torch.cos(math.pi * j * z))
        return torch.cat(features, dim=1)

    def forward(
        self,
        y: torch.Tensor,
        alpha: torch.Tensor,
        mach: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.net(self.features(y, alpha, mach))
        pr = output[:, 0:1]
        pi = output[:, 1:2]
        qr = output[:, 2:3]
        qi = output[:, 3:4]
        return torch.complex(pr, pi), torch.complex(qr, qi)


def physical_eta(alpha: float, mach: float) -> float:
    return alpha / math.sqrt(max(1.0 - mach**2, 1.0e-30))


def normalize_training_plan(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t").copy()
    required = {
        "chart_id",
        "output_dir",
        "mach_min",
        "mach_max",
        "eta_min",
        "eta_max",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Training plan missing columns: {missing}")
    return frame


def route_chart(
    frame: pd.DataFrame,
    alpha: float,
    mach: float,
) -> pd.Series:
    eta = physical_eta(alpha, mach)
    tolerance = 5.0e-10
    candidates = frame.loc[
        (pd.to_numeric(frame["mach_min"]) - tolerance <= mach)
        & (mach <= pd.to_numeric(frame["mach_max"]) + tolerance)
        & (pd.to_numeric(frame["eta_min"]) - tolerance <= eta)
        & (eta <= pd.to_numeric(frame["eta_max"]) + tolerance)
    ].copy()

    if candidates.empty:
        raise RuntimeError(
            f"No chart covers alpha={alpha}, M={mach}, eta={eta}."
        )

    candidates["area"] = (
        (
            pd.to_numeric(candidates["mach_max"])
            - pd.to_numeric(candidates["mach_min"])
        )
        * (
            pd.to_numeric(candidates["eta_max"])
            - pd.to_numeric(candidates["eta_min"])
        )
    )
    mach_center = 0.5 * (
        pd.to_numeric(candidates["mach_min"])
        + pd.to_numeric(candidates["mach_max"])
    )
    eta_center = 0.5 * (
        pd.to_numeric(candidates["eta_min"])
        + pd.to_numeric(candidates["eta_max"])
    )
    candidates["distance"] = (
        (mach - mach_center) ** 2 + (eta - eta_center) ** 2
    )
    return candidates.sort_values(
        ["area", "distance", "chart_id"],
        kind="mergesort",
    ).iloc[0]


def locate_checkpoint(
    chart: pd.Series,
    atlas_root: Path,
) -> Path:
    output_dir = Path(str(chart["output_dir"]))
    candidates = [
        output_dir / "model_state.pt",
        ROOT / output_dir / "model_state.pt",
        atlas_root / str(chart["chart_id"]) / "model_state.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    chart_id = str(chart["chart_id"]).lower()
    all_checkpoints = sorted(atlas_root.rglob("model_state.pt"))
    path_matches = [
        path for path in all_checkpoints
        if chart_id in str(path.parent).lower()
    ]
    if len(path_matches) == 1:
        return path_matches[0].resolve()

    # Last robust fallback: inspect checkpoint metadata.
    for path in all_checkpoints:
        try:
            checkpoint = torch.load(path, map_location="cpu")
            args = dict(checkpoint.get("args", {}))
            text = " ".join(
                [
                    str(args.get("chart_id", "")),
                    str(args.get("output_dir", "")),
                    str(path.parent),
                ]
            ).lower()
            if chart_id in text:
                return path.resolve()
        except Exception:
            continue

    raise FileNotFoundError(
        f"Could not locate model_state.pt for chart {chart['chart_id']} "
        f"under {atlas_root}."
    )


def infer_input_dimension(
    state_dict: dict[str, torch.Tensor],
) -> int:
    candidates: list[tuple[int, int]] = []
    for key, tensor in state_dict.items():
        if key.endswith(".weight") and tensor.ndim == 2:
            pieces = key.split(".")
            index = next(
                (int(piece) for piece in pieces if piece.isdigit()),
                10**9,
            )
            candidates.append((index, int(tensor.shape[1])))
    if not candidates:
        raise RuntimeError("No linear weight found in field_state_dict.")
    return min(candidates)[1]


def load_joint_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = dict(checkpoint.get("args", {}))
    state = checkpoint["field_state_dict"]

    input_dimension = infer_input_dimension(state)
    if input_dimension < 9 or (input_dimension - 7) % 2 != 0:
        raise RuntimeError(
            "Unsupported eta-aware input dimension: "
            f"{input_dimension} in {checkpoint_path}"
        )

    # Eta-aware input: 7 physical features + sin/cos pairs.
    n_freq = (input_dimension - 7) // 2
    print(f"Inferred eta-aware architecture: n_freq={n_freq}")

    mach_min = float(args["mach_min"])
    mach_max = float(args["mach_max"])
    eta_min = float(args["eta_min"])
    eta_max = float(args["eta_max"])
    alpha_corners = [
        eta * math.sqrt(max(1.0 - mach**2, 1.0e-14))
        for eta in (eta_min, eta_max)
        for mach in (mach_min, mach_max)
    ]

    field = EtaAwareFieldPQNet(
        ymax=float(args.get("ymax", 100.0)),
        alpha_min=min(alpha_corners),
        alpha_max=max(alpha_corners),
        mach_min=mach_min,
        mach_max=mach_max,
        eta_min=eta_min,
        eta_max=eta_max,
        width=int(args.get("width", 256)),
        depth=int(args.get("depth", 7)),
        n_freq=n_freq,
    ).to(device=device, dtype=torch.float64)
    field.load_state_dict(state, strict=True)
    field.eval()

    anchor_df = pd.DataFrame(checkpoint.get("anchor_df", {}))
    if anchor_df.empty:
        raise RuntimeError("Checkpoint has no anchor_df.")

    ci_net = CiAtlasNet(
        mach_min=mach_min,
        mach_max=mach_max,
        eta_min=eta_min,
        eta_max=eta_max,
        ci_init=float(anchor_df["ci"].mean()),
        width=int(args.get("ci_width", 96)),
        depth=int(args.get("ci_depth", 3)),
    ).to(device=device, dtype=torch.float64)
    ci_net.load_state_dict(checkpoint["ci_state_dict"], strict=True)
    ci_net.eval()
    return field, ci_net, args


def evaluate_direct_pq(
    field: nn.Module,
    ci_net: nn.Module,
    y: np.ndarray,
    alpha: float,
    mach: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    y_tensor = torch.tensor(
        np.asarray(y, dtype=float)[:, None],
        device=device,
        dtype=torch.float64,
    )
    alpha_tensor = torch.full_like(y_tensor, float(alpha))
    mach_tensor = torch.full_like(y_tensor, float(mach))
    with torch.no_grad():
        p, q = field(y_tensor, alpha_tensor, mach_tensor)
        ci = ci_net(alpha_tensor[:1], mach_tensor[:1])
    return (
        p.detach().cpu().numpy().reshape(-1),
        q.detach().cpu().numpy().reshape(-1),
        float(ci.detach().cpu().item()),
    )


def interp_complex(
    x_source: np.ndarray,
    values: np.ndarray,
    x_target: np.ndarray,
) -> np.ndarray:
    return (
        np.interp(x_target, x_source, np.real(values))
        + 1j * np.interp(x_target, x_source, np.imag(values))
    )


def trapz(values: np.ndarray, x: np.ndarray) -> complex:
    return np.trapezoid(values, x)


def phase_scale(
    source: np.ndarray,
    target: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> complex:
    denominator = trapz(
        np.conjugate(source[mask]) * source[mask],
        y[mask],
    )
    if abs(denominator) <= 1.0e-30:
        return 1.0 + 0.0j
    return complex(
        trapz(
            np.conjugate(source[mask]) * target[mask],
            y[mask],
        )
        / denominator
    )


def split_gep_vector(
    vector: np.ndarray,
    n_points: int,
    mach: float,
) -> dict[str, np.ndarray]:
    vector = np.asarray(vector, dtype=np.complex128)
    u = vector[:n_points]
    v = vector[n_points : 2 * n_points]
    p = vector[2 * n_points : 3 * n_points]
    rho = p * mach**2
    return {"p": p, "rho": rho, "u": u, "v": v}


def select_central_mode(
    eigenvalues: np.ndarray,
) -> int:
    values = np.asarray(eigenvalues, dtype=np.complex128)
    candidates = np.where(
        np.isfinite(values.real)
        & np.isfinite(values.imag)
        & (values.imag > 0.0)
        & (values.imag <= 2.0)
        & (np.abs(values.real) <= 0.05)
    )[0]
    if len(candidates) == 0:
        raise RuntimeError("No unstable central GEP mode found.")
    return int(candidates[np.argmax(values[candidates].imag)])


def direct_fields(
    y: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    alpha: float,
    mach: float,
    ci: float,
) -> dict[str, np.ndarray]:
    ubar = np.tanh(y)
    ubar_y = 1.0 - ubar**2
    c = 1j * ci
    denominator = ubar - c
    rho = mach**2 * p
    v = -q / (1j * alpha * denominator)
    u = -(ubar_y * v + 1j * alpha * p) / (
        1j * alpha * denominator
    )
    return {"p": p, "rho": rho, "u": u, "v": v}


def align_family(
    family: dict[str, np.ndarray],
    reference_p: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    scale = phase_scale(family["p"], reference_p, y, mask)
    return {key: scale * value for key, value in family.items()}


def save_outputs(
    output_dir: Path,
    result: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    y = result["y"]

    frame = pd.DataFrame({"y": y})
    for method in ("classic", "direct", "gep"):
        for field in FIELDS:
            values = result[method][field]
            frame[f"{field}_{method}_real"] = np.real(values)
            frame[f"{field}_{method}_imag"] = np.imag(values)
    for key in ("alpha", "Mach", "eta", "ci_classic", "ci_pinn", "ci_gep"):
        frame[key] = result[key]
    frame["chart_id"] = result["chart_id"]
    frame.to_csv(
        output_dir / "mode_M050_alpha0500_fields.csv",
        index=False,
    )

    metadata = {
        key: value
        for key, value in result.items()
        if key not in {"y", "classic", "direct", "gep"}
    }
    (output_dir / "mode_M050_alpha0500_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def plot_mode(
    output_stem: Path,
    result: dict[str, Any],
) -> None:
    # Méthode : couleur + type de trait.
    # Composante : les parties imaginaires portent de petits cercles ouverts.
    styles = {
        "classic": {
            "color": "black",
            "linestyle": "--",
            "linewidth_real": 3.2,
            "linewidth_imag": 2.6,
            "zorder": 30,
            "label": "Classical shooting",
        },
        "direct": {
            "color": "#3b73b9",
            "linestyle": "-.",
            "linewidth_real": 2.1,
            "linewidth_imag": 1.9,
            "zorder": 20,
            "label": "Direct PINN",
        },
        "gep": {
            "color": "#e6782d",
            "linestyle": "-",
            "linewidth_real": 2.4,
            "linewidth_imag": 2.1,
            "zorder": 25,
            "label": "PINN + GEP",
        },
    }

    panels = [
        ("p", r"Pressure $\hat p$"),
        ("rho", r"Density $\hat\rho$"),
        ("v", r"Transverse velocity $\hat v$"),
        ("u", r"Streamwise velocity $\hat u$"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 9.4),
        sharex=True,
    )
    axes = axes.ravel()
    y = result["y"]

    # Le classique est tracé en dernier pour rester visible lorsqu'il est
    # presque confondu avec la reconstruction GEP.
    drawing_order = ("direct", "gep", "classic")

    for axis, (field, title) in zip(axes, panels):
        for method in drawing_order:
            style = styles[method]
            values = result[method][field]

            # Partie réelle : aucun marqueur.
            axis.plot(
                y,
                np.real(values),
                color=style["color"],
                linewidth=style["linewidth_real"],
                linestyle=style["linestyle"],
                zorder=style["zorder"],
                alpha=1.0,
                solid_capstyle="round",
                dash_capstyle="round",
            )

            # Partie imaginaire : même style de méthode, petits cercles ouverts.
            axis.plot(
                y,
                np.imag(values),
                color=style["color"],
                linewidth=style["linewidth_imag"],
                linestyle=style["linestyle"],
                marker="o",
                markevery=32,
                markersize=2.8,
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=0.8,
                zorder=style["zorder"] + 1,
                alpha=0.95,
                solid_capstyle="round",
                dash_capstyle="round",
            )

        axis.axhline(
            0.0,
            color="0.72",
            linewidth=0.7,
            zorder=0,
        )
        axis.set_title(title, fontsize=14)
        axis.set_xlabel(r"$y$")
        axis.set_ylabel("Amplitude")
        axis.grid(alpha=0.22, zorder=0)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=styles[key]["color"],
            linewidth=styles[key]["linewidth_real"],
            linestyle=styles[key]["linestyle"],
            label=styles[key]["label"],
        )
        for key in ("classic", "direct", "gep")
    ]

    component_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            linewidth=2.2,
            linestyle="-",
            label="Real part",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            linewidth=2.0,
            linestyle="-",
            marker="o",
            markersize=3.2,
            markerfacecolor="white",
            markeredgecolor="0.25",
            label="Imaginary part",
        ),
    ]

    method_legend = fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.36, 0.944),
        ncol=3,
        frameon=False,
        title="Method",
        handlelength=3.1,
        columnspacing=1.8,
    )
    fig.add_artist(method_legend)

    fig.legend(
        handles=component_handles,
        loc="upper center",
        bbox_to_anchor=(0.82, 0.944),
        ncol=2,
        frameon=False,
        title="Component",
        handlelength=2.6,
    )

    fig.suptitle(
        (
            r"Subsonic mode comparison at $\alpha=0.500$ and $M=0.500$"
            "\n"
            rf"$c_i^{{class}}={result['ci_classic']:.6f}$, "
            rf"$c_i^{{PINN}}={result['ci_pinn']:.6f}$, "
            rf"$c_i^{{GEP}}={result['ci_gep']:.6f}$"
        ),
        fontsize=15,
        y=0.995,
    )

    fig.tight_layout(
        rect=(0.025, 0.025, 0.975, 0.87)
    )

    output_stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=320,
        bbox_inches="tight",
    )
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-plan",
        type=Path,
        default=Path(
            "archive/csv/assets/pinn_subsonic/joint_ci_mode_atlas_v2/training_plan.tsv"
        ),
    )
    parser.add_argument(
        "--atlas-root",
        type=Path,
        default=Path(
            "assets/pinn_subsonic/joint_ci_mode_atlas_v2"
        ),
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(
            "assets/pinn_subsonic/joint_ci_mode_final_assets_v3"
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--N", type=int, default=401)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    training_plan = (
        args.training_plan
        if args.training_plan.is_absolute()
        else ROOT / args.training_plan
    )
    atlas_root = (
        args.atlas_root
        if args.atlas_root.is_absolute()
        else ROOT / args.atlas_root
    )
    asset_root = (
        args.asset_root
        if args.asset_root.is_absolute()
        else ROOT / args.asset_root
    )

    plan = normalize_training_plan(training_plan)
    chart = route_chart(plan, args.alpha, args.mach)
    checkpoint_path = locate_checkpoint(chart, atlas_root)
    print(f"Routed chart: {chart['chart_id']}")
    print(f"Checkpoint: {checkpoint_path}")

    device = torch.device(args.device)
    field, ci_net, _ = load_joint_checkpoint(checkpoint_path, device)

    solver = NotebookStyleDenseGEPSolver(
        alpha=float(args.alpha),
        Mach=float(args.mach),
        n_points=int(args.N),
        mapping_kind="pin",
        mapping_scale=5.0,
        xi_max=0.98,
    )

    p_pinn, q_pinn, ci_pinn = evaluate_direct_pq(
        field,
        ci_net,
        solver.y,
        float(args.alpha),
        float(args.mach),
        device,
    )

    eigenvalues, eigenvectors = solver.solve_all()
    selected_index = select_central_mode(eigenvalues)
    selected_value = complex(eigenvalues[selected_index])

    classic_raw, ci_classic = load_classic_full_mode(
        float(args.alpha),
        float(args.mach),
    )
    y = np.asarray(classic_raw["y"], dtype=float)
    classic = {
        field_name: np.asarray(
            classic_raw[field_name],
            dtype=np.complex128,
        )
        for field_name in FIELDS
    }

    gep_native = split_gep_vector(
        eigenvectors[:, selected_index],
        solver.n_points,
        float(args.mach),
    )
    gep = {
        name: interp_complex(solver.y, gep_native[name], y)
        for name in FIELDS
    }

    p_direct = interp_complex(solver.y, p_pinn, y)
    q_direct = interp_complex(solver.y, q_pinn, y)
    direct = direct_fields(
        y,
        p_direct,
        q_direct,
        float(args.alpha),
        float(args.mach),
        float(ci_pinn),
    )

    mask = np.abs(y) <= 12.0
    if int(mask.sum()) < 20:
        mask = np.ones_like(y, dtype=bool)

    direct = align_family(direct, classic["p"], y, mask)
    gep = align_family(gep, classic["p"], y, mask)

    normalization = float(np.max(np.abs(classic["rho"])))
    if not math.isfinite(normalization) or normalization < 1.0e-30:
        normalization = 1.0
    for family in (classic, direct, gep):
        for name in FIELDS:
            family[name] = family[name] / normalization

    result = {
        "alpha": float(args.alpha),
        "Mach": float(args.mach),
        "eta": float(physical_eta(args.alpha, args.mach)),
        "chart_id": str(chart["chart_id"]),
        "checkpoint": str(checkpoint_path),
        "N": int(args.N),
        "ci_classic": float(ci_classic),
        "ci_pinn": float(ci_pinn),
        "ci_gep": float(selected_value.imag),
        "cr_gep": float(selected_value.real),
        "y": y,
        "classic": classic,
        "direct": direct,
        "gep": gep,
    }

    data_dir = asset_root / "data" / "article_M050_alpha0500"
    figure_stem = (
        asset_root
        / "figures"
        / "Fig_subsonic_mode_comparison_M050_alpha0500_classical_PINN_GEP"
    )
    save_outputs(data_dir, result)
    plot_mode(figure_stem, result)

    print(figure_stem.with_suffix(".pdf"))
    print(figure_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
