from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.subsonic.mstab17_subsonic_solver import Mstab17SubsonicSolver  # noqa: E402
from classical_solver.subsonic.robust_subsonic_shooting import RobustSubsonicShootingSolver  # noqa: E402
from src.models.kh_subsonic_pinn import build_fixed_mach_model_from_config, load_fixed_mach_state_dict_compat  # noqa: E402
from src.physics.kh_subsonic_residual import (  # noqa: E402
    base_velocity,
    base_velocity_derivative,
    dy_dxi,
    reconstruct_pressure_p_y_from_riccati,
    xi_to_y,
)


RUNS = [
    (
        "physics_only",
        "Physique pur",
        Path("model_saved/kh_subsonic_fixed_mach_M05_alpha010_080_riccati_pure_physics_reference"),
    ),
    (
        "hybrid_ci4",
        "Hybride 4 points",
        Path("model_saved/kh_subsonic_fixed_mach_M05_alpha010_080_ci_sparse_reference/hybrid_ci4_fixed"),
    ),
    (
        "hybrid_ci8",
        "Hybride 8 points",
        Path("model_saved/kh_subsonic_fixed_mach_M05_alpha010_080_ci_sparse_reference/hybrid_ci8_fixed"),
    ),
    (
        "hybrid_ci16",
        "Hybride 16 points",
        Path("model_saved/kh_subsonic_fixed_mach_M05_alpha010_080_ci_sparse_reference/hybrid_ci16_fixed"),
    ),
]

PALETTE = {
    "physics_only": "#c84c09",
    "hybrid_ci4": "#1f77b4",
    "hybrid_ci8": "#0b6e4f",
    "hybrid_ci16": "#8e44ad",
}

REFERENCE_CI_CANDIDATES = [
    ROOT_DIR / "assets/pinn_subsonic/mach_fixed/frozen_alpha_sweep_modefocus_best/ci_curve_vs_reference.csv",
    ROOT_DIR / "archive/repo_cleanup_2026-04-24/model_saved/kh_subsonic_fixed_mach_M05/ci_curve_vs_reference.csv",
]

MODE_REFERENCE_RUN = ROOT_DIR / "assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current"
MODE_REFERENCE_METRICS_CSV = MODE_REFERENCE_RUN / "modes/classic_vs_pinn_modes_overlay.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construit les comparaisons subsoniques M=0.5 en HTML/CSV sans matplotlib."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("assets/pinn_subsonic/experiment_M05_alpha010_080_reference_2026-06-21"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Par defaut: <experiment-root>/analysis_plotly",
    )
    parser.add_argument("--num-alpha", type=int, default=81)
    parser.add_argument("--mode-alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--n-y", type=int, default=1001)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def safe_slug(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def run_path(experiment_root: Path, relative_run_dir: Path) -> Path:
    return experiment_root / relative_run_dir


def load_model(run_dir: Path, device: torch.device):
    config_df = pd.read_csv(run_dir / "config.csv")
    history = pd.read_csv(run_dir / "history.csv")
    config = config_df.iloc[0]
    model = build_fixed_mach_model_from_config(config)
    state_dict = torch.load(run_dir / "model_best.pt", map_location=device)
    load_fixed_mach_state_dict_compat(model, state_dict)
    model.to(device)
    model.eval()
    return model, config, history


def solve_reference_curve(mach: float, alpha_min: float, alpha_max: float, num_alpha: int) -> pd.DataFrame:
    alpha_values = np.linspace(alpha_min, alpha_max, num_alpha)
    if abs(float(mach) - 0.5) < 1e-12:
        for candidate in REFERENCE_CI_CANDIDATES:
            if candidate.exists():
                df = pd.read_csv(candidate)
                if "ci_reference" not in df.columns:
                    continue
                alpha_ref = df["alpha"].to_numpy(dtype=float)
                ci_ref = df["ci_reference"].to_numpy(dtype=float)
                if float(alpha_min) >= float(alpha_ref.min()) and float(alpha_max) <= float(alpha_ref.max()):
                    return pd.DataFrame(
                        {
                            "alpha": alpha_values,
                            "ci_reference": np.interp(alpha_values, alpha_ref, ci_ref),
                        }
                    )

    rows = []
    for alpha in alpha_values:
        result = RobustSubsonicShootingSolver(alpha=float(alpha), Mach=float(mach)).solve()
        rows.append({"alpha": float(alpha), "ci_reference": float(result.ci)})
    return pd.DataFrame(rows)


def normalize_full_mode(y: np.ndarray, u: np.ndarray, v: np.ndarray, p: np.ndarray, rho: np.ndarray) -> dict[str, np.ndarray]:
    idx = int(np.argmax(np.abs(rho)))
    if np.abs(rho[idx]) > 0.0:
        phase = np.exp(-1j * np.angle(rho[idx]))
        u = u * phase
        v = v * phase
        p = p * phase
        rho = rho * phase

    if np.max(np.real(rho)) < abs(np.min(np.real(rho))):
        u = -u
        v = -v
        p = -p
        rho = -rho

    scale = max(np.max(np.abs(np.real(rho))), np.max(np.abs(np.imag(rho))), 1e-12)
    return {
        "y": np.asarray(y, dtype=float),
        "u": u / scale,
        "v": v / scale,
        "p": p / scale,
        "rho": rho / scale,
    }


def interp_complex(y_src: np.ndarray, f_src: np.ndarray, y_dst: np.ndarray) -> np.ndarray:
    return np.interp(y_dst, y_src, np.real(f_src)) + 1j * np.interp(y_dst, y_src, np.imag(f_src))


def load_classic_full_mode(alpha: float, mach: float) -> tuple[dict[str, np.ndarray], float]:
    solver = Mstab17SubsonicSolver(alpha=float(alpha), Mach=float(mach))
    result = solver.solve()
    sol_left, sol_right, _ = solver.get_trajectories(result.ci, ln_p_start_right=result.ln_p_start_right)

    y_left = np.asarray(sol_left.t)
    y_right = np.asarray(sol_right.t)
    k_left = np.asarray(sol_left.y[0])
    q_left = np.asarray(sol_left.y[1])
    ln_p_left = np.asarray(sol_left.y[2])
    phi_left = np.asarray(sol_left.y[3])
    k_right = np.asarray(sol_right.y[0])
    q_right = np.asarray(sol_right.y[1])
    ln_p_right = np.asarray(sol_right.y[2])
    phi_right = np.asarray(sol_right.y[3])

    abs_p_left = np.exp(ln_p_left)
    abs_p_right = np.exp(ln_p_right)
    phi_left_0 = solver._interp_component(0.0, sol_left, 3)
    phi_right_0 = solver._interp_component(0.0, sol_right, 3)
    phase_shift = phi_left_0 - phi_right_0

    p_left = abs_p_left * np.exp(1j * phi_left)
    p_right = abs_p_right * np.exp(1j * (phi_right + phase_shift))
    gamma_left = k_left + 1j * q_left
    gamma_right = k_right + 1j * q_right

    mask_left = y_left < 0.0
    y = np.concatenate([y_left[mask_left], y_right[::-1]])
    p = np.concatenate([p_left[mask_left], p_right[::-1]])
    gamma = np.concatenate([gamma_left[mask_left], gamma_right[::-1]])

    p_y = gamma * p
    c = 1j * float(result.ci)
    u_bar = np.tanh(y)
    du_bar = 1.0 / np.cosh(y) ** 2
    i_alpha = 1j * float(alpha)
    v = -p_y / (i_alpha * (u_bar - c))
    u = -(du_bar * v + i_alpha * p) / (i_alpha * (u_bar - c))
    rho = p * (float(mach) ** 2)
    return normalize_full_mode(y, u, v, p, rho), float(result.ci)


def load_pinn_full_mode(run_dir: Path, *, alpha: float, n_y: int, device: torch.device) -> tuple[dict[str, np.ndarray], float, float]:
    config = pd.read_csv(run_dir / "config.csv").iloc[0]
    model = build_fixed_mach_model_from_config(config)
    state_dict = torch.load(run_dir / "model_best.pt", map_location=device)
    load_fixed_mach_state_dict_compat(model, state_dict)
    model.to(device)
    model.eval()

    xi = torch.linspace(-0.98, 0.98, int(n_y), device=device).view(-1, 1)
    xi.requires_grad_(True)
    alpha_tensor = torch.full_like(xi, float(alpha))

    if str(config.get("mode_representation", "cartesian")) == "riccati":
        pr, pi, p_y, _, y_t = reconstruct_pressure_p_y_from_riccati(model, xi, alpha_tensor, anchor_xi=0.0)
    else:
        pred = model(xi, alpha_tensor)
        pr = pred[:, 0:1]
        pi = pred[:, 1:2]
        y_t = xi_to_y(xi, model.get_mapping_scale().detach())
        p_r_xi = torch.autograd.grad(pr, xi, grad_outputs=torch.ones_like(pr), create_graph=False, retain_graph=True)[0]
        p_i_xi = torch.autograd.grad(pi, xi, grad_outputs=torch.ones_like(pi), create_graph=False, retain_graph=True)[0]
        p_xi = torch.complex(p_r_xi, p_i_xi)
        y_xi = dy_dxi(xi, model.get_mapping_scale().detach())
        p_y = p_xi / y_xi

    p = torch.complex(pr, pi)

    ci = float(model.get_ci(torch.tensor([[alpha]], dtype=torch.float32, device=device)).item())
    mach = float(config["mach"])
    c = 1j * ci
    y = y_t[:, 0]
    u_bar = base_velocity(y)
    du_bar = base_velocity_derivative(y)
    i_alpha = 1j * float(alpha)
    v = -p_y[:, 0] / (i_alpha * (u_bar - c))
    u = -(du_bar * v + i_alpha * p[:, 0]) / (i_alpha * (u_bar - c))
    rho = p[:, 0] * (mach**2)

    fields = normalize_full_mode(
        y.detach().cpu().numpy(),
        u.detach().cpu().numpy(),
        v.detach().cpu().numpy(),
        p[:, 0].detach().cpu().numpy(),
        rho.detach().cpu().numpy(),
    )
    return fields, ci, mach


def compute_mode_metrics(classic: dict[str, np.ndarray], pinn: dict[str, np.ndarray], *, n_common: int = 1200, phase_threshold: float = 0.02) -> dict[str, float]:
    y_min = max(float(np.min(classic["y"])), float(np.min(pinn["y"])))
    y_max = min(float(np.max(classic["y"])), float(np.max(pinn["y"])))
    y_common = np.linspace(y_min, y_max, int(n_common), dtype=float)

    def rel(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12))

    p_c = interp_complex(classic["y"], classic["p"], y_common)
    p_p = interp_complex(pinn["y"], pinn["p"], y_common)
    rho_c = interp_complex(classic["y"], classic["rho"], y_common)
    rho_p = interp_complex(pinn["y"], pinn["rho"], y_common)
    u_c = interp_complex(classic["y"], classic["u"], y_common)
    u_p = interp_complex(pinn["y"], pinn["u"], y_common)
    v_c = interp_complex(classic["y"], classic["v"], y_common)
    v_p = interp_complex(pinn["y"], pinn["v"], y_common)

    amp_c = np.abs(p_c)
    amp_p = np.abs(p_p)
    phase_c = np.unwrap(np.angle(p_c))
    phase_p = np.unwrap(np.angle(p_p))
    phase_c -= phase_c[np.argmax(amp_c)]
    phase_p -= phase_p[np.argmax(amp_p)]
    mask = np.maximum(amp_c, amp_p) > float(phase_threshold)
    if np.any(mask):
        phase_diff = np.angle(np.exp(1j * (phase_p[mask] - phase_c[mask])))
        phase_rmse = float(np.sqrt(np.mean(phase_diff**2)))
    else:
        phase_rmse = float("nan")

    return {
        "p_rel": rel(p_c, p_p),
        "rho_rel": rel(rho_c, rho_p),
        "u_rel": rel(u_c, u_p),
        "v_rel": rel(v_c, v_p),
        "amp_rel": float(np.linalg.norm(amp_p - amp_c) / max(np.linalg.norm(amp_c), 1e-12)),
        "phase_rmse": phase_rmse,
    }


def build_reference_mode_cache(mode_alphas: list[float], n_y: int, device: torch.device) -> dict[float, tuple[dict[str, np.ndarray], float]]:
    metrics_df = pd.read_csv(MODE_REFERENCE_METRICS_CSV)
    cache: dict[float, tuple[dict[str, np.ndarray], float]] = {}
    for alpha in mode_alphas:
        row = metrics_df.loc[np.isclose(metrics_df["alpha"].to_numpy(dtype=float), float(alpha), atol=1e-12)]
        if row.empty:
            raise ValueError(f"Alpha={alpha:.3f} absent de la reference modale validee {MODE_REFERENCE_METRICS_CSV}.")
        reference_fields, _, _ = load_pinn_full_mode(MODE_REFERENCE_RUN, alpha=float(alpha), n_y=n_y, device=device)
        cache[float(alpha)] = (reference_fields, float(row.iloc[0]["ci_classic"]))
    return cache


def save_html(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")


def generate_individual_ci_assets(run_dir: Path, analysis_run_dir: Path, num_alpha: int, device: torch.device) -> pd.DataFrame:
    model, config, _ = load_model(run_dir, device)
    alpha_values = np.linspace(float(config["alpha_min"]), float(config["alpha_max"]), int(num_alpha))
    alpha_tensor = torch.tensor(alpha_values, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_pred = model.get_ci(alpha_tensor).cpu().numpy().reshape(-1)
    reference_df = solve_reference_curve(float(config["mach"]), float(config["alpha_min"]), float(config["alpha_max"]), int(num_alpha))
    ci_df = reference_df.copy()
    ci_df["ci_pinn"] = ci_pred
    ci_df["ci_abs_err"] = np.abs(ci_df["ci_pinn"] - ci_df["ci_reference"])
    ci_df.to_csv(analysis_run_dir / "ci_curve_vs_reference.csv", index=False)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("c_i(alpha)", "Erreur absolue sur c_i"))
    fig.add_trace(go.Scatter(x=ci_df["alpha"], y=ci_df["ci_reference"], name="Classique", line=dict(color="black", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ci_df["alpha"], y=ci_df["ci_pinn"], name="PINN", line=dict(color="#0b6e4f", width=3, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ci_df["alpha"], y=ci_df["ci_abs_err"], name="|err|", line=dict(color="#c84c09", width=3)), row=1, col=2)
    fig.update_layout(title=f"Comparaison ci | M={float(config['mach']):.3f}", height=460, width=1100)
    fig.update_xaxes(title_text="alpha", row=1, col=1)
    fig.update_yaxes(title_text="ci", row=1, col=1)
    fig.update_xaxes(title_text="alpha", row=1, col=2)
    fig.update_yaxes(title_text="|ci_pinn-ci_ref|", row=1, col=2)
    save_html(fig, analysis_run_dir / "ci_curve_vs_reference.html")
    return ci_df


def generate_mode_overlay_html(
    run_dir: Path,
    analysis_run_dir: Path,
    label: str,
    mode_alphas: list[float],
    classic_cache: dict[float, tuple[dict[str, np.ndarray], float]],
    n_y: int,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    field_titles = {
        "rho": "rho",
        "u": "u",
        "v": "v",
        "p": "p",
    }
    for alpha in mode_alphas:
        classic, ci_classic = classic_cache[float(alpha)]
        pinn, ci_pinn, mach = load_pinn_full_mode(run_dir, alpha=float(alpha), n_y=n_y, device=device)
        metrics = compute_mode_metrics(classic, pinn)
        rows.append(
            {
                "run_label": label,
                "alpha": float(alpha),
                "mach": float(mach),
                "ci_classic": float(ci_classic),
                "ci_pinn": float(ci_pinn),
                "ci_abs_err": abs(float(ci_pinn) - float(ci_classic)),
                **metrics,
            }
        )
        fig = make_subplots(rows=2, cols=2, subplot_titles=("rho", "u", "v", "p"))
        for idx, field_name in enumerate(("rho", "u", "v", "p"), start=1):
            row = 1 if idx <= 2 else 2
            col = 1 if idx % 2 == 1 else 2
            fig.add_trace(go.Scatter(x=classic["y"], y=np.real(classic[field_name]), name=f"{field_titles[field_name]} ref Re", line=dict(color="#1f77b4", width=2), showlegend=(idx == 1)), row=row, col=col)
            fig.add_trace(go.Scatter(x=classic["y"], y=np.imag(classic[field_name]), name=f"{field_titles[field_name]} ref Im", line=dict(color="#ff7f0e", width=2), showlegend=(idx == 1)), row=row, col=col)
            fig.add_trace(go.Scatter(x=pinn["y"], y=np.real(pinn[field_name]), name=f"{field_titles[field_name]} PINN Re", line=dict(color="#1f77b4", width=2, dash="dash"), showlegend=(idx == 1)), row=row, col=col)
            fig.add_trace(go.Scatter(x=pinn["y"], y=np.imag(pinn[field_name]), name=f"{field_titles[field_name]} PINN Im", line=dict(color="#ff7f0e", width=2, dash="dash"), showlegend=(idx == 1)), row=row, col=col)
        fig.update_layout(
            title=(
                f"{label} | alpha={alpha:.3f}, M={mach:.3f}<br>"
                f"ci classique={ci_classic:.5f} | ci PINN={ci_pinn:.5f} | err abs={abs(ci_pinn - ci_classic):.3e} | p_rel(ref)={metrics['p_rel']:.3e}"
            ),
            height=820,
            width=1100,
        )
        save_html(fig, analysis_run_dir / f"mode_overlay_alpha_{safe_slug(alpha)}.html")
    df = pd.DataFrame(rows)
    df.to_csv(analysis_run_dir / "mode_metrics.csv", index=False)
    return df


def save_global_ci_comparison(ci_frames: dict[str, tuple[str, pd.DataFrame]], output_dir: Path) -> None:
    base_key = next(iter(ci_frames))
    alpha_values = ci_frames[base_key][1]["alpha"].to_numpy(dtype=float)
    ci_ref = ci_frames[base_key][1]["ci_reference"].to_numpy(dtype=float)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Comparaison ci(alpha)", "Erreur absolue"))
    fig.add_trace(go.Scatter(x=alpha_values, y=ci_ref, name="Classique", line=dict(color="black", width=4)), row=1, col=1)

    rows = []
    merged = pd.DataFrame({"alpha": alpha_values, "ci_reference": ci_ref})
    for key, (label, df) in ci_frames.items():
        fig.add_trace(go.Scatter(x=df["alpha"], y=df["ci_pinn"], name=label, line=dict(color=PALETTE[key], width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["alpha"], y=df["ci_abs_err"], name=f"|err| {label}", line=dict(color=PALETTE[key], width=2), showlegend=False), row=2, col=1)
        merged[f"ci_{key}"] = df["ci_pinn"].to_numpy(dtype=float)
        merged[f"ci_abs_err_{key}"] = df["ci_abs_err"].to_numpy(dtype=float)
        rows.append(
            {
                "run_key": key,
                "run_label": label,
                "ci_mae": float(df["ci_abs_err"].mean()),
                "ci_max_abs_err": float(df["ci_abs_err"].max()),
                "ci_rmse": float(np.sqrt(np.mean(df["ci_abs_err"].to_numpy(dtype=float) ** 2))),
            }
        )
    fig.update_layout(title="Comparaison globale des 4 runs | M=0.5", height=850, width=1200)
    fig.update_xaxes(title_text="alpha", row=2, col=1)
    fig.update_yaxes(title_text="ci", row=1, col=1)
    fig.update_yaxes(title_text="|ci_pinn-ci_ref|", row=2, col=1)
    save_html(fig, output_dir / "comparison_all_runs_ci_curve.html")
    merged.to_csv(output_dir / "comparison_all_runs_ci_curve.csv", index=False)
    pd.DataFrame(rows).sort_values("ci_mae").to_csv(output_dir / "comparison_all_runs_ci_summary.csv", index=False)


def save_pressure_mode_overview(
    experiment_root: Path,
    output_dir: Path,
    mode_alphas: list[float],
    classic_cache: dict[float, tuple[dict[str, np.ndarray], float]],
    n_y: int,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for alpha in mode_alphas:
        classic, ci_classic = classic_cache[float(alpha)]
        fig = make_subplots(rows=len(RUNS), cols=2, subplot_titles=sum(([f"{label} amplitude", f"{label} phase"] for _, label, _ in RUNS), []))
        for row_idx, (key, label, relative_run_dir) in enumerate(RUNS, start=1):
            pinn, ci_pinn, _ = load_pinn_full_mode(run_path(experiment_root, relative_run_dir), alpha=float(alpha), n_y=n_y, device=device)
            metrics = compute_mode_metrics(classic, pinn)
            rows.append(
                {
                    "run_key": key,
                    "run_label": label,
                    "alpha": float(alpha),
                    "ci_classic": float(ci_classic),
                    "ci_pinn": float(ci_pinn),
                    "ci_abs_err": abs(float(ci_pinn) - float(ci_classic)),
                    **metrics,
                }
            )

            amp_classic = np.abs(classic["p"])
            amp_pinn = np.abs(pinn["p"])
            phase_classic = np.unwrap(np.angle(classic["p"]))
            phase_pinn = np.unwrap(np.angle(pinn["p"]))
            phase_classic -= phase_classic[np.argmax(amp_classic)]
            phase_pinn -= phase_pinn[np.argmax(amp_pinn)]

            fig.add_trace(go.Scatter(x=classic["y"], y=amp_classic, name="Reference", line=dict(color="black", width=3), showlegend=(row_idx == 1)), row=row_idx, col=1)
            fig.add_trace(go.Scatter(x=pinn["y"], y=amp_pinn, name=label, line=dict(color=PALETTE[key], width=2.5, dash="dash"), showlegend=(row_idx == 1)), row=row_idx, col=1)
            fig.add_trace(go.Scatter(x=classic["y"], y=phase_classic, name="Reference", line=dict(color="black", width=3), showlegend=False), row=row_idx, col=2)
            fig.add_trace(go.Scatter(x=pinn["y"], y=phase_pinn, name=label, line=dict(color=PALETTE[key], width=2.5, dash="dash"), showlegend=False), row=row_idx, col=2)
        fig.update_layout(title=f"Comparaison pression mode par mode | alpha={alpha:.3f}, M=0.5", height=320 * len(RUNS), width=1250)
        save_html(fig, output_dir / f"comparison_mode_pressure_alpha_{safe_slug(alpha)}.html")
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "comparison_all_runs_mode_metrics.csv", index=False)
    return df


def write_readme(output_dir: Path, mode_alphas: list[float]) -> None:
    lines = [
        "Assets HTML + CSV generated for the local M=0.5 comparison package.",
        "CI curves are compared against a classical reference ci(alpha) interpolated from an existing validated local asset.",
        "Mode overlays are compared against a validated modal reference run already audited against the classical solver at the requested alphas.",
        "",
        "Global assets:",
        "- comparison_all_runs_ci_curve.html",
        "- comparison_all_runs_ci_curve.csv",
        "- comparison_all_runs_ci_summary.csv",
        "- comparison_all_runs_mode_metrics.csv",
    ]
    for alpha in mode_alphas:
        lines.append(f"- comparison_mode_pressure_alpha_{safe_slug(alpha)}.html")
    lines.extend(
        [
            "",
            "Per-run assets:",
            "- ci_curve_vs_reference.html",
            "- ci_curve_vs_reference.csv",
            "- mode_overlay_alpha_<alpha>.html",
            "- mode_metrics.csv",
        ]
    )
    (output_dir / "README.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    experiment_root = args.experiment_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else (experiment_root / "analysis_plotly")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    mode_alphas = [float(value) for value in args.mode_alphas]
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Building validated modal reference cache for alphas={mode_alphas}", flush=True)
    classic_cache = build_reference_mode_cache(mode_alphas, int(args.n_y), device)
    print("Validated modal reference cache ready", flush=True)

    ci_frames: dict[str, tuple[str, pd.DataFrame]] = {}
    all_mode_frames = []
    for key, label, relative_run_dir in RUNS:
        run_dir = run_path(experiment_root, relative_run_dir)
        analysis_run_dir = output_dir / key
        analysis_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {label}", flush=True)
        ci_frames[key] = (label, generate_individual_ci_assets(run_dir, analysis_run_dir, int(args.num_alpha), device))
        all_mode_frames.append(
            generate_mode_overlay_html(
                run_dir,
                analysis_run_dir,
                label,
                mode_alphas,
                classic_cache,
                int(args.n_y),
                device,
            )
        )

    save_global_ci_comparison(ci_frames, output_dir)
    save_pressure_mode_overview(experiment_root, output_dir, mode_alphas, classic_cache, int(args.n_y), device)
    if all_mode_frames:
        pd.concat(all_mode_frames, ignore_index=True).to_csv(output_dir / "comparison_all_runs_mode_metrics_fullfields.csv", index=False)
    write_readme(output_dir, mode_alphas)
    print(output_dir)


if __name__ == "__main__":
    main()
