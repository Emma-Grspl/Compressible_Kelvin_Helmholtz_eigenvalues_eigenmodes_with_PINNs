from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
font_manager._get_macos_fonts = lambda: []
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
for candidate in (ROOT_DIR, SCRIPTS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from classical_solver.subsonic.mstab17_subsonic_solver import Mstab17SubsonicSolver  # noqa: E402
from classical_solver.subsonic.robust_subsonic_shooting import RobustSubsonicShootingSolver  # noqa: E402
from src.models.kh_subsonic_pinn import build_fixed_mach_model_from_config, load_fixed_mach_state_dict_compat  # noqa: E402
from src.physics.kh_subsonic_residual import (  # noqa: E402
    base_velocity,
    base_velocity_derivative,
    dy_dxi,
    reconstruct_pressure_from_riccati,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construit le package comparatif local pour les 4 runs subsoniques M=0.5 alpha in [0.1,0.8]."
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
        help="Par defaut: <experiment-root>/analysis",
    )
    parser.add_argument("--num-alpha", type=int, default=81)
    parser.add_argument("--mode-alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--n-y", type=int, default=1001)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def setup_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linestyle": "--",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


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
    rows = []
    for alpha in np.linspace(alpha_min, alpha_max, num_alpha):
        result = RobustSubsonicShootingSolver(alpha=float(alpha), Mach=float(mach)).solve()
        rows.append({"alpha": float(alpha), "ci_reference": float(result.ci)})
    return pd.DataFrame(rows)


def plot_history(history: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["loss"], label="loss totale")
    for key in ("loss_pde", "loss_bc", "loss_norm", "loss_phase", "loss_ci_supervision"):
        if key in history.columns:
            axes[0].plot(history["epoch"], history[key], alpha=0.8, label=key)
    axes[0].set_yscale("log")
    axes[0].set_title("Historique des losses")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)

    audited = history.dropna(subset=["audit_ci_mae"]) if "audit_ci_mae" in history.columns else pd.DataFrame()
    if not audited.empty:
        axes[1].plot(audited["epoch"], audited["audit_ci_mae"], label="audit ci MAE")
        if "audit_ci_max_abs" in audited.columns:
            axes[1].plot(audited["epoch"], audited["audit_ci_max_abs"], label="audit ci max abs")
        axes[1].set_yscale("log")
        axes[1].legend()
    axes[1].set_title("Audit spectral")
    axes[1].set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ci_curve(model, config: pd.Series, output_path: Path, *, num_alpha: int, device: torch.device) -> pd.DataFrame:
    alpha_values = np.linspace(float(config["alpha_min"]), float(config["alpha_max"]), int(num_alpha))
    alpha_tensor = torch.tensor(alpha_values, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_pred = model.get_ci(alpha_tensor).cpu().numpy().reshape(-1)

    reference_df = solve_reference_curve(
        mach=float(config["mach"]),
        alpha_min=float(config["alpha_min"]),
        alpha_max=float(config["alpha_max"]),
        num_alpha=int(num_alpha),
    )
    plot_df = reference_df.copy()
    plot_df["ci_pinn"] = ci_pred
    plot_df["ci_abs_err"] = np.abs(plot_df["ci_pinn"] - plot_df["ci_reference"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(plot_df["alpha"], plot_df["ci_reference"], label="Classique", linewidth=2.0, color="black")
    axes[0].plot(plot_df["alpha"], plot_df["ci_pinn"], "--", label="PINN", linewidth=2.0, color="#0b6e4f")
    axes[0].set_title(fr"$c_i(\alpha)$ a M={float(config['mach']):.3f}")
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$c_i$")
    axes[0].legend()

    axes[1].plot(plot_df["alpha"], plot_df["ci_abs_err"], color="#c84c09", linewidth=2.0)
    axes[1].set_title(r"Erreur absolue sur $c_i$")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$|c_i^{PINN}-c_i^{ref}|$")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_df


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


def compute_visible_xlim(y: np.ndarray, fields: list[np.ndarray], *, threshold_ratio: float = 0.02, min_half_width: float = 8.0) -> tuple[float, float]:
    envelope = np.zeros_like(y, dtype=float)
    for field in fields:
        envelope = np.maximum(envelope, np.abs(np.real(field)))
        envelope = np.maximum(envelope, np.abs(np.imag(field)))
    peak = float(np.max(envelope))
    if peak <= 0.0:
        return float(y[0]), float(y[-1])
    mask = envelope >= threshold_ratio * peak
    if not np.any(mask):
        return float(y[0]), float(y[-1])
    y_vis = y[mask]
    half_width = max(float(np.max(np.abs(y_vis))), float(min_half_width))
    return -half_width, half_width


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


def generate_individual_ci_assets(
    *,
    run_dir: Path,
    analysis_run_dir: Path,
    num_alpha: int,
    device: torch.device,
) -> pd.DataFrame:
    model, config, history = load_model(run_dir, device)
    analysis_run_dir.mkdir(parents=True, exist_ok=True)

    plot_history(history, analysis_run_dir / "history_diagnostics.png")
    ci_df = plot_ci_curve(
        model,
        config,
        analysis_run_dir / "ci_curve_vs_reference.png",
        num_alpha=num_alpha,
        device=device,
    )
    ci_df.to_csv(analysis_run_dir / "ci_curve_vs_reference.csv", index=False)
    return ci_df


def generate_mode_overlay_pdf(
    *,
    run_dir: Path,
    analysis_run_dir: Path,
    label: str,
    mode_alphas: list[float],
    n_y: int,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    output_pdf = analysis_run_dir / "classic_vs_pinn_modes_overlay_a030_a050_a070.pdf"
    output_csv = analysis_run_dir / "classic_vs_pinn_modes_overlay_a030_a050_a070.csv"

    with PdfPages(output_pdf) as pdf:
        for alpha in mode_alphas:
            classic, ci_classic = load_classic_full_mode(float(alpha), mach=0.5)
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

            x_limits_classic = compute_visible_xlim(
                classic["y"],
                [classic["rho"], classic["u"], classic["v"], classic["p"]],
            )
            x_limits_pinn = compute_visible_xlim(
                pinn["y"],
                [pinn["rho"], pinn["u"], pinn["v"], pinn["p"]],
            )
            x_limits = (
                min(x_limits_classic[0], x_limits_pinn[0]),
                max(x_limits_classic[1], x_limits_pinn[1]),
            )

            fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), sharex=False)
            fields = [
                ("rho", r"Density perturbation $\hat{\rho}$"),
                ("u", r"Streamwise velocity $\hat{u}$"),
                ("v", r"Normal velocity $\hat{v}$"),
                ("p", r"Pressure perturbation $\hat{p}$"),
            ]
            for ax, (field_name, title) in zip(axes.flat, fields):
                ax.plot(classic["y"], np.real(classic[field_name]), color="tab:blue", linewidth=1.6, label="Classique Re")
                ax.plot(classic["y"], np.imag(classic[field_name]), color="tab:orange", linewidth=1.6, label="Classique Im")
                ax.plot(pinn["y"], np.real(pinn[field_name]), "--", color="tab:blue", linewidth=1.6, label="PINN Re")
                ax.plot(pinn["y"], np.imag(pinn[field_name]), "--", color="tab:orange", linewidth=1.6, label="PINN Im")
                ax.set_title(title)
                ax.set_xlim(*x_limits)
                ax.legend()

            fig.suptitle(
                f"{label} | alpha={alpha:.3f}, M={mach:.3f}\n"
                f"ci classique={ci_classic:.5f} | ci PINN={ci_pinn:.5f} | "
                f"err abs={abs(ci_pinn - ci_classic):.3e} | p_rel={metrics['p_rel']:.3e}"
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=220, bbox_inches="tight")
            plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df


def save_global_ci_comparison(
    *,
    ci_frames: dict[str, tuple[str, pd.DataFrame]],
    output_dir: Path,
) -> None:
    base_key = next(iter(ci_frames))
    alpha_values = ci_frames[base_key][1]["alpha"].to_numpy(dtype=float)
    ci_ref = ci_frames[base_key][1]["ci_reference"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.5), sharex=True, constrained_layout=True)
    ax_curve, ax_err = axes

    ax_curve.plot(alpha_values, ci_ref, color="black", linewidth=2.4, label="Classique")

    palette = {
        "physics_only": "#c84c09",
        "hybrid_ci4": "#1f77b4",
        "hybrid_ci8": "#0b6e4f",
        "hybrid_ci16": "#8e44ad",
    }
    rows = []
    for key, (label, df) in ci_frames.items():
        color = palette[key]
        ci_pinn = df["ci_pinn"].to_numpy(dtype=float)
        ci_abs_err = df["ci_abs_err"].to_numpy(dtype=float)
        ax_curve.plot(alpha_values, ci_pinn, linewidth=2.0, color=color, label=label)
        ax_err.plot(alpha_values, ci_abs_err, linewidth=2.0, color=color, label=label)
        rows.append(
            {
                "run_key": key,
                "run_label": label,
                "ci_mae": float(np.mean(ci_abs_err)),
                "ci_max_abs_err": float(np.max(ci_abs_err)),
                "ci_rmse": float(np.sqrt(np.mean(ci_abs_err**2))),
            }
        )

    ax_curve.set_title(r"Comparaison $c_i(\alpha)$, $M=0.5$")
    ax_curve.set_ylabel(r"$c_i$")
    ax_curve.legend(ncol=2)

    ax_err.set_title(r"Erreur absolue $|c_i^{PINN}-c_i^{ref}|$")
    ax_err.set_xlabel(r"$\alpha$")
    ax_err.set_ylabel("Erreur abs.")
    ax_err.legend(ncol=2)

    fig.savefig(output_dir / "comparison_all_runs_ci_curve.png", bbox_inches="tight")

    summary = pd.DataFrame(rows).sort_values("ci_mae").reset_index(drop=True)
    summary.to_csv(output_dir / "comparison_all_runs_ci_summary.csv", index=False)

    merged = pd.DataFrame({"alpha": alpha_values, "ci_reference": ci_ref})
    for key, (_, df) in ci_frames.items():
        merged[f"ci_{key}"] = df["ci_pinn"].to_numpy(dtype=float)
        merged[f"ci_abs_err_{key}"] = df["ci_abs_err"].to_numpy(dtype=float)
    merged.to_csv(output_dir / "comparison_all_runs_ci_curve.csv", index=False)


def save_mode_overview_figures(
    *,
    experiment_root: Path,
    output_dir: Path,
    mode_alphas: list[float],
    n_y: int,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for alpha in mode_alphas:
        classic, ci_classic = load_classic_full_mode(float(alpha), mach=0.5)
        fig, axes = plt.subplots(len(RUNS), 2, figsize=(12.0, 3.0 * len(RUNS)), sharex=False)
        if len(RUNS) == 1:
            axes = np.asarray([axes])

        x_limits = [compute_visible_xlim(classic["y"], [classic["p"]])]
        pinn_cache: list[tuple[str, str, dict[str, np.ndarray], float]] = []
        for key, label, relative_run_dir in RUNS:
            pinn, ci_pinn, _ = load_pinn_full_mode(run_path(experiment_root, relative_run_dir), alpha=float(alpha), n_y=n_y, device=device)
            pinn_cache.append((key, label, pinn, ci_pinn))
            x_limits.append(compute_visible_xlim(pinn["y"], [pinn["p"]]))

        x_min = min(value[0] for value in x_limits)
        x_max = max(value[1] for value in x_limits)

        for row_idx, (key, label, pinn, ci_pinn) in enumerate(pinn_cache):
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

            ax_amp = axes[row_idx, 0]
            ax_phase = axes[row_idx, 1]

            ax_amp.plot(classic["y"], amp_classic, color="black", linewidth=2.0, label="Classique")
            ax_amp.plot(pinn["y"], amp_pinn, "--", color="#0b6e4f", linewidth=2.0, label="PINN")
            ax_amp.set_xlim(x_min, x_max)
            ax_amp.set_ylabel(label)
            ax_amp.set_title(r"Amplitude $|\hat{p}|$")
            ax_amp.legend()

            ax_phase.plot(classic["y"], phase_classic, color="black", linewidth=2.0, label="Classique")
            ax_phase.plot(pinn["y"], phase_pinn, "--", color="#0b6e4f", linewidth=2.0, label="PINN")
            ax_phase.set_xlim(x_min, x_max)
            ax_phase.set_title(r"Phase $\arg(\hat{p})$")
            ax_phase.legend()

        fig.suptitle(f"Comparaison modale pression | alpha={alpha:.3f}, M=0.5")
        fig.tight_layout()
        fig.savefig(output_dir / f"comparison_mode_pressure_alpha_{safe_slug(alpha)}.png", bbox_inches="tight")
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "comparison_all_runs_mode_metrics.csv", index=False)
    return df


def write_readme(output_dir: Path, mode_alphas: list[float]) -> None:
    lines = [
        "Assets generated for the local comparison package:",
        "- comparison_all_runs_ci_curve.png",
        "- comparison_all_runs_ci_curve.csv",
        "- comparison_all_runs_ci_summary.csv",
        "- comparison_all_runs_mode_metrics.csv",
    ]
    for alpha in mode_alphas:
        lines.append(f"- comparison_mode_pressure_alpha_{safe_slug(alpha)}.png")
    lines.extend(
        [
            "",
            "Each per-run subdirectory contains:",
            "- history_diagnostics.png",
            "- ci_curve_vs_reference.png",
            "- ci_curve_vs_reference.csv",
            "- classic_vs_pinn_modes_overlay_a030_a050_a070.pdf",
            "- classic_vs_pinn_modes_overlay_a030_a050_a070.csv",
        ]
    )
    (output_dir / "README.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    experiment_root = args.experiment_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else (experiment_root / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()
    device = torch.device(args.device)
    mode_alphas = [float(value) for value in args.mode_alphas]

    ci_frames: dict[str, tuple[str, pd.DataFrame]] = {}
    mode_frames: list[pd.DataFrame] = []

    for key, label, relative_run_dir in RUNS:
        run_dir = run_path(experiment_root, relative_run_dir)
        analysis_run_dir = output_dir / key
        ci_df = generate_individual_ci_assets(
            run_dir=run_dir,
            analysis_run_dir=analysis_run_dir,
            num_alpha=int(args.num_alpha),
            device=device,
        )
        ci_frames[key] = (label, ci_df)
        mode_df = generate_mode_overlay_pdf(
            run_dir=run_dir,
            analysis_run_dir=analysis_run_dir,
            label=label,
            mode_alphas=mode_alphas,
            n_y=int(args.n_y),
            device=device,
        )
        mode_frames.append(mode_df)

    save_global_ci_comparison(ci_frames=ci_frames, output_dir=output_dir)
    overview_df = save_mode_overview_figures(
        experiment_root=experiment_root,
        output_dir=output_dir,
        mode_alphas=mode_alphas,
        n_y=int(args.n_y),
        device=device,
    )

    if mode_frames:
        pd.concat(mode_frames, ignore_index=True).to_csv(output_dir / "comparison_all_runs_mode_metrics_fullfields.csv", index=False)
    overview_df.to_csv(output_dir / "comparison_all_runs_mode_metrics_pressure.csv", index=False)
    write_readme(output_dir, mode_alphas)

    print(output_dir)


if __name__ == "__main__":
    main()
