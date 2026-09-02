from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Script placé dans:
# code/plots/scripts/pinn_subsonic/plot_classical_subsonic_mode_four_fields.py
# Donc parents[3] = racine du repo.
ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "code"))

from src.scripts.classical.solve_mstab17_subsonic_solver import Mstab17SubsonicSolver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Génère une figure article avec les modes classiques subsoniques "
            "p, rho, v_hat, u_hat : parties réelle et imaginaire."
        )
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--match-y", type=float, default=1.0)
    parser.add_argument("--ci-min", type=float, default=1e-3)
    parser.add_argument("--ci-max", type=float, default=1.0)
    parser.add_argument("--n-scan", type=int, default=61)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/classical_solveur/modes"),
        help="Dossier de sortie pour l'asset article.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--save-pdf", action="store_true")

    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.015,
        help="Seuil relatif utilisé pour choisir automatiquement la fenêtre visible en y.",
    )
    parser.add_argument(
        "--min-half-width",
        type=float,
        default=8.0,
        help="Demi-largeur minimale affichée en y.",
    )
    return parser


def base_velocity(y: np.ndarray) -> np.ndarray:
    return np.tanh(y)


def base_velocity_derivative(y: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(y) ** 2


def sanitize_float(x: float) -> str:
    return f"{x:.3f}".replace(".", "p")


def normalize_phase_and_scale(
    p: np.ndarray,
    rho: np.ndarray,
    v: np.ndarray,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalisation purement visuelle :
    - phase fixée au maximum de |rho|
    - signe choisi pour avoir une partie réelle dominante positive
    - échelle fixée par rho
    """
    idx = int(np.argmax(np.abs(rho)))

    if np.abs(rho[idx]) > 0.0:
        phase = np.exp(-1j * np.angle(rho[idx]))
        p = p * phase
        rho = rho * phase
        v = v * phase
        u = u * phase

    if np.max(np.real(rho)) < abs(np.min(np.real(rho))):
        p = -p
        rho = -rho
        v = -v
        u = -u

    scale = max(
        np.max(np.abs(np.real(rho))),
        np.max(np.abs(np.imag(rho))),
        1e-12,
    )

    return p / scale, rho / scale, v / scale, u / scale


def compute_visible_xlim(
    y: np.ndarray,
    fields: list[np.ndarray],
    *,
    threshold_ratio: float,
    min_half_width: float,
) -> tuple[float, float]:
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

    y_visible = y[mask]
    half_width = max(float(np.max(np.abs(y_visible))), min_half_width)
    return -half_width, half_width


def reconstruct_classical_fields(
    solver: Mstab17SubsonicSolver,
    *,
    ci: float,
    ln_p_start_right: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruit les champs classiques à partir des trajectoires Riccati.

    Sorties :
    - y
    - p_hat
    - rho_hat
    - v_hat
    - u_hat
    """
    sol_left, sol_right, _ = solver.get_trajectories(
        ci,
        ln_p_start_right=ln_p_start_right,
    )

    if not sol_left.success or not sol_right.success:
        raise RuntimeError(
            f"Échec intégration trajectoires : "
            f"left={sol_left.message}, right={sol_right.message}"
        )

    # Alignement phase/amplitude au centre y=0.
    phi_left_0 = solver._interp_component(0.0, sol_left, 3)
    phi_right_0 = solver._interp_component(0.0, sol_right, 3)
    phase_shift = phi_left_0 - phi_right_0

    ln_left_0 = solver._interp_component(0.0, sol_left, 2)
    ln_right_0 = solver._interp_component(0.0, sol_right, 2)
    ln_amp_shift = ln_left_0 - ln_right_0

    # Branche gauche : y <= 0.
    y_left_all = np.asarray(sol_left.t)
    left_mask = y_left_all <= 0.0
    y_left = y_left_all[left_mask]
    k_left = np.asarray(sol_left.y[0])[left_mask]
    q_left = np.asarray(sol_left.y[1])[left_mask]
    ln_p_left = np.asarray(sol_left.y[2])[left_mask]
    phi_left = np.asarray(sol_left.y[3])[left_mask]

    gamma_left = k_left + 1j * q_left
    p_left = np.exp(ln_p_left) * np.exp(1j * phi_left)

    # Branche droite : y > 0, puis tri croissant.
    y_right_all = np.asarray(sol_right.t)
    right_mask = y_right_all > 0.0
    y_right = y_right_all[right_mask]
    k_right = np.asarray(sol_right.y[0])[right_mask]
    q_right = np.asarray(sol_right.y[1])[right_mask]
    ln_p_right = np.asarray(sol_right.y[2])[right_mask]
    phi_right = np.asarray(sol_right.y[3])[right_mask]

    order = np.argsort(y_right)
    y_right = y_right[order]
    k_right = k_right[order]
    q_right = q_right[order]
    ln_p_right = ln_p_right[order]
    phi_right = phi_right[order]

    gamma_right = k_right + 1j * q_right
    p_right = np.exp(ln_p_right + ln_amp_shift) * np.exp(
        1j * (phi_right + phase_shift)
    )

    # Fusion des deux branches.
    y = np.concatenate([y_left, y_right])
    gamma = np.concatenate([gamma_left, gamma_right])
    p = np.concatenate([p_left, p_right])

    # p_y = gamma p.
    p_y = gamma * p

    alpha = float(solver.alpha)
    mach = float(solver.Mach)
    c = 1j * float(ci)

    U = base_velocity(y)
    Up = base_velocity_derivative(y)

    denom = 1j * alpha * (U - c)

    # Formules linéarisées déjà utilisées dans les scripts de comparaison.
    v_hat = -p_y / denom
    u_hat = -(Up * v_hat + 1j * alpha * p) / denom
    rho_hat = p * mach**2

    p, rho_hat, v_hat, u_hat = normalize_phase_and_scale(
        p,
        rho_hat,
        v_hat,
        u_hat,
    )

    return y, p, rho_hat, v_hat, u_hat


def plot_four_fields(
    *,
    y: np.ndarray,
    p: np.ndarray,
    rho: np.ndarray,
    v_hat: np.ndarray,
    u_hat: np.ndarray,
    alpha: float,
    mach: float,
    ci: float,
    omega_i: float,
    stage1_mismatch: float,
    stage2_mismatch: float,
    output_png: Path,
    output_pdf: Path | None,
    dpi: int,
    threshold_ratio: float,
    min_half_width: float,
) -> None:
    fields = [
        (p, r"Pressure $\hat{p}$"),
        (rho, r"Density $\hat{\rho}$"),
        (v_hat, r"Transverse velocity $\hat{v}$"),
        (u_hat, r"Streamwise velocity $\hat{u}$"),
    ]

    xlim = compute_visible_xlim(
        y,
        [p, rho, v_hat, u_hat],
        threshold_ratio=threshold_ratio,
        min_half_width=min_half_width,
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), sharex=True)
    axes_flat = axes.ravel()

    for ax, (field, title) in zip(axes_flat, fields):
        ax.plot(y, np.real(field), label="Re")
        ax.plot(y, np.imag(field), "--", label="Im")
        ax.axhline(0.0, linewidth=0.8, alpha=0.5)
        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"$y$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        rf"Classical subsonic mode, $\alpha={alpha:.3f}$, $M={mach:.3f}$"
        "\n"
        rf"$c_i={ci:.6f}$, $\omega_i=\alpha c_i={omega_i:.6f}$, "
        rf"Riccati mismatch={stage1_mismatch:.2e}, amplitude={stage2_mismatch:.2e}",
        fontsize=11,
    )

    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")

    if output_pdf is not None:
        fig.savefig(output_pdf, bbox_inches="tight")

    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()

    solver = Mstab17SubsonicSolver(
        alpha=args.alpha,
        Mach=args.mach,
        match_y=args.match_y,
    )

    result = solver.solve(
        ci_min=args.ci_min,
        ci_max=args.ci_max,
        n_scan=args.n_scan,
    )

    if not result.success:
        print(
            "WARNING: le solveur n'a pas marqué success=True. "
            "La figure est quand même générée pour diagnostic."
        )

    y, p, rho, v_hat, u_hat = reconstruct_classical_fields(
        solver,
        ci=result.ci,
        ln_p_start_right=result.ln_p_start_right,
    )

    tag = f"a{sanitize_float(args.alpha)}_M{sanitize_float(args.mach)}"
    output_png = args.output_dir / f"classical_subsonic_mode_four_fields_{tag}.png"
    output_pdf = (
        args.output_dir / f"classical_subsonic_mode_four_fields_{tag}.pdf"
        if args.save_pdf
        else None
    )

    plot_four_fields(
        y=y,
        p=p,
        rho=rho,
        v_hat=v_hat,
        u_hat=u_hat,
        alpha=result.alpha,
        mach=result.Mach,
        ci=result.ci,
        omega_i=result.omega_i,
        stage1_mismatch=result.stage1_mismatch,
        stage2_mismatch=result.stage2_mismatch,
        output_png=output_png,
        output_pdf=output_pdf,
        dpi=args.dpi,
        threshold_ratio=args.threshold_ratio,
        min_half_width=args.min_half_width,
    )

    print(result)
    print(f"PNG écrit dans : {output_png}")
    if output_pdf is not None:
        print(f"PDF écrit dans : {output_pdf}")


if __name__ == "__main__":
    main()
