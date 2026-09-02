from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "code"))

from src.scripts.classical.solve_mstab17_subsonic_solver import Mstab17SubsonicSolver  # noqa: E402
from src.models.kh_subsonic_pinn import build_fixed_mach_model_from_config, load_fixed_mach_state_dict_compat  # noqa: E402
from src.physics.kh_subsonic_residual import (  # noqa: E402
    base_velocity,
    base_velocity_derivative,
    dy_dxi,
    reconstruct_pressure_p_y_from_riccati,
    xi_to_y,
)


CONFIGS = [
    {
        "key": "physics_only",
        "slug": "physique_pur",
        "label": "Physique pur",
        "run_dir": Path("model_saved/kh_subsonic_fixed_mach_M05_alpha010_080_riccati_pure_physics_reference"),
        "color": "#c84c09",
    },
    {
        "key": "hybrid_ci4",
        "slug": "hybride_4",
        "label": "Hybride 4",
        "run_dir": Path("model_saved/kh_subsonic_fixed_mach_M05_alpha010_080_ci_sparse_reference/hybrid_ci4_fixed"),
        "color": "#1f77b4",
    },
    {
        "key": "hybrid_ci8",
        "slug": "hybride_8",
        "label": "Hybride 8",
        "run_dir": Path("model_saved/kh_subsonic_fixed_mach_M05_alpha010_080_ci_sparse_reference/hybrid_ci8_fixed"),
        "color": "#0b6e4f",
    },
    {
        "key": "hybrid_ci16",
        "slug": "hybride_16",
        "label": "Hybride 16",
        "run_dir": Path("model_saved/kh_subsonic_fixed_mach_M05_alpha010_080_ci_sparse_reference/hybrid_ci16_fixed"),
        "color": "#8e44ad",
    },
]

MODE_ALPHAS = [0.3, 0.5, 0.7]
FIELD_ORDER = ["p", "rho", "v", "u"]
FIELD_LABELS = {
    "p": "pression",
    "rho": "rho",
    "v": "v_hat",
    "u": "u_hat",
}

LINE_COLORS = {
    "classic_re": (0, 84, 166),
    "classic_im": (0, 150, 136),
    "pinn_re": (200, 76, 9),
    "pinn_im": (142, 68, 173),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genere exactement les 10 figures subsoniques demandees.")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("assets/pinn_subsonic/experiment_M05_alpha010_080_reference_2026-06-21"),
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("assets/pinn_subsonic/experiment_M05_alpha010_080_reference_2026-06-21/analysis_plotly"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Par defaut: <analysis-dir>/exact_10figs_png",
    )
    parser.add_argument("--n-y", type=int, default=1201)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


FONT_TITLE = get_font(34)
FONT_SUBTITLE = get_font(24)
FONT_AXIS = get_font(20)
FONT_LEGEND = get_font(18)
FONT_SMALL = get_font(16)


def make_canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=(20, 20, 20), anchor=None) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def draw_line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color, width: int = 3) -> None:
    if len(points) >= 2:
        draw.line(points, fill=color, width=width)


def draw_plot_box(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> None:
    draw.rectangle([x0, y0, x1, y1], outline=(35, 35, 35), width=2)


def panel_coords(x, y, xlim, ylim, box):
    x0, y0, x1, y1 = box
    px = x0 + (x - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0) if xlim[1] != xlim[0] else x0
    py = y1 - (y - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0) if ylim[1] != ylim[0] else y1
    return float(px), float(py)


def draw_axes(draw: ImageDraw.ImageDraw, box, xlim, ylim, *, title=None, x_label=None, y_label=None, x_ticks=6, y_ticks=5):
    x0, y0, x1, y1 = box
    draw_plot_box(draw, x0, y0, x1, y1)
    grid = (228, 228, 228)
    text = (70, 70, 70)
    for i in range(x_ticks + 1):
        x = x0 + i * (x1 - x0) / x_ticks
        draw.line([(x, y0), (x, y1)], fill=grid, width=1)
        xv = xlim[0] + i * (xlim[1] - xlim[0]) / x_ticks
        draw_text(draw, (int(x), y1 + 8), f"{xv:.2f}", FONT_SMALL, fill=text, anchor="ma")
    for j in range(y_ticks + 1):
        y = y1 - j * (y1 - y0) / y_ticks
        draw.line([(x0, y), (x1, y)], fill=grid, width=1)
        yv = ylim[0] + j * (ylim[1] - ylim[0]) / y_ticks
        draw_text(draw, (x0 - 8, int(y)), f"{yv:.2f}", FONT_SMALL, fill=text, anchor="rm")
    if title:
        draw_text(draw, ((x0 + x1) // 2, y0 - 20), title, FONT_SUBTITLE, fill=text, anchor="ma")
    if x_label:
        draw_text(draw, ((x0 + x1) // 2, y1 + 38), x_label, FONT_AXIS, fill=text, anchor="ma")
    if y_label:
        draw_text(draw, (x0 - 55, (y0 + y1) // 2), y_label, FONT_AXIS, fill=text, anchor="mm")


def draw_legend(draw: ImageDraw.ImageDraw, items: list[tuple[str, tuple[int, int, int]]], x: int, y: int) -> None:
    dy = 28
    for idx, (label, color) in enumerate(items):
        yy = y + idx * dy
        draw.line([(x, yy), (x + 24, yy)], fill=color, width=4)
        draw_text(draw, (x + 34, yy), label, FONT_LEGEND, anchor="lm")


def colormap_magma(t: float) -> tuple[int, int, int]:
    t = min(max(float(t), 0.0), 1.0)
    anchors = [
        (0.0, (4, 0, 25)),
        (0.25, (84, 15, 109)),
        (0.5, (187, 55, 84)),
        (0.75, (249, 142, 8)),
        (1.0, (252, 253, 191)),
    ]
    for (a0, c0), (a1, c1) in zip(anchors[:-1], anchors[1:]):
        if t <= a1:
            r = (t - a0) / (a1 - a0) if a1 > a0 else 0.0
            return tuple(int(c0[k] + r * (c1[k] - c0[k])) for k in range(3))
    return anchors[-1][1]


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
        "u": np.asarray(u / scale, dtype=np.complex128),
        "v": np.asarray(v / scale, dtype=np.complex128),
        "p": np.asarray(p / scale, dtype=np.complex128),
        "rho": np.asarray(rho / scale, dtype=np.complex128),
    }


def interp_complex(y_src: np.ndarray, f_src: np.ndarray, y_dst: np.ndarray) -> np.ndarray:
    return np.interp(y_dst, y_src, np.real(f_src)) + 1j * np.interp(y_dst, y_src, np.imag(f_src))


def relative_l2(ref: np.ndarray, pred: np.ndarray) -> float:
    num = float(np.sqrt(np.mean(np.abs(pred - ref) ** 2)))
    den = float(np.sqrt(np.mean(np.abs(ref) ** 2)))
    return num / max(den, 1e-12)


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


def load_pinn_full_mode(run_dir: Path, *, alpha: float, n_y: int, device: torch.device) -> tuple[dict[str, np.ndarray], float]:
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
    return fields, ci


def build_mode_dataset(experiment_root: Path, device: torch.device, n_y: int):
    classic_modes: dict[float, dict[str, np.ndarray]] = {}
    ci_classic: dict[float, float] = {}
    for alpha in MODE_ALPHAS:
        fields, ci = load_classic_full_mode(alpha, mach=0.5)
        classic_modes[alpha] = fields
        ci_classic[alpha] = ci

    pinn_modes: dict[str, dict[float, dict[str, np.ndarray]]] = {}
    ci_pinn: dict[str, dict[float, float]] = {}
    for cfg in CONFIGS:
        run_dir = experiment_root / cfg["run_dir"]
        pinn_modes[cfg["key"]] = {}
        ci_pinn[cfg["key"]] = {}
        for alpha in MODE_ALPHAS:
            fields, ci = load_pinn_full_mode(run_dir, alpha=alpha, n_y=n_y, device=device)
            pinn_modes[cfg["key"]][alpha] = fields
            ci_pinn[cfg["key"]][alpha] = ci
    return classic_modes, ci_classic, pinn_modes, ci_pinn


def render_ci_curve_error_figure(cfg: dict, analysis_dir: Path, output_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(analysis_dir / cfg["key"] / "ci_curve_vs_reference.csv")
    image, draw = make_canvas(1600, 980)
    draw_text(draw, (800, 38), f"{cfg['label']} : ci classique vs PINN", FONT_TITLE, anchor="ma")

    top_box = (120, 100, 1480, 520)
    heat_box = (120, 650, 1480, 790)
    cbar_box = (1500, 650, 1540, 790)

    xlim = (float(df["alpha"].min()), float(df["alpha"].max()))
    ylim = (0.0, max(float(df["ci_reference"].max()), float(df["ci_pinn"].max())) * 1.05)
    draw_axes(draw, top_box, xlim, ylim, title="Courbe c_i(alpha)", x_label="alpha", y_label="c_i")

    ref_pts = [panel_coords(x, y, xlim, ylim, top_box) for x, y in zip(df["alpha"], df["ci_reference"])]
    pinn_pts = [panel_coords(x, y, xlim, ylim, top_box) for x, y in zip(df["alpha"], df["ci_pinn"])]
    draw_line(draw, ref_pts, (0, 0, 0), width=5)
    draw_line(draw, pinn_pts, hex_to_rgb(cfg["color"]), width=4)
    draw_legend(draw, [("Classique", (0, 0, 0)), ("PINN", hex_to_rgb(cfg["color"]))], 1180, 130)

    err = df["ci_abs_err"].to_numpy(dtype=float)
    emax = max(float(err.max()), 1e-12)
    draw_text(draw, (800, 600), "Heatmap erreur |c_i classique - c_i PINN|", FONT_SUBTITLE, anchor="ma")
    x0, y0, x1, y1 = heat_box
    draw_plot_box(draw, x0, y0, x1, y1)
    for i, value in enumerate(err):
        xa = x0 + i * (x1 - x0) / len(err)
        xb = x0 + (i + 1) * (x1 - x0) / len(err)
        color = colormap_magma(value / emax)
        draw.rectangle([xa, y0, xb, y1], fill=color, outline=color)
    draw_text(draw, ((x0 + x1) // 2, y1 + 32), "alpha", FONT_AXIS, anchor="ma")
    draw_text(draw, (x0, y1 + 8), f"{xlim[0]:.2f}", FONT_SMALL, anchor="la")
    draw_text(draw, (x1, y1 + 8), f"{xlim[1]:.2f}", FONT_SMALL, anchor="ra")

    cx0, cy0, cx1, cy1 = cbar_box
    for j in range(cy1 - cy0):
        t = 1.0 - j / max(cy1 - cy0 - 1, 1)
        color = colormap_magma(t)
        draw.line([(cx0, cy0 + j), (cx1, cy0 + j)], fill=color, width=1)
    draw_plot_box(draw, cx0, cy0, cx1, cy1)
    draw_text(draw, (cx1 + 10, cy0), f"{emax:.3f}", FONT_SMALL, anchor="lm")
    draw_text(draw, (cx1 + 10, cy1), "0.000", FONT_SMALL, anchor="lm")

    draw_text(draw, (120, 860), f"MAE = {df['ci_abs_err'].mean():.4f}", FONT_LEGEND)
    draw_text(draw, (360, 860), f"Max erreur = {df['ci_abs_err'].max():.4f}", FONT_LEGEND)
    image.save(output_dir / f"ci_error_{cfg['slug']}.png")
    return df


def render_modal_reconstruction_figure(
    cfg: dict,
    output_dir: Path,
    classic_modes: dict[float, dict[str, np.ndarray]],
    ci_classic: dict[float, float],
    pinn_modes: dict[str, dict[float, dict[str, np.ndarray]]],
    ci_pinn: dict[str, dict[float, float]],
) -> pd.DataFrame:
    image, draw = make_canvas(2400, 1950)
    draw_text(draw, (1200, 38), f"{cfg['label']} : reconstruction modale classique vs PINN", FONT_TITLE, anchor="ma")
    legend_items = [
        ("Classique Re", LINE_COLORS["classic_re"]),
        ("Classique Im", LINE_COLORS["classic_im"]),
        ("PINN Re", LINE_COLORS["pinn_re"]),
        ("PINN Im", LINE_COLORS["pinn_im"]),
    ]
    draw_legend(draw, legend_items, 1820, 70)

    rows = []
    left_margin = 90
    top_margin = 140
    panel_w = 520
    panel_h = 430
    col_gap = 35
    row_gap = 55

    for row_idx, alpha in enumerate(MODE_ALPHAS):
        classic = classic_modes[alpha]
        pinn = pinn_modes[cfg["key"]][alpha]
        ci_ref = ci_classic[alpha]
        ci_run = ci_pinn[cfg["key"]][alpha]
        draw_text(draw, (left_margin, top_margin + row_idx * (panel_h + row_gap) - 28), f"alpha = {alpha:.1f} | ci classique = {ci_ref:.3f} | ci PINN = {ci_run:.3f}", FONT_SUBTITLE)

        y_min = max(float(np.min(classic["y"])), float(np.min(pinn["y"])))
        y_max = min(float(np.max(classic["y"])), float(np.max(pinn["y"])))
        y_common = np.linspace(y_min, y_max, 1200)

        for col_idx, field_name in enumerate(FIELD_ORDER):
            box = (
                left_margin + col_idx * (panel_w + col_gap),
                top_margin + row_idx * (panel_h + row_gap),
                left_margin + col_idx * (panel_w + col_gap) + panel_w,
                top_margin + row_idx * (panel_h + row_gap) + panel_h - 110,
            )

            ref = interp_complex(classic["y"], classic[field_name], y_common)
            pred = interp_complex(pinn["y"], pinn[field_name], y_common)
            ref_re = np.real(ref)
            ref_im = np.imag(ref)
            pred_re = np.real(pred)
            pred_im = np.imag(pred)
            ymin = min(float(ref_re.min()), float(ref_im.min()), float(pred_re.min()), float(pred_im.min()))
            ymax = max(float(ref_re.max()), float(ref_im.max()), float(pred_re.max()), float(pred_im.max()))
            pad = 0.08 * max(ymax - ymin, 0.2)
            ylim = (ymin - pad, ymax + pad)
            xlim = (float(y_common.min()), float(y_common.max()))
            draw_axes(draw, box, xlim, ylim, title=FIELD_LABELS[field_name] if row_idx == 0 else None, x_label="y", y_label=None, x_ticks=6, y_ticks=5)

            curves = [
                (ref_re, LINE_COLORS["classic_re"]),
                (ref_im, LINE_COLORS["classic_im"]),
                (pred_re, LINE_COLORS["pinn_re"]),
                (pred_im, LINE_COLORS["pinn_im"]),
            ]
            for values, color in curves:
                pts = [panel_coords(x, y, xlim, ylim, box) for x, y in zip(y_common, values)]
                draw_line(draw, pts, color, width=3)

            l2 = relative_l2(ref, pred)
            rows.append(
                {
                    "configuration": cfg["label"],
                    "alpha": alpha,
                    "field": field_name,
                    "l2_rel": l2,
                }
            )
            draw_text(draw, (box[0], box[3] + 8), f"L2 rel = {l2:.3e}", FONT_SMALL)

    image.save(output_dir / f"modal_reconstruction_{cfg['slug']}.png")
    return pd.DataFrame(rows)


def render_recap_l2_ci(config_dfs: dict[str, pd.DataFrame], output_dir: Path) -> None:
    image, draw = make_canvas(1700, 950)
    draw_text(draw, (850, 40), "Recapitulatif erreurs L2 sur ci en fonction de alpha", FONT_TITLE, anchor="ma")
    box = (130, 120, 1580, 820)
    xlim = (min(float(df["alpha"].min()) for df in config_dfs.values()), max(float(df["alpha"].max()) for df in config_dfs.values()))
    ylim = (0.0, max(float(df["ci_abs_err"].max()) for df in config_dfs.values()) * 1.05)
    draw_axes(draw, box, xlim, ylim, title="Erreur L2 sur ci (scalaire : |ci_PINN - ci_classique|)", x_label="alpha", y_label="L2 ci")

    legend = []
    for cfg in CONFIGS:
        df = config_dfs[cfg["key"]]
        color = hex_to_rgb(cfg["color"])
        pts = [panel_coords(x, y, xlim, ylim, box) for x, y in zip(df["alpha"], df["ci_abs_err"])]
        draw_line(draw, pts, color, width=4)
        legend.append((cfg["label"], color))
    draw_legend(draw, legend, 1210, 150)
    image.save(output_dir / "recap_L2_ci.png")


def render_recap_l2_modal(field_error_df: pd.DataFrame, output_dir: Path) -> None:
    image, draw = make_canvas(2200, 1150)
    draw_text(draw, (1100, 40), "Recapitulatif erreurs L2 des reconstructions modales", FONT_TITLE, anchor="ma")

    field_panels = {
        "p": (90, 120, 1030, 500),
        "rho": (1130, 120, 2070, 500),
        "v": (90, 630, 1030, 1010),
        "u": (1130, 630, 2070, 1010),
    }
    alpha_vals = MODE_ALPHAS
    ymax = float(field_error_df["l2_rel"].max()) * 1.08
    legend = []

    for field_name, box in field_panels.items():
        sub = field_error_df[field_error_df["field"] == field_name].copy()
        draw_axes(draw, box, (min(alpha_vals), max(alpha_vals)), (0.0, ymax), title=FIELD_LABELS[field_name], x_label="alpha", y_label="L2 rel", x_ticks=2, y_ticks=5)
        for cfg in CONFIGS:
            part = sub[sub["configuration"] == cfg["label"]].sort_values("alpha")
            color = hex_to_rgb(cfg["color"])
            pts = [panel_coords(float(a), float(v), (min(alpha_vals), max(alpha_vals)), (0.0, ymax), box) for a, v in zip(part["alpha"], part["l2_rel"])]
            draw_line(draw, pts, color, width=4)
            for a, v in zip(part["alpha"], part["l2_rel"]):
                px, py = panel_coords(float(a), float(v), (min(alpha_vals), max(alpha_vals)), (0.0, ymax), box)
                r = 4
                draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline=color)
            if field_name == "p":
                legend.append((cfg["label"], color))
    draw_legend(draw, legend, 1650, 70)
    image.save(output_dir / "recap_L2_modal_fields.png")


def main() -> None:
    args = build_parser().parse_args()
    experiment_root = args.experiment_root.resolve()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else (analysis_dir / "exact_10figs_png")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Verification explicite des structures.
    config_dfs: dict[str, pd.DataFrame] = {}
    for cfg in CONFIGS:
        ci_csv = analysis_dir / cfg["key"] / "ci_curve_vs_reference.csv"
        df = pd.read_csv(ci_csv)
        required = {"alpha", "ci_reference", "ci_pinn", "ci_abs_err"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{ci_csv} is missing columns: {sorted(missing)}")
        config_dfs[cfg["key"]] = df

    classic_modes, ci_classic, pinn_modes, ci_pinn = build_mode_dataset(experiment_root, device, int(args.n_y))

    field_error_frames = []
    for cfg in CONFIGS:
        render_ci_curve_error_figure(cfg, analysis_dir, output_dir)
        field_error_frames.append(
            render_modal_reconstruction_figure(cfg, output_dir, classic_modes, ci_classic, pinn_modes, ci_pinn)
        )

    render_recap_l2_ci(config_dfs, output_dir)
    field_error_df = pd.concat(field_error_frames, ignore_index=True)
    field_error_df.to_csv(output_dir / "recap_L2_modal_fields.csv", index=False)
    render_recap_l2_modal(field_error_df, output_dir)

    # Resume structure verification.
    verify_rows = []
    for cfg in CONFIGS:
        verify_rows.append(
            {
                "configuration": cfg["label"],
                "n_alpha_ci": int(len(config_dfs[cfg["key"]])),
                "mode_alphas": ",".join(f"{alpha:.1f}" for alpha in MODE_ALPHAS),
                "fields": ",".join(FIELD_ORDER),
            }
        )
    pd.DataFrame(verify_rows).to_csv(output_dir / "data_structure_verification.csv", index=False)
    print(output_dir)


if __name__ == "__main__":
    main()
