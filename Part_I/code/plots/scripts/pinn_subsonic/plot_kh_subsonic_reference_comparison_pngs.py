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

from src.models.kh_subsonic_pinn import build_fixed_mach_model_from_config, load_fixed_mach_state_dict_compat  # noqa: E402
from src.physics.kh_subsonic_residual import (  # noqa: E402
    base_velocity,
    base_velocity_derivative,
    dy_dxi,
    reconstruct_pressure_p_y_from_riccati,
    xi_to_y,
)


RUNS = [
    ("physics_only", "Physique pur", "#c84c09"),
    ("hybrid_ci4", "Hybride 4 points", "#1f77b4"),
    ("hybrid_ci8", "Hybride 8 points", "#0b6e4f"),
    ("hybrid_ci16", "Hybride 16 points", "#8e44ad"),
]

REFERENCE_RUN = ROOT_DIR / "archive/csv/assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current"
REFERENCE_MODEL = ROOT_DIR / "models_saved/production/fixed_mach/reference/model_best.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rend des PNG lisibles a partir des CSV/weights du package comparatif subsonique.")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("assets/pinn_subsonic/experiment_M05_alpha010_080_reference_2026-06-21/analysis_plotly"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Par defaut: <analysis-dir>/static_png",
    )
    parser.add_argument("--mode-alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
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
    draw.rectangle([x0, y0, x1, y1], outline=(40, 40, 40), width=2)


def to_panel_coords(x, y, xlim, ylim, box):
    x0, y0, x1, y1 = box
    if xlim[1] == xlim[0]:
        px = x0
    else:
        px = x0 + (x - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
    if ylim[1] == ylim[0]:
        py = y1
    else:
        py = y1 - (y - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)
    return float(px), float(py)


def draw_axes_and_grid(draw: ImageDraw.ImageDraw, box, xlim, ylim, x_ticks=6, y_ticks=6, x_label=None, y_label=None, title=None):
    x0, y0, x1, y1 = box
    draw_plot_box(draw, x0, y0, x1, y1)
    grid = (225, 225, 225)
    text = (60, 60, 60)
    for i in range(x_ticks + 1):
        x = x0 + i * (x1 - x0) / x_ticks
        draw.line([(x, y0), (x, y1)], fill=grid, width=1)
        xv = xlim[0] + i * (xlim[1] - xlim[0]) / x_ticks
        draw_text(draw, (int(x), y1 + 8), f"{xv:.2f}", FONT_SMALL, fill=text, anchor="ma")
    for j in range(y_ticks + 1):
        y = y1 - j * (y1 - y0) / y_ticks
        draw.line([(x0, y), (x1, y)], fill=grid, width=1)
        yv = ylim[0] + j * (ylim[1] - ylim[0]) / y_ticks
        draw_text(draw, (x0 - 10, int(y)), f"{yv:.2f}", FONT_SMALL, fill=text, anchor="rm")
    if x_label:
        draw_text(draw, ((x0 + x1) // 2, y1 + 40), x_label, FONT_AXIS, fill=text, anchor="ma")
    if y_label:
        draw_text(draw, (x0 - 65, (y0 + y1) // 2), y_label, FONT_AXIS, fill=text, anchor="mm")
    if title:
        draw_text(draw, ((x0 + x1) // 2, y0 - 24), title, FONT_SUBTITLE, fill=text, anchor="ma")


def draw_legend(draw: ImageDraw.ImageDraw, items: list[tuple[str, tuple[int, int, int]]], x: int, y: int) -> None:
    dy = 28
    for idx, (label, color) in enumerate(items):
        yy = y + idx * dy
        draw.line([(x, yy), (x + 26, yy)], fill=color, width=4)
        draw_text(draw, (x + 36, yy), label, FONT_LEGEND, anchor="lm")


def render_global_ci_plot(analysis_dir: Path, output_dir: Path) -> None:
    df = pd.read_csv(analysis_dir / "comparison_all_runs_ci_curve.csv")
    image, draw = make_canvas(1800, 1200)
    draw_text(draw, (900, 40), "Comparaison lisible de c_i(alpha) a M = 0.5", FONT_TITLE, anchor="ma")

    top_box = (120, 110, 1680, 620)
    bottom_box = (120, 720, 1680, 1090)

    xlim = (float(df["alpha"].min()), float(df["alpha"].max()))
    y_top_vals = [df["ci_reference"].to_numpy(dtype=float)]
    y_err_vals = []
    for key, _, _ in RUNS:
        y_top_vals.append(df[f"ci_{key}"].to_numpy(dtype=float))
        y_err_vals.append(df[f"ci_abs_err_{key}"].to_numpy(dtype=float))
    ylim_top = (0.0, max(float(np.max(v)) for v in y_top_vals) * 1.05)
    ylim_err = (0.0, max(float(np.max(v)) for v in y_err_vals) * 1.05)

    draw_axes_and_grid(draw, top_box, xlim, ylim_top, x_label="alpha", y_label="c_i", title="Courbes c_i")
    draw_axes_and_grid(draw, bottom_box, xlim, ylim_err, x_label="alpha", y_label="erreur abs.", title="Erreur absolue |c_i PINN - c_i ref|")

    ref_pts = [to_panel_coords(x, y, xlim, ylim_top, top_box) for x, y in zip(df["alpha"], df["ci_reference"])]
    draw_line(draw, ref_pts, (0, 0, 0), width=5)
    legend = [("Classique", (0, 0, 0))]
    for key, label, color in RUNS:
        rgb = hex_to_rgb(color)
        pts = [to_panel_coords(x, y, xlim, ylim_top, top_box) for x, y in zip(df["alpha"], df[f"ci_{key}"])]
        draw_line(draw, pts, rgb, width=4)
        err_pts = [to_panel_coords(x, y, xlim, ylim_err, bottom_box) for x, y in zip(df["alpha"], df[f"ci_abs_err_{key}"])]
        draw_line(draw, err_pts, rgb, width=4)
        legend.append((label, rgb))
    draw_legend(draw, legend, 1260, 140)
    image.save(output_dir / "comparison_all_runs_ci_curve_readable.png")


def render_per_run_ci_plots(analysis_dir: Path, output_dir: Path) -> None:
    for key, label, color in RUNS:
        df = pd.read_csv(analysis_dir / key / "ci_curve_vs_reference.csv")
        image, draw = make_canvas(1600, 980)
        draw_text(draw, (800, 38), f"{label} : c_i classique vs PINN", FONT_TITLE, anchor="ma")
        top_box = (120, 100, 1480, 520)
        bottom_box = (120, 610, 1480, 900)
        xlim = (float(df["alpha"].min()), float(df["alpha"].max()))
        ylim_top = (0.0, max(float(df["ci_reference"].max()), float(df["ci_pinn"].max())) * 1.05)
        ylim_err = (0.0, float(df["ci_abs_err"].max()) * 1.05)
        draw_axes_and_grid(draw, top_box, xlim, ylim_top, x_label="alpha", y_label="c_i", title="Courbe c_i")
        draw_axes_and_grid(draw, bottom_box, xlim, ylim_err, x_label="alpha", y_label="erreur abs.", title="Erreur absolue")
        draw_line(draw, [to_panel_coords(x, y, xlim, ylim_top, top_box) for x, y in zip(df["alpha"], df["ci_reference"])], (0, 0, 0), width=5)
        draw_line(draw, [to_panel_coords(x, y, xlim, ylim_top, top_box) for x, y in zip(df["alpha"], df["ci_pinn"])], hex_to_rgb(color), width=4)
        draw_line(draw, [to_panel_coords(x, y, xlim, ylim_err, bottom_box) for x, y in zip(df["alpha"], df["ci_abs_err"])], hex_to_rgb(color), width=4)
        draw_legend(draw, [("Classique", (0, 0, 0)), (label, hex_to_rgb(color))], 1120, 130)
        draw_text(draw, (1240, 220), f"MAE = {df['ci_abs_err'].mean():.3f}", FONT_LEGEND)
        draw_text(draw, (1240, 252), f"Max = {df['ci_abs_err'].max():.3f}", FONT_LEGEND)
        image.save(output_dir / f"{key}_ci_curve_readable.png")


def render_mode_metric_bars(analysis_dir: Path, output_dir: Path) -> None:
    df = pd.read_csv(analysis_dir / "comparison_all_runs_mode_metrics_fullfields.csv")
    metrics = [("ci_abs_err", "Erreur sur c_i"), ("p_rel", "Erreur relative pression"), ("phase_rmse", "Erreur de phase RMS")]
    image, draw = make_canvas(1800, 1300)
    draw_text(draw, (900, 40), "Synthese modale lisible aux alphas 0.3, 0.5, 0.7", FONT_TITLE, anchor="ma")
    panel_w, panel_h = 500, 300
    positions = [(90, 120), (650, 120), (1210, 120)]
    alpha_values = sorted(df["alpha"].unique())
    run_keys = [key for key, _, _ in RUNS]
    run_labels = {key: label for key, label, _ in RUNS}
    run_colors = {key: hex_to_rgb(color) for key, _, color in RUNS}

    for (metric, title), (px, py) in zip(metrics, positions):
        panel = (px, py, px + panel_w, py + panel_h)
        sub = df.copy()
        ylim = (0.0, float(sub[metric].max()) * 1.15)
        draw_axes_and_grid(draw, panel, (-0.5, len(alpha_values) - 0.5), ylim, x_label="alpha", y_label=metric, title=title, x_ticks=max(1, len(alpha_values) - 1), y_ticks=5)
        group_width = 0.72
        bar_w = group_width / len(run_keys)
        for i, alpha in enumerate(alpha_values):
            for j, key in enumerate(run_keys):
                label = run_labels[key]
                val = float(sub[(sub["alpha"] == alpha) & (sub["run_label"] == label)][metric].iloc[0])
                x_left = i - group_width / 2 + j * bar_w
                x_right = x_left + bar_w * 0.9
                p0 = to_panel_coords(x_left, 0.0, (-0.5, len(alpha_values) - 0.5), ylim, panel)
                p1 = to_panel_coords(x_right, val, (-0.5, len(alpha_values) - 0.5), ylim, panel)
                x0, y0 = p0
                x1, y1 = p1
                draw.rectangle([x0, y1, x1, y0], fill=run_colors[key], outline=run_colors[key])
            draw_text(draw, (int(px + (i + 0.5) * panel_w / len(alpha_values)), py + panel_h + 12), f"{alpha:.1f}", FONT_SMALL, anchor="ma")

    legend_items = [(label, hex_to_rgb(color)) for _, label, color in RUNS]
    draw_legend(draw, legend_items, 120, 500)

    table_top = 600
    draw_text(draw, (900, table_top), "Valeurs numeriques clefs", FONT_SUBTITLE, anchor="ma")
    headers = ["Run", "alpha", "ci abs err", "p_rel", "phase_rmse"]
    xs = [120, 460, 650, 900, 1160]
    for x, h in zip(xs, headers):
        draw_text(draw, (x, table_top + 40), h, FONT_LEGEND)
    y = table_top + 80
    for _, row in df.iterrows():
        draw_text(draw, (xs[0], y), str(row["run_label"]), FONT_SMALL)
        draw_text(draw, (xs[1], y), f"{row['alpha']:.1f}", FONT_SMALL)
        draw_text(draw, (xs[2], y), f"{row['ci_abs_err']:.3f}", FONT_SMALL)
        draw_text(draw, (xs[3], y), f"{row['p_rel']:.3f}", FONT_SMALL)
        draw_text(draw, (xs[4], y), f"{row['phase_rmse']:.3f}", FONT_SMALL)
        y += 30
    image.save(output_dir / "comparison_mode_metrics_readable.png")


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
    return {"y": np.asarray(y, dtype=float), "u": u / scale, "v": v / scale, "p": p / scale, "rho": rho / scale}


def load_pinn_full_mode(
    run_dir: Path,
    *,
    alpha: float,
    n_y: int,
    device: torch.device,
    model_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    config = pd.read_csv(run_dir / "config.csv").iloc[0]
    model = build_fixed_mach_model_from_config(config)
    state_dict = torch.load(model_path or run_dir / "model_best.pt", map_location=device)
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


def render_pressure_mode_overlays(analysis_dir: Path, output_dir: Path, mode_alphas: list[float], n_y: int, device: torch.device) -> None:
    run_dirs = {key: analysis_dir.parent / {
        "physics_only": "model_saved/kh_subsonic_fixed_mach_M05_alpha010_080_riccati_pure_physics_reference",
        "hybrid_ci4": "model_saved/kh_subsonic_fixed_mach_M05_alpha010_080_ci_sparse_reference/hybrid_ci4_fixed",
        "hybrid_ci8": "model_saved/kh_subsonic_fixed_mach_M05_alpha010_080_ci_sparse_reference/hybrid_ci8_fixed",
        "hybrid_ci16": "model_saved/kh_subsonic_fixed_mach_M05_alpha010_080_ci_sparse_reference/hybrid_ci16_fixed",
    }[key] for key, _, _ in RUNS}
    reference_metrics = pd.read_csv(REFERENCE_RUN / "modes/classic_vs_pinn_modes_overlay.csv")
    for alpha in mode_alphas:
        ref_fields, ref_ci_model = load_pinn_full_mode(
            REFERENCE_RUN,
            alpha=float(alpha),
            n_y=n_y,
            device=device,
            model_path=REFERENCE_MODEL,
        )
        ci_classic = float(reference_metrics.loc[np.isclose(reference_metrics["alpha"], float(alpha)), "ci_classic"].iloc[0])
        image, draw = make_canvas(1900, 1650)
        draw_text(draw, (900, 36), f"Comparaison modale pression lisible | alpha={alpha:.1f}, M=0.5", FONT_TITLE, anchor="ma")
        draw_text(draw, (900, 78), "Reference noire = run modal valide contre le classique", FONT_LEGEND, anchor="ma")
        row_h = 360
        for idx, (key, label, color) in enumerate(RUNS):
            fields, ci_pinn = load_pinn_full_mode(run_dirs[key], alpha=float(alpha), n_y=n_y, device=device)
            amp_ref = np.abs(ref_fields["p"])
            amp_run = np.abs(fields["p"])
            phase_ref = np.unwrap(np.angle(ref_fields["p"]))
            phase_run = np.unwrap(np.angle(fields["p"]))
            phase_ref -= phase_ref[np.argmax(amp_ref)]
            phase_run -= phase_run[np.argmax(amp_run)]
            yvals = ref_fields["y"]
            top = 130 + idx * row_h
            amp_box = (150, top, 900, top + 210)
            phase_box = (1030, top, 1780, top + 210)
            xlim = (float(yvals.min()), float(yvals.max()))
            xmask = amp_ref > 0.02 * float(amp_ref.max())
            if np.any(xmask):
                xmin = float(yvals[xmask].min())
                xmax = float(yvals[xmask].max())
                xlim = (xmin, xmax)
            ylim_amp = (0.0, max(float(amp_ref.max()), float(amp_run.max())) * 1.05)
            phase_min = min(float(phase_ref.min()), float(phase_run.min()))
            phase_max = max(float(phase_ref.max()), float(phase_run.max()))
            pad = 0.1 * max(phase_max - phase_min, 0.2)
            ylim_phase = (phase_min - pad, phase_max + pad)
            draw_text(draw, (150, top - 36), f"{label} | ci ref={ci_classic:.3f} | ci PINN={ci_pinn:.3f}", FONT_SUBTITLE)
            draw_axes_and_grid(
                draw,
                amp_box,
                xlim,
                ylim_amp,
                x_label="y",
                y_label=None,
                title="Amplitude pression" if idx == 0 else None,
            )
            draw_axes_and_grid(
                draw,
                phase_box,
                xlim,
                ylim_phase,
                x_label="y",
                y_label=None,
                title="Phase pression" if idx == 0 else None,
            )
            ref_amp_pts = [to_panel_coords(x, y, xlim, ylim_amp, amp_box) for x, y in zip(ref_fields["y"], amp_ref)]
            run_amp_pts = [to_panel_coords(x, y, xlim, ylim_amp, amp_box) for x, y in zip(fields["y"], amp_run)]
            ref_phase_pts = [to_panel_coords(x, y, xlim, ylim_phase, phase_box) for x, y in zip(ref_fields["y"], phase_ref)]
            run_phase_pts = [to_panel_coords(x, y, xlim, ylim_phase, phase_box) for x, y in zip(fields["y"], phase_run)]
            draw_line(draw, ref_amp_pts, (0, 0, 0), width=4)
            draw_line(draw, run_amp_pts, hex_to_rgb(color), width=4)
            draw_line(draw, ref_phase_pts, (0, 0, 0), width=4)
            draw_line(draw, run_phase_pts, hex_to_rgb(color), width=4)
        draw_legend(draw, [("Reference modale validee", (0, 0, 0))] + [(label, hex_to_rgb(color)) for _, label, color in RUNS], 1360, 85)
        image.save(output_dir / f"comparison_mode_pressure_alpha_{str(alpha).replace('.', 'p')}_readable.png")


def main() -> None:
    args = build_parser().parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else (analysis_dir / "static_png")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    render_global_ci_plot(analysis_dir, output_dir)
    render_per_run_ci_plots(analysis_dir, output_dir)
    render_mode_metric_bars(analysis_dir, output_dir)
    render_pressure_mode_overlays(analysis_dir, output_dir, [float(v) for v in args.mode_alphas], int(args.n_y), device)
    print(output_dir)


if __name__ == "__main__":
    main()
