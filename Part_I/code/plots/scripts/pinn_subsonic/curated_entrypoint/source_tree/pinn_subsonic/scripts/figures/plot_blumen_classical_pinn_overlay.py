from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

ROOT = Path.cwd()

RELEASE_CANDIDATES = [
    ROOT / "assets/pinn_subsonic/csv/curated/pinn_subsonic/data/scientific_outputs/release_v1/data/Table_validation_pointwise_canonical.csv",
    ROOT / "assets/pinn_subsonic/release_v1/data/validation_pointwise_canonical.csv",
    ROOT / "assets/pinn_subsonic/local_atlas_v1/publication_assets_scientific_v2/data/validation_pointwise_canonical.csv",
]

BLUMEN_DIR_CANDIDATES = [
    ROOT / "KH_RT_Blumen/subsonic",
    ROOT / "pinn_subsonic/data/blumen/subsonic",
]

OUTDIR = ROOT / "assets/pinn_subsonic/article/ci"
OUTDIR.mkdir(parents=True, exist_ok=True)

LEVELS = [0.00, 0.05, 0.10, 0.15, 0.175]


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Aucun chemin trouvé parmi :\n" + "\n".join(str(p) for p in paths)
    )


def read_blumen_curve(path: Path):
    """
    On suppose que les CSV digitalisés sont en colonnes:
    alpha, Mach
    sans header ou avec header léger.
    """
    raw = pd.read_csv(path, header=None)
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(how="all")

    # on garde les deux premières colonnes numériques non vides
    numeric_cols = [c for c in raw.columns if raw[c].notna().sum() >= 2]
    if len(numeric_cols) < 2:
        return None

    sub = raw[numeric_cols[:2]].dropna().copy()
    sub.columns = ["alpha", "Mach"]

    sub = sub[
        np.isfinite(sub["alpha"])
        & np.isfinite(sub["Mach"])
    ].copy()

    if len(sub) < 2:
        return None

    return sub.sort_values(["Mach", "alpha"]).reset_index(drop=True)


def load_blumen_curves(blumen_dir: Path):
    curves = []
    for path in sorted(blumen_dir.glob("*.csv")):
        m = re.match(r"^([0-9]+(?:\.[0-9]+)?)$", path.stem)
        if not m:
            continue
        level = float(m.group(1))
        curve = read_blumen_curve(path)
        if curve is None:
            continue
        curves.append((level, curve, path.name))
    if not curves:
        raise RuntimeError(f"Aucune courbe Blumen exploitable dans {blumen_dir}")
    return curves


def build_field_interpolator(df: pd.DataFrame, value_col: str):
    pts = df[["Mach", "alpha"]].to_numpy()
    vals = df[value_col].to_numpy()

    lin = LinearNDInterpolator(pts, vals, fill_value=np.nan)
    nn = NearestNDInterpolator(pts, vals)

    return lin, nn


def evaluate_full_grid(df: pd.DataFrame, value_col: str, nx=500, ny=500):
    mach_min = float(df["Mach"].min())
    mach_max = float(df["Mach"].max())
    alpha_min = float(df["alpha"].min())
    alpha_max = float(df["alpha"].max())

    mx = np.linspace(mach_min, mach_max, nx)
    ay = np.linspace(alpha_min, alpha_max, ny)
    MX, AY = np.meshgrid(mx, ay)

    lin, nn = build_field_interpolator(df, value_col)
    Z = lin(MX, AY)
    Z = np.asarray(Z, dtype=float)

    mask = ~np.isfinite(Z)
    if np.any(mask):
        Z[mask] = nn(MX[mask], AY[mask])

    return MX, AY, Z


def plot_blumen_only(ax, curves):
    ax.set_title("(a) Blumen digitalisé")
    for level, curve, _ in curves:
        x = curve["Mach"].to_numpy()
        y = curve["alpha"].to_numpy()

        ax.plot(x, y, color="orange", lw=1.4, alpha=0.9)
        ax.scatter(
            x, y,
            s=16,
            color="orange",
            edgecolors="none",
            alpha=0.9,
        )

        mid = len(curve) // 2
        ax.text(
            x[mid], y[mid],
            f"{level:g}",
            color="darkorange",
            fontsize=9,
            ha="left",
            va="bottom",
        )

    ax.set_xlabel("Mach")
    ax.set_ylabel(r"$\alpha$")
    ax.grid(alpha=0.2)


def overlay_family(ax, curves, MX, AY, Z, family_label, line_color, line_style, title):
    ax.set_title(title)

    # Blumen en fond: points visibles + ligne
    for level, curve, _ in curves:
        x = curve["Mach"].to_numpy()
        y = curve["alpha"].to_numpy()
        ax.plot(x, y, color="orange", lw=1.0, alpha=0.85, zorder=3)
        ax.scatter(
            x, y,
            s=16,
            color="orange",
            edgecolors="none",
            alpha=0.95,
            zorder=4,
        )

    # Isolignes reconstruites
    cs = ax.contour(
        MX, AY, Z,
        levels=LEVELS,
        colors=line_color,
        linewidths=1.6,
        linestyles=line_style,
        zorder=2,
    )
    ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=8, colors=line_color)

    ax.set_xlabel("Mach")
    ax.set_ylabel(r"$\alpha$")
    ax.grid(alpha=0.2)

    # petite légende manuelle
    ax.plot([], [], color="orange", lw=2, label="Blumen digitalisé")
    ax.plot([], [], color=line_color, lw=2, linestyle=line_style, label=family_label)
    ax.legend(loc="best", frameon=True)


def main():
    release_csv = first_existing(RELEASE_CANDIDATES)
    blumen_dir = first_existing(BLUMEN_DIR_CANDIDATES)

    df = pd.read_csv(release_csv)

    needed = ["Mach", "alpha", "ci_ref", "ci_final"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Colonnes manquantes dans {release_csv}: {missing}"
        )

    df = df[
        np.isfinite(df["Mach"])
        & np.isfinite(df["alpha"])
        & np.isfinite(df["ci_ref"])
        & np.isfinite(df["ci_final"])
    ].copy()

    curves = load_blumen_curves(blumen_dir)

    MX_ref, AY_ref, Z_ref = evaluate_full_grid(df, "ci_ref")
    MX_fin, AY_fin, Z_fin = evaluate_full_grid(df, "ci_final")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), constrained_layout=True)

    plot_blumen_only(axes[0], curves)

    overlay_family(
        axes[1],
        curves,
        MX_ref, AY_ref, Z_ref,
        family_label="Solveur classique",
        line_color="black",
        line_style="-",
        title="(b) Blumen + classique",
    )

    overlay_family(
        axes[2],
        curves,
        MX_fin, AY_fin, Z_fin,
        family_label="PINN final",
        line_color="green",
        line_style="--",
        title="(c) Blumen + PINN final",
    )

    fig.suptitle(
        r"Comparaison des isolignes de croissance $c_i$",
        fontsize=18
    )

    pdf_path = OUTDIR / "SuppFig_Blumen_growth_rate_comparison.pdf"
    png_path = OUTDIR / "SuppFig_Blumen_growth_rate_comparison.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("release_csv:", release_csv)
    print("blumen_dir :", blumen_dir)
    print("wrote      :", pdf_path)
    print("wrote      :", png_path)


if __name__ == "__main__":
    main()
