from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


LEVELS = [0.05, 0.10, 0.15, 0.175]


def resolve_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise RuntimeError(
        f"Impossible de trouver la colonne pour {label}. "
        f"Colonnes disponibles: {list(df.columns)}"
    )


def load_canonical(asset_root: Path) -> pd.DataFrame:
    path = asset_root / "data" / "validation_pointwise_canonical.csv"
    df = pd.read_csv(path)

    mach_col = resolve_column(df, ["Mach", "M"], "Mach")
    alpha_col = resolve_column(df, ["alpha", "Alpha"], "alpha")
    ci_ref_col = resolve_column(
        df,
        ["ci_ref", "ci_classic", "ci_shooting", "ci_reference"],
        "ci classique",
    )
    ci_gep_col = resolve_column(
        df,
        ["ci_final", "pinn_matched_ci", "ci_gep", "ci_pinn_gep"],
        "ci PINN+GEP",
    )

    out = pd.DataFrame(
        {
            "Mach": pd.to_numeric(df[mach_col], errors="coerce"),
            "alpha": pd.to_numeric(df[alpha_col], errors="coerce"),
            "ci_ref": pd.to_numeric(df[ci_ref_col], errors="coerce"),
            "ci_gep": pd.to_numeric(df[ci_gep_col], errors="coerce"),
        }
    )

    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    # Déduplication si plusieurs lignes portent le même point (Mach, alpha)
    out = (
        out.groupby(["Mach", "alpha"], as_index=False)
        .mean()
        .sort_values(["Mach", "alpha"])
        .reset_index(drop=True)
    )

    # Domaine physique subsonique validé
    denom = np.sqrt(np.clip(1.0 - out["Mach"].to_numpy() ** 2, 1e-14, None))
    eta = out["alpha"].to_numpy() / denom
    mask = (
        (out["Mach"].to_numpy() >= 0.02)
        & (out["Mach"].to_numpy() <= 0.98)
        & (eta >= 0.02)
        & (eta <= 0.98)
    )
    out = out.loc[mask].reset_index(drop=True)

    return out


def interpolate_crossing(alpha: np.ndarray, values: np.ndarray, level: float):
    """
    Pour une coupe à Mach fixé, cherche le point alpha tel que values(alpha)=level.
    On privilégie une vraie traversée, sinon on prend un point très proche.
    """
    x = np.asarray(alpha, dtype=float)
    y = np.asarray(values, dtype=float)

    if len(x) < 2:
        return None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if len(x) < 2:
        return None

    # Cas où le niveau n'est même pas dans l'enveloppe de la coupe
    if level < np.nanmin(y) or level > np.nanmax(y):
        return None

    d = y - level

    # Vraies traversées
    candidate_idx = []
    for i in range(len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        d0, d1 = d[i], d[i + 1]

        if np.isclose(d0, 0.0, atol=1e-10):
            return float(x[i])
        if np.isclose(d1, 0.0, atol=1e-10):
            return float(x[i + 1])

        if d0 * d1 < 0.0:
            candidate_idx.append(i)

    if candidate_idx:
        # On choisit la traversée la plus "propre"
        best_i = min(
            candidate_idx,
            key=lambda i: abs(y[i] - level) + abs(y[i + 1] - level),
        )
        x0, x1 = x[best_i], x[best_i + 1]
        y0, y1 = y[best_i], y[best_i + 1]

        if np.isclose(y1, y0):
            return float(0.5 * (x0 + x1))

        alpha_star = x0 + (level - y0) * (x1 - x0) / (y1 - y0)
        return float(alpha_star)

    # Fallback : si pas de vraie traversée, on prend un point très proche
    k = int(np.argmin(np.abs(d)))
    if abs(d[k]) <= max(0.0025, 0.02 * level):
        return float(x[k])

    return None


def extract_level_curve(df: pd.DataFrame, value_col: str, level: float) -> pd.DataFrame:
    rows = []

    for mach, sub in df.groupby("Mach", sort=True):
        alpha_star = interpolate_crossing(
            sub["alpha"].to_numpy(float),
            sub[value_col].to_numpy(float),
            level,
        )
        if alpha_star is None:
            continue

        rows.append(
            {
                "Mach": float(mach),
                "alpha": float(alpha_star),
                "level": float(level),
                "source": value_col,
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    out = out.sort_values("Mach").reset_index(drop=True)
    return out


def label_curve(ax, curve: pd.DataFrame, level: float, color: str):
    if len(curve) < 6:
        return

    # Placement manuel simple pour éviter le paquet à droite
    frac_map = {
        0.05: 0.80,
        0.10: 0.58,
        0.15: 0.40,
        0.175: 0.28,
    }
    frac = frac_map.get(level, 0.5)
    idx = min(len(curve) - 1, max(0, int(frac * (len(curve) - 1))))

    x = curve.iloc[idx]["Mach"]
    y = curve.iloc[idx]["alpha"]

    ax.text(
        x + 0.006,
        y + 0.004,
        f"{level:g}",
        color=color,
        fontsize=11,
        rotation=-20,
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.2),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()

    asset_root = args.asset_root
    figure_dir = asset_root / "figures"
    data_dir = asset_root / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    df = load_canonical(asset_root)

    fig, ax = plt.subplots(figsize=(12.5, 8.2))

    export_rows = []

    for level in LEVELS:
        curve_ref = extract_level_curve(df, "ci_ref", level)
        curve_gep = extract_level_curve(df, "ci_gep", level)

        if len(curve_ref) > 0:
            ax.plot(
                curve_ref["Mach"],
                curve_ref["alpha"],
                color="black",
                linewidth=1.6,
                solid_capstyle="round",
                zorder=3,
            )
            ax.scatter(
                curve_ref["Mach"],
                curve_ref["alpha"],
                color="black",
                s=12,
                zorder=4,
            )
            label_curve(ax, curve_ref, level, "black")
            export_rows.append(curve_ref.assign(kind="classical"))

        if len(curve_gep) > 0:
            ax.plot(
                curve_gep["Mach"],
                curve_gep["alpha"],
                color="tab:orange",
                linewidth=2.6,
                linestyle="--",
                dash_capstyle="butt",
                zorder=2,
            )
            ax.scatter(
                curve_gep["Mach"],
                curve_gep["alpha"],
                color="tab:orange",
                s=12,
                zorder=2,
            )
            export_rows.append(curve_gep.assign(kind="pinn_gep"))

    # Frontière neutre
    mach_neutral = np.linspace(0.0, 1.0, 1000)
    alpha_neutral = np.sqrt(np.clip(1.0 - mach_neutral**2, 0.0, None))
    ax.plot(
        mach_neutral,
        alpha_neutral,
        color="tab:blue",
        linestyle=":",
        linewidth=2.2,
        zorder=1,
    )

    ax.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=1.8, label="Classical shooting"),
            Line2D([0], [0], color="tab:orange", linewidth=2.8, linestyle="--", label="PINN + GEP"),
            Line2D([0], [0], color="tab:blue", linewidth=2.2, linestyle=":", label="Neutral boundary"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=12,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Mach number $M$", fontsize=14)
    ax.set_ylabel(r"Wavenumber $\alpha$", fontsize=14)
    ax.set_title(r"Constant-$c_i$ isolines: classical shooting versus PINN + GEP", fontsize=18)
    ax.grid(alpha=0.22)

    fig.tight_layout()

    fig.savefig(figure_dir / "Fig_ci_isolines_classical_vs_PINN_GEP.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "Fig_ci_isolines_classical_vs_PINN_GEP.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    if export_rows:
        export = pd.concat(export_rows, ignore_index=True)
        export.to_csv(data_dir / "ci_isoline_points_localfix.csv", index=False)

    print(figure_dir / "Fig_ci_isolines_classical_vs_PINN_GEP.pdf")
    print(figure_dir / "Fig_ci_isolines_classical_vs_PINN_GEP.png")
    print(data_dir / "ci_isoline_points_localfix.csv")


if __name__ == "__main__":
    main()
