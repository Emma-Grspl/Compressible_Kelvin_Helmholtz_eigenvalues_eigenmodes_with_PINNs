from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Column resolution
# ============================================================

def find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise RuntimeError(
        f"Could not find column for {label}. "
        f"Available columns: {list(df.columns)}"
    )


FIELD_CANDIDATES = {
    "y": ["y", "Y", "coord_y"],
    "p_r": ["p_r", "p_re", "p_real", "Re_p", "p_real_aligned"],
    "p_i": ["p_i", "p_im", "p_imag", "Im_p", "p_imag_aligned"],
    "rho_r": ["rho_r", "rho_re", "rho_real", "Re_rho", "rho_real_aligned"],
    "rho_i": ["rho_i", "rho_im", "rho_imag", "Im_rho", "rho_imag_aligned"],
    "u_r": ["u_r", "u_re", "u_real", "Re_u", "u_real_aligned"],
    "u_i": ["u_i", "u_im", "u_imag", "Im_u", "u_imag_aligned"],
    "v_r": ["v_r", "v_re", "v_real", "Re_v", "v_real_aligned"],
    "v_i": ["v_i", "v_im", "v_imag", "Im_v", "v_imag_aligned"],
    "ci": ["ci", "c_i", "ci_pred", "ci_final", "ci_ref"],
}


def load_profile(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)

    cols = {}
    for key, candidates in FIELD_CANDIDATES.items():
        if key == "ci":
            # optional
            for name in candidates:
                if name in df.columns:
                    cols[key] = name
                    break
            continue
        cols[key] = find_column(df, candidates, key)

    df = df.sort_values(cols["y"]).reset_index(drop=True)

    out = {
        "path": csv_path,
        "df": df,
        "y": df[cols["y"]].to_numpy(float),
        "p": df[cols["p_r"]].to_numpy(float) + 1j * df[cols["p_i"]].to_numpy(float),
        "rho": df[cols["rho_r"]].to_numpy(float) + 1j * df[cols["rho_i"]].to_numpy(float),
        "u": df[cols["u_r"]].to_numpy(float) + 1j * df[cols["u_i"]].to_numpy(float),
        "v": df[cols["v_r"]].to_numpy(float) + 1j * df[cols["v_i"]].to_numpy(float),
        "ci": None,
    }

    if "ci" in cols:
        vals = df[cols["ci"]].dropna().unique()
        if len(vals) > 0:
            out["ci"] = float(vals[0])

    return out


# ============================================================
# Complex interpolation / alignment
# ============================================================

def interp_complex(y_old: np.ndarray, z_old: np.ndarray, y_new: np.ndarray) -> np.ndarray:
    zr = np.interp(y_new, y_old, np.real(z_old))
    zi = np.interp(y_new, y_old, np.imag(z_old))
    return zr + 1j * zi


def l2_complex_scale(reference: np.ndarray, candidate: np.ndarray, y: np.ndarray) -> complex:
    """
    Find A minimizing ||A*candidate - reference||_L2
    """
    num = np.trapz(np.conj(candidate) * reference, y)
    den = np.trapz(np.conj(candidate) * candidate, y)
    if abs(den) < 1e-30:
        return 1.0 + 0.0j
    return num / den


def align_to_reference(ref: dict, other: dict) -> dict:
    y_ref = ref["y"]

    p_i = interp_complex(other["y"], other["p"], y_ref)
    rho_i = interp_complex(other["y"], other["rho"], y_ref)
    u_i = interp_complex(other["y"], other["u"], y_ref)
    v_i = interp_complex(other["y"], other["v"], y_ref)

    A = l2_complex_scale(ref["p"], p_i, y_ref)

    return {
        "y": y_ref,
        "p": A * p_i,
        "rho": A * rho_i,
        "u": A * u_i,
        "v": A * v_i,
        "ci": other["ci"],
        "scale_factor": A,
    }


def normalize_family(classical: dict, direct: dict, gep: dict) -> tuple[dict, dict, dict]:
    amp = np.max(np.abs(classical["rho"]))
    if amp < 1e-30:
        amp = 1.0
    fac = 1.0 / amp

    def apply(d: dict) -> dict:
        return {
            **d,
            "p": fac * d["p"],
            "rho": fac * d["rho"],
            "u": fac * d["u"],
            "v": fac * d["v"],
        }

    return apply(classical), apply(direct), apply(gep)


# ============================================================
# Plotting
# ============================================================

def maybe_fmt_ci(ci: float | None) -> str:
    if ci is None:
        return "N/A"
    return f"{ci:.6f}"


def plot_family(
    classical: dict,
    direct: dict,
    gep: dict,
    alpha: float,
    mach: float,
    output_root: Path,
    basename: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    colors = {
        "Classical shooting": "black",
        "Direct PINN": "#1f77b4",
        "PINN + GEP": "#ff7f0e",
    }

    family = {
        "Classical shooting": classical,
        "Direct PINN": direct,
        "PINN + GEP": gep,
    }

    fields = [
        ("p", r"Pressure $\hat p$"),
        ("rho", r"Density $\hat \rho$"),
        ("v", r"Transverse velocity $\hat v$"),
        ("u", r"Streamwise velocity $\hat u$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), sharex=True)
    axes = axes.ravel()

    y = classical["y"]

    for ax, (field_key, title) in zip(axes, fields):
        for label, data in family.items():
            z = data[field_key]
            ax.plot(y, np.real(z), color=colors[label], lw=2.0, ls="-")
            ax.plot(y, np.imag(z), color=colors[label], lw=2.0, ls="--")

        ax.axhline(0.0, color="0.75", lw=0.8)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r"$y$")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)

    method_handles = [
        Line2D([0], [0], color=colors["Classical shooting"], lw=2.5, ls="-", label="Classical shooting"),
        Line2D([0], [0], color=colors["Direct PINN"], lw=2.5, ls="-", label="Direct PINN"),
        Line2D([0], [0], color=colors["PINN + GEP"], lw=2.5, ls="-", label="PINN + GEP"),
    ]
    comp_handles = [
        Line2D([0], [0], color="0.2", lw=2.0, ls="-", label="Real part"),
        Line2D([0], [0], color="0.2", lw=2.0, ls="--", label="Imaginary part"),
    ]

    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.34, 0.98),
        ncol=3,
        frameon=False,
        title="Method",
        fontsize=11,
        title_fontsize=12,
    )
    fig.legend(
        handles=comp_handles,
        loc="upper center",
        bbox_to_anchor=(0.83, 0.98),
        ncol=2,
        frameon=False,
        title="Component",
        fontsize=11,
        title_fontsize=12,
    )

    title = (
        rf"Subsonic mode comparison at $\alpha={alpha:.3f}$ and $M={mach:.3f}$" "\n"
        rf"$c_i^{{\mathrm{{class}}}}={maybe_fmt_ci(classical['ci'])}$, "
        rf"$c_i^{{\mathrm{{PINN}}}}={maybe_fmt_ci(direct['ci'])}$, "
        rf"$c_i^{{\mathrm{{GEP}}}}={maybe_fmt_ci(gep['ci'])}$"
    )
    fig.suptitle(title, fontsize=16, y=0.995)

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.90])

    pdf_path = output_root / f"{basename}.pdf"
    png_path = output_root / f"{basename}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(pdf_path)
    print(png_path)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical-csv", type=Path, required=True)
    parser.add_argument("--direct-csv", type=Path, required=True)
    parser.add_argument("--gep-csv", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--basename",
        type=str,
        default="Fig_subsonic_mode_comparison_M050_alpha0500_classical_PINN_GEP",
    )
    args = parser.parse_args()

    classical_raw = load_profile(args.classical_csv)
    direct_raw = load_profile(args.direct_csv)
    gep_raw = load_profile(args.gep_csv)

    classical = {
        "y": classical_raw["y"],
        "p": classical_raw["p"],
        "rho": classical_raw["rho"],
        "u": classical_raw["u"],
        "v": classical_raw["v"],
        "ci": classical_raw["ci"],
    }

    direct = align_to_reference(classical, direct_raw)
    gep = align_to_reference(classical, gep_raw)

    classical, direct, gep = normalize_family(classical, direct, gep)

    plot_family(
        classical=classical,
        direct=direct,
        gep=gep,
        alpha=args.alpha,
        mach=args.mach,
        output_root=args.output_root,
        basename=args.basename,
    )


if __name__ == "__main__":
    main()
