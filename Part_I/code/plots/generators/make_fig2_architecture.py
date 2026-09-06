from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTDIR = Path("assets/section3_pinn_spectral_atlas")
OUTDIR.mkdir(parents=True, exist_ok=True)

png_path = OUTDIR / "Fig_spectral_modal_architecture_subsonic.png"
pdf_path = OUTDIR / "Fig_spectral_modal_architecture_subsonic.pdf"

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

def box(x, y, w, h, text, fc="#f6f6f6", ec="#333333", fs=11, lw=1.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, wrap=True)
    return patch

def arrow(x1, y1, x2, y2, text=None, fs=10):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle='->', mutation_scale=14,
                          linewidth=1.5, color="#333333")
    ax.add_patch(arr)
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.02, text, ha="center", va="bottom", fontsize=fs)
    return arr

# Title
ax.text(0.5, 0.97, "Subsonic spectral atlas: final training pipeline",
        ha="center", va="top", fontsize=16, fontweight="bold")

# Input / routing
box(0.04, 0.72, 0.14, 0.12, "Input point\n$(M,\\eta,\\alpha)$", fc="#e8f1fb")
box(0.23, 0.72, 0.16, 0.12, "Chart routing\nselect local chart $\\mathcal{C}_k$", fc="#e8f1fb")
box(0.44, 0.72, 0.16, 0.12, "Local anchor set\n$\\mathcal{A}_{340}^{(k)}$", fc="#fff4db")

arrow(0.18, 0.78, 0.23, 0.78)
arrow(0.39, 0.78, 0.44, 0.78)

# Stage 1
box(0.05, 0.46, 0.24, 0.16,
    "Stage 1\nModal initialization\nwith fixed IDW spectral field\n$\\widetilde c_i(M,\\eta)$",
    fc="#eaf7ea", fs=11)
box(0.33, 0.46, 0.17, 0.16,
    "Modal network\n$\\gamma_\\theta(y;M,\\eta,\\alpha)$",
    fc="#eaf7ea", fs=11)
box(0.53, 0.46, 0.18, 0.16,
    "Modal loss\nRiccati equation\n+ boundary terms\n+ stabilization",
    fc="#eaf7ea", fs=11)

arrow(0.52, 0.72, 0.17, 0.62)
arrow(0.29, 0.54, 0.33, 0.54)
arrow(0.50, 0.54, 0.53, 0.54)

# Stage 2
box(0.05, 0.23, 0.24, 0.16,
    "Stage 2\nSpectral prefit\nfit a spectral network to\n$\\mathcal{A}_{340}^{(k)}$",
    fc="#f5eafa", fs=11)
box(0.33, 0.23, 0.17, 0.16,
    "Spectral network\n$c_{i,\\phi}(M,\\eta)$",
    fc="#f5eafa", fs=11)
box(0.53, 0.23, 0.18, 0.16,
    "Anchor loss\n$\\mathcal{L}_{c_i}^{(k)}$",
    fc="#f5eafa", fs=11)

arrow(0.52, 0.72, 0.17, 0.39)
arrow(0.29, 0.31, 0.33, 0.31)
arrow(0.50, 0.31, 0.53, 0.31)

# Stage 3
box(0.76, 0.38, 0.18, 0.24,
    "Stage 3\nJoint training\nspectral network + modal network\noptimized together",
    fc="#fdecec", fs=11)

arrow(0.71, 0.54, 0.76, 0.54)
arrow(0.71, 0.31, 0.76, 0.46)

# Coupling and outputs
box(0.76, 0.15, 0.18, 0.12,
    "Physics loss also acts on\n$c_{i,\\phi}(M,\\eta)$\nthrough the modal equations",
    fc="#fdecec", fs=10)

arrow(0.85, 0.38, 0.85, 0.27)

box(0.78, 0.74, 0.16, 0.12,
    "Outputs\ncontinuous $c_i(M,\\eta)$\nand modal field",
    fc="#e8f1fb", fs=11)

arrow(0.85, 0.62, 0.86, 0.74)

# Notes
ax.text(0.05, 0.08,
        "Important: IDW is used only for modal initialization. "
        "The final atlas uses a trainable spectral network.",
        fontsize=11, ha="left", va="center")
ax.text(0.05, 0.04,
        "The same local anchors are used for the spectral prefit and for the joint stage.",
        fontsize=11, ha="left", va="center")

plt.tight_layout()
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print("WROTE:", png_path)
print("WROTE:", pdf_path)
