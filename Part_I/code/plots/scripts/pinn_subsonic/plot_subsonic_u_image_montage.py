#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

out = Path("assets/pinn_subsonic/u_reconstruction_diagnostics_M050")
img_inv = pd.read_csv(out / "image_inventory.csv")

# On garde seulement les PNG/JPG, matplotlib ne lit pas directement les PDF.
cand = img_inv[
    (img_inv["u_candidate"] == True)
    & (img_inv["suffix"].isin([".png", ".jpg", ".jpeg"]))
].copy()

# Priorité aux fichiers qui parlent explicitement de u.
def score_path(s):
    low = s.lower()
    score = 0
    for k in ["u_", "_u", "u-", "u.", "velocity", "profile", "field", "mode", "diagnostic"]:
        if k in low:
            score += 1
    for a in ["a030", "a050", "a055", "a070", "alpha"]:
        if a in low:
            score += 1
    return score

cand["score"] = cand["path"].map(score_path)
cand = cand.sort_values(["method", "score", "path"], ascending=[True, False, True])

cand.to_csv(out / "u_image_candidates_ranked.csv", index=False)

pdf_path = out / "M050_u_image_montage.pdf"

with PdfPages(pdf_path) as pdf:
    if cand.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        ax.text(
            0.02, 0.98,
            "No PNG/JPG u/profile diagnostic images found.\n"
            "Need to rerun diagnostics with profile export.",
            va="top", ha="left", family="monospace"
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    else:
        for _, r in cand.iterrows():
            path = Path(r["path"])
            try:
                img = mpimg.imread(path)
            except Exception as e:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.axis("off")
                ax.text(0.02, 0.98, f"Could not read:\n{path}\n\n{e}", va="top")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                continue

            fig, ax = plt.subplots(figsize=(11, 7))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{r['method']} — {path.name}", fontsize=10)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

print("[OK] wrote", pdf_path)
print("[OK] wrote", out / "u_image_candidates_ranked.csv")
print(cand[["method", "path", "score"]].head(80).to_string(index=False))
