#!/usr/bin/env python3
"""Publish the curated Part II article figure package.

Outputs:
    article/assets/Fig_supersonic_branch_tracking_cr_ci.png
    article/assets/Fig_supersonic_blumen_isolines.png
    article/assets/Fig_blumen_cr_independent_validation.png
    article/assets/Fig_supersonic_representative_mode_M140_a018.png
    article/assets/Fig_supersonic_farfield_validation_M140_a018.png
    article/assets/Fig_supersonic_atlas_geometry.png
    article/assets/Fig_supersonic_anchor_budget_N76.png
    article/assets/Fig_supersonic_budget_branch_recovery.png
    article/assets/Fig_supersonic_T401_raw_vs_corrected.png
    article/assets/Fig_T401_matched_seeded_vs_generic.png
    article/assets/Fig_T401_failure_basin_map.png
    article/assets/Fig_supersonic_modal_overlay_M140_a018_real.png
    article/assets/Fig_supersonic_modal_overlay_M140_a018_imag.png
    article/assets/Fig_supersonic_computational_cost_and_robustness.png
    article/assets/Fig_S2a_blumen_pointwise_delta_ci.png
    article/assets/Fig_S2b_blumen_fixedM_delta_alpha.png
    article/assets/Fig_S4a_spectral_integration_convergence.png
    article/assets/Fig_S4b_spectral_box_convergence.png
    article/assets/Fig_S4c_matching_location_sensitivity.png
    article/assets/Fig_S4d_modal_convergence.png
    article/assets/Fig_independent_physics_residual_audit.png
    article/assets/Fig_multiseed_physics_vs_anchors_ablation.png
    article/assets/Fig_routing_boundary_discontinuity.png
    article/assets/Fig_S7_interchart_mismatch.png
    article/assets/Fig_COST500_threshold_sensitivity.png

Purpose:
    Copy existing validated scientific assets into the curated article package.

Source data:
    assets/article_sources/ and the paths recorded in article/FIGURE_MANIFEST.csv.

Scientific computation:
    This script does not retrain the neural atlas or recompute the classical
    campaign. It only copies previously validated, tracked assets.
"""

from __future__ import annotations

import csv
from pathlib import Path
import shutil


PART_II_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PART_II_ROOT / "article" / "FIGURE_MANIFEST.csv"
ARTICLE_ASSETS = PART_II_ROOT / "article" / "assets"


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ARTICLE_ASSETS.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = PART_II_ROOT / row["canonical_source"]
        destination = ARTICLE_ASSETS / row["filename"]
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
    print(f"published_figures={len(rows)}")


if __name__ == "__main__":
    main()
