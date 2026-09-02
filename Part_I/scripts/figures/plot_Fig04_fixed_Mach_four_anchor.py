"""
Article figure: Fig. 4
Output: article/assets/Fig04_fixed_Mach_four_anchor_vs_physics_only.png
Purpose: Compare four-anchor spectral supervision with the physics-only baseline.
Source data: archived fixed-Mach processed predictions and classical references.
Scientific computation: Publishes the validated processed asset; no training is run.
Original generator: code/src/scripts/utils/plot_ci4_vs_physics_modes.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/complementary_audits/final_figures/Fig04_fixed_Mach_four_anchor_vs_physics_only.png",
        "Fig04_fixed_Mach_four_anchor_vs_physics_only.png",
    )
