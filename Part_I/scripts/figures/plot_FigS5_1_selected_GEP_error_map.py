"""
Supplementary figure: Fig. S5.1
Output: article/assets/FigS5_1_selected_GEP_error_heatmap_N340.png
Purpose: Map selected GEP eigenvalue error against the classical reference.
Source data: results/complementary_audits/curated/map_1000_ci_checks.csv
Scientific computation: Publishes the validated processed asset; no GEP is run.
Original generator: scripts/analysis/complementary_audits/remake_fig8_percent_scale.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/complementary_audits/final_figures/FigS5_1_selected_GEP_error_heatmap_N340.png",
        "FigS5_1_selected_GEP_error_heatmap_N340.png",
    )
