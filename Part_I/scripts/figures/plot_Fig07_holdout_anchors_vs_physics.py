"""
Article figure: Fig. 7
Output: article/assets/Fig07_holdout_anchors_only_vs_physics_N340.png
Purpose: Compare anchors-only and physics-constrained holdout predictions.
Source data: assets/pinn_subsonic/csv/article_supplementary/N340/source_data/Table_Holdout384_pointwise_physics_vs_anchors_only.csv
Scientific computation: Publishes the validated processed asset; no training is run.
Original generator: code/src/scripts/utils/build_holdout_baseline_assets.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/complementary_audits/final_figures/Fig07_holdout_anchors_only_vs_physics_N340.png",
        "Fig07_holdout_anchors_only_vs_physics_N340.png",
    )
