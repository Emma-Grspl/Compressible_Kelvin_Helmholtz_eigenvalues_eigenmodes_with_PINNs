"""
Supplementary figure: Fig. S8.1
Output: article/assets/FigS8_1_near_neutral_error_scaling_N340.png
Purpose: Quantify error scaling near the neutral boundary.
Source data: results/complementary_audits/curated/validation_pointwise_canonical.csv
Scientific computation: Publishes the validated processed asset; no GEP is run.
Original generator: code/src/scripts/utils/build_data_assets.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/complementary_audits/final_figures/FigS8_1_near_neutral_error_scaling_N340.png",
        "FigS8_1_near_neutral_error_scaling_N340.png",
    )
