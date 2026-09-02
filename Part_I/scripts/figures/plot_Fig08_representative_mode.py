"""
Article figure: Fig. 8
Output: article/assets/Fig08_representative_mode_M05_a05_N340.png
Purpose: Compare representative direct neural and selected GEP modal fields.
Source data: assets/pinn_subsonic/csv/article/N340/Table_Data_representative_mode_M05_a05_N340.csv
Scientific computation: Publishes the validated processed asset; no GEP is run.
Original generator: code/src/scripts/utils/build_representative_mode_plot.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/complementary_audits/final_figures/Fig08_representative_mode_M05_a05_N340.png",
        "Fig08_representative_mode_M05_a05_N340.png",
    )
