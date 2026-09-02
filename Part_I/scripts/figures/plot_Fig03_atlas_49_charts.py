"""
Article figure: Fig. 3
Output: article/assets/Fig03_atlas_49_charts_Mach_alpha.png
Purpose: Show the 49-chart partition of the Mach-alpha domain.
Source data: assets/pinn_subsonic/csv/article/N340/Table_N340_training_plan_for_seams.csv
Scientific computation: Publishes the validated processed asset; no training is run.
Original generator: code/src/scripts/utils/build_data_assets.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/pinn_subsonic/png/article/N340/Fig_atlas_49_charts_Mach_alpha.png",
        "Fig03_atlas_49_charts_Mach_alpha.png",
    )
