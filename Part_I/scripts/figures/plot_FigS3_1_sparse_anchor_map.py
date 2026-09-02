"""
Supplementary figure: Fig. S3.1
Output: article/assets/FigS3_1_sparse_ci_anchor_map_Mach_alpha_N340.png
Purpose: Show the sparse scalar c_i anchors over the Mach-alpha domain.
Source data: assets/pinn_subsonic/csv/anchor_budget_runs/N340/Table_anchors.csv
Scientific computation: Publishes the validated processed asset; no training is run.
Original generator: code/src/scripts/utils/build_data_assets.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/pinn_subsonic/png/article/N340/Fig_sparse_ci_anchor_map_Mach_alpha_N340.png",
        "FigS3_1_sparse_ci_anchor_map_Mach_alpha_N340.png",
    )
