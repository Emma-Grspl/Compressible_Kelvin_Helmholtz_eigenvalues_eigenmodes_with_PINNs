"""
Article figure: Fig. 6
Output: article/assets/Fig06_atlas_routing_interface_consistency_N340.png
Purpose: Quantify consistency across deterministic atlas routing interfaces.
Source data: assets/pinn_subsonic/csv/article/N340/Table_atlas_overlap_consistency_N340.csv
Scientific computation: Publishes the validated processed asset; no training is run.
Original generator: code/src/scripts/utils/build_N340_overlap_asset.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/complementary_audits/final_figures/Fig06_atlas_routing_interface_consistency_N340.png",
        "Fig06_atlas_routing_interface_consistency_N340.png",
    )
