"""
Supplementary figure: Fig. S6.1
Output: article/assets/FigS6_1_direct_neural_vs_selected_GEP_modal_error_ecdf_20_N340.png
Purpose: Compare direct neural and selected GEP modal-error distributions.
Source data: assets/pinn_subsonic/csv/article/N340/Table_paired_modal_validation_20_N340.csv
Scientific computation: Publishes the validated processed asset; no GEP is run.
Original generator: code/src/scripts/utils/build_N340_modal_assets.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/complementary_audits/final_figures/FigS6_1_direct_neural_vs_selected_GEP_modal_error_ecdf_20_N340.png",
        "FigS6_1_direct_neural_vs_selected_GEP_modal_error_ecdf_20_N340.png",
    )
