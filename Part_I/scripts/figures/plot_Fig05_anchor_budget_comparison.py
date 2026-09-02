"""
Article figure: Fig. 5
Output: article/assets/Fig05_anchor_budget_comparison.png
Purpose: Compare spectral accuracy across sparse-anchor budgets.
Source data: assets/pinn_subsonic/csv/article/N340/Table_anchor_budget_comparison.csv
Scientific computation: Publishes the validated processed asset; no training is run.
Original generator: code/src/scripts/utils/build_anchor_budget_asset.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/complementary_audits/final_figures/Fig05_anchor_budget_comparison.png",
        "Fig05_anchor_budget_comparison.png",
    )
