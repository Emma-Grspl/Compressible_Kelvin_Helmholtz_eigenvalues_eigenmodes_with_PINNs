"""
Article figure: Fig. 9
Output: article/assets/Fig09_near_neutral_growth_rate_cuts_N340.png
Purpose: Show growth-rate cuts near the neutral boundary.
Source data: results/complementary_audits/curated/section4_results/Fig_near_neutral_growth_rate_cuts_N340_data.csv
Scientific computation: Publishes the validated processed asset; no GEP is run.
Original generator: scripts/analysis/complementary_audits/make_fig9_near_neutral_pro.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/complementary_audits/final_figures/Fig09_near_neutral_growth_rate_cuts_N340.png",
        "Fig09_near_neutral_growth_rate_cuts_N340.png",
    )
