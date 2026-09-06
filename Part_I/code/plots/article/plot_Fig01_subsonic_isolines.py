"""
Article figure: Fig. 1
Output: article/assets/Fig01_subsonic_isolines_blumen_vs_classical.png
Purpose: Compare digitized Blumen isolines with the classical shooting map.
Source data: assets/classic_subsonic/csv/data/Table_subsonic_hybrid_growth_map.csv
Scientific computation: Publishes the validated processed asset; no solve is run.
Original generator: archive/code/scripts/build_presentation_plots.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/classic_subsonic/png/article/reference_comparison/Fig01_subsonic_isolines_blumen_vs_classical.png",
        "Fig01_subsonic_isolines_blumen_vs_classical.png",
    )
