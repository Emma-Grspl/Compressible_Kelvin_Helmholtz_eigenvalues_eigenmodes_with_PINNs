"""
Supplementary figure: Fig. S1.2
Output: article/assets/FigS1_2_subsonic_error_map_classical.png
Purpose: Map the discrepancy between Blumen data and the classical reference.
Source data: assets/classic_subsonic/csv/data/Table_subsonic_blumen_shooting_metrics_by_point.csv
Scientific computation: Publishes the validated processed asset; no solve is run.
Original generator: archive/code/scripts/build_presentation_plots.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/classic_subsonic/png/article/reference_comparison/FigS1_2_subsonic_error_map_classical.png",
        "FigS1_2_subsonic_error_map_classical.png",
    )
