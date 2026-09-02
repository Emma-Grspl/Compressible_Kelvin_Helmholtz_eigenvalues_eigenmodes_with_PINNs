"""
Supplementary figure: Fig. S1.3
Output: article/assets/FigS1_3_classical_subsonic_convergence.png
Purpose: Report classical subsonic domain and resolution convergence.
Source data: assets/classic_subsonic/csv/article/convergence/tables/Table_classical_subsonic_convergence.csv
Scientific computation: Publishes the validated processed asset; no solve is run.
Original generator: code/plots/scripts/pinn_subsonic/plot_subsonic_classical_convergence_article.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/classic_subsonic/png/article/convergence/figures/Fig_classical_subsonic_convergence.png",
        "FigS1_3_classical_subsonic_convergence.png",
    )
