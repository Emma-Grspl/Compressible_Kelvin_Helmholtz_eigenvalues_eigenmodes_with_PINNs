"""
Supplementary figure: Fig. S4.1
Output: article/assets/FigS4_1_GEP_N_convergence.png
Purpose: Report resolution convergence of the independently selected GEP eigenpair.
Source data: assets/pinn_subsonic/csv/article/results_pinn/release_final/tables/Table_GEP_N_convergence.csv
Scientific computation: Publishes the corrected validated asset; no GEP is run.
Original generator: code/src/scripts/gep/selection/solve_joint_gep_n_convergence.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/complementary_audits/final_figures/FigS4_1_GEP_N_convergence.png",
        "FigS4_1_GEP_N_convergence.png",
    )
