"""
Supplementary figure: Fig. S1.1
Output: article/assets/FigS1_1_classical_subsonic_mode_four_fields_a0p500_M0p500.png
Purpose: Show the classical pressure, density, and velocity eigenmode fields.
Source data: classical solver output at Mach=0.5 and alpha=0.5.
Scientific computation: Publishes the validated processed asset; no solve is run.
Original generator: code/plots/scripts/pinn_subsonic/plot_classical_subsonic_mode_four_fields.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/classic_subsonic/png/modes/Fig_classical_subsonic_mode_four_fields_a0p500_M0p500.png",
        "FigS1_1_classical_subsonic_mode_four_fields_a0p500_M0p500.png",
    )
