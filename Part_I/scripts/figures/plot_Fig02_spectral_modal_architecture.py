"""
Article figure: Fig. 2
Output: article/assets/Fig02_spectral_modal_architecture_subsonic.png
Purpose: Summarize the spectral-modal architecture and independent GEP selection.
Source data: schematic generated from the documented Part I workflow.
Scientific computation: Publishes the validated processed asset; no solve is run.
Original generator: scripts/analysis/complementary_audits/make_fig2_architecture.py
"""

from _publish_validated_asset import publish_validated_asset


if __name__ == "__main__":
    publish_validated_asset(
        "assets/complementary_audits/final_figures/Fig02_spectral_modal_architecture_subsonic.png",
        "Fig02_spectral_modal_architecture_subsonic.png",
    )
