# Part II article figures

`article/assets/` contains exactly the 25 curated publication figures for the
supersonic study. `assets/` contains the broader scientific asset collection,
including classical references, atlas data, reviewer controls, and audit
outputs.

`FIGURE_MANIFEST.csv` is the machine-readable map from each final filename to
its canonical source, main source-data location, SHA-256 checksum, and the
publication entrypoint. `scripts/figures/publish_article_assets.py` republishes
the curated package by copying validated assets; it does not retrain a neural
model or recompute a shooting campaign.

Three supplementary figures were available only as authoritative PDF files in
the supplied supplementary package. Their required PNG publication assets were
rendered from those PDFs at 180 dpi and retain the source PDF alongside the
rendered canonical PNG under `assets/article_sources/`. This rendering detail
is identified in the manifest.
