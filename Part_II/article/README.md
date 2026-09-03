# Part II article figures

`article/assets/` contains exactly the 25 curated publication figures for the
supersonic study. `assets/` contains the broader scientific asset collection,
including classical references, atlas data, reviewer controls, and audit
outputs.

`FIGURE_MANIFEST.csv` is the machine-readable map from each final filename to
its canonical source, main source-data location, SHA-256 checksum, and upstream
scientific plotting provenance when that script is available in this repository.
`plot_script` therefore denotes a scientific generator when known, not a
universal publication wrapper. `NO_SCRIPT_FOUND` means that no upstream
generator can be assigned without guessing.

`scripts/figures/publish_article_assets.py` is a separate publication step: it
copies validated assets and does not retrain a neural model, recompute a
shooting campaign, or replace upstream scientific plotting provenance.

Three supplementary figures were available only as authoritative PDF files in
the supplied supplementary package. Their required PNG publication assets were
rendered from those PDFs at 180 dpi and retain the source PDF alongside the
rendered canonical PNG under `assets/article_sources/`. This rendering detail
is identified in the manifest.
