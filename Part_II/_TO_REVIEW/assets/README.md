Ce dossier contient uniquement les sorties de référence encore actives.

Organisation actuelle :

- `classic_subsonic/`
  Référence classique subsonique retenue.
- `classic_supersonic/`
  Référence classique supersonique, audits de branche et comparaison à Blumen.
- `pinn_subsonic/`
  Sorties de référence du PINN subsonique retenues pour `c_i` et le mode.
- `pinn_supersonic/`
  Réservé pour la suite. Pas encore de référence figée.

Sous-structures utilisées :

- `data/` : CSV et résumés
- `plots/` : figures de synthèse
- `modes/` : reconstructions modales
- `diagnostics/` : audits complémentaires utiles pour justifier la sélection
- `blumen_reference/` : données Blumen redigitalisées
- `shooting/` : sorties de référence par tir, validations visuelles et expériences ciblées
- `gep/` : diagnostics GEP conservés pour comparaison spectrale
- `frozen_fixed_mach_alpha_sweep_best/` : meilleur run haut-`alpha` ciblé conservé pour historique
- `mach_fixed/frozen_riccati_multibranch_ci_best/` : base 1D figée provisoire pour `c_i`, avec réserve explicite sur les modes à bas `alpha`

Statut du workspace actif :

- l'ancien `assets/blumen_gep/` n'existe plus
- les sorties de runs PINN obsolètes dans `model_saved/` ont été supprimées après gel du meilleur cas
- `pinn_supersonic/` reste volontairement vide de résultats, en attente du protocole supersonique

Les deux seules sources supersoniques Blumen conservées dans le repo de travail sont :

- `assets/classic_supersonic/csv/blumen_validation/supersonic/table_cr_datasets.csv`
- `assets/classic_supersonic/csv/blumen_validation/supersonic/table_ci_datasets.csv`
