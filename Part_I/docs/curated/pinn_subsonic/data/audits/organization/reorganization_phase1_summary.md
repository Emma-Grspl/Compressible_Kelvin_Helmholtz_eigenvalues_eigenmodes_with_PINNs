# Phase 1 - Audit de reorganisation PINN subsonique

## Etat initial

- `pinn_subsonic/` contient 484 fichiers pour environ 240 MiB et est entierement
  non suivi par Git dans le worktree courant.
- `assets/pinn_subsonic/` contient 100 fichiers presents et non suivis. Git
  signale par ailleurs 55 anciens fichiers suivis comme deja supprimes. Ces
  suppressions preexistantes ne sont ni restaurees ni modifiees par ce plan.
- Les racines actuelles ne respectent pas encore la structure cible :
  `pinn_subsonic/` a dix entrees de premier niveau et
  `assets/pinn_subsonic/` en a six, plus `.DS_Store`.
- 49 charts sont declares `resolved/present` par `model_manifest.csv` et les 49
  fichiers `model_state.pt` sont presents.
- 81 liens symboliques de compatibilite existent dans
  `pinn_subsonic/data/scientific_outputs/local_atlas_v1`. Ils ne seront pas
  reproduits dans l'arborescence canonique.

## Artefacts de securite

- `reorganization_plan.csv` : 776 fichiers inventories avec chemin source,
  destination proposee, categorie, statut Git, action et justification.
- `scientific_hashes_before.csv` : SHA256 de 381 fichiers `.pt`, `.pth`,
  `.npz`, `.npy`, `.csv`, `.pdf` et `.png`, avec identification des liens.
- `checkpoint_hashes_before.csv` : SHA256 et taille des 49 checkpoints.
- `figure_pair_audit.csv` : inventaire des couples PDF/PNG avant mouvement.

Le plan propose actuellement :

- 574 mouvements ;
- 191 archivages sans suppression ;
- 7 fichiers centraux conserves en place ;
- 4 fichiers ambigus, donc aucune execution de la phase 2.

## Decisions non ambigues importantes

- Les 49 checkpoints et leurs metadonnees vont vers
  `pinn_subsonic/atlas/charts/<chart_id>/`.
- Les manifests de charts et de couverture vont vers
  `pinn_subsonic/atlas/manifests/` ; les autres manifests scientifiques vont
  vers `pinn_subsonic/assets/manifests/`.
- Les scripts racine, `scripts/assets_v2/`, les launchers subsoniques et les
  scripts deja regroupes sous `pinn_subsonic/scripts/` ont une destination
  explicite training, plot ou audits.
- Les scripts et launchers classiques restent en place.
- Les anciennes figures rejetees, les variantes `backup_blue`, les bytecodes,
  les anciens constructeurs de liens et les doublons byte-identiques sont
  proposes a l'archive datee
  `archive/pinn_subsonic_reorganization_20260721/`.
- Les figures modales portant historiquement le meme nom mais representant des
  diagnostics differents sont toutes conservees sous des noms descriptifs
  distincts. Aucune version scientifique n'est ecrasee.
- La version canonique proposee de `code/plots/scripts/pinn_subsonic/canonical_source/source_tree/plot_longwave_mapping_audit.py` est celle
  de `pinn_subsonic/scripts/figures/`, car elle corrige le decodage des valeurs
  `xi` a nombre variable de chiffres.

## Figures incompletes avant reorganisation

L'arborescence actuellement visible sous `assets/pinn_subsonic` contient six
stems incomplets :

- `ci_supervision_needed_barplot` : PDF absent ;
- quatre variantes `Fig_modes_*_backup_blue` : PDF absent, mais ces variantes
  sont explicitement des sauvegardes et sont proposees a l'archive ;
- `comparaisons_modales/Fig06_representative_modes` : PNG absent dans ce
  dossier, mais un PNG correspondant existe dans le paquet scientifique local
  et le plan rassemble les deux formats sous un stem canonique.

La regeneration de `ci_supervision_needed_barplot` devra utiliser son script
Matplotlib afin de produire un PDF vectoriel ; aucune encapsulation du PNG
n'est proposee.

## Blocages restant a resoudre

### 1. `code/src/scripts/evaluation/evaluate_subsonic_atlas_offgrid.py`

Usage : validation off-grid des charts et chargement du fournisseur `CiGridIDW`.

Destination probable :
`pinn_subsonic/scripts/scripts_audits/validate_subsonic_atlas_offgrid.py`.

Blocage : importe le module absent
`scripts.dev.train_subsonic_seedGEP_pq2d_continuous_M_alpha_etaaware`.
Sans ce fichier, le loader de chart ne peut pas etre preserve ni teste.

### 2. `archive/code/scripts/assets_v2/build_blumen_exact_point_comparison.py`

Usage : comparaison pointwise Blumen/PINN/GEP et production de figures et
manifests.

Destinations possibles : `scripts_plot` pour le constructeur de figures, ou
`scripts_audits` selon la regle qui classe les comparaisons PINN/classique.

Blocage : depend de `CiGridIDW` fourni par le meme module d'entrainement absent
et de `build_exact4_article_modes.py`.

### 3. `archive/code/scripts/assets_v2/build_exact4_article_modes.py`

Usage : chargement des checkpoints, reconstruction des quatre champs et
generation des figures modales exactes pour l'article.

Destination probable : `pinn_subsonic/scripts/scripts_plot/`.

Blocage : importe deux modules absents :

- `scripts.dev.train_subsonic_seedGEP_pq2d_continuous_M_alpha_etaaware` ;
- `scripts.train_kh_subsonic_2d_pressure_pq_firstorder_mini`.

### 4. `archive/code/scripts/assets_v2/extract_mode_profiles_20.py`

Usage : extraction des profils modaux PINN direct et GEP final pour les 20
points de validation.

Destination probable : `pinn_subsonic/scripts/scripts_plot/`, car ses NPZ sont
des entrees directes des figures modales.

Blocage : importe les deux memes modules d'architecture absents que le script
precedent.

Les archives locales ZIP/TAR et les dossiers `archive/` ou `_quarantine/` ont
ete recherches sans retrouver ces deux modules. Il faut soit recuperer leurs
sources depuis la branche ou la machine qui a produit les checkpoints, soit
declarer explicitement ces quatre scripts comme historiques et les archiver.

## Etat de la phase 2

La phase 2 n'est pas executee, conformement a la consigne : le plan contient
encore quatre lignes `ambiguous`. Aucun fichier scientifique, script, asset ou
checkpoint n'a ete deplace ou renomme.
