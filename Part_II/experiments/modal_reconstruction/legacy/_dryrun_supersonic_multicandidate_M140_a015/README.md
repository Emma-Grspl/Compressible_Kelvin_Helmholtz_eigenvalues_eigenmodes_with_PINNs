# Supersonic Multicandidate Audit

Ce dossier contient un audit multi-candidats supersonique avec verrouillage Blumen,
selection stricte et robustesse de boite.

## Configuration

- `output_stem`: `_dryrun_supersonic_multicandidate_M140_a015`
- `candidate_source`: `auto`
- `max_candidates_per_point`: `6`
- `box_required`: `True`
- `dry_run_candidates`: `True`
- `append_validated`: `False`

## Critere strict

- `stage1 <= 0.05`
- `stage2 <= 0.0001`
- `|delta ci| <= 0.008`
- `|delta ci| / ci_ref <= 0.1`
- `|delta cr| <= 0.035` si `c_r` de reference est disponible
- `box_robustness_max_rel_l2 <= 0.15`
- `peak_shift <= 0.75` si disponible

## Points demandes

- `0.150000:1.400000`

## Sorties

- `all_candidates.csv` : tous les candidats testes ou prepares
- `accepted_points.csv` : un candidat accepte par point quand il existe
- `rejected_candidates.csv` : tous les candidats rejetes
- `point_summary.csv` : resume une ligne par point
- `run_config.json` : configuration du run

## Etat courant

- points resumes : `1`
- points avec candidat accepte : `0`
- catalogues existants considers : `40`

## Garantie de securite

- aucun point n'est ajoute automatiquement a `validated_modal_points`
- si `--append-validated` est active, seule une proposition locale est ecrite dans ce dossier
