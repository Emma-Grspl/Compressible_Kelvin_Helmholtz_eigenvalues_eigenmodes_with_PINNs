# Workflow Subsonique De Référence

Ce dossier contient plusieurs solveurs de tir subsoniques. La version de référence
à utiliser pour la thèse est désormais :

- [hybrid_subsonic_scan.py](${PROJECT_ROOT}/code/src/scripts/classical/solve_classic_subsonic_gep_shooting_scan.py)

## Principe

Le workflow subsonique verrouillé combine deux solveurs :

- [shooting_subsonic.py](${PROJECT_ROOT}/code/src/scripts/classical/solve_shooting_subsonic.py)
  Solveur principal, rapide, utilisé sur la majorité des points.
- [mstab17_subsonic_solver.py](${PROJECT_ROOT}/code/src/scripts/classical/solve_mstab17_subsonic_solver.py)
  Solveur de contrôle plus robuste près de la neutralité.

Le script hybride :
- résout toute la grille avec le solveur principal ;
- corrige seulement une bande proche de la frontière neutre ;
- produit la carte finale de référence.

## Statut

Ce workflow est considéré comme la référence subsonique actuelle.

Sorties de référence :
- [subsonic_hybrid_growth_map.csv](${PROJECT_ROOT}/assets/blumen_shooting_hybrid/subsonic_hybrid_growth_map.csv)
- [subsonic_hybrid_vs_blumen.png](${PROJECT_ROOT}/assets/blumen_shooting_hybrid/subsonic_hybrid_vs_blumen.png)
- [subsonic_hybrid_error_summary.json](${PROJECT_ROOT}/assets/blumen_shooting_hybrid/subsonic_hybrid_error_summary.json)

Le dernier run validé donne :
- `global_mae_omega ≈ 1.75e-3`
- `global_median_distance ≈ 4.61e-3`

## Scripts

Scripts principaux :
- [hybrid_subsonic_scan.py](${PROJECT_ROOT}/code/src/scripts/classical/solve_classic_subsonic_gep_shooting_scan.py)
  Scan de référence.
- [robust_subsonic_shooting.py](${PROJECT_ROOT}/code/src/scripts/classical/solve_robust_subsonic_shooting.py)
  Solveur point-à-point combinant les deux méthodes.

Scripts de support :
- [compare_subsonic_shooting_solvers.py](${PROJECT_ROOT}/archive/code/classical_solver/subsonic/compare_subsonic_shooting_solvers.py)
  Comparaison entre les deux solveurs.
- [plot_subsonic_error_map.py](${PROJECT_ROOT}/code/plots/scripts/classic_subsonic/plot_classic_subsonic_error_map.py)
  Carte d’erreur.
- [plot_subsonic_ci_map.py](${PROJECT_ROOT}/code/plots/scripts/classic_subsonic/plot_classic_subsonic_ci_map.py)
  Visualisation de `c_i`.

Scripts historiques conservés :
- [reconstruct_blumen_subsonic_shooting.py](${PROJECT_ROOT}/code/src/scripts/classical/solve_classic_blumen_subsonic_shooting_reconstruction.py)
- [reconstruct_blumen_subsonic_robust.py](${PROJECT_ROOT}/code/src/scripts/classical/solve_classic_blumen_subsonic_robust_reconstruction.py)

Ils restent utiles pour comparaison, mais ne sont plus le point d’entrée recommandé.

## Lancement Local

Petit test :

```bash
python3 code/src/scripts/classical/solve_classic_subsonic_gep_shooting_scan.py --num-mach 9 --num-alpha 9
```

Carte plus fine :

```bash
python3 code/src/scripts/classical/solve_classic_subsonic_gep_shooting_scan.py --num-mach 41 --num-alpha 41
```

## Lancement Jean Zay

Script Slurm de référence :
- [jz_submit_subsonic.slurm](${PROJECT_ROOT}/code/src/launch/slurm/jz_submit_subsonic.slurm)

Commande :

```bash
sbatch code/src/launch/slurm/jz_submit_subsonic.slurm
```

## Convention de travail

Pour la suite de la thèse :
- on considère le subsonique comme verrouillé avec ce workflow hybride ;
- les nouveaux développements portent d’abord sur le supersonique ;
- si une modification affecte le subsonique, elle doit être comparée à cette référence.
