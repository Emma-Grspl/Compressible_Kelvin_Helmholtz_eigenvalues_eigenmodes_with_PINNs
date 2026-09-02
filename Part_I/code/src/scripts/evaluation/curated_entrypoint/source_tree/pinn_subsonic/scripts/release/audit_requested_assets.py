#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--repo-root', default='.')
    p.add_argument('--output-dir', default='assets/pinn_subsonic/asset_audit')
    return p.parse_args()


def exists_all(root: Path, paths: list[str]):
    found = [p for p in paths if (root / p).exists()]
    return len(found) == len(paths), found


def exists_any(root: Path, paths: list[str]):
    found = [p for p in paths if (root / p).exists()]
    return bool(found), found


def add(rows, aid, family, title, status, found, expected, note=''):
    rows.append({
        'asset_id': aid,
        'family': family,
        'title': title,
        'status': status,
        'found_paths': ' | '.join(found),
        'expected_or_candidate_paths': ' | '.join(expected),
        'note': note,
    })


def main():
    a = args()
    root = Path(a.repo_root).resolve()
    out = root / a.output_dir
    out.mkdir(parents=True, exist_ok=True)

    sci = 'assets/pinn_subsonic/local_atlas_v1/publication_assets_scientific_v2'
    full = 'assets/pinn_subsonic/local_atlas_v1/publication_assets_fullrect_v1'
    off = 'assets/pinn_subsonic/local_atlas_v1/offgrid_validation_384'
    atlas = 'assets/pinn_subsonic/local_atlas_v1'
    rel = 'assets/pinn_subsonic/release_v1'
    rows = []

    groups = [
        ('A1','Atlas','Carte des charts avec statut scientifique',[f'{sci}/figures/Fig02a_atlas_status_footprints.pdf',f'{sci}/data/atlas_catalog_operational.csv']),
        ('A2','Atlas','Multiplicité de couverture',[f'{sci}/figures/Fig02b_coverage_multiplicity.pdf',f'{sci}/data/coverage_and_assignment_grid.csv']),
        ('A3','Atlas','Assignation finale des charts',[f'{sci}/figures/SuppFig03_final_chart_assignment.pdf',f'{sci}/data/coverage_and_assignment_grid.csv']),
        ('A4','Atlas','Carte du pipeline numérique',[f'{sci}/figures/Fig02c_operational_pipeline_map.pdf']),
        ('B1','c_i','Blumen vs PINN/GEP',[f'{sci}/figures/Fig04a_Blumen_PINN_GEP_overlay.pdf',f'{sci}/figures/Fig04b_Blumen_error_maps.pdf',f'{sci}/figures/Fig04c_Blumen_Mach_cuts.pdf',f'{sci}/data/blumen_ci_datasets.csv',f'{sci}/data/Blumen_interpolated_comparison.csv']),
        ('B2','c_i','Classique vs PINN direct',[f'{sci}/figures/Fig04_ci_direct_and_errors.pdf']),
        ('B3','c_i','Classique+GEP vs PINN-seeded GEP',[f'{sci}/figures/Fig05_ci_gep_parity_and_gain.pdf']),
        ('B4','c_i','Gain apporté par le GEP',[f'{sci}/figures/Fig05_ci_gep_parity_and_gain.pdf']),
        ('B5','c_i','Validation hors grille',[f'{rel}/figures/supplement/SuppFig_random_offgrid_validation.pdf',f'{rel}/validation/offgrid_validation_results_384_release.csv',f'{rel}/validation/offgrid_validation_release_summary.csv',f'{rel}/validation/offgrid_validation_release_checks.csv']),
        ('C1','Modes','PDF 20 points PINN direct vs classique',[f'{sci}/supplement/supp_modes_direct_PINN_vs_classic_20_points.pdf']),
        ('C2','Modes','PDF 20 points PINN+GEP vs classique+GEP',[f'{sci}/supplement/supp_modes_PINN_GEP_vs_classic_GEP_20_points.pdf']),
        ('C3','Modes','Sélection stratifiée des 20 points',[f'{sci}/data/validation_mode_points_20.csv']),
        ('C4','Modes','Heatmaps modales PINN direct p/rho/u/v',[f'{sci}/figures/SuppFig_direct_PINN_modal_error_heatmaps.pdf']),
        ('C5','Modes','Heatmaps modales finales p/rho/u/v',[f'{sci}/figures/Fig07_final_modal_error_heatmaps.pdf']),
        ('C6','Modes','Défaut d’overlap modal',[f'{sci}/figures/SuppFig_modal_overlap_defect.pdf']),
        ('D2','Audits GEP','Audit long-wave map5/map10/map15/map20',[f'{sci}/figures/SuppFig08_longwave_mapping_audit.pdf']),
        ('D3','Audits GEP','Audit near-neutral et continuation bidirectionnelle',[f'{sci}/figures/SuppFig09_near_neutral_continuation_audit.pdf',f'{sci}/figures/SuppFig10_bidirectional_continuation_overlap.pdf']),
        ('D4','Audits GEP','Corrections de branche de référence',[f'{rel}/audits/offgrid_reference_branch_corrections.csv',f'{rel}/audits/offgrid_0246_branch_resolution.csv']),
        ('FIG01','Figures article','Schéma de méthode',[f'{sci}/figures/Fig01_method.pdf']),
        ('FIG02','Figures article','Architecture et couverture',[f'{sci}/figures/Fig02_atlas_architecture_and_coverage.pdf']),
        ('FIG03','Figures article','Supervision sparse',[f'{sci}/figures/Fig03_sparse_spectral_supervision.pdf']),
        ('FIG04','Figures article','Validation c_i PINN/Blumen',[f'{sci}/figures/Fig04_ci_direct_and_errors.pdf',f'{sci}/figures/Fig04a_Blumen_PINN_GEP_overlay.pdf']),
        ('FIG05','Figures article','Validation c_i après GEP',[f'{sci}/figures/Fig05_ci_gep_parity_and_gain.pdf']),
        ('FIG06','Figures article','Modes représentatifs',[f'{sci}/figures/Fig06_representative_modes.pdf']),
        ('FIG07','Figures article','Erreurs modales finales',[f'{sci}/figures/Fig07_final_modal_error_heatmaps.pdf']),
        ('FIG08','Figures article','Régimes difficiles',[f'{sci}/figures/SuppFig07_GEP_N_convergence.pdf',f'{sci}/figures/SuppFig08_longwave_mapping_audit.pdf',f'{sci}/figures/SuppFig09_near_neutral_continuation_audit.pdf',f'{sci}/figures/SuppFig10_bidirectional_continuation_overlap.pdf']),
    ]
    for aid,fam,title,expected in groups:
        ok, found = exists_all(root, expected)
        add(rows, aid, fam, title, 'PRESENT' if ok else ('PARTIAL' if found else 'MISSING'), found, expected)

    # A5: explicit anchors required; fallback points are only partial.
    expected = [f'{sci}/figures/Fig03_sparse_spectral_supervision.pdf',f'{sci}/build_report.json']
    ok, found = exists_all(root, expected)
    status = 'PRESENT' if ok else ('PARTIAL' if found else 'MISSING')
    note = ''
    report = root / f'{sci}/build_report.json'
    if report.exists():
        try:
            data = json.loads(report.read_text())
            if not data.get('anchor_files'):
                status = 'PARTIAL'
                note = 'Figure fondée sur les points de validation faute de CSV explicite des ancres.'
        except Exception as exc:
            status = 'PARTIAL'; note = str(exc)
    add(rows,'A5','Atlas','Carte de supervision spectrale sparse',status,found,expected,note)

    # D1 policy: one of several accepted files.
    candidates = [f'{atlas}/gep_regime_policy_fullrect.csv',f'{full}/data/gep_regime_policy.csv']
    ok, found = exists_any(root, candidates)
    add(rows,'D1','Audits GEP','Politique numérique finale','PRESENT' if ok else 'MISSING',found,candidates)

    # D5: figure exists, but full N=201/301/401 may be incomplete.
    expected = [f'{sci}/figures/SuppFig07_GEP_N_convergence.pdf',f'{sci}/tables/GEP_N_convergence.csv']
    ok, found = exists_all(root, expected)
    status = 'PRESENT' if ok else ('PARTIAL' if found else 'MISSING')
    note = ''
    table = root / expected[1]
    if table.exists():
        try:
            df = pd.read_csv(table)
            text = ' '.join(map(str, df.columns)) + ' ' + df.astype(str).head(200).to_csv(index=False)
            if not all(str(n) in text for n in (201,301,401)):
                status = 'PARTIAL'
                note = 'La triplette demandée N=201/301/401 n’est pas entièrement identifiable.'
        except Exception as exc:
            status = 'PARTIAL'; note = str(exc)
    add(rows,'D5','Audits GEP','Convergence N=201/301/401',status,found,expected,note)

    table_groups = [
        ('E1','Catalogue des atlas',[f'{sci}/data/atlas_catalog_operational.csv']),
        ('E2','Métriques par atlas',[f'{atlas}/atlas_fullrect_gep_final_metrics_by_chart.csv',f'{full}/data/atlas_fullrect_gep_final_metrics_by_chart.csv']),
        ('E3','Résultats point par point',[f'{sci}/data/validation_pointwise_canonical.csv',f'{off}/offgrid_validation_results_384_release.csv']),
        ('E4','Table de couverture et assignation',[f'{sci}/data/coverage_and_assignment_grid.csv']),
        ('E5','Table des 20 points modaux',[f'{sci}/data/validation_mode_points_20.csv']),
        ('E6','Résumé global',[f'{full}/data/release_summary.json',f'{off}/offgrid_validation_release_summary.csv']),
    ]
    for aid,title,candidates in table_groups:
        ok, found = exists_any(root,candidates)
        add(rows,aid,'Tables',title,'PRESENT' if ok else 'MISSING',found,candidates)

    frame = pd.DataFrame(rows).sort_values(['family','asset_id']).reset_index(drop=True)
    csv_path = out / 'requested_assets_audit.csv'
    json_path = out / 'requested_assets_audit.json'
    frame.to_csv(csv_path,index=False)
    payload = {
        'repo_root': str(root),
        'counts': frame.status.value_counts().to_dict(),
        'missing_asset_ids': frame.loc[frame.status.eq('MISSING'),'asset_id'].tolist(),
        'partial_asset_ids': frame.loc[frame.status.eq('PARTIAL'),'asset_id'].tolist(),
        'rows': frame.to_dict(orient='records'),
    }
    json_path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(frame[['asset_id','family','title','status']].to_string(index=False))
    print('\nCounts:\n'+frame.status.value_counts().to_string())
    print('\nCSV :',csv_path)
    print('JSON:',json_path)
    if frame.status.eq('MISSING').any():
        raise SystemExit(2)


if __name__ == '__main__':
    main()
