#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

REQUIRED_NOW = [
    'figures/Fig02_atlas_architecture_and_coverage.pdf',
    'figures/Fig02a_atlas_status_footprints.pdf',
    'figures/Fig02b_coverage_multiplicity.pdf',
    'figures/SuppFig03_final_chart_assignment.pdf',
    'figures/Fig02c_operational_pipeline_map.pdf',
    'figures/Fig03_sparse_spectral_supervision.pdf',
    'figures/Fig04_ci_direct_and_errors.pdf',
    'figures/Fig05_ci_gep_parity_and_gain.pdf',
    'figures/SuppFig_random_offgrid_validation.pdf',
    'figures/Fig07_final_modal_error_heatmaps.pdf',
    'figures/SuppFig_modal_overlap_defect.pdf',
    'data/validation_mode_points_20.csv',
]
OPTIONAL_REQUIRING_EXTRA_INPUTS = [
    'figures/Fig04a_Blumen_PINN_GEP_overlay.pdf',
    'figures/Fig04b_Blumen_error_maps.pdf',
    'figures/Fig04c_Blumen_Mach_cuts.pdf',
    'figures/SuppFig_direct_PINN_modal_error_heatmaps.pdf',
    'supplement/supp_modes_direct_PINN_vs_classic_20_points.pdf',
    'supplement/supp_modes_PINN_GEP_vs_classic_GEP_20_points.pdf',
    'figures/Fig06_representative_modes.pdf',
    'figures/SuppFig08_longwave_mapping_audit.pdf',
    'figures/SuppFig07_GEP_N_convergence.pdf',
]


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--asset-dir',required=True);a=p.parse_args();root=Path(a.asset_dir)
    rows=[]
    for rel in REQUIRED_NOW:
        rows.append({'asset':rel,'class':'required_from_existing_release','present':(root/rel).exists()})
    for rel in OPTIONAL_REQUIRING_EXTRA_INPUTS:
        rows.append({'asset':rel,'class':'requires_source_or_extraction_campaign','present':(root/rel).exists()})
    df=pd.DataFrame(rows);print(df.to_string(index=False));(root/'asset_v2_check.csv').write_text(df.to_csv(index=False))
    report_path=root/'build_report.json'
    if report_path.exists():
        print('\nBuild report:');print(report_path.read_text())
    missing_required=df[(df['class']=='required_from_existing_release')&(~df.present)]
    if not missing_required.empty: raise SystemExit('Required assets are missing.')
    print('\nCORE SCIENTIFIC ASSETS: VALIDATED')
    if (~df[df['class']=='requires_source_or_extraction_campaign'].present).any():
        print('Some supplementary assets still require Blumen data, direct-mode profiles, or convergence-sweep CSVs.')

if __name__=='__main__':main()
