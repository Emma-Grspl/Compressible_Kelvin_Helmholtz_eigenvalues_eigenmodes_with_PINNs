#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from plots.scripts.pinn_subsonic.utils_asset_common import canonicalize_release, resolve_release_frame


def choose_closest(df: pd.DataFrame, M: float, eta: float, used: set[str], label: str) -> dict:
    pool=df[~df.point_id.isin(used)].copy()
    if pool.empty:
        raise RuntimeError('No unused validation points remain.')
    pool['_d']=(pool.Mach-M)**2+(pool.eta-eta)**2
    row=pool.sort_values(['_d','modal_error_final'],ascending=[True,False]).iloc[0].to_dict()
    row['selection_stratum']=label
    return row


def choose_worst(df: pd.DataFrame, mask: pd.Series, used: set[str], label: str) -> dict:
    pool=df[mask & ~df.point_id.isin(used)].copy()
    if pool.empty:
        pool=df[~df.point_id.isin(used)].copy()
    metric=pool.modal_error_final.fillna(pool.ci_final_abs_err)
    row=pool.loc[metric.idxmax()].to_dict()
    row['selection_stratum']=label
    return row


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',default='.')
    p.add_argument('--atlas-root',default='assets/pinn_subsonic/local_atlas_v1')
    p.add_argument('--output',default='assets/pinn_subsonic/local_atlas_v1/publication_assets_scientific_v2/data/validation_mode_points_20.csv')
    a=p.parse_args()
    repo=Path(a.repo_root).resolve(); atlas=repo/a.atlas_root
    _,raw=resolve_release_frame(atlas); df=canonicalize_release(raw)
    used:set[str]=set(); rows=[]

    # Named strata from the agreed scientific list.
    strata=[
        ('ultralow',lambda d:(d.Mach<=0.08)&(d.eta<=0.08)),
        ('low_Mach_low_eta',lambda d:(d.Mach<=0.20)&(d.eta<=0.25)),
        ('low_Mach_high_eta',lambda d:(d.Mach<=0.20)&(d.eta>=0.75)),
        ('interior_1',lambda d:(d.Mach.between(.25,.55))&(d.eta.between(.25,.55))),
        ('interior_2',lambda d:(d.Mach.between(.45,.75))&(d.eta.between(.45,.75))),
        ('high_Mach_interior',lambda d:(d.Mach>=.85)&(d.eta.between(.25,.75))),
        ('very_high_Mach',lambda d:(d.Mach>=.95)&(d.eta.between(.30,.80))),
        ('longwave_low_M',lambda d:(d.alpha<=.06)&(d.Mach<.80)),
        ('longwave_high_M',lambda d:(d.alpha<=.06)&(d.Mach>=.80)),
        ('extreme_longwave',lambda d:(d.Mach>=.92)&(d.eta<=.08)),
        ('near_neutral_low_M',lambda d:(d.eta>=.96)&(d.Mach<.40)),
        ('near_neutral_mid_M',lambda d:(d.eta>=.96)&(d.Mach.between(.40,.80))),
        ('near_neutral_high_M',lambda d:(d.eta>=.96)&(d.Mach>.80)),
        ('reference_corrected',lambda d:d.reference_status.eq('continuation_corrected_reference')),
        ('continuation_bidirectional',lambda d:d.continuation_status.eq('validated_bidirectional')),
        ('worst_ci_seed',lambda d:pd.Series(True,index=d.index)),
        ('worst_ci_final',lambda d:pd.Series(True,index=d.index)),
        ('worst_modal_final',lambda d:pd.Series(True,index=d.index)),
        ('lowest_overlap',lambda d:pd.Series(True,index=d.index)),
    ]

    for label,fn in strata:
        pool=df[~df.point_id.isin(used)].copy()
        if label=='worst_ci_seed':
            row=pool.loc[pool.ci_seed_abs_err.idxmax()].to_dict()
        elif label=='worst_ci_final':
            row=pool.loc[pool.ci_final_abs_err.idxmax()].to_dict()
        elif label=='worst_modal_final':
            row=pool.loc[pool.modal_error_final.idxmax()].to_dict()
        elif label=='lowest_overlap':
            row=pool.loc[pool.p_overlap_final.idxmin()].to_dict()
        else:
            row=choose_worst(df,fn(df),used,label)
        row['selection_stratum']=label; rows.append(row); used.add(str(row['point_id']))

    # Force OFFGRID_0246 into the selection, replacing the last generic point if needed.
    if 'OFFGRID_0246' in set(df.point_id) and 'OFFGRID_0246' not in used:
        row=df[df.point_id.eq('OFFGRID_0246')].iloc[0].to_dict(); row['selection_stratum']='branch_case_OFFGRID_0246'
        rows[-1]=row; used.add('OFFGRID_0246')

    # Complete to exactly 20 with high-error unused points.
    while len(rows)<20:
        pool=df[~df.point_id.isin({str(r['point_id']) for r in rows})]
        row=pool.loc[pool.modal_error_final.fillna(pool.ci_final_abs_err).idxmax()].to_dict()
        row['selection_stratum']=f'additional_worst_{len(rows)+1:02d}'; rows.append(row)

    out=pd.DataFrame(rows[:20])
    keep=['selection_stratum','point_id','sample_group','Mach','eta','alpha','chart_id','ci_ref','ci_seed','ci_final','ci_seed_abs_err','ci_final_abs_err','p_rel_final','rho_rel_final','u_rel_final','v_rel_final','p_overlap_final','continuation_status','reference_status']
    out=out[[c for c in keep if c in out.columns]]
    path=repo/a.output; path.parent.mkdir(parents=True,exist_ok=True); out.to_csv(path,index=False)
    print(out.to_string(index=False)); print(f'\nCreated: {path}')

if __name__=='__main__':
    main()
