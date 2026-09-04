#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def show_csv(path: Path)->str:
    try:
        f=pd.read_csv(path,nrows=3)
        return ','.join(f.columns)
    except Exception as exc:
        return f'ERROR:{exc}'


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--repo-root',default='.');p.add_argument('--atlas-root',default='assets/pinn_subsonic/local_atlas_v1');a=p.parse_args();repo=Path(a.repo_root).resolve();atlas=repo/a.atlas_root
    patterns={
        'Blumen':['*blumen*.csv','*Blumen*.csv'],
        'mode profiles':['*mode*.npz','*profile*.npz','*mode*.csv','*profile*.csv'],
        'mapping audits':['*map10*.csv','*map20*.csv','*mapping*audit*.csv','*longwave*.csv'],
        'N convergence':['*convergence*.csv','*N401*.csv','*n401*.csv'],
        'continuation paths':['continuation_paths.csv'],
    }
    rows=[]
    for category,pats in patterns.items():
        found=[]
        for pat in pats: found.extend(atlas.rglob(pat))
        for path in sorted(set(found))[:200]:
            rows.append({'category':category,'path':str(path),'columns':show_csv(path) if path.suffix.lower()=='.csv' else 'NPZ'})
    frame=pd.DataFrame(rows);print(frame.to_string(index=False) if not frame.empty else 'No optional source data discovered.')
    out=atlas/'publication_assets_scientific_v2'/'missing_input_inventory.csv';out.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(out,index=False);print(f'\nInventory: {out}')

if __name__=='__main__':main()
