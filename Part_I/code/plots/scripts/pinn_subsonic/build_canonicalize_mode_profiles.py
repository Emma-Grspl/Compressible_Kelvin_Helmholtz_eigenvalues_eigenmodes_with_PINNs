#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import pandas as pd


def main() -> None:
    p=argparse.ArgumentParser(description='Find existing per-point mode-profile CSV/NPZ files and copy them into the canonical layout.')
    p.add_argument('--points-csv',required=True)
    p.add_argument('--search-root',action='append',required=True,help='May be supplied multiple times.')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--pipeline',choices=['direct','gep'],required=True)
    a=p.parse_args()
    points=pd.read_csv(a.points_csv); dest=Path(a.output_dir)/a.pipeline; dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for pid in points.point_id.astype(str):
        hits=[]
        for root_s in a.search_root:
            root=Path(root_s)
            if not root.exists(): continue
            hits.extend(root.rglob(f'{pid}*.npz')); hits.extend(root.rglob(f'{pid}*.csv'))
        # Exclude summaries and paths, which are not modal profiles.
        hits=[h for h in hits if h.name not in {'summary.csv','continuation_paths.csv'} and 'summary' not in h.name.lower()]
        if not hits:
            rows.append({'point_id':pid,'status':'missing','source':''}); continue
        source=sorted(hits,key=lambda x:(len(str(x)),str(x)))[0]
        target=dest/f'{pid}{source.suffix.lower()}'; shutil.copy2(source,target)
        rows.append({'point_id':pid,'status':'copied','source':str(source),'target':str(target)})
    report=pd.DataFrame(rows); report.to_csv(dest/'canonicalization_report.csv',index=False); print(report.to_string(index=False))
    print('\nFound:',(report.status=='copied').sum(),'/',len(report))

if __name__=='__main__': main()
