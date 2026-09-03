#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path
import numpy as np
import pandas as pd

def locate(columns, variants):
    m={str(c).strip().lower():str(c) for c in columns}
    for v in variants:
        if v.lower() in m: return m[v.lower()]
    return None

def parse_level(value):
    nums=re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",str(value))
    return float(nums[-1]) if nums else np.nan

def load_blumen(path):
    f=pd.read_csv(path)
    mc=locate(f.columns,("Mach","M","mach_physical")); ac=locate(f.columns,("alpha","Alpha","wavenumber"))
    cc=locate(f.columns,("target_ci","ci_level","ci_value","ci","c_i","contour_level"))
    lc=locate(f.columns,("curve_label","label","series","source")); ic=locate(f.columns,("curve_id","curve","line_id"))
    if mc is None or ac is None: raise KeyError(f"Mach/alpha missing in {path}: {f.columns.tolist()}")
    o=f.copy(); o["blumen_row_id"]=np.arange(len(o),dtype=int)
    o["Mach"]=pd.to_numeric(o[mc],errors="coerce"); o["alpha"]=pd.to_numeric(o[ac],errors="coerce")
    o["target_ci"]=pd.to_numeric(o[cc],errors="coerce") if cc else np.nan
    o["curve_label"]=o[lc].astype(str) if lc else ""; o["curve_id_key"]=o[ic].astype(str) if ic else ""
    miss=~np.isfinite(o["target_ci"])
    if miss.any(): o.loc[miss,"target_ci"]=o.loc[miss,"curve_label"].map(parse_level)
    o=o.dropna(subset=["Mach","alpha","target_ci"]); o=o[o["Mach"]>=1.0].copy()
    o["curve_key"]=[f"{i}__{l}__ci_{c:.12g}" for i,l,c in o[["curve_id_key","curve_label","target_ci"]].itertuples(index=False)]
    return o.sort_values("blumen_row_id").reset_index(drop=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--blumen-csv",type=Path,required=True); p.add_argument("--work-root",type=Path,required=True); p.add_argument("--overwrite",action="store_true"); a=p.parse_args()
    root=a.work_root.resolve()
    if root.exists() and a.overwrite: shutil.rmtree(root)
    posdir=root/"targets/positive"; posdir.mkdir(parents=True,exist_ok=True)
    b=load_blumen(a.blumen_csv.resolve()); b.to_csv(root/"targets/blumen_supersonic_all.csv",index=False)
    pos=b[b.target_ci>0].copy(); neu=b[np.isclose(b.target_ci,0.0,atol=1e-14)].copy()
    rows=[]; groups=list(pos.groupby("curve_key",sort=False)); groups.sort(key=lambda kv:(float(kv[1].target_ci.iloc[0]),kv[0]))
    for idx,(key,sub) in enumerate(groups):
        path=posdir/f"curve_{idx:03d}.csv"; sub.sort_values("blumen_row_id").to_csv(path,index=False)
        rows.append({"array_index":idx,"curve_key":key,"target_ci":float(sub.target_ci.iloc[0]),"n_points":len(sub),"subset_csv":str(path)})
    pd.DataFrame(rows).to_csv(root/"targets/positive_curve_manifest.csv",index=False)
    neu=neu.sort_values("blumen_row_id").reset_index(drop=True); neu["array_index"]=np.arange(len(neu),dtype=int); neu.to_csv(root/"targets/neutral_point_manifest.csv",index=False)
    meta={"status":"PASS","n_positive_curves":len(rows),"n_positive_points":len(pos),"n_neutral_points":len(neu),"work_root":str(root)}
    (root/"targets/target_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print("=== BLUMEN SLURM TARGETS ==="); print(f"Positive curves : {len(rows)}"); print(f"Positive points : {len(pos)}"); print(f"Neutral points  : {len(neu)}"); print("TARGET STATUS   : PASS")
if __name__=="__main__": raise SystemExit(main())
