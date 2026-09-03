#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, subprocess, sys, uuid
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
@dataclass
class R:
    accepted: bool; alpha: float; cr: float; ci: float; residual_norm: float; workdir: str; message: str

def nearest(ref,M,a):
    w=ref.copy(); w["d"]=((w.Mach-M)/.05)**2+((w.alpha-a)/.02)**2; r=w.sort_values("d").iloc[0]; return float(r.cr),max(float(r.ci),1e-5)

def solve(repo,script,runs,ref,M,a,seed,py):
    if seed is None: seed=nearest(ref,M,a)
    cr,ci=seed; wd=runs/f"a_{a:.10f}_{uuid.uuid4().hex[:8]}"; wd.mkdir(parents=True,exist_ok=True)
    cmd=[py,"-u",str(script),"--repo",str(repo),"--Mach",f"{M:.17g}","--alpha",f"{a:.17g}","--seed-cr",f"{cr:.17g}","--seed-ci",f"{ci:.17g}","--reference-cr",f"{cr:.17g}","--reference-ci",f"{ci:.17g}","--cr-lower",f"{max(-1,cr-.2):.17g}","--cr-upper",f"{min(1.5,cr+.2):.17g}","--ci-lower","0","--ci-upper",f"{max(.08,ci+.08):.17g}","--Ly",("2000" if M>=1.8 else "500"),"--matching-y","1.0","--max-step","0.25","--rtol","1e-10","--atol","1e-12","--method","DOP853","--optimizer-xtol","1e-11","--optimizer-ftol","1e-11","--output-dir",str(wd)]
    q=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False); sm=wd/"summary.json"
    if q.returncode!=0 or not sm.is_file(): return R(False,a,math.nan,math.nan,math.nan,str(wd),f"returncode={q.returncode}")
    d=json.loads(sm.read_text()); e=d.get("optimized_eigenvalue",{}); t=d.get("root_test",{})
    cr=float(e.get("cr",math.nan)); ci=float(e.get("ci",math.nan)); res=float(t.get("residual_norm",math.nan)); ok=bool(t.get("accepted",False)) and np.isfinite(cr) and np.isfinite(ci) and ci>0 and np.isfinite(res) and res<=1e-8
    return R(ok,a,cr,ci,res,str(wd),"accepted" if ok else "not_accepted")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--array-index",type=int,required=True); p.add_argument("--reference-csv",type=Path,required=True); p.add_argument("--search-script",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--python-exe",default=sys.executable); p.add_argument("--scan-halfwidth",type=float,default=.08); p.add_argument("--scan-step",type=float,default=.005); p.add_argument("--bracket-tolerance",type=float,default=5e-5); a=p.parse_args()
    m=pd.read_csv(a.manifest); row=m[pd.to_numeric(m.array_index)==a.array_index]
    if len(row)!=1: raise RuntimeError(f"Expected one target for {a.array_index}, got {len(row)}")
    row=row.iloc[0]; M=float(row.Mach); ab=float(row.alpha)
    ref=pd.read_csv(a.reference_csv)
    for c in ("Mach","alpha","cr","ci"): ref[c]=pd.to_numeric(ref[c],errors="coerce")
    ref=ref.dropna(subset=["Mach","alpha","cr","ci"])
    root=a.output_root.resolve()/f"point_{a.array_index:03d}"; runs=root/"_solver_runs"; runs.mkdir(parents=True,exist_ok=True)
    lo=max(.03,ab-a.scan_halfwidth); hi=ab+a.scan_halfwidth; grid=np.arange(lo,hi+.5*a.scan_step,a.scan_step)
    last=None; firstfail=None; seed=None; calls=0
    for x in grid:
        rr=solve(a.repo.resolve(),a.search_script.resolve(),runs,ref,M,float(x),seed,a.python_exe); calls+=1
        if rr.accepted: last=rr; seed=(rr.cr,rr.ci)
        elif last is not None: firstfail=rr; break
    if last is None or firstfail is None:
        out={"status":"no_neutral_bracket","Mach":M,"target_ci":0.0,"alpha_blumen":ab,"alpha_classical":math.nan,"delta_alpha":math.nan,"neutral_alpha_lower":math.nan,"neutral_alpha_upper":math.nan,"classical_cr":math.nan,"classical_ci":math.nan,"residual_norm":math.nan,"n_solver_calls":calls}
    else:
        upper=firstfail.alpha
        for _ in range(24):
            if upper-last.alpha<=a.bracket_tolerance: break
            mid=.5*(last.alpha+upper); rr=solve(a.repo.resolve(),a.search_script.resolve(),runs,ref,M,mid,(last.cr,max(last.ci,1e-6)),a.python_exe); calls+=1
            if rr.accepted: last=rr
            else: upper=mid
        est=.5*(last.alpha+upper)
        out={"status":"converged_neutral_bracket","Mach":M,"target_ci":0.0,"alpha_blumen":ab,"alpha_classical":est,"delta_alpha":est-ab,"neutral_alpha_lower":last.alpha,"neutral_alpha_upper":upper,"neutral_alpha_uncertainty":.5*(upper-last.alpha),"classical_cr":last.cr,"classical_ci":last.ci,"residual_norm":last.residual_norm,"n_solver_calls":calls}
    out.update({"array_index":a.array_index,"blumen_row_id":int(row.blumen_row_id),"curve_key":str(row.curve_key),"curve_id_key":str(getattr(row,"curve_id_key","")),"curve_label":str(getattr(row,"curve_label",""))})
    root.mkdir(parents=True,exist_ok=True); pd.DataFrame([out]).to_csv(root/"neutral_result.csv",index=False); (root/"metadata.json").write_text(json.dumps(out,indent=2)+"\n")
    print("=== BLUMEN NEUTRAL POINT ==="); print(f"Mach            : {M}"); print(f"Blumen alpha    : {ab}"); print(f"Status          : {out['status']}"); print(f"Classical alpha : {out['alpha_classical']}")
if __name__=="__main__": raise SystemExit(main())
