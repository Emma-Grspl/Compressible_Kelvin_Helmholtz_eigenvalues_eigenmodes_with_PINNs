#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil, tarfile
from datetime import datetime,timezone
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load(paths):
    fs=[]
    for p in paths:
        try: f=pd.read_csv(p)
        except Exception: continue
        if not f.empty: fs.append(f)
    return pd.concat(fs,ignore_index=True,sort=False) if fs else pd.DataFrame()

def nums(f,cols):
    o=f.copy()
    for c in cols:
        if c in o: o[c]=pd.to_numeric(o[c],errors="coerce")
    return o

def overlay(d,out):
    fig,ax=plt.subplots(figsize=(9,7),constrained_layout=True); pos=d[d.target_ci>0].copy(); groups=list(pos.groupby("curve_key",sort=False)); levels=[float(g.target_ci.iloc[0]) for _,g in groups]; cm=plt.get_cmap("viridis"); colors={key:cm(.15+.75*i/max(1,len(groups)-1)) for i,(key,_) in enumerate(groups)}
    for key,s in groups:
        s=s.sort_values("blumen_row_id" if "blumen_row_id" in s.columns else "Mach"); x=float(s.target_ci.iloc[0]); col=colors[key]
        ax.plot(s.alpha_blumen,s.Mach,"--",lw=1,color=col); g=s.status.astype(str).str.startswith("converged"); ax.plot(s.loc[g,"alpha_classical"],s.loc[g,"Mach"],"-",lw=1.8,color=col,label=rf"$c_i={x:.4g}$")
    n=d[np.isclose(d.target_ci,0,atol=1e-14)].sort_values("Mach")
    if not n.empty:
        ax.plot(n.alpha_blumen,n.Mach,"--",lw=1.2,color="black",label=r"Blumen $c_i=0$"); g=n.status.astype(str).eq("converged_neutral_bracket"); ax.plot(n.loc[g,"alpha_classical"],n.loc[g,"Mach"],"-",lw=2,color="black",label=r"Classical $c_i=0$")
        if {"neutral_alpha_lower","neutral_alpha_upper"}.issubset(n.columns):
            lo=pd.to_numeric(n.loc[g,"neutral_alpha_lower"],errors="coerce"); up=pd.to_numeric(n.loc[g,"neutral_alpha_upper"],errors="coerce"); mm=n.loc[g,"Mach"]; ok=np.isfinite(lo)&np.isfinite(up); ax.fill_betweenx(mm.loc[ok],lo.loc[ok],up.loc[ok],color="black",alpha=.12)
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"Mach number $M$"); ax.set_title("Blumen digitization and reconstructed classical isolines\ndashed: Blumen; solid: classical"); ax.grid(True,alpha=.22); ax.legend(frameon=False,fontsize=8,ncols=2); fig.savefig(out,bbox_inches="tight"); plt.close(fig)

def heat(d,out):
    good=d[d.status.astype(str).isin(("converged_center","converged_bracket_endpoint","converged_root","converged_neutral_bracket"))].copy(); good=nums(good,("Mach","alpha_blumen","delta_alpha")); good=good.dropna(subset=["Mach","alpha_blumen","delta_alpha"])
    if good.empty: raise RuntimeError("No converged points for heatmap")
    lim=float(np.nanmax(np.abs(good.delta_alpha))) or 1e-12; fig,ax=plt.subplots(figsize=(8.8,6.8),constrained_layout=True); sc=ax.scatter(good.alpha_blumen,good.Mach,c=good.delta_alpha,cmap="coolwarm",vmin=-lim,vmax=lim,s=34,edgecolors="black",linewidths=.25); fig.colorbar(sc,ax=ax,label=r"$\Delta\alpha=\alpha_{classique}-\alpha_{Blumen}$"); ax.set_xlabel(r"$\alpha_{Blumen}$"); ax.set_ylabel(r"Mach number $M$"); ax.set_title("Geometric error of the Blumen isolines"); ax.grid(True,alpha=.2); fig.savefig(out,bbox_inches="tight"); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--work-root",type=Path,required=True); p.add_argument("--asset-root",type=Path,required=True); p.add_argument("--overwrite",action="store_true"); a=p.parse_args(); w=a.work_root.resolve(); r=a.asset_root.resolve()
    if r.exists() and a.overwrite: shutil.rmtree(r)
    r.mkdir(parents=True,exist_ok=True)
    pos=load(sorted(w.glob("positive/curve_*/blumen_true_classical_isolines.csv"))); neu=load(sorted(w.glob("neutral/point_*/neutral_result.csv")))
    if pos.empty: raise RuntimeError("No positive results found")
    if neu.empty: raise RuntimeError("No neutral results found")
    pos=nums(pos,("Mach","target_ci","alpha_blumen","alpha_classical","classical_cr","classical_ci","residual_norm","delta_alpha")); neu=nums(neu,("Mach","target_ci","alpha_blumen","alpha_classical","classical_cr","classical_ci","residual_norm","delta_alpha","neutral_alpha_lower","neutral_alpha_upper")); all=pd.concat([pos,neu],ignore_index=True,sort=False).sort_values(["target_ci","curve_key","Mach","alpha_blumen"])
    pos.to_csv(r/"blumen_positive_true_classical_isolines.csv",index=False); neu.to_csv(r/"blumen_neutral_true_classical_line.csv",index=False); all.to_csv(r/"blumen_true_classical_all_isolines.csv",index=False)
    overlay(all,r/"blumen_true_classical_isolines_overlay.pdf"); heat(all,r/"blumen_true_classical_delta_alpha_heatmap.pdf")
    summary=all.assign(converged=all.status.astype(str).str.startswith("converged")).groupby(["target_ci","curve_key"],dropna=False).agg(n_points=("Mach","size"),n_converged=("converged","sum"),mean_abs_delta_alpha=("delta_alpha",lambda x:float(np.nanmean(np.abs(x)))),max_abs_delta_alpha=("delta_alpha",lambda x:float(np.nanmax(np.abs(x))))).reset_index(); summary.to_csv(r/"blumen_true_classical_isolines_summary.csv",index=False)
    meta={"generated_at":datetime.now(timezone.utc).isoformat(),"n_positive_rows":len(pos),"n_neutral_rows":len(neu),"n_total_rows":len(all),"n_converged":int(all.status.astype(str).str.startswith("converged").sum()),"status":"PASS"}; (r/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    arch=r.with_suffix(".tar.gz");
    with tarfile.open(arch,"w:gz") as h: h.add(r,arcname=r.name)
    print("=== BLUMEN TRUE-ISOLINE ASSETS ==="); print(f"Positive rows : {len(pos)}"); print(f"Neutral rows  : {len(neu)}"); print(f"Converged     : {meta['n_converged']}"); print(f"Overlay       : {r/'blumen_true_classical_isolines_overlay.pdf'}"); print(f"Heatmap       : {r/'blumen_true_classical_delta_alpha_heatmap.pdf'}"); print(f"Archive       : {arch}"); print("ASSET STATUS  : PASS")
if __name__=="__main__": raise SystemExit(main())
