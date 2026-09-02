#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from plots.scripts.pinn_subsonic.utils_asset_common import configure_plotting, first_column, save_figure, write_json


def discover_csvs(root: Path, patterns: list[str]) -> list[Path]:
    out=[]
    for p in root.rglob('*.csv'):
        s=str(p).lower()
        if any(re.search(pattern,s) for pattern in patterns): out.append(p)
    return sorted(set(out))


def combine(paths: list[Path]) -> pd.DataFrame:
    frames=[]
    for p in paths:
        try: f=pd.read_csv(p)
        except Exception: continue
        f['_source_file']=str(p); frames.append(f)
    return pd.concat(frames,ignore_index=True,sort=False) if frames else pd.DataFrame()


def plot_mapping_audit(frame: pd.DataFrame, out: Path, report: dict) -> None:
    if frame.empty:
        report['missing'].append('No long-wave mapping-sweep CSV found.')
        return
    mcol=first_column(frame,['mapping_scale','map_scale','mapping','scale'],required=False)
    xicol=first_column(frame,['xi_max','xi','ximax'],required=False)
    ncol=first_column(frame,['N','n_points','grid_N'],required=False)
    errcol=first_column(frame,['ci_gep_abs_err','ci_abs_err','ci_error_abs','ci_err','abs_error'],required=False)
    ovcol=first_column(frame,['p_overlap','overlap','modal_overlap'],required=False)
    idcol=first_column(frame,['point_id','case_id','chart_id'],required=False)
    if mcol is None or errcol is None:
        report['missing'].append('Long-wave CSVs were found but lack mapping_scale and absolute-error columns.')
        return
    f=frame.copy(); f[mcol]=pd.to_numeric(f[mcol],errors='coerce'); f[errcol]=pd.to_numeric(f[errcol],errors='coerce')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    groups=[('',f)] if idcol is None else list(f.groupby(idcol))
    for name,g in groups:
        g=g.sort_values(mcol); label=str(name) if name!='' else None
        axes[0].plot(g[mcol],g[errcol],marker='o',lw=1,label=label)
        if ovcol is not None:
            axes[1].plot(g[mcol],pd.to_numeric(g[ovcol],errors='coerce'),marker='o',lw=1,label=label)
    axes[0].set_yscale('log'); axes[0].set(xlabel='mapping scale',ylabel=r'$|c_i-c_i^{ref}|$',title='Long-wave eigenvalue convergence')
    axes[1].set(xlabel='mapping scale',ylabel='pressure-mode overlap',title='Long-wave modal convergence')
    if len(groups)<=12 and idcol is not None: axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    fig.tight_layout(); save_figure(fig,out/'SuppFig08_longwave_mapping_audit')

    policy_cols=[c for c in [idcol,mcol,xicol,ncol,errcol,ovcol,'_source_file'] if c is not None]
    f[policy_cols].to_csv(out.parent/'tables'/'longwave_mapping_audit.csv',index=False)


def plot_N_audit(frame: pd.DataFrame, out: Path, report: dict) -> None:
    if frame.empty:
        report['missing'].append('No N=201/301/401 convergence CSV found.')
        return
    ncol=first_column(frame,['N','n_points','grid_N'],required=False)
    errcol=first_column(frame,['ci_gep_abs_err','ci_abs_err','ci_error_abs','ci_err','abs_error'],required=False)
    ovcol=first_column(frame,['p_overlap','overlap','modal_overlap'],required=False)
    idcol=first_column(frame,['point_id','case_id','chart_id'],required=False)
    if ncol is None or errcol is None:
        report['missing'].append('Convergence CSVs were found but lack N and absolute-error columns.')
        return
    f=frame.copy(); f[ncol]=pd.to_numeric(f[ncol],errors='coerce');f[errcol]=pd.to_numeric(f[errcol],errors='coerce')
    f=f[f[ncol].isin([201,301,401,501])]
    if f.empty:
        report['missing'].append('No N=201/301/401 rows found in convergence CSVs.')
        return
    fig,axes=plt.subplots(1,2,figsize=(11,4.8)); groups=[('',f)] if idcol is None else list(f.groupby(idcol))
    for name,g in groups:
        g=g.sort_values(ncol); label=str(name) if name!='' else None
        axes[0].plot(g[ncol],g[errcol],marker='o',label=label)
        if ovcol is not None: axes[1].plot(g[ncol],pd.to_numeric(g[ovcol],errors='coerce'),marker='o',label=label)
    axes[0].set_yscale('log');axes[0].set(xlabel='GEP grid size N',ylabel=r'$|c_i-c_i^{ref}|$',title='Eigenvalue convergence in N')
    axes[1].set(xlabel='GEP grid size N',ylabel='pressure-mode overlap',title='Modal convergence in N')
    if len(groups)<=12 and idcol is not None: axes[0].legend(fontsize=7);axes[1].legend(fontsize=7)
    fig.tight_layout();save_figure(fig,out/'SuppFig07_GEP_N_convergence')
    f.to_csv(out.parent/'tables'/'GEP_N_convergence.csv',index=False)


def plot_continuations(paths: list[Path], out: Path, report: dict) -> None:
    if not paths:
        report['missing'].append('No continuation_paths.csv files found.')
        return
    rows=[]
    for path in paths:
        try:f=pd.read_csv(path)
        except Exception:continue
        f['_point_id']=path.parent.name;f['_source_file']=str(path);rows.append(f)
    if not rows:
        report['missing'].append('Continuation path files could not be read.');return
    f=pd.concat(rows,ignore_index=True,sort=False)
    ecol=first_column(f,['eta','eta_step','eta_value']); ccol=first_column(f,['gep_ci','ci','ci_selected','eigenvalue_imag','imag_c']); pathcol=first_column(f,['path','direction','continuation_direction'],required=False); ovcol=first_column(f,['adjacent_overlap','overlap_previous','step_overlap'],required=False)
    selected=[]
    for pid,g in f.groupby('_point_id'):
        spread=pd.to_numeric(g[ccol],errors='coerce').max()-pd.to_numeric(g[ccol],errors='coerce').min(); selected.append((spread,pid))
    selected=[pid for _,pid in sorted(selected,reverse=True)[:8]]
    fig,axes=plt.subplots(2,4,figsize=(14,7),squeeze=False)
    for ax,pid in zip(axes.ravel(),selected):
        g=f[f._point_id.eq(pid)].copy(); groups=[('path',g)] if pathcol is None else list(g.groupby(pathcol))
        for name,h in groups:
            ax.plot(pd.to_numeric(h[ecol],errors='coerce'),pd.to_numeric(h[ccol],errors='coerce'),marker='o',ms=2.5,lw=1,label=str(name))
        ax.set(title=pid,xlabel=r'$\eta$',ylabel=r'$c_i$');ax.legend(fontsize=6)
    fig.suptitle('Near-neutral and branch-continuation audit');fig.tight_layout();save_figure(fig,out/'SuppFig09_near_neutral_continuation_audit')
    if ovcol is not None:
        fig,ax=plt.subplots(figsize=(8,5.5))
        for pid in selected:
            g=f[f._point_id.eq(pid)]
            ax.plot(pd.to_numeric(g[ecol],errors='coerce'),1-pd.to_numeric(g[ovcol],errors='coerce'),marker='.',lw=.9,label=pid)
        ax.set_yscale('log');ax.set(xlabel=r'$\eta$',ylabel='adjacent overlap defect',title='Continuation modal-overlap defect');ax.legend(fontsize=6,ncol=2)
        save_figure(fig,out/'SuppFig10_bidirectional_continuation_overlap')
    f.to_csv(out.parent/'tables'/'continuation_paths_all.csv',index=False)


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--repo-root',default='.');p.add_argument('--atlas-root',default='assets/pinn_subsonic/local_atlas_v1');p.add_argument('--output-dir',default='assets/pinn_subsonic/local_atlas_v1/publication_assets_scientific_v2');a=p.parse_args()
    configure_plotting();repo=Path(a.repo_root).resolve();atlas=repo/a.atlas_root;out=repo/a.output_dir;figdir=out/'figures';(out/'tables').mkdir(parents=True,exist_ok=True);figdir.mkdir(parents=True,exist_ok=True)
    report={'missing':[],'sources':{}}
    map_paths=discover_csvs(atlas,[r'map(5|10|15|20)',r'longwave.*(audit|sweep|mapping)',r'mapping.*(audit|sweep)'])
    n_paths=discover_csvs(atlas,[r'convergence',r'n201',r'n301',r'n401'])
    cont_paths=list(atlas.rglob('continuation_paths.csv'))
    report['sources']={'mapping':[str(p) for p in map_paths],'N_convergence':[str(p) for p in n_paths],'continuation_count':len(cont_paths)}
    plot_mapping_audit(combine(map_paths),figdir,report);plot_N_audit(combine(n_paths),figdir,report);plot_continuations(cont_paths,figdir,report)
    write_json(out/'numerical_audit_report.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
