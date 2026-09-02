#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plots.scripts.pinn_subsonic.utils_asset_common import (
    canonicalize_release,
    configure_plotting,
    find_named,
    first_column,
    numeric,
    resolve_release_frame,
    safe_rel_error,
    save_figure,
    scatter_map,
    write_json,
)


def load_rectangles(atlas_root: Path) -> pd.DataFrame:
    candidates = [
        atlas_root / 'publication_assets_fullrect_v1' / 'data' / 'atlas_catalog.csv',
        atlas_root / 'release_fullrect_v1' / 'atlas_catalog.csv',
        atlas_root / 'atlas_catalog.csv',
    ]
    frames = []
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            frames.append(frame)
            break
    if not frames:
        for name, family in [
            ('atlas_manifest.csv','core'),
            ('atlas_extension_plan_M005_098_eta005_098.csv','extension'),
            ('atlas_fullrect_plan_M002_098_eta002_098.csv','boundary_completion'),
        ]:
            path = atlas_root / name
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame['construction_family'] = family
            frames.append(frame)
    if not frames:
        raise FileNotFoundError('No atlas rectangle/catalog file found.')

    frame = pd.concat(frames, ignore_index=True, sort=False)
    aliases = {
        'chart_id': ['chart_id','atlas_id','name'],
        'M_min': ['M_min','mach_min','Mach_min'],
        'M_max': ['M_max','mach_max','Mach_max'],
        'eta_min': ['eta_min','Eta_min'],
        'eta_max': ['eta_max','Eta_max'],
    }
    out = pd.DataFrame()
    for dest, names in aliases.items():
        col = first_column(frame, names)
        out[dest] = frame[col]
    for dest, names, default in [
        ('priority',['priority'],0.0),
        ('chart_status',['chart_status','status','pipeline_status'],'validated'),
        ('pipeline_status',['pipeline_status','chart_status','status'],'PINN seed + GEP'),
        ('gep_regime',['gep_regime','regime'],'standard_N301'),
        ('construction_family',['construction_family','source'],'unknown'),
    ]:
        col = first_column(frame, names, required=False)
        out[dest] = default if col is None else frame[col]
    # Canonical scientific labels for legacy atlas catalogs whose status fields are empty.
    for text_col in ["chart_status", "pipeline_status", "gep_regime", "construction_family"]:
        values = out[text_col].astype("string").str.strip()
        missing = values.fillna("").str.lower().isin(
            ["", "nan", "none", "null", "<na>"]
        )
        out[text_col] = values.mask(missing, pd.NA)

    chart_id_upper = out["chart_id"].astype(str).str.upper()

    is_neutral3 = chart_id_upper.str.contains(
        "NEUTRAL3",
        regex=False,
    )

    is_extreme_longwave = (
        chart_id_upper.str.contains("ETAEDGE_HM2B", regex=False)
        | chart_id_upper.str.contains("VLOW_EXTREME", regex=False)
    )

    is_longwave = (
        chart_id_upper.str.contains("LMEDGE_VLOW", regex=False)
        | chart_id_upper.str.contains("ETAEDGE_", regex=False)
        | chart_id_upper.str.contains("ULTRALOW", regex=False)
    ) & ~is_neutral3 & ~is_extreme_longwave

    out["chart_status"] = "ci_only_validated_with_gep"
    out.loc[
        is_longwave | is_extreme_longwave,
        "chart_status",
    ] = "longwave_gep"
    out.loc[
        is_neutral3,
        "chart_status",
    ] = "near_neutral_continuation"

    out["pipeline_status"] = "PINN seed + GEP standard"
    out.loc[
        is_longwave,
        "pipeline_status",
    ] = "PINN seed + GEP long-wave"
    out.loc[
        is_extreme_longwave,
        "pipeline_status",
    ] = "PINN seed + GEP extreme long-wave"
    out.loc[
        is_neutral3,
        "pipeline_status",
    ] = "PINN seed + GEP near-neutral N401 + modal continuation"

    out["gep_regime"] = "standard_N301"
    out.loc[
        is_longwave,
        "gep_regime",
    ] = "longwave_map10_N301"
    out.loc[
        is_extreme_longwave,
        "gep_regime",
    ] = "extreme_longwave_map20_N301"
    out.loc[
        is_neutral3,
        "gep_regime",
    ] = "near_neutral_N401"

    out["construction_family"] = (
        out["construction_family"]
        .fillna("unknown")
        .astype(str)
    )

    for col in ['M_min','M_max','eta_min','eta_max','priority']:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out['area'] = (out['M_max']-out['M_min']) * (out['eta_max']-out['eta_min'])
    return out.dropna(subset=['M_min','M_max','eta_min','eta_max']).drop_duplicates('chart_id', keep='last')


def plot_atlas_status_footprints(rectangles: pd.DataFrame, stem: Path) -> None:
    from matplotlib.patches import Rectangle, Patch
    statuses=sorted(rectangles.chart_status.fillna('unknown').astype(str).unique())
    cmap=plt.get_cmap('tab10',max(1,len(statuses))); colors={s:cmap(i) for i,s in enumerate(statuses)}
    fig,ax=plt.subplots(figsize=(9.2,7.2))
    for _,r in rectangles.iterrows():
        status=str(r.chart_status); rect=Rectangle((r.M_min,r.eta_min),r.M_max-r.M_min,r.eta_max-r.eta_min,fill=False,lw=1.2,ec=colors[status])
        ax.add_patch(rect)
        if (r.M_max-r.M_min)*(r.eta_max-r.eta_min) < .03:
            ax.text((r.M_min+r.M_max)/2,(r.eta_min+r.eta_max)/2,str(r.chart_id),fontsize=5.5,ha='center',va='center',clip_on=True)
    ax.set(xlim=(.02,.98),ylim=(.02,.98),xlabel=r'Mach number $M$',ylabel=r'Scaled wavenumber $\eta$',title='Local PINN atlas charts by final scientific status')
    ax.legend(handles=[Patch(facecolor='none',edgecolor=colors[s],label=s) for s in statuses],loc='center left',bbox_to_anchor=(1.01,.5))
    save_figure(fig,stem)


def plot_coverage_grid(grid: pd.DataFrame, stem: Path) -> None:
    Mv=np.sort(grid.Mach.unique());Ev=np.sort(grid.eta.unique());arr=grid.pivot(index='eta',columns='Mach',values='coverage_count').reindex(index=Ev,columns=Mv).to_numpy()
    fig,ax=plt.subplots(figsize=(8.0,6.7));im=ax.imshow(arr,origin='lower',extent=[Mv.min(),Mv.max(),Ev.min(),Ev.max()],aspect='auto',interpolation='nearest',cmap='viridis',vmin=0,vmax=max(1,int(np.nanmax(arr))))
    ax.set(xlabel=r'Mach number $M$',ylabel=r'Scaled wavenumber $\eta$',title='Atlas coverage multiplicity');cb=fig.colorbar(im,ax=ax);cb.set_label('number of covering charts')
    save_figure(fig,stem)

def build_assignment(rectangles: pd.DataFrame, step: float=0.0025) -> pd.DataFrame:
    ms = np.round(np.arange(0.02, 0.9800001, step), 7)
    es = np.round(np.arange(0.02, 0.9800001, step), 7)
    M, E = np.meshgrid(ms, es)
    flat_m = M.ravel(); flat_e = E.ravel()
    n = len(flat_m)
    coverage = np.zeros(n, dtype=np.int16)
    selected = np.full(n, '', dtype=object)
    status = np.full(n, '', dtype=object)
    pipeline = np.full(n, '', dtype=object)
    regime = np.full(n, '', dtype=object)
    best_score = np.full(n, -np.inf)

    for _, r in rectangles.iterrows():
        inside = (
            (flat_m >= r.M_min-1e-12) & (flat_m <= r.M_max+1e-12) &
            (flat_e >= r.eta_min-1e-12) & (flat_e <= r.eta_max+1e-12)
        )
        coverage[inside] += 1
        # Priority first, then prefer the smallest local chart.
        score = float(r.priority) - 1e-3 * float(r.area)
        choose = inside & (score > best_score)
        best_score[choose] = score
        selected[choose] = str(r.chart_id)
        status[choose] = str(r.chart_status)
        pipeline[choose] = str(r.pipeline_status)
        regime[choose] = str(r.gep_regime)

    return pd.DataFrame({
        'Mach': flat_m,
        'eta': flat_e,
        'alpha': flat_e*np.sqrt(np.maximum(0.0,1-flat_m**2)),
        'coverage_count': coverage,
        'selected_chart': selected,
        'chart_status': status,
        'pipeline_mode': pipeline,
        'gep_regime': regime,
    })


def plot_atlas_multipanel(rectangles: pd.DataFrame, grid: pd.DataFrame, stem: Path) -> None:
    from matplotlib.patches import Rectangle, Patch
    fig,axes=plt.subplots(2,2,figsize=(12.5,10.0))
    statuses=sorted(rectangles.chart_status.fillna('unknown').astype(str).unique());cmap_s=plt.get_cmap('tab10',max(1,len(statuses)));colors={s:cmap_s(i) for i,s in enumerate(statuses)}
    ax=axes[0,0]
    for _,r in rectangles.iterrows(): ax.add_patch(Rectangle((r.M_min,r.eta_min),r.M_max-r.M_min,r.eta_max-r.eta_min,fill=False,lw=1.0,ec=colors[str(r.chart_status)]))
    ax.set(xlim=(.02,.98),ylim=(.02,.98),xlabel=r'$M$',ylabel=r'$\eta$',title='(a) charts by scientific status')
    ax.legend(handles=[Patch(facecolor='none',edgecolor=colors[v],label=v) for v in statuses],fontsize=6,loc='upper left')

    Mv=np.sort(grid.Mach.unique());Ev=np.sort(grid.eta.unique());cov=grid.pivot(index='eta',columns='Mach',values='coverage_count').reindex(index=Ev,columns=Mv).to_numpy()
    ax=axes[0,1];im=ax.imshow(cov,origin='lower',extent=[Mv.min(),Mv.max(),Ev.min(),Ev.max()],aspect='auto',interpolation='nearest',cmap='viridis',vmin=0,vmax=max(1,int(np.nanmax(cov))));fig.colorbar(im,ax=ax,label='covering charts');ax.set(xlabel=r'$M$',ylabel=r'$\eta$',title='(b) coverage multiplicity')

    for ax,column,title in [(axes[1,0],'selected_chart','(c) operational chart assignment'),(axes[1,1],'pipeline_mode','(d) operational inference pipeline')]:
        vals=sorted(v for v in grid[column].fillna('').unique() if v);codes={v:i for i,v in enumerate(vals)};pivot=grid.pivot(index='eta',columns='Mach',values=column).reindex(index=Ev,columns=Mv);arr=pivot.applymap(lambda x:codes.get(str(x),-1)).to_numpy();cmap=plt.get_cmap('tab20',max(1,len(vals)));ax.imshow(arr,origin='lower',extent=[Mv.min(),Mv.max(),Ev.min(),Ev.max()],aspect='auto',interpolation='nearest',cmap=cmap,vmin=-.5,vmax=max(.5,len(vals)-.5));ax.set(xlabel=r'$M$',ylabel=r'$\eta$',title=title)
        if column=='pipeline_mode': ax.legend(handles=[Patch(facecolor=cmap(codes[v]),label=v) for v in vals],fontsize=6,loc='upper left')
    fig.suptitle('Architecture and operational coverage of the local PINN atlas');fig.tight_layout();save_figure(fig,stem)

def plot_categorical_grid(grid: pd.DataFrame, column: str, title: str, stem: Path) -> None:
    values = sorted(v for v in grid[column].fillna('').unique() if v)
    code = {v:i for i,v in enumerate(values)}
    Mv = np.sort(grid.Mach.unique()); Ev = np.sort(grid.eta.unique())
    pivot = grid.pivot(index='eta',columns='Mach',values=column).reindex(index=Ev,columns=Mv)
    arr = pivot.applymap(lambda x: code.get(str(x), -1)).to_numpy()
    cmap = plt.get_cmap('tab20', max(1,len(values)))
    fig, ax = plt.subplots(figsize=(8.2,7.0))
    ax.imshow(arr,origin='lower',extent=[Mv.min(),Mv.max(),Ev.min(),Ev.max()],aspect='auto',interpolation='nearest',cmap=cmap,vmin=-0.5,vmax=max(0.5,len(values)-0.5))
    ax.set_xlabel(r'Mach number $M$'); ax.set_ylabel(r'Scaled wavenumber $\eta$'); ax.set_title(title)
    handles = [Patch(facecolor=cmap(code[v]), label=v) for v in values]
    if handles:
        ax.legend(handles=handles,loc='center left',bbox_to_anchor=(1.01,0.5),frameon=True)
    save_figure(fig, stem)


def plot_sparse_supervision(atlas_root: Path, release: pd.DataFrame, stem: Path, report: dict) -> None:
    anchor_rows = []
    anchor_sources = []
    for path in atlas_root.rglob('*anchor*.csv'):
        try:
            f = pd.read_csv(path)
        except Exception:
            continue
        mcol = first_column(f,['Mach','M','mach'],required=False)
        ecol = first_column(f,['eta','Eta'],required=False)
        if mcol is None or ecol is None:
            continue
        temp = pd.DataFrame({'Mach':pd.to_numeric(f[mcol],errors='coerce'),'eta':pd.to_numeric(f[ecol],errors='coerce')}).dropna()
        if temp.empty:
            continue
        temp['source'] = path.parent.name
        anchor_rows.append(temp)
        anchor_sources.append(str(path))
    if anchor_rows:
        anchors = pd.concat(anchor_rows,ignore_index=True).drop_duplicates(['Mach','eta'])
    else:
        # Fallback: points from the deterministic chart-validation table, clearly marked as fallback.
        points_path = atlas_root / 'atlas_fullrect_gep_final_all_points.csv'
        if points_path.exists():
            f = pd.read_csv(points_path)
            mcol = first_column(f,['Mach','M']); ecol=first_column(f,['eta','Eta'])
            anchors = pd.DataFrame({'Mach':pd.to_numeric(f[mcol],errors='coerce'),'eta':pd.to_numeric(f[ecol],errors='coerce')}).dropna().drop_duplicates()
            anchors['source']='chart validation points (anchor CSVs unavailable)'
            report['warnings'].append('No explicit anchor CSV was found; sparse-supervision figure uses chart-validation locations as a fallback.')
        else:
            report['missing'].append('Sparse supervision: no *anchor*.csv and no atlas_fullrect_gep_final_all_points.csv')
            return
    fig, ax = plt.subplots(figsize=(8.0,6.7))
    ax.scatter(anchors.Mach,anchors.eta,s=22,facecolors='none',edgecolors='tab:blue',linewidths=0.8,label=r'$c_i$ supervision/anchor locations')
    ax.scatter(release.Mach,release.eta,s=10,c='0.65',alpha=0.55,label='independent off-grid validation points')
    ax.set(xlim=(0.02,0.98),ylim=(0.02,0.98),xlabel=r'Mach number $M$',ylabel=r'Scaled wavenumber $\eta$',title=r'Sparse spectral supervision and independent validation')
    ax.legend(loc='best')
    save_figure(fig,stem)
    report['anchor_files']=anchor_sources


def parity(ax, ref, pred, label, marker, alpha=0.75):
    valid=np.isfinite(ref)&np.isfinite(pred)
    ax.scatter(ref[valid],pred[valid],s=24,marker=marker,alpha=alpha,label=label,edgecolors='none')


def build_ci_figures(df: pd.DataFrame, out: Path) -> None:
    eps=1e-14
    positive = pd.concat([df.ci_seed_abs_err,df.ci_final_abs_err]).replace(0,np.nan).dropna()
    norm=LogNorm(vmin=max(positive.min(),1e-8),vmax=max(positive.max(),1e-7)) if len(positive) else None

    fig,axes=plt.subplots(2,2,figsize=(11,9))
    scatter_map(axes[0,0],df,'ci_seed_abs_err',norm=norm,cmap='magma',title=r'PINN seed: $|c_i^{PINN}-c_i^{ref}|$',cbar_label='absolute error')
    scatter_map(axes[0,1],df,'ci_final_abs_err',norm=norm,cmap='magma',title=r'PINN-seeded GEP: $|c_i^{final}-c_i^{ref}|$',cbar_label='absolute error')
    scatter_map(axes[1,0],df,'ci_seed_rel_err',norm=LogNorm(vmin=max(df.ci_seed_rel_err.replace(0,np.nan).min(),1e-6),vmax=max(df.ci_seed_rel_err.max(),1e-5)),cmap='viridis',title='PINN seed relative error',cbar_label='relative error')
    scatter_map(axes[1,1],df,'ci_final_rel_err',norm=LogNorm(vmin=max(df.ci_final_rel_err.replace(0,np.nan).min(),1e-6),vmax=max(df.ci_final_rel_err.max(),1e-5)),cmap='viridis',title='Final GEP relative error',cbar_label='relative error')
    fig.suptitle(r'Off-grid growth-rate validation: direct PINN seed and GEP refinement')
    fig.tight_layout()
    save_figure(fig,out/'Fig04_ci_direct_and_errors')

    fig,axes=plt.subplots(1,2,figsize=(11,5))
    parity(axes[0],df.ci_ref.to_numpy(),df.ci_seed.to_numpy(),'PINN seed','o')
    parity(axes[0],df.ci_ref.to_numpy(),df.ci_final.to_numpy(),'PINN-seeded GEP','x')
    lo=np.nanmin([df.ci_ref.min(),df.ci_seed.min(),df.ci_final.min()]); hi=np.nanmax([df.ci_ref.max(),df.ci_seed.max(),df.ci_final.max()])
    axes[0].plot([lo,hi],[lo,hi],'k--',lw=1,label='$y=x$')
    axes[0].set(xlabel=r'Reference $c_i$',ylabel=r'Predicted/refined $c_i$',title='Parity plot'); axes[0].legend()
    gain=np.clip(df.gep_gain.replace([np.inf,-np.inf],np.nan),1e-2,1e5)
    scatter_map(axes[1],df.assign(gep_gain=gain),'gep_gain',norm=LogNorm(vmin=max(gain.dropna().min(),1e-2),vmax=max(gain.dropna().max(),1.0)),cmap='cividis',title='Error-reduction factor from GEP',cbar_label=r'$|e_{PINN}|/|e_{GEP}|$')
    fig.tight_layout(); save_figure(fig,out/'Fig05_ci_gep_parity_and_gain')

    fig,axes=plt.subplots(2,2,figsize=(11,8.5))
    bins=np.geomspace(max(min(df.ci_seed_abs_err.replace(0,np.nan).min(),df.ci_final_abs_err.replace(0,np.nan).min()),1e-8),max(df.ci_seed_abs_err.max(),df.ci_final_abs_err.max())*1.05,32)
    axes[0,0].hist(df.ci_seed_abs_err.clip(lower=bins[0]),bins=bins,alpha=.7,label='PINN seed')
    axes[0,0].hist(df.ci_final_abs_err.clip(lower=bins[0]),bins=bins,alpha=.7,label='GEP final')
    axes[0,0].set_xscale('log'); axes[0,0].set(xlabel='absolute error',ylabel='count',title='Absolute-error distribution'); axes[0,0].legend()
    axes[0,1].scatter(df.ci_seed_abs_err,df.ci_final_abs_err,s=20,alpha=.7); axes[0,1].set_xscale('log');axes[0,1].set_yscale('log'); diag_lo=max(min(df.ci_seed_abs_err.replace(0,np.nan).min(),df.ci_final_abs_err.replace(0,np.nan).min()),1e-10); diag_hi=max(df.ci_seed_abs_err.max(),df.ci_final_abs_err.max()); axes[0,1].plot([diag_lo,diag_hi],[diag_lo,diag_hi],'k--',lw=1);axes[0,1].set(xlabel='PINN seed absolute error',ylabel='GEP final absolute error',title='Pointwise GEP improvement')
    for ax,col,title in [(axes[1,0],'sample_group','Validation strata'),(axes[1,1],'chart_id','Selected charts')]:
        counts=df[col].replace('',np.nan).value_counts().head(12)
        counts.sort_values().plot.barh(ax=ax)
        ax.set(xlabel='number of points',title=title)
    fig.tight_layout(); save_figure(fig,out/'SuppFig_random_offgrid_validation')


def build_modal_heatmaps(df: pd.DataFrame, out: Path, report: dict) -> None:
    cols=[('p_rel_final',r'$p$'),('rho_rel_final',r'$\rho$'),('u_rel_final',r'$u$'),('v_rel_final',r'$v$')]
    fig,axes=plt.subplots(2,2,figsize=(11,9))
    for ax,(col,label) in zip(axes.ravel(),cols):
        vals=df[col].replace(0,np.nan).dropna()
        norm=LogNorm(vmin=max(vals.min(),1e-6),vmax=max(vals.max(),1e-5)) if len(vals) else None
        scatter_map(ax,df,col,norm=norm,cmap='viridis',title=fr'Final relative error: {label}',cbar_label='relative error')
    fig.tight_layout(); save_figure(fig,out/'Fig07_final_modal_error_heatmaps')

    defect=(1-df.p_overlap_final).clip(lower=1e-12)
    fig,ax=plt.subplots(figsize=(7.5,6.5))
    scatter_map(ax,df.assign(overlap_defect=defect),'overlap_defect',norm=LogNorm(vmin=max(defect.min(),1e-10),vmax=max(defect.max(),1e-9)),cmap='magma',title='Final pressure-mode overlap defect',cbar_label=r'$1-|\langle p_{GEP},p_{ref}\rangle|$')
    save_figure(fig,out/'SuppFig_modal_overlap_defect')

    direct_cols=['p_rel_direct','rho_rel_direct','u_rel_direct','v_rel_direct']
    if not df[direct_cols].notna().any().any():
        report['missing'].append('Direct PINN modal errors are absent from the release CSV; C4 requires a direct-mode extraction campaign.')
    else:
        fig,axes=plt.subplots(2,2,figsize=(11,9))
        for ax,col,label in zip(axes.ravel(),direct_cols,[r'$p$',r'$\rho$',r'$u$',r'$v$']):
            vals=df[col].replace(0,np.nan).dropna(); norm=LogNorm(vmin=max(vals.min(),1e-6),vmax=max(vals.max(),1e-5)) if len(vals) else None
            scatter_map(ax,df,col,norm=norm,cmap='plasma',title=fr'Direct PINN relative error: {label}',cbar_label='relative error')
        fig.tight_layout(); save_figure(fig,out/'SuppFig_direct_PINN_modal_error_heatmaps')


def load_blumen(repo: Path, explicit: str|None) -> Path|None:
    if explicit:
        p=Path(explicit); return p if p.exists() else None
    return find_named(repo,['blumen_ci_datasets.csv','blumen_ci_reference.csv'])


def build_blumen(blumen_path: Path|None, df: pd.DataFrame, out: Path, report: dict) -> None:
    if blumen_path is None:
        report['missing'].append('Blumen comparison: blumen_ci_datasets.csv was not found. Supply --blumen-csv PATH.')
        return
    b=pd.read_csv(blumen_path)
    mcol=first_column(b,['Mach','M','mach']); ccol=first_column(b,['ci','c_i','ci_reference','value'])
    acol=first_column(b,['alpha','Alpha'],required=False); ecol=first_column(b,['eta','Eta'],required=False)
    if acol is None and ecol is None:
        report['missing'].append(f'Blumen CSV lacks alpha or eta: {blumen_path}')
        return
    bM=pd.to_numeric(b[mcol],errors='coerce'); bci=pd.to_numeric(b[ccol],errors='coerce')
    if acol is not None:
        bx=pd.to_numeric(b[acol],errors='coerce'); xlabel=r'Physical wavenumber $\alpha$'; x_release=df.alpha; coordinate='alpha'
    else:
        bx=pd.to_numeric(b[ecol],errors='coerce'); xlabel=r'Scaled wavenumber $\eta$'; x_release=df.eta; coordinate='eta'
    valid=np.isfinite(bM)&np.isfinite(bx)&np.isfinite(bci)
    bwork=pd.DataFrame({'Mach':bM[valid],'x':bx[valid],'ci':bci[valid]})
    curve_col=first_column(b,['curve_id','isoline','level','dataset','branch'],required=False)
    fig,ax=plt.subplots(figsize=(8.2,6.6))
    if curve_col:
        temp=b.loc[valid].copy();temp['_M']=bM[valid];temp['_x']=bx[valid];temp['_ci']=bci[valid]
        for name,g in temp.groupby(curve_col):
            g=g.sort_values(['_M','_x']); ax.plot(g._x,g._ci,lw=1.2,alpha=.75,label=f'Blumen {name}')
    else:
        ax.scatter(bwork.x,bwork.ci,s=16,facecolors='none',edgecolors='k',label='Blumen reference')
    ax.scatter(x_release,df.ci_seed,s=15,alpha=.50,label='PINN seed')
    ax.scatter(x_release,df.ci_final,s=15,alpha=.50,label='PINN-seeded GEP')
    ax.set(xlabel=xlabel,ylabel=r'Growth-rate phase speed $c_i$',title='Blumen reference, direct PINN seed and GEP-refined prediction')
    ax.legend(ncol=2,fontsize=7)
    save_figure(fig,out/'Fig04a_Blumen_PINN_GEP_overlay')

    # Interpolate the digitized Blumen field only when the source has enough non-collinear samples.
    try:
        from scipy.interpolate import LinearNDInterpolator
        if len(bwork)>=12 and bwork[['Mach','x']].drop_duplicates().shape[0]>=12:
            interp=LinearNDInterpolator(bwork[['Mach','x']].to_numpy(),bwork.ci.to_numpy(),fill_value=np.nan)
            qx=df.alpha.to_numpy() if coordinate=='alpha' else df.eta.to_numpy()
            bref=np.asarray(interp(df.Mach.to_numpy(),qx),float)
            comp=df.copy();comp['ci_blumen']=bref
            comp['pinn_blumen_abs']=(comp.ci_seed-comp.ci_blumen).abs();comp['gep_blumen_abs']=(comp.ci_final-comp.ci_blumen).abs()
            comp['pinn_blumen_rel']=safe_rel_error(comp.ci_seed,comp.ci_blumen);comp['gep_blumen_rel']=safe_rel_error(comp.ci_final,comp.ci_blumen)
            finite=comp.ci_blumen.notna()
            if finite.sum()>=20:
                vals=pd.concat([comp.loc[finite,'pinn_blumen_abs'],comp.loc[finite,'gep_blumen_abs']]).replace(0,np.nan).dropna()
                norm=LogNorm(vmin=max(vals.min(),1e-8),vmax=max(vals.max(),1e-7))
                fig,axes=plt.subplots(2,2,figsize=(11,9))
                scatter_map(axes[0,0],comp[finite],'pinn_blumen_abs',norm=norm,cmap='magma',title='PINN seed vs interpolated Blumen',cbar_label='absolute error')
                scatter_map(axes[0,1],comp[finite],'gep_blumen_abs',norm=norm,cmap='magma',title='GEP final vs interpolated Blumen',cbar_label='absolute error')
                scatter_map(axes[1,0],comp[finite],'pinn_blumen_rel',norm=LogNorm(vmin=max(comp.loc[finite,'pinn_blumen_rel'].replace(0,np.nan).min(),1e-6),vmax=max(comp.loc[finite,'pinn_blumen_rel'].max(),1e-5)),cmap='viridis',title='PINN relative error vs Blumen',cbar_label='relative error')
                scatter_map(axes[1,1],comp[finite],'gep_blumen_rel',norm=LogNorm(vmin=max(comp.loc[finite,'gep_blumen_rel'].replace(0,np.nan).min(),1e-6),vmax=max(comp.loc[finite,'gep_blumen_rel'].max(),1e-5)),cmap='viridis',title='GEP relative error vs Blumen',cbar_label='relative error')
                fig.suptitle('Blumen comparison on the interpolation support');fig.tight_layout();save_figure(fig,out/'Fig04b_Blumen_error_maps')

                # One-dimensional cuts at representative Mach values.
                targets=[0.10,0.30,0.50,0.70,0.90,0.97]
                fig,axes=plt.subplots(2,3,figsize=(13,7.5),squeeze=False)
                for ax,M0 in zip(axes.ravel(),targets):
                    band=max(.015,0.02 if M0<.9 else .01)
                    br=bwork[(bwork.Mach-M0).abs()<=band].sort_values('x')
                    rr=comp[finite & ((comp.Mach-M0).abs()<=band)].sort_values(coordinate)
                    if not br.empty: ax.scatter(br.x,br.ci,s=18,facecolors='none',edgecolors='k',label='Blumen')
                    if not rr.empty:
                        xx=rr.alpha if coordinate=='alpha' else rr.eta
                        ax.scatter(xx,rr.ci_seed,s=14,label='PINN seed');ax.scatter(xx,rr.ci_final,s=14,label='GEP final')
                    ax.set(title=fr'$M\approx {M0:.2f}$',xlabel=xlabel,ylabel=r'$c_i$')
                axes[0,0].legend(fontsize=7);fig.tight_layout();save_figure(fig,out/'Fig04c_Blumen_Mach_cuts')
                comp[finite].to_csv(out.parent/'data'/'Blumen_interpolated_comparison.csv',index=False)
            else:
                report['warnings'].append('Blumen interpolation covers fewer than 20 release points; only overlay was generated.')
        else:
            report['warnings'].append('Blumen CSV does not contain enough samples for a 2-D interpolation; only overlay was generated.')
    except Exception as exc:
        report['warnings'].append(f'Blumen error maps/cuts were skipped: {exc}')
    report['blumen_csv']=str(blumen_path)

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo-root',default='.')
    parser.add_argument('--atlas-root',default='assets/pinn_subsonic/local_atlas_v1')
    parser.add_argument('--output-dir',default='assets/pinn_subsonic/local_atlas_v1/publication_assets_scientific_v2')
    parser.add_argument('--blumen-csv',default=None)
    args=parser.parse_args()
    configure_plotting()
    repo=Path(args.repo_root).resolve(); atlas=(repo/args.atlas_root).resolve(); out=(repo/args.output_dir).resolve()
    figdir=out/'figures'; datadir=out/'data'; figdir.mkdir(parents=True,exist_ok=True);datadir.mkdir(parents=True,exist_ok=True)
    report={'generated':[],'missing':[],'warnings':[]}

    release_path,raw=resolve_release_frame(atlas); df=canonicalize_release(raw)
    df.to_csv(datadir/'validation_pointwise_canonical.csv',index=False)
    rectangles=load_rectangles(atlas); rectangles.to_csv(datadir/'atlas_catalog_operational.csv',index=False)
    assignment=build_assignment(rectangles); assignment.to_csv(datadir/'coverage_and_assignment_grid.csv',index=False)
    plot_atlas_status_footprints(rectangles,figdir/'Fig02a_atlas_status_footprints')
    plot_coverage_grid(assignment,figdir/'Fig02b_coverage_multiplicity')
    plot_atlas_multipanel(rectangles,assignment,figdir/'Fig02_atlas_architecture_and_coverage')
    plot_categorical_grid(assignment,'selected_chart','Operational chart assignment',figdir/'SuppFig03_final_chart_assignment')
    plot_categorical_grid(assignment,'pipeline_mode','Operational inference pipeline',figdir/'Fig02c_operational_pipeline_map')
    plot_sparse_supervision(atlas,df,figdir/'Fig03_sparse_spectral_supervision',report)
    build_ci_figures(df,figdir)
    build_modal_heatmaps(df,figdir,report)
    build_blumen(load_blumen(repo,args.blumen_csv),df,figdir,report)

    summary={
        'release_csv':str(release_path),
        'n_points':int(len(df)),
        'n_charts':int(len(rectangles)),
        'coverage_min':int(assignment.coverage_count.min()),
        'coverage_max':int(assignment.coverage_count.max()),
        'ci_seed_abs_max':float(df.ci_seed_abs_err.max()),
        'ci_final_abs_max':float(df.ci_final_abs_err.max()),
        'modal_final_max':float(df.modal_error_final.max()),
        'p_overlap_final_min':float(df.p_overlap_final.min()),
        **report,
    }
    write_json(out/'build_report.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
