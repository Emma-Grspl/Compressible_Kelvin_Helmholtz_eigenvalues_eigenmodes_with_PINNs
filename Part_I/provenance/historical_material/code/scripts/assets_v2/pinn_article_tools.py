#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    torch = None


def rp(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def read_header(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def find_preferred(root: Path, filename: str) -> Path | None:
    preferred = [
        root / 'pinn_subsonic/data/scientific_outputs/release_v1/data' / filename,
        root / 'pinn_subsonic/data/scientific_outputs/local_atlas_v1/publication_assets_scientific_v2/data' / filename,
        root / 'pinn_subsonic/manifests' / filename,
    ]
    for path in preferred:
        if path.exists():
            return path
    matches = [p for p in root.rglob(filename) if 'article_audit' not in p.parts]
    if not matches:
        return None
    matches.sort(key=lambda p: ('release_v1' in p.parts, 'scientific_outputs' in p.parts, -len(p.parts)), reverse=True)
    return matches[0]


def linear_info(state: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    layers = []
    for key, tensor in state.items():
        m = re.fullmatch(r'net\.(\d+)\.weight', str(key))
        if m and hasattr(tensor, 'shape') and len(tuple(tensor.shape)) == 2:
            layers.append((int(m.group(1)), tuple(tensor.shape)))
    if not layers:
        return None, None, None
    layers.sort()
    return int(layers[0][1][1]), int(layers[-1][1][0]), max(len(layers)-1, 0)


def count_anchors(raw: Any) -> int | None:
    try:
        if isinstance(raw, pd.DataFrame):
            return len(raw)
        if isinstance(raw, dict):
            return len(pd.DataFrame(raw))
        if isinstance(raw, (list, tuple)):
            return len(raw)
    except Exception:
        pass
    return None


def audit_checkpoints(root: Path) -> pd.DataFrame:
    rows = []
    keys = [
        'mach_min','mach_max','eta_min','eta_max','ymax','width','depth','n_freq',
        'epochs','n_epochs','max_epochs','lr','learning_rate','optimizer','activation',
        'n_collocation','n_f','n_boundary','batch_size','amp_mask_frac',
        'ci_idw_eta_scale','ci_idw_mach_scale','ci_idw_power','seed','output_dir'
    ]
    for path in sorted(root.rglob('model_state.pt')):
        if 'article_audit' in path.parts:
            continue
        row = {'checkpoint_path': rp(path, root), 'chart_id': path.parent.name, 'size_MiB': path.stat().st_size/1024**2}
        if torch is None:
            row['load_status'] = 'torch_unavailable'
            rows.append(row)
            continue
        try:
            ckpt = torch.load(path, map_location='cpu')
            args = dict(ckpt.get('args', {}))
            inp, out, hidden = linear_info(ckpt.get('field_state_dict', {}))
            row.update({
                'load_status':'ok', 'architecture_input_dim':inp,
                'architecture_output_dim':out, 'hidden_linear_layers':hidden,
                'n_anchors':count_anchors(ckpt.get('anchor_df')),
                'field_family':'pQscaled' if 'qscaled' in str(args.get('output_dir','')).lower() else 'pq_or_unspecified'
            })
            for key in keys:
                if key in args:
                    value = args[key]
                    if isinstance(value, (str,int,float,bool)) or value is None:
                        row[key] = value
                    else:
                        row[key] = str(value)
            row['args_json'] = json.dumps({str(k): str(v) if not isinstance(v,(str,int,float,bool,type(None))) else v for k,v in args.items()}, sort_keys=True)
        except Exception as exc:
            row['load_status'] = f'failed:{type(exc).__name__}'
            row['load_error'] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def audit_csvs(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob('*.csv')):
        if 'article_audit' in path.parts:
            continue
        columns = read_header(path)
        if not columns:
            continue
        low = {str(c).lower() for c in columns}
        cats = []
        if any('epoch' in c for c in low) and any('loss' in c for c in low):
            cats.append('training_history')
        if any('anchor' in c for c in low):
            cats.append('anchor_ablation')
        if {'mach','alpha','ci_ref','ci_seed','ci_final'}.issubset(low):
            cats.append('validation_pointwise')
        if {'chart_id','mach','eta'}.issubset(low):
            cats.append('chart_manifest')
        if {'chart_id','mach','alpha'}.issubset(low) and any(c in low for c in ('ci_pred','ci_seed','ci_final')):
            cats.append('chart_prediction')
        if any(c == 'seed' or c.endswith('_seed') for c in low):
            cats.append('multiseed')
        if not cats:
            continue
        rows.append({
            'path':rp(path,root), 'size_MiB':path.stat().st_size/1024**2,
            'categories':';'.join(sorted(set(cats))), 'n_columns':len(columns),
            'columns':'|'.join(columns)
        })
    return pd.DataFrame(rows)


def audit_scripts(root: Path) -> pd.DataFrame:
    rows=[]
    token = re.compile(r'single|anchor|ablation|riccati|subsonic|atlas|pinn', re.I)
    for path in sorted(root.rglob('*.py')):
        if 'article_audit' in path.parts:
            continue
        try:
            text=path.read_text(encoding='utf-8',errors='ignore')
        except Exception:
            continue
        if not token.search(rp(path,root)) and not token.search(text):
            continue
        flags=sorted(set(re.findall(r'add_argument\(\s*["\'](--[^"\']+)',text)))
        rows.append({
            'path':rp(path,root),'size_KiB':path.stat().st_size/1024,
            'argparse_flags':'|'.join(flags),'mentions_anchor':bool(re.search('anchor',text,re.I)),
            'mentions_riccati':bool(re.search('riccati',text,re.I)),
            'mentions_seed':bool(re.search('seed',text,re.I))
        })
    return pd.DataFrame(rows)


def choose_anchor_summary(root: Path, csv_inv: pd.DataFrame) -> Path | None:
    if csv_inv.empty:
        return None
    candidates=[]
    for _,row in csv_inv.iterrows():
        if 'anchor_ablation' not in str(row['categories']):
            continue
        cols=set(str(row['columns']).lower().split('|'))
        score=sum(2 for c in ('anchor_strategy','anchor_label','best_ci_mae','best_p_rel','run_dir','seed') if c in cols)
        if 'ablation' in str(row['path']).lower(): score+=2
        candidates.append((score,root/str(row['path'])))
    return max(candidates,default=(None,None))[1]


def audit(root: Path, out: Path) -> None:
    out.mkdir(parents=True,exist_ok=True)
    ckpt=audit_checkpoints(root); ckpt.to_csv(out/'checkpoint_inventory.csv',index=False)
    csv_inv=audit_csvs(root); csv_inv.to_csv(out/'csv_inventory.csv',index=False)
    scripts=audit_scripts(root); scripts.to_csv(out/'python_script_inventory.csv',index=False)
    validation=find_preferred(root,'validation_pointwise_canonical.csv')
    anchor=choose_anchor_summary(root,csv_inv)
    histories=[]
    if not csv_inv.empty:
        histories=[str(p) for p in csv_inv.loc[csv_inv['categories'].str.contains('training_history',na=False),'path'].head(50)]
    status=[]
    status.append({'asset':'Table_PINN_hyperparameters','status':'ready' if not ckpt.empty and (ckpt.get('load_status')=='ok').any() else 'missing','source':'checkpoint_inventory.csv'})
    status.append({'asset':'Table_global_validation_metrics','status':'ready' if validation else 'missing','source':rp(validation,root) if validation else ''})
    status.append({'asset':'Fig_single_case_anchor_ablation','status':'partial' if anchor else 'missing','source':rp(anchor,root) if anchor else '','note':'final main figure also needs aligned modal profiles for each configuration'})
    status.append({'asset':'SuppFig_training_convergence','status':'ready_or_partial' if histories else 'missing','source':histories[0] if histories else ''})
    status.append({'asset':'SuppTable_multiseed_robustness','status':'ready_or_partial' if anchor else 'missing','source':rp(anchor,root) if anchor else ''})
    overlap_ready = (not csv_inv.empty and csv_inv['categories'].str.contains('chart_prediction',na=False).any())
    status.append({'asset':'SuppFig_chart_boundary_continuity','status':'partial' if overlap_ready else 'missing','source':'candidate chart-prediction CSVs' if overlap_ready else ''})
    pd.DataFrame(status).to_csv(out/'article_asset_status.csv',index=False)
    summary={
        'repo_root':str(root),'torch_available':torch is not None,
        'validation_pointwise':rp(validation,root) if validation else None,
        'anchor_ablation_summary':rp(anchor,root) if anchor else None,
        'training_history_candidates':histories,'assets':status
    }
    (out/'audit_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True),encoding='utf-8')
    print(pd.DataFrame(status).to_string(index=False))
    print('\nAudit written to',out)


def tex(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    try:
        text=frame.to_latex(index=False,escape=False,**kwargs)
    except Exception:
        text='% LaTeX export unavailable; use CSV.\n'
    path.write_text(text,encoding='utf-8')


def build_tables(root: Path, audit_dir: Path, out: Path, summary: dict) -> None:
    table_dir=out/'tables'; table_dir.mkdir(parents=True,exist_ok=True)
    ckpt_path=audit_dir/'checkpoint_inventory.csv'
    if ckpt_path.exists():
        df=pd.read_csv(ckpt_path)
        if 'load_status' in df.columns: df=df.loc[df['load_status']=='ok'].copy()
        cols=[c for c in ['chart_id','mach_min','mach_max','eta_min','eta_max','field_family','width','depth','hidden_linear_layers','n_freq','epochs','n_epochs','max_epochs','lr','learning_rate','optimizer','n_collocation','n_f','n_boundary','activation','n_anchors','ci_idw_eta_scale','ci_idw_mach_scale','ci_idw_power'] if c in df.columns and not df[c].isna().all()]
        if not df.empty and cols:
            t=df[cols].sort_values('chart_id')
            t.to_csv(table_dir/'Table_PINN_hyperparameters.csv',index=False)
            tex(t,table_dir/'Table_PINN_hyperparameters.tex',longtable=True)
            print('WROTE Table_PINN_hyperparameters')
    vrel=summary.get('validation_pointwise')
    vpath=root/vrel if vrel else None
    if vpath and vpath.exists():
        df=pd.read_csv(vpath)
        metrics=['ci_seed_abs_err','ci_final_abs_err','ci_seed_rel_err','ci_final_rel_err','p_rel_direct','rho_rel_direct','u_rel_direct','v_rel_direct','p_rel_final','rho_rel_final','u_rel_final','v_rel_final','p_overlap_direct','p_overlap_final']
        rows=[]
        for m in metrics:
            if m not in df.columns: continue
            x=pd.to_numeric(df[m],errors='coerce'); x=x[np.isfinite(x)]
            if x.empty: continue
            rows.append({'metric':m,'n':len(x),'median':x.median(),'mean':x.mean(),'q90':x.quantile(.9),'q95':x.quantile(.95),'max':x.max()})
        t=pd.DataFrame(rows)
        t.to_csv(table_dir/'Table_global_validation_metrics.csv',index=False)
        tex(t,table_dir/'Table_global_validation_metrics.tex',float_format='%.4e')
        print('WROTE Table_global_validation_metrics')


def group_col(df: pd.DataFrame) -> str | None:
    return next((c for c in ('anchor_label','anchor_strategy','anchor','configuration','config') if c in df.columns),None)


def first_col(df: pd.DataFrame, names: tuple[str,...]) -> str | None:
    return next((c for c in names if c in df.columns),None)


def plot_metric(ax: plt.Axes, df: pd.DataFrame, group: str, metric: str, title: str) -> None:
    labels=list(dict.fromkeys(df[group].astype(str)))
    for i,label in enumerate(labels):
        x=pd.to_numeric(df.loc[df[group].astype(str)==label,metric],errors='coerce'); x=x[np.isfinite(x)]
        if x.empty: continue
        jitter=np.linspace(-.12,.12,len(x)) if len(x)>1 else np.array([0.])
        ax.scatter(i+jitter,x,s=34,alpha=.75)
        ax.plot([i-.2,i+.2],[np.median(x),np.median(x)],lw=2)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=20,ha='right')
    ax.set_title(title); ax.grid(True,axis='y',alpha=.25)
    vals=pd.to_numeric(df[metric],errors='coerce'); vals=vals[np.isfinite(vals)&(vals>0)]
    if not vals.empty and vals.max()/vals.min()>100: ax.set_yscale('log')


def build_anchor_assets(root: Path, out: Path, summary: dict) -> None:
    rel=summary.get('anchor_ablation_summary')
    if not rel: print('SKIP anchor ablation: no summary found'); return
    path=root/rel
    if not path.exists(): print('SKIP anchor ablation: source missing'); return
    df=pd.read_csv(path); group=group_col(df)
    if group is None: print('SKIP anchor ablation: no configuration column'); return
    metrics=[]
    for names,title in [
        (('best_ci_mae','best_ci_abs_err','ci_abs_err','last_ci_mae','ci_mae'),'c_i error'),
        (('best_p_rel','p_rel','last_p_rel','modal_error','best_modal_error'),'pressure-mode error'),
        (('best_env','last_env','envelope_error'),'envelope error'),
        (('best_phase','last_phase','phase_error'),'phase error')]:
        c=first_col(df,names)
        if c: metrics.append((c,title))
    if len(metrics)<2: print('SKIP anchor ablation: fewer than two metrics'); return
    fig_dir=out/'figures'; table_dir=out/'tables'; fig_dir.mkdir(parents=True,exist_ok=True); table_dir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11,8.5)); flat=axes.ravel()
    for ax,(metric,title) in zip(flat,metrics): plot_metric(ax,df,group,metric,title)
    for ax in flat[len(metrics):]: ax.axis('off')
    fig.suptitle('Single-case spectral-anchor ablation — available runs',fontsize=14); fig.tight_layout()
    stem=fig_dir/'Fig_single_case_anchor_ablation_metrics'; fig.savefig(stem.with_suffix('.pdf'),bbox_inches='tight'); fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight'); plt.close(fig)
    rows=[]
    for label,sub in df.groupby(group,dropna=False):
        row={'configuration':str(label),'n_runs':len(sub)}
        for m,_ in metrics:
            x=pd.to_numeric(sub[m],errors='coerce'); x=x[np.isfinite(x)]
            if not x.empty: row.update({f'{m}_median':x.median(),f'{m}_q90':x.quantile(.9),f'{m}_max':x.max()})
        rows.append(row)
    t=pd.DataFrame(rows); t.to_csv(table_dir/'SuppTable_multiseed_robustness.csv',index=False); tex(t,table_dir/'SuppTable_multiseed_robustness.tex',float_format='%.4e')
    print('WROTE Fig_single_case_anchor_ablation_metrics')
    print('WROTE SuppTable_multiseed_robustness')
    print('NOTE: the final main ablation figure still needs aligned mode profiles for each configuration.')


def build_training_plot(root: Path, out: Path, summary: dict) -> None:
    usable=[]
    for rel in summary.get('training_history_candidates',[]):
        path=root/rel
        if not path.exists(): continue
        try: df=pd.read_csv(path)
        except Exception: continue
        epoch=next((c for c in df.columns if 'epoch' in c.lower()),None)
        losses=[c for c in df.columns if 'loss' in c.lower() and pd.to_numeric(df[c],errors='coerce').notna().sum()>=3]
        if epoch and losses: usable.append((path,df,epoch,losses))
        if len(usable)>=3: break
    if not usable: print('SKIP training convergence: no usable histories'); return
    fig,axes=plt.subplots(len(usable),1,figsize=(10.5,3.4*len(usable)),squeeze=False)
    for ax,(path,df,epoch,losses) in zip(axes.ravel(),usable):
        x=pd.to_numeric(df[epoch],errors='coerce')
        for c in losses[:6]:
            y=pd.to_numeric(df[c],errors='coerce'); mask=np.isfinite(x)&np.isfinite(y)&(y>0)
            if mask.sum()>=3: ax.plot(x[mask],y[mask],label=c)
        ax.set_yscale('log'); ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.set_title(rp(path,root)); ax.grid(True,alpha=.25); ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Representative PINN training convergence',fontsize=14); fig.tight_layout()
    fig_dir=out/'figures'; fig_dir.mkdir(parents=True,exist_ok=True); stem=fig_dir/'SuppFig_training_convergence'; fig.savefig(stem.with_suffix('.pdf'),bbox_inches='tight'); fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight'); plt.close(fig)
    print('WROTE SuppFig_training_convergence')


def build(root: Path, audit_dir: Path, out: Path) -> None:
    summary_path=audit_dir/'audit_summary.json'
    if not summary_path.exists(): raise FileNotFoundError(f'Run audit first: {summary_path}')
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    out.mkdir(parents=True,exist_ok=True)
    build_tables(root,audit_dir,out,summary)
    build_anchor_assets(root,out,summary)
    build_training_plot(root,out,summary)
    status_path=audit_dir/'article_asset_status.csv'
    if status_path.exists():
        df=pd.read_csv(status_path); df.loc[df['status']!='ready'].to_csv(out/'missing_or_partial_assets.csv',index=False)
    print('\nGenerated files:')
    for p in sorted(out.rglob('*')):
        if p.is_file(): print(rp(p,root))


def main() -> None:
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit'); a.add_argument('--repo-root',default='.'); a.add_argument('--output-dir',default='pinn_subsonic/article_audit')
    b=sub.add_parser('build-existing'); b.add_argument('--repo-root',default='.'); b.add_argument('--audit-dir',default='pinn_subsonic/article_audit'); b.add_argument('--output-dir',default='assets/pinn_subsonic/article/generated')
    args=parser.parse_args(); root=Path(args.repo_root).resolve()
    if args.command=='audit': audit(root,root/args.output_dir)
    else: build(root,root/args.audit_dir,root/args.output_dir)


if __name__=='__main__':
    main()
