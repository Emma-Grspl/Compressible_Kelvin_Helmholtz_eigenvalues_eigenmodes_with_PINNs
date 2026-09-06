#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from plots.scripts.pinn_subsonic.utils_asset_common import configure_plotting

FIELDS=('p','rho','u','v')


def complex_from_csv(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    direct=prefix
    if direct in frame.columns:
        return frame[direct].to_numpy(dtype=complex)
    r=f'{prefix}_real'; i=f'{prefix}_imag'
    if r not in frame.columns:
        raise KeyError(f'Missing {prefix} or {r}')
    imag=np.zeros(len(frame)) if i not in frame.columns else pd.to_numeric(frame[i],errors='coerce').to_numpy()
    return pd.to_numeric(frame[r],errors='coerce').to_numpy()+1j*imag


def load_profile(path: Path) -> dict[str,Any]:
    if path.suffix.lower()=='.npz':
        with np.load(path,allow_pickle=True) as z:
            return {k:z[k] for k in z.files}
    if path.suffix.lower()=='.csv':
        f=pd.read_csv(path); d={'y':pd.to_numeric(f['y'],errors='coerce').to_numpy()}
        for field in FIELDS:
            d[f'{field}_ref']=complex_from_csv(f,f'{field}_ref')
            d[f'{field}_pred']=complex_from_csv(f,f'{field}_pred')
        return d
    raise ValueError(path)


def find_profile(root: Path, point_id: str) -> Path|None:
    for suffix in ('.npz','.csv'):
        p=root/f'{point_id}{suffix}'
        if p.exists(): return p
    hits=list(root.rglob(f'{point_id}*.npz'))+list(root.rglob(f'{point_id}*.csv'))
    return hits[0] if hits else None


def phase_align(pred: np.ndarray, ref: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask=np.isfinite(pred)&np.isfinite(ref)&np.isfinite(y)
    if mask.sum()<2: return pred
    inner=np.trapz(np.conjugate(ref[mask])*pred[mask],y[mask])
    if abs(inner)==0: return pred
    return pred*np.exp(-1j*np.angle(inner))


def rel_l2(pred: np.ndarray, ref: np.ndarray, y: np.ndarray) -> float:
    num=np.trapz(np.abs(pred-ref)**2,y); den=np.trapz(np.abs(ref)**2,y)
    return float(np.sqrt(max(num,0)/max(den,1e-30)))


def make_pdf(points: pd.DataFrame, profile_root: Path, output: Path, title_prefix: str) -> tuple[int,list[str]]:
    output.parent.mkdir(parents=True,exist_ok=True); missing=[]; pages=0
    with PdfPages(output) as pdf:
        for _,row in points.iterrows():
            pid=str(row.point_id); path=find_profile(profile_root,pid)
            if path is None:
                missing.append(pid); continue
            d=load_profile(path); y=np.asarray(d['y'],float)
            fig,axes=plt.subplots(4,2,figsize=(11.2,12),sharex=True)
            metrics=[]
            for i,field in enumerate(FIELDS):
                ref=np.asarray(d[f'{field}_ref'],complex); pred=np.asarray(d[f'{field}_pred'],complex)
                pred=phase_align(pred,ref,y); metrics.append(f'{field}: {rel_l2(pred,ref,y):.3e}')
                axes[i,0].plot(y,ref.real,'k-',lw=1.4,label='reference'); axes[i,0].plot(y,pred.real,'--',lw=1.2,label='prediction')
                axes[i,1].plot(y,ref.imag,'k-',lw=1.4,label='reference'); axes[i,1].plot(y,pred.imag,'--',lw=1.2,label='prediction')
                axes[i,0].set_ylabel(field); axes[i,0].set_title(f'{field}: real part'); axes[i,1].set_title(f'{field}: imaginary part')
            axes[-1,0].set_xlabel(r'$y$'); axes[-1,1].set_xlabel(r'$y$')
            axes[0,0].legend(loc='best'); axes[0,1].legend(loc='best')
            info=f"M={row.Mach:.6f}, eta={row.eta:.6f}, alpha={row.alpha:.6f}"
            if 'ci_ref' in row and pd.notna(row.ci_ref): info+=f", ci_ref={row.ci_ref:.6g}"
            if 'ci_final' in row and pd.notna(row.ci_final): info+=f", ci_final={row.ci_final:.6g}"
            fig.suptitle(f'{title_prefix} — {pid}\n{info}\n'+'; '.join(metrics),fontsize=12)
            fig.tight_layout(rect=(0,0,1,.94)); pdf.savefig(fig,bbox_inches='tight'); plt.close(fig); pages+=1
    return pages,missing


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--points-csv',required=True)
    p.add_argument('--direct-profile-dir',required=True)
    p.add_argument('--gep-profile-dir',required=True)
    p.add_argument('--output-dir',required=True)
    a=p.parse_args(); configure_plotting()
    points=pd.read_csv(a.points_csv); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    n1,m1=make_pdf(points,Path(a.direct_profile_dir),out/'supp_modes_direct_PINN_vs_classic_20_points.pdf','Direct PINN versus classical reference')
    n2,m2=make_pdf(points,Path(a.gep_profile_dir),out/'supp_modes_PINN_GEP_vs_classic_GEP_20_points.pdf','PINN-seeded GEP versus classical GEP')
    report=pd.DataFrame({'pipeline':['direct','gep'],'pages':[n1,n2],'missing_point_ids':[';'.join(m1),';'.join(m2)]})
    report.to_csv(out/'mode_pdf_build_report.csv',index=False)
    print(report.to_string(index=False))
    if m1 or m2:
        raise SystemExit('Some mode profiles are missing; see mode_pdf_build_report.csv and contracts/mode_profile_schema.md')

if __name__=='__main__': main()
