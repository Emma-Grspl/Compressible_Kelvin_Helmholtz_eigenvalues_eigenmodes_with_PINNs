#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.scripts.pinn_subsonic.utils_asset_common import configure_plotting, save_figure
from plots.scripts.pinn_subsonic.active_implementation.source_tree.scripts.assets_v2.plot_mode_pdfs import FIELDS, find_profile, load_profile, phase_align


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument('--points-csv',required=True)
    p.add_argument('--direct-profile-dir',required=True)
    p.add_argument('--gep-profile-dir',required=True)
    p.add_argument('--output-stem',required=True)
    p.add_argument('--n-points',type=int,default=6)
    a=p.parse_args();configure_plotting()
    pts=pd.read_csv(a.points_csv).head(a.n_points)
    fig,axes=plt.subplots(len(pts),4,figsize=(13,2.25*len(pts)),squeeze=False,sharex=False)
    missing=[]
    for i,(_,row) in enumerate(pts.iterrows()):
        pid=str(row.point_id); pdirect=find_profile(Path(a.direct_profile_dir),pid); pgep=find_profile(Path(a.gep_profile_dir),pid)
        if pdirect is None or pgep is None:
            missing.append(pid);continue
        direct=load_profile(pdirect); gep=load_profile(pgep); y=np.asarray(direct['y'],float)
        for j,field in enumerate(FIELDS):
            ref=np.asarray(direct[f'{field}_ref'],complex)
            d=phase_align(np.asarray(direct[f'{field}_pred'],complex),ref,y)
            g=phase_align(np.asarray(gep[f'{field}_pred'],complex),ref,y)
            ax=axes[i,j]
            ax.plot(y,ref.real,'k-',lw=1.2,label='classical')
            ax.plot(y,d.real,'--',lw=1.0,label='PINN direct')
            ax.plot(y,g.real,':',lw=1.4,label='PINN+GEP')
            ax.set_title(field if i==0 else '')
            if j==0:
                ax.set_ylabel(f'{pid}\nreal part')
            if i==len(pts)-1:
                ax.set_xlabel(r'$y$')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.995))
    fig.suptitle('Representative modal profiles: classical, direct PINN and PINN-seeded GEP',y=1.015)
    fig.tight_layout(rect=(0,0,1,.97));save_figure(fig,Path(a.output_stem))
    if missing: raise SystemExit('Missing profiles for: '+', '.join(missing))

if __name__=='__main__':main()
