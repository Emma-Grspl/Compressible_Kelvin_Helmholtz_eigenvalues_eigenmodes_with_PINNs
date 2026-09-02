#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 320


def configure_plotting() -> None:
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': DPI,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.grid': False,
    })


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def first_column(frame: pd.DataFrame, aliases: Sequence[str], required: bool = True) -> str | None:
    for name in aliases:
        if name in frame.columns:
            return name
    if required:
        raise KeyError(f'None of these columns were found: {list(aliases)}. Available={list(frame.columns)}')
    return None


def numeric(frame: pd.DataFrame, aliases: Sequence[str], required: bool = True) -> pd.Series | None:
    name = first_column(frame, aliases, required=required)
    if name is None:
        return None
    return pd.to_numeric(frame[name], errors='coerce')


def text(frame: pd.DataFrame, aliases: Sequence[str], required: bool = True) -> pd.Series | None:
    name = first_column(frame, aliases, required=required)
    if name is None:
        return None
    return frame[name].astype(str)


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({'true','1','yes','y','pass','passed'})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def find_first(root: Path, candidates: Iterable[str]) -> Path | None:
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    return None


def find_named(root: Path, names: Iterable[str]) -> Path | None:
    names = set(names)
    for path in root.rglob('*'):
        if path.is_file() and path.name in names:
            return path
    return None


def safe_rel_error(pred: pd.Series, ref: pd.Series, floor: float = 1e-12) -> pd.Series:
    return (pred - ref).abs() / ref.abs().clip(lower=floor)


def nearest_row(frame: pd.DataFrame, M: float, eta: float, mcol: str='Mach', ecol: str='eta') -> pd.Series:
    dist = (frame[mcol].astype(float) - M) ** 2 + (frame[ecol].astype(float) - eta) ** 2
    return frame.loc[dist.idxmin()]


def resolve_release_frame(atlas_root: Path) -> tuple[Path, pd.DataFrame]:
    candidates = [
        atlas_root / 'offgrid_validation_384' / 'offgrid_validation_results_384_release.csv',
        atlas_root / 'release_fullrect_v1' / 'offgrid_validation_results_384_release.csv',
        atlas_root / 'publication_assets_fullrect_v1' / 'data' / 'validation_pointwise.csv',
    ]
    for path in candidates:
        if path.exists():
            return path, pd.read_csv(path)
    raise FileNotFoundError('Could not find off-grid release CSV. Tried:\n' + '\n'.join(map(str, candidates)))


def canonicalize_release(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out['point_id'] = text(frame, ['point_id','id','sample_id'])
    out['Mach'] = numeric(frame, ['Mach','M','mach'])
    out['eta'] = numeric(frame, ['eta','Eta'])
    alpha = numeric(frame, ['alpha','Alpha'], required=False)
    if alpha is None:
        alpha = out['eta'] * np.sqrt(np.maximum(0.0, 1.0 - out['Mach'] ** 2))
    out['alpha'] = alpha

    out['ci_ref'] = numeric(frame, ['ci_classic','ci_classic_raw','classic_ci','ci_reference','ci_ref'])
    out['ci_seed'] = numeric(frame, ['ci_seed','seed_ci','pinn_ci','ci_pinn'])
    out['ci_gep'] = numeric(frame, ['gep_ci','ci_gep','ci_independent_original','ci_independent'], required=False)
    out['ci_final'] = numeric(frame, ['ci_final','ci_forward','gep_ci','ci_gep'], required=False)
    if out['ci_gep'].isna().all():
        out['ci_gep'] = out['ci_final']
    if out['ci_final'].isna().all():
        out['ci_final'] = out['ci_gep']

    for dest, aliases in {
        'p_rel_final': ['p_rel_final','p_rel_cont_vs_classic','p_rel'],
        'rho_rel_final': ['rho_rel_final','rho_rel_cont_vs_classic','rho_rel'],
        'u_rel_final': ['u_rel_final','u_rel_cont_vs_classic','u_rel'],
        'v_rel_final': ['v_rel_final','v_rel_cont_vs_classic','v_rel'],
        'p_overlap_final': ['p_overlap_final','p_overlap_cont_vs_classic','p_overlap'],
        'p_rel_direct': ['p_rel_direct','pinn_p_rel','p_rel_seed'],
        'rho_rel_direct': ['rho_rel_direct','pinn_rho_rel','rho_rel_seed'],
        'u_rel_direct': ['u_rel_direct','pinn_u_rel','u_rel_seed'],
        'v_rel_direct': ['v_rel_direct','pinn_v_rel','v_rel_seed'],
        'p_overlap_direct': ['p_overlap_direct','pinn_p_overlap','p_overlap_seed'],
    }.items():
        series = numeric(frame, aliases, required=False)
        out[dest] = np.nan if series is None else series

    for dest, aliases in {
        'sample_group': ['sample_group','group','stratum'],
        'chart_id': ['chart_id','selected_chart','atlas_id'],
        'continuation_status': ['continuation_status'],
        'reference_status': ['reference_status'],
        'ci_final_source': ['ci_final_source'],
    }.items():
        series = text(frame, aliases, required=False)
        out[dest] = '' if series is None else series

    out['ci_seed_abs_err'] = (out['ci_seed'] - out['ci_ref']).abs()
    out['ci_seed_rel_err'] = safe_rel_error(out['ci_seed'], out['ci_ref'])
    out['ci_final_abs_err'] = (out['ci_final'] - out['ci_ref']).abs()
    out['ci_final_rel_err'] = safe_rel_error(out['ci_final'], out['ci_ref'])
    out['gep_gain'] = out['ci_seed_abs_err'] / out['ci_final_abs_err'].clip(lower=1e-14)
    out['modal_error_final'] = out[['p_rel_final','rho_rel_final','u_rel_final','v_rel_final']].max(axis=1)
    out['modal_error_direct'] = out[['p_rel_direct','rho_rel_direct','u_rel_direct','v_rel_direct']].max(axis=1)
    return out


def scatter_map(ax: plt.Axes, df: pd.DataFrame, value: str, *, norm=None, cmap='viridis', title='', cbar_label='', s=28):
    valid = df[['Mach','eta',value]].dropna()
    im = ax.scatter(valid['Mach'], valid['eta'], c=valid[value], s=s, cmap=cmap, norm=norm, edgecolors='0.25', linewidths=0.25)
    ax.set_xlabel(r'Mach number $M$')
    ax.set_ylabel(r'Scaled wavenumber $\eta$')
    ax.set_xlim(0.02,0.98)
    ax.set_ylim(0.02,0.98)
    ax.set_title(title)
    cb = ax.figure.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    return im
