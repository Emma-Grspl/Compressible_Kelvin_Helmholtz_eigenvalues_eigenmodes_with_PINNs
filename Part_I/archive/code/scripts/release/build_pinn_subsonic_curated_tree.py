#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd

CHECKPOINT_NAMES = ['model_state.pt','model_best.pt','best_model.pt','checkpoint_best.pt']
CONFIG_NAMES = ['config.yaml','config.yml','config.json','args.json','normalization.json','ci_anchors.csv','metrics.json','README.md']


def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--repo-root', default='.')
    p.add_argument('--code-dir', default='pinn_subsonic')
    p.add_argument('--asset-dir', default='assets/pinn_subsonic/release_v1')
    p.add_argument('--model-mode', choices=['hardlink','copy','symlink','none'], default='hardlink')
    p.add_argument('--clean', action='store_true')
    p.add_argument('--strict-models', action='store_true')
    return p.parse_args()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def cp(src: Path, dst: Path, inv: list[dict], category: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src,dst)
    inv.append({'category':category,'source':str(src),'destination':str(dst),'size_bytes':dst.stat().st_size})


def materialize(src: Path, dst: Path, mode: str, inv: list[dict]):
    if mode == 'none': return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink(): dst.unlink()
    if mode == 'hardlink':
        try: os.link(src,dst)
        except OSError: shutil.copy2(src,dst)
    elif mode == 'copy': shutil.copy2(src,dst)
    else: dst.symlink_to(src.resolve())
    inv.append({'category':'model checkpoint','source':str(src),'destination':str(dst),'size_bytes':dst.stat().st_size if dst.exists() else 0})


def first(root: Path, candidates: list[str]) -> Path | None:
    for rel in candidates:
        p = root / rel
        if p.exists(): return p
    return None


def catalog_path(root: Path) -> Path:
    p = first(root,[
        'assets/pinn_subsonic/local_atlas_v1/publication_assets_scientific_v2/data/atlas_catalog_operational.csv',
        'assets/pinn_subsonic/local_atlas_v1/publication_assets_fullrect_v1/data/atlas_catalog.csv',
        'assets/pinn_subsonic/local_atlas_v1/atlas_catalog.csv',
    ])
    if p is None: raise FileNotFoundError('Atlas catalogue not found')
    return p


def chart_dir(root: Path, row: pd.Series) -> Path | None:
    for col in ['path','chart_path','model_path','checkpoint_path']:
        if col not in row or pd.isna(row[col]): continue
        p = Path(str(row[col]))
        if not p.is_absolute(): p = root / p
        if p.is_file(): return p.parent
        if p.is_dir(): return p
    return None


def checkpoint(directory: Path) -> Path | None:
    for name in CHECKPOINT_NAMES:
        p = directory / name
        if p.is_file(): return p
    for p in sorted(directory.glob('*.pt')):
        if p.is_file() and 'optimizer' not in p.name.lower(): return p
    return None


def copy_glob(root: Path, pattern: str, dst: Path, inv: list[dict], category: str):
    for src in sorted(root.glob(pattern)):
        if src.is_file(): cp(src,dst/src.name,inv,category)


def write_readmes(code: Path, assets: Path):
    (code/'README.md').write_text('''# pinn_subsonic\n\nCurated software/model layer for the validated subsonic Kelvin–Helmholtz PINN atlas.\n\n- scripts/figures: plot and publication scripts\n- scripts/validation: validation and continuation scripts\n- scripts/release: audit/release scripts\n- configs: global and per-chart configuration\n- models/<chart_id>: curated model checkpoint and metadata\n- slurm: launch scripts\n- manifests: catalogues and checksums\n\nThe research workspace remains under assets/pinn_subsonic/local_atlas_v1.\n''',encoding='utf-8')
    (assets/'README.md').write_text('''# assets/pinn_subsonic/release_v1\n\nCurated scientific release.\n\n- figures/main\n- figures/supplement\n- tables\n- data\n- validation\n- audits\n- manifests\n- source_data/blumen/subsonic\n''',encoding='utf-8')


def main():
    a = cli(); root = Path(a.repo_root).resolve(); code = root/a.code_dir; assets = root/a.asset_dir
    if a.clean:
        for p in [code,assets]:
            if p.exists(): shutil.rmtree(p)
    dirs = [
        code/'scripts/figures',code/'scripts/validation',code/'scripts/release',code/'configs/global',code/'configs/charts',code/'models',code/'slurm',code/'manifests',code/'examples',code/'tests',
        assets/'figures/main',assets/'figures/supplement',assets/'tables',assets/'data',assets/'validation',assets/'audits',assets/'manifests',assets/'source_data/blumen/subsonic'
    ]
    for p in dirs: p.mkdir(parents=True,exist_ok=True)
    inv=[]

    copy_glob(root,'scripts/assets_v2/*.py',code/'scripts/figures',inv,'figure script')
    for name in ['validate_subsonic_atlas_offgrid.py','continue_subsonic_offgrid_neutral_N401.py','benchmark_subsonic_local_atlas_core_ci_seeded_gep_v2.py']:
        src=root/'scripts/dev'/name
        if src.is_file(): cp(src,code/'scripts/validation'/name,inv,'validation script')
    for name in ['audit_requested_assets.py','build_pinn_subsonic_curated_tree.py']:
        src=root/'scripts/release'/name
        if src.is_file(): cp(src,code/'scripts/release'/name,inv,'release script')
    copy_glob(root,'slurm/offgrid_validation/*.slurm',code/'slurm',inv,'Slurm launcher')

    atlas_root=root/'assets/pinn_subsonic/local_atlas_v1'
    for pattern in ['*.csv','*.json','*.yaml','*.yml']:
        for src in sorted(atlas_root.glob(pattern)):
            if src.is_file(): cp(src,code/'manifests'/src.name,inv,'manifest/policy')

    catp=catalog_path(root); cat=pd.read_csv(catp); chart_col='chart_id' if 'chart_id' in cat.columns else cat.columns[0]
    models=[]
    for _,row in cat.iterrows():
        cid=str(row[chart_col]); cdir=chart_dir(root,row); rec={'chart_id':cid,'chart_source':str(cdir) if cdir else '','checkpoint_source':'','checkpoint_status':'missing'}
        if cdir:
            ck=checkpoint(cdir)
            if ck:
                materialize(ck,code/'models'/cid/'model_state.pt',a.model_mode,inv)
                rec['checkpoint_source']=str(ck); rec['checkpoint_status']='present'
                meta={'chart_id':cid,'source_chart_directory':str(cdir),'source_checkpoint':str(ck),'checkpoint_sha256':digest(ck),'materialization_mode':a.model_mode}
                md=code/'models'/cid/'metadata.json'; md.parent.mkdir(parents=True,exist_ok=True); md.write_text(json.dumps(meta,indent=2),encoding='utf-8')
            for name in CONFIG_NAMES:
                src=cdir/name
                if src.is_file(): cp(src,code/'configs/charts'/cid/name,inv,'chart config')
        models.append(rec)
    model_df=pd.DataFrame(models); model_df.to_csv(code/'manifests/model_manifest.csv',index=False); cp(catp,code/'manifests/atlas_catalog_operational.csv',inv,'atlas catalogue')

    sci=atlas_root/'publication_assets_scientific_v2'
    if not sci.is_dir(): raise FileNotFoundError(sci)
    for src in sorted((sci/'figures').glob('*')):
        if src.is_file(): cp(src,(assets/'figures/supplement' if src.name.startswith('Supp') else assets/'figures/main')/src.name,inv,'publication figure')
    if (sci/'supplement').is_dir():
        for src in sorted((sci/'supplement').glob('*')):
            if src.is_file(): cp(src,assets/'figures/supplement'/src.name,inv,'supplementary PDF')
    for folder,dst,catname in [('tables',assets/'tables','table'),('data',assets/'data','source data')]:
        for src in sorted((sci/folder).glob('*')):
            if src.is_file(): cp(src,dst/src.name,inv,catname)
    for src in sorted(list(sci.glob('*.json'))+list(sci.glob('*.csv'))):
        if src.is_file(): cp(src,assets/'manifests'/src.name,inv,'build report')

    off=atlas_root/'offgrid_validation_384'
    for pattern in ['offgrid_validation_*release*.csv','offgrid_validation_points_384.csv','offgrid_continuation_N401_summary_55.csv']:
        for src in sorted(off.glob(pattern)):
            if src.is_file(): cp(src,assets/'validation'/src.name,inv,'validation data')
    for pattern in ['*branch*resolution*.csv','*reference_branch_corrections*.csv','*manual_audit*.csv','*continuation*targets*.csv']:
        for src in sorted(off.glob(pattern)):
            if src.is_file(): cp(src,assets/'audits'/src.name,inv,'audit data')
    for pattern in ['*mapping_audit*.csv','*N301_vs_N401*.csv','*near_neutral*.csv']:
        for src in sorted(atlas_root.glob(pattern)):
            if src.is_file(): cp(src,assets/'audits'/src.name,inv,'numerical audit')
    blumen=root/'KH_RT_Blumen/subsonic'
    if blumen.is_dir():
        for src in sorted(blumen.glob('*.csv')): cp(src,assets/'source_data/blumen/subsonic'/src.name,inv,'raw Blumen isoline')
    audit=root/'assets/pinn_subsonic/asset_audit'
    if audit.is_dir():
        for src in sorted(audit.glob('*')):
            if src.is_file(): cp(src,assets/'manifests'/src.name,inv,'asset audit')

    write_readmes(code,assets)
    invdf=pd.DataFrame(inv); invdf.to_csv(assets/'manifests/file_inventory.csv',index=False)
    lines=[]
    for base in [code,assets]:
        for p in sorted(base.rglob('*')):
            if p.is_file() and p.name!='checksums.sha256': lines.append(f'{digest(p)}  {p.relative_to(root)}')
    (assets/'manifests/checksums.sha256').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    missing=model_df[model_df.checkpoint_status.ne('present')]
    summary={'code_directory':str(code),'asset_directory':str(assets),'catalogue_rows':len(cat),'model_checkpoints_present':int(model_df.checkpoint_status.eq('present').sum()),'model_checkpoints_missing':len(missing),'files_materialized':len(invdf),'model_mode':a.model_mode}
    (assets/'manifests/build_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    if len(missing):
        print('\nMissing checkpoints:\n'+missing.to_string(index=False))
        if a.strict_models: raise SystemExit(3)


if __name__=='__main__': main()
