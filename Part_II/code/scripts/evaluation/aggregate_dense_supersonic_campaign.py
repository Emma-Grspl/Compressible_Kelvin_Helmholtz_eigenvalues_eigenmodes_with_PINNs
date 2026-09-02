#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False, suffix='.tmp') as handle:
        frame.to_csv(handle, index=False)
        temp = Path(handle.name)
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    config = json.loads(config_path.read_text())
    root = Path(config['output_root'])
    if not root.is_absolute():
        root = repo / root

    spectral_frames = []
    mode_frames = []
    for mach_dir in sorted(root.glob('M*')):
        spectral = mach_dir / 'spectral_points.csv'
        if spectral.exists():
            frame = pd.read_csv(spectral)
            frame['mach_directory'] = mach_dir.name
            spectral_frames.append(frame)
        mode = mach_dir / 'modes' / 'mode_summary.csv'
        if mode.exists():
            frame = pd.read_csv(mode)
            frame['mach_directory'] = mach_dir.name
            mode_frames.append(frame)

    if spectral_frames:
        spectral = pd.concat(spectral_frames, ignore_index=True, sort=False)
        atomic_csv(root / 'all_spectral_rows.csv', spectral)
        is_target = spectral['is_target'].astype(str).str.lower().isin(('true', '1'))
        targets = spectral[is_target].copy().sort_values(['Mach', 'alpha'])
        atomic_csv(root / 'dense_spectral_targets.csv', targets)
        retained = targets[targets['status'].astype(str).isin(('converged', 'anchor_converged'))].copy()
        atomic_csv(root / 'dense_spectral_retained.csv', retained)
        print(f'spectral rows: {len(spectral)}; targets: {len(targets)}; retained: {len(retained)}')
    else:
        print('No spectral files found.')

    if mode_frames:
        modes = pd.concat(mode_frames, ignore_index=True, sort=False).sort_values(['Mach', 'alpha'])
        atomic_csv(root / 'all_mode_summaries.csv', modes)
        print(f'mode summaries: {len(modes)}')
    else:
        print('No mode summaries found.')
    print(f'Written to: {root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
