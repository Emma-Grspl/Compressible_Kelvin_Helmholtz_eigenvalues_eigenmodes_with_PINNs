#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    asset_root = args.asset_root.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    if work_root.exists() and args.overwrite:
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    positive_path = asset_root / 'blumen_positive_true_classical_isolines.csv'
    neutral_path = asset_root / 'blumen_neutral_true_classical_line.csv'
    if not positive_path.is_file():
        raise FileNotFoundError(positive_path)
    if not neutral_path.is_file():
        raise FileNotFoundError(neutral_path)

    positive = pd.read_csv(positive_path)
    neutral = pd.read_csv(neutral_path)

    if 'blumen_row_id' not in positive.columns:
        positive['blumen_row_id'] = np.arange(len(positive), dtype=int)
    if 'blumen_row_id' not in neutral.columns:
        neutral['blumen_row_id'] = np.arange(len(neutral), dtype=int)

    positive_failed = positive.loc[
        ~positive['status'].astype(str).str.startswith('converged')
    ].copy().reset_index(drop=True)
    positive_failed['repair_index'] = np.arange(len(positive_failed), dtype=int)

    neutral_targets = neutral.copy().reset_index(drop=True)
    neutral_targets['repair_index'] = np.arange(len(neutral_targets), dtype=int)

    positive_failed.to_csv(work_root / 'positive_failed_manifest.csv', index=False)
    neutral_targets.to_csv(work_root / 'neutral_repair_manifest.csv', index=False)

    metadata = {
        'asset_root': str(asset_root),
        'work_root': str(work_root),
        'n_positive_failed': int(len(positive_failed)),
        'n_neutral_targets': int(len(neutral_targets)),
        'status': 'PASS',
    }
    (work_root / 'repair_target_metadata.json').write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )

    print('=== BLUMEN ISOLINE REPAIR TARGETS ===')
    print(f'Positive failed : {len(positive_failed)}')
    print(f'Neutral targets : {len(neutral_targets)}')
    print(f'Written to      : {work_root}')
    print('TARGET STATUS   : PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
