#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.evaluation.run_dense_supersonic_campaign as campaign


CONVERGED_POSITIVE = 'converged_repair'
CONVERGED_NEUTRAL = 'converged_neutral_bracket'
CONVERGED_ZERO = 'converged_neutral_at_alpha_zero'


def numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors='coerce')
    return result


def robust_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding='utf-8'))
    config.update(
        {
            'root_tolerance': 1.0e-8,
            'ci_floor': 1.0e-12,
            'ci_upper': max(float(config.get('ci_upper', 0.20)), 0.25),
            'cr_half_width': max(float(config.get('cr_half_width', 0.08)), 0.12),
            'ci_factor': max(float(config.get('ci_factor', 100.0)), 1000.0),
            'max_nfev': max(int(config.get('max_nfev', 100)), 180),
            'direct_ci_switch': max(float(config.get('direct_ci_switch', 1.0e-3)), 2.0e-3),
            'direct_ci_scale_floor': min(float(config.get('direct_ci_scale_floor', 1.0e-4)), 1.0e-5),
        }
    )
    return config


def nearest_reference_row(reference: pd.DataFrame, Mach: float, alpha: float) -> pd.Series:
    work = reference.copy()
    work['_distance'] = ((work['Mach'] - Mach) / 0.05) ** 2 + (
        (work['alpha'] - alpha) / 0.02
    ) ** 2
    return work.sort_values('_distance').iloc[0]


def branch_reference_rows(
    reference: pd.DataFrame, Mach: float, branch: str,
) -> list[pd.Series]:
    """
    Return candidate source rows for continuation in Mach.

    For the upper neutral boundary, the last accepted reference point is a
    poor transport seed: at a nearby Mach the same alpha can already lie
    beyond neutrality.  We therefore start from progressively deeper points
    inside the unstable branch, selected by a minimum ci and by fixed
    interior-alpha fallbacks.
    """
    mach_values = np.sort(reference['Mach'].dropna().unique())
    source_mach = float(mach_values[np.argmin(np.abs(mach_values - Mach))])
    sub = reference.loc[
        np.isclose(reference['Mach'], source_mach, atol=5e-10)
    ].copy()
    sub = sub.sort_values('alpha').reset_index(drop=True)
    if sub.empty:
        raise RuntimeError(f'No reference branch available near M={Mach}')

    if branch != 'upper':
        return [sub.iloc[0]]

    candidates: list[pd.Series] = []

    # Highest-alpha point that is still safely inside the unstable branch.
    for ci_threshold in (2.0e-2, 1.0e-2, 5.0e-3, 2.0e-3, 1.0e-3):
        eligible = sub.loc[sub['ci'] >= ci_threshold]
        if not eligible.empty:
            candidates.append(eligible.iloc[-1])

    # Fixed interior-alpha fallbacks are particularly useful for extrapolating
    # beyond M=1.90, where alpha=0.20 is already validated.
    for alpha_target in (0.20, 0.18, 0.15, 0.12, 0.10):
        candidates.append(
            sub.iloc[(sub['alpha'] - alpha_target).abs().argmin()]
        )

    # Final fallback: a point several grid cells below the upper endpoint.
    candidates.append(sub.iloc[max(0, len(sub) - 6)])
    candidates.append(sub.iloc[len(sub) // 2])

    deduplicated: list[pd.Series] = []
    seen: set[tuple[float, float, float]] = set()
    for row in candidates:
        key = (
            float(row['alpha']),
            float(row['cr']),
            float(row['ci']),
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(row)
    return deduplicated


def solve(
    *, Mach: float, alpha: float, seed_cr: float, seed_ci: float,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    seed_ci = max(
        float(seed_ci),
        max(float(config.get('ci_floor', 1e-12)) * 100.0, 1e-10),
    )
    result, attempts = campaign.solve_with_fallbacks(
        Mach=float(Mach),
        alpha=float(alpha),
        seed_cr=float(seed_cr),
        seed_ci=seed_ci,
        config=config,
    )
    if result is None:
        return None
    result = dict(result)
    result['n_attempts'] = len(attempts)
    return result


def solve_from_reference(
    reference: pd.DataFrame, Mach: float, alpha: float,
    config: dict[str, Any], seed: tuple[float, float] | None = None,
) -> dict[str, Any] | None:
    if seed is None:
        row = nearest_reference_row(reference, Mach, alpha)
        seed = (float(row['cr']), float(row['ci']))
    result = solve(
        Mach=Mach, alpha=alpha, seed_cr=seed[0], seed_ci=seed[1], config=config,
    )
    if result is not None:
        return result
    row = nearest_reference_row(reference, Mach, alpha)
    alternate = (float(row['cr']), float(row['ci']))
    if abs(alternate[0] - seed[0]) > 1e-12 or abs(alternate[1] - seed[1]) > 1e-12:
        return solve(
            Mach=Mach, alpha=alpha,
            seed_cr=alternate[0], seed_ci=alternate[1], config=config,
        )
    return None


def find_sign_bracket(values: list[dict[str, Any]], target_ci: float, alpha_blumen: float):
    usable = sorted(
        [v for v in values if v is not None and np.isfinite(v.get('ci', np.nan))],
        key=lambda item: float(item['alpha']),
    )
    candidates = []
    for left, right in zip(usable[:-1], usable[1:]):
        f_left = float(left['ci']) - target_ci
        f_right = float(right['ci']) - target_ci
        if f_left == 0.0:
            return left, left
        if f_right == 0.0:
            return right, right
        if np.sign(f_left) != np.sign(f_right):
            distance = abs(
                0.5 * (float(left['alpha']) + float(right['alpha'])) - alpha_blumen
            )
            candidates.append((distance, left, right))
    if not candidates:
        return None
    _, left, right = min(candidates, key=lambda item: item[0])
    return left, right


def solve_positive(
    *, reference: pd.DataFrame, Mach: float, alpha_blumen: float,
    target_ci: float, config: dict[str, Any], scan_step: float,
    max_halfwidth: float, alpha_tolerance: float, ci_tolerance: float,
) -> dict[str, Any]:
    center = solve_from_reference(reference, Mach, alpha_blumen, config)
    values: list[dict[str, Any]] = []
    if center is not None:
        values.append(center)
        if abs(float(center['ci']) - target_ci) <= ci_tolerance:
            return {
                'status': CONVERGED_POSITIVE,
                'Mach': Mach,
                'target_ci': target_ci,
                'alpha_blumen': alpha_blumen,
                'alpha_classical': float(center['alpha']),
                'delta_alpha': float(center['alpha']) - alpha_blumen,
                'classical_cr': float(center['cr']),
                'classical_ci': float(center['ci']),
                'delta_ci': float(center['ci']) - target_ci,
                'residual_norm': float(center['residual_norm']),
                'n_spectral_solves': 1,
                'message': 'center point already on requested level',
            }

    state = {
        -1: (float(center['cr']), float(center['ci'])) if center is not None else None,
        +1: (float(center['cr']), float(center['ci'])) if center is not None else None,
    }
    n_solves = 1 if center is not None else 0
    n_steps = int(math.ceil(max_halfwidth / scan_step))
    bracket = find_sign_bracket(values, target_ci, alpha_blumen)

    for index in range(1, n_steps + 1):
        if bracket is not None:
            break
        for direction in (-1, +1):
            alpha = alpha_blumen + direction * index * scan_step
            if alpha <= 1.0e-4 or alpha >= 0.60:
                continue
            result = solve_from_reference(
                reference, Mach, alpha, config, seed=state[direction],
            )
            n_solves += 1
            if result is not None:
                values.append(result)
                state[direction] = (float(result['cr']), float(result['ci']))
                bracket = find_sign_bracket(values, target_ci, alpha_blumen)
                if bracket is not None:
                    break

    if bracket is None:
        center_ci = float(center['ci']) if center is not None else math.nan
        return {
            'status': 'no_bracket_after_repair',
            'Mach': Mach,
            'target_ci': target_ci,
            'alpha_blumen': alpha_blumen,
            'alpha_classical': math.nan,
            'delta_alpha': math.nan,
            'classical_cr': math.nan,
            'classical_ci': math.nan,
            'delta_ci': math.nan,
            'ci_at_blumen_alpha': center_ci,
            'residual_norm': math.nan,
            'n_spectral_solves': n_solves,
            'message': f'No accepted sign bracket inside alpha_B +/- {max_halfwidth}',
        }

    left, right = bracket
    if left is right:
        final = left
    else:
        f_left = float(left['ci']) - target_ci
        for _ in range(40):
            if abs(float(right['alpha']) - float(left['alpha'])) <= alpha_tolerance:
                break
            midpoint = 0.5 * (float(left['alpha']) + float(right['alpha']))
            seed_source = left if abs(midpoint - float(left['alpha'])) <= abs(float(right['alpha']) - midpoint) else right
            trial = solve_from_reference(
                reference, Mach, midpoint, config,
                seed=(float(seed_source['cr']), float(seed_source['ci'])),
            )
            n_solves += 1
            if trial is None:
                opposite = right if seed_source is left else left
                trial = solve_from_reference(
                    reference, Mach, midpoint, config,
                    seed=(float(opposite['cr']), float(opposite['ci'])),
                )
                n_solves += 1
            if trial is None:
                break
            f_mid = float(trial['ci']) - target_ci
            if abs(f_mid) <= ci_tolerance:
                left = right = trial
                break
            if np.sign(f_mid) == np.sign(f_left):
                left = trial
                f_left = f_mid
            else:
                right = trial
        final = min(
            [left, right], key=lambda item: abs(float(item['ci']) - target_ci)
        )

    return {
        'status': CONVERGED_POSITIVE,
        'Mach': Mach,
        'target_ci': target_ci,
        'alpha_blumen': alpha_blumen,
        'alpha_classical': float(final['alpha']),
        'delta_alpha': float(final['alpha']) - alpha_blumen,
        'classical_cr': float(final['cr']),
        'classical_ci': float(final['ci']),
        'delta_ci': float(final['ci']) - target_ci,
        'residual_norm': float(final['residual_norm']),
        'n_spectral_solves': n_solves,
        'message': 'exact spectral level reconstructed by alpha bracketing',
    }


def transport_reference_seed(
    *, reference: pd.DataFrame, Mach: float, branch: str,
    config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """
    Transport an interior unstable root from a reference Mach to the target
    Mach with adaptive step halving.

    The previous implementation transported the last accepted point of the
    source branch.  That point is almost neutral, so the same alpha may already
    be stable at the target Mach.  The transport then failed even though the
    eigenbranch itself was perfectly recoverable.
    """
    failures: list[str] = []

    for row in branch_reference_rows(reference, Mach, branch):
        source_mach = float(row['Mach'])
        alpha = float(row['alpha'])
        current = {
            'Mach': source_mach,
            'alpha': alpha,
            'cr': float(row['cr']),
            'ci': float(row['ci']),
            'residual_norm': float(row.get('residual_norm', np.nan)),
        }

        if abs(source_mach - Mach) <= 5e-10:
            return alpha, current

        direction = 1.0 if Mach > source_mach else -1.0
        current_mach = source_mach
        step = min(0.01, abs(Mach - source_mach))
        minimum_step = 2.5e-4
        n_attempts = 0

        while abs(Mach - current_mach) > 5e-10:
            n_attempts += 1
            if n_attempts > 2000:
                failures.append(
                    f'alpha={alpha:.8g}: exceeded adaptive continuation limit'
                )
                break

            remaining = abs(Mach - current_mach)
            trial_step = min(step, remaining)
            target_mach = current_mach + direction * trial_step

            result = solve(
                Mach=float(target_mach),
                alpha=alpha,
                seed_cr=float(current['cr']),
                seed_ci=float(current['ci']),
                config=config,
            )

            if result is not None:
                current = result
                current_mach = float(target_mach)
                step = min(0.01, max(minimum_step, step * 1.35))
                continue

            step *= 0.5
            if step < minimum_step:
                failures.append(
                    f'alpha={alpha:.8g}: failed near M={current_mach:.8g} '
                    f'while targeting M={Mach:.8g}'
                )
                break
        else:
            return alpha, current

    detail = '; '.join(failures[-6:])
    raise RuntimeError(
        f'Could not transport {branch} interior seed to M={Mach}. {detail}'
    )


def solve_neutral(
    *, reference: pd.DataFrame, Mach: float, alpha_blumen: float,
    curve_label: str, curve_key: str, config: dict[str, Any],
    coarse_step: float, bracket_tolerance: float,
) -> dict[str, Any]:
    label = f'{curve_label} {curve_key}'.lower()
    branch = 'upper' if ('ci_sup' in label or alpha_blumen >= 0.15) else 'lower'
    start_alpha, current = transport_reference_seed(
        reference=reference, Mach=Mach, branch=branch, config=config,
    )
    n_solves = 0

    if branch == 'upper':
        lower = current
        alpha = start_alpha
        upper_alpha = math.nan
        while alpha < 0.65:
            trial_alpha = alpha + coarse_step
            trial = solve_from_reference(
                reference, Mach, trial_alpha, config,
                seed=(float(lower['cr']), float(lower['ci'])),
            )
            n_solves += 1
            if trial is None:
                upper_alpha = trial_alpha
                break
            lower = trial
            alpha = trial_alpha
        if not np.isfinite(upper_alpha):
            return {
                'status': 'no_upper_neutral_bracket', 'Mach': Mach,
                'target_ci': 0.0, 'alpha_blumen': alpha_blumen,
                'alpha_classical': math.nan, 'delta_alpha': math.nan,
                'neutral_alpha_lower': math.nan, 'neutral_alpha_upper': math.nan,
                'classical_cr': math.nan, 'classical_ci': math.nan,
                'residual_norm': math.nan, 'n_spectral_solves': n_solves,
                'neutral_branch': branch,
                'message': 'No accepted-to-failed transition found below alpha=0.65',
            }
        for _ in range(30):
            if upper_alpha - float(lower['alpha']) <= bracket_tolerance:
                break
            midpoint = 0.5 * (float(lower['alpha']) + upper_alpha)
            trial = solve_from_reference(
                reference, Mach, midpoint, config,
                seed=(float(lower['cr']), max(float(lower['ci']), 1e-10)),
            )
            n_solves += 1
            if trial is None:
                upper_alpha = midpoint
            else:
                lower = trial
        lower_alpha = float(lower['alpha'])
        estimate = 0.5 * (lower_alpha + upper_alpha)
        return {
            'status': CONVERGED_NEUTRAL, 'Mach': Mach, 'target_ci': 0.0,
            'alpha_blumen': alpha_blumen, 'alpha_classical': estimate,
            'delta_alpha': estimate - alpha_blumen,
            'neutral_alpha_lower': lower_alpha,
            'neutral_alpha_upper': upper_alpha,
            'neutral_alpha_uncertainty': 0.5 * (upper_alpha - lower_alpha),
            'classical_cr': float(lower['cr']),
            'classical_ci': float(lower['ci']),
            'residual_norm': float(lower['residual_norm']),
            'n_spectral_solves': n_solves, 'neutral_branch': branch,
            'message': 'last accepted / first failed upper-neutral bracket',
        }

    upper = current
    alpha = start_alpha
    failed_alpha = math.nan
    minimum_alpha = 1.0e-4
    while alpha > minimum_alpha:
        trial_alpha = max(minimum_alpha, alpha - coarse_step)
        trial = solve_from_reference(
            reference, Mach, trial_alpha, config,
            seed=(float(upper['cr']), max(float(upper['ci']), 1e-10)),
        )
        n_solves += 1
        if trial is None:
            failed_alpha = trial_alpha
            break
        upper = trial
        alpha = trial_alpha
        if alpha <= minimum_alpha + 1e-12:
            break

    if not np.isfinite(failed_alpha):
        estimate = 0.5 * minimum_alpha
        return {
            'status': CONVERGED_ZERO, 'Mach': Mach, 'target_ci': 0.0,
            'alpha_blumen': alpha_blumen, 'alpha_classical': estimate,
            'delta_alpha': estimate - alpha_blumen,
            'neutral_alpha_lower': 0.0,
            'neutral_alpha_upper': minimum_alpha,
            'neutral_alpha_uncertainty': 0.5 * minimum_alpha,
            'classical_cr': float(upper['cr']),
            'classical_ci': float(upper['ci']),
            'residual_norm': float(upper['residual_norm']),
            'n_spectral_solves': n_solves, 'neutral_branch': branch,
            'message': 'No lower cutoff above numerical alpha floor; boundary bracketed at alpha=0',
        }

    lower_alpha = failed_alpha
    for _ in range(30):
        if float(upper['alpha']) - lower_alpha <= bracket_tolerance:
            break
        midpoint = 0.5 * (lower_alpha + float(upper['alpha']))
        trial = solve_from_reference(
            reference, Mach, midpoint, config,
            seed=(float(upper['cr']), max(float(upper['ci']), 1e-10)),
        )
        n_solves += 1
        if trial is None:
            lower_alpha = midpoint
        else:
            upper = trial
    upper_alpha = float(upper['alpha'])
    estimate = 0.5 * (lower_alpha + upper_alpha)
    return {
        'status': CONVERGED_NEUTRAL, 'Mach': Mach, 'target_ci': 0.0,
        'alpha_blumen': alpha_blumen, 'alpha_classical': estimate,
        'delta_alpha': estimate - alpha_blumen,
        'neutral_alpha_lower': lower_alpha,
        'neutral_alpha_upper': upper_alpha,
        'neutral_alpha_uncertainty': 0.5 * (upper_alpha - lower_alpha),
        'classical_cr': float(upper['cr']), 'classical_ci': float(upper['ci']),
        'residual_norm': float(upper['residual_norm']),
        'n_spectral_solves': n_solves, 'neutral_branch': branch,
        'message': 'last failed / first accepted lower-neutral bracket',
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('positive', 'neutral'), required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--repair-index', type=int, required=True)
    parser.add_argument('--reference-csv', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--positive-scan-step', type=float, default=0.005)
    parser.add_argument('--positive-max-halfwidth', type=float, default=0.18)
    parser.add_argument('--alpha-tolerance', type=float, default=1.0e-5)
    parser.add_argument('--ci-tolerance', type=float, default=2.0e-5)
    parser.add_argument('--neutral-step', type=float, default=0.005)
    parser.add_argument('--neutral-bracket-tolerance', type=float, default=5.0e-5)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    selected = manifest.loc[
        pd.to_numeric(manifest['repair_index'], errors='coerce') == args.repair_index
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f'Expected exactly one repair target for index {args.repair_index}; found {len(selected)}'
        )
    row = selected.iloc[0]

    reference = numeric(
        pd.read_csv(args.reference_csv),
        ('Mach', 'alpha', 'cr', 'ci', 'residual_norm'),
    ).dropna(subset=['Mach', 'alpha', 'cr', 'ci'])
    config = robust_config(args.config)

    Mach = float(row['Mach'])
    alpha_blumen = float(row['alpha_blumen'])
    target_ci = float(row.get('target_ci', 0.0))
    curve_label = str(row.get('curve_label', ''))
    curve_key = str(row.get('curve_key', ''))

    if args.kind == 'positive':
        result = solve_positive(
            reference=reference, Mach=Mach, alpha_blumen=alpha_blumen,
            target_ci=target_ci, config=config,
            scan_step=args.positive_scan_step,
            max_halfwidth=args.positive_max_halfwidth,
            alpha_tolerance=args.alpha_tolerance,
            ci_tolerance=args.ci_tolerance,
        )
    else:
        result = solve_neutral(
            reference=reference, Mach=Mach, alpha_blumen=alpha_blumen,
            curve_label=curve_label, curve_key=curve_key, config=config,
            coarse_step=args.neutral_step,
            bracket_tolerance=args.neutral_bracket_tolerance,
        )

    for column in row.index:
        if column not in result:
            value = row[column]
            result[column] = value.item() if isinstance(value, np.generic) else value
    result['repair_index'] = args.repair_index
    result['repair_kind'] = args.kind

    point_root = args.output_root.resolve() / args.kind / f'point_{args.repair_index:03d}'
    point_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(point_root / 'result.csv', index=False)
    (point_root / 'metadata.json').write_text(
        json.dumps(campaign.json_safe(result), indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )

    print('=== BLUMEN ISOLINE REPAIR POINT ===')
    print(f'Kind            : {args.kind}')
    print(f'Repair index    : {args.repair_index}')
    print(f'Mach            : {Mach}')
    print(f'Blumen alpha    : {alpha_blumen}')
    print(f'Target ci       : {target_ci}')
    print(f'Status          : {result["status"]}')
    print(f'Classical alpha : {result.get("alpha_classical")}')
    print(f'Delta alpha     : {result.get("delta_alpha")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
