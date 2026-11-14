"""Read completed hidden-test outputs and implement the stated rank formula.

This is audit analysis, never imported by a submitted candidate or evaluator.
The real future cohort is unknown, so scores use two explicit audit cohorts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
EXPERIMENT = 'scaled_synthetic_cqa'
reference = json.loads((ROOT / 'scoring/reference.json').read_text())['experiments']
spec = reference[EXPERIMENT]
fixed = spec['results']['lvsa']
attempts = {}
failures = {}
for folder in sorted(AUDIT.iterdir()):
    if not folder.is_dir() or not (folder / 'run_result.json').exists():
        continue
    run = json.loads((folder / 'run_result.json').read_text())
    if run['test_exit_code']:
        failures[folder.name] = run
        continue
    file = folder / 'scores_test.json'
    if file.exists():
        attempts[folder.name] = json.loads(file.read_text())['experiments'][EXPERIMENT]['results']['candidate']

honest = [name for name in attempts if not name.startswith('shortcut_')]
finite = [name for name, result in attempts.items()
          if all(isinstance(v, (float, int)) and math.isfinite(v) for v in result.values())]
normalizers = {}
rankings = {}
contributions = {}
for cohort_name, cohort in [('honest', honest), ('all_finite', finite)]:
    scales = {}
    for metric in spec['metrics']:
        vals = [attempts[name][metric] for name in cohort if metric in attempts[name]]
        # A constant metric has no empirically defined scale. None of the
        # evaluated finite metrics is constant in these audit cohorts.
        scales[metric] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    normalizers[cohort_name] = scales
    totals = {}
    metric_contrib = {}
    for name, result in attempts.items():
        if name not in finite:
            totals[name] = None
            continue
        terms = {}
        for metric, definition in spec['metrics'].items():
            if metric not in result or metric not in fixed or not scales[metric]:
                continue
            direction = 1 if definition['higher_is_better'] else -1
            terms[metric] = (spec['weight'] * abs(definition['coefficient']) *
                direction * (result[metric]-fixed[metric]) / scales[metric])
        totals[name] = sum(terms.values())
        metric_contrib[name] = terms
    rankings[cohort_name] = {
        'absolute_vs_fixed_targets': totals,
        'delta_vs_existing_test_reference': {
            name: (value - totals['reference_existing'] if value is not None else None)
            for name, value in totals.items()},
    }
    contributions[cohort_name] = metric_contrib

seeds = [attempts[f'reference_seed{seed}'] for seed in (11, 12, 13)]
seed_spread = {metric: max(row[metric] for row in seeds) - min(row[metric] for row in seeds)
               for metric in spec['metrics']}
seed_ranks = {}
for cohort, rank in rankings.items():
    vals = [rank['delta_vs_existing_test_reference'][f'reference_seed{seed}'] for seed in (11, 12, 13)]
    seed_ranks[cohort] = {'min': min(vals), 'max': max(vals), 'spread': max(vals)-min(vals)}

output = {
    'rank_definition': 'sum weight * abs(coefficient) * direction * (value-fixed_reference) / population_stddev(cohort); missing metrics skipped',
    'normalizer_cohorts': {'honest': honest, 'all_finite': finite},
    'normalizers': normalizers, 'rankings': rankings, 'contributions': contributions,
    'seed_metric_spreads': seed_spread, 'seed_rank_spreads': seed_ranks,
    'metrics': attempts, 'failed_runs': failures,
    'reference_bound_metrics': [metric for metric in spec['metrics']
        if (metric != 'param_count_k' and attempts['reference_existing'][metric] == 1)
        or (metric == 'param_count_k' and attempts['reference_existing'][metric] == 0)],
    'declared_experiments_not_written_by_hidden_test': [name for name in reference if name != EXPERIMENT],
}
(AUDIT / 'summary.json').write_text(json.dumps(output, indent=2, allow_nan=True)+'\n')
lines = ['| Run | MRR | Hits@1 | Trace MRR | Trace path P@1 | Params k | Rank delta (all finite) | Rank delta (honest scales) |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
for name, result in attempts.items():
    def value(v):
        if v is None:
            return 'undefined'
        # Preserve a small positive shortcut gain instead of printing a tie.
        return f'{v:.6e}' if 0 < abs(v) < 0.000001 else f'{v:.6f}'
    lines.append('| '+name+' | '+' | '.join(value(result.get(k)) for k in
        ['mrr', 'hits1', 'trace_variable_mrr', 'trace_path_p1', 'param_count_k'])+' | '+
        ' | '.join(value(rankings[c]['delta_vs_existing_test_reference'][name]) for c in ['all_finite', 'honest'])+' |')
for name in failures:
    lines.append(f'| {name} | TEST FAILED; no scores written | | | | | | |')
(AUDIT / 'results.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
print('SEED METRIC SPREADS', json.dumps(seed_spread))
print('SEED RANK SPREADS', json.dumps(seed_ranks))
