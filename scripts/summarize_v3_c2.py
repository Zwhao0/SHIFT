"""Aggregate independent fresh runs without treating ranks as repetitions."""
import argparse
import json
from pathlib import Path
import statistics

from scripts.v3_benchmark_remote import stats


def aggregate(manifests):
    runs = [dict(row, directory=p.parent) for p in manifests
            for row in json.loads(p.read_text())['runs']]
    names = [row['suffix'] for row in runs]
    if len(names) != len(set(names)):
        raise ValueError('duplicate run across manifests')
    baseline = {r['suffix']: r for r in runs if r['kind'] == 'baseline'}
    result = dict(schema=1, runs=[], independent_conditions=[],
                  performance_pass=False,
                  policy='All steps retained. Complete healthy-A commit intervals are primary. '
                         'Sleep subtraction is descriptive additive accounting, not a measured '
                         'no-migration slow control. Rank spreads are not repetitions.')
    result['baselines']=[]
    for row in baseline.values():
        summary=json.loads((row['directory']/row['summary']).read_text())
        result['baselines'].append(dict(suffix=row['suffix'],nodes=row['nodes'],
            all_steps=summary['all_steps'],all_step_seconds=summary['all_step_seconds'],
            all_commit_intervals=summary['all_commit_intervals'],
            full_run_seconds=summary.get('full_run_healthy_A_seconds'),
            full_run_samples_per_second=summary.get('full_run_samples_per_second')))
    for row in runs:
        if row['kind'] != 'slow':
            continue
        s = json.loads((row['directory']/row['summary']).read_text())
        c = s['cycles'][0]
        pre = baseline[row['calibration_baseline']]['all_intervals']
        healthy,source,target=row['nodes'].split(',')
        post=next(r['all_intervals'] for r in baseline.values() if r['nodes']==','.join((healthy,target,source)))
        slow = [x['commit_interval_seconds'] for x in c['all_steps']
                if x['phase'] == 'preparation' and x['injected_slow_step']
                and x['commit_interval_seconds'] is not None]
        phases = c['phase_commit_intervals']
        item = {k: row[k] for k in ('suffix', 'nodes', 'slow_ms', 'target_factor')}
        item.update(normal_interval_seconds=pre['mean'],
                    preparation_slow_step_count=len(slow),
                    measured_preparation_slow_ratio=statistics.fmean(slow)/pre['mean'] if slow else None,
                    preparation_seconds=c['observed_preparation_seconds'],
                    request_to_first_joint_seconds=c['observed_request_to_first_joint_seconds'],
                    switch_seconds=c['switch_critical_seconds'],
                    switch_normal_steps=c['switch_critical_seconds']/pre['mean'],
                    freeze_seconds=c['freeze_to_joint_critical_seconds'],
                    all_phases=phases,
                    post_baseline_interval_seconds=post['mean'],
                    post_mean_change_percent=100*(phases['post_switch']['mean']/post['mean']-1),
                    all_injected_rank_steps=len(c['observed_delay_seconds']),
                    full_run_seconds=s.get('full_run_healthy_A_seconds'),
                    full_run_samples_per_second=s.get('full_run_samples_per_second'),
                    source_original_pids_alive_through_standby=c.get('source_original_pids_alive_through_standby'),
                    all_steps=s['all_steps'],
                    lifecycle=s['lifecycle'])
        before=phases.get('before_request')
        if before:
            item['within_run_normal_interval_seconds']=before['mean']
            item['within_run_normal_interval_count']=before['count']
            item['within_run_normal_vs_calibration_ratio']=before['mean']/pre['mean']
            item['measured_preparation_slow_ratio_within_run']=statistics.fmean(slow)/before['mean'] if slow else None
        if slow:
            item['estimated_preparation_residual_seconds'] = statistics.fmean(slow)-pre['mean']-row['slow_ms']/1000
        result['runs'].append(item)
    for nodes, factor in sorted({(r['nodes'], r['target_factor']) for r in result['runs']}):
        rows = [r for r in result['runs'] if r['nodes'] == nodes and r['target_factor'] == factor]
        condition = dict(nodes=nodes, target_factor=factor, independent_runs=len(rows),
                         suffixes=[r['suffix'] for r in rows])
        for field in ('switch_seconds', 'freeze_seconds', 'preparation_seconds', 'request_to_first_joint_seconds'):
            values = [r[field] for r in rows if r[field] is not None]
            condition[field] = dict(stats(values), sample_stdev=statistics.stdev(values) if len(values)>1 else None) if values else None
        result['independent_conditions'].append(condition)
    result['same_condition_repeat_coverage'] = len(result['independent_conditions']) == 6 and all(c['independent_runs'] >= 2 for c in result['independent_conditions'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifests', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    result = aggregate(a.manifests)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(runs=len(result['runs']), same_condition_repeat_coverage=result['same_condition_repeat_coverage'])))


if __name__ == '__main__':
    main()
