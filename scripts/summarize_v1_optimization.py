"""Reduce saved, independently audited recovery experiments; no GPU work."""
import json
from pathlib import Path
import statistics

from scripts.summarize_quick_validation import summarize


def describe(values):
    return dict(n=len(values), min=min(values), mean=statistics.fmean(values), max=max(values)) if values else dict(n=0)


def v1_details(value):
    timing = value['timing']
    lifecycle = timing['worker_lifecycle_events']
    spans, gaps = {}, []
    joint = value['faults'][0]['first_joint_step']
    for worker, events in lifecycle.items():
        by_kind = {}
        for event in events:
            by_kind.setdefault(event['kind'], []).append(event)
        if '_g1_' in worker:
            for label, left, right in [
                ('recv_to_load_callback', 'transport_recv', 'load_state_begin'),
                ('recv_to_loaded', 'transport_recv', 'state_loaded'),
                ('ready_to_restored', 'ready', 'restored'),
            ]:
                if left in by_kind and right in by_kind:
                    spans.setdefault(label, []).append(by_kind[right][0]['local_monotonic'] - by_kind[left][0]['local_monotonic'])
        else:
            commits = timing['worker_step_timings'][worker]['commits']
            gaps.extend(row['commit_interval_seconds'] for row in commits
                        if row['step'] == joint and row.get('commit_interval_seconds') is not None)
    return dict(local_spans={key: describe(values) for key, values in spans.items()},
                donor_commit_gap_to_first_joint=describe(gaps),
                controls={key: value['status'].get(key) for key in ['seed', 'trigger', 'post_steps', 'communication_timeout', 'cold_launcher', 'digest_engine', 'overlap_cpu_model_load', 'recovery_gradient_only', 'overlap_cold_cleanup']})


def main():
    root = Path(__file__).resolve().parents[1] / 'evidence/v1_optimization_20261104'
    runs = []
    for path in sorted(root.glob('*_timing.json')):
        value = json.loads(path.read_text())
        row = summarize(path)
        if row['type'] == 'v1':
            row['v1_details'] = v1_details(value)
        runs.append(row)
    report = dict(schema=1, runs=runs,
                  failures=[dict(file=path.name, name=json.loads(path.read_text())['name']) for path in sorted(root.glob('failure_*.json'))],
                  limitations=['Shared company workload changed; historical times are not matched performance controls.',
                               'Original 2s-deadline baseline failures retained. Later comparisons explicitly declare the 30s deadline.',
                               'Few runs and short post windows cannot establish absence of performance regression.',
                               'Controller intervals include SSH/polling; worker-local spans overlap and cannot be added across hosts.'])
    report['selected_profile'] = dict(cold_launcher=True, digest_engine='staged',
                                    overlap_cpu_model_load=True, communication_timeout=2,
                                    training_algorithm_changed=False)
    report['additional_candidates_rejected'] = [
        dict(name='o2_grad0', functional_pass=True, seconds=25.068779447115958,
             reason='Native gradient-only recovery is correct but slower; cold compute moved onto the critical path.'),
        dict(name='o3_early0', functional_pass=True, seconds=21.238761194050312,
             reason='0.025s difference from21.264s is within controller observation uncertainty.')]
    report['user_comparison_requirement'] = ('One common training engine/workload with different recovery decisions. '
        'Existing v1/v2/v4 entries, batches, data, audit frequency and physical placement differ; '
        'these quick tests do not establish cross-strategy performance rankings.')
    report['retest_decision'] = 'User canceled redundant reruns after exact restoration of the already audited selected sources.'
    (root/'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    for row in runs:
        stats = row['step_statistics']
        print(row['name'], row['functional_pass'],
              round(row['request_to_first_joint_observed_seconds'], 3),
              {key: round(stats[key]['max_worker_seconds'].get('mean', 0), 3)
               for key in ['warm_before_request', 'first_recovered_step', 'subsequent_recovered_steps']})


if __name__ == '__main__':
    main()
