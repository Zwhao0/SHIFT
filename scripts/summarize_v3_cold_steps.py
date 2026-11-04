"""Retain every cold-run step, retry and same-worker healthy commit interval."""
import argparse
import json
from pathlib import Path

from scripts.v3_benchmark_remote import stats


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def duration_stats(values):
    result = stats(values)
    result.pop('samples_per_second', None)
    return result


def summarize(status_path):
    status = json.loads(status_path.read_text())
    root = status_path.parent.parent / status['session']
    controller = rows(root / 'controller.jsonl')
    fault = next(e for e in controller if e['kind'] == 'rotation_verified')
    pre, joint = fault['committed_before_fault'], fault['first_joint_step']
    manifest = json.loads((root / 'physical_manifest.json').read_text())
    healthy = next(r for r in manifest if r['rid'] != fault['target'] and r['generation'] == 0)

    def events(role, rank):
        directory = root / ('physical_' + role['physical_node']) / Path(role['directory']).name
        return rows(directory / f"events_g{role['generation']}_r{rank}.jsonl")

    def phase(step):
        return ('before_fault' if step <= pre else 'during_recovery' if step < joint
                else 'first_joint' if step == joint else 'post_recovery')

    traces = {rank: events(healthy, rank) for rank in range(8)}
    commits = {rank: [e for e in trace if e['kind'] == 'commit'] for rank, trace in traces.items()}
    fields = sorted({k for trace in commits.values() for e in trace for k in e
                     if k == 'seconds' or k.endswith('_seconds')})
    all_steps = []
    for step in sorted({e['step'] for trace in commits.values() for e in trace}):
        group = [e for trace in commits.values() for e in trace if e['step'] == step]
        durations = {key: max(e[key] for e in group if isinstance(e.get(key), (int, float)))
                     for key in fields if any(isinstance(e.get(key), (int, float)) for e in group)}
        interval = durations.get('commit_interval_seconds')
        batches = {e['global_batch'] for e in group}
        if len(batches) != 1:
            raise ValueError(f'Healthy ranks disagree on batch at step {step}')
        batch = next(iter(batches))
        all_steps.append(dict(step=step, phase=phase(step), rank_count=len(group), global_batch=batch,
                              participants=sorted({e['participants'] for e in group}),
                              durations=durations,
                              interval_samples_per_second=batch/interval if interval else None))
    interval_differences = []
    full_elapsed = []
    for rank, trace in commits.items():
        for previous, current in zip(trace, trace[1:]):
            measured = current['monotonic'] - previous['monotonic']
            recorded = current.get('commit_interval_seconds')
            interval_differences.append(dict(rank=rank, step=current['step'], event_interval=measured,
                                             recorded_interval=recorded, difference_seconds=measured-recorded
                                             if recorded is not None else None))
        begin = next(e for e in traces[rank] if e['kind'] == 'begin')
        start = begin['monotonic'] - begin.get('boundary_sync_seconds', 0) - begin.get('quorum_seconds', 0)
        full_elapsed.append(trace[-1]['monotonic'] - start)
    phases = {}
    for label in ('before_fault', 'during_recovery', 'first_joint', 'post_recovery'):
        selected = [s for s in all_steps if s['phase'] == label]
        if not selected:
            continue
        interval = [s['durations']['commit_interval_seconds'] for s in selected
                    if 'commit_interval_seconds' in s['durations']]
        selected_ids = {s['step'] for s in selected if 'commit_interval_seconds' in s['durations']}
        rank_elapsed = [sum(e['commit_interval_seconds'] for e in trace if e['step'] in selected_ids)
                        for trace in commits.values()]
        interval_samples = sum(s['global_batch'] for s in selected if s['step'] in selected_ids)
        phases[label] = dict(steps=len(selected), step_seconds=duration_stats([s['durations']['seconds'] for s in selected]),
                             commit_intervals=duration_stats(interval) if interval else None,
                             interval_samples=interval_samples, same_rank_interval_elapsed_seconds=max(rank_elapsed),
                             effective_interval_samples_per_second=interval_samples/max(rank_elapsed) if interval_samples else None)
    attempts = []
    for role in manifest:
        for rank in range(8):
            for event in events(role, rank):
                if event['kind'] in ('commit', 'skipped'):
                    attempts.append(dict(physical_node=role['physical_node'], rid=role['rid'],
                                         generation=role['generation'], rank=rank,
                                         **{key: value for key, value in event.items()
                                            if key in ('kind', 'step', 'participants', 'contributed', 'finite_gradients',
                                                       'sample_ids', 'global_batch', 'fixed_batch') or key == 'seconds' or key.endswith('_seconds')}))
    return dict(schema=2, session=status['session'], healthy=healthy, committed_before_fault=pre,
                first_joint_step=joint, all_steps=all_steps, phases=phases, all_rank_attempts=attempts,
                interval_timestamp_differences=interval_differences,
                healthy_full_run_seconds=max(full_elapsed),
                healthy_full_run_samples=sum(s['global_batch'] for s in all_steps),
                healthy_full_run_samples_per_second=sum(s['global_batch'] for s in all_steps)/max(full_elapsed),
                performance_pass=False,
                policy='All steps, first step and skipped attempts retained. Phases are step-labelled by last observed '
                       'prefault commit and first joint, not exact fault timestamps. Complete intervals and full run '
                       'use each healthy worker clock. Recorded intervals sample commit_end before event emission; '
                       'event-interval differences retain logging timestamp jitter. Per-step maxima and component maxima may select different '
                       'ranks; components overlap and must not be summed. Eight ranks are not independent repeats. '
                       'Effective samples use recorded global_batch (including single-replica steps), never a constant 8. '
                       'Fault/Popen-to-joint wall-clock milestones remain separately in status with clock limitations.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('status', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.status)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(steps=len(result['all_steps']), attempts=len(result['all_rank_attempts']),
                          interval_timestamp_differences=len(result['interval_timestamp_differences']))))


if __name__ == '__main__':
    main()
