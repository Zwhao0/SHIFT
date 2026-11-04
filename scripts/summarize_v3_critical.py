"""Offline same-rank attribution of each benchmark's actual maximum-duration step."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(root, suffix):
    status_path = root / f'v3_b1_20261103_{suffix}/status.json'
    status = json.loads(status_path.read_text())
    if not status['done'] or not status['success']:
        raise ValueError('completed successful benchmark required')
    runs = []
    fields = ('schedule_seconds', 'gradient_check_seconds', 'decision_seconds',
              'optimizer_enqueue_seconds', 'final_sync_seconds')
    for run in status['rows']:
        sources, commits = {}, {}
        for rid in range(2):
            for rank in range(8):
                p = root / run['session'] / f'node{rid}/events_g0_r{rank}.jsonl'
                raw = p.read_bytes()
                sources[str(p.relative_to(root))] = hashlib.sha256(raw).hexdigest()
                events = [json.loads(s) for s in raw.splitlines() if s.strip()]
                rows = [e for e in events if e['kind'] == 'commit']
                if [e['step'] for e in rows] != list(range(1, 81)):
                    raise ValueError('missing/duplicate/extra commit')
                if any(e['kind'] == 'skipped' for e in events):
                    raise ValueError('skipped attempt')
                commits[rid, rank] = rows
        steps = []
        for step in range(1, 81):
            role = max(commits, key=lambda role: commits[role][step-1]['seconds'])
            e = commits[role][step-1]
            parts = {f: e[f] for f in fields}
            parts['pre_compute_seconds'] = e['seconds'] - e['compute_seconds']
            parts['post_compute_residual_seconds'] = e['compute_seconds'] - sum(parts[f] for f in fields)
            if parts['post_compute_residual_seconds'] < -1e-6:
                raise ValueError('negative partition residual')
            steps.append(dict(step=step, role=list(role), seconds=e['seconds'], parts=parts,
                              overlapping_gpu={f: e[f] for f in ('schedule_event_seconds',
                                'local_and_outer_event_seconds', 'optimizer_event_seconds') if f in e}))
        selected = steps[10:80]
        runs.append(dict(session=run['session'], mode=run['mode'], source_sha256=sources,
                         all_steps=steps, fixed11_80=dict(
                             seconds=statistics.fmean(e['seconds'] for e in selected),
                             parts={f: statistics.fmean(e['parts'][f] for e in selected) for f in selected[0]['parts']},
                             overlapping_gpu={f: statistics.fmean(e['overlapping_gpu'][f] for e in selected)
                                              for f in selected[0]['overlapping_gpu']})))
    return dict(source_status_sha256=hashlib.sha256(status_path.read_bytes()).hexdigest(), runs=runs,
                note='Each partition uses the same rank selected by maximum step seconds; all 80 steps retained. '
                'CPU intervals partition the reported duration (subtraction residual includes audit/metadata). '
                'GPU intervals overlap CPU and each other: never add them. This is attribution, not causal proof; '
                'rank/steps are not independent repeats.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix', required=True)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    a = p.parse_args()
    result = summarize(a.root, a.suffix)
    out = a.root / f'v3_b1_20261103_{a.suffix}/critical_attribution.json'
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(runs=len(result['runs']), output=str(out))))


if __name__ == '__main__':
    main()
