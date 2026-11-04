"""Recompute same-rank CUDA readiness windows from a diagnostic-only probe."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


FIELDS = ('producer_from_schedule_ms', 'local_done_from_schedule_ms', 'local_wait_ms',
          'producer_to_finalizer_ms', 'local_done_to_finalizer_ms', 'local_done_to_outer_done_ms')


def stats(values):
    values = sorted(values)
    return dict(count=len(values), mean=statistics.fmean(values),
                p50=statistics.median(values), p95=values[max(0, __import__('math').ceil(.95*len(values))-1)],
                minimum=values[0], maximum=values[-1])


def analyze(root):
    # Collected roots also contain physical-host copies; count logical slots once.
    files = [root/f'node{rid}'/f'events_g0_r{rank}.jsonl'
             for rid in range(2) for rank in range(8)]
    all_rows, ranks, sources = [], [], []
    for path in files:
        raw = path.read_bytes()
        sources.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
        events = [json.loads(line) for line in raw.splitlines()]
        rows = [row for row in events if row.get('kind') == 'bucket_ready_diagnostic']
        if sorted(row['step'] for row in rows) != list(range(1, 13)):
            raise ValueError('missing/duplicate diagnostic steps: '+str(path))
        commits = [row for row in events if row.get('kind') == 'commit']
        if sorted(row['step'] for row in commits) != list(range(1, 13)):
            raise ValueError('missing/duplicate commits: '+str(path))
        buckets = []
        for row in rows:
            if row['performance_evidence'] is not False:
                raise ValueError('diagnostic incorrectly labeled as performance')
            if [b['bucket'] for b in row['buckets']] != [0, 1]:
                raise ValueError('expected original two buckets')
            buckets += [dict(b, step=row['step']) for b in row['buckets']]
        all_rows += buckets
        ranks.append(dict(path=str(path), buckets={str(i):{
            key: stats([b[key] for b in buckets if b['bucket']==i]) for key in FIELDS}
            for i in (0, 1)}))
    return dict(performance_evidence=False, diagnostic_steps=192, sources=sources, ranks=ranks,
                all_steps={str(i):{key:stats([b[key] for b in all_rows if b['bucket']==i])
                                  for key in FIELDS} for i in (0,1)},
                bucket_numel={str(i):sorted({b['numel'] for b in all_rows if b['bucket']==i})
                              for i in (0,1)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = {root.name:analyze(root) for root in args.roots}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({name:row['all_steps'] for name,row in result.items()}))


if __name__ == '__main__':
    main()
