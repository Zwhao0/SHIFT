"""Small derived summary of an immutable fixed-tolerance comparison report."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    rows = []
    totals = {}
    for step, ranks in sorted(report['byrank'].items(), key=lambda item: int(item[0])):
        for rank, pair in sorted(ranks.items(), key=lambda item: int(item[0])):
            for category, result in pair.get('by_category', {}).items():
                metrics = result['metrics']
                row = dict(step=int(step), rank=int(rank), category=category,
                           passed=result['pass'], **{k: metrics.get(k) for k in
                           ('max_abs', 'max_rel', 'relative_l2', 'strict_equal')},
                           errors=result['errors'], top_differences=result['top_differences'])
                rows.append(row)
                total = totals.setdefault(category, dict(passed=0, failed=0, maxima={}))
                total['passed' if row['passed'] else 'failed'] += 1
                for metric in ('max_abs', 'max_rel', 'relative_l2'):
                    value = row[metric]
                    if value is not None and value > total['maxima'].get(metric, {}).get('value', -1):
                        total['maxima'][metric] = dict(value=value, step=int(step), rank=int(rank))
    return dict(source_sha256=hashlib.sha256(raw).hexdigest(), complete=report['complete'],
                passed=report['pass'], tolerances=report['tolerances'],
                rank_steps=sum(len(ranks) for ranks in report['byrank'].values()),
                by_category=totals, rows=rows,
                note='Derived summary; per-category maxima are maxima over rank-steps, not pooled L2. '
                     'Null metrics remain in rows. Source report contains all retained failure details.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.report)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('complete', 'passed', 'rank_steps', 'by_category')}))


if __name__ == '__main__':
    main()
