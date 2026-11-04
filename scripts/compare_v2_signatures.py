"""Compare nine-state initial and pre-cut signatures across audited V2 runs."""
import argparse
import hashlib
import json
from pathlib import Path


def signatures(root, include_migrated=False, initial_only=False):
    status = json.loads((root / 'status.json').read_text())
    assert status['done'] and status['functional_pass'], root.name
    cut = 0 if initial_only else status['goal'] if include_migrated else status['observed_cut']
    result = {}
    for binding in json.loads((root / 'physical_manifest.json').read_text()):
        role = binding['role']
        if role not in ('A', 'B') and not include_migrated:
            continue
        directory = root / ('physical_' + binding['physical_node']) / Path(binding['directory']).name
        for path in directory.glob('events_*.jsonl'):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if (row['kind'] == 'ready' and role in 'AB') or (row['kind'] == 'commit' and row['step'] <= cut):
                    step = 0 if row['kind'] == 'ready' else row['step']
                    key = ('B' if role=='C' else role, row['rank'], step)
                    assert key not in result, key
                    assert len(row['hashes']) == 9, key
                    result[key] = row['hashes']
    assert set(result) == {(role, rank, step) for role in 'AB' for rank in range(8) for step in range(cut + 1)}
    return cut, result


def compare(roots, include_migrated=False, initial_only=False):
    values = [signatures(root, include_migrated, initial_only) for root in roots]
    cut = min(value[0] for value in values)
    keys = sorted(key for key in values[0][1] if key[2] <= cut)
    digests = []
    mismatches = []
    for root, (_, value) in zip(roots, values):
        digests.append(dict(run=root.name, sha256=hashlib.sha256(json.dumps(
            [(key, value[key]) for key in keys], sort_keys=True).encode()).hexdigest()))
        mismatches.extend(dict(run=root.name, role=key[0], rank=key[1], step=key[2],
                               fields=[field for field in value[key] if value[key][field] != values[0][1][key][field]])
                          for key in keys if value[key] != values[0][1][key])
    return dict(passed=not mismatches, common_pre_cut_steps=cut if not include_migrated else None,
                common_committed_steps=cut, includes_migration=include_migrated,
                roles=2, ranks_per_role=8, state_fields=9,
                compared_worker_states=len(keys), signatures=digests, mismatches=mismatches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--root', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-migrated',action='store_true')
    parser.add_argument('--initial-only',action='store_true')
    args = parser.parse_args()
    assert len(args.runs) >= 2
    result = compare([args.root / run for run in args.runs], args.include_migrated, args.initial_only)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
