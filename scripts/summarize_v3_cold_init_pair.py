"""Same-placement fresh initialization pair, including all steps and audits."""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(root, baseline='p11base01', candidate='p11overlap01'):
    sources, runs, events_by_run = {}, [], []
    for suffix in (baseline, candidate):
        base = root/'results'/f'v3b1_{suffix}_01_shift'
        status = root/'results'/f'v3_b1_20261103_{suffix}'/'status.json'
        sources[str(status.relative_to(root))] = hashlib.sha256(status.read_bytes()).hexdigest()
        assert json.loads(status.read_text())['success'] is True
        roles, traces = [], {}
        for rid in range(2):
            directories = list(base.glob(f'physical_*/rid{rid}_g0'))
            assert len(directories) == 1
            directory = directories[0]
            launches = [json.loads(line) for line in (directory/'supervisor.jsonl').read_text().splitlines()]
            launch = next(e for e in launches if e['kind'] == 'launched')
            for rank in range(8):
                path = directory/f'events_g0_r{rank}.jsonl'
                raw = path.read_bytes()
                sources[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
                events = [json.loads(line) for line in raw.splitlines()]
                ready = next(e for e in events if e['kind'] == 'ready')
                phases = [e for e in events if e['kind'] == 'phase']
                commits = [e for e in events if e['kind'] == 'commit']
                assert [e['step'] for e in commits] == list(range(1,13))
                assert not any(e['kind'] == 'skipped' for e in events)
                assert all(e['participants'] == 2 and e['contributed'] and e['finite_gradients'] for e in commits)
                overlap = next((e for e in events if e['kind'] == 'cpu_model_load_overlap'), None)
                roles.append(dict(rid=rid, rank=rank, physical_node=ready['physical_node'],
                    process_to_ready_same_host_wall_seconds=ready['time']-launch['process_started_time'],
                    module_to_ready_monotonic_seconds=ready['monotonic']-phases[0]['monotonic']+phases[0]['seconds'],
                    startup_phases={e['name']:e['seconds'] for e in phases}, overlap=overlap,
                    import_overlap=next((e for e in events if e['kind'] == 'cpu_import_overlap'), None),
                    overlap_cpu_model_load=ready.get('overlap_cpu_model_load', False),
                    overlap_cpu_imports=ready.get('overlap_cpu_imports', False)))
                traces[rid,rank] = dict(initial=ready['hashes'], last=commits[-1]['hashes'],
                                       sample_ids=[e['sample_ids'] for e in commits])
        runs.append(dict(suffix=suffix, roles=roles))
        events_by_run.append(traces)
    assert events_by_run[0] == events_by_run[1], 'initial/final nine hashes or twelve sample plans differ'
    pairs = []
    for a, b in zip(runs[0]['roles'], runs[1]['roles']):
        assert (a['rid'],a['rank'],a['physical_node']) == (b['rid'],b['rank'],b['physical_node'])
        pairs.append(dict(rid=a['rid'], rank=a['rank'], physical_node=a['physical_node'],
            process_to_ready_saving_seconds=a['process_to_ready_same_host_wall_seconds']-b['process_to_ready_same_host_wall_seconds'],
            groups_plus_load_saving_seconds=sum(a['startup_phases'][k]-b['startup_phases'][k]
                                                for k in ('parallel_groups','full_model_load'))))
    by_host = {}
    for host in {r['physical_node'] for r in runs[0]['roles']}:
        durations = [max(r['process_to_ready_same_host_wall_seconds'] for r in run['roles'] if r['physical_node'] == host) for run in runs]
        by_host[host] = dict(baseline_seconds=durations[0], candidate_seconds=durations[1], saving_seconds=durations[0]-durations[1])
    return dict(schema=1, runs=runs, pairs=pairs, host_ready=by_host, initial_and_final_nine_hashes_exact=True,
        all_twelve_sample_plans_exact=True, rank_commits=384, performance_pass=False,
        sources_sha256=sources,
        policy='Single baseline/candidate chronological pair; ranks are not repetitions. All twelve steps kept. '
        'Same-host wall includes Popen launcher; module monotonic separately retained. No cross-host timestamp subtraction. '
        'Final hashes do not prove every intermediate update or flat equivalence. No fault/performance release gate pass.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', default='p11base01')
    parser.add_argument('--candidate', default='p11overlap01')
    args = parser.parse_args()
    for suffix in (args.baseline, args.candidate):
        if not suffix or not all(c.isalnum() or c in '_-' for c in suffix):
            parser.error('invalid suffix')
    report = analyze(Path(__file__).resolve().parents[1], args.baseline, args.candidate)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(host_ready=report['host_ready'], hashes_exact=True)))


if __name__ == '__main__':
    main()
