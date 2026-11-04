"""Reuse raw cold events and existing first-F/B traces; no GPU work."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def analyze(root):
    sources = {}
    cold = []
    for index in ('cold_paths.json', 'pdcold01-path.json'):
        path = root/'results/v3_p_20261103'/index
        sources[str(path.relative_to(root))] = sha(path.read_bytes())
        for run in json.loads(path.read_text())['runs']:
            ranks = []
            directory = (root/run['source']).parent
            for rank in range(8):
                p = directory/f'events_g1_r{rank}.jsonl'
                raw = p.read_bytes()
                sources[str(p.relative_to(root))] = sha(raw)
                if rank == run['selected_rank']:
                    assert sha(raw) == run['source_sha256']
                events = [json.loads(line) for line in raw.splitlines()]
                phases = {e['name']:e['seconds'] for e in events if e['kind'] == 'phase'}
                ready = next(e for e in events if e['kind'] == 'ready')
                loaded = next(e for e in events if e['kind'] == 'state_loaded')
                recv = next(e for e in events if e['kind'] == 'transport_recv')
                attempt = next(e for e in events if e['kind'] in ('commit','skipped'))
                joint = next(e for e in events if e['kind'] == 'commit' and e['step'] == run['first_joint_step'])
                ranks.append(dict(rank=rank, startup_phases=phases,
                    groups_plus_load_seconds=phases['parallel_groups']+phases['full_model_load'],
                    ideal_overlap_ceiling_seconds=min(phases['parallel_groups'], phases['full_model_load']),
                    recv_to_loaded_seconds=loaded['monotonic']-recv['monotonic'],
                    loaded_to_local_joint_seconds=joint['monotonic']-loaded['monotonic'],
                    ready_to_local_joint_seconds=joint['monotonic']-ready['monotonic'],
                    first_attempt={k:attempt[k] for k in ('kind','step','schedule_seconds','gradient_check_seconds','decision_seconds')}))
            cold.append(dict(session=run['session'], selected_rank=run['selected_rank'], ranks=ranks,
                historical_uncalibrated_process_to_global_joint_seconds=run['historical_wall_clock_process_to_global_joint_seconds']))
    traces = []
    for path in sorted((root/'results').glob('v3b1_panchor01_*/physical_*/rid0_g0/timeline_step2_r*.json.gz')):
        compressed = path.read_bytes()
        raw = gzip.decompress(compressed)
        meta_path = path.with_suffix('').with_suffix('.metadata.json')
        metadata = json.loads(meta_path.read_text())
        assert sha(compressed) == metadata['compressed_sha256']
        assert sha(raw) == metadata['raw_sha256']
        assert metadata['committed'] is True
        sources[str(path.relative_to(root))] = sha(compressed)
        sources[str(meta_path.relative_to(root))] = sha(meta_path.read_bytes())
        events = json.loads(raw)['traceEvents']
        scope = next(e for e in events if e.get('cat') == 'user_annotation' and e.get('name') == 'shift::schedule')
        a, b = scope['ts'], scope['ts']+scope['dur']
        selected = [e for e in events if e.get('cat') == 'cpu_op' and e.get('pid') == scope['pid'] and a <= e['ts'] < b]
        calls = {}
        for name in ('_native::bmm_triton_0', 'aten::_cudnn_attention_forward', 'aten::_cudnn_attention_backward'):
            rows = sorted([e for e in selected if e['name'] == name], key=lambda e:e['ts'])
            calls[name] = dict(count=len(rows), first_ms=rows[0]['dur']/1000 if rows else None,
                first_offset_ms=(rows[0]['ts']-a)/1000 if rows else None,
                subsequent_max_ms=max((e['dur']/1000 for e in rows[1:]), default=None))
        traces.append(dict(path=str(path.relative_to(root)), schedule_cpu_scope_ms=scope['dur']/1000, calls=calls))
    assert len(cold) == 4 and len(traces) == 6
    return dict(schema=1, performance_evidence=False, sources_sha256=sources, cold=cold, first_FB_traces=traces,
        policy='Each cold rank uses its own monotonic clock. Overlap ceiling is arithmetic, not measured saving; '
        'ranks are not repetitions and phase maxima must not be added. First skipped schedule overlaps receipt/apply. '
        'P6 traces are intrusive anchor/route-audit first F/B, not pure-cold recovery or performance samples. '
        'CPU operation durations include nested work/waits and do not prove compilation, GPU execution cost or removability. '
        'Historical process-to-global-joint wall-clock limitation retained. No workload/boundary/check changes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(Path(__file__).resolve().parents[1])
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(cold_runs=len(report['cold']), traces=len(report['first_FB_traces']),
                          checked_sources=len(report['sources_sha256']), performance_evidence=False)))


if __name__ == '__main__':
    main()
