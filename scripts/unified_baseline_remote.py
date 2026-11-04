"""Bounded batch8 no-fault baseline on the retained runtime; no fault injection."""
import argparse
import fcntl
import hashlib
import json
import statistics
import time
from pathlib import Path

from scripts import v1_acceptance_remote as remote
from scripts.quick_v1_remote import QuickV1Rotation, resource_snapshot, read_rows
from scripts.v1_cold_remote import config as cold_config
from shift_full.control import server_password, endpoints_from_env

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix', required=True)
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--audit', action='store_true')
    p.add_argument('--policy', choices=('v1','v2'), default='v1')
    p.add_argument('--post-layout', action='store_true', help='full-replica A/C timing control')
    a = p.parse_args()
    assert a.suffix and all(c.isalnum() or c in '_-' for c in a.suffix)
    assert a.steps >= 8
    lock = (ROOT/'runtime/v3-training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    remote.PASSWORD = server_password(remote.PRIVATE/'credentials')
    (remote.PRIVATE/'credentials').unlink()
    c = cold_config()
    c['train_args'] = [x for x in c['train_args'] if x != '--audit-state']
    c['train_args'] += ['--sync-quorum', '--fixed-batch', '--data', 'random']
    c['train_args'][c['train_args'].index('--communication-timeout')+1] = '30'
    if a.audit:
        c['train_args'] += ['--audit-state', '--reference-signatures']
    c.update(target_steps=a.steps, cold_launcher=True)
    c['env'].update(SHIFT_GENERIC_PRELOAD='0', SHIFT_COMMON_GPU_WARMUP='0')
    c['lighthouse_ip'] = endpoints_from_env()['node0'].ip
    if a.policy == 'v1':
        s = QuickV1Rotation('unified_'+a.suffix, c, 600, 20261104, backup_mode='cold')
        s.bindings = {(rid, 0): s.endpoints[node] for rid, node in enumerate(('node0', 'node3' if a.post_layout else 'node1'))}
        s.config['master_addr'] = s.endpoint_for(0).ip
    else:
        from scripts.quick_v2_remote import QuickV2Rotation, config as planned_config
        c = planned_config(9, 5, a.audit, a.audit, 'p2')
        if a.audit: c['train_args'].append('--reference-signatures')
        c.update(target_steps=a.steps, cold_launcher=True)
        s = QuickV2Rotation('unified_'+a.suffix, c, 600,
                            ['node0','node3','node1'] if a.post_layout else ['node0','node1','node3'])
    baseline = json.loads((ROOT/'runtime/v3-reproduce/company_gpu_baseline.json').read_text())
    report = dict(name=s.name, policy=a.policy, post_layout=a.post_layout,
                  placement={str(r):s.endpoint_for(r).physical_node for r in range(2)},
                  audit=a.audit, steps=a.steps, global_batch=8, data='random',
                  initialization_seed=2026, source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in (ROOT/'shift_full').glob('*.py')}, pass_all=False)
    started = time.monotonic()
    try:
        for e in s.endpoints.values(): s.ensure_connection(e)
        report['resources_before'] = resource_snapshot(s, baseline, 'before')
        if a.policy == 'v1':
            s.start()
            s.await_('all_stopped', lambda d: all(s.rows(d, r, k, 'stopped')
                     for r in range(2) for k in range(8)), timeout=480)
        else:
            from scripts.v3_planned_remote import complete
            s.start_services(1)
            for role in 'AB': s.launch_role(role)
            s.wait('ab_ready',lambda d:all(complete(s.rows_role(d,r,'ready')) for r in 'AB'),timeout=480)
            for role in 'AB': s.mark_role(role,'START')
            s.wait('all_stopped',lambda d:all(complete(s.rows_role(d,r,'stopped')) for r in 'AB'),timeout=480)
        s.event('complete', faults=0)
    except BaseException as error:
        report['error'] = type(error).__name__
    finally:
        try: s.finish()
        except BaseException as error: report['cleanup_error'] = type(error).__name__
        report['resources_after'] = resource_snapshot(s, baseline, 'after')
    report['controller_seconds'] = time.monotonic()-started
    signatures, per_step = {}, {}
    try:
        for rid in (0, 1):
            binding = s.bindings[rid, 0]
            directory = s.out/('physical_'+binding.physical_node)/s.subdir(rid, 0)
            for rank in range(8):
                rows = read_rows(directory/f'events_g0_r{rank}.jsonl')
                commits = [x for x in rows if x['kind'] == 'commit']
                assert [x['step'] for x in commits] == list(range(1, a.steps+1))
                for x in commits:
                    assert x['participants'] == 2 and x['contributed'] and x['finite_gradients'] and x['global_batch'] == 8
                    expected = [(x['step']-1)*8 + m*4 + rid*2 + rank%4//2 for m in range(2)]
                    assert x['sample_ids'] == expected
                    per_step.setdefault(x['step'], []).append(x['seconds'])
                signatures[f'{rid}/{rank}'] = [dict(step=x['step'], hashes=x['hashes'], sample_ids=x['sample_ids'])
                    for x in rows if x['kind'] == 'reference_signature']
                assert not any(x['kind'] == 'skipped' for x in rows)
        report['step_max_seconds'] = {k: max(v) for k, v in per_step.items()}
        report['steady_mean_seconds'] = statistics.fmean(max(v) for k, v in per_step.items() if k >= 4)
        if a.audit:
            for rank in range(8):
                assert len(signatures[f'0/{rank}']) == a.steps
                for x, y in zip(signatures[f'0/{rank}'], signatures[f'1/{rank}']):
                    assert x['hashes'] == y['hashes']
            (s.out/'selected_signatures.json').write_text(json.dumps(signatures, separators=(',', ':'))+'\n')
        report['pass_all'] = not report.get('error') and not report.get('cleanup_error')
    except BaseException as error:
        report['audit_error'] = type(error).__name__+': '+str(error)
    (s.out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','resources_before','resources_after') }), flush=True)
    return 0 if report['pass_all'] else 1


if __name__ == '__main__': raise SystemExit(main())
