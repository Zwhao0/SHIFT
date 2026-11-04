"""Compact retained summaries for a bounded round; input raw is regenerable."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from scripts.summarize_v3_timeline import analyze as timeline_analysis, occupancy


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostic(root):
    rows = []
    for path in sorted(root.glob('physical_node*/rid0_g0/timeline_step11_r*.json.gz')):
        report = timeline_analysis(path)
        events = json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
        scopes = [e for e in events if e.get('ph') == 'X' and 'dur' in e]
        schedule = next(e for e in scopes if e['name'] == 'shift::schedule')
        origin = schedule['ts']
        nccl = [e for e in scopes if e.get('cat') == 'kernel' and 'nccl' in e['name'].lower()]
        groups = {}
        for e in nccl:
            a = e.get('args', {})
            key = (a.get('Process Group Name', 'unlabelled'), a.get('Process Group Ranks', 'unknown'))
            groups.setdefault(key, []).append(e)
        large = [e for e in nccl if e.get('args', {}).get('In msg nelems', 0) > 5_000_000
                 or 'Process Group Name' not in e.get('args', {}) and 'AllReduce' in e['name']]
        outer = [e for e in scopes if e['name'] == 'torchft::manager::allreduce']
        rows.append(dict(rank=report['rank'] if 'rank' in report else int(path.name.split('_r')[1].split('.')[0]),
            trace_sha256=sha(path), phases=report['phases'], gpu_total=report['gpu_total'],
            outer_cpu_ms=[e['dur']/1000 for e in outer],
            communication_groups=[dict(group=k[0], ranks=k[1], **occupancy(v)) for k,v in groups.items()],
            large_allreduces=[dict(start_ms=(e['ts']-origin)/1000, duration_ms=e['dur']/1000,
                elements=e.get('args', {}).get('In msg nelems'),
                dtype=e.get('args', {}).get('dtype'), stream=e.get('args', {}).get('stream'),
                group=e.get('args', {}).get('Process Group Name'),
                ranks=e.get('args', {}).get('Process Group Ranks')) for e in large],
            memcpy_counts=dict(Counter(e['name'] for e in scopes if e.get('cat') == 'gpu_memcpy')),
            limitations=report['limitations']))
    if len(rows) != 2:
        raise ValueError('exactly two selected-rank traces required')
    analysis = json.loads((root/'analysis.json').read_text())
    return dict(session=root.name, performance_evidence=False, ranks=rows,
                config_sha256=sha(root/'config.json'),
                functional_pass=analysis['functional_pass'], all_step_seconds=analysis['step_seconds'],
                all_commit_intervals_seconds=analysis['intervals_seconds'],
                outer_mapping_limit='TorchFT internal-thread NCCL kernels lack CPU correlation/group labels; two unlabelled tail allreduces are temporal attribution, not exact call ancestry.')


def formal(root, suffix):
    status = json.loads((root/f'v3_b1_20261103_{suffix}/status.json').read_text())
    if not status['done']:
        raise ValueError('controller must have ended before summarizing')
    runs = []
    for entry in status['rows']:
        run = root/entry['session']
        analysis = json.loads((run/'analysis.json').read_text())
        control = []
        commits_by_step = {}
        sources = {}
        for rid in range(2):
            for rank in range(8):
                path = run/f'node{rid}/events_g0_r{rank}.jsonl'
                sources[str(path.relative_to(root))] = sha(path)
                rows = [json.loads(l) for l in path.read_text().splitlines()]
                for e in rows:
                    if e['kind'] == 'commit':
                        commits_by_step.setdefault(e['step'], []).append(e)
                commits = [e for e in rows if e['kind'] == 'commit' and 11 <= e['step'] <= 80]
                begins = [e for e in rows if e['kind'] == 'begin' and 10 <= e['step'] < 80]
                fields = ('decision_seconds','cross_dispatch_cpu_seconds','cross_wait_h2d_cpu_seconds')
                control.append(dict(role=[rid,rank],
                    commit_means_ms={f:statistics.fmean(e[f] for e in commits)*1000
                                     for f in fields if all(f in e for e in commits)},
                    boundary_means_ms={f:statistics.fmean(e[f] for e in begins)*1000
                                       for f in ('quorum_seconds','boundary_sync_seconds')}))
        partitions = []
        for step, candidates in sorted(commits_by_step.items()):
            if len(candidates) != 16:
                raise ValueError('complete run must have all sixteen roles')
            e = max(candidates, key=lambda x:x['seconds'])
            fields = ('schedule_seconds','gradient_check_seconds','decision_seconds',
                      'optimizer_enqueue_seconds','final_sync_seconds')
            parts = {f:e[f] for f in fields}
            parts.update(pre_compute_seconds=e['seconds']-e['compute_seconds'],
                         post_compute_residual_seconds=e['compute_seconds']-sum(parts.values()))
            partitions.append(dict(step=step,role=[e['rid'],e['rank']],seconds=e['seconds'],parts=parts))
        selected = partitions[10:80]
        means = {f:statistics.fmean(e['parts'][f] for e in selected) for f in selected[0]['parts']}
        runs.append(dict(session=entry['session'], mode=entry['mode'], functional_pass=analysis['functional_pass'],
            windows=analysis['windows'], commit_intervals=analysis['fixed11_80_commit_intervals'],
            elapsed_seconds=analysis['fixed11_80_elapsed_seconds'],
            all_step_seconds=analysis['step_seconds'], all_commit_intervals_seconds=analysis['intervals_seconds'],
            critical_rank_partitions=means, all_critical_rank_steps=partitions,
            per_role_control=control, source_sha256=sources,
            config_sha256=sha(run/'config.json'), unit_cleanup=json.loads((run/'unit_cleanup.json').read_text())))
    incomplete = []
    complete_names = {r['session'] for r in runs}
    for run in sorted(root.glob(f'v3b1_{suffix}_*')):
        if not run.is_dir() or run.name in complete_names:
            continue
        roles = []
        for path in sorted(run.glob('node[01]/events_g0_r*.jsonl')):
            events = [json.loads(l) for l in path.read_text().splitlines()]
            origin = events[0]['monotonic']
            fields = ('kind','step','error','seconds','schedule_seconds','gradient_check_seconds',
                      'decision_seconds','cross_calls','finite_gradients','participants','contributed',
                      'global_batch','commit_interval_seconds')
            attempts = [dict(relative_seconds=e['monotonic']-origin,
                             **{k:e[k] for k in fields if k in e})
                        for e in events if e['kind'] in ('begin','commit','skipped','reconfigure_begin','reconfigure_end')]
            roles.append(dict(role=str(path.relative_to(run)),source_sha256=sha(path),attempts=attempts,
                              commits=sum(e['kind']=='commit' for e in events),
                              skipped=sum(e['kind']=='skipped' for e in events)))
        incomplete.append(dict(session=run.name,performance_evidence=False,roles=roles,
            log_sha256={str(p.relative_to(run)):sha(p) for p in run.glob('physical_node*/rid*/torchrun*.log')},
            unit_cleanup=json.loads((run/'unit_cleanup.json').read_text())))
    return dict(done=status['done'],success=status['success'],planned_runs=len(status['schedule']),
                incomplete_runs=incomplete, failure_reason='Cross-replica NCCL aborted; should_commit rejected nine consecutive attempts' if incomplete else None,
                schedule=status['schedule'], seed=status['seed'], fixed_window=status['fixed_window'],
                complete_run_rank_steps=len(runs)*80*16, comparison=status.get('comparison'), runs=runs,
                attribution_note='CPU partitions use the same maximum-duration rank each step. GPU spans overlap and are never added. Failed attempts retained separately; no paired score for an incomplete schedule.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results'))
    p.add_argument('--diagnostics', nargs='+', required=True)
    p.add_argument('--suffix', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    output = dict(baseline_tag='v3-20261104.1', baseline_commit='8ee421cada7b1986fc0f0bed7d1c1dab42f7d1e3',
        training_algorithm_changed=False, performance_pass=False,
        diagnostics=[diagnostic(a.root/name) for name in a.diagnostics], formal=formal(a.root,a.suffix),
        source_sha256={name:sha(Path(name)) for name in
            ('shift_full/train.py','shift_full/bucket_finalizer.py','shift_full/compat.py',
             'scripts/v3_benchmark_remote.py','scripts/start_v3_benchmark.py','scripts/summarize_steady_round.py')},
        retention='All 80 step distributions, same-rank attribution and input SHAs retained. Regenerable raw/traces removed after analysis; summaries cannot reconstruct traces.')
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),runs=len(output['formal']['runs']),
                          comparison=output['formal']['comparison'])))


if __name__ == '__main__':
    main()
