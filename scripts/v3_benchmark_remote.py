"""Server-owned no-fault matched benchmarks; no acceptance.main or fault calls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess as sp
import time

from scripts import v1_acceptance_remote as remote
from shift_full.control import server_password

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = {'model', 'buffers', 'optimizer', 'scheduler', 'progress', 'cpu_rng', 'cuda_rng', 'python_rng', 'tracker_rng'}


def config(mode, steps, trace=False):
    c = json.loads((ROOT/f'configs/v1_bench_{"flat" if mode == "flat" else "shift"}.json').read_text())
    a = c['train_args']
    a[a.index('--mode')+1] = mode
    # Identical required audits and complete step boundaries in every mode.
    a += ['--audit-state', '--hash-every', str(min(20,steps))]
    if trace:
        a.append('--trace-events')
    c.update(target_steps=steps, fast_launch=False, persistent_cpu_launcher=False,
             single_gpu_visibility=False, audit_initial=True)
    c['env']['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    return c


class ServerOwnedRotation(remote.ServerRotation):
    """Use HostRotation I/O, but put every detached owner in its own cgroup."""
    def __init__(self, name, config, *args, **kwargs):
        engine=os.environ.get("SHIFT_DIGEST_ENGINE","reference")
        if engine not in ("reference","staged"):raise ValueError("invalid SHIFT_DIGEST_ENGINE")
        config.setdefault("env",{})["SHIFT_DIGEST_ENGINE"]=engine
        overlap = os.environ.get('SHIFT_OVERLAP_CPU_MODEL_LOAD', '0')
        if overlap not in ('0', '1'):raise ValueError('invalid SHIFT_OVERLAP_CPU_MODEL_LOAD')
        config['env']['SHIFT_OVERLAP_CPU_MODEL_LOAD'] = overlap
        imports = os.environ.get('SHIFT_OVERLAP_CPU_IMPORTS', '0')
        if imports not in ('0', '1'):raise ValueError('invalid SHIFT_OVERLAP_CPU_IMPORTS')
        config['env']['SHIFT_OVERLAP_CPU_IMPORTS'] = imports
        self.units = []
        super().__init__(name, config, *args, **kwargs)

    def docker(self, endpoint, cmd, timeout=45):
        if cmd[:2] != ['exec', '-d']:
            return super().docker(endpoint, cmd, timeout)
        endpoint = self.coerce_endpoint(endpoint)
        i, env, cwd = 1, {}, '/workspace/SHIFT'
        while cmd[i].startswith('-'):
            if cmd[i] == '-d':
                i += 1
            elif cmd[i] in ('-e', '-w'):
                key, value = cmd[i:i+2]
                if key == '-w': cwd = value
                else:
                    k, v = value.split('=', 1); env[k] = v
                i += 2
            else: raise ValueError('unsupported detached option')
        if cmd[i] != endpoint.container: raise ValueError('endpoint mismatch')
        args = list(cmd[i+1:])
        if args[0] == 'python': args[0] = '/usr/bin/python3'
        role = 'control' if 'shift_full.control_service' in args else 'rid'+args[args.index('--replica')+1]
        unit = f'shift-v3-{self.name}-{endpoint.physical_node}-{role}'
        env.update(PYTHONDONTWRITEBYTECODE='1', CPATH='/workspace/SHIFT/runtime/python-headers',
                   TMPDIR='/workspace/SHIFT/runtime/v1-recheck-bootstrap/cache',
                   XDG_CACHE_HOME='/workspace/SHIFT/runtime/v1-recheck-bootstrap/cache',
                   HF_HOME='/workspace/SHIFT/runtime/v1-recheck-bootstrap/cache/hf',
                   TRITON_CACHE_DIR='/workspace/SHIFT/runtime/v1-recheck-bootstrap/cache/triton')
        numerical = any(x in self.config['train_args'] for x in ('--reference-dump','--reference-anchor-save'))
        run = ['systemd-run', '--quiet', '--collect', '--unit='+unit,
               '--property=RuntimeMaxSec='+str(min(2400,max(600,self.seconds+60))), '--property=KillMode=control-group',
               '--property=TimeoutStopSec=15', '--property=LimitNOFILE=1048576',
               '--property=LimitFSIZE='+str(4*1024**3 if numerical else 134217728), '--working-directory='+cwd]
        run += ['--setenv='+k+'='+v for k,v in env.items()]
        run += args
        # Record the exact unit before launch so cleanup also handles a lost SSH ack.
        self.units.append((endpoint, unit))
        return self.ssh(endpoint, ['sudo', '-S', '-p', '', *run], timeout=timeout, sudo_input=True)

    def finish(self):
        try:
            super().finish()
        finally:
            errors = []
            for e, unit in reversed(self.units):
                try:
                    code = ('import subprocess; u='+repr(unit)+'; '
                            'subprocess.run(["systemctl","stop",u],capture_output=True,timeout=25); '
                            'r=subprocess.run(["systemctl","is-active",u],capture_output=True,text=True); '
                            'assert r.stdout.strip() not in ("active","activating","deactivating")')
                    self.ssh(e, ['sudo', '-S', '-p', '', 'python3', '-c', code], sudo_input=True)
                except Exception as exc: errors.append(dict(unit=unit, error=type(exc).__name__))
            (self.out/'unit_cleanup.json').write_text(json.dumps(dict(units=[u for _,u in self.units], errors=errors),indent=2)+'\n')
            if errors: raise RuntimeError('benchmark unit cleanup failed')


class NoFaultRotation(ServerOwnedRotation):
    def fault(self, *args, **kwargs):
        raise RuntimeError('fault injection is forbidden in the B1 benchmark')


def quantile(values, q):
    values = sorted(values)
    x = (len(values)-1)*q
    i = int(x)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(x-i)


def stats(values):
    return dict(count=len(values), mean=statistics.fmean(values), p50=quantile(values,.5),
                p95=quantile(values,.95), maximum=max(values), samples_per_second=8/statistics.fmean(values))


def analyze(root, steps):
    commits, initial, inventory = {}, {}, {}
    for rid in range(2):
        for rank in range(8):
            path = root/f'node{rid}'/f'events_g0_r{rank}.jsonl'
            events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            ready = [e for e in events if e['kind']=='ready']
            if len(ready)!=1: raise RuntimeError('unexpected ready count')
            initial[rid,rank] = ready[0]['hashes']
            inventory[f'{rid}:{rank}'] = ready[0]['state_inventory']
            rows = [e for e in events if e['kind']=='commit']
            begins = {e['step']:e for e in events if e['kind']=='begin'}
            for e in rows:
                start=begins.get(e['step']-1,{})
                for field in ('boundary_sync_seconds','quorum_seconds'):
                    if field in start: e[field]=start[field]
            if [e['step'] for e in rows] != list(range(1,steps+1)):
                raise RuntimeError('missing/duplicate/extra committed steps')
            if any(e['kind']=='skipped' for e in events): raise RuntimeError('skipped step in no-fault benchmark')
            for e in rows:
                if not (e['participants']==2 and e['contributed'] and e['finite_gradients'] and e['global_batch']==8):
                    raise RuntimeError('invalid joint commit')
                if e['step']%min(20,steps)==0 and set(e['hashes']) != SCHEMAS: raise RuntimeError('incomplete state audit')
                commits.setdefault(e['step'], {})[rid,rank] = e
    for rank in range(8):
        if initial[0,rank]!=initial[1,rank] or set(initial[0,rank])!=SCHEMAS:
            raise RuntimeError('initial state differs')
        for step, rows in commits.items():
            if rows[0,rank]['hashes'] != rows[1,rank]['hashes']:
                raise RuntimeError(f'joint audit differs at step {step} rank {rank}')
            if rows[0,rank]['sample_ids'] == rows[1,rank]['sample_ids']:
                raise RuntimeError('replicas consumed identical samples')
    seconds = {k:max(e['seconds'] for e in rows.values()) for k,rows in commits.items()}
    intervals = {k:max(e['commit_interval_seconds'] for e in rows.values()) for k,rows in commits.items() if k>1}
    windows = dict(first10=[v for k,v in seconds.items() if k<=10], full=list(seconds.values()))
    if steps>=80: windows['fixed11_80'] = [seconds[k] for k in range(11,81)]
    fields = ['boundary_sync_seconds','quorum_seconds','schedule_seconds','gradient_check_seconds',
              'decision_seconds','optimizer_enqueue_seconds','final_sync_seconds','intra_finish_cpu_seconds',
              'cross_dispatch_cpu_seconds','cross_wait_h2d_cpu_seconds','schedule_event_seconds',
              'local_and_outer_event_seconds','optimizer_event_seconds']
    # Per-rank aggregates retain attribution; sums across maxima are not a critical path.
    attribution = {}
    for role in next(iter(commits.values())):
        rows = [v[role] for k,v in commits.items() if 11<=k<=80]
        attribution['%d:%d'%role] = {f:statistics.fmean(e[f] for e in rows) for f in fields if rows and all(f in e for e in rows)}
    return dict(functional_pass=True, step_seconds=seconds, intervals_seconds=intervals,
                windows={k:stats(v) for k,v in windows.items()},
                fixed11_80_commit_intervals=stats([intervals[k] for k in range(11,81)]) if steps>=80 else None,
                fixed11_80_elapsed_seconds=max(commits[80][role]['monotonic']-commits[10][role]['monotonic']
                                              for role in commits[80]) if steps>=80 else None,
                state_inventory=inventory, attribution=attribution,
                note='All window steps retained, including hashes and slow steps; max rank duration and commit intervals reported separately.')


def matched_comparison(rows, baseline, candidate):
    """Pair chronological occurrences; full intervals/elapsed keep audited slow steps."""
    output = dict(baseline=baseline, candidate=candidate, metrics={})
    for metric, extract in (
        ('step_seconds', lambda r:r['windows']['fixed11_80']['mean']),
        ('commit_interval_seconds', lambda r:r['commit_intervals']['mean']),
        ('elapsed_seconds', lambda r:r['elapsed_seconds'])):
        means={mode:[extract(r) for r in rows if r['mode']==mode] for mode in (baseline,candidate)}
        if any(len(v)!=4 for v in means.values()):raise ValueError('need four complete runs per mode')
        paired=[100*(b/a-1) for a,b in zip(means[baseline],means[candidate])]
        mean=statistics.fmean(paired);half=3.182446*statistics.stdev(paired)/2
        output['metrics'][metric]=dict(run_values=means,pairs_percent=paired,paired_percent_mean=mean,
            paired_percent_t95=[mean-half,mean+half],
            pooled_overhead_percent=100*(statistics.fmean(means[candidate])/statistics.fmean(means[baseline])-1))
    output['confidence_note']='Exploratory t95 over four run pairs; all fixed11_80 steps retained, ranks/steps are not repetitions. Triad order and hashes frozen before launch. Framework vs adapter attribution uses comparable local paths, not full Titan equivalence.'
    return output


def run_case(name, mode, steps, seed, trace, cold_launcher=False,sync_quorum=False,eager_groups=False,ready_trace=False,timeline_trace=False):
    c=config(mode,steps,trace)
    if ready_trace:c['train_args'].append('--bucket-ready-trace')
    if timeline_trace:
        c['train_args'] += ['--timeline-trace-step', '11']
        # Profiler initialization/export delays selected ranks. No timing score;
        # keep bounded diagnostic workers while avoiding artificial FT timeouts.
        c['train_args'][c['train_args'].index('--communication-timeout')+1]='20'
    if sync_quorum:c['train_args'].append('--sync-quorum')
    if eager_groups:c['train_args'].append('--eager-groups')
    if cold_launcher:
        c['cold_launcher']=True
        c['train_args']=[arg for arg in c['train_args'] if arg!='--warmup-before-join']
        c['env'].update(SHIFT_GENERIC_PRELOAD='0',SHIFT_COMMON_GPU_WARMUP='0')
    s = NoFaultRotation(name, c, 550, seed, backup_mode='cold')
    try:
        s.start()
        s.await_('all_stopped',lambda d:all(s.rows(d,r,k,'stopped') for r in range(2) for k in range(8)),timeout=450)
        s.event('complete',faults=0)
    except BaseException as exc:
        s.event('failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        s.finish()
    result = analyze(s.out, steps)
    (s.out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix', required=True)
    p.add_argument('--schedule', choices=['probe','cold-probe','abba','native-abba','triad'], default='probe')
    p.add_argument('--seed', type=int, default=20261106)
    p.add_argument('--trace-events', action='store_true')
    p.add_argument('--bucket-ready-trace', action='store_true')
    p.add_argument('--timeline-trace', action='store_true')
    p.add_argument('--sync-quorum',action='store_true')
    p.add_argument('--eager-groups',action='store_true')
    a = p.parse_args()
    if a.bucket_ready_trace and (a.schedule!='probe' or not a.trace_events):
        p.error('bucket readiness is a trace-enabled formal-load probe only')
    if a.timeline_trace and (a.schedule!='probe' or not a.trace_events or a.bucket_ready_trace):
        p.error('timeline is a separate trace-enabled formal-load probe only')
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix): p.error('invalid suffix')
    out = ROOT/f'results/v3_b1_20261103_{a.suffix}'
    out.mkdir(exist_ok=False)
    lock = (ROOT/'runtime/v3-training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    remote.PASSWORD = server_password(remote.PRIVATE/'credentials')
    (remote.PRIVATE/'credentials').unlink()
    schedule = ['shift'] if a.schedule=='cold-probe' else ['flat','shift','nativeft'] if a.schedule=='probe' else (
        ['flat','shift','shift','flat','shift','flat','flat','shift'] if a.schedule=='abba' else
        ['shift','nativeft','nativeft','shift','nativeft','shift','shift','nativeft'] if a.schedule=='native-abba' else
        ['flat','nativeft','shift','shift','nativeft','flat','nativeft','shift','flat','flat','shift','nativeft'])
    status = dict(done=False, success=False, injections=0, schedule=schedule, seed=a.seed,
                  sync_quorum=a.sync_quorum,eager_groups=a.eager_groups, bucket_ready_diagnostic=a.bucket_ready_trace,timeline_diagnostic=a.timeline_trace, fixed_window=[11,80], trace_events=a.trace_events, rows=[], started=time.time())
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    try:
        for i,mode in enumerate(schedule):
            name=f'v3b1_{a.suffix}_{i+1:02d}_{mode}'
            result = run_case(name,mode,12 if a.schedule in ('probe','cold-probe') else 80,a.seed,a.trace_events,
                              cold_launcher=a.schedule=='cold-probe',sync_quorum=a.sync_quorum,eager_groups=a.eager_groups,ready_trace=a.bucket_ready_trace,timeline_trace=a.timeline_trace)
            status['rows'].append(dict(session=name,mode=mode,windows=result['windows'],
                commit_intervals=result.get('fixed11_80_commit_intervals'),
                elapsed_seconds=result.get('fixed11_80_elapsed_seconds')))
            (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        if a.schedule=='triad':
            status['comparisons'] = {baseline+'->'+candidate: matched_comparison(status['rows'],baseline,candidate)
                for baseline,candidate in [('flat','nativeft'),('nativeft','shift'),('flat','shift')]}
        elif a.schedule not in ('probe','cold-probe'):
            left,right = schedule[:2]
            means = {m:[r['windows']['fixed11_80']['mean'] for r in status['rows'] if r['mode']==m] for m in (left,right)}
            paired=[100*(means[right][i]/means[left][i]-1) for i in range(4)]
            half=3.182446*statistics.stdev(paired)/2
            status['comparison'] = dict(baseline=left,candidate=right,run_means=means,
                paired_percent_mean=statistics.fmean(paired),
                paired_percent_t95=[statistics.fmean(paired)-half,statistics.fmean(paired)+half],
                confidence_note='Exploratory t interval over four run pairs; steps are not independent repetitions.',
                pooled_overhead_percent=100*(statistics.fmean(means[right])/statistics.fmean(means[left])-1),
                run_stddev={m:statistics.stdev(v) for m,v in means.items()},
                pairs_percent=paired)
        status['success'] = True
    except BaseException as exc:
        status['error'] = type(exc).__name__+': '+str(exc)
        raise
    finally:
        status.update(done=True, finished=time.time())
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps({k:v for k,v in status.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    main()
