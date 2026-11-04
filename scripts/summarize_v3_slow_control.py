"""Same-rank, same-step slow/no-migration comparison; all raw steps retained."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from scripts.v3_benchmark_remote import stats


def read(root,suffix):
    status=json.loads((root/('v3_c1_20261103_'+suffix)/'status.json').read_text())
    if not status['done'] or not status['success']:raise ValueError('completed functional run required')
    directory=root/('v3c1_'+suffix)
    roles=json.loads((directory/'role_map.json').read_text())
    config=json.loads((directory/'config.json').read_text())
    source_sha={}
    data={}
    for role,rid in [('A',0),('B',1)]:
        data[role]={}
        for rank in range(8):
            p=directory/('physical_'+roles[role])/f'rid{rid}_g0/events_g0_r{rank}.jsonl'
            raw=p.read_bytes();source_sha[str(p)]=hashlib.sha256(raw).hexdigest()
            events=[json.loads(line) for line in raw.splitlines() if line.strip()]
            if any(e['kind']=='skipped' for e in events):raise ValueError('unexpected skipped attempt')
            ready=[e for e in events if e['kind']=='ready']
            if len(ready)!=1 or ready[0]['digest_engine']!='staged':raise ValueError('staged ready required')
            commits=[e for e in events if e['kind']=='commit']
            if not commits or [e['step'] for e in commits]!=list(range(1,len(commits)+1)):raise ValueError('missing/duplicate/backward commit')
            if any(not(e['participants']==2 and e['contributed'] and e['finite_gradients'] and e['global_batch']==8) for e in commits):raise ValueError('invalid joint commit')
            data[role][rank]=events
    a={rank:{e['step']:e for e in events if e['kind']=='commit'} for rank,events in data['A'].items()}
    delays={rank:[dict(step=e['step']+1,seconds=e['seconds']) for e in events if e['kind']=='planned_compute_delay'] for rank,events in data['B'].items()}
    common_delays=sorted(set.intersection(*(set(e['step'] for e in rows) for rows in delays.values())))
    return dict(suffix=suffix,roles=roles,config=config,A=a,delays=delays,common_delays=common_delays,source_sha256=source_sha)


def window(run,steps):
    ranks={rank:[events[k]['commit_interval_seconds'] for k in steps] for rank,events in run['A'].items()}
    if any(any(v is None for v in values) for values in ranks.values()):raise ValueError('complete intervals required')
    elapsed=max(map(sum,ranks.values()))
    return dict(steps=steps,rank_intervals=ranks,rank_mean={rank:statistics.fmean(values) for rank,values in ranks.items()},
        critical_same_rank_elapsed_seconds=elapsed,effective_samples_per_second=8*len(steps)/elapsed,
        per_step_max=stats([max(run['A'][rank][step]['commit_interval_seconds'] for rank in ranks) for step in steps]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True);p.add_argument('--control',required=True)
    p.add_argument('--migrations',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]/'results'
    runs={suffix:read(root,suffix) for suffix in [a.baseline,a.control,*a.migrations]}
    baseline,control=runs[a.baseline],runs[a.control]
    for run in runs.values():
        if run['roles']!=baseline['roles']:raise ValueError('physical placement differs')
        for flag in ('--seq-len','--hash-every','--bucket-size'):
            def value(x):args=x['config']['train_args'];return args[args.index(flag)+1]
            if value(run)!=value(baseline):raise ValueError('load/audit differs')
        if run['config']['env']!=baseline['config']['env']:raise ValueError('worker environment differs')
    rows=[]
    for suffix in a.migrations:
        run=runs[suffix]
        status=json.loads((root/('v3_c1_20261103_'+suffix)/'status.json').read_text())
        if not status['success']:raise ValueError('successful functional migration required')
        cut=status['cycles'][0]['cut']
        steps=sorted(set(run['common_delays'])&set(control['common_delays'])&set(range(2,cut+1)))
        if not steps:raise ValueError('no common injected preparation steps')
        m,c,b=(window(r,steps) for r in (run,control,baseline))
        rows.append(dict(suffix=suffix,cut=cut,matched_preparation=dict(migration=m,control=c,baseline=b),
            migration_vs_control_elapsed_percent=100*(m['critical_same_rank_elapsed_seconds']/c['critical_same_rank_elapsed_seconds']-1),
            migration_vs_control_rank_percent={rank:100*(m['rank_mean'][rank]/c['rank_mean'][rank]-1) for rank in m['rank_mean']},
            control_vs_normal_elapsed_ratio=c['critical_same_rank_elapsed_seconds']/b['critical_same_rank_elapsed_seconds'],
            switch_commit_interval=window(run,[cut+1])))
    report=dict(rows=rows,full_steady={suffix:window(runs[suffix],list(range(2,21))) for suffix in (a.baseline,a.control)},
        injections={suffix:dict(per_rank=run['delays'],all_ranks_common_steps=run['common_delays']) for suffix,run in runs.items()},
        sources={suffix:run['source_sha256'] for suffix,run in runs.items()},
        policy='No sleep subtraction. Same A-rank complete commit intervals at common actual injected steps; preparation excludes switch, separately retained. All original raw/summary steps remain. Controls n1; migration n2, shared control and drift prevent independent confidence or <=5/1% acceptance. Nominal old 2x is only a fixed sleep, not recalibrated 2x.')
    with a.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps([dict(suffix=r['suffix'],matched_steps=r['matched_preparation']['migration']['steps'],
        migration_vs_control_percent=r['migration_vs_control_elapsed_percent'],actual_control_vs_normal_ratio=r['control_vs_normal_elapsed_ratio']) for r in rows]))


if __name__=='__main__':main()
