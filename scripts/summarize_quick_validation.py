"""Summarize audited quick trials; keep worker samples in referenced artifacts."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'evidence/v4'


def stats(values):
    values=sorted(x for x in values if isinstance(x,(int,float)))
    return dict(n=len(values),mean=statistics.fmean(values),median=statistics.median(values),
                min=values[0],max=values[-1]) if values else dict(n=0)


def group(rows):
    steps={}
    gaps=[]
    for r in rows:
        steps.setdefault(r['step'],[]).append(r['seconds'])
        gap=r.get('commit_interval_seconds',r.get('commit_gap'))
        if gap is not None:gaps.append(gap)
    maxima={str(k):max(v) for k,v in sorted(steps.items())}
    return dict(worker_seconds=stats([r['seconds'] for r in rows]),
                worker_commit_intervals=stats(gaps),max_worker_seconds_by_step=maxima,
                max_worker_seconds=stats(list(maxima.values())),
                worker_samples_per_step={str(k):len(v) for k,v in sorted(steps.items())})


def segments(points):
    result=[]
    for (left,a),(right,b) in zip(points,points[1:]):
        if a is not None and b is not None:
            assert b>=a,(left,right,a,b)
            result.append(dict(start=left,end=right,seconds=b-a))
    return result


def local_durations(version,t,v):
    samples={}
    def add(label,key,row):
        value=row.get(key)
        if isinstance(value,(int,float)):samples.setdefault(label,[]).append(value)
    if version=='v4':
        for row in t['phase_events']:add(row['kind'],'seconds',row)
    elif version=='v1':
        for worker,rows in t['worker_phases'].items():
            if '_g1_' in worker:
                for row in rows:add('replacement/'+row['name'],'seconds',row)
        for worker,rows in t['worker_lifecycle_events'].items():
            for row in rows:
                for key in ['seconds','load_seconds','audit_seconds']:
                    add(('replacement/' if '_g1_' in worker else 'initial/')+row['kind']+'/'+key,key,row)
    else:
        for worker in t['workers']:
            if worker['role']=='C':
                for row in worker['phases']:add('candidate/'+row['name'],'seconds',row)
        for rows in v['worker_lifecycle_events'].values():
            for row in rows:
                for key in ['seconds','load_seconds','audit_seconds']:add(row['kind']+'/'+key,key,row)
    return {key:stats(values) for key,values in sorted(samples.items())}


def summarize(path):
    v=json.loads(path.read_text());t=v['timing'];name=v.get('name',v.get('raw_directory_name'))
    if name.startswith('v4_'):
        version='v4';f=v['functional'];trigger=t['migration_request']['observed_step'];cut=f['cuts'][0]
        joint=t['first_recovery_common_commit']['step'];goal=f['final_step'];rows=t['per_worker_commits']
        elapsed=t['request_to_first_common_commit_seconds'];audit=f['independent_audit']['pass_all']
        request=t['migration_request']['monotonic'];events=[e for e in t['phase_events'] if e['worker']=='g0_A0']
        def at(kind,step=None,epoch=None):
            return next((e['monotonic'] for e in events if e['kind']==kind and
                         (step is None or e.get('step')==step) and (epoch is None or e.get('epoch')==epoch)),None)
        first=t['node0_A0']['first_recovery_commit_proxy']['monotonic']
        phases=segments([('request',request),('boundary',at('boundary_selected',cut)),
            ('snapshot_ready',at('snapshot_ready',cut)),('freeze',at('freeze',cut)),
            ('candidate_validated_decision',at('switch_decision',cut)),('inner_connected',at('connected',epoch=1)),
            ('activation',at('activated',cut)),('first_common_step_A0_proxy',first),
            ('all_rank_commit_observed',t['controller_observation']['observed_monotonic'])])
        resource=f['resources_clean'];limitations=f['limitations']
    elif name.startswith('v1_'):
        version='v1';f=v['faults'][0];trigger=t['trigger_step'];cut=f['restored_step'];joint=f['first_joint_step'];goal=joint+t['post_steps']-1
        rows=[r for worker in t['worker_step_timings'].values() for r in worker['commits']]
        elapsed=t['fault_marker_to_first_joint_observation_seconds'];audit=v['independent_audit']['pass']
        observations=t['controller_observations']
        def at(kind):return observations.get(kind,{}).get('monotonic')
        phases=segments([('kill_marker_request',t['fault_marker_request_monotonic']),
            ('kill_ack_observed',at('fault_0_ack')),('old_cleanup_observed',at('old_cleanup_0')),
            ('replacement_ready_observed',at('replacement_ready_0')),('restored_observed',at('restored_0')),
            ('first_joint_observed',t['first_joint_commit_observation']['scan_end_monotonic'])])
        resource=v['status']['resource_pass'];limitations=['Cold spare; healthy replica can commit solo during recovery.',
            'Only first joint commit has nine-state hash audit; five post-recovery commits checked.',
            'Controller observation includes polling/SSH; wrapper import precedes train import phase.']
    else:
        version='v2';f=v['independent_audit'];trigger=t['requested_trigger'];cut=t['observed_cut'];joint=cut+1;goal=t['goal']
        rows=[r for w in t['workers'] for r in w['steps'] if r['kind']=='commit']
        elapsed=t['first_joint_observation']['request_to_observed_seconds'];audit=v['functional_pass']
        events=v['controller_events']
        def at(kind):return next((e['monotonic'] for e in events if e['kind']==kind),None)
        phases=segments([(k,at(k)) for k in ['REQUEST','c_local_ready','cut_selected','first_joint_commit_observed']])
        resource=all(x['company_unchanged'] and not x['extra_gpu'] for x in v['resources_after'].values())
        limitations=['P1f chooses actual cut after preparation; request trigger and actual cut differ.',
            'Hash every step, fixed random batch8; logical-rid RNG streams differ across DP replicas.',
            'Nine-state B-to-C restore; A/C shared model/optimizer/scheduler, membership and samples independently checked.']
    common=[r for r in rows if r.get('participants',2)==2 and r.get('contributed',True)]
    groups=dict(startup_steps_1_2=group([r for r in common if r['step']<=2]),
        warm_before_request=group([r for r in common if 3<=r['step']<=trigger]),
        request_to_cut_training=group([r for r in common if trigger<r['step']<joint]),
        first_recovered_step=group([r for r in common if r['step']==joint]),
        subsequent_recovered_steps=group([r for r in common if joint<r['step']<=goal]))
    if version=='v1':groups['healthy_solo_during_recovery']=group([r for r in rows if r.get('participants')==1])
    return dict(name=name,type=version,functional_pass=bool(audit),resources_clean=bool(resource),
        requested_trigger=trigger,actual_cut_or_restore_step=cut,first_joint_step=joint,goal=goal,
        request_to_first_joint_observed_seconds=elapsed,controller_stage_seconds=phases,
        local_stage_duration_statistics=local_durations(version,t,v),
        local_stage_note='Worker-local durations can overlap; extrema across ranks are not additive controller phases.',
        step_statistics=groups,evidence_file=path.name,evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        limitations=limitations)


def main():
    paths=[EVIDENCE/(x+'_timing.json') for x in ['q4_local','q4_cross','q1_0_r1','q1_1','q2_0','q2_1']]
    runs=[summarize(p) for p in paths]
    assert all(r['functional_pass'] and r['resources_clean'] for r in runs)
    value=dict(schema=1,manifest=json.loads((EVIDENCE/'quick_random_manifest.json').read_text()),runs=runs,
        timing_semantics='Controller same-host monotonic observation, including polling/collection latency. Per-step maxima reduce durations, never cross-host clocks. Warm window excludes fixed initial steps1-2; all samples retained in linked artifacts.',
        limitations=['Two diagnostic trials per type; no significance or speedup claim.',
            'Different topology/batch/hash/standby conditions across types; compare within declared workload.',
            'Shared company hosts; no independent host failure-domain claim.'])
    (EVIDENCE/'quick_validation.json').write_text(json.dumps(value,indent=2)+'\n')
    for r in runs:
        print(r['name'],r['step_statistics']['warm_before_request']['max_worker_seconds']['mean'],
              r['step_statistics']['subsequent_recovered_steps']['max_worker_seconds']['mean'],
              r['request_to_first_joint_observed_seconds'])


if __name__=='__main__':main()
