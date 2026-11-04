"""Read complete planned logs; retain all steps and local-clock durations."""
import argparse
import json
from pathlib import Path
from scripts.v3_benchmark_remote import stats
from scripts.v3_planned_remote import PlannedRotation,audit_lifecycle


def summarize(root,status):
    role_map=json.loads((root/'role_map.json').read_text());s=PlannedRotation.__new__(PlannedRotation)
    data={}
    for role,node in role_map.items():
        rid,g={'A':(0,0),'B':(1,0),'C':(1,1)}[role]
        directory=root/('physical_'+node)/f'rid{rid}_g{g}'
        if not directory.exists():continue
        data[role]={p.name:[json.loads(line) for line in p.read_text().splitlines() if line] for p in directory.glob('events_*.jsonl')}
    s.role_started=set(data)
    controller=[json.loads(line) for line in (root/'controller.jsonl').read_text().splitlines() if line]
    result=dict(schema=2,lifecycle=audit_lifecycle(s,data),cycles=[],clock_policy='Per-step durations use each worker clock; switch interval/freeze-to-commit use the same healthy A rank clock. Controller observations include polling delay. Rank spreads are not independent migration repetitions. Cross-host timestamps are not subtracted.',performance_claim=False,numerical_diagnostic=bool(status.get('anchor')))
    all_commits=[e for role in data for e in s.rows_role(data,role,'commit')]
    all_steps=[]
    for step in sorted({e['step'] for e in all_commits}):
        rows=[e for e in all_commits if e['step']==step]
        seconds=max(e['seconds'] for e in rows)
        intervals=[e['commit_interval_seconds'] for e in rows if e.get('planned_role')=='A' and e.get('commit_interval_seconds') is not None]
        interval=max(intervals) if intervals else None
        all_steps.append(dict(step=step,seconds=seconds,commit_interval_seconds=interval,
                              samples_per_second=8/seconds,interval_samples_per_second=8/interval if interval else None))
    result['all_steps']=all_steps
    result['all_step_seconds']=stats([e['seconds'] for e in all_steps]) if all_steps else None
    intervals=[e['commit_interval_seconds'] for e in all_steps if e['commit_interval_seconds'] is not None]
    result['all_commit_intervals']=stats(intervals) if intervals else None
    elapsed=[]
    for rank in range(8):
        commits=[e for e in s.rows_role(data,'A','commit') if e['rank']==rank]
        begins=[e for e in s.rows_role(data,'A','begin') if e['rank']==rank]
        if commits and begins:
            first=min(begins,key=lambda e:e['monotonic'])
            start=first['monotonic']-first['boundary_sync_seconds']-first['quorum_seconds']
            elapsed.append(max(e['monotonic'] for e in commits)-start)
    result['full_run_healthy_A_seconds']=max(elapsed) if elapsed else None
    result['full_run_samples_per_second']=8*len(all_steps)/max(elapsed) if elapsed else None
    result['measurement_policy']='Step seconds include iteration boundary/quorum, compute and required hash audit through commit. Complete same-rank commit intervals additionally retain interiteration work. First step has no prior commit interval but is retained in all_steps, all_step_seconds and full_run timing. Step durations use rank maxima; intervals use healthy A ranks only, excluding standby residence on a reactivated role. Neither is an independent repetition.'
    for cycle in status.get('cycles',[]):
        r=cycle['round'];cut=cycle['cut'];source=cycle['source'];target=cycle['target'];goal=cycle['goal']
        commits={role:s.rows_role(data,role,'commit',r) for role in ('A',source,target)}
        delays=s.rows_role(data,source,'planned_compute_delay',r)
        delayed_steps={e['step']+1 for e in delays}
        # Identify the first local observation of the request without combining clocks.
        requests=s.rows_role(data,'A','planned_request_seen',r)
        request_step=min((e['step']+1 for e in requests),default=cut)
        phases={'before_request':[],'preparation':[],'switch':[],'post_switch':[]}
        step_rows={}
        for step in sorted({e['step'] for rows in commits.values() for e in rows}):
            rows=[e for rs in commits.values() for e in rs if e['step']==step]
            seconds=max(e['seconds'] for e in rows)
            intervals=[e['commit_interval_seconds'] for e in rows if e.get('planned_role')=='A' and e.get('commit_interval_seconds') is not None]
            phase='before_request' if step<request_step else 'preparation' if step<=cut else 'switch' if step==cut+1 else 'post_switch'
            item=dict(step=step,phase=phase,seconds=seconds,samples_per_second=8/seconds,commit_interval_seconds=max(intervals) if intervals else None,interval_samples_per_second=8/max(intervals) if intervals else None,injected_slow_step=step in delayed_steps)
            step_rows[step]=item;phases[phase].append(seconds)
        stop=[];interval=[];gradient_wait=[]
        for rank in range(8):
            a={e['step']:e for e in commits['A'] if e['rank']==rank}
            freeze=[e for e in s.rows_role(data,'A','planned_final_freeze',r,cut) if e['rank']==rank]
            local=[e for e in s.rows_role(data,'A','planned_local_gradient_ready',r,cut) if e['rank']==rank]
            if cut in a and cut+1 in a:interval.append(a[cut+1]['monotonic']-a[cut]['monotonic'])
            if freeze and cut+1 in a:stop.append(a[cut+1]['monotonic']-freeze[0]['monotonic'])
            if freeze and local:gradient_wait.append(freeze[0]['monotonic']-local[0]['monotonic'])
        breakdown={}
        for role in ('A',source,target):
            by_kind={};transport_counts={}
            for e in [e for values in data[role].values() for e in values if e.get('planned_round')==r]:
                if e['kind']=='commit':continue
                transport_label=None
                if e['kind'] in ('transport_send','transport_recv'):
                    key=(e['kind'],e['rank']);index=transport_counts.get(key,0)
                    transport_counts[key]=index+1
                    transport_label=e['kind']+'.'+('precopy' if index==0 else 'gradient_tail' if index==1 else str(index))
                for field,value in e.items():
                    if (field=='seconds' or field.endswith('_seconds')) and isinstance(value,(int,float)):
                        key=e['kind'] if field=='seconds' else e['kind']+'.'+field
                        if e['kind']=='phase':key+='.'+str(e.get('phase',e.get('name','unknown')))
                        by_kind.setdefault(key,[]).append(value)
                        if transport_label and field=='seconds':by_kind.setdefault(transport_label,[]).append(value)
            for label,start,end in [('precopy_window_to_sent','planned_precopy_window_open','planned_precopy_sent'),
                                    ('target_load_to_relay_applied','planned_precopy_loaded','planned_gradient_received'),
                                    ('source_freeze_to_relay_sent','planned_final_freeze','planned_gradient_sent')]:
                spans=[]
                for rank in range(8):
                    local=[e for values in data[role].values() for e in values if e.get('planned_round')==r and e.get('rank')==rank]
                    a=next((e['monotonic'] for e in local if e['kind']==start),None)
                    b=next((e['monotonic'] for e in local if e['kind']==end),None)
                    if a is not None and b is not None:spans.append(b-a)
                if spans:by_kind[label]=spans
            breakdown[role]={k:dict(count=len(v),p50=stats(v)['p50'],p95=stats(v)['p95'],maximum=max(v)) for k,v in by_kind.items()}
        def observed(kind):
            return next((e['monotonic'] for e in controller if e['kind']==kind and 'monotonic' in e),None)
        request=observed('REQUEST') if r==0 else next((e['monotonic'] for e in controller if e['kind']=='round_prepare_requested' and e['round']==r),None)
        ready=observed('c_ready' if r==0 else f'r{r}_ready');joint=observed(f'r{r}_first_joint')
        source_alive=[]
        for rank in range(8):
            kinds=('ready','planned_retired_alive','planned_standby_verified')
            rows={k:[e for e in s.rows_role(data,source,k) if e['rank']==rank] for k in kinds}
            source_alive.append(all(rows.values()) and len({rows[k][-1]['pid'] for k in kinds})==1)
        result['cycles'].append(dict(**cycle,source_original_pids_alive_through_standby=all(source_alive),all_steps=list(step_rows.values()),phases={k:stats(v) for k,v in phases.items() if v},
            switch_critical_seconds=max(interval) if interval else None,switch_A_rank_seconds=interval,
            freeze_to_joint_critical_seconds=max(stop) if stop else None,freeze_to_joint_A_rank_seconds=stop,
            A_local_gradient_to_fence_rank_seconds=gradient_wait,observed_delay_seconds=[e['seconds'] for e in delays],durations=breakdown,
            catchup_policy='Full precopy plus current gradient/RNG/buffer tail relay; no multi-step replay. Target load-to-relay and transport_recv durations include waiting; precopy and gradient_tail transport labels follow the two ordered p2 transfers per rank.',
            phase_commit_intervals={k:stats([e['commit_interval_seconds'] for e in step_rows.values() if e['phase']==k and e['commit_interval_seconds'] is not None]) for k in phases if any(e['phase']==k and e['commit_interval_seconds'] is not None for e in step_rows.values())},
            observed_preparation_seconds=ready-request if request is not None and ready is not None else None,
            observed_request_to_first_joint_seconds=joint-request if request is not None and joint is not None else None))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('session',type=Path);p.add_argument('status',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=summarize(a.session,json.loads(a.status.read_text()));a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(lifecycle=report['lifecycle'],cycles=[{k:v for k,v in x.items() if k not in ('all_steps','durations','observed_delay_seconds')} for x in report['cycles']])))


if __name__=='__main__':main()
