"""Read-only same-rank C2 critical paths; no cross-machine timestamp arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(summary_path):
    summary=json.loads(summary_path.read_text())
    suffix=summary_path.name.removesuffix('-summary.json')
    root=summary_path.parent.parent/('v3c1_'+suffix)
    role_map=json.loads((root/'role_map.json').read_text())
    cut=summary['cycles'][0]['cut']
    result=dict(suffix=suffix,cut=cut,roles={},source_sha256={})
    for role,(rid,generation) in {'A':(0,0),'B':(1,0),'C':(1,1)}.items():
        directory=root/('physical_'+role_map[role])/f'rid{rid}_g{generation}'
        ranks=[]
        for rank in range(8):
            path=directory/f'events_g{generation}_r{rank}.jsonl'
            result['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            events=[json.loads(s) for s in path.read_text().splitlines() if s]
            def one(kind,step=None):
                rows=[e for e in events if e['kind']==kind and (step is None or e.get('step')==step)]
                if len(rows)!=1:raise ValueError(f'{path}:{kind}:{step}:expected one, got {len(rows)}')
                return rows[0]
            row=dict(rank=rank,segments={},components={})
            if role=='A':
                names=[('freeze','planned_final_freeze',cut),('fenced','planned_old_epoch_fenced',cut),
                    ('manager_closed','planned_old_manager_closed',cut),('activated','planned_epoch_activated',cut),
                    ('outer','planned_ac_outer_begin',cut),('joint','commit',cut+1)]
            elif role=='B':
                names=[('freeze','planned_final_freeze',cut),('fenced','planned_old_epoch_fenced',cut),
                    ('manager_closed','planned_old_manager_closed',cut),('gradient_sent','planned_gradient_sent',cut),
                    ('retired_alive','planned_retired_alive',cut)]
            else:
                names=[('preloaded','planned_precopy_loaded',cut),('state_ready','planned_state_ready',cut),
                    ('activated','planned_epoch_activated',cut),('ready','ready',None),
                    ('gradient_received','planned_gradient_received',cut),('outer','planned_ac_outer_begin',cut),('joint','commit',cut+1)]
                restored=one('restored',cut)
                row['components'].update(restore_load_seconds=restored['load_seconds'],restore_audit_seconds=restored['audit_seconds'])
                row['components']['initial_audit_seconds']=next(e['seconds'] for e in events if e['kind']=='phase' and e['name']=='initial_audit')
            milestones={name:one(kind,step)['monotonic'] for name,kind,step in names}
            row['milestones_monotonic']=milestones
            for (left,t0),(right,t1) in zip(list(milestones.items()),list(milestones.items())[1:]):
                if t1<t0:raise ValueError('milestones out of order')
                row['segments'][left+'->'+right]=t1-t0
            if role in ('A','C'):
                commit=one('commit',cut+1)
                row['components'].update({k:commit[k] for k in ('schedule_seconds','gradient_check_seconds','decision_seconds','seconds','commit_interval_seconds')})
                # schedule includes the handoff; these components overlap segments.
            if role=='A':
                row.update(freeze_to_joint_seconds=milestones['joint']-milestones['freeze'],
                    previous_commit_to_joint_seconds=milestones['joint']-one('commit',cut)['monotonic'])
            ranks.append(row)
        result['roles'][role]=ranks
    for label,key in [('freeze','freeze_to_joint_seconds'),('commit_interval','previous_commit_to_joint_seconds')]:
        critical=max(result['roles']['A'],key=lambda r:r[key])
        result[label+'_critical_A']=dict(rank=critical['rank'],seconds=critical[key],segments=critical['segments'])
    result['policy']='Every segment uses one worker monotonic clock. Adjacent segments partition only that local interval. C preloaded->state_ready includes wait for A/B; component timings overlap segments and are not added. No clock offset or cross-host subtraction. Eight ranks are not independent repetitions. Phase manager_init is excluded as it spans receiver waiting.'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summaries',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runs=[summarize(path) for path in a.summaries]
    with a.output.open('x') as f:json.dump(dict(schema=1,runs=runs),f,indent=2);f.write('\n')
    print(json.dumps(dict(runs=len(runs),output=str(a.output),freeze_critical=[dict(suffix=r['suffix'],**r['freeze_critical_A']) for r in runs])))


if __name__=='__main__':main()
