"""One target rank per cold run; monotonic adjacent segments, wall totals labelled."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(path):
    status=json.loads(path.read_text());root=path.parent.parent/status['session']
    manifest=json.loads((root/'physical_manifest.json').read_text())
    target=next(row for row in manifest if row['generation']==1)
    directory=root/('physical_'+target['physical_node'])/Path(target['directory']).name
    by_rank={}
    for rank in range(8):
        p=directory/f'events_g1_r{rank}.jsonl'
        es=[json.loads(line) for line in p.read_text().splitlines() if line]
        ready=next(e for e in es if e['kind']=='ready')
        by_rank[rank]=(es,ready,p)
    rank=max(by_rank,key=lambda r:by_rank[r][1]['monotonic'])
    es,ready,source=by_rank[rank]
    def event(kind,step=None):
        matches=[e for e in es if e['kind']==kind and (step is None or e.get('step')==step)]
        if len(matches)!=1:raise ValueError(f'{source}:{kind}:{step}:ambiguous')
        return matches[0]
    phases=[e for e in es if e['kind']=='phase' and e['monotonic']<=ready['monotonic']]
    imports=next(e for e in phases if e['name']=='python_imports')
    module_start=imports['monotonic']-imports['seconds']
    markers={'ready':ready,'request':event('recovery_request'),'recv_end':event('transport_recv'),
        'state_loaded':event('state_loaded'),'restored':event('restored'),
        'skipped_attempt':event('skipped'),'local_joint_step_commit':event('commit',status['first_joint_step'])}
    times={k:e['monotonic'] for k,e in markers.items()}
    segments={left+'->'+right:times[right]-times[left] for left,right in zip(list(times),list(times)[1:])}
    if any(x<0 for x in segments.values()):raise ValueError('milestones out of order')
    return dict(session=status['session'],target_node=target['physical_node'],selected_rank=rank,
        rank_selection='Latest target ready monotonic timestamp (all ranks share one physical host)',
        source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        status_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        historical_wall_clock_process_to_global_joint_seconds=status['process_start_to_joint_seconds'],
        module_start_to_ready_seconds=ready['monotonic']-module_start,
        ready_to_local_joint_step_seconds=times['local_joint_step_commit']-times['ready'],
        first_joint_step=status['first_joint_step'],milestones_monotonic=times,segments=segments,
        startup_phases={e['name']:e['seconds'] for e in phases},
        overlapping_components=dict(recv_seconds=markers['recv_end']['seconds'],load_seconds=markers['state_loaded']['load_seconds'],
            restore_audit_seconds=markers['restored']['audit_seconds'],
            skipped_attempt={k:markers['skipped_attempt'][k] for k in ('schedule_seconds','gradient_check_seconds','decision_seconds')}),
        policy='Adjacent segments use only this one rank monotonic clock; sum once. Step0 skipped schedule overlaps receive/load waits, so components are NOT added to segments. Local commit at globally identified first_joint_step is not the global 16-rank endpoint. Popen has only wall time; module_start excludes launcher/interpreter prefix. Historical process->global joint retains unsynchronised cross-host wall maxima.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('statuses',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runs=[summarize(path) for path in a.statuses]
    with a.output.open('x') as f:json.dump(dict(schema=1,runs=runs),f,indent=2);f.write('\n')
    print(json.dumps([dict(session=r['session'],rank=r['selected_rank'],module_to_ready=r['module_start_to_ready_seconds'],ready_to_local_joint=r['ready_to_local_joint_step_seconds']) for r in runs]))


if __name__=='__main__':main()
