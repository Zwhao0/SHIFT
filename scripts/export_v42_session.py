"""Export compact audited session evidence before deleting regenerable raw logs."""
import argparse
import inspect
import subprocess
import json
from pathlib import Path
from scripts.v41_control import command


def export(root,two_each=False,rejected=False):
    import hashlib
    import json
    from collections import defaultdict
    from scripts.verify_session import verify
    summary=json.loads((root/'summary.json').read_text())
    result=verify(root,expected_counts=dict(v1=2,v2=2,v4=2) if two_each else None)
    result['source_manifest_sha256']=hashlib.sha256(json.dumps(result['source_manifest'],sort_keys=True).encode()).hexdigest()
    result['verifier_sha256']=hashlib.sha256(Path('/workspace/SHIFT/scripts/verify_session.py').read_bytes()).hexdigest()
    if not rejected:assert result['pass_all'],result['errors']
    files=sorted(root.glob('raw/*/*.jsonl'))
    manifest={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
    if rejected:
        return dict(name=root.name,accepted=False,done=summary.get('done'),error=summary.get('error'),
                    audit_pass=result['pass_all'],audit_errors=result['errors'][:12],
                    completed_commands=[dict(policy=c['command']['policy'],epoch=c['sequence'],step=c['completed_step']) for c in summary.get('commands',[])],
                    resources_clean=summary.get('resources_clean'),cleanup_errors=summary.get('cleanup_errors'),
                    source_sha256=summary.get('source_sha256'),raw_manifest=manifest)
    rows={}
    for p in files:
        for line in p.read_text().splitlines():
            e=json.loads(line)
            rows.setdefault(p.stem,[]).append(e)
    commits={k:[e for e in es if e['kind']=='session_commit'] for k,es in rows.items()}
    active={(r,i):('g0_'+('A' if i<4 else 'B') if r==0 else 'g0_'+('D' if i<4 else 'E'),i%4) for r in range(2) for i in range(8)}
    def window(cohort,first,last,*,cid=None):
        steps=defaultdict(dict);gaps=defaultdict(dict)
        for rank,(label,device) in cohort.items():
            for e in commits[label+str(device)]:
                if first<=e['step']<=last and e.get('contributed') and e.get('participants')==2 and (cid is None or e.get('command_id')==cid):
                    steps[e['step']][rank]=e['seconds']
                    if e.get('commit_gap') is not None:gaps[e['step']][rank]=e['commit_gap']
        values={str(step):max(sample.values()) for step,sample in sorted(steps.items()) if len(sample)==16}
        gap_values={str(step):max(sample.values()) for step,sample in sorted(gaps.items()) if len(sample)==16}
        return dict(steps=list(map(int,values)),max_worker_seconds=values,mean_seconds=sum(values.values())/len(values) if values else None,
                    max_worker_commit_gap_seconds=gap_values,mean_commit_gap_seconds=sum(gap_values.values())/len(gap_values) if gap_values else None)
    initial_control=window(active,11,80)
    cases=[]
    for record in summary['commands']:
        c=record['command'];after=dict(active);label=record['target_actor']
        if c['policy'] in ('v1','v2'):
            for i in range(8):after[1,i]=(label,i)
        else:
            for i in range(4):after[0,4+i]=(label,i)
        observed=record['first_common_observation'];first=observed['step']
        case=dict(sequence=record['sequence'],command=c,completed_step=record['completed_step'],
                  request_to_first_joint_seconds=observed['request_seconds'],fault_to_first_joint_seconds=observed['fault_seconds'],
                  command_complete_including_post_seconds=record['controller_seconds'],
                  windows=dict(before_route_last6=window(active,max(4,c['at_step']-5),c['at_step']),
                               first_joint=window(after,first,first,cid=c['id']),
                               post4=window(after,first+1,first+4,cid=c['id'])))
        a0=commits['g0_A0'];local=[e for e in a0 if c['at_step']<=e['step']<=record['completed_step']]
        case['a0_max_commit_gap_seconds']=max((b['monotonic']-a['monotonic'] for a,b in zip(local,local[1:])),default=None)
        case['fault_proof']=record.get('fault_proof')
        adjacent=[e for e in a0 if record['completed_step']<=e['step']<=record['completed_step']+1]
        case['a0_completion_transition_gap_seconds']=adjacent[1]['monotonic']-adjacent[0]['monotonic'] if len(adjacent)==2 else None
        assert len(case['windows']['first_joint']['steps'])==1 and len(case['windows']['post4']['steps'])==4
        next_boundary=summary['commands'][len(cases)+1]['command']['at_step'] if len(cases)+1<len(summary['commands']) else summary['final_step']
        if record['completed_step']+80<=next_boundary:
            case['steady_full80']=window(after,record['completed_step']+1,record['completed_step']+80)
            case['steady_11_to_80']=window(after,record['completed_step']+11,record['completed_step']+80)
            assert len(case['steady_11_to_80']['steps'])==70
        cases.append(case);active=after
    signatures={f"{e['replica']}/{e['rank']}":e['hashes'] for es in rows.values() for e in es if e['kind']=='session_done'}
    resources={kind:json.loads((root/('resources_'+kind+'.json')).read_text()) for kind in ('before','after')}
    return dict(name=root.name,accepted=True,session=summary['session'],controller_pid=summary.get('controller_pid'),
                epoch=summary['epoch'],final_step=summary['final_step'],cases=cases,independent_audit=result,
                initial_native_11_to_80=initial_control if len(initial_control['steps'])==70 and summary['commands'][0]['command']['at_step']>=80 else None,
                source_sha256=summary.get('source_sha256'),raw_manifest=manifest,resources=resources,
                resources_clean=summary['resources_clean'],cleanup_errors=summary['cleanup_errors'],selected_signatures=signatures)


def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--two-each',action='store_true');p.add_argument('--rejected',action='store_true');a=p.parse_args()
    if not a.name.startswith('session_') or not all(c.isalnum() or c in '_-' for c in a.name):p.error('invalid session name')
    code='import json,sys\nsys.path.insert(0,"/workspace/SHIFT")\nfrom pathlib import Path\n'+inspect.getsource(export)+'\nprint(json.dumps(export(Path('+repr('/workspace/SHIFT/results/'+a.name)+'),two_each='+repr(a.two_each)+',rejected='+repr(a.rejected)+')))'
    try:packet=json.loads(command(code,True))
    except subprocess.CalledProcessError as error:raise RuntimeError((error.stderr or '')[-3500:]) from None
    base=Path(__file__).resolve().parents[1]/'evidence/v4_2';base.mkdir(exist_ok=True)
    signatures=packet.pop('selected_signatures',None)
    path=base/('rejected.json' if a.rejected else 'summary.json')
    document=json.loads(path.read_text()) if path.exists() else dict(base_commit='e4d4f51eef0ab7343ccc7af750b17a42644bfa25',runs=[])
    document['runs']=[r for r in document['runs'] if r['name']!=a.name]+[packet];path.write_text(json.dumps(document,indent=2)+'\n')
    if signatures:
        sp=base/'selected_signatures.json';selected=json.loads(sp.read_text()) if sp.exists() else {};selected[a.name]=signatures;sp.write_text(json.dumps(selected,separators=(',',':'))+'\n')
    print(json.dumps(dict(name=a.name,accepted=packet['accepted'],audit=packet.get('independent_audit',{}).get('pass_all'),resources_clean=packet.get('resources_clean'))))


if __name__=='__main__':main()
