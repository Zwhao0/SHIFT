"""CPU-only independent batch8 audits and compact timing extraction on node0."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from shift_full.reference_audit import REFERENCE_HASH_KEYS
from shift_full.training_core import sample_plan

ROOT=Path('/workspace/SHIFT/results')


def read(path): return json.loads(path.read_text())
def events(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(values):
    values=[v for v in values if isinstance(v,(float,int))]
    return dict(n=len(values),mean=statistics.fmean(values) if values else None,
                max=max(values) if values else None,min=min(values) if values else None)


def window(rows):
    by_step={}
    for x in rows: by_step.setdefault(x['step'],[]).append(x['seconds'])
    return dict(step_max_seconds={k:max(v) for k,v in sorted(by_step.items())},
                max_rank_step=stats([max(v) for v in by_step.values()]))


def clean(root):
    before,after=read(root/'resources_before.json'),read(root/'resources_after.json')
    assert set(before)==set(after)=={'node0','node1','node3'}
    loads={}
    for node in before:
        a,b=before[node],after[node]
        def pids(x):
            if 'company' in x: return set(x['company'])
            if 'company_pids' in x: return set(x['company_pids'])
            return {s.split(',')[0].strip() for s in x['gpu'] if 'VLLM::' in s}
        assert len(pids(a))==8 and pids(a)==pids(b)
        for x in (a,b):
            assert not x.get('extra_gpu') and not x.get('extra_pids') and not x.get('missing_company_pids')
        loads[node]={phase:x.get('gpu_load') for phase,x in (('before',a),('after',b))}
    for phase in ('before','after'):
        path=root/f'gpu_load_{phase}.json'
        if path.exists():
            for node,load in read(path).items(): loads[node][phase]=load
    return dict(company_pids_unchanged=True,company_pid_count_per_node=8,extra_gpu=0,gpu_load=loads)


def binding_rows(root):
    result={}
    for item in read(root/'physical_manifest.json'):
        for rank in range(8):
            p=root/('physical_'+item['physical_node'])/Path(item['directory']).name/f"events_g{item['generation']}_r{rank}.jsonl"
            if p.exists(): result[item['rid'],item['generation'],rank]=events(p)
    return result


def native_baseline(root,summary):
    rows={}
    for rid in (0,1):
        node=summary.get('placement',{'0':'node0','1':'node1'})[str(rid)]
        for rank in range(8): rows[rid,0,rank]=events(root/f'physical_{node}/rid{rid}_g0/events_g0_r{rank}.jsonl')
    commits=[]
    for (rid,g,rank),es in rows.items():
        cs=[x for x in es if x['kind']=='commit'];commits+=cs
        assert [x['step'] for x in cs]==list(range(1,summary['steps']+1))
        assert len({x['pid'] for x in es})==1
        assert not any(x['kind']=='skipped' for x in es)
        for x in cs:
            assert x['global_batch']==8 and x['participants']==2 and x['contributed'] and x['finite_gradients']
            assert x['sample_ids']==sample_plan(x['step']-1,replica=rid,local_data_rank=rank%4//2)
            if not summary['audit']: assert not x.get('hashes')
    result=dict(pass_all=True,commits=len(commits),windows=dict(full=window(commits),steady_step4_plus=window([x for x in commits if x['step']>=4])))
    if summary['audit']:
        signatures=read(root/'selected_signatures.json')
        for rank in range(8):
            a,b=signatures[f'0/{rank}'],signatures[f'1/{rank}']
            assert len(a)==len(b)==summary['steps']
            for x,y in zip(a,b):
                assert set(x['hashes'])==REFERENCE_HASH_KEYS|{'gradients'} and x['hashes']==y['hashes']
        result['signature_sha256']=sha(root/'selected_signatures.json')
        result['final_signatures']={k:v[-1]['hashes'] for k,v in signatures.items()}
    return result


def v4(root,summary):
    assert summary['unified'] and summary['plan']['microbatches']==2 and summary['topology']['global_batch']==8
    roles=summary.get('role_hosts')
    if roles is None:
        from shift_full.control import endpoints_from_env
        ips={e.ip:n for n,e in endpoints_from_env().items()}
        config=read(Path('/workspace/SHIFT/runtime/v3-reproduce')/(root.name+'.json'))
        roles={role:ips[ip] for role,ip in config['hosts'].items()}
    expected=dict(A='node0',B='node3' if summary.get('post_layout') else 'node0',D='node1',E='node1')
    assert all(roles[k]==v for k,v in expected.items())
    if summary.get('reference_only',False):
        if not summary['performance_no_audit']:
            from scripts.verify_v4_reference import verify
            result=verify(root);assert result['pass_all'],result['errors']
            result['signature_sha256']=sha(root/'reference_signatures.json')
            return result
        cs=[];steps=summary['plan']['cut']+summary['plan']['post_steps']
        for role in 'ABDE':
            for device in range(4):
                es=events(root/f'{role}{device}.jsonl');rank=device+(4 if role in 'BE' else 0);replica=int(role in 'DE')
                commits=[x for x in es if x['kind']=='commit'];cs+=commits
                assert [x['step'] for x in commits]==list(range(1,steps+1))
                assert len({x['pid'] for x in es})==1
                assert not any(x['kind'] in ('reference_step','reference_comparison','outer_gradient_audit') for x in es)
                for x in commits:
                    assert x['finite'] and x['phase']=='reference' and x['epoch']==0
                    assert x['sample_ids']==sample_plan(x['step']-1,replica=replica,local_data_rank=rank%4//2)
                outer=[x for x in es if x['kind']=='outer_commit']
                assert [x['step'] for x in outer]==list(range(steps))
                assert all(x['committed_step']==x['step']+1 and x['batches_committed']==2*(x['step']+1) for x in outer)
                done=[x for x in es if x['kind']=='done'];assert len(done)==1 and done[0]['identity_preserved'] and done[0]['step']==steps
        return dict(pass_all=True,commits=len(cs),windows=dict(full=window(cs),steady_step4_plus=window([x for x in cs if x['step']>=4])))
    assert roles['C']=='node3'
    from scripts.verify_v4_transaction import verify
    from scripts.quick_v4_timing import _load_worker_events,_timing_evidence
    result=verify(root);assert result['pass_all'],result['errors']
    timing=_timing_evidence(root,summary,_load_worker_events(root))
    result['timing']={k:timing[k] for k in ('request_to_first_common_commit_seconds','request_to_first_common_commit_proxy_seconds','controller_observation_matches_event_logs')}
    result['timing']['stages']={k:{f:v[f] for f in ('commit_count','seconds','commit_gap')} for k,v in timing['commit_stage_samples'].items()}
    result['timing']['node0_A0_stages']=timing['node0_A0']['stages']
    result['timing']['peer_outer_wait_seconds']=stats(result.pop('peer_outer_gradient_ready_seconds').values())
    commits=[x for es in _load_worker_events(root).values() for x in es if x['kind']=='commit']
    first=timing['first_recovery_common_commit']['step'];requested=summary['migration_request']['observed_step']
    result['timing']['fixed_windows']={
        'pre':window([x for x in commits if 4<=x['step']<=requested]),
        'during':window([x for x in commits if requested<x['step']<first]),
        'first_joint':window([x for x in commits if x['step']==first]),
        'post':window([x for x in commits if first<x['step']<=first+4])}

    result['final_signatures']={f"{x['replica']}/{x['rank']}":x['hashes'] for es in _load_worker_events(root).values() for x in es if x['kind']=='done'}
    return result


def recovery(root,status):
    rows=binding_rows(root);all_events=[x for es in rows.values() for x in es]
    if root.name.startswith('v1_'):
        from shift_full.verify_rotation import verify
        from scripts.quick_v1_remote import short_schema2_audit,performance_state_audit
        performance=bool(status.get('performance_no_audit'))
        result=verify(root,1,require_audit=not performance);assert result['pass'],result['errors']
        errors=performance_state_audit(root,5) if performance else short_schema2_audit(root,5)
        assert not errors,errors
        timing=read(root/'timing_summary.json');fault=read(root/'fault_results.json')[0]
        first=fault['first_joint_step'];trigger=status['trigger']
        selected=[x for x in events(root/'controller.jsonl') if x['kind']=='candidate_selected']
        actual_fault_boundary=selected[0]['committed_step']

        result=dict(pass_all=True,target_rid=fault['target'],actual_fault_boundary=actual_fault_boundary,first_joint_step=first,solo_steps=result['faults'][0]['healthy_solo_step_count'],
                    full_seconds=timing['fault_marker_to_first_joint_observation_seconds'])
    else:
        from scripts.quick_v2_remote import audit
        from scripts.v3_planned_remote import PlannedRotation
        from types import SimpleNamespace
        data={role:{f'events_g{g}_r{k}.jsonl':es for (rid,gen,k),es in rows.items() if rid==r and gen==g}
              for role,(r,g) in dict(A=(0,0),B=(1,0),C=(1,1)).items()}
        fake=SimpleNamespace(rows_role=lambda *args,**kw:PlannedRotation.rows_role(None,*args,**kw))
        cut,goal=status['observed_cut'],status['goal'];first=cut+1;trigger=status['requested_trigger']
        result=audit(fake,data,cut,goal,status['audit_every_step'],status['audit_state'],status['variant'])
        result.update(pass_all=True,full_seconds=status['request_to_observed_joint_seconds'])
        # New replicated-RNG contract: independently require all nine, beyond the legacy shared-model checks.
        for step in range(1,goal+1):
            peers={}
            for (rid,g,rank),es in rows.items():
                for x in es:
                    if x['kind']=='commit' and x['step']==step and x.get('hashes'): peers[rid,rank]=x['hashes']
            if peers:
                assert len(peers)==16
                for rank in range(8): assert set(peers[0,rank])==REFERENCE_HASH_KEYS and peers[0,rank]==peers[1,rank]
        if not status['audit_state']:
            finals={(rid,rank):x['hashes'] for (rid,g,rank),es in rows.items() for x in es if x['kind']=='final_audit' and x['step']==goal}
            for rank in range(8): assert set(finals[0,rank])==REFERENCE_HASH_KEYS and finals[0,rank]==finals[1,rank]
    commits=[x for x in all_events if x['kind']=='commit' and x.get('contributed')]
    result['windows']={
        'pre':window([x for x in commits if 4<=x['step']<=trigger]),
        'during':window([x for x in commits if trigger<x['step']<first]),
        'first_joint':window([x for x in commits if x['step']==first and x['participants']==2]),
        'post':window([x for x in commits if first<x['step']<=first+4 and x['participants']==2]),
        'post_extra_drain':window([x for x in commits if x['step']>first+4])}
    result['stage_durations']={kind:stats([x.get('seconds') for x in all_events if x['kind']==kind])
        for kind in sorted({x['kind'] for x in all_events if 'seconds' in x and x['kind'] not in ('commit','begin')})}
    result['last_validated_signatures']={f'{rid}/{g}/{rank}':next((x.get('hashes') for x in reversed(es) if x.get('hashes')),None)
                                for (rid,g,rank),es in rows.items()}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('name');a=p.parse_args()
    assert a.name and all(c.isalnum() or c in '_-' for c in a.name)
    root=ROOT/a.name;summary=read(root/('summary.json' if (root/'summary.json').exists() else 'status.json'))
    result=v4(root,summary) if a.name.startswith('v4_') else native_baseline(root,summary) if a.name.startswith('unified_') else recovery(root,summary)
    result.update(name=a.name,resources=clean(root))
    (root/'v41_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(name=a.name,pass_all=result['pass_all'])))


if __name__=='__main__':main()
