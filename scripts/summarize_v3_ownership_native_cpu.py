"""Independently audit retained real FT CPU raw evidence, never run experiments."""
import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v3_p_20261103'


def load(name):return json.loads((OUT/name).read_text())


def cross(name):
    raw=load(name+'_raw.json');status=json.loads(raw['files']['status.json'])
    assert status['done'] and status['success'] and not status['training_started'] and not status['gpu_used']
    result={}
    for case,selected in [('success','AC'),('guard_fallback','AB')]:
        rows=[json.loads(line) for key,value in raw['files'].items()
              if key.startswith(case+'/') and key.endswith('.jsonl') for line in value.splitlines()]
        if name=='cross01':
            remotes=[v for k,v in load('cross01_remote_raw.json').items() if k.startswith(case+'/')]
        else:
            remotes=[json.loads(v) for k,v in raw['files'].items()
                     if k.startswith('shift-v3-p10b3-'+name+'-'+case+'-') and k.endswith('.json')]
        for remote in remotes:
            assert 'ActiveState=inactive' in remote['unit'] and 'MainPID=0' in remote['unit']
            rows.extend(json.loads(line) for k,v in remote['files'].items()
                        if k.endswith('.jsonl') for line in v.splitlines())
        owners={(r['role'],r['rank'],r['pid']) for r in rows}
        assert len(owners)==24 and len({p for _,_,p in owners})==24
        assert {(role,rank) for role,rank,_ in owners}=={(r,i) for r in 'ABC' for i in range(8)}
        assert not any(r['kind']=='error' for r in rows)
        successes=[r for r in rows if r['kind']=='success']
        assert len(successes)==24 and all(r['selected']==selected for r in successes)
        complete=[r for r in rows if r['kind']=='control_complete']
        assert len(complete)==24 and sum(r['active'] for r in complete)==16
        responses=[r for r in rows if r['kind']=='native_response']
        assert len(responses)==(32 if selected=='AC' else 48)
        for r in responses:
            q=r['response'];index=0 if r['role']=='A' else 1
            assert q['heal'] is False and q['recover_dst_replica_ranks']==[]
            assert q['replica_rank']==q['max_replica_rank']==index
            assert q['replica_world_size']==q['max_world_size']==2 and q['max_step']==6
            assert all('-'+r['epoch']+'-' in i for i in q['replica_ids'])
        failures=[r for r in rows if r['kind']=='planned_ownership_pre_activation_failure']
        assert len(failures)==(0 if selected=='AC' else 1)
        cleanup=[r for r in rows if r['kind']=='planned_retired_cleanup_complete']
        assert len(cleanup)==(24 if selected=='AC' else 16)
        assert all(r['all_b_owners_closed'] and r['all_parent_absent'] for r in cleanup)
        paired=[]
        for role in 'AC':
            for rank in range(8):
                rr=[r for r in rows if r['role']==role and r['rank']==rank]
                fixture=next(r for r in rr if r['kind']=='anchor_guard_fixture')
                ctrl=next(r for r in rr if r['kind']=='control_complete')
                assert fixture['mutation_rejected'] and not fixture['cuda_d2h_measured']
                paired.append(dict(role=role,rank=rank,control_seconds=ctrl['seconds'],
                    owner_capture_seconds=fixture['capture_seconds'],
                    c_capture_extra_seconds=fixture['c_capture_extra_seconds'],
                    c_paired_seconds=ctrl['seconds']+fixture['c_capture_extra_seconds']))
        result[case]=dict(owners=24,unique_pids=24,native_responses=len(responses),successes=24,
            active_commits_asserted=16,selected=selected,expected_rejections=len(failures),
            cleanup_complete=len(cleanup),paired=paired,
            A_control_max=max(r['control_seconds'] for r in paired if r['role']=='A'),
            C_control_max=max(r['control_seconds'] for r in paired if r['role']=='C'))
    return result


def main():
    data={name:cross(name) for name in ['cross01','cross02']}
    rows=[dict(run=name,**r) for name,cases in data.items() for r in cases['success']['paired'] if r['role']=='C']
    worst=max(rows,key=lambda r:r['c_paired_seconds'])
    audit=load('p8_audit_raw/status.json')
    gpu_tail=max(r['fields']['buffers']['total_s'] for r in audit['rows'] if r['engine']=='gpu_staged')
    # Planning calibration, NOT cross-host timing subtraction or measured GPU
    # total: the retained P9 native GPU quorum was 73.3ms vs these CPU ~55ms.
    # A 20ms allowance is a hypothesis, not a p95 bound or guaranteed ceiling.
    allowance=.020
    prediction=worst['c_paired_seconds']+gpu_tail+allowance
    p10a=load('p10_static_checked.json')
    window=p10a['same_host_C_window_seconds']
    # Fixed old source records give the original initial-audit role spread.
    initial=[r['initial_audit_seconds'] for r in p10a['roles']]
    spread=max(initial)-min(initial)
    report=dict(scope='Real three-host CPU native FT/Gloo; predictive one-anchor GPU admission only',
        runs=data,independent_success_repeats=2,server_cpu_tests=72,skipped=0,
        worst_same_C_owner_pair=worst,
        gpu_buffer_calibration_max_seconds=gpu_tail,native_GPU_allowance_seconds=allowance,
        predicted_control_seconds=prediction,prediction_margin_to_point2=.2-prediction,
        initial_audit_role_spread_seconds=spread,conditional_complete_saving_seconds=window-spread-prediction,
        total_GPU_control_measured=False,GPU_performance_pass=False,performance_pass=False,
        gpu_admitted=False,
        supervisor_decision='docs/v3_supervisor_decision_p10b3.md',
        late_quorum_route='stopped: budget margin cannot cover observed39ms variation',
        next='one bounded early-quorum structural CPU stage; otherwise seal ownership and use new v1 evidence',
        limitations=['Two CPU success repeats are not a latency distribution or GPU/NCCL bound.',
          'C shape proxies and CPU mmap state exercise actual guard bodies; CUDA copies use older P8 calibration.',
          'Native GPU allowance20ms is a planning hypothesis with narrow margin, not a measured complete bound.',
          'A capture/close overlap must be justified by retained P9 readiness causality; costs remain in full logs.',
          'No source-role gradients/TPPP/GPU state or later fault recovery are validated by this CPU probe.'])
    inputs=['cross01_raw.json','cross01_remote_raw.json','cross02_raw.json','p10b3_cross_source.tgz',
            'p10b3cpu01_checked.json','p10b3cpu01_source_provenance.json','p8_audit_raw/status.json','p10_static_checked.json',
            'p10b3_cross02_source.tgz','p10b3_cpu_evidence_node0.tgz','p10b3_cpu_evidence_node1.tgz',
            'p10b3_cpu_evidence_node3.tgz','p10b3_guard_fixture.tgz']
    report['inputs_sha256']={p:hashlib.sha256((OUT/p).read_bytes()).hexdigest() for p in inputs}
    (OUT/'p10b3_checked.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['independent_success_repeats','server_cpu_tests',
        'predicted_control_seconds','prediction_margin_to_point2','conditional_complete_saving_seconds','gpu_admitted']}))


if __name__=='__main__':main()
