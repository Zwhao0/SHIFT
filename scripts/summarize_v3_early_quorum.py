"""Recheck retained real early-quorum CPU raw; no experiment execution."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v3_p_20261103'


def main():
    p=OUT/'p10b4_early02_raw.json';raw=json.loads(p.read_text())
    status=json.loads(raw['files']['status.json'])
    assert status['done'] and status['success'] and not status['gpu_used'] and not status['training_started']
    assert 'MainPID=0' in raw['unit'] and 'ActiveState=inactive' in raw['unit']
    source=json.loads((OUT/'p10b4_early02_source.json').read_text())
    assert json.loads(raw['files']['source_manifest.json'])==source['runner_sha256']
    cpu=json.loads((OUT/'p10b4_cpu02_provenance.json').read_text())
    assert all(r['current_source']==source['runner_sha256'] for r in cpu.values())
    assert json.loads(cpu['node0']['cpu']['cpu_status.json'])['success']
    assert 'Ran 72 tests' in cpu['node0']['cpu']['cpu_tests.log'] and '\nOK\n' in cpu['node0']['cpu']['cpu_tests.log']
    gpu=json.loads((OUT/'p8_audit_raw/status.json').read_text())
    tail=max(r['fields']['buffers']['total_s'] for r in gpu['rows'] if r['engine']=='gpu_staged')
    old=json.loads((OUT/'p10b3_checked.json').read_text())
    p10=json.loads((OUT/'p10_static_checked.json').read_text())
    window=p10['same_host_C_window_seconds'];spread=old['initial_audit_role_spread_seconds']
    cases={};budgets=[]
    for case,selected in [('success','AC'),('future_reject','AB'),('stale_k','AB'),('guard_fallback','AB')]:
        rows=[json.loads(l) for k,v in raw['files'].items() if k.startswith(case+'/') and k.endswith('.jsonl') for l in v.splitlines()]
        for k,v in raw['files'].items():
            if k.startswith('shift-v3-p10b4-early02-'+case+'-') and k.endswith('.json'):
                r=json.loads(v)
                assert 'MainPID=0' in r['unit'] and 'ActiveState=inactive' in r['unit']
                rows.extend(json.loads(l) for k2,v2 in r['files'].items() if k2.endswith('.jsonl') for l in v2.splitlines())
        assert len({(r['role'],r['rank'],r['pid']) for r in rows})==24
        assert {(r['role'],r['rank']) for r in rows}=={(r,i) for r in 'ABC' for i in range(8)}
        assert not any(r['kind']=='error' for r in rows)
        success=[r for r in rows if r['kind']=='success'];exits=[r for r in rows if r['kind']=='owner_exit']
        assert len(success)==len(exits)==24 and all(r['selected']==selected for r in success)
        assert all(r['parents_gone'] for r in exits)
        complete=[r for r in rows if r['kind']=='control_complete']
        assert len(complete)==24 and sum(r['active'] for r in complete)==16
        cleanup=[r for r in rows if r['kind']=='planned_retired_cleanup_complete']
        assert len(cleanup)==(24 if selected=='AC' else 16)
        assert all(r['all_b_owners_closed'] and r['all_parent_absent'] for r in cleanup)
        responses=[r for r in rows if r['kind']=='native_response']
        assert len(responses)==(32 if selected=='AC' else 48)
        for r in responses:
            q=r['response']
            assert all('-'+r['epoch']+'-' in s for s in q['replica_ids'])
            if case=='future_reject' and r['epoch']=='AC':continue
            assert q['max_step']==6 and not q['heal'] and not q['recover_dst_replica_ranks']
            assert q['replica_world_size']==q['max_world_size']==2
            assert q['replica_rank']==q['max_replica_rank']==(0 if r['role']=='A' else 1)
        rejects=[r for r in rows if r['kind']=='prospective_rejected']
        if case=='future_reject':
            assert len(rejects)==16 and all(r['future_done'] and r['configured']==-1 for r in rejects)
            assert all('requested recovery' in r['error'] for r in rejects)
        elif case=='stale_k':
            assert len(rejects)==1 and rejects[0]['role']=='C' and rejects[0]['rank']==0
            assert 'stale prospective boundary' in rejects[0]['error']
        else:assert not rejects
        failures=[r for r in rows if r['kind']=='planned_ownership_pre_activation_failure']
        assert len(failures)=={'success':0,'future_reject':16,'stale_k':1,'guard_fallback':1}[case]
        timing=[]
        for role in 'AC':
            for rank in range(8):
                rr=[r for r in rows if r['role']==role and r['rank']==rank];d={r['kind']:r for r in rr}
                fixture=d['anchor_guard_fixture'];assert fixture['mutation_rejected'] and not fixture['cuda_d2h_measured']
                assert len(d['early_initial_done']['hashes'])==9
                b=d['early_initial_begin']['monotonic'];e=d['early_initial_done']['monotonic']
                start=d['prospective_start']['monotonic'];done=d['prospective_future_done']['monotonic']
                row=dict(role=role,rank=rank,full_cpu_control_seconds=d['control_complete']['seconds'],
                    future_seconds=done-start,initial_tiny_audit_seconds=e-b,
                    measured_audit_future_overlap_seconds=max(0,min(e,done)-max(b,start)),
                    all_preparation_included_in_full_control=True,
                    cleanup_seconds=d['owner_exit']['monotonic']-d['cleanup_begin']['monotonic'])
                if case=='success':
                    ready=d['prospective_before_forward']['monotonic'];wait=d['prospective_wait_begin']['monotonic']
                    m=next(r for r in rr if r['kind']=='manager_constructed' and r['epoch']=='AC')
                    row.update(manager_construct_seconds=m['seconds'],
                        boundary_capture_calibration_seconds=fixture['capture_seconds'],
                        start_guard_and_submit_seconds=b-start,
                        before_forward_validation_seconds=max(0,ready-max(done,wait)),
                        post_ready_through_activation_seconds=d['control_complete']['monotonic']-ready,
                        post_compute_capture_seconds=fixture['c_capture_extra_seconds'])
                    if role=='C':
                        # Favorable planning hypothesis: all native future time
                        # hides in the original GPU initial audit, though only
                        # milliseconds actually overlap in this CPU probe.
                        # Keep every added nonoverlappable guard and capture.
                        nominal=sum(row[k] for k in ['manager_construct_seconds','boundary_capture_calibration_seconds',
                            'start_guard_and_submit_seconds','before_forward_validation_seconds',
                            'post_ready_through_activation_seconds','post_compute_capture_seconds'])
                        row.update(cpu_nonquorum_plan_seconds=nominal,
                            gpu_buffer_calibration_reserve_seconds=2*tail,
                            observed_control_variation_reserve_seconds=.039,
                            predicted_not_measured_control_seconds=nominal+2*tail+.039)
                        budgets.append(row)
                timing.append(row)
        cases[case]=dict(successes=24,active_commits_asserted=16,native_responses=len(responses),
            selected=selected,real_future_rejections=len(rejects),timing=timing,
            control_max_seconds={r:max(x['seconds'] for x in complete if x['role']==r) for r in 'ABC'})
    worst=max(budgets,key=lambda r:r['predicted_not_measured_control_seconds'])
    report=dict(scope='one bounded real3host CPU early-quorum stage; tiny-state semantics/anchor guard only',
        cases=cases,server_regression_tests=72,skipped=0,final_cpu_provenance='p10b4_cpu02_provenance.json',
        initial_GPU_audit_calibration_min_seconds=min(r['initial_audit_seconds'] for r in p10['roles']),
        real_GPU_buffer_calibration_max_seconds=tail,worst_same_C_owner_plan=worst,
        predicted_not_measured_control_seconds=worst['predicted_not_measured_control_seconds'],
        conditional_saving_seconds=window-spread-worst['predicted_not_measured_control_seconds'],
        full_GPU_budget_measured=False,gpu_admitted=False,performance_pass=False,
        route='sealed: early CPU structure works in tested scope; no robust <=200ms admission evidence',
        next='new evidence first: v1 pure-cold critical path; all normal/v1/v2 and independent gates remain',
        limitations=['CPU no-op CUDA and CPU RNG proxy; tiny ninehash is not full anchor/GPU recovery.',
            'Tiny ninehash state is independently constructed, not a real source B checkpoint transfer/load; Manager step6 is real.',
            'stale_k mutates owner progress/version adversarially, not a successful old-epoch training commit.',
            'No real CPU training schedule/TPPP; retained gradients are tiny lifecycle contributions.',
            'Audit/future overlap milliseconds measured; complete hiding in .577s GPU initial audit is a favorable hypothesis.',
            'Planning reserves use previous GPU device cost and observed39ms variation, not a p95 bound.',
            'Production trainer early factory/closure move deliberately not enabled; no new GPU run.',
            'early01 retained: semantic success24 but diagnostic parent property assertion after cleanup caused exit1.'])
    inputs=['p10b4_early01_raw.json','p10b4_early02_raw.json','p10b4_early01_source.tgz','p10b4_early02_source.tgz',
        'p10b4_early02_source.json','p10b4_cpu02_provenance.json','p8_audit_raw/status.json','p10b3_checked.json','p10_static_checked.json']
    report['inputs_sha256']={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in inputs}
    (OUT/'p10b4_checked.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['server_regression_tests','predicted_not_measured_control_seconds',
        'conditional_saving_seconds','gpu_admitted','route']}))


if __name__=='__main__':main()
