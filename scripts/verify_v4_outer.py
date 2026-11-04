"""Independent event audit for two-replica V4 Manager integration."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from shift_full.v4_protocol import state_channel_matches
from shift_full.reference_audit import REFERENCE_HASH_KEYS


def verify(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text())
    if not summary.get('outer_manager'):
        raise ValueError('outer Manager experiment required')
    if summary.get('reference_only'):
        from scripts.verify_v4_reference import verify as verify_reference
        return verify_reference(root)
    if summary.get('transactional_switch'):
        from scripts.verify_v4_transaction import verify as verify_transaction
        return verify_transaction(root)
    external=bool(summary.get('reference_run'))
    retained=summary.get('retain_outer_managers',False)
    overlap_peer=bool(summary.get('overlap_peer',False))
    rows={r:{d:[json.loads(line) for line in (root/f'{r}{d}.jsonl').read_text().splitlines() if line.strip()]
             for d in range(4)} for r in 'ABCDE'}
    errors=[]
    def check(ok,message):
        if not ok:errors.append(message)
    overlap_evidence_ok=overlap_peer
    peer_cut_quorum_wait_seconds={}
    def check_overlap(ok,message):
        nonlocal overlap_evidence_ok
        check(ok,message)
        if not ok:overlap_evidence_ok=False
    summary_cut=summary['plan']['cut']
    fail_restore_rank=summary.get('fail_restore_rank',-1)
    failed=fail_restore_rank!=-1
    outer_audit=bool(summary.get('outer_audit',False))
    boundary_steps=[]
    for role in 'ABDE':
        for device,events in rows[role].items():
            found=[x for x in events if x['kind']=='boundary_selected']
            check(len(found)==1,'unique boundary '+role+str(device))
            boundary_steps.extend(x['step'] for x in found)
    check(len(boundary_steps)==16 and len(set(boundary_steps))==1,'joint boundary selection')
    cut=boundary_steps[0] if boundary_steps else summary_cut
    check(cut==summary_cut,'boundary/summary cut agreement')
    post=summary['plan']['post_steps']
    gradient_audit_ok=None if not outer_audit else True

    def event_pairs(events):
        return [(x.get('outer_epoch'),x.get('step')) for x in events]

    def has_complete_hashes(value):
        return isinstance(value,dict) and set(value)==REFERENCE_HASH_KEYS

    def valid_tensor_comparison(comparison):
        if not isinstance(comparison,dict):
            return False
        if set(comparison)!= {'model','optimizer','all_buffers'}:
            return False
        for stats in comparison.values():
            if not isinstance(stats,dict) or not {'elements','unequal','max_abs'}<=set(stats):
                return False
            if not isinstance(stats['elements'],int) or stats['elements']<0:
                return False
            if stats['unequal']!=0 or stats['max_abs']!=0:
                return False
        return True

    for role,devices in rows.items():
        for device,events in devices.items():
            label=role+str(device);replica=int(role in 'DE');rank=device+(4 if role in 'BCE' else 0)
            check(events and all(x['role']==role and x['rank']==rank and x['replica']==replica for x in events),'binding '+label)
            check(len({x['pid'] for x in events})==1,'PID continuity '+label)
            for x in events:
                if 'hashes' in x:
                    check(has_complete_hashes(x['hashes']),'nine state hashes '+label+'/'+x['kind'])
            if role in 'ABDE':
                ready0=[x for x in events if x['kind']=='outer_ready' and x.get('outer_epoch')==0]
                check(len(ready0)==1,'epoch 0 Manager ready '+label)
            commits=[x for x in events if x['kind']=='commit']
            expected_steps={
                'before':list(range(1,cut+1)) if role!='C' else [],
                'oracle':list(range(cut+1,cut+post+1)) if summary['oracle'] and not external and role!='C' else [],
                'after':list(range(cut+1,cut+post+1)) if not failed and role!='B' else [],
            }
            for phase in ('before','oracle','after'):
                check([x['step'] for x in commits if x['phase']==phase]==expected_steps[phase],
                      'commit coverage '+label+'/'+phase)
            for x in commits:
                expected=[(x['step']-1)*summary['plan']['microbatches']*4+m*4+replica*2+device//2
                          for m in range(summary['plan']['microbatches'])]
                check(x['sample_ids']==expected and x['finite'],'sample/finite '+label+'/'+str(x['step']))
            if overlap_peer:
                after_commits=[x for x in commits if x.get('phase')=='after']
                check_overlap(all(x.get('global_activation_observed') is True for x in after_commits),
                              'post-activation commit marker '+label)

            expected_pairs=[]
            for phase in ('before','oracle','after'):
                epoch=1 if phase=='after' else 0
                expected_pairs.extend((epoch,step-1) for step in expected_steps[phase])
            outer=[x for x in events if x['kind']=='outer_commit']
            check(len(outer)==len(commits),'real Manager commit coverage '+label)
            check(event_pairs(outer)==expected_pairs and
                  [x['step']+1 for x in outer]==[x['step'] for x in commits] and
                  all(x['committed_step']==x['step']+1 and x['batches_committed']==2*(x['step']+1) for x in outer),
                  'Manager committed progress '+label)
            begins=[x for x in events if x['kind']=='outer_begin']
            check(event_pairs(begins)==expected_pairs and
                  all(x['participants']==2 and x['participating_rank']==replica for x in begins),
                  'fixed two-replica quorum '+label)
            if overlap_peer:
                requested=[x for x in events if x['kind']=='outer_quorum_requested']
                waits=[x for x in events if x['kind']=='outer_quorum_wait']
                check_overlap(event_pairs(requested)==expected_pairs,
                              'async quorum request coverage '+label)
                duration_ok=all(isinstance(x.get('seconds'),(int,float)) and
                                not isinstance(x.get('seconds'),bool) and
                                math.isfinite(x['seconds']) and x['seconds']>=0 for x in waits)
                check_overlap(event_pairs(waits)==expected_pairs and duration_ok,
                              'async quorum wait coverage '+label)
                for pair in expected_pairs:
                    positions=[]
                    for kind in ('outer_quorum_requested','outer_quorum_wait','outer_begin'):
                        found=[i for i,x in enumerate(events)
                               if x['kind']==kind and (x.get('outer_epoch'),x.get('step'))==pair]
                        check_overlap(len(found)==1,'unique async quorum event '+label+'/'+kind+'/'+str(pair))
                        positions.append(found[0] if len(found)==1 else -1)
                    if all(i>=0 for i in positions):
                        check_overlap(positions[0]<positions[1]<positions[2],
                                      'async quorum causal order '+label+'/'+str(pair))
                if role in 'DE':
                    cut_waits=[x for x in waits if x.get('outer_epoch')==1 and x.get('step')==cut]
                    check_overlap(len(cut_waits)==1,'peer cut quorum wait '+label)
                    if len(cut_waits)==1 and duration_ok:
                        peer_cut_quorum_wait_seconds[f'{role}/rank{rank}']=cut_waits[0]['seconds']
            audits=[x for x in events if x['kind']=='outer_gradient_audit']
            if outer_audit:
                audit_ok=(event_pairs(audits)==expected_pairs and all(x['exact'] and x['buckets']>0 for x in audits))
                check(audit_ok,'independent gradient AVG '+label)
                gradient_audit_ok=gradient_audit_ok and audit_ok
            ready=[x for x in events if x['kind']=='outer_ready']
            check(all(x['timeout_seconds']==summary['outer_timeout'] and x['quorum_timeout_seconds']==summary['outer_timeout']
                      for x in ready),'effective Manager deadlines '+label)
            check(len({x.get('outer_epoch') for x in ready})==len(ready) and
                  all(x.get('outer_epoch') in (0,1) for x in ready),'unique Manager generations '+label)
            if not failed:
                expected_epochs=([1] if role=='C' else [0] if role=='B' or retained or
                                 (overlap_peer and role in 'DE') else [0,1])
                check([x['outer_epoch'] for x in ready]==expected_epochs,'Manager generation coverage '+label)
            oracle_steps=[x for x in events if x['kind']=='oracle_step']
            expected_oracle_steps=(list(range(cut+1,cut+post+1))
                                   if summary['oracle'] and not external and not failed and role in 'ACDE' else [])
            check([x['step'] for x in oracle_steps]==expected_oracle_steps and
                  all(x.get('exact') is True for x in oracle_steps),
                  'oracle step correspondence '+label)
            if role=='B':
                check(sum(x['kind']=='old_owner_rejected' for x in events)==1,'source fence '+label)
            if failed:continue
            if role=='B':
                check(sum(x['kind']=='retired' for x in events)==1,'source retirement '+label)
                if summary['defer_source_local_close']:
                    observed=[x for x in events if x['kind']=='activation_observed']
                    closed=[x for x in events if x['kind']=='source_local_groups_closed']
                    check(len(observed)==len(closed)==1 and observed[0]['monotonic']<=closed[0]['monotonic']
                          and closed[0]['discarded']>0,'deferred source close '+label)
                continue
            done=[x for x in events if x['kind']=='done']
            check(len(done)==1 and done[0]['identity_preserved'] and done[0]['step']==cut+post,'done/identity '+label)
            restored=[x for x in events if x['kind']=='outer_restored']
            if (retained and role in 'ADE') or (overlap_peer and role in 'DE'):
                check(not restored,'retained Manager progress was not loaded '+label)
            else:
                check(len(restored)==1 and restored[0]['step']==cut and restored[0]['batches_committed']==2*cut
                      and restored[0]['outer_epoch']==1,'Manager progress restoration '+label)
            if role in 'DE':
                check(sum(x['kind']=='peer_inner_groups_retained' and x['identity_preserved'] for x in events)==1,
                      'peer inner PG preservation '+label)
                if overlap_peer:
                    peer_ready=[x for x in events if x['kind']=='peer_compute_ready']
                    peer_begin=[x for x in events if x['kind']=='peer_compute_begin']
                    peer_gradients=[x for x in events if x['kind']=='peer_native_gradient_ready']
                    check_overlap(len(peer_ready)==1 and peer_ready[0].get('step')==cut and
                                  peer_ready[0].get('identity_preserved') is True and
                                  has_complete_hashes(peer_ready[0].get('hashes')),
                                  'peer boundary compute-ready '+label)
                    check_overlap(len(peer_begin)==1 and peer_begin[0].get('step')==cut,
                                  'peer compute begin '+label)
                    check_overlap(len(peer_gradients)==1 and peer_gradients[0].get('step')==cut,
                                  'peer native gradient readiness '+label)
                    check_overlap(not any(x['kind']=='activated' for x in events),
                                  'peer worker must not emit activation '+label)
                    frozen=[x for x in events if x['kind']=='freeze']
                    check_overlap(len(frozen)==1 and len(peer_ready)==1 and
                                  frozen[0].get('hashes')==peer_ready[0].get('hashes'),
                                  'peer state preserved while computing '+label)
                    locations={}
                    for kind in ('outer_quorum_requested','peer_compute_begin',
                                 'peer_native_gradient_ready','outer_quorum_wait','outer_begin'):
                        found=[i for i,x in enumerate(events)
                               if x['kind']==kind and x.get('step')==cut and
                               (kind.startswith('peer_') or x.get('outer_epoch')==1)]
                        check_overlap(len(found)==1,'peer overlap marker '+label+'/'+kind)
                        locations[kind]=found[0] if len(found)==1 else -1
                    ordered=[locations[kind] for kind in
                             ('outer_quorum_requested','peer_compute_begin','peer_native_gradient_ready',
                              'outer_quorum_wait','outer_begin')]
                    if all(i>=0 for i in ordered):
                        check_overlap(ordered==sorted(ordered) and len(set(ordered))==len(ordered),
                                      'peer compute before async quorum wait '+label)
            if summary['oracle'] and done:
                comparison=done[0].get('comparison')
                exact=(comparison.get('exact') if isinstance(comparison,dict) else None)
                complete_exact=(isinstance(exact,dict) and set(exact)==REFERENCE_HASH_KEYS and all(exact.values()))
                tensors=(comparison.get('tensors') if isinstance(comparison,dict) else None)
                check(complete_exact and comparison.get('trajectory_exact') is True and
                      comparison.get('manager_exact') is True,
                      'final states and Manager '+label)
                if external:
                    check(comparison.get('method')=='independent_nine_state_sha256' and tensors=={},
                          'independent signature comparison method '+label)
                else:
                    check(valid_tensor_comparison(tensors),'full tensor oracle '+label)
    def event(role,device,kind):
        found=[x for x in rows[role][device] if x['kind']==kind]
        check(bool(found),kind+' '+role+str(device))
        return found[0] if found else {}
    if failed:
        injected=[x for es in rows['C'].values() for x in es if x['kind']=='restore_failure_injected']
        check(fail_restore_rank in range(4,8) and len(injected)==1 and
              injected[0].get('rank')==fail_restore_rank and
              injected[0].get('device')==fail_restore_rank-4 and
              injected[0].get('step')==cut and summary.get('expected_failure_pass'),
              'expected injected restore failure at configured rank')
        check(not any(x['kind']=='activated' for r in 'ACDE' for es in rows[r].values() for x in es),
              'no activation after failed restoration')
    else:
        check(not any(x['kind']=='restore_failure_injected' for es in rows['C'].values() for x in es),
              'unexpected restore failure injection')
        for device in range(4):
            source=event('B',device,'freeze');target=event('C',device,'restored')
            check(source.get('hashes')==target.get('hashes'),'complete source-target state '+str(device))
            sent=event('B',device,'state_sent');received=event('C',device,'state_received')
            check(bool(sent and received) and state_channel_matches(sent,received),'state channel '+str(device))
            for role in 'ADE':
                frozen=event(role,device,'freeze')
                if overlap_peer and role in 'DE':
                    active=event(role,device,'peer_compute_ready')
                    check(active.get('step')==cut and active.get('identity_preserved') is True and
                          frozen.get('hashes')==active.get('hashes'),
                          'retained peer state during compute '+role+str(device))
                else:
                    active=event(role,device,'activated')
                    check(frozen.get('hashes')==active.get('hashes'),
                          'retained state across Manager rebuild '+role+str(device))
        if overlap_peer:
            for role in 'AC':
                for device in range(4):
                    active_events=[x for x in rows[role][device] if x['kind']=='activated']
                    active=active_events[0] if active_events else {}
                    check_overlap(len(active_events)==1 and
                                  type(active.get('peer_compute_started')) is bool and
                                  type(active.get('peer_gradients_ready')) is bool,
                                  'peer activation markers '+role+str(device))
            a0=event('A',0,'activated')
            check_overlap(a0.get('peer_compute_started') is True,
                          'healthy compute started before global activation')
        for rank in range(8):
            role0='A' if rank<4 else 'C';role1='D' if rank<4 else 'E'
            left=event(role0,rank%4,'done').get('hashes',{})
            right=event(role1,rank%4,'done').get('hashes',{})
            check(all(left.get(k)==right.get(k) for k in ('model','optimizer','scheduler')),'DP state equality '+str(rank))
    if external:
        from scripts.verify_v4_reference import verify_comparisons
        errors.extend(verify_comparisons(root,summary,rows))
        check(summary.get('reference_audit',{}).get('pass_all') is True,'audited independent reference provenance')
    if retained and not failed:
        from scripts.verify_v4_reference import verify_retention
        retention_errors=verify_retention(rows,cut)
        errors.extend(retention_errors)
        if overlap_peer and retention_errors:overlap_evidence_ok=False
    elif overlap_peer and not failed:
        from scripts.verify_v4_reference import verify_retention
        retention_errors=verify_retention({'D':rows['D'],'E':rows['E']},cut)
        errors.extend(retention_errors)
        if retention_errors:overlap_evidence_ok=False
    before=json.loads((root/'resources_before.json').read_text());after=json.loads((root/'resources_after.json').read_text())
    required_nodes={'node0','node1','node3'}
    check(isinstance(before,dict) and set(before)==required_nodes and
          isinstance(after,dict) and set(after)==required_nodes,'resource node coverage')
    clean=(isinstance(before,dict) and isinstance(after,dict) and
           set(before)==required_nodes and set(after)==required_nodes and
           all(isinstance(before[n].get('company'),dict) and len(before[n]['company'])==8 and
               before[n]['company']==after[n].get('company') and
               not before[n].get('extra_gpu') and not after[n].get('extra_gpu')
               for n in required_nodes))
    check(clean,'resource/company PID audit')
    return dict(pass_all=not errors,errors=errors,resources_clean=clean,replicas=2,logical_workers=16,
                healthy_processes_preserved=0 if failed else 12,source_processes_fenced=4,cut=cut,post_steps=post,
                expected_failure=failed,oracle_rank_steps=0 if failed or not summary['oracle'] else 16*post,
                independent_gradient_audit=bool(outer_audit and gradient_audit_ok),
                gradient_audit_enabled=outer_audit,
                gradient_audit_status=('disabled' if not outer_audit else 'passed' if gradient_audit_ok else 'failed'),
                gradient_audit_cases=sum(x['kind']=='outer_gradient_audit' for devices in rows.values()
                                         for events in devices.values() for x in events),
                overlap_peer=overlap_peer,
                peer_compute_overlap_proven=bool(overlap_peer and overlap_evidence_ok),
                peer_cut_quorum_wait_seconds=peer_cut_quorum_wait_seconds,
                independent_fault_domains=False)


def summarize(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text())
    if summary.get('reference_only'):
        from scripts.verify_v4_reference import summarize as summarize_reference
        return summarize_reference(root)
    if summary.get('transactional_switch'):
        from scripts.verify_v4_transaction import summarize as summarize_transaction
        return summarize_transaction(root)
    audit=verify(root)
    if not audit['pass_all']:raise ValueError(audit['errors'])
    events=[json.loads(line) for role in 'ABCDE' for device in range(4)
            for line in (root/f'{role}{device}.jsonl').read_text().splitlines() if line.strip()]
    log_text='\n'.join(p.read_text(errors='replace') for p in root.rglob('*.log'))
    warning_patterns={
        'nccl_exit_without_explicit_destroy':'destroy_process_group() was not called',
        'barrier_device_inferred':'barrier(): using the device under current context',
        'torchft_nonblocking_timeout_default':'TORCH_NCCL_NONBLOCKING_TIMEOUT is not set',
        'flight_recorder_unavailable':'Unable to reset NCCL flight recorder',
        'future_deprecation_warning':'FutureWarning',
    }
    result={k:summary[k] for k in ('name','topology','plan','oracle','outer_manager','outer_audit','outer_timeout',
        'fail_restore_rank','functional_pass','oracle_exact','source_sha256','resources_clean','cleanup_errors')}
    result.update(reference_run=summary.get('reference_run'),reference_sha256=summary.get('reference_sha256'),
        retain_outer_managers=summary.get('retain_outer_managers',False),
        overlap_peer=bool(summary.get('overlap_peer',False)),
        peer_compute_overlap_proven=audit.get('peer_compute_overlap_proven',False),
        peer_cut_quorum_wait_seconds=audit.get('peer_cut_quorum_wait_seconds',{}),
        reference_comparison_cases=sum(x['kind']=='reference_comparison' for x in events),
        reference_comparison_failures=sum(x['kind']=='reference_comparison' and not x['exact'] for x in events),
        comparison_method='independent_nine_state_sha256' if summary.get('reference_run') else 'in_process_full_tensor_replay' if summary['oracle'] else 'none',
        independent_audit=audit,diagnostic_timing=summary.get('diagnostic_timing'),
        expected_failure_pass=summary.get('expected_failure_pass'),
        gradient_audit_cases=sum(x['kind']=='outer_gradient_audit' for x in events),
        gradient_audit_failures=sum(x['kind']=='outer_gradient_audit' and not x['exact'] for x in events),
        oracle_rank_steps=sum(x['kind']=='oracle_step' for x in events),
        oracle_failures=sum(x['kind']=='oracle_step' and not x['exact'] for x in events),
        tensor_unequal=sum(v['unequal'] for x in events if x['kind']=='done' and x['comparison']
                           for v in x['comparison']['tensors'].values()),
        warning_categories_present=[name for name,pattern in warning_patterns.items() if pattern in log_text],
        outer_manager_instances=sum(x['kind']=='outer_ready' for x in events),
        effective_deadlines=[dict(timeout_seconds=t,quorum_timeout_seconds=q,connect_timeout_seconds=c)
            for t,q,c in sorted({(x['timeout_seconds'],x['quorum_timeout_seconds'],x['connect_timeout_seconds'])
                                for x in events if x['kind']=='outer_ready'})],
        raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.jsonl')},
        limitations=[('two replicas share hosts; D/E compute before activation and wait at the shared two-replica quorum'
                      if summary.get('overlap_peer') else
                      'two replicas share hosts; both pause; healthy Managers retained' if summary.get('retain_outer_managers') else
                      'two replicas share hosts; both pause and rebuild outer Managers at migration'),
            'restore failure is bounded whole-job fail-stop; rollback and fault isolation unverified',
            '120s controlled defaults do not resolve the historical v3 2s outer deadline failure',
            'native NCCL teardown warnings are retained as categories; resource audit is separate',
            'raw events/logs removed after summarization; hashes cannot reconstruct trajectories'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    result=verify(a.directory);(a.directory/'independent_outer_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    if not result['pass_all']:raise SystemExit(1)
