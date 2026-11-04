"""Ensure the independent audit rejects missing Manager and DP evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.verify_v4_outer import summarize, verify
from shift_full.reference_audit import REFERENCE_HASH_KEYS


class OuterAuditTest(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        summary=dict(name='v4_outer_fixture',topology={},outer_manager=True,outer_audit=True,
                     outer_timeout=120,oracle=False,functional_pass=True,oracle_exact=None,
                     source_sha256={},resources_clean=True,cleanup_errors=[],
                     fail_restore_rank=-1,defer_source_local_close=True,plan=dict(cut=1,post_steps=1,microbatches=1))
        (self.root/'summary.json').write_text(json.dumps(summary))
        resources={name:dict(company={str(i):'VLLM' for i in range(8)},extra_gpu=[]) for name in ('node0','node1','node3')}
        for filename in ('resources_before.json','resources_after.json'):
            (self.root/filename).write_text(json.dumps(resources))
        for role in 'ABCDE':
            for device in range(4):
                replica=int(role in 'DE');rank=device+(4 if role in 'BCE' else 0)
                hashes={key:f'{rank}/{key}' for key in REFERENCE_HASH_KEYS};events=[]
                def add(kind,**fields):
                    events.append(dict(kind=kind,role=role,replica=replica,rank=rank,device=device,pid=100+rank,
                                       monotonic=len(events)+1,**fields))
                for epoch in ([0] if role=='B' else [1] if role=='C' else [0,1]):
                    add('outer_ready',outer_epoch=epoch,timeout_seconds=120,quorum_timeout_seconds=120,
                        connect_timeout_seconds=120,service_epoch=epoch)
                if role!='C':
                    add('boundary_selected',step=1)
                    add('freeze',hashes=hashes)
                phases=[('before',1)] if role=='B' else [('after',2)] if role=='C' else [('before',1),('after',2)]
                for phase,step in phases:
                    epoch=0 if phase=='before' else 1
                    add('outer_begin',outer_epoch=epoch,step=step-1,participants=2,participating_rank=replica)
                    add('outer_gradient_audit',outer_epoch=epoch,step=step-1,exact=True,buckets=1)
                    add('outer_commit',outer_epoch=epoch,step=step-1,committed_step=step,batches_committed=2*step)
                    add('commit',epoch=epoch,phase=phase,step=step,finite=True,
                        sample_ids=[(step-1)*4+replica*2+device//2])
                packet=dict(transport='packed_gpu',bytes=1088,gpu_bytes=1024,control_bytes=64,cpu_tensor_bytes=8,
                            hashes=hashes,oracle_hashes=None)
                if role=='B':
                    add('old_owner_rejected');add('state_sent',**packet)
                    add('activation_observed');add('source_local_groups_closed',discarded=12);add('retired')
                else:
                    if role=='C':add('restored',hashes=hashes);add('state_received',**packet)
                    add('outer_restored',step=1,batches_committed=2,outer_epoch=1)
                    add('activated',hashes=hashes)
                    if role in 'DE':add('peer_inner_groups_retained',identity_preserved=True)
                    add('done',step=2,identity_preserved=True,hashes=hashes,comparison=None)
                self.save(role,device,events)

    def save(self,role,device,events):
        (self.root/f'{role}{device}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))

    def change(self,kind,change):
        events=self.read('A',0)
        for event in events:
            if event['kind']==kind:change(event);break
        self.save('A',0,events)

    def read(self,role,device):
        return [json.loads(x) for x in (self.root/f'{role}{device}.jsonl').read_text().splitlines()]

    def update_summary(self,**fields):
        path=self.root/'summary.json';summary=json.loads(path.read_text());summary.update(fields)
        path.write_text(json.dumps(summary))

    def add_event(self,role,device,kind,**fields):
        events=self.read(role,device)
        replica=int(role in 'DE');rank=device+(4 if role in 'BCE' else 0)
        events.append(dict(kind=kind,role=role,replica=replica,rank=rank,device=device,
                           pid=100+rank,monotonic=len(events)+1,**fields))
        self.save(role,device,events)

    def enable_oracle(self):
        self.update_summary(oracle=True)
        for role in 'ABDE':
            replica=int(role in 'DE')
            for device in range(4):
                events=self.read(role,device)
                new=[]
                for event in events:
                    if event['kind']=='outer_begin' and event.get('outer_epoch')==1:
                        break
                    new.append(event)
                step=1
                new.extend([
                    dict(kind='outer_begin',role=role,replica=replica,rank=device+(4 if role in 'BCE' else 0),
                         device=device,pid=100+device+(4 if role in 'BCE' else 0),monotonic=100+len(new),
                         outer_epoch=0,step=step,participants=2,participating_rank=replica),
                    dict(kind='outer_gradient_audit',role=role,replica=replica,
                         rank=device+(4 if role in 'BCE' else 0),device=device,
                         pid=100+device+(4 if role in 'BCE' else 0),monotonic=101+len(new),
                         outer_epoch=0,step=step,exact=True,buckets=1),
                    dict(kind='outer_commit',role=role,replica=replica,
                         rank=device+(4 if role in 'BCE' else 0),device=device,
                         pid=100+device+(4 if role in 'BCE' else 0),monotonic=102+len(new),
                         outer_epoch=0,step=step,committed_step=step+1,batches_committed=2*(step+1)),
                    dict(kind='commit',role=role,replica=replica,
                         rank=device+(4 if role in 'BCE' else 0),device=device,
                         pid=100+device+(4 if role in 'BCE' else 0),monotonic=103+len(new),
                         epoch=0,phase='oracle',step=step+1,finite=True,
                         sample_ids=[4+replica*2+device//2]),
                ])
                if role!='B':
                    for event in events[len(new)-4:]:
                        pass
                    new.extend(events[len(new)-4:])
                    new.extend(events[len(new):])
                else:
                    new.extend(events[len(new):])
                self.save(role,device,new)
        for role in 'ACDE':
            for device in range(4):
                events=self.read(role,device)
                for event in events:
                    if event['kind']=='done':
                        event['comparison']=dict(
                            exact={key:True for key in REFERENCE_HASH_KEYS},
                            trajectory_exact=True,manager_exact=True,
                            tensors={name:dict(elements=1,unequal=0,max_abs=0.0)
                                     for name in ('model','optimizer','all_buffers')})
                self.save(role,device,events)
                self.add_event(role,device,'oracle_step',step=2,exact=True)

    def configure_failure(self):
        self.update_summary(fail_restore_rank=4,expected_failure_pass=True)
        for role in 'ABCDE':
            for device in range(4):
                events=self.read(role,device)
                events=[event for event in events if not
                        (event['kind']=='commit' and event.get('phase')=='after' or
                         event['kind'] in ('outer_begin','outer_gradient_audit','outer_commit') and
                         event.get('outer_epoch')==1 or
                         event['kind'] in ('activated','done','outer_restored','peer_inner_groups_retained'))]
                self.save(role,device,events)
        self.add_event('C',0,'restore_failure_injected',step=1)

    def enable_overlap_peer(self):
        self.update_summary(overlap_peer=True,retain_outer_managers=False)
        for role in 'ABCDE':
            for device in range(4):
                events=self.read(role,device)
                if role in 'DE':
                    events=[x for x in events if not
                            (x['kind']=='outer_ready' and x.get('outer_epoch')==1 or
                             x['kind'] in ('outer_restored','activated'))]
                    for event in events:
                        if event['kind']=='outer_ready':event['service_epoch']=0
                    rank=device+(4 if role=='E' else 0)
                    meta=dict(role=role,replica=1,rank=rank,device=device,pid=100+rank)
                    identity=dict(manager=10+rank,server=20+rank,client=30+rank,executor=40+rank,
                                  pg=50+rank,service_epoch=0)
                    training=dict(model=60+rank,optimizer=70+rank,ddp=80+rank,
                                  parameters=[90+rank],adam=[100+rank],buckets=[110+rank])
                    retained_fields=dict(meta,step=1,manager_state=dict(step=1,batches_committed=2),
                                         identity=identity,training_identity=training)
                    boundary=dict(kind='outer_retention_boundary',**retained_fields)
                    retained_event=dict(kind='outer_retained',**retained_fields,channel_reconnected=(role=='E'))
                    frozen_index=next(i for i,x in enumerate(events)
                                      if x['kind']=='outer_begin' and x.get('outer_epoch')==1)
                    peer_ready=dict(kind='peer_compute_ready',**meta,step=1,
                                    hashes=next(x['hashes'] for x in events if x['kind']=='freeze'),
                                    identity_preserved=True)
                    events[frozen_index:frozen_index]=[boundary,retained_event,peer_ready]
                else:
                    for event in events:
                        if event['kind']=='activated':
                            event['peer_compute_started']=(role=='A' and device==0)
                            event['peer_gradients_ready']=False
                for event in events:
                    if event['kind']=='commit' and event.get('phase')=='after':
                        event['global_activation_observed']=True
                rebuilt=[]
                for event in events:
                    if event['kind']!='outer_begin':
                        rebuilt.append(event)
                        continue
                    replica=int(role in 'DE');rank=device+(4 if role in 'BCE' else 0)
                    epoch=event['outer_epoch'];step=event['step']
                    marker_base=dict(role=role,replica=replica,rank=rank,device=device,pid=100+rank,
                                     outer_epoch=epoch,step=step)
                    if role in 'DE':
                        event['migration_namespace']=1 if epoch==1 and rank>=4 else 0
                        event['pg_configures']=2 if epoch==1 and role=='E' else 1
                    rebuilt.append(dict(kind='outer_quorum_requested',**marker_base))
                    peer_cut=(role in 'DE' and epoch==1 and step==1)
                    if peer_cut:
                        rebuilt.append(dict(kind='peer_compute_begin',role=role,replica=replica,
                                            rank=rank,device=device,pid=100+rank,step=1))
                        rebuilt.append(dict(kind='peer_native_gradient_ready',role=role,replica=replica,
                                            rank=rank,device=device,pid=100+rank,step=1))
                    rebuilt.append(dict(kind='outer_quorum_wait',seconds=0.025,**marker_base))
                    rebuilt.append(event)
                for index,event in enumerate(rebuilt,1):event['monotonic']=index
                self.save(role,device,rebuilt)

    def test_valid_two_replica_evidence_passes(self):
        self.assertTrue(verify(self.root)['pass_all'])

    def test_common_verifier_dispatches_to_outer_audit(self):
        from scripts.verify_v4 import verify as common_verify
        self.assertTrue(common_verify(self.root)['pass_all'])

    def test_manager_progress_mismatch_is_rejected(self):
        self.change('outer_commit',lambda x:x.update(batches_committed=99))
        self.assertFalse(verify(self.root)['pass_all'])

    def test_independent_avg_mismatch_is_rejected(self):
        self.change('outer_gradient_audit',lambda x:x.update(exact=False))
        self.assertFalse(verify(self.root)['pass_all'])

    def test_duplicate_sample_assignment_is_rejected(self):
        self.change('commit',lambda x:x.update(sample_ids=[2]))
        self.assertFalse(verify(self.root)['pass_all'])

    def test_replica_parameter_divergence_is_rejected(self):
        self.change('done',lambda x:x['hashes'].update(model='wrong'))
        self.assertFalse(verify(self.root)['pass_all'])

    def test_missing_resource_node_is_rejected(self):
        path=self.root/'resources_after.json';resources=json.loads(path.read_text());resources.pop('node3')
        path.write_text(json.dumps(resources))
        self.assertFalse(verify(self.root)['pass_all'])

    def test_duplicate_boundary_cannot_mask_missing_worker_boundary(self):
        a0=self.read('A',0);a0=[x for x in a0 if x['kind']!='boundary_selected'];self.save('A',0,a0)
        a1=self.read('A',1);boundary=next(x for x in a1 if x['kind']=='boundary_selected')
        a1.append(dict(boundary));self.save('A',1,a1)
        self.assertFalse(verify(self.root)['pass_all'])

    def test_boundary_must_match_summary_cut(self):
        summary=json.loads((self.root/'summary.json').read_text());summary['plan']['cut']=2
        (self.root/'summary.json').write_text(json.dumps(summary))
        self.assertFalse(verify(self.root)['pass_all'])

    def test_manager_begin_and_gradient_steps_must_match(self):
        self.change('outer_begin',lambda x:x.update(step=7))
        self.assertFalse(verify(self.root)['pass_all'])
        self.setUp()
        self.change('outer_gradient_audit',lambda x:x.update(step=7))
        self.assertFalse(verify(self.root)['pass_all'])

    def test_disabled_gradient_audit_is_reported_not_rejected(self):
        self.update_summary(outer_audit=False)
        for role in 'ABCDE':
            for device in range(4):
                self.save(role,device,[x for x in self.read(role,device)
                                       if x['kind']!='outer_gradient_audit'])
        result=verify(self.root)
        self.assertTrue(result['pass_all'])
        self.assertFalse(result['independent_gradient_audit'])
        self.assertEqual(result['gradient_audit_status'],'disabled')

    def test_nine_hash_keys_are_required(self):
        self.change('freeze',lambda x:x['hashes'].pop('tracker_rng'))
        self.assertFalse(verify(self.root)['pass_all'])

    def test_oracle_steps_and_comparison_sections_must_be_complete(self):
        self.enable_oracle()
        self.change('oracle_step',lambda x:x.update(step=9))
        self.assertFalse(verify(self.root)['pass_all'])
        self.setUp();self.enable_oracle()
        events=self.read('A',0)
        for event in events:
            if event['kind']=='done':event['comparison']['exact'].pop('tracker_rng')
        self.save('A',0,events)
        self.assertFalse(verify(self.root)['pass_all'])
        self.setUp();self.enable_oracle()
        events=self.read('A',0)
        for event in events:
            if event['kind']=='done':event['comparison']['tensors'].pop('all_buffers')
        self.save('A',0,events)
        self.assertFalse(verify(self.root)['pass_all'])

    def test_expected_failure_requires_configured_restore_rank(self):
        self.configure_failure()
        self.assertTrue(verify(self.root)['pass_all'])
        events=self.read('C',0);events=[x for x in events if x['kind']!='restore_failure_injected']
        self.save('C',0,events);self.add_event('C',1,'restore_failure_injected',step=1)
        self.assertFalse(verify(self.root)['pass_all'])

    def test_epoch_zero_ready_is_required_for_existing_replicas(self):
        self.configure_failure()
        events=self.read('A',0);events=[x for x in events if not
                                        (x['kind']=='outer_ready' and x.get('outer_epoch')==0)]
        self.save('A',0,events)
        self.assertFalse(verify(self.root)['pass_all'])

    def test_overlap_peer_evidence_and_summary_pass(self):
        self.enable_overlap_peer()
        report=verify(self.root)
        self.assertTrue(report['pass_all'],report['errors'])
        self.assertTrue(report['peer_compute_overlap_proven'])
        self.assertEqual(set(report['peer_cut_quorum_wait_seconds']),
                         {f'D/rank{i}' for i in range(4)}|{f'E/rank{i}' for i in range(4,8)})
        reduced=summarize(self.root)
        self.assertTrue(reduced['overlap_peer'])
        self.assertTrue(reduced['peer_compute_overlap_proven'])
        self.assertEqual(reduced['peer_cut_quorum_wait_seconds'],report['peer_cut_quorum_wait_seconds'])
        self.assertFalse(any('both pause' in item for item in reduced['limitations']))

    def test_overlap_peer_missing_marker_is_rejected(self):
        self.enable_overlap_peer()
        events=self.read('D',0);events=[x for x in events if x['kind']!='peer_compute_begin']
        self.save('D',0,events)
        report=verify(self.root)
        self.assertFalse(report['pass_all'])
        self.assertFalse(report['peer_compute_overlap_proven'])

    def test_overlap_peer_rejects_after_commit_without_activation_marker(self):
        self.enable_overlap_peer()
        events=self.read('E',0)
        for event in events:
            if event['kind']=='commit' and event.get('phase')=='after':
                event['global_activation_observed']=False
        self.save('E',0,events)
        self.assertFalse(verify(self.root)['pass_all'])

    def test_overlap_peer_requires_quorum_wait(self):
        self.enable_overlap_peer()
        events=self.read('E',0);events=[x for x in events if not
                                        (x['kind']=='outer_quorum_wait' and x.get('outer_epoch')==1 and
                                         x.get('step')==1)]
        self.save('E',0,events)
        report=verify(self.root)
        self.assertFalse(report['pass_all'])
        self.assertFalse(report['peer_compute_overlap_proven'])

    def test_overlap_peer_requires_compute_marker_before_activation(self):
        self.enable_overlap_peer()
        events=self.read('A',0)
        for event in events:
            if event['kind']=='activated':event['peer_compute_started']=False
        self.save('A',0,events)
        self.assertFalse(verify(self.root)['pass_all'])


if __name__=='__main__':unittest.main()
