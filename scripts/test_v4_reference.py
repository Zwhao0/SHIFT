"""Independent reference coverage, provenance and live Manager lifecycle guards."""
from concurrent.futures import Future
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from shift_full.reference_audit import REFERENCE_HASH_KEYS
from shift_full.v4_reference import (compare_signature, sample_ids, training_profile,
    validate_compatibility, validate_signatures)
from shift_full.v4_outer import MigrationNamespace, Outer
from scripts.verify_v4_reference import verify, verify_comparisons, verify_retention


def reference():
    return dict(schema=1, steps=2, seq_len=512, microbatches=4, seed=20261106,cross_node_tp=False,
        training_profile=training_profile({}),
        source_sha256={'train':'123'},workers={f'{r}/{k}':{str(s):dict(
            hashes={key:'a'*64 for key in REFERENCE_HASH_KEYS},sample_ids=sample_ids(s,r,k,4),
            outer_manager=dict(step=s,batches_committed=2*s)) for s in range(3)}
            for r in range(2) for k in range(8)})


class ReferenceTest(unittest.TestCase):
    def test_full_reference_and_exact_comparison(self):
        value=validate_signatures(reference());state=value['workers']['1/7']['2']
        self.assertTrue(compare_signature(value,replica=1,rank=7,step=2,
            hashes=state['hashes'],samples=state['sample_ids'],manager=state['outer_manager']))
        self.assertFalse(compare_signature(value,replica=1,rank=7,step=2,
            hashes={},samples=state['sample_ids'],manager=state['outer_manager']))

    def test_missing_trajectory_hash_worker_and_bad_sample_rejected(self):
        for change in (lambda v:v['workers'].pop('0/0'),
            lambda v:v['workers']['0/0'].pop('1'),
            lambda v:v['workers']['0/0']['1']['hashes'].pop('cuda_rng'),
            lambda v:v['workers']['0/0']['1'].update(sample_ids=[999]),
            lambda v:v['workers']['0/0']['1']['outer_manager'].update(batches_committed=1)):
            value=reference();change(value)
            with self.assertRaises(ValueError):validate_signatures(value)

    def test_wrong_source_and_dimensions_rejected(self):
        config=dict(plan=dict(seq_len=512,microbatches=4),seed=20261106)
        validate_compatibility(reference(),config,{'train':'123'})
        with self.assertRaises(ValueError):validate_compatibility(reference(),config,{'train':'456'})
        config['seed']=0
        with self.assertRaises(ValueError):validate_compatibility(reference(),config,{'train':'123'})

    def test_training_profile_is_required(self):
        value=reference();value.pop('training_profile')
        config=dict(plan=dict(seq_len=512,microbatches=4),seed=20261106)
        with self.assertRaisesRegex(ValueError,'training profile'):
            validate_compatibility(value,config,{'train':'123'})

    def test_async_profile_mismatch_rejected(self):
        config=dict(plan=dict(seq_len=512,microbatches=4),seed=20261106,
            outer_manager=True,outer_audit=True,overlap_peer=True)
        value=reference();value['training_profile']=training_profile(config)
        validate_compatibility(value,config,{'train':'123'})
        config['overlap_peer']=False
        with self.assertRaisesRegex(ValueError,'training profile'):
            validate_compatibility(value,config,{'train':'123'})

    def test_retain_profile_mismatch_rejected(self):
        config=dict(plan=dict(seq_len=512,microbatches=4),seed=20261106,
            outer_manager=True,retain_outer_managers=True,transactional_switch=True)
        value=reference();value['training_profile']=training_profile(config)
        validate_compatibility(value,config,{'train':'123'})
        config['retain_outer_managers']=False
        with self.assertRaisesRegex(ValueError,'training profile'):
            validate_compatibility(value,config,{'train':'123'})

    def test_uncovered_step_is_error(self):
        with self.assertRaises(ValueError):
            compare_signature(reference(),replica=0,rank=0,step=3,hashes={},samples=[],manager={})


class RetentionTest(unittest.TestCase):
    def outer(self, rank):
        value=Outer.__new__(Outer);future=Future();future.set_result(None)
        state=dict(step=2,batches_committed=4)
        value.manager=SimpleNamespace(_quorum_future=future,_quorum_id=7,
            state_dict=lambda:deepcopy(state),current_step=lambda:state['step'])
        value._retain=True;value._closed=False;value._step_started=None
        value._audit=False;value._async=False
        value.epoch=0;value.rank=rank;value.pg=SimpleNamespace(migration_epoch=0)
        value.reference_pg=None
        return value

    def test_unaffected_rank_keeps_quorum_and_channel(self):
        value=self.outer(0);manager=value.manager;pg=value.pg;value.migrate(1)
        self.assertIs(value.manager,manager);self.assertIs(value.pg,pg)
        self.assertEqual((manager._quorum_id,pg.migration_epoch,value.state_dict()['step']),(7,0,2))

    def test_affected_peer_requires_new_namespace_and_reconfigure(self):
        value=self.outer(4);value.migrate(1)
        self.assertIsNone(value.manager._quorum_id)
        self.assertEqual(value.pg.migration_epoch,1)
        with self.assertRaises(ValueError):value.migrate(1)

    def test_active_or_unfinished_quorum_cannot_migrate(self):
        value=self.outer(4);value._step_started=2
        with self.assertRaises(RuntimeError):value.migrate(1)
        value._step_started=None;value.manager._quorum_future=Future()
        with self.assertRaises(RuntimeError):value.migrate(1)

    def test_live_manager_cannot_rewind(self):
        value=self.outer(0)
        with self.assertRaises(RuntimeError):value.load_state_dict(dict(step=1,batches_committed=2))

    def test_namespace_changes_even_for_same_quorum_id(self):
        class Parent:
            def configure(self,store,*args,**kwargs):return store
        class Scoped(MigrationNamespace,Parent):pass
        pg=Scoped();before=pg.configure('store/quorum7/rank4');pg.migration_epoch=1
        after=pg.configure('store/quorum7/rank4')
        self.assertNotEqual(before,after);self.assertEqual(pg.configure_count,2)


class ReferenceEvidenceTest(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup);self.root=Path(directory.name)
        self.summary=dict(name='v4_ref',reference_only=True,outer_manager=True,outer_audit=True,
            overlap_peer=True,retain_outer_managers=True,transactional_switch=True,
            cross_node_tp=False,deterministic=False,functional_pass=True,error=None,
            cleanup_errors=[],outer_timeout=120,source_sha256={'train':'123'},
            plan=dict(cut=1,post_steps=1,seq_len=512,microbatches=4))
        self.reference=reference();self.reference['run']='v4_ref'
        self.reference['training_profile']=training_profile(self.summary)
        self.save('summary.json',self.summary);self.save('reference_signatures.json',self.reference)
        resources={n:dict(company={str(i):'VLLM' for i in range(8)},extra_gpu=[]) for n in ('node0','node1','node3')}
        self.save('resources_before.json',resources);self.save('resources_after.json',resources)
        self.rows={}
        for role in 'ABDE':
            self.rows[role]={}
            for device in range(4):
                replica=int(role in 'DE');rank=device+(4 if role in 'BE' else 0);events=[]
                def add(kind,**fields):
                    events.append(dict(kind=kind,role=role,device=device,rank=rank,replica=replica,pid=10+rank,**fields))
                add('outer_ready',outer_epoch=0,timeout_seconds=120,quorum_timeout_seconds=120,connect_timeout_seconds=120)
                for step in range(3):
                    add('reference_step',step=step,**self.reference['workers'][f'{replica}/{rank}'][str(step)])
                    if step:
                        add('outer_begin',step=step-1,participants=2,participating_rank=replica)
                        add('outer_gradient_audit',step=step-1,exact=True,buckets=1)
                        add('outer_commit',step=step-1,committed_step=step,batches_committed=2*step)
                        add('commit',step=step,phase='reference',epoch=0,finite=True,sample_ids=sample_ids(step,replica,rank,4))
                add('done',step=2,identity_preserved=True)
                self.rows[role][device]=events;self.save_events(role,device)

    def save(self,name,value):self.root.joinpath(name).write_text(json.dumps(value))
    def save_events(self,role,device):
        self.root.joinpath(f'{role}{device}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in self.rows[role][device]))

    def test_complete_no_migration_reference_passes(self):self.assertTrue(verify(self.root)['pass_all'])

    def test_missing_avg_and_rewind_are_rejected(self):
        events=self.rows['A'][0]
        events[:]=[x for x in events if x['kind']!='outer_gradient_audit'];self.save_events('A',0)
        self.assertFalse(verify(self.root)['pass_all'])
        events.append(dict(events[0],kind='outer_restored'));self.save_events('A',0)
        self.assertTrue(any('rewind' in e for e in verify(self.root)['errors']))

    def test_saved_signatures_must_match_raw_events(self):
        self.reference['workers']['0/0']['1']['hashes']['model']='b'*64
        self.save('reference_signatures.json',self.reference)
        self.assertFalse(verify(self.root)['pass_all'])

    def test_empty_resources_rejected(self):
        self.save('resources_before.json',{});self.save('resources_after.json',{})
        self.assertFalse(verify(self.root)['pass_all'])

    def test_comparison_does_not_trust_exact_flag(self):
        self.summary['reference_sha256']=hashlib.sha256((self.root/'reference_signatures.json').read_bytes()).hexdigest()
        rows={}
        for role,devices in self.rows.items():
            rows[role]={}
            for device,events in devices.items():
                rows[role][device]=[dict(x,kind='reference_initial' if x['step']==0 else 'reference_comparison',exact=True)
                    if x['kind']=='reference_step' else x for x in events]
        rows['A'][0][1]['hashes']={k:'c'*64 for k in REFERENCE_HASH_KEYS}
        self.assertTrue(verify_comparisons(self.root,self.summary,rows))

    def test_retention_rejects_rebuilt_manager_and_wrong_namespace(self):
        identity=dict(manager=1,server=2,client=3,executor=4,pg=5,service_epoch=0)
        training=dict(model=1,optimizer=2,ddp=3,parameters=[[4,5]],adam=[[6,7]],buckets=[[8,9]])
        events=[dict(kind='outer_ready',service_epoch=0),
            dict(kind='outer_retention_boundary',identity=identity,training_identity=training,
                 manager_state=dict(step=2,batches_committed=4),monotonic=1),
            dict(kind='outer_retained',identity=deepcopy(identity),training_identity=training,
                 manager_state=dict(step=2,batches_committed=4),monotonic=2),
            dict(kind='outer_begin',outer_epoch=1,migration_namespace=1,pg_configures=2)]
        rows={'E':{0:events}}
        self.assertEqual(verify_retention(rows,2),[])
        events[2]['identity']['manager']=99
        self.assertTrue(verify_retention(rows,2))
        events[2]['identity']['manager']=1;events[-1]['migration_namespace']=0
        self.assertTrue(verify_retention(rows,2))


if __name__=='__main__':unittest.main()
