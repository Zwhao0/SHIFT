"""Runtime admission must validate a real restore, independently of hash audit."""
import copy
from pathlib import Path
import unittest
from shift_full.v4_state import boundary_validation,validate_candidate_state
from shift_full.v4_transaction import Transaction


class RuntimeAdmissionTest(unittest.TestCase):
    def tx(self):return Transaction(1,'B','C',12,(4,5,6,7),validation='runtime')
    def report(self,rank):return dict(schema=1,epoch=1,rank=rank,cut=12,target='C',restored=True)

    def test_four_completed_restores_required(self):
        tx=self.tx()
        for rank in (4,5,6):self.assertTrue(tx.ready(1,rank,self.report(rank)))
        with self.assertRaises(ValueError):tx.decide()
        self.assertTrue(tx.ready(1,7,self.report(7)));self.assertEqual(tx.decide(),'replace')
        with self.assertRaises(ValueError):tx.ready(1,7,self.report(7))

    def test_wrong_identity_version_or_unfinished_restore_rejected(self):
        for key,value in [('epoch',0),('rank',5),('cut',11),('target','B'),('restored',False),('epoch',True)]:
            with self.subTest(key=key,value=value):
                tx=self.tx();report=self.report(4);report[key]=value
                self.assertFalse(tx.ready(1,4,report))
                for rank in (5,6,7):tx.ready(1,rank,self.report(rank))
                self.assertEqual(tx.decide(),'retain')

    def test_empty_hashes_cannot_approve_runtime_candidate(self):
        with self.assertRaises(ValueError):self.tx().candidate(1,4,{}, {})
        tx=Transaction(1,'B','C',12,(4,5,6,7))
        with self.assertRaises(ValueError):tx.ready(1,4,self.report(4))
        self.assertFalse(tx.candidate(1,4,{},{}))

    def test_explicit_modes_parse_through_both_real_entrypoints(self):
        from scripts.test_v41_defaults import parse_actual_main
        for name in ('start_v4.py','v4_remote.py'):
            parsed=parse_actual_main(Path(__file__).with_name(name),['--performance-no-audit','--boundary-validation','runtime','--digest-engine','staged','--final-reference-run','ref'])
            self.assertTrue(parsed.performance_no_audit)
            self.assertEqual((parsed.boundary_validation,parsed.digest_engine,parsed.final_reference_run),('runtime','staged','ref'))

    def test_policy_defaults_and_invalid_value(self):
        self.assertEqual(boundary_validation({}),'full')
        self.assertEqual(boundary_validation({'performance_no_audit':True}),'runtime')
        self.assertEqual(boundary_validation({'performance_no_audit':True,'boundary_validation':'full'}),'full')
        with self.assertRaises(ValueError):boundary_validation({'boundary_validation':'skip'})


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'use the installed server torch environment')
class StateStructureTest(unittest.TestCase):
    def setUp(self):
        self.stage=torch.nn.Linear(3,2);self.stage.register_buffer('count',torch.zeros(1))
        self.optimizer=torch.optim.AdamW(self.stage.parameters())
        self.stage(torch.ones(1,3)).sum().backward();self.optimizer.step()
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda step:1.)
        self.rng=dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.get_rng_state(),
                      python_rng=__import__('random').getstate(),tracker_rng={'tracker':torch.get_rng_state()})
        self.packet=dict(schema=2,source_replica=0,source_rank=4,step=12,progress={'updates':12},
                         model=self.stage.state_dict(),all_buffers=dict(self.stage.named_buffers()),
                         optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict(),
                         hashes={},v4_epoch=1,v4_owner='C',switch_round=0,
                         outer_manager=dict(step=12,batches_committed=24),**self.rng)
        self.packet=copy.deepcopy(self.packet)

    def validate(self,**kwargs):
        return validate_candidate_state(self.packet,rank=4,cut=12,epoch=1,target='C',round_id=0,
            source_replica=0,stage=self.stage,optimizer=self.optimizer,scheduler=self.scheduler,
            rng=self.rng,progress={'updates':0},**kwargs)

    def test_complete_packet_and_changed_values_are_structurally_valid(self):
        self.validate();self.packet['model']['weight'].add_(1);self.validate()

    def test_post_native_history_preserves_real_manager_counter(self):
        self.packet['outer_manager']['batches_committed']=18
        with self.assertRaises(ValueError):self.validate()
        self.validate(fixed_replicas=False)
        for bad in (11,25,True):
            self.packet['outer_manager']['batches_committed']=bad
            with self.assertRaises(ValueError):self.validate(fixed_replicas=False)

    def test_bad_model_or_adam_shape_is_rejected_before_load(self):
        for section in ('model','optimizer'):
            self.setUp()
            if section=='model':self.packet['model']['weight']=torch.zeros(2,4)
            else:next(iter(self.packet['optimizer']['state'].values()))['exp_avg']=torch.zeros(1)
            with self.subTest(section=section),self.assertRaises(ValueError):self.validate()

    def test_missing_buffer_rng_and_old_cursor_rejected(self):
        for section in ('all_buffers','torch_rng','step','outer_manager','v4_owner','source_replica'):
            self.setUp()
            if section=='all_buffers':self.packet[section]={}
            elif section=='torch_rng':self.packet[section]=torch.zeros(2,dtype=torch.uint8)
            elif section=='step':self.packet[section]=11
            elif section=='outer_manager':self.packet[section]['batches_committed']=22
            elif section=='v4_owner':self.packet[section]='B'
            else:self.packet[section]=1
            with self.subTest(section=section),self.assertRaises(ValueError):self.validate()


if __name__=='__main__':unittest.main()
