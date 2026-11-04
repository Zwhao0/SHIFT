"""Reject incomplete/stale role or relay packets before migration activation."""
import copy
import hashlib
import unittest
from shift_full.planned_state import validate_state, validate_relay, role_hosts, merge_role_state
from shift_full.reference_audit import REFERENCE_HASH_KEYS


def digest(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()


def state(step=7, epoch=1):
    return dict(schema=2,step=step,progress=dict(updates=step),source_rank=3,
                source_replica=1,model={},optimizer={},scheduler={},all_buffers={},
                torch_rng=b'cpu',cuda_rng=b'cuda',python_rng=(1,),tracker_rng={'tp':b'tp'},
                hashes={key:'hash' for key in REFERENCE_HASH_KEYS},planned_epoch=epoch,planned_role=1)


def packet(step=7,epoch=1):
    tail=state(step,epoch)
    return dict(schema=2,step=step,sample_step=step+1,planned_epoch=epoch,planned_role=1,
                sample_ids=[56,60],gradients=[1,2],gradient_hash=digest([1,2]),tail=tail,tail_hash=digest(tail))


class PlannedStateTest(unittest.TestCase):
    def validate(self,payload,epoch=1):
        return validate_state(payload,rank=3,step=7,epoch=epoch,audit=True)

    def test_complete_state_and_all_nine_hashes_required(self):
        self.validate(state())
        for field in ('schema','tracker_rng','all_buffers','python_rng','progress'):
            with self.subTest(field=field):
                value=state();del value[field]
                with self.assertRaises(ValueError):self.validate(value)
        value=state();del value['hashes']['tracker_rng']
        with self.assertRaises(ValueError):self.validate(value)

    def test_progress_rank_role_and_epoch_rejected_before_load(self):
        for fields in ({'step':6},{'source_rank':0},{'planned_epoch':0},{'planned_role':0},{'progress':{'updates':6}}):
            with self.subTest(fields=fields),self.assertRaises(ValueError):self.validate(dict(state(),**fields))

    def test_relay_checks_tail_as_well_as_gradients(self):
        value=packet()
        validate_relay(value,rank=3,step=7,epoch=1,digest=digest,audit=True)
        for mutate in (lambda p:p['tail']['tracker_rng'].update(tp=b'corrupt'),
                       lambda p:p['gradients'].append(3),
                       lambda p:p.update(sample_step=7),lambda p:p['tail'].update(planned_epoch=0)):
            value=packet();mutate(value)
            with self.assertRaises(ValueError):validate_relay(value,rank=3,step=7,epoch=1,digest=digest,audit=True)

    def test_version_and_rng_requirements_remain_without_audit(self):
        value=packet();del value['tail']['tracker_rng']
        with self.assertRaises(ValueError):validate_relay(value,rank=3,step=7,epoch=1,digest=digest,audit=False)

    def test_multiple_cycles_never_accept_previous_epoch_packet(self):
        for epoch in range(1,5):
            value=packet(7+epoch,epoch)
            validate_relay(value,rank=3,step=7+epoch,epoch=epoch,digest=digest,audit=True)
            with self.assertRaises(ValueError):validate_relay(value,rank=3,step=7+epoch,epoch=epoch+1,digest=digest,audit=True)

    def test_A_shared_state_and_B_persistent_buffers_have_coherent_hashes(self):
        donor=state();donor['model']={'weight':10,'persistent':1};donor['optimizer']={'adam':99}
        role={k:v for k,v in state().items() if k not in ('model','optimizer','scheduler','hashes','source_replica')}
        role.update(all_buffers={'persistent':2,'scratch':3},tracker_rng={'tp':b'B'})
        merged=merge_role_state(donor,role,rank=3,step=7,epoch=1,digest=digest,audit=True)
        self.assertEqual(merged['model'],{'weight':10,'persistent':2})
        self.assertEqual(merged['optimizer'],{'adam':99})
        self.assertEqual(merged['hashes']['model'],digest(merged['model']))
        self.assertEqual(merged['hashes']['tracker_rng'],digest(role['tracker_rng']))
        self.assertEqual(donor['model']['persistent'],1)

    def test_three_rotations_and_alternating_roles(self):
        nodes=['host0','host1','host2']
        for i in range(3):
            hosts=dict(zip(('A','B','C'),nodes[i:]+nodes[:i]))
            self.assertEqual(role_hosts(hosts,0),hosts)
            self.assertEqual(role_hosts(hosts,1),dict(A=hosts['A'],B=hosts['C'],C=hosts['B']))
            self.assertEqual(role_hosts(hosts,2),hosts)
        with self.assertRaises(ValueError):role_hosts(dict(A='same',B='same',C='third'),0)

if __name__=='__main__':unittest.main()
