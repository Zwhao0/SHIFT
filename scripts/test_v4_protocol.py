import unittest
from shift_full.v4_protocol import Ownership, Plan, state_channel_matches
from shift_full.reference_audit import REFERENCE_HASH_KEYS


class ProtocolTest(unittest.TestCase):
    def test_gpu_channel_requires_matching_bounded_control_and_payload(self):
        packet=dict(transport='packed_gpu',bytes=1040,gpu_bytes=1024,control_bytes=16,
                    cpu_tensor_bytes=8,hashes={'model':'a'},oracle_hashes={'model':'b'})
        self.assertTrue(state_channel_matches(packet,dict(packet)))
        for key,value in [('gpu_bytes',0),('bytes',1041),('control_bytes',17*1024**2),
                          ('cpu_tensor_bytes',5*1024**2),('hashes',{}),('oracle_hashes',None),
                          ('transport','cpu_socket')]:
            self.assertFalse(state_channel_matches(packet,dict(packet,**{key:value})))
        missing=dict(packet);del missing['gpu_bytes']
        self.assertFalse(state_channel_matches(packet,missing))

    def test_replacement_preserves_complete_logical_coverage(self):
        before={Plan.logical_rank(r,d) for r in 'AB' for d in range(4)}
        after={Plan.logical_rank(r,d) for r in 'AC' for d in range(4)}
        self.assertEqual(before,set(range(8)));self.assertEqual(before,after)

    def test_cross_node_tp_preserves_rank_coverage(self):
        self.assertEqual([Plan.logical_rank('A',d,True) for d in range(4)],[0,2,4,6])
        self.assertEqual([Plan.logical_rank('B',d,True) for d in range(4)],[1,3,5,7])
        self.assertEqual([Plan.logical_rank('C',d,True) for d in range(4)],[1,3,5,7])

    def test_old_source_and_early_joiner_cannot_commit(self):
        for role,epoch in [('B',1),('C',0),('A',2),('A',True)]:
            with self.assertRaises(ValueError):Plan.authorize(role,epoch)

    def test_retained_worker_cannot_revive_old_epoch(self):
        gate=Ownership();gate.authorize('A',0);gate.advance();gate.authorize('A',1)
        for role,epoch in [('A',0),('B',0),('B',1)]:
            with self.assertRaises(ValueError):gate.authorize(role,epoch)
        with self.assertRaises(ValueError):gate.advance()

    def test_stale_or_partial_state_never_admitted(self):
        value=dict(schema=2,model={},optimizer={'state':{0:{'step':4}}},scheduler={},
            all_buffers={},torch_rng=None,cuda_rng=None,python_rng=None,tracker_rng={},
            progress={'updates':4},source_rank=4,source_replica=0,step=4,
            hashes={key:'hash' for key in REFERENCE_HASH_KEYS},v4_epoch=1,v4_owner='C')
        Plan().admit(value,4)
        for patch in [dict(step=3),dict(source_rank=5),dict(v4_epoch=0),dict(v4_owner='B'),
                      dict(optimizer={'state':{}}),dict(hashes={'model':'hash'})]:
            with self.assertRaises(ValueError):Plan().admit(dict(value,**patch),4)


if __name__=='__main__':unittest.main()
