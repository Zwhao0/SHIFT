"""CPU regression: nonpersistent buffers and internal RNG must round-trip."""
import unittest
import torch
from shift_full.state_inventory import buffer_manifest, restore_buffers, tracker_snapshot, restore_tracker

class StateTest(unittest.TestCase):
    def test_nonpersistent_buffers_roundtrip_and_manifest_rejection(self):
        model=torch.nn.Module()
        model.register_buffer('cache',torch.tensor([3.]),persistent=False)
        model.register_buffer('scale',torch.tensor([4.]))
        self.assertNotIn('cache',model.state_dict())
        self.assertEqual({r['name']:r['persistent'] for r in buffer_manifest(model)},{'cache':False,'scale':True})
        frozen={n:b.clone() for n,b in model.named_buffers()}
        model.cache.add_(5)
        restore_buffers(model,frozen)
        self.assertEqual(model.cache.item(),3.)
        with self.assertRaisesRegex(RuntimeError,'manifest'):restore_buffers(model,{'scale':frozen['scale']})

    def test_internal_rng_snapshot_and_restore_do_not_alias(self):
        class Tracker:
            def get_states(self):return self.states
            def set_states(self,values):self.states=values
        tracker=Tracker();tracker.states={'model':torch.tensor([7],dtype=torch.uint8)}
        frozen=tracker_snapshot(tracker)
        tracker.states['model'].add_(1)
        self.assertEqual(frozen['model'].item(),7)
        restore_tracker(tracker,frozen);frozen['model'].add_(2)
        self.assertEqual(tracker.states['model'].item(),7)

    def test_ft02_managed_interface_avg_is_not_rescaled(self):
        from torchft.process_group import ManagedProcessGroup
        from unittest.mock import Mock
        manager=Mock();manager._pg=Mock()
        pg=ManagedProcessGroup.__new__(ManagedProcessGroup)
        pg._manager=manager
        value=torch.tensor([2.])
        result=pg.allreduce([value],torch.distributed.ReduceOp.AVG)
        manager.allreduce.assert_called_once_with(value,reduce_op=torch.distributed.ReduceOp.AVG)
        self.assertIs(result,manager.allreduce.return_value)

if __name__=='__main__':unittest.main()
