"""Batch8 membership invariants and the common optimizer vote boundary."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from shift_full.training_core import sample_plan, train_step, complete_step


class BatchTest(unittest.TestCase):
    def test_solo_and_joint_cover_same_batch_once(self):
        for step in range(4):
            for participants in (1, 2):
                samples = [sample for replica in range(participants) for data_rank in range(2)
                    for sample in sample_plan(step, participants=participants, replica=replica, local_data_rank=data_rank)]
                self.assertEqual(sorted(samples), list(range(step*8, (step+1)*8)))

    def test_third_contributor_is_rejected(self):
        with self.assertRaises(ValueError): sample_plan(0, participants=3)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'retained server runtime provides torch')
class StepTest(unittest.TestCase):
    def execute(self, commit, finite=True):
        parameter = torch.nn.Parameter(torch.tensor([2.]))
        parameter.main_grad = torch.tensor([1. if finite else float('nan')])
        stage = SimpleNamespace(parameters=lambda:[parameter])
        ddp = SimpleNamespace(buffers=[SimpleNamespace(buckets=[SimpleNamespace(grad_data=parameter.main_grad)])], expert_parallel_buffers=[])
        optimizer = torch.optim.AdamW([parameter], lr=.01, foreach=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=.98)
        progress = {'updates': 0}
        tx = train_step(stage, ddp, optimizer, scheduler, progress, execute=lambda:[], synchronize=False)
        computed = next(tx)
        self.assertEqual(computed.finite, finite)
        if commit and not finite:
            with self.assertRaises(RuntimeError): complete_step(tx, True)
        else:
            complete_step(tx, commit)
        return parameter, optimizer, scheduler, progress

    def test_rejected_vote_does_not_mutate_adam_scheduler_or_cursor(self):
        parameter, optimizer, scheduler, progress = self.execute(False)
        self.assertEqual(parameter.item(), 2.)
        self.assertFalse(optimizer.state)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(progress['updates'], 0)

    def test_committed_vote_updates_exactly_once(self):
        parameter, optimizer, scheduler, progress = self.execute(True)
        self.assertLess(parameter.item(), 2.)
        self.assertEqual(progress['updates'], 1)
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertEqual(optimizer.state[parameter]['step'].item(), 1)

    def test_nonfinite_cannot_commit(self):
        parameter, optimizer, scheduler, progress = self.execute(True, False)
        self.assertEqual(parameter.item(), 2.)
        self.assertFalse(optimizer.state)
        self.assertEqual(progress['updates'], 0)


if __name__ == '__main__': unittest.main()
