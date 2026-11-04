"""Local-only checks for the safe quick-v2 adapter; no GPU or SSH calls."""
import unittest
from unittest.mock import Mock

from scripts.quick_v2_remote import (SCHEMAS, WORKER_WRAPPER, audit, config,
                                     timing_samples)


class QuickV2Tests(unittest.TestCase):
    def test_config_is_single_cycle_b_to_c_p1f_with_fraction_wrapper(self):
        cfg = config(9, 5)
        args = cfg['train_args']
        self.assertEqual(cfg['entry_module'], 'quick_v2_worker')
        self.assertIn('--planned-variant', args)
        self.assertEqual(args[args.index('--planned-variant') + 1], 'p1f')
        self.assertEqual(args[args.index('--planned-source') + 1], 'B')
        self.assertEqual(args[args.index('--planned-cycles') + 1], '1')
        self.assertIn('--data', args)
        self.assertEqual(args[args.index('--data') + 1], 'random')
        self.assertIn('--hash-every', args)
        self.assertEqual(args[args.index('--hash-every') + 1], '1')
        self.assertIn('set_per_process_memory_fraction(0.12, rank)', WORKER_WRAPPER)
        self.assertIn('run_module("shift_full.planned_train"', WORKER_WRAPPER)

    def test_timing_keeps_per_rank_phase_step_and_handoff_rows(self):
        data = {'A': {'events_g0_r2.jsonl': [
            dict(kind='phase', name='model_load', seconds=1.25, monotonic=10),
            dict(kind='begin', step=8, quorum_seconds=.2, monotonic=11),
            dict(kind='commit', step=9, seconds=2.5, schedule_seconds=2.0, monotonic=12),
            dict(kind='planned_final_freeze', step=9, monotonic=13, hashes={'omit': 'x'})]}}
        worker, = timing_samples(data)
        self.assertEqual((worker['role'], worker['rank'], worker['generation']), ('A', 2, 0))
        self.assertEqual(worker['phases'][0]['seconds'], 1.25)
        self.assertEqual(worker['steps'][1]['schedule_seconds'], 2.0)
        self.assertNotIn('hashes', worker['handoff'][0])

    @staticmethod
    def sample(cut=9, goal=14):
        data = {role: {} for role in 'ABC'}
        def hashes(step, rank, role):
            return {key: f'{key}:{step}:{rank}'+('' if key in ('model','optimizer','scheduler') else ':'+role)
                    for key in SCHEMAS}
        def sample_ids(step, role, rank):
            replica = 0 if role == 'A' else 1
            data_rank = replica*2 + rank % 2
            return [(step-1)*8 + micro*4 + data_rank for micro in range(2)]
        for role in 'AB':
            rows = []
            for step in range(1, cut + 1):
                for rank in range(8):
                    rows.append(dict(kind='commit', step=step, rank=rank, hashes=hashes(step,rank,role),
                                     sample_ids=sample_ids(step,role,rank), fixed_batch=True, global_batch=8,
                                     participants=2, contributed=True, finite_gradients=True))
            if role == 'B':
                rows.extend(dict(kind='planned_state_sent', step=cut, rank=rank) for rank in range(8))
                rows.extend(dict(kind='planned_standby_verified', step=cut, rank=rank) for rank in range(8))
            data[role]['events_g0_r0.jsonl'] = rows
        c_rows = []
        for rank in range(8):
            restored_hashes = hashes(cut,rank,'B')
            c_rows.append(dict(kind='restored', step=cut, rank=rank, hashes_match=True,
                               optimizer_tensors=1, expected_hashes=restored_hashes,
                               actual_hashes=restored_hashes))
            c_rows.append(dict(kind='planned_state_ready', step=cut, rank=rank))
        for step in range(cut + 1, goal + 1):
            for rank in range(8):
                for role in 'AC':
                    row = dict(kind='commit', step=step, rank=rank, hashes=hashes(step,rank,role),
                               sample_ids=sample_ids(step,role,rank), fixed_batch=True,
                               epoch=1, physical_members=['A', 'C'], participants=2,
                               contributed=True, finite_gradients=True, global_batch=8)
                    if role == 'C':
                        c_rows.append(row)
                    else:
                        data[role].setdefault('events_g0_r0.jsonl', []).append(row)
        data['C']['events_g1_r0.jsonl'] = c_rows
        return data

    @staticmethod
    def fake_session():
        class S:
            @staticmethod
            def rows_role(data, role, kind, round_index=None, step=None):
                rows = [e for name, events in data.get(role, {}).items() if name.startswith('events_')
                        for e in events if e.get('kind') == kind]
                return [e for e in rows if step is None or e.get('step') == step]
        return S()

    def test_audit_accepts_complete_b_to_c_and_common_post_steps(self):
        result = audit(self.fake_session(), self.sample(), 9, 14)
        self.assertTrue(result['functional_pass'])
        self.assertEqual(result['post_steps'], 5)

    def test_audit_rejects_mismatched_joint_state(self):
        data = self.sample()
        next(row for row in data['C']['events_g1_r0.jsonl']
             if row.get('kind') == 'commit')['hashes']['model'] = 'bad'
        with self.assertRaisesRegex(RuntimeError, 'shared state hash mismatch'):
            audit(self.fake_session(), data, 9, 14)


if __name__ == '__main__':
    unittest.main()
