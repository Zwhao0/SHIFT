"""CPU-only checks for metadata groups and no-communication guards."""
from types import SimpleNamespace
import unittest

from shift_full.v4_metadata import MetadataGroup, metadata_groups


COMMUNICATION_METHODS = (
    'all_reduce', 'all_gather', 'all_gather_into_tensor', 'reduce_scatter',
    'reduce_scatter_tensor', '_all_gather_base', '_reduce_scatter_base',
    'all_to_all_single', 'broadcast', 'send', 'recv', 'isend', 'irecv',
    'batch_isend_irecv', 'barrier', 'init_process_group', 'new_group',
)


def sentinel(*args, **kwargs):
    return 'original'


class MetadataGroupsTest(unittest.TestCase):
    def make_dist(self):
        values = dict(get_world_size=sentinel, get_rank=sentinel,
                      get_process_group_ranks=sentinel)
        values.update({name: sentinel for name in COMMUNICATION_METHODS})
        return SimpleNamespace(**values)

    def test_geometry_for_all_eight_ranks(self):
        for rank in range(8):
            with self.subTest(rank=rank):
                with metadata_groups(self.make_dist(), SimpleNamespace(), rank) as groups:
                    stage = rank // 4
                    tp_rank = rank % 2
                    dp_rank = (rank % 4) // 2
                    self.assertEqual(groups['tp'].members, (rank // 2 * 2, rank // 2 * 2 + 1))
                    self.assertEqual(groups['tp'].rank, tp_rank)
                    self.assertEqual(groups['ep'].members, (stage * 4 + tp_rank, stage * 4 + tp_rank + 2))
                    self.assertEqual(groups['ep'].rank, dp_rank)
                    self.assertEqual(groups['dense'].members, groups['ep'].members)
                    self.assertEqual(groups['dense'].rank, dp_rank)
                    self.assertEqual(groups['expert_dp'].members, (rank,))
                    self.assertEqual(groups['expert_dp'].rank, 0)
                    self.assertEqual(groups['pp'].members, (rank % 4, rank % 4 + 4))
                    self.assertEqual(groups['pp'].rank, stage)
                    self.assertTrue(all(isinstance(group, MetadataGroup) for group in groups.values()))
                    self.assertTrue(all(group.size == len(group.members) for group in groups.values()))

    def test_queries_return_metadata_and_same_members_groups_are_distinct(self):
        dist = self.make_dist()
        state = SimpleNamespace()
        with metadata_groups(dist, state, 6) as groups:
            self.assertIsNot(groups['ep'], groups['dense'])
            self.assertEqual(groups['ep'].members, groups['dense'].members)
            self.assertEqual(dist.get_world_size(), 8)
            self.assertEqual(dist.get_rank(), 6)
            self.assertEqual(dist.get_process_group_ranks(), tuple(range(8)))
            self.assertEqual(dist.get_world_size(groups['tp']), 2)
            self.assertEqual(dist.get_rank(group=groups['ep']), 1)
            self.assertEqual(dist.get_process_group_ranks(group=groups['pp']), (2, 6))
            self.assertEqual(state.get_tensor_model_parallel_rank(), 0)
            self.assertEqual(state.get_tensor_model_parallel_world_size(extra=True), 2)
            self.assertIs(state.get_tensor_model_parallel_group(), groups['tp'])
            self.assertEqual(state.get_pipeline_model_parallel_rank('ignored'), 1)
            self.assertEqual(state.get_pipeline_model_parallel_world_size(), 2)
            self.assertIs(state.get_pipeline_model_parallel_group(), groups['pp'])
            self.assertEqual(state.get_expert_model_parallel_rank(), 1)
            self.assertEqual(state.get_expert_model_parallel_world_size(), 2)
            self.assertEqual(state.get_data_parallel_rank(), 1)
            self.assertEqual(state.get_data_parallel_world_size(), 2)
            self.assertIs(state.get_data_parallel_group(), groups['dense'])
            self.assertIs(state.get_data_modulo_expert_parallel_group(), groups['expert_dp'])
            self.assertTrue(state.model_parallel_is_initialized())
            self.assertFalse(state.is_pipeline_first_stage(ignore=True))
            self.assertTrue(state.is_pipeline_last_stage())
            for query in (dist.get_world_size, dist.get_rank, dist.get_process_group_ranks):
                with self.assertRaisesRegex(RuntimeError, 'metadata-only candidate cannot communicate'):
                    query(object())

    def test_communication_methods_raise(self):
        dist = self.make_dist()
        with metadata_groups(dist, SimpleNamespace(), 0):
            for name in COMMUNICATION_METHODS:
                with self.subTest(method=name), self.assertRaisesRegex(
                        RuntimeError, 'metadata-only candidate cannot communicate'):
                    getattr(dist, name)()

    def test_exception_restores_originals_and_removes_added_state_attributes(self):
        dist = self.make_dist()
        state = SimpleNamespace(get_tensor_model_parallel_rank=sentinel)
        original_dist = {name: getattr(dist, name) for name in vars(dist)}
        original_state = state.get_tensor_model_parallel_rank
        state_names = (
            'get_tensor_model_parallel_rank', 'get_tensor_model_parallel_world_size',
            'get_tensor_model_parallel_group', 'get_pipeline_model_parallel_rank',
            'get_pipeline_model_parallel_world_size', 'get_pipeline_model_parallel_group',
            'get_expert_model_parallel_rank', 'get_expert_model_parallel_world_size',
            'get_data_parallel_rank', 'get_data_parallel_world_size',
            'get_data_parallel_group', 'get_data_modulo_expert_parallel_group',
            'model_parallel_is_initialized', 'is_pipeline_first_stage',
            'is_pipeline_last_stage',
        )
        with self.assertRaisesRegex(LookupError, 'sentinel exception'):
            with metadata_groups(dist, state, 3):
                self.assertTrue(state.get_tensor_model_parallel_rank is not original_state)
                raise LookupError('sentinel exception')
        for name, original in original_dist.items():
            self.assertIs(getattr(dist, name), original)
        self.assertIs(state.get_tensor_model_parallel_rank, original_state)
        for name in state_names[1:]:
            self.assertFalse(hasattr(state, name), name)

    def test_invalid_logical_rank_is_rejected(self):
        for rank in (-1, 8, 1.0, True, '1', None):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                with metadata_groups(self.make_dist(), SimpleNamespace(), rank):
                    self.fail('invalid rank entered metadata context')


if __name__ == '__main__':
    unittest.main()
