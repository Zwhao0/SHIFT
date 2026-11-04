"""Registry lifecycle tests; real CUDA correctness is checked by V4 oracle runs."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from shift_full.v4_groups import LocalGroupCache, validate_runtime


class Group:
    def __init__(self):
        self.closed = False
    def _set_group_name(self, name):
        self.name = name
    def shutdown(self):
        self.closed = True


class ReuseTest(unittest.TestCase):
    def test_runtime_source_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'distributed_c10d.py';path.write_text('different runtime')
            module=SimpleNamespace(__file__=str(path))
            with self.assertRaisesRegex(RuntimeError,'pinned source'):
                validate_runtime(module,module,module)

    def setUp(self):
        self.rank = 0
        self.counter = 0
        self.world = SimpleNamespace(pg_names={},pg_map={},pg_backend_config={},
                                     pg_group_ranks={},pg_coalesce_state={},comms=[])
        def register(group, backend, store, name, config, rank_mapping=None, **_):
            self.world.pg_names[group] = name
            self.world.pg_map[group] = (backend, store)
            self.world.pg_backend_config[group] = config
            self.world.pg_group_ranks[group] = rank_mapping
        def name(*_, **__):
            self.counter += 1
            return str(self.counter)
        def create(ranks, **kwargs):
            n=name()
            if self.rank not in ranks:return None
            group=Group();register(group,kwargs.get('backend','nccl'),'original-store',n,'config')
            return group
        self.dist=SimpleNamespace(new_group=create,get_rank=lambda:self.rank,get_backend=lambda:'nccl')
        self.c10d=SimpleNamespace(_world=self.world,_process_group_name=name,_register_pg_in_world=register)
        self.cache=LocalGroupCache(self.dist,self.c10d,[__file__])

    def build(self, offset):
        return self.dist.new_group([offset,offset+2])

    def test_survivor_keeps_backend_and_updates_global_rank_mapping(self):
        with self.cache.creating():old=self.build(0)
        self.assertEqual(self.cache.detach(),1)
        # Simulate the pinned WORLD shutdown: only named groups are closed.
        for group in self.world.pg_names:group.shutdown()
        self.world.pg_map.clear();self.world.pg_group_ranks.clear()
        self.assertFalse(old.closed)
        self.rank=4
        with self.cache.creating(reuse=True):new=self.build(4)
        self.assertIs(new,old)
        self.assertEqual(self.world.pg_group_ranks[new],{4:0,6:1})
        self.assertEqual(self.world.pg_map[new][1],'original-store')
        self.assertEqual(self.cache.finish(),dict(reused=1,discarded=0))

    def test_cross_host_groups_are_never_preserved(self):
        with self.cache.creating():group=self.dist.new_group([0,4])
        self.assertFalse(self.cache.saved)
        self.assertIn(group,self.world.pg_names)

    def test_unused_standby_groups_are_closed(self):
        with self.cache.creating():group=self.build(0)
        self.cache.detach()
        self.assertEqual(self.cache.finish(),dict(reused=0,discarded=1))
        self.assertTrue(group.closed)

    def test_catalog_survives_multiple_replacements(self):
        with self.cache.creating():original=self.build(0)
        for offset in (4,0,4):
            self.cache.detach()
            for registry in (self.world.pg_names,self.world.pg_map,self.world.pg_group_ranks,
                             self.world.pg_backend_config):registry.clear()
            self.rank=offset
            with self.cache.creating(reuse=True):current=self.build(offset)
            self.assertIs(current,original)
            self.assertEqual(self.cache.finish(retain_catalog=True),dict(reused=1,discarded=0))
            self.assertFalse(original.closed)
            self.assertEqual(self.world.pg_group_ranks[original],{offset:0,offset+2:1})
        self.cache.detach();self.cache.finish()
        self.assertTrue(original.closed)

    def test_pending_coalescing_and_early_reuse_are_rejected(self):
        with self.assertRaises(RuntimeError):
            with self.cache.creating(reuse=True):pass
        with self.cache.creating():group=self.build(0)
        self.world.pg_coalesce_state[group]=['pending']
        with self.assertRaises(RuntimeError):self.cache.detach()
        self.assertIn(group,self.world.pg_names)

    def test_constructor_patch_is_restored_after_failure(self):
        original=self.dist.new_group
        with self.assertRaisesRegex(ValueError,'injected'):
            with self.cache.creating():raise ValueError('injected')
        self.assertIs(self.dist.new_group,original)


if __name__=='__main__':unittest.main()
