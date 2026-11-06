"""Logical profile invariants independent of a physical host's eight GPU slots."""
import unittest
from types import SimpleNamespace

from shift_full.session_layout import SessionLayout
from shift_full.session_engine import SessionRuntime
from shift_full.training_core import sample_plan
from shift_full.v4_transaction import Transaction
from shift_full.v4_groups import LocalGroupCache


class LayoutTests(unittest.TestCase):
    def test_four_and_eight_rank_profiles_cover_identical_global_sample_ids(self):
        for world in (4,8):
            layout=SessionLayout(world)
            self.assertEqual(layout.width,2*layout.tp)
            self.assertEqual(layout.affected,tuple(range(world//2,world)))
            for replica in (0,1):
                for stage in (0,1):
                    role=('A','B')[stage] if replica==0 else ('D','E')[stage]
                    ranks=[layout.rank(role,d) for d in range(layout.width)]
                    self.assertEqual(ranks,list(range(stage*layout.width,(stage+1)*layout.width)))
            covered={sample for replica in (0,1) for rank in range(layout.width)
                for sample in sample_plan(7,replica=replica,local_data_rank=layout.local_data_rank(rank))}
            self.assertEqual(covered,set(range(56,64)))
            self.assertEqual([layout.rank('D',d,True) for d in range(world)],list(range(world)))

    def test_small_profile_requires_all_two_real_candidate_restores(self):
        t=Transaction(1,'B','C',10,(2,3),validation='runtime',world_size=4)
        t.ready(1,2,dict(schema=1,epoch=1,rank=2,cut=10,target='C',restored=True))
        with self.assertRaises(ValueError):t.decide()
        t.ready(1,3,dict(schema=1,epoch=1,rank=3,cut=10,target='C',restored=True))
        self.assertEqual(t.decide(),'replace')
        with self.assertRaises(ValueError):Transaction(1,'B','C',10,(2,),world_size=4)
        with self.assertRaises(ValueError):Transaction(1,'B','C',10,(2,4),world_size=4)

    def test_partial_scope_depends_on_profile(self):
        r=SessionRuntime.__new__(SessionRuntime);r.layout=SessionLayout(4)
        r.c={'node_hosts':{'node1':'unused-private-host'}}
        c=SimpleNamespace(policy='v4',source_replica=0,ranks=(2,3),target='node1')
        r.validate_placement(c)
        c.ranks=(4,5,6,7)
        with self.assertRaises(ValueError):r.validate_placement(c)

    def test_command_declares_small_world_and_stage_scope(self):
        from shift_full.session_control import Command
        payload=dict(schema=1,session='small',id='c1',expected_epoch=0,action='migrate',
            policy='v2',source_replica=1,target='node0',replica_world_size=4)
        self.assertEqual(Command.from_dict(payload).ranks,(0,1,2,3))
        payload.update(policy='v4',source_replica=0,ranks=[2,3])
        self.assertEqual(Command.from_dict(payload).ranks,(2,3))
        payload['replica_world_size']=8
        with self.assertRaises(ValueError):Command.from_dict(payload)

    def test_two_host_roles_require_explicit_opt_in_and_distinct_source_target(self):
        from shift_full.planned_state import role_hosts
        roles=dict(A='host-a',B='host-b',C='host-a')
        with self.assertRaises(ValueError):role_hosts(roles,0)
        self.assertEqual(role_hosts(roles,0,allow_colocated=True),roles)
        self.assertEqual(role_hosts(roles,1,allow_colocated=True),dict(A='host-a',B='host-a',C='host-b'))
        with self.assertRaises(ValueError):role_hosts(dict(A='one',B='one',C='one'),0,allow_colocated=True)

    def test_group_cache_never_reuses_across_a_pipeline_boundary(self):
        self.assertEqual(LocalGroupCache.local_members((2,3),2),(0,1))
        self.assertIsNone(LocalGroupCache.local_members((1,2),2))
        self.assertEqual(LocalGroupCache.local_members((4,6),4),(0,2))

    def test_unsupported_world_and_out_of_scope_devices_rejected(self):
        for n in (True,2,6,16):
            with self.assertRaises(ValueError):SessionLayout(n)
        with self.assertRaises(ValueError):SessionLayout(4).rank('B',2)


if __name__=='__main__':unittest.main()
