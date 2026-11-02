"""CPU tests for the all-role joint-commit boundary used by failover."""
import unittest
import json
import subprocess
from unittest.mock import Mock
from .rotation import Rotation


class JointBoundaryTests(unittest.TestCase):
    def test_transient_read_disconnect_reconnects_without_losing_offsets(self):
        s=Rotation.__new__(Rotation);s._read_offsets={(0,1):{'supervisor.jsonl':20}};s._read_cache={}
        endpoint=Mock(physical_node='node1');s.endpoint_for=Mock(return_value=endpoint)
        s.directory=Mock(return_value='/results/session/rid0_g1');s.event=Mock();s.ensure_connection=Mock()
        packet=json.dumps(dict(rows={'supervisor.jsonl':[dict(kind='cleanup')]},offsets={'supervisor.jsonl':40}))
        s.execute_endpoint=Mock(side_effect=[subprocess.CalledProcessError(255,['ssh']),packet])
        result=s.read_binding(0,1)
        self.assertEqual(result['supervisor.jsonl'][0]['kind'],'cleanup')
        self.assertEqual(s._read_offsets[0,1]['supervisor.jsonl'],40)
        s.ensure_connection.assert_called_once_with(endpoint,force=True)

    def setUp(self):
        self.session=Rotation.__new__(Rotation)
        self.session.gen=[0,0]
        self.data=[{},{}]
        for rid in range(2):
            for rank in range(8):
                self.data[rid][f'events_g0_r{rank}.jsonl']=[
                    dict(kind='commit',step=step,participants=2,contributed=True,time=float(step))
                    for step in (1,2,3)]

    def test_fault_boundary_requires_every_rank_before_kill(self):
        self.data[1]['events_g0_r7.jsonl'][-2]['time']=2.5
        steps,_=self.session.common_commits(self.data,before_time=2.2)
        self.assertEqual(steps,[1])

    def test_recovery_metadata_commit_is_not_a_joint_update(self):
        self.data[1]['events_g0_r0.jsonl'][-1]['contributed']=False
        steps,_=self.session.common_commits(self.data,after=1)
        self.assertEqual(steps,[2])


if __name__=='__main__':
    unittest.main()
