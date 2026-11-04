"""No remote or GPU access: three-role I/O and failure cleanup invariants."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from scripts.v3_planned_remote import PlannedRotation,complete,run,audit_ownership,SCHEMAS
from shift_full.control import endpoints_from_env


class ControllerTests(unittest.TestCase):
    def test_production_ownership_audit_uses_real_event_schema(self):
        import copy
        events=[]
        for rank in range(8):
            descriptor=dict(sample_ids=[rank,rank+4])
            common=dict(rank=rank,step=6,planned_round=0)
            exact=dict(gradient_hash='g',parameter_gradient_hash='p',tail_hash='t',sample_ids=[rank,rank+4])
            for role,kind,fields in [
                ('B','planned_ownership_reserved',dict(descriptor=descriptor)),
                ('B','planned_ownership_b_closed',{}),
                ('B','planned_ownership_late_exact',dict(exact,exact=True)),
                ('C','planned_ownership_initial_audit',dict(hashes={k:'h' for k in SCHEMAS},hashes_match=True,before_forward=True,monotonic=1)),
                ('C','planned_ownership_computed',dict(exact,descriptor=descriptor,monotonic=2)),
                ('A','planned_ownership_activated',dict(members='AC')),
                ('C','planned_ownership_activated',dict(members='AC'))]:
                events.append(dict(common,role=role,kind=kind,**fields))
            for role in 'BC':
                for call in range(2):
                    events.append(dict(common,role=role,kind='routing',layer='gate',call=call,
                        topk_hash=str(call),sorted_topk_hash=str(call),min_boundary_margin=.1,
                        zero_margins=0,near_ties_1e6=0,tokens=512))
        s=Mock()
        s.rows_role.side_effect=lambda d,r,k,round,step:[e for e in d if e['role']==r and e['kind']==k]
        self.assertTrue(audit_ownership(s,events,6)['late_full_contribution_exact'])
        for kind,key,value in [('planned_ownership_initial_audit','before_forward',False),
                               ('planned_ownership_late_exact','gradient_hash','bad'),
                               ('routing','shadow',True)]:
            broken=copy.deepcopy(events);next(e for e in broken if e['kind']==kind)[key]=value
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):audit_ownership(s,broken,6)
        with self.assertRaises(RuntimeError):audit_ownership(s,events[1:],6)

    def session(self,root):
        s=PlannedRotation.__new__(PlannedRotation)
        s.root=s.out=Path(root);s.name='test';s.gen=[0,0];s.seconds=10
        eps=endpoints_from_env();s.endpoints=eps
        s.roles=dict(zip('ABC',(eps['node1'],eps['node3'],eps['node0'])))
        s.role_bindings={'A':(0,0),'B':(1,0),'C':(1,1)}
        s.bindings={s.role_bindings[r]:e for r,e in s.roles.items()}
        s.role_started=set('ABC');s.control_started=True
        s.started_endpoints={e.physical_node:e for e in s.roles.values()}
        s.units=[(e,'test-'+r) for r,e in s.roles.items()];s.log=io.StringIO()
        s.event=Mock();return s

    def test_complete_rejects_duplicate_rank(self):
        self.assertTrue(complete([dict(rank=i) for i in range(8)]))
        self.assertFalse(complete([dict(rank=i) for i in range(8)]+[dict(rank=0)]))

    def test_all_three_bindings_read_and_mark(self):
        with tempfile.TemporaryDirectory() as root:
            s=self.session(root);s.read_binding=Mock(side_effect=lambda rid,g:(rid,g));s.write_marker=Mock()
            self.assertEqual(s.data(),s.role_bindings)
            s.mark_role('C','START')
            s.write_marker.assert_called_once_with(s.roles['C'],'/workspace/SHIFT/results/test/rid1_g1','START','')
            self.assertEqual(s.endpoint_for(0).physical_node,'node1')

    def test_full_audit_reads_every_original_event(self):
        with tempfile.TemporaryDirectory() as root:
            s=self.session(root)
            original=[dict(kind='commit',rank=0,step=i) for i in range(30)]
            s.execute_endpoint=Mock(return_value=json.dumps(dict(rows={'events_g0_r0.jsonl':original})))
            s.read_binding=Mock(side_effect=AssertionError('bounded cache used'))
            d=s.data(full=True)
            self.assertEqual(len(s.rows_role(d,'A','commit')),30)
            self.assertEqual(s.execute_endpoint.call_count,3)

    def test_slow_steady_arms_injection_without_target_or_cut(self):
        s=Mock();s.rows_role.return_value=[dict(hashes={k:'x' for k in
            ('model','buffers','optimizer','scheduler','progress','cpu_rng','cuda_rng','python_rng','tracker_rng')},
            finite_gradients=True,participants=2)]
        s.wait.return_value={r:{'events':[]} for r in 'AB'}
        state={};run(s,'steady',1,'slow-compute',10,False,state,Mock(),20)
        self.assertEqual([c.args[0] for c in s.launch_role.call_args_list],['A','B'])
        s.set_key.assert_called_once_with('request','1')
        self.assertTrue(state['slow_control_no_migration'])
        self.assertEqual([c.args[0] for c in s.wait.call_args_list],['ab_ready','steady_two_steps','steady_complete'])

    def test_cleanup_continues_after_marker_and_collection_failure(self):
        with tempfile.TemporaryDirectory() as root:
            s=self.session(root);s.mark_role=Mock(side_effect=OSError('lost transport'))
            s.data=Mock(return_value={r:{'supervisor.jsonl':[dict(kind='cleanup')]} for r in 'ABC'})
            s.write_marker=Mock();s.ssh=Mock(return_value='inactive')
            s.execute_endpoint=Mock(side_effect=OSError('collection unavailable'));s.resource_check=Mock()
            with patch('scripts.v3_planned_remote.sp.run'):
                with self.assertRaisesRegex(RuntimeError,'cleanup/collection failed'):s.finish()
            stops=[call for call in s.ssh.call_args_list if '"systemctl","stop"' in call.args[1][-1]]
            self.assertEqual(len(stops),3)
            self.assertEqual(s.mark_role.call_count,3)
            report=json.loads((Path(root)/'unit_cleanup.json').read_text())
            self.assertEqual(len(report['units']),3)
            self.assertEqual(len(report['errors']),6)
            s.resource_check.assert_called_once_with('after')


if __name__=='__main__':unittest.main()
