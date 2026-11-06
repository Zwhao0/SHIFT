"""Regression guards at actual live-policy boundaries; no CUDA initialization."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shift_full.session_engine import SessionRuntime,SessionPolicy
from shift_full.v4_transaction import OwnerGate


class NativeRecoveryTest(unittest.TestCase):
    def test_switch_first_gradient_collective_waits_for_both_replicas(self):
        from shift_full.session_layout import SessionLayout
        for replica in (0,1):
            calls=[]
            r=SessionRuntime.__new__(SessionRuntime)
            r.handoff=None;r.in_protocol=True;r.replica=replica;r.rank=0;r.phase='after'
            r.protocol_plan=SimpleNamespace(cut=10);r.progress={'updates':10};r.layout=SessionLayout(4)
            r.event=lambda *args,**kw:None
            r.protocol_store=SimpleNamespace(set=lambda key,value:calls.append(('set',key)),
                wait=lambda keys:calls.append(('wait',keys)))
            r.ddp=SimpleNamespace(buffers=[],expert_parallel_buffers=[])
            r.outer=SimpleNamespace(reduce=lambda *args:calls.append(('reduce',None)))
            r.gradient_finalize(lambda:calls.append(('native',None)))
            self.assertEqual(calls[0][0],'native')
            self.assertEqual(calls[-1][0],'reduce')
            self.assertEqual(calls[-2],('wait',[f'gradients_ready/10/{a}/{b}' for a in range(2) for b in range(4)]))
            calls.clear();r.progress['updates']=11
            r.gradient_finalize(lambda:None)
            self.assertEqual(calls,[('reduce',None)])

    def test_v4_peer_defers_quorum_until_replacement_ready_but_not_compute(self):
        from shift_full.session_outer import SessionOuter
        calls=[]
        r=SimpleNamespace(in_protocol=True,replica=1,phase='after',protocol_plan=SimpleNamespace(cut=10),
            progress={'updates':10},healing=False,protocol_store=SimpleNamespace(wait=lambda keys:calls.append(('ready',keys))))
        o=SessionOuter.__new__(SessionOuter);o.runtime=r;o.replica=1;o.rank=0;o.refresh_boundary=lambda:None
        o.manager=SimpleNamespace(start_quorum=lambda **kw:calls.append(('start',kw)),wait_quorum=lambda:calls.append(('wait',None)),
            num_participants=lambda:2,participating_rank=lambda:1)
        o.begin(10);self.assertFalse(calls)
        calls.append(('compute',None));o.wait_quorum()
        self.assertEqual([x[0] for x in calls],['compute','ready','start','wait'])
        self.assertEqual(calls[1][1],['activated'])
        calls.clear();r.progress['updates']=11
        o.begin(11);self.assertEqual(calls[0][0],'start')

    def runtime(self):
        r=SessionRuntime.__new__(SessionRuntime)
        from shift_full.session_layout import SessionLayout
        r.layout=SessionLayout()
        r.progress={'updates':0};r.replica=1;r.rank=0;r.role='D';r.epoch=0;r.sequence=1;r.handoff=None
        r.gate=OwnerGate(0,'B');r.outer=SimpleNamespace(manager=SimpleNamespace(_use_async_quorum=True))
        r.optimizer=SimpleNamespace(zero_grad=lambda **kw:None);r.ddp=SimpleNamespace(zero_grad_buffer=lambda:None)
        r._set_command=lambda c:setattr(r,'command',c)
        r.base_store=SimpleNamespace(wait=lambda keys:self.assertFalse(r.outer.manager._use_async_quorum))
        return r

    def test_native_state_must_load_before_forward_and_context_step(self):
        r=self.runtime();r.start_native(SimpleNamespace(id='recover'))
        source={'updates':17,'weight':42}
        def begin(step):
            if not r.outer.manager._use_async_quorum:r.progress['updates']=source['updates'];r.weight=source['weight']
        r.outer.begin=begin
        context=SessionPolicy().begin(r)
        self.assertEqual(context['step'],17)
        self.assertEqual(r.weight,42)
        self.assertFalse(r.at_committed_boundary)

    def test_fenced_owner_cannot_start_an_attempt(self):
        r=self.runtime();r.replica=0;r.role='B';r.gate.fence()
        with self.assertRaises(ValueError):SessionPolicy().begin(r)

    def test_planned_epoch_is_checked_before_optimizer_vote(self):
        r=self.runtime();r.handoff=SimpleNamespace(authorize=lambda manager:(_ for _ in ()).throw(ValueError('old epoch')))
        r.outer.vote=lambda finite:self.fail('fenced policy reached the commit vote')
        with self.assertRaises(ValueError):SessionPolicy().vote(r,{},SimpleNamespace(finite=True))

    def test_activated_p2_does_not_reenter_precopy_boundary(self):
        r=self.runtime();r.in_protocol=False;r.command=SimpleNamespace(policy='v2')
        r.handoff=SimpleNamespace(activated=True,boundary=lambda manager:self.fail('reentered completed precopy'))
        completed=[];r._complete_command=lambda:completed.append(True)
        r.boundary();self.assertEqual(completed,[True])

    def test_cross_node_native_candidate_publishes_target_address(self):
        from shift_full.session_engine import run_session
        command=dict(schema=1,session='s',id='recover',expected_epoch=5,action='recover',policy='v1',source_replica=1,target='node3')
        r=SimpleNamespace(c=dict(session_candidate=True,session_command={'command':command,'sequence':6},node_hosts={'node3':'target-host'}),replica_hosts={'0':'healthy-host','1':'old-source-host'},retired=False,stopping=False,policy=object())
        r._set_command=lambda c:setattr(r,'command',c)
        r.bind_outer=lambda outer:setattr(r,'outer',outer)
        observed=[]
        def outer(runtime):
            observed.append((runtime.replica_hosts['1'],runtime.healing));return object()
        with patch('shift_full.session_engine.SessionRuntime',return_value=r),patch('shift_full.session_engine.SessionOuter',side_effect=outer),patch('shift_full.session_engine.run_loop'):
            run_session({})
        self.assertEqual(observed,[('target-host',True)])

    def test_replica_zero_native_candidate_keeps_replica_one_address(self):
        from shift_full.session_engine import run_session
        command=dict(schema=1,session='s',id='recover-zero',expected_epoch=4,
            action='recover',policy='v1',source_replica=0,target='node3')
        r=SimpleNamespace(c=dict(session_candidate=True,session_command={'command':command,'sequence':5},
            node_hosts={'node3':'replacement'}),replica_hosts={'0':'old-zero','1':'healthy-one'},
            retired=False,stopping=False,policy=object())
        r._set_command=lambda c:setattr(r,'command',c)
        observed=[];r.bind_outer=lambda outer:None
        def outer(runtime):observed.append(dict(runtime.replica_hosts));return object()
        with patch('shift_full.session_engine.SessionRuntime',return_value=r),patch('shift_full.session_engine.SessionOuter',side_effect=outer),patch('shift_full.session_engine.run_loop'):
            run_session({})
        self.assertEqual(observed,[{'0':'replacement','1':'healthy-one'}])

    def test_native_recovery_accepts_either_replica_but_planned_scope_stays_explicit(self):
        r=self.runtime();r.c={'node_hosts':{'node3':'target'}}
        for replica in (0,1):
            r.validate_placement(SimpleNamespace(policy='v1',source_replica=replica,target='node3'))
        with self.assertRaisesRegex(ValueError,'replica 1'):
            r.validate_placement(SimpleNamespace(policy='v2',source_replica=0,target='node3'))
        with self.assertRaises(ValueError):
            r.validate_placement(SimpleNamespace(policy='v1',source_replica=2,target='node3'))

    def test_recovered_replica_zero_updates_only_its_host_and_resets_planned_store(self):
        r=self.runtime();r.replica=0;r.progress={'updates':25};r.healing=True;r.planned_coord=object()
        r.outer.manager.num_participants=lambda:2
        r.epoch=4;r.gate=OwnerGate(4,'B')
        r.c={'node_hosts':{'node3':'new-zero'}};r.replica_hosts={'0':'old-zero','1':'one'}
        r.partial_owner='B';r.command=SimpleNamespace(policy='v1',expected_epoch=4,source_replica=0,target='node3')
        r.base_store=SimpleNamespace(check=lambda keys:True,get=lambda key:b'25',set=lambda *args:None)
        r._barrier=lambda key:None;r.event=lambda *args,**kwargs:None
        r.s=SimpleNamespace(training_identity=lambda:{})
        r._complete_command()
        self.assertEqual(r.replica_hosts,{'0':'new-zero','1':'one'})
        self.assertIsNone(r.planned_coord)
        self.assertEqual(r.epoch,5)

    def test_p2_coordinator_is_kept_alive_and_reused(self):
        r=self.runtime();r.planned_coord=None;r.event=lambda *a,**k:None
        r.training_state=SimpleNamespace(rng=lambda:{})
        r._planned_args=lambda *a,**kw:SimpleNamespace()
        coordinator=object();calls=[]
        class Handoff:
            def __init__(self,args,*other):
                calls.append(args)
                self.coord_raw=getattr(args,'_coord_raw',coordinator)
                self.coord=SimpleNamespace(set=lambda *a:None)
        with patch.dict('sys.modules',{'shift_full.overlap_handoff':SimpleNamespace(OverlapHandoff=Handoff)}):
            r.start_precopy(SimpleNamespace(id='first'))
            self.assertIs(r.planned_coord,coordinator)
            r.handoff=None
            r.start_precopy(SimpleNamespace(id='second'))
        self.assertIs(calls[1]._coord_raw,coordinator)

    def test_native_restore_refreshes_capture_boundary(self):
        from shift_full.session_outer import SessionOuter

        class FakeTensor:
            def __init__(self,value):self.value=value
            def detach(self):return self
            def clone(self):return FakeTensor(self.value)

        class FakeEvent:
            def __init__(self):self.recorded=False;self.synchronized=False
            def record(self):self.recorded=True
            def synchronize(self):
                if not self.recorded:raise RuntimeError('boundary event was not recorded')
                self.synchronized=True

        events=[]
        class FakeTrainingState:
            def __init__(self):self.rng_state={'counter':'initial'}
            def rng(self):return self.rng_state
            def restore(self,packet,*,clone_optimizer):
                self.rng_state=packet['rng']
                runtime.buffers['buf'].value=packet['buffer']
            def capture(self,*,rng,all_buffers,source_replica,source_rank):
                return dict(rng=dict(rng),buffers={k:v.value for k,v in all_buffers.items()},
                            source_replica=source_replica,source_rank=source_rank)
            def hashes(self,*args,**kwargs):return {}

        class FakeCUDA:
            @staticmethod
            def synchronize():pass
            @staticmethod
            def Event():
                event=FakeEvent();events.append(event);return event

        runtime=SimpleNamespace(replica=1,rank=0,audit=False,sequence=3,progress={'updates':12},
            buffers={'buf':FakeTensor('initial-buffer')},training_state=None,
            optimizer=SimpleNamespace(zero_grad=lambda **kwargs:None),
            ddp=SimpleNamespace(zero_grad_buffer=lambda:None),
            base_store=SimpleNamespace(set=lambda *args:None),event=lambda *args,**kwargs:None)
        runtime.stage=SimpleNamespace(named_buffers=lambda:list(runtime.buffers.items()))
        runtime.training_state=FakeTrainingState()
        fake_torch=SimpleNamespace(cuda=FakeCUDA)
        fake_state=SimpleNamespace(validate_state=lambda *args,**kwargs:None)
        outer=SessionOuter.__new__(SessionOuter);outer.runtime=runtime
        outer.replica=runtime.replica;outer.rank=runtime.rank;outer._boundary=None

        with patch.dict('sys.modules',{'torch':fake_torch,'shift_full.planned_state':fake_state}):
            outer.refresh_boundary()
            old_boundary=outer._boundary
            outer.restore(dict(rng={'counter':'restored'},buffer='restored-buffer',source_replica=0))
            self.assertIsNot(outer._boundary,old_boundary)
            packet=outer.capture()

        self.assertEqual(packet['rng'],{'counter':'restored'})
        self.assertEqual(packet['buffers'],{'buf':'restored-buffer'})
        self.assertTrue(events[-1].synchronized)


if __name__=='__main__':unittest.main()
