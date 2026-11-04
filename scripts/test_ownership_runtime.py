"""Runtime callbacks with concrete Futures and intercepted Manager methods.

No CUDA/PG claim; actual trainer tensor closures run in test_planned_payload.
"""
from concurrent.futures import Future
from types import SimpleNamespace
import threading
import unittest
from shift_full.ownership_protocol import Cut
from shift_full.ownership_runtime import ReservedB, completed_quorum, retain_contribution, forbid_quorum_recovery


class OwnershipRuntimeTest(unittest.TestCase):
    def manager(self):
        self.calls=[]
        future=Future();future.set_result(None)
        return SimpleNamespace(_quorum_future=future,_errored=None,_healing=False,
            _pending_state_dict=None,_recovery_event=None,_participating_replica_world_size=2,
            _participating_replica_rank=1,_pg=object(),current_step=lambda:6,
            shutdown=lambda:self.calls.append('shutdown'),
            allreduce=lambda *_:self.calls.append('illegal'),
            start_quorum=lambda:self.calls.append('illegal'),should_commit=lambda:self.calls.append('illegal'))

    def test_real_future_and_recovery_event_drain_before_reserve(self):
        m=self.manager();m._recovery_event=SimpleNamespace(synchronize=lambda:self.calls.append('recovery'))
        b=ReservedB(m,Cut(1,6,'run'),local_pgs=[object()])
        self.assertEqual(self.calls,['recovery']);self.assertIsNone(m._recovery_event)
        b.descriptor(lambda:self.calls.append('published'))
        for name in ('start_quorum','allreduce','should_commit','shutdown'):
            with self.assertRaises(RuntimeError):getattr(m,name)()
        t=threading.Thread(target=lambda:b.control(b.fence.cut,lambda _:self.calls.append('ack')))
        t.start();t.join()
        self.assertNotIn('shutdown',self.calls)
        b.drain_and_close(drain_local=lambda:self.calls.append('local'),finish_presend=lambda:self.calls.append('presend'))
        self.assertEqual(self.calls,['recovery','published','ack','local','presend','shutdown'])
        self.assertTrue(b.fence.closed)

    def test_nonquiescent_state_or_local_alias_rejected(self):
        for field,value in [('_quorum_future',Future()),('_errored',ValueError('x')),('_healing',True),
                            ('_pending_state_dict',{}),('_participating_replica_world_size',1),
                            ('_participating_replica_rank',None)]:
            m=self.manager();setattr(m,field,value)
            with self.subTest(field=field),self.assertRaises(RuntimeError):ReservedB(m,Cut(1,6,'r'),local_pgs=[])
        m=self.manager()
        with self.assertRaises(RuntimeError):ReservedB(m,Cut(1,6,'r'),local_pgs=[m._pg])
        with self.assertRaises(RuntimeError):completed_quorum(m,step=6,replica_index=0)
        with self.assertRaises(RuntimeError):completed_quorum(m,step=5)

    def test_future_failure_is_not_swallowed(self):
        m=self.manager();f=Future();f.set_exception(ValueError('quorum failed'));m._quorum_future=f
        with self.assertRaisesRegex(ValueError,'quorum failed'):ReservedB(m,Cut(1,6,'r'),local_pgs=[])

    def test_native_response_rejected_before_configuration_or_healing(self):
        good=dict(heal=False,recover_dst_replica_ranks=[],max_step=6,replica_world_size=2,
                  max_world_size=2,replica_rank=1,max_replica_rank=1)
        for updates in ({},{'heal':True},{'recover_dst_replica_ranks':[0]}, {'max_step':7},
                        {'replica_world_size':3},{'max_world_size':1},{'replica_rank':0},{'max_replica_rank':None}):
            m=self.manager();response=SimpleNamespace(**dict(good,**updates))
            m._client=SimpleNamespace(_quorum=lambda **kw:response,other=lambda:'delegate')
            forbid_quorum_recovery(m,1)
            if updates:
                with self.subTest(updates=updates),self.assertRaises(RuntimeError):m._client._quorum(step=6)
            else:self.assertIs(m._client._quorum(step=6),response)
            self.assertEqual(m._client.other(),'delegate')

    def test_missing_ack_presend_failure_and_publication_error_keep_revocation(self):
        m=self.manager();b=ReservedB(m,Cut(1,6,'r'),local_pgs=[])
        with self.assertRaises(ValueError):b.descriptor(lambda:(_ for _ in ()).throw(ValueError('publish')))
        with self.assertRaises(RuntimeError):b.drain_and_close(drain_local=lambda:None,finish_presend=lambda:None)
        b.control(b.fence.cut,lambda _:None)
        with self.assertRaises(TimeoutError):b.drain_and_close(drain_local=lambda:None,
            finish_presend=lambda:(_ for _ in ()).throw(TimeoutError('presend')))
        self.assertNotIn('shutdown',self.calls)
        with self.assertRaises(RuntimeError):m.allreduce(None)

    def contribution(self, failure=None):
        self.calls=[];self.config=SimpleNamespace(finalize_model_grads_func=object())
        self.ddp=SimpleNamespace(finish_grad_sync=lambda:self.calls.append('local'))
        def run(name):
            self.calls.append(name)
            if name==failure:raise ValueError(name)
        def execute():
            self.config.finalize_model_grads_func([self.ddp]);run('execute')
        return retain_contribution(config=self.config,ddp=self.ddp,audit_initial=lambda:run('initial'),
            clear=lambda:run('clear'),execute=execute,capture=lambda:run('capture'),unchanged=lambda:True,
            restore=lambda:run('restore'),verify=lambda:run('verify'))

    def test_success_retains_then_one_outer_and_native_tail(self):
        c=self.contribution();self.assertEqual(self.calls,['initial','clear','local','execute','capture'])
        previous=self.ddp.finish_grad_sync
        def native(models):models[0].finish_grad_sync();self.calls.append('tail')
        c.finish(native_finalizer=native,outer=lambda:self.calls.append('outer'))
        self.assertEqual(self.calls[-2:],['outer','tail']);self.assertEqual(self.calls.count('local'),1)
        self.assertIs(self.ddp.finish_grad_sync,previous)
        with self.assertRaises(RuntimeError):c.cancel()
        with self.assertRaises(RuntimeError):c.finish(native_finalizer=native,outer=lambda:None)

    def test_each_pre_activation_failure_restores_clears_and_audits(self):
        for failure in ('initial','execute','capture'):
            with self.subTest(failure=failure),self.assertRaisesRegex(ValueError,failure):self.contribution(failure)
            self.assertEqual(self.calls[-3:],['restore','clear','verify'])
        c=self.contribution();c.cancel();self.assertEqual(self.calls[-3:],['restore','clear','verify'])
        with self.assertRaises(RuntimeError):c.validate()

    def test_post_activation_outer_or_tail_error_never_rolls_back(self):
        for failure in ('outer','tail'):
            c=self.contribution();previous=self.ddp.finish_grad_sync
            def outer():
                self.calls.append('outer')
                if failure=='outer':raise ValueError('outer')
            def native(models):
                models[0].finish_grad_sync()
                if failure=='tail':raise ValueError('tail')
            with self.assertRaises(ValueError):c.finish(native_finalizer=native,outer=outer)
            self.assertEqual(c.state,'activated');self.assertIs(self.ddp.finish_grad_sync,previous)
            with self.assertRaises(RuntimeError):c.cancel()
            self.assertNotIn('restore',self.calls)

if __name__=='__main__':unittest.main()
