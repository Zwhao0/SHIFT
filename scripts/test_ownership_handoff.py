"""Actual production coordinator methods, concrete Futures/TCP, CPU only.

Training groups/Managers/CUDA are instrumented substitutes. This is NOT a
native Manager latency measurement or NCCL/GPU correctness evidence.
"""
from concurrent.futures import Future
from dataclasses import asdict
import json
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shift_full.ownership_handoff import OwnershipHandoff
from shift_full.ownership_runtime import retain_contribution
from shift_full.planned_handoff import EpochGate
from shift_full.ownership_protocol import Ownership


class Store:
    def __init__(self):
        self.data = {}
        self.cv = threading.Condition()

    def set(self, key, value):
        with self.cv:
            self.data[key] = value.encode() if isinstance(value, str) else value
            self.cv.notify_all()

    def check(self, keys):
        with self.cv:
            return all(key in self.data for key in keys)

    def get(self, key):
        with self.cv:
            if not self.cv.wait_for(lambda: key in self.data, timeout=3):
                raise TimeoutError(key)
            return self.data[key]

    def compare_set(self, key, expected, value):
        with self.cv:
            if self.data.get(key, b'') == expected.encode():
                self.set(key, value)
            return self.data.get(key, b'')


class CoordinatorTest(unittest.TestCase):
    def test_fixed_anchor_cut_selected_before_arm_and_late_notice_rejected(self):
        from shift_full.overlap_handoff import OverlapHandoff
        h=OverlapHandoff.__new__(OverlapHandoff)
        h.role,h.rank,h.local_world='A',0,8
        h.coord=Store();h.coord.set('fixed_cut','6')
        for r in range(8):h.coord.set(f'prepared/C/{r}','1')
        h.check=lambda:None;h.cancelled=False;h.request_seen=False;h.progress=dict(updates=1)
        events=[];h.emit=lambda kind,**kw:events.append((kind,kw))
        token=object()
        self.assertEqual(h.boundary(token),(token,False))
        self.assertEqual(h.coord.get('cut'),b'6')
        self.assertEqual(events,[('planned_cut_selected',dict(step=6))])
        del h.coord.data['cut'];h.progress['updates']=5
        with self.assertRaisesRegex(RuntimeError,'two-boundary notice'):h.boundary(token)

    def run_cut(self, world=2, failure=None):
        store = Store()
        store.set('cut', '6')
        store.set('ownership_nonce', 'cpu-test')
        trace, outcomes, errors, objects = [], {}, [], []
        trace_lock = threading.Lock()

        def record(kind, role, rank):
            with trace_lock:
                trace.append((kind, role, rank))

        class PG:
            _pg = object()
            def abort(self, **kw):
                # No A/C first commit can bypass this cross-role condition.
                assert store.check([f'ownership/B/{r}/closed' for r in range(world)])
                record('abort', self.role, self.rank)
                self._pg = None
            def shutdown(self):
                assert self._pg is None

        def pg(role, rank):
            value = PG();value.role, value.rank = role, rank
            return value

        def manager(role, rank, new=False, members='AC'):
            future = Future();future.set_result(None)
            m = SimpleNamespace(_quorum_future=future, _errored=None, _healing=False,
                _pending_state_dict=None, _recovery_event=None, _participating_replica_world_size=2,
                _participating_replica_rank=0 if role == 'A' else 1,
                _pg=pg(role, rank), current_step=lambda: 6, _shift_epoch=1 if new else 0)
            def shutdown():
                if not new and role == 'B':
                    assert ('presend', 'B', rank) in trace
                if not new and role == 'A':
                    assert store.check([f'ownership/B/{r}/fenced' for r in range(world)])
                record('close_new' if new else 'close_old', role, rank)
            m.shutdown = shutdown
            m._client=SimpleNamespace(_quorum=lambda **kw:SimpleNamespace(heal=False,
                recover_dst_replica_ranks=[],max_step=6,replica_world_size=2,max_world_size=2,
                replica_rank=0 if role=='A' else 1,max_replica_rank=0 if role=='A' else 1))
            def quorum(**kwargs):
                if new:
                    assert kwargs['allow_heal'] is False
                    m._client._quorum()
                record('new_quorum' if new else 'ILLEGAL_OLD',role,rank)
            m.start_quorum = quorum
            m.wait_quorum = lambda: None
            m.allreduce = lambda *_: record('outer' if new else 'ILLEGAL_OLD', role, rank)
            m.should_commit = lambda: record('commit' if new else 'ILLEGAL_OLD', role, rank)
            if new and role == 'C' and rank == 0 and failure == 'quorum':
                m._healing = True
            return m

        def worker(role, rank):
            h = OwnershipHandoff.__new__(OwnershipHandoff)
            objects.append(h)
            h.role, h.rank, h.local_world = role, rank, world
            h.cut, h.new_epoch, h.base_epoch, h.round, h.offset = 6, 1, 0, 0, 100
            h.coord = store;h.coord_prefix = '';h.control_host = '127.0.0.1'
            h.hosts = dict(A='127.0.0.1', B='127.0.0.1', C='127.0.0.1')
            h.args = SimpleNamespace(host_ip='127.0.0.1', audit_state=True)
            h.progress = dict(updates=6);h.gate = EpochGate();h.protocol = None
            h.errors=[];h.threads=[];h.deferred=True;h.cancelled=False
            h.retired_pgs=[];h.local_pgs=[object()];h.channels={};h.candidate=pg(role,rank)
            h.cleanup_thread=None;h.cleanup_cancel=threading.Event();h.device=0
            h.fallback=False;h.relay_done=False;h.old_epoch_revoked=False
            h.ownership_active=False;h.building_manager=False;h.epoch_store=object()
            h.reservation=h.contribution=None;h.initial_audit=None
            h.control_ready=threading.Event();h.reserved_ready=threading.Event()
            h.control_done=threading.Event();h.control_stop=threading.Event()
            h.control_error=None;h.wire=h.listener=None;h.control_thread=None
            h.presend=None
            h.emit=lambda kind, **kw: record(kind, role, rank)
            h.control_client=lambda seconds: store
            h.manifest=lambda: {}
            h.load=lambda payload: None
            h.capture_owner_guard=lambda: lambda: True
            samples = [50 + rank, 54 + rank]
            result = dict(gradient_hash='full-buckets', parameter_gradient_hash='full-parameters',
                          tail_hash='all-tail', sample_ids=samples,
                          descriptor=dict(replica_index=1))
            h.source_contribution=lambda *a: {k:result[k] for k in
                ('gradient_hash','parameter_gradient_hash','tail_hash','sample_ids')}
            payload=dict(step=6,manifest={},hashes={str(i): 'hash' for i in range(9)})
            h.transport=lambda:SimpleNamespace(recv_checkpoint=lambda *a: payload)
            def recompute(expected):
                if rank == 0 and failure == 'compute':
                    raise ValueError('actual callback rejected')
                config=SimpleNamespace(finalize_model_grads_func=None)
                ddp=SimpleNamespace(finish_grad_sync=lambda:record('local_C',role,rank))
                return retain_contribution(config=config, ddp=ddp,
                    audit_initial=lambda: record('initial_C',role,rank), clear=lambda:None,
                    execute=lambda:config.finalize_model_grads_func([ddp]),
                    capture=lambda:dict(result,initial_audit=dict(hashes=expected,before_forward=True)),
                    unchanged=lambda:not (rank==0 and failure=='guard'),
                    restore=lambda:record('rollback_C',role,rank),verify=lambda:None)
            h.ownership_recompute=recompute
            def make():
                return manager(role,rank,new=True,members='AB' if h.fallback else 'AC')
            old=manager(role,rank)
            try:
                if role in 'AB':
                    h.control_thread=threading.Thread(target=h._prepare_control)
                    h.control_thread.start()
                if role=='B':
                    h.reserve_descriptor(old,dict(replica_index=1),h.local_pgs)
                    # Original slow work stays present. Control ack can precede
                    # local completion; PG destruction is still forbidden.
                    time.sleep(.1)
                    record('presend',role,rank)
                    new, retired=h.at_gradient(old,[],make,samples,None)
                    outcomes[role,rank]='retired' if retired else 'AB'
                    if not retired:
                        h.authorize(new);new.allreduce(None)
                elif role=='A':
                    time.sleep(.01)
                    new, retired=h.at_gradient(old,[],make,samples,None)
                    h.authorize(new);new.allreduce(None)
                    outcomes[role,rank]='AB' if h.fallback else 'AC'
                else:
                    if not h.ready_receiver():
                        outcomes[role,rank]='cancelled'
                        return
                    new=h.finish_receiver(make)
                    if new is None:
                        outcomes[role,rank]='cancelled'
                        return
                    h.contribution.finish(native_finalizer=lambda models:models[0].finish_grad_sync(),
                                          outer=lambda:new.allreduce(None))
                    outcomes[role,rank]='AC'
                h.progress['updates']=7
                if outcomes[role,rank] not in ('retired','cancelled'):
                    h.committed()
                if store.check([f'ownership/{r}/{i}/committed' for r in 'AC' for i in range(world)]):
                    store.set('ac_three','1')
                h.finish_retired_cleanup()
            except BaseException as exc:
                errors.append((role,rank,repr(exc)))
                store.set('ac_three','1')
            finally:
                h.cancel_cleanup_on_exit()

        with patch('shift_full.ownership_handoff.torch.cuda.synchronize'), \
             patch('shift_full.ownership_handoff.torch.cuda.set_device'), \
             patch('shift_full.ownership_handoff.dist.TCPStore', return_value=object()), \
             patch('shift_full.ownership_handoff.validate_state'):
            threads=[threading.Thread(target=worker,args=(r,i),daemon=True)
                     for r in 'BAC' for i in range(world)]
            for t in threads:t.start()
            for t in threads:t.join(timeout=5)
            self.assertFalse(any(t.is_alive() for t in threads), (errors,trace))
        self.assertEqual(errors,[])
        self.assertFalse(any(k=='ILLEGAL_OLD' for k,_,_ in trace))
        self.assertTrue(all(h.control_thread is None or not h.control_thread.is_alive() for h in objects))
        for h in objects:
            if h.reservation:
                with self.assertRaises(RuntimeError):h.reservation.manager.allreduce(None)
        return store,trace,outcomes,objects

    def test_all_eight_roles_real_control_and_cleanup_barrier(self):
        store,trace,outcomes,objects=self.run_cut(world=8)
        self.assertEqual(list(outcomes.values()).count('retired'),8)
        self.assertEqual(list(outcomes.values()).count('AC'),16)
        self.assertEqual(sum(k=='local_C' for k,_,_ in trace),8)
        self.assertEqual(sum(k=='outer' for k,_,_ in trace),16)
        self.assertEqual(store.get('ownership_activation_AC'),b'AC')
        with self.assertRaises(RuntimeError):next(h for h in objects if h.role=='C').contribution.cancel()

    def test_compute_quorum_and_guard_failures_use_new_AB(self):
        for failure in ('compute','quorum','guard'):
            with self.subTest(failure=failure):
                store,trace,outcomes,_=self.run_cut(failure=failure)
                self.assertEqual(list(outcomes.values()).count('cancelled'),2)
                self.assertEqual(list(outcomes.values()).count('AB'),4)
                self.assertEqual(store.get('ownership_activation_AB'),b'AB')
                self.assertFalse(any(k=='outer' and r=='C' for k,r,_ in trace))
                self.assertEqual(sum(k=='outer' for k,_,_ in trace),4)

    def test_conflicting_or_stale_proof_rejected(self):
        h=OwnershipHandoff.__new__(OwnershipHandoff)
        h.coord=Store();h.new_epoch=1;h.cut=6;h.role='A';h.rank=0
        h.coord.set('ownership_nonce','unique')
        h.proof('drained',value=True);h.proof('drained',value=True)
        with self.assertRaises(RuntimeError):h.proof('drained',value=False)
        value=json.loads(h.coord.get('ownership/A/0/drained'))
        value['cut']['nonce']='old';h.coord.set('ownership/A/0/drained',json.dumps(value))
        with self.assertRaises(RuntimeError):h.read_proof('A',0,'drained')

    def test_AC_activation_deadline_publishes_irreversible_AB(self):
        h=OwnershipHandoff.__new__(OwnershipHandoff)
        h.role='A';h.rank=0;h.local_world=1;h.cut=6;h.new_epoch=1
        h.coord=Store();h.coord.set('ownership_nonce','timeout')
        h.emit=lambda *a,**kw:None
        h.protocol=Ownership(h.ticket(),1)
        for role,proofs in [('A',['drained']),('B',['fenced']),
                            ('C',['initial_audit','computed','gradient_audit'])]:
            for proof in proofs:h.protocol.record(h.ticket(),role,0,proof)
        h.protocol.choose('AC')
        with patch('shift_full.ownership_handoff.time.monotonic',side_effect=[0,100,100]):
            self.assertFalse(h.activate_manager(None,'AC'))
        self.assertEqual(h.coord.get('ownership_activation_AC'),b'AB')
        self.assertEqual(h.protocol.decision,'AB');self.assertIsNone(h.protocol.members)
        with self.assertRaises(RuntimeError):h.protocol.choose('AC')

    def test_late_fatal_stops_every_owner_before_more_dispatch(self):
        store=Store();store.set('ownership_fatal','late B mismatch')
        for role in 'ABC':
            h=OwnershipHandoff.__new__(OwnershipHandoff);h.role=role;h.coord=store
            with self.subTest(role=role),self.assertRaisesRegex(RuntimeError,'late B mismatch'):
                h.authorize(object())


if __name__=='__main__':unittest.main()
