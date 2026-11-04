"""CPU-only tests for the global optimizer commit fence in Outer.commit."""
from datetime import timedelta
from threading import Barrier, Lock, Thread
import time
import unittest
from types import SimpleNamespace

from shift_full.v4_outer import Outer


class FakeStore:
    def __init__(self):
        self.values={}
        self.lock=Lock()

    def set(self,key,value):
        with self.lock:self.values[key]=self._bytes(value)

    @staticmethod
    def _bytes(value):
        return value if isinstance(value,bytes) else str(value).encode()

    def get(self,key):
        with self.lock:return self.values[key]

    def compare_set(self,key,expected,value):
        expected=self._bytes(expected);value=self._bytes(value)
        with self.lock:
            current=self.values.get(key,b'')
            if current==expected:
                self.values[key]=value
                return value
            return current

    def check(self,keys):
        with self.lock:return all(key in self.values for key in keys)

    def keys(self):
        with self.lock:return tuple(self.values)


class GlobalCommitTest(unittest.TestCase):
    def make_outer(self,store,replica,rank,*,epoch=0,step=0,timeout=0.5,
                   native_vote=True,progress=None):
        if progress is None:progress=dict(step=step+1,batches_committed=2*(step+1))
        events=[]
        event_lock=Lock()
        native_calls=[]
        native_lock=Lock()
        def should_commit():
            with native_lock:native_calls.append(native_vote)
            return native_vote
        outer=Outer.__new__(Outer)
        outer.manager=SimpleNamespace(should_commit=should_commit,
                                      state_dict=lambda:dict(progress))
        outer._closed=False
        outer._step_started=step
        outer._reduced_step=step
        outer._commit_store=store
        outer._timeout=timedelta(seconds=timeout)
        outer.timeout=outer._timeout
        outer.epoch=epoch
        outer.replica=replica
        outer.rank=rank
        outer._effective_timeouts=dict(timeout_seconds=90.0,
            quorum_timeout_seconds=180.0,connect_timeout_seconds=20.0)
        outer.event_rows=events
        outer.native_calls=native_calls
        outer.event=lambda kind,**fields: self._record(event_lock,events,kind,replica,rank,fields)
        return outer

    @staticmethod
    def _record(lock,events,kind,replica,rank,fields):
        with lock:events.append(dict(kind=kind,replica=replica,rank=rank,**fields))

    def worker_set(self,store,*,epoch=0,step=0,timeout=0.5,false_vote=None):
        return [self.make_outer(store,replica,rank,epoch=epoch,step=step,timeout=timeout,
                    native_vote=not (false_vote==(replica,rank)))
                for replica in range(2) for rank in range(8)]

    def run_concurrently(self,outers,step,*,update_after_commit=False):
        start=Barrier(len(outers))
        outcomes=[];outcome_lock=Lock();updates=[]

        def run(outer):
            start.wait()
            try:
                outer.commit(step)
                with outcome_lock:
                    outcomes.append((outer.replica,outer.rank,None))
                    if update_after_commit:updates.append((outer.replica,outer.rank))
            except BaseException as error:
                with outcome_lock:outcomes.append((outer.replica,outer.rank,error))

        threads=[Thread(target=run,args=(outer,)) for outer in outers]
        for thread in threads:thread.start()
        for thread in threads:thread.join(timeout=2)
        self.assertFalse(any(thread.is_alive() for thread in threads),'commit threads must finish boundedly')
        return outcomes,updates

    def test_all_sixteen_native_votes_confirm_one_global_commit(self):
        store=FakeStore();outers=self.worker_set(store)
        outcomes,updates=self.run_concurrently(outers,0,update_after_commit=True)
        self.assertEqual(len(outcomes),16)
        self.assertTrue(all(error is None for _,_,error in outcomes))
        self.assertEqual(len(updates),16)
        self.assertEqual(sum(x['kind']=='outer_commit' for outer in outers for x in outer.event_rows),16)
        self.assertEqual(sum(key.endswith('/rejected') for key in store.keys()),0)
        self.assertEqual(sum('/vote/' in key for key in store.keys()),16)

    def test_one_native_false_rejects_already_true_peer_votes_without_optimizer_update(self):
        store=FakeStore();outers=self.worker_set(store,false_vote=(0,0))
        outcomes,updates=self.run_concurrently(outers,0,update_after_commit=True)
        self.assertEqual(len(outcomes),16)
        self.assertTrue(all(error is not None for _,_,error in outcomes))
        self.assertEqual(len(updates),0)
        self.assertEqual(sum(x['kind']=='outer_commit' for outer in outers for x in outer.event_rows),0)
        self.assertTrue(any(key.endswith('/rejected') for key in store.keys()))
        self.assertGreaterEqual(sum('/vote/' in key for key in store.keys()),15)
        self.assertEqual(sum(call for outer in outers for call in outer.native_calls),15)
        self.assertEqual(sum(len(outer.native_calls) for outer in outers),16)

    def test_missing_votes_fail_at_bounded_deadline(self):
        store=FakeStore();outer=self.make_outer(store,0,0,timeout=0.03)
        started=time.monotonic()
        with self.assertRaises(TimeoutError):outer.commit(0)
        elapsed=time.monotonic()-started
        self.assertGreaterEqual(elapsed,0.02)
        self.assertLess(elapsed,0.5)
        self.assertFalse(any(x['kind']=='outer_commit' for x in outer.event_rows))
        self.assertTrue(any(key.endswith('/rejected') for key in store.keys()))

    def test_old_epoch_or_old_step_votes_do_not_satisfy_current_commit(self):
        for stale_epoch,stale_step in ((0,2),(1,1)):
            with self.subTest(stale_epoch=stale_epoch,stale_step=stale_step):
                store=FakeStore()
                prefix=f'v4-opt-commit/{stale_epoch}/{stale_step}/'
                for replica in range(2):
                    for rank in range(8):
                        store.set(prefix+f'vote/{replica}/{rank}','1')
                outer=self.make_outer(store,0,0,epoch=1,step=2,timeout=0.03)
                with self.assertRaises(TimeoutError):outer.commit(2)
                self.assertFalse(any(x['kind']=='outer_commit' for x in outer.event_rows))

    def test_new_step_and_new_epoch_get_fresh_global_commit_sets(self):
        store=FakeStore();outers=self.worker_set(store)
        for epoch,step in ((0,0),(0,1),(1,2)):
            with self.subTest(epoch=epoch,step=step):
                if step:
                    for outer in outers:
                        outer.epoch=epoch
                        outer._step_started=step
                        outer._reduced_step=step
                        outer.manager.state_dict=lambda step=step:dict(
                            step=step+1,batches_committed=2*(step+1))
                outcomes,updates=self.run_concurrently(outers,step,update_after_commit=True)
                self.assertEqual(len(outcomes),16)
                self.assertTrue(all(error is None for _,_,error in outcomes))
                self.assertEqual(len(updates),16)
        self.assertEqual(sum(x['kind']=='outer_commit' for outer in outers for x in outer.event_rows),48)

    def test_bad_native_progress_is_rejected_before_true_vote(self):
        store=FakeStore()
        outer=self.make_outer(store,0,0,step=3,progress=dict(step=3,batches_committed=6))
        with self.assertRaises(RuntimeError):outer.commit(3)
        prefix='v4-opt-commit/0/3/'
        keys=store.keys()
        self.assertFalse(any(key.startswith(prefix+'vote/') for key in keys))
        self.assertIn(prefix+'rejected',keys)
        self.assertEqual(store.get(prefix+'decision'),b'abort')
        self.assertFalse(any(x['kind']=='outer_commit' for x in outer.event_rows))

    def test_late_timeout_observes_immutable_commit_decision(self):
        store=FakeStore();prefix='v4-opt-commit/0/0/'
        for replica in range(2):
            for rank in range(8):
                store.set(prefix+f'vote/{replica}/{rank}','1')
        self.assertEqual(store.compare_set(prefix+'decision','','commit'),b'commit')
        outer=self.make_outer(store,0,0,timeout=0)
        outer.commit(0)
        self.assertTrue(any(x['kind']=='outer_commit' for x in outer.event_rows))
        self.assertEqual(store.get(prefix+'decision'),b'commit')
        self.assertFalse(store.check([prefix+'rejected']))

    def test_abort_decision_cannot_be_changed_after_all_votes_arrive(self):
        store=FakeStore();prefix='v4-opt-commit/0/0/'
        self.assertEqual(store.compare_set(prefix+'decision','','abort'),b'abort')
        for replica in range(2):
            for rank in range(8):
                store.set(prefix+f'vote/{replica}/{rank}','1')
        outers=self.worker_set(store,timeout=0.1)
        outcomes,updates=self.run_concurrently(outers,0,update_after_commit=True)
        self.assertTrue(all(error is not None for _,_,error in outcomes))
        self.assertEqual(updates,[])
        self.assertEqual(sum(x['kind']=='outer_commit' for outer in outers for x in outer.event_rows),0)
        self.assertEqual(store.get(prefix+'decision'),b'abort')


if __name__=='__main__':
    unittest.main()
