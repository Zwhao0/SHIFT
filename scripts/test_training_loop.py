"""Behavior tests for the common runtime loop, using fake transactions."""
import unittest
from types import SimpleNamespace

from shift_full.training_loop import run_loop


class FakeTransaction:
    def __init__(self, runtime):
        self.runtime = runtime
        self.sends = []
        self.closed = False
        self.yielded = False

    def __next__(self):
        if self.yielded:
            raise StopIteration
        self.yielded = True
        return SimpleNamespace(finite=True)

    def send(self, commit):
        self.sends.append(commit)
        if len(self.sends) != 1:
            raise AssertionError('transaction sent more than once')
        if commit:
            self.runtime.adam_steps += 1
            self.runtime.progress['updates'] += 1
        raise StopIteration({'step': self.runtime.progress['updates']})

    def close(self):
        self.closed = True


class FakeRuntime:
    def __init__(self, policy=None):
        self.progress = {'updates': 0}
        self.policy = policy
        self.stop = False
        self.adam_steps = 0
        self.transactions = []
        self.events = []

    def should_stop(self):
        return self.stop

    def compute(self, context):
        transaction = FakeTransaction(self)
        self.transactions.append(transaction)
        return transaction

    def committed(self, context, computed, update):
        self.events.append(('committed', context, update))

    def skipped(self, context, computed):
        self.events.append(('skipped', context))


class FakePolicy:
    def __init__(self, votes=()):
        self.votes = iter(votes)
        self.begins = []
        self.vote_count = 0
        self.before_count = 0
        self.on_before = None
        self.on_vote = None

    def before_iteration(self, runtime):
        self.before_count += 1
        if self.on_before:
            return self.on_before(runtime)
        return True

    def begin(self, runtime):
        context = {'attempt': len(self.begins) + 1}
        self.begins.append(context)
        return context

    def vote(self, runtime, context, computed):
        self.vote_count += 1
        if self.on_vote:
            return self.on_vote(runtime, context, computed)
        return next(self.votes)


class TrainingLoopTest(unittest.TestCase):
    def test_rejection_retries_without_advancing_adam_or_progress(self):
        policy = FakePolicy([False, True])
        runtime = FakeRuntime(policy)

        self.assertEqual(run_loop(runtime, policy, until=1, max_attempts=2), 1)
        self.assertEqual(runtime.adam_steps, 1)
        self.assertEqual(runtime.progress['updates'], 1)
        self.assertEqual([tx.sends for tx in runtime.transactions], [[False], [True]])
        self.assertEqual([event[0] for event in runtime.events], ['skipped', 'committed'])

    def test_before_hook_policy_change_applies_to_upcoming_attempt(self):
        old = FakePolicy([True])
        new = FakePolicy([True])
        runtime = FakeRuntime(old)
        old.on_before = lambda rt: (setattr(rt, 'policy', new) or True)

        run_loop(runtime, old, until=1)
        self.assertEqual(old.begins, [])
        self.assertEqual(len(new.begins), 1)
        self.assertEqual(runtime.progress['updates'], 1)

    def test_vote_policy_change_waits_until_next_attempt(self):
        old = FakePolicy([True])
        new = FakePolicy([True])
        runtime = FakeRuntime(old)
        old.on_vote = lambda rt, context, computed: (setattr(rt, 'policy', new) or True)

        run_loop(runtime, old, until=2)
        self.assertEqual(len(old.begins), 1)
        self.assertEqual(old.vote_count, 1)
        self.assertEqual(len(new.begins), 1)
        self.assertEqual(new.vote_count, 1)

    def test_stop_before_attempt_does_not_commit(self):
        policy = FakePolicy([True])
        runtime = FakeRuntime(policy)
        runtime.stop = True

        self.assertEqual(run_loop(runtime, policy, until=1), 0)
        self.assertEqual(runtime.transactions, [])
        self.assertEqual(runtime.adam_steps, 0)
        self.assertEqual(runtime.progress['updates'], 0)

    def test_vote_exception_closes_unresolved_transaction(self):
        policy = FakePolicy()
        runtime = FakeRuntime(policy)

        def fail(*_):
            raise ValueError('vote failed')

        policy.on_vote = fail
        with self.assertRaisesRegex(ValueError, 'vote failed'):
            run_loop(runtime, policy, until=1)
        self.assertTrue(runtime.transactions[0].closed)
        self.assertEqual(runtime.transactions[0].sends, [])
        self.assertEqual(runtime.progress['updates'], 0)

    def test_attempt_limit_raises_after_rejected_votes(self):
        policy = FakePolicy([False, False])
        runtime = FakeRuntime(policy)

        with self.assertRaisesRegex(RuntimeError, 'max_attempts=2'):
            run_loop(runtime, policy, until=1, max_attempts=2)
        self.assertEqual([tx.sends for tx in runtime.transactions], [[False], [False]])
        self.assertEqual(runtime.progress['updates'], 0)

    def test_each_attempt_sends_exactly_once(self):
        policy = FakePolicy([False, True])
        runtime = FakeRuntime(policy)

        run_loop(runtime, policy, until=1)
        self.assertTrue(all(len(tx.sends) == 1 for tx in runtime.transactions))


if __name__ == '__main__':
    unittest.main()
