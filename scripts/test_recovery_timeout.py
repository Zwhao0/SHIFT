"""CPU-only regression of checkpoint/gradient deadline isolation."""
from datetime import timedelta
from types import SimpleNamespace
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch.distributed import Work
from shift_full.compat import BlockingNCCL, RecoveryNCCL, RecoveryWork, cancelling_timeout


class FakeWork(Work):
    pass


class Parent:
    def __init__(self):
        self.calls = []

    def send(self, tensors, peer, tag):
        self.calls.append(('send', peer, tag))
        return FakeWork()

    def recv(self, tensors, peer, tag):
        self.calls.append(('recv', peer, tag))
        return FakeWork()


class DeadlineTest(unittest.TestCase):
    def test_checkpoint_keeps_gradient_deadline(self):
        pg = RecoveryNCCL(timeout=timedelta(seconds=2))
        parent = Parent()
        pg._pg = parent
        for method in (pg.send, pg.recv):
            work = method([], 1, 7)
            self.assertEqual(work._timeout, timedelta(seconds=90))
            self.assertEqual(pg._timeout, timedelta(seconds=2))
        self.assertEqual(parent.calls, [('send', 1, 7), ('recv', 1, 7)])
        gradient = pg._wrap_work(FakeWork(), None)
        self.assertEqual(gradient._timeout, timedelta(seconds=2))
        with patch.object(parent, 'send', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                pg.send([], 1, 7)
        self.assertEqual(pg._timeout, timedelta(seconds=2))

    def test_exception_cancels_pinned_timer(self):
        cancelled = []
        @contextmanager
        def upstream_guard(callback, timeout):
            yield
            cancelled.append(True)
        with patch('torchft.process_group.context_timeout', upstream_guard):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with cancelling_timeout(lambda: None, timedelta(seconds=90)):
                    raise RuntimeError('injected')
        self.assertEqual(cancelled, [True])

    def test_failed_wait_does_not_schedule_late_stream_abort(self):
        pg = SimpleNamespace(abort=lambda: None)
        with patch('torchft.process_group.stream_timeout') as later:
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with RecoveryWork._stream_timeout(pg, timedelta(seconds=2)):
                    raise RuntimeError('injected')
            later.assert_not_called()

    def test_prepared_backend_inherits_recovery_guard(self):
        from shift_full.prepared_pg import PreparedNCCL
        self.assertTrue(issubclass(PreparedNCCL, RecoveryNCCL))

    def test_native_deadline_allows_checkpoint_budget(self):
        pg = BlockingNCCL(timeout=timedelta(seconds=2))
        observed = {}

        class Group:
            BackendType = SimpleNamespace(NCCL=1)
            def __init__(self, *args): pass
            def _set_default_backend(self, *args): pass
            def _register_backend(self, *args): pass

        class Native:
            @staticmethod
            def Options():
                return SimpleNamespace(config=SimpleNamespace())
            def __init__(self, store, rank, size, opts):
                observed['timeout'] = opts._timeout
                observed['blocking'] = opts.config.blocking
            def _set_sequence_number_for_group(self): pass
            def eager_connect_single_device(self, device): pass

        with patch('torch.distributed.ProcessGroup', Group), patch('torch.distributed.ProcessGroupNCCL', Native), patch('torch.cuda.current_device', return_value=0):
            pg._create_pg(None, 0, 2)
        self.assertEqual(observed['timeout'], timedelta(seconds=90))
        self.assertTrue(observed['blocking'])
        self.assertEqual(pg._timeout, timedelta(seconds=2))


if __name__ == '__main__':
    unittest.main()
