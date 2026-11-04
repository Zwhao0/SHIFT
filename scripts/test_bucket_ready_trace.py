"""Diagnostic lifecycle and native work ordering without CUDA dependencies."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest

from shift_full.bucket_ready_trace import BucketReadyTrace


class FakeCuda:
    def __init__(self):
        self.current = 'compute'
        self.log = []

    def Stream(self):
        name = object()
        return SimpleNamespace(
            wait_event=lambda event: self.log.append(('event_wait', self.current)),
            synchronize=lambda: self.log.append(('sync', name)))

    def Event(self, **kwargs):
        return SimpleNamespace(record=lambda: self.log.append(('record', self.current)),
                               elapsed_time=lambda other: 1.0)

    @contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous


class ReadyTraceTest(unittest.TestCase):
    def make(self):
        cuda = FakeCuda()
        bucket = SimpleNamespace(grad_data=SimpleNamespace(numel=lambda: 123),
                                 communication_handle=SimpleNamespace(
                                     wait=lambda: cuda.log.append(('local_wait', cuda.current))))
        bucket.start_grad_sync = lambda: cuda.log.append(('native', cuda.current))
        return cuda, bucket, BucketReadyTrace([bucket], cuda)

    def test_inactive_then_native_first_and_wait_only_on_diagnostic_stream(self):
        cuda, bucket, trace = self.make()
        bucket.start_grad_sync()
        self.assertEqual(cuda.log, [('native', 'compute')])
        cuda.log.clear()
        trace.begin()
        bucket.start_grad_sync()
        self.assertEqual([x[0] for x in cuda.log],
                         ['native', 'record', 'event_wait', 'local_wait', 'record'])
        self.assertIs(cuda.log[3][1], trace.streams[0])
        self.assertEqual(cuda.current, 'compute')
        rows = trace.collect(cuda.Event(), cuda.Event(), cuda.Event())
        self.assertEqual(rows[0]['numel'], 123)
        self.assertFalse(trace.active)
        trace.begin()
        bucket.start_grad_sync()
        self.assertEqual(len(trace.collect(cuda.Event(), cuda.Event(), cuda.Event())), 1)

    def test_missing_duplicate_and_nested_diagnostics_fail(self):
        cuda, bucket, trace = self.make()
        trace.begin()
        with self.assertRaisesRegex(RuntimeError, 'nested'): trace.begin()
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            trace.collect(cuda.Event(), cuda.Event(), cuda.Event())
        trace.begin()
        bucket.start_grad_sync()
        with self.assertRaisesRegex(RuntimeError, 'duplicate'): bucket.start_grad_sync()

    def test_discard_skipped_attempt_can_start_next_attempt(self):
        cuda, bucket, trace = self.make()
        trace.begin()
        bucket.start_grad_sync()
        trace.discard()
        self.assertFalse(trace.active)
        self.assertEqual(trace.rows, {})
        trace.begin()
        bucket.start_grad_sync()
        trace.collect(cuda.Event(), cuda.Event(), cuda.Event())


if __name__ == '__main__':
    unittest.main()
