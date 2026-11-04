"""Bucket readiness, native tail order, and exceptional callback lifetime."""
from types import SimpleNamespace
import unittest

from shift_full.bucket_finalizer import BucketFinalizer


class BucketFinalizerTest(unittest.TestCase):
    def make(self, hook):
        rows = []
        buckets = [SimpleNamespace(finish_grad_sync=lambda i=i: rows.append(('ready', i)))
                   for i in range(3)]
        def native():
            for bucket in buckets:
                bucket.finish_grad_sync()
            rows.append(('tail', None))
        runner = BucketFinalizer(native, [SimpleNamespace(buckets=buckets[:2]),
                                        SimpleNamespace(buckets=buckets[2:])], hook)
        return runner, buckets, rows

    def test_persistent_hooks_follow_each_native_ready_and_keep_tail(self):
        runner, buckets, rows = self.make(True)
        hooks = [bucket.finish_grad_sync for bucket in buckets]
        for _ in range(2):
            runner.run(lambda bucket: rows.append(('outer', buckets.index(bucket))))
        expected = [('ready', 0), ('outer', 0), ('ready', 1), ('outer', 1),
                    ('ready', 2), ('outer', 2), ('tail', None)]
        self.assertEqual(rows, expected * 2)
        self.assertEqual(hooks, [bucket.finish_grad_sync for bucket in buckets])
        buckets[0].finish_grad_sync()
        self.assertEqual(rows[-1], ('ready', 0))

    def test_no_hook_retains_native_tail_before_outer(self):
        runner, buckets, rows = self.make(False)
        runner.run(lambda bucket: rows.append(('outer', buckets.index(bucket))))
        self.assertEqual(rows, [('ready', 0), ('ready', 1), ('ready', 2), ('tail', None),
                                ('outer', 0), ('outer', 1), ('outer', 2)])

    def test_failure_clears_callback_and_does_not_dispatch_unready_buckets(self):
        runner, buckets, rows = self.make(True)
        def fail(_):
            raise RuntimeError('dispatch failed')
        with self.assertRaisesRegex(RuntimeError, 'dispatch failed'):
            runner.run(fail)
        self.assertIsNone(runner.enqueue)
        self.assertEqual(rows, [('ready', 0)])
        buckets[1].finish_grad_sync()
        self.assertEqual(rows[-1], ('ready', 1))


if __name__ == '__main__':
    unittest.main()
