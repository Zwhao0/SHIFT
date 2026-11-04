"""Bounded capture ordering, storage independence and single-step runner contract."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class ProbeRunnerTest(unittest.TestCase):
    def test_offline_bf16_rounding_and_tree_envelope(self):
        from scripts.summarize_v3_gradient_probe import bf16, possible_sums
        self.assertEqual(bf16(1 + 1/256), 1)
        self.assertEqual(bf16(1 + 3/256), 1 + 1/64)
        self.assertEqual(possible_sums((1, 1/128)), {1 + 1/128})
        self.assertIn(bf16(1 + 3/256), possible_sums((1, 1/256, 1/256, 1/256)))

    def test_single_formal_update_retains_audits_and_anchor(self):
        from scripts import v3_numerical_remote as n
        session = MagicMock()
        session.endpoint_for.return_value = 'same-host'
        with patch.object(n, 'STEPS', [2]), patch.object(n, 'NoFaultRotation', return_value=session) as rotation, patch.object(n, 'audit'):
            n.run_case('probe', 'shift', Path('/original-anchor'), reuse_anchor=True, dump=False, gradient_probe=True)
            c = rotation.call_args.args[1]
            self.assertEqual(c['target_steps'], 2)
            a = c['train_args']
            self.assertEqual(a[a.index('--seq-len')+1], '512')
            self.assertEqual(a[a.index('--gradient-probe-step')+1], '2')
            self.assertTrue({'--audit-state', '--reference-signatures', '--audit-routing', '--measure-step'} <= set(a))
            self.assertFalse({'--strict-fp32', '--reference-dump', '--timeline-trace-step', '--warmup-before-join'} & set(a))
            session.finish.assert_called_once()
        with self.assertRaises(ValueError):
            n.run_case('bad', 'shift', Path('/anchor'), reuse_anchor=True, dump=False, gradient_probe=True)


class ProbeStorageTest(unittest.TestCase):
    def check_device(self, device):
        import torch
        from shift_full.gradient_probe import GradientProbe
        from shift_full.bucket_finalizer import BucketFinalizer
        named = [('decoder.layers.0.self_attn.v_proj.bias', 64),
                 ('decoder.layers.0.mlp.shared_expert_gate.weight', 896),
                 ('decoder.layers.0.mlp.experts.0.gate_proj.weight', 1024)]
        params = []
        for name, size in named:
            p = torch.nn.Parameter(torch.zeros(size, dtype=torch.bfloat16, device=device))
            p.main_grad = torch.full_like(p, 4)
            p.allreduce = '.experts.' not in name
            params.append(p)
        stage = SimpleNamespace(named_parameters=lambda: zip([n for n, _ in named], params))
        bucket = SimpleNamespace(params=set(params), data_parallel_group='group', gradient_scaling_factor=.5)
        def start():
            for p in params: p.main_grad.mul_(.5)
        bucket.start_grad_sync = start
        bucket.finish_grad_sync = lambda: None
        buffers = [SimpleNamespace(buckets=[bucket])]
        progress = dict(updates=1)
        probe = GradientProbe(torch, SimpleNamespace(get_process_group_ranks=lambda _: [0, 2]), stage, buffers, 2, progress, {})
        runner = BucketFinalizer(lambda: bucket.finish_grad_sync(), buffers, True)
        bucket.start_grad_sync()
        def outer(_):
            for p in params: p.main_grad.add_(1)
        runner.run(outer)
        probe.capture('post_outer')
        for p in params: p.main_grad.add_(1)
        probe.capture('post_tail')
        result = probe.result()
        self.assertEqual(len(result['tensors']), 12)
        expected = dict(pre_local=4, post_local=2, post_outer=3, post_tail=4)
        for row in result['tensors']:
            self.assertEqual(set(row['values']), {expected[row['phase']]})
        for p in params: p.main_grad.zero_()
        self.assertEqual(result, probe.result())
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            probe.capture('post_tail')
        progress['updates'] = 2
        probe.capture('post_tail')  # disabled after the selected update

    def test_cpu_storage_and_order(self):
        self.check_device('cpu')

    def test_cuda_storage_and_order(self):
        import torch
        self.assertTrue(torch.cuda.is_available(), 'CUDA required on independent server test')
        self.check_device('cuda')


if __name__ == '__main__':
    unittest.main()
