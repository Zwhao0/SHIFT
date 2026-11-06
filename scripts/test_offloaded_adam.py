"""Exact CPU AdamW trajectory and restore checks, no CUDA initialization."""
import copy
import unittest

try:
    import torch
except ImportError:torch=None


@unittest.skipIf(torch is None,'requires retained Torch runtime')
class OffloadTests(unittest.TestCase):
    def test_chunked_finite_check_covers_nan_inf_and_last_partial_chunk(self):
        from types import SimpleNamespace
        from shift_full.training_core import gradients_finite
        values=torch.arange(11,dtype=torch.bfloat16)
        buffers=[SimpleNamespace(buckets=[SimpleNamespace(grad_data=values)])]
        self.assertTrue(gradients_finite(buffers,4))
        for index,bad in [(4,float('nan')),(10,float('inf'))]:
            old=values[index].clone();values[index]=bad
            self.assertFalse(gradients_finite(buffers,4));values[index]=old
    def test_scheduler_and_checkpoint_resume_match_cpu_adamw(self):
        from shift_full.offloaded_adam import OffloadedAdamW
        a=torch.nn.Parameter(torch.tensor([1.,-2.,3.]));b=torch.nn.Parameter(a.detach().clone())
        standard=torch.optim.AdamW([a],lr=.01,foreach=False)
        offloaded=OffloadedAdamW([b],lr=.01)
        sa=torch.optim.lr_scheduler.StepLR(standard,2,gamma=.9)
        sb=torch.optim.lr_scheduler.StepLR(offloaded,2,gamma=.9)
        for step in range(8):
            a.grad=torch.tensor([step+.5,2.,-1.]);b.grad=a.grad.clone()
            standard.step();sa.step();offloaded.step();sb.step()
            self.assertTrue(torch.equal(a,b))
            for name,tensor in standard.state_dict()['state'][0].items():
                self.assertTrue(torch.equal(tensor,offloaded.state_dict()['state'][0][name]))
            self.assertEqual(standard.param_groups[0]['lr'],offloaded.param_groups[0]['lr'])
            if step==3:
                payload=copy.deepcopy(offloaded.state_dict());scheduler=copy.deepcopy(sb.state_dict())
                b=torch.nn.Parameter(b.detach().clone())
                offloaded=OffloadedAdamW([b],lr=99.)
                sb=torch.optim.lr_scheduler.StepLR(offloaded,2,gamma=.9)
                offloaded.load_state_dict(payload);sb.load_state_dict(scheduler)

    def test_zero_grad_and_eager_template_do_not_update_parameters(self):
        from shift_full.offloaded_adam import OffloadedAdamW
        p=torch.nn.Parameter(torch.ones(4,dtype=torch.bfloat16));o=OffloadedAdamW([p])
        state=o.state_dict()['state'][0]
        self.assertEqual(set(state),{'step','exp_avg','exp_avg_sq'})
        self.assertEqual(state['exp_avg'].device.type,'cpu')
        self.assertEqual(state['exp_avg'].dtype,torch.bfloat16)
        o.zero_grad();self.assertTrue(torch.equal(p,torch.ones_like(p)))
        self.assertEqual(state['step'].item(),0)

    def test_bfloat16_moments_match_cpu_adamw(self):
        from shift_full.offloaded_adam import OffloadedAdamW
        a=torch.nn.Parameter(torch.tensor([1.,2.,3.],dtype=torch.bfloat16))
        b=torch.nn.Parameter(a.detach().clone())
        x=torch.optim.AdamW([a],lr=.03,foreach=False);y=OffloadedAdamW([b],lr=.03)
        for step in range(5):
            a.grad=torch.full_like(a,.3+step);b.grad=a.grad.clone();x.step();y.step()
            self.assertTrue(torch.equal(a,b))
            for key,value in x.state_dict()['state'][0].items():
                self.assertTrue(torch.equal(value,y.state_dict()['state'][0][key]))


if __name__=='__main__':unittest.main()
