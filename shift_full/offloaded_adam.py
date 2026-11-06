"""CPU AdamW state for bounded GPU memory; complete state remains checkpointed."""
import torch


class OffloadedAdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-4):
        params=list(params)
        self._gpu_params=params
        self._cpu_params=[torch.nn.Parameter(p.detach().to('cpu').clone(),requires_grad=p.requires_grad) for p in params]
        self._cpu_optimizer=torch.optim.AdamW(self._cpu_params,lr=lr,foreach=False)
        super().__init__(params,dict(self._cpu_optimizer.defaults))
        # Eager CPU-only initialization gives a cold candidate a full structural
        # template without doing a warmup update or allocating GPU Adam tensors.
        for p in self._cpu_params:
            self._cpu_optimizer.state[p]=dict(step=torch.tensor(0.,device='cpu'),
                exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
        self.state=self._cpu_optimizer.state

    def _sync_options(self):
        for gpu,cpu in zip(self.param_groups,self._cpu_optimizer.param_groups):
            cpu.update({key:value for key,value in gpu.items() if key!='params'})

    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:raise ValueError('offloaded validation profile does not use closures')
        self._sync_options()
        for gpu,cpu in zip(self._gpu_params,self._cpu_params):
            cpu.grad=None if gpu.grad is None else gpu.grad.detach().to('cpu')
        self._cpu_optimizer.step()
        for gpu,cpu in zip(self._gpu_params,self._cpu_params):gpu.copy_(cpu)
        self._cpu_optimizer.zero_grad(set_to_none=True)

    def zero_grad(self,set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self._cpu_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        self._sync_options()
        return self._cpu_optimizer.state_dict()

    @torch.no_grad()
    def load_state_dict(self,value):
        self._cpu_optimizer.load_state_dict(value)
        for gpu,cpu in zip(self.param_groups,self._cpu_optimizer.param_groups):
            gpu.update({key:value for key,value in cpu.items() if key!='params'})
        # TrainingState restores the model before restoring Adam. Reconstruct
        # CPU parameter copies from those restored GPU values, never stale warmup.
        for gpu,cpu in zip(self._gpu_params,self._cpu_params):cpu.copy_(gpu)
        self.state=self._cpu_optimizer.state
