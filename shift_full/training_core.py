"""Shared numerical training and complete state; policies own membership only.

No digest is evaluated by a normal step. The split transaction lets recovery
policies vote/fence after backward and before the sole optimizer implementation.
"""
from dataclasses import dataclass
import random
import time


def initialize_rng(logical_rank):
    """Replicated DP state, keyed by logical shard rather than physical GPU.

    Native TorchFT copies a complete healthy DP state on fail-stop recovery.
    Paired replicas therefore share the RNG stream for each logical shard;
    sample randomness is separately keyed by the globally committed sample ID.
    """
    import torch
    torch.manual_seed(70_000 + logical_rank)
    random.seed(80_000 + logical_rank)


def sample_plan(step, *, microbatches=2, participants=2, replica=0, local_data_rank=0,
                fixed_batch=True):
    if fixed_batch and participants not in (1, 2):
        raise ValueError('batch8 requires one or two complete DP replicas')
    count = microbatches * (2 // participants if fixed_batch else 1)
    width = 2 * participants if fixed_batch else 4
    return [step * microbatches * 4 + m * width + replica * 2 + local_data_rank
            for m in range(count)]


def make_batches(samples, *, seq_len, vocab_size, data='random', device='cuda'):
    import torch
    batches = []
    for sample in samples:
        if data == 'random':
            generator = torch.Generator(device=device).manual_seed(100_000 + sample)
            tokens = torch.randint(0, vocab_size, (1, seq_len+1), device=device, generator=generator)
        elif data == 'pattern':
            tokens = (torch.arange(seq_len+1, device=device).view(1, -1) + sample % 8) % 32 + 100
        else:
            raise ValueError('unknown training data')
        batches.append(dict(input_ids=tokens[:, :-1].contiguous(), next_token_labels=tokens[:, 1:].contiguous()))
    return batches


def partition_model(full, *, ep, tp, pp_rank, StageModel, patch_moe, patch_tensor_parallel):
    patch_moe(full, ep)
    stage = StageModel(full, pp_rank)
    manifest = patch_tensor_parallel(stage, tp)
    return stage, manifest


def configure_model(stage, *, adapter, dense, expert_dp, finalize, bucket_size=67_108_864,
                    lr=1e-4, scheduler_step=1000):
    import torch
    stage.cuda()
    ddp = adapter.wrap(stage, dense, expert_dp, bucket_size, config=stage.config)
    stage.config.no_sync_func = ddp.no_sync
    stage.config.finalize_model_grads_func = finalize
    import os
    if os.environ.get('SHIFT_CPU_ADAM')=='1':
        from .offloaded_adam import OffloadedAdamW
        optimizer=OffloadedAdamW(stage.parameters(),lr=lr)
    else:optimizer = torch.optim.AdamW(stage.parameters(), lr=lr, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step, gamma=.98)
    return ddp, optimizer, scheduler


class TrainingState:
    def __init__(self, stage, optimizer, scheduler, progress, tracker):
        self.stage, self.optimizer, self.scheduler = stage, optimizer, scheduler
        self.progress, self.tracker = progress, tracker

    def rng(self):
        import torch
        from .state_inventory import tracker_snapshot
        return dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    python_rng=random.getstate(), tracker_rng=tracker_snapshot(self.tracker))

    def restore_rng(self, value):
        import torch
        from .state_inventory import restore_tracker
        torch.set_rng_state(value['torch_rng']); torch.cuda.set_rng_state(value['cuda_rng'])
        random.setstate(value['python_rng']); restore_tracker(self.tracker, value['tracker_rng'])

    def hashes(self, rng=None, *, model_state=None, all_buffers=None):
        import torch
        from .train import digest
        torch.cuda.current_stream().synchronize()
        r = self.rng() if rng is None else rng
        return dict(model=digest(self.stage.state_dict() if model_state is None else model_state),
            optimizer=digest(self.optimizer.state_dict()), scheduler=digest(self.scheduler.state_dict()),
            progress=digest(self.progress), buffers=digest(dict(self.stage.named_buffers()) if all_buffers is None else all_buffers),
            cpu_rng=digest(r['torch_rng']), cuda_rng=digest(r['cuda_rng']),
            python_rng=digest(r['python_rng']), tracker_rng=digest(r['tracker_rng']))

    def capture(self, *, rng=None, all_buffers=None, **metadata):
        buffers = dict(self.stage.named_buffers()) if all_buffers is None else all_buffers
        model = self.stage.state_dict()
        for name, value in buffers.items():
            if name in model: model[name] = value
        return dict(schema=2, model=model, all_buffers=buffers,
            optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
            progress=dict(self.progress), step=self.progress['updates'],
            **(self.rng() if rng is None else rng), **metadata)

    def restore(self, value, *, clone_optimizer=False):
        import torch
        from torch.utils._pytree import tree_map
        from .state_inventory import restore_buffers
        if value['step'] != value['progress']['updates']:
            raise ValueError('state step differs from sample/update cursor')
        self.stage.load_state_dict(value['model'], strict=True)
        restore_buffers(self.stage, value['all_buffers'])
        optimizer = value['optimizer']
        if clone_optimizer:
            optimizer = tree_map(lambda t:t.detach().clone() if isinstance(t, torch.Tensor) else t, optimizer)
        self.optimizer.load_state_dict(optimizer); self.scheduler.load_state_dict(value['scheduler'])
        self.progress.clear(); self.progress.update(value['progress'])
        self.restore_rng(value)


@dataclass
class Computed:
    metrics: object
    finite: bool
    schedule_done: float
    gradient_checked: float


def gradients_finite(buffers,chunk_elements=8*1024*1024):
    """Same all-element predicate with bounded boolean/intermediate tensors."""
    import torch
    if type(chunk_elements) is not int or chunk_elements<1:raise ValueError('positive finite-check chunk required')
    checks=[]
    for buf in buffers:
        for bucket in buf.buckets:
            flat=bucket.grad_data.reshape(-1)
            checks.extend(torch.isfinite(flat[i:i+chunk_elements]).all() for i in range(0,flat.numel(),chunk_elements))
    return bool(torch.stack(checks).all()) if checks else True


def train_step(stage, ddp, optimizer, scheduler, progress, *, execute, synchronize=True,
               schedule_end=None, finite_scope=None, optimizer_scope=None, mark=None):
    """Compute once, yield for policy commit vote, then update exactly once.

    A false vote never mutates Adam, scheduler or committed sample progress.
    All normal and relay paths use this same gradient check and update path.
    """
    from contextlib import nullcontext
    import torch
    metrics = execute()
    for parameter in stage.parameters(): parameter.grad = parameter.main_grad
    if schedule_end: schedule_end()
    schedule_done = time.monotonic()
    buffers = ddp.buffers + ddp.expert_parallel_buffers
    with finite_scope() if finite_scope else nullcontext():
        import os
        finite=(gradients_finite(buffers) if os.environ.get('SHIFT_CPU_ADAM')=='1' else
                bool(torch.stack([torch.isfinite(b.grad_data).all() for buf in buffers for b in buf.buckets]).all()))
    computed = Computed(metrics, finite, schedule_done, time.monotonic())
    commit = yield computed
    if commit:
        if not finite: raise RuntimeError('nonfinite gradients cannot commit')
        if mark: mark('optimizer_begin')
        started = time.monotonic()
        with optimizer_scope() if optimizer_scope else nullcontext():
            optimizer.step(); scheduler.step()
        if mark: mark('optimizer_end')
        optimizer_done = time.monotonic()
        if synchronize: torch.cuda.current_stream().synchronize()
        sync_done = time.monotonic()
        progress['updates'] += 1
        return dict(optimizer_started=started, optimizer_done=optimizer_done, step_sync_done=sync_done)
    return None


def complete_step(transaction, commit):
    try: transaction.send(commit)
    except StopIteration as result: return result.value
    raise RuntimeError('training step yielded more than one policy vote')
