"""Minimal SHIFT integration: existing Qwen MoE, independent 8-GPU replicas."""
import argparse
from datetime import timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import socket
import threading
import time
import types

import torch
import torch.distributed as dist


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def digest(value):
    h = hashlib.sha256()

    def add(v):
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            h.update(str((str(t.dtype), tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                add(key)
                add(v[key])
        elif isinstance(v, (list, tuple)):
            h.update(str(type(v)).encode())
            for item in v:
                add(item)
        else:
            h.update(repr(v).encode() + b"\0")

    add(value)
    return h.hexdigest()


def tensor_count(value):
    if isinstance(value, torch.Tensor):
        return 1
    if isinstance(value, dict):
        return sum(tensor_count(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_count(v) for v in value)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', required=True)
    p.add_argument('--replica', type=int, required=True)
    p.add_argument('--generation', type=int, required=True)
    p.add_argument('--host-ip', required=True)
    p.add_argument('--lighthouse', required=True)
    p.add_argument('--root', default='/workspace/SHIFT')
    p.add_argument('--flash', default='/workspace/FLASH1022')
    p.add_argument('--max-seconds', type=int, default=900)
    args = p.parse_args()
    os.environ['FLASH_PARALLEL_LAYOUT'] = 'legacy'  # Native ordering INSIDE each replica.
    os.environ['TORCHFT_USE_OTEL'] = 'false'
    logging.basicConfig(level=logging.WARNING)
    rank = int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    assert dist.get_world_size() == 8
    out = Path(args.root) / 'results' / args.session
    out.mkdir(parents=True, exist_ok=True)
    log = (out / f'events_g{args.generation}_r{rank}.jsonl').open('a', buffering=1)
    lock = threading.Lock()

    def event(kind, **fields):
        row = dict(kind=kind, time=time.time(), rid=args.replica, rank=rank,
                   generation=args.generation, pid=os.getpid(), **fields)
        with lock:
            log.write(json.dumps(row, sort_keys=True) + '\n')

    import sys
    sys.path.insert(0, args.flash)
    from flash_v1 import megatron_adapter
    from flash_v1.pipeline_adapter import StageModel, forward_step
    from flash_v1.tensor_parallel import patch_tensor_parallel
    from flash_v1.train import load_model, patch_moe, MODEL_ID, REVISION
    topology_out = out / f'topology_g{args.generation}'
    topology_out.mkdir(parents=True, exist_ok=True)
    ep, expert_dp, dense = megatron_adapter.initialize_groups(topology_out, 2, 2, 2)
    state, _, _ = megatron_adapter.core()
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    assert dist.get_world_size(expert_dp) == 1
    assert dist.get_world_size(dense) == 2
    torch.manual_seed(2026)
    full = load_model(Path(args.flash) / 'models/qwen_moe')
    total_parameters = sum(v.numel() for v in full.parameters())
    patch_moe(full, ep)
    stage = StageModel(full, state.get_pipeline_model_parallel_rank())
    tp_manifest = patch_tensor_parallel(stage, state.get_tensor_model_parallel_group())
    del full
    stage.cuda()
    ddp = megatron_adapter.wrap(stage, dense, expert_dp, 2_097_152, config=stage.config)
    stage.config.no_sync_func = ddp.no_sync
    stage.config.finalize_model_grads_func = finalize_model_grads
    optimizer = torch.optim.AdamW(stage.parameters(), lr=1e-3, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.98)
    progress = {'updates': 0}
    manager = None

    def current_hashes():
        torch.cuda.synchronize()
        return dict(model=digest(stage.state_dict()),
                    optimizer=digest(optimizer.state_dict()),
                    scheduler=digest(scheduler.state_dict()))

    def export_state():
        torch.cuda.synchronize()
        step = manager.current_step()
        assert step == progress['updates'], (step, progress)
        payload = dict(model=cpu_tree(stage.state_dict()),
                       optimizer=cpu_tree(optimizer.state_dict()),
                       scheduler=scheduler.state_dict(), progress=dict(progress),
                       torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                       python_rng=random.getstate(), source_replica=args.replica,
                       source_rank=rank, step=step)
        payload['hashes'] = {key: digest(payload[key]) for key in ('model', 'optimizer', 'scheduler')}
        event('snapshot', step=step, hashes=payload['hashes'],
              optimizer_tensors=tensor_count(payload['optimizer'].get('state', {})))
        return payload

    def load_state(payload):
        assert payload['source_rank'] == rank
        assert payload['step'] == payload['progress']['updates']
        assert payload['step'] == manager.current_step()
        stage.load_state_dict(payload['model'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
        progress.update(payload['progress'])
        torch.set_rng_state(payload['torch_rng'])
        torch.cuda.set_rng_state(payload['cuda_rng'])
        random.setstate(payload['python_rng'])
        hashes = current_hashes()
        matches = hashes == payload['hashes']
        event('restored', step=payload['step'], source_replica=payload['source_replica'],
              source_rank=payload['source_rank'], hashes_match=matches,
              expected_hashes=payload['hashes'], actual_hashes=hashes,
              optimizer_tensors=tensor_count(optimizer.state_dict().get('state', {})))
        assert matches, 'Restored state differs from source snapshot'

    from torchft import Manager, ProcessGroupGloo
    from torchft.checkpointing import HTTPTransport

    class ReachableHTTPTransport(HTTPTransport):
        def address(self):
            return super().address().replace(socket.gethostname(), args.host_ip)

    class CompatibleProcessGroupGloo(ProcessGroupGloo):
        def _create_pg(self, store, rank, world_size):
            # NVIDIA's early 2.7 build lacks the optional flight-recorder
            # fields in Gloo options. Keep membership metadata on the wrapper,
            # but omit those tracing-only assignments while creating its PG.
            saved = self._global_ranks, self._group_rank
            try:
                self._global_ranks = self._group_rank = None
                return super()._create_pg(store, rank, world_size)
            finally:
                self._global_ranks, self._group_rank = saved

    manager = Manager(
        pg=CompatibleProcessGroupGloo(timeout=timedelta(seconds=10)),
        load_state_dict=load_state, state_dict=export_state,
        min_replica_size=1, use_async_quorum=False, init_sync=False,
        timeout=timedelta(seconds=90), quorum_timeout=timedelta(seconds=180),
        connect_timeout=timedelta(seconds=20),
        hostname=args.host_ip, lighthouse_addr=args.lighthouse,
        replica_id=f'shift{args.replica}',
        checkpoint_transport=ReachableHTTPTransport(timedelta(seconds=90), 0),
        max_retries=8)

    # The native finalizer waits intra-replica buckets, then sums tied PP grads.
    # Both dense and routed-expert buffers need the second reduction. In this
    # vendor, expert gradients use dense-DP normalization, retained unchanged.
    original_finish = ddp.finish_grad_sync
    buffers = list(ddp.buffers) + list(ddp.expert_parallel_buffers)
    cross_calls = [0]

    def finish_grad_sync(_self):
        original_finish()
        pending = []
        for buffer in buffers:
            for bucket in buffer.buckets:
                cpu = bucket.grad_data.detach().float().cpu()
                pending.append((bucket.grad_data, cpu, manager.allreduce(cpu)))
        for target, cpu, work in pending:
            work.wait()
            target.copy_(cpu)
        cross_calls[0] += len(pending)

    ddp.finish_grad_sync = types.MethodType(finish_grad_sync, ddp)
    initial = current_hashes()
    event('ready', physical_rank=args.replica * 8 + rank,
          layout='node_local_pp', tp=2, pp=2, ep=2, local_dense_dp=2,
          local_expert_dp=1, model_id=MODEL_ID, model_revision=REVISION,
          full_parameters=total_parameters, local_parameters=sum(v.numel() for v in stage.parameters()),
          tp_manifest=tp_manifest, hashes=initial, torch_version=torch.__version__,
          transport='HTTP_CPU', cross_gradient='Gloo_CPU_FP32')
    deadline = time.monotonic() + args.max_seconds
    while not (out / 'START').exists():
        if (out / 'STOP').exists() or time.monotonic() > deadline:
            raise RuntimeError('Stopped or timed out before START')
        time.sleep(0.1)
    schedule = get_forward_backward_func()
    attempts = 0
    try:
        while not (out / 'STOP').exists() and time.monotonic() < deadline:
            torch.cuda.synchronize()
            manager.start_quorum()  # Synchronous: apply healing BEFORE forward.
            step = manager.current_step()
            assert progress['updates'] == step
            event('begin', step=step, participants=manager.num_participants())
            optimizer.zero_grad(set_to_none=True)
            ddp.zero_grad_buffer()
            generator = torch.Generator(device='cuda').manual_seed(
                100_000 + step * 16 + args.replica * 2 + state.get_data_parallel_rank())
            batches = []
            for _ in range(2):
                tokens = torch.randint(0, stage.hf_config.vocab_size, (1, 17),
                                       device='cuda', generator=generator)
                batches.append(dict(input_ids=tokens[:, :-1].contiguous(),
                                    next_token_labels=tokens[:, 1:].contiguous()))
            started = time.monotonic()
            before_calls = cross_calls[0]
            metrics = schedule(forward_step_func=forward_step, data_iterator=iter(batches),
                               model=ddp, num_microbatches=2,
                               seq_length=16 + stage.extra_positions(1), micro_batch_size=1)
            for parameter in stage.parameters():
                parameter.grad = parameter.main_grad
            # A bad numeric result must never be counted as a valid update.
            finite = all(bool(torch.isfinite(b.grad_data).all()) for buf in buffers for b in buf.buckets)
            if not finite:
                manager.report_error(RuntimeError('nonfinite gradients'))
            commit = manager.should_commit()
            attempts += 1
            if commit:
                optimizer.step()
                scheduler.step()
                progress['updates'] += 1
                hashes = current_hashes()
                event('commit', step=manager.current_step(), updates=progress['updates'],
                      participants=manager.num_participants(), model_hash=hashes['model'],
                      optimizer_hash=hashes['optimizer'], scheduler_hash=hashes['scheduler'],
                      optimizer_tensors=tensor_count(optimizer.state_dict().get('state', {})),
                      cross_calls=cross_calls[0] - before_calls, finite_gradients=finite,
                      seconds=time.monotonic() - started,
                      loss=[float(m['loss']) for m in metrics] if metrics else [])
            else:
                event('skipped', step=step, error=str(manager.errored()))
            assert manager.current_step() == progress['updates']
        event('stopped', step=manager.current_step(), attempts=attempts)
    finally:
        # Peers may see STOP at slightly different iterations; the supervisor
        # performs bounded teardown of only its own process group.
        log.flush()


if __name__ == '__main__':
    main()
