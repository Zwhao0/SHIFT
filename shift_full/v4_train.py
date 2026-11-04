"""V4 boundary-rebuild baseline on the retained Qwen MoE and unmodified runtime.

One logical 8-rank model spans A(0..3)+B(4..7), then A+C. A preserves its
model, optimizer, DDP and parameter identities. All communication groups are
rebuilt at a drained update boundary. C overlaps imports/CPU model loading.
This is not yet TrainMover's sandbox, delta communicator setup, or DP isolation.
"""
import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import timedelta

from .v4_protocol import Ownership, Plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--role', choices=('store', 'host', 'A', 'B', 'C', 'D', 'E'), required=True)
    p.add_argument('--host-role', choices=('A', 'B', 'C', 'D', 'E'))
    p.add_argument('--device', type=int, default=0)
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text())
    out = Path(c['directory'])
    out.mkdir(parents=True, exist_ok=True)
    actor_prefix=f"g{c.get('initial_round',0)}_" if c.get('transactional_switch') and not c.get('reference_only') else ''
    if a.role == 'host':
        children, logs = [], []
        try:
            for device in range(4):
                log = (out / f'{actor_prefix}{a.host_role}{device}.log').open('wb')
                logs.append(log)
                children.append(subprocess.Popen([sys.executable, '-m', 'shift_full.v4_train', '--config', a.config,
                    '--role', a.host_role, '--device', str(device)], stdout=log, stderr=subprocess.STDOUT))
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError('owned worker failed')
                time.sleep(.2)
            if any(child.returncode for child in children):
                raise RuntimeError('owned worker failed')
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
            for log in logs:
                log.close()
        return

    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    timeout = timedelta(seconds=c.get('timeout', 120))
    store = dist.TCPStore(c['hosts']['A'], c['port'], None, a.role == 'store',
                          timeout=timeout, wait_for_workers=False)
    if a.role == 'store':
        outer_stores = [dist.TCPStore(c['hosts']['A'], c['port']+100+epoch*2+replica,
            None, True, timeout=timeout, wait_for_workers=False)
            for epoch in range(1 if c.get('reference_only') or c.get('retain_outer_managers') else 2) for replica in range(2)] if c.get('outer_manager') else []
        (out / 'store_ready').write_text(str(os.getpid()))
        until = time.monotonic() + c['seconds']
        while time.monotonic() < until and not (out / 'STOP_STORE').exists():
            time.sleep(.2)
        return

    base_store=store
    if c.get('transactional_switch'):
        store=dist.PrefixStore(f"round/{c.get('initial_round',0)}/",base_store)
    role, device = a.role, a.device
    initial_candidate=role==c.get('target_role','C')
    replica = int(role in ('D','E'))
    rank = Plan.logical_rank({'D':'A','E':'B'}.get(role,role), device, c.get('cross_node_tp',False))
    owner_role = 'A' if replica else role
    plan = Plan(**c['plan'])
    independent_reference = None
    if c.get('reference_file'):
        from .v4_reference import validate_signatures
        independent_reference = validate_signatures(json.loads(Path(c['reference_file']).read_text()))
    ownership = Ownership()
    log = (out / f'{actor_prefix}{role}{device}.jsonl').open('w', buffering=1)
    def event(kind, **fields):
        fields.setdefault('round',c.get('round_id',c.get('initial_round',0)))
        fields.setdefault('actor_round',c.get('initial_round',0))
        log.write(json.dumps(dict(kind=kind, role=role, rank=rank, device=device,
                                 replica=replica, pid=os.getpid(), time=time.time(), monotonic=time.monotonic(), **fields)) + '\n')

    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.12, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    sys.path.insert(0, c['flash'])
    from .compat import bootstrap_legacy_core
    bootstrap_legacy_core(c['flash'])
    from flash_v1 import megatron_adapter
    from flash_v1.pipeline_adapter import StageModel, forward_step
    from flash_v1.tensor_parallel import patch_tensor_parallel
    from flash_v1.train import load_model, patch_moe, MODEL_ID, REVISION
    from .train import digest
    from .compat import BlockingNCCL
    from .transport import PackedGPUTransport
    from torch.utils._pytree import tree_map
    from .state_inventory import restore_buffers, restore_tracker, tracker_snapshot
    state, _, _ = megatron_adapter.core()
    local_cache = None
    local_handles = None
    if c.get('reuse_local_groups') and not replica and (role != 'B' or c.get('defer_source_local_close')):
        from torch.distributed import distributed_c10d as c10d
        from .v4_groups import LocalGroupCache, validate_runtime
        validate_runtime(c10d, state, megatron_adapter)
        local_cache = LocalGroupCache(dist, c10d, (state.__file__, megatron_adapter.__file__))
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    tracker = get_cuda_rng_tracker()
    os.environ['FLASH_PARALLEL_LAYOUT'] = 'legacy'

    def connect(epoch, local=False):
        nonlocal local_handles
        started = time.monotonic()
        dist.init_process_group('nccl', store=dist.PrefixStore(f'v4/replica{replica}/{epoch}/', store),
                                rank=device if local else rank, world_size=4 if local else 8, timeout=timeout)
        directory = out / f'topology_{epoch}_{role}'
        directory.mkdir(exist_ok=True)
        with local_cache.creating(reuse=isinstance(epoch,int) and epoch>=1) if local_cache else nullcontext():
            groups = megatron_adapter.initialize_groups(directory, 2, 1 if local else 2, 2)
        assert state.get_pipeline_model_parallel_rank() == (0 if local else rank // 4)
        assert dist.get_world_size(groups[0]) == 2
        assert dist.get_world_size(groups[1]) == 1
        assert dist.get_world_size(groups[2]) == 2
        if local_cache:
            handles = (state.get_tensor_model_parallel_group(), *groups)
            if isinstance(epoch,int) and epoch>=1:
                if any(old is not new for old,new in zip(local_handles,handles)):
                    raise RuntimeError('local TP/EP/expert-DP/dense group identity changed')
                event('local_groups_reused', identity_preserved=True, **local_cache.finish(retain_catalog=bool(c.get('transactional_switch'))))
            else:
                local_handles = handles
        event('connected', epoch=epoch, seconds=time.monotonic()-started,
              pp=dist.get_process_group_ranks(state.get_pipeline_model_parallel_group()),
              tp=dist.get_process_group_ranks(state.get_tensor_model_parallel_group()),
              ep=dist.get_process_group_ranks(groups[0]), dense=dist.get_process_group_ranks(groups[2]))
        return groups

    def build(full, groups, pp_rank):
        ep, expert_dp, dense = groups
        patch_moe(full, ep)
        stage = StageModel(full, pp_rank)
        patch_tensor_parallel(stage, state.get_tensor_model_parallel_group())
        stage.cuda()
        ddp = megatron_adapter.wrap(stage, dense, expert_dp, 67_108_864, config=stage.config)
        stage.config.no_sync_func = ddp.no_sync
        stage.config.finalize_model_grads_func = finalize_model_grads
        optimizer = torch.optim.AdamW(stage.parameters(), lr=1e-4, foreach=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=.98)
        return stage, ddp, optimizer, scheduler

    def rebind(stage, ddp, old_groups, new_groups):
        replacements = {id(old):new for old,new in zip(old_groups,new_groups)}
        changed = []
        owners = [('ddp', ddp)] + list(stage.named_modules())
        for label, buffers in [('dense', ddp.buffers), ('expert', ddp.expert_parallel_buffers)]:
            for i, buffer in enumerate(buffers):
                owners.append((f'{label}.buffer{i}', buffer))
                owners.extend((f'{label}.buffer{i}.bucket{j}', b) for j,b in enumerate(buffer.buckets))
        for name, owner in owners:
            for key, value in vars(owner).copy().items():
                if id(value) in replacements:
                    setattr(owner, key, replacements[id(value)]); changed.append(name+'.'+key)
        return changed

    def disconnect(preserve=False, abort=False):
        started = time.monotonic()
        torch.cuda.synchronize()
        if not abort:dist.barrier()
        retained = local_cache.detach() if preserve else 0
        if abort:
            from torch.distributed import distributed_c10d as c10d
            for group in list(c10d._world.pg_names):
                try:backend=group._get_backend(torch.device('cuda'))
                except RuntimeError:backend=group._get_backend(torch.device('cpu'))
                if hasattr(backend,'abort'):backend.abort()
        state.destroy_model_parallel()
        dist.destroy_process_group()
        event('groups_closed', seconds=time.monotonic()-started, retained=retained,aborted=abort)

    state_channel = None
    state_thread = None
    state_errors = []
    def prepare_state_channel():
        nonlocal state_channel
        channel = None
        try:
            torch.cuda.set_device(device)
            # Separate client: a wait must not monopolize the training store.
            control = dist.TCPStore(c['hosts']['A'], c['port'], None, False,
                                    timeout=timeout, wait_for_workers=False)
            if role == 'B':
                control.wait([f'channel_start/{device}'])
            else:
                control.set(f'channel_start/{device}', '1')
            started = time.monotonic()
            event('state_channel_preparing')
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                channel = BlockingNCCL(timeout=timeout)
                channel.configure(f"{c['hosts']['A']}:{c['port']}/v4-state/{device}",
                                  'v4-state', 0 if role == 'B' else 1, 2)
                value = torch.ones(1, device='cuda')
                channel.allreduce([value], dist.AllreduceOptions()).wait(timeout)
                stream.synchronize()
                if value.item() != 2:
                    raise RuntimeError('state channel warmup mismatch')
            state_channel = channel
            event('state_channel_prepared', seconds=time.monotonic()-started)
            if role == 'B':
                control.set(f'channel_ready/{device}', '1')
            else:
                control.wait([f'channel_ready/{device}'])
        except BaseException as error:
            if channel is not None:
                channel.shutdown()
            state_errors.append(error)

    def check_state_channel():
        if state_errors:
            raise RuntimeError('state channel preparation failed') from state_errors[0]

    def transfer(packet=None):
        nonlocal state_channel
        # Independent pair PGs never include retained A or the training groups.
        started = time.monotonic()
        if state_thread:
            state_thread.join(timeout.total_seconds())
            if state_thread.is_alive():
                raise TimeoutError('state channel preparation did not finish')
        check_state_channel()
        channel = state_channel or BlockingNCCL(timeout=timeout)
        stats = {}
        def transport_event(kind, **fields):
            if kind in ('transport_send', 'transport_recv'):
                stats.update(fields)
            event(kind, **fields)
        try:
            if state_channel is None:
                channel.configure(f"{c['hosts']['A']}:{c['port']}/v4-state/{device}",
                                  'v4-state', 0 if role == 'B' else 1, 2)
                event('state_channel_connected', seconds=time.monotonic()-started)
            transport = PackedGPUTransport(channel, torch.device('cuda', device), timeout, transport_event)
            if role == 'B':
                transport.send_checkpoint([1], plan.cut, packet, timeout)
            else:
                packet = transport.recv_checkpoint(0, transport.metadata(), plan.cut, timeout)
            event('state_sent' if role == 'B' else 'state_received',
                  bytes=stats['gpu_bytes']+stats['control_bytes'],
                  gpu_bytes=stats['gpu_bytes'], control_bytes=stats['control_bytes'],
                  cpu_tensor_bytes=stats['cpu_tensor_bytes'],
                  hashes=packet['anchor']['hashes'],
                  oracle_hashes=packet['oracle']['hashes'] if packet['oracle'] is not None else None,
                  seconds=time.monotonic()-started, transport='packed_gpu')
            return packet
        finally:
            channel.shutdown()
            state_channel = None

    # The joiner imports independently; its CPU model load starts while AB train.
    prepared_stage = None
    metadata_handles=None
    if initial_candidate:
        event('standby_imported', model_id=MODEL_ID, revision=REVISION)
        store.wait(['prepare'])
        store.set(f'preparing/{device}', '1')
        started = time.monotonic()
        torch.manual_seed(c['seed']); random.seed(c['seed'])
        full = load_model(Path(c['flash']) / 'models/qwen_moe')
        event('standby_cpu_model_ready', seconds=time.monotonic()-started)
        full_parameters = sum(t.numel() for t in full.parameters())
        if c.get('cross_node_tp'):
            from .v4_metadata import metadata_groups
            with metadata_groups(dist,state,rank) as metadata_handles:
                local_groups=(metadata_handles['ep'],metadata_handles['expert_dp'],metadata_handles['dense'])
                prepared_stage=build(full,local_groups,rank//4)
            del full
            prepared_old=(metadata_handles['tp'],local_groups[0],local_groups[2],local_groups[1])
            event('standby_metadata_model_ready',allocated=torch.cuda.memory_allocated(),
                  scope='constructor metadata only; communication forbidden; no shadow training')
        elif c.get('gpu_prepare'):
            local_groups = connect('standby', local=True)
            # Construct the eventual PP1 DDP bucket policy on the isolated C group.
            state.set_pipeline_model_parallel_rank(1)
            state.set_pipeline_model_parallel_world_size(2)
            prepared_stage = build(full, local_groups, 1)
            del full
            prepared_old = (state.get_tensor_model_parallel_group(),local_groups[0],local_groups[2],local_groups[1])
            if c.get('gpu_warmup'):
                started = time.monotonic()
                warm_stage, warm_ddp, warm_opt, _ = prepared_stage
                warm_opt.zero_grad(set_to_none=True); warm_ddp.zero_grad_buffer()
                # This adapter-specific PP1 rehearsal uses synthetic upstream
                # activations. TP/EP exchanges remain wholly inside standby C.
                size = plan.seq_len + warm_stage.extra_positions(1)
                activation = torch.randn(size,1,warm_stage.hf_config.hidden_size,device='cuda')
                activation[plan.seq_len:] = 0
                activation.requires_grad_(True)
                warm_stage.set_input_tensor(activation)
                tokens = (torch.arange(plan.seq_len+1,device='cuda').view(1,-1)%32+100)
                logits, aux = warm_ddp(tokens[:,:-1].contiguous())
                loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),
                    tokens[:,1:].reshape(-1)) + warm_stage.hf_config.router_aux_loss_coef*aux
                loss.backward(); warm_ddp.finish_grad_sync()
                for parameter in warm_stage.parameters(): parameter.grad=parameter.main_grad
                warm_opt.step()
                warm_opt.zero_grad(set_to_none=True); warm_ddp.zero_grad_buffer()
                warm_stage.input_tensor = None
                del activation, logits, aux, loss, tokens
                torch.cuda.synchronize()
                event('standby_gpu_warmed',seconds=time.monotonic()-started,
                      scope='synthetic PP1 forward/backward/Adam; standby-local TP/EP',
                      allocated=torch.cuda.memory_allocated(),peak=torch.cuda.max_memory_allocated())
            torch.cuda.synchronize()
            event('standby_gpu_model_ready', allocated=torch.cuda.memory_allocated(),
                  training_iteration_warmup=bool(c.get('gpu_warmup')))
        if c.get('transactional_switch'):
            if not prepared_stage:raise ValueError('transactional candidate requires isolated preparation')
            groups=local_groups
            anchor=oracle=None
        else:
            if c.get('preconnect_state'):
                prepare_state_channel()
                check_state_channel()
            store.set(f'prepared/{device}', '1')
            store.wait(['switch'])
            plan = replace(plan, cut=int(store.get('actual_cut')))
            if prepared_stage:
                disconnect(preserve=local_cache is not None)
            received = transfer()
            anchor, oracle = received['anchor'], received.get('oracle')
            del received
            plan.admit(anchor, rank)
            groups = connect(1)
    else:
        groups = connect(0)
        torch.manual_seed(c['seed']); random.seed(c['seed'])
        full = load_model(Path(c['flash']) / 'models/qwen_moe')
        anchor = oracle = None

    ep, expert_dp, dense = groups
    if prepared_stage:
        stage, ddp, optimizer, scheduler = prepared_stage
        changed = [] if metadata_handles else rebind(stage,ddp,prepared_old,(state.get_tensor_model_parallel_group(),ep,dense,expert_dp))
        event('standby_rebound',attributes=changed)
    else:
        full_parameters = sum(t.numel() for t in full.parameters())
        stage, ddp, optimizer, scheduler = build(full, groups, state.get_pipeline_model_parallel_rank())
        del full
    progress = {'updates': 0}
    outer = None
    def rng():
        return dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    python_rng=random.getstate(), tracker_rng=tracker_snapshot(tracker))
    def restore_rng(value):
        torch.set_rng_state(value['torch_rng']); torch.cuda.set_rng_state(value['cuda_rng'])
        random.setstate(value['python_rng']); restore_tracker(tracker, value['tracker_rng'])
    def hashes():
        r = rng()
        return dict(model=digest(stage.state_dict()), optimizer=digest(optimizer.state_dict()),
                    scheduler=digest(scheduler.state_dict()), progress=digest(progress),
                    buffers=digest(dict(stage.named_buffers())), cpu_rng=digest(r['torch_rng']),
                    cuda_rng=digest(r['cuda_rng']), python_rng=digest(r['python_rng']),
                    tracker_rng=digest(r['tracker_rng']))
    def snapshot():
        started = time.monotonic()
        torch.cuda.synchronize()
        value = tree_map(lambda t: t.detach().clone() if isinstance(t, torch.Tensor) else t,
            dict(schema=2, model=stage.state_dict(), all_buffers=dict(stage.named_buffers()),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), progress=dict(progress),
            **rng(), source_replica=replica, source_rank=rank, step=progress['updates'], v4_epoch=1, v4_owner='C',
            outer_manager=outer.state_dict() if outer else None))
        value['hashes'] = hashes()
        torch.cuda.synchronize()
        event('snapshot_ready', step=progress['updates'], seconds=time.monotonic()-started,
              allocated=torch.cuda.memory_allocated())
        return value
    def restore(value):
        started = time.monotonic()
        stage.load_state_dict(value['model'], strict=True)
        restore_buffers(stage, value['all_buffers'])
        # Adam's same-device load can alias checkpoint tensors; keep snapshots
        # independent of future optimizer updates and release packed storage.
        optimizer.load_state_dict(tree_map(
            lambda t: t.detach().clone() if isinstance(t, torch.Tensor) else t,
            value['optimizer']))
        scheduler.load_state_dict(value['scheduler'])
        progress.clear(); progress.update(value['progress'])
        restore_rng(value)
        optimizer.zero_grad(set_to_none=True); ddp.zero_grad_buffer()
        torch.cuda.synchronize()
        actual = hashes()
        if actual != value['hashes']:
            raise RuntimeError('complete state restoration mismatch: '+str([k for k in actual if actual[k]!=value['hashes'][k]]))
        event('restored', step=progress['updates'], hashes_match=True, hashes=actual,
              seconds=time.monotonic()-started)
    identity = dict(model=id(stage), optimizer=id(optimizer), ddp=id(ddp),
                    parameter_ids=[id(t) for t in stage.parameters()], pointers=[t.data_ptr() for t in stage.parameters()])
    def identity_preserved():
        return identity == dict(model=id(stage), optimizer=id(optimizer), ddp=id(ddp),
            parameter_ids=[id(t) for t in stage.parameters()], pointers=[t.data_ptr() for t in stage.parameters()])
    def training_identity():
        return dict(model=id(stage),optimizer=id(optimizer),ddp=id(ddp),
            parameters=[(id(t),t.data_ptr()) for t in stage.parameters()],
            adam=[(id(t),t.data_ptr()) for state_value in optimizer.state.values()
                  for t in state_value.values() if isinstance(t,torch.Tensor)],
            buckets=[(id(b),b.grad_data.data_ptr()) for buf in ddp.buffers+ddp.expert_parallel_buffers for b in buf.buckets])
    comparison_start_step=anchor['step'] if anchor is not None else 0
    reference_trajectory = {}
    trajectory_exact = []
    current_phase=None
    native_finish = ddp.finish_grad_sync
    def finish_grad_sync():
        native_finish()
        if c.get('overlap_peer') and replica and current_phase=='after' and progress['updates']==plan.cut:
            torch.cuda.synchronize()
            store.set(f'peer_gradient_ready/{rank}','1')
            event('peer_native_gradient_ready',step=progress['updates'])
        outer_started=time.monotonic()
        if outer:
            outer.reduce([b for buf in ddp.buffers+ddp.expert_parallel_buffers for b in buf.buckets],
                         progress['updates'])
        if c.get('overlap_peer') and replica and current_phase=='after' and progress['updates']==plan.cut:
            event('peer_outer_gradient_ready',step=progress['updates'],seconds=time.monotonic()-outer_started)
    ddp.finish_grad_sync = finish_grad_sync
    def train(epoch, count, phase):
        nonlocal current_phase
        current_phase=phase
        schedule = get_forward_backward_func()
        previous = None
        for _ in range(count):
            check_state_channel()
            ownership.authorize(owner_role, epoch)
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True); ddp.zero_grad_buffer()
            step = progress['updates']
            if outer:
                saved_rng = rng()
                outer.begin(step)
                restore_rng(saved_rng)
            if role == 'B' and phase == 'before' and step == plan.cut-1 and c.get('slow_ms',0):
                event('planned_slow',step=step,delay_ms=c['slow_ms'])
                time.sleep(c['slow_ms']/1000)
            batches, sample_ids = [], []
            for micro in range(plan.microbatches):
                width = 4 if c.get('outer_manager') else 2
                sample = step * plan.microbatches * width + micro * width + replica*2 + state.get_data_parallel_rank()
                sample_ids.append(sample)
                tokens = (torch.arange(plan.seq_len+1, device='cuda').view(1,-1) + sample % 8) % 32 + 100
                batches.append(dict(input_ids=tokens[:,:-1].contiguous(), next_token_labels=tokens[:,1:].contiguous()))
            if c.get('overlap_peer') and replica and phase=='after' and step==plan.cut:
                store.set(f'peer_compute/{rank}','1')
                event('peer_compute_begin',step=step)
            metrics = schedule(forward_step_func=forward_step, data_iterator=iter(batches), model=ddp,
                num_microbatches=plan.microbatches, seq_length=plan.seq_len+stage.extra_positions(1), micro_batch_size=1)
            for parameter in stage.parameters():
                parameter.grad = parameter.main_grad
            buffers = ddp.buffers + ddp.expert_parallel_buffers
            finite = bool(torch.stack([torch.isfinite(b.grad_data).all() for buf in buffers for b in buf.buckets]).all())
            if not finite:
                raise RuntimeError('nonfinite gradients')
            ownership.authorize(owner_role, epoch)
            if outer:
                outer.commit(step)
            optimizer.step(); scheduler.step(); torch.cuda.synchronize()
            progress['updates'] += 1
            step_hashes = hashes() if (c.get('reference_only') or independent_reference or
                (c['oracle'] and phase in ('oracle','after'))) else None
            if c.get('reference_only'):
                event('reference_step',step=progress['updates'],hashes=step_hashes,sample_ids=sample_ids,
                      outer_manager=outer.state_dict())
            if independent_reference:
                from .v4_reference import compare_signature
                exact=compare_signature(independent_reference,replica=replica,rank=rank,
                    step=progress['updates'],hashes=step_hashes,samples=sample_ids,manager=outer.state_dict())
                trajectory_exact.append(exact)
                event('reference_comparison',step=progress['updates'],phase=phase,exact=exact,
                      hashes=step_hashes,sample_ids=sample_ids,outer_manager=outer.state_dict())
                if not exact:raise RuntimeError('independent reference mismatch')
            if phase == 'oracle':
                reference_trajectory[str(progress['updates'])] = dict(hashes=step_hashes,sample_ids=sample_ids,
                    outer_manager=outer.state_dict() if outer else None)
            elif phase == 'after' and oracle is not None:
                expected = oracle['trajectory'][str(progress['updates'])]
                exact = (step_hashes == expected['hashes'] and sample_ids == expected['sample_ids'] and
                         (outer.state_dict() if outer else None) == expected.get('outer_manager'))
                trajectory_exact.append(exact)
                event('oracle_step',step=progress['updates'],exact=exact)
            committed = time.monotonic()
            event('commit', epoch=epoch, phase=phase, step=progress['updates'], sample_ids=sample_ids,
                  seconds=committed-started, commit_gap=committed-previous if previous else None,
                  finite=True, global_activation_observed=store.check(['activated']),
                  loss=[float(v['loss']) for v in metrics] if metrics else [],
                  standby_preparing=store.check([f'preparing/{i}' for i in range(4)]),
                  standby_ready=store.check([f'prepared/{i}' for i in range(4)]),
                  allocated=torch.cuda.memory_allocated(), peak=torch.cuda.max_memory_allocated())
            previous = committed
            if role == 'A' and device == 0 and progress['updates'] == 1 and phase == 'before':
                store.set('prepare', '1')

    def finish():
        actual = snapshot()
        comparison = None
        if oracle is not None:
            comparison = dict(exact={k: actual['hashes'][k] == oracle['hashes'][k] for k in actual['hashes']}, tensors={})
            comparison['trajectory_exact'] = len(trajectory_exact)==plan.post_steps and all(trajectory_exact)
            comparison['manager_exact'] = actual['outer_manager'] == oracle['outer_manager']
            for section in ('model', 'optimizer', 'all_buffers'):
                total, unequal, max_abs = 0, 0, 0.0
                def compare(x, y):
                    nonlocal total, unequal, max_abs
                    if isinstance(x, torch.Tensor):
                        assert x.shape == y.shape and x.dtype == y.dtype
                        total += x.numel(); unequal += int((x != y).sum())
                        if x.numel():
                            max_abs = max(max_abs, float((x.to(torch.float64)-y.to(torch.float64)).abs().max()))
                    elif isinstance(x, dict):
                        assert x.keys() == y.keys()
                        for key in x: compare(x[key], y[key])
                    elif isinstance(x, (tuple, list)):
                        assert len(x) == len(y)
                        for xx, yy in zip(x,y): compare(xx,yy)
                    elif x != y:
                        raise RuntimeError('non-tensor oracle metadata mismatch')
                compare(actual[section], oracle[section])
                comparison['tensors'][section] = dict(elements=total, unequal=unequal, max_abs=max_abs)
        if independent_reference:
            expected=independent_reference['workers'][f'{replica}/{rank}'][str(progress['updates'])]
            comparison=dict(exact={k:actual['hashes'][k]==expected['hashes'][k] for k in actual['hashes']},
                trajectory_exact=len(trajectory_exact)==progress['updates']-comparison_start_step and all(trajectory_exact),
                manager_exact=actual['outer_manager']==expected['outer_manager'],tensors={},
                method='independent_nine_state_sha256')
        event('done', step=progress['updates'], comparison=comparison,
              identity_preserved=identity_preserved(), hashes=actual['hashes'])
        if outer:
            outer.close()
        disconnect()
        log.close()

    event('model_ready', full_parameters=full_parameters, local_parameters=sum(t.numel() for t in stage.parameters()),
          model_id=MODEL_ID, revision=REVISION)
    def configure_runtime(*,outer_value=...,plan_value=...,ownership_value=...,store_value=...,role_value=...,comparison_start=...):
        nonlocal outer,plan,ownership,store,owner_role,comparison_start_step
        if outer_value is not ...:outer=outer_value
        if plan_value is not ...:plan=plan_value
        if ownership_value is not ...:ownership=ownership_value
        if store_value is not ...:store=store_value
        if role_value is not ...:owner_role=role_value
        if comparison_start is not ...:comparison_start_step=comparison_start
    if c.get('transactional_switch') and not c.get('reference_only'):
        from .v4_transaction_train import run
        context=locals().copy()
        run(context)
        return
    if c.get('outer_manager') and role != 'C':
        from .v4_outer import Outer
        saved_rng = rng()
        outer = Outer(c,replica,rank,0,event)
        restore_rng(saved_rng)
    if c.get('reference_only') or (independent_reference and role != 'C'):
        initial_hashes=hashes()
        event('reference_step' if c.get('reference_only') else 'reference_initial',step=0,
              hashes=initial_hashes,sample_ids=[],outer_manager=outer.state_dict())
        if independent_reference:
            from .v4_reference import compare_signature
            if not compare_signature(independent_reference,replica=replica,rank=rank,step=0,
                    hashes=initial_hashes,samples=[],manager=outer.state_dict()):
                raise RuntimeError('independent initial state mismatch')
    if c.get('reference_only'):
        train(0,plan.cut+plan.post_steps,'reference')
        event('done',step=progress['updates'],comparison=None,identity_preserved=identity_preserved(),hashes=hashes())
        outer.close(); disconnect(); log.close()
        return
    if role == 'B' and c.get('preconnect_state'):
        state_thread = threading.Thread(target=prepare_state_channel, daemon=True)
        state_thread.start()
    if role != 'C':
        train(0, plan.cut, 'before')
        # Keep training while C prepares. Only rank 0 selects a common boundary;
        # a ready flag is broadcast after each completed, fully drained step.
        while True:
            ready = False
            if rank == 0:
                key = f'boundary_ready/{progress["updates"]}'
                if not replica:
                    ready = store.check([f'prepared/{i}' for i in range(4)])
                    if c.get('outer_manager'):
                        store.set(key,str(int(ready)))
                else:
                    ready = bool(int(store.get(key)))
            flag = torch.tensor(int(ready),device='cuda')
            dist.broadcast(flag,src=0)
            if flag.item():break
            train(0, 1, 'before')
        requested_cut = plan.cut
        plan = replace(plan, cut=progress['updates'])
        if rank == 0 and not replica:store.set('actual_cut',str(plan.cut))
        event('boundary_selected',requested_cut=requested_cut,step=plan.cut,standby_ready=True)
        event('cut_boundary',step=plan.cut)
        # Oracle replay is a diagnostic and is excluded from performance claims.
        anchor = snapshot()
        if c.get('overlap_peer') and replica:
            boundary_outer_identity=outer.identity()
            boundary_training_identity=training_identity()
            event('outer_retention_boundary',identity=boundary_outer_identity,training_identity=boundary_training_identity,
                  step=plan.cut,manager_state=outer.state_dict())
            event('freeze',step=plan.cut,hashes=hashes(),identity_preserved=identity_preserved())
            store.set(f'frozen/{replica}/{rank}','1')
            ownership.advance()
            outer.migrate(1)
            restore_rng(anchor)
            if outer.identity()!=boundary_outer_identity or training_identity()!=boundary_training_identity:
                raise RuntimeError('healthy peer Manager/training storage identity changed')
            event('outer_retained',step=plan.cut,identity=outer.identity(),training_identity=training_identity(),
                  manager_state=outer.state_dict(),channel_reconnected=rank>=4)
            event('peer_inner_groups_retained',identity_preserved=True)
            event('peer_compute_ready',step=plan.cut,hashes=hashes(),identity_preserved=identity_preserved())
            store.set(f'active/{replica}/{rank}','1')
            del anchor
            gc.collect();torch.cuda.empty_cache()
            train(1,plan.post_steps,'after')
            finish()
            return
        if c['oracle'] and not independent_reference:
            if outer:raise RuntimeError('live outer Manager reference rewind is forbidden')
            train(0, plan.post_steps, 'oracle')
            oracle = snapshot()
            oracle['trajectory'] = reference_trajectory
            if outer:
                dist.barrier(); outer.close(); outer = None
            restore(anchor)
        elif outer and (not c.get('retain_outer_managers') or role == 'B'):
            dist.barrier(); outer.close(); outer = None
        elif outer:
            dist.barrier()
            boundary_outer_identity = outer.identity()
            boundary_training_identity = training_identity()
            event('outer_retention_boundary',identity=boundary_outer_identity,training_identity=boundary_training_identity,
                  step=plan.cut,manager_state=outer.state_dict())
        store.wait([f'prepared/{i}' for i in range(4)])
        dist.barrier(); torch.cuda.synchronize()
        event('freeze', step=plan.cut, hashes=hashes(), identity_preserved=identity_preserved())
        if c.get('outer_manager'):
            store.set(f'frozen/{replica}/{rank}', '1')
        if rank == 0 and not replica:
            if c.get('outer_manager'):
                store.wait([f'frozen/{r}/{i}' for r in range(2) for i in range(8)])
            store.set('switch', '1')
        old_tp, old_ep = state.get_tensor_model_parallel_group(), ep
        old_dense, old_expert = dense, expert_dp
        if not replica:
            disconnect(preserve=local_cache is not None)
        else:
            store.wait(['switch'])
        ownership.advance()
        event('old_groups_closed', step=plan.cut)
        try:
            ownership.authorize(owner_role,0)
            raise AssertionError('old epoch revived')
        except ValueError:
            event('old_epoch_rejected',epoch=0,step=plan.cut)
        if role == 'B':
            # B is fenced before C may become active, and never runs another step.
            try:
                ownership.authorize('B', 1)
                raise AssertionError('retired owner admitted')
            except ValueError:
                event('old_owner_rejected', step=plan.cut, epoch=1)
            transfer(dict(anchor=anchor, oracle=oracle))
            del anchor, oracle
            gc.collect(); torch.cuda.empty_cache()
            store.wait(['activated'])
            if local_cache:
                event('activation_observed', step=plan.cut)
                started = time.monotonic()
                counts = local_cache.finish()
                event('source_local_groups_closed', seconds=time.monotonic()-started, **counts)
            event('retired', step=plan.cut)
            return
        if not replica:
            ep, expert_dp, dense = connect(1)
            changed = rebind(stage,ddp,(old_tp,old_ep,old_dense,old_expert),
                             (state.get_tensor_model_parallel_group(),ep,dense,expert_dp))
        else:
            changed = []
            if (old_tp,old_ep,old_dense,old_expert) != (state.get_tensor_model_parallel_group(),ep,dense,expert_dp):
                raise RuntimeError('peer replica inner groups changed')
            event('peer_inner_groups_retained',identity_preserved=True)
        if not identity_preserved():
            raise RuntimeError('healthy model/optimizer/DDP identity changed')
        # Group construction must not change the retained training RNG state.
        if c.get('retain_outer_managers'):
            restore_rng(anchor)
            if hashes()!=anchor['hashes'] or training_identity()!=boundary_training_identity:
                raise RuntimeError('retained training state/storage changed')
        else:
            restore(anchor)
        event('rebound', attributes=changed, identity_preserved=True)
    else:
        if c.get('fail_restore_rank',-1) == rank:
            event('restore_failure_injected',step=plan.cut)
            raise RuntimeError('injected pre-activation restore failure')
        restore(anchor)
        ownership.advance()

    if c.get('outer_manager'):
        from .v4_outer import Outer
        if c.get('retain_outer_managers') and role != 'C':
            outer.migrate(1)
            if outer.identity()!=boundary_outer_identity or outer.state_dict()!=anchor['outer_manager']:
                raise RuntimeError('retained Manager identity/progress changed')
            event('outer_retained',step=plan.cut,identity=outer.identity(),training_identity=training_identity(),
                  manager_state=outer.state_dict(),channel_reconnected=rank>=4)
        else:
            outer = Outer(c,replica,rank,1,event)
            outer.load_state_dict(anchor['outer_manager'])
        restore_rng(anchor)
    del anchor
    gc.collect(); torch.cuda.empty_cache()
    dist.barrier()
    active_fields=dict(epoch=1, step=plan.cut, hashes=hashes(), identity_preserved=identity_preserved(),
        peer_compute_started=store.check([f'peer_compute/{i}' for i in range(8)]),
        peer_gradients_ready=store.check([f'peer_gradient_ready/{i}' for i in range(8)]))
    event('activation_ready' if c.get('outer_manager') else 'activated', **active_fields)
    if c.get('outer_manager'):
        store.set(f'active/{replica}/{rank}', '1')
    if rank == 0 and not replica:
        if c.get('outer_manager'):
            store.wait([f'active/{r}/{i}' for r in range(2) for i in range(8)])
        store.set('activated', '1')
    if c.get('outer_manager'):
        store.wait(['activated'])
        event('activated', **active_fields)
    train(1, plan.post_steps, 'after')
    finish()


if __name__ == '__main__':
    main()
