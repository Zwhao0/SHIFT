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
from .session_layout import SessionLayout
from .training_core import (TrainingState, partition_model, configure_model, sample_plan,
    make_batches, train_step, complete_step, initialize_rng)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--role', choices=('store', 'host', 'A', 'B', 'C', 'D', 'E'), required=True)
    p.add_argument('--host-role', choices=('A', 'B', 'C', 'D', 'E'))
    p.add_argument('--device', type=int, default=0)
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text())
    if c.get('deterministic'):os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    out = Path(c['directory'])
    out.mkdir(parents=True, exist_ok=True)
    actor_prefix=f"g{c.get('actor_generation',c.get('initial_round',0))}_" if c.get('transactional_switch') and not c.get('reference_only') else ''
    if a.role == 'host':
        children, logs = [], []
        try:
            if c.get('flash_live'):
                layout = SessionLayout.from_config(c)
                observations = out / 'flash_native' / f'{actor_prefix}{a.host_role}'
                observations.mkdir(parents=True, exist_ok=True)
            for device in range(c.get('actor_devices',4)):
                log = (out / f'{actor_prefix}{a.host_role}{device}.log').open('wb')
                logs.append(log)
                env = None
                if c.get('flash_live'):
                    ranks = c.get('session_actor_ranks')
                    rank = ranks[device] if ranks else layout.rank(
                        a.host_role, device, c.get('candidate_full_dp', False))
                    env = dict(os.environ, RANK=str(c['session_replica'] * layout.world + rank),
                               FLASH_V4_OUTPUT_DIR=str(observations))
                children.append(subprocess.Popen([sys.executable, '-m', 'shift_full.v4_train', '--config', a.config,
                    '--role', a.host_role, '--device', str(device)], stdout=log, stderr=subprocess.STDOUT, env=env))
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

    worker_import_started=time.monotonic()
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    timeout = timedelta(seconds=c.get('timeout', 120))
    store = dist.TCPStore(c['hosts']['A'], c['port'], None, a.role == 'store',
                          timeout=timeout, wait_for_workers=False)
    if a.role == 'store':
        outer_stores = [dist.TCPStore(c['hosts']['A'], c['port']+100+epoch*2+replica,
            None, True, timeout=timeout, wait_for_workers=False)
            for epoch in range(c.get('session_services',1 if c.get('reference_only') or c.get('retain_outer_managers') else 2)) for replica in range(2)] if c.get('outer_manager') else []
        (out / 'store_ready').write_text(str(os.getpid()))
        until = time.monotonic() + c['seconds']
        while time.monotonic() < until and not (out / 'STOP_STORE').exists():
            time.sleep(.2)
        return

    base_store=store
    if c.get('transactional_switch'):
        store=dist.PrefixStore(f"round/{c.get('initial_round',0)}/",base_store)
    layout=SessionLayout.from_config(c)
    role, device = a.role, a.device
    initial_candidate=c.get('session_candidate',role==c.get('target_role','C'))
    replica = c.get('session_replica',int(role in ('D','E')))
    rank = Plan.logical_rank({'D':'A','E':'B'}.get(role,role),device,True) if c.get('cross_node_tp') else layout.rank(role,device,c.get('candidate_full_dp',False))
    if 'session_actor_ranks' in c:
        scope = layout.partial_scope(replica, c['session_actor_ranks'])
        rank = scope.ranks[device]
    owner_role = 'A' if replica else role
    plan = Plan(**c['plan'])
    from .v4_state import boundary_validation
    boundary_mode=boundary_validation(c)
    audit_boundary=boundary_mode=='full'
    final_reference=None
    if c.get('final_reference_file'):
        from .v4_reference import validate_signatures
        final_reference=validate_signatures(json.loads(Path(c['final_reference_file']).read_text()))
    independent_reference = None
    if c.get('reference_file'):
        from .v4_reference import validate_signatures
        independent_reference = validate_signatures(json.loads(Path(c['reference_file']).read_text()))
    ownership = Ownership()
    log = (out / f'{actor_prefix}{role}{device}.jsonl').open('w', buffering=1)
    def event(kind, **fields):
        fields.setdefault('round',c.get('round_id',c.get('initial_round',0)))
        fields.setdefault('actor_round',c.get('actor_generation',c.get('initial_round',0)))
        log.write(json.dumps(dict(kind=kind, role=role, rank=rank, device=device,
                                 replica=replica, pid=os.getpid(), time=time.time(), monotonic=time.monotonic(), **fields)) + '\n')

    event('worker_imported',seconds=time.monotonic()-worker_import_started)
    torch.cuda.set_device(device)
    if 'session_gpu_uuids' in c:
        gpu_uuid = str(torch.cuda.get_device_properties(device).uuid)
        if gpu_uuid.removeprefix('GPU-') != c['session_gpu_uuids'][device].removeprefix('GPU-'):
            raise RuntimeError('training worker received a different GPU than its reservation')
        event('worker_device_bound', gpu_uuid=gpu_uuid)
    torch.cuda.set_per_process_memory_fraction(.12, device)
    if c.get('deterministic'):torch.use_deterministic_algorithms(True)
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
        local_cache = LocalGroupCache(dist, c10d, (state.__file__, megatron_adapter.__file__),stage_width=layout.width)
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    tracker = get_cuda_rng_tracker()
    os.environ['FLASH_PARALLEL_LAYOUT'] = 'legacy'

    def connect(epoch, local=False):
        nonlocal local_handles
        started = time.monotonic()
        dist.init_process_group('nccl', store=dist.PrefixStore(f'v4/replica{replica}/{epoch}/', store),
                                rank=device if local else rank, world_size=(layout.world if c.get('candidate_full_dp') else layout.width) if local else layout.world, timeout=timeout)
        directory = out / f'topology_{epoch}_{role}'
        directory.mkdir(exist_ok=True)
        with local_cache.creating(reuse=isinstance(epoch,int) and epoch>=1) if local_cache else nullcontext():
            groups = megatron_adapter.initialize_groups(directory, 2, 1 if local and not c.get('candidate_full_dp') else 2, layout.tp)
        assert state.get_pipeline_model_parallel_rank() == (0 if local and not c.get('candidate_full_dp') else rank // layout.width)
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
        stage, _ = partition_model(full, ep=ep, tp=state.get_tensor_model_parallel_group(),
            pp_rank=pp_rank, StageModel=StageModel, patch_moe=patch_moe,
            patch_tensor_parallel=patch_tensor_parallel)
        ddp, optimizer, scheduler = configure_model(stage, adapter=megatron_adapter,
            dense=dense, expert_dp=expert_dp, finalize=finalize_model_grads)
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
    preparation_started=time.monotonic()
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
                prepared_stage=build(full,local_groups,rank//layout.width)
            del full
            prepared_old=(metadata_handles['tp'],local_groups[0],local_groups[2],local_groups[1])
            event('standby_metadata_model_ready',allocated=torch.cuda.memory_allocated(),
                  scope='constructor metadata only; communication forbidden; no shadow training')
        elif c.get('gpu_prepare'):
            local_groups = connect('standby', local=True)
            # Construct the eventual stage's DDP buckets on its isolated group.
            pp_rank = rank // layout.width
            if not c.get('candidate_full_dp'):
                state.set_pipeline_model_parallel_rank(pp_rank)
                state.set_pipeline_model_parallel_world_size(2)
            build_started=time.monotonic()
            prepared_stage = build(full, local_groups, pp_rank)
            event('standby_stage_built',seconds=time.monotonic()-build_started)
            del full
            prepared_old = (state.get_tensor_model_parallel_group(),local_groups[0],local_groups[2],local_groups[1])
            if c.get('gpu_warmup') and c.get('candidate_full_dp'):
                warm_stage,warm_ddp,warm_opt,warm_scheduler=prepared_stage
                warm_batches=make_batches(sample_plan(0,replica=replica,local_data_rank=state.get_data_parallel_rank()),seq_len=plan.seq_len,vocab_size=warm_stage.hf_config.vocab_size)
                warm_schedule=get_forward_backward_func()
                warm_progress={'updates':0}
                def warm_execute():
                    return warm_schedule(forward_step_func=forward_step,data_iterator=iter(warm_batches),model=warm_ddp,num_microbatches=2,seq_length=plan.seq_len+warm_stage.extra_positions(1),micro_batch_size=1)
                warm_step=train_step(warm_stage,warm_ddp,warm_opt,warm_scheduler,warm_progress,execute=warm_execute)
                next(warm_step);complete_step(warm_step,True)
                warm_opt.zero_grad(set_to_none=True);warm_ddp.zero_grad_buffer()
                del warm_batches
                event('standby_full_dp_warmed')
            if c.get('gpu_warmup') and not c.get('candidate_full_dp'):
                started = time.monotonic()
                warm_stage, warm_ddp, warm_opt, _ = prepared_stage
                warm_opt.zero_grad(set_to_none=True); warm_ddp.zero_grad_buffer()
                tokens = (torch.arange(plan.seq_len+1,device='cuda').view(1,-1)%32+100)
                inputs = tokens[:,:-1].contiguous()
                if pp_rank == 0:
                    output = warm_ddp(inputs)
                    loss = output[:plan.seq_len].float().square().mean()
                else:
                    size = plan.seq_len + warm_stage.extra_positions(1)
                    activation = torch.randn(size,1,warm_stage.hf_config.hidden_size,device='cuda')
                    activation[plan.seq_len:] = 0
                    activation.requires_grad_(True)
                    warm_stage.set_input_tensor(activation)
                    logits, aux = warm_ddp(inputs)
                    loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),
                        tokens[:,1:].reshape(-1)) + warm_stage.hf_config.router_aux_loss_coef*aux
                loss.backward(); warm_ddp.finish_grad_sync()
                for parameter in warm_stage.parameters(): parameter.grad=parameter.main_grad
                warm_opt.step()
                warm_opt.zero_grad(set_to_none=True); warm_ddp.zero_grad_buffer()
                warm_stage.input_tensor = None
                if pp_rank == 0:del output
                else:del activation, logits, aux
                del loss, tokens, inputs
                torch.cuda.synchronize()
                event('standby_gpu_warmed',seconds=time.monotonic()-started,
                      scope=f'synthetic PP{pp_rank} forward/backward/Adam with standby-local TP/EP',
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
    initialize_rng(rank)
    progress = {'updates': 0}
    outer = None
    training_state = TrainingState(stage, optimizer, scheduler, progress, tracker)
    rng, restore_rng = training_state.rng, training_state.restore_rng
    final_audit_started=False
    hash_calls=0
    def hashes():
        nonlocal hash_calls
        if c.get('performance_no_audit') and not audit_boundary and not final_audit_started:
            raise RuntimeError('development state hash entered the runtime timing window')
        started=time.monotonic()
        value=training_state.hashes();hash_calls+=1
        event('state_hash',step=progress['updates'],seconds=time.monotonic()-started,
              final_audit=final_audit_started,calls=hash_calls)
        return value
    def snapshot(*, audit=None, copy_state=True):
        if audit is None:audit=audit_boundary
        started = time.monotonic()
        torch.cuda.synchronize()
        value=training_state.capture(source_replica=replica, source_rank=rank, v4_epoch=1, v4_owner='C',
                outer_manager=outer.state_dict() if outer else None)
        if copy_state:value=tree_map(lambda t:t.detach().clone() if isinstance(t,torch.Tensor) else t,value)
        value['hashes'] = hashes() if audit else {}
        torch.cuda.synchronize()
        event('snapshot_ready', step=progress['updates'], seconds=time.monotonic()-started,
              allocated=torch.cuda.memory_allocated(),frozen_readonly=not copy_state)
        return value
    def restore(value):
        started = time.monotonic()
        training_state.restore(value, clone_optimizer=True)
        optimizer.zero_grad(set_to_none=True); ddp.zero_grad_buffer()
        torch.cuda.synchronize()
        actual = hashes() if audit_boundary else None
        if audit_boundary and actual != value['hashes']:
            raise RuntimeError('complete state restoration mismatch: '+str([k for k in actual if actual[k]!=value['hashes'][k]]))
        event('restored', step=progress['updates'], hashes_match=True if audit_boundary else None, hashes=actual,
              seconds=time.monotonic()-started)
        return actual
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
    session_runtime=None
    native_finish = ddp.finish_grad_sync
    def gradient_trace(point):
        if c.get('trace_step') != progress['updates']+1:return
        values=[b.grad_data for buf in ddp.buffers+ddp.expert_parallel_buffers for b in buf.buckets]
        fields=dict(bucket_hash=digest(values))
        if point=='optimizer':fields['parameters']={n:digest(p.main_grad) for n,p in stage.named_parameters()}
        event('gradient_trace',step=progress['updates']+1,point=point,**fields)
    def finish_grad_sync():
        if session_runtime is not None:return session_runtime.gradient_finalize(native_finish)
        native_finish()
        gradient_trace('native')
        if c.get('overlap_peer') and replica and current_phase=='after' and progress['updates']==plan.cut:
            torch.cuda.synchronize()
            store.set(f'peer_gradient_ready/{rank}','1')
            event('peer_native_gradient_ready',step=progress['updates'])
        outer_started=time.monotonic()
        if outer:
            outer.reduce([b for buf in ddp.buffers+ddp.expert_parallel_buffers for b in buf.buckets],
                         progress['updates'])
        gradient_trace('outer')
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
            if role == 'B' and phase == 'before' and step >= c.get('slow_start_step',plan.cut-1) and c.get('slow_ms',0):
                event('planned_slow',step=step,delay_ms=c['slow_ms'])
                time.sleep(c['slow_ms']/1000)
            sample_ids = sample_plan(step, microbatches=plan.microbatches,
                participants=2 if c.get('outer_manager') else 1, replica=replica,
                local_data_rank=state.get_data_parallel_rank(), fixed_batch=bool(c.get('outer_manager')))
            if not c.get('outer_manager'):
                sample_ids = [step*plan.microbatches*2+m*2+state.get_data_parallel_rank() for m in range(plan.microbatches)]
            batches = make_batches(sample_ids, seq_len=plan.seq_len, vocab_size=stage.hf_config.vocab_size,
                data=c.get('data','pattern'))
            if c.get('overlap_peer') and replica and phase=='after' and step==plan.cut:
                store.set(f'peer_compute/{rank}','1')
                event('peer_compute_begin',step=step)
            def execute_step():
                return schedule(forward_step_func=forward_step, data_iterator=iter(batches), model=ddp,
                    num_microbatches=len(batches), seq_length=plan.seq_len+stage.extra_positions(1), micro_batch_size=1)
            transaction = train_step(stage, ddp, optimizer, scheduler, progress, execute=execute_step)
            computed = next(transaction)
            metrics, finite = computed.metrics, computed.finite
            if not finite: raise RuntimeError('nonfinite gradients')
            ownership.authorize(owner_role, epoch)
            if outer:
                outer.commit(step)
            gradient_trace('optimizer')
            complete_step(transaction, True)
            step_hashes = hashes() if ((c.get('reference_only') and not c.get('performance_no_audit')) or independent_reference or
                (c['oracle'] and phase in ('oracle','after'))) else None
            if c.get('reference_only') and not c.get('performance_no_audit'):
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
        nonlocal final_audit_started
        if c.get('performance_no_audit'):
            audit_store=dist.PrefixStore('v4-final-audit/',base_store)
            event('final_audit_ready',step=progress['updates'])
            audit_store.set(f'{replica}/{rank}',str(progress['updates']))
            keys=[f'{r}/{i}' for r in range(2) for i in range(8)]
            audit_store.wait(keys)
            if any(int(audit_store.get(k))!=progress['updates'] for k in keys):
                raise RuntimeError('final audit requires a common completed window')
            event('final_audit_begin',step=progress['updates'])
        final_audit_started=True
        actual = snapshot(audit=True)
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
        if final_reference:
            expected=final_reference['workers'][f'{replica}/{rank}'].get(str(progress['updates']))
            if expected is None or actual['hashes']!=expected['hashes'] or actual['outer_manager']!=expected['outer_manager']:
                raise RuntimeError('independent final state mismatch')
            event('final_reference_comparison',step=progress['updates'],exact=True,hashes=actual['hashes'])
        event('done', step=progress['updates'], comparison=comparison,
              identity_preserved=identity_preserved(), hashes=actual['hashes'],state_hash_calls=hash_calls)
        if outer:
            outer.close()
        disconnect()
        log.close()

    if initial_candidate:event('standby_preparation_complete',seconds=time.monotonic()-preparation_started)
    event('validation_policy',boundary_validation=boundary_mode,digest_engine=os.environ.get('SHIFT_DIGEST_ENGINE','reference'))
    event('model_ready', full_parameters=full_parameters, local_parameters=sum(t.numel() for t in stage.parameters()),
          model_id=MODEL_ID, revision=REVISION)
    def configure_runtime(*,outer_value=...,plan_value=...,ownership_value=...,store_value=...,role_value=...,comparison_start=...,session_runtime_value=...):
        nonlocal outer,plan,ownership,store,owner_role,comparison_start_step,session_runtime
        if session_runtime_value is not ...:session_runtime=session_runtime_value
        if outer_value is not ...:outer=outer_value
        if plan_value is not ...:plan=plan_value
        if ownership_value is not ...:ownership=ownership_value
        if store_value is not ...:store=store_value
        if role_value is not ...:owner_role=role_value
        if comparison_start is not ...:comparison_start_step=comparison_start
    if c.get('unified_session'):
        from .session_engine import run_session
        run_session(locals().copy())
        return
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
    if (c.get('reference_only') and not c.get('performance_no_audit')) or (independent_reference and role != 'C'):
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
        event('done',step=progress['updates'],comparison=None,identity_preserved=identity_preserved(),hashes=hashes() if not c.get('performance_no_audit') else {})
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
