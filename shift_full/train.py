"""SHIFT training validation and matched flat-DP baseline on existing Qwen MoE."""
import time
_MODULE_STARTED = time.monotonic()
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
from shift_full.training_core import (TrainingState, partition_model, configure_model,
    sample_plan, make_batches, train_step, complete_step, initialize_rng)
from shift_full.compat import install_baby_event_fix, install_managed_callback_fix, bootstrap_legacy_core
from shift_full.reference_audit import (
    parse_reference_dump_steps,
    reference_anchor_path,
    reference_dump_payload,
    should_dump_reference,
    validate_reference_anchor,
)
if os.environ.get("SHIFT_BABY_COMPAT") == "1":
    install_baby_event_fix()  # Inherited by the isolated subprocess backend.
    install_managed_callback_fix()


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
    from shift_full.state_digest import configured_engine, digest_staged
    if configured_engine() == "staged":return digest_staged(value)
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
    p.add_argument('--lighthouse', default='http://192.0.2.10:29771')
    p.add_argument('--root', default='/workspace/SHIFT')
    p.add_argument('--flash', default='/workspace/FLASH1022')
    p.add_argument('--max-seconds', type=int, default=900)
    p.add_argument('--mode', choices=['shift', 'nativeft', 'flat'], default='shift')
    p.add_argument('--backend', choices=['nccl', 'unified', 'blocking', 'baby', 'gloo', 'hybrid'], default='hybrid')
    p.add_argument('--transport', choices=['gpu', 'http'], default='gpu')
    p.add_argument('--steps', '--target-steps', dest='steps', type=int, default=1000000)
    p.add_argument('--seq-len', type=int, default=16)
    p.add_argument('--microbatches', type=int, default=2)
    p.add_argument('--hash-every', type=int, default=0)
    p.add_argument('--audit-state', action='store_true')
    p.add_argument('--final-audit', action='store_true', help='development hashes after all measured commits')
    p.add_argument('--measure-step', action='store_true')
    p.add_argument('--sync-quorum', action='store_true')
    p.add_argument('--fixed-batch', action='store_true')
    p.add_argument('--fp32', action='store_true')
    p.add_argument('--data', choices=['pattern','random'], default='pattern')
    p.add_argument('--reference-dump', action='store_true')
    p.add_argument('--reference-dump-steps', type=parse_reference_dump_steps,
                   help='comma-separated committed update numbers to save as full tensors')
    p.add_argument('--reference-ranks', default='0,1,2,3,4,5,6,7')
    p.add_argument('--reference-anchor')
    p.add_argument('--reference-anchor-step', type=int, default=1)
    p.add_argument('--reference-anchor-save', help='directory for full schema-2 per-rank anchors')
    p.add_argument('--reference-anchor-save-step', type=int, default=1,
                   help='committed update to save as the schema-2 anchor')
    p.add_argument('--reference-signatures', action='store_true',
                   help='log compact full-state numerical signatures after every committed update')
    p.add_argument('--audit-routing', action='store_true')
    p.add_argument('--checkpoint-step', type=int, default=0)
    p.add_argument('--checkpoint-cold', action='store_true')
    p.add_argument('--resume-checkpoint', action='store_true')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--bucket-size', type=int, default=2_097_152)
    p.add_argument('--communication-timeout', type=float, default=10)
    p.add_argument('--profile-timings', action='store_true')
    p.add_argument('--trace-events', action='store_true', help='read CUDA events only after the existing step sync')
    p.add_argument('--bucket-ready-trace', action='store_true', help='diagnostic streams/events; not performance evidence')
    p.add_argument('--timeline-trace-step', type=int, default=0, help='one-step CPU/CUDA diagnostic on replica0 ranks0/4; not performance evidence')
    p.add_argument('--local-group-sync', action='store_true')
    p.add_argument('--eager-groups', action='store_true')
    p.add_argument('--recovery-timing', action='store_true', help='log bounded joining-thread milestones')
    p.add_argument('--gradient-probe-step', type=int, default=0, help='bounded single-update gradient slices; diagnostic only')
    p.add_argument('--bucket-hook', action='store_true')
    p.add_argument('--warmup-before-join', action='store_true')
    p.add_argument('--standby', choices=('cpu', 'gpu', 'generic'))
    p.add_argument('--strict-fp32', action='store_true')
    args = p.parse_args()
    if args.reference_dump_steps is not None and not args.reference_dump:
        p.error('--reference-dump-steps requires --reference-dump')
    try:
        reference_rank_set = frozenset(int(value.strip()) for value in args.reference_ranks.split(','))
    except ValueError:
        p.error('--reference-ranks must be comma-separated non-negative integers')
    if not reference_rank_set or any(value < 0 for value in reference_rank_set):
        p.error('--reference-ranks must be comma-separated non-negative integers')
    if args.reference_anchor_step < 0:
        p.error('--reference-anchor-step must be non-negative')
    if args.reference_anchor_save_step < 1:
        p.error('--reference-anchor-save-step must be positive')
    if args.strict_fp32:
        args.fp32=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
    rank = int(os.environ['LOCAL_RANK'])
    device_rank = int(os.environ.get('SHIFT_CUDA_DEVICE',rank))
    out = Path(os.environ.get('SHIFT_RESULT_DIR', str(Path(args.root) / 'results' / args.session)))
    out.mkdir(parents=True, exist_ok=True)
    log = (out / f'events_g{args.generation}_r{rank}.jsonl').open('a', buffering=1)
    lock = threading.Lock()

    def event(kind, **fields):
        row = dict(kind=kind, time=time.time(), monotonic=time.monotonic(), rid=args.replica, rank=rank,
                   generation=args.generation, physical_node=os.environ.get('SHIFT_PHYSICAL_NODE', args.host_ip),
                   host_ip=args.host_ip, pid=os.getpid(), **fields)
        with lock:
            log.write(json.dumps(row, sort_keys=True) + '\n')

    def phase(name, started):
        event('phase', name=name, seconds=time.monotonic()-started)
        return time.monotonic()

    def standby_gate(mode, **ready_fields):
        import resource
        fields = dict(
            cpu_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cuda_allocated_bytes=torch.cuda.memory_allocated() if mode=='gpu' else 0,
            role_manifest=role_manifest() if mode=='gpu' else None,
        )
        fields.update(ready_fields)
        event('standby_ready', mode=mode, **fields)
        until=time.monotonic()+args.max_seconds
        while not (out/'ACTIVATE').exists():
            if (out/'STOP').exists() or time.monotonic()>until:
                raise RuntimeError('standby stopped or timed out before activation')
            time.sleep(.05)
        event('standby_activated', mode=mode)

    if args.standby=='cpu':
        standby_gate('cpu')

    phase_started = phase('python_imports', _MODULE_STARTED)
    install_managed_callback_fix()
    if args.backend == 'baby':
        os.environ['SHIFT_BABY_COMPAT'] = '1'
        install_baby_event_fix()
        install_managed_callback_fix()
    legacy_namespace = bootstrap_legacy_core(args.flash) if os.environ.get('SHIFT_ISOLATED_RUNTIME') == '1' else False
    os.environ['FLASH_PARALLEL_LAYOUT'] = 'node_local_pp' if args.mode == 'flat' else 'legacy'  # Native ordering INSIDE each replica.
    os.environ['TORCHFT_USE_OTEL'] = 'false'
    logging.basicConfig(level=logging.WARNING)
    rank = int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(1)
    overlap_imports = os.environ.get('SHIFT_OVERLAP_CPU_IMPORTS', '0')
    if overlap_imports not in ('0', '1'):
        raise ValueError('invalid SHIFT_OVERLAP_CPU_IMPORTS')
    cpu_imports = None
    if overlap_imports == '1':
        if not legacy_namespace:
            raise RuntimeError('CPU import overlap requires the isolated legacy namespace')
        import sys
        sys.path.insert(0, args.flash)
        from flash_v1 import megatron_adapter
        from flash_v1.pipeline_adapter import StageModel, forward_step
        from flash_v1.tensor_parallel import patch_tensor_parallel
        from flash_v1.train import load_model, patch_moe, MODEL_ID, REVISION
        from shift_full.cold_imports import CPUImports
        cpu_imports = CPUImports(megatron_adapter)
    torch.cuda.set_device(device_rank)
    phase_started = phase('runtime_cuda_setup', phase_started)
    if cpu_imports is not None:
        imported_components, import_timing = cpu_imports.result()
        event('cpu_import_overlap', **import_timing,
              elapsed_overlap_seconds=max(0, min(import_timing['end_monotonic'], phase_started)-import_timing['start_monotonic']),
              performance_evidence=False)
        del cpu_imports
        phase_started = phase('cpu_import_join', phase_started)
    run_common_warmup = (
        args.standby == 'generic'
        or (args.standby is None and os.environ.get('SHIFT_COMMON_GPU_WARMUP') == '1')
    )
    if run_common_warmup:
        common_started = time.monotonic()
        from shift_full.common_warmup import warmup as common_warmup
        common_metrics = common_warmup(torch.device('cuda', device_rank))
        common_metrics['common_warmup_seconds'] = time.monotonic() - common_started
        if dist.is_initialized():
            raise RuntimeError('common GPU warmup unexpectedly initialized training process groups')
        if args.standby == 'generic':
            standby_gate('generic', model_loaded=False, training_groups_initialized=False,
                         **common_metrics)
        else:
            event('common_gpu_warmup', mode='active', model_loaded=False,
                  training_groups_initialized=False, **common_metrics)
        phase_started = time.monotonic()
    nccl_options = None
    if args.backend == 'unified':
        nccl_options = dist.ProcessGroupNCCL.Options()
        nccl_options.config.blocking = False
        original_new_group = dist.new_group
        def unified_new_group(*pos, **kwargs):
            backend = kwargs.get('backend', pos[2] if len(pos) > 2 else None)
            if backend in (None, 'nccl'):
                options = kwargs.get('pg_options') or dist.ProcessGroupNCCL.Options()
                options.config.blocking = False
                kwargs['pg_options'] = options
            return original_new_group(*pos, **kwargs)
        dist.new_group = unified_new_group
    dist.init_process_group('nccl', timeout=timedelta(seconds=180), pg_options=nccl_options,
                            device_id=torch.device('cuda',device_rank) if args.eager_groups else None)
    assert dist.get_world_size() == (16 if args.mode == 'flat' else 8)
    phase_started = phase('default_process_group', phase_started)

    import sys
    sys.path.insert(0, args.flash)
    from flash_v1 import megatron_adapter
    from flash_v1.pipeline_adapter import StageModel, forward_step
    from flash_v1.tensor_parallel import patch_tensor_parallel
    from flash_v1.train import load_model, patch_moe, MODEL_ID, REVISION
    phase_started = phase('adapter_imports', phase_started)
    if overlap_imports == '1':
        state, finalize_model_grads, get_forward_backward_func = imported_components
        del imported_components
    else:
        state, _, _ = megatron_adapter.core()
        from megatron.core.distributed import finalize_model_grads
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    phase_started = phase('megatron_imports', phase_started)
    topology_out = out / f'topology_g{args.generation}'
    topology_out.mkdir(parents=True, exist_ok=True)
    overlap_cpu_load = os.environ.get('SHIFT_OVERLAP_CPU_MODEL_LOAD', '0')
    if overlap_cpu_load not in ('0', '1'):
        raise ValueError('invalid SHIFT_OVERLAP_CPU_MODEL_LOAD')
    cpu_model_load = None
    if overlap_cpu_load == '1':
        # This pinned group/topology path consumes no Torch/Python RNG. Only
        # CPU checkpoint construction runs in the helper; every PG stays here.
        torch.manual_seed(2026)
        random.seed(2026)
        from shift_full.cold_cpu_load import CPUModelLoad
        cpu_model_load = CPUModelLoad(lambda:load_model(Path(args.flash) / 'models/qwen_moe'))
    new_group = dist.new_group
    def local_new_group(*pos, **kwargs):
        kwargs['use_local_synchronization'] = True
        return new_group(*pos, **kwargs)
    if args.local_group_sync: dist.new_group = local_new_group
    try:
        ep, expert_dp, dense = megatron_adapter.initialize_groups(topology_out, 2, 2, 2)
    finally:
        dist.new_group = new_group
    assert dist.get_world_size(expert_dp) == (2 if args.mode == 'flat' else 1)
    assert dist.get_world_size(dense) == (4 if args.mode == 'flat' else 2)
    phase_started = phase('parallel_groups', phase_started)
    # Reset CPU/CUDA/Python RNG after common prewarm and topology setup.
    if cpu_model_load is None:
        torch.manual_seed(2026)
        random.seed(2026)
        full = load_model(Path(args.flash) / 'models/qwen_moe')
    else:
        full, load_timing = cpu_model_load.result()
        event('cpu_model_load_overlap', **load_timing,
              elapsed_overlap_seconds=max(0, min(load_timing['end_monotonic'], phase_started)-load_timing['start_monotonic']),
              performance_evidence=False)
        del cpu_model_load
    phase_started = phase('full_model_load', phase_started)
    total_parameters = sum(v.numel() for v in full.parameters())
    if args.fp32:
        full.float()
    stage, tp_manifest = partition_model(full, ep=ep, tp=state.get_tensor_model_parallel_group(),
        pp_rank=state.get_pipeline_model_parallel_rank(), StageModel=StageModel,
        patch_moe=patch_moe, patch_tensor_parallel=patch_tensor_parallel)
    del full
    phase_started = phase('model_partition', phase_started)
    ddp, optimizer, scheduler = configure_model(stage, adapter=megatron_adapter, dense=dense,
        expert_dp=expert_dp, finalize=finalize_model_grads, bucket_size=args.bucket_size,
        lr=args.lr, scheduler_step=1000)
    phase_started = phase('cuda_ddp_optimizer', phase_started)
    def role_manifest():
        return dict(tp=2, pp=2, ep=2, etp=2, rank=rank,
                    shard_signature=digest([(n,tuple(p.shape),str(p.dtype)) for n,p in stage.named_parameters()]),
                    optimizer_groups=[dict(count=len(g['params']),lr=g['lr'],betas=g['betas'],eps=g['eps'],
                                           weight_decay=g['weight_decay']) for g in optimizer.param_groups])
    progress = {'updates': 0}
    warming_up = False
    if args.audit_routing:
        route_calls = {}
        for name, module in stage.named_modules():
            if name.endswith('.mlp.gate'):
                def route_hook(_module, _inputs, output, layer=name):
                    if warming_up: return
                    call=route_calls.get(layer,0);route_calls[layer]=call+1
                    scores=torch.softmax(output.float(),dim=-1)
                    indices=torch.topk(scores,stage.top_k,dim=-1).indices
                    boundary=torch.topk(scores,stage.top_k+1,dim=-1).values
                    margin=boundary[...,-2]-boundary[...,-1]
                    event('routing',step=progress['updates'],layer=layer,call=call,topk_hash=digest(indices),
                          sorted_topk_hash=digest(indices.sort(dim=-1).values),
                          min_boundary_margin=float(margin.min()),zero_margins=int((margin==0).sum()),
                          near_ties_1e6=int((margin<1e-6).sum()),tokens=margin.numel())
                module.register_forward_hook(route_hook)
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    from shift_full.state_inventory import buffer_manifest, restore_buffers, tracker_snapshot, restore_tracker
    rng_tracker = get_cuda_rng_tracker()
    boundary_rng = None
    audit_joint_pending = False
    manager = None
    reference_manager_state = None

    training_state = TrainingState(stage, optimizer, scheduler, progress, rng_tracker)
    rng_snapshot = training_state.rng
    current_hashes = training_state.hashes

    def export_state():
        nonlocal audit_joint_pending
        event('snapshot_begin',step=progress['updates'])
        torch.cuda.current_stream().synchronize()
        step = manager.current_step() if manager else progress['updates']
        assert step == progress['updates'], (step, progress)
        audit_joint_pending = True
        rng = boundary_rng['rng'] if boundary_rng and boundary_rng['step'] == step else rng_snapshot()
        frozen = boundary_rng if boundary_rng and boundary_rng['step'] == step else None
        if frozen:
            frozen['copy_ready'].synchronize()
        all_buffers = frozen['buffers'] if frozen else {n: b.detach().clone() for n,b in stage.named_buffers()}
        payload = training_state.capture(rng=rng, all_buffers=all_buffers,
            source_replica=args.replica, source_rank=rank)
        model_state = payload['model']
        payload['hashes'] = current_hashes(rng) if args.audit_state else {}
        if args.audit_state:
            payload['hashes'].update(model=digest(model_state), buffers=digest(all_buffers))
        if args.transport == 'http':
            payload = cpu_tree(payload)
        event('snapshot', step=step, hashes=payload['hashes'],
              optimizer_tensors=tensor_count(payload['optimizer'].get('state', {})))
        return payload

    def load_state(payload):
        nonlocal audit_joint_pending
        load_started = time.monotonic()
        if args.recovery_timing:
            event('load_state_begin', step=payload['step'])
        assert payload['source_rank'] == rank
        assert payload['step'] == payload['progress']['updates']
        assert payload['step'] == (manager.current_step() if manager else payload['step'])
        training_state.restore(payload)
        audit_joint_pending = True
        torch.cuda.current_stream().synchronize()
        loaded_at = time.monotonic()
        event('state_loaded',step=payload['step'],load_seconds=loaded_at-load_started,
              source_replica=payload['source_replica'],source_rank=payload['source_rank'])
        hashes = current_hashes() if args.audit_state else {}
        matches = hashes == payload['hashes']
        event('restored', step=payload['step'], load_seconds=loaded_at-load_started, audit_seconds=time.monotonic()-loaded_at, source_replica=payload['source_replica'],
              source_rank=payload['source_rank'], hashes_match=matches,
              expected_hashes=payload['hashes'], actual_hashes=hashes,
              optimizer_tensors=tensor_count(optimizer.state_dict().get('state', {})),
              cuda_allocated_bytes=torch.cuda.memory_allocated(device_rank), cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device_rank))
        assert matches, 'Restored state differs from source snapshot'

    # Reference anchors are applied only after Megatron's RNG tracker exists.
    # The same local-rank file is loaded by each replica on its own host.
    if args.reference_anchor:
        anchor_started = time.monotonic()
        anchor_path = reference_anchor_path(args.reference_anchor, rank, args.reference_anchor_step)
        anchor = torch.load(anchor_path, map_location='cpu', weights_only=False, mmap=True)
        validate_reference_anchor(anchor, expected_rank=rank, expected_step=args.reference_anchor_step)
        stage.load_state_dict(anchor['model'], strict=True)
        restore_buffers(stage, anchor['all_buffers'])
        optimizer.load_state_dict(anchor['optimizer'])
        scheduler.load_state_dict(anchor['scheduler'])
        progress.clear()
        progress.update(anchor['progress'])
        torch.set_rng_state(anchor['torch_rng'])
        torch.cuda.set_rng_state(anchor['cuda_rng'])
        random.setstate(anchor['python_rng'])
        restore_tracker(rng_tracker, anchor['tracker_rng'])
        reference_manager_state = dict(anchor['manager_state'])
        restored_hashes = current_hashes()
        if restored_hashes != anchor['hashes']:
            raise RuntimeError(
                f"schema-2 reference anchor restoration mismatch: "
                f"expected {anchor['hashes']}, got {restored_hashes}"
            )
        event('reference_anchor_loaded', step=progress['updates'], path=str(anchor_path),
              schema=2, hashes_match=True)
        del anchor
        phase_started = phase('reference_anchor_load', anchor_started)

    if args.warmup_before_join:
        warm_started=time.monotonic()
        saved_rng=rng_snapshot()
        before_hash=digest(stage.state_dict()) if args.audit_state else None
        optimizer.zero_grad(set_to_none=True);ddp.zero_grad_buffer()
        warm_batches=[]
        for micro in range(args.microbatches):
            tokens=(torch.arange(args.seq_len+1,device='cuda').view(1,-1)+micro)%32+100
            warm_batches.append(dict(input_ids=tokens[:,:-1].contiguous(),next_token_labels=tokens[:,1:].contiguous()))
        warming_up=True
        try:
            get_forward_backward_func()(forward_step_func=forward_step,data_iterator=iter(warm_batches),
                                        model=ddp,num_microbatches=args.microbatches,
                                        seq_length=args.seq_len+stage.extra_positions(1),micro_batch_size=1)
        finally:
            warming_up=False
        torch.cuda.current_stream().synchronize()
        optimizer.zero_grad(set_to_none=True);ddp.zero_grad_buffer()
        torch.set_rng_state(saved_rng['torch_rng']);torch.cuda.set_rng_state(saved_rng['cuda_rng']);random.setstate(saved_rng['python_rng']);restore_tracker(rng_tracker, saved_rng['tracker_rng'])
        if before_hash is not None: assert before_hash==digest(stage.state_dict()),'warmup changed model state'
        phase_started=phase('local_cuda_warmup',warm_started)

    if args.standby=='gpu':
        standby_gate('gpu')

    if not args.reference_anchor: initialize_rng(rank % 8)

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

    if args.mode != 'flat':
        if args.backend in ('nccl', 'unified', 'blocking', 'baby'):
            from torchft import ProcessGroupNCCL, ProcessGroupBabyNCCL
            from shift_full.compat import BlockingNCCL, RecoveryNCCL
            cls = ProcessGroupBabyNCCL if args.backend == 'baby' else BlockingNCCL if args.backend == 'blocking' else RecoveryNCCL
            pg = cls(timeout=timedelta(seconds=30 if args.backend == "baby" else args.communication_timeout))
        elif args.backend == 'hybrid':
            from shift_full.compat import HybridGloo
            pg = HybridGloo(timeout=timedelta(seconds=10))
        else:
            pg = CompatibleProcessGroupGloo(timeout=timedelta(seconds=10))
        native_configure=pg.configure
        def measured_configure(*pos, **kwargs):
            started=time.monotonic()
            world_size=pos[3] if len(pos)>3 else kwargs.get('world_size')
            quorum_id=pos[4] if len(pos)>4 else kwargs.get('quorum_id')
            event('reconfigure_begin',world_size=world_size,quorum_id=quorum_id)
            try:
                return native_configure(*pos,**kwargs)
            finally:
                event('reconfigure_end',world_size=world_size,quorum_id=quorum_id,seconds=time.monotonic()-started)
        pg.configure=measured_configure
        if args.transport == 'gpu':
            if args.backend == 'gloo':
                raise ValueError('GPU transport requires NCCL')
            from shift_full.transport import PackedGPUTransport
            transport = PackedGPUTransport(pg.gpu_pg if args.backend == 'hybrid' else pg, torch.device('cuda', device_rank), timedelta(seconds=90), event)
        else:
            transport = ReachableHTTPTransport(timedelta(seconds=90), 0)
        manager = Manager(pg=pg, load_state_dict=load_state, state_dict=export_state,
                          min_replica_size=1, use_async_quorum=not args.sync_quorum,
                          init_sync=False, timeout=timedelta(seconds=90),
                          quorum_timeout=timedelta(seconds=180), connect_timeout=timedelta(seconds=20),
                          hostname=args.host_ip, lighthouse_addr=args.lighthouse,
                          replica_id=f'shift{args.replica}', checkpoint_transport=transport, max_retries=8)
        if args.reference_anchor:
            manager.load_state_dict(reference_manager_state)
            if manager.current_step() != progress['updates']:
                raise RuntimeError(
                    f"reference anchor manager step {manager.current_step()} "
                    f"does not match progress {progress['updates']}"
                )

    if args.resume_checkpoint and args.generation > 0:
        resume_started = time.monotonic()
        payload = torch.load(out / f'checkpoint_r{rank}.pt', map_location='cpu', weights_only=False)
        if manager:
            manager.load_state_dict(dict(step=payload['step'], batches_committed=payload['step']*2))
        load_state(payload)
        event('checkpoint_resumed', step=progress['updates'], seconds=time.monotonic()-resume_started)

    managed_pg = None
    if args.mode == 'nativeft':
        from torchft.process_group import ManagedProcessGroup
        managed_pg = ManagedProcessGroup(manager)
    phase_started = phase('manager_init', phase_started)
    # original_finish is DDP's buffer-only sync. finalize_model_grads calls it
    # before the TP/PP tail, which starts only after this wrapper's final waits.
    # Both dense and expert buffers need outer AVG; this vendor's expert grads
    # retain their native dense-DP normalization.
    original_finish = ddp.finish_grad_sync
    buffers = list(ddp.buffers) + list(ddp.expert_parallel_buffers)
    gradient_probe = None
    if args.gradient_probe_step:
        if args.gradient_probe_step != 2 or args.steps != 2 or progress['updates'] != 1 or not args.reference_anchor or args.reference_dump or args.timeline_trace_step:
            raise ValueError('gradient probe requires original anchor single commit2 without dumps/profile')
        from shift_full.gradient_probe import GradientProbe
        gradient_probe = GradientProbe(torch, dist, stage, buffers, 2, progress,
            dict(replica=args.replica, rank=rank, physical_rank=args.replica*8+rank,
                 pp=state.get_pipeline_model_parallel_rank(), tp=dist.get_rank(state.get_tensor_model_parallel_group()),
                 ep=dist.get_rank(ep), global_layer_offset=state.get_pipeline_model_parallel_rank()*len(stage.decoder.layers),
                 sequence_parallel=stage.config.sequence_parallel,
                 tied_embeddings=stage.share_embeddings_and_output_weights, mode=args.mode))
    from shift_full.bucket_finalizer import BucketFinalizer
    bucket_finalizer = BucketFinalizer(original_finish, buffers, args.bucket_hook) if manager else None
    ready_trace = None
    if args.bucket_ready_trace:
        if not args.trace_events or not args.measure_step:
            raise ValueError('--bucket-ready-trace requires --trace-events --measure-step')
        from shift_full.bucket_ready_trace import BucketReadyTrace
        ready_trace = BucketReadyTrace([b for buf in buffers for b in buf.buckets], torch.cuda)
    cross_calls = [0]
    cross_timings = {}

    step_gpu_events = {}
    def gpu_mark(name):
        if args.trace_events:
            if name not in step_gpu_events: step_gpu_events[name] = torch.cuda.Event(enable_timing=True)
            step_gpu_events[name].record()

    def finish_grad_sync(_self):
        gpu_mark('local_finish_begin')
        if not manager:
            original_finish()
            if gradient_probe: gradient_probe.capture('post_outer')
            gpu_mark('outer_finish_end')
            return
        entered = time.monotonic()
        gpu_started = torch.cuda.Event(enable_timing=True) if args.profile_timings else None
        gpu_done = torch.cuda.Event(enable_timing=True) if args.profile_timings else None
        if gpu_started: gpu_started.record()
        pending = []
        d2h_seconds = dispatch_seconds = 0.0
        def enqueue(bucket):
            nonlocal d2h_seconds, dispatch_seconds
            if args.backend == 'baby': torch.cuda.current_stream().synchronize()
            before = time.monotonic()
            value = bucket.grad_data if args.backend not in ('gloo', 'hybrid') else bucket.grad_data.detach().float().cpu()
            d2h_seconds += time.monotonic()-before
            before = time.monotonic()
            work = (managed_pg.allreduce([value], dist.ReduceOp.AVG) if managed_pg is not None
                    else manager.allreduce(value))
            pending.append((bucket.grad_data, value, work))
            dispatch_seconds += time.monotonic()-before
        if args.bucket_hook:
            bucket_finalizer.run(enqueue)
        else:
            original_finish()
        intra_done = time.monotonic()
        if not args.bucket_hook:
            for bucket in bucket_finalizer.buckets: enqueue(bucket)
        waiting = time.monotonic()
        for target, cpu, work in pending:
            work.wait()
            if args.backend in ('gloo', 'hybrid'):
                target.copy_(cpu)
        if gradient_probe: gradient_probe.capture('post_outer')
        finished = time.monotonic()
        cross_timings.update(intra_finish_cpu_seconds=intra_done-entered,
                             cross_d2h_cpu_seconds=d2h_seconds, cross_dispatch_cpu_seconds=dispatch_seconds,
                             cross_wait_h2d_cpu_seconds=finished-waiting)
        if gpu_done:
            gpu_done.record(); gpu_done.synchronize()
            cross_timings['cross_gpu_seconds'] = gpu_started.elapsed_time(gpu_done)/1000
        gpu_mark('outer_finish_end')
        cross_calls[0] += len(pending)

    if manager or args.trace_events or gradient_probe:
        ddp.finish_grad_sync = types.MethodType(finish_grad_sync, ddp)
    initial = current_hashes() if args.audit_state else {}
    phase_started = phase('initial_audit', phase_started)
    event('ready', digest_engine=os.environ.get('SHIFT_DIGEST_ENGINE','reference'), overlap_cpu_model_load=overlap_cpu_load == '1', overlap_cpu_imports=overlap_imports == '1', physical_rank=args.replica * 8 + rank,
          layout='node_local_pp', tp=2, pp=2, ep=2, local_dense_dp=dist.get_world_size(dense),
          local_expert_dp=dist.get_world_size(expert_dp), model_id=MODEL_ID, model_revision=REVISION,
          full_parameters=total_parameters, local_parameters=sum(v.numel() for v in stage.parameters()),
          tp_manifest=tp_manifest, hashes=initial, torch_version=torch.__version__,
          cuda_version=torch.version.cuda, nccl_version=torch.cuda.nccl.version(),
          global_rank=dist.get_rank(), world_size=dist.get_world_size(), legacy_core_namespace=legacy_namespace,
          transport=args.transport, cross_gradient=args.backend, mode=args.mode, fp32=args.fp32,
          sync_quorum=args.sync_quorum,eager_groups=args.eager_groups,
          cuda_device=device_rank,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
          role_manifest=role_manifest(), state_schema=2,
          state_inventory=dict(buffers=buffer_manifest(stage), tracker_keys=sorted(rng_tracker.get_states()),
                               tracker_class=type(rng_tracker).__name__, snapshot='boundary buffers + RNG; parameter/Adam read-only until quorum finishes'),
          matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
          float32_matmul_precision=torch.get_float32_matmul_precision(),
          cpu_maxrss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss,
          cuda_allocated_bytes=torch.cuda.memory_allocated(device_rank), cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device_rank))
    deadline = time.monotonic() + args.max_seconds
    while not (out / 'START').exists():
        if (out / 'STOP').exists() or time.monotonic() > deadline:
            raise RuntimeError('Stopped or timed out before START')
        time.sleep(0.1)
    schedule = get_forward_backward_func()
    from contextlib import nullcontext
    timeline = None
    if args.timeline_trace_step and args.replica == 0 and rank in (0, 4):
        from shift_full.step_timeline import StepTimeline
        timeline = StepTimeline(torch, args.timeline_trace_step, out, rank)
    attempts = 0
    previous_commit_end = None
    recovery_requested = False
    joining_commit_seen = False
    def requested_stop():
        marker = out / 'STOP'
        if not marker.exists():
            return False
        text = marker.read_text().strip()
        return not text or progress['updates'] >= int(text)
    try:
        while not requested_stop() and time.monotonic() < deadline and progress['updates'] < args.steps:
            iteration_started = time.monotonic()
            if timeline: timeline.begin(progress['updates']+1)
            torch.cuda.current_stream().synchronize()
            boundary_done = time.monotonic()
            if manager:
                frozen_buffers = {n: b.detach().clone() for n,b in stage.named_buffers()}
                copy_ready = torch.cuda.Event()
                copy_ready.record()
                boundary_rng = dict(step=progress['updates'], rng=rng_snapshot(),
                                    buffers=frozen_buffers, copy_ready=copy_ready)
                if args.generation>0 and not recovery_requested:
                    event('recovery_request',step=progress['updates'])
                    recovery_requested=True
                manager.start_quorum()
                if args.fixed_batch:
                    if not args.sync_quorum:
                        raise ValueError('fixed batch requires synchronous quorum')
                    manager.wait_quorum()
            step = progress['updates']
            # num_participants waits for quorum. Avoid it before forward in the
            # dynamic-batch async path; data assignment does not depend on it.
            participants = manager.num_participants() if manager and (args.sync_quorum or args.fixed_batch) else 2 if not manager else None
            event('begin', step=step, participants=participants,
                  boundary_sync_seconds=boundary_done-iteration_started, quorum_seconds=time.monotonic()-boundary_done)
            optimizer.zero_grad(set_to_none=True)
            ddp.zero_grad_buffer()
            if args.mode == 'flat':
                data_rank = state.get_data_parallel_rank()
            else:
                replica_index = manager._participating_replica_rank if args.fixed_batch else args.replica
                data_rank = (replica_index or 0) * 2 + state.get_data_parallel_rank()
            sample_ids = sample_plan(step, microbatches=args.microbatches, participants=participants,
                replica=0 if args.mode=='flat' else (replica_index or 0), local_data_rank=state.get_data_parallel_rank(),
                fixed_batch=args.fixed_batch)
            count = len(sample_ids)
            batches = make_batches(sample_ids, seq_len=args.seq_len, vocab_size=stage.hf_config.vocab_size,
                data=args.data)
            if args.reference_dump or args.reference_signatures:
                event('batch_plan',step=step,sample_ids=sample_ids)
            started = time.monotonic()
            joining_timing = args.recovery_timing and args.generation > 0 and not joining_commit_seen
            if joining_timing:
                event('join_schedule_begin', step=step)
            if ready_trace: ready_trace.begin()
            gpu_mark('schedule_begin')
            before_calls = cross_calls[0]
            def execute_step():
                with timeline.scope('schedule') if timeline else nullcontext():
                    return schedule(forward_step_func=forward_step, data_iterator=iter(batches),
                        model=ddp, num_microbatches=count,
                        seq_length=args.seq_len + stage.extra_positions(1), micro_batch_size=1)
            transaction = train_step(stage, ddp, optimizer, scheduler, progress, execute=execute_step,
                synchronize=args.measure_step, schedule_end=lambda:gpu_mark('schedule_end'),
                finite_scope=(lambda:timeline.scope('finite')) if timeline else None,
                optimizer_scope=(lambda:timeline.scope('optimizer')) if timeline else None, mark=gpu_mark)
            computed = next(transaction)
            metrics = computed.metrics
            if gradient_probe:
                gradient_probe.capture('post_tail')
                event('gradient_probe', **gradient_probe.result())
            schedule_done = computed.schedule_done
            if joining_timing: event('join_schedule_end', step=step)
            finite = computed.finite
            if not finite:
                if manager: manager.report_error(RuntimeError('nonfinite gradients'))
                else: raise RuntimeError('nonfinite gradients in baseline')
            gradient_checked = computed.gradient_checked
            dump_selected = should_dump_reference(
                enabled=args.reference_dump,
                selected_steps=args.reference_dump_steps,
                next_step=progress['updates'] + 1,
                replica=args.replica,
                rank=rank,
                selected_ranks=reference_rank_set,
            )
            gradients = cpu_tree({name: parameter.main_grad for name, parameter in stage.named_parameters()}) if dump_selected else None
            decision_started = time.monotonic()
            if joining_timing:
                event('join_decision_begin', step=step)
            with timeline.scope('should_commit') if timeline else nullcontext():
                commit = manager.should_commit() if manager else True
            decision_done = time.monotonic()
            if joining_timing:
                event('join_decision_end', step=progress['updates'], commit=commit)
            gradient_signature_hash = None
            if args.reference_signatures and commit:
                gradient_signature_hash = digest(
                    {name: parameter.main_grad for name, parameter in stage.named_parameters()}
                )
            timings = dict(schedule_seconds=schedule_done-started, gradient_check_seconds=gradient_checked-schedule_done,
                           decision_seconds=decision_done-decision_started)
            attempts += 1
            if commit:
                update = complete_step(transaction, True)
                optimizer_done, step_sync_done = update['optimizer_done'], update['step_sync_done']
                if ready_trace:
                    event('bucket_ready_diagnostic', step=progress['updates'],
                          performance_evidence=False,
                          buckets=ready_trace.collect(step_gpu_events['schedule_begin'],
                              step_gpu_events['local_finish_begin'], step_gpu_events['outer_finish_end']))
                timings.update(optimizer_enqueue_seconds=optimizer_done-decision_done,
                               final_sync_seconds=step_sync_done-optimizer_done)
                if args.trace_events:
                    if not args.measure_step: raise ValueError('--trace-events requires --measure-step')
                    for label, begin, end in [('schedule', 'schedule_begin', 'schedule_end'),
                                               ('local_and_outer', 'local_finish_begin', 'outer_finish_end'),
                                               ('optimizer', 'optimizer_begin', 'optimizer_end')]:
                        timings[label+'_event_seconds'] = step_gpu_events[begin].elapsed_time(step_gpu_events[end])/1000
                actual_participants=manager.num_participants() if manager else 2
                contributed=manager.is_participating() if manager else True
                if contributed:
                    joining_commit_seen = True
                audit_joint=args.audit_state and audit_joint_pending and actual_participants>=2 and contributed
                hashes = current_hashes() if ((args.hash_every and progress['updates'] % args.hash_every == 0) or audit_joint) else {}
                if audit_joint: audit_joint_pending=False
                if gradients is not None:
                    dump_path = out / f'reference_step{progress["updates"]}_r{rank}.pt'
                    torch.save(reference_dump_payload(
                        gradients,
                        cpu_tree(stage.state_dict()),
                        cpu_tree(optimizer.state_dict()),
                        scheduler.state_dict(),
                    ), dump_path)
                if args.reference_signatures:
                    manager_step = manager.current_step() if manager else progress['updates']
                    if manager_step != progress['updates']:
                        raise RuntimeError(
                            f"reference signature manager step {manager_step} "
                            f"does not match progress {progress['updates']}"
                        )
                    signature_hashes = current_hashes()
                    signature_hashes['gradients'] = gradient_signature_hash
                    event('reference_signature', schema=1, step=progress['updates'],
                          input_step=step, manager_step=manager_step, progress=dict(progress), sample_ids=sample_ids,
                          hashes=signature_hashes)
                commit_end = time.monotonic()
                timings['commit_interval_seconds'] = commit_end-previous_commit_end if previous_commit_end else None
                previous_commit_end = commit_end
                event('commit', step=progress['updates'], updates=progress['updates'],
                      participants=actual_participants, contributed=contributed, hashes=hashes, model_hash=hashes.get('model'),
                      sample_ids=sample_ids, fixed_batch=args.fixed_batch,
                      global_batch=args.microbatches*4 if args.fixed_batch else args.microbatches*2*actual_participants,
                      optimizer_hash=hashes.get('optimizer'), scheduler_hash=hashes.get('scheduler'),
                      optimizer_tensors=tensor_count(optimizer.state_dict().get('state', {})),
                      cross_calls=cross_calls[0] - before_calls, finite_gradients=finite,
                      seconds=time.monotonic() - iteration_started, compute_seconds=time.monotonic()-started,
                      loss=[float(m['loss']) for m in metrics] if metrics else [], **timings, **cross_timings)
            else:
                complete_step(transaction, False)
                if ready_trace: ready_trace.discard()
                event('skipped', step=step, error=str(manager.errored()), **timings)
            if timeline:
                captured = timeline.end(commit)
                if captured: event('timeline_diagnostic', **captured)
            if manager:
                assert manager.current_step() == progress['updates']
            if (args.reference_anchor_save and commit
                    and progress['updates'] == args.reference_anchor_save_step):
                anchor_path = reference_anchor_path(args.reference_anchor_save, rank)
                anchor_path.parent.mkdir(parents=True, exist_ok=True)
                torch.cuda.current_stream().synchronize()
                anchor_step = manager.current_step() if manager else progress['updates']
                if anchor_step != progress['updates']:
                    raise RuntimeError(
                        f"reference anchor manager step {anchor_step} "
                        f"does not match progress {progress['updates']}"
                    )
                anchor_buffers = {name: value.detach().clone() for name, value in stage.named_buffers()}
                anchor_model = stage.state_dict()
                for name, value in anchor_buffers.items():
                    if name in anchor_model:
                        anchor_model[name] = value
                anchor_rng = rng_snapshot()
                anchor = cpu_tree(dict(
                    schema=2,
                    model=anchor_model,
                    all_buffers=anchor_buffers,
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    progress=dict(progress),
                    **anchor_rng,
                    source_replica=args.replica,
                    source_rank=rank,
                    step=anchor_step,
                ))
                anchor['manager_state'] = dict(
                    step=anchor_step,
                    batches_committed=anchor_step * 2,
                )
                anchor['hashes'] = current_hashes(anchor_rng)
                validate_reference_anchor(anchor, expected_rank=rank, expected_step=progress['updates'])
                temporary_path = anchor_path.with_name(f'.{anchor_path.name}.{os.getpid()}.tmp')
                torch.save(anchor, temporary_path)
                with temporary_path.open('rb') as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary_path, anchor_path)
                event('reference_anchor_saved', step=anchor_step, path=str(anchor_path),
                      schema=2, bytes=anchor_path.stat().st_size)
                del anchor
            if args.checkpoint_step and progress['updates'] == args.checkpoint_step and commit:
                torch.cuda.synchronize()
                start_save = time.monotonic()
                payload = cpu_tree(export_state())
                checkpoint = out / f'checkpoint_r{rank}.pt'
                torch.save(payload, checkpoint)
                with checkpoint.open('rb') as handle:
                    os.fsync(handle.fileno())
                event('checkpoint_saved', step=progress['updates'], seconds=time.monotonic()-start_save, bytes=checkpoint.stat().st_size)
                load_started = time.monotonic()
                loaded = torch.load(checkpoint, weights_only=False)
                stage.load_state_dict(loaded['model']); optimizer.load_state_dict(loaded['optimizer']); scheduler.load_state_dict(loaded['scheduler'])
                torch.cuda.synchronize()
                event('checkpoint_loaded', step=progress['updates'], seconds=time.monotonic()-load_started,
                      cache='warm_after_save', hashes_match=not args.audit_state or current_hashes()==payload.get('hashes'))
                del loaded
                if args.checkpoint_cold:
                    with checkpoint.open('rb') as handle:
                        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    load_started = time.monotonic()
                    loaded = torch.load(checkpoint, weights_only=False)
                    stage.load_state_dict(loaded['model']); optimizer.load_state_dict(loaded['optimizer']); scheduler.load_state_dict(loaded['scheduler'])
                    torch.cuda.current_stream().synchronize()
                    event('checkpoint_loaded_cold', step=progress['updates'], seconds=time.monotonic()-load_started,
                          cache='fadvise_dontneed_advisory', hashes_match=not args.audit_state or current_hashes()==payload.get('hashes'))
                    del loaded
                del payload
        if args.final_audit:
            event('final_audit', step=progress['updates'], hashes=current_hashes())
        event('stopped', step=progress['updates'], attempts=attempts)
        dist.barrier()
        if manager:
            manager.shutdown()
            manager._pg.abort(errored=False)
            manager._pg.shutdown()
        dist.destroy_process_group()
    finally:
        # Peers may see STOP at slightly different iterations; the supervisor
        # performs bounded teardown of only its own process group.
        if timeline: timeline.close()
        log.flush()


if __name__ == '__main__':
    main()
