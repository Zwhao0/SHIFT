"""Isolated planned-handoff fork of the v1 Qwen MoE harness."""
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
from shift_full.compat import install_baby_event_fix, install_managed_callback_fix, bootstrap_legacy_core
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
    p.add_argument('--mode', choices=['shift', 'flat'], default='shift')
    p.add_argument('--backend', choices=['nccl', 'unified', 'blocking', 'baby', 'gloo', 'hybrid'], default='hybrid')
    p.add_argument('--transport', choices=['gpu', 'http'], default='gpu')
    p.add_argument('--steps', '--target-steps', dest='steps', type=int, default=1000000)
    p.add_argument('--seq-len', type=int, default=16)
    p.add_argument('--microbatches', type=int, default=2)
    p.add_argument('--hash-every', type=int, default=0)
    p.add_argument('--audit-state', action='store_true')
    p.add_argument('--measure-step', action='store_true')
    p.add_argument('--sync-quorum', action='store_true')
    p.add_argument('--fixed-batch', action='store_true')
    p.add_argument('--fp32', action='store_true')
    p.add_argument('--data', choices=['pattern','random'], default='pattern')
    p.add_argument('--reference-dump', action='store_true')
    p.add_argument('--reference-signatures', action='store_true')
    p.add_argument('--reference-ranks', default='0,1,2,3,4,5,6,7')
    p.add_argument('--reference-anchor')
    p.add_argument('--reference-anchor-step', type=int, default=1)
    p.add_argument('--audit-routing', action='store_true')
    p.add_argument('--checkpoint-step', type=int, default=0)
    p.add_argument('--checkpoint-cold', action='store_true')
    p.add_argument('--resume-checkpoint', action='store_true')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--bucket-size', type=int, default=2_097_152)
    p.add_argument('--communication-timeout', type=float, default=10)
    p.add_argument('--profile-timings', action='store_true')
    p.add_argument('--local-group-sync', action='store_true')
    p.add_argument('--eager-groups', action='store_true')
    p.add_argument('--bucket-hook', action='store_true')
    p.add_argument('--warmup-before-join', action='store_true')
    p.add_argument('--standby', choices=('cpu', 'gpu'))
    p.add_argument('--strict-fp32', action='store_true')
    p.add_argument('--planned-role', choices=['A','B','C'], required=True)
    p.add_argument('--planned-variant', choices=['p0','p1','p1d','p1f','p2','steady'], required=True)
    p.add_argument('--planned-source', choices=['A','B'], default='B')
    p.add_argument('--planned-inject',choices=['healthy','prepare-fail','ready-timeout','stale-state','relay-reject','slow-compute'],default='healthy')
    p.add_argument('--planned-slow-ms',type=float,default=250)
    p.add_argument('--planned-cycles',type=int,default=1)
    p.add_argument('--planned-hosts',required=True,help='JSON mapping of initial A/B/C to three host addresses')
    p.add_argument('--planned-shadow-recompute',action='store_true',help='diagnostic only; original B relay remains sole contribution')
    p.add_argument('--planned-ownership-recompute',action='store_true',help='experimental default-off single-epoch contribution ownership')
    args = p.parse_args()
    if args.planned_shadow_recompute and args.planned_ownership_recompute:
        p.error('choose one recompute mode')
    if (args.planned_shadow_recompute or args.planned_ownership_recompute) and not (args.planned_variant=='p2' and args.planned_cycles==1
            and args.planned_source=='B' and args.planned_inject=='slow-compute'
            and (args.reference_anchor or (args.planned_role=='C' and args.generation>0))
            and args.reference_signatures and args.audit_state and args.audit_routing
            and args.fixed_batch and args.sync_quorum and args.mode=='shift'
            and args.seq_len==512 and args.microbatches==2 and not args.fp32 and not args.strict_fp32
            and args.data=='pattern' and args.backend=='nccl' and args.bucket_size==67108864):
        p.error('shadow requires one p2 B-source slow-compute formal-anchor audited fixed-batch cycle')
    if args.planned_cycles<1 or (args.planned_cycles>1 and args.planned_variant!='p2'):p.error('cycles require p2 and a positive count')
    from shift_full.planned_state import role_hosts
    try:args.planned_hosts=json.loads(args.planned_hosts);role_hosts(args.planned_hosts,0)
    except (ValueError,TypeError):p.error('planned-hosts requires three distinct A/B/C addresses')
    args.planned_initial_role=args.planned_role
    args.planned_round=0
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
        row.setdefault('planned_round',args.planned_round)
        with lock:
            log.write(json.dumps(row, sort_keys=True) + '\n')

    def phase(name, started):
        event('phase', name=name, seconds=time.monotonic()-started)
        return time.monotonic()

    def standby_gate(mode):
        import resource
        event('standby_ready', mode=mode, cpu_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              cuda_allocated_bytes=torch.cuda.memory_allocated() if mode=='gpu' else 0,
              role_manifest=role_manifest() if mode=='gpu' else None)
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
    torch.cuda.set_device(device_rank)
    phase_started = phase('runtime_cuda_setup', phase_started)
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
    state, _, _ = megatron_adapter.core()
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    phase_started = phase('megatron_imports', phase_started)
    topology_out = out / f'topology_g{args.generation}'
    topology_out.mkdir(parents=True, exist_ok=True)
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
    torch.manual_seed(2026)
    random.seed(2026)
    full = load_model(Path(args.flash) / 'models/qwen_moe')
    phase_started = phase('full_model_load', phase_started)
    total_parameters = sum(v.numel() for v in full.parameters())
    if args.fp32:
        full.float()
    patch_moe(full, ep)
    stage = StageModel(full, state.get_pipeline_model_parallel_rank())
    tp_manifest = patch_tensor_parallel(stage, state.get_tensor_model_parallel_group())
    del full
    phase_started = phase('model_partition', phase_started)
    stage.cuda()
    ddp = megatron_adapter.wrap(stage, dense, expert_dp, args.bucket_size, config=stage.config)
    stage.config.no_sync_func = ddp.no_sync
    stage.config.finalize_model_grads_func = finalize_model_grads
    optimizer = torch.optim.AdamW(stage.parameters(), lr=args.lr, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(os.environ.get('SHIFT_PLANNED_SCHEDULER_STEP','1000')), gamma=0.98)
    phase_started = phase('cuda_ddp_optimizer', phase_started)
    def role_manifest():
        return dict(tp=2, pp=2, ep=2, etp=2, rank=rank,
                    shard_signature=digest([(n,tuple(p.shape),str(p.dtype)) for n,p in stage.named_parameters()]),
                    optimizer_groups=[dict(count=len(g['params']),lr=g['lr'],betas=g['betas'],eps=g['eps'],
                                           weight_decay=g['weight_decay']) for g in optimizer.param_groups])
    progress = {'updates': 0}
    warming_up = False
    shadow_active = [False]
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
                          near_ties_1e6=int((margin<1e-6).sum()),tokens=margin.numel(),
                          **({'shadow':True} if shadow_active[0] else {}))
                module.register_forward_hook(route_hook)
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    from shift_full.state_inventory import buffer_manifest, restore_buffers, tracker_snapshot, restore_tracker
    from shift_full.planned_state import validate_state
    from shift_full.reference_audit import reference_anchor_path, validate_reference_anchor
    rng_tracker = get_cuda_rng_tracker()
    reference_manager_state = None
    boundary_rng = None
    audit_joint_pending = False
    manager = None

    def rng_snapshot():
        return dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    python_rng=random.getstate(), tracker_rng=tracker_snapshot(rng_tracker))

    def current_hashes(rng=None):
        torch.cuda.current_stream().synchronize()
        rng = rng if rng is not None else rng_snapshot()
        return dict(model=digest(stage.state_dict()), buffers=digest(dict(stage.named_buffers())), optimizer=digest(optimizer.state_dict()),
                    scheduler=digest(scheduler.state_dict()), progress=digest(progress),
                    cpu_rng=digest(rng['torch_rng']), cuda_rng=digest(rng['cuda_rng']),
                    python_rng=digest(rng['python_rng']), tracker_rng=digest(rng['tracker_rng']))

    def export_state():
        nonlocal audit_joint_pending
        event('snapshot_begin',step=progress['updates'])
        torch.cuda.current_stream().synchronize()
        step = manager.current_step() if manager else progress['updates']
        assert step == progress['updates'], (step, progress)
        audit_joint_pending = True
        rng = boundary_rng['rng'] if boundary_rng and boundary_rng['step'] == step else rng_snapshot()
        frozen = boundary_rng if boundary_rng and boundary_rng['step'] == step else None
        all_buffers = frozen['buffers'] if frozen else {n:b.detach().clone() for n,b in stage.named_buffers()}
        if frozen: frozen['copy_ready'].synchronize()
        model_state = stage.state_dict()
        for name,value in all_buffers.items():
            if name in model_state:model_state[name]=value
        payload = dict(schema=2, model=model_state, all_buffers=all_buffers,
                       optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict(), progress=dict(progress),
                       **rng, source_replica=args.replica,
                       source_rank=rank, step=step)
        payload['hashes'] = current_hashes(rng) if args.audit_state else {}
        if args.audit_state:payload['hashes'].update(model=digest(model_state),buffers=digest(all_buffers))
        if args.transport == 'http':
            payload = cpu_tree(payload)
        event('snapshot', step=step, hashes=payload['hashes'],
              optimizer_tensors=tensor_count(payload['optimizer'].get('state', {})))
        return payload

    def load_state(payload):
        nonlocal audit_joint_pending
        load_started = time.monotonic()
        validate_state(payload, rank=rank, audit=args.audit_state)
        assert payload['step'] == payload['progress']['updates']
        assert payload['step'] == (manager.current_step() if manager else payload['step'])
        stage.load_state_dict(payload['model'], strict=True)
        restore_buffers(stage,payload['all_buffers'])
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
        progress.clear();progress.update(payload['progress'])
        audit_joint_pending = True
        torch.set_rng_state(payload['torch_rng'])
        torch.cuda.set_rng_state(payload['cuda_rng'])
        random.setstate(payload['python_rng'])
        restore_tracker(rng_tracker,payload['tracker_rng'])
        torch.cuda.current_stream().synchronize()
        loaded_at = time.monotonic()
        event('state_loaded',step=payload['step'],load_seconds=loaded_at-load_started,
              source_replica=payload['source_replica'],source_rank=payload['source_rank'])
        hashes = current_hashes() if args.audit_state else {}
        matches = hashes == payload['hashes']
        event('restored', step=payload['step'], load_seconds=loaded_at-load_started, audit_seconds=time.monotonic()-loaded_at, source_replica=payload['source_replica'],
              source_rank=payload['source_rank'], hashes_match=matches,
              expected_hashes=payload['hashes'], actual_hashes=hashes,restored_learning_rates=[g['lr'] for g in optimizer.param_groups],
              optimizer_tensors=tensor_count(optimizer.state_dict().get('state', {})),
              cuda_allocated_bytes=torch.cuda.memory_allocated(device_rank), cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device_rank))
        assert matches, 'Restored state differs from source snapshot'

    if args.reference_anchor:
        anchor_started=time.monotonic()
        anchor_path=reference_anchor_path(args.reference_anchor,rank,args.reference_anchor_step)
        anchor=torch.load(anchor_path,map_location='cpu',weights_only=False,mmap=True)
        validate_reference_anchor(anchor,expected_rank=rank,expected_step=args.reference_anchor_step)
        stage.load_state_dict(anchor['model'],strict=True);restore_buffers(stage,anchor['all_buffers'])
        optimizer.load_state_dict(anchor['optimizer']);scheduler.load_state_dict(anchor['scheduler'])
        progress.clear();progress.update(anchor['progress'])
        torch.set_rng_state(anchor['torch_rng']);torch.cuda.set_rng_state(anchor['cuda_rng'])
        random.setstate(anchor['python_rng']);restore_tracker(rng_tracker,anchor['tracker_rng'])
        reference_manager_state=dict(anchor['manager_state'])
        if current_hashes()!=anchor['hashes']:raise RuntimeError('schema-2 reference anchor restoration mismatch')
        event('reference_anchor_loaded',step=progress['updates'],schema=2,hashes_match=True)
        del anchor
        phase_started=phase('reference_anchor_load',anchor_started)

    if args.warmup_before_join or ((args.planned_shadow_recompute or args.planned_ownership_recompute) and args.planned_role=='C'):
        warm_started=time.monotonic()
        saved_rng=rng_snapshot()
        saved_buffers={n:b.detach().clone() for n,b in stage.named_buffers()}
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
        torch.set_rng_state(saved_rng['torch_rng']);torch.cuda.set_rng_state(saved_rng['cuda_rng']);random.setstate(saved_rng['python_rng']);restore_tracker(rng_tracker,saved_rng['tracker_rng']);restore_buffers(stage,saved_buffers)
        if before_hash is not None: assert before_hash==digest(stage.state_dict()),'warmup changed model state'
        phase_started=phase('local_cuda_warmup',warm_started)

    # Distinct logical-rid RNG streams; model initialization is already complete.
    if not args.reference_anchor:
        torch.manual_seed(70000+100*args.replica+rank);random.seed(80000+100*args.replica+rank)
    export_state.role_metadata=lambda:dict(schema=2,all_buffers=dict(stage.named_buffers()),source_rank=rank,step=progress['updates'],progress=dict(progress),**rng_snapshot())
    from shift_full.planned_handoff import PlannedHandoff
    if args.planned_variant=='p2':
        from shift_full.overlap_handoff import OverlapHandoff as PlannedHandoff
    if args.planned_ownership_recompute:
        from shift_full.ownership_handoff import OwnershipHandoff as PlannedHandoff
    handoff=PlannedHandoff(args,rank,event,export_state,load_state,progress,role_manifest,rng_snapshot)
    forward_counter=[0]
    def count_forward(*_):forward_counter[0]+=1
    stage.register_forward_pre_hook(count_forward)
    if args.planned_ownership_recompute:
        # Include native local DDP/TP/PP/EP handles in the alias exclusion set.
        handoff.local_pgs=[dist.group.WORLD,ep,expert_dp,dense,
            state.get_tensor_model_parallel_group(),state.get_pipeline_model_parallel_group()]
        def source_contribution(samples,gradients):
            torch.cuda.current_stream().synchronize()
            tail=dict(export_state.role_metadata(),planned_epoch=handoff.new_epoch,planned_role=1)
            return dict(sample_ids=samples,gradient_hash=digest(gradients),
                parameter_gradient_hash=digest({n:p.main_grad for n,p in stage.named_parameters()}),tail_hash=digest(tail))
        handoff.source_contribution=source_contribution
        def capture_owner_guard():
            from torch.utils._pytree import tree_flatten
            def inventory():
                return [t for t in tree_flatten((list(stage.parameters()),dict(stage.named_buffers()),
                    optimizer.state_dict(),[b.grad_data for buf in list(ddp.buffers)+list(ddp.expert_parallel_buffers) for b in buf.buckets],
                    {n:p.main_grad for n,p in stage.named_parameters()}))[0] if isinstance(t,torch.Tensor)]
            versions=[(t,t._version) for t in inventory()]
            def metadata():
                return digest((export_state.role_metadata(),scheduler.state_dict(),optimizer.state_dict()['param_groups']))
            expected=metadata()
            def unchanged():
                tensors=inventory()
                return (len(tensors)==len(versions) and all(t is old and t._version==v for t,(old,v) in zip(tensors,versions))
                        and metadata()==expected)
            return unchanged
        handoff.capture_owner_guard=capture_owner_guard
    if args.planned_shadow_recompute and args.planned_role=='B':
        handoff.shadow_parameter_hash=lambda:digest({n:p.main_grad for n,p in stage.named_parameters()})
    if args.planned_role=='C':
        # Defined before ready_receiver, using only already-initialized local
        # state. No late Manager/finalizer/apply_tail closure is reachable here.
        def shadow_recompute(expected_hashes, *, retain=False):
            from shift_full.planned_shadow import validate_descriptor, shadow_transaction
            key='ownership_descriptor' if retain else 'shadow_descriptor'
            descriptor=json.loads(handoff.coord.get(f'{key}/{rank}'))
            sample_ids=validate_descriptor(descriptor,rank=rank,step=progress['updates'],epoch=handoff.new_epoch,
                seq_len=args.seq_len,microbatches=args.microbatches,data=args.data,vocab_size=stage.hf_config.vocab_size)
            if descriptor['local_data_rank']!=state.get_data_parallel_rank():
                raise ValueError('shadow local data role mismatch')
            owner=threading.get_ident();started=time.monotonic()
            saved_rng=rng_snapshot();saved_buffers={n:b.detach().clone() for n,b in stage.named_buffers()}
            saved_routes=dict(route_calls)
            local_buffers=list(ddp.buffers)+list(ddp.expert_parallel_buffers)
            def clear():
                optimizer.zero_grad(set_to_none=True);ddp.zero_grad_buffer()
            def execute():
                shadow_active[0]=not retain;route_calls.clear()
                if retain:event('planned_ownership_recompute_begin',step=handoff.cut)
                batches=[]
                for sample in sample_ids:
                    if args.data=='random':
                        generator=torch.Generator(device='cuda').manual_seed(100_000+sample)
                        tokens=torch.randint(0,stage.hf_config.vocab_size,(1,args.seq_len+1),device='cuda',generator=generator)
                    else:
                        tokens=(torch.arange(args.seq_len+1,device='cuda').view(1,-1)+sample%8)%32+100
                    batches.append(dict(input_ids=tokens[:,:-1].contiguous(),next_token_labels=tokens[:,1:].contiguous()))
                get_forward_backward_func()(forward_step_func=forward_step,data_iterator=iter(batches),model=ddp,
                    num_microbatches=len(sample_ids),seq_length=args.seq_len+stage.extra_positions(1),micro_batch_size=1)
            versions=[]
            retained_tail=[None]
            retained_metadata=[None]
            initial_audit=[None]
            def inventory():
                from torch.utils._pytree import tree_flatten
                return [t for t in tree_flatten((list(stage.parameters()),dict(stage.named_buffers()),optimizer.state_dict(),
                    [b.grad_data for buf in local_buffers for b in buf.buckets],
                    {n:p.main_grad for n,p in stage.named_parameters()}))[0] if isinstance(t,torch.Tensor)]
            def capture():
                torch.cuda.current_stream().synchronize();ready=time.monotonic()
                tail=dict(export_state.role_metadata(),planned_epoch=handoff.new_epoch,planned_role=1)
                result=dict(sample_ids=sample_ids,descriptor=descriptor,
                    gradient_hash=digest([b.grad_data for buf in local_buffers for b in buf.buckets]),
                    parameter_gradient_hash=digest({n:p.main_grad for n,p in stage.named_parameters()}),tail_hash=digest(tail))
                if retain:
                    result.update(initial_audit=initial_audit[0],started_mono=started)
                    versions.extend((t,t._version) for t in inventory())
                    retained_tail[0]=result['tail_hash']
                    retained_metadata[0]=digest((scheduler.state_dict(),optimizer.state_dict()['param_groups']))
                event('planned_ownership_computed' if retain else 'planned_shadow_computed',step=handoff.cut,preouter_seconds=ready-started,
                    capture_seconds=time.monotonic()-ready,gradient_hash=result['gradient_hash'],
                    parameter_gradient_hash=result['parameter_gradient_hash'],tail_hash=result['tail_hash'],
                    sample_ids=sample_ids,descriptor=descriptor,owner_thread=owner,
                    cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
                return result
            def restore():
                torch.cuda.current_stream().synchronize()
                shadow_active[0]=False;route_calls.clear();route_calls.update(saved_routes)
                restore_buffers(stage,saved_buffers)
                torch.set_rng_state(saved_rng['torch_rng']);torch.cuda.set_rng_state(saved_rng['cuda_rng'])
                random.setstate(saved_rng['python_rng']);restore_tracker(rng_tracker,saved_rng['tracker_rng'])
            def verify():
                hashes=current_hashes()
                buckets=[b for buf in local_buffers for b in buf.buckets]
                clean=all(not b.params_with_grad and b.communication_handle is None and not b.communication_issued for b in buckets)
                clean=clean and all(buf.is_last_microbatch for buf in local_buffers)
                clean=clean and all(p.grad is None and not p.grad_added_to_main_grad for p in stage.parameters() if p.requires_grad)
                clean=clean and all(not bool(buf.grad_data.count_nonzero()) for buf in local_buffers)
                event('planned_shadow_rollback',step=handoff.cut,hashes=hashes,hashes_match=hashes==expected_hashes,
                    ddp_clean=clean,seconds=time.monotonic()-started,owner_thread=threading.get_ident())
                if hashes!=expected_hashes or not clean or threading.get_ident()!=owner:
                    raise RuntimeError('shadow rollback state/DDP/owner mismatch')
            if retain:
                from shift_full.ownership_runtime import retain_contribution
                def audit_initial():
                    before=time.monotonic();hashes=current_hashes()
                    initial_audit[0]=dict(hashes=hashes,seconds=time.monotonic()-before,
                        boundary_step=handoff.cut,before_forward=True)
                    event('planned_ownership_initial_audit',step=handoff.cut,hashes=hashes,
                        hashes_match=hashes==expected_hashes,seconds=time.monotonic()-before,before_forward=True)
                    if hashes!=expected_hashes:raise RuntimeError('ownership initial state mismatch')
                def unchanged():
                    tail=dict(export_state.role_metadata(),planned_epoch=handoff.new_epoch,planned_role=1)
                    tensors=inventory()
                    return (len(tensors)==len(versions) and
                            all(t is previous and t._version==v for t,(previous,v) in zip(tensors,versions)) and
                            digest(tail)==retained_tail[0] and progress['updates']==handoff.cut and
                            digest((scheduler.state_dict(),optimizer.state_dict()['param_groups']))==retained_metadata[0])
                try:
                    return retain_contribution(config=stage.config,ddp=ddp,audit_initial=audit_initial,
                        execute=execute,capture=capture,unchanged=unchanged,restore=restore,clear=clear,verify=verify)
                finally:
                    shadow_active[0]=False
            return shadow_transaction(config=stage.config,ddp=ddp,execute=execute,capture=capture,
                restore=restore,clear=clear,verify=verify)
        if args.planned_shadow_recompute:handoff.shadow_recompute=shadow_recompute
        # Selected only by the opt-in production coordinator above.
        handoff.ownership_recompute=lambda expected:shadow_recompute(expected,retain=True)
    if args.planned_role=='C':
        if not handoff.ready_receiver():
            event('stopped',step=progress['updates'],attempts=0,reason='planned_cancelled')
            dist.barrier();dist.destroy_process_group();return

    from torchft import Manager, ProcessGroupGloo
    from torchft.manager import WorldSizeMode
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

    def make_manager():
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
        if handoff.activated and not getattr(handoff,'fallback',False):
            pg=handoff.candidate
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
        manager = Manager(pg=pg, load_state_dict=handoff.forbidden_load if getattr(handoff,'building_manager',False) else load_state, state_dict=export_state,
                          min_replica_size=2, world_size_mode=WorldSizeMode.FIXED_WITH_SPARES, use_async_quorum=not args.sync_quorum,
                          init_sync=False, timeout=timedelta(seconds=90),
                          quorum_timeout=timedelta(seconds=180), connect_timeout=timedelta(seconds=20),
                          hostname=args.host_ip, lighthouse_addr=args.lighthouse,
                          store_addr=args.host_ip,store_port=getattr(handoff,'manager_store_port',31600+handoff.offset) if handoff.activated else int(os.environ['MASTER_PORT']),
                          replica_id=(f'shift-own-{handoff.ticket().nonce}-{"AB" if handoff.fallback else "AC"}-{args.replica}'
                                      if args.planned_ownership_recompute and handoff.activated else f'shift{args.replica}'),
                          checkpoint_transport=transport, max_retries=8)
        if reference_manager_state is not None and not handoff.gate.epoch:
            manager.load_state_dict(reference_manager_state)
        elif handoff.gate.epoch:
            manager.load_state_dict(dict(step=progress['updates'],batches_committed=progress['updates']*2))

        manager._shift_epoch=handoff.gate.epoch
        return manager
    if args.mode=='shift' and not (args.planned_ownership_recompute and args.planned_role=='C'): manager=make_manager()

    if args.resume_checkpoint and args.generation > 0:
        resume_started = time.monotonic()
        payload = torch.load(out / f'checkpoint_r{rank}.pt', map_location='cpu', weights_only=False)
        if manager:
            manager.load_state_dict(dict(step=payload['step'], batches_committed=payload['step']*2))
        load_state(payload)
        event('checkpoint_resumed', step=progress['updates'], seconds=time.monotonic()-resume_started)

    if args.planned_role=='C' and not args.planned_ownership_recompute:
        event('planned_epoch_activated',step=progress['updates'],epoch=1)
    phase_started = phase('manager_init', phase_started)
    # The native finalizer waits intra-replica buckets, then sums tied PP grads.
    # Both dense and routed-expert buffers need the second reduction. In this
    # vendor, expert gradients use dense-DP normalization, retained unchanged.
    original_finish = ddp.finish_grad_sync
    buffers = list(ddp.buffers) + list(ddp.expert_parallel_buffers)
    cross_calls = [0]
    cross_timings = {}

    def finish_grad_sync(_self, *, local_finished=False):
        nonlocal manager
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
            pending.append((bucket.grad_data, value, manager.allreduce(value)))
            dispatch_seconds += time.monotonic()-before
        relay = args.planned_variant=='p2' and handoff.is_relay_step()
        if local_finished:
            handoff.authorize(manager)
            for buffer in buffers:
                for bucket in buffer.buckets:enqueue(bucket)
            handoff.emit('planned_ac_outer_begin',step=progress['updates'])
            intra_done=time.monotonic()
        elif relay:
            if args.planned_role!='C': original_finish()
            manager,retired=handoff.at_gradient(manager,[b.grad_data for buf in buffers for b in buf.buckets],make_manager,sample_ids,apply_tail)
            if retired:
                from shift_full.overlap_handoff import RetiredReplica
                raise RetiredReplica()
            for buffer in buffers:
                for bucket in buffer.buckets:enqueue(bucket)
            handoff.emit('planned_ac_outer_begin',step=progress['updates'])
            intra_done=time.monotonic()
        else:
            patched = []
            if args.bucket_hook:
                # Preserve both native DDP and buffer finalizers. Inject only after
                # each bucket's own finish has established local gradient readiness.
                for buffer in buffers:
                    for bucket in buffer.buckets:
                        original = bucket.finish_grad_sync
                        previous = bucket.__dict__.get('finish_grad_sync')
                        def finish_bucket(this, native=original):
                            native()
                            enqueue(this)
                        patched.append((bucket, previous))
                        bucket.finish_grad_sync = types.MethodType(finish_bucket, bucket)
            try:
                original_finish()
            finally:
                for bucket, previous in patched:
                    if previous is None: del bucket.finish_grad_sync
                    else: bucket.finish_grad_sync = previous
            intra_done = time.monotonic()
            if not args.bucket_hook:
                for buffer in buffers:
                    for bucket in buffer.buckets: enqueue(bucket)
        waiting = time.monotonic()
        for target, cpu, work in pending:
            work.wait()
            if args.backend in ('gloo', 'hybrid'):
                target.copy_(cpu)
        finished = time.monotonic()
        cross_timings.update(intra_finish_cpu_seconds=intra_done-entered,
                             cross_d2h_cpu_seconds=d2h_seconds, cross_dispatch_cpu_seconds=dispatch_seconds,
                             cross_wait_h2d_cpu_seconds=finished-waiting)
        if gpu_done:
            gpu_done.record(); gpu_done.synchronize()
            cross_timings['cross_gpu_seconds'] = gpu_started.elapsed_time(gpu_done)/1000
        cross_calls[0] += len(pending)

    def apply_tail(tail):
        validate_state(tail,rank=rank,step=progress['updates'],full=False,epoch=handoff.new_epoch)
        restore_buffers(stage,tail['all_buffers'])
        torch.set_rng_state(tail['torch_rng']);torch.cuda.set_rng_state(tail['cuda_rng']);random.setstate(tail['python_rng']);restore_tracker(rng_tracker,tail['tracker_rng'])
        return dict(tail,all_buffers=dict(stage.named_buffers()),progress=dict(progress),**rng_snapshot())

    if args.mode == 'shift':
        ddp.finish_grad_sync = types.MethodType(finish_grad_sync, ddp)
    if args.planned_ownership_recompute and args.planned_role=='C':
        manager=handoff.finish_receiver(make_manager)
        if manager is None:
            handoff.start_retired_cleanup();handoff.finish_retired_cleanup()
            event('stopped',step=progress['updates'],attempts=0,reason='ownership_pre_activation_AB')
            handoff.cancel_cleanup_on_exit();dist.barrier();dist.destroy_process_group();return
    initial = (handoff.initial_audit['hashes'] if args.planned_ownership_recompute and args.planned_role=='C'
               else current_hashes() if args.audit_state else {})
    phase_started = phase('ownership_activation' if args.planned_ownership_recompute and args.planned_role=='C' else 'initial_audit', phase_started)
    event('ready', digest_engine=os.environ.get('SHIFT_DIGEST_ENGINE','reference'), physical_rank=args.replica * 8 + rank,
          layout='node_local_pp', tp=2, pp=2, ep=2, local_dense_dp=dist.get_world_size(dense),
          local_expert_dp=dist.get_world_size(expert_dp), model_id=MODEL_ID, model_revision=REVISION,
          full_parameters=total_parameters, local_parameters=sum(v.numel() for v in stage.parameters()),
          tp_manifest=tp_manifest, hashes=initial, torch_version=torch.__version__,
          initial_audit_receipt=handoff.initial_audit if args.planned_ownership_recompute and args.planned_role=='C' else None,
          cuda_version=torch.version.cuda, nccl_version=torch.cuda.nccl.version(),
          global_rank=dist.get_rank(), world_size=dist.get_world_size(), legacy_core_namespace=legacy_namespace,
          transport=args.transport, cross_gradient=args.backend, mode=args.mode, fp32=args.fp32,
          cuda_device=device_rank,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
          role_manifest=role_manifest(), state_schema=2,
          state_inventory=dict(buffers=buffer_manifest(stage),tracker_keys=sorted(rng_tracker.get_states()),
                               tracker_class=type(rng_tracker).__name__,snapshot='boundary buffers/RNG; model/Adam read-only before relay'),
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
    attempts = 0
    last_commit_mono = None
    recovery_requested = False
    def requested_stop():
        limit=out/'TRAIN_UNTIL'
        if limit.exists() and progress['updates']>=int(limit.read_text()):return True
        marker = out / 'STOP'
        if not marker.exists():
            return False
        text = marker.read_text().strip()
        return not text or progress['updates'] >= int(text)
    def advance_round(wait=False):
        nonlocal handoff,manager
        import gc
        r=handoff.round+1
        if r>=args.planned_cycles:return False
        raw=handoff.coord_raw
        end=time.monotonic()+180
        while not (raw.check(['advance']) and int(raw.get('advance'))>=r):
            if not wait:return False
            if time.monotonic()>end:raise RuntimeError('next round was not requested')
            time.sleep(.02)
        old=handoff;old.finish_retired_cleanup()
        args._coord_raw=raw;args.planned_round=r
        # Only the second logical DP slot changes physical owner.
        initial_role=args.planned_initial_role
        args.planned_role='A' if initial_role=='A' else (initial_role if r%2==0 else ('C' if initial_role=='B' else 'B'))
        handoff=PlannedHandoff(args,rank,event,export_state,load_state,progress,role_manifest,rng_snapshot)
        del old;gc.collect()
        event('planned_round_started',step=progress['updates'],role=args.planned_role,epoch=handoff.gate.epoch,cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),threads=len(__import__('os').listdir('/proc/self/task')))
        if args.planned_role=='C':
            if not handoff.ready_receiver():raise RuntimeError('cycle target cancelled; coordinated stop required')
            manager=make_manager();event('planned_epoch_activated',step=progress['updates'],epoch=handoff.gate.epoch)
        return True
    try:
        while not requested_stop() and time.monotonic() < deadline and progress['updates'] < args.steps:
            if args.planned_cycles>1:advance_round()
            handoff.check()
            retained_step=(args.planned_ownership_recompute and args.planned_role=='C'
                and handoff.contribution.state=='activated' and not handoff.contribution.finish_started)
            iteration_started = handoff.contribution.result['started_mono'] if retained_step else time.monotonic()
            torch.cuda.current_stream().synchronize()
            boundary_done = time.monotonic()
            if not handoff.activated:
                boundary_rng = dict(step=progress['updates'], rng=rng_snapshot(),
                                    buffers={n:b.detach().clone() for n,b in stage.named_buffers()},copy_ready=torch.cuda.Event())
                boundary_rng['copy_ready'].record()
                manager,retired=handoff.boundary(manager)
                if retired: break
                if manager is None:
                    manager=make_manager()
                    event('planned_epoch_activated',step=progress['updates'],epoch=1)
            if manager and not retained_step:
                boundary_rng = dict(step=progress['updates'], rng=rng_snapshot(),
                                    buffers={n:b.detach().clone() for n,b in stage.named_buffers()},copy_ready=torch.cuda.Event())
                boundary_rng['copy_ready'].record()
                if args.generation>0 and not recovery_requested:
                    event('recovery_request',step=progress['updates'])
                    recovery_requested=True
                handoff.authorize(manager)
                manager.start_quorum(allow_heal=False) if args.planned_ownership_recompute and handoff.activated else manager.start_quorum()
                if args.fixed_batch:
                    if not args.sync_quorum:
                        raise ValueError('fixed batch requires synchronous quorum')
                    manager.wait_quorum()
            step = progress['updates']
            # num_participants waits for quorum. Avoid it before forward in the
            # dynamic-batch async path; data assignment does not depend on it.
            participants = manager.num_participants() if manager and (args.sync_quorum or args.fixed_batch) else 2 if not manager else None
            if participants!=2:raise RuntimeError('planned handoff requires exactly two contributors')
            event('begin', step=step, participants=participants,
                  boundary_sync_seconds=boundary_done-iteration_started, quorum_seconds=time.monotonic()-boundary_done)
            if not retained_step:
                optimizer.zero_grad(set_to_none=True)
                ddp.zero_grad_buffer()
            if args.mode == 'flat':
                data_rank = state.get_data_parallel_rank()
            else:
                replica_index = manager._participating_replica_rank if args.fixed_batch else args.replica
                data_rank = (replica_index or 0) * 2 + state.get_data_parallel_rank()
            count = args.microbatches * (2 // participants if args.fixed_batch else 1)
            dense_size = 2 * participants if args.fixed_batch else 4
            batches = []
            sample_ids = []
            for micro in range(0 if retained_step else count):
                sample = step * (args.microbatches * 4) + micro * dense_size + data_rank
                sample_ids.append(sample)
                if args.data == 'random':
                    generator = torch.Generator(device='cuda').manual_seed(100_000 + sample)
                    tokens = torch.randint(0, stage.hf_config.vocab_size, (1, args.seq_len + 1),
                                           device='cuda', generator=generator)
                else:
                    tokens = (torch.arange(args.seq_len + 1, device='cuda').view(1, -1)
                              + sample % 8) % 32 + 100
                batches.append(dict(input_ids=tokens[:, :-1].contiguous(),
                                    next_token_labels=tokens[:, 1:].contiguous()))
            if retained_step:sample_ids=handoff.contribution.result['sample_ids']
            if args.reference_dump or args.reference_signatures: event('batch_plan',step=step,sample_ids=sample_ids)
            if ((args.planned_shadow_recompute or args.planned_ownership_recompute) and args.planned_role=='B'
                    and handoff.prepared_step==progress['updates'] and handoff.is_relay_step()):
                descriptor=dict(schema=1,rank=rank,step=step,epoch=handoff.new_epoch,
                    participants=participants,replica_index=replica_index,local_data_rank=state.get_data_parallel_rank(),
                    sample_ids=sample_ids,seq_len=args.seq_len,microbatches=args.microbatches,
                    data=args.data,vocab_size=stage.hf_config.vocab_size,seed_base=100_000)
                if args.planned_ownership_recompute:
                    handoff.reserve_descriptor(manager,descriptor,handoff.local_pgs)
                else:
                    handoff.coord.set(f'shadow_descriptor/{rank}',json.dumps(descriptor))
                    event('planned_shadow_descriptor',step=step,descriptor=descriptor)
            started = handoff.contribution.result['started_mono'] if retained_step else time.monotonic()
            before_calls = cross_calls[0]
            before_forward=0 if retained_step else forward_counter[0]
            if args.planned_inject=='slow-compute' and args.planned_role=='B' and not handoff.activated and handoff.request_seen:
                delay_started=time.monotonic();time.sleep(args.planned_slow_ms/1000)
                event('planned_compute_delay',step=step,seconds=time.monotonic()-delay_started)
            try:
                if retained_step:
                    metrics=[]
                    handoff.contribution.finish(native_finalizer=stage.config.finalize_model_grads_func,
                        outer=lambda:finish_grad_sync(ddp,local_finished=True))
                elif args.planned_variant=='p2' and args.planned_role=='C' and handoff.is_relay_step():
                    metrics=[]
                    stage.config.finalize_model_grads_func([ddp])
                else:
                    metrics = schedule(forward_step_func=forward_step, data_iterator=iter(batches),
                                       model=ddp, num_microbatches=count,
                                       seq_length=args.seq_len + stage.extra_positions(1), micro_batch_size=1)
            except Exception as exc:
                from shift_full.overlap_handoff import RetiredReplica
                if not isinstance(exc,RetiredReplica):raise
                manager=None
                if args.planned_cycles>1 and advance_round(wait=True):continue
                break
            for parameter in stage.parameters():
                parameter.grad = parameter.main_grad
            # A bad numeric result must never be counted as a valid update.
            schedule_done = time.monotonic()
            finite = bool(torch.stack([torch.isfinite(b.grad_data).all() for buf in buffers for b in buf.buckets]).all())
            if not finite:
                if manager:
                    manager.report_error(RuntimeError('nonfinite gradients'))
                else:
                    raise RuntimeError('nonfinite gradients in baseline')
            gradient_checked = time.monotonic()
            gradients = cpu_tree({name: p.main_grad for name, p in stage.named_parameters()}) if args.reference_dump and args.replica == 0 and rank in {int(r) for r in args.reference_ranks.split(',')} else None
            decision_started = time.monotonic()
            handoff.authorize(manager)
            commit = manager.should_commit() if manager else True
            decision_done = time.monotonic()
            gradient_signature_hash = digest({name:p.main_grad for name,p in stage.named_parameters()}) if args.reference_signatures and commit else None
            timings = dict(schedule_seconds=schedule_done-started, gradient_check_seconds=gradient_checked-schedule_done,
                           decision_seconds=decision_done-decision_started)
            attempts += 1
            if commit:
                optimizer.step()
                scheduler.step()
                if args.measure_step:
                    torch.cuda.current_stream().synchronize()
                progress['updates'] += 1
                actual_participants=manager.num_participants() if manager else 2
                contributed=manager.is_participating() if manager else True
                audit_joint=args.audit_state and audit_joint_pending and actual_participants>=2 and contributed
                hashes = current_hashes() if ((args.hash_every and progress['updates'] % args.hash_every == 0) or audit_joint) else {}
                if audit_joint: audit_joint_pending=False
                if gradients is not None:
                    torch.save(dict(gradients=gradients, model=cpu_tree(stage.state_dict()),
                                    optimizer=cpu_tree(optimizer.state_dict()), scheduler=scheduler.state_dict()),
                               out / f'reference_step{progress["updates"]}_r{rank}.pt')
                if args.reference_signatures:
                    manager_step=manager.current_step() if manager else progress['updates']
                    if manager_step!=progress['updates']:raise RuntimeError('reference signature manager step mismatch')
                    signature_hashes=current_hashes();signature_hashes['gradients']=gradient_signature_hash
                    event('reference_signature',schema=1,step=progress['updates'],input_step=step,manager_step=manager_step,progress=dict(progress),sample_ids=sample_ids,hashes=signature_hashes)
                committed_at=time.monotonic()
                event('commit', commit_interval_seconds=None if last_commit_mono is None else committed_at-last_commit_mono,
                      step=progress['updates'], updates=progress['updates'],
                      participants=actual_participants, contributed=contributed, hashes=hashes, model_hash=hashes.get('model'),
                      sample_ids=sample_ids, fixed_batch=args.fixed_batch,
                      global_batch=args.microbatches*4 if args.fixed_batch else args.microbatches*2*actual_participants,
                      optimizer_hash=hashes.get('optimizer'), scheduler_hash=hashes.get('scheduler'),
                      optimizer_tensors=tensor_count(optimizer.state_dict().get('state', {})),
                      learning_rates=[g['lr'] for g in optimizer.param_groups],
                      local_forward_calls=forward_counter[0]-before_forward,
                      cross_calls=cross_calls[0] - before_calls, finite_gradients=finite,
                      seconds=time.monotonic() - iteration_started, compute_seconds=time.monotonic()-started,
                      loss=[float(m['loss']) for m in metrics] if metrics else [], **timings, **cross_timings, **handoff.commit_fields(manager))
                last_commit_mono=committed_at
                handoff.committed()
            else:
                event('skipped', step=step, error=str(manager.errored()), **timings)
            if manager:
                assert manager.current_step() == progress['updates']
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
                load_state(loaded)
                torch.cuda.synchronize()
                event('checkpoint_loaded', step=progress['updates'], seconds=time.monotonic()-load_started,
                      cache='warm_after_save', hashes_match=not args.audit_state or current_hashes()==payload.get('hashes'))
                del loaded
                if args.checkpoint_cold:
                    with checkpoint.open('rb') as handle:
                        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    load_started = time.monotonic()
                    loaded = torch.load(checkpoint, weights_only=False)
                    load_state(loaded)
                    torch.cuda.current_stream().synchronize()
                    event('checkpoint_loaded_cold', step=progress['updates'], seconds=time.monotonic()-load_started,
                          cache='fadvise_dontneed_advisory', hashes_match=not args.audit_state or current_hashes()==payload.get('hashes'))
                    del loaded
                del payload
        handoff.finish_retired_cleanup()
        event('stopped', step=progress['updates'], attempts=attempts)
        dist.barrier()
        if manager:
            manager.shutdown()
            manager._pg.abort(errored=False)
            manager._pg.shutdown()
        dist.destroy_process_group()
    finally:
        handoff.cancel_cleanup_on_exit()
        # Peers may see STOP at slightly different iterations; the supervisor
        # performs bounded teardown of only its own process group.
        log.flush()


if __name__ == '__main__':
    main()
