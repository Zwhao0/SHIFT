"""Actual trainer guard function bodies over existing CPU-mapped anchor state.

This measures inventory/version/metadata complexity, not CUDA D2H latency.
Gradient tensor objects are proxies with the real model's shapes; one bucket
per parameter overestimates bucket traversal, and no gradient contents hash is
omitted by the production guard (it checks their identity/version only).
"""
import ast
import os
from pathlib import Path
import time
from types import SimpleNamespace


def anchor_guards(root, rank, handoff, emit):
    import torch
    from torch.utils._pytree import tree_flatten
    from shift_full.planned_train import digest
    from shift_full.reference_audit import validate_reference_anchor
    anchor = Path(os.environ.get('SHIFT_OWNERSHIP_GUARD_FIXTURE',root/'runtime/v3-reference-num01'))/f'anchor_r{rank}.pt'
    payload = torch.load(anchor, map_location='cpu', weights_only=False, mmap=True)
    validate_reference_anchor(payload, expected_rank=rank, expected_step=1)
    buffers = payload['all_buffers']
    params = {n:t for n,t in payload['model'].items() if isinstance(t,torch.Tensor) and n not in buffers}
    for t in params.values():
        # No contents are read by a guard; don't allocate another full model.
        t.main_grad = torch.empty((),dtype=torch.float32).expand(t.shape)
    stage = SimpleNamespace(parameters=lambda:list(params.values()),
        named_parameters=lambda:list(params.items()), named_buffers=lambda:list(buffers.items()))
    optimizer = SimpleNamespace(state_dict=lambda:payload['optimizer'])
    scheduler = SimpleNamespace(state_dict=lambda:payload['scheduler'])
    local_buffers = [SimpleNamespace(buckets=[SimpleNamespace(grad_data=t.main_grad) for t in params.values()])]
    ddp = SimpleNamespace(buffers=local_buffers, expert_parallel_buffers=[])
    progress = dict(updates=handoff.cut)
    def metadata():
        return dict(schema=2, all_buffers=dict(buffers), source_rank=rank,
            step=handoff.cut, progress=dict(progress), torch_rng=payload['torch_rng'].clone(),
            cuda_rng=payload['cuda_rng'].clone(), python_rng=payload['python_rng'],
            tracker_rng={k:v.clone() for k,v in payload['tracker_rng'].items()})
    export = SimpleNamespace(role_metadata=metadata)
    source = ast.parse((root/'shift_full/planned_train.py').read_text())
    owner = next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='capture_owner_guard')
    shadow = next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='shadow_recompute')
    inventory = next(n for n in ast.walk(shadow) if isinstance(n,ast.FunctionDef) and n.name=='inventory')
    unchanged = next(n for n in ast.walk(shadow) if isinstance(n,ast.FunctionDef) and n.name=='unchanged')
    scope = dict(torch=torch, digest=digest, stage=stage, optimizer=optimizer,
        scheduler=scheduler, ddp=ddp, local_buffers=local_buffers, export_state=export,
        progress=progress, handoff=handoff)
    exec(compile(ast.Module(body=[owner,inventory,unchanged],type_ignores=[]),
                 'actual_planned_train_guard_bodies','exec'),scope)
    begin=time.monotonic(); owner_guard=scope['capture_owner_guard']()
    capture_seconds=time.monotonic()-begin
    begin=time.monotonic()
    scope['versions']=[(t,t._version) for t in scope['inventory']()]
    scope['retained_tail']=[digest(dict(metadata(),planned_epoch=handoff.new_epoch,planned_role=1))]
    scope['retained_metadata']=[digest((scheduler.state_dict(),optimizer.state_dict()['param_groups']))]
    c_capture_extra_seconds=time.monotonic()-begin
    c_guard=scope['unchanged']
    measurements=[]
    for _ in range(3):
        started=time.monotonic(); assert owner_guard(); owner_seconds=time.monotonic()-started
        started=time.monotonic(); assert c_guard(); c_seconds=time.monotonic()-started
        measurements.append(dict(owner_seconds=owner_seconds,c_seal_seconds=c_seconds))
    # Reject a real main_grad tensor identity change using extracted functions.
    first=next(iter(params.values()))
    first.main_grad=first.main_grad.clone()
    assert not owner_guard() and not c_guard()
    # Rebaseline the intentional rejection check; original anchor is mmap private
    # and never written. Production guards are newly captured per boundary too.
    owner_guard=scope['capture_owner_guard']()
    scope['versions']=[(t,t._version) for t in scope['inventory']()]
    counts={}
    for key in ['model','optimizer','all_buffers','tracker_rng']:
        tensors=[t for t in tree_flatten(payload[key])[0] if isinstance(t,torch.Tensor)]
        counts[key]=dict(tensors=len(tensors),bytes=sum(t.numel()*t.element_size() for t in tensors))
    emit('anchor_guard_fixture', anchor=str(anchor), anchor_bytes=anchor.stat().st_size,
         counts=counts, parameters=len(params), proxy_buckets=len(params),
         inventory_tensors=len(scope['versions']), capture_seconds=capture_seconds,
         c_capture_extra_seconds=c_capture_extra_seconds,
         shape_proxy=bool(os.environ.get('SHIFT_OWNERSHIP_GUARD_FIXTURE')),
         measurements=measurements, mutation_rejected=True, cuda_d2h_measured=False)
    return owner_guard,c_guard
