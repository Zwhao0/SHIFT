"""Byte-identical SHA audit with bounded pinned staging; no hash/state cache.

Caller must keep tensors immutable for the call, as for the reference audit.
Experimental: training selects it only with SHIFT_DIGEST_ENGINE=staged.
"""
import hashlib
import os
import torch


def configured_engine():
    engine=os.environ.get("SHIFT_DIGEST_ENGINE","reference")
    if engine not in ("reference","staged"):raise ValueError("invalid SHIFT_DIGEST_ENGINE")
    return engine


def digest_reference(value):
    h=hashlib.sha256()
    def add(v):
        if isinstance(v,torch.Tensor):
            t=v.detach().cpu().contiguous()
            h.update(str((str(t.dtype),tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,dict):
            for key in sorted(v,key=lambda k:(type(k).__name__,str(k))):add(key);add(v[key])
        elif isinstance(v,(list,tuple)):
            h.update(str(type(v)).encode())
            for item in v:add(item)
        else:h.update(repr(v).encode()+b'\0')
    add(value)
    return h.hexdigest()


def digest_staged(value, staging_bytes=64*1024*1024):
    """Keep original metadata/order and bytes; synchronize once per bounded slab.

    Staging is per call and released on return. CPU values are read without an
    extra tobytes copy; CUDA values copy to pinned memory on their current stream.
    Noncontiguous CUDA inputs may require one contiguous device temporary.
    """
    if staging_bytes<=0:raise ValueError('staging_bytes must be positive')
    h=hashlib.sha256();pending=[];streams={};slab=None;used=0
    def flush():
        nonlocal used
        for stream in streams.values():stream.synchronize()
        for part in pending:h.update(part)
        pending.clear();streams.clear();used=0
    def add(v):
        nonlocal slab,used
        if isinstance(v,torch.Tensor):
            t=v.detach()
            pending.append(str((str(t.dtype),tuple(t.shape))).encode())
            if t.device.type=='cpu':
                pending.append(memoryview(t.contiguous().reshape(-1).view(torch.uint8).numpy()))
            elif t.device.type=='cuda':
                if slab is None:slab=torch.empty(staging_bytes,dtype=torch.uint8,pin_memory=True)
                flat=t.contiguous().reshape(-1).view(torch.uint8)
                offset=0
                while offset<flat.numel():
                    if used==staging_bytes:flush()
                    count=min(staging_bytes-used,flat.numel()-offset)
                    target=slab[used:used+count]
                    target.copy_(flat[offset:offset+count],non_blocking=True)
                    stream=torch.cuda.current_stream(t.device)
                    streams[t.device,stream.cuda_stream]=stream
                    pending.append(memoryview(target.numpy()))
                    offset+=count;used+=count
            else:raise ValueError('audit supports CPU/CUDA tensors')
        elif isinstance(v,dict):
            for key in sorted(v,key=lambda k:(type(k).__name__,str(k))):add(key);add(v[key])
        elif isinstance(v,(list,tuple)):
            pending.append(str(type(v)).encode())
            for item in v:add(item)
        else:pending.append(repr(v).encode()+b'\0')
    try:add(value);flush()
    finally:
        # Also finish copies if traversal rejects a later value.
        for stream in streams.values():stream.synchronize()
    return h.hexdigest()
