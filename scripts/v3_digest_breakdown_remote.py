"""One bounded real-anchor audit decomposition; never starts training."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import time
import torch
from scripts.v3_digest_remote import gpu_rows
from shift_full.state_digest import digest_staged


def tokens(value):
    if isinstance(value, torch.Tensor):
        t=value.detach().contiguous()
        yield str((str(t.dtype),tuple(t.shape))).encode()
        yield memoryview(t.reshape(-1).view(torch.uint8).numpy())
    elif isinstance(value, dict):
        for key in sorted(value,key=lambda k:(type(k).__name__,str(k))):
            yield from tokens(key);yield from tokens(value[key])
    elif isinstance(value,(list,tuple)):
        yield str(type(value)).encode()
        for item in value:yield from tokens(item)
    else:yield repr(value).encode()+b'\0'


def inventory(value):
    row=dict(tensors=0,tensor_bytes=0,noncontiguous=0,by_dtype_device={})
    def visit(v):
        if isinstance(v,torch.Tensor):
            size=v.numel()*v.element_size();key=f'{v.dtype}/{v.device.type}'
            row['tensors']+=1;row['tensor_bytes']+=size
            row['noncontiguous']+=int(not v.is_contiguous())
            row['by_dtype_device'][key]=row['by_dtype_device'].get(key,0)+size
        elif isinstance(v,dict):
            for k,x in v.items():visit(k);visit(x)
        elif isinstance(v,(list,tuple)):
            for x in v:visit(x)
    visit(value);return row


def digest_timed(value,capacity=64*1024*1024):
    # Diagnostic copy of staged ordering. Explicit timing adds overhead;
    # compare to the untouched production implementation in separate rows.
    start=time.monotonic();h=hashlib.sha256();pending=[];streams={};slab=None;used=0
    m=dict(allocate_s=0.,enqueue_s=0.,wait_s=0.,sha_s=0.,flushes=0,pinned_requested_bytes=0)
    def flush():
        nonlocal used
        t=time.monotonic()
        for stream in streams.values():stream.synchronize()
        m['wait_s']+=time.monotonic()-t;t=time.monotonic()
        for part in pending:h.update(part)
        m['sha_s']+=time.monotonic()-t
        pending.clear();streams.clear();used=0;m['flushes']+=1
    def add(v):
        nonlocal slab,used
        if isinstance(v,torch.Tensor):
            t=v.detach();pending.append(str((str(t.dtype),tuple(t.shape))).encode())
            if t.device.type=='cpu':pending.append(memoryview(t.contiguous().reshape(-1).view(torch.uint8).numpy()))
            elif t.device.type=='cuda':
                if slab is None:
                    began=time.monotonic();slab=torch.empty(capacity,dtype=torch.uint8,pin_memory=True)
                    m['allocate_s']+=time.monotonic()-began;m['pinned_requested_bytes']=capacity
                flat=t.contiguous().reshape(-1).view(torch.uint8);offset=0
                while offset<flat.numel():
                    if used==capacity:flush()
                    count=min(capacity-used,flat.numel()-offset);target=slab[used:used+count]
                    began=time.monotonic();target.copy_(flat[offset:offset+count],non_blocking=True)
                    m['enqueue_s']+=time.monotonic()-began
                    stream=torch.cuda.current_stream(t.device);streams[t.device,stream.cuda_stream]=stream
                    pending.append(memoryview(target.numpy()));offset+=count;used+=count
            else:raise ValueError('unsupported device')
        elif isinstance(v,dict):
            for key in sorted(v,key=lambda k:(type(k).__name__,str(k))):add(key);add(v[key])
        elif isinstance(v,(list,tuple)):
            pending.append(str(type(v)).encode())
            for item in v:add(item)
        else:pending.append(repr(v).encode()+b'\0')
    try:add(value);flush()
    finally:
        for stream in streams.values():stream.synchronize()
    m['total_s']=time.monotonic()-start
    m['traversal_residual_s']=m['total_s']-sum(m[k] for k in ('allocate_s','enqueue_s','wait_s','sha_s'))
    return h.hexdigest(),m


def main():
    p=argparse.ArgumentParser();p.add_argument('--suffix',required=True);a=p.parse_args()
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix):p.error('invalid suffix')
    root=Path(__file__).resolve().parents[1]
    lock=(root/'runtime/v3-training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=root/f'results/v3_b1_20261103_{a.suffix}';out.mkdir(exist_ok=False)
    (root/'runtime/v1-controller/credentials').unlink(missing_ok=True)
    status=dict(done=False,success=False,training=False,injections=0,pid=os.getpid(),rows=[],started=time.time())
    def save():(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    try:
        before=gpu_rows();status['gpu_before']=before
        expected={r.split(',')[0].strip() for r in before if 'VLLM::' in r}
        if len(before)!=8 or len(expected)!=8:raise RuntimeError('expected only eight company GPU PIDs')
        torch.set_num_threads(1);torch.cuda.set_device(0)
        anchor=root/'runtime/v3-reference-num01/anchor_r6.pt';sha=hashlib.sha256()
        with anchor.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):sha.update(chunk)
        status['anchor_sha256']=sha.hexdigest()
        if status['anchor_sha256']!='75e2793553acfff49c0100aad600107594706d4324f881d0873a1912aa213120':
            raise RuntimeError('original anchor SHA mismatch')
        state=torch.load(anchor,map_location='cpu',weights_only=False,mmap=True)
        cpu=dict(model=state['model'],optimizer=state['optimizer'],scheduler=state['scheduler'],progress=state['progress'],
                 cpu_rng=state['torch_rng'],cuda_rng=state['cuda_rng'],python_rng=state['python_rng'],
                 buffers=state['all_buffers'],tracker_rng=state['tracker_rng'])
        def cuda_tree(v):
            if isinstance(v,torch.Tensor):return v.cuda() if v.ndim else v
            if isinstance(v,dict):return {k:cuda_tree(x) for k,x in v.items()}
            if isinstance(v,list):return [cuda_tree(x) for x in v]
            if isinstance(v,tuple):return tuple(cuda_tree(x) for x in v)
            return v
        gpu={k:cuda_tree(v) if k in ('model','optimizer','buffers') else v for k,v in cpu.items()}
        torch.cuda.synchronize()
        began=time.monotonic();prepared={k:list(tokens(v)) for k,v in cpu.items()}
        status['cpu_token_prepare_s']=time.monotonic()-began
        status['inventory']={k:dict(**inventory(v),serialized_bytes=sum(len(x) for x in prepared[k]),
                                    metadata_bytes=sum(len(x) for x in prepared[k] if isinstance(x,bytes))) for k,v in gpu.items()}
        status['layout']='Original rank6 CPU anchor; model/buffers/non-scalar Adam on CUDA, Adam step scalars CPU; RNG stays CPU. Isolated rank, no training contention.'
        def measure(name):
            torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            began=time.monotonic();hashes={};fields={}
            for k,v in (cpu if name=='cpu_staged' else gpu).items():
                t=time.monotonic()
                if name=='gpu_breakdown':hashes[k],fields[k]=digest_timed(v)
                elif name=='cpu_sha_only':
                    h=hashlib.sha256()
                    for part in prepared[k]:h.update(part)
                    hashes[k]=h.hexdigest()
                else:hashes[k]=digest_staged(v)
                fields.setdefault(k,dict(total_s=time.monotonic()-t))
            torch.cuda.synchronize()
            if hashes!=state['hashes']:raise RuntimeError('all nine recorded anchor hashes must match')
            return dict(engine=name,seconds=time.monotonic()-began,hashes=hashes,fields=fields,
                        cuda_peak_extra_bytes=torch.cuda.max_memory_allocated()-base,
                        process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        orders=[['gpu_staged','cpu_staged','cpu_sha_only','gpu_breakdown'],
                ['cpu_sha_only','gpu_breakdown','gpu_staged','cpu_staged'],
                ['cpu_staged','gpu_staged','gpu_breakdown','cpu_sha_only']]
        for repeat,order in enumerate(orders):
            for name in order:
                status['rows'].append(dict(repeat=repeat,**measure(name)));save()
        status['means']={name:statistics.fmean(r['seconds'] for r in status['rows'] if r['engine']==name) for name in orders[0]}
        status['hypothetical_sha_two_field_saving_s']=statistics.fmean(min(r['fields']['model']['total_s'],r['fields']['optimizer']['total_s']) for r in status['rows'] if r['engine']=='cpu_sha_only')
        status.update(hashes_exact=True,pinned_capacity_bytes=64*1024*1024,
                      limitation='CPU SHA is optimistic serial-resident lower bound; parallel speedup and multi-rank contention NOT measured.')
        del gpu,cpu,state,prepared;gc.collect();torch.cuda.empty_cache()
        after=gpu_rows();status['gpu_after_before_exit']=after
        company={r.split(',')[0].strip() for r in after if 'VLLM::' in r}
        extras={r.split(',')[0].strip() for r in after}-expected
        if company!=expected or extras-{str(os.getpid())}:raise RuntimeError('company baseline changed or unknown GPU load')
        status['success']=True
    except BaseException as exc:
        status['error']=type(exc).__name__+': '+str(exc);raise
    finally:status.update(done=True,finished=time.time());save();print(json.dumps({k:v for k,v in status.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
