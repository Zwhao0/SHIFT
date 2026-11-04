"""Server-owned, locked single-GPU audit microbenchmark; no training/faults."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import unittest
import torch
from shift_full.state_digest import digest_reference,digest_staged


def gpu_rows():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).splitlines()


def main():
    p=argparse.ArgumentParser();p.add_argument('--suffix',required=True);a=p.parse_args()
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix):p.error('invalid suffix')
    root=Path(__file__).resolve().parents[1]
    lock=(root/'runtime/v3-training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=root/f'results/v3_b1_20261103_{a.suffix}';out.mkdir(exist_ok=False)
    (root/'runtime/v1-controller/credentials').unlink(missing_ok=True)
    status=dict(done=False,success=False,training=False,injections=0,started=time.time(),pid=os.getpid(),rows=[])
    def save():(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    try:
        before=gpu_rows();status['gpu_before']=before
        expected={r.split(',')[0].strip() for r in before if 'VLLM::' in r}
        if len(expected)!=8 or len(before)!=8:raise RuntimeError('expected only eight company GPU PIDs; unknown load')
        torch.set_num_threads(1);torch.cuda.set_device(0)
        suite=unittest.defaultTestLoader.loadTestsFromName('scripts.test_state_digest')
        with (out/'tests.log').open('w') as stream:tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        status['tests']=dict(run=tests.testsRun,failures=len(tests.failures),errors=len(tests.errors),skipped=len(tests.skipped))
        if not tests.wasSuccessful() or tests.skipped:raise RuntimeError('CPU/CUDA digest compatibility failed')
        anchor=root/'runtime/v3-reference-num01/anchor_r6.pt'
        status['anchor']=str(anchor.relative_to(root));checksum=hashlib.sha256()
        with anchor.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):checksum.update(chunk)
        status['anchor_sha256']=checksum.hexdigest()
        state=torch.load(anchor,map_location='cpu',weights_only=False,mmap=True)
        def cuda_tree(v):
            if isinstance(v,torch.Tensor):return v.cuda() if v.ndim else v
            if isinstance(v,dict):return {k:cuda_tree(x) for k,x in v.items()}
            if isinstance(v,list):return [cuda_tree(x) for x in v]
            if isinstance(v,tuple):return tuple(cuda_tree(x) for x in v)
            return v
        fields={'model':cuda_tree(state['model']),'buffers':cuda_tree(state['all_buffers']),
            'optimizer':cuda_tree(state['optimizer'])}
        anchor_hashes={key:state['hashes'][key] for key in fields}
        status['layout']='Real rank6 anchor model/buffers and nonscalar Adam tensors on CUDA; scalar optimizer step on CPU. Isolated one rank, no training contention; not end-to-end speedup.'
        def measure(engine):
            torch.cuda.synchronize();allocated=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.monotonic();hashes={k:engine(v) for k,v in fields.items()};torch.cuda.synchronize()
            return dict(seconds=time.monotonic()-start,hashes=hashes,cuda_allocated_before_bytes=allocated,
                cuda_peak_extra_bytes=torch.cuda.max_memory_allocated()-allocated)
        expected_hashes=None
        for index,name in enumerate(['reference','staged','staged','reference','staged','reference','reference','staged']):
            row=measure(digest_reference if name=='reference' else digest_staged)
            if expected_hashes is None:expected_hashes=row['hashes']
            if row['hashes']!=expected_hashes:raise RuntimeError('real anchor digest differs')
            if row['hashes']!=anchor_hashes:raise RuntimeError('anchor recorded hashes differ')
            status['rows'].append(dict(engine=name,**row));save()
        means={name:statistics.fmean(r['seconds'] for r in status['rows'] if r['engine']==name) for name in ('reference','staged')}
        status.update(means=means,saved_seconds=means['reference']-means['staged'],hashes_exact=True,
            cuda_peak_bytes=torch.cuda.max_memory_allocated(),staging_bytes=64*1024*1024)
        del fields,state;gc.collect();torch.cuda.empty_cache()
        after=gpu_rows();company={r.split(',')[0].strip() for r in after if 'VLLM::' in r}
        extras={r.split(',')[0].strip() for r in after}-expected
        status['gpu_after_before_process_exit']=after
        if company!=expected or extras-{str(os.getpid())}:raise RuntimeError('GPU process baseline changed or unknown load')
        status['success']=True
    except BaseException as exc:
        status['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:status.update(done=True,finished=time.time());save();print(json.dumps(status),flush=True)


if __name__=='__main__':main()
