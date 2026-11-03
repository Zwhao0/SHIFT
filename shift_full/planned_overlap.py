"""Immutable model+nonempty Adam snapshot overlaps next forward/backward reads.

The writer waits for snapshot completion before optimizer.step. This component
probe does not claim network overlap or production MoE performance.
"""
import copy,hashlib,json,random,time
from pathlib import Path
import torch

def digest_tree(value):
 h=hashlib.sha256()
 def walk(v):
  if isinstance(v,torch.Tensor):h.update(v.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(v,dict):
   for k,x in v.items():h.update(str(k).encode());walk(x)
  elif isinstance(v,(list,tuple)):
   for x in v:walk(x)
  else:h.update(repr(v).encode())
 walk(value);return h.hexdigest()
def clone_tree(v):
 if isinstance(v,torch.Tensor):return v.detach().clone()
 if isinstance(v,dict):return {k:clone_tree(x) for k,x in v.items()}
 if isinstance(v,list):return [clone_tree(x) for x in v]
 if isinstance(v,tuple):return tuple(clone_tree(x) for x in v)
 return copy.deepcopy(v)
def allocate_tree(v):
 if isinstance(v,torch.Tensor):return torch.empty_like(v)
 if isinstance(v,dict):return {k:allocate_tree(x) for k,x in v.items()}
 if isinstance(v,list):return [allocate_tree(x) for x in v]
 if isinstance(v,tuple):return tuple(allocate_tree(x) for x in v)
 return copy.deepcopy(v)
def copy_into(src,dst):
 if isinstance(src,torch.Tensor):dst.copy_(src,non_blocking=True);return dst
 if isinstance(src,dict):return {k:copy_into(x,dst[k]) for k,x in src.items()}
 if isinstance(src,list):return [copy_into(x,y) for x,y in zip(src,dst)]
 if isinstance(src,tuple):return tuple(copy_into(x,y) for x,y in zip(src,dst))
 return copy.deepcopy(src)
def main():
 torch.set_num_threads(1);torch.cuda.set_device(0);torch.manual_seed(20261103);random.seed(20261103)
 torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 model=torch.nn.Linear(2048,2048,bias=False).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=False);sched=torch.optim.lr_scheduler.StepLR(opt,1000)
 copy_stream=torch.cuda.Stream();rows=[];snapshot=None;first_hash=None;storage=None;allocation_seconds=None
 for step in range(12):
  torch.cuda.synchronize();anchor=torch.cuda.Event(enable_timing=True);anchor.record();anchor.synchronize()
  cs=ce=None;cpu=time.monotonic()
  if step==2:
   begin=time.monotonic();storage=allocate_tree(dict(model=model.state_dict(),optimizer=opt.state_dict()));allocation_seconds=time.monotonic()-begin
  if step==3:
   metadata=dict(step=step,epoch=0,role=1,cpu_rng=torch.get_rng_state().clone(),cuda_rng=torch.cuda.get_rng_state().clone(),python_rng=random.getstate(),scheduler=copy.deepcopy(sched.state_dict()))
  opt.zero_grad(set_to_none=True);fs=torch.cuda.Event(enable_timing=True);fe=torch.cuda.Event(enable_timing=True);fs.record()
  gen=torch.Generator(device='cuda').manual_seed(100000+step*2+1)
  prediction=model(torch.randn(512,2048,generator=gen,device='cuda'))
  if step==3:
   enqueue_started=time.monotonic()
   # Linear forward/backward only READ weights; wait on the committed
   # boundary, not this forward, and fence the next optimizer writer.
   with torch.cuda.stream(copy_stream):
    copy_stream.wait_event(anchor);cs=torch.cuda.Event(enable_timing=True);ce=torch.cuda.Event(enable_timing=True);cs.record();snapshot=copy_into(dict(model=model.state_dict(),optimizer=opt.state_dict()),storage);snapshot['metadata']=metadata;ce.record()
   snapshot_cpu_enqueue=time.monotonic()-enqueue_started
  prediction.square().mean().backward();fe.record()
  if ce:torch.cuda.current_stream().wait_event(ce)
  opt.step();sched.step();torch.cuda.synchronize()
  row=dict(step=step+1,seconds=time.monotonic()-cpu,compute_ms=fs.elapsed_time(fe))
  if ce:
   row.update(copy_start_ms=anchor.elapsed_time(cs),copy_end_ms=anchor.elapsed_time(ce),compute_start_ms=anchor.elapsed_time(fs),compute_end_ms=anchor.elapsed_time(fe),snapshot_enqueue_seconds=snapshot_cpu_enqueue)
   row['overlap_ms']=max(0,min(row['copy_end_ms'],row['compute_end_ms'])-max(row['copy_start_ms'],row['compute_start_ms']))
   first_hash=digest_tree(snapshot);row['immutable_hash']=first_hash
  rows.append(row)
 unchanged=digest_tree(snapshot)==first_hash
 target=torch.nn.Linear(2048,2048,bias=False).cuda();targetopt=torch.optim.AdamW(target.parameters(),lr=1e-4,foreach=False)
 target.load_state_dict(snapshot['model']);targetopt.load_state_dict(snapshot['optimizer']);torch.cuda.synchronize()
 loaded_old=digest_tree(dict(model=target.state_dict(),optimizer=targetopt.state_dict()))==digest_tree(dict(model=snapshot['model'],optimizer=snapshot['optimizer']))
 final=clone_tree(dict(model=model.state_dict(),optimizer=opt.state_dict()));torch.cuda.synchronize();start=time.monotonic();target.load_state_dict(final['model']);targetopt.load_state_dict(final['optimizer']);torch.cuda.synchronize()
 final_load_seconds=time.monotonic()-start
 final_loaded=digest_tree(dict(model=target.state_dict(),optimizer=targetopt.state_dict()))==digest_tree(final)
 nbytes=sum(t.numel()*t.element_size() for v in (snapshot['model'],snapshot['optimizer']['state'][0]) for t in v.values() if isinstance(t,torch.Tensor))
 result=dict(variant='preallocated_copy_in_readonly_forward_backward_window',target_allocation_seconds=allocation_seconds,seed=20261103,rows=rows,immutable_after_nine_updates=unchanged,old_snapshot_loaded_exactly=loaded_old,final_loaded_exactly=final_loaded,old_version=3,current_version=12,lag=9,precopy_tensor_bytes=nbytes,final_tensor_bytes=nbytes,final_load_seconds=final_load_seconds,final_sync_reduction=0,cuda_peak_bytes=torch.cuda.max_memory_allocated(),limitation='single GPU component; no asynchronous network transfer; no delta/replay, final full state still required')
 assert unchanged and loaded_old and final_loaded
 out=Path('/workspace/SHIFT/results/planned_overlap');out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
