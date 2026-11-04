"""Bounded installed-torchFT microprobe: three physical replicas, two contributors."""
import argparse,copy,hashlib,json,os,time
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from shift_full.compat import install_managed_callback_fix

def main():
 p=argparse.ArgumentParser();p.add_argument('--rid',type=int);p.add_argument('--host');p.add_argument('--session');p.add_argument('--phase',choices=['shadow','candidate','p0','overlap','relay'],default='shadow');a=p.parse_args()
 if a.phase=='relay':
  from shift_full.overlap_probe import run
  return run(a)
 if a.phase in ('candidate','p0'):
  from shift_full.planned_components import run
  return run(a)
 torch.set_num_threads(1);torch.cuda.set_device(0);install_managed_callback_fix()
 out=Path('/workspace/SHIFT/results')/a.session;out.mkdir(parents=True,exist_ok=True)
 log=(out/'events.jsonl').open('w',buffering=1)
 def event(kind,**kw): log.write(json.dumps(dict(kind=kind,rid=a.rid,pid=os.getpid(),mono=time.monotonic(),**kw))+'\n')
 dist.init_process_group('nccl',init_method=f'tcp://{a.host}:31301',rank=0,world_size=1,timeout=timedelta(seconds=40))
 from torchft import Manager,ProcessGroupNCCL
 from torchft.manager import WorldSizeMode
 from torchft.checkpointing import HTTPTransport
 class HTTP(HTTPTransport):
  def address(self):
   import socket
   return super().address().replace(socket.gethostname(),a.host)
 model=torch.nn.Linear(16,8,bias=True).cuda();torch.manual_seed(20261103)
 with torch.no_grad():
  for t in model.parameters(): t.fill_(.125)
 opt=torch.optim.AdamW(model.parameters(),lr=1e-3,foreach=False);sched=torch.optim.lr_scheduler.StepLR(opt,1000)
 progress=0
 def state():
  torch.cuda.synchronize();return dict(model=copy.deepcopy(model.state_dict()),opt=copy.deepcopy(opt.state_dict()),sched=sched.state_dict(),step=progress)
 def load(v):
  nonlocal progress
  model.load_state_dict(v['model']);opt.load_state_dict(v['opt']);sched.load_state_dict(v['sched']);progress=v['step'];event('healed',step=progress,optimizer_states=len(opt.state))
 pg=ProcessGroupNCCL(timeout=timedelta(seconds=20));configs=[];configure=pg.configure
 def traced(*args,**kw):
  configs.append([args[2],args[3],args[4]]);event('configure',rank=args[2],world=args[3],quorum=args[4]);return configure(*args,**kw)
 pg.configure=traced
 m=Manager(pg=pg,state_dict=state,load_state_dict=load,min_replica_size=2,world_size_mode=WorldSizeMode.FIXED_WITH_SPARES,use_async_quorum=False,init_sync=False,rank=0,world_size=1,store_addr=a.host,store_port=31301,replica_id=f'probe{a.rid}',hostname=a.host,lighthouse_addr='http://192.0.2.10:31300',timeout=timedelta(seconds=30),quorum_timeout=timedelta(seconds=40),max_retries=0,checkpoint_transport=HTTP(timedelta(seconds=30),0))
 coord=dist.TCPStore('192.0.2.10',31302,3,a.rid==0,timeout=timedelta(seconds=40),wait_for_workers=False)
 event('ready')
 while not (out/'START').exists(): time.sleep(.02)
 # C joins after AB have nonempty Adam state, then skips ALL forward/backward.
 if a.rid==2:
  coord.wait(['JOIN'])
 for attempt in range(3000):
  m.start_quorum();m.wait_quorum();progress=m.current_step()
  active=m.is_participating();role=m.participating_rank();opt.zero_grad(set_to_none=True)
  if active:
   x=torch.arange(16,device='cuda',dtype=torch.float32).view(1,16)/16+role*.25+progress*.01
   model(x).square().sum().backward()
  grad=[]
  for t in model.parameters():
   if t.grad is None:t.grad=torch.zeros_like(t)
   w=m.allreduce(t.grad);w.wait();grad.append(t.grad.detach().cpu().tolist())
  commit=m.should_commit()
  if not commit: raise RuntimeError('commit rejected '+str(m.errored()))
  opt.step();sched.step();progress+=1;torch.cuda.synchronize()
  vals={k:v.detach().cpu().tolist() for k,v in model.state_dict().items()}
  moments={str(k):{n:t.detach().cpu().tolist() if isinstance(t,torch.Tensor) else t for n,t in v.items()} for k,v in opt.state_dict()['state'].items()}
  event('commit',step=progress,active=active,role=role,participants=m.num_participants(),physical_world=m._pg.size(),forward_calls=int(active),model_hash=hashlib.sha256(json.dumps(vals).encode()).hexdigest(),moments_hash=hashlib.sha256(json.dumps(moments).encode()).hexdigest(),grad_hash=hashlib.sha256(json.dumps(grad).encode()).hexdigest())
  if a.rid==0 and progress==2:coord.set('JOIN','1')
  if progress>=2000:break
 event('done',configs=configs,step=progress,allocated=torch.cuda.memory_allocated(),peak=torch.cuda.max_memory_allocated())
 m.shutdown();pg.abort(errored=False);pg.shutdown();dist.destroy_process_group()
if __name__=='__main__':main()
