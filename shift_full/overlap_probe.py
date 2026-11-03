"""Small real-NCCL/torchFT correctness probe for the same relay implementation."""
import copy,json,random,time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from shift_full.compat import install_managed_callback_fix
from shift_full.overlap_handoff import OverlapHandoff
from shift_full.planned_train import digest,tensor_count

def run(a):
 install_managed_callback_fix();torch.set_num_threads(1);torch.cuda.set_device(0)
 dist.init_process_group('nccl',init_method=f'tcp://{a.host}:31301',rank=0,world_size=1,timeout=timedelta(seconds=40))
 out=Path('/workspace/SHIFT/results')/a.session;log=(out/'events.jsonl').open('w',buffering=1)
 role=('A','B','C')[a.rid];rid=int(role!='A')
 def event(kind,**kw):log.write(json.dumps(dict(kind=kind,role=role,rid=rid,rank=0,pid=__import__('os').getpid(),monotonic=time.monotonic(),**kw))+'\n')
 args=SimpleNamespace(planned_role=role,planned_variant='p2',planned_source='B',host_ip=a.host,lighthouse='http://10.10.4.2:31300',audit_state=True)
 model=torch.nn.Linear(16,8).cuda()
 with torch.no_grad():
  for p in model.parameters():p.fill_(.125)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False);sched=torch.optim.lr_scheduler.StepLR(opt,1000)
 progress={'updates':0};manager=None
 torch.manual_seed(70000+rid);random.seed(80000+rid)
 def rng():return dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate())
 def hashes():return dict(model=digest(model.state_dict()),optimizer=digest(opt.state_dict()),scheduler=digest(sched.state_dict()),progress=digest(progress),**{k:digest(v) for k,v in rng().items()})
 def export():
  torch.cuda.synchronize();return dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),all_buffers=dict(model.named_buffers()),progress=dict(progress),step=progress['updates'],source_rank=0,source_replica=rid,hashes=hashes(),**rng())
 def tail(v):
  torch.set_rng_state(v['torch_rng']);torch.cuda.set_rng_state(v['cuda_rng']);random.setstate(v['python_rng'])
 def load(v):
  model.load_state_dict(v['model']);opt.load_state_dict(v['optimizer']);sched.load_state_dict(v['scheduler']);progress.update(v['progress']);tail(v)
  assert hashes()==v['hashes'];event('restored',step=progress['updates'],hashes_match=True,optimizer_tensors=tensor_count(opt.state_dict()['state']))
 h=OverlapHandoff(args,0,event,export,load,progress,lambda:dict(layout='Linear16x8'),rng)
 event('ready')
 while not (out/'START').exists():time.sleep(.01)
 if role=='C':h.ready_receiver()
 from torchft import Manager,ProcessGroupNCCL
 from torchft.manager import WorldSizeMode
 from shift_full.transport import PackedGPUTransport
 def make():
  pg=h.candidate if h.gate.epoch else ProcessGroupNCCL(timeout=timedelta(seconds=20))
  m=Manager(pg=pg,state_dict=export,load_state_dict=load,min_replica_size=2,world_size_mode=WorldSizeMode.FIXED_WITH_SPARES,use_async_quorum=False,init_sync=False,rank=0,world_size=1,store_addr=a.host,store_port=31600 if h.gate.epoch else 31301,replica_id=f'probe{rid}',hostname=a.host,lighthouse_addr=args.lighthouse,timeout=timedelta(seconds=30),quorum_timeout=timedelta(seconds=40),max_retries=0,checkpoint_transport=PackedGPUTransport(pg,torch.device('cuda'),timedelta(seconds=30)))
  if h.gate.epoch:m.load_state_dict(dict(step=progress['updates'],batches_committed=progress['updates']*2))
  m._shift_epoch=h.gate.epoch;return m
 manager=make()
 try:
  for attempt in range(40):
   if role=='A' and progress['updates']>=2 and h.coord.check(['prepared/C/0']):h.coord.set('arm','1')
   if not h.gate.epoch:manager,_=h.boundary(manager)
   manager.start_quorum();manager.wait_quorum();opt.zero_grad(set_to_none=True)
   step=progress['updates'];samples=[step*2+rid]
   if role=='C' and h.is_relay_step():
    for p in model.parameters():p.grad=torch.zeros_like(p)
   else:
    x=torch.arange(16,device='cuda').view(1,16)/16+rid*.25+step*.01
    model(x).square().sum().backward()
   grads=[p.grad for p in model.parameters()]
   if h.is_relay_step():
    manager,retired=h.at_gradient(manager,grads,make,samples,tail)
    if retired:manager=None;break
   for g in grads:manager.allreduce(g).wait()
   h.authorize(manager);assert manager.should_commit();opt.step();sched.step();progress['updates']+=1;torch.cuda.synchronize()
   event('commit',step=progress['updates'],hashes=hashes(),**h.commit_fields(manager));h.committed()
   if h.gate.epoch and progress['updates']>=h.cut+6:break
  h.finish_retired_cleanup();event('done',step=progress['updates'])
 finally:
  if manager:manager.shutdown();manager._pg.abort(errored=False);manager._pg.shutdown()
  h.cancel_cleanup_on_exit();dist.destroy_process_group()
