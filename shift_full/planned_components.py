"""Three-node planned AB->AC component probe; not a MoE benchmark."""
import copy,hashlib,json,os,threading,time
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from shift_full.compat import install_managed_callback_fix
from shift_full.prepared_pg import PreparedNCCL
from shift_full.transport import PackedGPUTransport

def run(a):
 torch.set_num_threads(1);torch.cuda.set_device(0);install_managed_callback_fix()
 out=Path('/workspace/SHIFT/results')/a.session;out.mkdir(parents=True,exist_ok=True);log=(out/'events.jsonl').open('w',buffering=1);lock=threading.Lock()
 def event(kind,**kw):
  with lock:log.write(json.dumps(dict(kind=kind,rid=a.rid,pid=os.getpid(),mono=time.monotonic(),**kw))+'\n')
 dist.init_process_group('nccl',init_method=f'tcp://{a.host}:31301',rank=0,world_size=1,timeout=timedelta(seconds=40))
 coord=dist.TCPStore('10.10.4.2',31302,3,a.rid==0,timeout=timedelta(seconds=60),wait_for_workers=False)
 from torchft import Manager,ProcessGroupNCCL
 from torchft.manager import WorldSizeMode
 # A real, distinguishable role-specific random sample sequence.
 torch.manual_seed(20261103);model=torch.nn.Linear(1024,1024,bias=False).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=False);sched=torch.optim.lr_scheduler.StepLR(opt,1000);progress=0
 def hashes():
  def h(v):
   buf=[]
   def walk(x):
    if isinstance(x,torch.Tensor):buf.append(x.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(x,dict):
     for k,v in x.items():buf.append(str(k).encode());walk(v)
    elif isinstance(x,(list,tuple)):
     for v in x:walk(v)
    else:buf.append(repr(x).encode())
   walk(v);return hashlib.sha256(b''.join(buf)).hexdigest()
  return dict(model=h(model.state_dict()),optimizer=h(opt.state_dict()),scheduler=h(sched.state_dict()))
 def state():
  torch.cuda.synchronize();return dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),scheduler=copy.deepcopy(sched.state_dict()),step=progress,source_rank=0,role=1,epoch=1)
 def load(v):
  nonlocal progress
  if v['epoch']!=1 or v['role']!=1 or v['step']!=int(coord.get('cut')):raise RuntimeError('stale state/epoch')
  model.load_state_dict(v['model']);opt.load_state_dict(v['optimizer']);sched.load_state_dict(v['scheduler']);progress=v['step'];torch.cuda.synchronize();event('state_loaded',step=progress,hashes=hashes())
 def manager(pg,rid,epoch):
  m=Manager(pg=pg,state_dict=state,load_state_dict=load,min_replica_size=2,world_size_mode=WorldSizeMode.FIXED_WITH_SPARES,use_async_quorum=False,init_sync=False,rank=0,world_size=1,store_addr=a.host,store_port=31301,replica_id=f'component{rid}',hostname=a.host,lighthouse_addr=f'http://10.10.4.2:{31300 if epoch==0 else 31310}',timeout=timedelta(seconds=30),quorum_timeout=timedelta(seconds=40),max_retries=0)
  m.load_state_dict(dict(step=progress,batches_committed=progress*2));return m
 candidate=None;candidate_thread=None;candidate_error=[];statepg=None;state_thread=None
 # Independent B-C state channel: not the future training group.
 def prepare_state():
  nonlocal statepg
  torch.cuda.set_device(0);store=dist.TCPStore('10.10.4.1',31304,2,a.rid==1,timeout=timedelta(seconds=40),wait_for_workers=False)
  pg=ProcessGroupNCCL(timeout=timedelta(seconds=30));pg._state_store=store;pg.configure('10.10.4.1:31304/state','state'+str(a.rid),0 if a.rid==1 else 1,2)
  v=torch.ones(1,device='cuda');pg.allreduce([v],dist.AllreduceOptions()).wait();torch.cuda.synchronize();statepg=pg;event('state_channel_ready')
 def prepare_candidate():
  nonlocal candidate
  try:
   torch.cuda.set_device(0)
   with torch.cuda.stream(torch.cuda.Stream()):
    # TCPStore lifetime is retained by the PG client and store object below.
    store=dist.TCPStore('10.10.4.2',31303,2,a.rid==0,timeout=timedelta(seconds=40),wait_for_workers=False)
    pg=PreparedNCCL(0 if a.rid==0 else 1,[0,1],event);pg._candidate_store=store
    pg.prepare('10.10.4.2:31303/candidate',f'component{0 if a.rid==0 else 1}',0,1);candidate=pg
  except BaseException as e:candidate_error.append(repr(e));event('candidate_error',error=repr(e))
 event('ready')
 while not (out/'START').exists():time.sleep(.02)
 if a.rid==2:coord.wait(['join'])
 if a.rid in (1,2):state_thread=threading.Thread(target=prepare_state);state_thread.start()
 if a.phase=='candidate' and a.rid in (0,2):candidate_thread=threading.Thread(target=prepare_candidate);candidate_thread.start()
 if a.rid==2:
  if candidate_thread:candidate_thread.join()
  state_thread.join()
  if candidate_error:raise RuntimeError(candidate_error)
  coord.set('c_ready','1');coord.wait(['cut'])
 else:
  oldpg=ProcessGroupNCCL(timeout=timedelta(seconds=20));m=manager(oldpg,a.rid,0)
  for attempt in range(2000):
   if progress==2 and a.rid==0:coord.set('join','1');event('request',step=progress)
   if a.rid==0 and coord.check(['c_ready']) and not coord.check(['cut']):coord.set('cut',str(progress+10));event('cut_selected',step=progress+10)
   if coord.check(['cut']) and progress==int(coord.get('cut')):break
   m.start_quorum();m.wait_quorum();opt.zero_grad(set_to_none=True)
   gen=torch.Generator(device='cuda').manual_seed(100000+progress*2+a.rid)
   x=torch.randn(128,1024,generator=gen,device='cuda');model(x).square().mean().backward()
   for t in model.parameters():m.allreduce(t.grad).wait()
   if not m.should_commit():raise RuntimeError('AB commit rejected')
   opt.step();sched.step();progress+=1;torch.cuda.synchronize();event('commit',step=progress,epoch=0,role=a.rid,sample=100000+(progress-1)*2+a.rid,physical_members=['A','B'],participants=2)
  if progress>=2000:raise RuntimeError('preparation did not complete')
  torch.cuda.synchronize();event('final_freeze',step=progress,hashes=hashes());m.shutdown();oldpg.abort(errored=False);oldpg.shutdown()
  if state_thread:state_thread.join()
  if candidate_thread:candidate_thread.join()
 # Final latest state; no stale snapshot is accepted.
 if a.rid in (1,2):
  tr=PackedGPUTransport(statepg,torch.device('cuda',0),timedelta(seconds=30),event)
  if a.rid==1:tr.send_checkpoint([1],progress,state(),timedelta(seconds=30));event('state_sent',step=progress,hashes=hashes())
  else:
   progress=int(coord.get('cut'));payload=tr.recv_checkpoint(0,'<packed-gpu>',progress,timedelta(seconds=30));load(payload);coord.set('loaded','1')
  statepg.abort(errored=False);statepg.shutdown()
 if a.rid==1:
  event('retired_alive',step=progress,epoch=0);coord.wait(['done']);event('done',step=progress,standby=True)
 else:
  coord.wait(['loaded'])
  if a.phase=='p0':prepare_candidate()
  if candidate_error:raise RuntimeError(candidate_error)
  m=manager(candidate,0 if a.rid==0 else 1,1);event('epoch_activated',step=progress,epoch=1)
  for i in range(3):
   m.start_quorum();m.wait_quorum();opt.zero_grad(set_to_none=True);role=0 if a.rid==0 else 1
   gen=torch.Generator(device='cuda').manual_seed(100000+progress*2+role);model(torch.randn(128,1024,generator=gen,device='cuda')).square().mean().backward()
   for t in model.parameters():m.allreduce(t.grad).wait()
   if not m.should_commit():raise RuntimeError('AC commit rejected')
   opt.step();sched.step();progress+=1;torch.cuda.synchronize();event('commit',step=progress,epoch=1,role=role,sample=100000+(progress-1)*2+role,physical_members=['A','C'],participants=m.num_participants(),hashes=hashes(),native_creations=candidate.native_creations)
  m.shutdown();candidate.abort(errored=False);candidate.shutdown();coord.set('done','1');event('done',step=progress,standby=False,allocated=torch.cuda.memory_allocated(),peak=torch.cuda.max_memory_allocated())
 dist.destroy_process_group()
