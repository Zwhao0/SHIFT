"""Application-level, update-boundary planned replacement (one epoch).

C has local groups/model/DDP first. Independent state channels and, for P1,
future AC PGs are prepared while AB runs. Old AB is drained before retirement.
"""
import copy,os,random,threading,time
from datetime import timedelta
import torch
import torch.distributed as dist
from shift_full.prepared_pg import PreparedNCCL
from shift_full.transport import PackedGPUTransport
from shift_full.planned_state import validate_state, role_hosts, merge_role_state

class EpochGate:
 def __init__(self,epoch=0):self.epoch=epoch;self.members=('A','B');self.step=None;self.ready=False
 def activate(self,epoch,step,loaded_step,ready):
  if epoch!=self.epoch+1 or step!=loaded_step or not ready:raise ValueError('stale epoch/state or unready target')
  self.epoch=epoch;self.members=('A','C');self.step=step;self.ready=True
 def authorize(self,owner,epoch):
  if epoch!=self.epoch or owner not in self.members:raise ValueError('retired owner or old epoch')
 def cancel(self):
  if self.epoch:raise ValueError('cannot cancel activated epoch')
  return self.members

class PlannedHandoff:
 def __init__(self,args,rank,event,export,load,progress,manifest,rng_snapshot):
  self.local_world=dist.get_world_size();self.args=args;self.rank=rank;self.device=torch.cuda.current_device();self.emit=event;self.export=export;self.load=load;self.progress=progress;self.role=args.planned_role;self.variant=args.planned_variant;self.source=args.planned_source;self.gate=EpochGate();self.threads=[];self.errors=[];self.channels={};self.candidate=None;self.cut=None;self.cancelled=False;self.request_seen=False;self.rng_snapshot=rng_snapshot;self.manifest=manifest;self.deferred=self.variant in ('p1d','p1f','p2');self.old_epoch_revoked=False;self.retired_pgs=[];self.cleanup_thread=None;self.cleanup_cancel=threading.Event()
  self.round=getattr(args,'planned_round',0);self.offset=20*self.round
  self.hosts=role_hosts(args.planned_hosts,self.round);self.control_host=self.hosts['A']
  self.source_host=self.hosts['B']
  self.coord_prefix=f'round/{self.round}' if getattr(args,'planned_cycles',1)>1 else ''
  self.coord_raw=getattr(args,'_coord_raw',None)
  if self.coord_raw is None:self.coord_raw=dist.TCPStore(self.control_host,31402,None,self.role=='A' and rank==0,timeout=timedelta(seconds=180),wait_for_workers=False)
  self.coord=dist.PrefixStore(self.coord_prefix,self.coord_raw) if self.coord_prefix else self.coord_raw
  self.epoch_store=dist.TCPStore(args.host_ip,31600+self.offset,None,rank==0,timeout=timedelta(seconds=60),wait_for_workers=False) if self.role in ('A','C') else None
  # One background worker establishes common PGs in a fixed order.
  # Source A/C both create state before candidate; C inserts B's role channel.
  if self.variant!='steady':self.spawn(self.prepare_all)
 @staticmethod
 def layout_manifest(value):
  if 'optimizer_groups' not in value:return value
  return dict(value,optimizer_groups=[{k:v for k,v in g.items() if k!='lr'} for g in value['optimizer_groups']])
 @property
 def activated(self):return bool(self.gate.epoch)
 def control_client(self,seconds):
  raw=dist.TCPStore(self.control_host,31402,None,False,timeout=timedelta(seconds=seconds))
  prefix=getattr(self,'coord_prefix','')
  return dist.PrefixStore(prefix,raw) if prefix else raw
 def prepare_all(self):
  if self.role=='C':
   fault=getattr(self.args,'planned_inject','healthy')
   if fault=='prepare-fail':
    self.coord.set(f'rejected/{self.rank}','1');self.emit('planned_prepare_rejected',step=self.progress['updates']);return
   if fault=='ready-timeout':
    self.emit('planned_prepare_delayed',step=self.progress['updates'])
    end=time.monotonic()+30
    while time.monotonic()<end:
     if self.coord.check(['cancel']):return
     time.sleep(.02)
   self.coord.set(f'local/C/{self.rank}','1')
  if self.role in (self.source,'C') and not self.prepare_channel('state',self.source,31404):return
  if self.source=='A' and self.role in ('B','C'):self.prepare_channel('role','B',31405)
  if self.variant in ('p1','p1d','p1f','p2') and self.role in ('A','C'):self.prepare_candidate()
 def spawn(self,fn):
  def wrapped():
   try:
    torch.cuda.set_device(self.device)
    with torch.cuda.stream(torch.cuda.Stream()):fn()
   except BaseException as e:self.errors.append(repr(e));self.emit('planned_prepare_error',error=repr(e));self.coord.set('prepare_error',repr(e))
  t=threading.Thread(target=wrapped);t.start();self.threads.append(t)
 def check(self):
  if self.errors or self.coord.check(['prepare_error']) or self.coord.check(['cleanup_error']):raise RuntimeError('planned preparation failed '+str(self.errors))
 def join(self):
  for t in self.threads:t.join()
  self.check()
 def prepare_channel(self,name,source,port):
  from torchft import ProcessGroupNCCL
  host=self.hosts[source]
  port+=self.offset
  store=dist.TCPStore(host,port,None,self.role==source and self.rank==0,timeout=timedelta(seconds=90),wait_for_workers=False)
  # TCPStore serializes requests per client: a background wait must not
  # monopolize the training thread's check/set connection.
  waiter=self.control_client(180)
  while not waiter.check([f'local/C/{self.rank}']):
   if waiter.check(['cancel']):return False
   time.sleep(.02)
  pg=ProcessGroupNCCL(timeout=timedelta(seconds=60));pg._owned_store=store
  pg.configure(f'{host}:{port}/{name}/{self.rank}',name,0 if self.role==source else 1,2)
  v=torch.ones(1,device='cuda');pg.allreduce([v],dist.AllreduceOptions()).wait();torch.cuda.current_stream().synchronize();self.channels[name]=pg
  if v.item()!=2:raise RuntimeError('state channel warmup mismatch')
  self.emit('planned_channel_ready',channel=name)
  return True
 def prepare_candidate(self):
  port=31403+self.offset
  store=dist.TCPStore(self.control_host,port,None,self.role=='A' and self.rank==0,timeout=timedelta(seconds=90),wait_for_workers=False)
  # TCPStore serializes requests per client: a background wait must not
  # monopolize the training thread's check/set connection.
  waiter=self.control_client(180)
  while not waiter.check([f'local/C/{self.rank}']):
   if waiter.check(['cancel']):return False
   time.sleep(.02)
  pg=PreparedNCCL(0 if self.role=='A' else 1,[self.rank,self.local_world+self.rank],self.emit,timeout=timedelta(seconds=60));pg._owned_store=store
  pg.prepare(f'{self.control_host}:{port}/ac/{self.rank}',f'shift{0 if self.role=="A" else 1}',self.rank,self.local_world);self.candidate=pg
 def ready_receiver(self):
  self.join();self.emit('planned_local_ready',manifest=self.manifest(),cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cpu_maxrss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss)
  self.coord.set(f'prepared/C/{self.rank}','1')
  while not self.coord.check(['cut']):
   if self.coord.check(['cancel']):
    self.close_prepared();self.emit('planned_cancelled',step=self.progress['updates']);return False
   time.sleep(.02)
  self.cut=int(self.coord.get('cut'));self.wait_freeze()
  self.final_transfer()
  self.coord.set(f'loaded/{self.rank}',str(self.cut));self.emit('planned_state_ready',step=self.cut,epoch=1)
  if self.variant=='p0':self.prepare_candidate()
  self.gate.activate(1,self.cut,self.progress['updates'],True)
  self.args.lighthouse=f'http://{self.control_host}:31410'
  self.start_retired_cleanup()
  return True
 def close_prepared(self):
  for pg in self.channels.values():pg.abort(errored=False);pg.shutdown()
  self.channels={}
  if self.candidate:self.candidate.abort(errored=False);self.candidate.shutdown();self.candidate=None
 def wait_freeze(self):self.coord.wait([f'freeze/{r}/{i}' for r in ('A','B') for i in range(self.local_world)])
 def metadata(self,payload=None):
  payload=payload or self.export()
  validate_state(payload,rank=self.rank,audit=self.args.audit_state)
  return dict({k:payload[k] for k in ('schema','torch_rng','cuda_rng','python_rng','tracker_rng','all_buffers','source_rank','step','progress')},planned_epoch=1,planned_role=1)
 def final_transfer(self):
  k=self.cut;timeout=timedelta(seconds=90)
  def transport(name):return PackedGPUTransport(self.channels[name],torch.device('cuda',self.device),timeout,self.emit)
  if self.source=='A' and self.role=='B':
   tr=transport('role');tr.send_checkpoint([1],k,self.metadata(),timeout)
  if self.role==self.source:
   if self.source=='A':self.coord.wait([f'role_loaded/{self.rank}'])
   payload=self.export();payload['planned_epoch']=1;payload['planned_role']=1;payload['manifest']=self.manifest()
   if payload['step']!=k:raise RuntimeError('donor version mismatch')
   tr=transport('state');tr.send_checkpoint([1],k,payload,timeout);self.emit('planned_state_sent',step=k,source=self.source,cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cpu_maxrss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss)
  if self.role=='C':
   role=None
   if self.source=='A':
    role=transport('role').recv_checkpoint(0,'<packed-gpu>',k,timeout)
    torch.cuda.current_stream().synchronize();self.coord.set(f'role_loaded/{self.rank}','1')
   payload=transport('state').recv_checkpoint(0,'<packed-gpu>',k,timeout)
   validate_state(payload,rank=self.rank,step=k,audit=self.args.audit_state,epoch=1)
   if self.layout_manifest(payload['manifest'])!=self.layout_manifest(self.manifest()):raise RuntimeError('state manifest mismatch')
   if role:
    from shift_full.planned_train import digest
    payload=merge_role_state(payload,role,rank=self.rank,step=k,epoch=1,digest=digest,audit=self.args.audit_state)
   payload['role_source_replica']=1
   self.emit('planned_restore_sources',step=k,shared_source=self.source,role_source='B')
   self.load(payload)
  if self.role!='C':self.coord.wait([f'loaded/{self.rank}'])
  if self.deferred:self.retired_pgs.extend(self.channels.values())
  else:
   for pg in self.channels.values():pg.abort(errored=False);pg.shutdown()
  self.channels={}
 def boundary(self,manager):
  self.check()
  if not self.cancelled and self.coord.check(['cancel']):
   self.join();self.close_prepared();self.gate.cancel();self.cancelled=True;self.emit('planned_cancelled',step=self.progress['updates'])
  if self.cancelled:return manager,False
  if not self.request_seen and self.coord.check(['request']):
   self.request_seen=True;self.emit('planned_request_seen',step=self.progress['updates'])
  if self.variant=='p1f' and self.role=='A' and self.rank==0 and self.coord.check(['arm']) and not self.coord.check(['cut']):
   if not self.coord.check([f'prepared/C/{rank}' for rank in range(self.local_world)]):raise RuntimeError('target ranks not prepared')
   # Synchronous per-step quorum/collectives bound peer lead to one step.
   # Two future committed steps leave one complete iteration for notification.
   selected=self.progress['updates']+2;self.coord.set('cut',str(selected));self.emit('planned_cut_selected',step=selected)
  if self.gate.epoch or not self.coord.check(['cut']):return manager,False
  k=int(self.coord.get('cut'))
  if self.progress['updates']<k:return manager,False
  if self.progress['updates']!=k:raise RuntimeError('missed handoff boundary')
  self.cut=k;self.join();torch.cuda.synchronize();dist.barrier();self.emit('planned_final_freeze',step=k,epoch=0,cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cpu_maxrss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss)
  self.coord.set(f'freeze/{self.role}/{self.rank}',str(k));self.wait_freeze()
  self.old_epoch_revoked=True;self.emit('planned_old_epoch_fenced',step=k)
  started=time.monotonic();manager.shutdown()
  self.emit('planned_old_manager_shutdown_complete',step=k,seconds=time.monotonic()-started)
  if self.deferred:self.retired_pgs.append(manager._pg)
  else:manager._pg.abort(errored=False);manager._pg.shutdown()
  if not self.deferred and manager._pg._pg is not None:raise RuntimeError('old training group still present')
  self.emit('planned_old_manager_closed',step=k,training_pg_absent=not self.deferred,server_closed=True,old_epoch_revoked=True)
  self.final_transfer()
  self.coord.wait([f'loaded/{i}' for i in range(self.local_world)])
  if self.role=='B':
   self.gate.activate(1,k,k,True)
   try:self.authorize(manager)
   except ValueError:self.emit('planned_old_commit_rejected',step=k,epoch=0)
   else:raise RuntimeError('retired owner accepted')
   self.emit('planned_retired_alive',step=k)
   self.start_retired_cleanup()
   self.coord.wait(['ac_three'])
   self.finish_retired_cleanup()
   self.emit('planned_standby_verified',step=k)
   return None,True
  if self.variant=='p0':self.prepare_candidate()
  self.gate.activate(1,k,k,True);self.args.lighthouse=f'http://{self.control_host}:31410'
  self.start_retired_cleanup()
  return None,False
 def authorize(self,manager):
  if self.old_epoch_revoked and manager._shift_epoch==0:raise ValueError('old epoch fenced')
  self.gate.authorize(self.role,manager._shift_epoch)
 def commit_fields(self,manager):
  self.gate.authorize(self.role,self.gate.epoch)
  return dict(native_pg_world=manager._pg.size(),native_pg_rank=manager._pg._rank,native_global_ranks=manager._pg._global_ranks,native_replica_id=manager._replica_id,native_pg_parent_identity=id(manager._pg.parent),epoch=self.gate.epoch,physical_members=list(self.gate.members),planned_role=self.role,logical_rid=0 if self.role=='A' else 1)
 def committed(self):
  if self.deferred and self.gate.epoch and self.progress['updates']==self.cut+1:
   self.coord.set(f'ac_first/{self.role}/{self.rank}','1')
  if self.gate.epoch and self.progress['updates']>=self.cut+3:
   self.coord.set(f'ac3/{self.role}/{self.rank}','1')
   if self.role=='A' and self.rank==0:
    self.coord.wait([f'ac3/{role}/{rank}' for role in ('A','C') for rank in range(self.local_world)])
    self.coord.set('ac_three','1')

 def start_retired_cleanup(self):
  if not self.deferred or self.cleanup_thread is not None:return
  def cleanup():
   try:
    torch.cuda.set_device(self.device)
    # A dedicated client avoids blocking the training thread's TCPStore.
    control=self.control_client(10)
    keys=[f'ac_first/{role}/{rank}' for role in ('A','C') for rank in range(getattr(self,'local_world',8))];deadline=time.monotonic()+45
    while not control.check(keys):
     if self.cleanup_cancel.wait(.02):return
     if time.monotonic()>deadline:raise RuntimeError('first AC cleanup trigger timed out')
    started=time.monotonic();self.emit('planned_retired_cleanup_begin',step=self.cut,first_commit_keys_ready=True)
    failures=[]
    for pg in self.retired_pgs:
     try:pg.abort(errored=False)
     except Exception as e:failures.append(repr(e))
     try:pg.shutdown()
     except Exception as e:failures.append(repr(e))
    if failures:raise RuntimeError('retired cleanup failures: '+str(failures))
    if any(pg._pg is not None for pg in self.retired_pgs):raise RuntimeError('retired PG still present')
    self.emit('planned_retired_cleanup_complete',step=self.cut,seconds=time.monotonic()-started,closed_pg_count=len(self.retired_pgs),all_parent_absent=True,cuda_allocated_bytes=torch.cuda.memory_allocated())
   except BaseException as e:
    self.errors.append(repr(e));self.emit('planned_cleanup_error',error=repr(e));self.coord.set('cleanup_error',repr(e))
  self.cleanup_thread=threading.Thread(target=cleanup,name='shift-retired-cleanup',daemon=True);self.cleanup_thread.start()
 def finish_retired_cleanup(self):
  if self.cleanup_thread is not None:
   self.cleanup_thread.join(timeout=60)
   if self.cleanup_thread.is_alive():
    self.cleanup_cancel.set();raise RuntimeError('retired cleanup timed out')
   self.check()

 def cancel_cleanup_on_exit(self):
  self.cleanup_cancel.set()
  if self.cleanup_thread is not None and self.cleanup_thread.is_alive():
   self.cleanup_thread.join(timeout=2)
   if self.cleanup_thread.is_alive():self.emit('planned_cleanup_incomplete_on_error')
