"""One-step B state pre-copy, then B-gradient ownership transfer before outer reduce.

Local DDP readiness precedes relay; native TP/PP finalization still follows the
AC reduction, exactly as in the existing wrapper. No in-flight collective swaps.
"""
import threading,time
from datetime import timedelta
import torch
import torch.distributed as dist
from shift_full.planned_handoff import PlannedHandoff,EpochGate
from shift_full.transport import PackedGPUTransport

class RetiredReplica(Exception):pass

class FrozenTransport(PackedGPUTransport):
 def freeze(self,payload,step):
  self.frozen=super()._pack(payload,step);self.frozen_step=step
  self.ready=torch.cuda.Event();self.ready.record()
 def _pack(self,payload,step):
  if step!=self.frozen_step:raise ValueError('frozen version mismatch')
  self.ready.wait()
  return self.frozen

class OverlapHandoff(PlannedHandoff):
 def __init__(self,*args,**kw):
  self.presend=None;self.prepared_step=None;self.relay_payload=None;self.relay_done=False;self.fallback=False
  super().__init__(*args,**kw)
  self.base_epoch=self.round;self.new_epoch=self.round+1;self.gate=EpochGate(self.base_epoch)
 @property
 def activated(self):return self.gate.epoch>self.base_epoch
 def transport(self,cls=PackedGPUTransport):return cls(self.channels['state'],torch.device('cuda',self.device),timedelta(seconds=60),self.emit)
 def boundary(self,manager):
  self.check()
  if self.coord.check(['cancel']):
   if not self.cancelled:
    self.join();self.close_prepared();self.cancelled=True;self.emit('planned_cancelled',step=self.progress['updates'])
   return manager,False
  if not self.request_seen and self.coord.check(['request']):
   self.request_seen=True;self.emit('planned_request_seen',step=self.progress['updates'])
  if self.role=='A' and self.rank==0 and self.coord.check(['arm']) and not self.coord.check(['cut']):
   if not self.coord.check([f'prepared/C/{r}' for r in range(self.local_world)]):raise RuntimeError('target not prepared')
   self.coord.set('cut',str(self.progress['updates']+2));self.emit('planned_cut_selected',step=self.progress['updates']+2)
  if not self.coord.check(['cut']):return manager,False
  self.cut=int(self.coord.get('cut'))
  if self.progress['updates']<self.cut:return manager,False
  if self.progress['updates']!=self.cut:raise RuntimeError('missed precopy boundary')
  if self.prepared_step is None:
   self.join();self.prepared_step=self.cut
   if self.role=='B':
    started=time.monotonic();payload=self.export();payload.update(planned_epoch=self.new_epoch,planned_role=1,manifest=self.manifest())
    if getattr(self.args,'planned_inject','healthy')=='stale-state' and self.rank==0:payload['planned_epoch']=self.base_epoch
    # B cannot update again: it retires at this iteration's gradient boundary.
    # Parameters/Adam remain read-only; buffers can change during forward.
    saved_buffers={name:t.detach().clone() for name,t in payload['all_buffers'].items()}
    payload['all_buffers']=saved_buffers
    for name,value in saved_buffers.items():
     if name in payload['model']:payload['model'][name]=value
    ready=torch.cuda.Event();ready.record()
    from torch.utils._pytree import tree_flatten
    versions=[(t,t._version) for t in tree_flatten(payload)[0] if isinstance(t,torch.Tensor)]
    tr=self.transport(FrozenTransport)
    self.emit('planned_precopy_window_open',step=self.cut,seconds=time.monotonic()-started)
    def send():
     try:
      torch.cuda.set_device(self.device)
      with torch.cuda.stream(torch.cuda.Stream()):
       ready.wait();packing=time.monotonic();tr.freeze(payload,self.cut)
       if any(t._version!=version for t,version in versions):raise RuntimeError('state mutated during precopy read window')
       self.emit('planned_precopy_frozen',step=self.cut,seconds=time.monotonic()-packing,mode='readonly_window')
       tr.send_checkpoint([1],self.cut,None,timedelta(seconds=60))
      del tr.frozen
      self.emit('planned_precopy_sent',step=self.cut)
     except BaseException as e:self.errors.append(repr(e));self.coord.set('prepare_error',repr(e))
    self.presend=threading.Thread(target=send);self.presend.start()
  return manager,False
 @staticmethod
 def layout_manifest(value):
  # Learning rate is restored state, not a shard/layout requirement.
  if 'optimizer_groups' not in value:return value
  return dict(value,optimizer_groups=[{k:v for k,v in g.items() if k!='lr'} for g in value['optimizer_groups']])
 def ready_receiver(self):
  self.join()
  if not self.coord.check([f'local/C/{self.rank}']):
   while not self.coord.check(['cancel']):time.sleep(.02)
   self.close_prepared();self.emit('planned_cancelled',step=0);return False
  self.emit('planned_local_ready',manifest=self.manifest());self.coord.set(f'prepared/C/{self.rank}','1')
  while not self.coord.check(['cut']):
   if self.coord.check(['cancel']):self.close_prepared();self.emit('planned_cancelled',step=0);return False
   time.sleep(.01)
  self.cut=int(self.coord.get('cut'))
  payload=self.transport().recv_checkpoint(0,'<packed-gpu>',self.cut,timedelta(seconds=60))
  if payload['step']!=self.cut or self.layout_manifest(payload['manifest'])!=self.layout_manifest(self.manifest()) or payload['planned_epoch']!=self.new_epoch:
   self.emit('planned_precopy_rejected',step=self.cut);self.coord.set('cancel','stale state')
  else:
   self.emit('planned_restore_sources',step=self.cut,shared_source='B',role_source='B',source_physical='C' if self.round%2 else 'B');self.load(payload)
   self.emit('planned_precopy_loaded',step=self.cut);self.coord.set(f'preloaded/{self.rank}',str(self.cut))
  del payload
  self.coord.wait(['decision'])
  if self.coord.get('decision')==b'abort':
   self.close_prepared();self.emit('planned_cancelled',step=self.cut);return False
  self.wait_freeze()
  self.relay_payload=self.transport().recv_checkpoint(0,'<packed-gpu>',self.cut,timedelta(seconds=60))
  if self.relay_payload['step']!=self.cut or self.relay_payload['sample_step']!=self.cut+1:raise RuntimeError('stale relay')
  from shift_full.planned_train import digest
  valid=not self.args.audit_state or digest(self.relay_payload['gradients'])==self.relay_payload['gradient_hash']
  self.coord.set(f'relay_{"ok" if valid else "rejected"}/{self.rank}','1')
  if not valid:self.emit('planned_relay_rejected',step=self.cut)
  self.coord.wait(['activation_decision'])
  if self.coord.get('activation_decision')==b'AB':
   self.close_prepared();self.emit('planned_activation_rejected',step=self.cut);self.relay_payload=None;return False
  self.coord.set(f'loaded/{self.rank}',str(self.cut));self.emit('planned_state_ready',step=self.cut,epoch=self.new_epoch)
  self.gate.activate(self.new_epoch,self.cut,self.progress['updates'],True);self.args.lighthouse=f'http://10.10.4.2:{31410+self.offset}'
  self.retired_pgs.extend(self.channels.values());self.channels={};self.start_retired_cleanup()
  return True
 def is_relay_step(self):return self.cut is not None and self.progress['updates']==self.cut and not self.relay_done
 def fence(self,manager):
  torch.cuda.synchronize();dist.barrier();self.emit('planned_final_freeze',step=self.cut,epoch=self.base_epoch)
  self.coord.set(f'freeze/{self.role}/{self.rank}',str(self.cut));self.wait_freeze()
  self.old_epoch_revoked=True;self.emit('planned_old_epoch_fenced',step=self.cut)
  started=time.monotonic();manager.shutdown();self.emit('planned_old_manager_shutdown_complete',step=self.cut,seconds=time.monotonic()-started)
  self.retired_pgs.append(manager._pg);self.emit('planned_old_manager_closed',step=self.cut,training_pg_absent=False,server_closed=True,old_epoch_revoked=True)
 def at_gradient(self,manager,gradients,make_manager,sample_ids,apply_tail):
  from shift_full.planned_train import digest
  if self.role=='C':
   packet=self.relay_payload
   if len(gradients)!=len(packet['gradients']):raise RuntimeError('relay bucket count mismatch')
   for dst,src in zip(gradients,packet['gradients']):
    if dst.shape!=src.shape or dst.dtype!=src.dtype:raise RuntimeError('relay bucket manifest mismatch')
    dst.copy_(src)
   if sample_ids!=packet['sample_ids']:raise RuntimeError('relay sample role mismatch')
   applied=apply_tail(packet['tail']);torch.cuda.current_stream().synchronize()
   actual=digest(gradients) if self.args.audit_state else None
   if actual!=packet['gradient_hash']:raise RuntimeError('gradient relay mismatch')
   expected_tail=digest(packet['tail']) if self.args.audit_state else None
   applied_tail=digest(applied) if self.args.audit_state and applied is not None else expected_tail
   if applied_tail!=expected_tail:raise RuntimeError('applied tail mismatch')
   self.emit('planned_gradient_received',applied_tail_hash=applied_tail,step=self.cut,sample_ids=sample_ids,gradient_hash=actual,tail_hash=digest(packet['tail']) if self.args.audit_state else None,forward_calls=0)
   self.relay_payload=None;self.relay_done=True
   return manager,False
  torch.cuda.current_stream().synchronize();self.emit('planned_local_gradient_ready',step=self.cut)
  self.coord.set(f'gradient_ready/{self.role}/{self.rank}','1')
  if self.role=='A' and self.rank==0:
   keys=[f'gradient_ready/{role}/{r}' for role in ('A','B') for r in range(self.local_world)]+[f'preloaded/{r}' for r in range(self.local_world)]
   end=time.monotonic()+60
   while not self.coord.check(keys) and not self.coord.check(['cancel']):
    if time.monotonic()>end:self.coord.set('cancel','precommit deadline');break
    time.sleep(.002)
   self.coord.set('decision','abort' if self.coord.check(['cancel']) else 'commit')
  self.coord.wait(['decision'])
  if self.coord.get('decision')==b'abort':
   if self.presend:
    self.presend.join(timeout=60)
    if self.presend.is_alive():raise RuntimeError('cannot resume writes while precopy is active')
   self.check();self.close_prepared();self.cancelled=True;self.relay_done=True
   self.emit('planned_cancelled',step=self.cut);return manager,False
  self.fence(manager)
  if self.role=='B':
   self.presend.join(timeout=60)
   if self.presend.is_alive():raise RuntimeError('precopy sender timed out')
   self.check();self.coord.wait([f'preloaded/{self.rank}'])
   tail=self.export.role_metadata() if hasattr(self.export,'role_metadata') else dict(self.metadata(),**self.rng_snapshot())
   packet=dict(step=self.cut,sample_step=self.cut+1,sample_ids=sample_ids,gradients=gradients,tail=tail,gradient_hash=digest(gradients) if self.args.audit_state else None)
   if getattr(self.args,'planned_inject','healthy')=='relay-reject' and self.rank==0:packet['gradient_hash']='invalid'
   self.transport().send_checkpoint([1],self.cut,packet,timedelta(seconds=60))
   self.emit('planned_gradient_sent',step=self.cut,sample_ids=sample_ids,gradient_hash=packet['gradient_hash'],tail_hash=digest(tail) if self.args.audit_state else None)
   self.coord.wait(['activation_decision'])
   if self.coord.get('activation_decision')==b'AB':return self.recover_ab(manager,make_manager)
   self.coord.wait([f'loaded/{self.rank}']);self.retired_pgs.extend(self.channels.values());self.channels={}
   self.gate.activate(self.new_epoch,self.cut,self.cut,True)
   try:self.authorize(manager)
   except ValueError:self.emit('planned_old_commit_rejected',step=self.cut,epoch=self.base_epoch)
   else:raise RuntimeError('old owner accepted')
   self.emit('planned_retired_alive',step=self.cut);self.start_retired_cleanup();self.coord.wait(['ac_three']);self.finish_retired_cleanup();self.emit('planned_standby_verified',step=self.cut)
   return None,True
  if self.rank==0:
   ok=[f'relay_ok/{r}' for r in range(self.local_world)];rejected=[f'relay_rejected/{r}' for r in range(self.local_world)]
   end=time.monotonic()+60
   while not self.coord.check(ok) and not any(self.coord.check([key]) for key in rejected):
    if time.monotonic()>end:raise RuntimeError('relay validation timed out')
    time.sleep(.002)
   self.coord.set('activation_decision','AC' if self.coord.check(ok) else 'AB')
  self.coord.wait(['activation_decision'])
  if self.coord.get('activation_decision')==b'AB':return self.recover_ab(manager,make_manager)
  self.coord.wait([f'loaded/{r}' for r in range(self.local_world)])
  self.gate.activate(self.new_epoch,self.cut,self.cut,True);self.args.lighthouse=f'http://10.10.4.2:{31410+self.offset}';self.start_retired_cleanup()
  manager=make_manager();manager.start_quorum();manager.wait_quorum();self.relay_done=True
  self.emit('planned_epoch_activated',step=self.cut,epoch=self.new_epoch)
  return manager,False

 def recover_ab(self,old_manager,make_manager):
  # No optimizer update has occurred. Preserve ready A/B gradients and state,
  # replace the closed Managers with a new epoch; never revive old handles.
  self.fallback=True;self.relay_done=True
  if self.candidate:self.retired_pgs.append(self.candidate);self.candidate=None
  self.retired_pgs.extend(self.channels.values());self.channels={}
  if self.epoch_store is None:self.epoch_store=dist.TCPStore(self.args.host_ip,31600+self.offset,None,self.rank==0,timeout=timedelta(seconds=60),wait_for_workers=False)
  self.gate.activate(self.new_epoch,self.cut,self.cut,True);self.gate.members=('A','B')
  self.args.lighthouse=f'http://10.10.4.2:{31411+self.offset}'
  manager=make_manager();manager.start_quorum();manager.wait_quorum();self.start_retired_cleanup()
  try:self.authorize(old_manager)
  except ValueError:pass
  else:raise RuntimeError('fallback revived old manager')
  self.emit('planned_new_epoch_ab_recovered',step=self.cut,epoch=self.new_epoch,old_manager_rejected=True)
  return manager,False

 def authorize(self,manager):
  if self.old_epoch_revoked and manager._shift_epoch==self.base_epoch:raise ValueError('old epoch fenced')
  self.gate.authorize(self.role,manager._shift_epoch)
 def commit_fields(self,manager):
  fields=super().commit_fields(manager)
  old='C' if self.round%2 else 'B';target='B' if self.round%2 else 'C'
  fields.update(physical_members=['A',old if self.fallback else target if self.activated else old],planned_round=self.round)
  return fields
 def committed(self):
  if not self.activated:return
  owner='C' if self.fallback and self.role=='B' else self.role
  if self.progress['updates']==self.cut+1:self.coord.set(f'ac_first/{owner}/{self.rank}','1')
  if self.progress['updates']>=self.cut+3:
   self.coord.set(f'ac3/{owner}/{self.rank}','1')
   if self.role=='A' and self.rank==0:
    self.coord.wait([f'ac3/{role}/{rank}' for role in ('A','C') for rank in range(self.local_world)]);self.coord.set('ac_three','1')
