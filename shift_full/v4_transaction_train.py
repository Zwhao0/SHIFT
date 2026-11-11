"""Repeatable, validated-before-switch intra-replica replacement with peer overlap."""
from dataclasses import replace
import gc
import json
import threading
import time
from types import SimpleNamespace

from .v4_transaction import OwnerGate, Transaction
from .session_layout import PartialScope, SessionLayout


def run(context):
    s=SimpleNamespace(**context);c=s.c;torch,dist=s.torch,s.dist
    layout=SessionLayout.from_config(c)
    role,rank,device,replica=s.role,s.rank,s.device,s.replica
    first=c.get('initial_round',0);cycles=c['cycles'];epoch=first
    source=c.get('source_role','B');target=c.get('target_role','C')
    session_scope = getattr(s, 'partial_scope', None)
    scope = session_scope or PartialScope(0, tuple(c.get('affected_ranks', layout.affected)), 0)
    is_peer = replica != scope.replica
    is_coordinator = scope.coordinates(replica, rank)
    channel_ranks = c.get('channel_ranks', scope.ranks)
    rebuild_outer = getattr(s, 'rebuild_outer', False)
    candidate = getattr(s, 'partial_candidate', False) if session_scope else not replica and role == target
    if session_scope:
        if candidate and not scope.owns(replica, rank):
            raise ValueError('candidate is outside the selected stage')
        role = target if candidate else source if scope.owns(replica, rank) else 'A'
    from .v4_state import boundary_validation,boundary_descriptor,validate_candidate_state
    mode=boundary_validation(c);audit=mode=='full'
    def boundary_hashes(packet=None):
        return (packet['hashes'] if packet is not None else s.hashes()) if audit else None
    gate=s.ownership if getattr(s,'session_protocol',False) else OwnerGate(first,source)
    outer=getattr(s,'outer',None);groups=s.groups
    outer_factory=getattr(s,'outer_factory',None)
    identity=s.identity_preserved
    def bind(**kwargs):s.configure_runtime(**kwargs)
    bind(ownership_value=gate,role_value='A' if is_peer else role)
    if not candidate:
        from .v4_outer import Outer
        saved=s.rng()
        if outer is None:outer=(outer_factory or Outer)(c,replica,rank,first,s.event)
        s.restore_rng(saved)
        bind(outer_value=outer)
        if s.independent_reference:
            from .v4_reference import compare_signature
            hashes=s.hashes()
            if not compare_signature(s.independent_reference,replica=replica,rank=rank,step=0,
                    hashes=hashes,samples=[],manager=outer.state_dict()):raise RuntimeError('independent initial state mismatch')
            s.event('reference_initial',step=0,hashes=hashes,sample_ids=[],outer_manager=outer.state_dict())

    class Channel:
        def __init__(self,store,round_id,is_source,kind='source'):
            self.store,self.round_id,self.is_source=store,round_id,is_source
            self.kind=kind
            self.pg=None;self.error=None;self.thread=None
        def prepare(self):
            started=time.monotonic()
            pg=None
            try:
                torch.cuda.set_device(device)
                if self.is_source:
                    while not self.store.check([f'{self.kind}_channel_start/{rank}']):time.sleep(.05)
                else:self.store.set(f'{self.kind}_channel_start/{rank}','1')
                stream=torch.cuda.Stream(device=device)
                with torch.cuda.stream(stream):
                    pg=s.BlockingNCCL(timeout=s.timeout)
                    pg.configure(f"{c['hosts']['A']}:{c['port']}/v4-switch-{self.kind}/{self.round_id+1}/{rank}",
                        f'v4-switch-{self.round_id+1}-{rank}',0 if self.is_source else 1,2)
                    value=torch.ones(1,device='cuda');pg.allreduce([value],dist.AllreduceOptions()).wait(s.timeout)
                    stream.synchronize()
                    if value.item()!=2:raise RuntimeError('candidate channel warmup mismatch')
                self.pg=pg
                s.event('switch_channel_prepared',round=self.round_id,epoch=self.round_id+1,seconds=time.monotonic()-started)
                if self.is_source:self.store.set(f'{self.kind}_channel_ready/{rank}','1')
                else:self.store.wait([f'{self.kind}_channel_ready/{rank}'])
            except BaseException as error:
                self.error=error
                if pg is not None:pg.shutdown()
        def start(self):
            self.thread=threading.Thread(target=self.prepare,daemon=True);self.thread.start()
        def transfer(self,packet=None):
            if self.thread:
                self.thread.join(s.timeout.total_seconds())
                if self.thread.is_alive():raise TimeoutError('candidate channel preparation deadline')
            if self.error:raise RuntimeError('candidate channel failed') from self.error
            if self.pg is None:self.prepare()
            if self.error:raise RuntimeError('candidate channel failed') from self.error
            stats={}
            def event(kind,**fields):
                if kind in ('transport_send','transport_recv'):stats.update(fields)
                s.event(kind,round=self.round_id,**fields)
            from .transport import PackedGPUTransport
            transport=PackedGPUTransport(self.pg,torch.device('cuda',device),s.timeout,event)
            cut=int(self.store.get('actual_cut'))
            try:
                if self.is_source:transport.send_checkpoint([1],cut,packet,s.timeout)
                else:packet=transport.recv_checkpoint(0,transport.metadata(),cut,s.timeout)
                if getattr(transport,'_stage_cpu',False):
                    # CPU leaf decoding can outlast the source's last send.
                    # Keep its NCCL channel alive until receive/copy has really
                    # finished, including the final CUDA stream synchronization.
                    ack=f'{self.kind}_delivered/{rank}'
                    if self.is_source:self.store.wait([ack],s.timeout)
                    else:self.store.set(ack,'1')
                s.event('state_sent' if self.is_source else 'state_received',round=self.round_id,
                    packet_source=self.kind,
                    step=cut,hashes=boundary_hashes(packet),oracle_hashes=None,transport='packed_gpu',
                    bytes=stats['gpu_bytes']+stats['control_bytes'],
                    **{k:stats[k] for k in ('gpu_bytes','control_bytes','cpu_tensor_bytes')})
                return packet
            finally:self.pg.shutdown();self.pg=None

    def serve_peer(packet,control,round_id):
        try:
            while not control.check(['packet_source']):time.sleep(.02)
            if control.get('packet_source')!=b'peer':return
            expected=json.loads(control.get(f'source_descriptor/{rank}'))
            if boundary_descriptor(packet)!=expected:
                raise RuntimeError('peer checkpoint version differs from the frozen source')
            expected_hashes=json.loads(control.get(f'source_hashes/{rank}')) if audit else None
            if audit and packet['hashes']!=expected_hashes:
                raise RuntimeError('peer checkpoint differs from the frozen logical source state')
            s.event('peer_state_authorized',round=round_id,step=packet['step'],
                hashes=boundary_hashes(packet),expected_hashes=expected_hashes,version_checked=True)
            relay=Channel(control,round_id,True,kind='peer');relay.prepare();relay.transfer(packet)
        except BaseException as error:
            control.set('recovery_error',type(error).__name__)
            s.event('peer_recovery_error',round=round_id,error=type(error).__name__)

    for round_id in range(first,cycles):
        c['round_id']=round_id
        store=dist.PrefixStore(f'round/{round_id}/',s.base_store)
        bind(store_value=store)
        next_epoch=round_id+1
        segment=c['between_steps'] if round_id+1<cycles else c['plan']['post_steps']
        if round_id==first:
            requested=c['plan']['cut'] if not candidate else None
        else:
            source=json.loads(store.get('placement'))['source'];target='C' if source=='B' else 'B'
            candidate=not replica and role==target
            requested=s.progress['updates']
        channel=None
        if not is_peer and role in (source,target):
            channel=Channel(store,round_id,role==source)
            if role==source:channel.start()

        if not candidate:
            plan=replace(s.plan,cut=requested,post_steps=segment)
            bind(plan_value=plan)
            s.train(epoch,max(0,requested-s.progress['updates']),'before')
            while True:
                ready=False
                if rank==0:
                    key=f'boundary_ready/{s.progress["updates"]}'
                    if not is_peer:
                        ready=store.check([f'prepared/{i}' for i in range(layout.width)])
                        store.set(key,str(int(ready)))
                    else:ready=bool(int(store.get(key)))
                flag=torch.tensor(int(ready),device='cuda');dist.broadcast(flag,src=0)
                if flag.item():break
                s.train(epoch,1,'before')
            cut=s.progress['updates'];plan=replace(plan,cut=cut);bind(plan_value=plan)
            if rank==0 and not is_peer:store.set('actual_cut',str(cut))
            s.event('boundary_selected',round=round_id,epoch=epoch,step=cut,requested_cut=requested)
            # The affected replica is frozen until activation/retirement. Its
            # snapshot can borrow read-only tensors; the peer needs a copy
            # because it overlaps the next iteration and supplies fallback.
            borrowed=bool(c.get('low_memory') and not is_peer)
            anchor=s.snapshot(copy_state=False) if borrowed else s.snapshot()
            anchor.update(v4_epoch=next_epoch,v4_owner=target,switch_round=round_id)
            from torch.utils._pytree import tree_flatten
            source_versions=[(t,t._version) for t in tree_flatten(anchor)[0] if isinstance(t,torch.Tensor)] if borrowed else []
            if role==source:
                store.set(f'source_descriptor/{rank}',json.dumps(boundary_descriptor(anchor)))
                if audit:store.set(f'source_hashes/{rank}',json.dumps(anchor['hashes']))
            boundary_outer=outer.identity();boundary_training=s.training_identity()
            s.event('freeze',round=round_id,step=cut,hashes=boundary_hashes(),identity_preserved=identity())
            s.event('outer_retention_boundary',round=round_id,step=cut,identity=boundary_outer,
                training_identity=boundary_training,manager_state=outer.state_dict())
            store.set(f'frozen/{replica}/{rank}','1')
            gate.fence()
            if is_peer:
                relay_thread=None
                if rank in c.get('affected_ranks',layout.affected):
                    relay_thread=threading.Thread(target=serve_peer,args=(anchor,store,round_id),daemon=True)
                    relay_thread.start()
                outer.migrate(next_epoch);s.restore_rng(anchor)
                gate.advance(next_epoch,source)
                s.event('outer_retained',round=round_id,step=cut,identity=outer.identity(),
                    training_identity=s.training_identity(),manager_state=outer.state_dict(),
                    channel_reconnected=rank in channel_ranks)
                s.event('peer_inner_groups_retained',round=round_id,identity_preserved=True)
                s.event('peer_compute_ready',round=round_id,step=cut,hashes=boundary_hashes(),identity_preserved=identity())
                store.set(f'active/{replica}/{rank}','1')
                del anchor;gc.collect();torch.cuda.empty_cache()
                s.train(next_epoch,segment,'after')
                if relay_thread:
                    relay_thread.join(c['restore_deadline']+10)
                    if relay_thread.is_alive():raise TimeoutError('peer recovery sender did not drain')
                gate.owner=json.loads(store.get('decision'))['owner']
                epoch=next_epoch
                if round_id+1<cycles:
                    next_store=dist.PrefixStore(f'round/{round_id+1}/',s.base_store)
                    next_store.wait(['placement'])
                continue
            if is_coordinator:
                store.wait([f'frozen/{r}/{i}' for r in range(2) for i in range(layout.world)])
                store.set('switch','1')
            store.wait(['switch'])
            if role==source:
                if c.get('source_stall_round',-1)==round_id:
                    try:
                        gate.authorize(role,epoch)
                        raise AssertionError('stalled source was not fenced')
                    except ValueError:s.event('old_owner_rejected',round=round_id,step=cut)
                    s.event('source_stalled',round=round_id,step=cut)
                    while True:time.sleep(.1)  # Exact owned unit is stopped by the controller.
                store.set(f'source_ready/{device}','1')
            if is_coordinator:
                until_source=time.monotonic()+c.get('source_deadline',2)
                while not store.check([f'source_ready/{i}' for i in range(layout.width)]) and time.monotonic()<until_source:
                    time.sleep(.01)
                packet_source='source' if store.check([f'source_ready/{i}' for i in range(layout.width)]) else 'peer'
                store.set('packet_source',packet_source)
                if packet_source=='peer':
                    s.event('source_recovery_selected',round=round_id,step=cut,source=source,target=target)
            packet_source=store.get('packet_source').decode()
            if role==source:
                if packet_source=='peer':
                    s.event('source_stalled',round=round_id,step=cut)
                    while True:time.sleep(.1)
                channel.transfer(anchor)
                if any(t._version!=version for t,version in source_versions):
                    raise RuntimeError('frozen source changed while transmitting borrowed state')
                if borrowed:s.event('source_readonly_verified',round=round_id,step=cut)
        else:
            # Candidate model/DDP was prepared in isolation by v4_train.
            channel.prepare()
            store.set(f'prepared/{device}','1');s.event('candidate_prepared',round=round_id)
            store.wait(['switch']);cut=int(store.get('actual_cut'))
            plan=replace(s.plan,cut=cut,post_steps=segment);bind(plan_value=plan,comparison_start=cut)
            packet_source=store.get('packet_source').decode()
            if packet_source=='peer':
                if channel.pg is not None:
                    channel.pg.abort();channel.pg.shutdown();channel.pg=None
                channel=Channel(store,round_id,False,kind='peer');channel.prepare()
            anchor=channel.transfer()
            from .planned_state import validate_state
            validate_state(anchor,rank=rank,step=cut,audit=audit)
            if not anchor['optimizer'].get('state'):raise ValueError('candidate needs complete Adam state')
            if anchor.get('v4_epoch')!=next_epoch or anchor.get('v4_owner')!=role:
                raise ValueError('stale candidate boundary')
            if c.get('stall_restore_rank',-1)==rank and round_id==c.get('fault_round',0):
                s.event('candidate_restore_stalled',round=round_id,step=cut)
                while not store.check(['decision']):time.sleep(.05)
            if not store.check(['decision']):
                actual=s.hashes() if audit else None
                readiness=None
                try:
                    structure_started=time.monotonic()
                    validate_candidate_state(anchor,rank=rank,cut=cut,epoch=next_epoch,target=role,
                        round_id=round_id,source_replica=1 - scope.replica if packet_source=='peer' else scope.replica,
                        stage=s.stage,optimizer=s.optimizer,scheduler=s.scheduler,rng=s.rng(),progress=s.progress,
                        fixed_replicas=not c.get('unified_session',False))
                    s.event('candidate_structure_validated',step=cut,seconds=time.monotonic()-structure_started)
                    s.restore(anchor)
                    if c.get('fail_restore_rank',-1)==rank and round_id==c.get('fault_round',0):
                        with torch.no_grad():next(s.stage.parameters()).add_(.125)
                    actual=s.hashes() if audit else None
                    valid=actual==anchor['hashes'] if audit else True
                    if valid:readiness=dict(schema=1,epoch=next_epoch,rank=rank,cut=cut,target=role,restored=True)
                    reason=None if valid else 'complete state signature mismatch'
                except BaseException as error:
                    valid=False;reason=type(error).__name__+': '+str(error)
                store.set(f'candidate/{device}',json.dumps(dict(rank=rank,actual=actual,
                    expected=boundary_hashes(anchor),valid=valid,reason=reason,readiness=readiness)))
                s.event('candidate_validated' if valid else 'candidate_rejected',round=round_id,
                    step=cut,hashes=actual,expected_hashes=boundary_hashes(anchor),reason=reason,readiness=readiness)

        if is_coordinator:
            ranks=tuple(c.get('affected_ranks',layout.affected))
            tx=Transaction(next_epoch,source,target,cut,ranks,validation=mode,world_size=layout.world)
            deadline=time.monotonic()+c['restore_deadline']
            while not store.check([f'candidate/{i}' for i in range(layout.width)]):
                if store.check(['recovery_error']):raise RuntimeError('peer could not supply the committed source state')
                if time.monotonic()>=deadline:break
                time.sleep(.05)
            timed_out=not store.check([f'candidate/{i}' for i in range(layout.width)])
            if not timed_out:
                for i in range(layout.width):
                    result=json.loads(store.get(f'candidate/{i}'))
                    if audit:tx.candidate(next_epoch,result['rank'],result['actual'] if result['valid'] else {},result['expected'])
                    else:tx.ready(next_epoch,result['rank'],result['readiness'] if result['valid'] else None)
            decision=tx.decide(cancel=timed_out or c.get('cancel_round',-1)==round_id)
            recovered_source=store.get('packet_source')==b'peer'
            if recovered_source and decision!='replace':raise RuntimeError('lost source has no validated replacement')
            selected=target if decision=='replace' else source
            store.set('decision',json.dumps(dict(decision=decision,owner=selected,epoch=next_epoch,
                cut=cut,timeout=timed_out,source_recovery=recovered_source)))
            s.event('switch_decision',round=round_id,step=cut,decision=decision,owner=selected,
                epoch=next_epoch,timeout=timed_out,candidate_results=tx.results,source_recovery=recovered_source,validation=mode)
            if recovered_source:
                # Decision metadata is immutable; annotate the selected recovery source before admission.
                store.set('source_recovered','1')
            if round_id+1<cycles:
                future=dist.PrefixStore(f'round/{round_id+1}/',s.base_store)
                future.set('placement',json.dumps(dict(source=selected)))
                future.set('prepare','1')
        outcome=json.loads(store.get('decision'));selected=outcome['owner']
        source_recovered=store.get('packet_source')==b'peer'
        if candidate and outcome['decision']=='retain':
            s.event('candidate_cancelled',round=round_id,step=cut)
            if dist.is_initialized():s.disconnect(preserve=False)
            s.event('standby_done',round=round_id,step=cut);s.log.close();return
        if role==source and outcome['decision']=='replace':
            # Only now may the old source close. It was kept intact for rollback.
            outer.close();bind(outer_value=None)
            s.disconnect(preserve=s.local_cache is not None)
            try:
                gate.authorize(role,epoch)
                raise AssertionError('retired source authorization revived')
            except ValueError:s.event('old_owner_rejected',round=round_id,step=cut)
            store.wait(['activated'])
            if s.local_cache:
                counts=s.local_cache.finish();s.event('source_local_groups_closed',round=round_id,**counts)
            s.event('retired',round=round_id,step=cut);s.log.close();return dict(retired=True)

        old=(s.metadata_handles['tp'] if candidate and s.metadata_handles else s.state.get_tensor_model_parallel_group(),
             groups[0],groups[2],groups[1])
        if outcome['decision']=='replace':
            if dist.is_initialized():s.disconnect(preserve=s.local_cache is not None,abort=source_recovered and not candidate)
            groups=s.connect(next_epoch)
            s.rebind(s.stage,s.ddp,old,(s.state.get_tensor_model_parallel_group(),groups[0],groups[2],groups[1]))
            if candidate and s.metadata_handles:
                from .v4_metadata import MetadataGroup
                owners=[s.ddp,*s.stage.modules()]
                for buffer in s.ddp.buffers+s.ddp.expert_parallel_buffers:owners.extend([buffer,*buffer.buckets])
                if any(isinstance(value,MetadataGroup) for owner in owners for value in vars(owner).values()):
                    raise RuntimeError('metadata-only group survived activation')
        else:s.event('inner_groups_retained_after_cancel',round=round_id,step=cut)
        replaced_outer = bool(rebuild_outer and not is_peer and outcome['decision'] == 'replace')
        if candidate or replaced_outer:
            from .v4_outer import Outer
            if not candidate:outer.close()
            outer=(outer_factory or Outer)(c,replica,rank,next_epoch,s.event);outer.load_state_dict(anchor['outer_manager'])
            bind(outer_value=outer)
        else:outer.migrate(next_epoch)
        s.restore_rng(anchor)
        validated_hashes=boundary_hashes()
        if (audit and validated_hashes!=anchor['hashes']) or not identity():raise RuntimeError('switch changed committed state')
        if not candidate and s.training_identity()!=boundary_training:raise RuntimeError('retained Adam/storage identity changed')
        gate.advance(next_epoch,selected);epoch=next_epoch
        event_kind = 'candidate_activated' if candidate else 'outer_rebuilt' if replaced_outer else 'outer_retained'
        s.event(event_kind,round=round_id,step=cut,
            epoch=epoch,decision=outcome['decision'],owner=selected,hashes=validated_hashes,identity_preserved=identity(),
            identity=outer.identity(),training_identity=s.training_identity(),manager_state=outer.state_dict(),
            channel_reconnected=rank in channel_ranks,
            readiness=dict(schema=1,epoch=next_epoch,rank=rank,cut=cut,target=selected,restored=True) if candidate else None)
        active=dict(round=round_id,epoch=epoch,step=cut,hashes=validated_hashes,identity_preserved=identity(),
            owner=selected,decision=outcome['decision'],
            peer_compute_started=store.check([f'peer_compute/{i}' for i in range(layout.world)]),
            peer_gradients_ready=store.check([f'peer_gradient_ready/{i}' for i in range(layout.world)]))
        store.set(f'active/{replica}/{rank}','1')
        if rank==0:
            store.wait([f'active/{r}/{i}' for r in range(2) for i in range(layout.world)]);store.set('activated','1')
        store.wait(['activated']);s.event('activated',**active)
        del anchor;gc.collect();torch.cuda.empty_cache()
        candidate=False
        s.train(epoch,segment,'after')
    s.finish()
    return dict(outer=outer,groups=groups,gate=gate,epoch=epoch,owner=gate.owner)
