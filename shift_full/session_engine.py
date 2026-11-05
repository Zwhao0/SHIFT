"""Persistent training runtime: one loop, live commands, three recovery policies."""
from datetime import timedelta
from types import SimpleNamespace
import json
import time

from .training_core import make_batches,sample_plan,train_step
from .training_loop import run_loop
from .session_control import Command,Router
from .session_outer import SessionOuter
from .v4_transaction import OwnerGate


class _Handler:
    def __init__(self,policy):self.policy=policy
    def request(self,runtime,command):
        if self.policy=='v1':runtime.start_native(command)
        elif self.policy=='v2':runtime.start_precopy(command)
        elif self.policy=='v4':runtime.start_partial(command)
        return dict(id=command.id,status='started')


class SessionPolicy:
    def before_iteration(self,r):
        r.at_committed_boundary=True
        r.boundary()
        return not r.stopping
    def begin(self,r):
        r.at_committed_boundary=False
        r.gate.authorize('A' if r.replica else r.role,r.epoch)
        started=time.monotonic()
        r.optimizer.zero_grad(set_to_none=True);r.ddp.zero_grad_buffer()
        r.outer.begin(r.progress['updates'])
        return dict(started=started,step=r.progress['updates'])
    def vote(self,r,context,computed):
        r.gate.authorize('A' if r.replica else r.role,r.epoch)
        if r.handoff:r.handoff.authorize(r.outer.manager)
        return r.outer.vote(computed.finite)


class SessionRuntime:
    def __init__(self,context):
        self.s=SimpleNamespace(**context)
        for name in ('c','torch','dist','role','rank','device','replica','stage','ddp','optimizer','scheduler',
                     'progress','training_state','base_store','groups'):
            setattr(self,name,getattr(self.s,name))
        self.session=self.c['session'];self.epoch=self.c.get('session_epoch',0)
        self.service=self.c.get('manager_service',0)
        self.store_services=self.c.get('manager_store_services',{'0':self.service,'1':self.service})
        self.channel_epochs=self.c.get('channel_epochs',[0]*8)
        self.replica_hosts=self.c['replica_hosts']
        self.audit=not self.c.get('performance_no_audit',False)
        self.policy=SessionPolicy();self.router=Router()
        for policy in ('v1','v2','v4'):self.router.register(policy,_Handler(policy))
        self.stopping=False;self.healing=False;self.at_committed_boundary=True
        self.command=None;self.sequence=self.c.get('command_sequence',0)
        self.handoff=None;self.planned_coord=None;self.current_samples=[];self.outer=None;self.phase='normal'
        self.protocol_plan=None;self.protocol_store=None
        self.in_protocol=False;self.retired=False;self.last_commit=None;self.skips=0
        self.gate=OwnerGate(self.epoch,self.c.get('partial_owner','B'))
        self.partial_owner=self.gate.owner
        self.boot=time.monotonic()
        self.s.configure_runtime(session_runtime_value=self,ownership_value=self.gate)

    def event(self,kind,**fields):
        fields.setdefault('session_epoch',self.epoch)
        fields.setdefault('command_id',self.command.id if self.command else None)
        self.s.event(kind,**fields)

    def bind_outer(self,outer):
        self.outer=outer;self.s.configure_runtime(outer_value=outer)

    def request_stop(self):self.stopping=True
    def should_stop(self):
        if time.monotonic()-self.boot>self.c['seconds']-30:raise TimeoutError('session worker bounded deadline')
        return self.stopping

    def validate_placement(self,command):
        if command.source_replica not in (0,1):raise ValueError('source replica must be 0 or 1')
        if command.policy in ('v1','v2') and command.source_replica!=1:
            raise ValueError('this runtime currently supports whole-DP operations on replica 1')
        if command.policy=='v4' and (command.source_replica!=0 or tuple(command.ranks)!=(4,5,6,7)):
            raise ValueError('partial placement currently replaces replica 0 PP1')
        if command.target not in self.c['node_hosts']:raise ValueError('unknown target node')

    def _barrier(self,key):
        self.base_store.set(key+f'/{self.replica}/{self.rank}','1')
        self.base_store.wait([key+f'/{r}/{i}' for r in range(2) for i in range(8)])

    def boundary(self):
        if self.in_protocol:return
        if self.command:
            if self.command.policy=='v2' and self.handoff and not self.handoff.activated:
                if self.handoff.role=='A' and self.rank==0 and not self.handoff.coord.check(['arm']):
                    if self.handoff.coord.check([f'prepared/C/{i}' for i in range(8)]):self.handoff.coord.set('arm','1')
                manager,retired=self.handoff.boundary(self.outer.manager)
                self.outer.manager=manager
                if retired:self.stopping=True;return
            self._complete_command()
            return
        count=int(self.base_store.get('session/command_count'))
        if count<=self.sequence:return
        raw=json.loads(self.base_store.get(f'session/command/{self.sequence+1}'))
        command=Command.from_dict(raw['command'])
        if command.at_step is not None and self.progress['updates']<command.at_step:return
        if command.at_step is not None and self.progress['updates']!=command.at_step:
            raise RuntimeError('missed live command boundary')
        self.sequence+=1;self.c['round_id']=self.sequence
        self.c['pending_command_metadata']=raw
        self._barrier(f'session/arrival/{self.sequence}')
        result=self.router.route(command,self)
        if not self.retired:self.event('session_command_routed',policy=command.policy,result=dict(status=result.status,value=result.value))

    def _set_command(self,command):
        self.command=command
        self.event('session_command_started',policy=command.policy,step=self.progress['updates'],
                   engine_identity=self.s.training_identity())

    def start_native(self,command):
        self._set_command(command);self.healing=True
        self.outer.manager._use_async_quorum=False
        self.base_store.wait([f'session/fault_released/{self.sequence}'])

    def _planned_args(self,command,*,candidate=False):
        host_a=self.replica_hosts['0'];host_b=self.replica_hosts['1'];host_c=self.c['node_hosts'][command.target]
        # The existing p2 protocol alternates B/C by its round index.
        round_id=self.sequence-1
        initial=dict(A=host_a,B=host_b,C=host_c)
        if round_id%2:initial['B'],initial['C']=initial['C'],initial['B']
        return SimpleNamespace(planned_role='C' if candidate else 'A' if self.replica==0 else 'B',
            planned_variant='p2',planned_source='B',planned_round=round_id,planned_cycles=1000,
            planned_hosts=initial,planned_initial_role='C' if candidate else 'A' if self.replica==0 else 'B',
            planned_inject='healthy',planned_shadow_recompute=False,planned_ownership_recompute=False,
            audit_state=self.audit,host_ip=host_c if candidate else host_a if self.replica==0 else host_b,
            lighthouse=f'http://{host_a}:{31410+20*round_id}',replica=self.replica,
            backend='nccl',transport='gpu',communication_timeout=2,sync_quorum=True)

    def start_precopy(self,command,*,candidate=False):
        from .overlap_handoff import OverlapHandoff
        self._set_command(command)
        args=self._planned_args(command,candidate=candidate)
        def export():
            value=self.training_state.capture(source_replica=self.replica,source_rank=self.rank)
            value['hashes']=self.training_state.hashes() if self.audit else {}
            return value
        def metadata():
            return dict(schema=2,all_buffers=dict(self.stage.named_buffers()),source_rank=self.rank,
                        step=self.progress['updates'],progress=dict(self.progress),**self.training_state.rng())
        export.role_metadata=metadata
        def load(value):
            self.training_state.restore(value,clone_optimizer=True)
            self.optimizer.zero_grad(set_to_none=True);self.ddp.zero_grad_buffer();self.torch.cuda.synchronize()
            if self.audit and self.training_state.hashes()!=value['hashes']:raise RuntimeError('p2 restore mismatch')
        def manifest():
            return dict(rank=self.rank,parameters=[(n,tuple(p.shape),str(p.dtype)) for n,p in self.stage.named_parameters()],
                        buffers=[(n,tuple(p.shape),str(p.dtype)) for n,p in self.stage.named_buffers()],
                        optimizer_groups=[dict(params=len(g['params'])) for g in self.optimizer.param_groups])
        if self.planned_coord is not None:args._coord_raw=self.planned_coord
        self.planned_args=args
        self.handoff=OverlapHandoff(args,self.rank,self.event,export,load,self.progress,manifest,self.training_state.rng)
        self.planned_coord=self.handoff.coord_raw
        self.handoff.coord.set('request','1')
        if candidate:
            if not self.handoff.ready_receiver():raise RuntimeError('p2 candidate rejected')
            self.outer=SessionOuter.__new__(SessionOuter)
            self.outer.runtime=self;self.outer.replica=self.replica;self.outer.rank=self.rank
            self.outer.epoch=self.epoch;self.outer._step_started=None;self.outer._closed=False
            self.outer.timeout=timedelta(seconds=2);self.outer._boundary=None
            self.outer.participants=2;self.outer.participating_rank=self.replica
            self.make_planned_manager()
            self.bind_outer(self.outer)

    def make_planned_manager(self):
        import torch
        from torchft import Manager
        from torchft.manager import WorldSizeMode
        from .transport import PackedGPUTransport
        h=self.handoff;a=self.planned_args;pg=h.candidate
        if pg is None:raise RuntimeError('p2 candidate communicator was not prepared')
        manager=Manager(pg=pg,load_state_dict=self.outer.restore,state_dict=self.outer.capture,
            min_replica_size=2,world_size_mode=WorldSizeMode.FIXED_WITH_SPARES,use_async_quorum=False,
            init_sync=False,timeout=timedelta(seconds=90),quorum_timeout=timedelta(seconds=180),
            connect_timeout=timedelta(seconds=20),hostname=a.host_ip,lighthouse_addr=a.lighthouse,
            rank=self.rank,world_size=8,store_addr=a.host_ip,store_port=31600+h.offset,
            replica_id=f'shift{self.replica}',checkpoint_transport=PackedGPUTransport(pg,torch.device('cuda',self.device),timedelta(seconds=90),self.event),max_retries=8)
        manager.load_state_dict(dict(step=self.progress['updates'],batches_committed=2*self.progress['updates']))
        manager._shift_epoch=h.gate.epoch
        self.outer.manager=manager;self.outer.pg=pg
        return manager

    def apply_tail(self,tail):
        from .state_inventory import restore_buffers
        restore_buffers(self.stage,tail['all_buffers']);self.training_state.restore_rng(tail)
        self.progress.clear();self.progress.update(tail['progress'])
        return dict(schema=2,all_buffers=dict(self.stage.named_buffers()),source_rank=self.rank,
                    step=self.progress['updates'],progress=dict(self.progress),
                    planned_epoch=tail['planned_epoch'],planned_role=1,**self.training_state.rng())

    def gradient_finalize(self,native_finish):
        h=self.handoff;relay=bool(h and h.is_relay_step())
        if not (relay and h.role=='C'):native_finish()
        peer_first=self.in_protocol and self.replica==1 and self.phase=='after' and self.protocol_plan and self.progress['updates']==self.protocol_plan.cut
        if peer_first:
            self.protocol_store.set(f'peer_gradient_ready/{self.rank}','1')
            self.event('peer_native_gradient_ready',step=self.progress['updates'])
        buckets=[b for buf in self.ddp.buffers+self.ddp.expert_parallel_buffers for b in buf.buckets]
        if relay:
            manager,retired=h.at_gradient(self.outer.manager,[b.grad_data for b in buckets],
                self.make_planned_manager,self.current_samples,self.apply_tail)
            if retired:
                from .overlap_handoff import RetiredReplica
                raise RetiredReplica()
            self.outer.manager=manager;self.outer.pg=manager._pg
        self.outer.reduce(buckets,self.progress['updates'])

    def compute(self,context):
        if self.in_protocol and self.replica==1 and self.phase=='after' and self.protocol_plan and self.progress['updates']==self.protocol_plan.cut:
            self.protocol_store.set(f'peer_compute/{self.rank}','1')
            self.event('peer_compute_begin',step=self.progress['updates'])
        r=self.outer
        self.current_samples=sample_plan(self.progress['updates'],microbatches=2,
            participants=r.participants,replica=r.participating_rank,
            local_data_rank=self.s.state.get_data_parallel_rank())
        batches=make_batches(self.current_samples,seq_len=self.c['plan']['seq_len'],
                             vocab_size=self.stage.hf_config.vocab_size,data='random')
        relay=bool(self.handoff and self.handoff.role=='C' and self.handoff.is_relay_step())
        def execute():
            if relay:
                self.stage.config.finalize_model_grads_func([self.ddp]);return []
            schedule=self.s.get_forward_backward_func()
            return schedule(forward_step_func=self.s.forward_step,data_iterator=iter(batches),model=self.ddp,
                num_microbatches=len(batches),seq_length=self.c['plan']['seq_len']+self.stage.extra_positions(1),micro_batch_size=1)
        return train_step(self.stage,self.ddp,self.optimizer,self.scheduler,self.progress,execute=execute)

    def committed(self,context,computed,update):
        self.at_committed_boundary=True;self.skips=0
        if self.handoff:self.handoff.committed()
        now=time.monotonic()
        hashes=self.training_state.hashes() if self.audit else None
        self.event('session_commit',step=self.progress['updates'],sample_ids=self.current_samples,
            participants=self.outer.manager.num_participants(),contributed=self.outer.manager.is_participating(),participating_rank=self.outer.manager.participating_rank(),
            finite=computed.finite,seconds=now-context['started'],
            commit_gap=now-self.last_commit if self.last_commit else None,
            hashes=hashes,phase=self.phase)
        self.last_commit=now
        if self.rank==0:self.base_store.set(f'session/progress/{self.replica}',str(self.progress['updates']))

    def skipped(self,context,computed):
        self.at_committed_boundary=True;self.skips+=1
        self.event('session_skipped',step=self.progress['updates'],finite=computed.finite)
        if self.skips>=8:raise RuntimeError('session exceeded native retry bound')

    def _complete_command(self):
        command=self.command;key=f'session/complete_step/{self.sequence}'
        if command.policy=='v1':
            ready=self.base_store.check([f'session/native_restored/{self.sequence}/{i}' for i in range(8)])
            if self.replica==0 and self.rank==0 and ready and self.outer.manager.num_participants()==2 and not self.base_store.check([key]):
                self.base_store.set(key,str(self.progress['updates']+5))
        elif command.policy=='v2' and self.handoff and self.handoff.activated:
            if self.replica==0 and self.rank==0 and not self.base_store.check([key]):self.base_store.set(key,str(self.handoff.cut+5))
        if not self.base_store.check([key]):return
        cut=int(self.base_store.get(key))
        if self.progress['updates']<cut:return
        if self.progress['updates']!=cut:raise RuntimeError('missed strategy completion boundary')
        self._barrier(f'session/completed/{self.sequence}')
        self.epoch=command.expected_epoch+1;self.healing=False
        self.replica_hosts['1']=self.c['node_hosts'][command.target]
        self.outer.manager._use_async_quorum=True
        self.gate.advance(self.epoch,self.partial_owner)
        self.outer.manager._shift_epoch=self.epoch
        if command.policy=='v2':
            old_handoff=self.handoff
            old_handoff.finish_retired_cleanup()
            self._barrier(f'session/planned_drained/{self.sequence}')
            self.outer.close()
            self._barrier(f'session/planned_managers_closed/{self.sequence}')
            self.handoff=None;self.service=self.sequence;self.channel_epochs=[self.epoch]*8
            self.store_services={'0':self.service,'1':self.service}
            self.replica_hosts['1']=self.c['node_hosts'][command.target]
            self._barrier(f'session/rebuild/{self.sequence}')
            self.bind_outer(SessionOuter(self))
            self._barrier(f'session/new_outer_ready/{self.sequence}')
        self.base_store.set(f'session/done/{self.sequence}/{self.replica}/{self.rank}',str(self.progress['updates']))
        self.event('session_command_done',policy=command.policy,step=self.progress['updates'],
                   engine_identity=self.s.training_identity())
        self.command=None

    def configure(self,**kwargs):
        self.s.configure_runtime(**kwargs)
        if kwargs.get('store_value',...) is not ...:self.protocol_store=kwargs['store_value']
        if kwargs.get('plan_value',...) is not ...:self.protocol_plan=kwargs['plan_value']
        if kwargs.get('outer_value',...) is not ...:self.outer=kwargs['outer_value']
        if kwargs.get('ownership_value',...) is not ...:self.gate=kwargs['ownership_value']

    def train_segment(self,epoch,count,phase):
        self.epoch=epoch;self.phase=phase
        run_loop(self,self.policy,until=self.progress['updates']+count,max_attempts=count+8)

    def start_partial(self,command,*,candidate=False):
        from .v4_transaction_train import run
        self._set_command(command);self.healing=False
        first=command.expected_epoch
        source=self.partial_owner;target='C' if source=='B' else 'B'
        self.c.update(initial_round=first,cycles=first+1,source_role=source,target_role=target,
                      between_steps=5,restore_deadline=10,source_deadline=2,
                      affected_ranks=[4,5,6,7],round_id=first)
        self.c['plan']=dict(self.c['plan'],cut=(command.at_step or self.progress['updates'])+2,post_steps=5)
        context=vars(self.s).copy()
        context.update(outer=self.outer,groups=self.groups,ownership=self.gate,train=self.train_segment,
            configure_runtime=self.configure,finish=lambda:None,session_protocol=True)
        def factory(c,replica,rank,epoch,event):
            value=SessionOuter(self);value.epoch=epoch;value.pg.migration_epoch=epoch;value.manager._shift_epoch=epoch;return value
        context['outer_factory']=factory
        self.in_protocol=True
        try:result=run(context)
        finally:self.in_protocol=False;self.phase='normal'
        if result and result.get('retired'):
            self.stopping=True;self.retired=True;return
        self.outer=result['outer'];self.groups=result['groups'];self.gate=result['gate']
        self.epoch=result['epoch'];self.partial_owner=result['owner']
        self.channel_epochs[4:8]=[self.epoch]*4
        self.base_store.set(f'session/done/{self.sequence}/{self.replica}/{self.rank}',str(self.progress['updates']))
        self.event('session_command_done',policy='v4',step=self.progress['updates'],engine_identity=self.s.training_identity())
        self.command=None

    def finish(self):
        self._barrier('session/final_ready')
        hashes=self.training_state.hashes()
        self.event('session_done',step=self.progress['updates'],hashes=hashes,engine_identity=self.s.training_identity())
        self.outer.close();self.s.disconnect();self.s.log.close()


def run_session(context):
    r=SessionRuntime(context)
    candidate=r.c.get('session_candidate')
    try:
        if candidate:
            raw=r.c['session_command'];command=Command.from_dict(raw['command'])
            r.sequence=raw['sequence'];r.c['round_id']=r.sequence
            if command.policy=='v2':r.start_precopy(command,candidate=True)
            elif command.policy=='v4':r.start_partial(command,candidate=True)
            elif command.policy=='v1':
                r._set_command(command);r.healing=True
                r.replica_hosts['1']=r.c['node_hosts'][command.target]
                r.bind_outer(SessionOuter(r))
        else:
            r.bind_outer(SessionOuter(r))
            r.base_store.set(f'session/ready/{r.replica}/{r.rank}','1')
            r.base_store.wait(['session/start'])
        if r.retired:return
        run_loop(r,r.policy,max_attempts=1_000_000)
        if r.retired or not r.stopping or r.c.get('retiring'):return
        r.finish()
    except BaseException as error:
        from .overlap_handoff import RetiredReplica
        if isinstance(error,RetiredReplica):
            r.event('session_retired',step=r.progress['updates']);r.s.disconnect();r.s.log.close();return
        raise
