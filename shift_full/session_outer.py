"""One persistent-session TorchFT adapter for ordinary, healing and planned steps."""
import time
from datetime import timedelta


class SessionOuter:
    def __init__(self,runtime,*,service=None):
        import torch
        from torchft import Manager
        from .compat import BlockingNCCL,install_managed_callback_fix
        from .v4_outer import MigrationNamespace
        from .transport import PackedGPUTransport
        install_managed_callback_fix()
        self.runtime=runtime;self.replica=runtime.replica;self.rank=runtime.rank
        self.epoch=runtime.epoch;self._step_started=None;self._closed=False
        self.timeout=timedelta(seconds=runtime.c.get('communication_timeout',2))
        service=runtime.service if service is None else service
        class ScopedNCCL(MigrationNamespace,BlockingNCCL):pass
        self.pg=ScopedNCCL(timeout=self.timeout);self.pg.migration_epoch=runtime.channel_epochs[self.rank]
        host=runtime.c['hosts']['A'];port=runtime.c['port']
        self.manager=Manager(pg=self.pg,state_dict=self.capture,load_state_dict=self.restore,
            min_replica_size=1,use_async_quorum=not runtime.healing,init_sync=False,max_retries=8,
            timeout=timedelta(seconds=90),quorum_timeout=timedelta(seconds=180),
            connect_timeout=timedelta(seconds=20),rank=self.rank,world_size=8,
            store_addr=host,store_port=port+100+runtime.store_services[str(self.replica)]*2+self.replica,
            hostname=runtime.replica_hosts[str(self.replica)],
            lighthouse_addr=f"http://{host}:{runtime.c.get('lighthouse_port',port+60)+service}",replica_id=f'shift{self.replica}',
            checkpoint_transport=PackedGPUTransport(self.pg,torch.device('cuda',runtime.device),timedelta(seconds=90),runtime.event))
        self.manager._shift_epoch=runtime.epoch
        self.manager.load_state_dict(dict(step=runtime.progress['updates'],batches_committed=runtime.progress['updates']*2))
        self._boundary=None;self.participants=2;self.participating_rank=self.replica
        self.runtime.event('session_outer_ready',service=service,identity=self.identity())

    def capture(self):
        boundary=self._boundary
        if boundary is None:raise RuntimeError('native export outside the immutable boundary')
        r=self.runtime
        boundary['ready'].synchronize()
        packet=r.training_state.capture(rng=boundary['rng'],all_buffers=boundary['buffers'],
            source_replica=self.replica,source_rank=self.rank)
        packet['hashes']=r.training_state.hashes(boundary['rng'],all_buffers=boundary['buffers']) if r.audit else {}
        return packet

    def refresh_boundary(self):
        import torch
        r=self.runtime
        rng=r.training_state.rng()
        buffers={n:b.detach().clone() for n,b in r.stage.named_buffers()}
        ready=torch.cuda.Event();ready.record()
        self._boundary=dict(rng=rng,buffers=buffers,ready=ready)

    def restore(self,packet):
        import torch
        from .planned_state import validate_state
        r=self.runtime
        validate_state(packet,rank=self.rank,audit=r.audit)
        r.training_state.restore(packet,clone_optimizer=True)
        r.optimizer.zero_grad(set_to_none=True);r.ddp.zero_grad_buffer();torch.cuda.synchronize()
        if r.audit and r.training_state.hashes()!=packet['hashes']:raise RuntimeError('native recovery state mismatch')
        self.refresh_boundary()
        r.base_store.set(f'session/native_restored/{r.sequence}/{r.rank}','1')
        r.event('session_native_restored',step=r.progress['updates'],source_replica=packet['source_replica'])

    def begin(self,step):
        r=self.runtime
        self.refresh_boundary()
        self.manager.start_quorum(allow_heal=r.healing)
        self._step_started=step
        if r.healing:
            self.manager.wait_quorum()
            self.participants=self.manager.num_participants()
            self.participating_rank=self.manager.participating_rank() or 0
        else:
            self.participants=2;self.participating_rank=self.replica

    def wait_quorum(self):
        self.manager.wait_quorum()
        n=self.manager.num_participants()
        if n not in (1,2) or (not self.runtime.healing and n!=2):raise RuntimeError('invalid session quorum size')
        self.participants=n;self.participating_rank=self.manager.participating_rank() or 0

    def reduce(self,buckets,step):
        import torch.distributed as dist
        self.wait_quorum()
        work=[self.manager.allreduce(b.grad_data,reduce_op=dist.ReduceOp.AVG) for b in buckets]
        for w in work:w.wait(self.timeout)

    def vote(self,finite):
        r=self.runtime
        if not finite:self.manager.report_error(RuntimeError('nonfinite gradients'))
        commit=self.manager.should_commit()
        if commit:
            step=r.progress['updates'];n=self.manager.num_participants()
            store=r.base_store;prefix=f'session-vote/{r.epoch}/{step}/{n}/'
            store.set(prefix+f'{self.replica}/{self.rank}','1')
            replicas=(self.replica,) if n==1 else (0,1)
            store.wait([prefix+f'{replica}/{rank}' for replica in replicas for rank in range(8)])
        self._step_started=None
        return bool(commit)

    def commit(self,step):
        if not self.vote(True):raise RuntimeError('session strategy rejected a required commit')

    def state_dict(self):return self.manager.state_dict()
    def load_state_dict(self,value):self.manager.load_state_dict(value)
    def identity(self):
        m=self.manager
        return dict(manager=id(m),server=id(m._manager),client=id(m._client),executor=id(m._executor),pg=id(self.pg),service_epoch=self.runtime.service)

    def migrate(self,epoch):
        if self._step_started is not None:raise RuntimeError('cannot migrate a live session step')
        if self.rank in (4,5,6,7):
            self.pg.migration_epoch=epoch;self.manager._quorum_id=None
        self.epoch=epoch;self.manager._shift_epoch=epoch

    def close(self):
        if self._closed:return
        self.manager.shutdown();self.pg.shutdown();self._closed=True
