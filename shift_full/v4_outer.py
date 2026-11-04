"""TorchFT outer-replica quorum for the V4 experiment.

Torch and TorchFT are imported only when ``Outer`` is constructed so the small
state/sample validators remain usable in CPU-only tooling.
"""

from collections.abc import Mapping
from datetime import timedelta
import time


def validate_progress(value):
    """Validate TorchFT's two-replica progress invariant and return a copy."""
    if not isinstance(value, Mapping):
        raise ValueError("outer progress must be a mapping")
    step = value.get("step")
    batches = value.get("batches_committed")
    if type(step) is not int or step < 0:
        raise ValueError("outer step must be a nonnegative integer")
    if type(batches) is not int or batches != 2 * step:
        raise ValueError("outer batches_committed must equal 2 * step")
    return {"step": step, "batches_committed": batches}


def validate_sample_role(replica, participating_rank):
    """Keep each replica attached to its fixed sample role (0 or 1)."""
    if type(replica) is not int or replica not in (0, 1):
        raise ValueError("outer replica must be 0 or 1")
    if type(participating_rank) is not int or participating_rank != replica:
        raise RuntimeError("outer quorum changed the fixed sample role")
    return replica


class MigrationNamespace:
    """Fence NCCL store keys even when the replica quorum ID stays unchanged."""
    migration_epoch = 0
    configure_count = 0

    def configure(self, store_addr, *args, **kwargs):
        result = super().configure(store_addr + f'/v4-migration/{self.migration_epoch}', *args, **kwargs)
        self.configure_count += 1
        return result


class _ForbiddenCheckpointTransport:
    """Metadata only; recovery transfers are a hard error in this experiment."""

    def metadata(self):
        return "<outer-recovery-forbidden>"

    def send_checkpoint(self, *_args, **_kwargs):
        raise RuntimeError("outer checkpoint send is forbidden")

    def recv_checkpoint(self, *_args, **_kwargs):
        raise RuntimeError("outer checkpoint receive is forbidden")

    def disallow_checkpoint(self):
        pass

    def shutdown(self, wait=True):
        del wait


class Outer:
    """One actual TorchFT Manager for one eight-rank replica and epoch."""

    def __init__(self, c, replica: int, rank: int, epoch: int, event):
        if not isinstance(c, Mapping):
            raise ValueError("outer config must be a mapping")
        if type(replica) is not int or replica not in (0, 1):
            raise ValueError("outer replica must be 0 or 1")
        if type(rank) is not int or rank not in range(8):
            raise ValueError("outer rank must be in 0..7")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("outer epoch must be a nonnegative integer")
        hosts = c.get("hosts")
        host = hosts.get("A") if isinstance(hosts, Mapping) else None
        port = c.get("port")
        timeout_seconds = c.get("outer_timeout", 120)
        if not isinstance(host, str) or not host:
            raise ValueError("outer config requires hosts['A']")
        if type(port) is not int or port < 1 or port > 65535:
            raise ValueError("outer config port must be in 1..65535")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("outer_timeout must be positive")

        self.replica = replica
        self.rank = rank
        self.epoch = epoch
        self.event = event
        self.timeout = timedelta(seconds=float(timeout_seconds))
        self._step_started = None
        self._reduced_step = None
        self._closed = False
        self._effective_timeouts = None
        self._torch = None
        self._dist = None
        self._commit_store = None
        self.reference_pg = None
        self._host, self._port = host, port
        self._audit = c.get('outer_audit', False)
        self._async = c.get('overlap_peer', False)
        self._pending_reference_epoch = epoch if self._audit and self._async else None
        self._quorum_checked_step = None
        self._retain = c.get('retain_outer_managers', False)
        self._affected_ranks = tuple(c.get('affected_ranks',range(4,8)))
        self._service_epoch = 0 if self._retain else epoch
        self.pg = None
        self.manager = None
        started = time.monotonic()

        # Runtime imports are deliberately delayed; this module's validators
        # work on machines that do not have Torch installed.
        import torch
        import torch.distributed as dist
        from torchft import Manager
        from torchft.manager import WorldSizeMode
        from shift_full.compat import BlockingNCCL, install_managed_callback_fix
        from shift_full.ownership_runtime import forbid_quorum_recovery

        self._torch, self._dist = torch, dist
        install_managed_callback_fix()
        if self._retain:
            class ScopedNCCL(MigrationNamespace, BlockingNCCL):
                pass
            self.pg = ScopedNCCL(timeout=self.timeout)
            self.pg.migration_epoch = epoch
        else:
            self.pg = BlockingNCCL(timeout=self.timeout)
        try:
            self._commit_store = dist.TCPStore(host,port,None,False,timeout=self.timeout,wait_for_workers=False)
            self.manager = Manager(
                pg=self.pg,
                load_state_dict=self._forbid_load_state_dict,
                state_dict=self._forbid_state_dict,
                min_replica_size=2,
                world_size_mode=WorldSizeMode.FIXED_WITH_SPARES,
                use_async_quorum=self._async,
                init_sync=False,
                max_retries=0,
                timeout=self.timeout,
                quorum_timeout=self.timeout,
                connect_timeout=self.timeout,
                rank=rank,
                world_size=8,
                store_addr=host,
                store_port=port + 100 + self._service_epoch * 2 + replica,
                lighthouse_addr=f"http://{host}:{port + 60 + self._service_epoch}",
                replica_id=f"v4outer{replica}",
                hostname=host,
                checkpoint_transport=_ForbiddenCheckpointTransport(),
            )
            self.manager.load_state_dict({"step": 0, "batches_committed": 0})
            forbid_quorum_recovery(self.manager, replica)
            self._effective_timeouts = dict(
                timeout_seconds=self.manager._timeout.total_seconds(),
                quorum_timeout_seconds=self.manager._quorum_timeout.total_seconds(),
                connect_timeout_seconds=self.manager._connect_timeout.total_seconds(),
            )

            if c.get("outer_audit", False) and not self._async:
                self.reference_pg = BlockingNCCL(timeout=self.timeout)
                self.reference_pg.configure(
                    f"{host}:{port}/v4-reference/{epoch}/{rank}",
                    f"v4-reference-{epoch}-{rank}",
                    replica,
                    2,
                )
        except BaseException as error:
            cleanup_errors = self._cleanup_partial()
            if cleanup_errors:
                details = "; ".join(
                    f"{type(item).__name__}: {item}" for item in cleanup_errors
                )
                raise RuntimeError(
                    f"outer initialization failed and partial cleanup failed: {details}"
                ) from error
            raise

        self.event(
            "outer_ready",
            outer_epoch=epoch,
            seconds=time.monotonic() - started,
            requested_timeout_seconds=self.timeout.total_seconds(),
            **self._timeout_fields(),
            audit=self._audit,
            service_epoch=self._service_epoch,
        )

    def identity(self):
        return dict(manager=id(self.manager),server=id(self.manager._manager),
                    client=id(self.manager._client),executor=id(self.manager._executor),
                    pg=id(self.pg),service_epoch=self._service_epoch)

    def migrate(self, epoch):
        """Retain the live Manager and rebuild only C/E's affected DP channels."""
        if not self._retain or self._closed or self._step_started is not None:
            raise RuntimeError('Manager retention requires an idle live retained outer')
        if type(epoch) is not int or epoch != self.epoch + 1:
            raise ValueError('expected the next migration epoch')
        future = self.manager._quorum_future
        if future is None or not future.done():
            raise RuntimeError('outer quorum must be fully drained before migration')
        future.result()
        before = self.state_dict()
        if self.rank in getattr(self,'_affected_ranks',range(4,8)):
            # Replica membership does not change. Force configure on the next
            # real quorum, using a fresh store namespace on both C and E.
            self.pg.migration_epoch = epoch
            self.manager._quorum_id = None
            if self.reference_pg is not None:
                self.reference_pg.shutdown()
                self.reference_pg = None
            if self._audit:
                self._pending_reference_epoch = epoch
                if not self._async:self._prepare_reference()
        self.epoch = epoch
        if self.state_dict() != before:
            raise RuntimeError('Manager migration changed committed progress')

    def _timeout_fields(self):
        if self._effective_timeouts is None:
            raise RuntimeError("effective outer timeouts are unavailable")
        return dict(self._effective_timeouts)

    def _prepare_reference(self):
        epoch = self._pending_reference_epoch
        if epoch is None:
            return
        from shift_full.compat import BlockingNCCL
        self.reference_pg = BlockingNCCL(timeout=self.timeout)
        self.reference_pg.configure(
            f'{self._host}:{self._port}/v4-reference/{epoch}/{self.rank}',
            f'v4-reference-{epoch}-{self.rank}', self.replica, 2)
        self._pending_reference_epoch = None

    @staticmethod
    def _forbid_state_dict():
        raise RuntimeError("outer checkpoint export is forbidden")

    @staticmethod
    def _forbid_load_state_dict(_value):
        raise RuntimeError("outer checkpoint restore is forbidden")

    def state_dict(self):
        if self.manager is None:
            raise RuntimeError("outer Manager is unavailable")
        return validate_progress(self.manager.state_dict())

    def load_state_dict(self, value):
        if self.manager is None or self._closed:
            raise RuntimeError("outer Manager is unavailable or closed")
        if self._step_started is not None:
            raise RuntimeError("cannot restore outer progress during a step")
        state = validate_progress(value)
        if state['step'] < self.manager.current_step():
            raise RuntimeError('cannot rewind a live outer Manager')
        self.manager.load_state_dict(state)
        self.event(
            "outer_restored",
            outer_epoch=self.epoch,
            step=state["step"],
            batches_committed=state["batches_committed"],
        )

    def begin(self, step):
        if self.manager is None or self._closed:
            raise RuntimeError("outer Manager is unavailable or closed")
        if self._step_started is not None:
            raise RuntimeError("outer step is already active")
        state = self.state_dict()
        if state["step"] != step or self.manager.current_step() != step:
            raise RuntimeError("outer Manager step does not match training step")

        started = time.monotonic()
        self.manager.start_quorum(allow_heal=False)
        self._step_started = step
        self._reduced_step = None
        self._quorum_checked_step = None
        self._quorum_started = started
        if self._async:
            self.event('outer_quorum_requested',outer_epoch=self.epoch,step=step)
        else:
            self._complete_quorum(step)

    def _complete_quorum(self, step):
        if self._quorum_checked_step == step:
            return
        wait_started = time.monotonic()
        self.manager.wait_quorum()
        wait_seconds = time.monotonic() - wait_started
        participants = self.manager.num_participants()
        participating_rank = self.manager.participating_rank()
        if participants != 2:
            raise RuntimeError(f"outer quorum requires two participants, got {participants}")
        validate_sample_role(self.replica, participating_rank)
        if self.manager.current_step() != step:
            raise RuntimeError("outer quorum changed the training step")

        self._quorum_checked_step = step
        if self._async:
            self.event('outer_quorum_wait',outer_epoch=self.epoch,step=step,seconds=wait_seconds)
        self.event(
            "outer_begin",
            outer_epoch=self.epoch,
            step=step,
            seconds=time.monotonic() - self._quorum_started,
            participants=participants,
            participating_rank=participating_rank,
            migration_namespace=getattr(self.pg, 'migration_epoch', None),
            pg_configures=getattr(self.pg, 'configure_count', None),
            **self._timeout_fields(),
        )

    def reduce(self, buckets, step):
        try:
            return self._reduce(buckets,step)
        except BaseException as error:
            self._reject_step(step,type(error).__name__)
            raise

    def _reduce(self, buckets, step):
        if self.manager is None or self._closed:
            raise RuntimeError("outer Manager is unavailable or closed")
        if self._step_started != step:
            raise RuntimeError("outer reduce requires the matching begun step")
        self._complete_quorum(step)
        self._prepare_reference()
        if self.manager.current_step() != step or self.manager.num_participants() != 2:
            raise RuntimeError("outer quorum is stale or has changed size")

        torch, dist = self._torch, self._dist
        count = 0
        exact = True
        for bucket in buckets:
            tensor = getattr(bucket, "grad_data", None)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("outer buckets must expose tensor grad_data")
            reference = tensor.detach().clone() if self.reference_pg is not None else None
            work = self.manager.allreduce(tensor, reduce_op=dist.ReduceOp.AVG)
            if not work.wait(self.timeout):
                raise RuntimeError("TorchFT outer allreduce failed")
            if self.manager.errored() is not None:
                raise RuntimeError("TorchFT outer allreduce reported an error")

            if reference is not None:
                options = dist.AllreduceOptions()
                options.reduceOp = dist.ReduceOp.SUM
                reference_work = self.reference_pg.allreduce([reference], options)
                if not reference_work.wait(self.timeout):
                    raise RuntimeError("outer reference allreduce failed")
                reference.div_(2)
                exact = bool(torch.equal(tensor, reference)) and exact
            count += 1

        if count == 0:
            raise ValueError("outer reduction requires at least one gradient bucket")
        if self.reference_pg is not None:
            self.event(
                "outer_gradient_audit",
                outer_epoch=self.epoch,
                step=step,
                exact=exact,
                buckets=count,
            )
            if not exact:
                raise RuntimeError("TorchFT AVG differs from native pair reference")
        self._reduced_step = step

    def commit(self, step):
        if self.manager is None or self._closed:
            raise RuntimeError("outer Manager is unavailable or closed")
        if self._step_started != step or self._reduced_step != step:
            raise RuntimeError("outer commit requires its reduced step")

        started = time.monotonic()
        try:
            if not self.manager.should_commit():
                raise RuntimeError("TorchFT outer rejected the step")
            state = self.state_dict()
            if state['step'] != step+1 or state['batches_committed'] != 2*(step+1):
                raise RuntimeError('TorchFT outer commit progress invariant failed')
            self._confirm_global_commit(step)
        except BaseException as error:
            self._reject_step(step,type(error).__name__)
            raise
        self._step_started = None
        self._reduced_step = None
        self.event(
            "outer_commit",
            outer_epoch=self.epoch,
            step=step,
            seconds=time.monotonic() - started,
            committed_step=state["step"],
            batches_committed=state["batches_committed"],
            global_fenced=True,
            **self._timeout_fields(),
        )

    def _commit_prefix(self,step):
        return f'v4-opt-commit/{self.epoch}/{step}/'

    def _reject_step(self,step,reason):
        if self._commit_store is not None:
            prefix=self._commit_prefix(step)
            self._commit_store.set(prefix+'rejected',reason)
            self._commit_store.compare_set(prefix+'decision','','abort')

    def _confirm_global_commit(self,step):
        """Manager's native commit vote is replica-local; Adam needs all 16."""
        store=self._commit_store
        if store is None:raise RuntimeError('global optimizer commit store unavailable')
        prefix=self._commit_prefix(step);rejected=prefix+'rejected';decision=prefix+'decision'
        store.set(prefix+f'vote/{self.replica}/{self.rank}','1')
        keys=[prefix+f'vote/{r}/{k}' for r in range(2) for k in range(8)]
        until=time.monotonic()+self.timeout.total_seconds()
        while True:
            timed_out=time.monotonic()>=until
            if store.check([decision]):
                value=store.get(decision)
            elif store.check([rejected]) or timed_out:
                value=store.compare_set(decision,'','abort')
            elif store.check(keys):
                value=store.compare_set(decision,'','commit')
            else:
                time.sleep(.005)
                continue
            if value in (b'commit','commit'):return
            if timed_out:raise TimeoutError('global optimizer commit deadline')
            raise RuntimeError('another worker rejected the optimizer commit')

    def _cleanup_partial(self):
        errors = []
        for obj in (self.manager, self.reference_pg, self.pg):
            if obj is None:
                continue
            try:
                obj.shutdown()
            except BaseException as error:
                errors.append(error)
        return errors

    def close(self):
        if self._closed:
            raise RuntimeError("outer already closed")
        if self._step_started is not None:
            raise RuntimeError("cannot close outer during an active step")

        started = time.monotonic()
        errors = []
        torch = self._torch
        sync_started = time.monotonic()
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except BaseException as error:
            errors.append(error)
        synchronize_seconds = time.monotonic() - sync_started

        manager_seconds = pg_seconds = reference_seconds = 0.0
        if self.manager is not None:
            t = time.monotonic()
            try:
                self.manager.shutdown()
            except BaseException as error:
                errors.append(error)
            manager_seconds = time.monotonic() - t
        if self.pg is not None:
            t = time.monotonic()
            try:
                self.pg.shutdown()
            except BaseException as error:
                errors.append(error)
            pg_seconds = time.monotonic() - t
        if self.reference_pg is not None:
            t = time.monotonic()
            try:
                self.reference_pg.shutdown()
            except BaseException as error:
                errors.append(error)
            reference_seconds = time.monotonic() - t

        self._closed = True
        self.event(
            "outer_closed",
            outer_epoch=self.epoch,
            seconds=time.monotonic() - started,
            synchronize_seconds=synchronize_seconds,
            manager_shutdown_seconds=manager_seconds,
            manager_pg_shutdown_seconds=pg_seconds,
            reference_pg_shutdown_seconds=reference_seconds,
            errors=[f"{type(error).__name__}: {error}" for error in errors],
            **self._timeout_fields(),
        )
        if errors:
            raise RuntimeError("outer cleanup failed") from errors[0]
