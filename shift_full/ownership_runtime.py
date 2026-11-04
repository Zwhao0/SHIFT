"""Default-off runtime building blocks. No GPU admission or coordinator here.

Unlike ownership_protocol these callbacks operate on actual Manager/finalizer
objects. CUDA drain and transport completion are supplied by the owner adapter.
"""
import threading
from shift_full.ownership_protocol import LocalFence
from shift_full.planned_shadow import LocalFinalizer


def forbid_quorum_recovery(manager, replica_index):
    """Inspect the native response BEFORE FT configures PGs or loads state."""
    client = manager._client
    class CheckedClient:
        def __getattr__(self, name):
            return getattr(client, name)

        def _quorum(self, *args, **kwargs):
            response = client._quorum(*args, **kwargs)
            if (response.heal or response.recover_dst_replica_ranks or
                    response.max_step != manager.current_step() or
                    response.replica_world_size != 2 or response.max_world_size != 2 or
                    response.replica_rank != replica_index or response.max_replica_rank != replica_index):
                raise RuntimeError('new ownership quorum requested recovery or changed sample role')
            return response
    manager._client = CheckedClient()


def completed_quorum(manager, *, step, replica_index=None):
    """Pinned FT0.2 only: do not wait on an unfinished future or apply healing."""
    future = manager._quorum_future
    if future is None or not future.done():
        raise RuntimeError('quorum still active')
    future.result()  # Propagate the original quorum failure.
    if (manager._errored is not None or manager._healing or
            manager._pending_state_dict is not None or manager.current_step() != step or
            manager._participating_replica_world_size != 2 or
            manager._participating_replica_rank not in (0, 1)):
        raise RuntimeError('non-quiescent or changed quorum')
    if replica_index is not None and manager._participating_replica_rank != replica_index:
        raise RuntimeError('sample role changed')
    # The future can be complete while recovery-stream device work remains.
    if manager._recovery_event is not None:
        manager._recovery_event.synchronize()
        manager._recovery_event = None


class ReservedB:
    """Owner reserves BEFORE descriptor/sleep; control thread only fences/acks.

    Never restores the intercepted methods. AB fallback needs a new Manager.
    This object does not abort any PG, including either end of old outer.
    """
    def __init__(self, manager, cut, *, local_pgs):
        completed_quorum(manager, step=cut.step)
        if any(pg is not None and (manager._pg is pg or getattr(manager._pg, 'parent', None) is pg)
               for pg in local_pgs):
            raise RuntimeError('outer aliases local training PG')
        self.manager, self.fence = manager, LocalFence(cut)
        self.native_shutdown = manager.shutdown
        self.closed = False
        self.fence.reserve(quorum_complete=True, state_readers_done=True)
        def forbidden(*args, **kwargs):
            self.fence.allow_old_outer()
            raise RuntimeError('reserved Manager is not reusable')
        for name in ('start_quorum', 'allreduce', 'should_commit', 'shutdown'):
            setattr(manager, name, forbidden)

    def descriptor(self, publish):
        self.fence._owner()
        publish()  # Already reserved even if publication raises.

    def control(self, cut, ack):
        ack(self.fence.control_fence(cut))

    def drain_and_close(self, *, drain_local, finish_presend):
        self.fence._owner()
        if self.closed:
            raise RuntimeError('duplicate Manager closure')
        drain_local()
        finish_presend()  # Must raise on timeout; never infer completion.
        self.fence.preouter(self.fence.cut, local_work_drained=True)
        if not self.fence.fenced:
            raise RuntimeError('control fence ack missing')
        self.native_shutdown()
        self.fence.close_old_manager()
        self.closed = True


class ComputedContribution:
    """C retains native-local-finished gradients until one new-epoch outer.

    Activation is irreversible BEFORE outer, even if native TP/PP then fails.
    Rollback is only legal before activation and always restores/clears/audits.
    """
    def __init__(self, *, ddp, result, unchanged, rollback):
        self.owner = threading.get_ident()
        self.ddp, self.result = ddp, result
        self.unchanged, self.rollback = unchanged, rollback
        self.state = 'computed'
        self.finish_started = False

    def _owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('training owner required')

    def validate(self):
        self._owner()
        if self.state != 'computed' or not self.unchanged():
            raise RuntimeError('computed contribution mutated or already consumed')

    def cancel(self):
        self._owner()
        if self.state not in ('computed', 'sealed'):
            raise RuntimeError('cannot roll back activated contribution')
        self.state = 'cancelled'
        self.rollback()

    def seal(self):
        """Last owner-thread audit before the all-role activation barrier."""
        self.validate()
        self.state = 'sealed'

    def activate(self):
        self._owner()
        if self.state != 'sealed':
            raise RuntimeError('activation requires a sealed contribution')
        self.state = 'activated'

    def finish(self, *, native_finalizer, outer):
        self._owner()
        if self.state == 'computed':
            self.seal()
            self.activate()
        if self.state != 'activated' or self.finish_started:
            raise RuntimeError('contribution already consumed or not activated')
        self.finish_started = True
        previous = self.ddp.finish_grad_sync
        calls = 0
        def outer_only():
            nonlocal calls
            self._owner()
            if calls:
                raise RuntimeError('duplicate outer')
            calls += 1
            outer()
        self.ddp.finish_grad_sync = outer_only
        try:
            native_finalizer([self.ddp])  # Outer, then original TP/PP tail once.
            if calls != 1:
                raise RuntimeError('native finalizer missed outer')
            self.state = 'finished'
        finally:
            self.ddp.finish_grad_sync = previous


def retain_contribution(*, config, ddp, audit_initial, clear, execute, capture,
                        unchanged, restore, verify):
    """Real receiver callback transaction; success intentionally does NOT clear."""
    native = config.finalize_model_grads_func
    local = LocalFinalizer(ddp)
    def rollback():
        try:
            restore()
        finally:
            try:
                clear()
            finally:
                verify()
    try:
        audit_initial()  # Fresh current nine hashes, before any F/B/zero_grad.
        config.finalize_model_grads_func = local
        clear()
        execute()
        if local.calls != 1:
            raise RuntimeError('recompute missed native local finish')
        result = capture()
        return ComputedContribution(ddp=ddp, result=result, unchanged=unchanged, rollback=rollback)
    except BaseException:
        config.finalize_model_grads_func = native
        rollback()
        raise
    finally:
        config.finalize_model_grads_func = native
