"""Runtime compatibility fixes for the pinned torchFT 0.2.0 wheel."""
from contextlib import contextmanager
from datetime import timedelta

import torch


def install_baby_event_fix():
    import torchft.process_group as groups
    import torchft.utils as utils

    def record_ipc_event():
        # 0.2.0 records the event but omits returning it. Both the producer
        # and consumer process require the event to preserve stream ordering.
        event = torch.cuda.Event(interprocess=True)
        torch.cuda.current_stream().record_event(event)
        event.synchronize()  # Hand out only completed IPC events; a killed child cannot strand a parent stream.
        return event

    groups.record_event = record_ipc_event
    utils.record_event = record_ipc_event


def install_managed_callback_fix():
    """Bind each TorchFT 0.2.0 managed callback to its own chain node.

    The upstream callback closes over the loop's ``managed_fut`` variable. A
    delayed CPU Future can therefore run after that variable has advanced to
    the tail, whose callback is ``None``. Bind only the node callback as a
    default argument; the shared ``value`` remains nonlocal so results flow
    through the chain in order. This monkeypatch is opt-in and changes only
    the in-process class, never the installed TorchFT files.
    """
    import torchft.manager as manager_module
    from torchft.utils import get_stream_context

    managed_work_class = manager_module._ManagedWork
    if getattr(managed_work_class, "_shift_managed_callback_fix_installed", False):
        return False

    def set_future_callback(self):
        if self._is_set_future_callback_called:
            return

        managed_fut = self._managed_fut_head
        managed_fut._fut = self._work.get_future()
        value = self._value

        is_future_wrapped = False
        while managed_fut._next:
            callback_for_node = managed_fut._callback

            def callback(fut, cb=callback_for_node):
                nonlocal value
                # Keep callback execution on the Work's stream, as upstream does.
                with get_stream_context(self._stream):
                    fut.wait()
                    assert cb
                    value = cb(manager_module._SimpleFuture(value))
                    return value

            assert managed_fut._fut
            fut = managed_fut._fut.then(callback)
            assert managed_fut._next
            managed_fut = managed_fut._next
            managed_fut._fut = fut

            if is_future_wrapped:
                continue

            managed_fut._fut = self._manager.wrap_future(managed_fut._fut, value)
            is_future_wrapped = True

        self._value = value
        self._is_set_future_callback_called = True

    managed_work_class._set_future_callback = set_future_callback
    managed_work_class._shift_managed_callback_fix_installed = True
    return True


def bootstrap_legacy_core(flash):
    """Load the real legacy Core without eager full-training/RETRO imports.

    This only provides the parent package namespace. Core modules and their
    computations remain the original files. Legacy top-level training helpers
    are outside this entrypoint; no TransformerEngine operation is emulated.
    """
    import importlib.machinery
    from pathlib import Path
    import sys
    import types

    if 'megatron' in sys.modules:
        return False
    package = Path(flash) / 'Greyhound-master/Megatron-LM/megatron'
    if not (package / 'core/__init__.py').is_file():
        raise FileNotFoundError(package)
    module = types.ModuleType('megatron')
    module.__path__ = [str(package)]
    module.__package__ = 'megatron'
    module.__file__ = str(package / '__init__.py')
    module.__spec__ = importlib.machinery.ModuleSpec('megatron', loader=None, is_package=True)
    module.__spec__.submodule_search_locations = module.__path__
    sys.modules['megatron'] = module
    return True


from torchft import ProcessGroupNCCL

@contextmanager
def cancelling_timeout(callback, timeout):
    """Cancel the pinned TorchFT timer even when the guarded call raises.

    TorchFT 0.2.0's context generator cancels only on normal exit. Resume it
    normally in finally so an old operation cannot later abort a new group.
    """
    from torchft.process_group import context_timeout
    guard = context_timeout(callback, timeout)
    guard.__enter__()
    try:
        yield
    finally:
        guard.__exit__(None, None, None)


from torchft.process_group import _WorkAcceleratorTimeout


class RecoveryWork(_WorkAcceleratorTimeout):
    @classmethod
    @contextmanager
    def _stream_timeout(cls, pg, timeout):
        from torchft.process_group import stream_timeout
        with cancelling_timeout(pg.abort, timeout):
            yield
        stream_timeout(pg.abort, timeout)


class RecoveryNCCL(ProcessGroupNCCL):
    """Keep fast gradient deadlines separate from checkpoint P2P deadlines."""
    checkpoint_timeout = timedelta(seconds=90)

    @contextmanager
    def _run_context(self):
        with cancelling_timeout(self.abort, self._timeout):
            yield

    def _wrap_work(self, work, opts):
        if not self._use_abort:
            return work
        timeout = self._timeout
        if hasattr(opts, 'timeout') and opts.timeout.total_seconds() > 0:
            timeout = opts.timeout
        return RecoveryWork(self, work, timeout)

    def _checkpoint_work(self, operation, tensors, peer, tag):
        # Never mutate the shared timeout: gradient submissions retain their
        # own deadline even if another thread is transferring a checkpoint.
        with cancelling_timeout(self.abort, self.checkpoint_timeout):
            work = operation(tensors, peer, tag)
        if not self._use_abort:
            return work
        return RecoveryWork(self, work, self.checkpoint_timeout)

    def send(self, tensors, dst_rank, tag):
        return self._checkpoint_work(self.parent.send, tensors, dst_rank, tag)

    def recv(self, tensors, src_rank, tag):
        return self._checkpoint_work(self.parent.recv, tensors, src_rank, tag)


class BlockingNCCL(RecoveryNCCL):
    def _create_pg(self, store, rank, world_size):
        from torch.distributed import ProcessGroupNCCL as NativeNCCL, ProcessGroup
        self._errored = None
        opts = NativeNCCL.Options()
        opts.config.blocking = True
        opts._timeout = max(self._timeout, self.checkpoint_timeout)
        group = ProcessGroup(store, rank, world_size)
        group._set_default_backend(ProcessGroup.BackendType.NCCL)
        backend = NativeNCCL(store, rank, world_size, opts)
        backend._set_sequence_number_for_group()
        backend.eager_connect_single_device(torch.device('cuda', torch.cuda.current_device()))
        group._register_backend(torch.device('cuda'), ProcessGroup.BackendType.NCCL, backend)
        return group


from torchft import ProcessGroupGloo

class HybridGloo(ProcessGroupGloo):
    """CPU gradient/control PG with a separate, idle-between-heals GPU state PG."""
    def __init__(self, timeout):
        self.gpu_pg = BlockingNCCL(timeout=timeout)
        super().__init__(timeout=timeout)

    def _create_pg(self, store, rank, world_size):
        saved = self._global_ranks, self._group_rank
        try:
            self._global_ranks = self._group_rank = None
            return super()._create_pg(store, rank, world_size)
        finally:
            self._global_ranks, self._group_rank = saved

    def configure(self, store_addr, *args, **kwargs):
        super().configure(store_addr, *args, **kwargs)
        self.gpu_pg.configure(store_addr + '/gpu-state', *args, **kwargs)

    def abort(self, errored=True):
        super().abort(errored=errored)
        self.gpu_pg.abort(errored=errored)

    def shutdown(self):
        super().shutdown()
        self.gpu_pg.shutdown()
