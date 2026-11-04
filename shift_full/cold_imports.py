"""Opt-in pinned CPU imports inside a fresh worker; join before all PG creation."""
from concurrent.futures import Future
import threading
import time


def megatron_components(adapter):
    state, _, _ = adapter.core()
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    return state, finalize_model_grads, get_forward_backward_func


class CPUImports:
    def __init__(self, adapter):
        self.future = Future()
        self.started = time.monotonic()
        self.finished = None

        def run():
            try:
                value = megatron_components(adapter)
            except BaseException as error:
                self.finished = time.monotonic()
                self.future.set_exception(error)
            else:
                self.finished = time.monotonic()
                self.future.set_result(value)

        self.thread = threading.Thread(target=run, name='shift-cpu-imports', daemon=True)
        self.thread.start()

    def result(self, timeout=45):
        waiting = time.monotonic()
        try:
            value = self.future.result(timeout=timeout)
        finally:
            self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise RuntimeError('cold CPU import thread did not terminate')
        return value, dict(import_seconds=self.finished-self.started,
                           wait_seconds=time.monotonic()-waiting,
                           start_monotonic=self.started, end_monotonic=self.finished)
