"""One fresh-worker CPU load; join before any model/CUDA consumer."""
from concurrent.futures import Future
import threading
import time


class CPUModelLoad:
    def __init__(self, load):
        self.future = Future()
        self.started = time.monotonic()
        self.finished = None

        def run():
            try:
                model = load()
                if any(t.device.type != 'cpu' for t in (*model.parameters(), *model.buffers())):
                    raise RuntimeError('cold background loader must return only CPU tensors')
            except BaseException as error:
                self.finished = time.monotonic()
                self.future.set_exception(error)
            else:
                self.finished = time.monotonic()
                self.future.set_result(model)

        self.thread = threading.Thread(target=run, name='shift-cpu-model-load', daemon=True)
        self.thread.start()

    def result(self, timeout=120):
        waiting = time.monotonic()
        model = self.future.result(timeout=timeout)
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise RuntimeError('cold CPU loader did not terminate')
        return model, dict(load_seconds=self.finished-self.started,
                           wait_seconds=time.monotonic()-waiting,
                           start_monotonic=self.started, end_monotonic=self.finished)
