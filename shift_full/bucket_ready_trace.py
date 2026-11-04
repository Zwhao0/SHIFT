"""Opt-in diagnostic only: observe native bucket readiness without dispatching work."""
from types import MethodType


class BucketReadyTrace:
    def __init__(self, buckets, cuda):
        self.cuda = cuda
        self.active = False
        self.rows = {}
        self.streams = [cuda.Stream() for _ in buckets]
        for index, bucket in enumerate(buckets):
            native = bucket.start_grad_sync

            def start(this, native=native, index=index):
                result = native()
                if self.active:
                    if index in self.rows:
                        raise RuntimeError('duplicate bucket start in diagnostic step')
                    ready = self.cuda.Event(enable_timing=True)
                    done = self.cuda.Event(enable_timing=True)
                    ready.record()
                    with self.cuda.stream(self.streams[index]):
                        self.streams[index].wait_event(ready)
                        if this.communication_handle is not None:
                            this.communication_handle.wait()
                        done.record()
                    self.rows[index] = (ready, done, this.grad_data.numel())
                return result

            bucket.start_grad_sync = MethodType(start, bucket)

    def begin(self):
        if self.active:
            raise RuntimeError('nested bucket diagnostic')
        self.rows = {}
        self.active = True

    def collect(self, schedule_begin, finalizer_begin, outer_done):
        # Call only after the original step synchronization. Extra diagnostic
        # stream waits belong to this diagnostic, never a performance score.
        self.active = False
        if len(self.rows) != len(self.streams):
            raise RuntimeError('missing bucket start in diagnostic step')
        for stream in self.streams:
            stream.synchronize()
        return [dict(bucket=index, numel=numel,
                     producer_from_schedule_ms=schedule_begin.elapsed_time(ready),
                     local_done_from_schedule_ms=schedule_begin.elapsed_time(done),
                     local_wait_ms=ready.elapsed_time(done),
                     producer_to_finalizer_ms=ready.elapsed_time(finalizer_begin),
                     local_done_to_finalizer_ms=done.elapsed_time(finalizer_begin),
                     local_done_to_outer_done_ms=done.elapsed_time(outer_done))
                for index, (ready, done, numel) in sorted(self.rows.items())]

    def discard(self):
        self.active = False
        for stream in self.streams:
            stream.synchronize()
        self.rows = {}
