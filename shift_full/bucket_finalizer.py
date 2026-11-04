"""Static Megatron bucket hooks; preserve native readiness and finalization order."""
from types import MethodType


class BucketFinalizer:
    def __init__(self, native_finish, buffers, hook):
        self.native_finish = native_finish
        self.buckets = tuple(bucket for buffer in buffers for bucket in buffer.buckets)
        self.hook = hook
        self.enqueue = None
        if hook:
            for bucket in self.buckets:
                native = bucket.finish_grad_sync

                def finish(this, native=native):
                    native()
                    if self.enqueue is not None:
                        self.enqueue(this)

                bucket.finish_grad_sync = MethodType(finish, bucket)

    def run(self, enqueue):
        if self.enqueue is not None:
            raise RuntimeError('nested bucket finalization')
        self.enqueue = enqueue
        try:
            self.native_finish()
            if not self.hook:
                for bucket in self.buckets:
                    enqueue(bucket)
        finally:
            self.enqueue = None
