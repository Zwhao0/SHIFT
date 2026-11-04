"""Default-off diagnostic helpers; no Manager, optimizer or outer collectives."""
import threading


def validate_descriptor(d, *, rank, step, epoch, seq_len, microbatches, data, vocab_size):
    expected = dict(schema=1, rank=rank, step=step, epoch=epoch,
                    seq_len=seq_len, microbatches=microbatches, data=data,
                    vocab_size=vocab_size, seed_base=100_000)
    if not isinstance(d, dict) or any(d.get(k) != v for k, v in expected.items()):
        raise ValueError('shadow descriptor version/role/workload mismatch')
    # The replica index comes from B's completed quorum, never a guessed rid.
    if (type(d.get('replica_index')) is not int or d['replica_index'] not in (0, 1)
            or d.get('participants') != 2 or d.get('local_data_rank') not in (0, 1)):
        raise ValueError('shadow requires a real two-participant fixed-batch quorum')
    data_rank = d['replica_index'] * 2 + d['local_data_rank']
    samples = [step * microbatches * 4 + micro * 4 + data_rank for micro in range(microbatches)]
    if d.get('sample_ids') != samples:
        raise ValueError('shadow sample sequence mismatch')
    return samples


class LocalFinalizer:
    """Exactly one call to the captured native local-DDP finish; no TP/PP tail."""
    def __init__(self, ddp):
        self.ddp = ddp
        self.finish = ddp.finish_grad_sync
        self.calls = 0
        self.owner = threading.get_ident()

    def __call__(self, models):
        if threading.get_ident() != self.owner or len(models) != 1 or models[0] is not self.ddp or self.calls:
            raise RuntimeError('invalid shadow local finalizer phase/owner')
        self.calls += 1
        self.finish()


def shadow_transaction(*, config, ddp, execute, capture, restore, clear, verify):
    """Restore the production finalizer and boundary state even on probe failure.

    execute/capture/restore/clear/verify all run synchronously on this owner.
    restore and clear must not skip each other when one raises. A failed verify
    prevents activation; callers retain the original error as its exception context.
    """
    native = config.finalize_model_grads_func
    local = LocalFinalizer(ddp)
    try:
        config.finalize_model_grads_func = local
        clear()
        execute()
        if local.calls != 1:
            raise RuntimeError('shadow schedule missed local finalizer')
        return capture()
    finally:
        config.finalize_model_grads_func = native
        try:
            restore()
        finally:
            try:
                clear()
            finally:
                verify()
