"""Explicit model buffers and Megatron RNG state for checkpoint boundaries."""
import copy


def buffer_manifest(model):
    rows = []
    for prefix, module in model.named_modules():
        for name, value in module._buffers.items():
            if value is not None:
                rows.append(dict(name=f'{prefix}.{name}' if prefix else name,
                                 shape=list(value.shape), dtype=str(value.dtype),
                                 persistent=name not in module._non_persistent_buffers_set))
    return rows


def restore_buffers(model, values):
    target = dict(model.named_buffers())
    if target.keys() != values.keys():
        raise RuntimeError('checkpoint buffer manifest mismatch')
    for name, value in values.items():
        target[name].copy_(value)


def tracker_snapshot(tracker):
    # get_states() may return the tracker's live dictionary. Do not retain it.
    return copy.deepcopy(tracker.get_states())


def restore_tracker(tracker, state):
    tracker.set_states(copy.deepcopy(state))
