"""Metadata-only process groups for cross-host V4 candidate construction."""
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class MetadataGroup:
    label: str
    members: tuple
    rank: int

    @property
    def size(self):
        return len(self.members)


@contextmanager
def _temporary_attributes(target, replacements):
    saved = []
    try:
        for name, replacement in replacements.items():
            try:
                original = getattr(target, name)
            except AttributeError:
                saved.append((name, False, None))
            else:
                saved.append((name, True, original))
            setattr(target, name, replacement)
        yield
    finally:
        for name, existed, original in reversed(saved):
            if existed:
                setattr(target, name, original)
            else:
                delattr(target, name)


def _fixed(value):
    def getter(*args, **kwargs):
        return value
    return getter


def _group_argument(args, kwargs):
    if 'group' in kwargs:
        return kwargs['group']
    return args[0] if args else None


@contextmanager
def metadata_groups(dist, state, logical_rank):
    """Expose fixed TP2/PP2/EP2 metadata while making communication impossible."""
    if type(logical_rank) is not int or logical_rank not in range(8):
        raise ValueError('logical_rank must be an int in 0..7')

    stage = logical_rank // 4
    tp_rank = logical_rank % 2
    ep_members = (stage * 4 + tp_rank, stage * 4 + tp_rank + 2)
    groups = {
        'tp': MetadataGroup('tp', (logical_rank // 2 * 2, logical_rank // 2 * 2 + 1), tp_rank),
        'ep': MetadataGroup('ep', ep_members, (logical_rank % 4) // 2),
        'expert_dp': MetadataGroup('expert_dp', (logical_rank,), 0),
        'dense': MetadataGroup('dense', ep_members, (logical_rank % 4) // 2),
        'pp': MetadataGroup('pp', (logical_rank % 4, logical_rank % 4 + 4), stage),
    }

    def metadata_value(group, kind):
        if group is None:
            if kind == 'size':
                return 8
            if kind == 'rank':
                return logical_rank
            return tuple(range(8))
        if not isinstance(group, MetadataGroup):
            raise RuntimeError('metadata-only candidate cannot communicate')
        if kind == 'size':
            return group.size
        if kind == 'rank':
            return group.rank
        return group.members

    dist_replacements = {
        'get_world_size': lambda *args, **kwargs: metadata_value(_group_argument(args, kwargs), 'size'),
        'get_rank': lambda *args, **kwargs: metadata_value(_group_argument(args, kwargs), 'rank'),
        'get_process_group_ranks': lambda *args, **kwargs: metadata_value(
            _group_argument(args, kwargs), 'members'),
    }
    communication_methods = (
        'all_reduce', 'all_gather', 'all_gather_into_tensor', 'reduce_scatter',
        'reduce_scatter_tensor', '_all_gather_base', '_reduce_scatter_base',
        'all_to_all_single', 'broadcast', 'send', 'recv', 'isend', 'irecv',
        'batch_isend_irecv', 'barrier', 'init_process_group', 'new_group',
    )

    def prohibited(*args, **kwargs):
        raise RuntimeError('metadata-only candidate cannot communicate')

    for name in communication_methods:
        try:
            getattr(dist, name)
        except AttributeError:
            continue
        dist_replacements[name] = prohibited

    pp = groups['pp']
    state_replacements = {
        'get_tensor_model_parallel_rank': _fixed(groups['tp'].rank),
        'get_tensor_model_parallel_world_size': _fixed(groups['tp'].size),
        'get_tensor_model_parallel_group': _fixed(groups['tp']),
        'get_pipeline_model_parallel_rank': _fixed(pp.rank),
        'get_pipeline_model_parallel_world_size': _fixed(pp.size),
        'get_pipeline_model_parallel_group': _fixed(pp),
        'get_expert_model_parallel_rank': _fixed(groups['ep'].rank),
        'get_expert_model_parallel_world_size': _fixed(groups['ep'].size),
        'get_data_parallel_rank': _fixed(groups['dense'].rank),
        'get_data_parallel_world_size': _fixed(groups['dense'].size),
        'get_data_parallel_group': _fixed(groups['dense']),
        'get_data_modulo_expert_parallel_group': _fixed(groups['expert_dp']),
        'model_parallel_is_initialized': _fixed(True),
        'is_pipeline_first_stage': _fixed(pp.rank == 0),
        'is_pipeline_last_stage': _fixed(pp.rank == pp.size - 1),
    }
    with _temporary_attributes(dist, dist_replacements):
        with _temporary_attributes(state, state_replacements):
            yield groups
