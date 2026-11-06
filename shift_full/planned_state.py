"""Versioned planned-migration payload checks, before any state is applied."""
from collections.abc import Mapping
from shift_full.reference_audit import REFERENCE_HASH_KEYS

ROLE_KEYS = frozenset(('schema', 'torch_rng', 'cuda_rng', 'python_rng', 'tracker_rng',
                       'all_buffers', 'source_rank', 'step', 'progress'))
STATE_KEYS = ROLE_KEYS | {'model', 'optimizer', 'scheduler', 'source_replica', 'hashes'}


def validate_state(payload, *, rank, step=None, full=True, audit=False, epoch=None):
    required = STATE_KEYS if full else ROLE_KEYS
    if not isinstance(payload, Mapping) or payload.get('schema') != 2 or required - payload.keys():
        raise ValueError('planned state requires complete schema 2 buffers/RNG/tracker state')
    if not isinstance(payload['progress'],Mapping):raise ValueError('planned progress must be a mapping')
    if payload['source_rank'] != rank or payload['progress'].get('updates') != payload['step']:
        raise ValueError('planned state rank/progress mismatch')
    if step is not None and payload['step'] != step:
        raise ValueError('planned state version mismatch')
    if epoch is not None and (payload.get('planned_epoch') != epoch or payload.get('planned_role') != 1):
        raise ValueError('planned epoch/role mismatch')
    mappings=('all_buffers','tracker_rng') + (('model','optimizer','scheduler','hashes') if full else ())
    if any(not isinstance(payload[k],Mapping) for k in mappings):
        raise ValueError('planned state dictionaries must be mappings')
    if full and audit and set(payload['hashes']) != REFERENCE_HASH_KEYS:
        raise ValueError('planned state requires all nine hashes')
    return payload


def merge_role_state(payload, role, *, rank, step, epoch, digest, audit):
    """A supplies shared parameters/Adam; B supplies buffers and RNG/cursor."""
    validate_state(payload,rank=rank,step=step,epoch=epoch,audit=audit)
    validate_state(role,rank=rank,step=step,epoch=epoch,full=False)
    merged=dict(payload,**role)
    merged['model']=dict(payload['model'])
    for name,value in role['all_buffers'].items():
        if name in merged['model']:merged['model'][name]=value
    merged['hashes']=dict(payload['hashes'])
    if audit:
        merged['hashes'].update(model=digest(merged['model']),buffers=digest(role['all_buffers']),
            progress=digest(role['progress']),cpu_rng=digest(role['torch_rng']),
            cuda_rng=digest(role['cuda_rng']),python_rng=digest(role['python_rng']),
            tracker_rng=digest(role['tracker_rng']))
    return merged


def validate_relay(packet, *, rank, step, epoch, digest, audit):
    required={'schema','planned_epoch','planned_role','step','sample_step','sample_ids','gradients','tail','gradient_hash','tail_hash'}
    if not isinstance(packet,Mapping) or required-packet.keys():raise ValueError('incomplete relay packet')
    if (packet.get('schema') != 2 or packet.get('planned_epoch') != epoch
            or packet.get('planned_role') != 1 or packet.get('step') != step
            or packet.get('sample_step') != step + 1):
        raise ValueError('stale relay epoch/role/version')
    validate_state(packet['tail'], rank=rank, step=step, full=False, epoch=epoch)
    if audit and (digest(packet['gradients']) != packet['gradient_hash']
                  or digest(packet['tail']) != packet['tail_hash']):
        raise ValueError('relay gradient/tail hash mismatch')
    return packet


def role_hosts(hosts, round_index, *, allow_colocated=False):
    if (not isinstance(hosts,Mapping) or set(hosts) != {'A', 'B', 'C'}
            or not all(isinstance(h,str) and h for h in hosts.values())
            or (len(set(hosts.values())) != 3 and not (allow_colocated and len(set(hosts.values()))==2 and hosts['B']!=hosts['C']))):
        raise ValueError('planned hosts require three distinct A/B/C addresses')
    return dict(hosts) if round_index % 2 == 0 else dict(A=hosts['A'], B=hosts['C'], C=hosts['B'])
