"""Compact, independent no-migration reference; no live Manager rewind."""
from collections.abc import Mapping

from .reference_audit import REFERENCE_HASH_KEYS
from .v4_outer import validate_progress


def training_sources(root):
    import hashlib
    from pathlib import Path
    root=Path(root)
    files=set((root/'shift_full').glob('v4_*.py'))
    files.update(root/'shift_full'/name for name in ('transport.py','compat.py','train.py',
        'training_core.py','state_inventory.py','state_digest.py','planned_state.py','reference_audit.py'))
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def training_profile(config):
    """Return the exact boolean execution profile used to build a reference."""
    fields = ('outer_manager', 'outer_audit', 'overlap_peer', 'retain_outer_managers',
              'transactional_switch', 'cross_node_tp', 'deterministic', 'unified', 'performance_no_audit')
    return {field: bool(config.get(field, False)) for field in fields}


def sample_ids(step, replica, rank, microbatches):
    return [(step - 1) * microbatches * 4 + m * 4 + replica * 2 + rank % 4 // 2
            for m in range(microbatches)] if step else []


def validate_signatures(value):
    if not isinstance(value, Mapping) or value.get('schema') != 1:
        raise ValueError('independent reference schema 1 required')
    steps = value.get('steps')
    if type(steps) is not int or steps < 1:
        raise ValueError('positive reference length required')
    workers = value.get('workers', {})
    if set(workers) != {f'{r}/{k}' for r in range(2) for k in range(8)}:
        raise ValueError('reference requires all sixteen logical workers')
    micro = value.get('microbatches')
    if type(micro) is not int or micro < 1:
        raise ValueError('positive reference microbatches required')
    for key, trajectory in workers.items():
        replica, rank = map(int, key.split('/'))
        if set(trajectory) != {str(i) for i in range(steps + 1)}:
            raise ValueError('reference trajectory is incomplete: ' + key)
        for number, state in trajectory.items():
            step = int(number)
            hashes = state.get('hashes', {})
            if set(hashes) != REFERENCE_HASH_KEYS or any(
                not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h)
                for h in hashes.values()
            ):
                raise ValueError('reference requires nine complete SHA256 signatures')
            if validate_progress(state.get('outer_manager'))['step'] != step:
                raise ValueError('reference Manager progress mismatch')
            if state.get('sample_ids') != sample_ids(step, replica, rank, micro):
                raise ValueError('reference sample cursor mismatch')
    return value


def compare_signature(reference, *, replica, rank, step, hashes, samples, manager):
    expected = reference['workers'][f'{replica}/{rank}'].get(str(step))
    if expected is None:
        raise ValueError(f'independent reference does not cover step {step}')
    return (hashes == expected['hashes'] and samples == expected['sample_ids']
            and manager == expected['outer_manager'])


def build_signatures(rows, summary):
    if not summary.get('reference_only') or not summary.get('functional_pass'):
        raise ValueError('successful no-migration reference required')
    workers = {}
    for role in 'ABDE':
        for device in range(4):
            events = [x for x in rows[role] if x['device'] == device and x['kind'] == 'reference_step']
            rank=2*device+int(role in 'BE') if summary.get('cross_node_tp') else device+(4 if role in 'BE' else 0)
            key = f'{int(role in "DE")}/{rank}'
            if len(events) != summary['plan']['cut'] + summary['plan']['post_steps'] + 1:
                raise ValueError('duplicate or missing reference step: ' + key)
            workers[key] = {str(x['step']):{k:x[k] for k in ('hashes','sample_ids','outer_manager')}
                            for x in events}
    return validate_signatures(dict(schema=1, run=summary['name'],
        steps=summary['plan']['cut'] + summary['plan']['post_steps'],
        seq_len=summary['plan']['seq_len'], microbatches=summary['plan']['microbatches'],
        seed=summary.get('initialization_seed',20261106),cross_node_tp=bool(summary.get('cross_node_tp')),
        training_profile=training_profile(summary),
        **({'data':summary['data']} if 'data' in summary else {}),
        source_sha256=summary['source_sha256'], workers=workers))


def validate_compatibility(reference, config, sources):
    validate_signatures(reference)
    expected_profile = training_profile(config)
    profile = reference.get('training_profile')
    if (not isinstance(profile, Mapping) or set(profile) != set(expected_profile) or
        any(type(profile[key]) is not bool or profile[key] != value
            for key, value in expected_profile.items())):
        raise ValueError('independent reference training profile mismatch')
    if (reference['seq_len'] != config['plan']['seq_len'] or
        reference['microbatches'] != config['plan']['microbatches'] or
        reference['seed'] != config['seed'] or reference.get('data','pattern') != config.get('data','pattern')):
        raise ValueError('independent reference dimensions/seed mismatch')
    if bool(reference.get('cross_node_tp'))!=bool(config.get('cross_node_tp')):
        raise ValueError('independent reference physical TP layout mismatch')
    for name, sha in sources.items():
        if reference['source_sha256'].get(name) != sha:
            raise ValueError('independent reference source mismatch: ' + name)
