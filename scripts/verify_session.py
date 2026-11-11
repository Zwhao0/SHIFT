"""CPU-only independent verifier for persistent mixed-policy session runs."""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from shift_full.reference_audit import REFERENCE_HASH_KEYS
from shift_full.session_layout import SessionLayout, partial_roles
from shift_full.training_core import sample_plan


def _int(x):
    return type(x) is int


def _finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def _configuration(summary, errors, *, world=8, tp=2):
    value = summary.get('configuration')
    if value is None:
        return dict(replica_world_size=world, tp=tp, pp=2, ep=2, batch=8, seq_len=None)
    if not isinstance(value, dict):
        errors.append('summary configuration must be an object')
        return dict(replica_world_size=world, tp=tp, pp=2, ep=2, batch=8, seq_len=None)
    configured_world = value.get('replica_world_size',8)
    if not _int(configured_world) or configured_world not in (4, 8):
        errors.append('configuration replica_world_size must be 4 or 8')
        configured_world = world
    wanted_tp = configured_world // 4
    for key, expected in (('tp', wanted_tp), ('pp', 2), ('ep', 2), ('batch', 8)):
        if not _int(value.get(key)) or value.get(key) != expected:
            errors.append(f'configuration {key} must be {expected} for replica_world_size={configured_world}')
    if not _int(value.get('seq_len')) or value.get('seq_len')<2:
        errors.append('configuration seq_len must be an integer at least two')
    return dict(replica_world_size=configured_world, tp=wanted_tp, pp=2, ep=2,
                batch=8, seq_len=value.get('seq_len'))


def _hashes(x):
    return (isinstance(x, dict) and set(x) == REFERENCE_HASH_KEYS and
            all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in x.values()))


def _logical_rank(role, device, replica, full_dp=False, *, world=8):
    width = world // 2
    if full_dp:
        return device
    if role in ('A', 'D'):
        return device
    if role in ('B', 'C', 'E'):
        return device + width
    return None


def _actor_rank(actor, device, world):
    if 'ranks' in actor:
        try:
            scope = SessionLayout(world).partial_scope(actor['replica'], actor['ranks'])
            if len(actor['devices']) != len(scope.ranks):
                return None
            return scope.ranks[device]
        except (KeyError, TypeError, IndexError, ValueError):
            return None
    full_dp = actor.get('role') == 'D' and actor.get('generation', 0) > 0 and len(actor.get('devices', [])) == world
    return _logical_rank(actor.get('role'), device, actor.get('replica'), full_dp, world=world)


def _read_json(path, errors, label):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('expected object')
        return value
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f'{label}: {exc}')
        return {}


def _event_rows(root, actors, errors, *, world=8, tp=2):
    rows, pids, expected = {}, {}, set()
    width = world // 2
    for label, d in actors.items():
        rows[label] = {}
        for device in range(len(d['devices'])):
            path = root / 'raw' / d['node'] / f'{label}{device}.jsonl'
            expected.add(path.resolve())
            events = []
            try:
                for line_no, line in enumerate(path.read_text().splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        if not isinstance(obj, dict):
                            raise ValueError('event is not an object')
                        events.append(obj)
                    except (ValueError, json.JSONDecodeError) as exc:
                        errors.append(f'malformed raw {path.name}:{line_no}: {exc}')
            except OSError as exc:
                errors.append(f'missing raw {path.name}: {exc}')
            rows[label][device] = events
            if not events:
                errors.append(f'empty actor log {path.name}')
            if d.get('pod_uid'):
                reserved = d.get('gpu_uuids')
                observed = [e.get('gpu_uuid') for e in events if e.get('kind') == 'worker_device_bound']
                if (not isinstance(reserved, list) or len(reserved) != len(d['devices'])
                        or not isinstance(reserved[device], str) or len(observed) != 1
                        or not isinstance(observed[0], str)
                        or observed[0].removeprefix('GPU-') != reserved[device].removeprefix('GPU-')):
                    errors.append(f'Lab GPU binding proof mismatch {path.name}')
            event_pids = set()
            for e in events:
                rank = _actor_rank(d, device, world)
                binding = (e.get('role'), e.get('replica'), e.get('device'),
                           e.get('rank'), e.get('actor_round'))
                wanted = (d['role'], d['replica'], device, rank, d['generation'])
                if binding != wanted:
                    errors.append(f'actor binding {path.name}/{e.get("kind")}')
                if any(not _int(e.get(k)) for k in ('replica','device','rank','actor_round')):
                    errors.append(f'invalid actor integer binding {path.name}/{e.get("kind")}')
                if not _int(e.get('round')) or e['round'] < 0:
                    errors.append(f'invalid event round {path.name}')
                pid = e.get('pid')
                if not _int(pid) or pid <= 0:
                    errors.append(f'invalid PID {path.name}')
                else:
                    event_pids.add(pid)
                mono = e.get('monotonic')
                if not _finite(mono):
                    errors.append(f'invalid monotonic timestamp {path.name}')
                for key in ('hashes', 'expected_hashes'):
                    if e.get(key) is not None and not _hashes(e[key]):
                        errors.append(f'incomplete nine-state hash {path.name}/{e.get("kind")}')
            if len(event_pids) != 1:
                errors.append(f'PID continuity {path.name}')
            elif event_pids:
                pid = (d.get('pod_uid'), next(iter(event_pids)))
                if pid in pids:
                    errors.append(f'PID reused across actors {path.name}')
                pids[pid] = path.name
            commits = [e for e in events if e.get('kind') == 'session_commit']
            steps = []
            for e in commits:
                step = e.get('step')
                if not _int(step) or step < 1:
                    errors.append(f'invalid commit step {path.name}')
                    continue
                steps.append(step)
                participants = e.get('participants')
                part_rank = e.get('participating_rank') or 0
                if not _int(participants) or participants not in (1, 2) or not _int(part_rank) or part_rank not in (0, 1):
                    errors.append(f'invalid commit participation {path.name}/step{step}')
                elif _int(rank):
                    try:
                        expected_samples = sample_plan(step - 1, microbatches=2,
                            participants=participants, replica=part_rank,
                            local_data_rank=(rank % width) // tp)
                        if e.get('sample_ids') != expected_samples:
                            errors.append(f'sample plan mismatch {path.name}/step{step}')
                    except (TypeError, ValueError):
                        errors.append(f'invalid sample plan {path.name}/step{step}')
                if not _finite(e.get('seconds')) or e['seconds'] < 0:
                    errors.append(f'invalid commit duration {path.name}/step{step}')
            if len(steps) != len(set(steps)):
                errors.append(f'duplicate commit step {path.name}')
            ordered = sorted(set(steps))
            if any(b != a + 1 for a, b in zip(ordered, ordered[1:])):
                errors.append(f'commit step gap {path.name}')
    actual = {p.resolve() for p in (root / 'raw').glob('*/*.jsonl')}
    if actual != expected:
        errors.append('raw actor file inventory mismatch')
    return rows


def _flatten(rows):
    return [(label, device, e) for label, devices in rows.items()
            for device, events in devices.items() for e in events]


def _active_map(summary, actors, errors, *, world=8):
    active = {}
    value = summary.get('active')
    if not isinstance(value, dict):
        errors.append('final active mapping missing')
        return active
    for replica in range(2):
        for rank in range(world):
            key = f'{replica}/{rank}'
            pair = value.get(key)
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                errors.append(f'active mapping missing {key}')
                continue
            label, device = pair
            d = actors.get(label)
            if not _int(device) or not d or device < 0 or device >= len(d['devices']):
                errors.append(f'invalid active mapping {key}')
                continue
            if d['replica'] != replica or _actor_rank(d, device, world) != rank:
                errors.append(f'active logical rank mismatch {key}')
            active[(replica, rank)] = (label, device)
    if len(value) != 2 * world:
        errors.append(f'active mapping must contain exactly {2 * world} workers')
    physical = []
    for (replica, rank), (label, device) in active.items():
        d = actors[label]
        physical.append((d['node'], d['devices'][device]))
    if len(physical) != len(set(physical)):
        errors.append('final active physical GPU collision')
    return active


def _initial_map(actors, errors, *, world=8):
    result = {}
    width = world // 2
    wanted = {
        **{(0,r):('g0_A',r) for r in range(width)},
        **{(0,r):('g0_B',r-width) for r in range(width,world)},
        **{(1,r):('g0_D',r) for r in range(width)},
        **{(1,r):('g0_E',r-width) for r in range(width,world)},
    }
    for key, pair in wanted.items():
        label, device = pair
        if label not in actors or device >= len(actors.get(label, {}).get('devices', [])):
            errors.append(f'initial actor missing for {key[0]}/{key[1]}')
        else:
            result[key] = pair
    return result


def _resource_check(root, errors, *, lab=False, actors=None):
    before = _read_json(root / 'resources_before.json', errors, 'resources_before unavailable')
    after = _read_json(root / 'resources_after.json', errors, 'resources_after unavailable')
    if len(before) not in (2, 3) or set(before) != set(after):
        errors.append('resource evidence must cover the same two or three nodes')
        return
    for node in sorted(before):
        b, a = before[node], after[node]
        if lab:
            uid = b.get('pod_uid')
            if (b.get('scope') != 'lab_pod' or a.get('scope') != 'lab_pod'
                    or not isinstance(uid, str) or not uid or a.get('pod_uid') != uid
                    or any(d.get('pod_uid') != uid for d in actors.values() if d.get('node') == node)):
                errors.append(f'{node}: Lab Pod identity is inconsistent')
            if b.get('extra_gpu') != [] or a.get('extra_gpu') != []:
                errors.append(f'{node}: session-owned actor remained live')
            continue
        bc, ac = b.get('company'), a.get('company')
        if not isinstance(bc, dict) or len(bc) != 8:
            errors.append(f'{node}: expected eight company PIDs before run')
        if not isinstance(ac, dict) or len(ac) != 8:
            errors.append(f'{node}: expected eight company PIDs after run')
        if bc != ac:
            errors.append(f'{node}: company PIDs changed')
        if b.get('extra_gpu') != [] or a.get('extra_gpu') != []:
            errors.append(f'{node}: extra GPU process present')


def _lab_gpu_inventory(summary, actors, errors):
    inventory = summary.get('runner_gpu_uuids')
    if inventory is None:  # Older Lab evidence records only actor bindings.
        return
    if (not isinstance(inventory, dict) or set(inventory) != set(summary.get('nodes', ()))
            or any(not isinstance(slots, list) or not slots
                   or any(not isinstance(uid, str) or not uid for uid in slots)
                   for slots in inventory.values())):
        errors.append('invalid Lab runner GPU inventory')
        return
    all_devices = [uid for slots in inventory.values() for uid in slots]
    if len(all_devices) != len(set(all_devices)):
        errors.append('overlapping Lab runner GPU UUIDs')
    for label, actor in actors.items():
        devices = inventory.get(actor['node'], ())
        indices = actor['devices']
        if (any(not _int(index) or not 0 <= index < len(devices) for index in indices)
                or [devices[index] for index in indices] != actor.get('gpu_uuids')):
            errors.append(f'{label}: actor GPU binding differs from the runner inventory')


def _event_index(flat):
    index = defaultdict(list)
    for label, device, e in flat:
        index[e.get('kind')].append((label, device, e))
    return index


def _rank_map(events, *, field='rank'):
    result = defaultdict(list)
    for item in events:
        e = item[2]
        result[(e.get('replica'), e.get(field))].append(item)
    return result


def _require_rank_set(errors, events, ranks, message, replica=None):
    selected = [x for x in events if replica is None or x[2].get('replica') == replica]
    seen = [x[2].get('rank') for x in selected]
    if sorted(seen) != sorted(ranks):
        errors.append(message)
    return selected


def _actor_events(rows, label, kind=None):
    result = []
    for device, events in rows.get(label, {}).items():
        for e in events:
            if kind is None or e.get('kind') == kind:
                result.append((label, device, e))
    return result


def _exact_readiness(e, *, epoch, rank, cut, target):
    return e.get('readiness') == dict(schema=1, epoch=epoch, rank=rank,
                                      cut=cut, target=target, restored=True)


def _verify_v2(cmdrec, before, after, rows, actors, flat, errors, *, world=8):
    command = cmdrec.get('command', {})
    seq = cmdrec.get('sequence')
    if not isinstance(command, dict) or not _int(seq):
        errors.append('v2 source handoff: invalid command metadata')
        return
    # The runtime round field is the session command sequence; the reused
    # PlannedHandoff's internal round remains sequence-1 in its manager state.
    cut_id = seq
    source_labels = {before[(1, r)][0] for r in range(world) if (1, r) in before}
    target = cmdrec.get('target_actor')
    if not target or target not in actors:
        errors.append('v2 source handoff: candidate actor missing')
        return
    candidate = _actor_events(rows, target)
    def source_kind(kind):
        return [item for label in source_labels for item in _actor_events(rows, label, kind)]
    # Cut is chosen by the original A rank zero; all source/candidate evidence must agree.
    cuts = [e for _, _, e in flat if e.get('kind') == 'planned_cut_selected' and
            e.get('command_id') == command.get('id')]
    if len(cuts) != 1:
        cuts = [e for _, _, e in flat if e.get('kind') == 'planned_cut_selected' and
                e.get('round') == cut_id and e.get('replica') == 0 and e.get('rank') == 0]
    if len(cuts) != 1 or not _int(cuts[0].get('step')):
        errors.append('v2 source handoff: planned cut selection missing')
        cut = None
    else:
        cut = cuts[0]['step']
    for kind in ('planned_precopy_window_open', 'planned_precopy_frozen', 'planned_precopy_sent'):
        evs = source_kind(kind)
        if cut is not None:
            evs = [x for x in evs if x[2].get('step') == cut and x[2].get('round') == cut_id]
        if sorted(x[2].get('rank') for x in evs) != list(range(world)):
            errors.append(f'v2 source handoff: {kind} missing across {world} ranks')
    for kind in ('planned_precopy_loaded', 'planned_state_ready',
                 'planned_gradient_received'):
        evs = [x for x in candidate if x[2].get('kind') == kind and
               x[2].get('round') == cut_id and (cut is None or x[2].get('step') == cut)]
        if sorted(x[2].get('rank') for x in evs) != list(range(world)):
            errors.append(f'v2 candidate handoff: {kind} missing across {world} ranks')
        if kind == 'planned_state_ready' and any(
                x[2].get('epoch') != command.get('expected_epoch', 0) + 1 for x in evs):
            errors.append(f'v2 candidate handoff epoch mismatch: {kind}')
    healthy_labels = {before[(0, r)][0] for r in range(world) if (0, r) in before}
    activated = [item for label in healthy_labels for item in _actor_events(rows, label, 'planned_epoch_activated')
                 if item[2].get('round') == cut_id and (cut is None or item[2].get('step') == cut)]
    if sorted(x[2].get('rank') for x in activated) != list(range(world)) or any(
            x[2].get('epoch') != command.get('expected_epoch', 0) + 1 for x in activated):
        errors.append(f'v2 healthy cohort activation missing across {world} ranks')
    sent = source_kind('planned_gradient_sent')
    received = [x for x in candidate if x[2].get('kind') == 'planned_gradient_received']
    if cut is not None:
        sent = [x for x in sent if x[2].get('step') == cut and x[2].get('round') == cut_id]
        received = [x for x in received if x[2].get('step') == cut and x[2].get('round') == cut_id]
    if sorted(x[2].get('rank') for x in sent) != list(range(world)) or sorted(x[2].get('rank') for x in received) != list(range(world)):
        errors.append(f'v2 gradient handoff missing across {world} ranks')
    else:
        sm = {x[2]['rank']: x[2] for x in sent}
        rm = {x[2]['rank']: x[2] for x in received}
        for rank in range(world):
            if sm[rank].get('sample_ids') != rm[rank].get('sample_ids'):
                errors.append(f'v2 gradient sample handoff mismatch rank {rank}')
            if rm[rank].get('forward_calls') != 0:
                errors.append(f'v2 relay performed target forward rank {rank}')
    retired = source_kind('session_retired')
    if sorted(x[2].get('rank') for x in retired) != list(range(world)):
        errors.append(f'v2 old owner retirement missing across {world} ranks')
    # Candidate uses D/full-DP; it must become the replica-1 active cohort.
    d = actors[target]
    if d.get('role') != 'D' or d.get('replica') != 1 or len(d.get('devices', [])) != world:
        errors.append(f'v2 candidate must be a {world}-rank replica-1 D actor')
    if d.get('generation') != seq:
        errors.append('v2 candidate generation mismatch')
    if after and any(after.get((1, r)) != (target, r) for r in range(world)):
        errors.append('v2 candidate did not become the full active replica')


def _verify_v4(cmdrec, before, after, rows, actors, flat, errors, boundary_mode, *, world=8):
    command = cmdrec.get('command', {})
    seq, epoch = cmdrec.get('sequence'), command.get('expected_epoch')
    if not isinstance(command, dict) or not _int(seq) or not _int(epoch):
        errors.append('v4 transaction: invalid command metadata')
        return
    cut_round = epoch
    width = world // 2
    try:
        scope = SessionLayout(world).partial_scope(command.get('source_replica'), command.get('ranks') or ())
    except (TypeError, ValueError):
        errors.append('v4 transaction requires one complete stage')
        return
    stage_ranks = scope.ranks
    old_labels = {before[(scope.replica, r)][0] for r in stage_ranks if (scope.replica, r) in before}
    old_roles = {actors[label]['role'] for label in old_labels if label in actors}
    source_role, target_role = partial_roles(next(iter(old_roles))) if len(old_roles) == 1 else (None, None)
    target = cmdrec.get('target_actor')
    if not target or target not in actors:
        errors.append('v4 transaction: candidate actor missing')
        return
    td = actors[target]
    if (target_role is None or td.get('role') != target_role or td.get('replica') != scope.replica or
            len(td.get('devices', [])) != width or
            tuple(_actor_rank(td, d, world) for d in range(width)) != stage_ranks or
            cmdrec.get('source_role', source_role) != source_role or cmdrec.get('target_role', target_role) != target_role):
        errors.append('v4 candidate does not match the selected stage and owner')
    def match(kind, pred):
        return [x for x in flat if x[2].get('kind') == kind and pred(x[2])]
    boundaries = [x for x in flat if x[2].get('kind') == 'boundary_selected' and
                   x[2].get('round') == cut_round]
    freezes = [x for x in flat if x[2].get('kind') == 'freeze' and x[2].get('round') == cut_round]
    all_workers = {(r,k) for r in range(2) for k in range(world)}
    if len(boundaries) != 2*world or { (e.get('replica'), e.get('rank')) for _,_,e in boundaries } != all_workers:
        errors.append(f'v4 boundary evidence must cover all {2*world} workers')
    if len(freezes) != 2*world or {(e.get('replica'), e.get('rank')) for _,_,e in freezes} != all_workers:
        errors.append(f'v4 freeze evidence must cover all {2*world} workers')
    cuts = {e.get('step') for _,_,e in boundaries}
    if len(cuts) != 1 or not all(_int(x) for x in cuts):
        errors.append('v4 boundary cut is not common')
        cut = None
    else:
        cut = next(iter(cuts))
    if cut is not None and any(e.get('step') != cut for _,_,e in freezes):
        errors.append('v4 freeze cut mismatch')
    if 0 in stage_ranks:
        retained = set(range(world)) - set(stage_ranks)
        rebuilt = match('outer_rebuilt', lambda e: e.get('round') == cut_round and e.get('replica') == scope.replica
                        and e.get('epoch') == epoch + 1 and (cut is None or e.get('step') == cut))
        if (cmdrec.get('outer_store_service') != seq or len(rebuilt) != len(retained)
                or {e.get('rank') for _,_,e in rebuilt} != retained):
            errors.append('v4 PP0 lacks fresh Store and retained-stage Manager rebuild evidence')
    decisions = match('switch_decision', lambda e: e.get('round') == cut_round and scope.coordinates(e.get('replica'), e.get('rank')))
    if len(decisions) != 1:
        errors.append('v4 switch decision missing')
    else:
        de = decisions[0][2]
        if de.get('decision') != 'replace' or de.get('owner') != target_role or de.get('validation') != boundary_mode:
            errors.append(f'v4 switch decision is not {boundary_mode} replace to {target_role}')
        if de.get('epoch') != epoch + 1 or (cut is not None and de.get('step') != cut):
            errors.append('v4 switch decision epoch/cut mismatch')
        results = de.get('candidate_results')
        if not isinstance(results, dict) or set(results) != {str(k) for k in stage_ranks} or any(results.get(str(k)) is not True for k in stage_ranks):
            errors.append(f'v4 switch decision lacks {width} successful candidate results')
    validated = _actor_events(rows, target, 'candidate_validated')
    validated = [x for x in validated if x[2].get('round') == cut_round and
                 (cut is None or x[2].get('step') == cut)]
    if sorted(x[2].get('rank') for x in validated) != list(stage_ranks):
        errors.append(f'v4 candidate validation missing for {width} ranks')
    for _, _, e in validated:
        if not _exact_readiness(e, epoch=epoch+1,
                rank=e.get('rank'), cut=cut, target=target_role):
            errors.append(f'v4 candidate readiness mismatch rank {e.get("rank")}')
    activated = _actor_events(rows, target, 'candidate_activated')
    activated = [x for x in activated if x[2].get('round') == cut_round and
                 (cut is None or x[2].get('step') == cut)]
    if sorted(x[2].get('rank') for x in activated) != list(stage_ranks):
        errors.append(f'v4 candidate activation missing for {width} ranks')
    for _, _, e in activated:
        if not _exact_readiness(e, epoch=epoch+1, rank=e.get('rank'), cut=cut, target=target_role):
            errors.append(f'v4 activation readiness mismatch rank {e.get("rank")}')
    retired = [x for label in old_labels for x in _actor_events(rows, label, 'retired')]
    retired = [x for x in retired if x[2].get('round') == cut_round and
               (cut is None or x[2].get('step') == cut)]
    if sorted(x[2].get('rank') for x in retired) != list(stage_ranks):
        errors.append(f'v4 old owner retirement missing for {width} ranks')
    # Packed state channels must connect each frozen source rank to its candidate.
    for kind, label_set, want in (('state_sent', old_labels, list(stage_ranks)),
                                  ('state_received', {target}, list(stage_ranks))):
        evs = [x for label in label_set for x in _actor_events(rows, label, kind)
               if x[2].get('round') == cut_round and (cut is None or x[2].get('step') == cut)]
        if sorted(x[2].get('rank') for x in evs) != want:
            errors.append(f'v4 transport missing {kind} for {width} ranks')
        for _, _, e in evs:
            if e.get('transport') != 'packed_gpu' or not _int(e.get('bytes')) or e['bytes'] <= 0:
                errors.append(f'v4 transport metadata invalid {kind} rank {e.get("rank")}')
            if e.get('packet_source') != 'source':
                errors.append(f'v4 transport source metadata invalid {kind} rank {e.get("rank")}')
            if not _int(e.get('gpu_bytes')) or not _int(e.get('control_bytes')) or e.get('bytes') != e.get('gpu_bytes', -1) + e.get('control_bytes', -2):
                errors.append(f'v4 transport byte accounting invalid {kind} rank {e.get("rank")}')
    for _, _, e in match('peer_state_authorized', lambda x: x.get('round') == cut_round and
                          (cut is None or x.get('step') == cut)):
        if e.get('version_checked') is not True:
            errors.append(f'v4 peer source metadata was not version-checked rank {e.get("rank")}')
    for kind in ('peer_compute_ready', 'peer_compute_begin', 'peer_native_gradient_ready'):
        evs = [x for x in flat if x[2].get('kind') == kind and x[2].get('replica') == 1 - scope.replica and
               (cut is None or x[2].get('step') == cut)]
        if sorted(x[2].get('rank') for x in evs) != list(range(world)):
            errors.append(f'v4 peer continuation marker missing: {kind}')
    act_keys = set()
    for (replica, rank), (label, device) in after.items():
        if replica != scope.replica:
            continue
        evs = [e for e in rows.get(label, {}).get(device, []) if e.get('kind') == 'activated' and
               e.get('round') == cut_round and (cut is None or e.get('step') == cut)]
        if len(evs) != 1:
            errors.append(f'v4 activated evidence missing/duplicate {replica}/{rank}')
            continue
        e = evs[0]
        if e.get('decision') != 'replace' or e.get('owner') != target_role or e.get('epoch') != epoch+1:
            errors.append(f'v4 active owner metadata mismatch {replica}/{rank}')
        act_keys.add((replica, rank))
    if act_keys != {(scope.replica,k) for k in range(world)}:
        errors.append('v4 activated evidence must cover the affected replica')
    if td.get('generation') != seq:
        errors.append('v4 candidate generation mismatch')
    if after and any(after.get((scope.replica, rank)) != (target, device)
                     for device, rank in enumerate(stage_ranks)):
        errors.append('v4 candidate did not become owner of the selected stage')


def _verify_v1(cmdrec, before, after, rows, actors, flat, errors, *, world=8, lab=False):
    command = cmdrec.get('command', {})
    seq = cmdrec.get('sequence')
    target = cmdrec.get('target_actor')
    source_replica = command.get('source_replica') if isinstance(command, dict) else None
    if not _int(source_replica) or source_replica not in (0, 1):
        errors.append('v1 source replica must be 0 or 1')
        return
    healthy_replica = 1 - source_replica
    expected_kind = 'observed_exit' if lab else 'SIGKILL'
    if cmdrec.get('fault_kind') != expected_kind:
        errors.append(f'v1 recovery lacks {expected_kind} fault evidence')
    proof = cmdrec.get('fault_proof')
    old_pairs = {r: before.get((source_replica, r)) for r in range(world)}
    old_labels = {pair[0] for pair in old_pairs.values() if pair}
    pids_by_label = {}
    for label in old_labels:
        pids_by_label[label] = {e.get('pid') for events in rows.get(label, {}).values()
                                for e in events if _int(e.get('pid'))}
    expected_pids = {(actors.get(label, {}).get('pod_uid'), pid)
                     for label, pids in pids_by_label.items() for pid in pids}
    if not isinstance(proof, list) or not proof:
        errors.append('v1 recovery lacks source death proof')
    else:
        proved_pids = set()
        proved_units = set()
        units_by_label = {label: actors.get(label, {}).get('unit') for label in old_labels}
        labels_by_unit = {unit: label for label, unit in units_by_label.items()
                          if isinstance(unit, str) and unit}
        for item in proof:
            if (not isinstance(item, dict) or (not lab and item.get('signal') != 'SIGKILL') or
                    (lab and not _int(item.get('returncode'))) or
                    item.get('all_workers_dead') is not True or
                    not isinstance(item.get('worker_pids'), list) or
                    not _int(item.get('main_pid')) or item.get('main_pid') <= 0):
                errors.append('v1 source death proof is incomplete')
                continue
            unit = item.get('unit')
            label = labels_by_unit.get(unit)
            if not label or unit in proved_units:
                errors.append('v1 source death proof unit does not uniquely match a source actor')
                continue
            if lab and item.get('pod_uid') != actors[label].get('pod_uid'):
                errors.append('v1 source death proof is from a different Pod')
                continue
            if lab and cmdrec.get('normal_source_exit') is True and item.get('returncode') != 0:
                errors.append('v1 source did not exit normally')
            proved_units.add(unit)
            pids = item['worker_pids']
            expected_actor_pids = pids_by_label.get(label, set())
            if (len(pids) != len(expected_actor_pids) or
                    any(not _int(pid) or pid <= 0 for pid in pids) or
                    set(pids) != expected_actor_pids):
                errors.append(f'v1 source death proof PID set mismatch for {label}')
            if len(pids) != len(set(pids)):
                errors.append(f'v1 source death proof repeats worker PID for {label}')
            proved_pids.update((actors[label].get('pod_uid'), pid) for pid in pids if _int(pid))
        if (proved_units != set(labels_by_unit) or proved_pids != expected_pids or
                len(expected_pids) != world):
            errors.append('v1 source death proof PID set mismatch')
    if not target or target not in actors:
        errors.append('v1 recovery candidate actor missing')
        return
    d = actors[target]
    if d.get('role') != 'D' or d.get('replica') != source_replica or len(d.get('devices', [])) != world:
        errors.append(f'v1 replacement must be a {world}-rank D actor on the source replica')
    if after and any(after.get((source_replica, r)) != (target, r) for r in range(world)):
        errors.append('v1 replacement did not become the full active replica')
    if cmdrec.get('prepare_v1'):
        prepared = _actor_events(rows, target, 'session_native_prepared')
        if (cmdrec.get('candidate_prepared_before_exit') is not True
                or sorted(event[2].get('rank') for event in prepared) != list(range(world))):
            errors.append('v1 warm candidate lacks pre-exit readiness for all ranks')
    restores = _actor_events(rows, target, 'session_native_restored')
    if sorted(x[2].get('rank') for x in restores) != list(range(world)) or any(x[2].get('source_replica') != healthy_replica for x in restores):
        errors.append(f'v1 native source handoff missing across {world} ranks')
    if before:
        for rank in range(world):
            old = before.get((source_replica, rank))
            new = after.get((source_replica, rank))
            if not old or not new or old[0] == new[0]:
                errors.append(f'v1 source not replaced rank {rank}')
                continue
            old_events = rows.get(old[0], {}).get(old[1], [])
            commits = [e.get('step') for e in old_events if e.get('kind') == 'session_commit' and _int(e.get('step'))]
            if commits and max(commits) > command.get('at_step', math.inf):
                errors.append(f'v1 killed source advanced past arrival rank {rank}')
            if any(e.get('kind') == 'session_done' for e in old_events):
                errors.append(f'v1 killed source incorrectly finalized rank {rank}')
            if lab and cmdrec.get('normal_source_exit') is True:
                exits = [e for e in old_events if e.get('kind') == 'session_source_exit'
                         and e.get('step') == command.get('at_step') and e.get('reason') == 'committed_boundary'
                         and e.get('command_id') == command.get('id')]
                if len(exits) != 1:
                    errors.append(f'v1 source lacks committed-boundary exit evidence rank {rank}')
    # The opposite replica is healthy and must keep its mapping and process IDs.
    for rank in range(world):
        old, new = before.get((healthy_replica, rank)), after.get((healthy_replica, rank))
        if not old or not new or old != new:
            errors.append(f'v1 healthy worker mapping changed rank {rank}')
            continue
        def pid_set(pair):
            return {e.get('pid') for e in rows.get(pair[0], {}).get(pair[1], []) if _int(e.get('pid'))}
        if len(pid_set(old)) != 1 or pid_set(old) != pid_set(new):
            errors.append(f'v1 healthy worker PID changed rank {rank}')


def _verify(root, expected_counts=None, *, world=8, tp=2):
    """Return independent acceptance findings; never raises for bad evidence."""
    root = Path(root)
    errors = []
    summary = _read_json(root / 'summary.json', errors, 'summary unavailable')
    config = _configuration(summary, errors, world=world, tp=tp)
    world, tp = config['replica_world_size'], config['tp']
    width = world // 2
    for key, expected in (('done', True), ('error', None), ('resources_clean', True)):
        if summary.get(key) is not expected:
            errors.append(f'summary {key} must be {expected!r}')
    if summary.get('cleanup_errors') != []:
        errors.append('summary cleanup_errors must be empty')
    audit = summary.get('audit')
    if type(audit) is not bool:
        errors.append('summary audit flag missing')
        audit = False
    descriptors = summary.get('actors')
    actors = {}
    if not isinstance(descriptors, dict):
        errors.append('summary actor descriptors missing')
        descriptors = {}
    for label, d in descriptors.items():
        if (not isinstance(d, dict) or d.get('role') not in ('A','B','C','D','E') or
            not _int(d.get('replica')) or d.get('replica') not in (0,1) or
            not _int(d.get('generation')) or d.get('generation') < 0 or
            not isinstance(d.get('node'), str) or not isinstance(d.get('unit'), str) or
            not isinstance(d.get('devices'), list)):
            errors.append(f'invalid actor descriptor {label}')
            continue
        if label != f"g{d['generation']}_{d['role']}":
            errors.append(f'actor label mismatch {label}')
        if 'ranks' in d:
            allowed = (d['role'] in ('B', 'C') and d['generation'] > 0 and len(d['devices']) == width
                       and all(_actor_rank(d, device, world) is not None for device in range(width)))
        elif d['role'] == 'D':
            allowed = ((d['generation'] == 0 and d['replica'] == 1 and
                        d['devices'] == list(range(width))) or
                       (d['generation'] > 0 and d['replica'] in (0,1) and
                        len(d['devices']) == world) )
        elif d['role'] == 'A':
            allowed = d['replica'] == 0 and d['devices'] == list(range(width))
        elif d['role'] in ('B','C','E'):
            allowed = (d['replica'] == (1 if d['role'] == 'E' else 0) and
                       (len(d['devices']) == width if d['generation']>0 and d['role'] in ('B','C')
                        else d['devices'] == list(range(width, world))))
        else:
            allowed = False
        if not allowed or len(set(d['devices'])) != len(d['devices']) or any(not _int(x) or x not in range(8) for x in d['devices']):
            errors.append(f'invalid actor placement {label}')
        if d['role'] == 'D' and d['generation'] > 0 and len(d['devices']) != world:
            errors.append(f'generation D candidate must be full DP {label}')
        actors[label] = d
    rows = _event_rows(root, actors, errors, world=world, tp=tp)
    flat = _flatten(rows)
    final_active = _active_map(summary, actors, errors, world=world)
    lab = summary.get('backend') == 'lab_pod'
    _resource_check(root, errors, lab=lab, actors=actors)
    if lab:
        _lab_gpu_inventory(summary, actors, errors)
    if not actors:
        errors.append('no session actors')
    if summary.get('boundary_validation') not in (None, 'runtime', 'full'):
        errors.append('invalid boundary_validation')
    if summary.get('boundary_validation') == 'runtime' and audit:
        errors.append('runtime boundary mode conflicts with audit=True')
    if summary.get('boundary_validation') == 'full' and not audit:
        errors.append('full boundary mode conflicts with audit=False')
    boundary_mode = summary.get('boundary_validation') or ('full' if audit else 'runtime')
    index = _event_index(flat)
    # Commit signatures and sample coverage are checked independently of policy events.
    by_step = defaultdict(list)
    logical_hashes = defaultdict(list)
    for label, device, e in index.get('session_commit', []):
        if audit:
            if not _hashes(e.get('hashes')):
                errors.append(f'full audit commit lacks nine hashes {label}{device}/step{e.get("step")}')
        elif 'hashes' not in e or e.get('hashes') is not None:
            errors.append(f'runtime commit carries hashes {label}{device}/step{e.get("step")}')
        if not e.get('contributed'):
            continue
        key = (e.get('replica'), e.get('rank'), e.get('step'))
        by_step[e.get('step')].append((key, e))
        if audit and _hashes(e.get('hashes')):
            logical_hashes[key].append(e['hashes'])
    for step, rows_at_step in by_step.items():
        participants = {e.get('participants') for _, e in rows_at_step}
        if len(participants) != 1 or next(iter(participants), None) not in (1,2):
            errors.append(f'global sample coverage participation mismatch step {step}')
            continue
        expected = set(range((step-1)*8, step*8))
        actual = {sid for _, e in rows_at_step for sid in (e.get('sample_ids') or []) if _int(sid)}
        if actual != expected:
            errors.append(f'global sample coverage mismatch step {step}')
        seen_keys = [key for key, _ in rows_at_step]
        if len(seen_keys) != len(set(seen_keys)):
            errors.append(f'duplicate contributing logical worker step {step}')
    if audit:
        common = [step for step, vals in by_step.items()
        if {(key[0], key[1]) for key, _ in vals} == {(r,k) for r in range(2) for k in range(world)}]
        if not common:
            errors.append(f'full audit has no common {2*world}-worker commit step')
        for step in common:
            for rank in range(world):
                a = logical_hashes.get((0,rank,step), [])
                b = logical_hashes.get((1,rank,step), [])
                if len(a) != 1 or len(b) != 1 or a[0] != b[0]:
                    errors.append(f'full audit paired commit hash mismatch rank {rank}/step{step}')
    # Runtime boundary events cannot smuggle hashes into timed windows.
    runtime_kinds = {'boundary_selected','freeze','outer_retention_boundary','outer_retained',
        'peer_state_authorized','state_sent','state_received','candidate_validated',
        'candidate_rejected','candidate_activated','peer_compute_ready','activated',
        'snapshot_ready','final_audit_ready','final_audit_begin'}
    if not audit:
        for label, device, e in flat:
            if e.get('kind') in runtime_kinds and any(e.get(k) is not None for k in ('hashes','expected_hashes','oracle_hashes')):
                errors.append(f'runtime boundary hash leakage {label}{device}/{e.get("kind")}')
            if e.get('kind') == 'state_hash' and e.get('final_audit') is not True:
                errors.append(f'runtime state hash outside final audit {label}{device}')
    # Final logical worker set and paired terminal hashes are mandatory for all modes.
    final = {}
    for (replica, rank), (label, device) in final_active.items():
        evs = [e for e in rows.get(label, {}).get(device, []) if e.get('kind') == 'session_done']
        if len(evs) != 1:
            errors.append(f'final session_done missing/duplicate {replica}/{rank}')
            continue
        e = evs[0]
        if not _hashes(e.get('hashes')):
            errors.append(f'final nine-state hash missing {replica}/{rank}')
        final[(replica,rank)] = e
    for rank in range(world):
        a, b = final.get((0,rank)), final.get((1,rank))
        if a and b and (a.get('step') != b.get('step') or a.get('hashes') != b.get('hashes')):
            ah, bh = a.get('hashes'), b.get('hashes')
            diff = sorted(k for k in REFERENCE_HASH_KEYS
                          if not isinstance(ah, dict) or not isinstance(bh, dict) or ah.get(k) != bh.get(k))
            errors.append(f'final paired logical state mismatch rank {rank}: {",".join(diff)}')
    if final and (len({e.get('step') for e in final.values()}) != 1):
        errors.append('final session_done steps are not common')
    commands = summary.get('commands', [])
    if not isinstance(commands, list):
        errors.append('summary commands must be a list')
        commands = []
    if commands:
        active = _initial_map(actors, errors, world=world)
        policies = [x.get('command', {}).get('policy') if isinstance(x, dict) else None for x in commands]
        if any(policy not in ('v1','v2','v4') for policy in policies):
            errors.append('command policy must be one of v1, v2 or v4')
        if summary.get('epoch') != len(commands):
            errors.append('session epoch must equal the number of completed commands')
    else:
        active = final_active
        if summary.get('epoch') != 0:
            errors.append('no-command session must remain at epoch 0')
        initial = _initial_map(actors, errors, world=world)
        if active and active != initial:
            errors.append('no-command session active map is not initial cohort')
    source_manifest = []
    for i, rec in enumerate(commands):
        if not isinstance(rec, dict):
            errors.append('invalid command record')
            continue
        c = rec.get('command', {})
        if not isinstance(c, dict):
            errors.append('invalid command metadata')
            continue
        policy = c.get('policy')
        if rec.get('status') != 'completed':
            errors.append(f'{policy} command is not completed')
        if rec.get('sequence') != i+1 or c.get('expected_epoch') != i:
            errors.append(f'{policy} command sequence/epoch mismatch')
        source_replica = c.get('source_replica')
        source_allowed = _int(source_replica) and (
            source_replica in (0,1) if policy in ('v1', 'v4') else source_replica == 1)
        if not source_allowed:
            errors.append(f'{policy} source replica mismatch')
        wanted_action = 'recover' if policy == 'v1' else 'migrate'
        if c.get('action') != wanted_action:
            errors.append(f'{policy} action mismatch')
        if policy in ('v1','v2') and tuple(c.get('ranks') or range(world)) != tuple(range(world)):
            errors.append(f'{policy} command must replace the full DP replica')
        source_ranks = range(world)
        expected_role = 'D'
        if policy == 'v4':
            try:
                scope = SessionLayout(world).partial_scope(source_replica, c.get('ranks') or ())
                source_ranks = scope.ranks
            except (TypeError, ValueError):
                source_ranks = ()
                errors.append('v4 command must replace one complete stage')
            old_roles = {actors[pair[0]]['role'] for rank in source_ranks
                         if (pair := active.get((source_replica, rank))) and pair[0] in actors}
            expected_role = partial_roles(next(iter(old_roles)))[1] if len(old_roles) == 1 else None
        target_label = rec.get('target_actor')
        target_desc = actors.get(target_label)
        if (not target_desc or target_desc.get('role') != expected_role or
                target_desc.get('generation') != rec.get('sequence') or
                target_desc.get('node') != c.get('target') or
                rec.get('devices') != target_desc.get('devices')):
            errors.append(f'{policy} target actor descriptor does not match command')
        source_map = {(source_replica, r): active.get((source_replica,r)) for r in source_ranks}
        source_labels = sorted({pair[0] for pair in source_map.values() if pair})
        source_manifest.append(dict(sequence=rec.get('sequence'), policy=policy,
            epoch=c.get('expected_epoch'), source_replica=source_replica,
            source_labels=source_labels,
            sources={f'{replica}/{rank}': dict(label=pair[0], device=pair[1],
                node=actors[pair[0]]['node'], role=actors[pair[0]]['role'],
                generation=actors[pair[0]]['generation'])
                for (replica,rank), pair in source_map.items() if pair and pair[0] in actors},
            target_actor=target_label,
            target_node=target_desc.get('node') if target_desc else None))
        if not isinstance(c.get('id'), str) or not _int(c.get('at_step')):
            errors.append(f'{policy} command metadata incomplete')
        if not _int(rec.get('completed_step')) or rec.get('completed_step',0) <= c.get('at_step',-1):
            errors.append(f'{policy} completed step invalid')
        if i and _int(commands[i-1].get('completed_step')) and _int(c.get('at_step')) and c['at_step'] <= commands[i-1]['completed_step']:
            errors.append('command steps overlap or are not continuous')
        common = rec.get('first_common_observation')
        if not isinstance(common, dict) or not _finite(common.get('request_seconds')):
            errors.append(f'{policy} first_common_observation missing request timing')
        elif common['request_seconds'] < 0:
            errors.append(f'{policy} negative request timing')
        if isinstance(common, dict):
            observed_step = common.get('step')
            wanted_common = dict(active)
            target_label = rec.get('target_actor')
            if policy in ('v1','v2'):
                replace_replica = source_replica if policy == 'v1' else 1
                for rank in range(world): wanted_common[(replace_replica,rank)] = (target_label,rank)
            elif policy == 'v4':
                for device, rank in enumerate(source_ranks):
                    wanted_common[(source_replica,rank)] = (target_label,device)
            wanted_keys = {(replica,rank,label,device) for (replica,rank),(label,device) in wanted_common.items()}
            observed_rows = []
            for label, device, e in flat:
                if (e.get('kind') == 'session_commit' and e.get('command_id') == c.get('id') and
                    e.get('step') == observed_step and e.get('participants') == 2 and
                    e.get('contributed') is True and
                    (e.get('replica'),e.get('rank'),label,device) in wanted_keys):
                    observed_rows.append(e)
            coverage = {(e.get('replica'), e.get('rank')) for e in observed_rows}
            if (not _int(observed_step) or len(observed_rows) != 2*world or
                    coverage != {(r,k) for r in range(2) for k in range(world)} or
                    not _finite(common.get('observed_monotonic'))):
                errors.append(f'{policy} first_common_observation lacks {2*world}-worker commit proof')
            requested = rec.get('requested_monotonic')
            if (_finite(requested) and _finite(common.get('observed_monotonic')) and
                    abs(common.get('request_seconds', 0) - (common['observed_monotonic']-requested)) > .02):
                errors.append(f'{policy} request timing does not match controller clocks')
        fault = common.get('fault_seconds') if isinstance(common, dict) else None
        if policy == 'v1' and (not _finite(fault) or fault < 0):
            errors.append('v1 first_common_observation missing fault timing')
        if policy == 'v1' and isinstance(common, dict) and _finite(rec.get('fault_monotonic')) and _finite(common.get('observed_monotonic')):
            if abs(fault - (common['observed_monotonic']-rec['fault_monotonic'])) > .02:
                errors.append('v1 fault timing does not match controller clocks')
        if policy in ('v2','v4') and fault is not None:
            errors.append(f'{policy} must not report fault timing')
        expected_map = dict(active)
        if policy in ('v2','v1'):
            target = rec.get('target_actor')
            replace_replica = source_replica if policy == 'v1' else 1
            for r in range(world): expected_map[(replace_replica,r)] = (target,r)
        elif policy == 'v4':
            target = rec.get('target_actor')
            for device, rank in enumerate(source_ranks):
                expected_map[(source_replica,rank)] = (target,device)
        if policy == 'v2':
            _verify_v2(rec, active, expected_map, rows, actors, flat, errors, world=world)
        elif policy == 'v4':
            _verify_v4(rec, active, expected_map, rows, actors, flat, errors,
                       boundary_mode, world=world)
        elif policy == 'v1':
            _verify_v1(rec, active, expected_map, rows, actors, flat, errors, world=world, lab=lab)
        else:
            errors.append(f'unsupported session command policy {policy}')
        active = expected_map
        # Active-cohort command completion evidence is worker-local and step exact.
        cid = c.get('id')
        for (replica, rank), (label, device) in active.items():
            evs = [e for e in rows.get(label, {}).get(device, []) if e.get('kind') == 'session_command_done' and
                   e.get('command_id') == cid and e.get('policy') == policy]
            if len(evs) != 1 or evs[0].get('step') != rec.get('completed_step'):
                errors.append(f'{policy} command completion event missing {replica}/{rank}')
    # The controller's committed final map must equal the command-derived map.
    if commands and active and active != final_active:
        errors.append('summary final active map disagrees with command sequence')
    policy_counts = {policy: sum(1 for rec in commands
                        if isinstance(rec, dict) and isinstance(rec.get('command'), dict) and
                        rec['command'].get('policy') == policy)
                     for policy in ('v1','v2','v4')}
    if expected_counts is not None:
        if not isinstance(expected_counts, dict) or any(k not in policy_counts or not _int(v) or v < 0
                                                        for k,v in expected_counts.items()):
            errors.append('expected_counts must map known policies to nonnegative integers')
        else:
            wanted_counts = {policy: expected_counts.get(policy, 0) for policy in policy_counts}
            if wanted_counts != policy_counts:
                errors.append(f'policy counts mismatch: expected {wanted_counts}, observed {policy_counts}')
    # Timing summaries use only per-worker durations/monotonic deltas; never subtract host clocks.
    policy_ids = {rec.get('command', {}).get('id') for rec in commands if isinstance(rec, dict)}
    first_at = commands[0].get('command', {}).get('at_step') if commands else None
    last_done = commands[-1].get('completed_step') if commands else None
    phase_values = defaultdict(lambda: defaultdict(list))
    for label, device, e in index.get('session_commit', []):
        step, duration = e.get('step'), e.get('seconds')
        if not _int(step) or not _finite(duration):
            continue
        if commands and e.get('command_id') in policy_ids:
            phase = 'during'
        elif commands and _int(first_at) and step <= first_at:
            phase = 'pre'
        elif commands and _int(last_done) and step > last_done:
            phase = 'post'
        elif commands:
            phase = 'during'
        else:
            phase = 'pre'
        d = actors.get(label, {})
        rank = _actor_rank(d, device, world)
        if _int(rank):
            phase_values[phase][(d.get('replica'),rank)].append(float(duration))
    timing = {}
    for phase in ('pre','during','post'):
        means = [sum(xs)/len(xs) for xs in phase_values[phase].values() if xs]
        timing[phase] = dict(rank_count=len(means), mean_rank_mean_seconds=(sum(means)/len(means) if means else None),
                             max_rank_mean_seconds=(max(means) if means else None))
    return dict(pass_all=not errors, errors=errors, configuration=config, policy_counts=policy_counts,
                source_manifest=source_manifest,
                statistics=dict(session=summary.get('session'), actors=len(actors),
                    final_workers=len(final), commands=len(commands),
                    commits=len(index.get('session_commit', [])), timing=timing))


def verify(root, expected_counts=None, *, world=8, tp=2):
    """Return findings for valid or malformed input without leaking exceptions."""
    try:
        return _verify(root, expected_counts, world=world, tp=tp)
    except Exception as exc:
        return dict(pass_all=False, errors=[f'verifier input failure: {type(exc).__name__}: {exc}'],
                    configuration=None, policy_counts=dict(v1=0,v2=0,v4=0), source_manifest=[],
                    statistics=dict(session=None, actors=0, final_workers=0,
                                    commands=0, commits=0, timing={}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='results/session_NAME directory')
    parser.add_argument('--two-each', action='store_true',
                        help='require exactly two completed commands for each policy')
    args = parser.parse_args()
    result = verify(args.root, expected_counts=({'v1':2,'v2':2,'v4':2} if args.two_each else None))
    audit_path = args.root / 'session_audit.json'
    temp = args.root / 'session_audit.tmp'
    temp.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    temp.replace(audit_path)
    print(json.dumps(dict(pass_all=result['pass_all'], errors=result['errors'],
                          policy_counts=result['policy_counts'], source_manifest=result['source_manifest'],
                          statistics=result['statistics']), sort_keys=True))
    raise SystemExit(0 if result['pass_all'] else 1)


if __name__ == '__main__':
    main()
