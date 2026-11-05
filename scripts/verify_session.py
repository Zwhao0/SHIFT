"""CPU-only independent verifier for persistent mixed-policy session runs."""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from shift_full.reference_audit import REFERENCE_HASH_KEYS
from shift_full.training_core import sample_plan


def _int(x):
    return type(x) is int


def _finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def _hashes(x):
    return (isinstance(x, dict) and set(x) == REFERENCE_HASH_KEYS and
            all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in x.values()))


def _logical_rank(role, device, replica, full_dp=False):
    if full_dp:
        return device
    if role in ('A', 'D'):
        return device
    if role in ('B', 'C', 'E'):
        return device + 4
    return None


def _read_json(path, errors, label):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('expected object')
        return value
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f'{label}: {exc}')
        return {}


def _event_rows(root, actors, errors):
    rows, pids, expected = {}, {}, set()
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
            event_pids = set()
            for e in events:
                full_dp = d['role'] == 'D' and len(d['devices']) == 8
                rank = _logical_rank(d['role'], device, d['replica'], full_dp)
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
                pid = next(iter(event_pids))
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
                            local_data_rank=rank % 4 // 2)
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


def _active_map(summary, actors, errors):
    active = {}
    value = summary.get('active')
    if not isinstance(value, dict):
        errors.append('final active mapping missing')
        return active
    for replica in range(2):
        for rank in range(8):
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
            full_dp = d['role'] == 'D' and len(d['devices']) == 8
            if d['replica'] != replica or _logical_rank(d['role'], device, replica, full_dp) != rank:
                errors.append(f'active logical rank mismatch {key}')
            active[(replica, rank)] = (label, device)
    if len(value) != 16:
        errors.append('active mapping must contain exactly 16 workers')
    physical = []
    for (replica, rank), (label, device) in active.items():
        d = actors[label]
        physical.append((d['node'], d['devices'][device]))
    if len(physical) != len(set(physical)):
        errors.append('final active physical GPU collision')
    return active


def _initial_map(actors, errors):
    result = {}
    wanted = {
        **{(0,r):('g0_A',r) for r in range(4)},
        **{(0,r):('g0_B',r-4) for r in range(4,8)},
        **{(1,r):('g0_D',r) for r in range(4)},
        **{(1,r):('g0_E',r-4) for r in range(4,8)},
    }
    for key, pair in wanted.items():
        label, device = pair
        if label not in actors or device >= len(actors.get(label, {}).get('devices', [])):
            errors.append(f'initial actor missing for {key[0]}/{key[1]}')
        else:
            result[key] = pair
    return result


def _resource_check(root, errors):
    before = _read_json(root / 'resources_before.json', errors, 'resources_before unavailable')
    after = _read_json(root / 'resources_after.json', errors, 'resources_after unavailable')
    if len(before) != 3 or set(before) != set(after):
        errors.append('resource evidence must cover the same three nodes')
        return
    for node in sorted(before):
        b, a = before[node], after[node]
        bc, ac = b.get('company'), a.get('company')
        if not isinstance(bc, dict) or len(bc) != 8:
            errors.append(f'{node}: expected eight company PIDs before run')
        if not isinstance(ac, dict) or len(ac) != 8:
            errors.append(f'{node}: expected eight company PIDs after run')
        if bc != ac:
            errors.append(f'{node}: company PIDs changed')
        if b.get('extra_gpu') != [] or a.get('extra_gpu') != []:
            errors.append(f'{node}: extra GPU process present')


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


def _verify_v2(cmdrec, before, after, rows, actors, flat, errors):
    command = cmdrec.get('command', {})
    seq = cmdrec.get('sequence')
    if not isinstance(command, dict) or not _int(seq):
        errors.append('v2 source handoff: invalid command metadata')
        return
    # The runtime round field is the session command sequence; the reused
    # PlannedHandoff's internal round remains sequence-1 in its manager state.
    cut_id = seq
    source_labels = {before[(1, r)][0] for r in range(8) if (1, r) in before}
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
        if sorted(x[2].get('rank') for x in evs) != list(range(8)):
            errors.append(f'v2 source handoff: {kind} missing across eight ranks')
    for kind in ('planned_precopy_loaded', 'planned_state_ready',
                 'planned_gradient_received'):
        evs = [x for x in candidate if x[2].get('kind') == kind and
               x[2].get('round') == cut_id and (cut is None or x[2].get('step') == cut)]
        if sorted(x[2].get('rank') for x in evs) != list(range(8)):
            errors.append(f'v2 candidate handoff: {kind} missing across eight ranks')
        if kind == 'planned_state_ready' and any(
                x[2].get('epoch') != command.get('expected_epoch', 0) + 1 for x in evs):
            errors.append(f'v2 candidate handoff epoch mismatch: {kind}')
    healthy_labels = {before[(0, r)][0] for r in range(8) if (0, r) in before}
    activated = [item for label in healthy_labels for item in _actor_events(rows, label, 'planned_epoch_activated')
                 if item[2].get('round') == cut_id and (cut is None or item[2].get('step') == cut)]
    if sorted(x[2].get('rank') for x in activated) != list(range(8)) or any(
            x[2].get('epoch') != command.get('expected_epoch', 0) + 1 for x in activated):
        errors.append('v2 healthy cohort activation missing across eight ranks')
    sent = source_kind('planned_gradient_sent')
    received = [x for x in candidate if x[2].get('kind') == 'planned_gradient_received']
    if cut is not None:
        sent = [x for x in sent if x[2].get('step') == cut and x[2].get('round') == cut_id]
        received = [x for x in received if x[2].get('step') == cut and x[2].get('round') == cut_id]
    if sorted(x[2].get('rank') for x in sent) != list(range(8)) or sorted(x[2].get('rank') for x in received) != list(range(8)):
        errors.append('v2 gradient handoff missing across eight ranks')
    else:
        sm = {x[2]['rank']: x[2] for x in sent}
        rm = {x[2]['rank']: x[2] for x in received}
        for rank in range(8):
            if sm[rank].get('sample_ids') != rm[rank].get('sample_ids'):
                errors.append(f'v2 gradient sample handoff mismatch rank {rank}')
            if rm[rank].get('forward_calls') != 0:
                errors.append(f'v2 relay performed target forward rank {rank}')
    retired = source_kind('session_retired')
    if sorted(x[2].get('rank') for x in retired) != list(range(8)):
        errors.append('v2 old owner retirement missing across eight ranks')
    # Candidate uses D/full-DP; it must become the replica-1 active cohort.
    d = actors[target]
    if d.get('role') != 'D' or d.get('replica') != 1 or len(d.get('devices', [])) != 8:
        errors.append('v2 candidate must be an eight-rank replica-1 D actor')
    if d.get('generation') != seq:
        errors.append('v2 candidate generation mismatch')
    if after and any(after.get((1, r)) != (target, r) for r in range(8)):
        errors.append('v2 candidate did not become the full active replica')


def _verify_v4(cmdrec, before, after, rows, actors, flat, errors, boundary_mode):
    command = cmdrec.get('command', {})
    seq, epoch = cmdrec.get('sequence'), command.get('expected_epoch')
    if not isinstance(command, dict) or not _int(seq) or not _int(epoch):
        errors.append('v4 transaction: invalid command metadata')
        return
    cut_round = epoch
    old_labels = {before[(0, r)][0] for r in range(4, 8) if (0, r) in before}
    old_roles = {actors[label]['role'] for label in old_labels if label in actors}
    target_role = ({'B','C'} - old_roles).pop() if len(old_roles) == 1 and old_roles <= {'B','C'} else None
    target = cmdrec.get('target_actor')
    if not target or target not in actors:
        errors.append('v4 transaction: candidate actor missing')
        return
    td = actors[target]
    if target_role is None or td.get('role') != target_role or td.get('replica') != 0 or len(td.get('devices', [])) != 4:
        errors.append('v4 candidate must replace the current B/C owner with four PP1 ranks')
    def match(kind, pred):
        return [x for x in flat if x[2].get('kind') == kind and pred(x[2])]
    boundaries = [x for x in flat if x[2].get('kind') == 'boundary_selected' and
                   x[2].get('round') == cut_round]
    freezes = [x for x in flat if x[2].get('kind') == 'freeze' and x[2].get('round') == cut_round]
    if len(boundaries) != 16 or { (e.get('replica'), e.get('rank')) for _,_,e in boundaries } != {(r,k) for r in range(2) for k in range(8)}:
        errors.append('v4 boundary evidence must cover all 16 workers')
    if len(freezes) != 16 or {(e.get('replica'), e.get('rank')) for _,_,e in freezes} != {(r,k) for r in range(2) for k in range(8)}:
        errors.append('v4 freeze evidence must cover all 16 workers')
    cuts = {e.get('step') for _,_,e in boundaries}
    if len(cuts) != 1 or not all(_int(x) for x in cuts):
        errors.append('v4 boundary cut is not common')
        cut = None
    else:
        cut = next(iter(cuts))
    if cut is not None and any(e.get('step') != cut for _,_,e in freezes):
        errors.append('v4 freeze cut mismatch')
    decisions = match('switch_decision', lambda e: e.get('round') == cut_round and e.get('replica') == 0 and e.get('rank') == 0)
    if len(decisions) != 1:
        errors.append('v4 switch decision missing')
    else:
        de = decisions[0][2]
        if de.get('decision') != 'replace' or de.get('owner') != target_role or de.get('validation') != boundary_mode:
            errors.append(f'v4 switch decision is not {boundary_mode} replace to {target_role}')
        if de.get('epoch') != epoch + 1 or (cut is not None and de.get('step') != cut):
            errors.append('v4 switch decision epoch/cut mismatch')
        results = de.get('candidate_results')
        if not isinstance(results, dict) or {str(k) for k in range(4, 8)} - set(results) or any(results.get(str(k)) is not True for k in range(4, 8)):
            errors.append('v4 switch decision lacks four successful candidate results')
    validated = _actor_events(rows, target, 'candidate_validated')
    validated = [x for x in validated if x[2].get('round') == cut_round and
                 (cut is None or x[2].get('step') == cut)]
    if sorted(x[2].get('rank') for x in validated) != list(range(4, 8)):
        errors.append('v4 candidate validation missing for four ranks')
    for _, _, e in validated:
        if not _exact_readiness(e, epoch=epoch+1,
                rank=e.get('rank'), cut=cut, target=target_role):
            errors.append(f'v4 candidate readiness mismatch rank {e.get("rank")}')
    activated = _actor_events(rows, target, 'candidate_activated')
    activated = [x for x in activated if x[2].get('round') == cut_round and
                 (cut is None or x[2].get('step') == cut)]
    if sorted(x[2].get('rank') for x in activated) != list(range(4, 8)):
        errors.append('v4 candidate activation missing for four ranks')
    for _, _, e in activated:
        if not _exact_readiness(e, epoch=epoch+1, rank=e.get('rank'), cut=cut, target=target_role):
            errors.append(f'v4 activation readiness mismatch rank {e.get("rank")}')
    retired = [x for label in old_labels for x in _actor_events(rows, label, 'retired')]
    retired = [x for x in retired if x[2].get('round') == cut_round and
               (cut is None or x[2].get('step') == cut)]
    if sorted(x[2].get('rank') for x in retired) != list(range(4, 8)):
        errors.append('v4 old owner retirement missing for four ranks')
    # Four packed state channels must connect each frozen source rank to its candidate.
    for kind, label_set, want in (('state_sent', old_labels, list(range(4, 8))),
                                  ('state_received', {target}, list(range(4, 8)))):
        evs = [x for label in label_set for x in _actor_events(rows, label, kind)
               if x[2].get('round') == cut_round and (cut is None or x[2].get('step') == cut)]
        if sorted(x[2].get('rank') for x in evs) != want:
            errors.append(f'v4 transport missing {kind} for four ranks')
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
        evs = [x for x in flat if x[2].get('kind') == kind and x[2].get('replica') == 1 and
               (cut is None or x[2].get('step') == cut)]
        if sorted(x[2].get('rank') for x in evs) != list(range(8)):
            errors.append(f'v4 peer continuation marker missing: {kind}')
    act_keys = set()
    for (replica, rank), (label, device) in after.items():
        if replica != 0:
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
    if act_keys != {(0,k) for k in range(8)}:
        errors.append('v4 activated evidence must cover the new replica-0 cohort')
    if td.get('generation') != seq:
        errors.append('v4 candidate generation mismatch')
    if after and any(after.get((0, r)) != (target, r-4) for r in range(4, 8)):
        errors.append('v4 candidate did not become replica-0 PP1 owner')


def _verify_v1(cmdrec, before, after, rows, actors, flat, errors):
    command = cmdrec.get('command', {})
    seq = cmdrec.get('sequence')
    target = cmdrec.get('target_actor')
    if cmdrec.get('fault_kind') != 'SIGKILL':
        errors.append('v1 recovery lacks SIGKILL fault evidence')
    proof = cmdrec.get('fault_proof')
    old_labels = {before[(1, r)][0] for r in range(8) if (1, r) in before}
    expected_pids = {e.get('pid') for label in old_labels for events in rows.get(label, {}).values()
                     for e in events if _int(e.get('pid'))}
    if not isinstance(proof, list) or not proof:
        errors.append('v1 recovery lacks source death proof')
    else:
        proved_pids = set()
        for item in proof:
            if (not isinstance(item, dict) or item.get('signal') != 'SIGKILL' or
                    item.get('all_workers_dead') is not True or
                    not isinstance(item.get('worker_pids'), list) or
                    not _int(item.get('main_pid')) or item.get('main_pid') <= 0):
                errors.append('v1 source death proof is incomplete')
                continue
            pids = item['worker_pids']
            if len(pids) != 8 or any(not _int(pid) or pid <= 0 for pid in pids):
                errors.append('v1 source death proof must contain exact eight worker PIDs')
            proved_pids.update(pids)
        if proved_pids != expected_pids or len(expected_pids) != 8:
            errors.append('v1 source death proof PID set mismatch')
    if not target or target not in actors:
        errors.append('v1 recovery candidate actor missing')
        return
    d = actors[target]
    if d.get('role') != 'D' or d.get('replica') != 1 or len(d.get('devices', [])) != 8:
        errors.append('v1 replacement must be an eight-rank replica-1 D actor')
    if after and any(after.get((1, r)) != (target, r) for r in range(8)):
        errors.append('v1 replacement did not become the full active replica')
    restores = _actor_events(rows, target, 'session_native_restored')
    if sorted(x[2].get('rank') for x in restores) != list(range(8)) or any(x[2].get('source_replica') != 0 for x in restores):
        errors.append('v1 native source handoff missing across eight ranks')
    if before:
        for rank in range(8):
            old = before.get((1, rank))
            new = after.get((1, rank))
            if not old or not new or old[0] == new[0]:
                errors.append(f'v1 source not replaced rank {rank}')
                continue
            old_events = rows.get(old[0], {}).get(old[1], [])
            commits = [e.get('step') for e in old_events if e.get('kind') == 'session_commit' and _int(e.get('step'))]
            if commits and max(commits) > command.get('at_step', math.inf):
                errors.append(f'v1 killed source advanced past arrival rank {rank}')
            if any(e.get('kind') == 'session_done' for e in old_events):
                errors.append(f'v1 killed source incorrectly finalized rank {rank}')
    # Replica 0 is the healthy cohort and must keep the same process IDs.
    for rank in range(8):
        old, new = before.get((0, rank)), after.get((0, rank))
        if not old or not new or old != new:
            errors.append(f'v1 healthy worker mapping changed rank {rank}')
            continue
        def pid_set(pair):
            return {e.get('pid') for e in rows.get(pair[0], {}).get(pair[1], []) if _int(e.get('pid'))}
        if len(pid_set(old)) != 1 or pid_set(old) != pid_set(new):
            errors.append(f'v1 healthy worker PID changed rank {rank}')


def _verify(root, expected_counts=None):
    """Return independent acceptance findings; never raises for bad evidence."""
    root = Path(root)
    errors = []
    summary = _read_json(root / 'summary.json', errors, 'summary unavailable')
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
        allowed = (d['role'] == 'D' and d['replica'] == 1 and len(d['devices']) in (4,8)) or (d['role'] in ('A','B','C','E') and len(d['devices']) == 4 and d['replica'] == (1 if d['role']=='E' else 0))
        if not allowed or len(set(d['devices'])) != len(d['devices']) or any(not _int(x) or x not in range(8) for x in d['devices']):
            errors.append(f'invalid actor placement {label}')
        if d['role'] == 'D' and d['generation'] > 0 and len(d['devices']) != 8:
            errors.append(f'generation D candidate must be full DP {label}')
        actors[label] = d
    rows = _event_rows(root, actors, errors)
    flat = _flatten(rows)
    final_active = _active_map(summary, actors, errors)
    _resource_check(root, errors)
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
                  if {(key[0], key[1]) for key, _ in vals} == {(r,k) for r in range(2) for k in range(8)}]
        if not common:
            errors.append('full audit has no common 16-worker commit step')
        for step in common:
            for rank in range(8):
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
    for rank in range(8):
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
        active = _initial_map(actors, errors)
        policies = [x.get('command', {}).get('policy') if isinstance(x, dict) else None for x in commands]
        if any(policy not in ('v1','v2','v4') for policy in policies):
            errors.append('command policy must be one of v1, v2 or v4')
        if summary.get('epoch') != len(commands):
            errors.append('session epoch must equal the number of completed commands')
    else:
        active = final_active
        if summary.get('epoch') != 0:
            errors.append('no-command session must remain at epoch 0')
        initial = _initial_map(actors, errors)
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
        wanted_source = 0 if policy == 'v4' else 1
        if c.get('source_replica') != wanted_source:
            errors.append(f'{policy} source replica mismatch')
        wanted_action = 'recover' if policy == 'v1' else 'migrate'
        if c.get('action') != wanted_action:
            errors.append(f'{policy} action mismatch')
        if policy in ('v1','v2') and tuple(c.get('ranks') or range(8)) != tuple(range(8)):
            errors.append(f'{policy} command must replace the full DP replica')
        if policy == 'v4' and tuple(c.get('ranks') or ()) != (4,5,6,7):
            errors.append('v4 command must replace PP1 ranks 4..7')
        old_roles = {actors[label]['role'] for rank in range(4,8)
                     if policy == 'v4' and (0,rank) in active
                     for label in (active[(0,rank)][0],) if label in actors}
        target_role = ({'B','C'} - old_roles).pop() if len(old_roles) == 1 and old_roles <= {'B','C'} else None
        expected_role = target_role if policy == 'v4' else 'D'
        target_label = rec.get('target_actor')
        target_desc = actors.get(target_label)
        if (not target_desc or target_desc.get('role') != expected_role or
                target_desc.get('generation') != rec.get('sequence') or
                target_desc.get('node') != c.get('target') or
                rec.get('devices') != target_desc.get('devices')):
            errors.append(f'{policy} target actor descriptor does not match command')
        source_replica = 0 if policy == 'v4' else 1
        source_ranks = range(4,8) if policy == 'v4' else range(8)
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
                for rank in range(8): wanted_common[(1,rank)] = (target_label,rank)
            elif policy == 'v4':
                for rank in range(4,8): wanted_common[(0,rank)] = (target_label,rank-4)
            wanted_keys = {(replica,rank,label,device) for (replica,rank),(label,device) in wanted_common.items()}
            observed_rows = []
            for label, device, e in flat:
                if (e.get('kind') == 'session_commit' and e.get('command_id') == c.get('id') and
                    e.get('step') == observed_step and e.get('participants') == 2 and
                    e.get('contributed') is True and
                    (e.get('replica'),e.get('rank'),label,device) in wanted_keys):
                    observed_rows.append(e)
            coverage = {(e.get('replica'), e.get('rank')) for e in observed_rows}
            if (not _int(observed_step) or len(observed_rows) != 16 or
                    coverage != {(r,k) for r in range(2) for k in range(8)} or
                    not _finite(common.get('observed_monotonic'))):
                errors.append(f'{policy} first_common_observation lacks 16-worker commit proof')
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
            for r in range(8): expected_map[(1,r)] = (target,r)
        elif policy == 'v4':
            target = rec.get('target_actor')
            for r in range(4,8): expected_map[(0,r)] = (target,r-4)
        if policy == 'v2':
            _verify_v2(rec, active, expected_map, rows, actors, flat, errors)
        elif policy == 'v4':
            _verify_v4(rec, active, expected_map, rows, actors, flat, errors, boundary_mode)
        elif policy == 'v1':
            _verify_v1(rec, active, expected_map, rows, actors, flat, errors)
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
        rank = _logical_rank(d.get('role'), device, d.get('replica',0), d.get('role')=='D' and len(d.get('devices',[]))==8)
        if _int(rank):
            phase_values[phase][(d.get('replica'),rank)].append(float(duration))
    timing = {}
    for phase in ('pre','during','post'):
        means = [sum(xs)/len(xs) for xs in phase_values[phase].values() if xs]
        timing[phase] = dict(rank_count=len(means), mean_rank_mean_seconds=(sum(means)/len(means) if means else None),
                             max_rank_mean_seconds=(max(means) if means else None))
    return dict(pass_all=not errors, errors=errors, policy_counts=policy_counts,
                source_manifest=source_manifest,
                statistics=dict(session=summary.get('session'), actors=len(actors),
                    final_workers=len(final), commands=len(commands),
                    commits=len(index.get('session_commit', [])), timing=timing))


def verify(root, expected_counts=None):
    """Return findings for valid or malformed input without leaking exceptions."""
    try:
        return _verify(root, expected_counts)
    except Exception as exc:
        return dict(pass_all=False, errors=[f'verifier input failure: {type(exc).__name__}: {exc}'],
                    policy_counts=dict(v1=0,v2=0,v4=0), source_manifest=[],
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
