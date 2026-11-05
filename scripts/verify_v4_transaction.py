"""Independent CPU audit for multi-round transactional V4 switch runs."""
import hashlib
import json
import math
from pathlib import Path

from shift_full.reference_audit import REFERENCE_HASH_KEYS
from shift_full.v4_protocol import Plan, state_channel_matches


def _is_int(value):
    return type(value) is int


def _finite_nonnegative(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except (OverflowError, TypeError, ValueError):
        return False


def _valid_hashes(value):
    return (isinstance(value, dict) and set(value) == REFERENCE_HASH_KEYS and
            all(isinstance(v, str) and len(v) == 64 and
                all(c in '0123456789abcdef' for c in v) for v in value.values()))


def _valid_sha256(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value))


_RUNTIME_HASHLESS_KINDS = frozenset({
    'boundary_selected', 'freeze', 'outer_retention_boundary', 'outer_retained',
    'peer_state_authorized', 'state_sent', 'state_received',
    'candidate_validated', 'candidate_rejected', 'candidate_activated',
    'peer_compute_ready', 'activated', 'snapshot_ready',
    'final_audit_ready', 'final_audit_begin',
})


def _runtime_readiness_ok(event, *, epoch, rank, cut, target):
    readiness = event.get('readiness')
    expected = dict(schema=1, epoch=epoch, rank=rank, cut=cut,
                    target=target, restored=True)
    return (isinstance(readiness, dict) and readiness == expected and
            all(type(readiness.get(key)) is int
                for key in ('schema', 'epoch', 'rank', 'cut')) and
            readiness.get('target') == target and
            readiness.get('restored') is True)


def _state_channel_ok(sent, received):
    try:
        return state_channel_matches(sent, received)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _actor_key(generation, role):
    return f'g{generation}_{role}'


def _raw_hashes(root):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.glob('*.jsonl'))}


def _load_events(root, actors, errors, cross_node_tp=False):
    rows = {}
    expected_files = set()
    all_pids = {}
    for label, descriptor in actors.items():
        generation, role = descriptor['generation'], descriptor['role']
        rows[label] = {}
        for device in range(4):
            name = f'{label}{device}.jsonl'
            expected_files.add(name)
            path = root / name
            events = []
            try:
                for line_number, line in enumerate(path.read_text().splitlines(), 1):
                    if line.strip():
                        try:
                            value = json.loads(line)
                            if not isinstance(value, dict):
                                raise ValueError('event must be an object')
                            events.append(value)
                        except (ValueError, json.JSONDecodeError) as exc:
                            errors.append(f'malformed event {name}:{line_number}: {exc}')
            except OSError:
                errors.append('missing raw actor file ' + name)
            rows[label][device] = events
            rank = Plan.logical_rank({'D': 'A', 'E': 'B'}.get(role, role), device, cross_node_tp)
            replica = int(role in ('D', 'E'))
            if not events:
                errors.append('empty actor file ' + name)
            pids = {x.get('pid') for x in events}
            if len(pids) != 1 or None in pids or not all(_is_int(x) and x > 0 for x in pids):
                errors.append('PID continuity ' + name)
            elif pids:
                pid = next(iter(pids))
                if pid in all_pids:
                    errors.append('PID reused across actors ' + name)
                all_pids[pid] = name
            for event in events:
                binding = (event.get('role'), event.get('rank'), event.get('replica'),
                           event.get('device'), event.get('actor_round'))
                expected = (role, rank, replica, device, generation)
                if (binding != expected or
                        not all(_is_int(event.get(k)) for k in ('rank', 'replica', 'device', 'actor_round'))):
                    errors.append('actor binding ' + name + '/' + str(event.get('kind')))
                round_id = event.get('round')
                if not _is_int(round_id) or round_id < 0:
                    errors.append('invalid event round ' + name)
                for key in ('step', 'epoch', 'outer_epoch'):
                    if key in event and not _is_int(event[key]):
                        if not (key=='epoch' and event.get('kind')=='connected' and event[key]=='standby'):
                            errors.append('invalid integer event field ' + name + '/' + key)
                for key in ('hashes', 'expected_hashes'):
                    if key in event and event[key] is not None and not _valid_hashes(event[key]):
                        errors.append('incomplete nine-state hash ' + name + '/' + str(event.get('kind')))
    actual_files = {p.name for p in root.glob('*.jsonl')}
    if actual_files != expected_files:
        errors.append('raw actor file inventory mismatch')
    return rows


def verify(root):
    """Check event evidence, switch ownership, commits, signatures and resources."""
    root = Path(root)
    errors = []

    def check(ok, message):
        if not ok:
            errors.append(message)

    try:
        summary = json.loads((root / 'summary.json').read_text())
    except (OSError, ValueError) as exc:
        summary = {}
        errors.append('summary unavailable: ' + str(exc))

    check(summary.get('transactional_switch') is True, 'transactional switch mode')
    boundary_validation = summary.get('boundary_validation', 'full')
    check(boundary_validation in ('full', 'runtime'), 'boundary validation mode')
    if boundary_validation not in ('full', 'runtime'):
        boundary_validation = 'full'
    check(summary.get('functional_pass') is True and summary.get('error') in (None, ''),
          'functional completion')
    check(summary.get('resources_clean') is True and summary.get('cleanup_errors') == [],
          'reported clean resource teardown')
    check(summary.get('overlap_peer') is True and summary.get('outer_manager') is True,
          'shared outer Manager and peer overlap enabled')
    cross_node_tp = summary.get('cross_node_tp', False)
    check(type(cross_node_tp) is bool, 'cross-node TP mode flag')
    cross_node_tp = cross_node_tp is True
    affected_ranks = {1, 3, 5, 7} if cross_node_tp else {4, 5, 6, 7}
    actors_value = summary.get('actors')
    actors = {}
    if not isinstance(actors_value, list):
        errors.append('actor inventory missing')
        actors_value = []
    for item in actors_value:
        if (not isinstance(item, dict) or not _is_int(item.get('generation')) or
                item.get('generation') < 0 or item.get('role') not in ('A', 'B', 'C', 'D', 'E')):
            errors.append('invalid actor descriptor')
            continue
        label = _actor_key(item['generation'], item['role'])
        if label in actors:
            errors.append('duplicate actor descriptor ' + label)
        actors[label] = dict(generation=item['generation'], role=item['role'])
    rows = _load_events(root, actors, errors, cross_node_tp)

    if boundary_validation == 'runtime':
        for label, devices in rows.items():
            for device, worker_events in devices.items():
                for event in worker_events:
                    if (event.get('kind') in _RUNTIME_HASHLESS_KINDS and
                            any(event.get(key) is not None
                                for key in ('hashes', 'expected_hashes', 'oracle_hashes'))):
                        errors.append('runtime boundary carries a state hash ' + label + str(device) + '/' +
                                      str(event.get('kind')))

    cycles = summary.get('cycles')
    plan = summary.get('plan')
    if not _is_int(cycles) or cycles < 1:
        errors.append('positive cycle count required')
        cycles = max(1, len([x for x in actors if x.startswith('g')]))
    if not isinstance(plan, dict):
        errors.append('training plan missing')
        plan = {}
    first_cut = plan.get('cut')
    post_steps = plan.get('post_steps')
    microbatches = plan.get('microbatches', 4)
    between = summary.get('between_steps', 0)
    check(_is_int(first_cut) and first_cut > 0, 'positive initial cut')
    check(_is_int(post_steps) and post_steps > 0, 'positive final post steps')
    check(_is_int(microbatches) and microbatches == (2 if summary.get('unified') else 4), 'microbatches match configured global batch')
    check(_is_int(between) and between >= 0, 'nonnegative between-round steps')
    topology = summary.get('topology')
    check(isinstance(topology, dict) and topology.get('global_batch') == (8 if summary.get('unified') else 16),
          'fixed global batch matches unified/legacy layout')
    first_cut = first_cut if _is_int(first_cut) and first_cut > 0 else 1
    post_steps = post_steps if _is_int(post_steps) and post_steps > 0 else 1
    between = between if _is_int(between) and between >= 0 else 0
    cuts = [first_cut + round_id * between for round_id in range(cycles)]

    def events(label, device, kind=None, round_id=None):
        values = rows.get(label, {}).get(device, [])
        return [x for x in values if (kind is None or x.get('kind') == kind) and
                (round_id is None or x.get('round') == round_id)]

    def actor_for_worker(replica, rank, owner=('B', 0)):
        if cross_node_tp:
            device = rank // 2 if rank % 2 == 0 else (rank - 1) // 2
            if replica:
                return ('g0_D' if rank % 2 == 0 else 'g0_E'), device
            if rank % 2 == 0:
                return 'g0_A', device
            return _actor_key(owner[1], owner[0]), device
        if replica:
            if rank < 4:
                return 'g0_D', rank
            return 'g0_E', rank - 4
        if rank < 4:
            return 'g0_A', rank
        return _actor_key(owner[1], owner[0]), rank - 4

    # One A0 decision determines the exact active source/candidate matrix each round.
    decisions = {}
    for round_id in range(cycles):
        found = [x for label in rows for device in rows[label] for x in rows[label][device]
                 if x.get('kind') == 'switch_decision' and x.get('round') == round_id]
        check(len(found) == 1, f'unique decision round {round_id}')
        if found:
            decision = found[0]
            check((decision.get('role'), decision.get('rank'), decision.get('replica'),
                   decision.get('device'), decision.get('actor_round')) == ('A', 0, 0, 0, 0),
                  f'A0 decision owner round {round_id}')
            check(decision.get('epoch') == round_id + 1 and _is_int(decision.get('step')),
                  f'decision epoch/cut round {round_id}')
            if boundary_validation == 'runtime':
                check(decision.get('validation') == 'runtime',
                      f'runtime decision validation mode round {round_id}')
            minimum_cut = first_cut if round_id == 0 else cuts[round_id - 1] + between
            check(_is_int(decision.get('step')) and decision.get('step') >= minimum_cut,
                  f'decision cut does not precede requested progress round {round_id}')
            if _is_int(decision.get('step')) and decision.get('step') >= minimum_cut:
                cuts[round_id] = decision['step']
            decisions[round_id] = decision
    check(len([x for label in rows for device in rows[label] for x in rows[label][device]
               if x.get('kind') == 'switch_decision']) == cycles, 'no repeated or late switch decision')
    final_step = cuts[-1] + post_steps
    check(summary.get('final_step') == final_step, 'final step matches actual transaction cuts')

    source = ('B', 0)
    round_state = []
    expected_actors = {_actor_key(0, role) for role in 'ABCDE'}
    for round_id in range(cycles):
        source_role, source_generation = source
        target_role = 'C' if source_role == 'B' else 'B'
        candidate = (target_role, round_id)
        expected_actors.add(_actor_key(candidate[1], candidate[0]))
        decision = decisions.get(round_id, {})
        outcome = decision.get('decision')
        timeout = decision.get('timeout')
        source_recovery = decision.get('source_recovery', False)
        owner = decision.get('owner')
        result_map = decision.get('candidate_results')
        check(outcome in ('replace', 'retain'), f'decision value round {round_id}')
        check(type(timeout) is bool, f'decision timeout boolean round {round_id}')
        check(type(source_recovery) is bool, f'immutable source recovery decision flag round {round_id}')
        check(owner == (target_role if outcome == 'replace' else source_role),
              f'decision owner round {round_id}')
        if not isinstance(result_map, dict):
            result_map = {}
            errors.append(f'candidate results missing round {round_id}')
        normalized_results = {}
        for rank, result in result_map.items():
            try:
                logical_rank = int(rank)
            except (TypeError, ValueError):
                errors.append(f'invalid candidate rank round {round_id}')
                continue
            check(str(logical_rank) == str(rank) and logical_rank in affected_ranks and
                  type(result) is bool, f'invalid candidate result round {round_id}/{rank}')
            normalized_results[logical_rank] = result
        cancel_round = summary.get('cancel_round') == round_id
        if timeout is True:
            check(not normalized_results, f'timed-out round has no completed decision results {round_id}')
        if outcome == 'replace':
            check(timeout is False and not cancel_round and
                  normalized_results == {rank: True for rank in affected_ranks},
                  f'four successful candidate validations required round {round_id}')
            check(not source_recovery or (outcome == 'replace' and not timeout and not cancel_round),
                  f'source recovery requires committed replacement round {round_id}')
            source = candidate
        elif outcome == 'retain':
            check(not source_recovery, f'source recovery cannot retain lost source round {round_id}')
            check(timeout is True or cancel_round or
                  (set(normalized_results) == affected_ranks and not all(normalized_results.values())),
                  f'retain requires timeout/cancel/invalid candidate round {round_id}')
        round_state.append(dict(round=round_id, cut=cuts[round_id], source=(source_role, source_generation),
                                target=target_role, candidate=candidate, decision=outcome,
                                timeout=timeout, source_recovery=source_recovery,
                                results=normalized_results,
                                active_after=source if outcome == 'retain' else candidate))
        if outcome == 'replace':
            source = candidate

    check(set(actors) == expected_actors, 'actor generations match round ownership matrix')

    reported_decisions = summary.get('decisions')
    check(isinstance(reported_decisions, list) and len(reported_decisions) == cycles,
          'summary preserves immutable switch decisions')
    if isinstance(reported_decisions, list):
        for round_id, decision in decisions.items():
            matches = [x for x in reported_decisions
                       if isinstance(x, dict) and x.get('round') == round_id]
            immutable_fields = ('step', 'decision', 'owner', 'epoch', 'timeout',
                                'candidate_results', 'source_recovery')
            if boundary_validation == 'runtime':
                immutable_fields += ('validation',)
            check(len(matches) == 1 and all(matches[0].get(key) == decision.get(key)
                  for key in immutable_fields),
                  f'summary switch decision immutable round {round_id}')

    # Event-level binding and full nine-component signatures.
    for label, devices in rows.items():
        for device, worker_events in devices.items():
            check(bool(worker_events), 'worker event stream ' + label + str(device))
            for event in worker_events:
                check(_is_int(event.get('round')) and 0 <= event['round'] < cycles,
                      'event round range ' + label + str(device))
                for key in ('hashes', 'expected_hashes'):
                    if key in event and event[key] is not None:
                        check(_valid_hashes(event[key]), 'nine hashes ' + label + str(device) + '/' + str(event.get('kind')))

    # Every logical rank has an exact commit history with the fixed batch/sample mapping.
    expected_commits = {}
    for replica in range(2):
        for rank in range(8):
            expected_commits[(replica, rank)] = []
            label, device = actor_for_worker(replica, rank, ('B', 0))
            expected_commits[(replica, rank)].extend(
                (step, 'before', 0, label, device) for step in range(1, cuts[0] + 1))
            for state in round_state:
                if state['round'] > 0:
                    requested = cuts[state['round'] - 1] + between
                    label, device = actor_for_worker(replica, rank, state['source'])
                    expected_commits[(replica, rank)].extend(
                        (step, 'before', state['round'], label, device)
                        for step in range(requested + 1, cuts[state['round']] + 1))
                end = cuts[state['round']] + between if state['round'] + 1 < cycles else final_step
                label, device = actor_for_worker(replica, rank, state['active_after'])
                expected_commits[(replica, rank)].extend(
                    (step, 'after', state['round'], label, device)
                    for step in range(cuts[state['round']] + 1, end + 1))

    worker_commits = {(r, k): [] for r in range(2) for k in range(8)}
    for (replica, rank), expected in expected_commits.items():
        # Collect from all actors, then compare event ownership and exact phase schedule.
        for actor_label, devices in rows.items():
            for actor_device, worker_events in devices.items():
                for commit in (x for x in worker_events if x.get('kind') == 'commit'):
                    if (commit.get('replica'), commit.get('rank')) == (replica, rank):
                        worker_commits[(replica, rank)].append((actor_label, actor_device, commit))
        got = sorted(worker_commits[(replica, rank)], key=lambda item: (item[2].get('step', -1), item[2].get('round', -1)))
        expected_rows = sorted(expected, key=lambda item: item[0])
        check([x[2].get('step') for x in got] == list(range(1, final_step + 1)),
              f'commit coverage/duplicates replica{replica}/rank{rank}')
        check(len(got) == final_step, f'unique commit count replica{replica}/rank{rank}')
        check([(x[2].get('step'), x[2].get('phase'), x[2].get('round'), x[0], x[1]) for x in got] ==
              [(x[0], x[1], x[2], x[3], x[4]) for x in expected_rows],
              f'commit owner/phase schedule replica{replica}/rank{rank}')
        for actor_label, actor_device, commit in got:
            step = commit.get('step')
            check(_is_int(step) and commit.get('phase') in ('before', 'after'),
                  f'integer commit step/phase replica{replica}/rank{rank}')
            exp_samples = ([ (step - 1) * microbatches * 4 + micro * 4 + replica * 2 + (rank % 4) // 2
                             for micro in range(microbatches)] if _is_int(step) and step > 0 else None)
            check(commit.get('sample_ids') == exp_samples and commit.get('finite') is True,
                  f'sample IDs/finite replica{replica}/rank{rank}/step{step}')
            if commit.get('phase') == 'after':
                check(commit.get('global_activation_observed') is True,
                      f'post-switch activation marker {actor_label}{actor_device}/step{step}')
                round_id = commit.get('round')
                role = actors.get(actor_label, {}).get('role')
                if role not in ('D', 'E') and _is_int(round_id):
                    markers = [x for x in rows.get(actor_label, {}).get(actor_device, [])
                               if x.get('kind') == 'activated' and x.get('round') == round_id]
                    worker_events = rows.get(actor_label, {}).get(actor_device, [])
                    check(len(markers) == 1 and worker_events.index(markers[0]) < worker_events.index(commit),
                          f'post-switch commit follows activation {actor_label}{actor_device}/step{step}')

    # Manager begin/reduce/commit records must match each worker's planned commits exactly.
    retention_identity_by_actor = {}
    for label, devices in rows.items():
        role = actors[label]['role']
        replica = int(role in ('D', 'E'))
        for device, worker_events in devices.items():
            commits = sorted((x for x in worker_events if x.get('kind') == 'commit'),
                             key=lambda x: x.get('step', -1))
            expected_pairs = []
            for commit in commits:
                step = commit.get('step')
                epoch = sum(1 for cut in cuts if _is_int(step) and cut < step)
                expected_pairs.append((epoch, step - 1 if _is_int(step) else None))
            def pairs(kind):
                return [(x.get('outer_epoch'), x.get('step')) for x in worker_events if x.get('kind') == kind]
            check(pairs('outer_begin') == expected_pairs,
                  f'fixed two-participant outer begin {label}{device}')
            check(all(x.get('participants') == 2 and x.get('participating_rank') == replica
                      and _is_int(x.get('participants')) and _is_int(x.get('participating_rank'))
                      for x in worker_events if x.get('kind') == 'outer_begin'),
                  f'fixed participant rank {label}{device}')
            for kind in ('outer_quorum_requested', 'outer_quorum_wait'):
                check(pairs(kind) == expected_pairs, f'async quorum {kind} coverage {label}{device}')
            waits = [x for x in worker_events if x.get('kind') == 'outer_quorum_wait']
            check(all(_finite_nonnegative(x.get('seconds')) for x in waits),
                  f'async quorum duration {label}{device}')
            outer = [x for x in worker_events if x.get('kind') == 'outer_commit']
            check([(x.get('outer_epoch'), x.get('step')) for x in outer] == expected_pairs,
                  f'outer commit coverage {label}{device}')
            commit_by_step = {x.get('step'): x for x in commits}
            check(len(commit_by_step) == len(commits), f'duplicate training steps {label}{device}')
            for entry in outer:
                step0 = entry.get('step')
                commit = commit_by_step.get(step0 + 1) if _is_int(step0) else None
                check(entry.get('global_fenced') is True and commit is not None and
                      _is_int(entry.get('committed_step')) and _is_int(entry.get('batches_committed')) and
                      entry.get('committed_step') == step0 + 1 and
                      entry.get('batches_committed') == 2 * (step0 + 1),
                      f'global fenced Manager progress {label}{device}/step{step0}')

    # Round-specific ownership, transfer, retention and peer-overlap proof.
    peer_waits = {}
    expected_source_stops = []
    for state in round_state:
        r, cut = state['round'], state['cut']
        source_role, source_generation = state['source']
        target_role = state['target']
        candidate_label = _actor_key(r, target_role)
        source_label = _actor_key(source_generation, source_role)
        decision = decisions.get(r, {})
        replace = state['decision'] == 'replace'
        recovery = state['source_recovery']
        recovery_events = [x for label in rows for device in rows[label] for x in rows[label][device]
                           if x.get('kind') == 'source_recovery_selected' and x.get('round') == r]
        check(len(recovery_events) == (1 if recovery else 0),
              f'unique peer recovery selection round {r}')
        if recovery:
            selected = recovery_events[0] if recovery_events else {}
            check((selected.get('role'), selected.get('rank'), selected.get('replica'),
                   selected.get('device'), selected.get('actor_round')) == ('A', 0, 0, 0, 0) and
                  selected.get('step') == cut and selected.get('source') == source_role and
                  selected.get('target') == target_role,
                  f'peer recovery selection binds source/target round {r}')
            a0_events = rows.get('g0_A', {}).get(0, [])
            switch = [x for x in a0_events if x.get('kind') == 'switch_decision' and x.get('round') == r]
            if selected and switch:
                check(a0_events.index(selected) < a0_events.index(switch[0]),
                      f'peer recovery selected before immutable decision round {r}')
            expected_source_stops.append(dict(round=r, generation=source_generation,
                role=source_role, unit_stopped=True))
        active_labels = {'g0_A', 'g0_D', 'g0_E', source_label}
        boundary_rows = []
        for label, devices in rows.items():
            for device, worker_events in devices.items():
                for event in worker_events:
                    if event.get('kind') == 'boundary_selected' and event.get('round') == r:
                        boundary_rows.append((label, device, event))
        expected_boundary = set()
        for label in active_labels:
            if label not in rows:
                continue
            expected_boundary.update((label, device) for device in range(4))
        check({(label, device) for label, device, _ in boundary_rows} == expected_boundary and
              len(boundary_rows) == 16 and
              {(event.get('replica'), event.get('rank')) for _, _, event in boundary_rows} ==
              {(replica, rank) for replica in range(2) for rank in range(8)},
              f'16 active boundary selections round {r}')
        check(all(event.get('step') == cut for _, _, event in boundary_rows),
              f'common boundary cut round {r}')
        requested_cut = first_cut if r == 0 else cuts[r - 1] + between
        check(all(event.get('requested_cut') == requested_cut for _, _, event in boundary_rows),
              f'boundary respects requested progress round {r}')

        for label in ('g0_A', 'g0_D', 'g0_E', source_label):
            for device in range(4):
                boundary = events(label, device, 'outer_retention_boundary', r)
                frozen = events(label, device, 'freeze', r)
                check(len(boundary) == len(frozen) == 1 and
                      boundary[0].get('step') == cut and frozen[0].get('step') == cut and
                      frozen[0].get('identity_preserved') is True,
                      f'active freeze/retention boundary {label}{device}/round{r}')
                if label in ('g0_A', 'g0_D', 'g0_E') or not replace:
                    retained = events(label, device, 'outer_retained', r)
                    check(len(boundary) == len(retained) == 1,
                          f'retained Manager event {label}{device}/round{r}')
                    if boundary and retained:
                        before, after = boundary[0], retained[0]
                        identity = before.get('identity')
                        training = before.get('training_identity')
                        required_identity = {'manager', 'server', 'client', 'executor', 'pg', 'service_epoch'}
                        required_training = {'model', 'optimizer', 'ddp', 'parameters', 'adam', 'buckets'}
                        check(isinstance(identity, dict) and set(identity) == required_identity and
                              all(identity.get(k) for k in ('manager', 'server', 'client', 'executor', 'pg')) and
                              identity.get('service_epoch') == 0 and
                              identity == after.get('identity'),
                              f'Manager identity retention {label}{device}/round{r}')
                        check(isinstance(training, dict) and set(training) == required_training and
                              all(training.get(k) for k in ('model', 'optimizer', 'ddp', 'parameters', 'adam', 'buckets')) and
                              training == after.get('training_identity'),
                              f'training storage retention {label}{device}/round{r}')
                        identity_key = (label, device)
                        previous = retention_identity_by_actor.get(identity_key)
                        if previous is not None:
                            check(identity == previous[0] and training == previous[1],
                                  f'identity stable across rounds {label}{device}/round{r}')
                        retention_identity_by_actor[identity_key] = (identity, training)
                        progress = {'step': cut, 'batches_committed': 2 * cut}
                        before_state, after_state = before.get('manager_state'), after.get('manager_state')
                        check(isinstance(before_state, dict) and isinstance(after_state, dict) and
                              all(_is_int(v) for v in before_state.values()) and
                              all(_is_int(v) for v in after_state.values()) and
                              before_state == progress and after_state == progress,
                              f'retained Manager progress {label}{device}/round{r}')
                        rank = retained[0].get('rank')
                        check(retained[0].get('channel_reconnected') is (rank in affected_ranks),
                              f'affected channel reconnect scope {label}{device}/round{r}')
                else:
                    check(not events(label, device, 'outer_retained', r),
                          f'retired source not retained round {r}')

        for role in ('A', source_role):
            label = 'g0_A' if role == 'A' else source_label
            for device in range(4):
                retained_groups = events(label, device, 'inner_groups_retained_after_cancel', r)
                check(len(retained_groups) == (0 if replace else 1),
                      f'inner groups cancel boundary {label}{device}/round{r}')
        for role in ('D', 'E'):
            label = _actor_key(0, role)
            for device in range(4):
                frozen = events(label, device, 'freeze', r)
                peer_ready = events(label, device, 'peer_compute_ready', r)
                groups = events(label, device, 'peer_inner_groups_retained', r)
                check(len(peer_ready) == len(groups) == 1 and
                      peer_ready[0].get('step') == cut and peer_ready[0].get('identity_preserved') is True and
                      len(frozen) == 1 and peer_ready[0].get('hashes') == frozen[0].get('hashes') and
                      len(groups) == 1 and groups[0].get('identity_preserved') is True,
                      f'peer state/groups retained {label}{device}/round{r}')
                compute = events(label, device, 'peer_compute_begin', r)
                native_ready = events(label, device, 'peer_native_gradient_ready', r)
                outer_ready = events(label, device, 'peer_outer_gradient_ready', r)
                wait = [x for x in events(label, device, 'outer_quorum_wait', r)
                        if x.get('outer_epoch') == r + 1 and x.get('step') == cut]
                request = [x for x in events(label, device, 'outer_quorum_requested', r)
                           if x.get('outer_epoch') == r + 1 and x.get('step') == cut]
                begin = [x for x in events(label, device, 'outer_begin', r)
                         if x.get('outer_epoch') == r + 1 and x.get('step') == cut]
                check(len(compute) == len(native_ready) == len(outer_ready) == len(wait) ==
                      len(request) == len(begin) == 1 and
                      compute[0].get('step') == native_ready[0].get('step') ==
                      outer_ready[0].get('step') == cut and
                      _finite_nonnegative(outer_ready[0].get('seconds')) and
                      _finite_nonnegative(wait[0].get('seconds')) and
                      outer_ready[0]['seconds'] + 1e-6 >= wait[0]['seconds'],
                      f'peer outer-gradient wait evidence {label}{device}/round{r}')
                if all((compute, native_ready, wait, begin, outer_ready)):
                    worker_events = rows[label][device]
                    order = [worker_events.index(x) for x in
                             (request[0], compute[0], native_ready[0], wait[0], begin[0], outer_ready[0])]
                    check(order == sorted(order) and len(set(order)) == len(order),
                          f'peer compute/quorum causal order {label}{device}/round{r}')
                    peer_waits[f'round{r}/{role}/rank{outer_ready[0].get("rank")}'] = outer_ready[0]['seconds']

        # The candidate is prepared/transferred every round, but it can run only after a full replace.
        for device in range(4):
            candidate_events = events(candidate_label, device)
            prepared = events(candidate_label, device, 'candidate_prepared', r)
            received = events(candidate_label, device, 'state_received', r)
            source_sent = events(source_label, device, 'state_sent', r)
            source_freeze = events(source_label, device, 'freeze', r)
            peer_label = 'g0_E'
            peer_freeze = events(peer_label, device, 'freeze', r)
            peer_auth = events(peer_label, device, 'peer_state_authorized', r)
            peer_sent = events(peer_label, device, 'state_sent', r)
            if recovery:
                sent = peer_sent
                source_stalled = events(source_label, device, 'source_stalled', r)
                check(len(prepared) == len(received) == len(source_freeze) == len(peer_freeze) ==
                      len(peer_auth) == len(peer_sent) == len(source_stalled) == 1 and not source_sent and
                      source_freeze[0].get('step') == peer_freeze[0].get('step') == cut and
                      peer_auth[0].get('step') == peer_sent[0].get('step') ==
                      received[0].get('step') == cut and
                      peer_auth[0].get('hashes') == peer_auth[0].get('expected_hashes') ==
                      source_freeze[0].get('hashes') == peer_freeze[0].get('hashes') and
                      (boundary_validation != 'runtime' or peer_auth[0].get('version_checked') is True) and
                      peer_sent[0].get('hashes') == peer_freeze[0].get('hashes') and
                      received[0].get('hashes') == peer_sent[0].get('hashes') and
                      peer_sent[0].get('packet_source') == received[0].get('packet_source') == 'peer' and
                      _state_channel_ok(peer_sent[0], received[0]),
                      f'peer recovery state matches source/E freezes {source_label}/E/{candidate_label}/device{device}/round{r}')
                if peer_auth and peer_sent and peer_freeze:
                    peer_events = rows[peer_label][device]
                    check(peer_events.index(peer_freeze[0]) < peer_events.index(peer_auth[0]) <
                          peer_events.index(peer_sent[0]),
                          f'peer freeze and authorization precede relay {peer_label}{device}/round{r}')
                if prepared and received:
                    candidate_events_for_device = rows[candidate_label][device]
                    check(candidate_events_for_device.index(prepared[0]) <
                          candidate_events_for_device.index(received[0]),
                          f'candidate prepares before peer state arrival {candidate_label}{device}/round{r}')
            else:
                sent = source_sent
                check(len(prepared) == len(received) == len(sent) == len(source_freeze) == 1 and
                      not peer_auth and not peer_sent and
                      received[0].get('step') == sent[0].get('step') == cut and
                      sent[0].get('hashes') == source_freeze[0].get('hashes') and
                      sent[0].get('hashes') == received[0].get('hashes') and
                      sent[0].get('packet_source') == received[0].get('packet_source') == 'source' and
                      _state_channel_ok(sent[0], received[0]),
                      f'source-target state channel {source_label}/{candidate_label}/device{device}/round{r}')
                if prepared and received:
                    candidate_events_for_device = rows[candidate_label][device]
                    check(candidate_events_for_device.index(prepared[0]) <
                          candidate_events_for_device.index(received[0]),
                          f'candidate prepares before source state arrival {candidate_label}{device}/round{r}')
            validated = events(candidate_label, device, 'candidate_validated', r)
            rejected = events(candidate_label, device, 'candidate_rejected', r)
            rank = Plan.logical_rank({'D': 'A', 'E': 'B'}.get(target_role, target_role),
                                     device, cross_node_tp)
            check(len(validated) + len(rejected) <= 1,
                  f'unique candidate validation {candidate_label}{device}/round{r}')
            if validated:
                if boundary_validation == 'runtime':
                    check(len(sent) == 1 and validated[0].get('step') == cut and
                          _runtime_readiness_ok(validated[0], epoch=r + 1, rank=rank,
                                                cut=cut, target=target_role),
                          f'candidate runtime readiness matches boundary {candidate_label}{device}/round{r}')
                else:
                    check(len(sent) == 1 and validated[0].get('step') == cut and
                          validated[0].get('hashes') == validated[0].get('expected_hashes') == sent[0].get('hashes'),
                          f'candidate validation matches source {candidate_label}{device}/round{r}')
            if rejected:
                if boundary_validation == 'runtime':
                    check(len(sent) == 1 and rejected[0].get('step') == cut and
                          rejected[0].get('hashes') is None and
                          rejected[0].get('expected_hashes') is None and
                          isinstance(rejected[0].get('reason'), str) and bool(rejected[0]['reason']),
                          f'rejected runtime candidate lacks a clear reason {candidate_label}{device}/round{r}')
                else:
                    check(len(sent) == 1 and rejected[0].get('step') == cut and
                          _valid_hashes(rejected[0].get('hashes')) and
                          rejected[0].get('expected_hashes') == sent[0].get('hashes') and
                          rejected[0].get('hashes') != rejected[0].get('expected_hashes') and
                          isinstance(rejected[0].get('reason'), str) and bool(rejected[0]['reason']),
                          f'rejected candidate state is incomplete or mismatched {candidate_label}{device}/round{r}')
            if not state['timeout']:
                expected_result = state['results'].get(rank)
                check(expected_result is (len(validated) == 1),
                      f'candidate result matches event rank{rank}/round{r}')
                check(len(validated) + len(rejected) == 1,
                      f'candidate reports all ranks rank{rank}/round{r}')
            candidate_activated = events(candidate_label, device, 'candidate_activated', r)
            activated = events(candidate_label, device, 'activated', r)
            if replace:
                # Earlier emitters recorded full state at the following global
                # activation event, rather than repeating it at local admission.
                admission=dict(activated[0]) if activated else {}
                if candidate_activated:admission.update(candidate_activated[0])
                readiness_ok = (boundary_validation != 'runtime' or
                    (len(candidate_activated) == 1 and
                     _runtime_readiness_ok(candidate_activated[0], epoch=r + 1, rank=rank,
                                           cut=cut, target=target_role)))
                check(len(sent) == len(validated) == len(candidate_activated) == len(activated) == 1 and
                      not rejected and candidate_activated[0].get('step') == cut and
                      readiness_ok and
                      admission.get('decision') == 'replace' and
                      admission.get('owner') == target_role and
                      admission.get('hashes') == sent[0].get('hashes') and
                      candidate_activated[0].get('channel_reconnected') is
                          (candidate_activated[0].get('rank') in affected_ranks) and
                      admission.get('identity_preserved') is True,
                      f'validated candidate activation {candidate_label}{device}/round{r}')
                candidate_sequence = candidate_events
                kinds = [x.get('kind') for x in candidate_sequence]
                if 'candidate_validated' in kinds and 'candidate_activated' in kinds:
                    check(kinds.index('candidate_validated') < kinds.index('candidate_activated'),
                          f'candidate validation precedes activation {candidate_label}{device}/round{r}')
                if 'outer_ready' in kinds and 'candidate_activated' in kinds and 'activated' in kinds:
                    check(kinds.index('outer_ready') < kinds.index('candidate_activated') < kinds.index('activated'),
                          f'candidate activation follows Manager readiness {candidate_label}{device}/round{r}')
                ready = events(candidate_label, device, 'outer_ready', r)
                check(len(ready) == 1 and ready[0].get('outer_epoch') == r + 1,
                      f'candidate Manager ready only after replacement {candidate_label}{device}/round{r}')
            else:
                cancelled = events(candidate_label, device, 'candidate_cancelled', r)
                standby = events(candidate_label, device, 'standby_done', r)
                check(len(cancelled) == len(standby) == 1 and
                      not candidate_activated and not activated and
                      not events(candidate_label, device, 'outer_ready', r) and
                      not events(candidate_label, device, 'outer_begin', r) and
                      not events(candidate_label, device, 'outer_commit', r) and
                      not events(candidate_label, device, 'commit', r),
                      f'retained candidate stays standby {candidate_label}{device}/round{r}')
                if cancelled and standby:
                    kinds = [x.get('kind') for x in candidate_events]
                    check(kinds.index('candidate_cancelled') < kinds.index('standby_done'),
                          f'candidate cancellation before standby done {candidate_label}{device}/round{r}')

            if replace:
                old_rejected = events(source_label, device, 'old_owner_rejected', r)
                retired = events(source_label, device, 'retired', r)
                if recovery:
                    stalled = events(source_label, device, 'source_stalled', r)
                    check(len(old_rejected) == len(stalled) == 1 and not retired and
                          not events(source_label, device, 'state_sent', r) and
                          old_rejected[0].get('step') == stalled[0].get('step') == cut and
                          not events(source_label, device, 'activated', r) and
                          not any(x.get('step', 0) > cut for x in events(source_label, device, 'commit')),
                          f'failed source fenced, stalled and stopped {source_label}{device}/round{r}')
                    if old_rejected and stalled:
                        source_events = rows[source_label][device]
                        check(source_events.index(old_rejected[0]) < source_events.index(stalled[0]),
                              f'old source rejection precedes controlled stall {source_label}{device}/round{r}')
                else:
                    check(len(old_rejected) == len(retired) == 1 and
                          not events(source_label, device, 'source_stalled', r) and
                          old_rejected[0].get('step') == retired[0].get('step') == cut and
                          len(events(source_label, device, 'state_sent', r)) == 1 and
                          not events(source_label, device, 'activated', r) and
                          not any(x.get('step', 0) > cut for x in events(source_label, device, 'commit')),
                          f'old source fenced and retired {source_label}{device}/round{r}')
                    if old_rejected and retired:
                        source_events = rows[source_label][device]
                        check(source_events.index(old_rejected[0]) < source_events.index(retired[0]),
                              f'old source rejection precedes retirement {source_label}{device}/round{r}')
            else:
                check(not events(source_label, device, 'old_owner_rejected', r) and
                      not events(source_label, device, 'retired', r),
                      f'retained source remains authorized {source_label}{device}/round{r}')

        for device in range(4):
            active_labels_for_round = ['g0_A'] + ([source_label] if not replace else [])
            for label in active_labels_for_round:
                activated = events(label, device, 'activated', r)
                check(len(activated) == 1, f'active actor activation marker {label}{device}/round{r}')
            for label in ('g0_D', 'g0_E'):
                check(not events(label, device, 'activated', r),
                      f'peer worker uses compute-ready marker {label}{device}/round{r}')
        a0_activated = events('g0_A', 0, 'activated', r)
        check(len(a0_activated) == 1 and a0_activated[0].get('peer_compute_started') is True,
              f'A0 observes peer compute before activation round {r}')

    reported_stops = summary.get('source_recovery_stops', [])
    check(isinstance(reported_stops, list), 'source recovery stop report is a list')
    if isinstance(reported_stops, list):
        normalized_stops = []
        for stop in reported_stops:
            if not isinstance(stop, dict):
                errors.append('source recovery stop report contains a non-object')
                continue
            normalized_stops.append({key: stop.get(key) for key in
                                     ('round', 'generation', 'role', 'unit_stopped')})
            pre, post = stop.get('pre'), stop.get('post')
            check({'round', 'generation', 'role', 'unit_stopped'}.issubset(stop) and
                  type(stop.get('round')) is int and type(stop.get('generation')) is int and
                  stop.get('role') in ('B', 'C') and stop.get('unit_stopped') is True,
                  'source recovery stop record binds exact old source and confirms stop')
            check(type(stop.get('stop_returncode')) is int and stop.get('stop_returncode') == 0 and
                  isinstance(pre, dict) and pre.get('LoadState') == 'loaded' and
                  pre.get('ActiveState') == 'active' and pre.get('SubState') == 'running' and
                  type(pre.get('MainPID')) is int and pre.get('MainPID') > 0 and
                  isinstance(post, dict) and post.get('ActiveState') == 'inactive' and
                  post.get('SubState') == 'dead' and type(post.get('MainPID')) is int and
                  post.get('MainPID') == 0,
                  'source recovery unit was running and stopped cleanly')
        expected_source_stops.sort(key=lambda x: x['round'])
        normalized_stops.sort(key=lambda x: x.get('round') if type(x.get('round')) is int else -1)
        check(normalized_stops == expected_source_stops,
              'controller reports exactly the stopped old source units')
    if expected_source_stops:
        scope = summary.get('source_recovery_scope')
        check(isinstance(scope, dict) and
              scope.get('failure_window') == 'after_freeze_before_ready' and
              scope.get('transport_failure') == 'fail_stop',
              'source recovery scope states freeze/ready boundary and transport fail-stop limit')

    # Readiness belongs only to initial active Managers and actually replaced candidates.
    for label, descriptor in actors.items():
        role = descriptor['role']
        for device in range(4):
            ready = [x for x in rows[label][device] if x.get('kind') == 'outer_ready']
            if role in ('A', 'D', 'E') or label == 'g0_B':
                expected_epochs = [0]
            else:
                activated_rounds = [state['round'] for state in round_state
                                    if state['candidate'] == (role, descriptor['generation']) and
                                    state['decision'] == 'replace']
                expected_epochs = [r + 1 for r in activated_rounds]
            check([x.get('outer_epoch') for x in ready] == expected_epochs,
                  f'outer Manager readiness generations {label}{device}')

    # Reference mode is checked from the independent manifest and every raw signature tuple.
    external = bool(summary.get('reference_run'))
    if external:
        reference_path = root / 'reference_signatures.json'
        reference = None
        try:
            raw = reference_path.read_bytes()
            check(hashlib.sha256(raw).hexdigest() == summary.get('reference_sha256'),
                  'reference manifest SHA256')
            from shift_full.v4_reference import compare_signature, validate_signatures
            reference = validate_signatures(json.loads(raw))
        except (OSError, ValueError, KeyError) as exc:
            errors.append('independent reference invalid: ' + str(exc))
        check(summary.get('oracle_exact') is True, 'summary reference comparison result')
        for label, descriptor in actors.items():
            role = descriptor['role']
            for device in range(4):
                replica = int(role in ('D', 'E'))
                rank = Plan.logical_rank({'D': 'A', 'E': 'B'}.get(role, role), device, cross_node_tp)
                worker_events = rows[label][device]
                initial = [x for x in worker_events if x.get('kind') == 'reference_initial']
                if descriptor['generation'] == 0 and role in ('A', 'B', 'D', 'E'):
                    check(len(initial) == 1 and initial[0].get('step') == 0,
                          f'initial reference state {label}{device}')
                    if reference is not None and initial:
                        try:
                            exact = compare_signature(reference, replica=replica, rank=rank, step=0,
                                hashes=initial[0].get('hashes'), samples=initial[0].get('sample_ids'),
                                manager=initial[0].get('outer_manager'))
                            check(exact and initial[0].get('sample_ids') == [],
                                  f'initial reference signature {label}{device}')
                        except (ValueError, KeyError) as exc:
                            errors.append('initial reference coverage ' + label + str(device) + ': ' + str(exc))
                else:
                    check(not initial, f'candidate has no pre-cut reference initial {label}{device}')
                commits = [x for x in worker_events if x.get('kind') == 'commit']
                comparisons = [x for x in worker_events if x.get('kind') == 'reference_comparison']
                check([x.get('step') for x in comparisons] == [x.get('step') for x in commits],
                      f'reference comparison coverage {label}{device}')
                for comparison in comparisons:
                    if reference is None:
                        continue
                    try:
                        exact = compare_signature(reference, replica=replica, rank=rank,
                            step=comparison['step'], hashes=comparison.get('hashes'),
                            samples=comparison.get('sample_ids'), manager=comparison.get('outer_manager'))
                        check(exact is True and comparison.get('exact') is True,
                              f'independent signature comparison {label}{device}/step{comparison.get("step")}')
                    except (ValueError, KeyError) as exc:
                        errors.append('reference comparison coverage ' + label + str(device) + ': ' + str(exc))
    else:
        check(not (root / 'reference_signatures.json').exists() or not summary.get('reference_sha256'),
              'unrequested reference manifest')

    # Final ownership, optional terminal reference, all sixteen done records,
    # identity and cross-replica state equality.
    if round_state:
        final_owner = round_state[-1]['active_after']
    else:
        final_owner = ('B', 0)
    expected_done = {_actor_key(0, 'A'), _actor_key(0, 'D'), _actor_key(0, 'E'),
                     _actor_key(final_owner[1], final_owner[0])}
    final_reference_run = summary.get('final_reference_run')
    final_reference_sha256 = summary.get('final_reference_sha256')
    final_reference_enabled = final_reference_run is not None or final_reference_sha256 is not None
    final_reference = None
    final_reference_path = root / 'final_reference_signatures.json'
    check((final_reference_run is None) == (final_reference_sha256 is None),
          'final reference run/SHA are paired')
    if final_reference_enabled:
        check(isinstance(final_reference_run, str) and bool(final_reference_run),
              'final reference run name')
        check(_valid_sha256(final_reference_sha256), 'final reference SHA256 format')
        try:
            raw = final_reference_path.read_bytes()
            check(hashlib.sha256(raw).hexdigest() == final_reference_sha256,
                  'final reference manifest SHA256')
            from shift_full.v4_reference import validate_signatures
            final_reference = validate_signatures(json.loads(raw))
            expected_final_reference_run = (final_reference_run if final_reference_run.startswith('v4_')
                                            else 'v4_' + final_reference_run)
            check(final_reference.get('run') == expected_final_reference_run,
                  'final reference run name matches manifest')
            check(final_reference.get('steps', 0) >= final_step,
                  'final reference covers final step')
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append('independent final reference invalid: ' + str(exc))
    else:
        check(not final_reference_path.exists(), 'unrequested final reference manifest')
    done_by_worker = {}
    done_total = []
    for label, devices in rows.items():
        for device, worker_events in devices.items():
            done = [x for x in worker_events if x.get('kind') == 'done']
            role = actors[label]['role']
            replica = int(role in ('D', 'E'))
            rank = Plan.logical_rank({'D': 'A', 'E': 'B'}.get(role, role), device, cross_node_tp)
            if label in expected_done:
                check(len(done) == 1 and done[0].get('step') == final_step and
                      done[0].get('identity_preserved') is True and _valid_hashes(done[0].get('hashes')),
                      f'final done/identity {label}{device}')
                if done:
                    done_by_worker[(replica, rank)] = done[0]
                    done_total.append(done[0])
                    if boundary_validation == 'runtime':
                        ready_markers = [x for x in worker_events if x.get('kind') == 'final_audit_ready']
                        begin_markers = [x for x in worker_events if x.get('kind') == 'final_audit_begin']
                        commits = [x for x in worker_events if x.get('kind') == 'commit']
                        hash_calls = [x for x in worker_events if x.get('kind') == 'state_hash']
                        check(len(ready_markers) == len(begin_markers) == 1 and
                              ready_markers[0].get('step') == begin_markers[0].get('step') == final_step,
                              f'final audit markers cover active worker {label}{device}')
                        check(len(hash_calls) == 1 and _is_int(done[0].get('state_hash_calls')) and
                              done[0].get('state_hash_calls') == 1 and
                              hash_calls[0].get('step') == final_step and
                              hash_calls[0].get('final_audit') is True and
                              _is_int(hash_calls[0].get('calls')) and hash_calls[0].get('calls') == 1,
                              f'one final state hash only {label}{device}')
                        if len(ready_markers) == len(begin_markers) == len(hash_calls) == 1 and commits:
                            check(worker_events.index(commits[-1]) < worker_events.index(ready_markers[0]) <
                                  worker_events.index(begin_markers[0]) < worker_events.index(hash_calls[0]) <
                                  worker_events.index(done[0]),
                                  f'final audit markers follow final commit before done {label}{device}')
                        else:
                            check(False, f'final audit follows final commit {label}{device}')
                    final_comparisons = [x for x in worker_events
                                         if x.get('kind') == 'final_reference_comparison']
                    if final_reference_enabled:
                        check(len(final_comparisons) == 1 and
                              final_comparisons[0].get('step') == final_step and
                              final_comparisons[0].get('exact') is True and
                              _valid_hashes(final_comparisons[0].get('hashes')) and
                              final_comparisons[0].get('hashes') == done[0].get('hashes'),
                              f'final reference event matches done {label}{device}')
                        if final_reference is not None:
                            expected_final = final_reference['workers'].get(f'{replica}/{rank}', {}).get(str(final_step))
                            check(expected_final is not None and
                                  done[0].get('hashes') == expected_final.get('hashes'),
                                  f'final reference signature replica{replica}/rank{rank}/step{final_step}')
                            if (len(final_comparisons) == 1 and expected_final is not None and
                                    boundary_validation == 'runtime'):
                                begin_markers = [x for x in worker_events if x.get('kind') == 'final_audit_begin']
                                check(len(begin_markers) == 1 and
                                      worker_events.index(begin_markers[0]) < worker_events.index(final_comparisons[0]) <
                                      worker_events.index(done[0]),
                                      f'final reference comparison follows audit begin {label}{device}')
                    else:
                        check(not final_comparisons,
                              f'unrequested final reference event {label}{device}')
                    comparison = done[0].get('comparison')
                    if isinstance(comparison, dict):
                        exact = comparison.get('exact')
                        if external or exact is not None:
                            check(isinstance(exact, dict) and set(exact) == REFERENCE_HASH_KEYS and
                                  all(type(v) is bool and v for v in exact.values()) and
                                  comparison.get('trajectory_exact') is True and
                                  comparison.get('manager_exact') is True,
                                  f'final nine-hash/manager comparison {label}{device}')
                    elif external:
                        check(False, f'final reference comparison missing {label}{device}')
            else:
                check(not done, f'inactive actor emitted done {label}{device}')
                if boundary_validation == 'runtime':
                    check(not any(x.get('kind') in ('final_audit_ready', 'final_audit_begin')
                                  for x in worker_events),
                          f'inactive actor emitted final audit marker {label}{device}')
                    check(not any(x.get('kind') == 'state_hash' for x in worker_events),
                          f'inactive actor emitted state hash {label}{device}')
    check(len(done_total) == 16 and set(done_by_worker) == {(r, k) for r in range(2) for k in range(8)},
          'exact sixteen-worker final coverage')
    for rank in range(8):
        left = done_by_worker.get((0, rank), {}).get('hashes', {})
        right = done_by_worker.get((1, rank), {}).get('hashes', {})
        check(all(left.get(key) == right.get(key) for key in ('model', 'optimizer', 'scheduler')),
              f'final DP model/optimizer/scheduler equality rank{rank}')

    # The three monitored nodes must return to the same eight company PIDs with no extras.
    resources_clean = False
    try:
        before = json.loads((root / 'resources_before.json').read_text())
        after = json.loads((root / 'resources_after.json').read_text())
        nodes = {'node0', 'node1', 'node3'}
        resources_clean = (isinstance(before, dict) and isinstance(after, dict) and
            set(before) == set(after) == nodes and
            all(isinstance(before[n].get('company'), dict) and
                len(before[n]['company']) == 8 and before[n]['company'] == after[n].get('company') and
                not before[n].get('extra_gpu') and not after[n].get('extra_gpu') for n in nodes))
    except (OSError, ValueError, AttributeError):
        resources_clean = False
    check(resources_clean, 'three-node company PID/resource audit')

    try:
        auditor_sha = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    except OSError:
        auditor_sha = None
    raw_hashes = _raw_hashes(root)
    return dict(pass_all=not errors, errors=errors, resources_clean=resources_clean,
        transactional_switch=True, boundary_validation=boundary_validation,
        cycles=cycles, cuts=cuts, final_step=final_step,
        decisions=[decisions[r].get('decision') for r in sorted(decisions)],
        logical_workers=16, commit_count=sum(len(v) for v in worker_commits.values()),
        peer_outer_gradient_ready_seconds=peer_waits, reference_run=summary.get('reference_run'),
        actual_auditor_sha256=auditor_sha, raw_sha256=raw_hashes)


def summarize(root):
    root = Path(root)
    audit = verify(root)
    if not audit['pass_all']:
        raise ValueError(audit['errors'])
    summary = json.loads((root / 'summary.json').read_text())
    all_events=[json.loads(line) for path in root.glob('g*_*.jsonl')
                for line in path.read_text().splitlines() if line.strip()]
    final_signatures={f'{x["replica"]}/{x["rank"]}':x['hashes'] for x in all_events if x['kind']=='done'}
    return dict(name=summary.get('name'), transactional_switch=True,
        boundary_validation=audit['boundary_validation'],
        cycles=audit['cycles'], cuts=audit['cuts'], final_step=audit['final_step'],
        decisions=audit['decisions'], functional_pass=True, resources_clean=audit['resources_clean'],
        cleanup_errors=summary.get('cleanup_errors'), reference_run=summary.get('reference_run'),
        reference_sha256=summary.get('reference_sha256'),
        final_reference_run=summary.get('final_reference_run'),
        final_reference_sha256=summary.get('final_reference_sha256'),
        plan=summary.get('plan'),topology=summary.get('topology'),cross_node_tp=bool(summary.get('cross_node_tp')),
        oracle_exact=summary.get('oracle_exact'),source_sha256=summary.get('source_sha256'),
        fail_restore_rank=summary.get('fail_restore_rank'),stall_restore_rank=summary.get('stall_restore_rank'),
        cancel_round=summary.get('cancel_round'),restore_deadline=summary.get('restore_deadline'),fault_round=summary.get('fault_round'),
        source_stall_round=summary.get('source_stall_round'),source_deadline=summary.get('source_deadline'),
        source_recovery_scope=summary.get('source_recovery_scope'),source_recovery_stops=summary.get('source_recovery_stops',[]),
        switch_decisions=summary.get('decisions'),
        gradient_audit_cases=sum(x['kind']=='outer_gradient_audit' for x in all_events),
        gradient_audit_failures=sum(x['kind']=='outer_gradient_audit' and not x.get('exact') for x in all_events),
        reference_comparison_cases=sum(x['kind']=='reference_comparison' for x in all_events),
        reference_comparison_failures=sum(x['kind']=='reference_comparison' and not x.get('exact') for x in all_events),
        outer_manager_instances=sum(x['kind']=='outer_ready' for x in all_events),
        final_signatures=final_signatures,
        limitations=['replicas share hosts; independent host fault domains are not demonstrated',
            'signatures and snapshots perturb timing; no general communication replay or zero-memory claim',
            'peer checkpoint recovery covers after-freeze/before-ready source loss; transfer failures fail-stop',
            'raw events and logs removed after independent audit; SHA256 signatures cannot reconstruct trajectories'],
        peer_outer_gradient_ready_seconds=audit['peer_outer_gradient_ready_seconds'],
        independent_audit=audit, actual_auditor_sha256=audit['actual_auditor_sha256'],
        raw_sha256=audit['raw_sha256'])
