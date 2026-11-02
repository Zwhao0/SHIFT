"""Offline validator for collected three-node SHIFT rotation evidence."""

import argparse
import json
import math
from pathlib import Path
import statistics


RANKS = range(8)
HASH_FIELDS = {
    "model", "optimizer", "scheduler", "progress",
    "cpu_rng", "cuda_rng", "python_rng",
}


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value):
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _read_json(path, errors, label):
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError:
        errors.append(f"missing {label}: {path}")
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {label} {path}: {exc}")
    return None


def _read_jsonl(path, errors, required=False):
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"invalid JSONL {path}:{lineno}: {exc}")
                    continue
                if isinstance(row, dict) and row.get("kind") != "phase":
                    rows.append(row)
                elif not isinstance(row, dict):
                    errors.append(f"non-object JSONL row {path}:{lineno}")
    except FileNotFoundError:
        if required:
            errors.append(f"missing JSONL: {path}")
    except OSError as exc:
        errors.append(f"cannot read JSONL {path}: {exc}")
    return rows


def _relative(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _manifest_path(root, entry):
    node = str(entry["physical_node"])
    rid = _integer(entry["rid"])
    generation = _integer(entry["generation"])
    directory = Path(str(entry.get("directory", ""))).name
    names = [directory, f"rid{rid}_g{generation}"]
    candidates = [root / f"physical_{node}" / name for name in names if name]
    candidates.append(root / f"physical_{node}")
    return next((path for path in candidates if path.is_dir()), candidates[0])


def _load_evidence(root, errors):
    manifest = _read_json(root / "physical_manifest.json", errors, "physical manifest")
    if not isinstance(manifest, list):
        errors.append("physical_manifest.json must be a list of rid/generation/node mappings")
        manifest = []

    mapping = {}
    node_names = set()
    for item in manifest:
        if not isinstance(item, dict):
            errors.append("physical manifest contains a non-object entry")
            continue
        rid, generation, node = _integer(item.get("rid")), _integer(item.get("generation")), item.get("physical_node")
        if rid not in (0, 1) or generation is None or generation < 0 or not isinstance(node, str) or not node:
            errors.append(f"invalid physical manifest entry: {item}")
            continue
        key = (rid, generation)
        if key in mapping:
            errors.append(f"duplicate physical manifest binding for rid{rid} generation{generation}")
            continue
        mapping[key] = {**item, "rid": rid, "generation": generation, "physical_node": node}
        node_names.add(node)

    if len(node_names) != 3:
        errors.append(f"physical coverage requires three nodes; found {sorted(node_names)}")
    if {rid for rid, _ in mapping} != {0, 1}:
        errors.append(f"physical manifest must cover both logical rids; found {sorted({rid for rid, _ in mapping})}")

    roles = {}
    supervisors = {}
    ready_nodes = set()
    for key, item in sorted(mapping.items()):
        rid, generation = key
        directory = _manifest_path(root, item)
        supervisors[key] = _read_jsonl(directory / "supervisor.jsonl", errors, required=True)
        for rank in RANKS:
            path = directory / f"events_g{generation}_r{rank}.jsonl"
            rows = _read_jsonl(path, errors, required=True)
            valid = []
            for row in rows:
                row_rid = _integer(row.get("rid"))
                row_generation = _integer(row.get("generation"))
                row_rank = _integer(row.get("rank"))
                if row_rid is not None and row_rid != rid:
                    errors.append(f"{_relative(path, root)} has rid {row_rid}, expected {rid}")
                if row_generation is not None and row_generation != generation:
                    errors.append(f"{_relative(path, root)} has generation {row_generation}, expected {generation}")
                if row_rank is not None and row_rank != rank:
                    errors.append(f"{_relative(path, root)} has rank {row_rank}, expected {rank}")
                valid.append(row)
            roles[rid, generation, rank] = valid
            if any(row.get("kind") == "ready" for row in valid):
                ready_nodes.add(item["physical_node"])

    if ready_nodes != node_names:
        errors.append(f"physical nodes without a worker ready event: {sorted(node_names - ready_nodes)}")

    controller = _read_jsonl(root / "controller.jsonl", errors, required=True)
    faults = _read_json(root / "fault_results.json", errors, "fault results")
    if not isinstance(faults, list):
        errors.append("fault_results.json must be a list")
        faults = []
    return mapping, roles, supervisors, controller, faults


def _matching_rows(events, kind, **fields):
    return [row for row in events if row.get("kind") == kind and all(row.get(k) == v for k, v in fields.items())]


def _event_time(row):
    return _number(row.get("time"))


def _commit_rows(roles, rid, generation, rank, *, participants=None,
                 after_time=None, before_time=None, after_step=None):
    result = {}
    for row in roles.get((rid, generation, rank), []):
        if row.get("kind") != "commit" or row.get("contributed") is not True:
            continue
        if participants is not None and _integer(row.get("participants")) != participants:
            continue
        step, timestamp = _integer(row.get("step")), _event_time(row)
        if step is None or timestamp is None:
            continue
        if after_step is not None and step <= after_step:
            continue
        if after_time is not None and timestamp < after_time:
            continue
        if before_time is not None and timestamp > before_time:
            continue
        old = result.get(step)
        if old is None or (_event_time(old) or float("-inf")) < timestamp:
            result[step] = row
    return result


def _common_steps(roles, active_generations, rids, *, participants=2,
                  after_step=None, after_time=None, before_time=None):
    per_role = {}
    sets = []
    for rid in rids:
        generation = active_generations[rid]
        for rank in RANKS:
            commits = _commit_rows(roles, rid, generation, rank, participants=participants,
                                   after_step=after_step, after_time=after_time,
                                   before_time=before_time)
            per_role[rid, rank] = commits
            sets.append(set(commits))
    common = set.intersection(*sets) if sets else set()
    return sorted(common), per_role


def _hashes_ok(row):
    expected = row.get("expected_hashes", row.get("expected"))
    actual = row.get("actual_hashes", row.get("actual"))
    valid_shape = (isinstance(expected, dict) and isinstance(actual, dict)
                   and set(expected) == HASH_FIELDS and set(actual) == HASH_FIELDS)
    return bool(valid_shape and expected == actual and row.get("hashes_match") is True)


def _percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    index = max(0, min(len(values) - 1, math.ceil(p * len(values)) - 1))
    return values[index]


def _distribution(values):
    values = [float(value) for value in values if _number(value) is not None]
    if not values:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(values), "min": min(values),
        "median": statistics.median(values), "p95": _percentile(values, .95),
        "max": max(values),
    }


def _max_gap(times, start=None, end=None):
    points = sorted(set(float(t) for t in times if _number(t) is not None))
    gaps = [b - a for a, b in zip(points, points[1:])
            if (start is None or b >= start) and (end is None or a <= end)]
    return max(gaps) if gaps else None


def _role_hashes_match(roles, step, errors, require_audit):
    checked = 0
    all_ok = True
    for rank in RANKS:
        pair = []
        for rid in (0, 1):
            row = roles[(rid, rank)].get(step)
            pair.append(row.get("hashes") if row else None)
        if all(isinstance(item, dict) for item in pair):
            if not require_audit and pair[0] == {} and pair[1] == {}:
                continue
            if set(pair[0]) != HASH_FIELDS or set(pair[1]) != HASH_FIELDS or pair[0] != pair[1]:
                all_ok = False
                errors.append(f"first joint step {step} rank {rank}: state hashes are missing fields or differ")
            else:
                checked += 1
        elif require_audit:
            all_ok = False
            errors.append(f"first joint step {step} rank {rank}: missing 7-field hashes")
        else:
            all_ok = False
            errors.append(f"first joint step {step} rank {rank}: missing paired hash dictionaries")
    return all_ok and (checked == 8 if require_audit else True), checked


def _verify_fault(index, fault, mapping, roles, supervisors, controller, active_generations,
                  require_audit):
    errors = []
    result = {"index": index, "errors": errors}
    target = _integer(fault.get("target"))
    if target not in (0, 1):
        errors.append("fault result has invalid target rid")
        return result, None
    healthy = 1 - target
    result.update(target=target, healthy_rid=healthy)
    current = dict(active_generations)

    injected_rows = _matching_rows(controller, "injected", index=index)
    if len(injected_rows) != 1:
        errors.append(f"expected one controller injected event; found {len(injected_rows)}")
        injected = injected_rows[-1] if injected_rows else {}
    else:
        injected = injected_rows[0]
    if injected and _integer(injected.get("target")) != target:
        errors.append("controller injected target disagrees with fault result")
    fault_time = _number(injected.get("time", fault.get("fault_time")))
    if fault_time is None:
        errors.append("missing actual fault_time")
    result["fault_time"] = fault_time

    new_generation = _integer(fault.get("generation"))
    old_generation = _integer(injected.get("generation"))
    if old_generation is None:
        old_generation = current[target]
    if old_generation != current[target]:
        errors.append(f"injected generation {old_generation} does not match active generation {current[target]}")
    if new_generation is None:
        new_generation = old_generation + 1
    if new_generation != old_generation + 1:
        errors.append(f"generation must advance exactly one: {old_generation} -> {new_generation}")
    result["generation"] = new_generation
    result["old_generation"] = old_generation

    old_binding = mapping.get((target, old_generation))
    new_binding = mapping.get((target, new_generation))
    if old_binding is None:
        errors.append(f"physical manifest lacks retired rid{target} generation{old_generation}")
    if new_binding is None:
        errors.append(f"physical manifest lacks replacement rid{target} generation{new_generation}")
    healthy_binding = mapping.get((healthy, current[healthy]))
    if old_binding and new_binding and old_binding["physical_node"] == new_binding["physical_node"]:
        errors.append("replacement reuses the retired physical node")
    if healthy_binding and new_binding and healthy_binding["physical_node"] == new_binding["physical_node"]:
        errors.append("replacement is placed on the still-active healthy node")
    nodes = {item["physical_node"] for item in mapping.values()}
    idle_nodes = nodes - ({old_binding["physical_node"]} if old_binding else set())
    idle_nodes -= ({healthy_binding["physical_node"]} if healthy_binding else set())
    if len(idle_nodes) != 1 or (new_binding and new_binding["physical_node"] not in idle_nodes):
        errors.append(f"replacement is not the unique idle physical node: {sorted(idle_nodes)}")
    if old_binding and fault.get("old_node") and old_binding["physical_node"] != fault["old_node"]:
        errors.append("fault result old_node disagrees with physical manifest")
    if new_binding and fault.get("new_node") and new_binding["physical_node"] != fault["new_node"]:
        errors.append("fault result new_node disagrees with physical manifest")
    if injected.get("physical_node") and old_binding and injected["physical_node"] != old_binding["physical_node"]:
        errors.append("injected physical_node disagrees with physical manifest")
    result["old_node"] = old_binding["physical_node"] if old_binding else fault.get("old_node")
    result["new_node"] = new_binding["physical_node"] if new_binding else fault.get("new_node")
    result["pre_fault_generations"] = current

    selected_rows = _matching_rows(controller, "candidate_selected", index=index)
    if len(selected_rows) != 1:
        errors.append(f"expected one candidate_selected event; found {len(selected_rows)}")
    elif new_binding:
        selected = selected_rows[0]
        candidates = selected.get("candidates")
        if not isinstance(candidates, list) or candidates != [new_binding["physical_node"]]:
            errors.append("controller candidates must contain only the selected idle node")
        if _integer(selected.get("target")) != target:
            errors.append("controller selected target disagrees with fault result")
        if selected.get("physical_node") != new_binding["physical_node"]:
            errors.append("controller selected physical_node disagrees with physical manifest")
        if _integer(selected.get("generation")) != new_generation:
            errors.append("controller selected generation disagrees with physical manifest")
        if _integer(selected.get("previous_generation")) != old_generation:
            errors.append("controller selected previous_generation disagrees with active binding")

    if fault.get("lighthouse_alive") is not True:
        errors.append("fault result does not confirm Lighthouse remained alive")
    result["lighthouse_alive"] = fault.get("lighthouse_alive") is True

    restored = {rank: _matching_rows(roles.get((target, new_generation, rank), []), "restored")
                for rank in RANKS}
    restored_rows = [row for rows in restored.values() for row in rows]
    for row in restored_rows:
        step = _integer(row.get("step"))
        if step is None or step <= 0:
            errors.append(f"restored event has invalid step: {step}")
        if (_integer(row.get("optimizer_tensors")) or 0) <= 0:
            errors.append("restored event has empty or unreported AdamW state")
        if require_audit and not _hashes_ok(row):
            errors.append(f"restored state hash audit failed or is incomplete for rank {row.get('rank')}")
        if row.get("hashes_match") is False:
            errors.append(f"restored state hash mismatch for rank {row.get('rank')}")
    for rank, rows in restored.items():
        if not rows:
            errors.append(f"replacement has no restored event for rank {rank}")
    latest_restored = {rank: max(rows, key=lambda row: _event_time(row) or float("-inf"))
                       for rank, rows in restored.items() if rows}
    restored_steps = {_integer(row.get("step")) for row in latest_restored.values()}
    if len(latest_restored) != 8 or len(restored_steps) != 1 or None in restored_steps:
        errors.append("replacement ranks do not report one consistent restored step")
        restored_step = None
    else:
        restored_step = restored_steps.pop()
    result["restored_step"] = restored_step
    result["restored_rank_count"] = len(latest_restored)
    restore_times = [_event_time(row) for row in latest_restored.values()]
    restored_time = max((t for t in restore_times if t is not None), default=None)
    result["restored_time"] = restored_time
    result["restore_hashes_ok"] = (
        bool(restored_rows) and len(latest_restored) == 8
        and (all(_hashes_ok(row) for row in restored_rows) if require_audit
             else all(row.get("hashes_match") is not False for row in restored_rows))
    )

    if restored_step is not None and fault_time is not None:
        steps, _ = _common_steps(roles, current, (0, 1), participants=2, before_time=fault_time)
        pre_fault_step = steps[-1] if steps else None
        result["highest_joint_step_before_fault"] = pre_fault_step
        if pre_fault_step is None:
            errors.append("no contributed participants=2 step is present on all 16 raw ranks before fault_time")
        elif pre_fault_step > restored_step:
            errors.append(f"RPO violation: pre-fault joint step {pre_fault_step} exceeds restored step {restored_step}")
        claimed = _integer(fault.get("committed_before_fault"))
        result["controller_committed_before_fault"] = claimed
        # Also preserve every complete joint update ever logged by the retired
        # generation. This stronger, clock-independent bound covers commits
        # finishing while SIGKILL is being delivered through the process tree.
        retired_steps, _ = _common_steps(roles, current, (0, 1), participants=2)
        last_retired_joint = retired_steps[-1] if retired_steps else None
        result['highest_joint_step_recorded_by_retired_generation'] = last_retired_joint
        if last_retired_joint is not None and last_retired_joint > restored_step:
            errors.append(f'RPO violation: retired joint step {last_retired_joint} exceeds restored step {restored_step}')

    controller_request_time = _number(fault.get("request_time"))
    if controller_request_time is None:
        restart_rows = _matching_rows(controller, "restart_requested", index=index)
        if restart_rows:
            controller_request_time = _event_time(restart_rows[-1])
    result["controller_request_time"] = controller_request_time
    if controller_request_time is None or restored_time is None:
        controller_request_to_restored = None
        errors.append("cannot compute controller launch/activate-to-restored timing")
    else:
        controller_request_to_restored = restored_time - controller_request_time
        if controller_request_to_restored < 0:
            errors.append("restored precedes controller launch/activate request")
    result["controller_request_to_restored_seconds"] = controller_request_to_restored

    recovery_request_times = []
    state_loaded_times = []
    for rank in RANKS:
        rank_events = roles.get((target, new_generation, rank), [])
        requests = [_event_time(row) for row in rank_events if row.get("kind") == "recovery_request"]
        loads = [_event_time(row) for row in rank_events if row.get("kind") == "state_loaded"]
        requests = [value for value in requests if value is not None]
        loads = [value for value in loads if value is not None]
        if not requests:
            errors.append(f"replacement rank {rank} has no train recovery_request event")
        else:
            recovery_request_times.append(min(requests))
        if not loads:
            errors.append(f"replacement rank {rank} has no raw state_loaded event")
        else:
            state_loaded_times.append(max(loads))
    train_request_time = min(recovery_request_times, default=None)
    state_loaded_time = max(state_loaded_times, default=None)
    result["train_recovery_request_time"] = train_request_time
    result["state_loaded_time"] = state_loaded_time
    result["state_loaded_rank_count"] = len(state_loaded_times)
    if train_request_time is None or state_loaded_time is None:
        train_request_to_state_loaded = None
    else:
        train_request_to_state_loaded = state_loaded_time - train_request_time
        if train_request_to_state_loaded < 0:
            errors.append("raw state_loaded precedes train recovery_request")
    result["train_recovery_request_to_state_loaded_seconds"] = train_request_to_state_loaded

    joint_step = None
    joint_time = None
    if restored_step is not None and restored_time is not None:
        post_restore_generations = dict(current)
        post_restore_generations[target] = new_generation
        post_steps, post_roles = _common_steps(
            roles, post_restore_generations, (0, 1), participants=2, after_step=restored_step,
            after_time=restored_time,
        )
        if not post_steps:
            errors.append("no common contributed participants=2 step after full restore")
        else:
            joint_step = post_steps[0]
            rows = [post_roles[rid, rank][joint_step] for rid in (0, 1) for rank in RANKS]
            joint_time = max((_event_time(row) for row in rows if _event_time(row) is not None), default=None)
            hashes_ok, checked = _role_hashes_match(post_roles, joint_step, errors, require_audit)
            result["first_joint_hashes_match"] = hashes_ok
            result["first_joint_hash_roles"] = checked
    result["first_joint_step"] = joint_step
    result["joint_time"] = joint_time
    if fault.get("first_joint_step") is not None and joint_step is not None:
        if _integer(fault.get("first_joint_step")) != joint_step:
            errors.append("fault_results first_joint_step disagrees with raw rank events")
    if joint_time is not None and fault_time is not None:
        result["fault_to_joint_seconds"] = joint_time - fault_time
        if result["fault_to_joint_seconds"] < 0:
            errors.append("first joint commit precedes fault_time (clock/order inconsistency)")
    else:
        result["fault_to_joint_seconds"] = None

    # Track the healthy generation throughout failover. Cold recovery must
    # show solo commits; warm/hot recovery may rejoin before the first one.
    healthy_generation = current[healthy]
    healthy_ready = {}
    healthy_pid_ok = True
    for rank in RANKS:
        rows = _matching_rows(roles.get((healthy, healthy_generation, rank), []), "ready")
        before = [row for row in rows if fault_time is None or (_event_time(row) or float("inf")) <= fault_time]
        if not before:
            errors.append(f"healthy rid{healthy} rank{rank} has no ready event before fault")
            healthy_pid_ok = False
            continue
        healthy_ready[rank] = before[-1].get("pid")
        if not isinstance(healthy_ready[rank], int) or healthy_ready[rank] <= 1:
            errors.append(f"healthy rid{healthy} rank{rank} has invalid ready pid")
            healthy_pid_ok = False
    solo_steps = []
    backup_mode = fault.get("backup_mode", "cold")
    solo_required = backup_mode == "cold"
    if fault_time is not None and joint_time is not None:
        solo_sets = []
        for rank in RANKS:
            commits = _commit_rows(roles, healthy, healthy_generation, rank,
                                   participants=1, after_time=fault_time,
                                   before_time=joint_time)
            solo_sets.append(set(commits))
            for row in _commit_rows(roles, healthy, healthy_generation, rank,
                                    after_time=fault_time, before_time=joint_time).values():
                if row.get("pid") != healthy_ready.get(rank):
                    errors.append(f"healthy rid{healthy} rank{rank} PID changed during failover")
                    healthy_pid_ok = False
        solo_steps = sorted(set.intersection(*solo_sets)) if solo_sets else []
        if solo_required and not solo_steps:
            errors.append("healthy replica made no common contributed participants=1 progress before rejoin")
    elif solo_required:
        errors.append("cannot verify required cold-recovery solo progress without a joint-rejoin time")
    result["healthy_pids_unchanged"] = healthy_pid_ok
    result["backup_mode"] = backup_mode
    result["healthy_solo_progress_required"] = solo_required
    result["healthy_solo_step_count"] = len(solo_steps)
    result["healthy_solo_first_step"] = solo_steps[0] if solo_steps else None
    result["healthy_solo_last_step"] = solo_steps[-1] if solo_steps else None
    result["healthy_progress"] = bool(solo_steps)

    if fault_time is not None and joint_time is not None:
        failover_gaps = []
        whole_gaps = []
        for rank in RANKS:
            rank_rows = [row for row in roles.get((healthy, healthy_generation, rank), [])
                         if row.get("kind") == "commit" and row.get("contributed") is True]
            times = [_event_time(row) for row in rank_rows]
            gap = _max_gap(times, fault_time, joint_time)
            if gap is not None:
                failover_gaps.append(gap)
            full_gap = _max_gap(times)
            if full_gap is not None:
                whole_gaps.append(full_gap)
        result["healthy_max_commit_gap_fault_to_joint_seconds"] = max(failover_gaps) if failover_gaps else None
        result["healthy_max_commit_gap_whole_generation_seconds"] = max(whole_gaps) if whole_gaps else None
    else:
        result["healthy_max_commit_gap_fault_to_joint_seconds"] = None
        result["healthy_max_commit_gap_whole_generation_seconds"] = None

    cleanup_rows = _matching_rows(supervisors.get((target, old_generation), []), "cleanup")
    result["retired_cleanup_empty"] = bool(cleanup_rows) and all(row.get("own_children") == [] for row in cleanup_rows)
    if not cleanup_rows:
        errors.append(f"retired supervisor rid{target} generation{old_generation} has no cleanup event")
    elif not result["retired_cleanup_empty"]:
        errors.append(f"retired supervisor rid{target} generation{old_generation} has unreaped children")

    active_generations[target] = new_generation
    return result, active_generations


def verify(result_dir, expected_faults, minimum_duration=None, require_audit=True):
    root = Path(result_dir)
    errors = []
    if not root.is_dir():
        return {"pass": False, "session": str(root), "errors": [f"result directory not found: {root}"]}
    mapping, roles, supervisors, controller, faults = _load_evidence(root, errors)
    joint_hashes={}
    for (rid,generation,rank),rows in roles.items():
        commits=[row for row in rows if row.get('kind')=='commit']
        steps=[row.get('step') for row in commits]
        if any(not isinstance(s,int) for s in steps) or any(b<=a for a,b in zip(steps,steps[1:])):
            errors.append(f'nonmonotone committed progress rid{rid} generation{generation} rank{rank}')
        for row in commits:
            if row.get('participants')==2 and row.get('contributed') and row.get('hashes'):
                joint_hashes.setdefault(row['step'],{})[rid,rank]=row['hashes']
            if row.get('finite_gradients') is not True:
                errors.append(f'nonfinite or unverified committed gradients rid{rid} generation{generation} rank{rank}')
            losses=row.get('loss',[])
            if isinstance(losses,list) and any(_number(v) is None for v in losses):
                errors.append(f'nonfinite committed loss rid{rid} generation{generation} rank{rank}')
            if 'global_batch' in row and row['global_batch'] not in (4,8):
                errors.append(f'unexpected global batch rid{rid} generation{generation} rank{rank}')
    audited_joint_steps=[]
    for step,states in sorted(joint_hashes.items()):
        if len(states)!=16:
            continue
        audited_joint_steps.append(step)
        for rank in RANKS:
            if set(states[0,rank])!=HASH_FIELDS or states[0,rank]!=states[1,rank]:
                errors.append(f'steady-state joint hash mismatch step{step} rank{rank}')
    if require_audit:
        for rank in RANKS:
            initial=[_matching_rows(roles.get((rid,0,rank),[]),'ready') for rid in (0,1)]
            if not all(initial) or set(initial[0][-1].get('hashes',{}))!=HASH_FIELDS or initial[0][-1]['hashes']!=initial[1][-1].get('hashes'):
                errors.append(f'initial full-state mismatch rank{rank}')

    if len(faults) != expected_faults:
        errors.append(f"fault_results has {len(faults)} faults; expected {expected_faults}")
    injected = [row for row in controller if row.get("kind") == "injected"]
    if len(injected) != expected_faults:
        errors.append(f"controller has {len(injected)} injected events; expected {expected_faults}")
    fault_indices = [_integer(row.get("index")) for row in injected]
    sorted_fault_indices = sorted(fault_indices, key=lambda value: -1 if value is None else value)
    if sorted_fault_indices != list(range(expected_faults)):
        errors.append(f"controller injection indices are {sorted_fault_indices}, expected 0..{expected_faults - 1}")

    window_starts = [row for row in controller if row.get("kind") == "fault_window_start"]
    completes = [row for row in controller if row.get("kind") == "complete"]
    if len(window_starts) != 1:
        errors.append(f"expected one fault_window_start event; found {len(window_starts)}")
        window = window_starts[-1] if window_starts else {}
    else:
        window = window_starts[0]
    if len(completes) != 1:
        errors.append(f"expected one complete event; found {len(completes)}")
        complete = completes[-1] if completes else {}
    else:
        complete = completes[0]
    requested = _integer(window.get("requested_faults"))
    actual = _integer(complete.get("faults"))
    requested_window = _number(window.get("duration"))
    actual_window = _number(complete.get("fault_window_seconds"))
    if requested is not None and requested != expected_faults:
        errors.append(f"fault_window_start requested_faults={requested}, expected {expected_faults}")
    if actual is not None and actual != expected_faults:
        errors.append(f"complete reports {actual} faults, expected {expected_faults}")
    if actual_window is None:
        errors.append("complete event lacks fault_window_seconds")
    elif requested_window is not None and actual_window + 1e-6 < requested_window:
        errors.append(f"fault window lasted {actual_window:.3f}s, requested {requested_window:.3f}s")
    if minimum_duration is not None:
        if actual_window is None or actual_window + 1e-6 < minimum_duration:
            errors.append(f"fault window is shorter than minimum {minimum_duration:.3f}s")
    schedule_late = [row for row in controller if row.get("kind") == "schedule_late"]
    if schedule_late:
        errors.append(f"controller recorded {len(schedule_late)} schedule_late event(s)")

    # Check final cleanup for every collected supervisor, including current
    # generations. Fault-specific checks below identify retired generations.
    supervisor_cleanup_count = 0
    for key, rows in supervisors.items():
        cleanup = _matching_rows(rows, "cleanup")
        if not cleanup:
            errors.append(f"supervisor rid{key[0]} generation{key[1]} lacks cleanup event")
        for row in cleanup:
            supervisor_cleanup_count += 1
            if row.get("own_children") != []:
                errors.append(f"supervisor rid{key[0]} generation{key[1]} cleanup has own_children={row.get('own_children')!r}")

    control_cleanup = []
    candidates = [root / "control" / "cleanup.json"]
    candidates.extend(sorted(root.glob("physical_node*/control/cleanup.json")))
    seen = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        value = _read_json(path, errors, "control cleanup")
        if isinstance(value, dict):
            control_cleanup.append({"path": _relative(path, root), "returncode": value.get("returncode")})
            # control_service terminates Lighthouse during normal cleanup; on
            # POSIX that expected SIGTERM exit is reported as -15.
            if value.get("returncode") not in (0, -15):
                errors.append(f"control cleanup returned {value.get('returncode')!r}: {_relative(path, root)}")
    if not control_cleanup:
        errors.append("control cleanup.json was not collected")
    if any(row.get("kind") == "control_cleanup_failed" for row in controller):
        errors.append("controller recorded control_cleanup_failed")

    active_generations = {0: 0, 1: 0}
    verified_faults = []
    metrics = {
        "fault_to_joint_seconds": [],
        "controller_request_to_restored_seconds": [],
        "train_recovery_request_to_state_loaded_seconds": [],
        "healthy_max_commit_gap_fault_to_joint_seconds": [],
        "healthy_max_commit_gap_whole_generation_seconds": [],
    }
    indexed_faults = {}
    for fault in faults:
        if not isinstance(fault, dict):
            errors.append("fault_results contains a non-object entry")
            continue
        index = _integer(fault.get("index"))
        if index is None or index in indexed_faults:
            errors.append(f"fault_results has missing or duplicate index: {fault.get('index')!r}")
            continue
        indexed_faults[index] = fault
    if sorted(indexed_faults) != list(range(expected_faults)):
        errors.append(f"fault_results indices are {sorted(indexed_faults)}, expected 0..{expected_faults - 1}")
    if expected_faults >= 10:
        target_coverage = {_integer(item.get("target")) for item in indexed_faults.values()}
        if target_coverage != {0, 1}:
            errors.append(f"10+ fault rotations must target both rids; found {sorted(target_coverage, key=lambda value: -1 if value is None else value)}")

    for index in range(expected_faults):
        fault = indexed_faults.get(index)
        if fault is None:
            continue
        checked, updated = _verify_fault(index, fault, mapping, roles, supervisors,
                                         controller, active_generations, require_audit)
        verified_faults.append(checked)
        errors.extend(f"fault {index}: {message}" for message in checked["errors"])
        if updated is not None:
            active_generations = updated
        for key in metrics:
            value = checked.get(key)
            if value is not None:
                metrics[key].append(value)

    return {
        "pass": not errors,
        "session": root.name,
        "require_audit": require_audit,
        "timestamp_note": "Raw wall-clock event timestamps are used; cross-host clock skew is not corrected.",
        "coverage": {
            "physical_nodes": sorted({item["physical_node"] for item in mapping.values()}),
            "logical_rids": sorted({rid for rid, _ in mapping}),
            "bindings": len(mapping),
        },
        "fault_window": {
            "requested_faults": requested,
            "completed_faults": actual,
            "requested_seconds": requested_window,
            "completed_seconds": actual_window,
            "schedule_late_events": len(schedule_late),
        },
        "control_cleanup": control_cleanup,
        "supervisor_cleanup_events": supervisor_cleanup_count,
        "audited_joint_steps": audited_joint_steps,
        "audited_joint_role_pairs": len(audited_joint_steps)*8,
        "timing_percentiles": {key: _distribution(values) for key, values in metrics.items()},
        "faults": verified_faults,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expected-faults", type=int, required=True)
    parser.add_argument("--minimum-duration", type=float)
    parser.add_argument("--require-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    if args.expected_faults < 1:
        parser.error("--expected-faults must be positive")
    if args.minimum_duration is not None and args.minimum_duration < 0:
        parser.error("--minimum-duration must be nonnegative")

    report = verify(args.result_dir, args.expected_faults, args.minimum_duration, args.require_audit)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
