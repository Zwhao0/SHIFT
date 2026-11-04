#!/usr/bin/env python3
"""Audit a SHIFT fault session from its JSONL evidence (CPU-only)."""

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


RANKS = range(8)
CPU_TENSOR_LIMIT = 4 * 1024 * 1024
EVENT_FILE = re.compile(r"events_g(\d+)_r(\d+)\.jsonl$")


def read_jsonl(path):
    rows, errors = [], []
    try:
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("row is not an object")
                    rows.append(row)
                except Exception as exc:
                    errors.append(f"{path}:{lineno}: {exc}")
    except OSError as exc:
        errors.append(f"{path}: {exc}")
    return rows, errors


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def event_time(row):
    return row.get("time") if number(row.get("time")) else None


def distribution(values):
    values = [float(v) for v in values if number(v)]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def int_field(row, key):
    value = row.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def row_key(row, fallback):
    rid, gen, rank = (int_field(row, k) for k in ("rid", "generation", "rank"))
    return (fallback[0] if rid is None else rid,
            fallback[1] if gen is None else gen,
            fallback[2] if rank is None else rank)


def hashes_for_commit(row):
    hashes = row.get("hashes")
    if isinstance(hashes, dict):
        return {k: hashes.get(k) for k in ("model", "optimizer", "scheduler")}
    return {k: row.get(k + "_hash") for k in ("model", "optimizer", "scheduler")}


def usable_hashes(values):
    return all(isinstance(values.get(k), str) and values[k] for k in ("model", "optimizer", "scheduler"))


def flatten_numbers(value):
    if number(value):
        yield float(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_numbers(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--expected-faults", type=int, required=True)
    parser.add_argument("--minimum-duration", type=float, default=0.0, metavar="SECONDS")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_faults < 1 or args.minimum_duration < 0:
        parser.error("--expected-faults must be positive and --minimum-duration nonnegative")

    checks = []

    def check(name, ok, details):
        checks.append({"name": name, "pass": bool(ok), "details": details})

    root = args.session_dir
    parse_errors = []
    controller, errs = read_jsonl(root / "controller.jsonl")
    parse_errors.extend(errs)
    supervisors = {}
    for rid in (0, 1):
        path = root / f"node{rid}" / "supervisor.jsonl"
        supervisors[rid], errs = read_jsonl(path)
        parse_errors.extend(errs)

    events = []
    identity_errors = []
    for path in sorted(root.glob("node*/events_g*_r*.jsonl")):
        match = EVENT_FILE.fullmatch(path.name)
        node = re.fullmatch(r"node(\d+)", path.parent.name)
        if not match or not node:
            identity_errors.append(f"unrecognized event path: {path}")
            continue
        fallback = (int(node.group(1)), int(match.group(1)), int(match.group(2)))
        rows, errs = read_jsonl(path)
        parse_errors.extend(errs)
        for row in rows:
            key = row_key(row, fallback)
            if key != fallback:
                identity_errors.append(f"{path.name}: event identity {key} differs from file {fallback}")
            events.append((row, key, path))

    check("input", not parse_errors and not identity_errors, {
        "jsonl_errors": parse_errors,
        "event_file_identity_errors": identity_errors,
        "event_files": len({str(item[2]) for item in events}),
    })

    # Raw physical evidence must prove that a spare, rather than the failed
    # machine, supplied the replacement generation. Logical folders alone
    # cannot establish this distinction.
    if (root / 'physical_node3').exists():
        physical_errors = []
        counts = {}
        expected = {'physical_node0': (0, '192.0.2.10'),
                    'physical_node1': (1, '192.0.2.11'),
                    'physical_node3': (1, '192.0.2.13')}
        for label, (rid, ip) in expected.items():
            paths = sorted((root / label).glob('events_g*_r*.jsonl'))
            counts[label] = len(paths)
            if len(paths) < 8 or len(paths) % 8:
                physical_errors.append(f'{label}: incomplete rank file groups: {len(paths)}')
            for path in paths:
                rows, errors = read_jsonl(path)
                physical_errors.extend(errors)
                for row in rows:
                    gen = int_field(row, 'generation')
                    if row.get('rid') != rid or row.get('host_ip') != ip:
                        physical_errors.append(f'{label}/{path.name}: wrong rid/host identity')
                        break
                    if (gen is None or gen < 0 or (label == 'physical_node1' and gen != 0)
                            or (label == 'physical_node3' and gen < 1)):
                        physical_errors.append(f'{label}/{path.name}: unexpected generation {gen}')
                        break
        check('new_physical_machine_takeover', not physical_errors,
              {'rank_files': counts, 'errors': physical_errors})

    kind_rows = defaultdict(list)
    for row in controller:
        kind_rows[row.get("kind")].append(row)
    injected = sorted(kind_rows.get("injected", []), key=lambda e: int_field(e, "index") if int_field(e, "index") is not None else -1)
    complete = kind_rows.get("complete", [])
    failed = kind_rows.get("failed", [])
    check("controller_complete_and_fault_count", len(complete) == 1 and not failed and len(injected) == args.expected_faults, {
        "complete_events": len(complete), "failed_events": len(failed),
        "injected_count": len(injected), "expected_faults": args.expected_faults,
        "failed": failed,
    })

    controller_events = defaultdict(list)
    controller_positions = {id(row): index for index, row in enumerate(controller)}
    for row in controller:
        controller_events[row.get("kind")].append(row)
    protocol_errors = []
    injections_by_index = {}
    restart_by_index = {}
    for row in injected:
        idx = int_field(row, "index")
        if idx is None or idx in injections_by_index:
            protocol_errors.append(f"injected event has missing/duplicate index: {row}")
        else:
            injections_by_index[idx] = row
    if sorted(injections_by_index) != list(range(args.expected_faults)):
        protocol_errors.append(f"injection indexes are {sorted(injections_by_index)}, expected 0..{args.expected_faults - 1}")
    for row in controller_events.get("restart_requested", []):
        idx = int_field(row, "index")
        if idx is not None:
            if idx in restart_by_index:
                protocol_errors.append(f"duplicate restart_requested for index {idx}")
            restart_by_index[idx] = row

    stages = []
    for idx in range(max(args.expected_faults, 0)):
        inj = injections_by_index.get(idx)
        restart = restart_by_index.get(idx)
        immediate = restart is not None and restart.get('restart_policy') == 'immediate'
        names = [f"prefault_{idx}", f"fault_{idx}_ack"]
        if not immediate:
            names.append(f"fault_{idx}_survivor_progress")
        names += [f"fault_{idx}_restored", f"fault_{idx}_rejoined"]
        stage_rows = {name: controller_events.get(name, []) for name in names}
        missing = [name for name, rows in stage_rows.items() if len(rows) != 1]
        if missing:
            protocol_errors.append(f"fault {idx}: missing/duplicate protocol stages {missing}")
        if restart is None:
            protocol_errors.append(f"fault {idx}: missing restart_requested")
        stage_times = []
        if inj and restart:
            stage_times = [event_time(stage_rows[name][0]) if len(stage_rows[name]) == 1 else None
                           for name in (f"prefault_{idx}",)]
            stage_times += [event_time(inj)]
            stage_times += [event_time(stage_rows[f"fault_{idx}_ack"][0]) if len(stage_rows[f"fault_{idx}_ack"]) == 1 else None]
            if not immediate:
                stage_times += [event_time(stage_rows[f"fault_{idx}_survivor_progress"][0]) if len(stage_rows[f"fault_{idx}_survivor_progress"]) == 1 else None]
            stage_times += [event_time(restart)]
            stage_times += [event_time(stage_rows[f"fault_{idx}_restored"][0]) if len(stage_rows[f"fault_{idx}_restored"]) == 1 else None]
            stage_times += [event_time(stage_rows[f"fault_{idx}_rejoined"][0]) if len(stage_rows[f"fault_{idx}_rejoined"]) == 1 else None]
            # injected.time is copied from the remote supervisor; every other
            # timestamp here belongs to the workstation. Compare only the local
            # clock, and verify causality by the single-writer controller order.
            local_times = stage_times[:1] + stage_times[2:]
            ordered_rows = [stage_rows[f'prefault_{idx}'][0], stage_rows[f'fault_{idx}_ack'][0], inj] if not missing else []
            if ordered_rows and not immediate: ordered_rows.append(stage_rows[f'fault_{idx}_survivor_progress'][0])
            if ordered_rows: ordered_rows += [restart, stage_rows[f'fault_{idx}_restored'][0], stage_rows[f'fault_{idx}_rejoined'][0]]
            positions = [controller_positions[id(row)] for row in ordered_rows]
            if (any(t is None for t in stage_times)
                    or any(b < a for a, b in zip(local_times, local_times[1:]) if a is not None and b is not None)
                    or any(b <= a for a, b in zip(positions, positions[1:]))):
                protocol_errors.append(f"fault {idx}: controller protocol order invalid: times={local_times}, positions={positions}")
            target, old_gen = int_field(inj, "target"), int_field(inj, "generation")
            if target not in (0, 1) or old_gen is None:
                protocol_errors.append(f"fault {idx}: injection target/generation invalid: {inj}")
            if int_field(restart, "target") != target or int_field(restart, "generation") != old_gen + 1:
                protocol_errors.append(f"fault {idx}: restart target/generation does not follow injection")
            if target in (0, 1):
                sup_injected = [r for r in supervisors[target] if r.get("kind") == "injected"
                                and int_field(r, "generation") == old_gen]
                sup_restarts = [r for r in supervisors[target] if r.get("kind") == "restart_requested"
                                and int_field(r, "generation") == int_field(restart, "generation")]
                if len(sup_injected) != 1:
                    protocol_errors.append(f"fault {idx}: target supervisor injected count={len(sup_injected)}")
                elif event_time(sup_injected[0]) is not None and event_time(inj) is not None and abs(event_time(sup_injected[0]) - event_time(inj)) > 1e-6:
                    protocol_errors.append(f"fault {idx}: controller/supervisor injection timestamps differ")
                if len(sup_restarts) != 1:
                    protocol_errors.append(f"fault {idx}: target supervisor restart_requested count={len(sup_restarts)}")
            if int_field(restart, "target") in (0, 1) and int_field(restart, "generation") is not None:
                restart_by_index[idx] = restart
        stages.append({"index": idx, "present": not missing and restart is not None, "timestamps": stage_times})
    check("repeated_fault_protocol", not protocol_errors and len(stages) == args.expected_faults, {
        "faults": stages, "errors": protocol_errors,
    })

    # Supervisor lifecycle and generation timelines.
    supervisor_errors = []
    physical_logs = sorted(root.glob('physical_node*/supervisor.jsonl'))
    lifecycles = {}
    if physical_logs:
        for path in physical_logs:
            rows, errors = read_jsonl(path)
            supervisor_errors.extend(errors)
            lifecycles[path.parent.name] = rows
    else:
        lifecycles = {f'node{rid}': supervisors.get(rid, []) for rid in (0, 1)}
    for label, rows in lifecycles.items():
        if not rows:
            supervisor_errors.append(f"{label}: supervisor.jsonl missing or empty")
            continue
        bad = [r for r in rows if r.get("kind") in ("unexpected_exit", "supervisor_error")]
        if bad:
            supervisor_errors.append(f"{label}: unexpected supervisor events {bad}")
        if rows[-1].get("kind") != "cleanup":
            supervisor_errors.append(f"{label}: final event is {rows[-1].get('kind')!r}, expected cleanup")
        elif rows[-1].get("own_children") != []:
            supervisor_errors.append(f"{label}: cleanup own_children={rows[-1].get('own_children')!r}")
    check("supervisor_lifecycle", not supervisor_errors, {"errors": supervisor_errors})

    supervisor_launches = {}
    for rid, rows in supervisors.items():
        supervisor_launches[rid] = sorted(
            [r for r in rows if r.get("kind") == "launched" and int_field(r, "generation") is not None],
            key=lambda r: event_time(r) if event_time(r) is not None else -1,
        )

    by_key = defaultdict(list)
    by_kind = defaultdict(list)
    for row, key, path in events:
        by_key[key].append(row)
        by_kind[row.get("kind")].append((row, key, path))

    # Restore evidence must carry full hashes and match the exact source snapshot.
    snapshot_index = defaultdict(list)
    for row, key, _ in by_kind.get("snapshot", []):
        snapshot_index[(key[0], key[2], int_field(row, "step"))].append(row)
    restores = by_kind.get("restored", [])
    restore_errors = []
    restore_groups = {}
    for idx in range(args.expected_faults):
        inj, restart = injections_by_index.get(idx), restart_by_index.get(idx)
        if not inj or not restart:
            continue
        target, new_gen = int_field(restart, "target"), int_field(restart, "generation")
        if target not in (0, 1) or new_gen is None:
            continue
        for rank in RANKS:
            matches = [(r, k) for r, k, _ in restores if k == (target, new_gen, rank)]
            restore_groups[(target, new_gen, rank)] = matches
            if len(matches) != 1:
                restore_errors.append(f"fault {idx} target={target} generation={new_gen} rank={rank}: restored count={len(matches)}")
                continue
            row, key = matches[0]
            step = int_field(row, "step")
            expected, actual = row.get("expected_hashes"), row.get("actual_hashes")
            if step is None or step <= 0:
                restore_errors.append(f"fault {idx} rank={rank}: restored step is not positive")
            if row.get("hashes_match") is not True:
                restore_errors.append(f"fault {idx} rank={rank}: hashes_match is not true")
            if not isinstance(row.get("optimizer_tensors"), int) or row.get("optimizer_tensors", 0) <= 0:
                restore_errors.append(f"fault {idx} rank={rank}: optimizer_tensors is not positive")
            if not isinstance(expected, dict) or not expected or expected != actual:
                restore_errors.append(f"fault {idx} rank={rank}: expected_hashes and actual_hashes are absent or differ")
            source = int_field(row, "source_replica")
            if source not in (0, 1):
                source = 1 - target
            candidates = snapshot_index.get((source, rank, step), []) if step is not None else []
            if not expected or not any(s.get("hashes") == expected for s in candidates):
                restore_errors.append(f"fault {idx} rank={rank}: no source snapshot.hashes match rid={source}, step={step}")
    check("all_rank_restore_hashes", not restore_errors, {
        "required_ranks_per_restart": 8, "restored_events": len(restores), "errors": restore_errors,
    })

    # Commit sequence, update counts, finite gradients, and per-generation PID stability.
    sequence_errors, commit_errors, pid_errors = [], [], []
    commit_groups = defaultdict(list)
    for (rid, gen, rank), rows in by_key.items():
        for row in rows:
            if row.get("kind") == "commit":
                commit_groups[(rid, gen, rank)].append(row)
    for key, rows in sorted(commit_groups.items()):
        rows.sort(key=lambda r: (event_time(r) if event_time(r) is not None else float("inf"), int_field(r, "step") or -1))
        steps = [int_field(r, "step") for r in rows]
        restore = next((r for r, _ in restore_groups.get(key, [])), None)
        first_expected = int_field(restore, "step") + 1 if restore and int_field(restore, "step") is not None else (1 if key[1] == 0 else None)
        if any(s is None for s in steps):
            sequence_errors.append(f"rid/gen/rank={key}: commit step missing")
        elif first_expected is None:
            sequence_errors.append(f"rid/gen/rank={key}: nonzero generation has no restored step")
        elif steps and (steps[0] != first_expected or any(b != a + 1 for a, b in zip(steps, steps[1:]))):
            sequence_errors.append(f"rid/gen/rank={key}: steps {steps[0]}..{steps[-1]} are not consecutive from {first_expected}")
        for row in rows:
            if int_field(row, "updates") != int_field(row, "step") or int_field(row, "step") is None:
                commit_errors.append(f"rid/gen/rank={key} step={row.get('step')}: updates does not equal step")
            if row.get("finite_gradients") is not True:
                commit_errors.append(f"rid/gen/rank={key} step={row.get('step')}: finite_gradients is not true")
    for (rid, gen, rank), rows in by_key.items():
        pids = {r.get("pid") for r in rows if r.get("pid") is not None}
        if len(pids) > 1:
            pid_errors.append(f"rid/gen/rank={(rid, gen, rank)}: multiple pids {sorted(pids, key=str)}")
    check("consecutive_commit_steps", not sequence_errors, {"groups": len(commit_groups), "errors": sequence_errors})
    check("commit_updates_and_finite_gradients", not commit_errors, {"commit_count": sum(map(len, commit_groups.values())), "errors": commit_errors})
    check("pid_stable_within_generation", not pid_errors, {"errors": pid_errors})

    # Validate a post-fault solo window for each surviving replica.
    solo_errors, fault_metrics = [], []
    for idx in range(args.expected_faults):
        inj, restart = injections_by_index.get(idx), restart_by_index.get(idx)
        if not inj or not restart:
            solo_errors.append(f"fault {idx}: cannot validate survivor without injection/restart markers")
            continue
        target, old_gen = int_field(inj, "target"), int_field(inj, "generation")
        t0, tr = event_time(inj), event_time(restart)
        if target not in (0, 1) or old_gen is None or t0 is None or tr is None:
            solo_errors.append(f"fault {idx}: invalid target/generation/timestamps")
            continue
        healthy = 1 - target
        immediate = restart.get('restart_policy') == 'immediate'
        rejoined = controller_events.get(f'fault_{idx}_rejoined', [])
        observation_end = event_time(rejoined[0]) if immediate and rejoined else tr
        launches = [r for r in supervisor_launches.get(healthy, []) if event_time(r) is not None and event_time(r) <= t0]
        active_gen = int_field(launches[-1], "generation") if launches else None
        per_rank = {}
        solo_times, first_solo_pids = [], {}
        for rank in RANKS:
            rows = by_key.get((healthy, active_gen, rank), []) if active_gen is not None else []
            commits = [r for r in rows if r.get("kind") == "commit" and r.get("participants") == 1
                       and event_time(r) is not None and t0 < event_time(r) < observation_end]
            per_rank[rank] = len(commits)
            solo_times.extend(event_time(r) for r in commits if event_time(r) is not None)
            pids = {r.get("pid") for r in rows if r.get("pid") is not None}
            if len(pids) != 1:
                solo_errors.append(f"fault {idx}: healthy rid={healthy} rank={rank} generation={active_gen} PID set={sorted(pids, key=str)}")
            continued = [r for r in rows if r.get('kind') == 'commit'
                         and event_time(r) is not None and t0 < event_time(r) < observation_end]
            if not commits and not (immediate and continued):
                solo_errors.append(f"fault {idx}: no participants=1 commit for healthy rid={healthy} rank={rank} after kill before restart")
            else:
                cpids = {r.get("pid") for r in (commits or continued)}
                if len(cpids) != 1:
                    solo_errors.append(f"fault {idx}: survivor rank={rank} solo commit PIDs changed: {sorted(cpids, key=str)}")
                first_solo_pids[rank] = next(iter(cpids))
        first_solo = min(solo_times) if solo_times else None
        if not any(per_rank.values()) and not immediate:
            solo_errors.append(f"fault {idx}: no post-kill solo commits")

        new_gen = int_field(restart, "generation")
        restored_times = [event_time(r) for rank in RANKS for r, _ in restore_groups.get((target, new_gen, rank), []) if event_time(r) is not None]
        last_restored = max(restored_times) if restored_times else None
        fault_metrics.append({
            "index": idx, "target": target, "generation": old_gen, "healthy_rid": healthy,
            "healthy_generation": active_gen, "kill_time": t0,
            "first_solo_time": first_solo,
            "kill_to_first_solo_seconds": first_solo - t0 if first_solo is not None else None,
            "restart_requested_time": tr, "last_restored_time": last_restored,
            "restart_to_last_restored_seconds": last_restored - tr if last_restored is not None else None,
            "kill_to_last_restored_seconds": last_restored - t0 if last_restored is not None else None,
            "solo_commits_by_rank": per_rank,
            "survivor_pid_by_rank": first_solo_pids,
        })
    check("survivor_solo_progress_and_pid", not solo_errors, {"faults": fault_metrics, "errors": solo_errors})

    # Match every restore receive to its source send, including exact byte/tensor counts.
    send_events = [r for r, _, _ in by_kind.get("transport_send", [])]
    recv_events = [r for r, _, _ in by_kind.get("transport_recv", [])]
    transfer_errors, paired_sends, paired_recvs = [], set(), set()
    transfer_pairs = []
    for idx in range(args.expected_faults):
        restart = restart_by_index.get(idx)
        if not restart:
            continue
        target, gen = int_field(restart, "target"), int_field(restart, "generation")
        if target not in (0, 1) or gen is None:
            continue
        for rank in RANKS:
            matches = restore_groups.get((target, gen, rank), [])
            if len(matches) != 1:
                continue
            restored, _ = matches[0]
            step = int_field(restored, "step")
            src = int_field(restored, "source_replica")
            if src not in (0, 1):
                src = 1 - target
            recv_matches = [r for r in recv_events if int_field(r, "rid") == target and int_field(r, "generation") == gen
                            and int_field(r, "rank") == rank and int_field(r, "step") == step]
            if len(recv_matches) != 1:
                transfer_errors.append(f"fault {idx} rank={rank}: transport_recv count={len(recv_matches)} for restored step={step}")
                continue
            recv = recv_matches[0]
            src_snaps = snapshot_index.get((src, rank, step), [])
            source_gens = {int_field(s, "generation") for s in src_snaps}
            send_matches = [r for r in send_events if int_field(r, "rid") == src and int_field(r, "rank") == rank
                            and int_field(r, "step") == step and (not source_gens or int_field(r, "generation") in source_gens)]
            if len(send_matches) != 1:
                transfer_errors.append(f"fault {idx} rank={rank}: source transport_send count={len(send_matches)} for rid={src}, step={step}")
                continue
            send = send_matches[0]
            paired_sends.add(id(send)); paired_recvs.add(id(recv))
            for name, row in (("send", send), ("recv", recv)):
                if not number(row.get("gpu_bytes")) or row["gpu_bytes"] <= 0:
                    transfer_errors.append(f"fault {idx} rank={rank}: {name} gpu_bytes is not positive")
                if not isinstance(row.get("tensors"), int) or row.get("tensors", 0) <= 0:
                    transfer_errors.append(f"fault {idx} rank={rank}: {name} tensors is not positive")
            if send.get("gpu_bytes") != recv.get("gpu_bytes") or send.get("tensors") != recv.get("tensors"):
                transfer_errors.append(f"fault {idx} rank={rank}: send/recv counts differ: gpu_bytes {send.get('gpu_bytes')}/{recv.get('gpu_bytes')}, tensors {send.get('tensors')}/{recv.get('tensors')}")
            transfer_pairs.append({"fault": idx, "rank": rank, "source_rid": src, "target_rid": target,
                                   "step": step, "gpu_bytes": recv.get("gpu_bytes"), "tensors": recv.get("tensors")})
    unpaired_recvs = [r for r in recv_events if id(r) not in paired_recvs]
    unpaired_sends = [r for r in send_events if id(r) not in paired_sends]
    if unpaired_recvs:
        transfer_errors.append(f"unmatched transport_recv events: {len(unpaired_recvs)}")
    if unpaired_sends:
        transfer_errors.append(f"unmatched transport_send events: {len(unpaired_sends)}")
    check("gpu_transport_bytes_and_tensor_counts", bool(send_events and recv_events) and not transfer_errors, {
        "send_events": len(send_events), "recv_events": len(recv_events), "matched_restores": len(transfer_pairs),
        "gpu_bytes_sent": sum(r.get("gpu_bytes", 0) for r in send_events if number(r.get("gpu_bytes"))),
        "gpu_bytes_received": sum(r.get("gpu_bytes", 0) for r in recv_events if number(r.get("gpu_bytes"))),
        "tensors_sent": sum(r.get("tensors", 0) for r in send_events if isinstance(r.get("tensors"), int)),
        "tensors_received": sum(r.get("tensors", 0) for r in recv_events if isinstance(r.get("tensors"), int)),
        "pairs": transfer_pairs, "errors": transfer_errors,
    })

    cpu_errors = []
    transport_events = [(r, k) for kind in ("transport_send", "transport_recv") for r, k, _ in by_kind.get(kind, [])]
    for row, key in transport_events:
        size = row.get("cpu_tensor_bytes")
        if not isinstance(size, int) or size < 0 or size > CPU_TENSOR_LIMIT:
            cpu_errors.append(f"{row.get('kind')} rid/gen/rank={key} step={row.get('step')}: cpu_tensor_bytes={size!r}, limit={CPU_TENSOR_LIMIT}")
    check("cpu_tensor_bytes_limit", bool(transport_events) and not cpu_errors, {
        "limit_bytes": CPU_TENSOR_LIMIT, "transport_events": len(transport_events), "errors": cpu_errors,
    })

    # At least three shared two-replica steps after the last restoration; compare
    # model/optimizer/scheduler hashes rank by rank whenever sampled.
    joint_errors, joint_details = [], {}
    latest_target, latest_gen, barrier = None, None, None
    if args.expected_faults and args.expected_faults - 1 in restart_by_index:
        last = restart_by_index[args.expected_faults - 1]
        latest_target, latest_gen = int_field(last, "target"), int_field(last, "generation")
        times = [event_time(r) for rank in RANKS for r, _ in restore_groups.get((latest_target, latest_gen, rank), []) if event_time(r) is not None]
        barrier = max(times) if times else None
    complete_time = event_time(complete[0]) if len(complete) == 1 else None
    active_gens = {}
    for rid in (0, 1):
        launched = [r for r in supervisor_launches.get(rid, []) if complete_time is None or (event_time(r) is not None and event_time(r) <= complete_time)]
        active_gens[rid] = int_field(launched[-1], "generation") if launched else None
    common_by_rank = {}
    hash_modes = []
    for rank in RANKS:
        rank_steps = []
        per_replica = {}
        for rid in (0, 1):
            gen = active_gens[rid]
            rows = by_key.get((rid, gen, rank), []) if gen is not None else []
            per_replica[rid] = {int_field(r, "step"): r for r in rows if r.get("kind") == "commit"
                                and r.get("participants") == 2 and int_field(r, "step") is not None
                                and (barrier is None or (event_time(r) is not None and event_time(r) > barrier))}
        rank_steps = sorted(set(per_replica[0]) & set(per_replica[1]))
        common_by_rank[rank] = rank_steps
    common_steps = sorted(set.intersection(*(set(common_by_rank[r]) for r in RANKS))) if all(common_by_rank.values()) else []
    hash_checked_steps, hash_dict_complete = [], bool(common_steps)
    hash_seen = False
    for step in common_steps:
        step_ok = True
        for rank in RANKS:
            pair = []
            for rid in (0, 1):
                gen = active_gens[rid]
                row = next((r for r in by_key.get((rid, gen, rank), []) if r.get("kind") == "commit"
                            and int_field(r, "step") == step and r.get("participants") == 2
                            and (barrier is None or (event_time(r) is not None and event_time(r) > barrier))), None)
                pair.append(row)
            left, right = (hashes_for_commit(row) if row else {} for row in pair)
            left_ok, right_ok = usable_hashes(left), usable_hashes(right)
            if not (left_ok and right_ok):
                step_ok = False
                hash_dict_complete = False
                if left_ok or right_ok:
                    joint_errors.append(f"step={step} rank={rank}: one replica has incomplete model/optimizer/scheduler hash sample")
                continue
            hash_seen = True
            if not all(left[k] == right[k] for k in ("model", "optimizer", "scheduler")):
                step_ok = False
                joint_errors.append(f"step={step} rank={rank}: model/optimizer/scheduler hashes differ")
            if not all(isinstance(row.get("hashes"), dict) and usable_hashes(row.get("hashes", {})) for row in pair):
                hash_dict_complete = False
        if step_ok:
            hash_checked_steps.append(step)
    joint_mode = "full" if hash_dict_complete and len(hash_checked_steps) == len(common_steps) and len(hash_checked_steps) >= 3 else "sampled"
    if len(common_steps) < 3:
        joint_errors.append(f"only {len(common_steps)} common participants=2 steps after recovery; need at least 3")
    if barrier is None:
        joint_errors.append("last restoration barrier is missing")
    if not hash_seen:
        joint_errors.append("no common post-recovery model/optimizer/scheduler hash sample is available")
    joint_details = {
        "active_generations": active_gens, "last_restoration_barrier": barrier,
        "common_participants2_steps": common_steps, "hash_mode": joint_mode,
        "hash_checked_steps": hash_checked_steps, "hash_sample_count": len(hash_checked_steps),
        "common_steps_by_rank": common_by_rank, "errors": joint_errors,
    }
    check("post_recovery_joint_hashes", not joint_errors, joint_details)

    # Duration and summary metrics.
    windows, completions = controller_events.get("fault_window_start", []), complete
    duration = None
    duration_errors = []
    if len(windows) != 1 or len(completions) != 1 or event_time(windows[0]) is None or event_time(completions[0]) is None:
        duration_errors.append("fault_window_start/complete timestamp missing or duplicated")
    else:
        duration = event_time(completions[0]) - event_time(windows[0])
        if duration < args.minimum_duration:
            duration_errors.append(f"duration {duration:.3f}s is below minimum {args.minimum_duration:.3f}s")
    check("fault_window_duration", not duration_errors, {
        "duration_seconds": duration, "minimum_seconds": args.minimum_duration, "errors": duration_errors,
    })

    send_seconds = [r.get("seconds") for r in send_events]
    recv_seconds = [r.get("seconds") for r in recv_events]
    recovery_timing = defaultdict(list)
    for row, _, _ in events:
        for key, value in row.items():
            low = key.lower()
            if number(value) and ("load" in low or "audit" in low) and ("second" in low or low.endswith("_sec")):
                recovery_timing[key].append(value)
    skipped = defaultdict(int)
    for row, key, _ in events:
        if "skip" in str(row.get("kind", "")).lower():
            source = row.get("source_rid", row.get("src_rid", row.get("source_replica", row.get("source", row.get("rid", key[0])))))
            skipped[str(source)] += 1
    loss_events = sorted(
        (row for rows in commit_groups.values() for row in rows
         if row.get("contributed") is not False and row.get("loss")),
        key=lambda row: event_time(row) if event_time(row) is not None else float("inf"),
    )
    loss_values = [value for row in loss_events for value in flatten_numbers(row.get("loss"))]
    metrics = {
        "fault_window_duration_seconds": duration,
        "faults": fault_metrics,
        "transport_seconds": {"send": distribution(send_seconds), "recv": distribution(recv_seconds)},
        "recovery_load_audit_seconds": {key: distribution(values) for key, values in sorted(recovery_timing.items())},
        "source_skipped_events": dict(skipped),
        "committed_loss": {
            "contributed_false_excluded": True, "samples": len(loss_values),
            "first": loss_values[0] if loss_values else None,
            "last": loss_values[-1] if loss_values else None,
        },
    }

    passed = all(c["pass"] for c in checks)
    report = {
        "session_dir": str(root), "expected_faults": args.expected_faults,
        "pass": passed, "checks": checks, "metrics": metrics,
    }
    output = args.output if args.output is not None else root / "report.json"
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"cannot write report {output}: {exc}", file=sys.stderr)
        return 1
    failed_names = [c["name"] for c in checks if not c["pass"]]
    print(json.dumps({"pass": passed, "checks": len(checks), "failed": failed_names,
                      "duration_seconds": duration, "report": str(output)}, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
