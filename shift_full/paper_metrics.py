#!/usr/bin/env python3
"""Summarize recovery and healthy-replica timing from a local SHIFT session."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .latency import distribution, is_number, read_jsonl, time_of


RANKS = set(range(8))
EVENT_FILE = re.compile(r"events_g(\d+)_r(\d+)\.jsonl$")


def int_field(row, key):
    value = row.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def delta(end, start):
    if is_number(end) and is_number(start) and end >= start:
        return round(float(end) - float(start), 6)
    return None


def load_events(root):
    events = []
    for path in sorted(root.glob("node*/events_g*_r*.jsonl")):
        file_match = EVENT_FILE.fullmatch(path.name)
        node_match = re.fullmatch(r"node(\d+)", path.parent.name)
        if not file_match or not node_match:
            continue
        file_gen, file_rank = map(int, file_match.groups())
        file_rid = int(node_match.group(1))
        for row in read_jsonl(path):
            rid = int_field(row, "rid")
            gen = int_field(row, "generation")
            rank = int_field(row, "rank")
            events.append((row, file_rid if rid is None else rid,
                           file_gen if gen is None else gen,
                           file_rank if rank is None else rank))
    return events


def complete_replica_steps(events, rid, generation):
    grouped = defaultdict(dict)
    for row, event_rid, gen, rank in events:
        step = int_field(row, "step")
        participants = int_field(row, "participants")
        at = time_of(row)
        if (row.get("kind") == "commit" and event_rid == rid and gen == generation
                and rank in RANKS and step is not None and participants in (1, 2)
                and row.get("contributed") is True and at is not None):
            grouped[step][rank] = (participants, at)
    result = {}
    for step, ranks in grouped.items():
        if set(ranks) != RANKS:
            continue
        participant_counts = {value[0] for value in ranks.values()}
        if len(participant_counts) == 1:
            result[step] = {"participants": participant_counts.pop(),
                            "time": max(value[1] for value in ranks.values())}
    return result


def complete_joint_steps(events, active_generations):
    left = complete_replica_steps(events, 0, active_generations[0])
    right = complete_replica_steps(events, 1, active_generations[1])
    joint = {}
    for step in left.keys() & right.keys():
        if left[step]["participants"] == right[step]["participants"] == 2:
            joint[step] = max(left[step]["time"], right[step]["time"])
    return joint, left, right


def run_options(root, events):
    config = {}
    for path in sorted(root.glob("node*/provenance.json")):
        try:
            config = json.loads(path.read_text(encoding="utf-8")).get("config", {})
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        if isinstance(config, dict):
            break
        config = {}
    train_args = config.get("train_args")

    def value(key, flag):
        if key in config and isinstance(config[key], bool):
            return config[key]
        if isinstance(train_args, list):
            return flag in train_args
        return None

    audit = value("audit_state", "--audit-state")
    if audit is None:
        audit = any(row.get("kind") == "ready" and bool(row.get("hashes"))
                    for row, _, _, _ in events)
    return {
        "audit_state": audit,
        "fast_launch": value("fast_launch", "--fast-launch"),
        "persistent_cpu_launcher": value("persistent_cpu_launcher", "--persistent-cpu-launcher"),
        "warmup_before_join": value("warmup_before_join", "--warmup-before-join"),
        "single_gpu_visibility": value("single_gpu_visibility", "--single-gpu-visibility"),
    }


def stage_distributions(events):
    sends, recvs, loads, audits, reconfigures = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    begins, snapshots = defaultdict(list), defaultdict(list)
    for row, rid, gen, rank in events:
        key = f"rid{rid}/rank{rank}"
        kind = row.get("kind")
        if kind == "transport_send" and is_number(row.get("seconds")):
            sends[key].append(row["seconds"])
        elif kind == "transport_recv" and is_number(row.get("seconds")):
            recvs[key].append(row["seconds"])
        elif kind == "state_loaded" and is_number(row.get("load_seconds")):
            loads[key].append(row["load_seconds"])
        elif kind == "restored" and is_number(row.get("audit_seconds")):
            audits[key].append(row["audit_seconds"])
        elif kind == "reconfigure_end" and is_number(row.get("seconds")):
            reconfigures[key].append(row["seconds"])
        elif kind in ("snapshot_begin", "snapshot") and int_field(row, "step") is not None:
            target = begins if kind == "snapshot_begin" else snapshots
            target[(rid, gen, rank, int_field(row, "step"))].append(time_of(row))
    export = []
    for key, starts in begins.items():
        ends = snapshots.get(key, [])
        starts = sorted(value for value in starts if is_number(value))
        ends = sorted(value for value in ends if is_number(value))
        export.extend(end - start for start, end in zip(starts, ends) if end >= start)
    return {
        "snapshot_export_seconds": distribution(export),
        "state_load_seconds_by_rank": {key: distribution(values) for key, values in sorted(loads.items())},
        "audit_seconds_by_rank": {key: distribution(values) for key, values in sorted(audits.items())},
        "reconfigure_seconds_by_rank": {key: distribution(values) for key, values in sorted(reconfigures.items())},
        "gpu_transport_send_seconds_by_rank": {key: distribution(values) for key, values in sorted(sends.items())},
        "gpu_transport_recv_seconds_by_rank": {key: distribution(values) for key, values in sorted(recvs.items())},
    }


def analyze(root):
    controller = read_jsonl(root / "controller.jsonl")
    events = load_events(root)
    injected = {int_field(row, "index"): row for row in controller
                if row.get("kind") == "injected" and int_field(row, "index") is not None}
    restarts = {int_field(row, "index"): row for row in controller
                if row.get("kind") == "restart_requested" and int_field(row, "index") is not None}
    restored = {}
    for row in controller:
        match = re.fullmatch(r"fault_(\d+)_restored", str(row.get("kind", "")))
        if match:
            restored[int(match.group(1))] = row

    per_fault = []
    for index, fault in sorted(injected.items()):
        target = int_field(fault, "target")
        old_gen = int_field(fault, "generation")
        fault_time = time_of(fault)
        if target not in (0, 1) or old_gen is None:
            per_fault.append({"index": index, "target": target, "missing_evidence": ["target_or_generation"]})
            continue
        target_gen = old_gen + 1
        survivor = 1 - target
        restart = restarts.get(index)
        req_rows = [(row, rank) for row, rid, gen, rank in events
                    if rid == target and gen == target_gen and rank in RANKS
                    and row.get("kind") == "recovery_request"]
        request_time = min((time_of(row) for row, _ in req_rows if time_of(row) is not None), default=None)
        loaded_rows = defaultdict(list)
        for row, rid, gen, rank in events:
            if rid == target and gen == target_gen and rank in RANKS and row.get("kind") == "state_loaded":
                at = time_of(row)
                if at is not None and (request_time is None or at >= request_time):
                    loaded_rows[rank].append((at, int_field(row, "step")))
        first_loaded = {rank: min(rows) for rank, rows in loaded_rows.items() if rows}
        all_loaded = max((value[0] for value in first_loaded.values()), default=None) if len(first_loaded) == 8 else None
        loaded_steps = {value[1] for value in first_loaded.values() if value[1] is not None}

        active = restored.get(index, {}).get("generations")
        if not (isinstance(active, list) and len(active) == 2
                and all(isinstance(value, int) and not isinstance(value, bool) for value in active)):
            active = [old_gen, old_gen]
            active[target] = target_gen
        else:
            active = list(active)
        previous_restores = [(time_of(row), row.get("generations")) for row in controller
                             if re.fullmatch(r"fault_\d+_restored", str(row.get("kind", "")))
                             and time_of(row) is not None and fault_time is not None
                             and time_of(row) < fault_time]
        if previous_restores:
            previous = max(previous_restores, key=lambda item: item[0])[1]
            if isinstance(previous, list) and len(previous) == 2 and isinstance(previous[survivor], int):
                active[survivor] = previous[survivor]
        active[target] = target_gen

        shrink_by_rank = defaultdict(list)
        if fault_time is not None and request_time is not None:
            for row, rid, gen, rank in events:
                at = time_of(row)
                if (rid == survivor and gen == active[survivor] and rank in RANKS
                        and row.get("kind") == "reconfigure_end"
                        and int_field(row, "world_size") == 1 and at is not None
                        and fault_time <= at < request_time):
                    shrink_by_rank[rank].append(at)
        first_shrink = {rank: min(times) for rank, times in shrink_by_rank.items() if times}
        shrink_barrier = max(first_shrink.values()) if len(first_shrink) == 8 else None
        failure_handling = delta(shrink_barrier, fault_time)
        request_to_loaded = delta(all_loaded, request_time)
        active_components = (round(failure_handling + request_to_loaded, 6)
                             if failure_handling is not None and request_to_loaded is not None else None)

        joint_steps, healthy_steps_by_step, _ = complete_joint_steps(events, active)
        eligible_joint = sorted(step for step in joint_steps if not loaded_steps or step > max(loaded_steps))
        first_joint_step = eligible_joint[0] if eligible_joint else (min(joint_steps) if joint_steps else None)
        first_joint_time = joint_steps.get(first_joint_step) if first_joint_step is not None else None
        healthy_steps = complete_replica_steps(events, survivor, active[survivor])
        first_solo_time = min((item["time"] for item in healthy_steps.values()
                               if item["participants"] == 1 and item["time"] >= fault_time), default=None) if fault_time is not None else None
        solo_before_request = sorted((step, item["time"]) for step, item in healthy_steps.items()
                                     if item["participants"] == 1 and fault_time is not None
                                     and request_time is not None and fault_time <= item["time"] < request_time)
        solo_intervals = [right[1] - left[1] for left, right in zip(solo_before_request, solo_before_request[1:])
                          if right[0] == left[0] + 1 and right[1] >= left[1]]
        baseline = distribution(solo_intervals)["median"]
        solo_count = len(solo_before_request) if request_time is not None else None

        healthy_joint_time = healthy_steps.get(first_joint_step, {}).get("time") if first_joint_step is not None else None
        gap_values = []
        if request_time is not None and healthy_joint_time is not None:
            ordered = sorted((step, item["time"]) for step, item in healthy_steps.items())
            for (left_step, left), (right_step, right) in zip(ordered, ordered[1:]):
                if (right_step == left_step + 1 and left <= healthy_joint_time
                        and right >= request_time and right >= left):
                    gap_values.append(right - left)
        max_gap = max(gap_values) if gap_values else None
        excess = round(max_gap - baseline, 6) if max_gap is not None and baseline is not None else None

        missing = []
        if request_time is None:
            missing.append("recovery_request")
        if all_loaded is None:
            missing.append("state_loaded_all_8_ranks")
        if shrink_barrier is None:
            missing.append("healthy_world_size_1_reconfigure_end_all_8_ranks_before_request")
        if first_joint_time is None:
            missing.append("complete_joint_commit")
        if not healthy_steps:
            missing.append("healthy_commit_steps")
        per_fault.append({
            "index": index, "target": target, "target_generation": target_gen,
            "failed_physical_node": fault.get('physical_node'),
            "replacement_physical_node": restart.get('physical_node') if restart else None,
            "new_physical_machine": (fault['physical_node'] != restart['physical_node']
                                     if restart and fault.get('physical_node') and restart.get('physical_node') else None),
            "healthy_replica": survivor, "healthy_generation": active[survivor],
            "fault_to_restart_requested_seconds": delta(time_of(restart), fault_time) if restart else None,
            "fault_to_first_solo_commit_seconds": delta(first_solo_time, fault_time),
            "fault_to_first_joint_commit_seconds": delta(first_joint_time, fault_time),
            "first_joint_step": first_joint_step,
            "request_to_all_state_loaded_seconds": request_to_loaded,
            "failure_handling_seconds": failure_handling,
            "failure_handling_rank_count": len(first_shrink),
            "failure_handling_barrier_time": shrink_barrier,
            "active_recovery_components_seconds": active_components,
            "state_loaded_rank_count": len(first_loaded),
            "fault_to_request_seconds": delta(request_time, fault_time),
            "healthy_solo_steps_fault_to_request": solo_count,
            "healthy_max_adjacent_commit_gap_request_to_joint_seconds": round(max_gap, 6) if max_gap is not None else None,
            "healthy_gap_interval_count": len(gap_values),
            "healthy_pre_request_solo_interval_median_seconds": baseline,
            "healthy_pre_request_solo_interval_count": len(solo_intervals),
            "healthy_gap_excess_over_baseline_seconds": excess,
            "missing_evidence": missing,
        })

    metric_keys = (
        "fault_to_restart_requested_seconds", "fault_to_first_solo_commit_seconds",
        "fault_to_first_joint_commit_seconds", "request_to_all_state_loaded_seconds",
        "failure_handling_seconds", "active_recovery_components_seconds",
        "fault_to_request_seconds", "healthy_solo_steps_fault_to_request",
        "healthy_max_adjacent_commit_gap_request_to_joint_seconds", "healthy_gap_interval_count",
        "healthy_pre_request_solo_interval_median_seconds", "healthy_pre_request_solo_interval_count",
        "healthy_gap_excess_over_baseline_seconds",
    )
    summary = {key: distribution([fault.get(key) for fault in per_fault]) for key in metric_keys}
    options = run_options(root, events)
    return {
        "session": root.name,
        "clock_note": "跨节点 wall clock 未校验；fault/request/joint 端到端差值和窗口边界可能含时钟偏差；健康副本相邻 commit 间隔在同一副本时钟内计算。",
        "run_options": options,
        "sample_counts": {
            "faults": len(per_fault),
            "request_to_all_state_loaded": sum(item.get("request_to_all_state_loaded_seconds") is not None for item in per_fault),
            "failure_handling": sum(item.get("failure_handling_seconds") is not None for item in per_fault),
            "active_recovery_components": sum(item.get("active_recovery_components_seconds") is not None for item in per_fault),
            "fault_to_first_joint_commit": sum(item.get("fault_to_first_joint_commit_seconds") is not None for item in per_fault),
        },
        "metric_definitions": {
            "failure_handling_seconds": "fault to the latest of 8 healthy-rank first world_size=1 reconfigure_end events before target recovery_request; includes detection/arbitration/shrink, excludes subsequent solo forward/backward.",
            "active_recovery_components_seconds": "per-fault sum of failure_handling_seconds and request_to_all_state_loaded_seconds; excludes the interval between those stages (including continued healthy training and target cold start), so it is not continuous wall-clock RTO.",
        },
        "per_fault": per_fault,
        "summary_by_fault": summary,
        "stage_distributions": stage_distributions(events),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.session_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
