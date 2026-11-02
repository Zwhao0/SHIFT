#!/usr/bin/env python3
"""Summarize per-fault recovery latency from local SHIFT JSONL evidence."""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


FAULT_STAGES = (
    "fault_to_restart_requested",
    "target_launch_to_all_ready",
    "all_ready_to_all_restored",
    "all_restored_to_first_common_commit",
    "fault_to_first_common_commit",
    "fault_to_solo",
)
EVENT_FILE = re.compile(r"events_g(\d+)_r(\d+)\.jsonl$")


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_jsonl(path):
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
    except (OSError, json.JSONDecodeError):
        return []
    return rows


def time_of(row):
    value = row.get("time")
    return float(value) if is_number(value) else None


def quantile(values, probability):
    """Linear interpolation, matching the session analyzer's percentile convention."""
    values = sorted(float(value) for value in values if is_number(value))
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def distribution(values):
    values = [float(value) for value in values if is_number(value)]
    if not values:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p95": quantile(values, 0.95),
        "max": max(values),
    }


def phase_duration(row):
    for key in ("seconds", "duration_seconds", "elapsed_seconds", "duration"):
        if is_number(row.get(key)):
            return float(row[key])
    starts = ("start_time", "started_at", "start", "begin_time")
    ends = ("end_time", "finished_at", "end", "finish_time")
    start = next((row[key] for key in starts if is_number(row.get(key))), None)
    end = next((row[key] for key in ends if is_number(row.get(key))), None)
    if start is not None and end is not None and end >= start:
        return float(end - start)
    return None


def phase_label(row):
    for key in ("name", "phase", "stage", "event"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return "unspecified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()
    root = args.session_dir

    controller = read_jsonl(root / "controller.jsonl")
    supervisors = {}
    for path in root.glob("node*/supervisor.jsonl"):
        match = re.fullmatch(r"node(\d+)", path.parent.name)
        if match:
            supervisors[int(match.group(1))] = read_jsonl(path)

    events = []
    for path in sorted(root.glob("node*/events_g*_r*.jsonl")):
        file_match = EVENT_FILE.fullmatch(path.name)
        node_match = re.fullmatch(r"node(\d+)", path.parent.name)
        if not file_match or not node_match:
            continue
        rid = int(node_match.group(1))
        file_gen, file_rank = map(int, file_match.groups())
        for row in read_jsonl(path):
            row_rid = row.get("rid", rid)
            row_gen = row.get("generation", file_gen)
            row_rank = row.get("rank", file_rank)
            if all(isinstance(value, int) and not isinstance(value, bool)
                   for value in (row_rid, row_gen, row_rank)):
                events.append((row, row_rid, row_gen, row_rank))

    by_kind = defaultdict(list)
    for row in controller:
        by_kind[row.get("kind")].append(row)
    injections = {row.get("index"): row for row in by_kind["injected"]
                  if isinstance(row.get("index"), int)}
    restarts = {row.get("index"): row for row in by_kind["restart_requested"]
                if isinstance(row.get("index"), int)}
    controller_restore = {}
    for row in controller:
        match = re.fullmatch(r"fault_(\d+)_restored", str(row.get("kind", "")))
        if match:
            controller_restore[int(match.group(1))] = row

    per_fault = []
    for index in sorted(injections):
        injection = injections[index]
        restart = restarts.get(index)
        target = injection.get("target")
        old_generation = injection.get("generation")
        new_generation = restart.get("generation") if restart else None
        fault_time = time_of(injection)
        restart_time = time_of(restart) if restart else None

        active = controller_restore.get(index, {}).get("generations")
        if not (isinstance(active, list) and len(active) == 2
                and all(isinstance(value, int) for value in active)):
            active = None
        if active is None and target in (0, 1) and isinstance(old_generation, int):
            active = [old_generation, old_generation]
            if isinstance(new_generation, int):
                active[target] = new_generation

        target_events = [item for item in events
                         if item[1] == target and item[2] == new_generation]
        target_ranks = set(range(8))
        rank_ids = {rank for _, rid, gen, rank in events if rid == target and gen == new_generation}
        if not rank_ids:
            target_ranks = set()
        ready_times = [time_of(row) for row, _, _, rank in target_events
                       if row.get("kind") == "ready" and rank in target_ranks]
        restore_rows = [row for row, _, _, rank in target_events
                        if row.get("kind") == "restored" and rank in target_ranks]
        all_ready = max(ready_times) if len(ready_times) == len(target_ranks) and ready_times else None
        restore_times = [time_of(row) for row in restore_rows if time_of(row) is not None]
        all_restored = max(restore_times) if len(restore_rows) == len(target_ranks) and len(restore_times) == len(target_ranks) else None

        launch_rows = [row for row in supervisors.get(target, [])
                       if row.get("kind") == "launched" and row.get("generation") == new_generation]
        launch = time_of(launch_rows[-1]) if launch_rows else None

        joint_time = None
        if active is not None and all_restored is not None:
            joint_commits = defaultdict(dict)
            for row, rid, generation, rank in events:
                if (rid in (0, 1) and generation == active[rid] and rank in range(8)
                        and row.get("kind") == "commit" and row.get("participants") == 2
                        and time_of(row) is not None and time_of(row) >= all_restored):
                    joint_commits[row.get("step")][(rid, rank)] = time_of(row)
            complete = [(max(times.values()), step) for step, times in joint_commits.items()
                        if len(times) == 16]
            if complete:
                joint_time = min(complete)[0]

        solo_time = None
        if target in (0, 1) and isinstance(old_generation, int) and fault_time is not None:
            survivor = 1 - target
            survivor_generation = active[survivor] if active is not None else old_generation
            solo_commits = defaultdict(dict)
            for row, rid, generation, rank in events:
                if (rid == survivor and generation == survivor_generation and rank in range(8)
                        and row.get("kind") == "commit" and row.get("participants") == 1
                        and time_of(row) is not None and time_of(row) >= fault_time):
                    solo_commits[row.get("step")][rank] = time_of(row)
            complete = [max(times.values()) for times in solo_commits.values() if len(times) == 8]
            if complete:
                solo_time = min(complete)

        def delta(end, start):
            return round(end - start, 6) if is_number(end) and is_number(start) and end >= start else None

        per_fault.append({
            "index": index,
            "target": target,
            "fault_to_restart_requested": delta(restart_time, fault_time),
            "target_launch_to_all_ready": delta(all_ready, launch),
            "all_ready_to_all_restored": delta(all_restored, all_ready),
            "all_restored_to_first_common_commit": delta(joint_time, all_restored),
            "fault_to_first_common_commit": delta(joint_time, fault_time),
            "fault_to_solo": delta(solo_time, fault_time),
        })

    stages = {name: distribution([fault[name] for fault in per_fault]) for name in FAULT_STAGES}
    timings = defaultdict(list)
    for row, _, _, _ in events:
        kind = row.get("kind")
        if kind in ("transport_send", "transport_recv") and is_number(row.get("seconds")):
            timings[kind].append(float(row["seconds"]))
        elif kind == "restored":
            for key in ("load_seconds", "audit_seconds"):
                if is_number(row.get(key)):
                    timings[key].append(float(row[key]))

    phase_timings = defaultdict(list)
    phase_counts = defaultdict(int)
    for row, _, _, _ in events:
        if row.get("kind") == "phase":
            name = phase_label(row)
            phase_counts[name] += 1
            duration = phase_duration(row)
            if duration is not None:
                phase_timings[name].append(duration)

    output = {
        "session": root.name,
        "clock_note": "跨节点 wall clock 未校验；跨节点故障/共同提交端到端时长可能含时钟偏差。",
        "sample_counts": {
            "faults": len(per_fault),
            "fault_to_solo": sum(fault["fault_to_solo"] is not None for fault in per_fault),
        },
        "per_fault": per_fault,
        "stages": stages,
        "rank_timings": {name: distribution(values) for name, values in sorted(timings.items())},
    }
    if phase_counts:
        output["startup_phase_events"] = {
            name: {"events": phase_counts[name], **distribution(phase_timings[name])}
            for name in sorted(phase_counts)
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
