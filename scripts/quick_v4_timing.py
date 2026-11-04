"""Audit and extract compact post-run timing evidence for a V4 switch run.

Run on node0 after the remote raw directory has been collected. This script is
CPU-only; it never starts, prepares, or commits a training worker.
"""
import argparse
import json
import math
import re
import time
from pathlib import Path


_ACTOR_FILE = re.compile(r"^(g\d+_[ABCDE])(\d+)\.jsonl$")
_STAGE_KINDS = {
    "standby_imported", "standby_cpu_model_ready", "standby_gpu_warmed",
    "state_channel_preparing", "state_channel_prepared", "state_channel_connected",
    "switch_channel_prepared", "boundary_selected", "freeze", "snapshot_ready",
    "state_sent", "state_received", "transport_send", "transport_recv",
    "restored", "connected", "groups_closed", "outer_ready", "outer_restored",
    "outer_retained", "candidate_prepared", "candidate_validated", "candidate_rejected",
    "candidate_activated", "activated", "source_recovery_selected", "switch_decision",
    "retired", "source_stalled", "peer_compute_ready", "done",
}
_DURATION_CLOCK = {
    "transport_send": "worker-local perf_counter duration",
    "transport_recv": "worker-local perf_counter duration",
}


def _read_json(path):
    return json.loads(path.read_text())


def _finite_number(value):
    return (type(value) in (int, float) and math.isfinite(value))


def _load_worker_events(root):
    workers = {}
    for path in sorted(root.glob("g*_*.jsonl")):
        match = _ACTOR_FILE.match(path.name)
        if not match:
            continue
        worker = match.group(1) + match.group(2)
        events = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError(f"event must be an object: {path.name}:{line_number}")
            events.append(event)
        workers[worker] = events
    if not workers:
        raise ValueError(f"no transactional actor JSONL files found under {root}")
    return workers


def _event_timing(worker, event, local_origin):
    kind = event.get("kind")
    monotonic = event.get("monotonic")
    seconds = event.get("seconds")
    row = {key: event[key] for key in (
        "kind", "role", "rank", "replica", "device", "round", "actor_round",
        "step", "phase", "epoch", "outer_epoch", "monotonic", "seconds",
    ) if key in event}
    row["worker"] = worker
    row["clock_scope"] = "worker-local monotonic; compare only within this worker/host"
    if _finite_number(monotonic) and _finite_number(local_origin):
        row["since_local_origin_seconds"] = monotonic - local_origin
    if _finite_number(monotonic):
        row["end_monotonic"] = monotonic
    if _finite_number(seconds):
        row["duration_seconds"] = seconds
        row["duration_clock"] = _DURATION_CLOCK.get(kind, "worker-local monotonic instrumentation")
        # A number of events store elapsed time and emit a monotonic endpoint,
        # not a start timestamp. Keep the derived start explicitly approximate.
        row["start_monotonic_estimate"] = monotonic - seconds if _finite_number(monotonic) else None
        row["start_estimated_from_duration"] = True
    return row


def _logical_after_commits(workers):
    by_step = {}
    for worker, events in workers.items():
        for event in events:
            if event.get("kind") != "commit" or event.get("round") != 0 or event.get("phase") != "after":
                continue
            if type(event.get("step")) is not int:
                continue
            replica, rank = event.get("replica"), event.get("rank")
            if type(replica) is int and type(rank) is int:
                by_step.setdefault(event["step"], {}).setdefault((replica, rank), []).append((worker, event))
    shared = sorted(step for step, commits in by_step.items() if len(commits) == 16)
    if not shared:
        return None, by_step
    step = shared[0]
    return {
        "step": step,
        "phase": "after",
        "round": 0,
        "logical_worker_count": len(by_step[step]),
        "logical_workers": [f"{replica}/{rank}" for replica, rank in sorted(by_step[step])],
        "all_workers_have_commit": True,
    }, by_step


def _stats(values):
    valid = [value for value in values if _finite_number(value)]
    if not valid:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {"count": len(valid), "mean": sum(valid) / len(valid),
            "min": min(valid), "max": max(valid)}


def _classify_commit_samples(workers, commits, summary, common):
    request = summary.get("migration_request")
    request_mono = request.get("monotonic") if isinstance(request, dict) else None
    request_step = request.get("observed_step") if isinstance(request, dict) else None
    cut = next((event.get("step") for events in workers.values() for event in events
                if event.get("kind") == "boundary_selected" and event.get("round") == 0), None)
    if type(cut) is not int:
        cut = summary.get("plan", {}).get("cut")
    first_step = (common or {}).get("step")
    values = {name: [] for name in (
        "normal_pre_request", "standby_preparing", "first_recovered", "subsequent_after")}
    basis_by_worker = {}
    stage_by_key = {}
    for worker, events in workers.items():
        role = worker.split("_")[-1][0]
        same_node0 = role in (("A", "B") if summary.get("unified") else ("A", "D"))
        basis_by_worker[worker] = ("same-host monotonic relative to node0 migration_request"
                                   if same_node0 else
                                   "logical step relative to node0 request.observed_step and cut")
        for event in events:
            if event.get("kind") != "commit":
                continue
            if event.get("round") == 0 and event.get("phase") == "before":
                step = event.get("step")
                if same_node0 and _finite_number(request_mono) and _finite_number(event.get("monotonic")):
                    if event["monotonic"] <= request_mono:
                        stage = "normal_pre_request"
                    elif type(cut) is int and type(step) is int and step <= cut:
                        stage = "standby_preparing"
                    else:
                        stage = None
                elif type(step) is int and type(request_step) is int and step <= request_step:
                    stage = "normal_pre_request"
                elif type(step) is int and type(request_step) is int and type(cut) is int and request_step < step <= cut:
                    stage = "standby_preparing"
                else:
                    stage = None
            elif event.get("phase") == "after" and type(event.get("step")) is int and first_step is not None:
                if event.get("round") == 0 and event.get("step") == first_step:
                    stage = "first_recovered"
                elif event.get("step") > first_step:
                    stage = "subsequent_after"
                else:
                    stage = None
            else:
                stage = None
            if stage:
                stage_by_key[(worker, event.get("round"), event.get("step"),
                              event.get("phase"), event.get("monotonic"))] = stage
                values[stage].append((worker, event))
    for row in commits:
        key = (row.get("worker"), row.get("round"), row.get("step"),
               row.get("phase"), row.get("monotonic"))
        if key in stage_by_key:
            row["timing_stage"] = stage_by_key[key]

    result = {}
    for stage, rows in values.items():
        by_worker = {}
        for worker, event in rows:
            sample = by_worker.setdefault(worker, {"seconds": [], "commit_gap": [], "commit_count": 0})
            sample["commit_count"] += 1
            sample["seconds"].append(event.get("seconds"))
            sample["commit_gap"].append(event.get("commit_gap"))
        result[stage] = {
            "commit_count": len(rows),
            "seconds": _stats([event.get("seconds") for _, event in rows]),
            "commit_gap": _stats([event.get("commit_gap") for _, event in rows]),
            "classification_basis_by_worker": {worker: basis_by_worker[worker] for worker in sorted(by_worker)},
            "per_worker": {worker: {"commit_count": sample["commit_count"],
                "seconds": _stats(sample["seconds"]), "commit_gap": _stats(sample["commit_gap"])}
                for worker, sample in sorted(by_worker.items())},
        }
    return result


def _first_event(events, kind, *, round_id=0, step=None, epoch=None):
    for event in events:
        if event.get("kind") != kind:
            continue
        if round_id is not None and event.get("round") != round_id:
            continue
        if step is not None and event.get("step") != step:
            continue
        if epoch is not None and event.get("epoch", event.get("outer_epoch")) != epoch:
            continue
        return event
    return None


def _node0_a0_timing(workers, request, common):
    events = workers.get("g0_A0", [])
    request_mono = request.get("monotonic") if isinstance(request, dict) else None
    cut = None
    for event in events:
        if event.get("kind") == "boundary_selected" and event.get("round") == 0:
            cut = event.get("step")
            break
    stage_kinds = {
        "boundary_selected", "snapshot_ready", "freeze", "outer_retained",
        "candidate_activated", "activated", "connected", "outer_ready",
        "outer_restored", "commit", "done",
    }
    selected = [event for event in events if event.get("kind") in stage_kinds and
                event.get("round", 0) == 0 and
                (event.get("kind") not in ("boundary_selected", "snapshot_ready", "freeze") or
                 cut is None or event.get("step") == cut)]
    timeline = []
    for event in selected:
        row = _event_timing("g0_A0", event, None)
        if _finite_number(request_mono) and _finite_number(event.get("monotonic")):
            row["since_migration_request_seconds"] = event["monotonic"] - request_mono
        timeline.append(row)

    def delta(start, end):
        if not isinstance(start, dict) or not isinstance(end, dict):
            return None
        a, b = start.get("monotonic"), end.get("monotonic")
        return b - a if _finite_number(a) and _finite_number(b) else None

    boundary = _first_event(events, "boundary_selected", step=cut)
    snapshot = _first_event(events, "snapshot_ready", step=cut)
    frozen = _first_event(events, "freeze", step=cut)
    activation = _first_event(events, "activated", step=cut)
    first_commit = None
    if common:
        first_commit = next((e for e in events if e.get("kind") == "commit" and
            e.get("round") == 0 and e.get("phase") == "after" and e.get("step") == common["step"]), None)
    connected = next((e for e in events if e.get("kind") == "connected" and e.get("epoch") == 1), None)
    proxy_seconds = None
    if first_commit and _finite_number(request_mono) and _finite_number(first_commit.get("monotonic")):
        proxy_seconds = first_commit["monotonic"] - request_mono
    return {
        "worker": "g0_A0",
        "clock_scope": "node0 worker/controller monotonic; same-host differences only",
        "cut_step": cut,
        "first_recovery_common_step": (common or {}).get("step"),
        "first_recovery_commit_proxy": _event_timing("g0_A0", first_commit, None) if first_commit else None,
        "request_to_a0_commit_proxy_seconds": proxy_seconds,
        "a0_proxy_semantics": "A0 commit at the first step independently confirmed on all 16 logical ranks; this is a single-worker proxy, not the slowest-rank timestamp",
        "stages": {
            "request_to_boundary_seconds": (boundary["monotonic"] - request_mono
                if boundary and _finite_number(boundary.get("monotonic")) and _finite_number(request_mono) else None),
            "boundary_to_snapshot_event_seconds": delta(boundary, snapshot),
            "snapshot_reported_seconds": snapshot.get("seconds") if snapshot else None,
            "snapshot_event_to_freeze_seconds": delta(snapshot, frozen),
            "a0_rebuild_connected_seconds": connected.get("seconds") if connected else None,
            "activation_to_first_after_commit_seconds": delta(activation, first_commit),
        },
        "timeline": timeline,
    }


def _observation(summary, root):
    summary_value = summary.get("first_common_observation")
    file_value = None
    observation_path = root / "first_common_observation.json"
    if observation_path.exists():
        file_value = _read_json(observation_path)
    selected = summary_value if isinstance(summary_value, dict) else file_value
    return selected, {
        "selected_from": "summary.json" if isinstance(summary_value, dict) else
            "first_common_observation.json" if isinstance(file_value, dict) else None,
        "summary_value_present": isinstance(summary_value, dict),
        "file_value_present": isinstance(file_value, dict),
        "summary_file_match": (summary_value == file_value if isinstance(summary_value, dict) and
                                isinstance(file_value, dict) else None),
    }


def _timing_evidence(root, summary, workers):
    commits = []
    phase_events = []
    for worker, events in sorted(workers.items()):
        boundaries = [e for e in events if e.get("kind") == "boundary_selected" and e.get("round") == 0]
        origin_event = boundaries[0] if boundaries else next((e for e in events if e.get("round") == 0), None)
        local_origin = origin_event.get("monotonic") if origin_event else None
        previous_monotonic = None
        for event in events:
            monotonic = event.get("monotonic")
            since_previous = (monotonic - previous_monotonic
                              if _finite_number(monotonic) and _finite_number(previous_monotonic) else None)
            if event.get("kind") == "commit":
                row = _event_timing(worker, event, local_origin)
                for key in ("step", "phase", "seconds", "commit_gap"):
                    if key in event:
                        row[key] = event[key]
                if since_previous is not None:
                    row["seconds_since_previous_logged_event"] = since_previous
                commits.append(row)
            elif event.get("kind") in _STAGE_KINDS or "seconds" in event:
                row = _event_timing(worker, event, local_origin)
                if since_previous is not None:
                    row["seconds_since_previous_logged_event"] = since_previous
                    row["previous_event_interval_only"] = True
                phase_events.append(row)
            if _finite_number(monotonic):
                previous_monotonic = monotonic
    commits.sort(key=lambda x: (x.get("round", -1), x.get("replica", -1),
                               x.get("rank", -1), x.get("step", -1)))

    common, by_step = _logical_after_commits(workers)
    commit_stage_samples = _classify_commit_samples(workers, commits, summary, common)
    request = summary.get("migration_request")
    observation, observation_meta = _observation(summary, root)
    observed_step = observation.get("first_joint_step") if isinstance(observation, dict) else None
    observed_mono = observation.get("observed_monotonic") if isinstance(observation, dict) else None
    request_mono = request.get("monotonic") if isinstance(request, dict) else None
    observation_matches_events = (common is not None and observed_step == common["step"])
    observed_elapsed = None
    if observation_matches_events and _finite_number(request_mono) and _finite_number(observed_mono):
        candidate = observed_mono - request_mono
        if candidate >= 0:
            observed_elapsed = candidate

    transfer = [event for event in phase_events if event.get("kind") in (
        "state_sent", "state_received", "transport_send", "transport_recv",
        "switch_channel_prepared", "restored", "connected", "outer_ready",
        "outer_restored", "candidate_prepared", "candidate_validated",
        "candidate_activated", "activated", "source_recovery_selected", "switch_decision",
    )]
    coverage = None
    if common:
        step_events = by_step[common["step"]]
        coverage = {
            "logical_workers": 16,
            "observed_worker_keys": common["logical_workers"],
            "worker_keys_complete": len(step_events) == 16,
            "commit_records": sum(len(rows) for rows in step_events.values()),
            "common_step": common["step"],
        }

    return {
        "migration_request": ({key: request[key] for key in ("monotonic", "observed_step") if key in request}
                              if isinstance(request, dict) else None),
        "first_recovery_common_commit": common,
        "common_commit_coverage": coverage,
        "controller_observation": observation,
        "controller_observation_metadata": observation_meta,
        "controller_observation_matches_event_logs": observation_matches_events,
        "request_to_first_common_commit_seconds": observed_elapsed,
        "request_to_first_common_commit_clock_scope": "node0 controller monotonic; includes observer polling and collection latency",
        "request_to_first_common_commit_proxy_seconds": _node0_a0_timing(workers, request, common)["request_to_a0_commit_proxy_seconds"],
        "node0_A0": _node0_a0_timing(workers, request, common),
        "per_worker_commits": commits,
        "commit_stage_samples": commit_stage_samples,
        "phase_events": phase_events,
        "transfer_restore_rebuild_events_local_only": transfer,
        "event_count": sum(map(len, workers.values())),
        "actor_count": len(workers),
        "limitations": [
            "Cross-host worker monotonic timestamps are never subtracted or max-reduced; event completeness is checked by logical rank and step.",
            "Controller-observed end-to-end time uses node0 observer monotonic and includes its polling and collection latency.",
            "A0 timing is explicitly a same-node proxy for the first 16-rank common step, not the slowest rank.",
            "Snapshot/restore hashes, transport checks, event writes, and audit hashing add overhead; audit pass durations are reported separately.",
            "Two quick repetitions are diagnostic and do not support statistical inference or a significance claim.",
        ],
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n")


def run(root, output=None):
    root = Path(root)
    summary_path = root / "summary.json"
    summary = _read_json(summary_path)
    if not summary.get("transactional_switch"):
        raise ValueError("transactional V4 raw directory required")

    from scripts.verify_v4_outer import verify as verify_outer
    from scripts.verify_v4_transaction import verify as verify_transaction
    from scripts.summarize_v4 import summarize as summarize_v4

    audit_runs = []
    audit_results = []
    for name, verifier in (("verify_v4_outer.verify", verify_outer),
                           ("verify_v4_transaction.verify", verify_transaction)):
        started = time.monotonic()
        try:
            result = verifier(root)
        except Exception as exc:
            result = {"pass_all": False, "errors": [type(exc).__name__ + ": " + str(exc)]}
        ended = time.monotonic()
        audit_results.append(result)
        audit_runs.append({"name": name, "start_monotonic": started,
                           "end_monotonic": ended, "seconds": ended - started,
                           "pass_all": result.get("pass_all") is True})

    outer_result, transaction_result = audit_results
    _write_json(root / "independent_outer_audit.json", outer_result)
    _write_json(root / "independent_transaction_audit.json", transaction_result)
    same_audit = (outer_result == transaction_result)
    audit_status = {
        "runs": audit_runs,
        "outer_and_transaction_results_match": same_audit,
        "both_pass": all(run_item["pass_all"] for run_item in audit_runs),
        "note": "verify_v4_outer.verify dispatches transactional runs to verify_v4_transaction.verify; the direct second call is retained as an explicit consistency check.",
    }
    if not audit_status["both_pass"]:
        raise ValueError("independent V4 audit failed; see independent_outer_audit.json and independent_transaction_audit.json")
    if not same_audit:
        raise ValueError("outer and transaction audit results disagree")

    summarize_started = time.monotonic()
    functional = summarize_v4(root)
    summarize_ended = time.monotonic()
    extraction_started = time.monotonic()
    workers = _load_worker_events(root)
    timing = _timing_evidence(root, summary, workers)
    extraction_ended = time.monotonic()
    timing["post_run_costs"] = {
        "outer_verify_seconds": audit_runs[0]["seconds"],
        "transaction_verify_seconds": audit_runs[1]["seconds"],
        "summarize_seconds": summarize_ended - summarize_started,
        "timing_extraction_seconds": extraction_ended - extraction_started,
        "clock_scope": "node0 controller monotonic",
        "note": "Each independent verifier and summarize_v4 performs raw reads/hashing; these CPU/I/O costs are post-run and excluded from worker commit timings.",
    }
    result = {
        "schema": 1,
        "experiment": "V4 transactional switch quick functional and timing evidence",
        "raw_directory_name": root.name,
        "functional": functional,
        "audit": audit_status,
        "timing": timing,
    }
    output = Path(output) if output is not None else root / "quick_v4_timing.json"
    _write_json(output, result)
    return {"output": str(output), "functional_pass": functional.get("functional_pass"),
            "resources_clean": functional.get("resources_clean"),
            "first_common_step": (timing.get("first_recovery_common_commit") or {}).get("step"),
            "request_to_first_common_commit_seconds": timing.get("request_to_first_common_commit_seconds"),
            "audits_pass": audit_status["both_pass"], "output_bytes": output.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_directory", type=Path)
    parser.add_argument("--output", type=Path,
                        help="result JSON path (default: RAW_DIRECTORY/quick_v4_timing.json)")
    args = parser.parse_args()
    print(json.dumps(run(args.raw_directory, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
