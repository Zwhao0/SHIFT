#!/usr/bin/env python3
"""Verify one fail-stop recovery evidence bundle (stdlib only)."""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


RANKS = range(8)
HASH_KEYS = ("model", "optimizer", "scheduler")


def expected_files(root):
    files = [root / "controller.jsonl"]
    files += [root / "node0" / f"events_g0_r{r}.jsonl" for r in RANKS]
    files += [
        root / "node1" / f"events_g{g}_r{r}.jsonl"
        for g in (0, 1)
        for r in RANKS
    ]
    return files


def parse_time(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def as_int(value):
    if isinstance(value, bool):
        return None
    try:
        n = int(value)
        return n if str(value).strip() == str(n) or isinstance(value, int) else None
    except (TypeError, ValueError):
        return None


def generation_number(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"g?(\d+)", value.strip(), re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def valid_hashes(value):
    return isinstance(value, dict) and all(
        key in value and value[key] is not None for key in HASH_KEYS
    )


def hash_values(value):
    return {key: value[key] for key in HASH_KEYS} if valid_hashes(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path, help="evidence session directory")
    args = parser.parse_args()
    root = args.session_dir

    checks = {}
    metrics = {}
    counts = {"files_expected": 25, "events_total": 0, "by_kind": {}}
    paths = expected_files(root)
    missing = [str(path) for path in paths if not path.is_file()]
    checks["required_files_present"] = {"pass": not missing, "missing": missing}

    events_by_file = {}
    parse_errors = []
    for path in paths:
        parsed = []
        if path.is_file():
            with path.open("r", encoding="utf-8") as stream:
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict) or not isinstance(
                            event.get("kind"), str
                        ):
                            raise ValueError("expected object with string kind")
                        parsed.append(event)
                        counts["events_total"] += 1
                        kind = event["kind"]
                        counts["by_kind"][kind] = counts["by_kind"].get(kind, 0) + 1
                    except (json.JSONDecodeError, ValueError) as exc:
                        parse_errors.append(
                            {"file": str(path), "line": line_no, "error": str(exc)}
                        )
        events_by_file[path] = parsed
    checks["jsonl_valid"] = {"pass": not parse_errors, "errors": parse_errors}

    common_fields = {"kind", "time", "rid", "rank", "generation", "pid"}
    event_fields = {
        "commit": {"step", "participants", "model_hash", "optimizer_hash", "scheduler_hash", "updates"},
        "snapshot": {"step", "hashes", "optimizer_tensors"},
        "restored": {"step", "source_replica", "source_rank", "hashes_match", "optimizer_tensors"},
    }
    schema_errors = []
    for path, records in events_by_file.items():
        for i, event in enumerate(records, 1):
            required = common_fields | event_fields.get(event["kind"], set())
            absent = sorted(required - event.keys())
            if absent or parse_time(event.get("time")) is None:
                schema_errors.append({"file": str(path), "event": i, "missing": absent,
                                      "invalid_time": parse_time(event.get("time")) is None})
    checks["event_schema_valid"] = {"pass": not schema_errors, "errors": schema_errors}

    def events(path, kind=None):
        result = events_by_file.get(path, [])
        return result if kind is None else [e for e in result if e.get("kind") == kind]

    def path_for(node, generation, rank):
        if node == "node0":
            return root / "node0" / f"events_g0_r{rank}.jsonl"
        return root / "node1" / f"events_g{generation}_r{rank}.jsonl"

    # Validate events against their file identities to prevent rank/generation mixups.
    identity_errors = []
    for path, records in events_by_file.items():
        match = re.search(r"events_g(\d+)_r(\d+)\.jsonl$", path.name)
        if not match:
            continue
        gen, rank = map(int, match.groups())
        for i, event in enumerate(records, 1):
            if "rank" in event and as_int(event["rank"]) != rank:
                identity_errors.append({"file": str(path), "event": i, "field": "rank"})
            if "generation" in event and generation_number(event["generation"]) != gen:
                identity_errors.append(
                    {"file": str(path), "event": i, "field": "generation"}
                )
    checks["event_identity_matches_file"] = {
        "pass": not identity_errors,
        "errors": identity_errors,
    }

    controller = events(root / "controller.jsonl")
    injected = [e for e in controller if e.get("kind") == "injected"]
    inject_time = parse_time(injected[0].get("time")) if injected else None
    checks["exactly_one_injection"] = {
        "pass": len(injected) == 1 and inject_time is not None,
        "count": len(injected),
        "time": injected[0].get("time") if injected else None,
    }
    metrics["injection_time"] = injected[0].get("time") if injected else None

    def timed_commits(path, participants, before):
        found = []
        for event in events(path, "commit"):
            event_time = parse_time(event.get("time"))
            if (
                inject_time is not None
                and event_time is not None
                and ((event_time < inject_time) if before else (event_time > inject_time))
                and as_int(event.get("participants")) == participants
            ):
                found.append(event)
        return found

    pre = {}
    post_single = {}
    for node in ("node0", "node1"):
        for rank in RANKS:
            pre[f"{node}_rank{rank}"] = len(timed_commits(path_for(node, 0, rank), 2, True))
    for rank in RANKS:
        p = path_for("node0", 0, rank)
        post_single[rank] = len(timed_commits(p, 1, False))
    checks["pre_failure_two_replica_commits"] = {
        "pass": inject_time is not None and all(n >= 3 for n in pre.values()),
        "min_required": 3,
        "by_rank": pre,
    }
    checks["post_failure_survivor_commits"] = {
        "pass": inject_time is not None and all(n >= 3 for n in post_single.values()),
        "min_required": 3,
        "by_rank": post_single,
    }
    metrics["pre_failure_participants2_by_rank"] = pre
    metrics["post_failure_participants1_by_rank"] = post_single

    restore_metrics = {}
    restore_steps = {}
    restore_ok = True
    for rank in RANKS:
        target = events(path_for("node1", 1, rank), "restored")
        per_rank = []
        if not target:
            restore_ok = False
            restore_metrics[rank] = {"count": 0, "valid": 0}
            continue
        for event in target:
            step = as_int(event.get("step"))
            src_rank = as_int(event.get("source_rank"))
            src_replica = event.get("source_replica")
            node0_records = events(path_for("node0", 0, rank))
            node0_rids = {
                str(e["rid"]) for e in node0_records if e.get("rid") is not None
            }
            src_replica_text = str(src_replica).strip().lower()
            source_is_node0 = (
                src_replica_text in {"0", "g0", "node0", "node0/g0", "replica0"}
                or str(src_replica) in node0_rids
            )
            valid = (
                step is not None
                and step > 0
                and inject_time is not None
                and parse_time(event.get("time")) is not None
                and parse_time(event.get("time")) > inject_time
                and event.get("hashes_match") is True
                and as_int(event.get("optimizer_tensors")) is not None
                and as_int(event.get("optimizer_tensors")) > 0
                and src_rank == rank
                and source_is_node0
            )
            candidates = [
                e
                for e in events(path_for("node0", 0, rank), "snapshot")
                if as_int(e.get("step")) == step
                and valid_hashes(e.get("hashes"))
            ]
            snap = candidates[-1] if candidates else None
            if snap is None:
                valid = False
            else:
                snap_hashes = snap["hashes"]
                for key in ("expected_hashes", "actual_hashes", "hashes"):
                    extra = event.get(key)
                    if extra is not None and hash_values(extra) != hash_values(snap_hashes):
                        valid = False
                expected = event.get("expected_hashes")
                actual = event.get("actual_hashes")
                if expected is not None and actual is not None and hash_values(expected) != hash_values(actual):
                    valid = False
            per_rank.append({
                "step": step,
                "valid": valid,
                "snapshot_found": snap is not None,
                "source_replica": src_replica,
                "source_replica_is_node0": source_is_node0,
            })
            if not valid:
                restore_ok = False
            elif step is not None:
                restore_steps[rank] = max(restore_steps.get(rank, 0), step)
        restore_metrics[rank] = {"count": len(target), "valid": sum(x["valid"] for x in per_rank), "events": per_rank}
        if len(restore_steps) < rank + 1:
            restore_ok = False
    checks["restored_state_matches_snapshot"] = {
        "pass": restore_ok,
        "by_rank": restore_metrics,
    }
    metrics["restored_step_by_rank"] = restore_steps
    checks["recovery_snapshot_step_consistent"] = {
        "pass": len(restore_steps) == 8 and len(set(restore_steps.values())) == 1,
        "steps": restore_steps,
    }

    joint = {}
    joint_pass = True
    for rank in RANKS:
        base_step = restore_steps.get(rank)
        if base_step is None:
            joint[rank] = {"common_steps": 0, "valid_steps": []}
            joint_pass = False
            continue
        left = {}
        right = {}
        for event in events(path_for("node0", 0, rank), "commit"):
            step = as_int(event.get("step"))
            if step is not None and step > base_step:
                left.setdefault(step, []).append(event)
        for event in events(path_for("node1", 1, rank), "commit"):
            step = as_int(event.get("step"))
            if step is not None and step > base_step:
                right.setdefault(step, []).append(event)
        common = sorted(set(left) & set(right))
        valid_steps = []
        for step in common:
            # Require a single unambiguous record per rank/step on both replicas.
            if len(left[step]) != 1 or len(right[step]) != 1:
                continue
            a, b = left[step][0], right[step][0]
            if (
                as_int(a.get("participants")) == 2
                and as_int(b.get("participants")) == 2
                and as_int(a.get("updates")) == step
                and as_int(b.get("updates")) == step
                and all(a.get(f"{k}_hash") == b.get(f"{k}_hash") for k in HASH_KEYS)
                and all(a.get(f"{k}_hash") is not None for k in HASH_KEYS)
            ):
                valid_steps.append(step)
        joint[rank] = {"common_steps": len(common), "valid_steps": valid_steps}
        if len(valid_steps) < 3:
            joint_pass = False
    checks["post_recovery_common_commits_match"] = {
        "pass": joint_pass,
        "min_required": 3,
        "by_rank": joint,
    }
    metrics["post_recovery_valid_common_steps_by_rank"] = {
        rank: data["valid_steps"] for rank, data in joint.items()
    }

    pid_metrics = {}
    pid_ok = True
    for rank in RANKS:
        p = path_for("node0", 0, rank)
        ready = events(p, "ready")
        commits = events(p, "commit")
        ready_pids = {str(e.get("pid")) for e in ready if e.get("pid") is not None}
        commit_pids = {str(e.get("pid")) for e in commits if e.get("pid") is not None}
        valid = bool(ready_pids) and len(ready_pids) == 1 and all(
            e.get("pid") is not None and str(e.get("pid")) in ready_pids for e in commits
        )
        pid_metrics[rank] = {
            "ready_count": len(ready),
            "commit_count": len(commits),
            "ready_pids": sorted(ready_pids),
            "commit_pids": sorted(commit_pids),
        }
        pid_ok = pid_ok and valid
    checks["healthy_node0_pid_stable"] = {"pass": pid_ok, "by_rank": pid_metrics}

    numeric_ok = True
    changed = {}
    for node, gen in (("node0", 0), ("node1", 0), ("node1", 1)):
        for rank in RANKS:
            path = path_for(node, gen, rank)
            commits = events(path, "commit")
            numeric_ok = numeric_ok and bool(commits) and all(
                e.get("finite_gradients") is True and e.get("cross_calls", 0) > 0
                and e.get("optimizer_tensors", 0) > 0 and e.get("updates") == e.get("step")
                for e in commits)
            ready = events(path, "ready")
            initial_hash = ready[0].get("hashes", {}).get("model") if ready else None
            changed[f"{node}_g{gen}_r{rank}"] = bool(initial_hash) and any(
                e.get("model_hash") != initial_hash for e in commits)
    checks["committed_updates_finite_and_nonempty"] = {"pass": numeric_ok}
    checks["all_model_shards_changed"] = {"pass": all(changed.values()), "by_rank": changed}

    cleanups = {}
    for node in ("node0", "node1"):
        path = root / node / "supervisor.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        cleanups[node] = [e for e in rows if e.get("kind") == "cleanup"]
    checks["supervisors_cleaned_up"] = {
        "pass": all(rows and not rows[-1].get("own_children") for rows in cleanups.values()),
        "by_node": cleanups,
    }

    if inject_time is not None:
        first_solo = [parse_time(e.get("time"))
                      for r in RANKS for e in timed_commits(path_for("node0", 0, r), 1, False)]
        restore_times = [parse_time(e.get("time")) for r in RANKS
                         for e in events(path_for("node1", 1, r), "restored")]
        restarts = [parse_time(e.get("time")) for e in controller if e.get("kind") == "restart_requested"]
        metrics["kill_to_first_solo_commit_seconds"] = min(first_solo) - inject_time if first_solo else None
        metrics["kill_to_all_restored_seconds"] = max(restore_times) - inject_time if len(restore_times) >= 8 else None
        metrics["restart_request_to_all_restored_seconds"] = max(restore_times) - restarts[0] if restarts and len(restore_times) >= 8 else None

    result = {
        "pass": all(item["pass"] for item in checks.values()),
        "checks": checks,
        "metrics": metrics,
        "counts": counts,
    }
    report_path = root / "report.json"
    try:
        report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"FAIL: cannot write {report_path}: {exc}")
        return 1

    failed = [name for name, item in checks.items() if not item["pass"]]
    status = "PASS" if result["pass"] else "FAIL"
    print(f"{status}: {len(checks) - len(failed)}/{len(checks)} checks; events={counts['events_total']}; report={report_path}")
    if failed:
        print("failed_checks=" + ",".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
