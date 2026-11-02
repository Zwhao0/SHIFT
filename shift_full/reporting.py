"""CPU-only endurance and ABBA benchmark summaries from SHIFT JSONL evidence.

Usage: python -m shift_full.reporting --endurance SESSIONDIR [--bench-flat A B
--bench-shift A B] --output DIR. Matplotlib is imported only when a plot has
usable data. Missing values are omitted; failed or incomplete sessions are
never marked complete.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
import sys


EVENT_FILE = re.compile(r"events_g(\d+)_r(\d+)\.jsonl$")
NODE_DIR = re.compile(r"node(\d+)$")
RANKS_PER_NODE = 8
WORLD_RANKS = 16
FAILURE_KINDS = {"failed", "failure", "unexpected_exit", "supervisor_error", "runtime_error"}


def finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def event_time(row):
    value = row.get("time") if isinstance(row, dict) else None
    if finite_number(value):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.timestamp()
            except ValueError:
                return None
    return None


def read_jsonl(path):
    rows, errors = [], []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("row must be a JSON object")
                    rows.append(row)
                except (json.JSONDecodeError, ValueError) as exc:
                    errors.append(f"{path}:{line_no}: {exc}")
    except OSError as exc:
        errors.append(f"{path}: {exc}")
    return rows, errors


def flatten_losses(value):
    if finite_number(value):
        yield float(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_losses(item)


def quantile(values, fraction):
    values = sorted(float(value) for value in values if finite_number(value))
    if not values:
        return None
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def describe(values):
    values = [float(value) for value in values if finite_number(value)]
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": quantile(values, 0.95),
    }


def event_files(inputs):
    files = set()
    for root in inputs:
        root = Path(root)
        if not root.is_dir():
            continue
        files.update(root.glob("events_g*_r*.jsonl"))
        files.update(root.glob("node*/events_g*_r*.jsonl"))
        if NODE_DIR.fullmatch(root.name):
            files.update(root.glob("events_g*_r*.jsonl"))
    return sorted(files)


def physical_rank_for(path, row):
    match = EVENT_FILE.fullmatch(path.name)
    rank = int(match.group(2)) if match else row.get("rank")
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < 8:
        return None
    node_match = NODE_DIR.fullmatch(path.parent.name)
    if node_match:
        node = int(node_match.group(1))
    else:
        node = row.get("rid")
        if not isinstance(node, int) or isinstance(node, bool):
            physical = row.get("physical_rank")
            return physical if isinstance(physical, int) and 0 <= physical < WORLD_RANKS else None
    physical = node * RANKS_PER_NODE + rank
    return physical if 0 <= physical < WORLD_RANKS else None


def load_events(inputs):
    rows, errors, found_ranks, paths = [], [], set(), []
    for path in event_files(inputs):
        paths.append(str(path))
        match = EVENT_FILE.fullmatch(path.name)
        rank = int(match.group(2)) if match else None
        node_match = NODE_DIR.fullmatch(path.parent.name)
        if node_match and rank is not None:
            physical = int(node_match.group(1)) * RANKS_PER_NODE + rank
            if 0 <= physical < WORLD_RANKS:
                found_ranks.add(physical)
        data, parse_errors = read_jsonl(path)
        errors.extend(parse_errors)
        for row in data:
            physical = physical_rank_for(path, row)
            if physical is not None:
                found_ranks.add(physical)
            rows.append((row, physical, path))
    return rows, errors, sorted(found_ranks), paths


def load_controller(inputs):
    candidates = []
    for root in inputs:
        root = Path(root)
        if root.is_file() and root.name == "controller.jsonl":
            candidates.append(root)
            continue
        if root.is_dir():
            candidates.append(root / "controller.jsonl")
            node_match = NODE_DIR.fullmatch(root.name)
            if node_match:
                candidates.append(root.parent / "controller.jsonl")
            candidates.extend(root.glob("node*/controller.jsonl"))
    candidates = sorted({path for path in candidates if path.is_file()})
    if not candidates:
        return [], ["controller.jsonl is missing"], []
    # Prefer the session-level journal; node copies often duplicate it.
    session_level = [path for path in candidates if not NODE_DIR.fullmatch(path.parent.name)]
    use = session_level[:1] if session_level else candidates
    rows, errors, seen = [], [], set()
    for path in use:
        data, parse_errors = read_jsonl(path)
        errors.extend(parse_errors)
        for row in data:
            key = json.dumps(row, sort_keys=True, separators=(",", ":"))
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows, errors, [str(path) for path in use]


def failure_rows(controller, events):
    result = [row for row in controller if str(row.get("kind", "")).lower() in FAILURE_KINDS]
    result.extend(row for row, _, _ in events
                  if str(row.get("kind", "")).lower() in FAILURE_KINDS)
    return result


def controller_complete(controller):
    return any(row.get("kind") == "complete" for row in controller) or any(
        row.get("kind") == "controller_stopped" and row.get("completed") is True
        for row in controller
    )


def rank4_commits(events, physical_rank):
    return sorted(
        (row for row, physical, _ in events
         if physical == physical_rank and row.get("kind") == "commit"
         and (row.get("rank", physical_rank % 8) == 4)),
        key=lambda row: event_time(row) if event_time(row) is not None else float("inf"),
    )


def endurance_summary(session_dir):
    root = Path(session_dir)
    inputs = [root]
    events, event_errors, ranks, files = load_events(inputs)
    controller, controller_errors, controller_files = load_controller(inputs)
    errors = event_errors + controller_errors
    failures = failure_rows(controller, events)
    controller_done = controller_complete(controller)
    replica0_rank4 = rank4_commits(events, 4)
    committed_ranks = sorted({physical for row, physical, _ in events
                              if physical is not None and row.get("kind") == "commit"})
    all_pp1 = [row for row, physical, _ in events
               if row.get("kind") == "commit" and physical in (4, 12)
               and row.get("rank", physical % 8) == 4]
    contributed, uncontributed, unknown = [], [], []
    for row in all_pp1:
        losses = list(flatten_losses(row.get("loss")))
        if row.get("contributed") is True:
            contributed.extend(losses)
        elif row.get("contributed") is False:
            uncontributed.extend(losses)
        else:
            unknown.extend(losses)

    timestamps = [event_time(row) for row in replica0_rank4 if event_time(row) is not None]
    timestamps += [event_time(row) for row in controller if event_time(row) is not None]
    origin = min(timestamps) if timestamps else None
    curve = []
    for row in replica0_rank4:
        timestamp = event_time(row)
        if timestamp is None or origin is None:
            continue
        losses = list(flatten_losses(row.get("loss")))
        curve.append({
            "time": timestamp,
            "minutes": (timestamp - origin) / 60.0,
            "step": row.get("step"),
            "participants": row.get("participants") if finite_number(row.get("participants")) else None,
            "loss_mean": statistics.fmean(losses) if losses else None,
            "loss_values": losses,
            "contributed": row.get("contributed"),
        })
    injections = [row for row in controller if row.get("kind") == "injected"]
    recovery_controller = [row for row in controller if isinstance(row.get("kind"), str)
                          and ("restored" in row["kind"] or "rejoined" in row["kind"]
                               or row["kind"] in {"all_8_restored", "recovery_complete"})]
    recovery_worker = [row for row, _, _ in events if row.get("kind") == "restored"]
    recovery = recovery_controller if recovery_controller else recovery_worker

    missing_nodes = []
    for node in (0, 1):
        if not any(Path(path).parent.name == f"node{node}" for path in files):
            missing_nodes.append(node)
    complete_data = (not failures and controller_done and not errors
                     and len(ranks) == WORLD_RANKS and len(committed_ranks) == WORLD_RANKS
                     and bool(replica0_rank4))
    if failures:
        status = "failed"
    elif not files or not replica0_rank4:
        status = "empty"
    elif complete_data:
        status = "complete"
    else:
        status = "incomplete"
    reasons = []
    if failures:
        reasons.append("failure events present")
    if not controller_done:
        reasons.append("no controller completion event")
    if errors:
        reasons.append("JSONL/controller input errors")
    if len(ranks) != WORLD_RANKS:
        reasons.append(f"rank evidence covers {len(ranks)}/{WORLD_RANKS} ranks")
    if len(committed_ranks) != WORLD_RANKS:
        reasons.append(f"commit evidence covers {len(committed_ranks)}/{WORLD_RANKS} ranks")
    if not replica0_rank4:
        reasons.append("no replica0 rank4 commit data")

    return {
        "session_dir": str(root),
        "status": status,
        "complete_data": complete_data,
        "completeness_reasons": reasons,
        "controller_complete": controller_done,
        "failure_events": failures,
        "input_errors": errors,
        "controller_files": controller_files,
        "event_files": files,
        "rank_evidence": {"count": len(ranks), "physical_ranks": ranks,
                          "committed_rank_count": len(committed_ranks),
                          "committed_physical_ranks": committed_ranks,
                          "missing_nodes": missing_nodes},
        "plot_origin_time": origin,
        "replica0_rank4_curve": curve,
        "pp1_loss_mean": {
            "mean": statistics.fmean(contributed) if contributed else None,
            "samples": len(contributed),
            "scope": "all physical PP1 rank4 commit loss values with contributed is true",
            "contributed_false_excluded": {"mean": statistics.fmean(uncontributed) if uncontributed else None,
                                            "samples": len(uncontributed)},
            "contributed_unspecified_excluded": {"mean": statistics.fmean(unknown) if unknown else None,
                                                  "samples": len(unknown)},
        },
        "events_for_plot": {
            "injected": [{"time": event_time(row), "kind": row.get("kind"), "index": row.get("index"),
                           "target": row.get("target"), "generation": row.get("generation")}
                          for row in injections if event_time(row) is not None],
            "recovery": [{"time": event_time(row), "kind": row.get("kind"), "index": row.get("index"),
                           "target": row.get("target"), "generation": row.get("generation")}
                         for row in recovery if event_time(row) is not None],
        },
    }


def normalize_session_inputs(two_paths):
    first, second = map(Path, two_paths)
    first_node = NODE_DIR.fullmatch(first.name)
    second_node = NODE_DIR.fullmatch(second.name)
    if (first_node and second_node and first.parent.resolve() == second.parent.resolve()
            and {int(first_node.group(1)), int(second_node.group(1))} == {0, 1}):
        return [{"paths": [first, second], "label": first.parent.name}]
    return [{"paths": [first], "label": first.name}, {"paths": [second], "label": second.name}]


def bench_session(inputs, label, condition, warmup_commits=10):
    events, event_errors, ranks, files = load_events(inputs)
    controller, controller_errors, controller_files = load_controller(inputs)
    errors = event_errors + controller_errors
    failures = failure_rows(controller, events)
    done = controller_complete(controller)
    commits = []
    committed_steps = set()
    for row, physical, path in events:
        if row.get("kind") != "commit" or physical is None:
            continue
        step = row.get("step", row.get("updates"))
        if not isinstance(step, int) or isinstance(step, bool):
            continue
        committed_steps.add(step)
        seconds = row.get("seconds")
        iteration = row.get("iteration")
        if not finite_number(seconds) and isinstance(iteration, dict):
            seconds = iteration.get("seconds")
        if not finite_number(seconds) and finite_number(row.get("iteration_seconds")):
            seconds = row["iteration_seconds"]
        if not finite_number(seconds) or seconds <= 0:
            continue
        commits.append({"step": step, "physical_rank": physical, "seconds": float(seconds),
                        "time": event_time(row), "generation": row.get("generation"),
                        "path": str(path)})

    unique_steps = sorted(committed_steps)
    warmup_steps = set(unique_steps[:warmup_commits])
    by_step_rank = {}
    by_step = {}
    for row in commits:
        if row["step"] in warmup_steps:
            continue
        key = (row["step"], row["physical_rank"])
        prior = by_step_rank.get(key)
        candidate_order = (row["generation"] if isinstance(row["generation"], int) else -1,
                           row["time"] if row["time"] is not None else -1)
        prior_order = ((prior["generation"] if isinstance(prior["generation"], int) else -1),
                       prior["time"] if prior["time"] is not None else -1) if prior else None
        if prior is None or candidate_order >= prior_order:
            by_step_rank[key] = row
            by_step.setdefault(row["step"], {})[row["physical_rank"]] = row["seconds"]

    step_seconds = []
    complete_steps = []
    partial_steps = {}
    for step in unique_steps:
        if step in warmup_steps:
            continue
        ranks_at_step = by_step.get(step, {})
        partial_steps[str(step)] = len(ranks_at_step)
        if set(ranks_at_step) == set(range(WORLD_RANKS)):
            complete_steps.append(step)
            step_seconds.append(max(ranks_at_step.values()))

    has_full_steps = bool(step_seconds)
    post_warmup_steps = [step for step in unique_steps if step not in warmup_steps]
    all_steps_have_16 = bool(post_warmup_steps) and all(
        partial_steps.get(str(step), 0) == WORLD_RANKS for step in post_warmup_steps
    )
    complete_data = (not failures and not errors and done and len(ranks) == WORLD_RANKS
                     and has_full_steps and all_steps_have_16)
    if failures:
        status = "failed"
    elif not files or not commits:
        status = "empty"
    elif not done or errors or len(ranks) != WORLD_RANKS:
        status = "incomplete"
    elif not has_full_steps:
        status = "insufficient_steps_after_warmup"
    elif not all_steps_have_16:
        status = "incomplete"
    else:
        status = "complete"
    dist = describe(step_seconds)
    mean = dist["mean"]
    return {
        "label": label,
        "condition": condition,
        "session_dirs": [str(path) for path in inputs],
        "status": status,
        "complete_data": complete_data,
        "controller_complete": done,
        "failure_event_count": len(failures),
        "input_errors": errors,
        "controller_files": controller_files,
        "event_files": files,
        "rank_evidence": {"count": len(ranks), "physical_ranks": ranks},
        "raw_global_steps": len(unique_steps),
        "warmup_global_commits_dropped": min(warmup_commits, len(unique_steps)),
        "complete_steps_after_warmup": complete_steps,
        "post_warmup_steps_with_missing_rank_data": [
            step for step in post_warmup_steps if partial_steps.get(str(step), 0) != WORLD_RANKS
        ],
        "partial_rank_counts_after_warmup": partial_steps,
        "iteration_seconds_max_over_16_ranks": dist,
        "steps_per_second": (1.0 / mean) if mean else None,
        "seconds_by_step": step_seconds,
    }


def percent_change(value, baseline):
    if not finite_number(value) or not finite_number(baseline) or baseline == 0:
        return None
    return (float(value) / float(baseline) - 1.0) * 100.0


def benchmark_summary(flat_args, shift_args):
    if flat_args is None and shift_args is None:
        return None, []
    sessions = []
    by_condition = {"flat": [], "shift": []}
    for condition, values in (("flat", flat_args), ("shift", shift_args)):
        if values is None:
            continue
        for index, input_group in enumerate(normalize_session_inputs(values), 1):
            report = bench_session(input_group["paths"], input_group["label"], condition)
            report["session_index"] = index
            by_condition[condition].append(report)
            sessions.append(report)

    usable = {condition: [row for row in rows if row["complete_data"]]
              for condition, rows in by_condition.items()}
    condition_stats = {}
    for condition, rows in usable.items():
        all_seconds = [value for row in rows for value in row["seconds_by_step"]]
        summary = describe(all_seconds)
        summary["steps_per_second"] = (1.0 / summary["mean"] if summary["mean"] else None)
        summary["complete_sessions"] = len(rows)
        condition_stats[condition] = summary

    flat_rate = condition_stats.get("flat", {}).get("steps_per_second")
    shift_rate = condition_stats.get("shift", {}).get("steps_per_second")
    combined = {
        "flat": condition_stats.get("flat"),
        "shift": condition_stats.get("shift"),
        "throughput_percent_change_shift_vs_flat": percent_change(shift_rate, flat_rate),
        "mean_step_seconds_percent_change_shift_vs_flat": percent_change(
            condition_stats.get("shift", {}).get("mean"), condition_stats.get("flat", {}).get("mean")),
        "formula": "(shift/base - 1)*100; throughput uses pooled complete post-warmup step/s",
    }
    baseline_mean_rate = flat_rate
    per_shift = []
    for row in by_condition["shift"]:
        session_rate = row["steps_per_second"] if row["complete_data"] else None
        per_shift.append({
            "label": row["label"], "session_index": row["session_index"],
            "status": row["status"], "steps_per_second": session_rate,
            "throughput_percent_vs_mean_flat": percent_change(session_rate, baseline_mean_rate),
        })

    abba_order = []
    flat_rows, shift_rows = by_condition["flat"], by_condition["shift"]
    if len(flat_rows) == 2 and len(shift_rows) == 2:
        abba_order = [
            {"condition": "flat", "label": flat_rows[0]["label"]},
            {"condition": "shift", "label": shift_rows[0]["label"]},
            {"condition": "shift", "label": shift_rows[1]["label"]},
            {"condition": "flat", "label": flat_rows[1]["label"]},
        ]
    report = {
        "warmup_policy": "drop first 10 unique global commit steps per session, then require all 16 physical ranks and use max iteration.seconds per step",
        "abba_order": abba_order,
        "abba_complete": (len(flat_rows) == 2 and len(shift_rows) == 2
                          and all(row["complete_data"] for row in flat_rows + shift_rows)),
        "sessions": [{key: value for key, value in row.items() if key != "seconds_by_step"}
                     for row in sessions],
        "individual_shift_sessions_vs_flat_mean": per_shift,
        "combined": combined,
        "combined_comparison_complete": bool(usable["flat"] and usable["shift"]),
        "timing_excludes_incomplete_or_failed_sessions": True,
        "flops_or_token_throughput_reported": False,
    }
    csv_rows = []
    for row in sessions:
        dist = row["iteration_seconds_max_over_16_ranks"]
        csv_rows.append({
            "condition": row["condition"], "session_index": row["session_index"],
            "label": row["label"], "status": row["status"],
            "complete_data": row["complete_data"],
            "post_warmup_steps": dist["count"], "mean_seconds": dist["mean"],
            "median_seconds": dist["median"], "p95_seconds": dist["p95"],
            "steps_per_second": row["steps_per_second"],
        })
    return report, csv_rows


def plot_endurance(summary, output):
    curve = summary["replica0_rank4_curve"]
    if not curve:
        return {"available": False, "reason": "no timed replica0 rank4 commits"}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"available": False, "reason": f"matplotlib unavailable: {exc}"}

    output.mkdir(parents=True, exist_ok=True)
    fig, loss_axis = plt.subplots(figsize=(11, 5.5))
    participant_axis = loss_axis.twinx()
    x = [row["minutes"] for row in curve]
    loss_points = [(row["minutes"], row["loss_mean"]) for row in curve
                   if row["loss_mean"] is not None]
    part_points = [(row["minutes"], row["participants"]) for row in curve
                   if row["participants"] is not None]
    if loss_points:
        loss_axis.plot([point[0] for point in loss_points], [point[1] for point in loss_points],
                       marker=".", linewidth=1, color="#2463a6", label="rank4 commit loss (mean of event values)")
    if part_points:
        participant_axis.scatter([point[0] for point in part_points], [point[1] for point in part_points],
                                 s=12, color="#d17a00", label="participants")
    origin = summary.get("plot_origin_time")
    for row in summary["events_for_plot"]["injected"]:
        minutes = (row["time"] - origin) / 60.0 if origin is not None else None
        if minutes is not None:
            loss_axis.axvline(minutes, color="#b22222", linestyle="--", alpha=0.8)
            loss_axis.text(minutes, 0.98, f"injected {row.get('index', '')}", rotation=90,
                           va="top", ha="right", transform=loss_axis.get_xaxis_transform(), fontsize=8)
    for row in summary["events_for_plot"]["recovery"]:
        minutes = (row["time"] - origin) / 60.0 if origin is not None else None
        if minutes is not None:
            loss_axis.axvline(minutes, color="#238636", linestyle=":", alpha=0.8)
            loss_axis.text(minutes, 0.98, f"recovery {row.get('index', '')}", rotation=90,
                           va="top", ha="left", transform=loss_axis.get_xaxis_transform(), fontsize=8)
    status = summary["status"].upper()
    loss_axis.set_title(f"Endurance: replica 0 rank 4 — {status}")
    loss_axis.set_xlabel("Elapsed time (minutes; actual event timestamps)")
    loss_axis.set_ylabel("Loss")
    participant_axis.set_ylabel("Participants")
    handles1, labels1 = loss_axis.get_legend_handles_labels()
    handles2, labels2 = participant_axis.get_legend_handles_labels()
    if handles1 or handles2:
        loss_axis.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=8)
    fig.tight_layout()
    png, svg = output / "endurance.png", output / "endurance.svg"
    fig.savefig(png, dpi=160)
    fig.savefig(svg)
    plt.close(fig)
    return {"available": True, "png": str(png), "svg": str(svg),
            "complete_data": summary["complete_data"],
            "success_marker_drawn": False}


def plot_bench(summary, output):
    if summary is None:
        return {"available": False, "reason": "no benchmark sessions supplied"}
    rows = [row for row in summary["sessions"] if row["complete_data"]]
    if not rows:
        return {"available": False, "reason": "no complete benchmark sessions; no comparison chart drawn"}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"available": False, "reason": f"matplotlib unavailable: {exc}"}
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"flat": "#4c78a8", "shift": "#f28e2b"}
    labels = [f"{row['condition']}\n{row['label']}" for row in rows]
    seconds = [row["iteration_seconds_max_over_16_ranks"]["mean"] for row in rows]
    rates = [row["steps_per_second"] for row in rows]
    positions = list(range(len(rows)))
    axes[0].bar(positions, seconds, color=[colors[row["condition"]] for row in rows])
    axes[0].set_ylabel("Mean step seconds (max over 16 ranks)")
    axes[1].bar(positions, rates, color=[colors[row["condition"]] for row in rows])
    axes[1].set_ylabel("Steps per second")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("ABBA benchmark — complete sessions only")
    fig.tight_layout()
    png, svg = output / "benchmark_abba.png", output / "benchmark_abba.svg"
    fig.savefig(png, dpi=160)
    fig.savefig(svg)
    plt.close(fig)
    return {"available": True, "png": str(png), "svg": str(svg),
            "sessions_plotted": [row["label"] for row in rows],
            "failed_sessions_plotted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endurance", required=True, type=Path, metavar="SESSIONDIR")
    parser.add_argument("--bench-flat", nargs=2, type=Path, metavar=("DIR1", "DIR2"))
    parser.add_argument("--bench-shift", nargs=2, type=Path, metavar=("DIR1", "DIR2"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    endurance = endurance_summary(args.endurance)
    benchmark, csv_rows = benchmark_summary(args.bench_flat, args.bench_shift)
    plots = {
        "endurance": plot_endurance(endurance, args.output),
        "benchmark": plot_bench(benchmark, args.output),
    }
    summary = {
        "endurance": endurance,
        "benchmarks": benchmark,
        "plots": plots,
        "method": {
            "time_axis": "actual event timestamps, minutes since earliest timed rank4/controller event; no synthetic rows",
            "endurance_curve": "replica0 physical rank4 commits; loss is per-event mean when loss is a list; participants shown at event timestamps",
            "pp1_loss_mean": "all physical rank4 commits with contributed=true; false and unspecified are separately counted and excluded",
            "bench_latency": "drop first 10 unique global steps per session; require all 16 physical ranks; take max seconds per step",
            "tokens_or_flops": "not estimated",
        },
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if csv_rows:
        csv_path = args.output / "bench_sessions.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(json.dumps({
        "summary": str(summary_path),
        "endurance_status": endurance["status"],
        "endurance_points": len(endurance["replica0_rank4_curve"]),
        "benchmark_comparison_complete": bool(benchmark and benchmark["combined_comparison_complete"]),
        "plots": plots,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
