#!/usr/bin/env python3
"""Offline aggregate for independent planned-handoff trials."""
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "planned_handoff" / "aggregate.json"
BOOTSTRAP_SEED = 20261103
BOOTSTRAP_N = 5000
EXPECTED_PER_CELL = 5
METRICS = {
    "gap_seconds": "last_ab_to_first_ac_seconds",
    "extra_gap_seconds": "extra_gap_seconds",
    "request_to_first_ac_seconds": "request_to_first_ac_observed_controller_seconds",
}


def expected_trials(phase):
    if phase == "audit":
        return [
            (f"ph_audit_{i + 1:02d}_p1_{'B' if i % 2 == 0 else 'A'}", "p1",
             "B" if i % 2 == 0 else "A")
            for i in range(10)
        ]
    order = ("p0", "p1", "p1", "p0", "p0", "p1", "p1", "p0", "p0", "p1")
    jobs = [(variant, source) for source in ("B", "A") for variant in order]
    return [
        (f"ph_timing_{i + 1:02d}_{variant}_{source}", variant, source)
        for i, (variant, source) in enumerate(jobs)
    ]


def read_manifest(phase):
    path = RESULTS / "planned_handoff" / f"{phase}_suite.jsonl"
    rows = defaultdict(list)
    if not path.is_file():
        return rows, f"missing suite manifest: {path.relative_to(ROOT)}"
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        return rows, f"cannot read {path.relative_to(ROOT)}: {exc}"
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            rows[row["name"]].append(row)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            return rows, f"invalid {path.relative_to(ROOT)}:{line_no}: {exc}"
    return rows, None


def valid_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_trial(name, variant, source, phase, manifests):
    path = RESULTS / name
    analysis_path = path / "planned_analysis.json"
    reasons = []
    if not analysis_path.is_file():
        return None, [f"{name}: missing planned_analysis.json"]
    try:
        data = json.loads(analysis_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"{name}: invalid analysis JSON: {exc}"]
    if not isinstance(data, dict):
        return None, [f"{name}: analysis JSON is not an object"]

    if data.get("session") != name or data.get("variant") != variant or data.get("source") != source:
        reasons.append(f"{name}: session/variant/source metadata mismatch")
    if data.get("audit") is not (phase == "audit"):
        reasons.append(f"{name}: audit flag does not match {phase} phase")
    if data.get("passed") is not True or data.get("all_checks_pass") is not True:
        reasons.append(f"{name}: planned analysis did not pass")
    checks = data.get("checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        false_checks = sorted(key for key, value in checks.items() if value is not True) if isinstance(checks, dict) else []
        reasons.append(f"{name}: missing or false acceptance checks {false_checks}")
    # These checks provide evidence beyond the logged static physical_members label.
    for key in ("native_training_membership", "no_b_in_ac_pg", "candidate_reused_once", "b_alive_through_three_ac"):
        if not isinstance(checks, dict) or checks.get(key) is not True:
            reasons.append(f"{name}: missing physical-membership evidence check {key}")

    manifest_rows = manifests.get(name, [])
    if len(manifest_rows) != 1:
        reasons.append(f"{name}: expected one suite-manifest row, found {len(manifest_rows)}")
    else:
        row = manifest_rows[0]
        if row.get("independent_launch") is not True or row.get("same_A_continuous_rotation") is not False:
            reasons.append(f"{name}: independent-launch metadata missing or inconsistent")
        if row.get("exit_code") != 0 or row.get("checks_pass") is not True:
            reasons.append(f"{name}: suite-manifest run did not pass")
        if row.get("variant") != variant or row.get("source") != source:
            reasons.append(f"{name}: suite-manifest matrix metadata mismatch")

    for field in METRICS.values():
        if not valid_number(data.get(field)):
            reasons.append(f"{name}: missing/non-finite metric {field}")
    for field in ("last_ab_to_first_ac_seconds", "request_to_first_ac_observed_controller_seconds"):
        if valid_number(data.get(field)) and data[field] <= 0:
            reasons.append(f"{name}: non-positive timing metric {field}")
    for field in ("normal_intervals_seconds", "ac_intervals_seconds"):
        values = data.get(field)
        if not isinstance(values, list) or not values or not all(valid_number(v) and v > 0 for v in values):
            reasons.append(f"{name}: missing/invalid interval series {field}")
    for field in ("ab_commits_while_c_prepared_lower_bound", "ab_commits_after_candidate_ready"):
        if field in data and (not valid_number(data[field]) or data[field] < 0):
            reasons.append(f"{name}: invalid preparation progress metric {field}")
    if reasons:
        return None, reasons
    return data, []


def numeric_summary(values):
    values = [float(v) for v in values if valid_number(v)]
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def interval_summary(trials, field):
    per_run = [data[field] for data in trials]
    run_medians = [statistics.median(values) for values in per_run if values]
    pooled = [value for values in per_run for value in values]
    return {
        "runs": len(run_medians),
        "median_of_run_medians_seconds": statistics.median(run_medians) if run_medians else None,
        "pooled_median_seconds": statistics.median(pooled) if pooled else None,
        "pooled_n_intervals": len(pooled),
    }


def summarize_cell(observed, passing, expected_n):
    cell = {"n": len(passing), "observed_n": len(observed), "expected_n": expected_n,
            "sessions": [row["session"] for row in passing]}
    for output_name, field in METRICS.items():
        cell[output_name] = numeric_summary([row[field] for row in passing])
    for field in ("normal_intervals_seconds", "ac_intervals_seconds"):
        cell[field] = interval_summary(passing, field)
    for field in ("ab_commits_while_c_prepared_lower_bound", "ab_commits_after_candidate_ready"):
        present = [row[field] for row in passing if field in row]
        if present:
            cell[field] = numeric_summary(present)
    return cell


def percentile(sorted_values, p):
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    fraction = position - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def bootstrap_median_delta(p0, p1, rng):
    """Independent two-sample percentile bootstrap; delta is median(P0)-median(P1)."""
    if len(p0) != EXPECTED_PER_CELL or len(p1) != EXPECTED_PER_CELL:
        return {"delta_median_seconds": None, "ci_95_seconds": None,
                "reason": f"requires {EXPECTED_PER_CELL} passing runs in both cells"}
    delta = statistics.median(p0) - statistics.median(p1)
    draws = []
    for _ in range(BOOTSTRAP_N):
        m0 = statistics.median(rng.choice(p0) for _ in p0)
        m1 = statistics.median(rng.choice(p1) for _ in p1)
        draws.append(m0 - m1)
    draws.sort()
    return {"delta_median_seconds": delta,
            "ci_95_seconds": [percentile(draws, 0.025), percentile(draws, 0.975)]}


def summarize_phase(phase, expected, manifests):
    issues = []
    manifest_issue = manifests[1]
    if manifest_issue:
        issues.append(manifest_issue)
    manifest_rows = manifests[0]
    expected_names = {name for name, _, _ in expected}
    found_names = {p.name for p in RESULTS.glob(f"ph_{phase}_*") if p.is_dir()}
    extras = sorted(found_names - expected_names)
    if extras:
        issues.append(f"unexpected {phase} session directories: {extras}")
    manifest_extras = sorted(set(manifest_rows) - expected_names)
    if manifest_extras:
        issues.append(f"unexpected {phase} suite-manifest rows: {manifest_extras}")
    matrix_ok = not extras and not manifest_extras and manifest_issue is None

    loaded = {}
    pass_by_cell = defaultdict(list)
    observed_by_cell = defaultdict(list)
    for name, variant, source in expected:
        data, errors = load_trial(name, variant, source, phase, manifest_rows)
        if data is not None:
            loaded[name] = data
            observed_by_cell[(variant, source)].append(data)
            pass_by_cell[(variant, source)].append(data)
        else:
            if (RESULTS / name / "planned_analysis.json").is_file():
                try:
                    bad_data = json.loads((RESULTS / name / "planned_analysis.json").read_text())
                    observed_by_cell[(variant, source)].append(bad_data)
                except Exception:
                    pass
            issues.extend(errors)

    cells = {}
    variants = ("p1",) if phase == "audit" else ("p0", "p1")
    for variant in variants:
        for source in ("A", "B"):
            key = f"{variant}_{source}"
            passing = pass_by_cell[(variant, source)]
            observed = observed_by_cell[(variant, source)]
            cells[key] = summarize_cell(observed, passing, EXPECTED_PER_CELL)
            if len(passing) != EXPECTED_PER_CELL:
                matrix_ok = False
                issues.append(f"{phase} {key}: {len(passing)}/{EXPECTED_PER_CELL} passing runs")

    expected_n = 10 if phase == "audit" else 20
    if len(loaded) != expected_n:
        matrix_ok = False
    manifest_runs = len(set(manifest_rows) & expected_names)
    return {"expected_trials": expected_n, "manifest_runs": manifest_runs,
            "passing_trials": len(loaded),
            "matrix_complete": matrix_ok, "cells": cells,
            "issues": sorted(set(issues)), "passing_rows": loaded}


def steady_reference(session):
    base = RESULTS / session / "physical_node0"
    issues = []
    events = []
    files = sorted(base.glob("events*.jsonl")) if base.is_dir() else []
    if not files:
        return {"complete": False, "issues": [f"{session}: missing physical_node0/events*.jsonl"]}
    for path in files:
        try:
            lines = path.read_text().splitlines()
        except OSError as exc:
            issues.append(f"{session}: cannot read {path.name}: {exc}")
            continue
        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(f"{path.name}:{line_no}: {exc}")
                continue
            if not isinstance(event, dict):
                issues.append(f"{path.name}:{line_no}: event is not an object")
            elif event.get("kind") == "commit" and valid_number(event.get("step")) and 6 <= event["step"] <= 50:
                events.append(event)
    by_step = defaultdict(list)
    for event in events:
        if valid_number(event.get("rank")) and valid_number(event.get("monotonic")):
            by_step[event["step"]].append(event)
        else:
            issues.append(f"{session}: commit missing rank/monotonic")
    step_max = {}
    for step in range(6, 51):
        rows = by_step.get(step, [])
        ranks = [row["rank"] for row in rows]
        if sorted(ranks) != list(range(8)):
            issues.append(f"{session}: step {step} rank coverage/count is {ranks}")
        elif all(valid_number(row.get("monotonic")) for row in rows):
            step_max[str(step)] = max(row["monotonic"] for row in rows)
    intervals = [step_max[str(step)] - step_max[str(step - 1)]
                 for step in range(7, 51) if str(step) in step_max and str(step - 1) in step_max]
    if len(intervals) != 44:
        issues.append(f"{session}: expected 44 step-6..50 intervals, found {len(intervals)}")
    if any(not valid_number(value) or value <= 0 for value in intervals):
        issues.append(f"{session}: non-positive/non-finite steady interval")
    return {"complete": not issues, "window_steps": [6, 50],
            "rank_completion_rule": "max monotonic timestamp across ranks 0..7 per step; intervals between adjacent step maxima",
            "max_rank_monotonic_by_step": step_max,
            "intervals_seconds": numeric_summary(intervals), "issues": issues}


def main():
    issues = []
    audit = summarize_phase("audit", expected_trials("audit"), read_manifest("audit"))
    timing = summarize_phase("timing", expected_trials("timing"), read_manifest("timing"))
    if not audit["matrix_complete"]:
        issues.extend(audit["issues"])
    if not timing["matrix_complete"]:
        issues.extend(timing["issues"])

    rng = random.Random(BOOTSTRAP_SEED)
    comparisons = {}
    for source in ("A", "B"):
        comparisons[source] = {}
        for metric_name in METRICS:
            metric_field = METRICS[metric_name]
            p0 = [row[metric_field] for row in timing["passing_rows"].values()
                  if row.get("variant") == "p0" and row.get("source") == source]
            p1 = [row[metric_field] for row in timing["passing_rows"].values()
                  if row.get("variant") == "p1" and row.get("source") == source]
            comparisons[source][metric_name] = bootstrap_median_delta(p0, p1, rng)

    steady = {name: steady_reference(name) for name in ("ph_steady_01", "ph_native_01")}
    for name, summary in steady.items():
        if not summary["complete"]:
            issues.extend(summary["issues"])

    aggregate = {
        "status": "complete" if not issues else "incomplete",
        "all_required_checks_pass": not issues,
        "incomplete_reasons": sorted(set(issues)),
        "protocol_acceptance_scope": {
            "independent_launches": audit["manifest_runs"],
            "expected_independent_launches": 10,
            "same_A_continuous_rotations": 0,
            "description": "Each audit session is a separate start with one AB-to-AC handoff; the target is ten starts, not ten rotations in one persistent A process.",
        },
        "timing_scope": {"independent_launches": timing["manifest_runs"],
                         "expected_independent_launches": 20,
                         "expected_per_variant_source_cell": EXPECTED_PER_CELL},
        "bootstrap": {"resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED,
                      "method": "independent two-sample percentile bootstrap of median differences",
                      "delta_sign": "median(P0) - median(P1); positive means lower P1 metric"},
        "audit": {k: v for k, v in audit.items() if k != "passing_rows"},
        "timing": {k: v for k, v in timing.items() if k != "passing_rows"},
        "p0_minus_p1_by_source": comparisons,
        "steady_reference": steady,
        "membership_evidence_note": (
            "Static physical_members labels are not treated as sole proof. Passing runs must also include native_training_membership, no_b_in_ac_pg, candidate_reused_once, and b_alive_through_three_ac checks. Logical global ranks encode the inherited replica id, so this summary does not claim they alone identify physical B versus C."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": aggregate["status"], "output": str(OUT),
                      "audit_passing": audit["passing_trials"], "audit_expected": audit["expected_trials"],
                      "timing_passing": timing["passing_trials"], "timing_expected": timing["expected_trials"],
                      "incomplete_reasons": aggregate["incomplete_reasons"]}, indent=2))
    return 0 if aggregate["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
