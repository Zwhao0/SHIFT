#!/usr/bin/env python3
"""Summarize flat-vs-SHIFT short-trajectory evidence using only the stdlib."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


MODES = ("flat", "shift")
HASH_FIELDS = (
    "cpu_rng", "cuda_rng", "model", "optimizer", "progress", "python_rng", "scheduler",
)
EXPECTED_ROLES = {(rid, rank) for rid in (0, 1) for rank in range(8)}


def _read_events(directory):
    rows = []
    errors = []
    files = sorted(directory.glob("node[01]/events_g*_r*.jsonl"))
    for path in files:
        try:
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except (ValueError, UnicodeError) as exc:
                        errors.append(f"{path}:{line_number}: {type(exc).__name__}")
                        continue
                    if isinstance(value, dict) and value.get("kind") in {
                        "ready", "batch_plan", "routing", "commit",
                    }:
                        rows.append(value)
        except OSError as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return rows, errors, len(files)


def _integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _role(row):
    rid, rank = _integer(row.get("rid")), _integer(row.get("rank"))
    return (rid, rank) if rid is not None and rank is not None else None


def _generation_zero(rows):
    return [row for row in rows if _integer(row.get("generation", 0)) == 0]


def _ready(rows):
    selected = {}
    duplicates = []
    for row in _generation_zero(rows):
        if row.get("kind") != "ready":
            continue
        key = _role(row)
        if key is None:
            duplicates.append("ready row missing rid/rank")
        elif key in selected:
            duplicates.append(f"duplicate ready role rid{key[0]}/rank{key[1]}")
        else:
            selected[key] = row
    return selected, duplicates


def _initial_hash_summary(mode_rows):
    ready = {mode: _ready(mode_rows[mode]) for mode in MODES}
    missing_roles = {}
    unexpected_roles = {}
    malformed_hash_roles = {mode: [] for mode in MODES}
    duplicate_roles = {mode: ready[mode][1] for mode in MODES}
    for mode in MODES:
        missing_roles[mode] = sorted(EXPECTED_ROLES - set(ready[mode][0]))
        unexpected_roles[mode] = sorted(set(ready[mode][0]) - EXPECTED_ROLES)
        for role, row in ready[mode][0].items():
            hashes = row.get("hashes")
            if not isinstance(hashes, dict) or not set(HASH_FIELDS).issubset(hashes):
                malformed_hash_roles[mode].append(list(role))

    flat_rows, shift_rows = ready["flat"][0], ready["shift"][0]
    mismatched = []
    for role in sorted(EXPECTED_ROLES & flat_rows.keys() & shift_rows.keys()):
        left, right = flat_rows[role].get("hashes"), shift_rows[role].get("hashes")
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        if any(left.get(name) != right.get(name) for name in HASH_FIELDS):
            mismatched.append(list(role))
    same = (not any(missing_roles.values()) and not any(unexpected_roles.values())
            and not any(malformed_hash_roles.values())
            and not any(duplicate_roles.values()) and not mismatched)
    return {
        "pass": same,
        "expected_role_count": len(EXPECTED_ROLES),
        "hash_fields": list(HASH_FIELDS),
        "roles_present": {mode: len(ready[mode][0]) for mode in MODES},
        "missing_roles": {mode: [list(role) for role in values]
                          for mode, values in missing_roles.items()},
        "unexpected_roles": {mode: [list(role) for role in values]
                             for mode, values in unexpected_roles.items()},
        "malformed_hash_roles": malformed_hash_roles,
        "duplicate_ready_roles": duplicate_roles,
        "mismatched_roles": mismatched,
    }


def _keyed_events(rows, kind, key_fields, value_field):
    result = {}
    duplicates = []
    for row in _generation_zero(rows):
        if row.get("kind") != kind:
            continue
        key = []
        for field in key_fields:
            value = _integer(row.get(field)) if field in ("rid", "rank", "step", "call") else row.get(field)
            key.append(value)
        key = tuple(key)
        if any(value is None for value in key):
            duplicates.append(f"{kind} row has incomplete key {key!r}")
        elif key in result:
            duplicates.append(f"duplicate {kind} key {key!r}")
        else:
            result[key] = row.get(value_field)
    return result, duplicates


def _batch_summary(mode_rows, expected_plan_steps):
    maps = {}
    duplicates = {}
    for mode in MODES:
        maps[mode], duplicates[mode] = _keyed_events(
            mode_rows[mode], "batch_plan", ("rid", "rank", "step"), "sample_ids",
        )
    flat, shift = maps["flat"], maps["shift"]
    flat_keys, shift_keys = set(flat), set(shift)
    common = flat_keys & shift_keys
    differences = [
        {"rid": key[0], "rank": key[1], "step": key[2],
         "flat_sample_ids": flat[key], "shift_sample_ids": shift[key]}
        for key in sorted(common) if flat[key] != shift[key]
    ]
    missing_flat = sorted(shift_keys - flat_keys)
    missing_shift = sorted(flat_keys - shift_keys)
    invalid_values = {mode: [list(key) for key, value in maps[mode].items()
                             if not isinstance(value, list) or not value]
                      for mode in MODES}
    coverage_errors = []
    for mode in MODES:
        keys = maps[mode]
        absent = [list((rid, rank, step)) for rid, rank in sorted(EXPECTED_ROLES)
                  for step in sorted(expected_plan_steps) if (rid, rank, step) not in keys]
        if absent:
            coverage_errors.append({"mode": mode, "missing_role_steps": absent})
    passed = (not differences and not missing_flat and not missing_shift
              and not any(duplicates.values()) and not coverage_errors
              and not any(invalid_values.values()))
    return {
        "pass": passed,
        "flat_count": len(flat), "shift_count": len(shift), "common_count": len(common),
        "different_sample_ids": differences[:50],
        "missing_from_flat": [list(key) for key in missing_flat[:50]],
        "missing_from_shift": [list(key) for key in missing_shift[:50]],
        "coverage_errors": coverage_errors,
        "invalid_sample_ids": invalid_values,
        "duplicate_keys": duplicates,
    }


def _routing_summary(mode_rows, expected_plan_steps):
    maps = {}
    duplicates = {}
    for mode in MODES:
        maps[mode], duplicates[mode] = _keyed_events(
            mode_rows[mode], "routing", ("rid", "rank", "step", "layer", "call"), "topk_hash",
        )
    flat, shift = maps["flat"], maps["shift"]
    flat_keys, shift_keys = set(flat), set(shift)
    common = flat_keys & shift_keys
    unequal = [key for key in common if flat[key] != shift[key]]
    flat_only, shift_only = flat_keys - shift_keys, shift_keys - flat_keys
    by_step = defaultdict(lambda: {"flat": 0, "shift": 0, "compared": 0,
                                   "matched": 0, "different": 0,
                                   "missing_from_flat": 0, "missing_from_shift": 0})
    for key in flat_keys:
        by_step[key[2]]["flat"] += 1
    for key in shift_keys:
        by_step[key[2]]["shift"] += 1
    for key in common:
        row = by_step[key[2]]
        row["compared"] += 1
        row["matched" if flat[key] == shift[key] else "different"] += 1
    for key in flat_only:
        by_step[key[2]]["missing_from_shift"] += 1
    for key in shift_only:
        by_step[key[2]]["missing_from_flat"] += 1
    invalid_values = {mode: [list(key) for key, value in maps[mode].items()
                             if not isinstance(value, str) or not value]
                      for mode in MODES}
    coverage_errors = {}
    for mode in MODES:
        present = {(key[0], key[1], key[2]) for key in maps[mode]}
        absent = [list((rid, rank, step)) for rid, rank in sorted(EXPECTED_ROLES)
                  for step in sorted(expected_plan_steps) if (rid, rank, step) not in present]
        if absent:
            coverage_errors[mode] = absent
    first = sorted(set(unequal) | set(flat_only) | set(shift_only),
                   key=lambda k:(k[2],k[0],k[1],k[3],k[4]))[:30]
    route_rows={mode:{(e['rid'],e['rank'],e['step'],e['layer'],e['call']):e
                     for e in mode_rows[mode] if e.get('kind')=='routing'} for mode in MODES}
    selection_sets={}
    for key in sorted(common):
        a=route_rows['flat'][key].get('sorted_topk_hash');b=route_rows['shift'][key].get('sorted_topk_hash')
        if isinstance(a,str) and isinstance(b,str):
            item=selection_sets.setdefault(str(key[2]),dict(compared=0,different=0))
            item['compared']+=1;item['different']+=int(a!=b)
    margins={}
    for mode in MODES:
        margins[mode]={}
        for step in sorted(expected_plan_steps):
            rows=[e for e in route_rows[mode].values() if e['step']==step and 'min_boundary_margin' in e]
            if rows:
                margins[mode][str(step)]=dict(min_boundary_margin=min(e['min_boundary_margin'] for e in rows),
                    zero_margins=sum(e['zero_margins'] for e in rows),
                    near_ties_1e6=sum(e['near_ties_1e6'] for e in rows),
                    audited_tokens=sum(e['tokens'] for e in rows))
    differences = []
    for rid, rank, step, layer, call in first:
        key = (rid, rank, step, layer, call)
        differences.append({
            "rid": rid, "rank": rank, "step": step, "layer": layer, "call": call,
            "flat_topk_hash": flat.get(key), "shift_topk_hash": shift.get(key),
            "flat_min_boundary_margin": route_rows['flat'].get(key,{}).get('min_boundary_margin'),
            "shift_min_boundary_margin": route_rows['shift'].get(key,{}).get('min_boundary_margin'),
        })
    return {
        "pass": (bool(flat_keys or shift_keys) and not unequal and not flat_only and not shift_only
                 and not any(duplicates.values()) and not any(invalid_values.values())
                 and not coverage_errors),
        "flat_count": len(flat), "shift_count": len(shift), "common_count": len(common),
        "different_count": len(unequal), "missing_from_flat_count": len(shift_only),
        "missing_from_shift_count": len(flat_only),
        "per_step": {str(step): by_step[step] for step in sorted(by_step)},
        "first_differences": differences,
        "first_divergent_pre_update_step": min((k[2] for k in unequal),default=None),
        "boundary_diagnostics": margins,
        "selection_set_comparison_per_step": selection_sets,
        "invalid_hash_keys": invalid_values,
        "coverage_errors": coverage_errors,
        "duplicate_keys": duplicates,
    }


def _batch_progress_summary(mode_rows, expected_commit_steps):
    maps = {}
    invalid = {}
    for mode in MODES:
        values, duplicates = _keyed_events(
            mode_rows[mode], "commit", ("rid", "rank", "step"), "participants",
        )
        maps[mode] = values
        invalid[mode] = []
        rows = [row for row in _generation_zero(mode_rows[mode]) if row.get("kind") == "commit"]
        for row in rows:
            if _integer(row.get("participants")) != 2 or _integer(row.get("global_batch")) != 8:
                invalid[mode].append({
                    "rid": _integer(row.get("rid")), "rank": _integer(row.get("rank")),
                    "step": _integer(row.get("step")),
                    "participants": row.get("participants"), "global_batch": row.get("global_batch"),
                })
        invalid[mode].extend({"duplicate": item} for item in duplicates)
    same_keys = set(maps["flat"]) == set(maps["shift"])
    coverage_errors = {}
    for mode in MODES:
        absent = [list((rid, rank, step)) for rid, rank in sorted(EXPECTED_ROLES)
                  for step in sorted(expected_commit_steps) if (rid, rank, step) not in maps[mode]]
        if absent:
            coverage_errors[mode] = absent
    return {
        "pass": (same_keys and not invalid["flat"] and not invalid["shift"]
                 and not coverage_errors),
        "expected_participants": 2,
        "expected_global_batch": 8,
        "commit_count": {mode: len(maps[mode]) for mode in MODES},
        "matching_role_step_keys": same_keys,
        "coverage_errors": coverage_errors,
        "invalid_rows": invalid,
    }


def _comparison_summary(path):
    errors = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        return {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(data, dict):
        return {"pass": False, "error": "comparison JSON root must be an object"}
    byrank = data.get("byrank")
    if not isinstance(byrank, dict):
        return {"pass": False, "error": "comparison JSON lacks byrank object"}

    category_stats = defaultdict(lambda: {"observations": 0, "failed_pairs": [],
                                          "maximum_abs": None, "maximum_relative_l2": None})
    pair_pass = {}
    for step, ranks in byrank.items():
        if not isinstance(ranks, dict):
            errors.append(f"byrank step {step!r} is not an object")
            continue
        for rank, pair in ranks.items():
            name = f"step{step}/rank{rank}"
            if not isinstance(pair, dict):
                errors.append(f"{name} is not an object")
                continue
            pair_pass[name] = pair.get("pass") is True
            categories = pair.get("by_category", {})
            if not isinstance(categories, dict):
                categories = {}
            for category in ("gradients", "model", "optimizer", "scheduler"):
                summary = category_stats[category]
                value = categories.get(category)
                if not isinstance(value, dict):
                    summary["failed_pairs"].append(name)
                    continue
                summary["observations"] += 1
                if value.get("pass") is not True:
                    summary["failed_pairs"].append(name)
                metrics = value.get("metrics", {})
                if not isinstance(metrics, dict):
                    continue
                for source, destination in (("max_abs", "maximum_abs"),
                                            ("relative_l2", "maximum_relative_l2")):
                    metric = metrics.get(source)
                    if isinstance(metric, (int, float)):
                        current = summary[destination]
                        summary[destination] = metric if current is None else max(current, metric)
    checks = data.get("checks", {})
    if not isinstance(checks, dict):
        checks = {}
    check = checks.get("all_rank_step_comparisons_pass", {})
    failed_pairs = check.get("failed_pairs", []) if isinstance(check, dict) else []
    categories_out = {}
    for name, summary in category_stats.items():
        categories_out[name] = {
            "pass": not summary["failed_pairs"],
            "observations": summary["observations"],
            "failed_pairs": sorted(set(summary["failed_pairs"])),
            "maximum_abs": summary["maximum_abs"],
            "maximum_relative_l2": summary["maximum_relative_l2"],
        }
    return {
        "pass": data.get("pass") is True,
        "steps": data.get("steps"), "ranks": data.get("ranks"),
        "tolerances": data.get("tolerances"),
        "comparator_checks": checks,
        "failed_pairs": failed_pairs,
        "pair_count": len(pair_pass),
        "by_category": categories_out,
        "errors": errors,
    }


def _step_coverage(comparison):
    raw_steps = comparison.get("steps")
    if (not isinstance(raw_steps, list) or not raw_steps
            or any(isinstance(step, bool) or not isinstance(step, int) or step < 1
                   for step in raw_steps)
            or len(set(raw_steps)) != len(raw_steps)):
        return None, None
    commit_steps = sorted(raw_steps)
    return commit_steps, [step - 1 for step in commit_steps]


def summarize(baseline_dir, shift_dir, comparison_json):
    directories = {"flat": baseline_dir, "shift": shift_dir}
    mode_rows, inputs = {}, {}
    for mode, directory in directories.items():
        mode_rows[mode], errors, file_count = _read_events(directory)
        inputs[mode] = {"directory": str(directory), "jsonl_files": file_count,
                        "event_rows": len(mode_rows[mode]), "read_errors": errors}
    initial = _initial_hash_summary(mode_rows)
    comparison = _comparison_summary(comparison_json)
    commit_steps, plan_steps = _step_coverage(comparison)
    coverage_defined = commit_steps is not None
    commit_steps = commit_steps or []
    plan_steps = plan_steps or []
    batches = _batch_summary(mode_rows, set(plan_steps))
    routing = _routing_summary(mode_rows, set(plan_steps))
    progress = _batch_progress_summary(mode_rows, set(commit_steps))
    no_input_errors = not any(inputs[mode]["read_errors"] for mode in MODES)
    checks = {
        "input_read": {"pass": no_input_errors},
        "same_initial_state": {"pass": initial["pass"]},
        "step_coverage": {"pass": coverage_defined},
        "same_batch_plans": {"pass": batches["pass"]},
        "same_routing": {"pass": routing["pass"]},
        "participants_and_global_batch": {"pass": progress["pass"]},
        "numerical_comparison": {"pass": comparison.get("pass") is True},
    }
    return {
        "pass": all(item["pass"] for item in checks.values()),
        "inputs": inputs,
        "checks": checks,
        "step_coverage": {
            "numerical_commit_steps": commit_steps,
            "pre_update_batch_and_routing_steps": plan_steps,
        },
        "initial_state": initial,
        "batch_plans": batches,
        "routing": routing,
        "participants_and_global_batch": progress,
        "numerical_comparison": comparison,
        "interpretation": (
            "These observations compare the captured trajectories; they do not prove a root cause. "
            "FP32 reduction and accumulation order may differ even when sample plans and routing agree."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("shift_dir", type=Path)
    parser.add_argument("comparison_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.baseline_dir, args.shift_dir, args.comparison_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "output": str(args.output),
                      "checks": report["checks"]}, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
