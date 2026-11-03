"""Offline correctness summary for run_overlap_cycles output."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODES = ("node0", "node1", "node3")
RANKS = set(range(8))


def analyze(path):
    checks, failures = {}, []

    def check(name, ok, detail=None):
        checks[name] = bool(ok)
        if not ok:
            failures.append({"check": name, "detail": detail})

    status = json.loads((path / "cycle_status.json").read_text())
    cycles = status["cycles"]
    audit = status["audit"]
    events, supervisors = {}, {}
    read_errors = []
    for node in NODES:
        folder = path / f"physical_{node}"
        events[node] = []
        files = sorted(folder.rglob("events*.jsonl"))
        if not files:
            read_errors.append(f"{node}: missing event files")
        for file in files:
            for line_no, line in enumerate(file.read_text().splitlines(), 1):
                if line.strip():
                    try:
                        events[node].append(json.loads(line))
                    except ValueError as exc:
                        read_errors.append(f"{file.name}:{line_no}: {exc}")
        supervisor_file = folder / "supervisor.jsonl"
        supervisors[node] = [json.loads(x) for x in supervisor_file.read_text().splitlines() if x.strip()]
    check("logs_read", not read_errors, read_errors)
    check("status_success", status["success"] and len(cycles) == status["cycles_requested"] and
          all(c["success"] and c["three_ac_steps"] and c["source_standby_verified"] and c["audit_verified"] == audit for c in cycles))
    def rows(node, kind, round_index=None, step=None, epoch=None):
        return [e for e in events[node] if e["kind"] == kind and
                (round_index is None or e.get("planned_round") == round_index) and
                (step is None or e.get("step") == step) and (epoch is None or e.get("epoch") == epoch)]

    def ranks_ok(items):
        count = Counter(e.get("rank") for e in items)
        return set(count) == RANKS and all(count[r] == 1 for r in RANKS)
    def by_rank(items):
        return {e["rank"]: e for e in items}
    cuts, reports = [], []
    status_rotation_ok = True
    for r, cycle in enumerate(cycles):
        source, target = ("node1", "node3") if r % 2 == 0 else ("node3", "node1")
        cut, post, final = cycle["cut"], cycle["post_steps"], cycle["post_step"]
        status_rotation_ok &= cycle["round"] == r and cycle["source"] == source and cycle["target"] == target
        status_rotation_ok &= final == cut + post
        cuts.append(cut)
        report = {"round": r, "source": source, "target": target, "cut": cut, "post_steps": post, "post_step": final}

        check(f"r{r}_cut_selected", any(e["step"] == cut for e in rows("node0", "planned_cut_selected", r)))
        check(f"r{r}_target_ready", ranks_ok(rows(target, "planned_local_ready", r)))
        standby = rows(source, "planned_standby_verified", r)
        check(f"r{r}_source_standby", ranks_ok(standby) and all(e["step"] == cut for e in standby))
        old_a = rows("node0", "commit", r, cut, r)
        old_s = rows(source, "commit", r, cut, r)
        check(f"r{r}_old_cut_commits", ranks_ok(old_a) and ranks_ok(old_s))
        restores = rows(target, "restored", r, cut)
        restore_map = by_rank(restores)
        restore_ok = ranks_ok(restores) and all(e.get("source_replica") == 1 and
            e.get("source_rank") == rank and e.get("optimizer_tensors", 0) > 0
            for rank, e in restore_map.items())
        restore_sources = by_rank(rows(target, "planned_restore_sources", r, cut))
        donor = "B" if r % 2 == 0 else "C"
        restore_ok &= set(restore_sources) == RANKS and all(
            e.get("source_physical") == donor for e in restore_sources.values())
        if audit:
            restore_ok &= all(e.get("hashes_match") is True and
                              e.get("expected_hashes") == e.get("actual_hashes") and
                              all(e["actual_hashes"].get(k) for k in ("model", "optimizer", "scheduler"))
                              for e in restores)
        check(f"r{r}_restore_from_rotating_replica1", restore_ok)
        first_three = {}
        for node in ("node0", target):
            for step in range(cut + 1, cut + 4):
                first_three[node, step] = rows(node, "commit", r, step, r + 1)
        check(f"r{r}_first_three_ac_commits", all(ranks_ok(v) for v in first_three.values()))
        relay_first = first_three[target, cut + 1]
        check(f"r{r}_target_relay_skips_forward", ranks_ok(relay_first) and
              all(e.get("local_forward_calls") == 0 for e in relay_first))
        old_close_ok = True
        for node in ("node0", source):
            closed = rows(node, "planned_old_manager_closed", r)
            old_close_ok &= ranks_ok(closed) and all(e.get("server_closed") and e.get("old_epoch_revoked") for e in closed)
        pg_ok = all(ranks_ok(cleaned) and all(e.get("all_parent_absent") is True for e in cleaned)
                    for node in NODES for cleaned in [rows(node, "planned_retired_cleanup_complete", r)])
        check(f"r{r}_old_pg_reclaimed", old_close_ok and pg_ok)
        if audit:
            sent = by_rank(rows(source, "planned_gradient_sent", r, cut))
            received = by_rank(rows(target, "planned_gradient_received", r, cut))
            relay_ok = set(sent) == set(received) == RANKS and all(
                sent[k].get("gradient_hash") == received[k].get("gradient_hash") and
                sent[k].get("sample_ids") == received[k].get("sample_ids") and
                sent[k].get("tail_hash") == received[k].get("applied_tail_hash") and received[k].get("forward_calls") == 0
                for k in RANKS)
            check(f"r{r}_gradient_tail_relay_exact", relay_ok)
            state_ok = True
            for step in range(cut + 1, cut + 4):
                aa, tt = by_rank(first_three["node0", step]), by_rank(first_three[target, step])
                state_ok &= all(k in aa and k in tt and all(aa[k]["hashes"].get(h) and
                    aa[k]["hashes"][h] == tt[k]["hashes"].get(h)
                    for h in ("model", "optimizer", "scheduler")) for k in RANKS)
            check(f"r{r}_ac_model_adam_scheduler_exact", state_ok)
        start_kind = "ready" if r == 0 else "planned_round_started"
        start = rows("node0", start_kind, r)
        report["A_start_resources"] = [{k: e[k] for k in ("rank", "step", "cuda_allocated_bytes",
            "cuda_reserved_bytes", "cuda_peak_allocated_bytes", "threads") if k in e}
            for e in sorted(start, key=lambda x: x["rank"])]
        check(f"r{r}_A_resource_snapshot", ranks_ok(start))
        report["A_end_resources"] = [{"kind": e["kind"], **{k: e[k] for k in ("rank", "step",
            "cuda_allocated_bytes", "cuda_reserved_bytes", "cuda_peak_allocated_bytes", "threads") if k in e}}
            for kind in ("planned_final_freeze", "planned_retired_cleanup_complete") for e in rows("node0", kind, r)]
        reports.append(report)
    check("cycle_rotation_and_cuts", status_rotation_ok and
          len(cuts) == len(set(cuts)) and all(a < b for a, b in zip(cuts, cuts[1:])))
    final_step = cycles[-1]["post_step"]
    commits = {n: rows(n, "commit") for n in NODES}
    a_steps = sorted({e["step"] for e in commits["node0"]})
    check("A_steps_contiguous", a_steps == list(range(1, final_step + 1)))
    wrong_owner, missing, samples, bad_commits, bad_native, bad_forward = [], [], [], [], [], []
    relay_first_steps = {(c["target"], c["cut"] + 1) for c in cycles}
    for step in range(1, final_step + 1):
        epoch = next((i for i, cut in enumerate(cuts) if step <= cut), len(cuts))
        owner = "node1" if epoch % 2 == 0 else "node3"
        active = {n: [e for e in commits[n] if e.get("step") == step and e.get("epoch") == epoch]
                  for n in ("node0", owner)}
        if not all(ranks_ok(v) for v in active.values()):
            missing.append({"step": step, "epoch": epoch,
                            "ranks": {n: sorted({e.get("rank") for e in v}) for n, v in active.items()}})
        ids = {x for rows_ in active.values() for e in rows_ for x in e.get("sample_ids", [])}
        if ids != set(range((step - 1) * 8, step * 8)):
            samples.append({"step": step, "ids": sorted(ids)})
        for node in NODES:
            for e in [x for x in commits[node] if x.get("step") == step]:
                if node not in ("node0", owner) or e.get("epoch") != epoch:
                    wrong_owner.append({"node": node, "step": step, "epoch": e.get("epoch")})
                members = ["A", "B" if epoch % 2 == 0 else "C"]
                if (e.get("participants") != 2 or e.get("contributed") is not True or
                    e.get("global_batch") != 8 or e.get("finite_gradients") is not True or
                    e.get("cross_calls", 0) <= 0 or e.get("physical_members") != members):
                    bad_commits.append({"node": node, "rank": e.get("rank"), "step": step})
                pg_rank = 0 if node == "node0" else 1
                if (e.get("native_pg_world") != 2 or e.get("native_pg_rank") != pg_rank or
                    e.get("native_global_ranks") != [e["rank"], e["rank"] + 8]):
                    bad_native.append({"node": node, "rank": e.get("rank"), "step": step})
                no_forward = (node, step) in relay_first_steps
                calls = e.get("local_forward_calls")
                if (no_forward and calls != 0) or (not no_forward and (calls is None or calls <= 0)):
                    bad_forward.append({"node": node, "rank": e.get("rank"), "step": step})
    check("active_rank_coverage", not missing, missing)
    check("fixed_batch_sample_union", not samples, samples)
    check("no_old_owner_or_wrong_epoch_commit", not wrong_owner and all(
        e["step"] <= final_step for n in NODES for e in commits[n]), wrong_owner)
    check("two_contributors_batch8_real_gradient_membership", not bad_commits, bad_commits[:20])
    check("native_world2_rank_mapping", not bad_native, bad_native[:20])
    check("relay_forward_calls", not bad_forward, bad_forward[:20])
    pid_changes = []
    for node in NODES:
        for rank in RANKS:
            pids = {e.get("pid") for e in events[node] if e.get("rank") == rank}
            if len(pids) != 1:
                pid_changes.append({"node": node, "rank": rank, "pids": sorted(pids)})
    check("physical_rank_pid_stable", not pid_changes, pid_changes)
    cleanup_bad = []
    for node in NODES:
        cleanup = [e for e in supervisors[node] if e.get("kind") == "cleanup"]
        if not any(e.get("trainer_returncode") == 0 and e.get("own_children") == [] for e in cleanup):
            cleanup_bad.append(node)
    check("supervisor_cleanup", not cleanup_bad, cleanup_bad)
    for i, report in enumerate(reports[:-1]):
        current = {e["rank"]: e for e in report["A_start_resources"]}
        following = {e["rank"]: e for e in reports[i + 1]["A_start_resources"]}
        report["A_resource_delta_to_next_start"] = [
            {"rank": rank, **{k: following[rank][k] - current[rank][k]
                               for k in ("cuda_allocated_bytes", "cuda_reserved_bytes", "threads")
                               if k in current[rank] and k in following[rank]}}
            for rank in sorted(RANKS)]
    if reports:
        reports[-1]["A_resource_delta_to_next_start"] = None

    a_records = commits["node0"]
    result = {"session": path.name, "audit": audit, "cycles": reports,
              "all_checks_pass": all(checks.values()), "checks": checks, "failures": failures,
              "A_commit_records": len(a_records), "A_unique_steps": len(a_steps),
              "resource_note": "raw per-rank values; CUDA peak is a peak, not a leak threshold"}
    (path / "cycle_analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_checks_pass": result["all_checks_pass"], "cycles": len(cycles),
        "A_commit_records": result["A_commit_records"], "A_unique_steps": result["A_unique_steps"],
        "A_start_resources": [{"round": c["round"], "rank0": next(
            (e for e in c["A_start_resources"] if e["rank"] == 0), None)} for c in reports]}, sort_keys=True))
    return 0 if result["all_checks_pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="session name under results/ or result directory")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    path = Path(args.session)
    if not path.is_dir():
        path = args.results / args.session
    return analyze(path.resolve())


if __name__ == "__main__":
    sys.exit(main())
