"""Run repeated full-MoE B/C relay cycles with all three physical hosts resident."""

import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_planned_handoff import PlannedSession, run_job  # noqa: E402


NODES = ("node0", "node1", "node3")


def rows(session, data, node, kind, round_index):
    return [row for row in session.events(data, node, kind)
            if row.get("planned_round") == round_index]


def ranks_complete(events):
    return {row.get("rank") for row in events} == set(range(8))


def round_key(cycles, round_index, key):
    return f"round/{round_index}/{key}" if cycles > 1 else key


def wait_round(session, label, node, kind, round_index, timeout=300, predicate=None):
    def ready(data):
        found = rows(session, data, node, kind, round_index)
        return ranks_complete(found) and (predicate is None or predicate(found))

    return session.wait(label, lambda data: ready(data), timeout=timeout)


def write_status(session, name, cycles, post_steps, audit, results, error=None):
    payload = {
        "session": name,
        "cycles_requested": cycles,
        "post_steps": post_steps,
        "audit": audit,
        "success": len(results) == cycles and all(item.get("success") for item in results),
        "cycles": results,
    }
    if error is not None:
        payload["error"] = error
    target = session.out / "cycle_status.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)


def start_lighthouses(session, pool, python, name, cycles, futures):
    endpoint=session.endpoints['node0']
    code='''import subprocess,time
from pathlib import Path
root=Path(DIRECTORY);ps=[];streams=[]
try:
 for r in range(CYCLES):
  stream=(root/f'lighthouse_round{r}.log').open('w');streams.append(stream)
  p=subprocess.Popen(['/workspace/SHIFT/runtime/torch214/bin/python','/workspace/SHIFT/vendor/bin/torchft_lighthouse','--bind','[::]:'+str(31410+20*r),'--min_replicas','2','--heartbeat_timeout_ms','1000','--join_timeout_ms','1000','--quorum_tick_ms','100'],stdout=stream,stderr=subprocess.STDOUT);ps.append(p)
 time.sleep(.5)
 if any(p.poll() is not None for p in ps):raise RuntimeError('Lighthouse startup failed')
 for r,p in enumerate(ps):(root/f'NEW_LIGHTHOUSE_{r}.ready').write_text(str(p.pid))
 while not (root/'STOP_NEW_LIGHTHOUSE_0').exists():
  if any(p.poll() is not None for p in ps):raise RuntimeError('Lighthouse exited')
  time.sleep(.1)
finally:
 for p in ps:
  if p.poll() is None:p.terminate()
 for p in ps:
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:p.kill();p.wait()
 for stream in streams:stream.close()
'''.replace('DIRECTORY',repr('/workspace/SHIFT/results/'+name)).replace('CYCLES',str(cycles))
    # One SSH poller owns all servers; avoid the SSH MaxSessions ceiling.
    futures[0]=pool.submit(run_job,session,endpoint,['env','TOKIO_WORKER_THREADS=2','TORCHFT_USE_OTEL=false',python,'-c',code],1920)
    ready_paths = {
        round_index: f"/workspace/SHIFT/results/{name}/NEW_LIGHTHOUSE_{round_index}.ready"
        for round_index in range(cycles)
    }
    end = min(session.deadline, time.monotonic() + 120)
    while time.monotonic() < end:
        for round_index, future in futures.items():
            if future.done():
                future.result()
        # Keep the remote check stdlib-only and inspect every round marker in one SSH call.
        check = (
            "from pathlib import Path; import json; paths=" + repr(ready_paths) +
            "; print(json.dumps({int(k):Path(v).exists() for k,v in paths.items()}))"
        )
        raw = session.execute_endpoint(endpoint, ["python", "-c", check], timeout=30)
        state = json.loads(raw.strip().splitlines()[-1])
        if all(state.values()):
            session.event("new_lighthouses_ready", ports=[31410 + 20 * r for r in range(cycles)])
            return
        time.sleep(.25)
    raise TimeoutError("new Lighthouse startup timed out")


def audit_cycle(session, data, round_index, source, target, cut, final_step):
    restores = [row for row in rows(session, data, target, "restored", round_index)
                if row.get("step") == cut]
    if not ranks_complete(restores) or any(not row.get("hashes_match") or
                                           row.get("optimizer_tensors", 0) <= 0
                                           for row in restores):
        raise AssertionError(f"round {round_index}: target state restore audit failed")

    sent = [row for row in rows(session, data, source, "planned_gradient_sent", round_index)
            if row.get("step") == cut]
    received = [row for row in rows(session, data, target, "planned_gradient_received", round_index)
                if row.get("step") == cut]
    if not ranks_complete(sent) or not ranks_complete(received):
        raise AssertionError(f"round {round_index}: incomplete gradient relay records")
    sent_by_rank = {row["rank"]: row for row in sent}
    received_by_rank = {row["rank"]: row for row in received}
    for rank in range(8):
        out = sent_by_rank[rank]
        inc = received_by_rank[rank]
        if (out.get("gradient_hash") != inc.get("gradient_hash") or
                out.get("sample_ids") != inc.get("sample_ids") or
                out.get("tail_hash") != inc.get("applied_tail_hash") or
                inc.get("forward_calls") != 0):
            raise AssertionError(f"round {round_index}, rank {rank}: relay audit mismatch")

    for step in range(cut + 1, final_step + 1):
        a_rows = [row for row in rows(session, data, "node0", "commit", round_index)
                  if row.get("step") == step and row.get("epoch") == round_index + 1]
        t_rows = [row for row in rows(session, data, target, "commit", round_index)
                  if row.get("step") == step and row.get("epoch") == round_index + 1]
        if not ranks_complete(a_rows) or not ranks_complete(t_rows):
            raise AssertionError(f"round {round_index}, step {step}: incomplete audited commits")
        a_by_rank = {row["rank"]: row for row in a_rows}
        t_by_rank = {row["rank"]: row for row in t_rows}
        for rank in range(8):
            ah = a_by_rank[rank].get("hashes", {})
            th = t_by_rank[rank].get("hashes", {})
            if any(not ah.get(key) or ah.get(key) != th.get(key)
                   for key in ("model", "optimizer", "scheduler")):
                raise AssertionError(f"round {round_index}, step {step}, rank {rank}: state hash mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--post-steps", type=int, default=10)
    parser.add_argument("--scheduler-step",type=int)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("name may contain only letters, digits, '_' and '-'")
    if args.cycles < 1 or args.post_steps < 3:
        parser.error("cycles must be positive and post-steps must be at least 3")
    if 31410 + 20 * (args.cycles - 1) > 65535:
        parser.error("cycle Lighthouse ports exceed the TCP port range")

    base = json.loads((ROOT / "configs/v1_bench_shift.json").read_text())
    base.update(entry_module="shift_full.planned_train", target_steps=100000,
                lighthouse_port=31400, master_port=31500, min_replicas=2,
                heartbeat_timeout_ms=1000, join_timeout_ms=1000,
                quorum_tick_ms=100, recycle_after_worker_fault=False,
                external_lighthouse=False)
    train_args = base["train_args"]
    train_args[train_args.index("--communication-timeout") + 1] = "30"
    train_args[train_args.index("--max-seconds") + 1] = "1800"
    train_args += ["--sync-quorum", "--fixed-batch", "--data", "random",
                   "--planned-variant", "p2", "--planned-source", "B",
                   "--planned-cycles", str(args.cycles)]
    if args.audit:
        train_args += ["--audit-state", "--hash-every", "1"]
    if args.scheduler_step:
        if args.scheduler_step<1:parser.error("scheduler step must be positive")
        base["env"]["SHIFT_PLANNED_SCHEDULER_STEP"]=str(args.scheduler_step)
    base["env"]["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = "60"
    base["audit_initial"] = args.audit

    session = PlannedSession(args.name, base, 1800, takeover_node3=True)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    lighthouse_jobs = {}
    results = []
    current = None
    error = None
    try:
        from scripts.verify_overlap_source import verify
        verify(session.out/'deployed_source.json')
        for endpoint in session.endpoints.values():
            session.ensure_connection(endpoint)

        preflight = {}
        for node, endpoint in session.endpoints.items():
            raw = session.execute_endpoint(
                endpoint,
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            )
            preflight[node] = raw.strip()
        (session.out / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
        if any(preflight.values()):
            raise RuntimeError("GPU compute load present; refusing experiment")

        for role in ("A", "B", "C"):
            session.launch_role(role, base)
        target0 = "node3"
        data = session.wait(
            "initial_groups_ready",
            lambda d: all(len(session.events(d, node, "ready")) == 8 for node in ("node0", "node1"))
            and ranks_complete(rows(session, d, target0, "planned_local_ready", 0)),
            timeout=300,
        )

        start_lighthouses(session, pool, base["python"], args.name, args.cycles, lighthouse_jobs)
        for node in NODES:
            session.mark_endpoint(session.endpoints[node], "START")
        session.event("start_released", physical_nodes=list(NODES), cycles=args.cycles)

        for round_index in range(args.cycles):
            source, target = (("node1", "node3") if round_index % 2 == 0
                             else ("node3", "node1"))
            post = args.post_steps if round_index == args.cycles - 1 else 3
            current = {
                "round": round_index, "source": source, "target": target,
                "cut": None, "post_steps": post, "post_step": None,
                "success": False,
            }
            results.append(current)
            write_status(session, args.name, args.cycles, args.post_steps, args.audit, results)

            if round_index > 0:
                session.set_key("advance", round_index)
                session.wait(
                    f"round_{round_index}_target_ready",
                    lambda d, r=round_index, t=target: ranks_complete(
                        rows(session, d, t, "planned_local_ready", r)),
                    timeout=300,
                )

            session.set_key(round_key(args.cycles, round_index, "arm"), "1")
            selected = session.wait(
                f"round_{round_index}_cut_selected",
                lambda d, r=round_index: bool(rows(session, d, "node0", "planned_cut_selected", r)),
                timeout=300,
            )
            cut_events = rows(session, selected, "node0", "planned_cut_selected", round_index)
            cut_events.sort(key=lambda row: row.get("time", 0))
            cut = int(cut_events[-1]["step"])
            final_step = cut + post
            current.update(cut=cut, post_step=final_step)

            if round_index == args.cycles - 1:
                for node in NODES:
                    session.mark_endpoint(session.endpoints[node], "TRAIN_UNTIL", str(final_step))

            data = session.wait(
                f"round_{round_index}_ac_three_commits",
                lambda d, r=round_index, s=cut + 3: all(
                    ranks_complete([row for row in rows(session, d, node, "commit", r)
                                    if row.get("step") == s and row.get("epoch") == r + 1])
                    for node in ("node0", target)),
                timeout=600,
            )
            data = wait_round(session, f"round_{round_index}_source_standby", source,
                              "planned_standby_verified", round_index, timeout=300)

            if round_index == args.cycles - 1:
                data = session.wait(
                    f"round_{round_index}_final_step",
                    lambda d, r=round_index, s=final_step: all(
                        ranks_complete([row for row in rows(session, d, node, "commit", r)
                                        if row.get("step") == s and row.get("epoch") == r + 1])
                        for node in ("node0", target)),
                    timeout=600,
                )

            if args.audit:
                audit_cycle(session, data, round_index, source, target, cut, final_step)
            current.update(
                three_ac_steps=True,
                source_standby_verified=True,
                audit_verified=args.audit,
                success=True,
            )
            write_status(session, args.name, args.cycles, args.post_steps, args.audit, results)
            session.event("round_complete", round=round_index, source=source, target=target,
                          cut=cut, post_step=final_step, audited=args.audit)
        session.event("CYCLES_PASS", cycles=args.cycles, audit=args.audit)
    except Exception as exc:
        error = repr(exc)
        if current is not None:
            current["error"] = error
        write_status(session, args.name, args.cycles, args.post_steps, args.audit, results, error)
        raise
    finally:
        for round_index in lighthouse_jobs:
            try:
                session.mark_endpoint(session.endpoints["node0"],
                                      f"STOP_NEW_LIGHTHOUSE_{round_index}")
            except Exception as exc:
                session.event("new_lighthouse_stop_marker_error", round=round_index,
                              error=repr(exc))
        for round_index, future in lighthouse_jobs.items():
            try:
                future.result(timeout=45)
            except Exception as exc:
                session.event("new_lighthouse_cleanup_error", round=round_index,
                              error=repr(exc))
        pool.shutdown(wait=True)
        session.finish()


if __name__ == "__main__":
    main()
