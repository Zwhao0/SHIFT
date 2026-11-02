#!/usr/bin/env python3
"""Summarize the cold/CPU/GPU standby comparison from collected evidence."""
import argparse, json, math, statistics
from pathlib import Path

MODES = ("cold", "cpu", "gpu")
METRICS = ("preparation_seconds", "fault_to_joint_seconds", "request_to_ready_seconds",
           "recovery_request_to_state_loaded_seconds", "state_loaded_to_joint_seconds",
           "healthy_max_adjacent_gap_seconds", "healthy_max_adjacent_gap_preparation_seconds",
           "detection_shrink_seconds")

def dist(values):
    v = sorted(x for x in values if isinstance(x, (int, float)))
    if not v: return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {"count": len(v), "min": v[0], "median": statistics.median(v),
            "p95": v[max(0, math.ceil(.95*len(v))-1)], "max": v[-1]}

def events(session, manifest, rid, generation):
    row = next((x for x in manifest if x.get("rid")==rid and x.get("generation")==generation), None)
    if not row: return [], "binding missing"
    directory = session / ("physical_" + row["physical_node"]) / f"rid{rid}_g{generation}"
    rows=[]
    for rank in range(8):
        path=directory/f"events_g{generation}_r{rank}.jsonl"
        if not path.is_file(): return rows, f"rank {rank} event file missing"
        for line in path.read_text().splitlines():
            try: item=json.loads(line)
            except ValueError: continue
            if isinstance(item,dict): rows.append(item)
    return rows, None

def max_gap_window(times, start, end):
    t=sorted(set(x for x in times if isinstance(x,(int,float))))
    gaps=[b-a for a,b in zip(t,t[1:]) if b>=start and a<=end]
    return max(gaps) if gaps else None

def resource(value):
    if not isinstance(value,dict): return None
    p=value.get("processes") or {}
    return {"gpus":[{"index":g.get("index"),"memory_used_mib":g.get("memory_used_mib")}
                      for g in value.get("gpus",[]) if isinstance(g,dict)],
            "processes":{k:p.get(k) for k in ("count","states","vmrss_kib_sum",
                                                "cpu_utime_ticks_sum","cpu_stime_ticks_sum")}}

def summarize_trial(root, entry):
    name, mode=entry["session"], entry["mode"]; controller=entry.get("controller",{}); verified=entry.get("verified",{})
    session=root.parent/name
    fault=verified.get("fault_time",controller.get("fault_time")); joint=verified.get("joint_time",controller.get("joint_time"))
    req=controller.get("request_time"); ready=controller.get("ready_time")
    loaded=controller.get("state_loaded_time"); recovery=controller.get("recovery_request_time")
    raw_manifest=json.loads((session/"physical_manifest.json").read_text())
    manifest=raw_manifest if isinstance(raw_manifest,list) else []
    target=verified.get("target",controller.get("target")); healthy=verified.get("healthy_rid",1-target if target in (0,1) else None)
    generations=verified.get("pre_fault_generations",{})
    healthy_gen=int(generations.get(str(healthy),generations.get(healthy,0))) if healthy is not None else 0
    raw,raw_error=events(session,manifest,healthy,healthy_gen) if healthy is not None else ([],"healthy rid missing")
    rank_times={}
    if fault is not None and joint is not None:
        for rank in range(8):
            ts=[x.get("time") for x in raw if x.get("kind")=="reconfigure_end" and x.get("rank")==rank
                and x.get("world_size")==1 and isinstance(x.get("time"),(int,float)) and fault<x["time"]<joint]
            if ts: rank_times[rank]=min(ts)
    shrink=(max(rank_times.values())-fault) if len(rank_times)==8 else None
    prep_event=None
    controller_path=session/"controller.jsonl"
    if controller_path.is_file():
        for line in controller_path.read_text().splitlines():
            try: item=json.loads(line)
            except ValueError: continue
            if item.get("kind")=="standby_prepared" and item.get("index")==controller.get("index"):
                prep_event=item; break
    prep_seconds=prep_event.get("preparation_seconds",controller.get("preparation_seconds")) if prep_event else None
    prep_end=prep_event.get("time") if prep_event else None
    prep_start=prep_end-prep_seconds if isinstance(prep_end,(int,float)) and isinstance(prep_seconds,(int,float)) else None
    prep_gaps=[]
    if prep_start is not None:
        for rank in range(8):
            ts=[x["time"] for x in raw if x.get("kind")=="commit" and x.get("rank")==rank
                and x.get("contributed") is True and isinstance(x.get("time"),(int,float))]
            gap=max_gap_window(ts,prep_start,prep_end)
            if gap is not None: prep_gaps.append(gap)
    prep_gap=max(prep_gaps) if len(prep_gaps)==8 else None
    new_rid=target; new_gen=int(controller.get("generation",verified.get("generation",0)))
    new_rows,ready_error=events(session,manifest,new_rid,new_gen) if new_rid in (0,1) else ([],"target missing")
    ready_rows={x.get("rank"):x for x in new_rows if x.get("kind")=="ready"}
    standby=controller.get("standby_resources",[])
    idle,prepared,recovered=(resource(controller.get(k)) for k in
                             ("idle_resources","prepared_resources","recovered_resources"))
    ip=(controller.get("idle_resources") or {}).get("processes",{})
    pp=(controller.get("prepared_resources") or {}).get("processes",{})
    delta={k:(pp.get(k)-ip.get(k) if isinstance(pp.get(k),(int,float)) and isinstance(ip.get(k),(int,float)) else None)
            for k in ("count","vmrss_kib_sum","cpu_utime_ticks_sum","cpu_stime_ticks_sum")}
    metrics={"preparation_seconds":controller.get("preparation_seconds"),
             "fault_to_joint_seconds":verified.get("fault_to_joint_seconds",controller.get("fault_to_joint_seconds")),
             "request_to_ready_seconds":ready-req if isinstance(ready,(int,float)) and isinstance(req,(int,float)) else None,
             "recovery_request_to_state_loaded_seconds":loaded-recovery if isinstance(loaded,(int,float)) and isinstance(recovery,(int,float)) else None,
             "state_loaded_to_joint_seconds":joint-loaded if isinstance(joint,(int,float)) and isinstance(loaded,(int,float)) else None,
             "healthy_max_adjacent_gap_seconds":verified.get("healthy_max_commit_gap_fault_to_joint_seconds"),
             "healthy_max_adjacent_gap_preparation_seconds":prep_gap,
             "detection_shrink_seconds":shrink}
    return {"session":name,"trial":entry.get("trial"),"metrics":metrics,
            "detection_shrink":{"healthy_rid":healthy,"generation":healthy_gen,
                "ranks_found":sorted(rank_times),"reason":None if shrink is not None else (raw_error or "all 8 ranks need world_size=1 reconfigure_end between fault and joint")},
            "resources":{"idle":idle,"prepared":prepared,"recovered":recovered,
                "prepared_minus_idle_process_counters":delta,
                "standby_rows":[{"rank":x.get("rank"),"cuda_allocated_bytes":x.get("cuda_allocated_bytes"),
                                  "cpu_maxrss_kib":x.get("cpu_maxrss_kib")} for x in standby if isinstance(x,dict)],
                "recovered_ready_cuda_allocated_bytes":{str(r):x.get("cuda_allocated_bytes") for r,x in ready_rows.items()}},
            "raw_reference":{"session_dir":str(session),"fault_index":controller.get("index"),
                             "reconfigure_read_error":raw_error,"ready_read_error":ready_error},
            "standby_preparation_window":{"start":prep_start,"end":prep_end,
                "duration_seconds":prep_seconds,"timestamp_note":"start is event time minus preparation_seconds; end is controller event time",
                "healthy_ranks_with_gap":len(prep_gaps),"reason":None if prep_gap is not None else
                    ("cold mode has no standby_prepared event" if mode=="cold" else "need adjacent healthy commits for all 8 ranks in preparation window")}}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",type=Path,default=Path("results/v1/backup_timings.json")); p.add_argument("--output",type=Path,default=Path("results/v1/backup_summary.json")); a=p.parse_args()
    entries=json.loads(a.input.read_text()); root=a.input.parent
    grouped={}
    for mode in MODES:
        trials=[summarize_trial(root,e) for e in entries if e.get("mode")==mode]
        grouped[mode]={"trial_count":len(trials),"expected_trials":3,"complete":len(trials)==3,
                       "metrics":{k:dist([t["metrics"][k] for t in trials]) for k in METRICS},"trials":trials}
    report={"modes":grouped,"standby_preparation_timing_note":"Window start is controller standby_prepared time minus preparation_seconds; this approximates start at event emission precision.",
            "extra_standby_communication":{"bytes":0,"scope":"structural, before ACTIVATE only",
             "basis":"standby gate waits before Manager construction; no bandwidth measurement is claimed"}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

if __name__=="__main__": main()
