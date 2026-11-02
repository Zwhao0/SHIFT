#!/usr/bin/env python3
import json, subprocess, sys, tarfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shift_full.control import Session, endpoints_from_env, CONTAINER_ROOT
MODES = ("flat", "shift")
RANKS = range(8)
HASH_FIELDS={"model","optimizer","scheduler","progress","cpu_rng","cuda_rng","python_rng"}
def route_map(rows, replay=False):
    grouped={}
    for e in rows:
        if e.get("kind")=="routing" and e.get("generation")==0 and e.get("step")==2:
            grouped.setdefault((e.get("rid"),e.get("rank"),e.get("layer")),[]).append(e)
    out={}
    for (rid,rank,layer), events in grouped.items():
        for i,e in enumerate(sorted(events,key=lambda x:x.get("call",-1))):
            logical,ordinal=(i%2,i//2) if replay else (rid,i); out[(logical,rank,layer,ordinal)]={k:e.get(k) for k in ("topk_hash","sorted_topk_hash")}
    return out
def selected(rows, kind, step, field):
    return {(e.get("rid"),e.get("rank")):e.get(field) for e in rows if e.get("kind")==kind and e.get("generation")==0 and e.get("step")==step}
def build_config(mode, ip):
    cfg=json.loads((ROOT/"configs/v1_trajectory_shift.json").read_text()); args=cfg["train_args"]; clean=[]; i=0
    while i<len(args):
        a=args[i]
        if a=="--reference-dump": i+=1; continue
        if a in ("--reference-anchor","--reference-anchor-step","--microbatches"):
            clean += [a,f"{CONTAINER_ROOT}/results/v1_trajectory_{mode}_01" if a=="--reference-anchor" else "2"]; i+=2; continue
        clean.append(a); i+=1
    for f,v in (("--reference-anchor",f"{CONTAINER_ROOT}/results/v1_trajectory_{mode}_01"),("--reference-anchor-step","2"),("--microbatches","2")):
        if f not in clean: clean += [f,v]
    cfg.update(target_steps=3, min_replicas=1, lighthouse_ip=ip, external_lighthouse=False, train_args=clean)
    return cfg
def read_state(session, endpoint, attempts=4):
    for attempt in range(attempts):
        try: session.read_endpoint(endpoint,0,rid=0); return session._read_cache[endpoint.physical_node]
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
            if isinstance(exc,subprocess.CalledProcessError) and exc.returncode!=255: raise
            session.ensure_connection(endpoint, force=True); time.sleep(1)
    raise TimeoutError("short event reads failed after reconnects")
def marker(session, endpoint, name):
    path=f"{CONTAINER_ROOT}/results/{session.name}/{name}"; code=f"from pathlib import Path; p=Path({path!r}); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+'.tmp'); t.write_text(''); t.replace(p)"
    for attempt in range(2):
        try: return session.execute_endpoint(endpoint,["python","-c",code],timeout=30)
        except subprocess.CalledProcessError as exc:
            if exc.returncode!=255 or attempt: raise
            session.ensure_connection(endpoint,force=True)
def clean(data):
    return all(has_event(data,"stopped",r) for r in RANKS) and any(e.get("kind")=="cleanup" and e.get("own_children")==[] for e in data.get("supervisor.jsonl", []))
def has_event(data, kind, rank):
    return any(e.get("kind")==kind and e.get("rid")==0 and e.get("rank")==rank and e.get("generation")==0 for e in data.get(f"events_g0_r{rank}.jsonl", []))
def wait_for(session, endpoint, predicate, timeout, label, allow_failure=False):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        data=read_state(session,endpoint); bad=[e for e in data.get("supervisor.jsonl",[]) if e.get("kind") in ("unexpected_exit","supervisor_error")]
        if bad and not allow_failure: raise RuntimeError("supervisor failed: "+str(bad[-1].get("kind")))
        if predicate(data): return data
        time.sleep(.5)
    raise TimeoutError(label)
def safe_archive(session, endpoint, name, out):
    remote_dir=f"{CONTAINER_ROOT}/results/{name}"; archive=f"{CONTAINER_ROOT}/results/{name}-logs.tgz"; host=f"{endpoint.host_root}/results/{name}-node1.tgz"
    session.execute_endpoint(endpoint,["tar","-czf",archive,"--exclude=*.pt","-C",remote_dir,"."],timeout=120)
    session.ssh(endpoint,["mkdir","-p",endpoint.host_root+"/results"],timeout=30); session.docker(endpoint,["cp",endpoint.container+":"+archive,host],timeout=120); local=out/"node1.tgz"
    subprocess.run(["scp","-q","-o",f"ControlPath={ROOT/'tmp'/endpoint.socket}",f"test@{endpoint.ip}:{host}",str(local)],check=True,timeout=180)
    with tarfile.open(local,"r:gz") as tar: tar.getmembers(); session.event("archive_collected",physical_node=endpoint.physical_node,archive=local.name,bytes=local.stat().st_size)
def original_rows(mode):
    return [json.loads(x) for rid in (0,1) for rank in RANKS for x in (ROOT/f"results/v1_trajectory_{mode}_01/node{rid}/events_g0_r{rank}.jsonl").read_text().splitlines() if x.strip()]
def source_check(rows):
    commits=selected(rows,"commit",2,"hashes"); plans=selected(rows,"batch_plan",2,"sample_ids")
    equal=[(isinstance(a:=commits.get((0,r)),dict) and isinstance(b:=commits.get((1,r)),dict) and set(a)==set(b)==HASH_FIELDS and a==b) for r in RANKS]
    routes=route_map(rows); plan_ok=all(isinstance(plans.get((r,k)),list) and len(plans[(r,k)])==2 for r in (0,1) for k in RANKS); ordered=all(isinstance(x.get("topk_hash"),str) for x in routes.values())
    return {"commit2_hash_pairs_equal":sum(equal),"expected_hash_pairs":8,"seven_hash_fields":sorted(HASH_FIELDS),"sample_plans":len(plans),"expected_sample_plans":16,"step2_routes":len(routes),"expected_step2_routes":368,"ordered_hashes_present":ordered,"pass":all(equal) and plan_ok and len(plans)==16 and len(routes)==368 and ordered}
def run_replay(mode, endpoints):
    name=f"v1_route_replay_{mode}_01"; endpoint=endpoints["node1"]; session=Session(name,build_config(mode,endpoint.ip),900,endpoints=endpoints)
    errors=[]; cleanup_ok=True; attempted=False
    try:
        session.ensure_connection(endpoint)
        check=f"from pathlib import Path; p=Path({(CONTAINER_ROOT+'/results/v1_trajectory_'+mode+'_01')!r}); assert all((p/f'reference_step2_r{{r}}.pt').is_file() for r in range(8))"
        session.execute_endpoint(endpoint,["python","-c",check],timeout=30)
        session.event("single_replica_plan", rid=0, generation=0, physical_node="node1", mode="shift",
                      anchor_mode=mode,anchor_step=2,target_steps=3,microbatches=2,fixed_batch=True,global_batch=8,interleave_rids=[0,1])
        attempted = True; session.launched = True
        session.started_endpoints[endpoint.physical_node] = endpoint
        session._endpoint_generations.setdefault(endpoint.physical_node, set()).add(0)
        session.start_supervisor(0, endpoint, 0)
        wait_for(session,endpoint,lambda d:all(has_event(d,"ready",r) for r in RANKS),240,"ready timeout"); session.event("all_ready",ranks=8); marker(session,endpoint,"START")
        wait_for(session,endpoint,lambda d:all(has_event(d,"stopped",r) for r in RANKS),240,"workers timeout")
    except Exception as exc:
        errors.append(type(exc).__name__); session.event("replay_error",error=type(exc).__name__)
    finally:
        if attempted:
            try: marker(session,endpoint,"STOP")
            except Exception as exc: errors.append("STOP marker: "+type(exc).__name__)
            try: wait_for(session,endpoint,clean,90,"owned-child cleanup timeout",allow_failure=True)
            except Exception as exc: errors.append("cleanup: "+type(exc).__name__); cleanup_ok=False
        session.event("complete",cleanup_errors=errors)
        manifest={"endpoints":[{"physical_node":"node1","ip":endpoint.ip,"logical_rids":[0],"generations":[0],"container":endpoint.container,"runtime":endpoint.runtime,"namespace":endpoint.namespace or None,"host_root":endpoint.host_root}],"bindings":[{"rid":0,"generation":0,"physical_node":"node1"}]}
        session.out.joinpath("physical_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        if attempted:
            try: safe_archive(session,endpoint,name,session.out)
            except Exception as exc: errors.append("archive: "+type(exc).__name__); session.event("collection_error",error=type(exc).__name__)
        session.log.close()
    rows={}
    if (session.out/'node1.tgz').exists():
        with tarfile.open(session.out/'node1.tgz') as tar:
            for member in tar.getmembers():
                filename=Path(member.name).name
                if member.isfile() and filename.startswith("events_g0_r") and filename.endswith(".jsonl"):
                    rows[filename]=[json.loads(x) for x in tar.extractfile(member).read().decode().splitlines() if x.strip()]
    return {"session":name,"mode":mode,"errors":errors,"cleanup_pass":cleanup_ok,"events":rows}
def compare(sources, source_checks, replay):
    results={}; route_maps={}
    for mode in MODES:
        source=sources[mode]; current=[e for rows in replay[mode]["events"].values() for e in rows]
        src,cur=route_map(source),route_map(current,replay=True); route_maps[mode]=cur
        same_keys=set(src)==set(cur); matched=sum(src[k].get("topk_hash")==cur[k].get("topk_hash") for k in set(src)&set(cur))
        originals=selected(source,"batch_plan",2,"sample_ids"); commits=selected(current,"commit",3,"sample_ids")
        samples={f"rid{rid}/rank{r}":{"match":isinstance(commits.get((0,r)),list) and commits[(0,r)][rid::2]==originals.get((rid,r)),"original":originals.get((rid,r)),"replay_commit3_interleave":commits.get((0,r))[rid::2] if isinstance(commits.get((0,r)),list) else None} for rid in (0,1) for r in RANKS}
        results[mode]={"ordered_hash_matches":matched,"source_routes":len(src),"replay_routes":len(cur),"route_keys_match":same_keys,
            "ordered_reproduction_pass":same_keys and matched==368 and len(src)==368 and len(cur)==368,"sample_ids_by_rank":samples,
            "sample_ids_pass":all(x["match"] for x in samples.values()),"missing_ordered_hash":sum(not isinstance(v.get("topk_hash"),str) for v in src.values())+sum(not isinstance(v.get("topk_hash"),str) for v in cur.values()),
            "missing_sorted_hash":sum(not isinstance(v.get("sorted_topk_hash"),str) for v in cur.values())}
    common=set(route_maps["flat"])&set(route_maps["shift"]); membership=[]; order_only=[]
    for key in sorted(common):
        f,s=route_maps["flat"][key],route_maps["shift"][key]
        row={"rid":key[0],"rank":key[1],"layer":key[2],"ordinal":key[3],"flat_ordered":f.get("topk_hash"),"shift_ordered":s.get("topk_hash"),"flat_sorted":f.get("sorted_topk_hash"),"shift_sorted":s.get("sorted_topk_hash")}
        if isinstance(f.get("sorted_topk_hash"),str) and isinstance(s.get("sorted_topk_hash"),str):
            if f["sorted_topk_hash"]!=s["sorted_topk_hash"]: membership.append(row)
            elif f.get("topk_hash")!=s.get("topk_hash"): order_only.append(row)
    valid=len(common)==368 and all(source_checks[m]["pass"] and not replay[m]["errors"] and results[m]["ordered_reproduction_pass"] and results[m]["sample_ids_pass"] and results[m]["missing_ordered_hash"]==0 and results[m]["missing_sorted_hash"]==0 and replay[m]["cleanup_pass"] for m in MODES)
    return {"diagnostic_valid":valid,"source_input_checks":source_checks,"replay_requirements":results,"common_route_keys":len(common),"membership_change_count":len(membership),"order_only_count":len(order_only),"membership_examples":membership[:20],"order_only_examples":order_only[:20],"interpretation":"One replica interleaves both inputs: route ordinals 0/2 map to original rid0 microbatches 0/1; 1/3 map to rid1. Valid only if all 368 ordered hashes and 16 sample plans reproduce. Sorted top-k differences indicate membership changes; ordered-only differences indicate selection order changes. Diagnostic only, not a trajectory or performance substitute."}
def main():
    out=ROOT/"results/v1"; out.mkdir(parents=True,exist_ok=True); endpoints=endpoints_from_env(); sources={m:original_rows(m) for m in MODES}; checks={m:source_check(sources[m]) for m in MODES}
    invalid={m:x for m,x in checks.items() if not x["pass"]}
    collisions=[f"v1_route_replay_{m}_01" for m in MODES if (ROOT/"results"/f"v1_route_replay_{m}_01").exists()]
    if invalid or collisions:
        errs=(["source commit2 hashes/sample plans/routes must be complete and equal"] if invalid else [])+["existing replay session: "+x for x in collisions]; report={"diagnostic_valid":False,"source_input_checks":checks,"preflight_errors":errs}; (out/"route_membership.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        print(json.dumps({"diagnostic_valid":False,"preflight_errors":errs})); return 1
    probe=Session.__new__(Session); probe.root=ROOT; probe.endpoints=endpoints; probe._sudo_password=None; endpoint=endpoints["node1"]; probe.ensure_connection(endpoint)
    paths=[f"{CONTAINER_ROOT}/results/v1_trajectory_{m}_01" for m in MODES]
    code=f"from pathlib import Path; assert 'sorted_topk_hash' in Path({(CONTAINER_ROOT+'/shift_full/train.py')!r}).read_text(); ps={[str(p) for p in paths]!r}; assert all(all((Path(p)/f'reference_step2_r{{r}}.pt').is_file() for r in range(8)) for p in ps)"
    for attempt in range(3):
        try: probe.execute_endpoint(endpoint,["python","-c",code],timeout=45); break
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
            if isinstance(exc,subprocess.CalledProcessError) and exc.returncode!=255: raise
            if attempt==2: raise
            probe.ensure_connection(endpoint,force=True)
    runs={m:run_replay(m,endpoints) for m in MODES}; report=compare(sources,checks,runs); report["replays"]={m:{k:v for k,v in runs[m].items() if k!="events"} for m in MODES}; (out/"route_membership.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"diagnostic_valid":report["diagnostic_valid"],"membership_changes":report["membership_change_count"],"order_only":report["order_only_count"]})); return 0 if report["diagnostic_valid"] else 1
if __name__=="__main__": raise SystemExit(main())
