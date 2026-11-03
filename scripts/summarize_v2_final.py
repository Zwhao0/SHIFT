#!/usr/bin/env python3
"""Offline final matrix validator and summary writer; never launches training."""
import json,random,statistics,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; INDEX=ROOT/'results'/'planned_handoff'
OUT=ROOT/'results'/'v2'/'summary.json'; N=10; BOOT=5000; SEED=20261103
INPUT_ERRORS=[]
ABBA=['p1','p1f','p1f','p1','p1','p1f','p1f','p1','p1','p1f']
METRICS=('gap_s','extra_gap_s','freeze_to_begin_s','request_first_ac_s','post50_total_s')
def read_json(p): return json.loads(p.read_text())
def read_rows(p):
 out=[]
 if not p.is_file(): return out
 for i,line in enumerate(p.read_text().splitlines(),1):
  if line.strip():
   try: out.append(json.loads(line))
   except json.JSONDecodeError: INPUT_ERRORS.append(f'{p.name}:{i}: incomplete/invalid JSONL row')
 return out
def event_rows(path,node):
 out=[]
 for p in sorted((path/f'physical_{node}').glob('events*.jsonl')): out.extend(read_rows(p))
 return out
def pct(xs,p):
 xs=sorted(xs); x=(len(xs)-1)*p; lo=int(x); hi=min(lo+1,len(xs)-1)
 return xs[lo]+(xs[hi]-xs[lo])*(x-lo)
def dist(xs):
 return None if not xs else {'n':len(xs),'median':statistics.median(xs),'min':min(xs),'max':max(xs)}
def all_true(a): return a.get('all_checks_pass') is True and bool(a.get('checks')) and all(v is True for v in a['checks'].values())
def inspect(row,phase):
 errors=[]; name=row.get('name'); path=ROOT/'results'/str(name)
 if not name or not path.is_dir(): return None,[f'missing session directory: {name}']
 try: status=read_json(path/'planned_status.json'); analysis=read_json(path/'planned_analysis.json')
 except Exception as e: return None,[f'{name}: missing/invalid status or analysis ({type(e).__name__})']
 variant=row.get('variant'); post=6 if phase=='audit' else 50
 for key,val in [('source','B'),('variant','p1f' if phase=='audit' else variant)]:
  if row.get(key)!=val or status.get(key)!=val or analysis.get(key)!=val: errors.append(f'{name}: {key} mismatch')
 if phase=='timing' and variant not in ('p1','p1f'): errors.append(f'{name}: unexpected timing variant')
 if status.get('post_steps')!=post or analysis.get('post_steps')!=post: errors.append(f'{name}: expected post_steps={post}')
 if row.get('independent_launch') is not True or row.get('exit_code')!=0: errors.append(f'{name}: launch/exit metadata invalid')
 if row.get('same_A_continuous_rotation') is True: errors.append(f'{name}: same-A continuous rotation is not an independent session')
 if row.get('checks_pass') is not True or not all_true(analysis): errors.append(f'{name}: analyzer checks are not all true')
 if status.get('passed') is not True or analysis.get('passed') is not True: errors.append(f'{name}: status/analysis did not pass')
 checks=analysis.get('checks',{})
 if variant=='p1f':
  for k in ('retired_cleanup_trigger','retired_cleanup_by_third_ac','cleanup_after_first_joint_local_commit'):
   if checks.get(k) is not True: errors.append(f'{name}: {k} not verified')
 c=event_rows(path,'node3'); restored=[r for r in c if r.get('kind')=='restored']; sources=[r for r in c if r.get('kind')=='planned_restore_sources']
 if {r.get('rank') for r in restored}!=set(range(8)) or len(restored)!=8: errors.append(f'{name}: C restored rank coverage')
 if any(r.get('source_replica')!=1 or r.get('source_rank')!=r.get('rank') for r in restored): errors.append(f'{name}: C restore source/rank mismatch')
 if {r.get('rank') for r in sources}!=set(range(8)) or len(sources)!=8 or any(r.get('shared_source')!='B' for r in sources): errors.append(f'{name}: shared state source is not B on all C ranks')
 native={}
 for node in ('node0','node3'):
  rows=event_rows(path,node); ready=[r for r in rows if r.get('kind')=='candidate_pg_ready']; reused=[r for r in rows if r.get('kind')=='candidate_pg_reused']
  ok=len(ready)==len(reused)==8 and all(r.get('native_creations')==1 for r in ready+reused) and {r.get('rank') for r in ready}==set(range(8)) and {r.get('rank') for r in reused}==set(range(8))
  native[node]=ok
  if not ok: errors.append(f'{name}: {node} candidate native PG creation/reuse != 1 per rank')
 vals={'gap_s':analysis.get('last_ab_to_first_ac_seconds'),'extra_gap_s':analysis.get('extra_gap_seconds'),
  'freeze_to_begin_s':analysis.get('last_ab_to_first_ac_begin_seconds'),
  'request_first_ac_s':analysis.get('request_to_first_ac_observed_controller_seconds'),
  'post50_total_s':analysis.get('post_ac_total_seconds') if phase=='timing' else None,'prep_ab_progress_lower_bound':analysis.get('ab_commits_while_c_prepared_lower_bound'),
  'post49_intervals_s':analysis.get('ac_intervals_seconds',[]) if phase=='timing' else [],'native_pg_creations_one_per_rank':native}
 if phase=='timing' and len(vals['post49_intervals_s'])!=49: errors.append(f'{name}: expected 49 post50 intervals')
 if any(not isinstance(vals[k],(int,float)) for k in METRICS if phase=='timing' or k!='post50_total_s'): errors.append(f'{name}: required timing metric missing')
 return {'name':name,'variant':variant,'source':'B','post_steps':post,'validated':not errors,'validation_errors':errors,'metrics':vals,'analyzer_checks':checks},errors
def boot_ci(a,b):
 rng=random.Random(SEED); d=[]
 for _ in range(BOOT): d.append(statistics.median(rng.choices(a,k=len(a)))-statistics.median(rng.choices(b,k=len(b))))
 return {'difference':'median(p1) - median(p1f)','estimate':statistics.median(a)-statistics.median(b),'ci95':[pct(d,.025),pct(d,.975)],'resamples':BOOT,'seed':SEED}
def summarize(runs):
 out={}
 for v in ('p1','p1f'):
  rs=[r for r in runs if r['variant']==v]; row={'n':len(rs),'metrics':{}}
  for k in METRICS:
   q=dist([r['metrics'][k] for r in rs]); row['metrics'][k]=q
  intervals=[x for r in rs for x in r['metrics']['post49_intervals_s']]
  row['post49_intervals']=None if not intervals else {'n':len(intervals),'median':statistics.median(intervals),'p95':pct(intervals,.95),'max':max(intervals)}
  row['post49_first3_by_session']={r['name']:r['metrics']['post49_intervals_s'][:3] for r in rs}
  row['post49_first3_position_medians']=[statistics.median(r['metrics']['post49_intervals_s'][i] for r in rs) for i in range(3)] if rs and all(len(r['metrics']['post49_intervals_s'])>=3 for r in rs) else None
  row['prep_ab_progress_lower_bound']=dist([r['metrics']['prep_ab_progress_lower_bound'] for r in rs if isinstance(r['metrics']['prep_ab_progress_lower_bound'],(int,float))])
  row['native_pg_creations_one_per_rank']=all(all(r['metrics']['native_pg_creations_one_per_rank'].values()) for r in rs) if rs else False
  out[v]=row
 return out
def main():
 audit=read_rows(INDEX/'v2_final_audit.jsonl'); timing=read_rows(INDEX/'v2_final_timing.jsonl')
 valid=[]; errors=INPUT_ERRORS[:]; all_names=[]
 for phase,rows in (('audit',audit),('timing',timing)):
  seen=set()
  for r in rows:
   if r.get('name') in seen: errors.append(f'duplicate {phase} session: {r.get("name")}')
   if r.get('name') in all_names: errors.append(f'session reused across matrices: {r.get("name")}')
   all_names.append(r.get('name'))
   seen.add(r.get('name')); item,bad=inspect(r,phase); errors.extend(bad)
   if item: item['phase']=phase; valid.append(item)
  if len(rows)!=N: errors.append(f'{phase} matrix has {len(rows)}/{N} rows')
 if [r.get('variant') for r in timing]!=ABBA[:len(timing)]: errors.append('timing order is not the specified ABBA sequence')
 for v in ('p1','p1f'):
  n=sum(r.get('variant')==v for r in timing)
  if n!=5: errors.append(f'timing {v} count is {n}/5')
 if sum(r.get('variant')=='p1f' for r in audit)!=10: errors.append('audit p1f count is not 10')
 errors.extend(x for x in INPUT_ERRORS if x not in errors)
 counts={'audit_rows':len(audit),'timing_rows':len(timing),'audit_p1f':sum(r.get('variant')=='p1f' for r in audit),'timing_p1':sum(r.get('variant')=='p1' for r in timing),'timing_p1f':sum(r.get('variant')=='p1f' for r in timing),'audit_independent':sum(r.get('independent_launch') is True for r in audit),'timing_independent':sum(r.get('independent_launch') is True for r in timing)}
 perf=[r for r in valid if r['phase']=='timing' and r['validated']]; summary=summarize(perf)
 bootstrap={k:boot_ci([r['metrics'][k] for r in perf if r['variant']=='p1'],[r['metrics'][k] for r in perf if r['variant']=='p1f']) for k in METRICS if sum(r['variant']=='p1' for r in perf)==5 and sum(r['variant']=='p1f' for r in perf)==5}
 status='complete' if counts['audit_rows']==counts['timing_rows']==10 and not errors else ('failed' if counts['audit_rows']==counts['timing_rows']==10 else 'incomplete')
 result={'status':status,'counts':counts,'validation_errors':errors,'audit_sessions':[r for r in valid if r['phase']=='audit'],'timing_summary_by_variant':summary,'p1_minus_p1f_bootstrap':bootstrap,
  'method':{'timing_order_expected':ABBA,'independent_sessions':'requires unique session directories and independent_launch=true per index row; reused_existing count is reported.','same_A_10_cycles_used':any(r.get('same_A_continuous_rotation') is True for r in audit+timing),'metric_clocks':'gap/freeze/post intervals use planned_analysis A-host monotonic; request-to-first-AC uses controller monotonic.','post49_definition':'49 A-host commit intervals from first AC commit through post step 50; first three retained per session and position medians shown.','negative_extra_gap':'kept signed; it does not mean zero downtime.','state_scope':'latest full state is copied from B at the final synchronized boundary; this is not full MoE pre-copy or outer-parameter pre-gradient continuation.'},
  'reused_existing_sessions':[r.get('name') for r in audit+timing if r.get('reused_existing') is True]}
 OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'status':status,'counts':counts,'validation_errors':errors},ensure_ascii=False))
 if status!='complete': raise SystemExit(2 if status=='incomplete' else 1)
if __name__=='__main__': main()
