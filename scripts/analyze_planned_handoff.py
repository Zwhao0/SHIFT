"""Offline full-rank protocol/state checks and one-host commit timing."""
import argparse,json,statistics
from pathlib import Path

def analyze(path):
 path=Path(path);status=json.loads((path/'planned_status.json').read_text());cut=status['cut'];post=status.get('post_steps',3);rows={}
 for node in ('node0','node1','node3'):
  rows[node]=[json.loads(line) for f in (path/('physical_'+node)).glob('events*.jsonl') for line in f.read_text().splitlines() if line.strip()]
 def get(node,kind,rank=None):return [r for r in rows[node] if r['kind']==kind and (rank is None or r['rank']==rank)]
 checks={}
 checks['graceful_cleanup']=all(any(r['kind']=='cleanup' and r['trainer_returncode']==0 and r['own_children']==[] for l in (path/('physical_'+n)/'supervisor.jsonl').read_text().splitlines() for r in [json.loads(l)]) for n in rows)
 checks['rank_coverage']=all({r['rank'] for r in get(n,'commit')}==set(range(8)) for n in rows)
 checks['two_contributors_batch8']=all(r['participants']==2 and r['global_batch']==8 and r['contributed'] for n in rows for r in get(n,'commit'))
 checks['real_gradient_path']=all(r['finite_gradients'] and r['cross_calls']>0 for n in rows for r in get(n,'commit'))
 checks['members']=all(r['physical_members']==(['A','B'] if r['epoch']==0 else ['A','C']) and (n!='node1' or r['epoch']==0) and (n!='node3' or r['epoch']==1) for n in rows for r in get(n,'commit'))
 checks['continuous_progress']=all([r['step'] for r in get(n,'commit',rank)]==list(range(1 if n!='node3' else cut+1,(cut if n=='node1' else cut+post)+1)) for n in rows for rank in range(8))
 checks['native_training_membership']=all(r['native_pg_world']==2 and r['native_pg_rank']==(0 if n=='node0' else 1) and r['native_global_ranks']==[r['rank'],8+r['rank']] for n in rows for r in get(n,'commit'))
 checks['no_b_in_ac_pg']=not get('node1','candidate_pg_ready') and all(len(get(n,'planned_old_manager_closed'))==8 and all(r['training_pg_absent'] or (r.get('server_closed') and r.get('old_epoch_revoked') and len(get(n,'planned_retired_cleanup_complete'))==8 and all(x['all_parent_absent'] for x in get(n,'planned_retired_cleanup_complete'))) for r in get(n,'planned_old_manager_closed')) for n in ('node0','node1'))
 checks['a_pid_unchanged']=all(len({r['pid'] for r in get('node0','ready',rank)+get('node0','commit',rank)})==1 for rank in range(8))
 checks['b_alive_through_three_ac']=len(get('node1','planned_standby_verified'))==8 and len(get('node1','planned_old_commit_rejected'))==8
 checks['candidate_reused_once']=all(len(get(n,'candidate_pg_ready',rank))==len(get(n,'candidate_pg_reused',rank))==1 and get(n,'candidate_pg_reused',rank)[0]['native_creations']==1 for n in ('node0','node3') for rank in range(8))
 if status['variant'] in ('p1','p1d','p1f'):
  checks['candidate_ready_before_freeze']=all(get('node0','candidate_pg_ready',rank)[0]['monotonic']<get('node0','planned_final_freeze',rank)[0]['monotonic'] for rank in range(8))
 else:
  checks['candidate_built_after_freeze']=all(get('node0','candidate_pg_ready',rank)[0]['monotonic']>get('node0','planned_final_freeze',rank)[0]['monotonic'] for rank in range(8))
 if status['variant'] in ('p1d','p1f'):
  checks['retired_cleanup_trigger']=all(len(get(n,'planned_retired_cleanup_begin'))==8 and all(r['first_commit_keys_ready'] for r in get(n,'planned_retired_cleanup_begin')) for n in rows)
  checks['retired_cleanup_by_third_ac']=all(max(r['monotonic'] for r in get(n,'planned_retired_cleanup_complete'))<=max(r['monotonic'] for r in get(n,'commit') if r['step']==cut+3) for n in ('node0','node3'))
  checks['cleanup_after_first_joint_local_commit']=all(min(r['monotonic'] for r in get(n,'planned_retired_cleanup_begin'))>=max(r['monotonic'] for r in get(n,'commit') if r['step']==cut+1) for n in ('node0','node3'))
 checks['sample_sequence']=True
 for step in range(1,cut+post+1):
  nodes=('node0','node1') if step<=cut else ('node0','node3')
  samples={x for n in nodes for r in get(n,'commit') if r['step']==step for x in r['sample_ids']}
  if samples!=set(range((step-1)*8,step*8)):checks['sample_sequence']=False
 checks['logical_data_role_preserved']=True
 for rank in range(8):
  aoffset=None;boffset=None
  for n in ('node0','node1','node3'):
   for r in get(n,'commit',rank):
    offsets=[x-(r['step']-1)*8 for x in r['sample_ids']]
    if n=='node0':
     aoffset=offsets if aoffset is None else aoffset
     valid=offsets==aoffset and offsets[0] in (0,1)
    elif n=='node1':
     boffset=offsets if boffset is None else boffset
     valid=offsets==boffset and offsets[0] in (2,3)
    else:valid=offsets==boffset
    checks['logical_data_role_preserved'] &= valid
 if status['audit']:
  checks['declared_donor_restored']=all(get('node3','restored',rank)[0]['source_replica']==(1 if status['source']=='B' else 0) and get('node3','restored',rank)[0]['source_rank']==rank for rank in range(8))
  checks['full_restore']=all(get('node3','restored',rank)[0]['hashes_match'] and get('node3','restored',rank)[0]['optimizer_tensors']>0 for rank in range(8))
  checks['ac_state_exact']=all(all(len({next(r for r in get(n,'commit',rank) if r['step']==step)['hashes'][key] for n in ('node0','node3')})==1 for key in ('model','optimizer','scheduler')) for step in range(cut+1,cut+4) for rank in range(8))
  checks['b_rng_inherited']=all(all(get('node3','restored',rank)[0]['actual_hashes'][key]==next(r for r in get('node1','snapshot',rank) if r['step']==cut)['hashes'][key] for key in ('cpu_rng','cuda_rng','python_rng','buffers')) for rank in range(8))
  checks['shared_state_compatible_before_cut']=all(all(next(r for r in get('node0','commit',rank) if r['step']==cut)['hashes'][key]==next(r for r in get('node1','commit',rank) if r['step']==cut)['hashes'][key] for key in ('model','optimizer','scheduler')) for rank in range(8))
  checks['distinct_rid_rng']=all(get('node0','ready',rank)[0]['hashes']['cuda_rng']!=get('node1','ready',rank)[0]['hashes']['cuda_rng'] for rank in range(8))
 acommits=get('node0','commit');committed={step:max(r['monotonic'] for r in acommits if r['step']==step) for step in range(1,cut+post+1)}
 normal=[committed[k]-committed[k-1] for k in range(2,cut+1)];gap=committed[cut+1]-committed[cut];steady=statistics.median(normal[-min(10,len(normal)):])
 ctrl=[json.loads(l) for l in (path/'controller.jsonl').read_text().splitlines()];request=next(r for r in ctrl if r['kind']=='REQUEST');observed=next(r for r in ctrl if r['kind']=='ac_three_commits')
 # Request->first AC is bounded by controller response snapshots, not cross-host subtraction.
 selected=next(r for r in ctrl if r['kind']=='FINAL_BOUNDARY_SELECTED')
 ready=max(r['monotonic'] for r in get('node0','candidate_pg_ready'))
 before=sum(t<ready for step,t in committed.items() if step<=cut)
 first=next((r for r in ctrl if r['kind']=='FIRST_AC_COMMIT'),None)
 request_rows=get('node0','planned_request_seen');request_step=max((r['step'] for r in request_rows),default=2)
 ready_progress=next((r['step'] for r in ctrl if r['kind']=='AB_PROGRESS_WHEN_C_READY'),None)
 first_begin=max(next(r for r in get('node0','begin',rank) if r['step']==cut)['monotonic'] for rank in range(8))
 post_intervals=[committed[k]-committed[k-1] for k in range(cut+2,cut+post+1)]
 result=dict(post_ac_max_interval_seconds=max(post_intervals),post_ac_p95_interval_seconds=statistics.quantiles(post_intervals,n=20,method='inclusive')[18],last_ab_to_first_ac_begin_seconds=first_begin-committed[cut],post_ac_total_seconds=committed[cut+post]-committed[cut],request_to_first_ac_observed_controller_seconds=first['mono']-request['mono'] if first else None,ab_commits_while_c_prepared_lower_bound=ready_progress-request_step if ready_progress is not None else None,session=path.name,**status,checks=checks,all_checks_pass=all(checks.values()),last_ab_to_first_ac_seconds=gap,normal_commit_median_seconds=steady,extra_gap_seconds=gap-steady,normal_intervals_seconds=normal,ac_intervals_seconds=[committed[k]-committed[k-1] for k in range(cut+2,cut+post+1)],ab_commits_before_candidate_ready=before,ab_commits_after_candidate_ready=cut-before,ab_commits_total=cut,request_to_boundary_selected_controller_seconds=selected['mono']-request['mono'],request_to_three_ac_observed_controller_seconds=observed['mono']-request['mono'],timing_note='AB/AC commit gaps use max rank timestamp on A only; controller times are same controller monotonic; audit/timing separated')
 (path/'planned_analysis.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');a=p.parse_args()
 for path in a.paths:
  r=analyze(path);print(json.dumps({k:v for k,v in r.items() if k not in ('normal_intervals_seconds','checks')}));assert r['all_checks_pass'],r['checks']
