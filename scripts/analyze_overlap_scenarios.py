"""Offline trajectory audit of four rejected/cancelled candidate scenarios."""
import argparse,json
from pathlib import Path

def analyze(path):
 path=Path(path);s=json.loads((path/'scenario_status.json').read_text());rows={}
 for node in ('node0','node1','node3'):
  rows[node]=[json.loads(l) for f in (path/('physical_'+node)).glob('events*.jsonl') for l in f.read_text().splitlines() if l.strip()]
 def get(node,kind,rank=None):return [r for r in rows[node] if r['kind']==kind and (rank is None or r['rank']==rank)]
 checks={};end=s['final_step'];fallback=s['scenario']=='relay-reject';cut=s.get('cut',end)
 checks['no_C_commit']=not get('node3','commit')
 checks['A_B_progress']=all([r['step'] for r in get(n,'commit',rank)]==list(range(1,end+1)) for n in ('node0','node1') for rank in range(8))
 checks['two_real_contributors']=all(r['participants']==2 and r['contributed'] and r['global_batch']==8 and r['finite_gradients'] and r['cross_calls']>0 and r['native_pg_world']==2 and r['native_pg_rank']==(0 if n=='node0' else 1) for n in ('node0','node1') for r in get(n,'commit'))
 checks['AB_members_epoch']=all(r['physical_members']==['A','B'] and r['epoch']==int(fallback and r['step']>cut) for n in ('node0','node1') for r in get(n,'commit'))
 checks['no_process_restart']=all(len({r['pid'] for r in get(n,'commit',rank)})==1 for n in ('node0','node1') for rank in range(8))
 checks['samples']=all({x for n in ('node0','node1') for r in get(n,'commit') if r['step']==step for x in r['sample_ids']}==set(range((step-1)*8,step*8)) for step in range(1,end+1))
 if s['audit']:
  checks['AB_state_exact']=all(all(next(r for r in get('node0','commit',rank) if r['step']==step)['hashes'][k]==next(r for r in get('node1','commit',rank) if r['step']==step)['hashes'][k] for k in ('model','optimizer','scheduler')) for rank in range(8) for step in range(cut+1,end+1))
 if fallback:
  checks['new_epoch_old_handle_rejected']=all(len(get(n,'planned_new_epoch_ab_recovered'))==8 and all(r['old_manager_rejected'] for r in get(n,'planned_new_epoch_ab_recovered')) for n in ('node0','node1'))
  checks['C_rejected']=len(get('node3','planned_activation_rejected'))==8 and bool(get('node3','planned_relay_rejected'))
 else:
  checks['all_cancelled']=all(len(get(n,'planned_cancelled'))==8 for n in rows)
 checks['clean_exit']=all(any(r.get('kind')=='cleanup' and r['trainer_returncode']==0 and r['own_children']==[] for l in (path/('physical_'+n)/'supervisor.jsonl').read_text().splitlines() for r in [json.loads(l)]) for n in rows)
 after=[end-r['step'] for n in ('node0','node1') for r in get(n,'planned_cancelled')]
 out=dict(passed=all(checks.values()),scenario=s['scenario'],checks=checks,min_commits_after_rank_cancel=min(after) if after else None,final_step=end)
 (path/'scenario_analysis.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');a=p.parse_args()
 for path in a.paths:
  r=analyze(path);print(json.dumps(r));assert r['passed']
