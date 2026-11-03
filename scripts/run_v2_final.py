"""Bounded final B-donor audit and ABBA timing; GPU experiments are serial."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_planned_handoff import analyze

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['audit','timing'],required=True);a=p.parse_args();out=ROOT/'results/planned_handoff';manifest=out/('v2_final_'+a.phase+'.jsonl')
 jobs=['p1f']*10 if a.phase=='audit' else ['p1','p1f','p1f','p1','p1','p1f','p1f','p1','p1','p1f']
 for i,v in enumerate(jobs,1):
  existing=a.phase=='audit' and i==1;name='v2_p1f_B_audit01' if existing else f'v2_final_{a.phase}_{i:02d}_{v}'
  start=time.monotonic();code=0
  if not existing:
   cmd=[sys.executable,'scripts/run_planned_handoff.py','--name',name,'--variant',v,'--source','B','--post-steps','6' if a.phase=='audit' else '50']+(['--audit'] if a.phase=='audit' else [])
   with (out/(name+'.console.log')).open('w') as stream:code=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
  row=dict(name=name,variant=v,source='B',reused_existing=existing,independent_launch=True,exit_code=code,seconds=time.monotonic()-start)
  if not code:
   r=analyze(ROOT/'results'/name);row.update(checks_pass=r['all_checks_pass'],gap=r['last_ab_to_first_ac_seconds'],extra=r['extra_gap_seconds'],freeze_to_begin=r['last_ab_to_first_ac_begin_seconds'],post_max=r['post_ac_max_interval_seconds'],post50_total=r['post_ac_total_seconds'])
  with manifest.open('a') as stream:stream.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
  if code or not row.get('checks_pass'):raise SystemExit(1)
if __name__=='__main__':main()
